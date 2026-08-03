"""
DiT バックボーン (DiT1D) の単体テスト。

smoke_test が見るのは形状の整合だけで、「情報が実際に流れているか」は見ていない。
ここで検証するのは、DiT 構成に寄せたことで新しく入った経路が本当に効いているか:

    1. 形状        : 条件あり / 無条件 (CFG) の両経路で (B,12,96) を返す
    2. patchify    : unpatchify(patchify(x)) == x。★DiT 固有（Tang には無い）。
                     時間順が入れ替わっても形は合うので、ここで押さえないと
                     「時刻がずれた学習」が静かに走る
    3. 時刻の位置符号: 時間方向に一定な入力を入れたとき、内部スロットの出力が時刻で
                     変わること。位置符号だけを 0 にした同じモデルと比べて切り分ける
    4. 条件付け     : cond_idx を変えると出力が変わる。drop_mask=True の行は
                     cond_idx=None と厳密に一致する（CFG の無条件経路の同一性）
    5. adaLN-Zero  : 初期状態で gate=0 のため★ブロック全体が恒等写像であること。
                     Tang の adaLN が「変調だけ恒等」なのに対し、DiT はブロックごと
                     恒等（残差の枝が閉じた状態から始まる）で、性質として強い
    6. 中間特徴     : features() が h1/h2/h3 を返し、★3点とも同一解像度であること
                     （DiT は等方的。U-Net 2本の 96/48/24 とは意味が違う）
    7. 拡散との整合 : agg.Diffusion.loss が有限値、q_sample(t=0) が x0 を返す

★ 出口の零初期化について:
    DiT1D は DiTBlock.ada_ln / FinalLayer の2層を零初期化する。学習開始時を
    恒等から始めるための意図的な設計だが、そのままだと出力が恒等的に 0 なので
    3 と 4 は「差が 0」になって何も測れない。よって _wake_up() で零初期化された層だけを
    小さな乱数で埋めてから測る。これはテスト用の細工であって、学習経路は変更しない。
    （テスト 5 だけは零初期化そのものを見るので _wake_up を掛けない）

    uv run python src/models/DDPM_Aggregate_DiT/test_backbone.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dit = _load("dit_backbone", REPO_ROOT / "src" / "models" / "DDPM_Aggregate_DiT" / "model.py")
agg = dit.agg

DEV = dit.DEVICE
B = 8
NUM_SLOTS, NUM_ACT = dit.NUM_SLOTS, dit.NUM_ACT


def _wake_up(model: nn.Module, seed: int = 0) -> nn.Module:
    """零初期化された層を小さな乱数で埋める。テスト3・4 で差を測れるようにするため。

    pos_enc は buffer なので parameters() に現れず、ここで壊れることはない。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if bool((p == 0).all()):
                p.copy_(torch.randn(p.shape, generator=g).to(p.device) * 0.05)
    return model


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(B, dit.IN_CH, NUM_SLOTS, device=DEV)
    t = torch.full((B,), 500, device=DEV, dtype=torch.long)
    cond = torch.as_tensor(agg.cond_grid()[:B], device=DEV)
    return x, t, cond


def test_shapes():
    m = dit.DiT1D().to(DEV).eval()
    x, t, cond = _inputs()
    assert m(x, t, cond).shape == x.shape
    assert m(x, t, None).shape == x.shape
    assert m.in_channels == dit.IN_CH == NUM_ACT
    print("1. 形状: OK")


def test_patchify_roundtrip():
    """★patchify/unpatchify が時間順を保って往復すること。

    P>1 でも成り立たなければならない（トークン内のスロット順が崩れると、
    形は合ったまま「時刻がずれたスケジュール」を学習してしまう）。
    """
    x, _, _ = _inputs()
    for patch in (1, 2, 4):
        m = dit.DiT1D(patch=patch).to(DEV).eval()
        tokens = m.patchify(x)
        assert tokens.shape == (B, NUM_SLOTS // patch, patch * dit.IN_CH)
        assert torch.allclose(m.unpatchify(tokens), x), f"patch={patch} で往復が壊れている"
    print("2. patchify 往復: OK (patch=1,2,4)")


def _shift_gap(model: nn.Module, k: int, seed: int = 0) -> float:
    """gap(k) = ‖ε(roll(x,k)) − roll(ε(x),k)‖ / ‖ε(x)‖。位置符号が無いモデルは ≈0。"""
    x, t, cond = _inputs(seed)
    with torch.no_grad():
        base = model(x, t, cond)
        rolled = model(torch.roll(x, shifts=k, dims=2), t, cond)
        return (rolled - torch.roll(base, shifts=k, dims=2)).norm().item() / base.norm().item()


def _interior_variation(model: nn.Module, seed: int = 0) -> float:
    """時間方向に一定な入力を与えたとき、内部スロットの出力がどれだけ時刻で変わるか。

    入力が時刻に依らないので、出力が時刻で変わる理由は「モデルが時刻を知っている」
    ことしかない。Conv1d のゼロパディング由来の位置情報は境界に集中するので、
    中央 (16..80 = 08:00-24:00 相当) だけを見ればパディングの寄与と切り分けられる。
    clock_diagnostics の B4 が境界/内部を分けて測るのと同じ考え方。
    """
    torch.manual_seed(seed)
    v = torch.randn(4, dit.IN_CH, 1, device=DEV).repeat(1, 1, NUM_SLOTS)
    t = torch.full((4,), 500, device=DEV, dtype=torch.long)
    cond = torch.as_tensor(agg.cond_grid()[:4], device=DEV)
    with torch.no_grad():
        out = model(v, t, cond)
    return (out[:, :, 16:80].std(dim=2).mean() / out.std()).item()


def test_positional_encoding():
    """★時刻の位置符号が効いていること。

    判定は「同じ重みの同じモデルで、位置符号だけを 0 にしたら内部変動が落ちるか」。
    別アーキ (現行 UNet1D) との比較だと初期化と構造の違いが混ざるので、PE の寄与だけを
    切り出す。現行 UNet1D の値は文脈として並記する（≈0 = 時計が無いことの定量確認）。

    ★ DiT は畳み込みを一切持たないので、PE を外すと出力は完全にシフト等変になる
      （境界パディングという「漏れ」の経路すら無い）。U-Net より落ち方が急なはず。
    """
    m = _wake_up(dit.DiT1D().to(DEV).eval())
    v_with, g_with = _interior_variation(m), _shift_gap(m, k=24)
    with torch.no_grad():                       # 位置符号だけを取り除く
        m.pos_enc.zero_()
    v_without, g_without = _interior_variation(m), _shift_gap(m, k=24)

    legacy = _wake_up(agg.UNet1D().to(DEV).eval())
    print(f"3. 内部スロットの時刻変動: PEあり={v_with:.4f}  PEなし={v_without:.4f}"
          f"  比={v_with / max(v_without, 1e-9):.1f}x   (参考: 現行UNet1D={_interior_variation(legacy):.4f})")
    print(f"   巡回シフト gap (k=24=6時間): PEあり={g_with:.4f}  PEなし={g_without:.4f}"
          f"   (参考: 現行UNet1D={_shift_gap(legacy, k=24):.4f})")
    assert v_with > 0.25, f"位置符号が効いていない (内部変動={v_with:.4f})"
    assert v_with > 2 * v_without, \
        f"位置符号を外しても内部変動が変わらない ({v_with:.4f} vs {v_without:.4f})"
    print("3. 時刻の位置符号: OK")


def test_conditioning():
    m = _wake_up(dit.DiT1D().to(DEV).eval())
    x, t, cond = _inputs()
    other = torch.roll(cond, shifts=1, dims=0)

    with torch.no_grad():
        base = m(x, t, cond)
        gap = (m(x, t, other) - base).norm().item() / base.norm().item()
        # drop_mask を全立てすると無条件経路と厳密に一致すること（CFG の前提）
        drop = torch.ones(B, dtype=torch.bool, device=DEV)
        dropped = m(x, t, cond, drop)
        uncond = m(x, t, None)

    print(f"4. 条件を入れ替えたときの相対差 = {gap:.4f}")
    assert gap > 1e-3, f"条件が出力に効いていない (gap={gap:.6f})"
    assert torch.allclose(dropped, uncond, atol=1e-5), \
        "drop_mask=True の行が無条件経路と一致しない（CFG が壊れる）"
    print("4. 条件付け: OK")


def test_adaln_zero_init():
    """★adaLN-Zero: 変調 Linear が零初期化され、ブロックが初期状態で恒等であること。

    gate も同じ Linear から出るので、零初期化は「変調が恒等」より強く
    「ブロックそのものが恒等（残差の枝が閉じている）」を意味する。DiT が深くしても
    学習が壊れない理由がここなので、恒等性そのものを直接測る。
    """
    m = dit.DiT1D()
    blocks = [b for b in m.modules() if isinstance(b, dit.DiTBlock)]
    assert len(blocks) == m.depth, f"ブロック数が depth と違う ({len(blocks)} vs {m.depth})"
    for blk in blocks:
        assert bool((blk.ada_ln.weight == 0).all()) and bool((blk.ada_ln.bias == 0).all())
    assert bool((m.final.ada_ln.weight == 0).all()) and bool((m.final.linear.weight == 0).all())

    # 恒等であること: c を変えても出力が x のまま
    # （Dropout が乱数を引かないよう eval にしてから測る）
    blk = blocks[0].eval()
    x = torch.randn(2, 16, m.hidden)
    c1, c2 = torch.randn(2, m.hidden), torch.randn(2, m.hidden)
    with torch.no_grad():
        assert torch.allclose(blk(x, c1), x), "零初期化なのにブロックが恒等でない"
        assert torch.allclose(blk(x, c2), x)

    # 出口も零なので、初期状態の ε̂ は恒等的に 0
    x, t, cond = _inputs()
    m = m.to(DEV).eval()
    with torch.no_grad():
        assert bool((m(x, t, cond) == 0).all()), "出口の零初期化が効いていない"
    print(f"5. adaLN-Zero 零初期化: OK (ブロック {len(blocks)} 本が恒等、ε̂=0)")


def test_features():
    m = dit.DiT1D().to(DEV).eval()
    x, t, cond = _inputs()
    with torch.no_grad():
        feats = m.features(x, t, cond)
    assert set(feats) == {"h1", "h2", "h3"}

    # ★DiT は等方的なので3点とも同じ解像度。深さだけが違う
    res = [feats[k].shape[2] for k in ("h1", "h2", "h3")]
    assert res == [m.n_tokens] * 3, f"DiT の中間特徴は全て同一解像度のはず: {res}"
    assert m.tap_idx == sorted(set(m.tap_idx)), f"タップ位置が重複している: {m.tap_idx}"
    assert m.tap_idx[-1] == m.depth - 1, "最深タップが最終ブロックでない"

    # キー集合は現行バックボーンと同じであること（clock_diagnostics の前提）。
    # 解像度は一致しない（U-Net は 96/48/24）ので、そこは比較しない
    with torch.no_grad():
        legacy = agg.UNet1D().to(DEV).eval().features(x, t, cond)
    assert set(feats) == set(legacy)
    print(f"6. 中間特徴: OK (解像度 {res}, タップ位置 {m.tap_idx}, "
          f"参考: 現行UNet1D={[legacy[k].shape[2] for k in ('h1', 'h2', 'h3')]})")


def test_diffusion_integration():
    m = dit.DiT1D().to(DEV).eval()
    d = agg.Diffusion()
    _, _, cond = _inputs()
    sched = torch.randint(0, NUM_ACT, (B, NUM_SLOTS), device=DEV)

    loss = d.loss(m, sched, cond)
    assert torch.isfinite(loss), f"loss が有限でない: {loss}"

    x0 = agg.sched_to_x0(sched)
    t0 = torch.zeros(B, dtype=torch.long, device=DEV)
    assert (d.q_sample(x0, t0, torch.zeros_like(x0)) - x0).abs().max() < 1e-4

    s = d.ddim_sample(m, cond, guidance_scale=1.0, steps=5)
    assert s.shape == (B, NUM_SLOTS) and 0 <= int(s.min()) and int(s.max()) < NUM_ACT
    print(f"7. 拡散との整合: OK (loss={loss.item():.4f})")


if __name__ == "__main__":
    m0 = dit.DiT1D()
    print(f"device={DEV}  size={m0.size}  patch={m0.patch}  depth={m0.depth}  "
          f"hidden={m0.hidden}  heads={m0.heads}  tokens={m0.n_tokens}  "
          f"params={sum(p.numel() for p in m0.parameters()):,}")
    test_shapes()
    test_patchify_roundtrip()
    test_positional_encoding()
    test_conditioning()
    test_adaln_zero_init()
    test_features()
    test_diffusion_integration()
    print("\nall tests passed")
