"""
Tang バックボーン (TangUNet1D) の単体テスト。

smoke_test が見るのは形状の整合だけで、「情報が実際に流れているか」は見ていない。
ここで検証するのは、Tang 構成に寄せたことで新しく入った経路が本当に効いているか:

    1. 形状        : 条件あり / 無条件 (CFG) の両経路で (B,12,96) を返す
    2. 時刻の位置符号: 時間方向に一定な入力を入れたとき、内部スロットの出力が時刻で
                     変わること。位置符号だけを 0 にした同じモデルと比べて切り分ける
    3. 条件付け     : cond_idx を変えると出力が変わる。drop_mask=True の行は
                     cond_idx=None と厳密に一致する（CFG の無条件経路の同一性）
    4. adaLN        : 初期状態で γ=0, β=0（変調が恒等）。零初期化が効いていること
    5. 中間特徴     : features() が h1/h2/h3 を解像度 96/48/24 で返す
                     （clock_diagnostics が両バックボーンで同じ表を出せる条件）
    6. 拡散との整合 : agg.Diffusion.loss が有限値、q_sample(t=0) が x0 を返す

★ 出口の零初期化について:
    TangUNet1D は out_fc を、adaLN は変調 Linear を零初期化する。学習開始時を
    恒等から始めるための意図的な設計だが、そのままだと出力が恒等的に 0 なので
    2 と 3 は「差が 0」になって何も測れない。よって _wake_up() で零初期化された層だけを
    小さな乱数で埋めてから測る。これはテスト用の細工であって、学習経路は変更しない。

    uv run python src/models/DDPM_Aggregate_Tang/test_backbone.py
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


tang = _load("tang_backbone", REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Tang" / "model.py")
agg = tang.agg

DEV = tang.DEVICE
B = 8
NUM_SLOTS, NUM_ACT = tang.NUM_SLOTS, tang.NUM_ACT
K_COND = len(agg.COND_SPEC)


def _wake_up(model: nn.Module, seed: int = 0) -> nn.Module:
    """零初期化された層を小さな乱数で埋める。テスト2・3 で差を測れるようにするため。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if bool((p == 0).all()):
                p.copy_(torch.randn(p.shape, generator=g).to(p.device) * 0.05)
    return model


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(B, tang.IN_CH, NUM_SLOTS, device=DEV)
    t = torch.full((B,), 500, device=DEV, dtype=torch.long)
    cond = torch.as_tensor(agg.cond_grid()[:B], device=DEV)
    return x, t, cond


def test_shapes():
    m = tang.TangUNet1D().to(DEV).eval()
    x, t, cond = _inputs()
    assert m(x, t, cond).shape == x.shape
    assert m(x, t, None).shape == x.shape
    assert m.in_channels == tang.IN_CH == NUM_ACT
    print("1. 形状: OK")


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
    v = torch.randn(4, tang.IN_CH, 1, device=DEV).repeat(1, 1, NUM_SLOTS)
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
    """
    m = _wake_up(tang.TangUNet1D().to(DEV).eval())
    v_with, g_with = _interior_variation(m), _shift_gap(m, k=24)
    with torch.no_grad():                       # 位置符号だけを取り除く
        m.pos_enc.zero_()
    v_without, g_without = _interior_variation(m), _shift_gap(m, k=24)

    legacy = _wake_up(agg.UNet1D().to(DEV).eval())
    print(f"2. 内部スロットの時刻変動: PEあり={v_with:.4f}  PEなし={v_without:.4f}"
          f"  比={v_with / max(v_without, 1e-9):.1f}x   (参考: 現行UNet1D={_interior_variation(legacy):.4f})")
    print(f"   巡回シフト gap (k=24=6時間): PEあり={g_with:.4f}  PEなし={g_without:.4f}"
          f"   (参考: 現行UNet1D={_shift_gap(legacy, k=24):.4f})")
    assert v_with > 0.25, f"位置符号が効いていない (内部変動={v_with:.4f})"
    assert v_with > 2 * v_without, \
        f"位置符号を外しても内部変動が変わらない ({v_with:.4f} vs {v_without:.4f})"
    print("2. 時刻の位置符号: OK")


def test_conditioning():
    m = _wake_up(tang.TangUNet1D().to(DEV).eval())
    x, t, cond = _inputs()
    other = torch.roll(cond, shifts=1, dims=0)

    with torch.no_grad():
        base = m(x, t, cond)
        gap = (m(x, t, other) - base).norm().item() / base.norm().item()
        # drop_mask を全立てすると無条件経路と厳密に一致すること（CFG の前提）
        drop = torch.ones(B, dtype=torch.bool, device=DEV)
        dropped = m(x, t, cond, drop)
        uncond = m(x, t, None)

    print(f"3. 条件を入れ替えたときの相対差 = {gap:.4f}")
    assert gap > 1e-3, f"条件が出力に効いていない (gap={gap:.6f})"
    assert torch.allclose(dropped, uncond, atol=1e-5), \
        "drop_mask=True の行が無条件経路と一致しない（CFG が壊れる）"
    print("3. 条件付け: OK")


def test_adaln_zero_init():
    """adaLN の変調 Linear が零初期化されていること（学習開始時が恒等）。"""
    m = tang.TangUNet1D()
    mods = [mod for mod in m.modules()
            if isinstance(mod, tang.DoubleConv1D) and mod.mod is not None]
    assert len(mods) == 6, f"adaLN の注入点は Tang と同じ Down/Up の6箇所のはず (実際 {len(mods)})"
    for dc in mods:
        assert dc.mod is not None
        assert bool((dc.mod.weight == 0).all()) and bool((dc.mod.bias == 0).all())

    # 恒等であること: emb を変えても出力が変わらない
    # （Dropout が乱数を引かないよう eval にしてから測る）
    dc = mods[0].eval()
    x = torch.randn(2, dc.conv1.in_channels, 16)
    e1, e2 = torch.randn(2, tang.EMB_DIM), torch.randn(2, tang.EMB_DIM)
    with torch.no_grad():
        assert torch.allclose(dc(x, e1), dc(x, e2)), "零初期化なのに emb が出力を動かしている"
    print(f"4. adaLN 零初期化: OK (注入点 {len(mods)} 箇所)")


def test_features():
    m = tang.TangUNet1D().to(DEV).eval()
    x, t, cond = _inputs()
    with torch.no_grad():
        feats = m.features(x, t, cond)
    assert set(feats) == {"h1", "h2", "h3"}
    res = [feats[k].shape[2] for k in ("h1", "h2", "h3")]
    assert res == [96, 48, 24], f"解像度が現行 UNet1D と揃っていない: {res}"

    # 現行バックボーンと同じキー・同じ解像度であること（clock_diagnostics の前提）
    with torch.no_grad():
        legacy = agg.UNet1D().to(DEV).eval().features(x, t, cond)
    assert set(feats) == set(legacy)
    assert res == [legacy[k].shape[2] for k in ("h1", "h2", "h3")]
    print(f"5. 中間特徴: OK (解像度 {res})")


def test_diffusion_integration():
    m = tang.TangUNet1D().to(DEV).eval()
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
    print(f"6. 拡散との整合: OK (loss={loss.item():.4f})")


if __name__ == "__main__":
    print(f"device={DEV}  width_scale={tang.WIDTH_SCALE}  "
          f"channels={tang.channels(tang.WIDTH_SCALE)}")
    test_shapes()
    test_positional_encoding()
    test_conditioning()
    test_adaln_zero_init()
    test_features()
    test_diffusion_integration()
    print("\nall tests passed")
