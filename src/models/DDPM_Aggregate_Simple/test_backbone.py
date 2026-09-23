"""
DDPM_Aggregate_Simple の単体テスト。

model.py の smoke_test が見るのは形状と定数の整合が中心なので、ここでは
「4つの変更が意図どおりに効いているか」と「変更していないはずの部分が
DDPM_Aggregate からずれていないか」を検証する:

    1. 形状        : 条件あり / 無条件 (CFG) の両経路で (B,12,96) を返す
    2. データ表現   : ★sched_to_x0 が {0,1}。argmax で往復する。
                     逆過程の clamp が [0,1] で、[-1,1] に戻っていない
    3. 時刻埋め込み : ★MLP を持たず、sinusoidal が直接 256次元で出る。
                     かつ t を変えると出力が実際に変わる（時刻情報が届いている）
    4. 条件付け     : cond_idx を変えると出力が変わる。drop_mask=True の行は
                     cond_idx=None と厳密に一致する（CFG の無条件経路の同一性）
    5. DDIM / EMA   : ★どちらも存在しない。チェックポイントのキーは "model" と "config" のみ
    6. 中間特徴     : features() が h1/h2/h3 を解像度 96/48/24 で返す
                     （clock_diagnostics が同じ表を出せる条件）
    7. 逆過程       : T を短くした ancestral が最後まで走り、正しい範囲のラベルを返す
    8. 原本との差分 : ★DDPM_Aggregate.UNet1D との違いが time_mlp だけであること。
                     自己完結（コピー）なので、意図しない差分が混入していないかを固定する
    9. 時計 (--clock) : φ の直交性、零初期化の時点で時計なしと出力が一致すること、
                     解像度 96/48/24 の位置の対応、保存して読み直したときの構造

★ 出口の零初期化について:
    UNet1D は out_conv を零初期化するので、そのままでは出力が恒等的に 0 になり
    3 と 4 は「差が 0」になって何も測れない。よって _wake_up() で out_conv だけを
    小さな乱数で埋めてから測る。テスト用の細工であって、学習経路は変更しない。

    uv run python src/models/DDPM_Aggregate_Simple/test_backbone.py
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 動的ロードしたモジュールは型チェッカから中身が見えないので Any で受ける
sm: Any = _load("simple_model", Path(__file__).resolve().parent / "model.py")
DEVICE = "cpu"          # テストは決定性重視で CPU 固定


def _wake_up(model: Any, seed: int = 0) -> Any:
    """零初期化された out_conv を小さな乱数で埋める（出力を 0 でなくするため）。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.out_conv.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def _model(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return _wake_up(sm.UNet1D().to(DEVICE).eval(), seed)


def _inputs(batch: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, sm.IN_CH, sm.NUM_SLOTS, generator=g)
    t = torch.full((batch,), 500, dtype=torch.long)
    c = torch.as_tensor(sm.cond_grid()[:batch], dtype=torch.long)
    return x, t, c


def test_shapes():
    m, (x, t, c) = _model(), _inputs()
    with torch.no_grad():
        assert m(x, t, c).shape == x.shape
        assert m(x, t, None).shape == x.shape          # 無条件 (CFG) 経路
    print("  1. 形状 (条件あり / 無条件): OK")


def test_onehot01_encoding():
    """★変更1: データ表現が {0,1} であること。"""
    sched = torch.randint(0, sm.NUM_ACT, (8, sm.NUM_SLOTS))
    x0 = sm.sched_to_x0(sched)

    assert x0.shape == (8, sm.IN_CH, sm.NUM_SLOTS)
    assert float(x0.min()) == 0.0 and float(x0.max()) == 1.0, \
        f"値域が {{0,1}} でない (min={float(x0.min())}, max={float(x0.max())})"
    assert torch.equal(x0.sum(dim=1), torch.ones(8, sm.NUM_SLOTS)), \
        "各スロットの one-hot の和が 1 でない"
    assert torch.equal(x0.argmax(dim=1), sched), "argmax でスケジュールに戻らない"

    # 逆過程の clamp が [0,1] であること（[-1,1] に戻っていないことの検出）。
    # x0_hat が下限に張り付く状況を作り、負値が残らないことを見る
    d = sm.Diffusion(device=DEVICE)
    ti = sm.T_STEPS - 1
    xt = torch.zeros(2, sm.IN_CH, sm.NUM_SLOTS)
    eps_big = torch.full_like(xt, 5.0)             # x0_hat が大きく負に振れる向き
    x0_hat = (xt - d.sqrt_1m_acp[ti] * eps_big) / d.sqrt_acp[ti]
    assert float(x0_hat.min()) < 0.0               # clamp 前は負
    x0_hat.clamp_(0.0, 1.0)
    assert float(x0_hat.min()) == 0.0, "clamp 下限が 0 でない（[-1,1] のままになっている）"
    print("  2. データ表現 {0,1} と clamp [0,1]: OK")


def test_time_embedding_direct():
    """★変更3: sinusoidal を MLP なしで直接足していること。"""
    t = torch.tensor([0, 250, 999])
    emb = sm.timestep_embedding(t)
    assert emb.shape == (3, sm.TIME_EMB_DIM), \
        f"時刻埋め込みが {sm.TIME_EMB_DIM} 次元で出ない: {tuple(emb.shape)}"

    m = _model()
    assert not hasattr(m, "time_mlp"), "time_mlp が残っている"
    assert not any("time_mlp" in k for k in m.state_dict()), \
        "state_dict に time_mlp のパラメータが残っている"

    # 時刻情報が実際に出力へ届いていること（MLP を外して経路が切れていないか）
    x, _, c = _inputs()
    with torch.no_grad():
        y_early = m(x, torch.full((x.size(0),), 10, dtype=torch.long), c)
        y_late  = m(x, torch.full((x.size(0),), 900, dtype=torch.long), c)
    gap = (y_early - y_late).abs().mean().item()
    assert gap > 1e-4, f"t を変えても出力が変わらない (平均差 {gap:.2e}) = 時刻経路が切れている"
    print(f"  3. 時刻埋め込み直結 (t=10 vs 900 の平均差 {gap:.4f}): OK")


def test_conditioning():
    m = _model()
    x, t, _ = _inputs()
    c0 = torch.zeros(x.size(0), len(sm.COND_SPEC), dtype=torch.long)
    c1 = torch.as_tensor(sm.cond_grid()[-x.size(0):], dtype=torch.long)

    with torch.no_grad():
        y0, y1 = m(x, t, c0), m(x, t, c1)
        gap = (y0 - y1).abs().mean().item()
        assert gap > 1e-5, f"cond_idx を変えても出力が変わらない (平均差 {gap:.2e})"

        # drop_mask を全立てすると無条件経路と厳密に一致すること（CFG の前提）
        drop = torch.ones(x.size(0), dtype=torch.bool)
        y_drop = m(x, t, c1, drop)
        y_none = m(x, t, None)
    assert torch.allclose(y_drop, y_none, atol=1e-6), \
        "drop_mask=True の行が無条件経路と一致しない（CFG が壊れる）"
    print(f"  4. 条件付け (群を変えた平均差 {gap:.4f}) と CFG 無条件経路の一致: OK")


def test_no_ddim_no_ema():
    """★変更2・4: DDIM と EMA を持たないこと。"""
    d = sm.Diffusion(device=DEVICE)
    assert not hasattr(d, "ddim_sample"), "Diffusion に ddim_sample が残っている"
    for name in ["DDIM_STEPS", "DDIM_ETA", "SAMPLER"]:
        assert not hasattr(sm, name), f"{name} が残っている"

    assert not hasattr(sm, "EMA"), "EMA クラスが残っている"
    assert not hasattr(sm, "EMA_DECAY"), "EMA_DECAY が残っている"

    # group_pool / sanity_check / load_pretrained がサンプラ・EMA 引数を持たないこと
    import inspect
    for fn, banned in [(sm.group_pool, ("sampler", "ddim_steps", "eta")),
                       (sm.sanity_check, ("sampler", "ddim_steps")),
                       (sm.load_pretrained, ("use_ema",)),
                       (sm.run_epoch, ("ema",))]:
        params = inspect.signature(fn).parameters
        for b in banned:
            assert b not in params, f"{fn.__name__} に {b} 引数が残っている"

    # チェックポイントの契約: 重みはキー "model"、出所の記録はキー "config"。EMA の重みは持たない
    src = inspect.getsource(sm.train)
    assert '{"model": model.state_dict(),' in src, "保存するチェックポイントの形が変わっている"
    # ★保存する dict のリテラルを丸ごと固定する（EMA の重みなど第3のキーが入ると落ちる）。
    #   '"ema"' の有無では判定できない。wandb の config に "ema": False があるため
    assert '"config": {"kernel_size": KERNEL_SIZE, "clock": clock}}' in src, \
        "チェックポイントの config が変わっている"
    print("  5. DDIM / EMA を持たない: OK")


def test_features():
    """clock_diagnostics が期待する中間特徴の契約。"""
    m = _model()
    x, t, c = _inputs()
    with torch.no_grad():
        f = m.features(x, t, c)
    assert set(f) == {"h1", "h2", "h3"}
    assert f["h1"].shape == (4, sm.BASE_CH, 96)
    assert f["h2"].shape == (4, sm.BASE_CH * 2, 48)
    assert f["h3"].shape == (4, sm.BASE_CH * 2, 24)
    assert m.in_channels == sm.IN_CH
    print("  6. 中間特徴 h1/h2/h3 (96/48/24): OK")


def test_reverse_process():
    """T を短くした ancestral が最後まで走り、正しい範囲のラベルを返すこと。

    本番の T=1000 は重いので、モジュール定数を一時的に差し替えて逆過程の
    ループそのものを検証する（DDIM が無くなった分、ここが唯一の経路になる）。
    """
    orig_t = sm.T_STEPS
    try:
        sm.T_STEPS = 5
        m = _model()
        d = sm.Diffusion(device=DEVICE)
        ci = torch.as_tensor(sm.cond_grid()[:4], dtype=torch.long)
        s = d.sample(m, ci, guidance_scale=sm.GUIDANCE_SCALE)
    finally:
        sm.T_STEPS = orig_t

    assert s.shape == (4, sm.NUM_SLOTS)
    assert int(s.min()) >= 0 and int(s.max()) < sm.NUM_ACT
    print("  7. 逆過程 (T=5 の ancestral): OK")


def test_diff_against_baseline():
    """★DDPM_Aggregate.UNet1D との構造差が time_mlp だけであること。

    自己完結（コピー）なので、共通部分に意図しない差分が混入していないかを
    ここで固定する。バックボーンを比べたときの差の解釈が変わるため。
    """
    agg: Any = _load("agg_baseline", REPO_ROOT / "src/models/DDPM_Aggregate/model.py")

    torch.manual_seed(0)
    a = agg.UNet1D()
    torch.manual_seed(0)
    b = sm.UNet1D()

    keys_a, keys_b = set(a.state_dict()), set(b.state_dict())
    only_a = {k for k in keys_a - keys_b}
    only_b = keys_b - keys_a
    assert only_b == set(), f"原本に無いパラメータが増えている: {sorted(only_b)}"
    assert all(k.startswith("time_mlp.") for k in only_a), \
        f"time_mlp 以外の差分がある: {sorted(k for k in only_a if not k.startswith('time_mlp.'))}"

    n_a = sum(p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    n_mlp = sum(p.numel() for p in a.time_mlp.parameters())
    assert n_a - n_b == n_mlp, f"パラメータ差が time_mlp 分と一致しない ({n_a - n_b} != {n_mlp})"

    # 変更していない定数が一致すること
    for name in ["NUM_SLOTS", "NUM_ACT", "IN_CH", "D_GROUPS", "T_STEPS",
                 "BETA_START", "BETA_END", "BASE_CH", "DROPOUT", "ATTN_HEADS",
                 "TIME_EMB_DIM", "P_UNCOND", "GUIDANCE_SCALE", "BATCH_SIZE",
                 "LR", "VAL_RATIO", "SEED", "EARLY_STOP_PATIENCE", "DAY_FILTER"]:
        assert getattr(sm, name) == getattr(agg, name), \
            f"{name} が原本とずれている: {getattr(sm, name)} != {getattr(agg, name)}"

    print(f"  8. 原本との差分は time_mlp のみ "
          f"({n_a:,} -> {n_b:,}, -{n_mlp:,} params): OK")


def test_clock():
    """★--clock の契約（時計つき UNet1D）。

    (a) φ は 8 行 × 96 スロットで、行どうしが直交する（離散フーリエの直交性 φφᵀ = 48·I）
    (b) 追加されるパラメータは 11 個の clock_proj だけ
    (c) 零初期化の時点では、時計なしのモデルと出力がビット単位で一致する
    (d) clock_proj を動かすと出力が変わり、48 解像度のバイアスは 96 解像度を 2 つおきに
        取ったものと一致する（ds1 の出力位置 j がスロット 2j にあたるという規約）
    (e) 保存して load_pretrained で読むと、時計つきの構造で組み直される
    """
    import tempfile

    # (a) φ の形と直交性
    phi = sm.clock_features()
    assert phi.shape == (sm.CLOCK_DIM, sm.NUM_SLOTS)
    gram = phi @ phi.T
    assert torch.allclose(gram, torch.eye(sm.CLOCK_DIM) * sm.NUM_SLOTS / 2, atol=1e-4), \
        "φ の行が直交していない"

    # (b) 増えるパラメータは clock_proj だけ
    torch.manual_seed(0)
    base = sm.UNet1D().to(DEVICE).eval()
    torch.manual_seed(0)
    clk = sm.UNet1D(clock=True).to(DEVICE).eval()
    only_clk = set(clk.state_dict()) - set(base.state_dict())
    assert only_clk and all(".clock_proj." in k for k in only_clk), sorted(only_clk)
    assert set(base.state_dict()) - set(clk.state_dict()) == set()
    n_blocks = sum(1 for m in clk.modules() if isinstance(m, sm.ResBlock1D))
    assert n_blocks == 11 and len(only_clk) == 2 * n_blocks
    assert not sm.state_has_clock(base.state_dict()) and sm.state_has_clock(clk.state_dict())
    n_extra = sum(p.numel() for p in clk.parameters()) - sum(p.numel() for p in base.parameters())

    # (c) 零初期化の時点で一致。時計なしの重みを時計つきへ流し込み、clock_proj は零のまま
    missing, unexpected = clk.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and set(missing) == only_clk
    _wake_up(base)
    _wake_up(clk)
    x, t, c = _inputs()
    with torch.no_grad():
        y_base, y_clk = base(x, t, c), clk(x, t, c)
    assert torch.equal(y_base, y_clk), "零初期化の時計つきが時計なしと一致しない"

    # (d) clock_proj を動かすと出力が変わる。解像度間の位置の対応
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in clk.named_parameters():
            if ".clock_proj." in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        y_moved = clk(x, t, c)
        b96, b48, b24 = clk.d1a.clock_bias(96), clk.d1a.clock_bias(48), clk.d1a.clock_bias(24)
    assert not torch.equal(y_base, y_moved), "clock_proj を動かしても出力が変わらない"
    assert b96.shape == (1, sm.BASE_CH, 96) and b48.shape == (1, sm.BASE_CH, 48)
    assert torch.allclose(b48, b96[:, :, ::2]) and torch.allclose(b24, b96[:, :, ::4])

    # (e) 保存 -> load_pretrained / build_unet_for_ckpt で時計つきの構造に戻る
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clock.pt"
        torch.save({"model": clk.state_dict(), "config": {"clock": True}}, path)
        loaded = sm.load_pretrained(path).to(DEVICE)
        assert loaded.clock and sm.build_unet_for_ckpt(path).clock
        with torch.no_grad():
            assert torch.equal(loaded(x, t, c), y_moved), "読み直した時計つきの出力が変わった"
        torch.save({"model": base.state_dict()}, path)      # config の無い古い形式
        assert not sm.load_pretrained(path).clock

    print(f"  9. 時計 (φ 直交・零初期化で一致・解像度の対応・読み直し, +{n_extra:,} params): OK")


if __name__ == "__main__":
    print("DDPM_Aggregate_Simple backbone tests")
    test_shapes()
    test_onehot01_encoding()
    test_time_embedding_direct()
    test_conditioning()
    test_no_ddim_no_ema()
    test_features()
    test_reverse_process()
    test_diff_against_baseline()
    test_clock()
    print("test_backbone: OK")
