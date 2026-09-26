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
    9. 時刻符号 (--clock) : φ の直交性、零初期化の時点で時刻符号なしと出力が一致すること、
                     解像度 96/48/24 の位置の対応、保存して読み直したときの構造
   10. 反復 (--seed) : 学習の乱数だけを変え、学習/評価の分割は SEED で固定のまま
   11. 行動者率の項 (--rate-lam) : ★loss() が従来の式と厳密に一致すること（Stage 2 が呼ぶ）、
                     rate_lam=0 の 1 更新が従来のループと一致すること、v(t) の境目、
                     u = −(x̂0 − x0) の換算、eps_hat=eps で 0、Stage 2 が loss_terms を呼ばないこと
   12. 構造の指定 (ArchSpec) : 倍音 K=48 が 96 スロットの全関数を張ること、Transformer 型が
                     timestep_embedding と一致すること、零初期化で時刻符号なしと出力が一致すること、
                     config["arch"] から構造が戻ること、矛盾する指定を弾くこと
   13. 計算ブロック : RoPE は回転 0 で nn.MultiheadAttention と一致し、位置をずらしても出力が不変、
                     96 解像度の attention は attn1 / u1_attn だけを足すこと、
                     条件×時刻のバイアスは零初期化で一致し、学習前から勾配が流れること

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
    assert ('"config": {"kernel_size": KERNEL_SIZE, "clock": arch.has_clock,\n'
            '                               "arch": asdict(arch), "seed": seed,\n'
            '                               "rate_lam": rate_lam, "rate_snr_gamma": rate_gamma}}'
            ) in src, "チェックポイントの config が変わっている"
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
    """★--clock の契約（時刻符号つき UNet1D）。

    (a) φ は 8 行 × 96 スロットで、行どうしが直交する（離散フーリエの直交性 φφᵀ = 48·I）
    (b) 追加されるパラメータは 11 個の clock_proj だけ
    (c) 零初期化の時点では、時刻符号なしのモデルと出力がビット単位で一致する
    (d) clock_proj を動かすと出力が変わり、48 解像度のバイアスは 96 解像度を 2 つおきに
        取ったものと一致する（ds1 の出力位置 j がスロット 2j にあたるという規約）
    (e) 保存して load_pretrained で読むと、時刻符号つきの構造で組み直される
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

    # (c) 零初期化の時点で一致。時刻符号なしの重みを時刻符号つきへ流し込み、clock_proj は零のまま
    missing, unexpected = clk.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and set(missing) == only_clk
    _wake_up(base)
    _wake_up(clk)
    x, t, c = _inputs()
    with torch.no_grad():
        y_base, y_clk = base(x, t, c), clk(x, t, c)
    assert torch.equal(y_base, y_clk), "零初期化の時刻符号つきが時刻符号なしと一致しない"

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

    # (e) 保存 -> load_pretrained / build_unet_for_ckpt で時刻符号つきの構造に戻る
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clock.pt"
        torch.save({"model": clk.state_dict(), "config": {"clock": True}}, path)
        loaded = sm.load_pretrained(path).to(DEVICE)
        assert loaded.clock and sm.build_unet_for_ckpt(path).clock
        with torch.no_grad():
            assert torch.equal(loaded(x, t, c), y_moved), "読み直した時刻符号つきの出力が変わった"
        torch.save({"model": base.state_dict()}, path)      # config の無い古い形式
        assert not sm.load_pretrained(path).clock

    print(f"  9. 時刻符号 (φ 直交・零初期化で一致・解像度の対応・読み直し, +{n_extra:,} params): OK")


def _copy_into(dst: Any, src: Any) -> set[str]:
    """src の重みを dst へ流し込み、dst にだけある（流し込まれなかった）キーを返す。"""
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected, unexpected
    return set(missing)


def test_arch_spec():
    """★ArchSpec の契約（時刻符号の種類と次元、構造の保存と復元）。

    (a) 倍音 K=48 の φ に定数の行を足すと階数 96（96 スロットの全関数を張る）。
        sin(πs) の行は恒等的に 0
    (b) Transformer 型の φ は timestep_embedding(0..95, 96) の転置そのもの
    (c) どちらの種類も零初期化の時点で時刻符号なしと出力がビット単位で一致する
    (d) config["arch"] を持つ ckpt から同じ構造が戻る。config["arch"] が無く
        入力次元が CLOCK_DIM と違う ckpt は構造を決められないので例外
    (e) 矛盾する指定（範囲外の K、時刻符号なしのランク、clock=True と arch の併用）を弾く
    """
    import tempfile
    from dataclasses import asdict

    # (a) 倍音 K=48
    h48 = sm.ArchSpec(clock_kind="harmonic", clock_harmonics=48)
    phi = sm.time_features(h48)
    assert phi.shape == (96, sm.NUM_SLOTS) and h48.clock_dim == 96
    assert torch.allclose(phi[-1], torch.zeros(sm.NUM_SLOTS), atol=1e-4), "sin(πs) の行が 0 でない"
    full = torch.cat([torch.ones(1, sm.NUM_SLOTS), phi], dim=0).double()
    assert int(torch.linalg.matrix_rank(full)) == sm.NUM_SLOTS, "倍音 K=48 が全関数を張らない"

    # (b) Transformer 型
    tf = sm.ArchSpec(clock_kind="transformer")
    phi_tf = sm.time_features(tf)
    ref = sm.timestep_embedding(torch.arange(sm.NUM_SLOTS), sm.CLOCK_TRANSFORMER_DIM).T
    assert phi_tf.shape == (sm.CLOCK_TRANSFORMER_DIM, sm.NUM_SLOTS) and torch.equal(phi_tf, ref)

    # (c) 零初期化で時刻符号なしと一致
    torch.manual_seed(0)
    base = sm.UNet1D().to(DEVICE).eval()
    x, t, c = _inputs()
    for arch in (h48, tf):
        model = sm.UNet1D(arch=arch).to(DEVICE).eval()
        only = _copy_into(model, base)
        assert only and all(".clock_proj." in k for k in only), sorted(only)
        _wake_up(base)
        _wake_up(model)
        with torch.no_grad():
            assert torch.equal(base(x, t, c), model(x, t, c)), f"{arch} が零初期化で一致しない"

    # (d) 保存と復元
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arch.pt"
        for arch in (sm.ArchSpec(), h48, tf):
            model = sm.UNet1D(arch=arch)
            torch.save({"model": model.state_dict(), "config": {"arch": asdict(arch)}}, path)
            assert sm.build_unet_for_ckpt(path).arch == arch
            assert sm.load_pretrained(path).arch == arch
        # config["arch"] の無い K=48 は種類を決められない
        torch.save({"model": sm.UNet1D(arch=h48).state_dict()}, path)
        try:
            sm.build_unet_for_ckpt(path)
        except ValueError:
            pass
        else:
            raise AssertionError("config['arch'] の無い K=48 の ckpt を黙って読んだ")

    # (e) 矛盾する指定を弾く
    bad_specs = [dict(clock_kind="harmonic", clock_harmonics=0),
                 dict(clock_kind="harmonic", clock_harmonics=49),
                 dict(clock_kind="none", cond_clock_rank=4),
                 dict(clock_kind="sundial")]
    for kwargs in bad_specs:
        try:
            sm.ArchSpec(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"矛盾する指定を弾かなかった: {kwargs}")
    try:
        sm.UNet1D(clock=True, arch=h48)
    except ValueError:
        pass
    else:
        raise AssertionError("clock=True と arch の併用を弾かなかった")

    n_h48 = sum(p.numel() for p in sm.UNet1D(arch=h48).parameters())
    n_base = sum(p.numel() for p in base.parameters())
    print(f"  12. ArchSpec (K=48 全基底・Transformer 型・零初期化で一致・保存と復元, "
          f"K=48 は +{n_h48 - n_base:,} params): OK")


def test_blocks():
    """★ArchSpec の計算ブロック（attn_rope / attn96 / cond_clock_rank）の契約。

    (a) RoPE: 位置を全て 0 にすると nn.MultiheadAttention と一致する。全位置を同じ量だけ
        ずらしても出力は変わらない（内積が位置の差だけで決まる）。重みのキーは増えない
    (b) attn96: 増えるキーは attn1 / u1_attn だけ
    (c) cond_clock: 増えるキーは cond_clock_* だけで、零初期化の時点で出力が一致する。
        混ぜる重み a が 0 でも、1 回の逆伝播で a に勾配が流れる（学習が始まる）
    """
    torch.manual_seed(0)
    block = sm.AttnBlock1D(sm.BASE_CH * 2, rope=True).eval()
    h = torch.randn(3, 48, sm.BASE_CH * 2)
    pos = sm.slot_positions(48, "cpu")
    with torch.no_grad():
        ref, _ = block.attn(h, h, h, need_weights=False)
        zero = block.rope_attention(h, torch.zeros_like(pos))
        a0, a7 = block.rope_attention(h, pos), block.rope_attention(h, pos + 7.0)
    assert torch.allclose(zero, ref, atol=1e-5), "回転 0 の RoPE が MultiheadAttention と一致しない"
    assert torch.allclose(a0, a7, atol=1e-4), "位置をずらすと RoPE の出力が変わった（相対位置でない）"
    assert not torch.allclose(a0, zero, atol=1e-4), "RoPE の回転が効いていない"
    assert torch.equal(pos[:3], torch.tensor([0.0, 2.0, 4.0])), "48 解像度の位置がスロット 2j でない"

    h48 = sm.ArchSpec(clock_kind="harmonic", clock_harmonics=48)
    torch.manual_seed(0)
    base = sm.UNet1D(arch=h48).eval()
    x, t, c = _inputs()
    n_base = sum(p.numel() for p in base.parameters())

    rope = sm.UNet1D(arch=sm.ArchSpec(clock_kind="harmonic", clock_harmonics=48, attn_rope=True))
    assert set(rope.state_dict()) == set(base.state_dict()), "RoPE で重みのキーが変わった"

    attn96 = sm.UNet1D(arch=sm.ArchSpec(clock_kind="harmonic", clock_harmonics=48,
                                        attn_rope=True, attn96=True)).eval()
    extra = set(attn96.state_dict()) - set(base.state_dict())
    assert extra and all(k.startswith(("attn1.", "u1_attn.")) for k in extra), sorted(extra)
    with torch.no_grad():
        assert attn96(x, t, c).shape == (x.size(0), sm.IN_CH, sm.NUM_SLOTS)
    n_attn96 = sum(p.numel() for p in attn96.parameters())

    cc = sm.UNet1D(arch=sm.ArchSpec(clock_kind="harmonic", clock_harmonics=48,
                                    cond_clock_rank=4)).eval()
    only = _copy_into(cc, base)
    assert only and all(".cond_clock_" in k for k in only), sorted(only)
    _wake_up(base)
    _wake_up(cc)
    with torch.no_grad():
        assert torch.equal(base(x, t, c), cc(x, t, c)), "零初期化の条件×時刻のバイアスで出力が変わった"
    cc.train()
    cc.zero_grad()
    cc(x, t, c).pow(2).mean().backward()
    grad = cc.d1a.cond_clock_mix.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0, "混ぜる重み a に勾配が流れない"
    n_cc = sum(p.numel() for p in cc.parameters())

    print(f"  13. 計算ブロック (RoPE の一致と相対性・attn96 は +{n_attn96 - n_base:,}・"
          f"条件×時刻 R=4 は +{n_cc - n_base:,} params, 零初期化で一致・勾配が流れる): OK")


def test_seed_keeps_split():
    """--seed は学習の乱数だけを変え、学習/評価の分割（split_indices）は変えないこと。

    ★分割が変わると Stage 2 の val_epsilon_mse と暗記チェックの参照集合が別物になり、
      反復実験の差が「種の差」でなく「データの差」を含んでしまう。
    """
    import inspect
    before = sm.split_indices(1000)
    torch.manual_seed(12345)                       # 学習側の乱数をどう動かしても
    after = sm.split_indices(1000)
    assert all((x == y).all() for x, y in zip(before, after)), "split_indices が大域の乱数に依存している"
    src = inspect.getsource(sm.train)
    assert "torch.manual_seed(seed)" in src and "torch.manual_seed(SEED)" not in src
    assert "manual_seed(SEED)" in inspect.getsource(sm.split_indices)
    print("  10. --seed は学習の乱数だけを変え、分割は SEED で固定: OK")


def _old_loss(diffusion: Any, model: Any, sched: torch.Tensor,
              cond_idx: torch.Tensor) -> torch.Tensor:
    """--rate-lam 導入前の Diffusion.loss を書き写した参照実装（比較専用）。"""
    x0 = sm.sched_to_x0(sched)
    t = torch.randint(0, sm.T_STEPS, (x0.size(0),), device=x0.device)
    eps = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, eps)
    drop_mask = torch.rand(x0.size(0), device=x0.device) < sm.P_UNCOND
    eps_hat = model(x_t, t, cond_idx, drop_mask)
    return torch.nn.functional.mse_loss(eps_hat, eps)


def test_rate_loss():
    """★行動者率の項 L_rate（Diffusion.rate_loss / loss_terms, --rate-lam）。

    Stage 2 は Diffusion.loss をリハーサル項と val に使い、make_loaders で ATUS を読む。
    その 2 つの挙動が 1 ビットも変わっていないことを最初に固定する。
    """
    import inspect
    d = sm.Diffusion(device=DEVICE)
    sched = torch.randint(0, sm.NUM_ACT, (8, sm.NUM_SLOTS), generator=torch.Generator().manual_seed(1))
    cond = torch.as_tensor(sm.cond_grid()[:8], dtype=torch.long)

    # (a) loss() は従来の式と同じ乱数・同じ値。loss_terms の eps も一致する
    model = _model(0).train()
    torch.manual_seed(7)
    ref = _old_loss(d, model, sched, cond)
    torch.manual_seed(7)
    new = d.loss(model, sched, cond)
    torch.manual_seed(7)
    terms = d.loss_terms(model, sched, cond)
    assert torch.equal(ref, new), f"loss() が従来の式とずれた ({float(ref)} vs {float(new)})"
    assert torch.equal(ref, terms["eps"]), "loss_terms の eps が loss() と一致しない"

    # (b) rate_lam=0 の 1 更新は、従来の学習ループ（loss().backward()）と同じ重みになる
    ds = sm.ScheduleDataset(cond, sched)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    m_old, m_new = _model(3).train(), _model(3).train()
    o_old = torch.optim.AdamW(m_old.parameters(), lr=sm.LR, weight_decay=0.0)
    o_new = torch.optim.AdamW(m_new.parameters(), lr=sm.LR, weight_decay=0.0)
    torch.manual_seed(11)
    for c_b, s_b in loader:
        o_old.zero_grad()
        _old_loss(d, m_old, s_b, c_b).backward()
        o_old.step()
    torch.manual_seed(11)
    # run_epoch はバッチをモジュール変数 DEVICE へ送るので、テストの間だけ CPU に揃える
    saved_device, sm.DEVICE = sm.DEVICE, DEVICE
    try:
        sm.run_epoch(m_new, d, loader, o_new, rate_lam=0.0)
    finally:
        sm.DEVICE = saved_device
    for (name, p_old), p_new in zip(m_old.named_parameters(), m_new.parameters()):
        assert torch.equal(p_old, p_new), f"rate_lam=0 の更新が従来とずれた: {name}"

    # (c) v(t) の境目: SNR >= γ の t は 1/√SNR、それより大きい t は 1/√γ
    snr = d.acp / (1.0 - d.acp)
    for gamma in (1.0, 0.01):
        dg = sm.Diffusion(device=DEVICE, rate_gamma=gamma)
        low, high = snr >= gamma, snr < gamma
        assert torch.allclose(dg.rate_v[low], snr[low].rsqrt())
        assert torch.allclose(dg.rate_v[high], torch.full_like(dg.rate_v[high], gamma ** -0.5))
    assert int(torch.nonzero(snr >= 1.0).max()) == 258, "γ=1 の境目が t=258 でない"

    # (d) SNR >= γ の t では u = v·(eps_hat − eps) が −(x̂0 − x0) に一致する（x0 空間の残差）
    g = torch.Generator().manual_seed(5)
    x0 = sm.sched_to_x0(sched)
    eps = torch.randn(x0.shape, generator=g)
    eps_hat = eps + 0.1 * torch.randn(x0.shape, generator=g)
    t = torch.full((8,), 100, dtype=torch.long)
    x_t = d.q_sample(x0, t, eps)
    x0_hat = (x_t - d.sqrt_1m_acp[t][:, None, None] * eps_hat) / d.sqrt_acp[t][:, None, None]
    u = d.rate_v[t][:, None, None] * (eps_hat - eps)
    assert torch.allclose(u, -(x0_hat - x0), atol=1e-5), "u が x0 空間の残差になっていない"

    # (e) eps_hat = eps なら L_rate も split 推定も 0。偏りを足すと m² に一致する
    rate, split = d.rate_loss(eps, eps, t)
    assert float(rate) == 0.0 and float(split) == 0.0
    bias = torch.full_like(eps, 0.2)
    rate, split = d.rate_loss(eps + bias, eps, t)
    expect = float((d.rate_v[100] * 0.2) ** 2)
    assert abs(float(rate) - expect) < 1e-7 and abs(float(split) - expect) < 1e-7

    # (f) Stage 2 は loss() と既定の make_loaders だけを使う（L_rate も drop_last も入らない）
    assert inspect.signature(sm.make_loaders).parameters["drop_last"].default is False
    assert inspect.signature(sm.Diffusion).parameters["rate_gamma"].default == sm.RATE_SNR_GAMMA
    assert sm.RATE_LAM == 0.0
    s2 = (Path(__file__).resolve().parent / "stage2_finetune.py").read_text(encoding="utf-8")
    assert "loss_terms" not in s2 and "drop_last" not in s2 and "rate_gamma" not in s2, \
        "stage2_finetune が L_rate の経路を使っている"
    print("  11. 行動者率の項 (loss 不変・λ=0 の更新一致・v(t) の境目 t=258・x0 換算・0 の検算): OK")


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
    test_seed_keeps_split()
    test_rate_loss()
    test_arch_spec()
    test_blocks()
    print("test_backbone: OK")
