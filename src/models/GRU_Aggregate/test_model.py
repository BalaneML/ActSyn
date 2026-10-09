"""
GRU_Aggregate の単体テスト（計画書 §7）

    a. 因果性      : スロット s の logits が a_{≥s} に依存しない。step を 96 回回すと forward と同じ logits になる
    b. 損失        : weighted_ce が手で書いた重み付き交差エントロピーと一致する
    c. 総量の仕組み : slot_bias の勾配が Σ_i w_i (p − y) / (96 Σ w) と一致する。
                     学習前（out_proj 零初期化）の予測が init_output_bias に渡した行動者率と一致する
    d. CFG         : 条件を落とした行は cond_idx=None と一致する。g = 1.0 の CFG は条件付きの logits と一致する
    e. 生成        : 同じ種で再現する。値域は [0, 12)。draw_categorical の頻度が確率に合う。
                     write_pool_csv → load_sample_pool で往復する
    f. teacher forcing の行動者率 : teacher_forced_rates が手で計算した加重平均と一致する（分割して通しても同じ）
    g. 保存と読み込み : 同じ構造・同じ出力が戻る。本番の幅（H = 128）でパラメータ数が 217,288
    h. slot_bias の補正（§9） : 履歴に依らない分布なら bias_update の 1 回で目標に一致する。
                     generated_rates が手計算と一致する。calibrate_slot_bias は slot_bias だけを変える
    i. weight decay の範囲（§10） : param_groups は slot_bias と 1 次元のパラメータに weight decay を掛けない。
                     weight decay だけを効かせた 1 step で、slot_bias は動かず重み行列は縮む
    j. slot_bias なしの構造（§12） : slot_bias は 0 の buffer で学習しない。out_proj.bias の勾配が
                     Σ_i w_i Σ_s (p − y) / (96 Σ w) と一致する。学習前の予測は全スロットで 1 日平均の活動シェア。
                     step と forward が一致する。保存と読み込みで構造が戻る。補正と補正の保存先は ValueError

★ out_proj は零初期化なので、そのままでは logits が slot_bias だけになり a・d が何も測れない。
  _wake_up で out_proj を小さな乱数で埋めてから測る（テスト用の細工で、学習経路は変えない）。

    .venv/bin/python src/models/GRU_Aggregate/test_model.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm: Any = _load("gru_aggregate_model", Path(__file__).resolve().parent / "model.py")
cur: Any = _load("gru_test_stage2_curves",
                 REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage2_curves.py")
DEVICE = "cpu"          # 決定性のため CPU 固定
SMALL = 32              # テスト用の幅


def _model(seed: int = 0) -> Any:
    """小さな幅のモデル。out_proj と slot_bias を乱数で埋める（_wake_up）"""
    torch.manual_seed(seed)
    m = gm.GRUScheduler(hidden=SMALL).to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.out_proj.weight.copy_(torch.randn(m.out_proj.weight.shape, generator=g) * 0.5)
        m.slot_bias.copy_(torch.randn(m.slot_bias.shape, generator=g))
    return m.eval()


def _inputs(batch: int = 6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(cond_idx (B, 3), sched (B, 96), weight (B,)) の乱数の入力"""
    g = torch.Generator().manual_seed(seed)
    grid = torch.as_tensor(gm.sm.cond_grid(), dtype=torch.long)
    cond = grid[torch.randint(0, gm.D_GROUPS, (batch,), generator=g)]
    sched = torch.randint(0, gm.NUM_ACT, (batch, gm.NUM_SLOTS), generator=g)
    weight = torch.rand(batch, generator=g) * 5000 + 100
    return cond, sched, weight


def test_causality() -> None:
    """a. スロット s の logits は a_{≥s} に依存しない。step の繰り返しは forward と一致する"""
    m = _model()
    cond, sched, _ = _inputs()
    with torch.no_grad():
        logits = m(gm.shift_right(sched), cond)
        k = 40
        changed = sched.clone()
        changed[:, k:] = (sched[:, k:] + 1) % gm.NUM_ACT
        logits2 = m(gm.shift_right(changed), cond)
        assert torch.equal(logits[:, :k + 1], logits2[:, :k + 1]), "未来の活動が logits に漏れている"
        assert not torch.allclose(logits[:, k + 1], logits2[:, k + 1]), "直前の活動が届いていない"

        c_emb = m.embed_cond(cond, cond.size(0))
        a_prev = gm.shift_right(sched)
        state = None
        for s in range(gm.NUM_SLOTS):
            ls, state = m.step(a_prev[:, s], s, c_emb, state)
            assert torch.allclose(ls, logits[:, s], atol=1e-5), f"step と forward がスロット {s} で違う"
    print("  a. 因果性・step と forward の一致: OK")


def test_loss() -> None:
    """b. weighted_ce が手で書いた重み付き交差エントロピーと一致する"""
    m = _model()
    cond, sched, weight = _inputs()
    with torch.no_grad():
        logits = m(gm.shift_right(sched), cond)
        logp = F.log_softmax(logits, dim=-1).gather(-1, sched[..., None])[..., 0]    # (B, 96)
        manual = -(weight[:, None] * logp).sum() / (gm.NUM_SLOTS * weight.sum())
        got = gm.weighted_ce(logits, sched, weight)
        assert torch.allclose(got, manual, rtol=1e-6), f"{float(got)} != {float(manual)}"
        assert torch.equal(gm.batch_loss(m, cond, sched, weight), got)
        # 重みの定数倍で不変（正規化が Σw で入っている）
        assert torch.allclose(gm.weighted_ce(logits, sched, weight * 7.0), got, rtol=1e-6)
    print(f"  b. 重み付き交差エントロピー ({float(got):.4f} nats): OK")


def test_slot_bias_gradient() -> None:
    """c. ∂L/∂slot_bias = Σ_i w_i (p − y) / (96 Σ w)。学習前の予測は行動者率"""
    m = _model()
    cond, sched, weight = _inputs()
    loss = gm.batch_loss(m, cond, sched, weight)
    loss.backward()
    with torch.no_grad():
        p = torch.softmax(m(gm.shift_right(sched), cond), dim=-1)                   # (B, 96, 12)
        y = F.one_hot(sched, gm.NUM_ACT).float()
        expected = (weight[:, None, None] * (p - y)).sum(0) / (gm.NUM_SLOTS * weight.sum())
    assert m.slot_bias.grad is not None
    err = float((m.slot_bias.grad - expected).abs().max())
    assert err < 1e-7, f"slot_bias の勾配が式と違う (最大差 {err:.2e})"

    # 学習前: out_proj が零なので softmax(slot_bias) = 行動者率（和が 1 なら log の往復で戻る）
    fresh = gm.GRUScheduler(hidden=SMALL)
    rates = gm.sm.population_rates(sched.numpy(), weight.double().numpy())          # (12, 96)
    fresh.init_output_bias(rates)
    tf = gm.teacher_forced_rates(fresh, sched.numpy(), cond.numpy(), weight.double().numpy())
    floor = rates.clamp_min(gm.RATE_FLOOR)
    target = (floor / floor.sum(0, keepdim=True)).double().numpy()
    assert np.allclose(tf, target, atol=1e-6), "学習前の予測が行動者率と一致しない"
    print(f"  c. slot_bias の勾配 (最大差 {err:.1e})・学習前の予測 = 行動者率: OK")


def test_cfg() -> None:
    """d. 条件を落とした行は cond_idx=None と一致する。g = 1.0 は条件付きの logits"""
    m = _model()
    with torch.no_grad():
        m.cond_null.copy_(torch.randn(SMALL))          # 零のままだと区別の検査が甘くなる
    cond, sched, _ = _inputs()
    drop = torch.tensor([True, False, True, False, False, True])
    a_prev = gm.shift_right(sched)
    with torch.no_grad():
        l_drop = m(a_prev, cond, drop)
        l_none = m(a_prev, None)
        l_cond = m(a_prev, cond)
    assert torch.equal(l_drop[drop], l_none[drop]), "落とした行が条件なしと一致しない"
    assert torch.equal(l_drop[~drop], l_cond[~drop]), "落としていない行が条件付きと一致しない"
    assert not torch.allclose(l_cond[drop], l_none[drop]), "条件が logits に届いていない"
    # l_u + 1·(l_c − l_u) は丸めで最下位ビットが動く。sample は g = 1.0 で条件なしの経路を計算しない
    assert torch.allclose(gm.guided_logits(l_cond, l_none, 1.0), l_cond, atol=1e-6)
    assert torch.allclose(gm.guided_logits(l_cond, l_none, 0.0), l_none, atol=1e-6)

    # 生成: g ≠ 1 では条件なしの経路を別の隠れ状態で回す。g = 1 と違う系列になる
    g1 = gm.sample(m, cond, 1.0, torch.Generator().manual_seed(0))
    g2 = gm.sample(m, cond, 1.25, torch.Generator().manual_seed(0))
    assert not torch.equal(g1, g2), "g = 1.25 が g = 1.0 と同じ系列を返した"
    print("  d. 条件の落とし方・logits の CFG: OK")


def test_sampling() -> None:
    """e. 再現性・値域・引く頻度・CSV の往復"""
    m = _model()
    cond, _, _ = _inputs(batch=8)
    a = gm.sample(m, cond, 1.0, torch.Generator().manual_seed(3))
    b = gm.sample(m, cond, 1.0, torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "同じ種で生成が再現しない"
    assert a.dtype == torch.long and a.shape == (8, gm.NUM_SLOTS)
    assert int(a.min()) >= 0 and int(a.max()) < gm.NUM_ACT

    # 逆関数法の頻度が確率に合う（2 万回で ±0.01）
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05] + [0.0] * (gm.NUM_ACT - 4))
    logits = probs.clamp_min(1e-12).log().expand(20000, -1)
    draw = gm.draw_categorical(logits, torch.Generator().manual_seed(0))
    freq = torch.bincount(draw, minlength=gm.NUM_ACT).double() / len(draw)
    assert float((freq - probs.double()).abs().max()) < 0.01, f"引く頻度が確率と違う: {freq[:4]}"

    pool = gm.group_pool(m, 2, 1.0, seed=5)
    assert np.array_equal(pool, gm.group_pool(m, 2, 1.0, seed=5))
    assert pool.shape == (gm.D_GROUPS, 2, gm.NUM_SLOTS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pool.csv"
        gm.sm.write_pool_csv(pool, path)
        assert np.array_equal(cur.load_sample_pool(path), pool), "CSV の往復で壊れた"
    print("  e. 生成の再現性・値域・頻度・CSV の往復: OK")


def test_teacher_forced_rates() -> None:
    """f. teacher_forced_rates = Σ_i w_i softmax(logits_i) / Σ w（分割して通しても同じ）"""
    m = _model()
    cond, sched, weight = _inputs(batch=7)
    with torch.no_grad():
        p = torch.softmax(m(gm.shift_right(sched), cond), dim=-1).double()           # (B, 96, 12)
    w = weight.double()
    manual = ((w[:, None, None] * p).sum(0) / w.sum()).T.numpy()                     # (12, 96)
    got = gm.teacher_forced_rates(m, sched.numpy(), cond.numpy(), w.numpy(), batch=3)
    assert got.shape == (gm.NUM_ACT, gm.NUM_SLOTS)
    assert np.allclose(got, manual, atol=1e-6), "teacher_forced_rates が手計算と違う"
    assert np.allclose(got.sum(axis=0), 1.0)
    print("  f. teacher forcing の行動者率: OK")


def test_save_load() -> None:
    """g. 保存した ckpt から同じ構造・同じ出力が戻る"""
    m = _model()
    cond, sched, _ = _inputs()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gru.pt"
        gm.save_ckpt(m, path, seed=7, epoch=3)
        ckpt = torch.load(path, map_location="cpu")
        assert set(ckpt) == {"model", "config"}
        assert ckpt["config"]["epoch"] == 3 and ckpt["config"]["seed"] == 7
        assert "phi" not in ckpt["model"], "φ は保存しない buffer のはず"
        back = gm.load_model(path, DEVICE)
    assert back.hidden == SMALL and back.num_layers == gm.NUM_LAYERS and not back.training
    with torch.no_grad():
        assert torch.equal(back(gm.shift_right(sched), cond), m(gm.shift_right(sched), cond))
    assert gm.count_params(gm.GRUScheduler()) == 217_288          # H = 128（計画書 §10）
    assert gm.ckpt_path(42).name == "gru_aggregate.pt" and gm.ckpt_path(43).name == "gru_aggregate_s43.pt"
    assert gm.pool_path(42, 1.25).name == "gru_aggregate_samples_g1.25.csv"
    assert gm.ckpt_path(43, calib_guidance=1.0).name == "gru_aggregate_s43_cal.pt"
    assert gm.pool_path(42, calib_guidance=1.0).name == "gru_aggregate_samples_cal.csv"
    assert gm.ckpt_path(43, calib_guidance=1.25).name == "gru_aggregate_s43_calg1.25.pt"
    assert gm.pool_path(42, 1.25, calib_guidance=1.25).name == "gru_aggregate_samples_calg1.25_g1.25.csv"
    assert gm.pool_path(42, 1.25, calib_guidance=1.0).name == "gru_aggregate_samples_cal_g1.25.csv"
    print("  g. 保存と読み込み・パラメータ数・保存先: OK")


def test_calibration() -> None:
    """h. bias_update の 1 回で目標に一致する（履歴に依らない分布）。補正は slot_bias だけを変える"""
    g = torch.Generator().manual_seed(0)
    bias = torch.randn(gm.NUM_SLOTS, gm.NUM_ACT, generator=g, dtype=torch.float64)
    gen = torch.softmax(bias, dim=-1).T.numpy()                                   # (12, 96)
    target = torch.softmax(torch.randn(gm.NUM_SLOTS, gm.NUM_ACT, generator=g, dtype=torch.float64),
                           dim=-1).T.numpy()
    new = torch.softmax(bias + torch.as_tensor(gm.bias_update(gen, target, step=1.0, eps=0.0)), dim=-1).T
    assert np.allclose(new.numpy(), target, atol=1e-12), "1 回の更新で目標に一致しない"

    rng = np.random.default_rng(0)
    pool = rng.integers(0, gm.NUM_ACT, size=(gm.D_GROUPS, 5, gm.NUM_SLOTS))
    pi = rng.random(gm.D_GROUPS)
    pi /= pi.sum()
    manual = np.zeros((gm.NUM_ACT, gm.NUM_SLOTS))
    for d in range(gm.D_GROUPS):
        for c in range(gm.NUM_ACT):
            manual[c] += pi[d] * (pool[d] == c).mean(axis=0)
    assert np.allclose(gm.generated_rates(pool, pi), manual), "generated_rates が手計算と違う"

    m = _model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    hist = gm.calibrate_slot_bias(m, manual, pi, 1.25, iters=2, n_per_group=2, seed=0, verbose=False)
    after = m.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == {"slot_bias"}, f"slot_bias 以外が変わった: {changed}"
    assert len(hist) == 3 and [h["iter"] for h in hist] == [0, 1, 2]
    print("  h. slot_bias の補正（1 回で目標に一致・slot_bias だけを変える）: OK")


def test_param_groups() -> None:
    """i. weight decay は重み行列だけに掛かり、slot_bias と 1 次元のパラメータには掛からない"""
    m = _model()
    decay, no_decay = gm.param_groups(m, 0.5)
    assert decay["weight_decay"] == 0.5 and no_decay["weight_decay"] == 0.0
    ids = {name: id(p) for name, p in m.named_parameters()}
    decay_ids = {id(p) for p in decay["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert decay_ids.isdisjoint(no_decay_ids) and decay_ids | no_decay_ids == set(ids.values()), "漏れか重複がある"
    assert ids["slot_bias"] in no_decay_ids, "slot_bias に weight decay が掛かっている"
    assert ids["cond_null"] in no_decay_ids and ids["gru.bias_hh_l0"] in no_decay_ids
    assert ids["out_proj.weight"] in decay_ids and ids["act_embed.weight"] in decay_ids

    # 勾配 0 で 1 step: AdamW の更新は weight decay だけになる
    opt = torch.optim.AdamW(gm.param_groups(m, 0.5), lr=0.1)
    before = {k: v.detach().clone() for k, v in m.named_parameters()}
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    after = dict(m.named_parameters())
    assert torch.equal(after["slot_bias"], before["slot_bias"]), "weight decay で slot_bias が動いた"
    assert torch.allclose(after["out_proj.weight"], before["out_proj.weight"] * (1 - 0.1 * 0.5))
    print("  i. weight decay の範囲（slot_bias と 1 次元には掛けない）: OK")


def test_no_slot_bias() -> None:
    """j. slot_bias なしの構造: 0 の buffer・out_proj.bias の勾配・学習前の予測・step・保存・補正の禁止"""
    torch.manual_seed(0)
    m = gm.GRUScheduler(hidden=SMALL, use_slot_bias=False)
    names = {name for name, _ in m.named_parameters()}
    assert "slot_bias" not in names and "out_proj.bias" in names, f"パラメータの構成が違う: {sorted(names)}"
    assert torch.equal(m.slot_bias, torch.zeros(gm.NUM_SLOTS, gm.NUM_ACT))
    assert gm.count_params(gm.GRUScheduler(use_slot_bias=False)) == 217_288 - gm.NUM_SLOTS * gm.NUM_ACT + gm.NUM_ACT
    no_decay = {id(p) for p in gm.param_groups(m, 0.5)[1]["params"]}
    assert m.out_proj.bias is not None and id(m.out_proj.bias) in no_decay, "out_proj.bias に weight decay が掛かる"

    # 学習前: out_proj.weight が零なので、全スロットで softmax(out_proj.bias) = 1 日平均の活動シェア
    cond, sched, weight = _inputs()
    rates = gm.sm.population_rates(sched.numpy(), weight.double().numpy())          # (12, 96)
    m.init_output_bias(rates)
    tf = gm.teacher_forced_rates(m, sched.numpy(), cond.numpy(), weight.double().numpy())
    share = rates.mean(dim=1).clamp_min(gm.RATE_FLOOR)
    share = (share / share.sum()).double().numpy()
    assert np.allclose(tf, np.repeat(share[:, None], gm.NUM_SLOTS, axis=1), atol=1e-6), "学習前の予測がシェアと違う"

    # ∂L/∂out_proj.bias = Σ_i w_i Σ_s (p − y) / (96 Σ w)（スロットの和だけを縛る）
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        m.out_proj.weight.copy_(torch.randn(m.out_proj.weight.shape, generator=g) * 0.5)
    m.train(False)
    loss = gm.batch_loss(m, cond, sched, weight)
    loss.backward()
    with torch.no_grad():
        p = torch.softmax(m(gm.shift_right(sched), cond), dim=-1)                   # (B, 96, 12)
        y = F.one_hot(sched, gm.NUM_ACT).float()
        expected = (weight[:, None, None] * (p - y)).sum(dim=(0, 1)) / (gm.NUM_SLOTS * weight.sum())
    assert m.out_proj.bias.grad is not None
    err = float((m.out_proj.bias.grad - expected).abs().max())
    assert err < 1e-7, f"out_proj.bias の勾配が式と違う (最大差 {err:.2e})"

    # step の繰り返しは forward と一致する
    with torch.no_grad():
        logits = m(gm.shift_right(sched), cond)
        c_emb = m.embed_cond(cond, cond.size(0))
        a_prev = gm.shift_right(sched)
        state = None
        for s in range(gm.NUM_SLOTS):
            ls, state = m.step(a_prev[:, s], s, c_emb, state)
            assert torch.allclose(ls, logits[:, s], atol=1e-5), f"step と forward がスロット {s} で違う"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gru.pt"
        gm.save_ckpt(m, path, seed=7)
        back = gm.load_model(path, DEVICE)
    assert not back.use_slot_bias and "slot_bias" not in {n for n, _ in back.named_parameters()}
    with torch.no_grad():
        assert torch.equal(back(gm.shift_right(sched), cond), logits)

    assert gm.ckpt_path(42, use_slot_bias=False).name == "gru_aggregate_no_slot_bias.pt"
    assert gm.ckpt_path(43, use_slot_bias=False).name == "gru_aggregate_no_slot_bias_s43.pt"
    assert gm.pool_path(43, 1.25, use_slot_bias=False).name == "gru_aggregate_samples_no_slot_bias_s43_g1.25.csv"
    for call in (lambda: gm.ckpt_path(42, calib_guidance=1.0, use_slot_bias=False),
                 lambda: gm.calibrate_slot_bias(m, rates.double().numpy(), np.full(gm.D_GROUPS, 1 / gm.D_GROUPS),
                                                iters=1, n_per_group=1, verbose=False)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("slot_bias なしの構造で補正が通った")
    print(f"  j. slot_bias なしの構造（out_proj.bias の勾配 最大差 {err:.1e}）: OK")


if __name__ == "__main__":
    print("GRU_Aggregate tests")
    test_causality()
    test_loss()
    test_slot_bias_gradient()
    test_cfg()
    test_sampling()
    test_teacher_forced_rates()
    test_save_load()
    test_calibration()
    test_param_groups()
    test_no_slot_bias()
    print("test_model: OK")
