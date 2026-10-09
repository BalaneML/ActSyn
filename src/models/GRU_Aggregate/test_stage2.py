"""
GRU_Aggregate の Stage 2 の単体テスト（計画書 stage2_plan.md §8）

    a. fit_additive が、既知の足し算の効果で作った log 比から群ごとの δ を復元する（NaN セルを除いても）
    b. δ = 0 の group_pool が、δ を渡さない group_pool と同じ乱数で同じ出力になる
    c. held-out 群の log 比を書き換えても、当てはめた δ が変わらない（held-out が最小二乗に入らない）
    d. CFG で δ を足すのは、条件付き・条件なしの両方の logits に足すのと同じ（g によらず 1 回だけ効く）
    e. to_group_bias の群 d の値が、sm.cond_grid()[d] の属性の係数の和になる

    .venv/bin/python src/models/GRU_Aggregate/test_stage2.py
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s2: Any = _load("gru_stage2", Path(__file__).resolve().parent / "stage2.py")
gm: Any = s2.gm
sm: Any = s2.sm


def _random_shift(seed: int = 0) -> Any:
    """基準の水準を 0 にした乱数の δ"""
    rng = np.random.default_rng(seed)
    z = s2.JapanShift.zeros()
    return s2.JapanShift(base=rng.normal(size=z.base.shape),
                         sex=np.concatenate([np.zeros_like(z.sex[:1]), rng.normal(size=z.sex[1:].shape)]),
                         age=np.concatenate([np.zeros_like(z.age[:1]), rng.normal(size=z.age[1:].shape)]),
                         emp=np.concatenate([np.zeros_like(z.emp[:1]), rng.normal(size=z.emp[1:].shape)]))


def _small_model() -> Any:
    """小さな幅の GRU（out_proj を乱数で埋める）"""
    torch.manual_seed(0)
    m = gm.GRUScheduler(hidden=16)
    with torch.no_grad():
        m.out_proj.weight.copy_(torch.randn(m.out_proj.weight.shape) * 0.5)
        m.cond_null.copy_(torch.randn(16))
    return m.eval()


def test_fit_recovers() -> None:
    """a. 足し算の効果から作った log 比を当てはめると群ごとの δ が戻る"""
    true = _random_shift()
    log_ratio = true.to_group_bias().transpose(0, 2, 1).copy()                 # (28, 12, 96)
    rng = np.random.default_rng(1)
    for c, s in zip(rng.integers(0, sm.NUM_ACT, 40), rng.integers(0, sm.NUM_SLOTS, 40)):
        log_ratio[rng.choice(sm.D_GROUPS, 3, replace=False), c, s] = np.nan    # 非公表セル
    fit = s2.fit_additive(log_ratio, np.ones(sm.D_GROUPS, dtype=bool))
    err = float(np.abs(fit.to_group_bias() - true.to_group_bias()).max())
    assert err < 1e-9, f"群ごとの δ が戻らない（最大差 {err:.2e}）"
    print(f"  a. 足し算の効果の復元（NaN セルあり、最大差 {err:.1e}）: OK")


def test_zero_shift_identity() -> None:
    """b. δ = 0 を渡しても、渡さないときと同じ系列になる"""
    m = _small_model()
    a = gm.group_pool(m, 2, 1.25, seed=3)
    b = gm.group_pool(m, 2, 1.25, seed=3, group_bias=s2.JapanShift.zeros().to_group_bias())
    assert np.array_equal(a, b), "δ = 0 で系列が変わった"
    print("  b. δ = 0 は従来の生成と同じ: OK")


def test_heldout_excluded() -> None:
    """c. held-out 群の log 比を書き換えても当てはめが変わらない"""
    true = _random_shift(2)
    log_ratio = true.to_group_bias().transpose(0, 2, 1).copy()
    use = np.ones(sm.D_GROUPS, dtype=bool)
    held = [0, 4, 7, 25]
    use[held] = False
    fit1 = s2.fit_additive(log_ratio, use)
    log_ratio[held] += 5.0
    fit2 = s2.fit_additive(log_ratio, use)
    assert np.array_equal(fit1.to_group_bias(), fit2.to_group_bias()), "held-out 群が当てはめに入っている"
    # 主効果だけなので、held-out 群の δ も残りの群から戻る
    err = float(np.abs(fit1.to_group_bias() - true.to_group_bias()).max())
    assert err < 1e-9, f"held-out 群の δ が属性の効果から戻らない（最大差 {err:.2e}）"
    print("  c. held-out 群は当てはめに入らず、その δ は属性の効果から決まる: OK")


def test_cfg_once() -> None:
    """d. δ を CFG の後に 1 回足す = 両方の logits に足す（全行で同じ δ なら slot_bias に足すのと同じ）"""
    m = _small_model()
    g = torch.Generator().manual_seed(0)
    delta = torch.randn(sm.NUM_SLOTS, sm.NUM_ACT, generator=g) * 0.5
    cond = torch.as_tensor(sm.cond_grid()[:6], dtype=torch.long)
    extra = delta.expand(cond.size(0), -1, -1).contiguous()
    got = gm.sample(m, cond, 1.25, torch.Generator().manual_seed(5), extra)
    shifted = _small_model()
    with torch.no_grad():
        shifted.slot_bias.add_(delta)                                          # 両方の枝に入る
    want = gm.sample(shifted, cond, 1.25, torch.Generator().manual_seed(5))
    assert torch.equal(got, want), "δ が CFG の両方の枝に足したのと違う"
    print("  d. CFG の後に足す δ = 両方の logits に足す δ: OK")


def test_group_bias_sum() -> None:
    """e. 群 d の δ = base + sex[g] + age[a] + emp[e]"""
    sh = _random_shift(3)
    gb = sh.to_group_bias()
    for d, (g, a, e) in enumerate(sm.cond_grid()):
        want = sh.base + sh.sex[g] + sh.age[a] + sh.emp[e]
        assert np.array_equal(gb[d], want), f"群 {d} の δ が係数の和と違う"
    print("  e. to_group_bias は属性の係数の和: OK")


if __name__ == "__main__":
    print("GRU_Aggregate Stage 2 tests")
    test_fit_recovers()
    test_zero_shift_identity()
    test_heldout_excluded()
    test_cfg_once()
    test_group_bias_sum()
    print("test_stage2: OK")
