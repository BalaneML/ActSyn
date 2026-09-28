"""
GRU_Aggregate の Stage 2（重みを更新する版）の単体テスト

    a. rake_weights が、既知の指数傾けで作った目標の行動者率を再現する
    b. 目標が生成の行動者率そのものなら、重みは一様（実効標本数の割合 1）
    c. rehearsal_target は STEP = 0 か gen_g = A* なら gen_1 のまま、gen_g = gen_1 かつ STEP = 1 なら A*。各スロットで和 1
    d. rehearsal_part は held-out 群の系列を入れず、教師群ごとの重みの和が等しい
    e. 微調整 1 回で、リハーサルの重み付き交差エントロピーが下がる（小さなモデル）

    .venv/bin/python src/models/GRU_Aggregate/test_stage2_finetune.py
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


ft: Any = _load("gru_stage2_finetune", Path(__file__).resolve().parent / "stage2_finetune.py")
gm: Any = ft.gm
sm: Any = ft.sm


def _pool(n: int = 400, seed: int = 0) -> Any:
    """群ごとに乱数の系列を並べたプール, (28, n, 96)"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, sm.NUM_ACT, size=(sm.D_GROUPS, n, sm.NUM_SLOTS)).astype(np.int64)


def test_rake_recovers_tilt() -> None:
    """a. 指数傾け w* ∝ exp(Σ_s η[s, x_s]) の重み付きの率を目標にすると、その率が再現される"""
    pool = _pool()
    rng = np.random.default_rng(1)
    eta = rng.normal(scale=0.3, size=(sm.D_GROUPS, sm.NUM_SLOTS, sm.NUM_ACT))
    rows = np.arange(sm.D_GROUPS)[:, None, None]
    w_true = np.exp(eta[rows, np.arange(sm.NUM_SLOTS)[None, None, :], pool].sum(axis=2))
    target = ft.weighted_slot_rates(pool, w_true)
    w = ft.rake_weights(pool, target, iters=200)
    err = float(np.abs(ft.weighted_slot_rates(pool, w) - target).max())
    assert err < 1e-4, f"目標の率が再現されない（最大差 {err:.2e}）"
    print(f"  a. 指数傾けの目標を再現（最大差 {err:.1e}）: OK")


def test_rake_identity() -> None:
    """b. 目標 = 生成の率なら重みは一様"""
    pool = _pool()
    target = ft.weighted_slot_rates(pool, np.ones(pool.shape[:2]))
    w = ft.rake_weights(pool, target)
    ess = ft.ess_fraction(w)
    assert np.allclose(w, 1.0) and np.allclose(ess, 1.0), "傾けなしで重みが一様にならない"
    print("  b. 目標 = 生成の率なら重みは一様: OK")


def test_rehearsal_target() -> None:
    """c. 傾けない場合と、A* まで動かす場合"""
    rng = np.random.default_rng(2)

    def rates() -> Any:
        return rng.dirichlet(np.ones(sm.NUM_ACT), size=(sm.D_GROUPS, sm.NUM_SLOTS)).transpose(0, 2, 1)

    gen_1, gen_g, a_star = rates(), rates(), rates()
    cases = (("STEP = 0", ft.rehearsal_target(gen_1, gen_g, a_star, 0.0), gen_1),
             ("gen_g = A*", ft.rehearsal_target(gen_1, a_star, a_star, 0.5), gen_1),
             ("gen_g = gen_1・STEP = 1", ft.rehearsal_target(gen_1, gen_1, a_star, 1.0), a_star))
    for name, t, want in cases:
        assert np.allclose(t.sum(axis=1), 1.0), f"{name}: スロットごとの和が 1 でない"
        err = float(np.abs(t - want).max())
        assert err < 1e-3, f"{name}: 期待した率にならない（最大差 {err:.2e}）"
    print("  c. rehearsal_target の端点と正規化: OK")


def test_rehearsal_excludes_heldout() -> None:
    """d. held-out 群の系列が入らず、教師群ごとの重みの和が等しい"""
    pool = _pool(n=16)
    w = np.random.default_rng(3).uniform(0.1, 5.0, size=pool.shape[:2])
    teacher = np.ones(sm.D_GROUPS, dtype=bool)
    held = [0, 4, 7, 25]
    teacher[held] = False
    part = ft.rehearsal_part(pool, w, teacher)
    d = sm.cond_to_d(part.cond_idx)
    assert not np.isin(d, held).any(), "held-out 群の系列が入っている"
    sums = np.bincount(d, weights=part.weight, minlength=sm.D_GROUPS)[teacher]
    assert np.allclose(sums, 1.0), "教師群ごとの重みの和が揃っていない"
    assert np.array_equal(part.sched, pool[teacher].reshape(-1, sm.NUM_SLOTS)), "系列と群の対応がずれている"
    print("  d. held-out 群を入れず、群ごとの重みの和が等しい: OK")


def test_finetune_lowers_loss() -> None:
    """e. 1 回の微調整で、同じリハーサルの重み付き交差エントロピーが下がる"""
    torch.manual_seed(0)
    model = gm.GRUScheduler(hidden=16)
    pool = gm.group_pool(model, 8, 1.0, seed=0)
    w = np.random.default_rng(4).uniform(0.1, 5.0, size=pool.shape[:2])
    loader = gm.make_loader(ft.rehearsal_part(pool, w, np.ones(sm.D_GROUPS, dtype=bool)), shuffle=False)
    before = gm.run_epoch(model, loader)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        gm.run_epoch(model, loader, opt)
    after = gm.run_epoch(model, loader)
    assert after < before, f"損失が下がらない（{before:.4f} → {after:.4f}）"
    print(f"  e. 微調整で重み付き交差エントロピーが下がる（{before:.4f} → {after:.4f}）: OK")


if __name__ == "__main__":
    print("GRU_Aggregate Stage 2 finetune tests")
    test_rake_recovers_tilt()
    test_rake_identity()
    test_rehearsal_target()
    test_rehearsal_excludes_heldout()
    test_finetune_lowers_loss()
    print("test_stage2_finetune: OK")
