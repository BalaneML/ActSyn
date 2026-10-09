"""
test_slot_rate_errors.py
========================
slot_rate_errors.py の検証。床の判定は両方向を確かめる:
    - 1 区間だけずらした生成 → その区間が最大誤差・Top-1 になり、床の外
    - 完全なモデルと同じ雑音だけの生成 → 床の内（大半の活動で）
    - 2 段の床の間の誤差 → between
平均誤差・種ごとの表・k:
    - 1 日を通して一定に多い → 平均誤差 ≈ MAE で床の外。山の時刻だけずれる（+ と − が並ぶ）→ 平均誤差 ≈ 0 で MAE だけ大きい
    - seed_error_table は種ごとに最大 |誤差| の時刻を決める。seeds の長さが違えば ValueError
    - k = 5 の上位 3 つは k = 3 と同じ

使い方:
    .venv/bin/python src/eval/test_slot_rate_errors.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sre = _load("slot_rate_errors", HERE / "slot_rate_errors.py")

N_ACT, N_SLOT, N_SEED, N_FLOOR = 12, 96, 5, 200
NOISE = 0.002
ACTS = [f"A{c}" for c in range(N_ACT)]


def _real(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.05, 0.5, size=(N_ACT, N_SLOT))


def _noisy(real: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return real + rng.normal(0.0, NOISE, size=(n, *real.shape))


def _floor_curves(real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """完全なモデルの曲線を、モデルと同じ N_SEED 本の平均で N_FLOOR 回作る, -> (N_FLOOR, 12, 96)"""
    return _noisy(real, N_FLOOR * N_SEED, rng).reshape(N_FLOOR, N_SEED, N_ACT, N_SLOT).mean(axis=1)


def test_slot_label() -> None:
    assert sre.slot_label(0) == "04:00"
    assert sre.slot_label(32) == "12:00"
    assert sre.slot_label(81) == "00:15"          # 04:00 + 20h15m = 翌 00:15


def test_spike_detected() -> None:
    rng = np.random.default_rng(0)
    real = _real(rng)
    floor = sre.floor_errors(real, _floor_curves(real, rng))
    gen = _noisy(real, N_SEED, rng)
    gen[:, 3, 32] -= 0.05                           # 活動 3 の 12:00 を 5pt 低くする
    gen[:, 3, 40] += 0.03                           # 活動 3 の 14:00 を 3pt 高くする
    tab = sre.activity_error_table(real, gen, ACTS, floor, floor)
    row = tab[tab["activity"] == "A3"].iloc[0]
    assert row["max_time"] == "12:00" and abs(row["max_err_pt"] + 5.0) < 0.5 and row["max_position"] == "above_high"
    assert row["err_seed_min_pt"] <= row["max_err_pt"] <= row["err_seed_max_pt"]
    top = sre.topk_error_table(real, gen, ACTS, floor, floor, k=3)
    t3 = top[top["activity"] == "A3"].reset_index(drop=True)
    assert len(t3) == 3 and list(t3["time"][:2]) == ["12:00", "14:00"]
    assert list(t3["position"]) == ["above_high", "above_high", "below_low"]
    assert (np.diff(t3["err_pt"].abs()) <= 0).all()


def test_between_floors() -> None:
    rng = np.random.default_rng(3)
    real = _real(rng)
    low = sre.floor_errors(real, _floor_curves(real, rng))
    high = {k: v * 10.0 for k, v in low.items()}    # 上限側の床を 10 倍に広げる
    gen = _noisy(real, N_SEED, rng)
    gen[:, 0, 50] += 0.02                           # 下限側の床（約 0.3pt）より大きく、上限側（約 3pt）より小さい
    row = sre.activity_error_table(real, gen, ACTS, low, high).iloc[0]
    assert row["max_slot"] == 50 and row["max_position"] == "between"


def test_noise_inside_floor() -> None:
    rng = np.random.default_rng(1)
    real = _real(rng)
    floor = sre.floor_errors(real, _floor_curves(real, rng))
    tab = sre.activity_error_table(real, _noisy(real, N_SEED, rng), ACTS, floor, floor)
    assert int((tab["max_position"] != "below_low").sum()) <= 2   # 95% 分位なので 12 活動で 0〜1 個程度
    assert (tab["mae_pt"] <= tab["mae_floor_low_pt"] * 1.5).all()


def test_nan_cells_skipped() -> None:
    rng = np.random.default_rng(2)
    real = _real(rng)
    real[5, :10] = np.nan
    gen = _noisy(real, N_SEED, rng)
    gen[:, 5, :10] = 0.9                            # NaN のセルは大きくずれていても数えない
    floor = sre.floor_errors(real, _noisy(real, 50, rng))
    tab = sre.activity_error_table(real, gen, ACTS, floor, floor)
    assert int(tab[tab["activity"] == "A5"]["max_slot"].iloc[0]) >= 10
    top = sre.topk_error_table(real, gen, ACTS, floor, floor, k=3)
    assert (top[top["activity"] == "A5"]["slot"] >= 10).all()


def test_bias_vs_mae() -> None:
    rng = np.random.default_rng(4)
    real = _real(rng)
    floor = sre.floor_errors(real, _floor_curves(real, rng))
    assert floor["bias_pt"].shape == (N_ACT,) and (floor["bias_pt"] <= floor["mae_pt"] + 1e-12).all()
    gen = _noisy(real, N_SEED, rng)
    gen[:, 2, :] += 0.01                            # 活動 2 は 1 日を通して 1pt 多い（総量のずれ）
    gen[:, 7, 30] += 0.03                           # 活動 7 は 11:30 に +3pt、11:45 に −3pt（時刻のずれ）
    gen[:, 7, 31] -= 0.03
    tab = sre.activity_error_table(real, gen, ACTS, floor, floor).set_index("activity")
    assert abs(tab.loc["A2", "bias_pt"] - 1.0) < 0.1 and abs(tab.loc["A2", "mae_pt"] - 1.0) < 0.1
    assert tab.loc["A2", "bias_position"] == "above_high"
    assert abs(tab.loc["A7", "bias_pt"]) < 0.05 and tab.loc["A7", "mae_pt"] > 0.05
    assert tab.loc["A7", "max_position"] == "above_high"


def test_seed_table() -> None:
    rng = np.random.default_rng(5)
    real = _real(rng)
    gen = _noisy(real, N_SEED, rng)
    gen[1, 4, 60] += 0.04                           # 種 1 本だけ、活動 4 の 19:00 を 4pt 高くする
    seeds = [42, 43, 44, 45, 46]
    tab = sre.seed_error_table(real, gen, ACTS, seeds)
    assert len(tab) == N_SEED * N_ACT and set(tab["seed"]) == set(seeds)
    row = tab[(tab["seed"] == 43) & (tab["activity"] == "A4")].iloc[0]
    assert row["max_time"] == "19:00" and abs(row["max_err_pt"] - 4.0) < 0.5 and row["max_abs_pt"] > 3.5
    others = tab[(tab["seed"] != 43) & (tab["activity"] == "A4")]
    assert (others["max_abs_pt"] < 1.5).all()
    assert np.allclose(tab["max_abs_pt"], tab["max_err_pt"].abs())
    try:
        sre.seed_error_table(real, gen, ACTS, seeds[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("seeds の長さの違いを見逃した")


def test_topk_k5() -> None:
    rng = np.random.default_rng(6)
    real = _real(rng)
    floor = sre.floor_errors(real, _floor_curves(real, rng))
    gen = _noisy(real, N_SEED, rng)
    top3 = sre.topk_error_table(real, gen, ACTS, floor, floor, k=3)
    top5 = sre.topk_error_table(real, gen, ACTS, floor, floor, k=5)
    assert sre.TOP_K == 3 and len(top5) == 5 * N_ACT
    for name in ACTS:
        t3 = top3[top3["activity"] == name]["slot"].tolist()
        t5 = top5[top5["activity"] == name]["slot"].tolist()
        assert t5[:3] == t3 and len(set(t5)) == 5


if __name__ == "__main__":
    for fn in (test_slot_label, test_spike_detected, test_between_floors, test_noise_inside_floor,
               test_nan_cells_skipped, test_bias_vs_mae, test_seed_table, test_topk_k5):
        fn()
        print(f"ok  {fn.__name__}")
