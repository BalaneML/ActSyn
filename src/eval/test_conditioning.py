"""
test_conditioning.py
================
conditioning.py の検証。test_feasibility.py と同じ形式（main() に素の assert ＋ 診断 print）。

重要な項目:
    (3) 条件を無視した生成（群ラベルをシャッフル）を**検出できる**こと。
        検出できない指標は「条件付けが効いている」の証拠にならない。
    (4) 群間の差が縮んだ生成（全群に平均的な個票）を separation_ratio が拾うこと。
    (5) conditioning_accuracy が「実データ並み」「条件無視」「型どおり」を
        3方向に区別できること。特に **gen > holdout（型どおり）も欠陥**である。

使い方:
    uv run python src/eval/test_conditioning.py
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


cond = _load("conditioning", HERE / "conditioning.py")

# 合成データの設定: 4群 = gender(2) × employment(2)、活動は5分類
N_ACT = 5
SLEEP, MEALS, WORK, LEISURE, SHOPPING = range(N_ACT)
N_GROUPS = 4
ATTR_GRID = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)  # (gender, employment)
ATTR_NAMES = ["gender", "employment"]
LABELS = [f"g{g},e{e}" for g, e in ATTR_GRID]


def make_person(gender: int, employment: int, rng: np.random.Generator) -> np.ndarray:
    """群の属性で中身が変わる合成個票。employment は WORK、gender は SHOPPING に効く。"""
    row = np.full(96, SLEEP, dtype=np.int64)
    row[24:28] = MEALS                                   # 誰でも昼食
    if employment == 1:
        start = 16 + int(rng.integers(0, 4))
        row[start:start + 32] = WORK                     # 働く人は WORK ブロック
    else:
        row[20:52] = LEISURE
    if gender == 1 and rng.random() < 0.8:
        s = 60 + int(rng.integers(0, 8))
        row[s:s + 4] = SHOPPING                          # 女性群は SHOPPING が多い
    return row


def make_pool(counts: list[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, groups = [], []
    for d, n in enumerate(counts):
        g, e = ATTR_GRID[d]
        for _ in range(n):
            rows.append(make_person(int(g), int(e), rng))
            groups.append(d)
    return np.stack(rows), np.asarray(groups, dtype=np.int64)


def test_group_profiles():
    """(1) 群ごとのシェアと本数。"""
    sched, groups = make_pool([10, 10, 10, 10], seed=0)
    prof, counts = cond.group_profiles(sched, groups, N_ACT, N_GROUPS)
    assert counts.tolist() == [10, 10, 10, 10]
    assert np.allclose(prof.sum(axis=1), 1.0), prof.sum(axis=1)
    # 働く群だけ WORK があり、働かない群には無い
    assert prof[1][WORK] > 0.3 and prof[0][WORK] == 0.0, prof[:, WORK]
    # 0人の群は全0行
    prof2, counts2 = cond.group_profiles(sched[groups == 0], groups[groups == 0],
                                         N_ACT, N_GROUPS)
    assert counts2.tolist() == [10, 0, 0, 0] and prof2[1].sum() == 0.0
    print("  (1) group_profiles: OK")


def test_jsd():
    """(2) JSD の性質（同一で0、排他で ln2）。"""
    p = np.array([0.5, 0.5, 0.0])
    assert abs(cond._jsd(p, p)) < 1e-12
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert abs(cond._jsd(a, b) - np.log(2)) < 1e-12, cond._jsd(a, b)
    assert np.isnan(cond._jsd(np.zeros(3), p))
    print("  (2) _jsd: OK  (同一=0, 排他=ln2, 空=nan)")


def test_profile_comparison():
    """(3) ★条件を無視した生成を検出できること。"""
    real, d_real = make_pool([120] * 4, seed=1)
    good, d_good = make_pool([120] * 4, seed=2)          # 同じ条件付き分布
    # 条件無視: 群ラベルだけ振り直す（中身は同じプール）
    bad_d = np.random.default_rng(3).permutation(d_good)

    ok = cond.group_profile_comparison(real, good, d_real, d_good, N_ACT, N_GROUPS,
                                       LABELS, n_boot=80, seed=0)
    ng = cond.group_profile_comparison(real, good, d_real, bad_d, N_ACT, N_GROUPS,
                                       LABELS, n_boot=80, seed=0)
    assert (ok["verdict"] == "床の内").all(), ok
    assert (ng["verdict"] == "★床の外").all(), ng
    assert (ng["jsd"] > ok["jsd"]).all()
    print(f"  (3) group_profile_comparison: OK  (正しい条件 JSD={ok['jsd'].mean():.4f} / "
          f"条件無視 JSD={ng['jsd'].mean():.4f})")


def test_separation():
    """(4) ★群間の差が縮んだ生成を separation_ratio が拾うこと。"""
    real, d_real = make_pool([120] * 4, seed=1)
    good, d_good = make_pool([120] * 4, seed=2)
    bad_d = np.random.default_rng(3).permutation(d_good)

    s_ok = cond.separation_summary(real, good, d_real, d_good, N_ACT, N_GROUPS)
    s_ng = cond.separation_summary(real, good, d_real, bad_d, N_ACT, N_GROUPS)
    assert 0.9 <= s_ok["separation_ratio"] <= 1.1, s_ok
    assert s_ng["separation_ratio"] < 0.2, s_ng
    assert s_ok["n_pairs"] == 6 and s_ok["n_groups_used"] == 4
    print(f"  (4) separation_summary: OK  (正しい条件 ratio={s_ok['separation_ratio']:.3f} / "
          f"条件無視 ratio={s_ng['separation_ratio']:.3f})")


def test_conditioning_accuracy():
    """(5) ★実データ並み / 条件無視 / 型どおり を区別できること。"""
    tr, d_tr = make_pool([150] * 4, seed=1)
    ho, d_ho = make_pool([60] * 4, seed=4)
    gen, d_gen = make_pool([150] * 4, seed=2)
    bad_d = np.random.default_rng(3).permutation(d_gen)

    ok = cond.conditioning_accuracy(tr, d_tr, ho, d_ho, gen, d_gen,
                                    ATTR_GRID, ATTR_NAMES, N_ACT)
    ng = cond.conditioning_accuracy(tr, d_tr, ho, d_ho, gen, bad_d,
                                    ATTR_GRID, ATTR_NAMES, N_ACT)
    assert (ok["balanced[gen]"] > 0.75).all(), ok
    assert (ok["gen - holdout"].abs() < 0.15).all(), ok
    assert (ng["balanced[gen]"] < 0.65).all(), ng           # 条件無視 -> 当たらない
    assert (ng["gen - holdout"] < -0.2).all(), ng

    # 型どおり（例外の無い生成）は holdout を上回る
    rigid, d_rigid = make_pool([150] * 4, seed=2)
    rigid[:, :] = np.stack([make_person(int(ATTR_GRID[d][0]), int(ATTR_GRID[d][1]),
                                        np.random.default_rng(0)) for d in d_rigid])
    st = cond.conditioning_accuracy(tr, d_tr, ho, d_ho, rigid, d_rigid,
                                    ATTR_GRID, ATTR_NAMES, N_ACT)
    assert (st["balanced[gen]"] >= ok["balanced[gen]"] - 1e-9).all(), st
    print(f"  (5) conditioning_accuracy: OK  (実並み gen={ok['balanced[gen]'].mean():.3f} / "
          f"条件無視 {ng['balanced[gen]'].mean():.3f} / 型どおり {st['balanced[gen]'].mean():.3f})")


def test_attribute_contrast():
    """(6) 属性コントラストの符号と大きさ。"""
    real, d_real = make_pool([120] * 4, seed=1)
    gen, d_gen = make_pool([120] * 4, seed=2)
    act_names = ["SLEEP", "MEALS", "WORK", "LEISURE", "SHOPPING"]
    tab = cond.attribute_contrast(real, gen, d_real, d_gen, ATTR_GRID, 1,
                                  N_ACT, act_names)
    work = tab[tab.activity == "WORK"].iloc[0]
    assert work["real Δ分/日 (1-0)"] > 400, work      # employment=1 は WORK が 8時間増える
    assert 0.8 < work["gen/real"] < 1.2, work
    leisure = tab[tab.activity == "LEISURE"].iloc[0]
    assert leisure["real Δ分/日 (1-0)"] < -400, leisure
    print("  (6) attribute_contrast: OK  "
          f"(WORK Δ real={work['real Δ分/日 (1-0)']:.0f}分 gen={work['gen Δ分/日 (1-0)']:.0f}分)")


def main():
    print("conditioning のテスト")
    test_group_profiles()
    test_jsd()
    test_profile_comparison()
    test_separation()
    test_conditioning_accuracy()
    test_attribute_contrast()
    print("\ntest_conditioning: OK")


if __name__ == "__main__":
    main()
