"""
SA（比較 a）の単体テスト

    a. グループの数え方: 円環で数える・途中に入ってよい活動を含める・主の活動を含まない区間は数えない
    a2. 局所的な差分: スロット 1 つを変えたときのグループ数の増減が、全部数え直した差と一致する
    b. 人数の丸め: 各スロットの和がちょうど N で、厳密な人数との差が 1 未満
    c. 近傍は時刻別行動者率を保つ: SA の後も、スロットごとの活動の人数が初期解と同じ
    d. 差分の更新が正しい: 履歴の最良の (f1, f2, f3) が、最良解から最初から計算し直した値と一致する
    e. 再現性: 同じ種なら同じ解、違う種なら違う解
    f. 入力: 20 分類を 12 分類にまとめると既存の A* と一致する（和を 1 に揃えた分だけずれる）

    .venv/bin/python src/models/SA_Aggregate/test_sa.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sa_core as core  # noqa: E402
import sa_run as run  # noqa: E402
import sa_targets as tg  # noqa: E402

R_SLEEP, R_COMMUTE, R_WORK = 0, 1, 2


def _row(segments: list[tuple[int, int]]) -> np.ndarray:
    """[(活動, スロット数), ...] を 96 スロットの 1 日にする"""
    row = np.concatenate([np.full(n, a, dtype=np.int8) for a, n in segments])
    assert row.shape[0] == 96
    return row


def _groups(row: np.ndarray, r: int) -> int:
    return core.count_groups(row, core.RULE_MEMBER[r], int(core.RULE_MAIN[r]))


def test_count_groups() -> None:
    # 04:00 始まりの 1 日：朝まで睡眠 → 起きる → 夜にまた睡眠。円環なので 1 グループ
    row = _row([(core.SLEEP, 10), (core.REST, 70), (core.SLEEP, 16)])
    assert _groups(row, R_SLEEP) == 1
    # 昼寝があると 2 グループ → 違反 1
    row = _row([(core.SLEEP, 10), (core.REST, 30), (core.SLEEP, 4), (core.REST, 36), (core.SLEEP, 16)])
    assert _groups(row, R_SLEEP) == 2
    # 通勤の途中の移動はグループを割らない（R2 に移動を入れた）
    row = _row([(core.SLEEP, 12), (core.COMMUTE, 2), (core.TRAVEL, 1), (core.COMMUTE, 2), (core.WORK, 40),
                (core.COMMUTE, 4), (core.REST, 35)])
    assert _groups(row, R_COMMUTE) == 2
    # 仕事の途中の食事はグループを割らない。休養も途中に入ってよい
    row = _row([(core.SLEEP, 12), (core.WORK, 20), (core.MEAL, 4), (core.WORK, 20), (core.REST, 40)])
    assert _groups(row, R_WORK) == 1
    # 途中に入ってよい活動だけの区間は数えない（仕事を含まない食事）
    row = _row([(core.SLEEP, 12), (core.MEAL, 4), (core.SLEEP, 2), (core.WORK, 20), (core.SLEEP, 58)])
    assert _groups(row, R_WORK) == 1
    # 1 日中睡眠なら 1 グループ、その活動が無ければ 0
    assert _groups(np.zeros(96, dtype=np.int8), R_SLEEP) == 1
    assert _groups(np.zeros(96, dtype=np.int8), R_WORK) == 0
    # 違反数は上限を超えた分だけ数える（通勤 3 回 → 1、仕事 0 回 → 0）
    row = _row([(core.SLEEP, 12), (core.COMMUTE, 2), (core.REST, 10), (core.COMMUTE, 2), (core.REST, 10),
                (core.COMMUTE, 2), (core.REST, 58)])
    assert core.person_violation(row, core.RULE_MEMBER, core.RULE_MAIN, core.RULE_CAP) == 1
    print("a. グループの数え方: OK")


def test_group_delta() -> None:
    """a2. 局所的な差分 group_delta が、全部数え直した差と一致する"""
    rng = np.random.default_rng(0)
    n_checked = 0
    for trial in range(4000):
        # ランダムな 1 日だと区間が短すぎるので、少数の活動からなる長い区間も混ぜる
        if trial % 2 == 0:
            row = rng.integers(0, 20, 96).astype(np.int8)
        else:
            acts = rng.choice([core.SLEEP, core.COMMUTE, core.WORK, core.MEAL, core.TRAVEL, core.REST], 6)
            row = np.repeat(acts, 16).astype(np.int8)
            flip = rng.integers(0, 96, 8)
            row[flip] = rng.integers(0, 20, 8)
        for _ in range(10):
            t = int(rng.integers(0, 96))
            b = int(rng.integers(0, 20))
            for r in range(len(core.RULE_MAIN)):
                member, main = core.RULE_MEMBER[r], int(core.RULE_MAIN[r])
                before = core.count_groups(row, member, main)
                new = row.copy()
                new[t] = b
                assert core.group_delta(row, t, b, member, main) == core.count_groups(new, member, main) - before
                n_checked += 1
    # 1 日中ほぼ睡眠（t 以外がすべて member）の場合
    row = np.zeros(96, dtype=np.int8)
    for b in range(20):
        new = row.copy()
        new[7] = b
        assert core.group_delta(row, 7, b, core.RULE_MEMBER[0], 0) == core.count_groups(new, core.RULE_MEMBER[0], 0) - 1
    print(f"a2. 局所的な差分: OK（{n_checked:,} 通り）")


def test_round_slot_counts() -> None:
    rates = tg.load_rates20()
    for d in (0, 5, 27):
        for n in (37, 2000):
            c = tg.round_slot_counts(rates[d], n)
            assert (c.sum(axis=0) == n).all()
            assert np.abs(c - rates[d] * n).max() < 1.0
    print("b. 人数の丸め: OK")


def _solve(variant: str, seed: int, d: int = 5, n: int = 60, k: int = 3) -> tuple[np.ndarray, np.ndarray,
                                                                               np.ndarray, np.ndarray]:
    rates = tg.load_rates20()
    part = tg.load_participation20()
    counts = tg.round_slot_counts(rates[d], n)
    X0 = run.initial_solution(counts, np.random.default_rng(seed))
    sna = np.round(part[d] * n).astype(np.int64)
    best, hist = core.anneal(X0.copy(), sna, core.F3_WEIGHT, core.RULE_MEMBER, core.RULE_MAIN, core.RULE_CAP,
                             core.RULE_RELEVANT, variant == "rules", k, run.T_INIT, run.T_CONV,
                             run.COOLING_RATE, seed)
    return X0, best, hist, sna


def test_anneal() -> None:
    for variant in ("rules", "norules"):
        use_rules = variant == "rules"
        X0, best, hist, sna = _solve(variant, seed=1)
        # c. スロットごとの人数が変わらない
        for s in range(96):
            assert np.array_equal(np.bincount(best[:, s], minlength=20), np.bincount(X0[:, s], minlength=20))
        # d. 差分で積んだ最良の値が、最初から計算した値と一致する
        f = core.objectives(best, sna, core.F3_WEIGHT, core.RULE_MEMBER, core.RULE_MAIN, core.RULE_CAP, use_rules)
        assert tuple(int(x) for x in hist[-1, 4:7]) == f, (variant, hist[-1, 4:7], f)
        f0 = core.objectives(X0, sna, core.F3_WEIGHT, core.RULE_MEMBER, core.RULE_MAIN, core.RULE_CAP, use_rules)
        assert f <= f0                                    # タプルの比較 = 辞書式
        if not use_rules:
            assert f[0] == 0
        # e. 再現性
        _, best2, _, _ = _solve(variant, seed=1)
        _, best3, _, _ = _solve(variant, seed=2)
        assert np.array_equal(best, best2)
        assert not np.array_equal(best, best3)
        print(f"c〜e. anneal（{variant}）: OK  初期 {f0} → 最良 {f}")


def test_inputs_match_a_star() -> None:
    spec = importlib.util.spec_from_file_location(
        "stage2_targets", tg.REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage2_targets.py")
    assert spec is not None and spec.loader is not None
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    a12 = st.load_stula_targets()["group_rates_tbl"]
    r20 = tg.load_rates20()
    r12 = np.zeros_like(a12)
    for k in range(tg.NUM_ACT20):
        r12[:, tg.STULA20_TO_COMMON[k]] += r20[:, k]
    assert np.abs(r12 - a12).max() < 5e-4
    p = tg.load_participation20()
    assert p.shape == (28, 20) and (p >= 0).all() and (p <= 1).all()
    print(f"f. 入力: OK  max|Σ20→12 − A*| = {np.abs(r12 - a12).max():.2e}")


if __name__ == "__main__":
    test_count_groups()
    test_group_delta()
    test_round_slot_counts()
    test_anneal()
    test_inputs_match_a_star()
    print("すべて通過")
