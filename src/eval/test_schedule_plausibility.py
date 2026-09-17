"""
test_schedule_plausibility.py
=============================
schedule_plausibility.py の検証。test_feasibility.py と同じ形式
（main() に素の assert ＋ 診断 print）。

重要な項目:
    (2) 循環エピソードが配列の端をまたいで 1 本に繋がること。
        04:00 起点では夜の睡眠が必ず端をまたぐので、ここが線形のままだと
        主睡眠が 2 本に割れ、main_sleep_mean_min が半分になる。
        本モジュールが存在する理由そのものなので、線形版との差も測る。
    (4) 日跨ぎの時刻窓（22:00-08:00）の判定。
        04:00 起点のオフセットを間違えると窓が 4 時間ずれる。

使い方:
    .venv/bin/python3 src/eval/test_schedule_plausibility.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sp = _load("schedule_plausibility", HERE / "schedule_plausibility.py")
im = _load("individual_metrics", HERE / "individual_metrics.py")


def _row(pairs: list[tuple[int, int, int]], base: int = 8) -> np.ndarray:
    """(開始, 終了, 活動) から 1 個票を作る。既定の埋め草は LEISURE_SOCIAL=8。"""
    row = np.full(96, base, dtype=np.int64)
    for s, e, a in pairs:
        row[s:e] = a
    return row


def test_circular_runs():
    """(1) run 分解の基本。"""
    # 全スロットが同じ活動 -> 1 本、長さ 96
    row = np.zeros(96, dtype=np.int64)
    start, length = sp.longest_circular_block(row[None, :], 0)
    assert (start[0], length[0]) == (0, 96), (start[0], length[0])
    assert sp.circular_episode_counts(row[None, :], 0).tolist() == [1]

    # 活動を 1 スロットも持たない -> (-1, 0)
    start, length = sp.longest_circular_block(row[None, :], 7)
    assert (start[0], length[0]) == (-1, 0)
    assert sp.circular_episode_counts(row[None, :], 7).tolist() == [0]

    # 離れた 2 本。長い方が採られる
    row = _row([(10, 14, 7), (40, 50, 7)])
    start, length = sp.longest_circular_block(row[None, :], 7)
    assert (start[0], length[0]) == (40, 10), (start[0], length[0])
    assert sp.circular_episode_counts(row[None, :], 7).tolist() == [2]
    print("  (1) 基本の run 分解: OK")


def test_wrap_merge():
    """(2) ★端をまたぐブロックが 1 本に繋がること。"""
    # slot 88..95 と 0..7 -> 円周上は 16 スロット 1 本、開始は 88
    row = _row([(88, 96, 0), (0, 8, 0)])
    start, length = sp.longest_circular_block(row[None, :], 0)
    assert (start[0], length[0]) == (88, 16), (start[0], length[0])
    assert sp.circular_episode_counts(row[None, :], 0).tolist() == [1]

    # 線形版は同じ個票を 2 本に割る（このモジュールが要る理由）
    linear = im.episode_lengths_by_act(row[None, :], 12)[0]
    assert sorted(linear.tolist()) == [8, 8], linear.tolist()

    # 端をまたぐ本と、またがない長い本が同居する場合は長い方
    row = _row([(90, 96, 0), (0, 2, 0), (40, 60, 0)])
    start, length = sp.longest_circular_block(row[None, :], 0)
    assert (start[0], length[0]) == (40, 20), (start[0], length[0])
    assert sp.circular_episode_counts(row[None, :], 0).tolist() == [2]
    print("  (2) 端をまたぐ連結: OK（線形版は 16 スロットを 8+8 に割る）")


def test_activity_span():
    """(3) 広がり＝全スロットを覆う最短の円弧。"""
    # 昼休みで割れた仕事: 32..40 と 44..56 -> 広がりは 32..56 の 24 スロット
    row = _row([(32, 40, 2), (44, 56, 2)])
    start, length = sp.activity_span(row[None, :], 2)
    assert (start[0], length[0]) == (32, 24), (start[0], length[0])
    # 最長ブロックは割れた側の長い方だけ
    b_start, b_len = sp.longest_circular_block(row[None, :], 2)
    assert (b_start[0], b_len[0]) == (44, 12)

    # 端をまたぐ広がり: 90..96 と 0..6
    row = _row([(90, 96, 2), (0, 6, 2)])
    start, length = sp.activity_span(row[None, :], 2)
    assert (start[0], length[0]) == (90, 12), (start[0], length[0])

    # 持たない個票 -> (-1, 0)。全スロット同一 -> (0, 96)
    assert sp.activity_span(np.full((1, 96), 8, dtype=np.int64), 2)[1][0] == 0
    assert sp.activity_span(np.full((1, 96), 2, dtype=np.int64), 2)[1][0] == 96
    print("  (3) activity_span: OK")


def test_clock_window():
    """(4) ★日跨ぎの時刻窓。04:00 起点のオフセットを間違えると 4 時間ずれる。"""
    assert sp.slot_to_clock_min(np.array([0]))[0] == 240.0       # slot 0 = 04:00
    assert sp.slot_to_clock_min(np.array([80]))[0] == 0.0        # slot 80 = 00:00
    # ★slot 95 は 23:45 ではなく翌 03:45。04:00 起点なので最終スロットは翌日側にある
    assert sp.slot_to_clock_min(np.array([95]))[0] == 225.0
    assert sp.slot_to_clock_min(np.array([72]))[0] == 1320.0     # slot 72 = 22:00

    night = sp.NIGHT_WINDOW                                      # 22:00-08:00
    cm = np.array([23 * 60, 2 * 60, 7 * 60 + 59, 8 * 60, 12 * 60, 21 * 60 + 59])
    assert sp.in_window(cm, night).tolist() == [True, True, True, False, False, False]

    day = sp.DAYTIME_WINDOW                                      # 06:00-20:00
    # slot 8 = 06:00 から 56 スロット = 20:00 ちょうど。終端に接するので収まる
    assert sp.arc_within_window(np.array([8]), np.array([56]), day)[0]
    assert not sp.arc_within_window(np.array([8]), np.array([57]), day)[0]
    # 区間なしは False
    assert not sp.arc_within_window(np.array([-1]), np.array([0]), day)[0]
    # 端をまたぐ夜勤は昼窓に収まらない
    assert not sp.arc_within_window(np.array([88]), np.array([16]), day)[0]
    # 22:00-08:00 の窓には収まる（slot 72 = 22:00 から 40 スロット = 08:00）
    assert sp.arc_within_window(np.array([72]), np.array([40]), night)[0]
    print("  (4) 時刻窓の判定: OK")


def test_weighting():
    """(5) 重みが率に効くこと。分母は保持者であること。"""
    # 1 人目: 夜に寝る / 2 人目: 昼に寝る
    night = _row([(72, 96, 0), (0, 16, 0)])
    noon = _row([(24, 64, 0)])
    S = np.stack([night, noon])
    assert abs(sp.sleep_plausibility(S)["main_sleep_nocturnal"] - 0.5) < 1e-12
    w = np.array([3.0, 1.0])
    assert abs(sp.sleep_plausibility(S, w)["main_sleep_nocturnal"] - 0.75) < 1e-12

    # WORK の率は保持者が分母。1 人だけが昼に働き、もう 1 人は WORK を持たない
    S = np.stack([_row([(32, 56, 2)]), _row([])])
    wk = sp.work_plausibility(S)
    assert abs(wk["work_holder_rate"] - 0.5) < 1e-12
    assert abs(wk["work_block_daytime"] - 1.0) < 1e-12, wk["work_block_daytime"]
    print("  (5) 重みと分母: OK")


def main():
    print("test_schedule_plausibility")
    test_circular_runs()
    test_wrap_merge()
    test_activity_span()
    test_clock_window()
    test_weighting()

    # --- 実 ATUS 平日での基準値（Stage2_design.md §9 の表の出所）---
    sm = _load("simple_model",
               REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "model.py")
    _, S, w, _ = sm.load_data()
    summary = sp.plausibility_summary(S, w)
    print(f"\n--- 実 ATUS 2024 平日 (N={len(S)}, TUFINLWGT 重み付き) の基準値 ---")
    for k, v in summary.items():
        print(f"  {k:24s} {v:.4f}")

    # 設計書に転記した実測値と一致すること
    expect = {
        "sleep_holder_rate":     0.9996,
        "main_sleep_nocturnal":  0.9748,
        "main_sleep_share_ok":   0.9893,
        "main_sleep_mean_min":   530.32,
        "work_holder_rate":      0.5097,
        "work_block_daytime":    0.8521,
        "work_span_daytime":     0.7543,
        "work_span_mean_min":    531.99,
        "meals_count_ok":        0.6611,
        "meals_count_mean":      1.8934,
    }
    for k, v in expect.items():
        tol = 0.01 if v > 1 else 2e-3
        assert abs(summary[k] - v) < tol, (k, summary[k], v)
    print("  基準値の再現: OK")

    print("\ntest_schedule_plausibility: OK")


if __name__ == "__main__":
    main()
