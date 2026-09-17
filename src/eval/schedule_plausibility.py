"""
schedule_plausibility.py
========================
「活動スケジュールとして人間らしいか」を**個人単位の述語**で測る。

なぜ別モジュールか:
    集計カーブのピーク位置は妥当性の証拠にならない。群内で 96 スロットを人の間で
    独立にシャッフルしても、群別の時刻別行動者率は 1 セルも変わらない
    (src/eval/teacher_free_space.py、群別周辺の最大差 0.000e+00) のに、切替回数の
    平均は 12.89 -> 50.59 へ動く。つまり「昼に WORK のピークが立つ」ことと
    「その人が昼に働いている」ことは別である。妥当性は個人ごとの述語として定義し、
    実データでの発生率と比べる。

なぜ循環エピソードか:
    日記日は 04:00 起点なので、夜の睡眠は必ず配列の端をまたぐ
    (slot 76 = 23:00 から slot 95 = 03:45 まで進み、slot 0 = 04:00 へ続く)。
    individual_metrics.episode_lengths は配列を線形に走査するので、この 1 本を
    2 本に割ってしまう。本モジュールは 96 スロットを円周として扱う。

★共通12分類の SLEEP_PERSONAL は「睡眠」と「身の回りの用事」の和 (STULA 01+02) で、
  純粋な睡眠ではない。そのため「主睡眠」ではなく「最長 SLEEP_PERSONAL ブロック」と呼ぶ。

判定の考え方 (feasibility.py と同じ):
    ゼロ違反を要求しない。実データでも最長 WORK ブロックが 06:00-20:00 に収まるのは
    85.4% で、夜勤・交替制が残りを占める。実データでの発生率を基準にし、生成側が
    そこからどれだけ離れるかで見る。

使い方:
    .venv/bin/python3 src/eval/test_schedule_plausibility.py

    sp = <importlib で src/eval/schedule_plausibility.py をロード>
    sp.plausibility_summary(sched, w)
    sp.compare_plausibility(sched_real, sched_gen, w_real, w_gen)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

NUM_SLOTS = 96
SLOT_MIN = 15
START_HOUR = 4
MINUTES_PER_DAY = 24 * 60

# 共通12分類 (crosswalk_atus_stula.Common) のうち本モジュールが名前で参照するもの
SLEEP_PERSONAL = 0
MEALS = 1
WORK = 2

# --- 窓と閾値（実 ATUS 2024 平日 N=3,736、TUFINLWGT 重み付きの実測を根拠に置く）---
# 最長 SLEEP_PERSONAL ブロックの中心がこの窓に入る人は 97.54%
NIGHT_WINDOW = (22 * 60, 8 * 60)
# 最長 WORK ブロックがこの窓に収まる WORK 保持者は 85.36%、
# WORK の広がり（下の activity_span）が収まる保持者は 75.53%
DAYTIME_WINDOW = (6 * 60, 20 * 60)
# 最長 SLEEP_PERSONAL ブロックが総 SLEEP_PERSONAL 時間に占める割合の下限。実測 98.93%
MAIN_SLEEP_SHARE_MIN = 0.5
# MEALS のエピソード本数がこの範囲に入る人は 66.11%
MEALS_EPISODES_RANGE = (2, 4)

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


def _norm_w(n: int, w: FloatArr | None) -> FloatArr:
    """重みを合計 1 に正規化する。None なら一様重み。

    Args:
        n: 個票数
        w: 重み, dtype=float64, (n,)。None なら一様

    Returns:
        正規化済みの重み, dtype=float64, (n,)
    """
    if w is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    wn = np.asarray(w, dtype=np.float64)
    return wn / wn.sum()


# ============================================================
# 1. 循環エピソード分解
# ============================================================
def _circular_runs(mask: BoolArr) -> tuple[IntArr, IntArr, IntArr]:
    """真偽マスクを円周上の連続 run に分解する。

    やり方:
        1. 各行の両端をゼロで挟み、差分が +1 の位置を run の開始、-1 の位置を
           run の終端の次とする（線形の run 分解）
        2. 行の先頭と末尾がどちらも True で、かつ run が 2 本以上ある行は、
           末尾の run に先頭の run を繋いで 1 本にする（円周の連結）

    Args:
        mask: 対象スロットの真偽値, dtype=bool, (N, NUM_SLOTS) = (N, 96)

    Returns:
        (row, start, length) の3本。いずれも dtype=int64, (R,)。
        row[i] が run の属する行、start[i] が開始スロット、length[i] がスロット数。
        長さ 0 の run は含まない。行あたりの run は先頭から順に並ぶ。
    """
    n = mask.shape[0]
    padded = np.zeros((n, NUM_SLOTS + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)                       # (N, 97)

    row, start = np.nonzero(diff == 1)                   # run 開始（行, スロット）
    _, end = np.nonzero(diff == -1)                      # run 終端の次（スロット）
    row = row.astype(np.int64)
    start = start.astype(np.int64)
    length = (end - start).astype(np.int64)

    # 行ごとの最初と最後の run の位置。np.nonzero は行優先で返すので row は昇順
    first = np.searchsorted(row, np.arange(n), side="left")
    last = np.searchsorted(row, np.arange(n), side="right") - 1
    n_runs = last - first + 1
    # ★run が 1 本だけの行を除く。全スロットが True の行で先頭と末尾を繋ぐと
    #   同じ run を二重に数えてしまう
    merge = np.asarray(mask[:, 0] & mask[:, -1] & (n_runs >= 2))
    if merge.any():
        f, l = first[merge], last[merge]
        length[l] = length[l] + length[f]                # 末尾の run に先頭を繋ぐ
        length[f] = 0                                    # 先頭の run は消す

    keep = length > 0
    return row[keep], start[keep], length[keep]


def _longest_true_run(mask: BoolArr) -> tuple[IntArr, IntArr]:
    """各行について、円周上で最長の True の run を返す。

    Args:
        mask: 対象スロットの真偽値, dtype=bool, (N, NUM_SLOTS) = (N, 96)

    Returns:
        (start, length)。dtype=int64, いずれも (N,)。
        True が 1 つも無い行は (-1, 0)。
    """
    n = mask.shape[0]
    row, start, length = _circular_runs(mask)

    # 行ごとに長さ降順へ並べ、各行の先頭を採る
    order = np.lexsort((-length, row))
    row_s, start_s, length_s = row[order], start[order], length[order]

    best_start = np.full(n, -1, dtype=np.int64)
    best_len = np.zeros(n, dtype=np.int64)
    if row_s.size:
        present = np.unique(row_s)
        head = np.searchsorted(row_s, present, side="left")
        best_start[present] = start_s[head]
        best_len[present] = length_s[head]
    return best_start, best_len


def longest_circular_block(sched: IntArr, act: int) -> tuple[IntArr, IntArr]:
    """各個票について、活動 act の循環上で最長のブロックを返す。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        act: 対象の活動インデックス, 範囲 [0, 12)

    Returns:
        (start, length)。dtype=int64, いずれも (N,)。
        start は開始スロット、length はスロット数。
        活動 act を 1 スロットも持たない個票は (-1, 0)。
    """
    return _longest_true_run(np.asarray(np.asarray(sched) == act))


def circular_episode_counts(sched: IntArr, act: int) -> IntArr:
    """各個票の、活動 act の循環エピソード本数。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        act: 対象の活動インデックス, 範囲 [0, 12)

    Returns:
        エピソード本数, dtype=int64, (N,)
    """
    sched = np.asarray(sched)
    row, _, _ = _circular_runs(np.asarray(sched == act))
    return np.bincount(row, minlength=sched.shape[0]).astype(np.int64)


def activity_span(sched: IntArr, act: int) -> tuple[IntArr, IntArr]:
    """活動 act の全スロットを覆う、円周上で最短の区間（＝広がり）。

    最長ブロックとの違い:
        最長ブロックは休憩で割れると短くなる。広がりは「最初の仕事の開始から
        最後の仕事の終了まで」を測るので、昼休みで割れても縮まない。
        断片化した個票は広がりが 1 日全体に膨らむので、両方を見ると
        「まとまって働いたか」と「昼の時間帯に働いたか」を分けられる。

    やり方:
        円周上で act でないスロットの最長 run（＝最大の空白）を外した残りが、
        全 act スロットを覆う最短の円弧である。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        act: 対象の活動インデックス, 範囲 [0, 12)

    Returns:
        (start, length)。dtype=int64, いずれも (N,)。
        活動 act を 1 スロットも持たない個票は (-1, 0)。
    """
    mask = np.asarray(np.asarray(sched) == act)
    gap_start, gap_len = _longest_true_run(np.asarray(~mask))   # 最大の空白

    span_len = NUM_SLOTS - gap_len
    span_start = (gap_start + gap_len) % NUM_SLOTS
    has = mask.any(axis=1)
    all_act = mask.all(axis=1)                            # 空白が無い行は開始を 0 に置く
    span_start = np.where(all_act, 0, span_start)
    return (np.where(has, span_start, -1).astype(np.int64),
            np.where(has, span_len, 0).astype(np.int64))


# ============================================================
# 2. 時刻窓の判定
# ============================================================
def slot_to_clock_min(slot: IntArr | int) -> FloatArr:
    """スロット番号を時計の分（0 = 00:00）へ直す。04:00 起点を戻す。

    Args:
        slot: スロット番号, dtype=int64, 任意形状。範囲 [0, 96)

    Returns:
        時計の分, dtype=float64, 入力と同じ形状。範囲 [0, 1440)
    """
    return (np.asarray(slot, dtype=np.float64) * SLOT_MIN
            + START_HOUR * 60) % MINUTES_PER_DAY


def _window_minutes(window: tuple[int, int]) -> int:
    """時刻窓の長さ（分）。lo == hi は 1 日全体とみなす。

    Args:
        window: (開始分, 終了分)。lo > hi なら日跨ぎの窓

    Returns:
        窓の長さ（分）, 範囲 (0, 1440]
    """
    lo, hi = window
    return (hi - lo) % MINUTES_PER_DAY or MINUTES_PER_DAY


def in_window(clock_min: FloatArr, window: tuple[int, int]) -> BoolArr:
    """時刻が窓に入るか。日跨ぎの窓（22:00-08:00 など）も扱う。

    Args:
        clock_min: 時計の分, dtype=float64, (N,)。範囲 [0, 1440)
        window: (開始分, 終了分)。終端は含まない

    Returns:
        窓に入るか, dtype=bool, (N,)
    """
    lo, _ = window
    offset = (np.asarray(clock_min, dtype=np.float64) - lo) % MINUTES_PER_DAY
    return np.asarray(offset < _window_minutes(window))


def arc_within_window(start: IntArr, length: IntArr,
                      window: tuple[int, int]) -> BoolArr:
    """区間（円弧）が丸ごと窓に収まるか。

    区間の始点を窓の開始からの相対位置に直し、そこに区間長を足しても窓から
    はみ出さないかで判定する。これは日跨ぎの窓でもそのまま成り立つ。

    Args:
        start: 開始スロット, dtype=int64, (N,)。-1 は区間なし
        length: スロット数, dtype=int64, (N,)。0 は区間なし
        window: (開始分, 終了分)。終端は含まない

    Returns:
        丸ごと収まるか, dtype=bool, (N,)。区間なしの個票は False
    """
    lo, _ = window
    offset = (slot_to_clock_min(start) - lo) % MINUTES_PER_DAY
    fits = offset + np.asarray(length, dtype=np.float64) * SLOT_MIN <= _window_minutes(window)
    return np.asarray((np.asarray(length) > 0) & fits)


# ============================================================
# 3. 個人単位の妥当性
# ============================================================
def _rate_among(flag: BoolArr, holder: BoolArr, wn: FloatArr) -> float:
    """holder が True の個票の中で、flag も True の重み付き割合。

    Args:
        flag: 判定結果, dtype=bool, (N,)
        holder: 分母に入れる個票, dtype=bool, (N,)
        wn: 正規化済み重み, dtype=float64, (N,)

    Returns:
        割合。holder が 1 つも無ければ NaN
    """
    denom = float(wn[holder].sum())
    if denom <= 0.0:
        return float("nan")
    return float(wn[holder & flag].sum() / denom)


def _mean_among(x: FloatArr, holder: BoolArr, wn: FloatArr) -> float:
    """holder が True の個票に限った重み付き平均。

    Args:
        x: 対象の値, dtype=float64, (N,)
        holder: 分母に入れる個票, dtype=bool, (N,)
        wn: 正規化済み重み, dtype=float64, (N,)

    Returns:
        平均。holder が 1 つも無ければ NaN
    """
    denom = float(wn[holder].sum())
    if denom <= 0.0:
        return float("nan")
    return float((wn[holder] * np.asarray(x, dtype=np.float64)[holder]).sum() / denom)


def sleep_plausibility(sched: IntArr, w: FloatArr | None = None) -> dict[str, float]:
    """「夜まとまって寝ているか」。最長 SLEEP_PERSONAL ブロックで測る。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        w: 個票の重み, dtype=float64, (N,)。None なら一様

    Returns:
        指標名 -> 値。率は SLEEP_PERSONAL 保持者に限った値
            sleep_holder_rate       SLEEP_PERSONAL を 1 スロット以上持つ割合
            main_sleep_nocturnal    最長ブロックの中心が NIGHT_WINDOW に入る割合
            main_sleep_share_ok     最長ブロック / 総 SLEEP_PERSONAL が閾値以上の割合
            main_sleep_share_mean   最長ブロック / 総 SLEEP_PERSONAL の平均
            main_sleep_mean_min     最長ブロックの平均長（分）
    """
    sched = np.asarray(sched)
    wn = _norm_w(sched.shape[0], w)
    start, length = longest_circular_block(sched, SLEEP_PERSONAL)
    holder = length > 0

    total = np.asarray(sched == SLEEP_PERSONAL).sum(axis=1).astype(np.float64)
    share = np.divide(length.astype(np.float64), total,
                      out=np.zeros(sched.shape[0], dtype=np.float64), where=total > 0)
    center = (slot_to_clock_min(start)
              + length.astype(np.float64) * SLOT_MIN / 2.0) % MINUTES_PER_DAY

    return {
        "sleep_holder_rate":     float(wn[holder].sum()),
        "main_sleep_nocturnal":  _rate_among(in_window(center, NIGHT_WINDOW), holder, wn),
        "main_sleep_share_ok":   _rate_among(np.asarray(share >= MAIN_SLEEP_SHARE_MIN),
                                             holder, wn),
        "main_sleep_share_mean": _mean_among(share, holder, wn),
        "main_sleep_mean_min":   _mean_among(length.astype(np.float64) * SLOT_MIN,
                                             holder, wn),
    }


def work_plausibility(sched: IntArr, w: FloatArr | None = None) -> dict[str, float]:
    """「昼に働いているか」。最長 WORK ブロックと WORK の広がりで測る。

    ★率の分母は WORK 保持者である。全体に対する割合ではないので、
      work_holder_rate と併せて読む。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        w: 個票の重み, dtype=float64, (N,)。None なら一様

    Returns:
        指標名 -> 値
            work_holder_rate      WORK を 1 スロット以上持つ割合（＝日次行動者率）
            work_block_daytime    最長ブロックが DAYTIME_WINDOW に収まる割合
            work_span_daytime     広がりが DAYTIME_WINDOW に収まる割合
            work_block_mean_min   最長ブロックの平均長（分）
            work_span_mean_min    広がりの平均長（分）
    """
    sched = np.asarray(sched)
    wn = _norm_w(sched.shape[0], w)
    start, length = longest_circular_block(sched, WORK)
    span_start, span_len = activity_span(sched, WORK)
    holder = length > 0

    return {
        "work_holder_rate":    float(wn[holder].sum()),
        "work_block_daytime":  _rate_among(arc_within_window(start, length, DAYTIME_WINDOW),
                                           holder, wn),
        "work_span_daytime":   _rate_among(
            arc_within_window(span_start, span_len, DAYTIME_WINDOW), holder, wn),
        "work_block_mean_min": _mean_among(length.astype(np.float64) * SLOT_MIN, holder, wn),
        "work_span_mean_min":  _mean_among(span_len.astype(np.float64) * SLOT_MIN, holder, wn),
    }


def meals_plausibility(sched: IntArr, w: FloatArr | None = None) -> dict[str, float]:
    """「食事が 1 日に数回あるか」。MEALS のエピソード本数で測る。

    ★実データでも範囲内は 66.1% しかない（0 回 5.7% / 1 回 27.8%）ので、
      単独では弱い指標である。断片化で本数が跳ね上がる場合の検出に使う。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        w: 個票の重み, dtype=float64, (N,)。None なら一様

    Returns:
        指標名 -> 値
            meals_count_ok    エピソード本数が MEALS_EPISODES_RANGE に入る割合
            meals_count_mean  エピソード本数の平均（全個票）
    """
    sched = np.asarray(sched)
    wn = _norm_w(sched.shape[0], w)
    counts = circular_episode_counts(sched, MEALS)
    lo, hi = MEALS_EPISODES_RANGE
    return {
        "meals_count_ok":   float((wn * ((counts >= lo) & (counts <= hi))).sum()),
        "meals_count_mean": float((wn * counts).sum()),
    }


def plausibility_summary(sched: IntArr, w: FloatArr | None = None) -> dict[str, float]:
    """睡眠・仕事・食事の妥当性指標をまとめて返す。

    Args:
        sched: 活動スケジュール, dtype=int64, (N, NUM_SLOTS) = (N, 96)
        w: 個票の重み, dtype=float64, (N,)。None なら一様

    Returns:
        指標名 -> 値。sleep_plausibility / work_plausibility / meals_plausibility の和
    """
    return {**sleep_plausibility(sched, w),
            **work_plausibility(sched, w),
            **meals_plausibility(sched, w)}


def plausibility_stats() -> dict[str, Callable[[IntArr, FloatArr | None], float]]:
    """individual_metrics.compare_with_band に渡す {指標名: 関数} を返す。

    compare_with_band は 1 つの float を返す関数しか受け取れないので、
    まとめて返す 3 つの関数から指標を 1 つずつ取り出す薄い包みを作る。

    Returns:
        指標名 -> (sched, w) を取り float を返す関数
    """
    families: list[tuple[Callable[[IntArr, FloatArr | None], dict[str, float]],
                         tuple[str, ...]]] = [
        (sleep_plausibility, ("sleep_holder_rate", "main_sleep_nocturnal",
                              "main_sleep_share_ok", "main_sleep_share_mean",
                              "main_sleep_mean_min")),
        (work_plausibility,  ("work_holder_rate", "work_block_daytime",
                              "work_span_daytime", "work_block_mean_min",
                              "work_span_mean_min")),
        (meals_plausibility, ("meals_count_ok", "meals_count_mean")),
    ]

    def pick(fn: Callable[[IntArr, FloatArr | None], dict[str, float]],
             key: str) -> Callable[[IntArr, FloatArr | None], float]:
        return lambda sched, w=None: fn(sched, w)[key]

    return {key: pick(fn, key) for fn, keys in families for key in keys}


def compare_plausibility(real: IntArr, gen: IntArr,
                         w_real: FloatArr | None = None,
                         w_gen: FloatArr | None = None) -> pd.DataFrame:
    """実データと生成を並べた比較表。

    ★重みの基準を揃えること。生成側は群一様のプールなので、実データと同じ
      群構成へ individual_metrics.group_reweight で寄せてから渡す。

    Args:
        real: 実データの活動スケジュール, dtype=int64, (N_real, 96)
        gen: 生成の活動スケジュール, dtype=int64, (N_gen, 96)
        w_real: 実データの重み, dtype=float64, (N_real,)。None なら一様
        w_gen: 生成の重み, dtype=float64, (N_gen,)。None なら一様

    Returns:
        列 metric / real / gen / ratio の DataFrame。ratio = gen / real
    """
    r = plausibility_summary(real, w_real)
    g = plausibility_summary(gen, w_gen)
    return pd.DataFrame({
        "metric": list(r),
        "real": [r[k] for k in r],
        "gen": [g[k] for k in r],
        "ratio": [g[k] / r[k] if r[k] else float("nan") for k in r],
    })
