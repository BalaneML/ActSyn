"""
individual_metrics.py
================
活動スケジュール個票（1人1日 = 96スロットの活動インデックス列）の評価指標。

これまで個票指標は notebooks/eval_cvae/05_selfcontained_eval.ipynb・
notebooks/eval_japan_pretrain/pretrain_individual_analysis.ipynb などに
コピペで散在していた（notebooks/ は gitignore 対象なので追跡資産がゼロ）。
本モジュールはそれらを単一定義に集約する。

設計方針:
    - 活動分類数 n_act と活動名は引数で受ける。モジュールに分類定数を焼き込まない
      （共通12分類・ATUS17分類・NHTS10分類のどれにも同じ関数を使うため）。
    - トップレベル依存は numpy / pandas / scipy / matplotlib のみ。torch は
      holdout_split の内部でのみ import する（学習側と疎結合に保つ）。
    - 重み w は survey weight（ATUS の TUFINLWGT）。None なら一様重み。

★ 時刻とスロットの対応（ここを間違えると別の時間帯を測る）:
    スケジュールは 04:00 開始。slot 0 = 04:00-04:15, slot 80 = 00:00-00:15,
    slot 95 = 03:45-04:00。したがって 00:00-04:00 は slot 80..95（末尾16スロット）で、
    slot 0..15 は 04:00-08:00 である。旧 DDPM/model.py が深夜チェックを s72..s95
    （22:00-04:00）と書いているのも同じ理由。

★ 切替の境界の定義:
    境界 t は slot t-1 と slot t の間（t = 1..95, 計95本）を指し、
    slot t の開始時刻に起きたものとして時間帯に割り当てる。
    日跨ぎ境界（slot 95 -> 翌日 slot 0）は n_switches に含めない。
    日記日の開始と終了の整合は wrap_closure_rate で別に測る。

★ 3つの「平均継続時間」は別の量なので別の名前で出す（duration_table 参照）:
    ep_mean_min     1エピソードあたりの平均長
    actor_mean_min  その活動を1回以上した人の1日合計時間の平均（= 行動者平均時間）
    pop_mean_min    全員の1日合計時間の平均（= time_share × 1440）
    実 ATUS 平日 val の WORK では順に 219.8 / 470.9 / 217.2 分。混同すると別の主張になる。

使い方:
    # 自己テスト
    uv run python src/eval/test_individual_metrics.py

    # 呼び出し側（ノートブック等）
    im = <importlib で src/eval/individual_metrics.py をロード>
    tr_idx, va_idx = im.holdout_split(len(sched))        # make_loaders と同一分割
    im.switch_stats(sched[va_idx], w[va_idx])
    im.switches_by_band(sched[va_idx], w[va_idx])
    im.duration_table(sched[va_idx], n_act, act_names, w[va_idx])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import ks_2samp, wasserstein_distance

# ============================================================
# 1. 時間軸の定数
# ============================================================
NUM_SLOTS  = 96
SLOT_MIN   = 15
START_HOUR = 4      # スケジュールの先頭スロットの時刻
MIN_PER_DAY = 24 * 60

IntArr   = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]


def roll_slots() -> int:
    """00:00 開始で表示するための np.roll 量 (= 16)。"""
    return START_HOUR * 60 // SLOT_MIN


def slot_clock_min(slot: int | IntArr) -> IntArr:
    """slot -> その開始時刻の「0時起点の分」。slot 80 -> 0 (= 00:00)。"""
    return np.asarray((START_HOUR * 60 + np.asarray(slot) * SLOT_MIN) % MIN_PER_DAY)


def fmt_time(slot: int) -> str:
    """slot -> その開始時刻 "HH:MM"。"""
    m = int(slot_clock_min(slot))
    return f"{m // 60:02d}:{m % 60:02d}"


def _norm_w(n: int, w: FloatArr | None) -> FloatArr:
    """重みを和1に正規化して返す。None なら一様。"""
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    if len(w) != n:
        raise ValueError(f"weights の長さ {len(w)} が個票数 {n} と一致しない")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights の総和が正でない")
    return w / s


# ============================================================
# 2. 系列分解
# ============================================================
def to_segments(seq: IntArr, act_names: list[str]) -> list[tuple[str, int, int]]:
    """1日の状態列を (活動名, 開始slot, 終了slot) の連続区間に分解する。"""
    segs: list[tuple[str, int, int]] = []
    s = 0
    for t in range(1, len(seq) + 1):
        if t == len(seq) or seq[t] != seq[s]:
            segs.append((act_names[int(seq[s])], s, t))
            s = t
    return segs


def episode_lengths(sched: IntArr) -> IntArr:
    """全個票のエピソード長（スロット数）を連結した (M,)。

    model.py:629 の fragmentation_stats は sched[:3000] で打ち切るが、
    ここでは打ち切らない（全個票を使う）。このため sanity_check の print と
    エピソード長の値が僅かにずれ得る。
    """
    sched = np.asarray(sched)
    out: list[int] = []
    for row in sched:
        c = 1
        for t in range(1, sched.shape[1]):
            if row[t] == row[t - 1]:
                c += 1
            else:
                out.append(c)
                c = 1
        out.append(c)
    return np.asarray(out, dtype=np.int64)


def episode_lengths_by_act(sched: IntArr, n_act: int) -> list[IntArr]:
    """活動ごとのエピソード長 (スロット数)。返り値 out[a] が活動 a の長さ配列。"""
    sched = np.asarray(sched)
    buf: list[list[int]] = [[] for _ in range(n_act)]
    for row in sched:
        c = 1
        for t in range(1, sched.shape[1]):
            if row[t] == row[t - 1]:
                c += 1
            else:
                buf[int(row[t - 1])].append(c)
                c = 1
        buf[int(row[-1])].append(c)
    return [np.asarray(b, dtype=np.int64) for b in buf]


def switch_matrix(sched: IntArr) -> npt.NDArray[np.bool_]:
    """(N, 95) bool。列 j が境界 t = j+1（slot j と slot j+1 の間）の切替有無。

    n_switches と switches_by_band の共通の土台。日跨ぎ境界は含まない。
    """
    sched = np.asarray(sched)
    return sched[:, 1:] != sched[:, :-1]


def n_switches(sched: IntArr) -> IntArr:
    """個人別の切替回数 (N,)。境界95本のうち切替が起きた本数。"""
    return switch_matrix(sched).sum(axis=1).astype(np.int64)


# ============================================================
# 3. 軸1: 切替回数の分布統計
# ============================================================
@overload
def weighted_quantile(x: FloatArr, q: float,
                      w: FloatArr | None = ...) -> float: ...
@overload
def weighted_quantile(x: FloatArr, q: list[float],
                      w: FloatArr | None = ...) -> FloatArr: ...
def weighted_quantile(x: FloatArr, q: float | list[float],
                      w: FloatArr | None = None) -> Any:
    """重み付き分位点。w=None のとき np.percentile（線形補間）と一致する。

    np.percentile は重みを取れないので、重み付き経験分布
    CDF_i = (cumsum(w) - 0.5*w) / sum(w) を線形補間する。
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    xs = x[order]
    ws = _norm_w(len(x), w)[order]
    cdf = np.cumsum(ws) - 0.5 * ws
    # 一様重みのとき cdf = (i + 0.5)/n となり np.percentile の (i)/(n-1) と
    # 座標系が違うので、端点を 0 / 1 に張り直してから補間する
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0]) if cdf[-1] > cdf[0] else np.zeros_like(cdf)
    return np.interp(q, cdf, xs)


def switch_stats(sched: IntArr, w: FloatArr | None = None) -> dict:
    """切替回数の分布統計。平均だけでなく分散・裾まで出す。

    平均が実データと一致していても分散や裾が違う場合を分離するのが目的。
    cv (= std/mean) が実データより小さければモード崩壊寄り、
    大きければ断片化の個人差が過大。

    ★ var / std は重み付き経験分布の母分散（ddof=0）。重みを持つ場合に
      ddof=1（標本分散）は一意に定まらないため ddof=0 に固定する。
      非重みの np.var(ddof=1) とは N/(N-1) 倍だけ違う
      （実 ATUS 平日 val: ddof=0 で 22.869、ddof=1 で 22.930）。
    """
    sw = n_switches(sched).astype(np.float64)
    wn = _norm_w(len(sw), w)
    mean = float((wn * sw).sum())
    var  = float((wn * (sw - mean) ** 2).sum())
    std  = float(np.sqrt(var))
    q = weighted_quantile(sw, [0.25, 0.50, 0.75, 0.90, 0.95], w)
    return {
        "mean": mean, "var": var, "std": std,
        "cv": float(std / mean) if mean > 0 else float("nan"),
        "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]),
        "p90": float(q[3]), "p95": float(q[4]),
        "min": float(sw.min()), "max": float(sw.max()),
    }


def switch_dist_compare(real: IntArr, gen: IntArr,
                        w_real: FloatArr | None = None,
                        w_gen: FloatArr | None = None) -> dict:
    """切替回数（1次元量）の分布距離。emd は重み付き、ks は非重み付き。"""
    sr = n_switches(real).astype(np.float64)
    sg = n_switches(gen).astype(np.float64)
    emd = float(wasserstein_distance(sr, sg,
                                     u_weights=None if w_real is None else np.asarray(w_real),
                                     v_weights=None if w_gen is None else np.asarray(w_gen)))
    # scipy の検定結果クラスは実行時に組み立てられるので静的には属性が見えない
    ks: Any = ks_2samp(sr, sg)
    return {"emd": emd, "ks": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}


def bootstrap_ci(x: FloatArr, w: FloatArr | None = None, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """重み付き平均の bootstrap 信頼区間。標本が薄い（val N=373）ときの解釈に必須。"""
    x = np.asarray(x, dtype=np.float64)
    wn = _norm_w(len(x), w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    # 重みも一緒にリサンプリングして各反復内で再正規化する
    xs, ws = x[idx], wn[idx]
    means = (xs * ws).sum(axis=1) / ws.sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ============================================================
# 4. 軸2: 時間帯別の切替回数
# ============================================================
def _boundary_clock_min() -> IntArr:
    """境界 t = 1..95 の発生時刻（0時起点の分）。境界 t は slot t の開始時刻に起きる。"""
    return slot_clock_min(np.arange(1, NUM_SLOTS))


def switches_by_band(sched: IntArr, w: FloatArr | None = None,
                     band_hours: int = 2) -> pd.DataFrame:
    """時間帯別の切替。00:00 起点で band_hours 刻みに集計する。

    ★ 帯ごとの境界本数が一定でないので switch_rate（境界1本あたりの切替確率）を
      必ず併記する。境界は t = 1..95 の95本で、04:00 の位置にあたる t = 0 は
      境界を持たない。このため 04:00 を含む帯だけ境界が1本少ない
      （band_hours=2 なら 04:00-06:00 が7本、他は8本）。
    """
    if MIN_PER_DAY % (band_hours * 60) != 0:
        raise ValueError(f"band_hours={band_hours} が24時間を割り切らない")
    sm = switch_matrix(sched)
    wn = _norm_w(len(sm), w)
    clock = _boundary_clock_min()
    band = clock // (band_hours * 60)
    n_band = MIN_PER_DAY // (band_hours * 60)

    rows = []
    for b in range(n_band):
        msk = band == b
        n_bnd = int(msk.sum())
        per_person = float((wn * sm[:, msk].sum(axis=1)).sum())
        start_min = b * band_hours * 60
        end_min = (b + 1) * band_hours * 60
        rows.append({
            "band_start_hour": start_min // 60,
            "label": f"{start_min // 60:02d}:{start_min % 60:02d}-{end_min // 60:02d}:{end_min % 60:02d}",
            "n_boundaries": n_bnd,
            "switch_rate": per_person / n_bnd if n_bnd else float("nan"),
            "switches_per_person": per_person,
        })
    return pd.DataFrame(rows)


def night_switches(sched: IntArr, w: FloatArr | None = None,
                   start_hour: int = 0, end_hour: int = 4) -> dict:
    """深夜帯（既定 00:00-04:00 = slot 80..95, 境界16本）の切替。

    実データでは深夜帯にほぼ切替が起きない（実 ATUS 平日 val で 0.231 回/人 =
    全切替の 1.8%、02:00-04:00 は境界あたり 0.0081）。per-slot 独立に抽出する
    モデルは残差エントロピーを時刻に無関係にばらまくので、この帯でこそ比が跳ねる。
    全体平均の比では見えない失敗を捕まえる診断軸。
    """
    sm = switch_matrix(sched)
    wn = _norm_w(len(sm), w)
    clock = _boundary_clock_min()
    msk = (clock >= start_hour * 60) & (clock < end_hour * 60)
    n_bnd = int(msk.sum())
    per_person = float((wn * sm[:, msk].sum(axis=1)).sum())
    total = float((wn * sm.sum(axis=1)).sum())
    return {
        "label": f"{start_hour:02d}:00-{end_hour:02d}:00",
        "n_boundaries": n_bnd,
        "switches_per_person": per_person,
        "switch_rate": per_person / n_bnd if n_bnd else float("nan"),
        "share_of_all_switches": per_person / total if total > 0 else float("nan"),
    }


def wrap_closure_rate(sched: IntArr, w: FloatArr | None = None) -> float:
    """日跨ぎ境界（slot 95 -> 翌日 slot 0）で活動が変わる個票の割合。

    日記日は 04:00 に始まり 04:00 に終わる。n_switches は境界95本しか数えず
    この境界を含めないので、開始と終了の整合を別指標として出す。
    実 ATUS 平日 val = 0.068。
    """
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    return float((wn * (sched[:, -1] != sched[:, 0])).sum())


# ============================================================
# 5. 軸3: 活動分類ごとの継続時間
# ============================================================
def duration_table(sched: IntArr, n_act: int, act_names: list[str],
                   w: FloatArr | None = None) -> pd.DataFrame:
    """活動別の継続時間。3つの「平均継続時間」を別の列として出す。

    ep_mean_min    1エピソードあたりの平均長（分）。エピソード単位の平均で、
                   重みは掛けない（エピソードは個人に1対多で紐づくため）
    actor_mean_min 行動者平均時間。その活動を1回以上した人の1日合計時間の重み付き平均。
                   docs の dur_rmse（行動者平均スロット数）と同じ定義
    pop_mean_min   全員平均時間。= time_share × 1440
    恒等式 pop_mean_min = actor_mean_min × participation が成り立つ。
    """
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    by_act = episode_lengths_by_act(sched, n_act)
    n_person = len(sched)

    rows = []
    for a in range(n_act):
        lens = by_act[a]
        tot_min = (sched == a).sum(axis=1) * float(SLOT_MIN)   # 個人別の1日合計時間
        did = (sched == a).any(axis=1)
        part = float(wn[did].sum())
        actor_mean = float((wn[did] * tot_min[did]).sum() / part) if part > 0 else 0.0
        rows.append({
            "activity": act_names[a],
            "ep_mean_min": float(lens.mean() * SLOT_MIN) if len(lens) else 0.0,
            "ep_median_min": float(np.median(lens) * SLOT_MIN) if len(lens) else 0.0,
            "ep_p90_min": float(np.percentile(lens, 90) * SLOT_MIN) if len(lens) else 0.0,
            "n_episodes_per_person": len(lens) / n_person,
            "actor_mean_min": actor_mean,
            "pop_mean_min": float((wn * tot_min).sum()),
            "participation": part,
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. 分布量
# ============================================================
def time_share(sched: IntArr, n_act: int, w: FloatArr | None = None) -> FloatArr:
    """活動別の時間シェア (n_act,)。和は1。"""
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    onehot = np.eye(n_act, dtype=np.float64)[sched]          # (N, 96, n_act)
    return np.einsum("n,nta->a", wn, onehot) / sched.shape[1]


def participation(sched: IntArr, n_act: int, w: FloatArr | None = None) -> FloatArr:
    """日次参加率 (n_act,)。その活動を1日に1回以上した人の割合。"""
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    return np.array([float((wn * (sched == a).any(axis=1)).sum()) for a in range(n_act)])


def participation_by_slot(sched: IntArr, n_act: int,
                          w: FloatArr | None = None) -> FloatArr:
    """時刻別行動者率 (96, n_act)。各スロットの活動確率で、スロットごとの和は1。"""
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    onehot = np.eye(n_act, dtype=np.float64)[sched]          # (N, 96, n_act)
    return np.einsum("n,nta->ta", wn, onehot)


# ============================================================
# 7. 要約と比較
# ============================================================
def fragmentation_summary(sched: IntArr, w: FloatArr | None = None) -> dict:
    """断片化の要約。switch_stats にエピソード長と単一スロット率を加える。

    switch_mean は model.py:622 fragmentation_stats の "switches" と同じ量。
    エピソード長は打ち切りをしない点だけが違う（episode_lengths の docstring 参照）。
    """
    st = switch_stats(sched, w)
    runs = episode_lengths(sched)
    return {
        "switch_mean": st["mean"], "switch_var": st["var"], "switch_std": st["std"],
        "switch_cv": st["cv"], "switch_median": st["median"],
        "switch_q25": st["q25"], "switch_q75": st["q75"],
        "switch_p90": st["p90"], "switch_p95": st["p95"],
        "switch_min": st["min"], "switch_max": st["max"],
        "ep_len_mean": float(runs.mean()),
        "ep_len_median": float(np.median(runs)),
        "single_slot_ratio": float((runs == 1).mean()),
        "wrap_closure_rate": wrap_closure_rate(sched, w),
    }


def compare_frag(real: IntArr, gen: IntArr, w_real: FloatArr | None = None,
                 w_gen: FloatArr | None = None) -> pd.DataFrame:
    """断片化要約の real / gen / 比。"""
    r = fragmentation_summary(real, w_real)
    g = fragmentation_summary(gen, w_gen)
    return pd.DataFrame({
        "metric": list(r.keys()),
        "real": [r[k] for k in r],
        "gen": [g[k] for k in r],
        "ratio": [g[k] / r[k] if r[k] != 0 else float("nan") for k in r],
    })


def compare_activity(real: IntArr, gen: IntArr, n_act: int, act_names: list[str],
                     w_real: FloatArr | None = None,
                     w_gen: FloatArr | None = None) -> pd.DataFrame:
    """活動別の時間シェアと日次参加率の real / gen / 差。"""
    sr, sg = time_share(real, n_act, w_real), time_share(gen, n_act, w_gen)
    pr, pg = participation(real, n_act, w_real), participation(gen, n_act, w_gen)
    return pd.DataFrame({
        "activity": act_names,
        "share_real": sr, "share_gen": sg, "share_abs_diff": np.abs(sr - sg),
        "part_real": pr, "part_gen": pg, "part_abs_diff": np.abs(pr - pg),
    })


# ============================================================
# 7b. ノイズ床 (null band) ★
# ============================================================
def null_band(sched_real: IntArr, stat_fn, n_boot: int = 300, seed: int = 0,
              w: FloatArr | None = None,
              alpha: float = 0.05) -> tuple[float, float, float]:
    """実データを復元抽出して統計量の (lo, median, hi) を返す＝ノイズ床。

    「gen と real の値が違う」だけでは差の証拠にならない。実データ自身を
    同じ N で取り直したときにどれだけ動くかを測り、その区間の外に出て初めて
    乖離と言える。

    ★ 床は「実データ側推定の精度」で作る。gen の N をいくら増やしても
      床は縮まない（縮むのは gen 側の推定誤差だけ）。
    ★ 基準は train を渡すこと。薄いホールドアウト (val N=373) を渡すと
      床が広がりすぎて何も検出できなくなる（実測で train の 2.8 倍の幅）。

    stat_fn: (sched, w) -> float、または (sched) -> float。
    """
    sched_real = np.asarray(sched_real)
    n = len(sched_real)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = _call_stat(stat_fn, sched_real[idx], None if w is None else np.asarray(w)[idx])
    lo, md, hi = np.quantile(vals, [alpha / 2, 0.5, 1 - alpha / 2])
    return float(lo), float(md), float(hi)


def null_band_2sample(sched_real: IntArr, stat_fn, n_gen: int, n_boot: int = 100,
                      seed: int = 0, w: FloatArr | None = None,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """2標本統計量（bigram_jsd / EMD など）のノイズ床 (lo, median, hi)。

    null_band は1標本統計量用で、real と gen の2つを取る統計量には使えない。
    ここが答えるのは「**生成器が完全だったら**、有限標本のゆらぎだけで
    どれだけの値が出るか」。観測 JSD がこの区間の中なら、乖離の証拠はない。

    ★ 実データ側と生成側の**両方**を実データから引き直す。片方（生成側）だけを
      引き直すと実データ側の推定誤差が床に入らず、床を過小に見積もる。
    ★ 実データの重み w は再抽出の確率として使い、抽出後は一様重みで評価する
      （生成側と同じ扱いにするため）。

    stat_fn: (real, gen) -> float
    """
    sched_real = np.asarray(sched_real)
    n_real = len(sched_real)
    p = _norm_w(n_real, w)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i_real = rng.choice(n_real, n_real, replace=True, p=p)
        i_gen  = rng.choice(n_real, n_gen,  replace=True, p=p)
        vals[b] = float(stat_fn(sched_real[i_real], sched_real[i_gen]))
    lo, md, hi = np.quantile(vals, [alpha / 2, 0.5, 1 - alpha / 2])
    return float(lo), float(md), float(hi)


def _call_stat(stat_fn, sched: IntArr, w: FloatArr | None) -> float:
    """stat_fn を (sched, w) でも (sched) でも呼べるようにする。"""
    try:
        return float(stat_fn(sched, w))
    except TypeError:
        return float(stat_fn(sched))


def compare_with_band(sched_real: IntArr, sched_gen: IntArr, stats: dict,
                      n_boot: int = 300, seed: int = 0,
                      w_real: FloatArr | None = None,
                      w_gen: FloatArr | None = None,
                      alpha: float = 0.05) -> pd.DataFrame:
    """複数の統計量を一度にノイズ床と突き合わせる。

    列: metric / real / ci_lo / ci_hi / gen / ratio / verdict
    verdict が "★床の外" の行だけを主張の根拠に使う。

    ★ リサンプルは1回だけ回して全統計量を同時に評価する（統計量ごとに
      n_boot 回まわすと指標数だけ遅くなるため）。

    stats: {表示名: stat_fn}。stat_fn は (sched, w) または (sched) を取り float を返す。
    """
    sched_real = np.asarray(sched_real)
    n = len(sched_real)
    rng = np.random.default_rng(seed)
    names = list(stats.keys())
    boot = np.empty((n_boot, len(names)), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = sched_real[idx]
        sub_w = None if w_real is None else np.asarray(w_real)[idx]
        for j, k in enumerate(names):
            boot[b, j] = _call_stat(stats[k], sub, sub_w)

    lo = np.quantile(boot, alpha / 2, axis=0)
    hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    real = np.array([_call_stat(stats[k], sched_real, w_real) for k in names])
    gen = np.array([_call_stat(stats[k], sched_gen, w_gen) for k in names])
    outside = (gen < lo) | (gen > hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(real != 0, gen / real, np.nan)
    return pd.DataFrame({
        "metric": names, "real": real, "ci_lo": lo, "ci_hi": hi,
        "gen": gen, "ratio": ratio,
        "verdict": np.where(outside, "★床の外", "床の内"),
    })


def group_reweight(group_d: IntArr, w_ref: FloatArr, d_ref: IntArr,
                   n_groups: int) -> FloatArr:
    """生成側の各行に「実データの群別重み付きシェア / 群内本数」の重みを与える。

    生成プールの群構成を実データの群構成へ合わせるだけで、群内の重みの
    ばらつきは再現しない（生成側に個人重みが無いため）。
    model.py:654-655 の sanity_check の w_gen と同じ作り方。
    """
    wr = np.asarray(w_ref, dtype=np.float64)
    wr = wr / wr.sum()
    share = np.array([wr[np.asarray(d_ref) == d].sum() for d in range(n_groups)])
    cnt = np.bincount(np.asarray(group_d), minlength=n_groups).astype(float)
    per = np.divide(share, cnt, out=np.zeros_like(share), where=cnt > 0)
    return per[np.asarray(group_d)]


# ============================================================
# 8. train / val 分割の再現
# ============================================================
def holdout_split(n: int, val_ratio: float = 0.1,
                  seed: int = 42) -> tuple[IntArr, IntArr]:
    """DDPM_Aggregate/model.py:212-229 make_loaders と同一の train/val 分割。

    ★ 必ず torch.randperm(n, generator=torch.Generator().manual_seed(seed)) を使う。
      numpy の RNG では別の並びになり、「学習に使っていない個票」という前提が
      静かに壊れる。N=3736 / val_ratio=0.1 で train 3363 / val 373。
    """
    import torch   # 学習側と疎結合に保つためローカル import

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * val_ratio)
    return perm[n_val:].astype(np.int64), perm[:n_val].astype(np.int64)


# ============================================================
# 8b. ばらつき (dispersion) ★
# ============================================================
# 標準の diversity 指標は 96スロット×12分類では飽和して使えない。実測（ATUS平日 train
# vs AggDDPM 生成 N=3,730）:
#     unique 率        real 1.0000 / gen 1.0000   <- 全個票が一意。検出力ゼロ
#     top1 占有        real 0.0003 / gen 0.0003   <- 同上
#     平均 Hamming     real 0.5804 / gen 0.5736   <- 1% しか違わない
# 「平均は合うが個人間のばらつきが縮む」タイプの歪みはこれらでは捕まらないので、
# 個人単位の要約量ごとに分散・cv・分位を出す形にする。
def _per_person_summaries(sched: IntArr, n_act: int) -> dict[str, FloatArr]:
    """個人単位のスカラー要約量を集める。dispersion_profile の土台。"""
    sched = np.asarray(sched)
    out: dict[str, FloatArr] = {"switches": n_switches(sched).astype(np.float64)}
    # 1人が1日に触れるユニーク活動数
    out["n_unique_act"] = np.array(
        [len(np.unique(row)) for row in sched], dtype=np.float64)
    for a in range(n_act):
        out[f"minutes[{a}]"] = (sched == a).sum(axis=1).astype(np.float64) * SLOT_MIN
        # 活動 a のエピソード本数（run の開始回数）。先頭スロットも開始として数える
        is_a = sched == a
        starts = is_a[:, 0].astype(np.int64) + (is_a[:, 1:] & ~is_a[:, :-1]).sum(axis=1)
        out[f"episodes[{a}]"] = starts.astype(np.float64)
    return out


def dispersion_profile(sched: IntArr, n_act: int, act_names: list[str],
                       w: FloatArr | None = None) -> pd.DataFrame:
    """個人単位の要約量ごとに 平均/分散/cv/分位 を出す。

    「平均は合っているのに分散が縮んでいる」を活動レベルで検出するための表。
    var / std は重み付き経験分布の母分散 (ddof=0)。switch_stats と同じ規約。
    """
    per = _per_person_summaries(sched, n_act)
    wn = _norm_w(len(sched), w)
    rows = []
    for key, x in per.items():
        mean = float((wn * x).sum())
        var = float((wn * (x - mean) ** 2).sum())
        q = weighted_quantile(x, [0.25, 0.5, 0.75, 0.9], w)
        label = key
        if "[" in key:
            kind, a = key.split("[")
            label = f"{kind}:{act_names[int(a.rstrip(']'))]}"
        rows.append({
            "quantity": label, "mean": mean, "var": var,
            "std": float(np.sqrt(var)),
            "cv": float(np.sqrt(var) / mean) if mean > 0 else float("nan"),
            "q25": float(q[0]), "median": float(q[1]),
            "q75": float(q[2]), "p90": float(q[3]),
        })
    return pd.DataFrame(rows)


def compare_dispersion(real: IntArr, gen: IntArr, n_act: int, act_names: list[str],
                       w_real: FloatArr | None = None,
                       w_gen: FloatArr | None = None) -> pd.DataFrame:
    """dispersion_profile の real / gen 比較。分散比 var_ratio が主役。"""
    r = dispersion_profile(real, n_act, act_names, w_real)
    g = dispersion_profile(gen, n_act, act_names, w_gen)
    return pd.DataFrame({
        "quantity": r["quantity"],
        "mean_real": r["mean"], "mean_gen": g["mean"],
        "mean_ratio": g["mean"] / r["mean"].replace(0, np.nan),
        "var_real": r["var"], "var_gen": g["var"],
        "var_ratio": g["var"] / r["var"].replace(0, np.nan),
        "cv_real": r["cv"], "cv_gen": g["cv"],
    })


def pairwise_distance_dist(sched: IntArr, sample_pairs: int = 20000,
                           seed: int = 0) -> dict:
    """個票間ハミング距離の分布（平均だけでなく分位と std）。

    平均だけだと real / gen が 1% しか違わず判別できない（モジュール冒頭のコメント）。
    分布が実データより狭ければ「似た個票ばかり」を意味する。
    """
    sched = np.asarray(sched)
    n = len(sched)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, sample_pairs)
    j = rng.integers(0, n, sample_pairs)
    ok = i != j                                    # 自分同士の対を除く
    d = (sched[i[ok]] != sched[j[ok]]).mean(axis=1)
    q = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(d.mean()), "std": float(d.std()),
            "p05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "p95": float(q[4]),
            "iqr": float(q[3] - q[1])}


def group_dispersion(sched: IntArr, groups: IntArr, n_groups: int,
                     sample_pairs: int = 4000, seed: int = 0) -> pd.DataFrame:
    """群内のばらつき。条件付けが強すぎて群内が潰れていないかを見る。"""
    sched, groups = np.asarray(sched), np.asarray(groups)
    rows = []
    for d in range(n_groups):
        m = groups == d
        if m.sum() < 2:
            continue
        sub = sched[m]
        pd_ = pairwise_distance_dist(sub, min(sample_pairs, m.sum() ** 2), seed)
        sw = n_switches(sub).astype(np.float64)
        rows.append({"group_d": d, "n": int(m.sum()),
                     "hamming_mean": pd_["mean"], "hamming_std": pd_["std"],
                     "switch_mean": float(sw.mean()), "switch_var": float(sw.var())})
    return pd.DataFrame(rows)


# ============================================================
# 8c. bigram（遷移）★
# ============================================================
def _bigram_counts(sched: IntArr, n_act: int, w: FloatArr | None,
                   exclude_diagonal: bool) -> FloatArr:
    """遷移 (cur, next) の重み付き頻度 (n_act, n_act)。bigram_matrix / _row_mass の土台。

    (cur, next) を1つの整数キーに畳んで bincount する。スロットごとの np.add.at
    ループより二桁速く、null_band_2sample が5万行のプールに対して現実的に回る。
    """
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    n_bnd = sched.shape[1] - 1
    pair = (sched[:, :-1].astype(np.int64) * n_act + sched[:, 1:]).ravel()
    M = np.bincount(pair, weights=np.repeat(wn, n_bnd),
                    minlength=n_act * n_act).reshape(n_act, n_act).astype(np.float64)
    if exclude_diagonal:
        np.fill_diagonal(M, 0.0)
    return M


def bigram_matrix(sched: IntArr, n_act: int, w: FloatArr | None = None,
                  exclude_diagonal: bool = True) -> FloatArr:
    """活動遷移 bigram の条件付き行列 (n_act, n_act)。行 i が P(next | cur=i)。

    境界は t = 1..95 の95本（n_switches と同じ土台）。日跨ぎ境界は含まない。

    ★ exclude_diagonal=True（既定）は対角＝「同じ活動に留まる」を除く。
      対角を含めると滞在確率（0.87 前後）が支配的で遷移構造が見えない。
      実測: 同じデータで bigram_jsd は 対角込み 0.0013 / 対角除く 0.0050 と
      桁が変わる。どちらの定義かを必ず明示すること。
    出現しない行は 0 のまま返す（正規化しない）。
    """
    M = _bigram_counts(sched, n_act, w, exclude_diagonal)
    row = M.sum(axis=1, keepdims=True)
    return np.divide(M, row, out=np.zeros_like(M), where=row > 0)


def bigram_jsd(real: IntArr, gen: IntArr, n_act: int,
               w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
               exclude_diagonal: bool = True, average: str = "row") -> float:
    """活動遷移 bigram 分布の JSD。

    average="row"  : 行ごとの条件付き分布の JSD を、実データ側の行の出現頻度で
                     重み付け平均する（既定）。docs の「bigram 分布の JSD 平均」。
    average="joint": 同時分布 (n_act^2) を平坦化して1つの JSD を取る。

    ★ docs/CLAUDE_EXPERIMENT.md の 0.065〜0.102 は擬似日本データでの別設定の値で、
      本関数の返り値と直接比較してはならない。
    """
    from scipy.spatial.distance import jensenshannon

    Pr = bigram_matrix(real, n_act, w_real, exclude_diagonal)
    Pg = bigram_matrix(gen, n_act, w_gen, exclude_diagonal)
    if average == "joint":
        # 行の重みを戻して同時分布にする
        wr = _row_mass(real, n_act, w_real, exclude_diagonal)
        wg = _row_mass(gen, n_act, w_gen, exclude_diagonal)
        jr = (Pr * wr[:, None]).ravel()
        jg = (Pg * wg[:, None]).ravel()
        return float(jensenshannon(jr, jg, base=2) ** 2)
    if average != "row":
        raise ValueError(f"average は 'row' か 'joint'。受け取った値: {average}")

    mass = _row_mass(real, n_act, w_real, exclude_diagonal)
    num, den = 0.0, 0.0
    for i in range(n_act):
        if Pr[i].sum() <= 0 or Pg[i].sum() <= 0:
            continue                       # 片方に出現しない行は平均から外す
        j = float(jensenshannon(Pr[i], Pg[i], base=2) ** 2)
        num += mass[i] * j
        den += mass[i]
    return num / den if den > 0 else float("nan")


def _row_mass(sched: IntArr, n_act: int, w: FloatArr | None,
              exclude_diagonal: bool) -> FloatArr:
    """bigram の行（現在の活動）ごとの出現質量。和1に正規化して返す。"""
    m = _bigram_counts(sched, n_act, w, exclude_diagonal).sum(axis=1)
    s = m.sum()
    return m / s if s > 0 else m


# ============================================================
# 8d. 暗記率（プライバシー / 過学習）★
# ============================================================
def nn_distances(gen: IntArr, ref: IntArr, k: int = 2, sample: int = 2000,
                 seed: int = 0, chunk: int = 2000) -> IntArr:
    """gen の各行から ref への上位 k 近傍ハミング距離 (N, k)。

    ref 全体との距離行列はメモリに乗らないので chunk 単位で走らせ、
    上位 k だけを保持する。距離は「一致しないスロット数」(0..96)。
    """
    gen, ref = np.asarray(gen), np.asarray(ref)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(gen), min(sample, len(gen)), replace=False)
    G = gen[idx]
    T = gen.shape[1]
    best = np.full((len(G), k), T + 1, dtype=np.int64)
    for s in range(0, len(ref), chunk):
        block = ref[s:s + chunk]
        d = (G[:, None, :] != block[None, :, :]).sum(axis=2)      # (len(G), c)
        merged = np.concatenate([best, d], axis=1)
        best = np.sort(merged, axis=1)[:, :k]
    return best


def memorization(gen: IntArr, train: IntArr, holdout: IntArr | None = None,
                 sample: int = 2000, seed: int = 0, near_thresh: int = 2) -> dict:
    """暗記の診断。train と holdout の両方に対して測り、その差を見る。

    ★ 要点は「gen が train にだけ有意に近いか」。両方に同じだけ近いなら
      それはデータ分布に近いだけで暗記ではない。
      DCR  = 最近傍距離（小さいほど暗記寄り）
      NNDR = 最近傍/第2近傍の比（小さいほど特定の1個票に張り付いている）
    """
    def _stats(ref, label):
        d = nn_distances(gen, ref, k=2, sample=sample, seed=seed)
        d1, d2 = d[:, 0].astype(np.float64), d[:, 1].astype(np.float64)
        nndr = np.divide(d1, d2, out=np.ones_like(d1), where=d2 > 0)
        keys = set(map(tuple, ref))
        exact = float(np.mean([tuple(r) in keys for r in gen]))
        return {
            f"exact_copy_rate[{label}]": exact,
            f"near_copy_rate(<={near_thresh})[{label}]": float((d1 <= near_thresh).mean()),
            f"DCR_mean[{label}]": float(d1.mean()),
            f"DCR_p05[{label}]": float(np.quantile(d1, 0.05)),
            f"DCR_median[{label}]": float(np.median(d1)),
            f"NNDR_mean[{label}]": float(nndr.mean()),
            f"NNDR_p05[{label}]": float(np.quantile(nndr, 0.05)),
        }

    out = _stats(train, "train")
    if holdout is not None:
        out.update(_stats(holdout, "holdout"))
        # train の方が近ければ暗記の兆候。holdout との差を明示する
        out["DCR_gap(holdout-train)"] = out["DCR_mean[holdout]"] - out["DCR_mean[train]"]
    return out


# ============================================================
# 9. 描画
# ============================================================
def setup_japanese_font() -> None:
    """日本語ラベルが豆腐にならないようフォントを設定する。"""
    plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Maru Gothic Pro",
                                   "YuGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def act_colors(n_act: int) -> tuple[list, ListedColormap]:
    """活動分類の色（tab20）。CVAE 版ノートブックと同じ色対応を保つ。"""
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(n_act)]
    return colors, ListedColormap(colors)


def _save(fig, path: str | Path | None) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")


def plot_timelines(seqs, titles: list[str], n_act: int, act_names: list[str],
                   path: str | Path | None = None, suptitle: str = "") -> None:
    """1人1行のスケジュール帯グラフ。行=個人, 列=スロット, 色=活動分類。"""
    colors, listed = act_colors(n_act)
    disp = np.stack([np.roll(s, roll_slots()) for s in seqs])   # 00:00 開始表示

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(seqs) + 1.8))
    ax.imshow(disp, aspect="auto", cmap=listed, vmin=0, vmax=n_act - 1,
              interpolation="nearest")
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(titles, fontsize=8)
    ax.set_xticks(np.arange(0, NUM_SLOTS + 1, 8) - 0.5)
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 2)])
    ax.set_xlim(-0.5, NUM_SLOTS - 0.5)
    ax.set_xlabel("時刻 (00:00開始表示)")
    ax.legend(handles=[Patch(color=colors[i], label=act_names[i]) for i in range(n_act)],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.set_title(suptitle)
    fig.tight_layout()
    _save(fig, path)
    plt.show()


def plot_real_vs_gen(real_seqs, gen_seqs, n_act: int, act_names: list[str],
                     path: str | Path | None = None, title: str = "") -> None:
    """実個票（上）と生成個票（下）を同一図に並べる。行ラベルに切替回数を出す。"""
    colors, listed = act_colors(n_act)
    real_seqs, gen_seqs = np.asarray(real_seqs), np.asarray(gen_seqs)
    seqs = np.concatenate([real_seqs, gen_seqs], axis=0)
    sw = n_switches(seqs)
    n_r = len(real_seqs)
    titles = ([f"real #{i} (sw={s})" for i, s in enumerate(sw[:n_r])]
              + [f"gen  #{i} (sw={s})" for i, s in enumerate(sw[n_r:])])

    disp = np.stack([np.roll(s, roll_slots()) for s in seqs])
    fig, ax = plt.subplots(figsize=(12, 0.35 * len(seqs) + 1.8))
    ax.imshow(disp, aspect="auto", cmap=listed, vmin=0, vmax=n_act - 1,
              interpolation="nearest")
    ax.axhline(n_r - 0.5, color="k", lw=2)          # real / gen の境界
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(titles, fontsize=8)
    ax.set_xticks(np.arange(0, NUM_SLOTS + 1, 8) - 0.5)
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 2)])
    ax.set_xlim(-0.5, NUM_SLOTS - 0.5)
    ax.set_xlabel("時刻 (00:00開始表示)")
    ax.legend(handles=[Patch(color=colors[i], label=act_names[i]) for i in range(n_act)],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    plt.show()


def plot_switch_hist(real: IntArr, gen: IntArr, w_real: FloatArr | None = None,
                     w_gen: FloatArr | None = None, path: str | Path | None = None,
                     title: str = "切替回数の分布") -> None:
    """切替回数のヒストグラム（共通bin）と ECDF の2パネル。平均に縦線を引く。"""
    sr = n_switches(real).astype(np.float64)
    sg = n_switches(gen).astype(np.float64)
    wr, wg = _norm_w(len(sr), w_real), _norm_w(len(sg), w_gen)
    hi = int(max(sr.max(), sg.max()))
    bins = np.arange(-0.5, hi + 1.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(sr, bins=bins, weights=wr, alpha=0.55, label="real", color="tab:blue")
    ax.hist(sg, bins=bins, weights=wg, alpha=0.55, label="gen", color="tab:orange")
    ax.axvline(float((wr * sr).sum()), color="tab:blue", ls="--", lw=1.5)
    ax.axvline(float((wg * sg).sum()), color="tab:orange", ls="--", lw=1.5)
    ax.set_xlabel("1日の切替回数"); ax.set_ylabel("割合")
    ax.set_title("ヒストグラム (破線 = 平均)"); ax.legend()

    ax = axes[1]
    for s, w, lab, c in ((sr, wr, "real", "tab:blue"), (sg, wg, "gen", "tab:orange")):
        o = np.argsort(s)
        ax.step(s[o], np.cumsum(w[o]), where="post", label=lab, color=c)
    ax.set_xlabel("1日の切替回数"); ax.set_ylabel("累積割合")
    ax.set_title("ECDF"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    plt.show()


def plot_switch_by_band(real: IntArr, gen: IntArr, w_real: FloatArr | None = None,
                        w_gen: FloatArr | None = None, band_hours: int = 2,
                        path: str | Path | None = None,
                        title: str = "時間帯別 切替率") -> None:
    """境界あたり切替率の折れ線（real vs gen）。深夜帯 00:00-04:00 を網掛け＋対数軸で拡大。"""
    br = switches_by_band(real, w_real, band_hours)
    bg = switches_by_band(gen, w_gen, band_hours)
    x = br["band_start_hour"].to_numpy()

    # 網掛けは深夜帯 00:00-04:00 の帯だけを囲む。点は帯の開始時刻に置かれるので、
    # 右端は 02:00-04:00 と 04:00-06:00 の中点にする（04:00 の点にかけない）
    night_right = 4 - band_hours / 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, logy in ((axes[0], False), (axes[1], True)):
        ax.plot(x, br["switch_rate"], "o-", label="real", color="tab:blue")
        ax.plot(x, bg["switch_rate"], "s-", label="gen", color="tab:orange")
        ax.axvspan(-band_hours / 2, night_right, color="0.85", zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(br["label"], rotation=60, fontsize=7)
        ax.set_ylabel("境界1本あたりの切替確率")
        ax.grid(alpha=0.3); ax.legend()
        if logy:
            ax.set_yscale("log")
            ax.set_title("同（対数軸）: 深夜帯の差を拡大")
        else:
            ax.set_title("網掛け = 00:00-04:00 (slot 80..95)")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    plt.show()


def plot_duration(real: IntArr, gen: IntArr, n_act: int, act_names: list[str],
                  w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                  path: str | Path | None = None,
                  title: str = "活動別 継続時間") -> None:
    """上: 活動別エピソード長の箱ひげ図（real/gen 並置）。下: 行動者平均時間の棒グラフ。"""
    er = episode_lengths_by_act(real, n_act)
    eg = episode_lengths_by_act(gen, n_act)
    dr = duration_table(real, n_act, act_names, w_real)
    dg = duration_table(gen, n_act, act_names, w_gen)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    ax = axes[0]
    pos = np.arange(n_act)
    b1 = ax.boxplot([e * SLOT_MIN if len(e) else [0] for e in er],
                    positions=pos - 0.18, widths=0.32, patch_artist=True, showfliers=False)
    b2 = ax.boxplot([e * SLOT_MIN if len(e) else [0] for e in eg],
                    positions=pos + 0.18, widths=0.32, patch_artist=True, showfliers=False)
    for b, c in ((b1, "tab:blue"), (b2, "tab:orange")):
        for p in b["boxes"]:
            p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_xticks(pos); ax.set_xticklabels(act_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("エピソード長 (分)")
    ax.set_title("エピソード長の分布 (青=real, 橙=gen, 外れ値非表示)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(pos - 0.18, dr["actor_mean_min"], width=0.32, label="real", color="tab:blue", alpha=0.75)
    ax.bar(pos + 0.18, dg["actor_mean_min"], width=0.32, label="gen", color="tab:orange", alpha=0.75)
    ax.set_xticks(pos); ax.set_xticklabels(act_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("行動者平均時間 (分)")
    ax.set_title("行動者平均時間 = その活動を1回以上した人の1日合計時間の平均")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    plt.show()
