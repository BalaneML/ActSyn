"""
feasibility.py
================
活動スケジュール個票の「構造的にありえるか」の診断。

なぜ別モジュールか:
    集計 JSD が一致していても非現実的な個人は隠れる
    （docs/CLAUDE_EXPERIMENT.md G3 / Shone & Hillel）。分布の一致とは別の軸なので
    individual_metrics.py（分布指標）から分ける。

★ 判定の考え方:
    ゼロ違反を要求しない。実データにも「TRAVEL が1本だけ」の個票が 7.7% ある。
    実データでの発生率を基準にし、生成側がそれをどれだけ超えるかで見る。
    閾値は下の定数にまとめ、実測値を根拠としてコメントに書く。

★ 共通12分類には HOME が無い:
    notebooks/eval_cvae/05_selfcontained_eval.ipynb の anomaly_rate は NHTS 10分類で
    HOME=0 / TRAVEL=9 を固定前提にしていた。共通12分類では在宅活動が
    SLEEP_PERSONAL / HOUSEWORK / LEISURE_SOCIAL などに散るため「HOME に始まり
    HOME に終わる」ルールは移植できない。代わりに
    「移動が往復で閉じているか」と「睡眠が細切れにされていないか」を見る。

使い方:
    uv run python src/eval/test_feasibility.py

    fe = <importlib で src/eval/feasibility.py をロード>
    fe.travel_pairing(sched)
    fe.night_intrusion(sched)
    fe.feasibility_summary(sched, w)
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

NUM_SLOTS = 96
SLOT_MIN = 15
START_HOUR = 4

# 共通12分類 (crosswalk_atus_stula.Common) のうち本モジュールが名前で参照するもの
SLEEP_PERSONAL = 0
TRAVEL = 7

# --- 閾値（実 ATUS 平日 train N=3,363 での実測値を根拠に置く） ---
# 睡眠に挟まれた別活動を「割り込み」とみなす最大長。2スロット = 30分。
# 夜中の授乳・トイレなどは実データにもあるので、率の比較で見る
NIGHT_INTRUSION_MAX_SLOTS = 2
# 深夜帯の定義。スケジュールは 04:00 開始なので 00:00-04:00 = slot 80..95
NIGHT_START_HOUR, NIGHT_END_HOUR = 0, 4
# 連続 TRAVEL がこれを超えたら異常（実データの TRAVEL エピソード平均は 30.8 分、
# p90 で 60 分。180 分 = 12 スロットは長距離移動を許容する緩い上限）
MAX_TRAVEL_MIN = 180

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]


def _norm_w(n: int, w: FloatArr | None) -> FloatArr:
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    if len(w) != n:
        raise ValueError(f"weights の長さ {len(w)} が個票数 {n} と一致しない")
    return w / w.sum()


def _night_slots() -> IntArr:
    """深夜帯のスロット番号。04:00 開始なので 00:00-04:00 は slot 80..95。"""
    clock = (START_HOUR * 60 + np.arange(NUM_SLOTS) * SLOT_MIN) % (24 * 60)
    return np.where((clock >= NIGHT_START_HOUR * 60) & (clock < NIGHT_END_HOUR * 60))[0]


def episode_count_by_person(sched: IntArr, act: int) -> IntArr:
    """個人ごとの活動 act のエピソード本数 (N,)。"""
    sched = np.asarray(sched)
    is_a = sched == act
    return (is_a[:, 0].astype(np.int64)
            + (is_a[:, 1:] & ~is_a[:, :-1]).sum(axis=1)).astype(np.int64)


# ============================================================
# 1. 移動の往復対応 ★（Step 1 で見つかった最大の欠陥）
# ============================================================
def travel_pairing(sched: IntArr, w: FloatArr | None = None,
                   travel_act: int = TRAVEL) -> pd.DataFrame:
    """移動エピソード本数の分布。往復が対になっているかを見る。

    実データでは移動は往復で対になるため 2本が最頻で、1本だけは稀。
    実 ATUS 平日 train の実測:
        0本 0.256 / 1本 0.077 / 2本 0.332 / 3本 0.145 / 4本以上 0.190
    AggDDPM 生成では 1本だけが 0.239（3.1倍）＝「出かけたまま帰ってこない」個票。

    返り値の行: n_travel = 0,1,2,3,"4+" と、要約行 odd / mean / participation。
    """
    c = episode_count_by_person(sched, travel_act)
    wn = _norm_w(len(c), w)
    rows = []
    for k in range(4):
        rows.append({"n_travel": str(k), "share": float(wn[c == k].sum())})
    rows.append({"n_travel": "4+", "share": float(wn[c >= 4].sum())})
    return pd.DataFrame(rows)


def travel_pairing_summary(sched: IntArr, w: FloatArr | None = None,
                           travel_act: int = TRAVEL) -> dict:
    """travel_pairing のスカラー要約。ノイズ床（compare_with_band）に渡せる形。"""
    c = episode_count_by_person(sched, travel_act)
    wn = _norm_w(len(c), w)
    moved = c > 0
    return {
        "travel_single_rate": float(wn[c == 1].sum()),      # ★主指標
        "travel_zero_rate": float(wn[c == 0].sum()),
        "travel_odd_rate": float(wn[moved & (c % 2 == 1)].sum()),
        "travel_mean": float((wn * c).sum()),
        "travel_participation": float(wn[moved].sum()),
        "travel_actor_mean": (float((wn[moved] * c[moved]).sum() / wn[moved].sum())
                              if moved.any() else 0.0),
    }


# ============================================================
# 2. 睡眠への割り込み ★
# ============================================================
def night_intrusion(sched: IntArr, w: FloatArr | None = None,
                    sleep_act: int = SLEEP_PERSONAL,
                    max_slots: int = NIGHT_INTRUSION_MAX_SLOTS) -> dict:
    """深夜帯で睡眠に挟まれた短い別活動の出現率。

    Step 1 の課題3（深夜帯の切替 +15%）の機序。深夜帯の切替で最も増えたのは
    MEALS->SLEEP_PERSONAL（実 0.0039 → 生成 0.0123 回/人、3.2倍）で、
    「睡眠の途中に短い食事が割り込む」形をしている。

    判定: 深夜スロットのうち、前後がともに sleep_act で長さ <= max_slots の
    別活動エピソードを1つでも持つ個票の割合。
    """
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    night = set(_night_slots().tolist())
    hit = np.zeros(len(sched), dtype=bool)
    n_intr = np.zeros(len(sched), dtype=np.int64)

    for i, row in enumerate(sched):
        s = 0
        for t in range(1, NUM_SLOTS + 1):
            if t == NUM_SLOTS or row[t] != row[s]:
                length = t - s
                if (row[s] != sleep_act and length <= max_slots
                        and s - 1 >= 0 and t < NUM_SLOTS
                        and row[s - 1] == sleep_act and row[t] == sleep_act
                        and any(u in night for u in range(s, t))):
                    hit[i] = True
                    n_intr[i] += 1
                s = t
    return {
        "night_intrusion_rate": float((wn * hit).sum()),
        "night_intrusion_per_person": float((wn * n_intr).sum()),
    }


# ============================================================
# 3. まとめ
# ============================================================
def feasibility_summary(sched: IntArr, w: FloatArr | None = None) -> dict:
    """ルール別の違反率と「いずれか該当」率。

    ゼロ違反は要求しない。実データでの発生率と比べるための表。
    """
    sched = np.asarray(sched)
    wn = _norm_w(len(sched), w)
    night = _night_slots()

    flags: dict[str, npt.NDArray[np.bool_]] = {}

    # 移動が奇数本（往復が閉じていない）。0本は「移動しない日」なので除く
    c_tr = episode_count_by_person(sched, TRAVEL)
    flags["travel_odd"] = (c_tr > 0) & (c_tr % 2 == 1)

    # 長すぎる連続 TRAVEL
    max_tr = np.zeros(len(sched), dtype=np.int64)
    for i, row in enumerate(sched):
        run, best = 0, 0
        for t in range(NUM_SLOTS):
            run = run + 1 if row[t] == TRAVEL else 0
            best = max(best, run)
        max_tr[i] = best
    flags[f"long_travel(>{MAX_TRAVEL_MIN}min)"] = max_tr > MAX_TRAVEL_MIN // SLOT_MIN

    # 深夜帯に睡眠が一度も現れない
    flags["no_sleep_at_night"] = ~(sched[:, night] == SLEEP_PERSONAL).any(axis=1)

    # 日跨ぎ非整合（04:00 の活動が日の始まりと終わりで食い違う）
    flags["wrap_mismatch"] = sched[:, -1] != sched[:, 0]

    out = {k: float((wn * v).sum()) for k, v in flags.items()}

    # 睡眠への割り込み（個票単位のフラグを再計算せず night_intrusion を使う）
    ni = night_intrusion(sched, w)
    out["night_intrusion"] = ni["night_intrusion_rate"]

    any_flag = np.zeros(len(sched), dtype=bool)
    for v in flags.values():
        any_flag |= v
    out["ANY"] = float((wn * any_flag).sum())
    out.update(travel_pairing_summary(sched, w))
    return out
