"""
NHTS 2022 トリップ列 -> 96スロット活動状態列 への変換

仕様:
    - 活動状態 10カテゴリ (WHYTRP1S 9分類 + 移動)
    - 1日は午前4:00起点 (経過分 0..1439)。STRTTIME/ENDTIME(時計時刻)を経過分に変換
    - 1分解像度で構築 -> 15分スロットを多数決で離散化 -> 長さ96
    - 初期活動(最初のトリップ前)は WHYFROM を優先して推定 (なければ在宅)。
    - クレンジング: 欠損コードを持つ個人を除外 / WHYTO-WHYFROM 矛盾を検出。
"""

import numpy as np
import pandas as pd
from enum import IntEnum

N_SLOTS = 96  # 24時間 * 60分 / 15分
SLOT_MIN = 15  # スロットの長さ(分)
DAY_MIN = 1440  # 1日の長さ(分)
DAY_START_MIN = 240  # 4:00 AM(240分)

# 活動カテゴリの定義
class Act(IntEnum):
    HOME=0
    WORK=1
    SCHOOL=2
    MEDICAL=3
    SHOP=4
    SOCIAL=5
    TRANSPORT=6
    MEALS=7
    OTHER=8
    TRAVEL=9
NUM_ACT = len(Act)

# WHYTRP1S(9分類) -> 活動状態。移動(9)はトリップ区間用なので含めない。
WHYTRP1S_TO_ACT = {
    1: Act.HOME, 10: Act.WORK, 20: Act.SCHOOL, 30: Act.MEDICAL,
    40: Act.SHOP, 50: Act.SOCIAL, 70: Act.TRANSPORT, 80: Act.MEALS, 97: Act.OTHER,
}

# NHTS 欠損/非該当コード
MISSING = {-1, -7, -8, -9}


def clock_to_daymin(t):
    """時計時刻(HHMM, 0000..2359) -> 4:00起点の経過分(0..1439)。"""
    t = int(t)
    m = (t // 100) * 60 + (t % 100)  # minutes = hh * 60 + mm
    return m - DAY_START_MIN if m >= DAY_START_MIN else m + DAY_MIN - DAY_START_MIN  # 深夜(0:00..3:59) or 4:00..23:59


def build_one(person_trips: pd.DataFrame) -> tuple[np.ndarray | None, str | None, int]:
    """1人のトリップ列(SEQ_TRIPID順, DataFrame) -> (sched[96] or None, reason or None, lost_short)。
        reason が None でなければ除外対象(その場合 sched は None)。"""
    df = person_trips.sort_values("SEQ_TRIPID")  # df: 1人分のトリップ列(SEQ_TRIPID順)

    # クレンジング: 時刻の欠損
    if df["STRTTIME"].isin(MISSING).any() or df["ENDTIME"].isin(MISSING).any():
        return None, "missing_time", 0

    # 経過分へ変換
    starts = df["STRTTIME"].map(clock_to_daymin).to_numpy()  # 出発時刻
    ends   = df["ENDTIME"].map(clock_to_daymin).to_numpy()  # 到着時刻
    why    = df["WHYTRP1S"].to_numpy()  # トリップの目的(WHYTRP1S)
    whyfrom= df["WHYFROM"].to_numpy()  # 出発活動(WHYFROM)
    whyto  = df["WHYTO"].to_numpy()  # 到着活動(WHYTO)
    n = len(df)  # トリップ数

    # クレンジング: WHYTO(前) と WHYFROM(次) の矛盾検出
    # トリップi の到着活動(WHYTO) と トリップi+1 の出発活動(WHYFROM) は同じ場所であるべき
    for i in range(n - 1):
        a, b = whyto[i], whyfrom[i+1]
        if a in MISSING or b in MISSING:
            continue
        if a != b:
            return None, "whyto_whyfrom_mismatch", 0

    # 初期活動の推定: 最初のトリップの WHYFROM を優先 
    first_from = whyfrom[0]
    if first_from in MISSING or first_from == 1:
        init_act = Act.HOME
    else:
        init_act = WHYTRP1S_TO_ACT.get(int(first_from), Act.OTHER)

    # 1分解像度タイムライン構築 
    tl = np.full(DAY_MIN, int(init_act), dtype=np.int8)  # 4:00起点の経過分(0..1439)をインデックスとする活動状態列 (初期値は最初の活動)
    for i in range(n):
        s, e = int(starts[i]), int(ends[i])
        act = int(WHYTRP1S_TO_ACT.get(int(why[i]), Act.OTHER))
        # トリップ区間を移動で埋める(範囲内のみ)
        if 0 <= s < e <= DAY_MIN:
            tl[s:e] = int(Act.TRAVEL)
        elif 0 <= s < DAY_MIN:  # 端のクリップ
            tl[s:min(e, DAY_MIN)] = int(Act.TRAVEL)
        # 到着後〜次トリップ出発(または1日の終わり)を到着活動で埋める
        nxt = int(starts[i+1]) if i + 1 < n else DAY_MIN
        nxt = max(nxt, min(e, DAY_MIN))
        e_clip = min(e, DAY_MIN)
        if e_clip < nxt <= DAY_MIN:
            tl[e_clip:nxt] = act

    # 15分スロットへ多数決離散化
    sched = np.empty(N_SLOTS, dtype=np.int8)
    lost_short = 0  # 離散化で消えた短時間活動
    for k in range(N_SLOTS):
        seg = tl[k*SLOT_MIN:(k+1)*SLOT_MIN]
        counts = np.bincount(seg, minlength=NUM_ACT)  # 15分セグメント内で出てきた各Actの数をカウント
        sched[k] = counts.argmax()  # 一番多いActをこのセグメントのスロットに登録
        # スロット内に複数活動があり、少数派が総取りで消えた数を数える
        if (counts > 0).sum() > 1:
            lost_short += int(counts.sum() - counts.max())
    return sched, None, lost_short


# How to pre-process
def main():
    print("loading trip data ...")
    trip = pd.read_csv("../../data/raw/NHTS2022/tripv2pub.csv",
                        usecols=["HOUSEID","PERSONID","SEQ_TRIPID","STRTTIME","ENDTIME",
                                "WHYTRP1S","WHYTO","WHYFROM","FRSTHM"])
    grouped = trip.groupby(["HOUSEID","PERSONID"], sort=False)
    print(f"persons with trips: {grouped.ngroups}")

    schedules, keys = [], []
    reasons = {"missing_time":0, "whyto_whyfrom_mismatch":0}  # クレンジング理由のカウンタ
    total_lost_short = 0

    for (hh, pid), g in grouped:
        sched, reason, lost = build_one(g)

        # クレンジング
        if reason is not None:
            reasons[reason] += 1
            continue

        schedules.append(sched)
        keys.append((hh, pid))
        total_lost_short += lost

    S = np.stack(schedules)
    K = pd.DataFrame(keys, columns=["HOUSEID","PERSONID"])
    np.save("../../data/processed/schedules.npy", S)
    K.to_csv("../../data/processed/schedule_keys.csv", index=False)

    print("\n=== 結果サマリ ===")
    print(f"採用: {len(schedules)} 人  形状: {S.shape}")
    print(f"除外(時刻欠損): {reasons['missing_time']}")
    print(f"除外(WHYTO/WHYFROM矛盾): {reasons['whyto_whyfrom_mismatch']}")
    print(f"離散化で吸収された短時間活動(延べ分): {total_lost_short}")

    # 妥当性チェック: 時間帯別の活動構成(全体)
    print("\n=== 時間帯別 活動構成(列=4:00起点スロット, 抜粋) ===")
    for label, k in [("06:00", 8), ("09:00", 20), ("12:00", 32), ("14:00", 40), ("18:00", 56), ("22:00", 72)]:
        col = S[:, k]
        dist = np.bincount(col, minlength=NUM_ACT) / len(col)
        top = sorted([(Act(i).name, dist[i]) for i in range(NUM_ACT)], key=lambda x:-x[1])[:3]
        print(f"  {label}: " + ", ".join(f"{n}={p:.0%}" for n,p in top))

if __name__ == "__main__":
    main()