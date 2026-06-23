"""
NHTS 2022 前処理パイプライン
NHTS生データ -> 活動スケジュール(96スロット) + 条件ベクトル(5属性) -> 統合CSV

処理の流れ:
    1. トリップ列 -> 96スロット活動状態列 (トリップ保有者)
    2. トリップ0の個人 -> 全96スロット在宅 (終日在宅者)
    3. 条件ベクトル (5属性) を全個人について構築
    4. 役割「その他」を除外し、条件とスケジュールを統合してCSV保存

スケジュール仕様:
    - 活動状態 10カテゴリ (WHYTRP1S 9分類 + 移動)
    - 1日は午前4:00起点 (経過分 0..1439), STRTTIME/ENDTIME(時計時刻)を経過分に変換
    - 1分解像度で構築 -> 15分スロットを多数決で離散化 -> 長さ96
    - 初期活動 (最初のトリップ前) は WHYFROM を優先して推定 (なければ在宅)
    - クレンジング: 欠損コードを持つ個人を除外 / WHYTO-WHYFROM 矛盾を検出
    - 終日在宅者 (トリップ0) は全スロット在宅として追加

条件5属性:
    age                : R_AGE (5..92)
    gender             : R_SEX (欠損<0 は R_SEX_IMP で補完), 1=male, 2=female -> {0:male,1:female}
    num_member         : HHSIZE (1..10)
    role_household_type: 8分類, 世帯単位で R_RELAT と Spouse有無, R_SEX から判定 ※改良すべき
    worker_status      : WORKER==1 -> 1(就業), それ以外(2 および -1=子供非該当) -> 0(非就業)

role_household_type の 8 カテゴリ(index):
    0 単独世帯(男性)  1 単独世帯(女性) 2 夫・男親  3 妻・女親
    4 子供(男性)     5 子供(女性)     6 親(男性) 7 親(女性)
    ※ Brother/Sister(4),Other relative(5),Not related(6) は -1(その他) とし除外
"""

import os
from enum import IntEnum
import numpy as np
import pandas as pd


# ===== パス =====
TRIP_PATH      = "../../data/raw/NHTS2022/tripv2pub.csv"
PERSON_PATH    = "../../data/raw/NHTS2022/perv2pub.csv"
HOUSEHOLD_PATH = "../../data/raw/NHTS2022/hhv2pub.csv"
OUT_DIR        = "../../data/processed"

# ===== スケジュール定数 =====
N_SLOTS = 96         # 24時間 * 60分 / 15分
SLOT_MIN = 15        # スロットの長さ(分)
DAY_MIN = 1440       # 1日の長さ(分)
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

# WHYTRP1S(9分類) -> 活動状態, 移動(9)はトリップ区間用なので含めない
WHYTRP1S_TO_ACT = {
    1: Act.HOME, 10: Act.WORK, 20: Act.SCHOOL, 30: Act.MEDICAL,
    40: Act.SHOP, 50: Act.SOCIAL, 70: Act.TRANSPORT, 80: Act.MEALS, 97: Act.OTHER,
}

# NHTS 欠損/非該当コード
MISSING = {-1, -7, -8, -9}

# ===== 条件ベクトル定数 =====
# R_RELAT コード
SELF, SPOUSE, CHILD, PARENT = 7, 1, 2, 3

# TODO: 3世代の家族構成の時，第1世代 or 第3世代が世帯主のときうまく動作しない
ROLE = {
    "single_m":0, "single_f":1, "husband":2, "wife":3,
    "child_m":4, "child_f":5, "parent_m":6, "parent_f":7, "other":-1,
}


# ==================================================================
# スケジュール構築
# ==================================================================
def clock_to_daymin(t):
    """時計時刻(HHMM, 0000..2359) -> 4:00起点の経過分(0..1439)"""
    t = int(t)
    m = (t // 100) * 60 + (t % 100)  # minutes = hh * 60 + mm
    return m - DAY_START_MIN if m >= DAY_START_MIN else m + DAY_MIN - DAY_START_MIN


def build_one(person_trips: pd.DataFrame) -> tuple[np.ndarray | None, str | None, int]:
    """ 1人のトリップ列 (SEQ_TRIPID順, DataFrame) -> (sched[96] or None, reason or None, lost_short)
        reason が None でなければ除外対象 (その場合 sched は None)"""
    df = person_trips.sort_values("SEQ_TRIPID")

    # クレンジング: 時刻の欠損
    if df["STRTTIME"].isin(MISSING).any() or df["ENDTIME"].isin(MISSING).any():
        return None, "missing_time", 0

    # 経過分へ変換
    starts = df["STRTTIME"].map(clock_to_daymin).to_numpy()
    ends   = df["ENDTIME"].map(clock_to_daymin).to_numpy()
    why    = df["WHYTRP1S"].to_numpy()
    whyfrom= df["WHYFROM"].to_numpy()
    whyto  = df["WHYTO"].to_numpy()
    n = len(df)

    # クレンジング: WHYTO(前) と WHYFROM(次) の矛盾検出
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
    tl = np.full(DAY_MIN, int(init_act), dtype=np.int8)
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
        counts = np.bincount(seg, minlength=NUM_ACT)
        sched[k] = counts.argmax()
        if (counts > 0).sum() > 1:
            lost_short += int(counts.sum() - counts.max())
    return sched, None, lost_short


def build_schedules(trip: pd.DataFrame):
    """ 
    トリップ保有者全員のスケジュールを構築
    -> (S[trip保有者数, 96], keys_df, reasons, total_lost
    """
    grouped = trip.groupby(["HOUSEID","PERSONID"], sort=False)
    schedules, keys = [], []
    reasons = {"missing_time":0, "whyto_whyfrom_mismatch":0}
    total_lost = 0
    for (hh, pid), g in grouped:
        sched, reason, lost = build_one(g)
        if reason is not None:
            reasons[reason] += 1
            continue
        schedules.append(sched)
        keys.append((hh, pid))
        total_lost += lost
    S = np.stack(schedules)
    K = pd.DataFrame(keys, columns=["HOUSEID","PERSONID"])
    return S, K, reasons, total_lost


# ==================================================================
# 条件ベクトル構築
# ==================================================================
def resolve_gender(r_sex, r_sex_imp):
    s = r_sex if r_sex not in MISSING else r_sex_imp
    # 1=male->0, 2=female->1
    return 0 if s == 1 else 1


def build_conditions(keys_df: pd.DataFrame, per: pd.DataFrame, hh: pd.DataFrame) -> pd.DataFrame:
    """keys_df (HOUSEID,PERSONID) の各個人について5属性を構築して返す"""
    df = keys_df.merge(per, on=["HOUSEID","PERSONID"], how="left")  # keysに含まれる個人のみ
    df = df.merge(hh, on="HOUSEID", how="left")

    # 世帯ごとに Spouse が存在するか(role判定に必要)
    spouse_present = (
        df.assign(is_sp=(df["R_RELAT"]==SPOUSE))
            .groupby("HOUSEID")["is_sp"].any()
    )
    df["spouse_in_hh"] = df["HOUSEID"].map(spouse_present)


    # TODO: 3世代の端っこ世帯主を含む世帯に対応する必要がある
    # 世帯内役割推定アルゴリズム
    def role_of(row):
        rel = int(row["R_RELAT"])  # 続柄
        g = resolve_gender(int(row["R_SEX"]), int(row["R_SEX_IMP"]))  # 性別 (0=m,1=f)
        hhsize = int(row["HHSIZE"])  # 世帯構成人数
        if rel == SELF or rel == SPOUSE:
            if hhsize == 1:
                # 単独世帯
                return ROLE["single_m"] if g==0 else ROLE["single_f"]
            else:
                # 複数人世帯: 夫婦なら husband/wife、Spouseなしの self はひとり親として吸収
                return ROLE["husband"] if g==0 else ROLE["wife"]
        elif rel == CHILD:
            return ROLE["child_m"] if g==0 else ROLE["child_f"]
        elif rel == PARENT:
            return ROLE["parent_m"] if g==0 else ROLE["parent_f"]
        else:  # 4,5,6,-9 など
            return ROLE["other"]

    
    # 条件ベクトル構築
    df["age"] = df["R_AGE"].astype(int)
    df["gender"] = [resolve_gender(int(s), int(si)) for s,si in zip(df["R_SEX"].astype(int), df["R_SEX_IMP"].astype(int))]
    df["num_member"] = df["HHSIZE"].astype(int)
    df["role_household_type"] = df.apply(role_of, axis=1)
    df["worker_status"] = (df["WORKER"]==1).astype(int)

    cond = df[["HOUSEID","PERSONID","age","gender","num_member","role_household_type","worker_status"]]
    return cond


# ==================================================================
# メイン: 全処理の統合
# ==================================================================
def main():
    print("loading ...")
    trip = pd.read_csv(TRIP_PATH,
                        usecols=["HOUSEID","PERSONID","SEQ_TRIPID","STRTTIME","ENDTIME",
                                "WHYTRP1S","WHYTO","WHYFROM","FRSTHM"])
    per = pd.read_csv(PERSON_PATH,
                        usecols=["HOUSEID","PERSONID","R_AGE","R_SEX","R_SEX_IMP","WORKER","R_RELAT"])
    hh = pd.read_csv(HOUSEHOLD_PATH, usecols=["HOUSEID","HHSIZE"])

    # --- 1. トリップ保有者のスケジュール構築 ---
    S_trip, keys_trip, reasons, total_lost = build_schedules(trip)
    print(f"persons with trips: {len(keys_trip) + sum(reasons.values())}")
    print(f"  採用(トリップ保有): {len(keys_trip)}  除外: {reasons}")

    # --- 2. トリップ0の個人 -> 終日在宅(全スロット在宅) ---
    per_keys = per[["HOUSEID","PERSONID"]].drop_duplicates()
    trip_all = trip[["HOUSEID","PERSONID"]].drop_duplicates()
    m = per_keys.merge(trip_all, on=["HOUSEID","PERSONID"], how="left", indicator=True)
    keys_home = m[m["_merge"]=="left_only"][["HOUSEID","PERSONID"]].reset_index(drop=True)
    S_home = np.full((len(keys_home), N_SLOTS), int(Act.HOME), dtype=np.int8)
    print(f"  終日在宅(トリップ0): {len(keys_home)}")

    # --- 3. 条件ベクトル構築(トリップ保有 + 終日在宅 まとめて) ---
    keys_all = pd.concat([keys_trip, keys_home], ignore_index=True)
    S_all = np.vstack([S_trip, S_home])
    cond_all = build_conditions(keys_all, per, hh).reset_index(drop=True)
    assert len(cond_all) == len(S_all) == len(keys_all)

    # --- 4. 役割「その他」を除外して統合 ---
    mask = (cond_all["role_household_type"] != -1).to_numpy()
    cond_all = cond_all[mask].reset_index(drop=True)
    S_all = S_all[mask]
    print(f"  役割「その他」除外後: {len(cond_all)}")

    # 行対応の検証(条件とスケジュールが同一個人・同一順序であること)
    keys_after = keys_all[mask].reset_index(drop=True)
    assert (cond_all[["HOUSEID","PERSONID"]].to_numpy()
            == keys_after[["HOUSEID","PERSONID"]].to_numpy()).all()

    # --- 統合CSV保存 ---
    slot_cols = [f"s{k}" for k in range(N_SLOTS)]
    S_df = pd.DataFrame(S_all, columns=slot_cols)
    merged = pd.concat([cond_all, S_df], axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f"{OUT_DIR}/schedules.npy", S_all)
    merged.to_csv(f"{OUT_DIR}/dataset_merged.csv", index=False)

    # --- 結果サマリ ---
    n_trip_kept = int(mask[:len(keys_trip)].sum())
    n_home_kept = int(mask[len(keys_trip):].sum())
    print("\n=== 結果サマリ ===")
    print(f"最終: {len(merged)} 人  (トリップ保有 {n_trip_kept} + 終日在宅 {n_home_kept})")
    print(f"統合CSV: {merged.shape}  -> {OUT_DIR}/dataset_merged.csv")
    print(f"離散化で吸収された短時間活動(延べ分): {total_lost}")

    # 妥当性チェック: 時間帯別の活動構成
    print("\n=== 時間帯別 活動構成(4:00起点スロット) ===")
    for label, k in [("06:00",8),("09:00",20),("12:00",32),("14:00",40),("18:00",56),("22:00",72)]:
        col = S_all[:, k]
        dist = np.bincount(col, minlength=NUM_ACT) / len(col)
        top = sorted([(Act(i).name, dist[i]) for i in range(NUM_ACT)], key=lambda x:-x[1])[:3]
        print(f"  {label}: " + ", ".join(f"{n}={p:.0%}" for n,p in top))
    print(f"\n在宅スロット比率(全体): {(S_all==int(Act.HOME)).mean():.1%}")

    # 妥当性チェック: 条件側の分布
    print("\n=== role_household_type 分布 ===")
    names = {0:"単独(男)",1:"単独(女)",2:"夫・男親",3:"妻・女親",
            4:"子供(男)",5:"子供(女)",6:"親(男)",7:"親(女)"}
    for k, v in cond_all["role_household_type"].value_counts().sort_index().items():
        print(f"  {names[int(k)]:9s}: {v}")  #type: ignore
    print("=== worker_status 分布 ===", dict(cond_all["worker_status"].value_counts()))
    print("=== age 範囲 ===", cond_all["age"].min(), "..", cond_all["age"].max())


if __name__ == "__main__":
    main()