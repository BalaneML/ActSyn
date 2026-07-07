"""
ATUS 2024 前処理パイプライン
ATUS生データ(time-use diary) -> 活動スケジュール(96スロット) + 条件属性 -> 統合CSV

データ入手 (手動ダウンロード):
    https://www.bls.gov/tus/data/datafiles-2024.htm から以下の zip を取得し、展開した
        .dat を data/raw/opened/ATUS2024/ に配置する:(※ .dat はカンマ区切りCSV)
        atusact-2024.zip  
            -> atusact_2024.dat   (Activity file: 1行=1エピソード)
        atusresp-2024.zip 
            -> atusresp_2024.dat  (Respondent file: 重み・曜日・就業状態)
        atusrost-2024.zip 
            -> atusrost_2024.dat  (Roster file: 年齢・性別)
        atuscps-2024.zip  
            -> atuscps-2024/atuscps_2024.dat (CPS file: 地域変数
            GEREG=センサス4地域, GESTFIPS=州FIPS, GTMETSTA=都市圏区分
            デモグラ属性はCPS第8回面接から継承 [articles/atus-users-guide])

処理の流れ:
    1. Activity file を diary (TUCASEID) ごとに 04:00起点 1440分タイムラインへ展開
        - エピソードは TUACTIVITY_N 順に連続しており TUACTDUR24 の合計が 1440 分
        - 開始位置は累積 TUACTDUR24 で決定 (時計時刻パースより頑健)
    2. tier1=50 (uncodeable) の処理:
        - diary内の合計が DROP_UNCODEABLE_MIN 分以上 -> その diary を除外
        - 未満 -> 直前エピソードの活動で埋める (先頭なら直後の活動で埋める)
    3. 15分スロット多数決で 96 スロットへ離散化
    4. Roster (TULINENO==1 = 回答者本人) から age/gender、
        Respondent から重み TUFINLWGT・曜日 TUDIARYDAY・就業状態 TELFS を結合

スケジュール仕様:
    - 活動 17 カテゴリ = ATUS Coding Lexicon 第1層 (01..16 + 18 Traveling)
    - diary day は 04:00 -> 翌04:00
    - 1分解像度で構築 -> 15分スロットを多数決で離散化 -> 長さ96

条件・付帯属性:
    age         : TEAGE (15..85)
    gender      : TESEX 1=male->0, 2=female->1
    day_of_week : TUDIARYDAY 1=Sunday .. 7=Saturday (将来の条件追加用に保持)
    telfs       : TELFS 労働力状態の生コード (将来の条件追加用に保持)
    region      : GEREG 1=Northeast 2=Midwest 3=South 4=West (CPS由来)
    state_fips  : GESTFIPS 州FIPSコード (CPS由来; σ_min(P) の地域セル層別用)
    metro       : GTMETSTA 1=metropolitan 2=non-metro 3=not identified (CPS由来)

出力:
    data/processed/atus2024_weighted_dataset.csv
        (TUCASEID, TUFINLWGT, age, gender, day_of_week, telfs, region, state_fips, metro, s0..s95)
    data/processed/atus2024_episodes.csv (連続時刻を保持した生エピソード; 将来の連続表現用)
        (TUCASEID, TUACTIVITY_N, start_min, stop_min, duration_min, tier1, tier2, tewhere, act_idx, trcode)
        (start/stop_min は 04:00 起点の経過分。tier-50 処理「前」の生データ)

場所コード TEWHERE (劣化パイプライン degrade_triplike.py で使用):
    -1 = 非収集 (personal care=tier01 は場所を尋ねない仕様。実データで tier01/50 のみ)
    1  = 自宅・庭
    2..11, 30..89 = 職場・他人の家・店舗など自宅外
    12..21 = 移動手段 (車運転/同乗/徒歩/バス等; ほぼ tier18 の移動エピソード)
"""

from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

# ===== パス (cwd非依存: スクリプト位置からリポジトリルートを解決) =====
REPO_ROOT = Path(__file__).resolve().parents[4]   # src/common/preprocess/atus -> repo root
YEAR      = 2024                                   # 対象年 (年切替はここを変更)
RAW_DIR   = REPO_ROOT / "data" / "raw" / "opened" / f"ATUS{YEAR}"
ACT_PATH  = RAW_DIR / f"atusact_{YEAR}.dat"
RESP_PATH = RAW_DIR / f"atusresp_{YEAR}.dat"
ROST_PATH = RAW_DIR / f"atusrost_{YEAR}.dat"
CPS_PATH  = RAW_DIR / f"atuscps-{YEAR}" / f"atuscps_{YEAR}.dat"
OUT_DIR   = REPO_ROOT / "data" / "processed"
OUT_DATASET  = OUT_DIR / f"atus{YEAR}_weighted_dataset.csv"
OUT_EPISODES = OUT_DIR / f"atus{YEAR}_episodes.csv"

# ===== スケジュール定数 =====
N_SLOTS  = 96      # 24時間 * 60分 / 15分
SLOT_MIN = 15      # スロットの長さ(分)
DAY_MIN  = 1440    # 1日の長さ(分); diary は 04:00 -> 翌04:00

# uncodeable (tier1=50) がこの分数以上ある diary は除外、未満なら前の活動で埋める
# ATUS2024 では tier50 保有 diary が 15.9% (中央値60分) と多く、閾値30分だと除外12.4%。
# 除外率を~3%に抑えつつ品質を保つ値として120分を採用 (30: 12.4% / 60: 8.2% / 120: 3.0%)
DROP_UNCODEABLE_MIN = 120  # 分数
UNCODEABLE_TIER = 50  # Uncodeableの実コード


# 活動カテゴリ: ATUS Coding Lexicon 第1層 17分類 (tier 01..16 + 18)
class Act(IntEnum):
    PERSONAL_CARE       = 0   # 01 (睡眠，身だしなみなど)
    HOUSEHOLD           = 1   # 02 (HH Activity; 家事，食事準備，片付けなど)
    CARE_HH_MEMBERS     = 2   # 03 (世帯内人物の身体介助，教育，健康ケア，遊び相手，送迎など)
    CARE_NONHH          = 3   # 04 (同上の世帯外人物などが対象)
    WORK                = 4   # 05 (本業，副業，勤務，勤務関連の活動，求人，面接など)
    EDUCATION           = 5   # 06 (資格，学位のための受講，興味関心での受講，宿題，課外活動，研究など)
    CONSUMER_PURCHASES  = 6   # 07 (食品，ガソリン，外食，その他の買い物など)
    PROF_PERSONAL_SVC   = 7   # 08 (保育，金融，法律，医療，介護，理美容，不動産，獣医サービスなどの利用)
    HOUSEHOLD_SVC       = 8   # 09 (清掃代行，調理代行，衣類クリーニングなど)
    GOVERNMENT_SVC      = 9   # 10 (警察，消防，社会福祉サービス利用など)
    EATING_DRINKING     = 10  # 11 (食事，飲酒，待ち時間も含むなど)
    SOCIAL_LEISURE      = 11  # 12 (交流，パーティ参加，リラックスなど)
    SPORTS_EXERCISE     = 12  # 13 (エアロビ，スポーツなど)
    RELIGIOUS           = 13  # 14 (宗教活動など)
    VOLUNTEER           = 14  # 15 (社会福祉など)
    TELEPHONE           = 15  # 16 (家族，友人，教育機関などへの電話)
    TRAVEL              = 16  # 18 (上記の各活動に紐づく移動時間)
NUM_ACT = len(Act)

# ATUSコードをインデックスにする
TIER1_TO_IDX = {t: i for i, t in enumerate(list(range(1, 17)) + [18])}  # {1:0,...,16:15,18:16}


# ==================================================================
# スケジュール構築
# ==================================================================
def clean_tier1(dur: np.ndarray, tier1: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """diary単位のクレンジング -> (tier1修正版 or None, 除外reason or None)

    T(time-use版)と T'(擬似trip版, degrade_triplike.py)で同一の採否判定を共有し、
    両データセットの diary 集合を厳密に一致させる (D1-a のペア比較の前提)。
    """
    # 24h丸め後の合計が1日に一致しない diary は除外
    if dur.sum() != DAY_MIN:
        return None, "duration_sum_mismatch"

    # tier-50 (uncodeable) 処理
    unc = tier1 == UNCODEABLE_TIER
    if unc.any():
        if dur[unc].sum() >= DROP_UNCODEABLE_MIN:
            return None, "uncodeable_over_threshold"
        # 閾値未満: 直前エピソードの活動で埋める (先頭なら直後の非50で埋める)
        tier1 = tier1.copy()
        for i in np.flatnonzero(unc):
            if i > 0:
                tier1[i] = tier1[i - 1]
            else:
                nxt = np.flatnonzero(~unc)
                tier1[i] = tier1[nxt[0]]

    # 想定外の tier コード (欠損等) を持つ diary は除外
    if not np.isin(tier1, list(TIER1_TO_IDX)).all():
        return None, "unknown_tier1"
    return tier1, None


def slots_from_categories(dur: np.ndarray, cats: np.ndarray, num_cat: int) -> tuple[np.ndarray, int]:
    """エピソード列(duration, カテゴリidx) -> 96スロット多数決離散化 -> (sched[96], lost_short)"""
    # 1分解像度タイムライン: 累積 duration で 04:00起点 1440分を敷き詰める
    tl = np.empty(DAY_MIN, dtype=np.int8)
    pos = 0
    for d, c in zip(dur, cats):
        tl[pos:pos + d] = c
        pos += d

    # 15分スロットへ多数決離散化 (NHTS前処理と同一パターン)
    sched = np.empty(N_SLOTS, dtype=np.int8)
    lost_short = 0  # 離散化で消えた短時間活動(延べ分)
    for k in range(N_SLOTS):
        seg = tl[k * SLOT_MIN:(k + 1) * SLOT_MIN]
        counts = np.bincount(seg, minlength=num_cat)
        sched[k] = counts.argmax()
        if (counts > 0).sum() > 1:
            lost_short += int(counts.sum() - counts.max())
    return sched, lost_short


def build_one(diary: pd.DataFrame) -> tuple[np.ndarray | None, str | None, int]:
    """1 diary (TUCASEID, TUACTIVITY_N順) -> (sched[96] or None, 除外reason or None, lost_short)"""
    dur   = diary["TUACTDUR24"].to_numpy()
    tier1, reason = clean_tier1(dur, diary["TUTIER1CODE"].to_numpy())
    if tier1 is None:
        return None, reason, 0
    cats = np.array([TIER1_TO_IDX[int(t)] for t in tier1], dtype=np.int8)
    sched, lost_short = slots_from_categories(dur, cats, NUM_ACT)
    return sched, None, lost_short


def build_schedules(act: pd.DataFrame):
    """全 diary のスケジュール構築 -> (S[N,96], case_ids, reasons, total_lost)"""
    reasons = {"duration_sum_mismatch": 0, "uncodeable_over_threshold": 0, "unknown_tier1": 0}
    schedules, case_ids = [], []
    total_lost = 0
    for case_id, g in act.groupby("TUCASEID", sort=False):
        sched, reason, lost = build_one(g)
        if reason is not None:
            reasons[reason] += 1
            continue
        schedules.append(sched)
        case_ids.append(case_id)
        total_lost += lost
    S = np.stack(schedules)
    return S, np.asarray(case_ids), reasons, total_lost


# ==================================================================
# エピソードファイル (連続時刻保持; tier-50処理前の生データ)
# ==================================================================
def build_episodes(act: pd.DataFrame) -> pd.DataFrame:
    ep = act[["TUCASEID", "TUACTIVITY_N", "TUACTDUR24", "TUTIER1CODE", "TUTIER2CODE",
              "TEWHERE", "TRCODE"]].copy()
    # 04:00起点の経過分: 累積 TUACTDUR24 から導出 (エピソードは連続していて隙間がない)
    cum = ep.groupby("TUCASEID")["TUACTDUR24"].cumsum()
    ep["stop_min"]  = cum
    ep["start_min"] = cum - ep["TUACTDUR24"]
    ep["act_idx"]   = ep["TUTIER1CODE"].map(TIER1_TO_IDX).fillna(-1).astype(int)  # tier50等は-1
    ep = ep.rename(columns={"TUACTDUR24": "duration_min", "TUTIER1CODE": "tier1",
                            "TUTIER2CODE": "tier2", "TEWHERE": "tewhere", "TRCODE": "trcode"})
    return ep[["TUCASEID", "TUACTIVITY_N", "start_min", "stop_min",
               "duration_min", "tier1", "tier2", "tewhere", "act_idx", "trcode"]]


# ==================================================================
# 読み込み・条件結合 (degrade_triplike.py と共有)
# ==================================================================
def load_activity() -> pd.DataFrame:
    """Activity file を (TUCASEID, TUACTIVITY_N) 順で読む"""
    act = pd.read_csv(ACT_PATH, usecols=["TUCASEID", "TUACTIVITY_N", "TUACTDUR24", "TUSTARTTIM",
                                         "TUTIER1CODE", "TUTIER2CODE", "TEWHERE", "TRCODE"])
    return act.sort_values(["TUCASEID", "TUACTIVITY_N"])


# 統合CSVに書き出す条件・重み列 (degrade_triplike.py と共有)
COND_COLS = ["TUCASEID", "TUFINLWGT", "age", "gender", "day_of_week", "telfs",
             "region", "state_fips", "metro"]


def attach_conditions(case_ids: np.ndarray) -> pd.DataFrame:
    """採用 diary の case_ids に Roster/Respondent/CPS の条件・重み・地域を結合"""
    resp = pd.read_csv(RESP_PATH, usecols=["TUCASEID", "TUFINLWGT", "TUDIARYDAY", "TELFS"])
    rost = pd.read_csv(ROST_PATH, usecols=["TUCASEID", "TULINENO", "TEAGE", "TESEX"])
    cps  = pd.read_csv(CPS_PATH, usecols=["TUCASEID", "TULINENO", "GEREG", "GESTFIPS", "GTMETSTA"])
    respondent = rost[rost["TULINENO"] == 1].copy()  # TULINENO==1 が diary 回答者本人
    cps_resp   = cps[cps["TULINENO"] == 1].drop(columns="TULINENO")
    cond = pd.DataFrame({"TUCASEID": case_ids}).merge(respondent, on="TUCASEID", how="left")
    cond = cond.merge(resp, on="TUCASEID", how="left")
    cond = cond.merge(cps_resp, on="TUCASEID", how="left")
    assert not cond[["TEAGE", "TESEX", "TUFINLWGT", "GESTFIPS"]].isna().any().any(), \
        "roster/resp/cps 結合に欠損"
    assert (cond["TUCASEID"].to_numpy() == case_ids).all()

    cond["age"]         = cond["TEAGE"].astype(int)
    cond["gender"]      = (cond["TESEX"] == 2).astype(int)  # 1=male->0, 2=female->1
    cond["day_of_week"] = cond["TUDIARYDAY"].astype(int)    # 1=Sun .. 7=Sat
    cond["telfs"]       = cond["TELFS"].astype(int)
    cond["region"]      = cond["GEREG"].astype(int)         # 1=NE 2=MW 3=South 4=West
    cond["state_fips"]  = cond["GESTFIPS"].astype(int)
    cond["metro"]       = cond["GTMETSTA"].astype(int)      # 1=metro 2=non-metro 3=unidentified
    return cond


# ==================================================================
# メイン
# ==================================================================
def main():
    print("loading ...")
    act = load_activity()

    # 妥当性: 先頭エピソードは 04:00 開始 (diary day の定義)
    first = act.groupby("TUCASEID", sort=False).first()
    n_bad_start = int((first["TUSTARTTIM"] != "04:00:00").sum())
    print(f"diaries: {act['TUCASEID'].nunique()}  episodes: {len(act)}")
    print(f"  先頭エピソードが04:00開始でない diary: {n_bad_start}")

    # --- 1-3. スケジュール構築 ---
    S, case_ids, reasons, total_lost = build_schedules(act)
    print(f"  採用: {len(case_ids)}  除外: {reasons}")

    # --- 4. 条件・重みの結合 ---
    cond = attach_conditions(case_ids)

    # --- 統合CSV保存 ---
    slot_cols = [f"s{k}" for k in range(N_SLOTS)]
    merged = pd.concat(
        [cond[COND_COLS].reset_index(drop=True), pd.DataFrame(S, columns=slot_cols)],
        axis=1,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DATASET, index=False)

    episodes = build_episodes(act)
    episodes.to_csv(OUT_EPISODES, index=False)

    # --- 結果サマリ ---
    print("\n=== 結果サマリ ===")
    print(f"最終: {len(merged)} 人  統合CSV: {merged.shape} -> {OUT_DATASET}")
    print(f"エピソードCSV: {episodes.shape} -> {OUT_EPISODES}")
    print(f"離散化で吸収された短時間活動(延べ分): {total_lost}")

    print("\n=== 17クラス 時間シェア ===")
    share = np.bincount(S.ravel(), minlength=NUM_ACT) / S.size
    for a in Act:
        print(f"  {a.name:19s}: {share[a]:6.1%}")

    print("\n=== 時間帯別 活動構成(4:00起点スロット) ===")
    for label, k in [("06:00", 8), ("09:00", 20), ("12:00", 32), ("18:00", 56), ("22:00", 72), ("03:00", 92)]:
        dist = np.bincount(S[:, k], minlength=NUM_ACT) / len(S)
        top = sorted([(Act(i).name, dist[i]) for i in range(NUM_ACT)], key=lambda x: -x[1])[:3]
        print(f"  {label}: " + ", ".join(f"{n}={p:.0%}" for n, p in top))

    print("\n=== 条件分布 ===")
    print("age 範囲:", merged["age"].min(), "..", merged["age"].max())
    print("gender:", dict(merged["gender"].value_counts().sort_index()))
    print("day_of_week (1=Sun..7=Sat):", dict(merged["day_of_week"].value_counts().sort_index()))


if __name__ == "__main__":
    main()
