"""
クロスウォーク: ATUS 17分類 (tier1) ↔ 社会生活基本調査 調査票A 20分類 → 共通12分類

方針 ([[concepts/activity-schema-alignment]] / 計画ファイル ステップ3):
    - 両側の全コードを共通分類に割り当てる (スケジュールは全スロット必須のため)
    - 意味的に対応が取れないコードは OTHER_X に集約し、**マッチ・評価から除外**する
      (`Other` の非対称罠: 両側の OTHER_X は構成が違うので「揃った」と扱わない)
    - フラグ: ①=1対1で比較可能 / ②=粗視化して一致 / ③=比較不能→OTHER_X

共通12分類と対応表 (③の根拠は右端):
    idx 名称             社基調A(20分類)                 ATUS(tier1)        フラグ/根拠
    0   SLEEP_PERSONAL   01睡眠+02身の回りの用事         01 Personal Care   ② ATUSは睡眠と身支度を分けない
    1   MEALS            03食事                          11 Eating&Drinking ①
    2   WORK             05仕事                          05 Work            ①
    3   SCHOOL           06学業                          06 Education       ① (在学者の学業)
    4   HOUSEWORK        07家事                          02 Household       ② ATUSは家計管理等を含む
    5   CAREGIVING       08介護・看護+09育児             03 HH員ケア+04 非HH員ケア ② 対象者軸が異なるため統合
    6   SHOPPING         10買い物                        07 Consumer Purch. ②
    7   TRAVEL           04通勤・通学+11移動(その他)     18 Traveling       ② ATUS tier1は移動目的を分けない
    8   LEISURE_SOCIAL   12TV等+13休養+15趣味娯楽+18交際 12 Social&Leisure+16 Telephone ② 電話は交際相当
    9   SPORTS           16スポーツ                      13 Sports/Exercise ①
    10  VOLUNTEER        17ボランティア・社会参加        15 Volunteer       ①
    11  OTHER_X (除外)   14学習自己啓発+19受診療養+20その他  08専門サービス+09家事サービス+10政府サービス+14宗教 ③
        - 19受診療養↔ATUS tier08 の医療サブ(0804)は tier1 粒度で分離不能
        - 14学習自己啓発は ATUS 側で tier06/12 と分離不能
        - ATUS 14宗教は社基調に対応分類が無い (その他/趣味に散る)
        - 両側 OTHER_X の中身は異なる → 比較不能。時間シェア: ATUS≈1.3% / 社基調≈2-4%

使い方:
    - main(): クロスウォーク表CSVを出力し、ATUS 96スロットデータセットを共通12分類へ写像
      (atus2024_common12_dataset.csv; T と同一フォーマット)
    - stula_to_common(df): parse_timeband.py の wide 出力の行動20分類を共通12分類へ集約
      (行動者率は主行動で排他的なので率の単純和が共通分類の率になる)
"""

import sys
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "atus"))

PROC_DIR    = REPO_ROOT / "data" / "processed"
ATUS_IN     = PROC_DIR / "atus2024_weighted_dataset.csv"
ATUS_OUT    = PROC_DIR / "atus2024_common12_dataset.csv"
XWALK_OUT   = PROC_DIR / "stula" / "crosswalk_atus_stula.csv"

N_SLOTS = 96


class Common(IntEnum):
    SLEEP_PERSONAL = 0
    MEALS          = 1
    WORK           = 2
    SCHOOL         = 3
    HOUSEWORK      = 4
    CAREGIVING     = 5
    SHOPPING       = 6
    TRAVEL         = 7
    LEISURE_SOCIAL = 8
    SPORTS         = 9
    VOLUNTEER      = 10
    OTHER_X        = 11   # ③比較不能 → マッチ・評価から除外
NUM_COMMON = len(Common)
EXCLUDED = {Common.OTHER_X}

# ATUS 17分類 (preprocess.py の act_idx 0..16) -> 共通分類
ATUS_TO_COMMON = {
    0:  Common.SLEEP_PERSONAL,   # PERSONAL_CARE (tier01)
    1:  Common.HOUSEWORK,        # HOUSEHOLD (tier02)
    2:  Common.CAREGIVING,       # CARE_HH_MEMBERS (tier03)
    3:  Common.CAREGIVING,       # CARE_NONHH (tier04)
    4:  Common.WORK,             # WORK (tier05)
    5:  Common.SCHOOL,           # EDUCATION (tier06)
    6:  Common.SHOPPING,         # CONSUMER_PURCHASES (tier07)
    7:  Common.OTHER_X,          # PROF_PERSONAL_SVC (tier08) ③医療のみ分離不能
    8:  Common.OTHER_X,          # HOUSEHOLD_SVC (tier09) ③
    9:  Common.OTHER_X,          # GOVERNMENT_SVC (tier10) ③
    10: Common.MEALS,            # EATING_DRINKING (tier11)
    11: Common.LEISURE_SOCIAL,   # SOCIAL_LEISURE (tier12)
    12: Common.SPORTS,           # SPORTS_EXERCISE (tier13)
    13: Common.OTHER_X,          # RELIGIOUS (tier14) ③社基調に対応なし
    14: Common.VOLUNTEER,        # VOLUNTEER (tier15)
    15: Common.LEISURE_SOCIAL,   # TELEPHONE (tier16) 交際相当
    16: Common.TRAVEL,           # TRAVEL (tier18)
}

# 社基調A 20分類 (コード接頭辞 '01'..'20') -> 共通分類
STULA_TO_COMMON = {
    "01": Common.SLEEP_PERSONAL,  # 睡眠
    "02": Common.SLEEP_PERSONAL,  # 身の回りの用事
    "03": Common.MEALS,           # 食事
    "04": Common.TRAVEL,          # 通勤・通学
    "05": Common.WORK,            # 仕事
    "06": Common.SCHOOL,          # 学業
    "07": Common.HOUSEWORK,       # 家事
    "08": Common.CAREGIVING,      # 介護・看護
    "09": Common.CAREGIVING,      # 育児
    "10": Common.SHOPPING,        # 買い物
    "11": Common.TRAVEL,          # 移動 (通勤・通学を除く)
    "12": Common.LEISURE_SOCIAL,  # テレビ・ラジオ・新聞・雑誌
    "13": Common.LEISURE_SOCIAL,  # 休養・くつろぎ
    "14": Common.OTHER_X,         # 学習・自己啓発・訓練 ③ATUS側で分離不能
    "15": Common.LEISURE_SOCIAL,  # 趣味・娯楽
    "16": Common.SPORTS,          # スポーツ
    "17": Common.VOLUNTEER,       # ボランティア・社会参加
    "18": Common.LEISURE_SOCIAL,  # 交際・付き合い
    "19": Common.OTHER_X,         # 受診・療養 ③tier1粒度で分離不能
    "20": Common.OTHER_X,         # その他 ③非対称バケット
}

FLAGS = {  # 共通分類ごとの比較可能性フラグ (論文のクロスウォーク表に記載)
    Common.SLEEP_PERSONAL: "②", Common.MEALS: "①", Common.WORK: "①", Common.SCHOOL: "①",
    Common.HOUSEWORK: "②", Common.CAREGIVING: "②", Common.SHOPPING: "②", Common.TRAVEL: "②",
    Common.LEISURE_SOCIAL: "②", Common.SPORTS: "①", Common.VOLUNTEER: "①", Common.OTHER_X: "③除外",
}


def stula_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """parse_timeband.py の wide 出力 (activity=20分類) -> 共通12分類へ率を集約"""
    df = df[df["activity"].str.match(r"^(0[1-9]|1\d|20)_")].copy()
    df["common"] = df["activity"].str[:2].map(lambda c: STULA_TO_COMMON[c].name)
    keys = [c for c in df.columns if c not in ("activity", "common", "population_k")
            and not c.startswith("s")]
    slot_cols = [f"s{j}" for j in range(N_SLOTS)]
    out = df.groupby(keys + ["common"], observed=True)[slot_cols].sum(min_count=1).reset_index()
    return out


def atus_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """ATUS 96スロットデータセット (act_idx 0..16) -> 共通12分類版"""
    slot_cols = [f"s{j}" for j in range(N_SLOTS)]
    lut = np.array([int(ATUS_TO_COMMON[a]) for a in range(17)], dtype=np.int8)
    out = df.copy()
    out[slot_cols] = lut[df[slot_cols].to_numpy()]
    return out


def main():
    # --- クロスウォーク表 (論文用・再現性貢献) ---
    rows = [{"common_idx": int(c), "common": c.name, "flag": FLAGS[c],
             "stula_codes": "+".join(k for k, v in STULA_TO_COMMON.items() if v == c),
             "atus_tier1_idx": "+".join(str(k) for k, v in ATUS_TO_COMMON.items() if v == c)}
            for c in Common]
    XWALK_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(XWALK_OUT, index=False)
    print(f"crosswalk表 -> {XWALK_OUT}")

    # --- ATUS 共通分類版データセット ---
    df = pd.read_csv(ATUS_IN)
    out = atus_to_common(df)
    out.to_csv(ATUS_OUT, index=False)
    slot_cols = [f"s{j}" for j in range(N_SLOTS)]
    S_before, S_after = df[slot_cols].to_numpy(), out[slot_cols].to_numpy()
    print(f"ATUS共通分類版: {out.shape} -> {ATUS_OUT}")

    # 検証: 写像前後の時間シェア保存 (各共通分類のシェア = 元分類シェアの和)
    share17 = np.bincount(S_before.ravel(), minlength=17) / S_before.size
    share12 = np.bincount(S_after.ravel(), minlength=NUM_COMMON) / S_after.size
    print("\n=== 共通12分類 時間シェア (ATUS側) ===")
    for c in Common:
        expect = share17[[a for a, v in ATUS_TO_COMMON.items() if v == c]].sum()
        ok = "OK" if abs(share12[c] - expect) < 1e-12 else "NG"
        print(f"  {c.name:15s} [{FLAGS[c]}]: {share12[c]:6.1%} (保存チェック {ok})")

    # 参考: 社基調側の共通分類シェア (平日・全国・総数; 率の全スロット平均)
    stula_path = PROC_DIR / "stula" / "timeband_weekday.csv"
    if stula_path.exists():
        st = pd.read_csv(stula_path)
        total = st[(st["region"] == "00_全国") & (st["gender"] == "0_総数")
                   & (st["employment"] == "0_総数") & (st["age_class"] == "00_総数")]
        stc = stula_to_common(total)
        print("\n=== 共通12分類 時間シェア (社基調 平日・全国・総数) ===")
        for _, r in stc.iterrows():
            share = np.nanmean(r[slot_cols].to_numpy(dtype=float)) / 100
            print(f"  {r['common']:15s}: {share:6.1%}")


if __name__ == "__main__":
    main()
