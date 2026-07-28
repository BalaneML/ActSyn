"""
クロスウォーク: ATUS 17分類 (tier1) ↔ 社会生活基本調査 調査票A 20分類 → 共通12分類

方針 ([[concepts/activity-schema-alignment]] / 計画ファイル ステップ3):
    - 両側の全コードを共通分類に割り当てる (スケジュールは全スロット必須のため)
    - 意味的に対応が取れないコードは OTHER_X に集約し、マッチ・評価から除外する
        (Other の非対称: 両側の OTHER_X は構成が違うので「揃った」と扱わない)
    - フラグ (①②③の判定基準):
        ① 両側が1対1に対応し、残存非対称が他分類との相殺で吸収される
        ② 複数コードを粗視化して対応させた、または1対1だが内容例示レベルで境界が一致しない
        ③ 対応物が無い → OTHER_X (マッチ・評価から除外)

数値の基準 (本モジュールが記載する時間シェアは2種類あり、混同しないこと):
    (A) スロット加重: 96スロット × TUFINLWGT・平日 (day_of_week 2..6)。
        共通12分類の公表表 (crosswalk_atus_stula.csv/.md) と本docstringの12分類レベルの数値。
        モデル (japan_match_experiment.py) が学習・評価に用いる量と同一基準。
    (B) エピソード加重: TUFINLWGT × duration_min の人時シェア・平日。
        tier2/tier3 粒度の数値 (RESIDUAL の pt 値、電話の内訳など) はこちら。
        12分類レベルでは (A) と 0.03pt 以内で一致する。
    社基調側はいずれも 平日・全国・総数の行動者率の全スロット平均。

共通12分類と対応表 (③の根拠は右端):
    idx 名称             社基調A(20分類)                  ATUS(tier1)                                   フラグ/根拠
    0   SLEEP_PERSONAL   01睡眠+02身の回りの用事           01 Personal Care                              ② ATUSは睡眠と身支度を分けない
    1   MEALS            03食事                         11 Eating&Drinking                             ② 社基調は同席者・目的で食事を3分割
    2   WORK             05仕事                         05 Work                                        ① 仕事中の移動はTRAVEL側で相殺
    3   SCHOOL           06学業                         06 Education                                   ② 趣味的講座・部活で境界が不一致
    4   HOUSEWORK        07家事                         02 Household                                   ② ATUSは家計管理等を含む
    5   CAREGIVING       08介護・看護+09育児              03 HH員ケア+04 非HH員ケア                        ② 対象者軸が異なるため統合
    6   SHOPPING         10買い物                        07 Consumer Purch.                            ②
    7   TRAVEL           04通勤・通学+11移動(その他)       18 Traveling                                   ② ATUS tier1は移動目的を分けない
    8   LEISURE_SOCIAL   12TV等+13休養+15趣味娯楽+18交際   12 Social&Leisure                              ②
    9   SPORTS           16スポーツ                      13 Sports/Exercise                            ① 残存非対称は観覧・鑑賞 0.06pt のみ
    10  VOLUNTEER        17ボランティア・社会参加           15 Volunteer                                 ② 社基調17は政治・布教・投票を含む
    11  OTHER_X (除外)   14学習自己啓発+19受診療養+20その他  08専門サービス+09家事サービス+10政府サービス
                                                       +14宗教+16電話                                 ③
        - 19受診療養↔ATUS tier08 の医療サブ(0804)は tier1 粒度で分離不能
        - 14学習自己啓発は ATUS 側で tier06/12 と分離不能
        - ATUS 14宗教は社基調に対応分類が無い (その他/趣味に散る)
        - ATUS 16電話は「手段軸」の分類で社基調 20分類に対応物が無い (下記)
        - 両側 OTHER_X の中身は異なる → 比較不能。時間シェア(A): ATUS 1.50% / 社基調 2.31%

    ATUS 16 Telephone を OTHER_X とする根拠 (2026-07 判断):
        ATUS 16 は「通話」という**手段軸**のカテゴリだが、社基調 20分類に手段軸の分類は無く、
        通話は相手・内容により 13(家族)/18(友人)/07(銀行・役所)/10(買い物)/09(保育者) へ分散する。
        軸が対応しないものは統合しないという方針 (ATUS 08「取引軸」を OTHER_X とするのと同じ基準)。
        コスト: tier3 内訳(B)では 160102 友人(43.0%)+160101 家族(36.6%)+160199 n.e.c.(13.0%) = 92.6% が
        社基調側では LEISURE_SOCIAL (18交際「友達との電話・会話」/ 13休養「家族との団らん」) に落ちる。
        すなわち平日 0.52pt の比較可能な時間をマッチ・評価から除外している。
        残り 7.4% は 160105 専門サービス(4.3%)/160104 販売員(1.2%)/160108 官公庁(1.0%)/
        160106,107 家事・保育サービス(0.8%)/160201 待機(0.2%) で、
        社基調では HOUSEWORK・SHOPPING・CAREGIVING へ行く。
        参考(A): この移動で乖離は LEISURE_SOCIAL -1.17→-1.74pt、OTHER_X -1.37→-0.80pt と相殺し、
        この2分類の絶対差の和は 2.54pt のまま変わらない
        (表全体の絶対差合計 13.25pt とは別物)。数値改善ではなく軸の一貫性を優先した判断である。
        なお社基調は 1スロット1ラベル(主行動のみ)・15分粒度のため短い通話は記録されにくく、
        ATUS のエピソード実測 (16 は平日 8.2分/日) との測定粒度差も両者を比較しにくくしている。

    MEALS が②である根拠 (両側の内容例示・コーディング規則の突き合わせ):
        ATUS 11 のカテゴリ定義は "all eating and drinking ... whether the respondent was
        alone, with others" (tu2025coderules.pdf) で食事を一切分割しない。一方 社基調は
        別表2 の備考で食事を同席者・目的により 03/13/18 へ3分割する。ラベルは1対1だが中身が違う。
        - 間食(おやつ)・お茶の時間・食休み: ATUS 110101 / 社基調 13休養  → MEALS↔LEISURE_SOCIAL
        - 交際のための飲食(知人と飲食):     ATUS 110101 / 社基調 18交際  → MEALS↔LEISURE_SOCIAL
          (ATUS は会話部分のみ 120101 へ分離する: cat.11 rule 2)
        上記2件は両側とも分離不能で、いずれも ATUS 11 を相対的に膨らませる向き。
        観測値(A)は ATUS 4.60% vs 社基調 6.73% (-2.13pt) だが、定義差が差を縮める向きに効くため
        真の行動差はこれより大きい → 差の大きさを主張する場合は下限として扱う。
        逆向き(ATUS 11 を削る)のリークは小さい (B):
        - 外食の注文・会計・ウェイターとの会話: ATUS 070103→SHOPPING / 社基調は03に一体   0.088%
        - 仕事としての飲食: ATUS 050202→WORK / 社基調 03「仕事場での食事」に含む          0.002%
          (ATUS は食事を面接時に pre-code するためこの経路はほぼ発火しない)

使い方:
    - main(): ATUS 96スロットデータセットを共通12分類へ写像し
        (atus2024_stula_common12_dataset.csv; T と同一フォーマット・全曜日を保持)、
        論文貼付用のクロスウォーク表を CSV と Markdown で出力する
        (crosswalk_atus_stula.csv / .md; いずれも平日スコープ)
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
from preprocess import NUM_ACT, TIER1_TO_IDX  # noqa: E402  ATUS側の分類定義は preprocess.py が唯一の出所

PROC_DIR    = REPO_ROOT / "data" / "processed"
ATUS_IN     = PROC_DIR / "atus2024" / "atus2024_weighted_dataset.csv"
ATUS_OUT    = PROC_DIR / "atus2024" / "atus2024_stula_common12_dataset.csv"
XWALK_OUT   = PROC_DIR / "stula" / "crosswalk_atus_stula.csv"
XWALK_MD    = PROC_DIR / "stula" / "crosswalk_atus_stula.md"   # 論文貼付用 Markdown 表

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
    15: Common.OTHER_X,          # TELEPHONE (tier16) ③手段軸のため社基調に対応物なし
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

# act_idx -> ATUS tier1 実コード。preprocess.py の TIER1_TO_IDX から導出する
# (定義を複製すると tier1 構成が変わったとき黙って食い違うため)
ATUS_TIER1_CODE = {i: f"{t:02d}" for t, i in TIER1_TO_IDX.items()}

# act_idx -> ATUS tier1 の英語名 (論文表の表示用; コードは ATUS_TIER1_CODE 側が唯一の出所)
ATUS_TIER1_NAME = {
    0:  "Personal Care",                1:  "Household Activities",
    2:  "Caring HH Members",            3:  "Caring NonHH Members",
    4:  "Work & Work-Related",          5:  "Education",
    6:  "Consumer Purchases",           7:  "Prof. & Personal Care Svcs",
    8:  "Household Services",           9:  "Government Svcs & Civic",
    10: "Eating and Drinking",          11: "Socializing/Relaxing/Leisure",
    12: "Sports/Exercise/Recreation",   13: "Religious and Spiritual",
    14: "Volunteer Activities",         15: "Telephone Calls",
    16: "Traveling",
}
assert set(ATUS_TIER1_NAME) == set(ATUS_TIER1_CODE) == set(ATUS_TO_COMMON) == set(range(NUM_ACT))

STULA_NAMES = {
    "01": "睡眠",        "02": "身の回りの用事", "03": "食事",   "04": "通勤・通学",
    "05": "仕事",        "06": "学業",          "07": "家事",   "08": "介護・看護",
    "09": "育児",        "10": "買い物",        "11": "移動(通勤・通学を除く)",
    "12": "テレビ・ラジオ・新聞・雑誌",          "13": "休養・くつろぎ",
    "14": "学習・自己啓発・訓練(学業以外)",      "15": "趣味・娯楽",
    "16": "スポーツ",    "17": "ボランティア活動・社会参加活動",
    "18": "交際・付き合い", "19": "受診・療養",  "20": "その他",
}

# tier1/20分類の粒度では解消できない残存非対称 (両側の内容例示・コーディング規則の突き合わせによる)
# pt 値は基準(B) エピソード加重 (TUFINLWGT × duration_min・平日) の人時シェア。
# ただし12分類レベルで完結する値 (WORK 行) のみ基準(A) スロット加重で、その旨を明記している
RESIDUAL = {
    Common.SLEEP_PERSONAL:
        "睡眠/身支度の分割は比較不能(睡眠+4.9pt・身支度-2.9pt と逆向き); "
        "うたたね(社基調13); 理美容・エステ(ATUS 0805は08に在籍); "
        "自宅療養(ATUS 0103は01に在籍); サウナ(ATUS 120301)",
    Common.MEALS:
        "間食・お茶(社基調13); 交際飲食(社基調18); 外食の会計(ATUS 070103). "
        "定義差は日米差を縮める向き → 差は下限として扱う",
    Common.WORK:
        "仕事中の移動(社基調05・ATUS 1805でTRAVEL); 求職活動(ATUS 0504は05に在籍・社基調20). "
        "差は WORK単独 -1.60pt → WORK+TRAVEL合算 -0.95pt と縮む(基準A)",
    Common.SCHOOL:
        "趣味的講座・自動車教習(社基調14・ATUS 060102/060101); "
        "必修外の部活(社基調15/16・ATUS 0602)",
    Common.HOUSEWORK:
        "ペットの世話(ATUS 0206 0.60pt・社基調15); 銀行/市役所(ATUS 0802・1001); "
        "送迎(ATUS 030112・1803); 就学後の子供の世話(ATUS 0301); "
        "趣味の園芸・日曜大工(ATUS 0205・0203)",
    Common.CAREGIVING:
        "分割軸が直交(社基調=被介護者属性 vs ATUS=世帯内外); "
        "家族以外・無報酬・非組織の介護(社基調17); 就学後の子供の世話(社基調07). 差は上限. "
        "ATUS 0405(世帯外成人への生活援助 0.18pt)は社基調では17へ行くが、"
        "96スロットが tier1 粒度のため分離できず CAREGIVING に残している",
    Common.SHOPPING:
        "外食の会計・注文(ATUS 070103 が MEALS から流入)",
    Common.TRAVEL:
        "仕事中の移動・送迎の流入(上記); ATUS 1801・1899 データコード 0.23pt を含む",
    Common.LEISURE_SOCIAL:
        "ペットの世話・スポーツ観戦が流出(ATUS 0206・1302); 冠婚葬祭(ATUS 140101); "
        "TV等による学習(社基調14); うたたね・間食が流入; ATUS 16電話の社会的通話を除外(0.52pt)",
    Common.SPORTS:
        "観覧・鑑賞(ATUS 1302 0.06pt・社基調15). 歩行の扱いは両側の定義が完全一致",
    Common.VOLUNTEER:
        "社基調17は社会参加活動(投票・布教・政治・労働運動)を含むがATUSは10・14へ分散; "
        "家族以外への無報酬の生活援助(ATUS 0405 0.18pt)はCAREGIVING側に残る(上記). 差は下限",
    Common.OTHER_X:
        "両側の構成が非対称。tier1制約により ATUS 0103・0504 が流出、0805・0802 が流入しない",
}

FLAGS = {  # 共通分類ごとの比較可能性フラグ (論文のクロスウォーク表に記載)
    Common.SLEEP_PERSONAL: "②",
    Common.MEALS:          "②",
    Common.WORK:           "①",
    Common.SCHOOL:         "②",
    Common.HOUSEWORK:      "②",
    Common.CAREGIVING:     "②",
    Common.SHOPPING:       "②",
    Common.TRAVEL:         "②",
    Common.LEISURE_SOCIAL: "②",
    Common.SPORTS:         "①",
    Common.VOLUNTEER:      "②",
    Common.OTHER_X:        "③除外",
}


def stula_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    parse_timeband.py の wide 出力 (activity=20分類) -> 共通12分類へ率を集約
    """
    df = df[df["activity"].str.match(r"^(0[1-9]|1\d|20)_")].copy()
    df["common"] = df["activity"].str[:2].map(lambda c: STULA_TO_COMMON[str(c)].name)
    keys = [c for c in df.columns if c not in ("activity", "common", "population_k")
            and not c.startswith("s")]
    slot_cols = [f"s{j}" for j in range(N_SLOTS)]
    grp = df.groupby(keys + ["common"], observed=True)
    out = grp[slot_cols].sum()
    # 非公表 ('…'→NaN) の成分を含むセルは部分和にせず NaN へ伝播させる
    # ('-'=行動者ゼロは parse 時点で 0.0 のため、ここに来る NaN は真の欠損のみ)
    out = out.mask(grp[slot_cols].count().lt(grp.size(), axis=0))
    return out.reset_index()


def atus_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    ATUS 96スロットデータセット (act_idx 0..16) -> 共通12分類版
    """
    slot_cols = [f"s{j}" for j in range(N_SLOTS)]
    lut = np.array([int(ATUS_TO_COMMON[a]) for a in range(NUM_ACT)], dtype=np.int8)
    out = df.copy()
    out[slot_cols] = lut[df[slot_cols].to_numpy()]
    return out


def build_markdown(rows: list[dict], n_atus: int) -> str:
    """クロスウォーク表を Markdown (対応表 + 残存非対称) で組み立てる"""
    L = ["# クロスウォーク: ATUS tier1 ↔ 社会生活基本調査A 20分類 → 共通12分類", "",
         f"時間シェアは両側とも**平日**。ATUS は 96スロットを `TUFINLWGT` で加重・n={n_atus}、",
         "社基調は全国・総数の行動者率の全スロット平均。",
         "モデル (`japan_match_experiment.py`) が学習・評価に用いる量と同一基準である。",
         "",
         "本表は平日のみを集計している。写像後のデータセット"
         " `atus2024_stula_common12_dataset.csv` は全曜日を保持するため、",
         "利用側で `day_of_week` 2..6 に絞ること。",
         "",
         "| # | 共通分類 | 社基調A 20分類 | ATUS tier1 | ATUS | 社基調 | 差 | フラグ |",
         "|---|---|---|---|---:|---:|---:|:---:|"]
    for r in rows:
        L.append(
            f"| {r['common_idx']} | {r['common']} | "
            f"{r['stula_codes']} {r['stula_names']} | "
            f"{r['atus_tier1']} {r['atus_names']} | "
            f"{r['share_atus_weekday']:.2%} | {r['share_stula_weekday']:.2%} | "
            f"{r['diff_pt']*100:+.2f}pt | {r['flag']} |")
    tot = sum(abs(r["diff_pt"]) for r in rows)
    L += [f"| | **絶対差合計** | | | | | **{tot*100:.2f}pt** | |", "",
          "フラグの判定基準:",
          "- ① 両側が1対1に対応し、残存非対称が他分類との相殺で吸収される",
          "- ② 複数コードを粗視化して対応させた、"
          "または1対1だが内容例示レベルで境界が一致しない",
          "- ③ 対応物が無い → OTHER_X (マッチ・評価から除外)",
          "", "## 残存非対称 (tier1 / 20分類の粒度では解消できない)", "",
          "両側の内容例示 (社基調 別表2「行動の種類の内容例示一覧」/ ATUS Lexicon) と",
          "コーディング規則 (`tu2025coderules.pdf` の Category definition) を突き合わせて特定したもの。",
          "pt 値は ATUS エピソードの `TUFINLWGT` × 分 で加重した平日の人時シェア"
          " (12分類レベルでは上表と 0.03pt 以内で一致)。", ""]
    for r in rows:
        L.append(f"- **{r['common']}** — {r['residual_asymmetry']}")
    return "\n".join(L) + "\n"


def main():
    # --- ATUS 共通分類版データセット ---
    # 出力CSVは全曜日を保持する (スコープ選択は利用側の責務;
    #  japan_match_experiment.py が読み込み時に day_of_week 2..6 で絞る)
    df = pd.read_csv(ATUS_IN)
    out = atus_to_common(df)
    out.to_csv(ATUS_OUT, index=False)
    slot_cols = [f"s{j}" for j in range(N_SLOTS)]
    print(f"ATUS共通分類版: {out.shape} -> {ATUS_OUT}")

    # 以降の印字は平日 (day_of_week 2..6) に限定する。
    # 社基調側の参考シェアが平日表であり、全曜日の ATUS と並べると
    # 曜日構成差 (土日は睡眠が長い) を日米差と誤読するため。
    wd = df["day_of_week"].between(2, 6).to_numpy()
    S_before, S_after = df.loc[wd, slot_cols].to_numpy(), out.loc[wd, slot_cols].to_numpy()

    # 検証: 写像前後の時間シェア保存 (各共通分類のシェア = 元分類シェアの和)。
    # LUT が漏れなく写しているかの確認が目的なので非加重で行う
    share17 = np.bincount(S_before.ravel(), minlength=NUM_ACT) / S_before.size
    share12 = np.bincount(S_after.ravel(), minlength=NUM_COMMON) / S_after.size

    # 公表表に載せるシェアは TUFINLWGT 加重 (基準A)。
    # 社基調側は母集団代表の公表率であり、japan_match_experiment.py も加重で学習するため、
    # 日米差を論じる対象は加重値でなければならない
    W = np.repeat(df.loc[wd, "TUFINLWGT"].to_numpy()[:, None], N_SLOTS, axis=1)
    share12_w = np.bincount(S_after.ravel(), weights=W.ravel(), minlength=NUM_COMMON) / W.sum()

    print(f"\n=== 共通12分類 時間シェア (ATUS 平日・{wd.sum()}人) ===")
    for c in Common:
        expect = share17[[a for a, v in ATUS_TO_COMMON.items() if v == c]].sum()
        ok = "OK" if abs(share12[c] - expect) < 1e-12 else "NG"
        print(f"  {c.name:15s} [{FLAGS[c]}]: 加重 {share12_w[c]:6.2%}"
              f"  非加重 {share12[c]:6.2%} (保存チェック {ok})")

    # 社基調側の共通分類シェア (平日・全国・総数; 率の全スロット平均)。
    # この表なしでは差の列が作れない = 公表表を出せないので、欠如は明示的なエラーにする
    stula_path = PROC_DIR / "stula" / "timeband_weekday.csv"
    if not stula_path.exists():
        raise FileNotFoundError(
            f"社基調の平日時間帯表がありません: {stula_path}\n"
            "先に parse_timeband.py を実行すること")
    st = pd.read_csv(stula_path)
    total = st[(st["region"] == "00_全国") & (st["gender"] == "0_総数")
                & (st["employment"] == "0_総数") & (st["age_class"] == "00_総数")]
    share_st = {Common[r["common"]]: np.nanmean(r[slot_cols].to_numpy(dtype=float)) / 100
                for _, r in stula_to_common(total).iterrows()}

    # --- クロスウォーク表 (論文用・再現性貢献) ---
    # 両側とも平日。ATUS は TUFINLWGT 加重 (基準A)。
    rows = []
    for c in Common:
        sc = [k for k, v in STULA_TO_COMMON.items() if v == c]
        ac = [k for k, v in ATUS_TO_COMMON.items() if v == c]
        rows.append({
            "common_idx":          int(c),
            "common":              c.name,
            "flag":                FLAGS[c],
            "stula_codes":         "+".join(sc),
            "stula_names":         " / ".join(STULA_NAMES[k] for k in sc),
            "atus_tier1":          "+".join(ATUS_TIER1_CODE[a] for a in ac),
            "atus_names":          " / ".join(ATUS_TIER1_NAME[a] for a in ac),
            "share_atus_weekday":  round(float(share12_w[c]), 5),
            "share_stula_weekday": round(share_st[c], 5),
            "diff_pt":             round(float(share12_w[c]) - share_st[c], 5),
            "residual_asymmetry":  RESIDUAL[c],
        })
    XWALK_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(XWALK_OUT, index=False)

    md = build_markdown(rows, int(wd.sum()))
    XWALK_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"crosswalk表 -> {XWALK_OUT}\n            -> {XWALK_MD}")


if __name__ == "__main__":
    main()
