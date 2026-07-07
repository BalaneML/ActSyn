"""
D1-a 劣化パイプライン: ATUS time-use diary -> 擬似trip版データセット T'

目的 (実験デザイン D1-a, 計画ファイル参照):
    同一のATUS diary から「測定基盤」だけを trip-based 調査相当に劣化させた訓練セットを作り、
    time-use版 T (preprocess.py の出力) との同一データ内比較で測定基盤の効果を因果識別する。
    diary の採否判定 (clean_tier1) は preprocess.py と共有し、T と T' の個人集合を厳密に一致させる。

劣化ルール (trip-based 調査が観測できるものだけを残す):
    1. tier1=18 (移動)                  -> TRAVEL
    2. 自宅 (TEWHERE=1) / 非収集 (=-1)  -> HOME 残差
       ※ TEWHERE=-1 は personal care (睡眠等; 場所を尋ねない仕様)。trip-based 調査は
         在宅活動の中身を観測できない、という測定基盤の欠落を正確に再現する。
       ※ tier-50 埋め処理後のエピソードも TEWHERE は原値 (-1) を使う。
    3. それ以外 (自宅外)                -> 目的カテゴリ (variant により2通り)

2つの variant を同時出力:
    - triplike18 (18クラス): HOME + 自宅外 tier01..16 + TRAVEL
        在宅の意味情報「だけ」を破壊する最小操作 = 測定基盤の単一要因操作 (D1-a の主変数)
    - nhts10 (10クラス): NHTS WHYTRP1S 相当へ写像 (nhts/preprocess.py の Act と同じ並び)
        自宅外の目的粒度も NHTS 水準に粗視化 = 実NHTSとの比較 (D1-b) に接続する完全劣化版
        クロスウォーク (自宅外エピソードのみ):
            tier05->WORK / tier06,14->SCHOOL (WHYTRP1S20 は宗教活動を含む) / tier07->SHOP /
            tier08&tier2=4(医療サービス)->MEDICAL / tier11->MEALS / tier12,13->SOCIAL /
            tier01,02,03,04,09,10,15,16 / tier08その他 -> OTHER
        ※ TRANSPORT(送迎) は NHTS ではトリップ目的だが ATUS では移動エピソード(tier18)に
          吸収されるため T' では常に空。クロスウォーク表の「比較不能セル」として扱う。

出力:
    data/processed/atus2024_triplike18_dataset.csv
    data/processed/atus2024_nhts10_dataset.csv
        いずれも T と同一フォーマット: TUCASEID, TUFINLWGT, age, gender, day_of_week, telfs, s0..s95

検証 (実行時に自動実施):
    - T' の diary 集合が T (atus2024_weighted_dataset.csv) と完全一致すること (assert)
    - T'=HOME スロットの T 側活動内訳 (残差化が在宅系活動を潰していることの確認)
    - サンプル diary の T vs T' 目視用ダンプ (RLE表記)
"""

from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (
    Act, NUM_ACT, N_SLOTS, TIER1_TO_IDX, OUT_DATASET, OUT_DIR, COND_COLS,
    load_activity, attach_conditions, clean_tier1, slots_from_categories,
)

OUT_TRIPLIKE18 = OUT_DIR / "atus2024_triplike18_dataset.csv"
OUT_NHTS10     = OUT_DIR / "atus2024_nhts10_dataset.csv"

WHERE_HOME          = 1    # TEWHERE: 自宅・庭
WHERE_NOT_COLLECTED = -1   # TEWHERE: 非収集 (personal care)
TRAVEL_TIER         = 18
MEDICAL_TIER2       = 4    # tier08 の下位: 0804 = medical care services


# ===== variant A: triplike18 (在宅の意味情報だけを破壊) =====
class ActT18(IntEnum):
    HOME                    = 0
    OOH_PERSONAL_CARE       = 1   # 自宅外 tier 01 (理論上ほぼ空: tier01は場所非収集->HOME)
    OOH_HOUSEHOLD           = 2
    OOH_CARE_HH_MEMBERS     = 3
    OOH_CARE_NONHH          = 4
    OOH_WORK                = 5
    OOH_EDUCATION           = 6
    OOH_CONSUMER_PURCHASES  = 7
    OOH_PROF_PERSONAL_SVC   = 8
    OOH_HOUSEHOLD_SVC       = 9
    OOH_GOVERNMENT_SVC      = 10
    OOH_EATING_DRINKING     = 11
    OOH_SOCIAL_LEISURE      = 12
    OOH_SPORTS_EXERCISE     = 13
    OOH_RELIGIOUS           = 14
    OOH_VOLUNTEER           = 15
    OOH_TELEPHONE           = 16
    TRAVEL                  = 17
NUM_T18 = len(ActT18)

# tier01..16 -> OOH_* (tier t は index t)
TIER1_TO_T18 = {t: t for t in range(1, 17)}


# ===== variant B: nhts10 (NHTS WHYTRP1S 相当; nhts/preprocess.py の Act と同じ並び) =====
class ActN10(IntEnum):
    HOME      = 0
    WORK      = 1
    SCHOOL    = 2
    MEDICAL   = 3
    SHOP      = 4
    SOCIAL    = 5
    TRANSPORT = 6   # 送迎: ATUSでは移動側に吸収され常に空 (比較不能セル)
    MEALS     = 7
    OTHER     = 8
    TRAVEL    = 9
NUM_N10 = len(ActN10)

TIER1_TO_N10 = {
    1: ActN10.OTHER, 2: ActN10.OTHER, 3: ActN10.OTHER, 4: ActN10.OTHER,
    5: ActN10.WORK, 6: ActN10.SCHOOL, 7: ActN10.SHOP, 8: ActN10.OTHER,  # tier08はtier2=4のみMEDICAL
    9: ActN10.OTHER, 10: ActN10.OTHER, 11: ActN10.MEALS, 12: ActN10.SOCIAL,
    13: ActN10.SOCIAL, 14: ActN10.SCHOOL, 15: ActN10.OTHER, 16: ActN10.OTHER,
}


def degrade_categories(tier1: np.ndarray, tier2: np.ndarray, tewhere: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """クレンジング済み tier1 列 -> (T18カテゴリ列, N10カテゴリ列)"""
    n = len(tier1)
    c18 = np.empty(n, dtype=np.int8)
    c10 = np.empty(n, dtype=np.int8)
    for i in range(n):
        t = int(tier1[i])
        if t == TRAVEL_TIER:
            c18[i] = ActT18.TRAVEL
            c10[i] = ActN10.TRAVEL
        elif tewhere[i] in (WHERE_HOME, WHERE_NOT_COLLECTED):
            c18[i] = ActT18.HOME
            c10[i] = ActN10.HOME
        else:
            c18[i] = TIER1_TO_T18[t]
            if t == 8 and tier2[i] == MEDICAL_TIER2:
                c10[i] = ActN10.MEDICAL
            else:
                c10[i] = TIER1_TO_N10[t]
    return c18, c10


def build_degraded(act: pd.DataFrame):
    """全 diary -> (S18[N,96], S10[N,96], case_ids)。採否は preprocess.clean_tier1 と同一。"""
    s18_all, s10_all, case_ids = [], [], []
    for case_id, g in act.groupby("TUCASEID", sort=False):
        dur = g["TUACTDUR24"].to_numpy()
        tier1, reason = clean_tier1(dur, g["TUTIER1CODE"].to_numpy())
        # if reason is not None:
        #     continue
        if tier1 is None:
            return None, reason, 0
        c18, c10 = degrade_categories(tier1, g["TUTIER2CODE"].to_numpy(), g["TEWHERE"].to_numpy())
        s18, _ = slots_from_categories(dur, c18, NUM_T18)
        s10, _ = slots_from_categories(dur, c10, NUM_N10)
        s18_all.append(s18)
        s10_all.append(s10)
        case_ids.append(case_id)
    return np.stack(s18_all), np.stack(s10_all), np.asarray(case_ids)


def rle(sched: np.ndarray, enum_cls) -> str:
    """スロット列を RLE 表記に (目視検証用): 'PERSONAL_CARE*28 | WORK*16 | ...'"""
    parts, run_val, run_len = [], int(sched[0]), 1
    for v in sched[1:]:
        if int(v) == run_val:
            run_len += 1
        else:
            parts.append(f"{enum_cls(run_val).name}*{run_len}")
            run_val, run_len = int(v), 1
    parts.append(f"{enum_cls(run_val).name}*{run_len}")
    return " | ".join(parts)


def main():
    print("loading ...")
    act = load_activity()
    S18, S10, case_ids = build_degraded(act)

    # --- 検証1: T との diary 集合一致 (D1-a ペア比較の前提) ---
    t_dataset = pd.read_csv(OUT_DATASET)
    assert (t_dataset["TUCASEID"].to_numpy() == case_ids).all(), \
        "T' の diary 集合が T と一致しない (preprocess.py を先に実行し、クレンジング規則を揃えること)"
    print(f"採用 diary: {len(case_ids)} (T と完全一致を確認)")

    # --- 条件・重み結合 + 保存 (T と同一フォーマット) ---
    cond = attach_conditions(case_ids)
    cond_cols = cond[COND_COLS].reset_index(drop=True)
    slot_cols = [f"s{k}" for k in range(N_SLOTS)]
    for S, out_path in [(S18, OUT_TRIPLIKE18), (S10, OUT_NHTS10)]:
        merged = pd.concat([cond_cols, pd.DataFrame(S, columns=slot_cols)], axis=1)
        merged.to_csv(out_path, index=False)
        print(f"saved: {merged.shape} -> {out_path}")

    # --- 検証2: 残差化の確認 (T'=HOME スロットの T 側活動内訳) ---
    S_t = t_dataset[slot_cols].to_numpy(dtype=np.int8)
    home_mask = S18 == ActT18.HOME
    print(f"\n=== T'(triplike18) HOME シェア: {home_mask.mean():.1%} (スロットベース) ===")
    print("T'=HOME スロットの T 側活動内訳 (上位; 在宅系が潰されていることの確認):")
    counts = np.bincount(S_t[home_mask], minlength=NUM_ACT) / home_mask.sum()
    for i in np.argsort(-counts)[:6]:
        print(f"  {Act(i).name:19s}: {counts[i]:6.1%}")

    # 逆向き: T'がHOME以外なのにTが在宅系(personal care)のスロットは少ないはず
    leak = (~home_mask & (S18 != ActT18.TRAVEL) & (S_t == Act.PERSONAL_CARE)).mean()
    print(f"T'=自宅外活動 かつ T=PERSONAL_CARE のスロット率 (漏れの目安): {leak:.2%}")

    # --- 検証3: クラス時間シェア ---
    for name, S, enum_cls, num in [("triplike18", S18, ActT18, NUM_T18), ("nhts10", S10, ActN10, NUM_N10)]:
        print(f"\n=== {name} 時間シェア ===")
        share = np.bincount(S.ravel(), minlength=num) / S.size
        for a in enum_cls:
            print(f"  {a.name:22s}: {share[a]:6.1%}")

    # --- 検証4: サンプル diary の目視ダンプ (T vs T') ---
    rng = np.random.default_rng(0)
    print("\n=== サンプル diary (RLE; 04:00起点 15分スロット) ===")
    for idx in rng.choice(len(case_ids), size=2, replace=False):
        print(f"\n- TUCASEID={case_ids[idx]}")
        print(f"  T  : {rle(S_t[idx], Act)}")
        print(f"  T18: {rle(S18[idx], ActT18)}")
        print(f"  N10: {rle(S10[idx], ActN10)}")


if __name__ == "__main__":
    main()
