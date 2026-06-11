"""
NHTS 2022 社会属性 -> 条件ベクトル構築 (5属性)

5属性:
    age                : R_AGE (5..92)
    gender             : R_SEX (欠損<0 は R_SEX_IMP で補完), 1=male, 2=female -> {0:male,1:female}
    num_member         : HHSIZE (1..10)
    role_household_type: 8分類, 世帯単位で R_RELAT と Spouse有無, R_SEX から判定
    worker_status      : WORKER==1 -> 1(就業), それ以外(2 および -1=子供非該当) -> 0(非就業)

role_household_type の 8 カテゴリ(index):
    0 単独世帯(男性)  1 単独世帯(女性) 2 夫・男親  3 妻・女親
    4 子供(男性)     5 子供(女性)     6 親(男性) 7 親(女性)
    ※ Brother/Sister(4),Other relative(5),Not related(6) は -1(その他) とし、後段で扱いを決める
"""

import numpy as np
import pandas as pd

# R_RELAT コード
SELF, SPOUSE, CHILD, PARENT = 7, 1, 2, 3
MISSING = {-1, -7, -8, -9}

ROLE = {
    "single_m":0, "single_f":1, "husband":2, "wife":3,
    "child_m":4, "child_f":5, "parent_m":6, "parent_f":7, "other":-1,
}

PERSON_PATH = "../../data/raw/NHTS2022/perv2pub.csv"
HOUSEHOLD_PATH = "../../data/raw/NHTS2022/hhv2pub.csv"
KEYS_PATH = "../../data/processed/schedule_keys.csv"


def resolve_gender(r_sex, r_sex_imp):
    s = r_sex if r_sex not in MISSING else r_sex_imp
    # 1=male->0, 2=female->1
    return 0 if s == 1 else 1


def build_conditions(person_path: str, household_path: str, keys_path: str) -> pd.DataFrame:
    per = pd.read_csv(person_path,
                        usecols=["HOUSEID","PERSONID","R_AGE",
                                "R_SEX","R_SEX_IMP","WORKER","R_RELAT"])
    hh  = pd.read_csv(household_path, usecols=["HOUSEID","HHSIZE"])

    # スケジュールを作れた個人だけに絞る
    keys = pd.read_csv(keys_path)
    per = keys.merge(per, on=["HOUSEID","PERSONID"], how="left")
    per = per.merge(hh, on="HOUSEID", how="left")

    # 世帯ごとに Spouse が存在するか(role判定に必要)
    spouse_present = (
        per.assign(is_sp=(per["R_RELAT"]==SPOUSE))
            .groupby("HOUSEID")["is_sp"].any()
    )
    per["spouse_in_hh"] = per["HOUSEID"].map(spouse_present)

    def role_of(row):
        rel = int(row["R_RELAT"])
        g = resolve_gender(int(row["R_SEX"]), int(row["R_SEX_IMP"]))  # 0=m,1=f
        hhsize = int(row["HHSIZE"])
        if rel == SELF or rel == SPOUSE:
            # 単独 or 夫婦の一方 or 配偶者なしの親(ひとり親)
            if hhsize == 1:
                return ROLE["single_m"] if g==0 else ROLE["single_f"]
            # 複数人世帯: 夫婦なら husband/wife、Spouseなしの self はひとり親として husband/wife に吸収
            return ROLE["husband"] if g==0 else ROLE["wife"]
        elif rel == CHILD:
            return ROLE["child_m"] if g==0 else ROLE["child_f"]
        elif rel == PARENT:
            return ROLE["parent_m"] if g==0 else ROLE["parent_f"]
        else:  # 4,5,6,-9 など
            return ROLE["other"]

    per["age"] = per["R_AGE"].astype(int)
    per["gender"] = [resolve_gender(int(s), int(si)) for s,si in zip(per["R_SEX"], per["R_SEX_IMP"])]
    per["num_member"] = per["HHSIZE"].astype(int)
    per["role_household_type"] = per.apply(role_of, axis=1)
    per["worker_status"] = (per["WORKER"]==1).astype(int)  # 1=就業, それ以外=0

    cond = per[["HOUSEID","PERSONID","age","gender","num_member","role_household_type","worker_status"]]
    return cond

def main():
    cond = build_conditions(PERSON_PATH, HOUSEHOLD_PATH, KEYS_PATH)
    cond.to_csv("../../data/processed/conditions.csv", index=False)

    print(f"条件ベクトル構築: {len(cond)} 人")
    print("\n=== role_household_type 分布 ===")
    names = {0:"単独(男)",1:"単独(女)",2:"夫・男親",3:"妻・女親",
            4:"子供(男)",5:"子供(女)",6:"親(男)",7:"親(女)",-1:"その他"}
    for k,v in cond["role_household_type"].value_counts().sort_index().items():
        print(f"  {names[int(k)]:9s}: {v}") # type: ignore
    print("\n=== worker_status 分布 ===", dict(cond["worker_status"].value_counts()))
    print("=== gender 分布 ===", dict(cond["gender"].value_counts()))
    print("=== age 範囲 ===", cond["age"].min(), "..", cond["age"].max())

    # 妥当性チェック: 子供(role 4/5)は worker=0 のはず
    child = cond[cond["role_household_type"].isin([4,5])]
    print(f"\n子供のうち worker=1(就業)の数: {int((child['worker_status']==1).sum())} / {len(child)}")
    # その他(-1)がどれくらいか
    print(f"role=その他(-1) の数: {int((cond['role_household_type']==-1).sum())}")

if __name__ == "__main__":
    main()