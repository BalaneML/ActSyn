"""
D2-a 集計表シミュレータ + σ_min(P) 識別可能性診断

目的 (実験デザイン D2-a, 計画ファイル参照):
    ATUS個票を「擬似日本」に見立て、社会生活基本調査・時間帯編と同形式の
    時刻別行動者率テーブルを自作する。個票 ground truth があるので、
    集計マッチ学習の復元誤差と ATLAS の理論予測 1/σ_min(P) の関係を検証できる。

生成する集計は2系統 (計画ファイル「日本側の集計教師供給源」に対応):
    (i) 群別テーブル: 社基調「男女, 年齢階級別」相当。
        デモグラ群 d = gender × age_class (× daytype) ごとの時刻別行動者率 μ(d) を直接公表する形式。
    (ii) セル混合テーブル: 社基調「都道府県別」等の地域集計に相当。
        セル g (層別変数の組) ごとの行動者率 ν(g) = Σ_d p(d|g) μ(d) のみを公表する形式。
        群分布 μ(d) の復元可能性は構成行列 P = [p(d|g)] の σ_min で決まる (ATLAS Condition 1)。

公表形式の模倣:
    社基調・時間帯編は行動者率を % 1桁 (0.1%単位) で公表 → rate_pub 列に丸め値を併記し、
    丸め誤差の感度分析 (計画ファイル「公表集計の測定誤差」) に使えるようにする。

σ_min(P) 診断:
    - P は G×D 行列 (行=セル g の重み付きデモグラ構成比, 行和=1)
    - フルカラムランク (σ_min > 0 かつ G ≥ D) が識別可能性の必要条件、
      復元誤差は 1/σ_min(P) で増幅される (ATLAS Theorem 1)
    - σ_min はセル数・セル標本サイズと交絡するため、セル数 G と最小セル標本数も併記する
      (交絡統制は分析側で行う; 実験デザイン D2-a)

注意:
    地域変数 (region/state_fips/metro) は atuscps_2024.dat (CPSファイル) 由来で、
    preprocess.attach_conditions が結合する。列が無い入力 (atuscps 未結合の旧CSV) では
    地域系セル層別は自動スキップされる。

入出力:
    入力: data/processed/atus2024_weighted_dataset.csv (T; --dataset で T'/N10 に切替可)
    出力: data/processed/aggregates/<データセットstem>/ (T / T' の相互上書き防止)
        timeuse_rates_by_group.csv          (i) 群別 時刻別行動者率
        timeuse_rates_by_cell_<spec>.csv    (ii) セル混合 時刻別行動者率 (スイープ各層別)
        constitution_P_<spec>.csv           構成行列 P
        sigma_min_sweep.csv                 層別スイープの σ_min 診断表
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/common/aggregates -> repo root
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "atus2024_weighted_dataset.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

N_SLOTS = 96
SLOT_COLS = [f"s{k}" for k in range(N_SLOTS)]

# 年齢階級: 社基調の公表10歳階級に合わせる (ATUSは15..85)
AGE_BINS   = [15, 25, 35, 45, 55, 65, 75, 200]
AGE_LABELS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

# デモグラ群 d の定義 (2粒度; σ_min スイープで比較)
DEMO_SPECS = {
    "sex_x_age7": ["gender", "age_class"],    # D = 2×7 = 14
    "sex_x_age3": ["gender", "age3_class"],   # D = 2×3 = 6 (粗い群)
}

# セル層別 g のスイープ。地域系 (region/state_fips/metro) は atuscps 結合後のみ利用可 —
# 入力に列が無ければ実行時にスキップする。state×daytype が社基調「都道府県×曜日」の最近傍。
CELL_SPECS = {
    "daytype":                 ["daytype"],                              # G=3
    "daytype_x_emp":           ["daytype", "employed"],                  # G=6
    "daytype_x_emp_x_sex":     ["daytype", "employed", "gender"],        # G=12
    "dow_x_emp":               ["day_of_week", "employed"],              # G=14
    "dow_x_emp_x_sex":         ["day_of_week", "employed", "gender"],    # G=28
    "region_x_daytype":        ["region", "daytype"],                    # G=12
    "region_x_daytype_x_emp":  ["region", "daytype", "employed"],        # G=24
    "state":                   ["state_fips"],                           # G≈51
    "state_x_daytype":         ["state_fips", "daytype"],                # G≈153 (都道府県×曜日相当)
}


def derive_strata(df: pd.DataFrame) -> pd.DataFrame:
    """層別に使う派生変数を付与"""
    df = df.copy()
    df["age_class"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df["age3_class"] = pd.cut(df["age"], bins=[15, 35, 65, 200],
                              labels=["15-34", "35-64", "65+"], right=False)
    # 曜日 1=Sun..7=Sat -> 平日/土/日 (社基調時間帯編の曜日区分)
    df["daytype"] = np.select([df["day_of_week"] == 1, df["day_of_week"] == 7],
                              ["sunday", "saturday"], default="weekday")
    # TELFS 1=就業(at work), 2=就業(absent) -> 就業; それ以外 (失業・非労働力) -> 非就業
    df["employed"] = df["telfs"].isin([1, 2]).astype(int)
    return df


def participation_rates(df: pd.DataFrame, by: list[str], n_act: int) -> pd.DataFrame:
    """重み付き時刻別行動者率: by の各セル × slot × activity -> rate

    rate = Σ_{i∈セル, s_it=a} w_i / Σ_{i∈セル} w_i  (行動者率; 各slotで活動aをしている人の割合)
    rate_pub = %表記 0.1単位への丸め (社基調の公表形式の模倣)
    """
    long = df.melt(id_vars=by + ["TUFINLWGT"], value_vars=SLOT_COLS,
                   var_name="slot", value_name="act")
    long["slot"] = long["slot"].str.removeprefix("s").astype(int)
    w_cell = long.groupby(by + ["slot"], observed=True)["TUFINLWGT"].sum()
    w_act  = long.groupby(by + ["slot", "act"], observed=True)["TUFINLWGT"].sum()
    rates = (w_act / w_cell).rename("rate").reset_index()
    # 全 (セル×slot×act) の完全な直積に展開 (出現しない活動は rate=0; 公表表と同じ密な形)
    full = pd.MultiIndex.from_product(
        [rates[c].unique() for c in by] + [range(N_SLOTS), range(n_act)],
        names=by + ["slot", "act"])
    rates = rates.set_index(by + ["slot", "act"]).reindex(full, fill_value=0.0).reset_index()
    rates["rate_pub"] = (rates["rate"] * 100).round(1)  # % 0.1単位
    return rates


def constitution_matrix(df: pd.DataFrame, cell_cols: list[str], demo_cols: list[str]
                        ) -> tuple[pd.DataFrame, dict]:
    """構成行列 P (G×D, 行=セル g のデモグラ構成比) と診断値を返す"""
    # デモグラ群を結合キー1列に畳む (セル層別と同じ変数を含む場合の名前衝突を回避)
    demo_key = df[demo_cols].astype(str).agg("_".join, axis=1).rename("_demo")
    w = df.assign(_demo=demo_key).groupby(cell_cols + ["_demo"], observed=True)["TUFINLWGT"] \
          .sum().unstack("_demo", fill_value=0.0)
    P = w.div(w.sum(axis=1), axis=0)

    n_cell = df.groupby(cell_cols, observed=True).size()  # セル標本サイズ (交絡の報告用)
    sv = np.linalg.svd(P.to_numpy(), compute_uv=False)
    G, D = P.shape
    sigma_min = float(sv[-1]) if G >= D else 0.0  # G<D は原理的にランク落ち
    diag = {
        "G": G, "D": D,
        "sigma_min": sigma_min,
        "error_amp": float("inf") if sigma_min == 0 else 1.0 / sigma_min,
        "rank": int(np.linalg.matrix_rank(P.to_numpy())),
        "full_col_rank": bool(G >= D and np.linalg.matrix_rank(P.to_numpy()) == D),
        "min_cell_n": int(n_cell.min()),
    }
    return P, diag


def main():
    ap = argparse.ArgumentParser(description="社基調時間帯編スタイル集計 + σ_min(P) 診断")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="processed dataset CSV (T / triplike18 / nhts10)")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    df = derive_strata(df)
    n_act = int(df[SLOT_COLS].to_numpy().max()) + 1
    out_dir = OUT_DIR / args.dataset.stem  # データセット別に分離 (T / T' の相互上書き防止)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {args.dataset.name}  N={len(df)}  活動クラス数={n_act}")

    # --- (i) 群別テーブル (社基調「男女, 年齢階級別」相当) ---
    group_rates = participation_rates(df, ["daytype", "gender", "age_class"], n_act)
    path = out_dir / "timeuse_rates_by_group.csv"
    group_rates.to_csv(path, index=False)
    print(f"(i) 群別 時刻別行動者率: {group_rates.shape} -> {path}")

    # --- (ii) セル混合テーブル + σ_min(P) スイープ ---
    print("\n=== σ_min(P) 識別可能性スイープ (ATLAS Cond.1: full column rank, 誤差増幅 1/σ_min) ===")
    rows = []
    for cell_name, cell_cols in CELL_SPECS.items():
        if not set(cell_cols) <= set(df.columns):
            print(f"  [skip] {cell_name}: 列 {cell_cols} が無い (atuscps 未結合の入力)")
            continue
        cell_rates = participation_rates(df, cell_cols, n_act)
        cell_rates.to_csv(out_dir / f"timeuse_rates_by_cell_{cell_name}.csv", index=False)
        for demo_name, demo_cols in DEMO_SPECS.items():
            P, diag = constitution_matrix(df, cell_cols, demo_cols)
            P.to_csv(out_dir / f"constitution_P_{cell_name}__{demo_name}.csv")
            rows.append({"cell_spec": cell_name, "demo_spec": demo_name, **diag})

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "sigma_min_sweep.csv", index=False)
    with pd.option_context("display.width", 200):
        print(sweep.to_string(index=False,
                              formatters={"sigma_min": "{:.4f}".format,
                                          "error_amp": lambda v: "inf" if np.isinf(v) else f"{v:.1f}"}))
    print("\n判定: full_col_rank=True のセル層別だけが「集計のみからの群分布復元」の必要条件を満たす。")
    print("      σ_min はセル数 G・最小セル標本数と交絡するため、比較は同 G 内で行うこと (D2-a)。")


if __name__ == "__main__":
    main()
