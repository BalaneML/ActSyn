"""
stage2_targets.py
=================
Stage 2 の教師テンソル A*(28,12,96) と、28群表への採点（Stage2_design.md §3, §9）

教師は e-Stat が誰でも落とせる公表統計表 1 ファイルだけから作る。日本側の個票は
1 本も使わない。この分離が Stage 2 の主張を成立させている。

    調査   : 社会生活基本調査 令和3年(2021) 調査票A
    表     : 第8-1表「曜日,男女,ふだんの就業状態,行動の種類,年齢,時刻区分別行動者率(15歳以上)」
    範囲   : region == "00_全国"、平日、15歳以上人口の 99.6% を覆う28群
             （ふだんの就業状態が不詳の 411千人 = 0.38% は公表表に区分が無く対象外）

前処理の分担（§3.3 の P1〜P8）:
    P1〜P4  上流の src/common/preprocess/stula/parse_timeband.py が担う。
            欠損マーカーの分離（'-'→0.0 / '…','X'→NaN）、日界の回転（0:00起点→04:00起点）、
            率が非定義な行の除去、標本なし層の復元。★ここに書き写さない
    P5〜P8  本ファイル。行動20→12（単純和）、年齢15→7（人口加重平均）、
            単位変換（%→[0,1]）、全国・28群への切り出し

★他のモデルフォルダを import しない（§10.1）。CVAE_Aggregate/japan_match_experiment.py
  から load_stula_targets / nan_renorm_pop_weights / eval_against / d_index を移植してある。
  移植元は `from model import AggCVAE` という素の import を持ち、sys.path の順序次第で
  別の model.py を掴む。Stage 2 は自己完結にしてこの経路を断つ。
  依存してよいのは src/common/preprocess/stula（共通の分類定義）だけである。

★torch を import しない。教師の構築は numpy/pandas だけで完結する。

使い方:
    from stage2_targets import load_stula_targets, eval_both
    tgt = load_stula_targets()                      # {"group_rates_tbl": (28,12,96), "pop": (2,7,2)}
    rows = eval_both(pool_to_rates(pool), tgt)      # 11act / 12act の2行
"""
import sys
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, NUM_COMMON, EXCLUDED, stula_to_common  # noqa: E402

STULA_DIR = REPO_ROOT / "data" / "processed" / "stula"
DEFAULT_TABLE = "timeband_weekday"

# 群定義。model.py:171 と同一でなければならない（test_stage2 の (d_idx) が担保する）
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E          # 28
NUM_SLOTS = 96
SLOT_COLS = [f"s{j}" for j in range(NUM_SLOTS)]

# 社会生活基本調査の5歳15区分 → モデルの10歳7区分
AGE15_TO_7 = {i: min((i - 1) // 2, 6) for i in range(1, 16)}


def d_index(g: int, a: int, e: int) -> int:
    """(性, 年齢7区分, 就業) -> 群インデックス 0..27。model.py:171 と同じ規則。"""
    return g * (N_A * N_E) + a * N_E + e


def mask_12act() -> npt.NDArray[np.bool_]:
    """損失と評価の12活動マスク（OTHER_X を含む）。実際に最適化する対象そのもの。"""
    return np.ones(NUM_COMMON, dtype=bool)


def mask_11act() -> npt.NDArray[np.bool_]:
    """既存の報告値と比較するための11活動マスク（OTHER_X を除く）。

    crosswalk_atus_stula.EXCLUDED の規約は「両側の OTHER_X は構成が違うので
    『揃った』と扱わない」という評価の比較可能性の話であって、損失に使うかとは別問題。
    Stage 2 の損失は12チャネルで、評価だけ両方を出す（§6 修正3）。
    """
    m = np.ones(NUM_COMMON, dtype=bool)
    for c in EXCLUDED:
        m[int(c)] = False
    return m


# ============================================================
# 教師 A* の構築（P5〜P8）
# ============================================================
def load_stula_targets(name: str = DEFAULT_TABLE) -> dict:
    """timeband CSV -> {"group_rates_tbl": (28,12,96), "pop": (2,7,2)}。率は[0,1]、NaN=非公表。

    ★移植元にあった margin_ga / margin_ge / pi_e は移植しない。あれらは公表2表
      （性×年齢 / 性×就業）への周辺マッチ用で、28群クロス表を直接教師にする
      Stage 2 では使わない。pop は eval_against が dev_* を作るのに要るので残す。

    ★全国・平日では非公表セルは 0 個である（欠測は11大都市圏の145層でだけ起きる）。
      NaN 経路は地域軸へ拡張したときのために残してあり、現状は1セルも通らない。
    """
    df = pd.read_csv(STULA_DIR / f"{name}.csv")
    df = cast(pd.DataFrame, df[df["region"] == "00_全国"])       # P8: 全国のみ
    com = stula_to_common(df)                                    # P5: 行動20 -> 12（単純和）
    com["g"] = com["gender"].map({"1_男": 0, "2_女": 1})          # type: ignore
    com["e"] = com["employment"].map({"1_有業者": 1, "2_無業者": 0})  # type: ignore
    com["a15"] = com["age_class"].str[:2].map(
        lambda c: int(c) if isinstance(c, str) and c.isdigit() and c != "00" else np.nan)
    com["a7"] = com["a15"].map(AGE15_TO_7)                        # type: ignore
    com["ci"] = com["common"].map({c.name: int(c) for c in Common})  # type: ignore

    # 人口（推定人口・千人）は 行動="00_総数" の行にしか入っていない。
    # 人口は層の属性であって行動の属性ではないため。これが総数行を捨てられない理由で、
    # 率は非定義でも人口の唯一の供給源になっている
    pop_src = cast(pd.DataFrame, df[df["activity"].str.startswith("00_")])
    pop_src = pop_src.assign(
        g=pop_src["gender"].map({"1_男": 0, "2_女": 1}),            # type: ignore
        e=pop_src["employment"].map({"1_有業者": 1, "2_無業者": 0}),  # type: ignore
        a15=pop_src["age_class"].str[:2].map(
            lambda c: int(c) if isinstance(c, str) and c.isdigit() and c != "00" else np.nan),
    ).dropna(subset=["g", "e", "a15"])
    pop15 = np.zeros((N_G, 16, N_E))          # a15 は 1..15（index 0 は使わない）
    for (g, a15, e), grp in pop_src.groupby(["g", "a15", "e"]):   # type: ignore
        pop15[int(g), int(a15), int(e)] = grp["population_k"].iloc[0]  # type: ignore
    # 5歳2区分ぶんの人口を足して10歳7区分へ。率でなく人数なので単純加算でよい
    pop = np.zeros((N_G, N_A, N_E))
    for a15, a7 in AGE15_TO_7.items():
        pop[:, a7, :] += pop15[:, a15, :]

    # P6: 年齢15 -> 7 は人口加重平均。単純平均だと人口が4分の1の高校生層が
    # 半分の発言権を持ってしまう（男・有業・仕事・15時台で 39.79% vs 47.24%、差 7.4pt）。
    # 公表の再掲行 R3_65歳以上 との照合で、加重平均が 0.009pt・単純平均が 9.256pt ずれる
    sub = com.dropna(subset=["g", "e", "a7"])
    g_i = sub["g"].to_numpy(dtype=np.int64)
    a_i = sub["a7"].to_numpy(dtype=np.int64)
    e_i = sub["e"].to_numpy(dtype=np.int64)
    c_i = sub["ci"].to_numpy(dtype=np.int64)
    a15_i = sub["a15"].to_numpy(dtype=np.int64)
    w_row = pop15[g_i, a15_i, e_i]                                  # 行の重み ＝ 5歳区分の人口
    vals = sub[SLOT_COLS].to_numpy(dtype=float) / 100.0             # P7: % -> [0,1]、(行数, 96)

    # 非公表はスロット単位で分子・分母の両方から外す。全行が非公表なら 0/0 = NaN のまま残る
    published = ~np.isnan(vals)
    wsum = np.zeros((N_G, N_A, N_E, NUM_COMMON, NUM_SLOTS))
    acc = np.zeros((N_G, N_A, N_E, NUM_COMMON, NUM_SLOTS))
    # ★np.add.at は重複インデックスを順に足し込む。同じ (性,年齢7,就業,活動) に
    #   5歳2区分ぶんの行が落ちるので、通常の acc[idx] += では後の行が前の行を上書きしてしまう
    idx = (g_i, a_i, e_i, c_i)
    np.add.at(acc, idx, np.where(published, w_row[:, None] * vals, 0.0))
    np.add.at(wsum, idx, np.where(published, w_row[:, None], 0.0))
    with np.errstate(invalid="ignore"):
        group_rates_tbl = (acc / wsum).reshape(D_GROUPS, NUM_COMMON, NUM_SLOTS)

    return {"group_rates_tbl": group_rates_tbl, "pop": pop}


# ============================================================
# 28群表への採点
# ============================================================
def nan_renorm_pop_weights(grp_tbl: np.ndarray, pi_d: np.ndarray) -> np.ndarray:
    """セル (活動,時刻) ごとに非公表 NaN の群を除外して再正規化した人口重み (28,12,96)。

    全群 NaN のセルは 0/0 = NaN（評価マスク m で除外されるので害はない）。
    """
    w = np.where(np.isnan(grp_tbl), 0.0, pi_d[:, None, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return w / w.sum(axis=0)


def eval_against(mu_hat: np.ndarray, tgt: dict,
                 mask_c: npt.NDArray[np.bool_]) -> dict:
    """μ̂ (D, 12*96 act-major) vs 公表の群別行動者率。NaN と mask_c 外を除外して採点する。

    D は教師テンソルの群数。全28群でも、LGO で絞った部分集合でも同じ定義で採点する。

    MAE と RMSE を必ず併記する。教師セルは率<0.01 のセルが約半数を占める強い偏りがあり、
    両者は別のものを測るため:
        MAE  = 平均何ポイントずれるか。解釈可能な主指標だが、「稀な活動をほぼゼロと出す」
                だけで下がる（多様性を犠牲にすると得をする）
        RMSE = 二乗和が誤差上位セルに集中するため、実質「ピーク帯（WORK昼・SLEEP深夜）が
                どれだけ合うか」の指標。裾に敏感な補助指標として残す
        MSE  = RMSE の二乗で順位情報は同一。学習損失と単位を揃えて読むためだけに置く

    ★rate_* と dev_* は別物。rate_* は群ごとの率そのものの誤差、dev_* は人口平均からの
      群のズレの誤差 ＝ 条件付け能力。dev_* は共通成分が引かれるぶん必ず小さくなる。

    ★mask 列を返す。このリポジトリは数値の出所取り違えを2回起こしているので、
      11act/12act のどちらで測った値かを機械的に区別できる形にしておく（§9.4）。
    """
    # ★群数は教師テンソルから読む。28群固定にしないのは、LGO で教師群と held-out 群を
    #   分けて採点するときに同じ定義をもう一度書かずに済ませるため（stage2_select）
    grp_tbl = tgt["group_rates_tbl"]                        # (D,12,96)
    n_d = grp_tbl.shape[0]
    mh = mu_hat.reshape(n_d, NUM_COMMON, NUM_SLOTS)
    pop = np.asarray(tgt["pop"]).reshape(-1)
    pi_d = pop / pop.sum()
    m = ~np.isnan(grp_tbl) & mask_c[None, :, None]
    err = (mh - grp_tbl)[m]
    mse = float((err ** 2).mean())

    # 群偏差: 非公表 NaN の群をセルごとに除外して π を再正規化し、モデル側の平均も
    # 同じ群集合で取る（教師とモデルで比較対象の「人口平均」を揃える）
    W = nan_renorm_pop_weights(grp_tbl, pi_d)
    dev = ((mh - np.nansum(mh * W, axis=0)) - (grp_tbl - np.nansum(grp_tbl * W, axis=0)))[m]

    per_act: dict[str, float] = {}
    for c in Common:
        if not mask_c[int(c)]:
            continue
        sel = m[:, int(c)]
        e = (mh - grp_tbl)[:, int(c)][sel]
        mae_c = float(np.abs(e).mean())
        # 相対誤差 = mae / 教師平均率。量の水準が違う活動を同じ土俵に載せる。
        # 例: SLEEP_PERSONAL 0.0542/0.3836 = 0.14 に対し TRAVEL 0.0249/0.0347 = 0.72
        q_bar = float(grp_tbl[:, int(c)][sel].mean())
        per_act[f"mae_{c.name}"] = mae_c
        per_act[f"rmse_{c.name}"] = float(np.sqrt((e ** 2).mean()))
        per_act[f"rel_{c.name}"] = mae_c / q_bar if q_bar > 0 else float("nan")

    return {"mask": "12act" if bool(mask_c.all()) else "11act",
            "n_cells": int(m.sum()),
            "rate_mae": float(np.abs(err).mean()),
            "rate_mse": mse,
            "rate_rmse": float(np.sqrt(mse)),
            "max_abs_err": float(np.abs(err).max()),
            "dev_mae": float(np.abs(dev).mean()),
            "dev_rmse": float(np.sqrt((dev ** 2).mean())),
            **per_act}


def eval_both(mu_hat: np.ndarray, tgt: dict) -> list[dict]:
    """12act（学習目的そのもの）と 11act（既存の報告値と比較可能）の2行を返す。

    どちらを主指標にするかではなく、両方出して mask 列で区別する（§6 修正3）。
    """
    return [eval_against(mu_hat, tgt, mask_12act()),
            eval_against(mu_hat, tgt, mask_11act())]
