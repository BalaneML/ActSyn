"""
sa_targets.py
=============
SA（比較 a）の入力を、社会生活基本調査の公表統計 2 表から作る（SA_setup.md §2）

    時刻別行動者率  timeband_weekday.csv      → rates20 (28, 20, 96)
    日次行動者率    timeuse_participation.csv → part20  (28, 20)

どちらも 20 分類のまま返す。12 分類へは評価のときに STULA20_TO_COMMON でまとめる。
群の番号は d = g·14 + a·2 + e（g: 0=男 1=女、a: 年齢 7 区分、e: 0=無業 1=有業）で、
stage2_targets.load_stula_targets の reshape(N_G, N_A, N_E) と同じ並びである。

★個票は 1 本も読まない。torch も import しない。
"""
import sys
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import STULA_TO_COMMON  # noqa: E402

STULA_DIR = REPO_ROOT / "data" / "processed" / "stula"

N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E          # 28
NUM_ACT20 = 20
NUM_SLOTS = 96
SLOT_COLS = [f"s{j}" for j in range(NUM_SLOTS)]

# 社会生活基本調査の 5 歳 15 区分 → モデルの 10 歳 7 区分（stage2_targets と同じ）
AGE15_TO_7 = {i: min((i - 1) // 2, 6) for i in range(1, 16)}

# ★列は cast(pd.Series, ...) で Series に絞ってから map する。df["col"] はスタブ上 Series と
#   DataFrame の union なので、辞書を渡すと DataFrame.map（callable のみ）に解決されて型の警告になる
GENDER_CODE = {"1_男": 0, "2_女": 1}
EMPLOY_CODE = {"1_有業者": 1, "2_無業者": 0}

# 20 分類の番号（0 始まり, k = 活動コード − 1） → 共通 12 分類
STULA20_TO_COMMON: npt.NDArray[np.int64] = np.array(
    [int(STULA_TO_COMMON[f"{k + 1:02d}"]) for k in range(NUM_ACT20)], dtype=np.int64)


def group_index(g: int, a: int, e: int) -> int:
    """属性 (性, 年齢 7 区分, 就業) → 群の番号 d"""
    return (g * N_A + a) * N_E + e


def load_rates20(name: str = "timeband_weekday") -> npt.NDArray[np.float64]:
    """時刻別行動者率を 20 分類のまま 28 群へまとめる

    年齢 15 → 7 区分は 5 歳区分の推定人口による加重平均（load_stula_targets の P6 と同じ）。
    公表値は四捨五入されているので、スロットごとの 20 活動の和は 1 からわずかにずれる。
    初期解の人数を決めるために、ここで和を 1 に揃える。

    Args:
        name: data/processed/stula/ の CSV の名前（拡張子なし）

    Returns:
        rates20 (28, 20, 96)。各 (d, s) で活動の和が 1

    Raises:
        ValueError: 非公表（NaN）のセルがあるとき。全国・平日には無い前提
    """
    df = pd.read_csv(STULA_DIR / f"{name}.csv")
    df = cast(pd.DataFrame, df[df["region"] == "00_全国"])
    g = cast(pd.Series, df["gender"]).map(GENDER_CODE)
    e = cast(pd.Series, df["employment"]).map(EMPLOY_CODE)
    a15 = cast(pd.Series, cast(pd.Series, df["age_class"]).str[:2].map(
        lambda c: int(c) if isinstance(c, str) and c.isdigit() and c != "00" else np.nan))
    act = cast(pd.Series, cast(pd.Series, df["activity"]).str[:2].map(
        lambda c: int(c) if isinstance(c, str) and c.isdigit() and c != "00" else np.nan))
    keep = g.notna() & e.notna() & a15.notna()

    # 推定人口は 行動 = "00_総数" の行にだけ入っている
    pop15 = np.zeros((N_G, 16, N_E))
    is_total = keep & (df["activity"] == "00_総数")
    for gi, ai, ei, p in zip(g[is_total], a15[is_total], e[is_total], df.loc[is_total, "population_k"]):
        pop15[int(gi), int(ai), int(ei)] = float(p)

    rows = keep & act.notna() & (act <= NUM_ACT20)
    vals = df.loc[rows, SLOT_COLS].to_numpy(dtype=np.float64) / 100.0
    if np.isnan(vals).any():
        raise ValueError(f"{name}: 全国の時刻別行動者率に非公表のセルがある")
    g_i = g.loc[rows].to_numpy(dtype=np.int64)
    a15_i = a15.loc[rows].to_numpy(dtype=np.int64)
    e_i = e.loc[rows].to_numpy(dtype=np.int64)
    k_i = act.loc[rows].to_numpy(dtype=np.int64) - 1
    a7_i = np.array([AGE15_TO_7[int(x)] for x in a15_i], dtype=np.int64)
    w = pop15[g_i, a15_i, e_i]

    acc = np.zeros((N_G, N_A, N_E, NUM_ACT20, NUM_SLOTS))
    wsum = np.zeros((N_G, N_A, N_E, NUM_ACT20, NUM_SLOTS))
    # ★同じセルに 5 歳 2 区分ぶんの行が落ちるので np.add.at で足し込む
    np.add.at(acc, (g_i, a7_i, e_i, k_i), w[:, None] * vals)
    np.add.at(wsum, (g_i, a7_i, e_i, k_i), np.broadcast_to(w[:, None], vals.shape))
    rates = (acc / wsum).reshape(D_GROUPS, NUM_ACT20, NUM_SLOTS)
    return rates / rates.sum(axis=1, keepdims=True)


def load_participation20(name: str = "timeuse_participation") -> npt.NDArray[np.float64]:
    """日次行動者率（第70-3表）を 28 群 × 20 活動で返す, [0, 1]

    平日・全国・健康状態 総数の行を使う。年齢は 7 区分で公表されているので畳み込みは要らない。

    Returns:
        part20 (28, 20)

    Raises:
        ValueError: 28 群がそろわないとき、または NaN があるとき
    """
    df = pd.read_csv(STULA_DIR / f"{name}.csv")
    df = cast(pd.DataFrame, df[(df["daytype"] == "2_平日") & (df["region"] == "00_全国")
                                & (df["health"] == "0_総数")])
    g = cast(pd.Series, df["gender"]).map(GENDER_CODE)
    e = cast(pd.Series, df["employment"]).map(EMPLOY_CODE)
    a7 = cast(pd.Series, df["age_class"]).str[:1].map(
        lambda c: int(c) - 1 if isinstance(c, str) and c in "1234567" else np.nan)
    keep = g.notna() & e.notna() & a7.notna() & ~cast(pd.Series, df["age_class"]).str.startswith("R")
    sub = df[keep]
    if len(sub) != D_GROUPS:
        raise ValueError(f"{name}: 平日・全国の 28 群がそろわない（{len(sub)} 行）")
    part = np.full((D_GROUPS, NUM_ACT20), np.nan)
    act_cols = [f"a{k + 1:02d}" for k in range(NUM_ACT20)]
    for gi, ai, ei, row in zip(g[keep], a7[keep], e[keep], cast(pd.DataFrame, sub[act_cols]).to_numpy(dtype=np.float64)):
        part[group_index(int(gi), int(ai), int(ei))] = row / 100.0
    if np.isnan(part).any():
        raise ValueError(f"{name}: 日次行動者率に NaN がある")
    return part


def round_slot_counts(rates20: npt.NDArray[np.float64], n_people: int) -> npt.NDArray[np.int64]:
    """スロットごとの人数を最大剰余法で整数にする（SA_setup.md §4.1 手順 1）

    Args:
        rates20: (20, 96) 1 群ぶんの時刻別行動者率。各スロットの和が 1
        n_people: 群の人数 N

    Returns:
        counts (20, 96)。各スロットの和がちょうど N
    """
    exact = rates20 * n_people
    counts = np.floor(exact).astype(np.int64)
    for s in range(NUM_SLOTS):
        short = n_people - int(counts[:, s].sum())
        # 小数部の大きい活動から 1 人ずつ足す。同点は活動番号の小さい方を先にする
        order = np.argsort(-(exact[:, s] - counts[:, s]), kind="stable")
        counts[order[:short], s] += 1
    return counts


if __name__ == "__main__":
    r = load_rates20()
    p = load_participation20()
    print("rates20", r.shape, "part20", p.shape)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stage2_targets", REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage2_targets.py")
    assert spec is not None and spec.loader is not None
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    a12 = st.load_stula_targets()["group_rates_tbl"]
    r12 = np.zeros_like(a12)
    for k in range(NUM_ACT20):
        r12[:, STULA20_TO_COMMON[k]] += r[:, k]
    print("max |sum20->12 − A*| =", float(np.abs(r12 - a12).max()))
