"""
atus_group_rates.py
===================
★ATUS 個票から「群別・時刻別行動者率」(28群, 12活動, 96スロット) を作る唯一の出所。

これは Stage 2 の教師テンソル A* と**まったく同じ量を米国側で測ったもの**である。
両者の差が、zero-shot の誤差のうち「ドメインギャップ」の成分にあたる。

    A*         社会生活基本調査の公表表 (日本)   stage2_targets.load_stula_targets()
    本モジュール ATUS 個票 (米国)                group_rates()

軸の規約は A* に完全に揃えてある:
    群 d      28 = 性2 × 年齢7区分 × 就業2、d = g*(N_A*N_E) + a*N_E + e
    活動 c    共通12分類。並びは crosswalk_atus_stula.Common が唯一の出所
    スロット s 96 (15分刻み)、s0 = 04:00-04:15。ATUS の diary が元から 04:00 起点
    対象      平日のみ (day_of_week 2..6)。A* も平日表なので揃う

★群あたりの標本は薄い。最小 17人 / 最大 329人 (N=3,736)。
  n=17 の群では非ゼロ率の最小刻みが 1/17 = 0.059 になる一方、A* のセルは
  中央値 0.0112・48.1% が 0.01 以下である (Stage2_design.md §3.6)。
  つまり ATUS 側の分解能のほうが教師のセル値より粗い群が存在する。
  率だけを見て「ズレている」と言えないので、group_counts() の n_d / n_eff を必ず併記する。

使い方:
    .venv/bin/python3 src/eval/test_atus_group_rates.py   # テスト
    書き出しは src/common/astar/export_atus_rates.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "atus2024" / "atus2024_stula_common12_dataset.csv"

# 共通12分類の定義は crosswalk_atus_stula が唯一の出所。ここに並びを書き写さない
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, NUM_COMMON  # noqa: E402

NUM_SLOTS = 96
SLOT_COLS = [f"s{j}" for j in range(NUM_SLOTS)]

# 群定義 d = g*(N_A*N_E) + a*N_E + e。DDPM_Aggregate_Simple/model.py:174 と同一
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E       # 28

# ATUS TUDIARYDAY は 1=日曜 .. 7=土曜。平日は 2..6
WEEKDAY_CODES = [2, 3, 4, 5, 6]
WEIGHT_COL = "TUFINLWGT"

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]


def load_atus_weekday(path: Path = DATA_PATH) -> tuple[IntArr, IntArr, FloatArr]:
    """ATUS 平日の (スケジュール (N,96), 群インデックス (N,), 調査ウェイト (N,))。

    群への割り当ては model.py:115-118 の COND_SPEC と同じ式を使う
    (年齢は15歳起点10歳刻みで上を 75+ に潰す、telfs 1|2 = 有業)。
    """
    df = pd.read_csv(path)
    df = cast(pd.DataFrame, df[df["day_of_week"].isin(WEEKDAY_CODES)])
    g = df["gender"].to_numpy().astype(np.int64)
    a = np.clip((df["age"].to_numpy() - 15) // 10, 0, N_A - 1).astype(np.int64)
    e = np.isin(df["telfs"].to_numpy(), [1, 2]).astype(np.int64)
    d = (g * (N_A * N_E) + a * N_E + e).astype(np.int64)
    sched = df[SLOT_COLS].to_numpy().astype(np.int64)
    w = df[WEIGHT_COL].to_numpy().astype(np.float64)
    return sched, d, w


def group_rates(sched: IntArr, groups: IntArr,
                w: FloatArr | None = None) -> FloatArr:
    """群別の時刻別行動者率 (28, 12, 96)。w=None なら非加重。

    各 (d, s) で 12 活動の和は 1（1スロット1ラベルなので排他的）。
    標本が 1 人もいない群は NaN で埋める（A* の非公表セルと同じ扱いにできる）。

    ★w=None の経路は「0/1 の和を取ってから 1 回だけ割る」。einsum に 1/n の重みを
      渡す形と数学的には同じだが、浮動小数の加算順に依存して最下位ビットが動く。
      teacher_free_space.run() は群内シャッフルの前後でこの関数の値が
      「厳密に 0 差」であることを assert しているので、整数和の経路を崩してはいけない。
    """
    sched = np.asarray(sched)
    out = np.full((D_GROUPS, NUM_COMMON, NUM_SLOTS), np.nan, dtype=np.float64)
    for d in range(D_GROUPS):
        idx = np.where(groups == d)[0]
        if len(idx) == 0:
            continue
        onehot = np.eye(NUM_COMMON, dtype=np.float64)[sched[idx]]   # (n, 96, 12)
        if w is None:
            # 0/1 の和は float64 で厳密（n < 2^53）。加算順に依存しない
            out[d] = onehot.sum(axis=0).T / len(idx)
        else:
            wd = np.asarray(w, dtype=np.float64)[idx]
            s = wd.sum()
            if s <= 0:
                raise ValueError(f"群 {d} のウェイト総和が正でない")
            out[d] = np.einsum("n,nsc->cs", wd / s, onehot)
    return out


def group_counts(groups: IntArr, w: FloatArr) -> tuple[IntArr, FloatArr, FloatArr]:
    """(n_d (28,), wsum_d (28,), n_eff_d (28,))。

    n_eff = (Σw)² / Σw² は Kish の有効標本数で、加重後に残る実質的な分解能を表す。
    非加重なら n_eff = n_d、重みがばらつくほど n_eff < n_d になる。
    率の標準誤差はおよそ sqrt(p(1-p)/n_eff) なので、A* との差を読むときの尺度になる。
    """
    w = np.asarray(w, dtype=np.float64)
    n_d = np.bincount(groups, minlength=D_GROUPS).astype(np.int64)
    wsum = np.bincount(groups, weights=w, minlength=D_GROUPS).astype(np.float64)
    wsq = np.bincount(groups, weights=w ** 2, minlength=D_GROUPS).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eff = np.where(wsq > 0, wsum ** 2 / wsq, 0.0).astype(np.float64)
    return n_d, wsum, n_eff


def to_act_major(rates: FloatArr) -> FloatArr:
    """(28, 12, 96) -> (28, 12*96) act-major（インデックス c*96 + s）。

    DDPM_Aggregate_Simple/model.py:883 pool_to_rates と同じ並びなので、
    stage2_targets.eval_against(mu_hat, tgt, mask_c) にそのまま渡せる。
    """
    return np.asarray(rates, dtype=np.float64).reshape(rates.shape[0], NUM_COMMON * NUM_SLOTS)


def describe(sched: IntArr, groups: IntArr, w: FloatArr) -> pd.DataFrame:
    """群ごとの標本数・ウェイトシェア・分解能の一覧。率を読む前に見るための表。"""
    n_d, wsum, n_eff = group_counts(groups, w)
    return pd.DataFrame({
        "d": np.arange(D_GROUPS),
        "n_d": n_d,
        "n_eff": n_eff,
        "wshare": wsum / wsum.sum(),
        "nshare": n_d / n_d.sum(),
        "min_step": 1.0 / np.maximum(n_eff, 1.0),   # 表現できる率の最小刻み
    })


if __name__ == "__main__":
    sched, groups, w = load_atus_weekday()
    tbl = describe(sched, groups, w)
    print(f"N={len(sched)}  群={D_GROUPS}  活動={NUM_COMMON}  スロット={NUM_SLOTS}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(tbl.to_string(index=False))
    print(f"\nn_d  min={tbl['n_d'].min()} (d={tbl['n_d'].idxmin()})  "
          f"max={tbl['n_d'].max()} (d={tbl['n_d'].idxmax()})")
