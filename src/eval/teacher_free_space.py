"""
teacher_free_space.py
=====================
★集計教師（時刻別行動者率）が縛れていない量を測る。

Stage2_design.md §9.3「前提：教師は日次の量を縛っていない」の数値を出す唯一の
出所である。このスクリプトを回さずに §9.3 の表の値を書き換えてはいけない。

何を測るか:
    Stage 2 の教師は 28群 × 12活動 × 96スロット の「各スロットの行動者率」であり、
    これは個票の同時分布ではなく **スロットごとの周辺** である。周辺を厳密に保った
    まま同時分布だけを壊す操作を作れば、「教師が縛れていない自由度」を直接測れる。

    操作 = 群内・列ごと独立シャッフル
        群 d に属する人の集合の中で、スロット s の列を人の間で独立に並べ替える。
        これを 96 スロットすべてに、群ごとに別々の置換で行う。
        → 群別の周辺（= 教師テンソル）は 1 セルも変わらない。
        → 個人の中でのスロット間の対応（= 同時分布）だけが消える。

    周辺が保存されていることは実行時に assert で検査する（許容 0）。

読み取り方:
    導出できる量   総平均時間 = 15分 × Σ_s rate(s,c)  … 周辺の線形写像なので不変
    導出できない量 日次行動者率 P(∃s: x_s = c)、行動者平均時間、派生時刻
                   … 同時分布に依存するので、シャッフルで動く。動いた幅が
                     「教師の外にある情報量」であり、非循環な評価軸（§9.3 軸 B）が
                     成立する根拠になる。

使い方:
    .venv/bin/python src/eval/teacher_free_space.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
import atus_group_rates as ag  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "processed" / "atus2024" / "atus2024_stula_common12_dataset.csv"

NUM_SLOTS = 96
SLOT_MIN = 15
SLOT_COLS = [f"s{i}" for i in range(NUM_SLOTS)]

# 群定義 d = g*(N_A*N_E) + a*N_E + e。DDPM_Aggregate_Simple/model.py と同一
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E

# 共通12分類 (crosswalk_atus_stula.Common) の並び
ACT_NAMES = ["SLEEP_PERSONAL", "MEALS", "WORK", "SCHOOL", "HOUSEWORK", "CAREGIVING",
             "SHOPPING", "TRAVEL", "LEISURE_SOCIAL", "SPORTS", "VOLUNTEER", "OTHER_X"]
NUM_ACT = len(ACT_NAMES)

# ATUS TUDIARYDAY は 1=日曜 .. 7=土曜。平日は 2..6
WEEKDAY_CODES = [2, 3, 4, 5, 6]

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]


def load_atus_weekday() -> tuple[IntArr, IntArr]:
    """ATUS 平日の (スケジュール (N,96), 群インデックス (N,)) を返す。"""
    df = pd.read_csv(DATA_PATH)
    df = cast(pd.DataFrame, df[df["day_of_week"].isin(WEEKDAY_CODES)])
    g = df["gender"].to_numpy().astype(np.int64)
    a = np.clip((df["age"].to_numpy() - 15) // 10, 0, N_A - 1).astype(np.int64)
    e = np.isin(df["telfs"].to_numpy(), [1, 2]).astype(np.int64)
    d = (g * (N_A * N_E) + a * N_E + e).astype(np.int64)
    sched = df[SLOT_COLS].to_numpy().astype(np.int64)
    return sched, d


def shuffle_within_group(sched: IntArr, groups: IntArr, seed: int = 0) -> IntArr:
    """群内で、96 スロットの列を人の間で独立に並べ替える。

    群別の周辺（各群・各スロットの行動別割合）は保存される。
    """
    rng = np.random.default_rng(seed)
    out = sched.copy()
    for d in range(D_GROUPS):
        idx = np.where(groups == d)[0]
        if len(idx) < 2:
            continue
        for s in range(NUM_SLOTS):
            out[idx, s] = sched[idx[rng.permutation(len(idx))], s]
    return out


def group_marginals(sched: IntArr, groups: IntArr) -> FloatArr:
    """群別の時刻別行動者率 (D, 12, 96)。これが教師テンソルと同じ量。

    ★定義は atus_group_rates.group_rates が唯一の出所。ここは非加重で呼ぶだけ。
      あちらの非加重経路は 0/1 の和を1回だけ割るので、人の並び順が変わっても
      値は厳密に一致する。run() の「周辺が保存されている」assert がそれに依存している。
    """
    return ag.group_rates(sched, groups)


def daily_stats(sched: IntArr) -> tuple[FloatArr, FloatArr]:
    """(日次行動者率 (12,), 行動者平均時間・分 (12,))。

    日次行動者率 = その日に一度でもその活動をした人の割合 = P(∃s: x_s = c)。
    ★周辺（時刻別行動者率）からは決まらない量である。
    """
    part = np.zeros(NUM_ACT, dtype=np.float64)
    doer_min = np.zeros(NUM_ACT, dtype=np.float64)
    for c in range(NUM_ACT):
        on = sched == c
        did = on.any(axis=1)
        part[c] = did.mean()
        n_doer = int(did.sum())
        doer_min[c] = on.sum() * SLOT_MIN / n_doer if n_doer > 0 else np.nan
    return part, doer_min


def mean_switches(sched: IntArr) -> float:
    """1 人あたりの活動切替回数の平均。これも周辺からは決まらない。"""
    return float((sched[:, 1:] != sched[:, :-1]).sum(axis=1).mean())


def run(seed: int = 0) -> pd.DataFrame:
    sched, groups = load_atus_weekday()
    shuffled = shuffle_within_group(sched, groups, seed=seed)

    # ★教師が保存されていることの検査。ここが 0 でなければ以降の数値に意味は無い。
    #   厳密な 0 を要求してよい: 平均の分子は 0.0/1.0 の float64 和なので、
    #   加算順が変わっても値は変わらない（2^53 までの整数は float64 で厳密）。
    gap = float(np.abs(group_marginals(sched, groups)
                       - group_marginals(shuffled, groups)).max())
    assert gap == 0.0, f"群別の周辺が保存されていない (最大差 {gap:.3e})"

    part_r, doer_r = daily_stats(sched)
    part_s, doer_s = daily_stats(shuffled)
    total_r = (sched[:, :, None] == np.arange(NUM_ACT)).mean(axis=(0, 1)) * NUM_SLOTS * SLOT_MIN
    total_s = (shuffled[:, :, None] == np.arange(NUM_ACT)).mean(axis=(0, 1)) * NUM_SLOTS * SLOT_MIN

    print(f"N={len(sched)}  群別の周辺 最大差 {gap:.3e}  (0 なら教師は完全に同一)")
    df = pd.DataFrame({
        "act": ACT_NAMES,
        "total_min_real": total_r,          # 周辺から決まる量（不変のはず）
        "total_min_shuf": total_s,
        "part_real": part_r,                # 周辺から決まらない量
        "part_shuf": part_s,
        "part_ratio": part_s / np.maximum(part_r, 1e-12),
        "doer_min_real": doer_r,
        "doer_min_shuf": doer_s,
    }).sort_values("part_real").reset_index(drop=True)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sw_r, sw_s = mean_switches(sched), mean_switches(shuffled)
    print(f"\n総平均時間の最大差 {np.abs(total_r - total_s).max():.3e} 分"
          f"  … 周辺の線形写像なので動かない")
    print(f"切替回数 平均: 実 {sw_r:.2f} → 群内シャッフル {sw_s:.2f}  ({sw_s / sw_r:.1f} 倍)")
    return df


if __name__ == "__main__":
    run()
