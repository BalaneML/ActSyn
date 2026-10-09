"""
grid_summary.py
===============
学習型の時刻符号で、層数 {1, 2} × 幅 {64, 128} × weight decay {0.01, 0.1, 1} の 12 構成（eval_curves.GRID_SIZES）を
LSTM と GRU のそれぞれで 1 つの表と 1 枚の図にまとめる（2026-10-06 ユーザー指示「組み合わせの結果を見たい」）。
eval_curves.py → eval_vs_gru.py の出力を読むだけで、新しく測るものはない。

作るもの:
    data/processed/aggregates/stage1_grid_summary.csv   セル × 構成の 1 行。主な指標の種平均と範囲・床の外の数・判定
    figures/stage1_grid_summary.png                      行 = セル、列 = 指標（val・12 活動の MAE・15 分で終わる割合）、
                                                         横軸 = weight decay、系列 = (層数, 幅)
      ★色は幅（H = 64 は紫、H = 128 は青。eval_curves.WIDTH_COLORS と同じ 2 色で validate_palette を通る）、
        線の種類と印は層数（1 層は実線と丸、2 層は破線と四角）。色だけに頼らない
      ★印は種平均、縦の線は種の最小〜最大

データフロー:

```mermaid
flowchart TD
    LONG["stage1_lstm_vs_gru_long.csv<br/>val・系列・群（arm × 種）"] --> ST["summary_table()"]
    SEED["stage1_lstm_curve_seed.csv<br/>種ごとの 12 活動の MAE"] --> ST
    ERR["stage1_lstm_curve_errors.csv<br/>種平均の曲線の MAE・床の外"] --> ST
    TRN["stage1_lstm_vs_gru_training.csv<br/>最良 epoch"] --> ST
    JDG["stage1_lstm_vs_gru_judge.csv<br/>判定 C1〜C3"] --> ST
    ST --> CSV["stage1_grid_summary.csv"]
    LONG --> FIG["plot_grid(per_seed)<br/>figures/stage1_grid_summary.png"]
    SEED --> FIG
```

使い方（eval_curves.py → eval_vs_gru.py の後に回す）:
    .venv/bin/python src/models/LSTM_Aggregate/grid_summary.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/LSTM_Aggregate -> repo root
FIG_DIR = Path(__file__).resolve().parent / "figures"
AGG_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（eval_curves.py と同じ規則）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# arm の定義（GRID_SIZES・grid_arm）と色の出所
ec: Any = _load("lstm_eval_curves", Path(__file__).resolve().parent / "eval_curves.py")

# 図の 3 つの指標: (列名, 縦軸の名前)。値は種ごと（per_seed_table の列）
PLOT_METRICS: tuple[tuple[str, str], ...] = (
    ("val_ce_unweighted", "val の交差エントロピー（重みなし）"),
    ("mae_pt_mean12", "12 活動の MAE の平均（pt、種ごと）"),
    ("single_slot_ratio", "15 分で終わるエピソードの割合"))
# 系列 = (層数, 幅)。色は幅、線の種類と印は層数
HIDDEN_COLORS: dict[int, str] = dict(zip(ec.GRID_HIDDENS, ec.WIDTH_COLORS, strict=True))
LAYER_STYLES: dict[int, tuple[str, str]] = {1: ("-", "o"), 2: ("--", "s")}
SERIES_OFFSETS: dict[tuple[int, int], float] = {(1, 64): -0.12, (1, 128): -0.04, (2, 64): 0.04, (2, 128): 0.12}


def per_seed_table() -> pd.DataFrame:
    """12 構成の種ごとの値（val・系列・群・予測の平らさ・12 活動の MAE・最良 epoch）

    Returns:
        列 cell / num_layers / hidden / weight_decay / arm / seed と、long・seed・training の指標
    """
    long = pd.read_csv(AGG_DIR / "stage1_lstm_vs_gru_long.csv")
    seed = pd.read_csv(AGG_DIR / "stage1_lstm_curve_seed.csv")
    training = pd.read_csv(AGG_DIR / "stage1_lstm_vs_gru_training.csv")
    mae = pd.DataFrame(seed.groupby(["arm", "seed"], as_index=False).agg(mae_pt_mean12=("mae_pt", "mean")))
    keys = pd.DataFrame([{"cell": cell, "num_layers": n, "hidden": h, "weight_decay": wd,
                          "arm": ec.grid_arm(cell, n, h, wd)}
                         for cell in ec.CELL_MODULES for n, h, wd in ec.GRID_SIZES])
    table = keys.merge(long, on="arm", how="left").merge(mae, on=["arm", "seed"], how="left")
    return table.merge(training.reindex(columns=["arm", "seed", "best_epoch"]), on=["arm", "seed"], how="left")


def summary_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """セル × 構成の 1 行の表

    Args:
        per_seed: per_seed_table の戻り値

    Returns:
        列 cell / num_layers / hidden / weight_decay / arm と、主な指標の種平均・最小・最大、
        curve_mae_mean12（種平均の曲線の 12 活動の MAE の平均）、n_above_{bias,mae,max}（上限側の床を超えた活動の数）、
        c1_pass / c2_pass / c3_pass
    """
    errors = pd.read_csv(AGG_DIR / "stage1_lstm_curve_errors.csv")
    judge = pd.read_csv(AGG_DIR / "stage1_lstm_vs_gru_judge.csv")
    metrics = ["val_ce_unweighted", "mae_pt_mean12", "single_slot_ratio", "switch_mean", "switch_emd",
               "separation_ratio", "val_entropy", "val_stay_gap", "best_epoch"]
    keys = ["cell", "num_layers", "hidden", "weight_decay", "arm"]
    stats = per_seed.groupby(keys, sort=False)[metrics].agg(["mean", "min", "max"])
    stats.columns = [f"{m}_{s}" for m, s in stats.columns]
    table = stats.reset_index()
    curve = pd.DataFrame(errors.groupby("arm", as_index=False).agg(curve_mae_mean12=("mae_pt", "mean")))
    above = pd.DataFrame(errors.assign(**{f"n_above_{q}": errors[f"{q}_position"].eq("above_high").astype(int)
                                          for q in ("bias", "mae", "max")})
                         .groupby("arm", as_index=False)[["n_above_bias", "n_above_mae", "n_above_max"]].sum())
    table = table.merge(curve, on="arm", how="left").merge(above, on="arm", how="left")
    return table.merge(judge.reindex(columns=["arm", "c1_pass", "c2_pass", "c3_pass"]), on="arm", how="left")


def plot_grid(per_seed: pd.DataFrame, atus_single_slot: float, out: Path) -> None:
    """行 = セル、列 = PLOT_METRICS、横軸 = weight decay、系列 = (層数, 幅) の図

    Args:
        per_seed: per_seed_table の戻り値
        atus_single_slot: ATUS 実の 15 分で終わるエピソードの割合（その列に横線で引く）
        out: 保存先
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = ec.cmp.rd._figure_module()
    fm.setup_fonts()
    cells = tuple(ec.CELL_MODULES)
    x_of = {wd: float(i) for i, wd in enumerate(ec.GRID_WEIGHT_DECAYS)}          # weight decay は等間隔に置く
    fig, axes = plt.subplots(len(cells), len(PLOT_METRICS), figsize=(5.0 * len(PLOT_METRICS), 3.9 * len(cells)),
                             squeeze=False)
    for r, cell in enumerate(cells):
        for c, (metric, ylabel) in enumerate(PLOT_METRICS):
            ax = axes[r][c]
            for n in ec.GRID_LAYERS:
                for h in ec.GRID_HIDDENS:
                    linestyle, marker = LAYER_STYLES[n]
                    color = HIDDEN_COLORS[h]
                    xs, means, lows, highs = [], [], [], []
                    for wd in ec.GRID_WEIGHT_DECAYS:
                        v = per_seed.loc[(per_seed["cell"] == cell) & (per_seed["num_layers"] == n)
                                         & (per_seed["hidden"] == h) & (per_seed["weight_decay"] == wd),
                                         metric].to_numpy(dtype=np.float64)
                        xs.append(x_of[wd] + SERIES_OFFSETS[(n, h)])
                        means.append(float(v.mean()))
                        lows.append(float(v.min()))
                        highs.append(float(v.max()))
                    ax.vlines(xs, lows, highs, color=color, lw=1.4, alpha=0.75)
                    ax.plot(xs, means, linestyle=linestyle, marker=marker, color=color, lw=1.6, ms=7,
                            markeredgecolor="white", markeredgewidth=0.8, label=f"{n} 層・H={h}")
            if metric == "single_slot_ratio":
                ax.axhline(atus_single_slot, color=fm.COLOR_ATUS, lw=0.9, ls=":", label="ATUS 実")
            ax.set_xticks(list(x_of.values()))
            ax.set_xticklabels([f"{wd:g}" for wd in ec.GRID_WEIGHT_DECAYS])
            ax.set_xlim(-0.4, len(ec.GRID_WEIGHT_DECAYS) - 0.6)
            ax.grid(color="#e4e3dd", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.set_title(f"{ec.CELL_LABELS[cell]}：{ylabel}", fontsize=11)
            if r == len(cells) - 1:
                ax.set_xlabel("weight decay", fontsize=10)
    fig.suptitle("層数・幅・weight decay の組み合わせ", fontsize=14, y=0.995)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[grid_summary] 図: {out}")


def main() -> None:
    """表と図を出力する"""
    per_seed = per_seed_table()
    missing = list(dict.fromkeys(str(arm) for arm, seed in zip(per_seed["arm"], per_seed["seed"]) if pd.isna(seed)))
    if missing:
        raise FileNotFoundError(f"eval_vs_gru.py の出力に無い arm がある（先に評価を回す）: {missing}")
    table = summary_table(per_seed)
    table.to_csv(AGG_DIR / "stage1_grid_summary.csv", index=False)
    show = ["cell", "num_layers", "hidden", "weight_decay", "val_ce_unweighted_mean", "mae_pt_mean12_mean",
            "curve_mae_mean12", "single_slot_ratio_mean", "switch_emd_max", "val_entropy_mean", "best_epoch_mean",
            "n_above_bias", "n_above_mae", "n_above_max", "c1_pass", "c2_pass", "c3_pass"]
    ec.cmp._show("12 構成の主な指標（種平均）", table[show].round(4))
    ref = pd.read_csv(AGG_DIR / "stage1_lstm_vs_gru_ref.csv")
    atus_single = float(ref.loc[ref["name"] == "atus_single_slot_ratio", "value"].iloc[0])
    plot_grid(per_seed, atus_single, FIG_DIR / "stage1_grid_summary.png")


if __name__ == "__main__":
    main()
