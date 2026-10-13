"""
plot_stage2_curves.py
=====================
LSTM_Aggregate の Stage 2（stage2_agg.py）の時刻別行動者率を描く。GRU_Aggregate/plot_stage2_curves.py と同じ形の図

    stage2_{セル}_curves{変種}.png          全国（28 群を日本人口で加重）の 12 活動
    stage2_{セル}_curves_groups{変種}.png   GRU と同じ 4 群 × 4 活動（gsp.GROUP_EXAMPLES × gsp.GROUP_ACTS）。
                                            Fine-tuned は 28 群すべてを教師にした E1（この 4 群も教師に入っている）
    stage2_cells_curves{変種}.png           --compare: LSTM と GRU の Fine-tuned を重ねた全国の 12 活動
    stage2_{セル}_learning_curves{変種}.png  --history: E1 の学習曲線（L_agg・L_reh・勾配の cos）
    stage2_cells_learning_curves{変種}.png   --history --compare: LSTM と GRU の学習曲線を重ねたもの
    stage2_{セル}_mae_curve{変種}_s2wd{値}.png --mae-curve: 評価のプールで測った A* との誤差の学習曲線を、
                                            Stage 2 の weight decay の値ごとに重ねたもの
    {セル} は lstm / gru_minimal、{変種} は stage2_agg.variant_tag（例: _lam0.01）

★率は stage2_agg.run が評価のプール（2000 本/群、種 12345）から保存した npz を読む。生成し直さない。
  npz が無いか、出所の ckpt より古ければ止める（生成プールの取り違えを防ぐ）

データフロー:

```mermaid
flowchart LR
    ZS["s2.rates_path(seed, 'zeroshot', '', cell)<br/>Pre-trained (28, 12, 96)"] --> NAT["cur.weighted_slot_rates<br/>全国の曲線 (12, 96)"]
    FT["s2.rates_path(seed, 'all', variant, cell)<br/>Fine-tuned (28, 12, 96)"] --> NAT
    TGT["tgt['group_rates_tbl']<br/>A* (28, 12, 96)"] --> NAT
    ATUS["cur.atus_real_curve(tgt)"] --> NAT
    NAT --> F1["plot_national"]
    ZS --> F2["plot_groups"]
    FT --> F2
    TGT --> F2
    HIST["load_history(seed, variant, steps, cell)<br/>run_dir(seed, 'all', variant, cell)/history.csv"] --> F3["plot_history"]
    PV["pretrained_val(seed, cell)<br/>Stage 1 の config['best_val']"] --> F3
    SC["load_step_curve(seed, variant, steps, cell)<br/>s2.step_curve_path(seed, 'all', variant, cell)"] --> F4["plot_mae_curve"]
```

使い方:
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --lam 0.01
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --lam 0.01 --seeds 42 43 44 45 46
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --cell gru_minimal --lam 0.01
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --compare --lam 0.01 --seeds 42 43 44 45 46
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --history --compare --lam 0.01 --seeds 42 43 44 45 46
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --mae-curve 0 0.01 --lam 0.01 --seeds 42 43 44 45 46
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent / "figures"

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名ファイルの取り違えを防ぐ）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s2: Any = _load("lstm_stage2_agg", Path(__file__).resolve().parent / "stage2_agg.py")
# 群・活動の選び方、教師と ATUS の色と名前は GRU の Stage 2 の図と同じ
gsp: Any = _load("gru_plot_stage2_curves", REPO_ROOT / "src" / "models" / "GRU_Aggregate" / "plot_stage2_curves.py")
cur: Any = s2.cur
cmp: Any = gsp.cmp
agr: Any = gsp.agr

# ★色はセルごとに固定する。LSTM の青緑は stage1_gru_compare.ARM_COLOR["lstm"]、GRU の赤は
#   eval_curves.COLORS["gru_minimal"]（Stage 1 の図と同じ）。Stage 2 の前後は線の種類で分ける（Pre-trained = 破線）
CELL_COLORS: dict[str, str] = {"lstm": cmp.ARM_COLOR["lstm"], "gru_minimal": "#e34948"}
CELL_LABELS: dict[str, str] = {"lstm": "LSTM", "gru_minimal": "GRU"}
DASH = (0, (3, 2))
HISTORY_WINDOW = 50   # 学習曲線の移動平均の幅（step）


def load_rates(seed: int, run: str, variant: str, cell: str = s2.CELL) -> FloatArr:
    """stage2_agg.run が保存した群別の時刻別行動者率を読む, -> (28, 12, 96)

    Args:
        seed: 種
        run: "zeroshot"（Pre-trained）または "all"（E1）
        variant: stage2_agg.variant_tag の印。zeroshot では使わない
        cell: 再帰の層 lstm / gru_minimal

    Raises:
        FileNotFoundError: npz が無いとき
        RuntimeError: npz が出所の ckpt より古いとき（別のモデルの率を読むおそれ）
    """
    path = s2.rates_path(seed, run, variant, cell)
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に stage2_agg.py を回す（{cell}、種 {seed}、{run}）")
    src = (s2.stage1_ckpt(seed, cell) if run == "zeroshot"
           else s2.ck.ckpt_path(s2.run_dir(seed, run, variant, cell), s2.STEPS))
    if src.exists() and path.stat().st_mtime < src.stat().st_mtime:
        raise RuntimeError(f"{path.name} が {src.name} より古い。stage2_agg.py で採点し直す")
    return cur.load_rates_npz(path)


def national(seeds: tuple[int, ...], run: str, variant: str, tgt: dict, cell: str = s2.CELL) -> FloatArr:
    """種ごとの全国の曲線, -> (種の数, 12, 96)"""
    return np.asarray([cur.weighted_slot_rates(load_rates(s, run, variant, cell), tgt) for s in seeds])


def plot_national(tgt: dict, seeds: tuple[int, ...], variant: str, label_ft: str, out: Path,
                  cell: str = s2.CELL, compare: bool = False) -> None:
    """全国（日本人口加重）の 12 活動。種が 2 本以上なら Fine-tuned に最小〜最大の帯を付ける

    Args:
        tgt: 教師
        seeds: 平均する種
        variant: stage2_agg.variant_tag の印
        label_ft: Fine-tuned の凡例。compare では「{セル} 」を頭に付けて使う
        out: 図の保存先
        cell: 描くセル。compare では使わない
        compare: True なら LSTM と GRU の Fine-tuned を重ねる（Pre-trained は描かない）
    """
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cur.slot_hours()
    teacher = cur.weighted_slot_rates(tgt["group_rates_tbl"], tgt)
    atus = cur.atus_real_curve(tgt)
    cells = tuple(s2.CELLS) if compare else (cell,)
    pre = {k: national(seeds, "zeroshot", "", tgt, k).mean(axis=0) for k in cells}
    ft = {k: national(seeds, "all", variant, tgt, k) for k in cells}
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.plot(hours, atus[c], color=gsp.COLOR_ATUS, lw=1.4, label=gsp.LABEL_ATUS)
        ax.plot(hours, teacher[c], color=gsp.COLOR_TEACHER, lw=2.6, label=gsp.LABEL_TEACHER, solid_capstyle="round")
        for k in cells:
            color = CELL_COLORS[k]
            if not compare:
                ax.plot(hours, pre[k][c], color=color, lw=1.4, ls=DASH, label=f"{CELL_LABELS[k]} Pre-trained")
            if len(seeds) > 1:
                ax.fill_between(hours, ft[k][:, c].min(axis=0), ft[k][:, c].max(axis=0), color=color, alpha=0.18,
                                lw=0)
            ax.plot(hours, ft[k][:, c].mean(axis=0), color=color, lw=1.8,
                    label=f"{CELL_LABELS[k]} {label_ft}" if compare else label_ft)
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)
    fig.suptitle("日本全国の時刻別行動者率", fontsize=15, y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # ★compare は系列が 4 本で 1 行に収まらないので 2 列に折り返す
    fig.legend(handles, labels, loc="upper center", ncol=2 if compare else len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if compare else 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[stage2 curves] 図: {out}")


def plot_groups(tgt: dict, seed: int, variant: str, label_ft: str, out: Path, cell: str = s2.CELL) -> None:
    """gsp.GROUP_EXAMPLES × gsp.GROUP_ACTS。Fine-tuned は E1（28 群すべてを教師）の種 seed"""
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cur.slot_hours()
    a_star = tgt["group_rates_tbl"]
    atus = np.asarray(agr.group_rates(*cmp.setup()[0]), dtype=np.float64)
    pre = load_rates(seed, "zeroshot", "", cell)
    ft = load_rates(seed, "all", variant, cell)
    color = CELL_COLORS[cell]
    groups, acts = gsp.GROUP_EXAMPLES, gsp.GROUP_ACTS
    fig, axes = plt.subplots(len(groups), len(acts), figsize=(17, 3.3 * len(groups) + 1.0))
    for i, d in enumerate(groups):
        for j, act in enumerate(acts):
            c = cur.ACT_NAMES.index(act)
            ax = axes[i][j]
            ax.plot(hours, atus[d, c], color=gsp.COLOR_ATUS, lw=1.3, label=gsp.LABEL_ATUS)
            ax.plot(hours, a_star[d, c], color=gsp.COLOR_TEACHER, lw=2.4, label=gsp.LABEL_TEACHER)
            ax.plot(hours, pre[d, c], color=color, lw=1.4, ls=DASH, label=f"{CELL_LABELS[cell]} Pre-trained")
            ax.plot(hours, ft[d, c], color=color, lw=1.8, label=label_ft)
            fm.style_axis(ax, 4.0, 28.0, 4.0)
            ax.set_ylim(bottom=0.0)
            if i == 0:
                ax.set_title(f"{cur.ACT_JA[act]}（{act}）", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{cmp.group_label(d)}\n行動者率", fontsize=10)
            if i == len(groups) - 1:
                ax.set_xlabel("時刻", fontsize=10)
    fig.suptitle("群ごとの日本の時刻別行動者率", fontsize=15, y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[stage2 curves] 図: {out}")


def load_history(seed: int, variant: str, steps: int, cell: str = s2.CELL) -> pd.DataFrame:
    """stage2_agg.train_stage2 が E1 で保存した 1 更新ごとの記録を読む

    Args:
        seed: 種
        variant: stage2_agg.variant_tag の印
        steps: 学習した step 数（記録の行数と照らす）
        cell: 再帰の層 lstm / gru_minimal

    Returns:
        history.csv（列 step・L_agg・L_reh・L_reh_val・tug_cos ほか）

    Raises:
        FileNotFoundError: history.csv が無いとき
        RuntimeError: 行数が steps と違うとき（途中で止まった実行を読むおそれ）
    """
    path = s2.run_dir(seed, "all", variant, cell) / "history.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に stage2_agg.py を回す（{cell}、種 {seed}）")
    history = pd.read_csv(path)
    if len(history) != steps:
        raise RuntimeError(f"{path} の行数 {len(history)} が step 数 {steps} と違う")
    return history


def pretrained_val(seed: int, cell: str = s2.CELL) -> float:
    """Stage 1 の学習記録にある ATUS の val の最小値（Stage 2 の L_reh_val と同じ関数で測った値）"""
    ckpt = torch.load(s2.stage1_ckpt(seed, cell), map_location="cpu", weights_only=False)
    return float(ckpt["config"]["best_val"])


def plot_history(seeds: tuple[int, ...], variant: str, steps: int, cells: tuple[str, ...], out: Path) -> None:
    """Stage 2 の学習曲線。L_agg・L_reh（train と val）・勾配の cos の 3 面

    Note:
        1. L_agg・L_reh・tug_cos は 1 更新ごとの値が揺れるので、HISTORY_WINDOW step の移動平均を描く
           （L_agg は移動平均の前の値も薄く描く）。L_reh_val は VAL_EVERY step ごとの値をそのまま描く
        2. ★L_agg は更新の前に、その step で引いた D_SUB 群だけで測った値。評価のプールの rate_mse_split とは別物
        3. 種が 2 本以上なら、種ごとの曲線の中央値を線、最小〜最大を帯で描く
        4. L_reh の破線は Pre-trained の val（種の中央値）

    Args:
        seeds: 描く種
        variant: stage2_agg.variant_tag の印
        steps: 学習した step 数
        cells: 描くセル
        out: 図の保存先
    """
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for cell in cells:
        color, name = CELL_COLORS[cell], CELL_LABELS[cell]
        hists = [load_history(s, variant, steps, cell) for s in seeds]
        step = hists[0]["step"].to_numpy()
        smooth = {k: np.asarray([h[k].rolling(HISTORY_WINDOW, min_periods=1).mean() for h in hists])
                  for k in ("L_agg", "L_reh", "tug_cos")}                                  # (種の数, steps)
        val_rows = hists[0]["L_reh_val"].notna().to_numpy()
        val = np.asarray([h["L_reh_val"].to_numpy()[val_rows] for h in hists])           # (種の数, steps / VAL_EVERY + 1)

        def band(ax: Any, x: npt.NDArray[Any], y: FloatArr, label: str, lw: float = 1.8, alpha: float = 1.0) -> None:
            if len(seeds) > 1:
                ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=color, alpha=0.18 * alpha, lw=0)
            ax.plot(x, np.median(y, axis=0), color=color, lw=lw, alpha=alpha, label=label)

        if len(seeds) == 1:
            axes[0].plot(step, hists[0]["L_agg"], color=color, lw=0.6, alpha=0.2)
        band(axes[0], step, smooth["L_agg"], name)
        band(axes[1], step, smooth["L_reh"], f"{name} train", lw=1.0, alpha=0.6)
        band(axes[1], step[val_rows], val, f"{name} val")
        axes[1].axhline(float(np.median([pretrained_val(s, cell) for s in seeds])), color=color, lw=1.0, ls=DASH,
                        label=f"{name} Pre-trained の val")
        band(axes[2], step, smooth["tug_cos"], name)
    axes[0].set_yscale("log")
    axes[2].axhline(0.0, color="#888888", lw=0.8)
    for ax, title in zip(axes, ("L_agg", "L_reh（ATUS の交差エントロピー）", "L_agg と L_reh の勾配の cos")):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("step", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Stage 2 の学習曲線", fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[stage2 curves] 図: {out}")


def load_step_curve(seed: int, variant: str, steps: int, cell: str = s2.CELL) -> pd.DataFrame:
    """stage2_agg.step_curve が書いた E1 の学習曲線を、step × 指標の横持ちにして読む

    Raises:
        FileNotFoundError: CSV が無いとき
        RuntimeError: CSV が最後の ckpt より古いとき（別の学習の曲線を読むおそれ）
    """
    path = s2.step_curve_path(seed, "all", variant, cell)
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に stage2_agg.py --step-curve を回す（{cell}、種 {seed}）")
    last = s2.ck.ckpt_path(s2.run_dir(seed, "all", variant, cell), steps)
    if last.exists() and path.stat().st_mtime < last.stat().st_mtime:
        raise RuntimeError(f"{path.name} が {last.name} より古い。--step-curve で測り直す")
    table = pd.read_csv(path)
    table = table[(table["mask"] == "12act") & table["eval_kind"].isin(["in-teacher", "all"])]
    return table.pivot_table(index="step", columns="metric", values="value")


def plot_mae_curve(seeds: tuple[int, ...], arms: list[tuple[str, str]], steps: int, out: Path,
                   cell: str = s2.CELL) -> None:
    """評価のプールで測った A* との誤差の学習曲線（E1）。28 群の rate_mae・rate_mse_split と全国の曲線の MSE

    Note:
        1. 種が 2 本以上なら、線が種の中央値、帯が最小〜最大
        2. 設定（arm）は線の種類で分ける。1 本目が実線、2 本目が破線

    Args:
        seeds: 描く種
        arms: (凡例の名前, variant_tag の印) の list
        steps: 学習した step 数
        out: 図の保存先
        cell: 再帰の層 lstm / gru_minimal
    """
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    color = CELL_COLORS[cell]
    panels = (("rate_mae", "A* との MAE（pt、28 群）", 100.0, False),
              ("rate_mse_split", "rate_mse_split（28 群）", 1.0, True),
              ("national_curve_mse", "全国の曲線の MSE", 1.0, True))
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.6))
    for i, (label, variant) in enumerate(arms):
        curves = [load_step_curve(s, variant, steps, cell) for s in seeds]
        step = curves[0].index.to_numpy()
        ls = "-" if i == 0 else DASH
        for ax, (metric, _, scale, _) in zip(axes, panels):
            y = np.asarray([c[metric].to_numpy() for c in curves]) * scale          # (種の数, step の数)
            if len(seeds) > 1:
                ax.fill_between(step, y.min(axis=0), y.max(axis=0), color=color, alpha=0.15, lw=0)
            ax.plot(step, np.median(y, axis=0), color=color, lw=1.8, ls=ls, label=label)
    for ax, (_, title, _, log) in zip(axes, panels):
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("step", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle("Stage 2 の A* との誤差の学習曲線", fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[stage2 curves] 図: {out}")


def main() -> None:
    """CLI"""
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="LSTM_Aggregate の Stage 2 の時刻別行動者率の図")
    ap.add_argument("--cell", choices=tuple(s2.CELLS), default=s2.CELL)
    ap.add_argument("--compare", action="store_true", help="LSTM と GRU の Fine-tuned を重ねた全国の図だけを描く")
    ap.add_argument("--lam", type=float, default=s2.LAM)
    ap.add_argument("--tau", type=float, default=s2.TAU)
    ap.add_argument("--bptt", type=int, default=s2.BPTT)
    ap.add_argument("--steps", type=int, default=s2.STEPS)
    ap.add_argument("--lr-cond", type=float, default=s2.LR_COND)
    ap.add_argument("--lr-time", type=float, default=s2.LR_TIME)
    ap.add_argument("--lr-rest", type=float, default=s2.LR_REST)
    ap.add_argument("--weight-decay", type=float, default=s2.WEIGHT_DECAY, help="Stage 2 の weight decay")
    ap.add_argument("--agg-loss", choices=s2.AGG_LOSSES, default=s2.AGG_LOSS, help="集計損失 mse / split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[s2.LGO_SEED],
                    help="全国の図で平均する種。群ごとの図は最初の種")
    ap.add_argument("--history", action="store_true",
                    help="学習曲線（E1 の history.csv）だけを描く。--compare と併せると LSTM と GRU を重ねる")
    ap.add_argument("--mae-curve", type=float, nargs="+", default=None, metavar="WD",
                    help="A* との誤差の学習曲線（--step-curve の CSV）を、Stage 2 の weight decay の値ごとに重ねて描く")
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    variant = s2.variant_tag(args.lam, args.tau, args.bptt, args.steps, args.lr_cond, args.lr_time, args.lr_rest,
                             args.weight_decay, args.agg_loss)
    if args.mae_curve is not None:
        def tag_of(wd: float) -> str:
            return s2.variant_tag(args.lam, args.tau, args.bptt, args.steps, args.lr_cond, args.lr_time,
                                  args.lr_rest, wd, args.agg_loss)
        arms = [(f"{CELL_LABELS[args.cell]} weight decay {wd:g}", tag_of(wd)) for wd in args.mae_curve]
        wd_txt = "-".join(f"{wd:g}" for wd in args.mae_curve)
        plot_mae_curve(seeds, arms, args.steps,
                       FIG_DIR / f"stage2_{args.cell}_mae_curve{tag_of(s2.WEIGHT_DECAY)}_s2wd{wd_txt}.png", args.cell)
        return
    if args.history:
        cells = tuple(s2.CELLS) if args.compare else (args.cell,)
        prefix = "cells" if args.compare else args.cell
        plot_history(seeds, variant, args.steps, cells, FIG_DIR / f"stage2_{prefix}_learning_curves{variant}.png")
        return
    seed_txt = f"種 {seeds[0]}" if len(seeds) == 1 else f"種 {len(seeds)} 本の平均、帯は最小〜最大"
    # 学習率と step 数は既定と違うときだけ凡例に書く（variant_tag と同じ規則）
    lrs = (args.lr_cond, args.lr_time, args.lr_rest)
    setting = f"λ = {args.lam:g}"
    if lrs != (s2.LR_COND, s2.LR_TIME, s2.LR_REST):
        setting += f"、学習率 {args.lr_cond:g}" if len(set(lrs)) == 1 else "、学習率 " + "/".join(f"{x:g}" for x in lrs)
    if args.steps != s2.STEPS:
        setting += f"、{args.steps} step"
    if args.weight_decay != s2.WEIGHT_DECAY:
        setting += f"、weight decay {args.weight_decay:g}"
    tgt = s2.st.load_stula_targets()
    if args.compare:
        # 2 つのセルは同じ設定なので、凡例には設定を書かない（設定は報告書の表に書く）
        plot_national(tgt, seeds, variant, f"Fine-tuned（{seed_txt}）",
                      FIG_DIR / f"stage2_cells_curves{variant}.png", compare=True)
        return
    name = CELL_LABELS[args.cell]
    plot_national(tgt, seeds, variant, f"{name} Fine-tuned（{setting}、{seed_txt}）",
                  FIG_DIR / f"stage2_{args.cell}_curves{variant}.png", cell=args.cell)
    plot_groups(tgt, seeds[0], variant, f"{name} Fine-tuned（{setting}、種 {seeds[0]}）",
                FIG_DIR / f"stage2_{args.cell}_curves_groups{variant}.png", cell=args.cell)


if __name__ == "__main__":
    main()
