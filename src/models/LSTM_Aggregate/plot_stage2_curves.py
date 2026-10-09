"""
plot_stage2_curves.py
=====================
LSTM_Aggregate の Stage 2（stage2_agg.py）の時刻別行動者率を描く。GRU_Aggregate/plot_stage2_curves.py と同じ形の図

    stage2_lstm_curves{変種}.png          全国（28 群を日本人口で加重）の 12 活動
    stage2_lstm_curves_groups{変種}.png   GRU と同じ 4 群 × 4 活動（gsp.GROUP_EXAMPLES × gsp.GROUP_ACTS）。
                                          Fine-tuned は 28 群すべてを教師にした E1（この 4 群も教師に入っている）
    {変種} は stage2_agg.variant_tag（例: _lam0.01）

★率は stage2_agg.run が評価のプール（2000 本/群、種 12345）から保存した npz を読む。生成し直さない。
  npz が無いか、出所の ckpt より古ければ止める（生成プールの取り違えを防ぐ）

データフロー:

```mermaid
flowchart LR
    ZS["s2.rates_path(seed, 'zeroshot', '')<br/>Pre-trained (28, 12, 96)"] --> NAT["cur.weighted_slot_rates<br/>全国の曲線 (12, 96)"]
    FT["s2.rates_path(seed, 'all', variant)<br/>Fine-tuned (28, 12, 96)"] --> NAT
    TGT["tgt['group_rates_tbl']<br/>A* (28, 12, 96)"] --> NAT
    ATUS["cur.atus_real_curve(tgt)"] --> NAT
    NAT --> F1["plot_national"]
    ZS --> F2["plot_groups"]
    FT --> F2
    TGT --> F2
```

使い方:
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --lam 0.01
    .venv/bin/python src/models/LSTM_Aggregate/plot_stage2_curves.py --lam 0.01 --seeds 42 43 44 45 46
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

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

# ★LSTM の青緑（stage1_gru_compare.ARM_COLOR["lstm"]）。Stage 2 の前後は線の種類で分ける（Pre-trained = 破線）
COLOR_LSTM: str = cmp.ARM_COLOR["lstm"]
DASH = (0, (3, 2))


def load_rates(seed: int, run: str, variant: str) -> FloatArr:
    """stage2_agg.run が保存した群別の時刻別行動者率を読む, -> (28, 12, 96)

    Args:
        seed: 種
        run: "zeroshot"（Pre-trained）または "all"（E1）
        variant: stage2_agg.variant_tag の印。zeroshot では使わない

    Raises:
        FileNotFoundError: npz が無いとき
        RuntimeError: npz が出所の ckpt より古いとき（別のモデルの率を読むおそれ）
    """
    path = s2.rates_path(seed, run, variant)
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に stage2_agg.py を回す（種 {seed}、{run}）")
    src = (s2.stage1_ckpt(seed) if run == "zeroshot"
           else s2.ck.ckpt_path(s2.run_dir(seed, run, variant), s2.STEPS))
    if src.exists() and path.stat().st_mtime < src.stat().st_mtime:
        raise RuntimeError(f"{path.name} が {src.name} より古い。stage2_agg.py で採点し直す")
    return cur.load_rates_npz(path)


def national(seeds: tuple[int, ...], run: str, variant: str, tgt: dict) -> FloatArr:
    """種ごとの全国の曲線, -> (種の数, 12, 96)"""
    return np.asarray([cur.weighted_slot_rates(load_rates(s, run, variant), tgt) for s in seeds])


def plot_national(tgt: dict, seeds: tuple[int, ...], variant: str, label_ft: str, out: Path) -> None:
    """全国（日本人口加重）の 12 活動。種が 2 本以上なら Fine-tuned に最小〜最大の帯を付ける"""
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cur.slot_hours()
    teacher = cur.weighted_slot_rates(tgt["group_rates_tbl"], tgt)
    atus = cur.atus_real_curve(tgt)
    pre = national(seeds, "zeroshot", "", tgt).mean(axis=0)
    ft = national(seeds, "all", variant, tgt)
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.plot(hours, atus[c], color=gsp.COLOR_ATUS, lw=1.4, label=gsp.LABEL_ATUS)
        ax.plot(hours, teacher[c], color=gsp.COLOR_TEACHER, lw=2.6, label=gsp.LABEL_TEACHER, solid_capstyle="round")
        ax.plot(hours, pre[c], color=COLOR_LSTM, lw=1.4, ls=DASH, label="LSTM Pre-trained")
        if len(seeds) > 1:
            ax.fill_between(hours, ft[:, c].min(axis=0), ft[:, c].max(axis=0), color=COLOR_LSTM, alpha=0.18, lw=0)
        ax.plot(hours, ft[:, c].mean(axis=0), color=COLOR_LSTM, lw=1.8, label=label_ft)
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)
    fig.suptitle("日本全国の時刻別行動者率", fontsize=15, y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[stage2 curves] 図: {out}")


def plot_groups(tgt: dict, seed: int, variant: str, label_ft: str, out: Path) -> None:
    """gsp.GROUP_EXAMPLES × gsp.GROUP_ACTS。Fine-tuned は E1（28 群すべてを教師）の種 seed"""
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cur.slot_hours()
    a_star = tgt["group_rates_tbl"]
    atus = np.asarray(agr.group_rates(*cmp.setup()[0]), dtype=np.float64)
    pre = load_rates(seed, "zeroshot", "")
    ft = load_rates(seed, "all", variant)
    groups, acts = gsp.GROUP_EXAMPLES, gsp.GROUP_ACTS
    fig, axes = plt.subplots(len(groups), len(acts), figsize=(17, 3.3 * len(groups) + 1.0))
    for i, d in enumerate(groups):
        for j, act in enumerate(acts):
            c = cur.ACT_NAMES.index(act)
            ax = axes[i][j]
            ax.plot(hours, atus[d, c], color=gsp.COLOR_ATUS, lw=1.3, label=gsp.LABEL_ATUS)
            ax.plot(hours, a_star[d, c], color=gsp.COLOR_TEACHER, lw=2.4, label=gsp.LABEL_TEACHER)
            ax.plot(hours, pre[d, c], color=COLOR_LSTM, lw=1.4, ls=DASH, label="LSTM Pre-trained")
            ax.plot(hours, ft[d, c], color=COLOR_LSTM, lw=1.8, label=label_ft)
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


def main() -> None:
    """CLI"""
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="LSTM_Aggregate の Stage 2 の時刻別行動者率の図")
    ap.add_argument("--lam", type=float, default=s2.LAM)
    ap.add_argument("--tau", type=float, default=s2.TAU)
    ap.add_argument("--bptt", type=int, default=s2.BPTT)
    ap.add_argument("--steps", type=int, default=s2.STEPS)
    ap.add_argument("--lr-cond", type=float, default=s2.LR_COND)
    ap.add_argument("--lr-time", type=float, default=s2.LR_TIME)
    ap.add_argument("--lr-rest", type=float, default=s2.LR_REST)
    ap.add_argument("--seeds", type=int, nargs="+", default=[s2.LGO_SEED],
                    help="全国の図で平均する種。群ごとの図は最初の種")
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    variant = s2.variant_tag(args.lam, args.tau, args.bptt, args.steps, args.lr_cond, args.lr_time, args.lr_rest)
    seed_txt = f"種 {seeds[0]}" if len(seeds) == 1 else f"種 {len(seeds)} 本の平均、帯は最小〜最大"
    # 学習率と step 数は既定と違うときだけ凡例に書く（variant_tag と同じ規則）
    lrs = (args.lr_cond, args.lr_time, args.lr_rest)
    setting = f"λ = {args.lam:g}"
    if lrs != (s2.LR_COND, s2.LR_TIME, s2.LR_REST):
        setting += f"、学習率 {args.lr_cond:g}" if len(set(lrs)) == 1 else "、学習率 " + "/".join(f"{x:g}" for x in lrs)
    if args.steps != s2.STEPS:
        setting += f"、{args.steps} step"
    label_ft = f"LSTM Fine-tuned（{setting}、{seed_txt}）"
    tgt = s2.st.load_stula_targets()
    plot_national(tgt, seeds, variant, label_ft, FIG_DIR / f"stage2_lstm_curves{variant}.png")
    plot_groups(tgt, seeds[0], variant, f"LSTM Fine-tuned（{setting}、種 {seeds[0]}）",
                FIG_DIR / f"stage2_lstm_curves_groups{variant}.png")


if __name__ == "__main__":
    main()
