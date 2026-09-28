"""
plot_stage2_curves.py
=====================
GRU_Aggregate の Stage 2 の時刻別行動者率を描く（計画書 docs/stage2_plan.md）。

    stage2_gru_curves.png         全国（28 群を日本人口で加重）の 12 活動
    stage2_gru_curves_groups.png  GROUP_EXAMPLES の 4 群 × GROUP_ACTS の 4 活動。GRU Stage 2 は
                                  その群を教師から外した fold（E2）の δ で生成した曲線（教師に使っていない群）

★評価（stage2.evaluate）は生成した率を保存しないので、保存した δ（stage2.shift_path）から、
  評価と同じ乱数（stage2.EVAL_SEED）・同じ本数（stage2.EVAL_N）で生成し直す。
  生成し直した率は RATES_DIR にキャッシュし、δ の npz より古ければ作り直す。

データフロー:

```mermaid
flowchart LR
    SH["s2.shift_path(seed, run)<br/>δ の npz"] --> LD["load_shift"]
    LD --> RT["stage2_rates(seed, run)<br/>gm.group_pool(..., group_bias) → (28, 12, 96)"]
    CK["gm.ckpt_path(seed, calib_guidance=1.25)"] --> RT
    RT --> NAT["cur.weighted_slot_rates<br/>全国の曲線 (12, 96)"]
    TGT["tgt['group_rates_tbl']<br/>A*"] --> NAT
    DD["DDPM の step 200 の率<br/>s2.DDPM_STEP200_RATES"] --> NAT
    NAT --> F1["plot_national"]
    RT --> F2["plot_groups"]
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/plot_stage2_curves.py
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent / "figures"
RATES_DIR = REPO_ROOT / "outputs" / "generated"

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


s2: Any = _load("gru_stage2", Path(__file__).resolve().parent / "stage2.py")
gm: Any = s2.gm
sm: Any = s2.sm
cur: Any = s2.cur
cmp: Any = _load("gru_s2plot_compare", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_gru_compare.py")
agr: Any = cmp.agr

# 色（stage1_gru_compare.ARM_COLOR と同じ対応。GRU 補正後は赤紫、DDPM Transformer 型は橙）。
# ★Stage 2 の前後は線の種類で分ける（Pre-trained = 破線、Fine-tuned = 実線）
COLOR_GRU = cmp.ARM_COLOR["gru_calg125"]
COLOR_DDPM = cmp.ARM_COLOR["ddpm_tf96"]
COLOR_TEACHER = "#1a1a19"
COLOR_ATUS = "#9a9a94"
LABEL_TEACHER = "日本の教師（社会生活基本調査）"
LABEL_ATUS = "ATUS 実データ（米国）"
# 群ごとの図の群と活動（日米差の大きい活動）
# 群は plot_schedules.EXAMPLE_GROUPS と同じ 4 群（男 35-44 有業・女 35-44 無業・女 15-24 無業・男 75+ 無業）
GROUP_EXAMPLES: tuple[int, ...] = (sm.d_index(0, 2, 1), sm.d_index(1, 2, 0), sm.d_index(1, 0, 0), sm.d_index(0, 6, 0))
GROUP_ACTS: tuple[str, ...] = ("SLEEP_PERSONAL", "MEALS", "WORK", "HOUSEWORK")


def load_shift(seed: int, run: str) -> Any:
    """stage2.JapanShift.save で保存した δ を読む"""
    with np.load(s2.shift_path(seed, run)) as z:
        return s2.JapanShift(base=z["base"], sex=z["sex"], age=z["age"], emp=z["emp"])


def stage2_rates(seed: int, run: str) -> FloatArr:
    """評価と同じ乱数・本数で生成し直した群別の時刻別行動者率, -> (28, 12, 96)

    Args:
        seed: 学習の種
        run: "zeroshot"（δ = 0）/ "all" / "fold{K}"
    """
    cache = RATES_DIR / f"gru_stage2{gm.run_suffix(seed)}_{run}_rates.npz"
    src = gm.ckpt_path(seed, calib_guidance=s2.CALIB_G) if run == "zeroshot" else s2.shift_path(seed, run)
    if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
        with np.load(cache) as z:
            return np.asarray(z["rates"], dtype=np.float64)
    model = gm.load_model(gm.ckpt_path(seed, calib_guidance=s2.CALIB_G))
    bias = None if run == "zeroshot" else load_shift(seed, run).to_group_bias()
    pool = gm.group_pool(model, s2.EVAL_N, s2.GUIDANCE, seed=s2.EVAL_SEED, group_bias=bias)
    rates = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
    np.savez_compressed(cache, rates=rates)
    return rates


def plot_national(tgt: dict, out: Path) -> None:
    """全国（日本人口加重）の 12 活動。GRU は種 5 本の平均の線と最小〜最大の帯"""
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cur.slot_hours()
    teacher = cur.weighted_slot_rates(tgt["group_rates_tbl"], tgt)
    atus = cur.atus_real_curve(tgt)
    gru_zs = np.mean([cur.weighted_slot_rates(stage2_rates(s, "zeroshot"), tgt) for s in s2.SEEDS], axis=0)
    gru_s2 = np.asarray([cur.weighted_slot_rates(stage2_rates(s, "all"), tgt) for s in s2.SEEDS])
    ddpm_s2 = cur.weighted_slot_rates(cur.load_rates_npz(s2.DDPM_STEP200_RATES), tgt)
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.plot(hours, atus[c], color=COLOR_ATUS, lw=1.4, label=LABEL_ATUS)
        ax.plot(hours, teacher[c], color=COLOR_TEACHER, lw=2.6, label=LABEL_TEACHER, solid_capstyle="round")
        ax.plot(hours, ddpm_s2[c], color=COLOR_DDPM, lw=1.5, label="DDPM Fine-tuned (λ=0.003, step200)")
        ax.plot(hours, gru_zs[c], color=COLOR_GRU, lw=1.4, ls=(0, (3, 2)), label="GRU Pre-trained")
        ax.fill_between(hours, gru_s2[:, c].min(axis=0), gru_s2[:, c].max(axis=0), color=COLOR_GRU, alpha=0.18, lw=0)
        ax.plot(hours, gru_s2[:, c].mean(axis=0), color=COLOR_GRU, lw=1.8,
                label="GRU Fine-tuned（種 5 本の平均、帯は最小〜最大）")
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


def fold_of(tgt: dict) -> dict[int, int]:
    """群 → その群を教師から外した fold"""
    return {d: k for k, held in enumerate(s2.lgo.stratified_folds(tgt["pop"])) for d in held}


def plot_groups(tgt: dict, out: Path) -> None:
    """GROUP_EXAMPLES × GROUP_ACTS。GRU Stage 2 はその群を教師から外した fold の δ（種 42）"""
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cur.slot_hours()
    a_star = tgt["group_rates_tbl"]
    real = cmp.setup()[0]
    atus = np.asarray(agr.group_rates(*real), dtype=np.float64)
    zs = stage2_rates(s2.LGO_SEED, "zeroshot")
    folds = fold_of(tgt)
    fig, axes = plt.subplots(len(GROUP_EXAMPLES), len(GROUP_ACTS), figsize=(17, 3.3 * len(GROUP_EXAMPLES) + 1.0))
    for i, d in enumerate(GROUP_EXAMPLES):
        held = stage2_rates(s2.LGO_SEED, f"fold{folds[d]}")
        for j, act in enumerate(GROUP_ACTS):
            c = cur.ACT_NAMES.index(act)
            ax = axes[i][j]
            ax.plot(hours, atus[d, c], color=COLOR_ATUS, lw=1.3, label=LABEL_ATUS)
            ax.plot(hours, a_star[d, c], color=COLOR_TEACHER, lw=2.4, label=LABEL_TEACHER)
            ax.plot(hours, zs[d, c], color=COLOR_GRU, lw=1.4, ls=(0, (3, 2)), label="GRU Pre-trained")
            ax.plot(hours, held[d, c], color=COLOR_GRU, lw=1.8, label="GRU Fine-tuned（この群を教師から外した fold）")
            fm.style_axis(ax, 4.0, 28.0, 4.0)
            ax.set_ylim(bottom=0.0)
            if i == 0:
                ax.set_title(f"{cur.ACT_JA[act]}（{act}）", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{cmp.group_label(d)}\n行動者率", fontsize=10)
            if i == len(GROUP_EXAMPLES) - 1:
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
    """2 枚の図を描く"""
    import matplotlib
    matplotlib.use("Agg")
    tgt = s2.st.load_stula_targets()
    plot_national(tgt, FIG_DIR / "stage2_gru_curves.png")
    plot_groups(tgt, FIG_DIR / "stage2_gru_curves_groups.png")


if __name__ == "__main__":
    main()
