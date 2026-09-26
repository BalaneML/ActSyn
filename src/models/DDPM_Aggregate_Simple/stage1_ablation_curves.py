"""
stage1_ablation_curves.py
=========================
Stage 1 アブレーションの arm ごとの時刻別行動者率（種の平均）を、ATUS 実データと重ねて描く。

    行        全日（04:00〜翌04:00）        昼（10:00〜14:00）
    食事      MEALS の人口加重曲線          同じ曲線の拡大
    仕事      WORK の人口加重曲線           同じ曲線の拡大
    仕事(有業) 有業の群だけの WORK          同じ曲線の拡大

★曲線は評価（src/eval/diagnostics/stage1_clock_replicates.py）と同じ定義で作る:
  群内は生成の本数で平均、群間は日本の人口 `tgt["pop"]` で重み付ける（stage2_curves.weighted_slot_rates）。
  有業の行は有業の群だけを人口加重する（stage1_clock_replicates.employment_slot_value と同じ）。
★arm の生成プールの名前は stage1_arms.arm_suffix から引く（唯一の出所）。種の平均を描く。
★配色は dataviz の既定カテゴリ順（青・橙・アクア・黄・マゼンタ）で arm の並び順に固定して割り当てる。
  validate_palette.js（light）で CVD 分離 9.1・通常視 19.6 を確認済み。背景とのコントラストが
  3:1 未満の色があるので、凡例を必ず付け、数値は報告書の表で渡す。ATUS 実データは黒の太線。
★タイトルは名前だけにする（指標の注記行は載せない）。

データフロー:

```mermaid
flowchart LR
    ARMS["stage1_arms.arm_suffix(arm, seed)"] --> CSV["生成プール CSV × 種"]
    CSV --> LP["stage2_curves.load_sample_pool → pool_to_slot_rates<br/>rates (28, 12, 96)"]
    LP --> AVG["種で平均した rates"]
    AVG --> POP["weighted_slot_rates → curve (12, 96)"]
    AVG --> EMP["有業の群だけ人口加重 → work_employed (96,)"]
    ATUS["atus_real_curve / group_rates"] --> FIG
    POP --> FIG["3 行 × 2 列（全日・昼の拡大）"]
    EMP --> FIG
    FIG --> PNG["docs/figures/stage1_ablation_{tag}.png"]
```

使い方:
    .venv/bin/python src/models/DDPM_Aggregate_Simple/stage1_ablation_curves.py \\
        --arms clock clock_h12 clock_h48 clock_tf96 --tag stage1 --title "段1 時刻符号の種類"
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
FIG_DIR = HERE / "docs" / "figures"

# dataviz の既定カテゴリ順（light）。arm の並び順に固定で割り当てる（順位で塗り直さない）
ARM_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
COLOR_ATUS = "#1a1a19"
ZOOM_HOURS = (10.0, 14.0)

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage2_select.py と同じ規則）。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rep: Any = _load("stage1_ablation_replicates",
                 REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_clock_replicates.py")
cur: Any = rep.cur
LABEL_ATUS = cur.ATUS_REAL_LABEL        # "ATUS実データ"


def hour_label(h: float) -> str:
    """04:00 起点の 0-36 時間軸の値 → 時計表示（28 → "4:00"）"""
    return f"{int(h) % 24}:00"


def setup_fonts() -> None:
    """日本語フォントを設定する。無ければ例外で止める（文字化けした図を残さない）"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    jp = [f for f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic",
                      "IPAexGothic", "Noto Sans CJK JP") if f in names]
    if not jp:
        raise RuntimeError("日本語フォントが見つからない。タイトルと活動名が文字化けするので止める")
    plt.rcParams["font.family"] = jp[0]
    plt.rcParams["axes.unicode_minus"] = False


def employed_curve(rates: FloatArr, pi_d: FloatArr, act: str) -> FloatArr:
    """有業の群だけを人口加重した 1 活動の曲線, -> (96,)"""
    c = cur.ACT_NAMES.index(act)
    sel_d = (np.arange(len(pi_d)) % rep.sm.N_E) == 1
    w = pi_d[sel_d] / pi_d[sel_d].sum()
    return np.asarray((w[:, None] * rates[sel_d, c, :]).sum(axis=0), dtype=np.float64)


def arm_rates(label: str, seeds: list[int]) -> FloatArr:
    """arm の群別行動者率を種で平均する, -> (28, 12, 96)"""
    stack = []
    for seed in seeds:
        path = rep.pool_csv(label, seed)
        if not path.exists():
            raise FileNotFoundError(f"生成プールが無い ({label}, seed={seed}): {path}")
        stack.append(cur.pool_to_slot_rates(cur.load_sample_pool(path)))
    return np.asarray(np.mean(stack, axis=0), dtype=np.float64)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, help="arm ラベル（最大 5 本）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--tag", required=True, help="出力 docs/figures/stage1_ablation_{tag}.png")
    ap.add_argument("--title", required=True)
    args = ap.parse_args()
    if len(args.arms) > len(ARM_COLORS):
        raise ValueError(f"arm は {len(ARM_COLORS)} 本まで（色を生成しない）: {len(args.arms)}")

    tgt = cur.st.load_stula_targets()
    pop = np.asarray(tgt["pop"], dtype=np.float64).reshape(-1)
    pi_d = pop / pop.sum()
    atus_curve = cur.atus_real_curve(tgt)
    sched_w, groups_w, w_w = rep.agr.load_atus_weekday()
    atus_rates = rep.agr.group_rates(sched_w, groups_w, w_w)

    # (行の名前, 実データの曲線, arm ごとの曲線を返す関数)
    rows: list[tuple[str, FloatArr, Any]] = [
        ("食事（MEALS）", atus_curve[cur.ACT_NAMES.index("MEALS")],
         lambda r: cur.weighted_slot_rates(r, tgt)[cur.ACT_NAMES.index("MEALS")]),
        ("仕事（WORK）", atus_curve[cur.ACT_NAMES.index("WORK")],
         lambda r: cur.weighted_slot_rates(r, tgt)[cur.ACT_NAMES.index("WORK")]),
        ("仕事・有業の群（WORK, employed）", employed_curve(atus_rates, pi_d, "WORK"),
         lambda r: employed_curve(r, pi_d, "WORK")),
    ]
    rates = {label: arm_rates(label, args.seeds) for label in args.arms}

    setup_fonts()
    hours = cur.slot_hours()
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 3.4 * len(rows) + 0.9))
    for i, (name, real, fn) in enumerate(rows):
        for j, (lo, hi) in enumerate(((4.0, 28.0), ZOOM_HOURS)):
            ax = axes[i][j]
            lines = [real] + [fn(rates[label]) for label in args.arms]
            ax.plot(hours, real, color=COLOR_ATUS, lw=2.6, label=LABEL_ATUS,
                    solid_capstyle="round")
            for k, label in enumerate(args.arms):
                ax.plot(hours, lines[k + 1], color=ARM_COLORS[k], lw=1.6, label=label,
                        solid_capstyle="round", solid_joinstyle="round")
            ax.set_xlim(lo, hi)
            ticks = np.arange(lo, hi + 0.01, 4.0 if j == 0 else 1.0)
            ax.set_xticks(ticks)
            ax.set_xticklabels([hour_label(float(h)) for h in ticks])
            if j == 0:                 # 全日は 0 から。昼の拡大は差を読むため表示範囲の値に合わせる
                ax.set_ylim(bottom=0.0)
            else:
                shown = np.concatenate([ln[(hours >= lo) & (hours <= hi)] for ln in lines])
                pad = 0.08 * float(shown.max() - shown.min())
                ax.set_ylim(float(shown.min()) - pad, float(shown.max()) + pad)
            ax.grid(color="#e4e3dd", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.set_title(name + ("" if j == 0 else "・昼の拡大"), fontsize=11)
            if j == 0:
                ax.set_ylabel("行動者率", fontsize=10)
            if i == len(rows) - 1:
                ax.set_xlabel("時刻", fontsize=10)
    fig.suptitle(args.title, fontsize=15, y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.968), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"stage1_ablation_{args.tag}.png"
    fig.savefig(out, dpi=150)
    print(f"[figure] {out}")


if __name__ == "__main__":
    main()
