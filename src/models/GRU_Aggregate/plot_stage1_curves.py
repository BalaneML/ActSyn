"""
plot_stage1_curves.py
=====================
Stage 1 の時刻別行動者率を、ATUS・社会生活基本調査・GRU・DDPM（Pre-trained）で 1 枚に重ねて描く。

    凡例                 中身                                                                     種
    ATUS                 ATUS 2024 平日の実データ                                                  —
    社会生活基本調査     教師 A*（st.load_stula_targets()["group_rates_tbl"]）                     —
    GRU                  H = 128・wd = 1、slot_bias の補正なし、g = 1.25。構造は --gru で選ぶ           42〜46
                           slot_bias     : slot_bias あり（gm.pool_path(seed, 1.25)）
                           no_slot_bias  : slot_bias なし（計画書 §12、gm.pool_path(seed, 1.25, use_slot_bias=False)）
    DDPM（Pre-trained）  clock_tf96_rate3：Transformer 型の時刻符号＋ε-MSE＋行動者率損失 λ = 3、    42〜44
                         g = 1.25（Stage 1 アブレーション段 3 のプール）

★28 群は 4 本とも ATUS の群シェア pi_atus（TUFINLWGT）でまとめる（compare.us_weighted_slot_rates）。
  Stage 1 の評価（米国加重）と同じで、曲線の差は群の中の違いだけになる。
★線は種平均、薄い帯は種の最小〜最大。生成プールは ckpt より古ければ止める（取り違えの防止）。
★読み込み・まとめ方・図の体裁は stage1_gru_compare と共有する（写し書きしない）。
★GRU の青と DDPM の橙は、色覚の検証（dataviz の validate_palette）の全検査を通る。
  ATUS（黒の太線）と社会生活基本調査（灰の破線）は、線の太さと線種でも区別する。

データフロー:

```mermaid
flowchart TD
    SET["compare.setup()<br/>real（ATUS 実）, pi_atus (28,)"] --> AT["compare.us_weighted_slot_rates(compare.agr.group_rates(*real), pi_atus)<br/>atus (12, 96)"]
    TGT["compare.st.load_stula_targets()['group_rates_tbl']<br/>a_star (28, 12, 96)"] --> SC["compare.us_weighted_slot_rates(a_star, pi_atus)<br/>stula (12, 96)"]
    GP["compare.load_pool_g(GRU_ARMS[args.gru], seed, GRU_GUIDANCE)<br/>gru_pools（種 42〜46）"] --> GC["seed_curves(gru_pools, pi_atus)<br/>gru (5, 12, 96)"]
    DP["ddpm_pool(seed)<br/>ddpm_pools（種 42〜44）"] --> DC["seed_curves(ddpm_pools, pi_atus)<br/>ddpm (3, 12, 96)"]
    SET --> GC
    SET --> DC
    AT --> FIG["plot_curves(atus, stula, gru, ddpm, out)"]
    SC --> FIG
    GC --> FIG
    DC --> FIG
    FIG --> PNG["OUT_PNG[args.gru]"]
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/plot_stage1_curves.py                     # slot_bias あり
    .venv/bin/python src/models/GRU_Aggregate/plot_stage1_curves.py --gru no_slot_bias  # slot_bias なし

    前提: GRU の補正なし・g = 1.25 のプール（種 42〜46）を作っておく
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42 --pool-only --guidance 1.25                  # 種 43〜46 も
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42 --pool-only --guidance 1.25 --no-slot-bias   # 同上
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/GRU_Aggregate -> repo root
FIG_DIR = Path(__file__).resolve().parent / "figures"
# GRU の構造（--gru）→ 出力 PNG
OUT_PNG: dict[str, Path] = {"slot_bias": FIG_DIR / "stage1_curves_atus_stula_gru_ddpm.png",
                            "no_slot_bias": FIG_DIR / "stage1_curves_atus_stula_gru_no_slot_bias_ddpm.png"}

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


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


# Stage 1 の読み込み・米国加重のまとめ方・図の体裁の出所
compare: Any = _load("gru_aggregate_stage1_gru_compare",
                     REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_gru_compare.py")

# GRU の構造（--gru）→ compare の arm 名。どちらも slot_bias の補正はしていない
GRU_ARMS: dict[str, str] = {"slot_bias": "gru", "no_slot_bias": "gru_no_slot_bias"}
GRU_GUIDANCE = 1.25                    # GRU の生成の CFG の強さ
GRU_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DDPM_LABEL = "clock_tf96_rate3@g1.25"  # stage1_clock_replicates のラベル（arm 名 @ CFG の強さ）
DDPM_SEEDS: tuple[int, ...] = (42, 43, 44)

# 凡例の文字（ユーザー指定）
LABEL_ATUS = "ATUS"
LABEL_STULA = "社会生活基本調査"
LABEL_GRU = "GRU"
LABEL_DDPM = "DDPM（Pre-trained）"
# 色と線。Stage 1 の他の図と同じ割り当て（ATUS は黒の太線、社会生活基本調査は灰の破線）
COLOR_ATUS = "#1a1a19"
COLOR_STULA = "#6f6e69"
COLOR_GRU: str = compare.ARM_COLOR["gru"]           # "#2a78d6"
COLOR_DDPM: str = compare.ARM_COLOR["ddpm_tf96"]    # "#eb6834"
BAND_ALPHA = 0.14                                   # 種の範囲の帯の濃さ（stage1_gru_compare.plot_curves と同じ）


def ddpm_pool(seed: int) -> IntArr:
    """DDPM（clock_tf96_rate3、g = 1.25）の群別プールを読む（ckpt より古ければ止める）

    Args:
        seed: 学習の種

    Returns:
        群別プール, dtype=int64, (28, M, 96)

    Raises:
        FileNotFoundError: プールが無いとき
    """
    path = compare.rep.pool_csv(DDPM_LABEL, seed)
    if not path.exists():
        raise FileNotFoundError(f"DDPM のプールが無い (seed={seed}): {path}")
    compare.rep.check_fresh(path, compare.rep.ckpt_path(DDPM_LABEL, seed))
    return np.asarray(compare.cur.load_sample_pool(path), dtype=np.int64)


def seed_curves(pools: list[IntArr], pi_atus: FloatArr) -> FloatArr:
    """種ごとの群別プールを、ATUS の群シェアでまとめた時刻別行動者率にする

    Args:
        pools: 種ごとの群別プール, 各 (28, M, 96)
        pi_atus: ATUS の群シェア, (28,)

    Returns:
        種ごとの時刻別行動者率, dtype=float64, (種の数, 12, 96)
    """
    return np.stack([compare.us_weighted_slot_rates(compare.cur.pool_to_slot_rates(p), pi_atus) for p in pools])


def plot_curves(atus: FloatArr, stula: FloatArr, gru: FloatArr, ddpm: FloatArr, out: Path) -> None:
    """12 活動の時刻別行動者率を 4×3 の小倍数に描く

    Note:
        1. 帯（種の最小〜最大）を先に描き、線を上に重ねる。凡例は線だけ（ATUS・社会生活基本調査・GRU・DDPM の順）
        2. 活動ごとに y 軸を独立させる（睡眠は 1.0 近く、ボランティアは 0.01 未満）

    Args:
        atus: ATUS 実, (12, 96)
        stula: 社会生活基本調査, (12, 96)
        gru: GRU の種ごとの曲線, (種の数, 12, 96)
        ddpm: DDPM の種ごとの曲線, (種の数, 12, 96)
        out: 出力 PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = compare.rd._figure_module()
    fm.setup_fonts()
    hours = compare.cur.slot_hours()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(compare.cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        for stack, color in ((gru, COLOR_GRU), (ddpm, COLOR_DDPM)):
            ax.fill_between(hours, stack[:, c].min(axis=0), stack[:, c].max(axis=0), color=color,
                            alpha=BAND_ALPHA, lw=0)
        ax.plot(hours, atus[c], color=COLOR_ATUS, lw=2.6, label=LABEL_ATUS, solid_capstyle="round")
        ax.plot(hours, stula[c], color=COLOR_STULA, lw=1.8, ls="--", label=LABEL_STULA)
        for stack, color, label in ((gru, COLOR_GRU, LABEL_GRU), (ddpm, COLOR_DDPM, LABEL_DDPM)):
            ax.plot(hours, stack[:, c].mean(axis=0), color=color, lw=1.6, label=label,
                    solid_capstyle="round", solid_joinstyle="round")
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{compare.cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)
    fig.suptitle("Stage 1 の時刻別行動者率", fontsize=15, y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[stage1_curves] 図: {out}")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="Stage 1 の時刻別行動者率（ATUS・社会生活基本調査・GRU・DDPM）")
    ap.add_argument("--gru", choices=list(GRU_ARMS), default="slot_bias",
                    help="GRU の構造。slot_bias = slot_bias あり（既定）、no_slot_bias = slot_bias なし（計画書 §12）")
    ap.add_argument("--out", type=Path, default=None, help="出力 PNG, default=OUT_PNG[--gru]")
    args = ap.parse_args()
    arm = GRU_ARMS[args.gru]
    out: Path = args.out or OUT_PNG[args.gru]

    real, pi_atus, _ = compare.setup()
    atus = compare.us_weighted_slot_rates(compare.agr.group_rates(*real), pi_atus)
    a_star = np.asarray(compare.st.load_stula_targets()["group_rates_tbl"], dtype=np.float64)
    stula = compare.us_weighted_slot_rates(a_star, pi_atus)
    gru_pools = [compare.load_pool_g(arm, s, GRU_GUIDANCE) for s in GRU_SEEDS]
    ddpm_pools = [ddpm_pool(s) for s in DDPM_SEEDS]
    for s, p in zip(GRU_SEEDS, gru_pools):
        print(f"[stage1_curves] {LABEL_GRU}（{args.gru}）種 {s}: {compare.pool_csv_g(arm, s, GRU_GUIDANCE).name}"
              f"（群あたり {p.shape[1]} 本）")
    for s, p in zip(DDPM_SEEDS, ddpm_pools):
        print(f"[stage1_curves] {LABEL_DDPM} 種 {s}: {compare.rep.pool_csv(DDPM_LABEL, s).name}（群あたり {p.shape[1]} 本）")
    plot_curves(atus, stula, seed_curves(gru_pools, pi_atus), seed_curves(ddpm_pools, pi_atus), out)


if __name__ == "__main__":
    main()
