"""
stage1_rate_curves.py
=====================
Stage 1 の生成プールの時刻別行動者率を、ATUS 実データと従来の損失（λ=0）に重ねて描く。

対象は行動者率の項 L_rate（model.py の --rate-lam / --rate-gamma）を入れた条件である。
条件（arm）ごとに PDF を 1 本書く。各 PDF の中身は、全体 1 ページと 28 群のページ。

★1 ページに載せる系列は 3 本（ATUS実データ・従来の損失・その条件）に限る。
  生成を 5 本重ねると、配色が色覚の全組検証を通らない。4 本目以降は分けて描く。
★全体ページは、全系列の 28 群を ATUS の調査ウェイト TUFINLWGT の群シェアでまとめる（米国の人口構成）。
  Stage 1 の目標は ATUS なので、生成も同じ構成でまとめ、差を「群の中の違い」だけにする。
  stage1_clock_replicates の key_*（日本の人口 π_d でまとめる）とは値が少し違う。
★生成プールは mtime を重みと照合してから読む（stage2_teacher_fit.load_pool_csv）。
  別の重みで作った古いプールを使った前例があるため。

データフロー:

```mermaid
flowchart LR
    CSV["生成 CSV（群あたり 256 本）<br/>ddpm_simple_pretrain_samples_clock{tag}.csv"] --> LP["tf.load_pool_csv(path, label, ckpt)<br/>mtime を重みと照合 → rates (28,12,96)"]
    ATUS["tf.load_atus_rates()<br/>atus_rates (28,12,96), atus_share (28,)"] --> OV
    LP --> OV["overall: np.einsum('d,dcs->cs', atus_share, rates)<br/>全系列を同じ群シェアでまとめる"]
    LP --> GR["群 d: rates[d] (12,96)"]
    ATUS --> GR
    OV --> FIG["render_page(title, curves)<br/>4x3 の小倍数"]
    GR --> FIG
    FIG --> PDF["outputs/figures/stage1_rate/{tag}.pdf<br/>overall_{tag}.png"]
```

使い方:
    .venv/bin/python src/models/DDPM_Aggregate_Simple/stage1_rate_curves.py
    .venv/bin/python src/models/DDPM_Aggregate_Simple/stage1_rate_curves.py --arms rate3
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
GEN_DIR = REPO_ROOT / "outputs" / "generated"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"
OUT_DIR = REPO_ROOT / "outputs" / "figures" / "stage1_rate"

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage2_teacher_fit.py と同じ規則）。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 読み込み（mtime 照合つき）・ATUS 実データ・群ラベル・フォントは stage2_teacher_fit と共有する
tf: Any = _load("rate_stage2_teacher_fit", HERE / "stage2_teacher_fit.py")
cv: Any = tf.cv

# 条件の接尾辞 → 凡例のラベル。接尾辞は model.py の保存先の規則（_clock の後ろ）と同じ
ARMS: dict[str, str] = {
    "rate1": "行動者率の項あり（λ=1）",
    "rate3": "行動者率の項あり（λ=3）",
    "rate10": "行動者率の項あり（λ=10）",
    "rate0.1g0.01": "行動者率の項あり（λ=0.1, γ=0.01）",
}
LABEL_ATUS = tf.LABEL_ATUS                  # "ATUS実データ"
LABEL_BASE = "従来の損失（λ=0）"

# 系列の役割 → (色, 線幅)。正解の ATUS は黒の太線、生成 2 本は検証済みの 2 色（青・橙）。
# stage2_teacher_fit の 教師A* / Pre-trained / Fine-tuned と同じ色の割り当て
# ★noclock（時刻符号なし）は stage1_compare_curves だけが使う 3 本目。青・橙・#199e70 の 3 色は
#   全組の色覚検証を通る（worst CVD ΔE 8.4）。Stage 1 の図では ATUS が黒なので #199e70 は空いている
# ★atus は atus_vs_stula_curves だけが使う。教師A* を target（黒）にしたとき、ATUS は
#   stage2_teacher_fit と同じ #199e70 で描く。noclock と同じ色だが、同じ図には載らない
ROLE_STYLE: dict[str, tuple[str, float]] = {
    "target": (tf.COLOR_TEACHER, 2.6),
    "base": (tf.COLOR_PRETRAINED, 1.5),
    "arm": (tf.COLOR_FINETUNED, 1.5),
    "noclock": ("#199e70", 1.5),
    "atus": (tf.COLOR_ATUS, 1.5),
    "stula": ("#6f6e69", 1.8),
}
# 役割 → 線種。載っていない役割は実線。
# ★stula（社会生活基本調査）は Stage 1 の図に載せる日本側の参照線。生成 3 本が色を使い切るので、
#   色ではなく灰色の破線で区別する（灰色は背景に対して 3:1 以上）
ROLE_LINESTYLE: dict[str, str] = {"stula": "--"}


def pool_paths(tag: str) -> tuple[Path, Path]:
    """条件の接尾辞から (生成 CSV, 重み) のパスを返す。tag="" は従来の損失（時刻符号あり・種 42）。

    Args:
        tag: ARMS のキー、または ""（従来の損失）

    Returns:
        (生成 CSV のパス, 重みのパス)。存在するかは確かめない
    """
    suffix = "_clock" + (f"_{tag}" if tag else "")
    return (GEN_DIR / f"ddpm_simple_pretrain_samples{suffix}.csv",
            CKPT_DIR / f"ddpm_simple_pretrain_common12_weekday{suffix}.pt")


def plot_page(ax_grid: Any, curves: list[tuple[str, str, FloatArr]]) -> None:
    """12 活動の時刻別行動者率を 4x3 の小倍数の各軸へ描く。

    Note:
        ★活動ごとに y 軸を独立させる。睡眠は 1.0 近くまで行き、ボランティアは
          0.01 未満なので、共通軸にすると小さい活動が潰れる（stage2_teacher_fit と同じ）。
        ★curves の順に描く。正解の ATUS を先に太線で描き、生成 2 本を上に細線で重ねる。

    Args:
        ax_grid: plt.subplots(4, 3) が返す軸の 2 次元配列
        curves: (役割, 凡例のラベル, 時刻別行動者率 (12, 96)) の並び。役割は ROLE_STYLE のキー
    """
    hours = cv.slot_hours()
    ticks = np.arange(4, 29, 4)
    for c, name in enumerate(cv.ACT_NAMES):
        ax = ax_grid[c // 3][c % 3]
        for role, label, cur in curves:
            color, lw = ROLE_STYLE[role]
            ax.plot(hours, cur[c], color=color, lw=lw, ls=ROLE_LINESTYLE.get(role, "-"),
                    label=label, solid_capstyle="round", solid_joinstyle="round")
        ax.set_title(f"{cv.ACT_JA[name]}（{tf.ACT_EN[name]}）", fontsize=11)
        ax.set_xlim(4.0, 28.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tf._hour_label(float(h)) for h in ticks])
        ax.set_ylim(bottom=0.0)
        ax.grid(color="#e4e3dd", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8.5)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)


def render_page(title: str, curves: list[tuple[str, str, FloatArr]], legend_ncol: int = 3) -> Any:
    """1 ページ（タイトル・凡例・4x3 の小倍数）を作って Figure を返す。

    Args:
        title: 図のタイトル
        curves: (役割, 凡例のラベル, 時刻別行動者率 (12, 96)) の並び
        legend_ncol: 凡例の列数, default=3

    Returns:
        matplotlib の Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    plot_page(axes, curves)
    fig.suptitle(title, fontsize=16, y=0.993)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=legend_ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.968), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="描く条件（既定は全部）")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    atus_rates, atus_share = tf.load_atus_rates()          # (28,12,96), (28,)
    base_csv, base_ckpt = pool_paths("")
    base = tf.load_pool_csv(base_csv, LABEL_BASE, base_ckpt)

    def overall(rates: FloatArr) -> FloatArr:
        """28 群を ATUS の群シェアでまとめる, (28,12,96) -> (12,96)"""
        return np.einsum("d,dcs->cs", atus_share, rates)

    tf._setup_fonts()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag in args.arms:
        csv_path, ckpt_path = pool_paths(tag)
        arm = tf.load_pool_csv(csv_path, ARMS[tag], ckpt_path)
        pages: list[tuple[str, list[tuple[str, str, FloatArr]]]] = [(
            "時刻別行動者率（全体：28群のATUS調査ウェイト加重平均）",
            [("target", LABEL_ATUS, overall(atus_rates)),
             ("base", LABEL_BASE, overall(base.rates)),
             ("arm", ARMS[tag], overall(arm.rates))])]
        for d in range(cv.st.D_GROUPS):
            pages.append((f"時刻別行動者率（{tf.group_label(d)}）",
                          [("target", LABEL_ATUS, atus_rates[d]),
                           ("base", LABEL_BASE, base.rates[d]),
                           ("arm", ARMS[tag], arm.rates[d])]))

        pdf_path = args.out_dir / f"{tag}.pdf"
        with PdfPages(pdf_path) as pdf:
            for i, (title, curves) in enumerate(pages):
                fig = render_page(title, curves)
                if i == 0:
                    fig.savefig(args.out_dir / f"overall_{tag}.png", dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"[stage1_rate_curves] {tag}: {pdf_path}（{len(pages)} ページ）")


if __name__ == "__main__":
    main()
