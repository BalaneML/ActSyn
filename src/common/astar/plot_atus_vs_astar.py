"""
plot_atus_vs_astar.py
=====================
時刻別行動者率を図にする — ATUS（米国・個票）と教師 A*（日本・公表表）の重ね描き

同じ量を2つのドメインで測ったものを、同じ軸の上に置く:
    ATUS   atus_group_rates.group_rates()      個票を TUFINLWGT で加重
    A*     stage2_targets.load_stula_targets()  社会生活基本調査 第8-1表

★どちらも CSV から読み戻さず、その場で組み直す。書き出し済みの CSV
  (A_star_weekday.csv / A_atus_weekday_*.csv) は成果物であってキャッシュではないので、
  古い版を掴む事故を作らない。

★全国へ畳むときの重みは既定で「日本の群人口 π_d」で両者に共通に使う（--weight-scheme jp_pop）。
  重みを揃えないと、曲線の差に「群構成の日米差」が混ざる。
  ATUS 自身のウェイト構成で見たいときは --weight-scheme atus を渡す。

図は3種類:
    national   全国に畳んだ12活動の曲線（ATUS 実線 / A* 破線）
    groups     1活動を28群にばらした曲線。各パネルに n_d を出す
    heatmap    群 × 時刻 の差 (ATUS − A*) を活動ごとに12枚

使い方:
    .venv/bin/python3 src/common/astar/plot_atus_vs_astar.py
    .venv/bin/python3 src/common/astar/plot_atus_vs_astar.py --kind groups --act WORK
    .venv/bin/python3 src/common/astar/plot_atus_vs_astar.py --no-teacher   # ATUS 単独
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))                                # atus_group_rates
sys.path.insert(0, str(REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"))    # stage2_targets
sys.path.insert(0, str(Path(__file__).resolve().parent))                           # export_astar
import atus_group_rates as ag  # noqa: E402
import stage2_targets as st  # noqa: E402
from export_astar import ACT_JA, AGE7, EMP, SEX  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "imgs" / "atus_vs_astar"
ROLL = 16                       # 00:00 開始で表示するための np.roll 量（= 4時間 / 15分）
C_ATUS, C_STAR = "#1f77b4", "#d62728"

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> ModuleType:
    """同名ファイルの取り違えを避けるためファイル直ロードする（repo 共通の作法）。"""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


im = _load("plot_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")


# ============================================================
# 1. 材料をそろえる
# ============================================================
def load_pair(weight_scheme: str) -> tuple[FloatArr, FloatArr, FloatArr, npt.NDArray[np.int64]]:
    """(ATUS (28,12,96), A* (28,12,96), 全国へ畳む重み pi_d (28,), n_d (28,))。"""
    sched, groups, w = ag.load_atus_weekday()
    atus = ag.group_rates(sched, groups, w)
    star = np.asarray(st.load_stula_targets()["group_rates_tbl"], dtype=np.float64)

    n_d, wsum, _ = ag.group_counts(groups, w)
    if weight_scheme == "jp_pop":
        pop = np.asarray(st.load_stula_targets()["pop"], dtype=np.float64).reshape(-1)
        pi_d = pop / pop.sum()
    elif weight_scheme == "atus":
        pi_d = wsum / wsum.sum()
    else:
        raise ValueError(f"未知の weight_scheme: {weight_scheme}")
    return atus, star, pi_d, n_d


def _clock_axis(ax, ylabel: str = "行動者率", xlabel: str = "") -> None:
    """x 軸を 00:00-24:00 の時計にする（表示は np.roll 済みの配列が前提）。"""
    ax.set_xticks(np.arange(0, ag.NUM_SLOTS + 1, 16))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 4)])
    ax.set_xlim(0, ag.NUM_SLOTS - 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)


def _roll(curve: FloatArr) -> FloatArr:
    """04:00 起点の配列を 00:00 起点の表示順に直す。"""
    return np.roll(curve, ROLL)


# ============================================================
# 2. 全国に畳んだ12活動の曲線
# ============================================================
def plot_national(atus: FloatArr, star: FloatArr, pi_d: FloatArr,
                  path: Path | None = None, with_teacher: bool = True) -> None:
    """12活動 × (ATUS vs A*) の曲線。群は pi_d で畳む。

    パネル見出しの Σ|Δ| は、その活動の96スロットぶんの絶対差の合計（単位: 率 × スロット）。
    """
    na = np.einsum("d,dcs->cs", pi_d, atus)
    ns = np.einsum("d,dcs->cs", pi_d, star)

    fig, axes = plt.subplots(6, 2, figsize=(12, 15), squeeze=False)
    x = np.arange(ag.NUM_SLOTS)
    for c in ag.Common:
        ax = axes.ravel()[int(c)]
        ax.plot(x, _roll(na[int(c)]), lw=1.7, color=C_ATUS, label="ATUS（米国・個票）")
        if with_teacher:
            ax.plot(x, _roll(ns[int(c)]), lw=1.7, ls="--", color=C_STAR, label="A*（日本・公表表）")
            gap = float(np.abs(na[int(c)] - ns[int(c)]).sum())
            ax.set_title(f"{c.name}  {ACT_JA[c]}   Σ|Δ|={gap:.2f}", fontsize=10)
        else:
            ax.set_title(f"{c.name}  {ACT_JA[c]}", fontsize=10)
        _clock_axis(ax, xlabel="時刻" if int(c) >= ag.NUM_COMMON - 2 else "")
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle("時刻別行動者率（全国・平日）" + ("  ATUS vs A*" if with_teacher else "  ATUS"),
                 fontsize=13)
    fig.tight_layout()
    im._save(fig, path)
    plt.close(fig)


# ============================================================
# 3. 1活動を28群にばらす
# ============================================================
def plot_groups(atus: FloatArr, star: FloatArr, n_d: npt.NDArray[np.int64], act: str,
                path: Path | None = None, with_teacher: bool = True) -> None:
    """1活動について28群ぶんの曲線を並べる。

    ★各パネルの n は ATUS の標本数。n が小さい群では曲線が階段状になり、
      刻みは 1/n（最小の群で 1/17 = 0.059）。曲線の粗さは「ズレ」ではなく分解能である。
    """
    c = int(ag.Common[act])
    fig, axes = plt.subplots(7, 4, figsize=(16, 16), squeeze=False, sharey=True)
    x = np.arange(ag.NUM_SLOTS)
    for d in range(ag.D_GROUPS):
        g, rem = divmod(d, ag.N_A * ag.N_E)
        a, e = divmod(rem, ag.N_E)
        ax = axes.ravel()[d]
        ax.plot(x, _roll(atus[d, c]), lw=1.4, color=C_ATUS, label="ATUS")
        if with_teacher:
            ax.plot(x, _roll(star[d, c]), lw=1.4, ls="--", color=C_STAR, label="A*")
        ax.set_title(f"d={d}  {SEX[g]}{AGE7[a]}{EMP[e]}  n={int(n_d[d])}", fontsize=9)
        _clock_axis(ax, ylabel="行動者率" if d % 4 == 0 else "",
                    xlabel="時刻" if d >= ag.D_GROUPS - 4 else "")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"群別の時刻別行動者率 — {act}（{ACT_JA[ag.Common[act]]}）・平日", fontsize=13)
    fig.tight_layout()
    im._save(fig, path)
    plt.close(fig)


# ============================================================
# 4. 群 × 時刻 の差のヒートマップ
# ============================================================
def plot_heatmap(atus: FloatArr, star: FloatArr, path: Path | None = None) -> None:
    """活動ごとに (28群 × 96スロット) の差 ATUS − A* を1枚ずつ。

    赤 = ATUS のほうが高い、青 = A* のほうが高い。色の範囲は全活動で共通にする
    （活動ごとに正規化すると、小さい活動の微差が大きく見える）。
    """
    diff = atus - star                                   # (28, 12, 96)
    vmax = float(np.nanpercentile(np.abs(diff), 99))
    fig, axes = plt.subplots(3, 4, figsize=(18, 9), squeeze=False)
    for c in ag.Common:
        ax = axes.ravel()[int(c)]
        img = ax.imshow(np.roll(diff[:, int(c), :], ROLL, axis=1), aspect="auto",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{c.name}  {ACT_JA[c]}", fontsize=10)
        ax.set_xticks(np.arange(0, ag.NUM_SLOTS + 1, 16))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 4)])
        ax.set_yticks(np.arange(0, ag.D_GROUPS, 2))
        # 左端の列だけにラベルを出す。全パネルに出すと隣のパネルの目盛と重なる
        ax.set_ylabel("群 d" if int(c) % 4 == 0 else "")
        ax.set_xlabel("時刻" if int(c) >= ag.NUM_COMMON - 4 else "")
    fig.colorbar(img, ax=axes, shrink=0.6, label="行動者率の差 (ATUS − A*)")
    fig.suptitle(f"群 × 時刻 の差（平日）  色の範囲 ±{vmax:.3f} = |Δ| の99パーセンタイル",
                 fontsize=13)
    im._save(fig, path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="時刻別行動者率を ATUS と A* で重ねて図にする")
    ap.add_argument("--kind", choices=["national", "groups", "heatmap", "all"], default="all")
    ap.add_argument("--act", default="WORK", help="--kind groups で描く活動（共通12分類の名前）")
    ap.add_argument("--weight-scheme", choices=["jp_pop", "atus"], default="jp_pop",
                    help="全国へ畳む重み。jp_pop は日本の群人口を両者に共通に使う")
    ap.add_argument("--no-teacher", action="store_true", help="A* を重ねず ATUS だけ描く")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    im.setup_japanese_font()
    atus, star, pi_d, n_d = load_pair(args.weight_scheme)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with_teacher = not args.no_teacher
    suffix = "" if with_teacher else "_atus_only"
    print(f"重み: {args.weight_scheme}   教師の重ね描き: {with_teacher}")

    if args.kind in ("national", "all"):
        p = args.out_dir / f"national_curves{suffix}.png"
        plot_national(atus, star, pi_d, p, with_teacher)
        print(f"  -> {p}")
    if args.kind in ("groups", "all"):
        p = args.out_dir / f"groups_{args.act}{suffix}.png"
        plot_groups(atus, star, n_d, args.act, p, with_teacher)
        print(f"  -> {p}")
    if args.kind in ("heatmap", "all") and with_teacher:
        p = args.out_dir / "diff_heatmap.png"
        plot_heatmap(atus, star, p)
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
