"""
plot_trajectory.py
==================
DDPM_Minimal の Stage 1 の途中の ckpt ごとの指標（trajectory_pools.py の出力）を 1 枚の図にする
（Stage2_Common/docs/Stage1_ddpm_minimal_traj_results.md §3）

描くもの（横軸はどれも epoch）:
    1. val の ε の二乗誤差（val 最良の ckpt の config に残る history。1 epoch ごと）
    2. 12 活動の割合の総変動距離（ATUS 平日の実データから）。ほかの 3 モデルの Pre-trained の範囲を帯で重ねる
    3. 睡眠・身の回りの用事の割合。ATUS 平日の実データの値を横線で重ねる
途中の ckpt は生成の種 POOL_SEED の 1 点、val 最良の ckpt は生成の種 2 つの 2 点（白抜き）で描く

```mermaid
flowchart LR
    CSV["stage1_traj_{stem}.csv<br/>（trajectory_pools.py）"] --> LR["load_run(stem)"]
    CK["{stem}.pt の config['history']"] --> VH["val_history(stem)"]
    NPZ["stage2_common_{e0_stem}_rates.npz<br/>rates (28, 12, 96)"] --> PT["pretrained_tv(real)"]
    REAL["tp.load_real()<br/>real.shares (12,)"] --> PT
    LR --> PL["plot(out_path)"]
    VH --> PL
    PT --> PL
    REAL --> PL
    PL --> PNG["Stage2_Common/figures/stage1_traj_ddpm_minimal.png"]
```

使い方:
    .venv/bin/python src/models/DDPM_Minimal/plot_trajectory.py

出力:
    src/models/Stage2_Common/figures/stage1_traj_ddpm_minimal.png
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同じファイルが既に載っていればそれを返す）"""
    cached = sys.modules.get(name)
    cached_file = getattr(cached, "__file__", None)
    if cached is not None and cached_file is not None and Path(cached_file).resolve() == path.resolve():
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp: Any = _load("ddpm_minimal_trajectory_pools", HERE / "trajectory_pools.py")
sm: Any = tp.sm

FIG_PATH = REPO_ROOT / "src" / "models" / "Stage2_Common" / "figures" / "stage1_traj_ddpm_minimal.png"
GEN_DIR = REPO_ROOT / "outputs" / "generated"
SLEEP = "SLEEP_PERSONAL"

# 色は dataviz のスキルの既定の配色の 1〜3 番（3 色の全組で色覚の差の検査に合格。3 番は地との対比が 3:1 未満なので
# 凡例と報告書の表で補う）。文字は系列の色を使わない
TEXT = "#0b0b0b"
TEXT_SUB = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
REF_FILL = "#d9d8d3"


class Run(NamedTuple):
    """図に描く 1 本の Stage 1"""
    stem: str      # ckpt の stem（trajectory_pools.py の --stem と同じ）
    label: str     # 凡例の名前
    color: str     # 線と点の色


RUNS = (
    Run("ddpm_minimal_ch16_wd0.01_traj", "学習率 1e-3（回し直し、途中の ckpt つき）", "#2a78d6"),
    Run("ddpm_minimal_ch16_wd0.01_lr0.0002", "学習率 2e-4（途中の ckpt つき）", "#eb6834"),
)
ORIGINAL = Run("ddpm_minimal_ch16_wd0.01", "学習率 1e-3（本番。E0・E1 の起点）", "#1baf7a")
# ほかの 3 モデルの Pre-trained（共通 Stage 2 の E0、種 42）。GAN は起点 final（結果の報告書 §6 と同じ）
PRETRAINED_E0 = {
    "LSTM": "lstm_time_learned_h64_wd0.01_seed42_zeroshot",
    "VAE": "vae_minimal_z16_h222_wd0.01_seed42_zeroshot",
    "GAN": "gan_minimal_final_z16_h222_seed42_zeroshot",
}

FloatArr = npt.NDArray[np.float64]


def setup_fonts() -> None:
    """日本語フォントを設定する。無ければ例外で止める（文字化けした図を残さない）"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    jp = [f for f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic",
                      "IPAexGothic", "Noto Sans CJK JP") if f in names]
    if not jp:
        raise RuntimeError("日本語フォントが見つからない。軸と凡例が文字化けするので止める")
    plt.rcParams["font.family"] = jp[0]
    plt.rcParams["axes.unicode_minus"] = False


def load_run(stem: str) -> pd.DataFrame:
    """trajectory_pools.py の表を読む

    Raises:
        FileNotFoundError: 表が無いとき
    """
    path = tp.OUT_DIR / f"stage1_traj_{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い（trajectory_pools.py --stem {stem} で作る）")
    return pd.read_csv(path)


def val_history(stem: str) -> tuple[FloatArr, FloatArr]:
    """val 最良の ckpt の config に残る学習曲線

    Returns:
        (epoch, dtype=float64, (E,), val の ε の二乗誤差, dtype=float64, (E,))
    """
    config = torch.load(tp.CKPT_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)["config"]
    hist = config["history"]
    return np.asarray(hist["epoch"], dtype=np.float64), np.asarray(hist["val"], dtype=np.float64)


def pretrained_tv(real: Any) -> dict[str, float]:
    """ほかのモデルの Pre-trained の活動の割合の、ATUS 平日の実データからの総変動距離

    Note:
        E0 の rates (28, 12, 96) を時刻で平均して群ごとの割合にし、ATUS の群構成（調査ウェイトの和の比）で
        重み付けする。trajectory_pools.pool_metrics（群ごとに im.group_reweight で重み付け）と同じ値になる

    Args:
        real: tp.RealData（ATUS 平日の実データ）

    Returns:
        {モデル名: 総変動距離}

    Raises:
        FileNotFoundError: E0 の rates が無いとき
    """
    wn = real.w / real.w.sum()
    group_share = np.array([wn[real.d == g].sum() for g in range(sm.D_GROUPS)])     # (28,)
    out: dict[str, float] = {}
    for name, stem in PRETRAINED_E0.items():
        path = GEN_DIR / f"stage2_common_{stem}_rates.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} が無い（共通 Stage 2 の E0 の出力）")
        rates = np.load(path)["rates"]                                             # (28, 12, 96)
        shares = (group_share[:, None] * rates.mean(axis=2)).sum(axis=0)           # (12,)
        out[name] = float(0.5 * np.abs(shares - real.shares).sum())
    return out


def plot(out_path: Path = FIG_PATH) -> Path:
    """3 段の図を描いて保存する

    Args:
        out_path: 保存先, default=FIG_PATH

    Returns:
        保存したパス
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    setup_fonts()
    real = tp.load_real()
    sleep_real = float(real.shares[list(sm.ACT_NAMES).index(SLEEP)])
    others = pretrained_tv(real)
    tv_lo, tv_hi = min(others.values()), max(others.values())

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.6), sharex=True, facecolor=SURFACE)
    ax_val, ax_tv, ax_sleep = axes
    titles = ("val の ε の二乗誤差", "活動の割合の総変動距離（ATUS 平日の実データから）", "睡眠・身の回りの用事の割合")
    for ax, title in zip(axes, titles):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=11, color=TEXT, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SUB, labelsize=9)

    for run in RUNS:
        epoch, val = val_history(run.stem)
        ax_val.plot(epoch, val, color=run.color, linewidth=1.6)
    ax_val.set_yscale("log")
    val_ticks = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    ax_val.set_yticks(val_ticks, [f"{v:g}" for v in val_ticks])
    ax_val.minorticks_off()

    ax_tv.axhspan(tv_lo, tv_hi, color=REF_FILL, linewidth=0)
    ax_tv.annotate(f"ほかの 3 モデルの Pre-trained（{tv_lo:.3f}〜{tv_hi:.3f}）", xy=(1000, tv_hi), xytext=(0, 4),
                   textcoords="offset points", ha="right", va="bottom", fontsize=9, color=TEXT_SUB)
    ax_sleep.axhline(sleep_real, color=TEXT_SUB, linewidth=1.0)
    ax_sleep.annotate(f"ATUS 平日の実データ（{sleep_real:.3f}）", xy=(1000, sleep_real), xytext=(0, -4),
                      textcoords="offset points", ha="right", va="top", fontsize=9, color=TEXT_SUB)

    for run in (*RUNS, ORIGINAL):
        table = load_run(run.stem)
        is_best = table["is_best"].astype(bool).to_numpy()
        inter = table.loc[~is_best].sort_values(by="epoch")
        best = table.loc[is_best]
        for ax, col in ((ax_tv, "tv_atus"), (ax_sleep, f"share_{SLEEP}")):
            if len(inter):
                ax.plot(inter["epoch"], inter[col], color=run.color, linewidth=1.6, marker="o", markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0)
            ax.scatter(best["epoch"], best[col], s=46, facecolors=SURFACE, edgecolors=run.color, linewidths=1.6,
                       marker="D" if run is ORIGINAL else "o", zorder=3)
    ax_tv.set_ylim(0, 0.5)
    ax_sleep.set_ylim(0, 0.45)
    ax_sleep.set_xlim(0, 1010)
    ax_sleep.set_xlabel("epoch", fontsize=9, color=TEXT_SUB)

    handles = [Line2D([], [], color=run.color, linewidth=1.6, marker="o", markersize=4.5,
                      markeredgecolor=SURFACE, label=run.label) for run in RUNS]
    handles.append(Line2D([], [], color=ORIGINAL.color, linewidth=0, marker="D", markersize=6,
                          markerfacecolor=SURFACE, markeredgewidth=1.6, label=ORIGINAL.label))
    handles.append(Line2D([], [], color=TEXT_SUB, linewidth=0, marker="o", markersize=6,
                          markerfacecolor=SURFACE, markeredgewidth=1.6, label="val 最良の ckpt（生成の種 2 つ）"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.962), ncol=2, frameon=False,
               fontsize=9, labelcolor=TEXT)
    fig.suptitle("DDPM（幅 16）Stage 1 の途中の ckpt", fontsize=13, color=TEXT, x=0.07, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] 書いた: {out_path}  ほかの Pre-trained の総変動距離: "
          f"{ {k: round(v, 3) for k, v in others.items()} }")
    return out_path


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="DDPM_Minimal の Stage 1 の途中の ckpt ごとの指標の図")
    ap.add_argument("--out", type=Path, default=FIG_PATH)
    plot(ap.parse_args().out)


if __name__ == "__main__":
    main()
