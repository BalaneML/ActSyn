"""
plot_schedules.py
=================
生成した活動スケジュールを個票レベルで描く（1 人 1 行、横軸 04:00 → 翌 04:00、色 = 活動）。

ATUS 実データ・GRU 補正後（gru_calg125: g = 1.25 で補正・生成）・DDPM Transformer 型を横に 3 列並べる。

    stage1_gru_schedules.png         米国加重の母集団から N_POPULATION 人ずつ
    stage1_gru_schedules_groups.png  EXAMPLE_GROUPS の群ごとに N_GROUP 人ずつ

人の引き方（3 列で同じ規則）:
    ATUS 実 : TUFINLWGT に比例して重複なしで引く（群ごとの図では群の中で引く）
    生成     : 群の重み pi_atus（ATUS の TUFINLWGT の群ごとの和）÷ 群の本数 を各行の重みにして重複なしで引く
               （群ごとの図では群の中から一様に引く）
    乱数の種は SAMPLE_SEED で固定する

行の並べ方: 「睡眠・身の回り」以外の活動を最初に始めたスロットが早い順、同じなら最後に終えたスロットが早い順。

★12 活動の色は src/viz/aggddpm_architecture.ACT_COLORS（モデル図と同じ対応）。12 色すべてを色覚の違いの
  もとで見分けることはできないので、凡例を必ず付ける。

データフロー:

```mermaid
flowchart LR
    REAL["cmp.setup<br/>ATUS 実 (sched, d, w)"] --> ROWS["source_rows<br/>(sched, w)"]
    POOL["cmp.load_pool_g(arm, seed, g)<br/>pool (28, 256, 96)"] --> ROWS
    ROWS --> PICK["pick_rows<br/>重み付きで N 人を重複なしで引く"]
    PICK --> ORD["day_start_order<br/>最初に活動を始めた順"]
    ORD --> FIG["draw_panel<br/>imshow(cmap=ACT_CMAP)"]
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/plot_schedules.py
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent / "figures"

IntArr = npt.NDArray[np.int64]
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


cmp: Any = _load("gru_schedules_compare", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_gru_compare.py")
arch: Any = _load("gru_schedules_arch", REPO_ROOT / "src" / "viz" / "aggddpm_architecture.py")
cur: Any = cmp.cur
sm: Any = cmp.sm

# 並べるもの: (arm, CFG の強さ g, 種)。"atus" は ATUS 実データ
SOURCES: tuple[tuple[str, float, int], ...] = (("atus", 0.0, 0), ("gru_calg125", 1.25, 42), ("ddpm_tf96", 1.25, 42))
SOURCE_TITLES: dict[str, str] = {"atus": "ATUS 実データ", "gru_calg125": "GRU 補正後（g=1.25、種 42）",
                                 "ddpm_tf96": "DDPM Transformer 型（g=1.25、種 42）"}
N_POPULATION = 100
N_GROUP = 30
SAMPLE_SEED = 0
# 群ごとの図に出す群（sm.d_index(性, 年齢 7 区分, 就業)）。生活の形が大きく違う 4 群
EXAMPLE_GROUPS: tuple[int, ...] = (sm.d_index(0, 2, 1), sm.d_index(1, 2, 0), sm.d_index(1, 0, 0),
                                   sm.d_index(0, 6, 0))
SLEEP = cur.ACT_NAMES.index("SLEEP_PERSONAL")


def source_rows(real: tuple[IntArr, IntArr, FloatArr], pi_atus: FloatArr, src: str, g: float, seed: int,
                group: int | None) -> tuple[IntArr, FloatArr]:
    """1 列ぶんの個票と、引くときの重み

    Args:
        real: ATUS 実の (sched, d, w)（cmp.setup の 1 つ目）
        pi_atus: 米国加重の群の重み, (28,)
        src: "atus" または cmp.pool_csv_g の arm
        g: CFG の強さ（"atus" では使わない）
        seed: 学習の種（"atus" では使わない）
        group: 群インデックス。None なら母集団（米国加重）

    Returns:
        (スケジュール (N, 96), 引く重み (N,))
    """
    if src == "atus":
        sched, d, w = real
        keep = np.ones(len(sched), dtype=bool) if group is None else d == group
        return sched[keep], w[keep]
    pool = cmp.load_pool_g(src, seed, g)
    n_d, m, n_s = pool.shape
    if group is not None:
        return pool[group], np.ones(m, dtype=np.float64)
    d = np.repeat(np.arange(n_d), m)
    return pool.reshape(n_d * m, n_s), pi_atus[d] / m


def pick_rows(sched: IntArr, w: FloatArr, n: int, rng: np.random.Generator) -> IntArr:
    """重み w に比例して n 人を重複なしで引く"""
    return sched[rng.choice(len(sched), n, replace=False, p=w / w.sum())]


def day_start_order(sched: IntArr) -> IntArr:
    """「睡眠・身の回り」以外を最初に始めたスロットが早い順（同じなら最後に終えたスロットが早い順）の並び"""
    active = sched != SLEEP
    any_active = active.any(axis=1)
    start = np.where(any_active, active.argmax(axis=1), sm.NUM_SLOTS)
    end = np.where(any_active, sm.NUM_SLOTS - 1 - active[:, ::-1].argmax(axis=1), -1)
    return np.lexsort((end, start))


def draw_panel(ax: Any, sched: IntArr, show_x: bool) -> None:
    """1 人 1 行の帯を 1 軸に描く（行は day_start_order の順）"""
    ax.imshow(sched[day_start_order(sched)], aspect="auto", cmap=arch.ACT_CMAP, vmin=0, vmax=sm.NUM_ACT - 1,
              interpolation="nearest")
    ticks = np.arange(0, sm.NUM_SLOTS + 1, 16)                         # 4 時間ごと
    ax.set_xticks(ticks - 0.5, [f"{(4 + t // 4) % 24}:00" for t in ticks] if show_x else [])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def add_legend(fig: Any) -> None:
    """12 活動の凡例を図の下に置く（色だけに活動の同定を負わせない）"""
    from matplotlib.patches import Patch
    handles = [Patch(color=arch.ACT_COLORS[c], label=cur.ACT_JA[name]) for c, name in enumerate(cur.ACT_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))


def plot_population(real: tuple[IntArr, IntArr, FloatArr], pi_atus: FloatArr, out: Path) -> None:
    """米国加重の母集団から N_POPULATION 人ずつ（3 列）"""
    import matplotlib.pyplot as plt
    cmp.rd._figure_module().setup_fonts()
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(17, 8.5))
    for ax, (src, g, seed) in zip(axes, SOURCES):
        sched, w = source_rows(real, pi_atus, src, g, seed, None)
        draw_panel(ax, pick_rows(sched, w, N_POPULATION, np.random.default_rng(SAMPLE_SEED)), show_x=True)
        ax.set_title(SOURCE_TITLES[src], fontsize=12)
        ax.set_xlabel("時刻", fontsize=10)
    axes[0].set_ylabel(f"{N_POPULATION} 人（活動を始めた時刻の早い順）", fontsize=10)
    fig.suptitle("活動スケジュールの個票", fontsize=14, y=0.995)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[schedules] 図: {out}")


def plot_groups(real: tuple[IntArr, IntArr, FloatArr], pi_atus: FloatArr, out: Path) -> None:
    """EXAMPLE_GROUPS の群ごとに N_GROUP 人ずつ（行 = 群、列 = 3 つの出所）"""
    import matplotlib.pyplot as plt
    cmp.rd._figure_module().setup_fonts()
    fig, axes = plt.subplots(len(EXAMPLE_GROUPS), len(SOURCES), figsize=(17, 3.2 * len(EXAMPLE_GROUPS) + 1.2))
    for i, group in enumerate(EXAMPLE_GROUPS):
        for j, (src, g, seed) in enumerate(SOURCES):
            ax = axes[i][j]
            sched, w = source_rows(real, pi_atus, src, g, seed, group)
            draw_panel(ax, pick_rows(sched, w, N_GROUP, np.random.default_rng(SAMPLE_SEED)),
                       show_x=i == len(EXAMPLE_GROUPS) - 1)
            if i == 0:
                ax.set_title(SOURCE_TITLES[src], fontsize=12)
            if j == 0:
                ax.set_ylabel(cmp.group_label(group), fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("時刻", fontsize=10)
    fig.suptitle("群ごとの活動スケジュールの個票", fontsize=14, y=0.995)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.975))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[schedules] 図: {out}")


def main() -> None:
    """2 枚の図を描く"""
    import matplotlib
    matplotlib.use("Agg")
    real, pi_atus, _ = cmp.setup()
    plot_population(real, pi_atus, FIG_DIR / "stage1_gru_schedules.png")
    plot_groups(real, pi_atus, FIG_DIR / "stage1_gru_schedules_groups.png")


if __name__ == "__main__":
    main()
