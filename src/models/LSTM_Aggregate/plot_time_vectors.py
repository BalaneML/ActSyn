"""
plot_time_vectors.py
====================
時刻符号を足した最低限の構成（LSTM・GRU × fixed / learned）で、学習後の時刻のベクトル time(s) が
1 日のどこで大きく変わるかを見る（参考の診断。判定には使わない）。

量:
    step[s] = ‖time(s+1) − time(s)‖ / spread,   s = 0..94
    spread  = sqrt(mean_s ‖time(s) − mean_s time(s)‖²)   （1 日の中でのばらつき。モデルごとの大きさをそろえる）
    ★step[s] は、スロット s と s+1 の境目（時刻 = スロット s+1 の開始）の値
    ★time(s) は time_input の出力そのもの（fixed は W φ(s) + b、learned は E[s]）。入力に足される量を比べる
    ★1 日の最後と最初（3:45 → 4:00）は同じ日の中で隣り合わないので測らない

作るもの:
    data/processed/aggregates/stage1_time_vector_steps.csv   cell × time_enc × seed × 境目の step
    figures/stage1_time_vector_steps.png                       セルごとの step（種平均の線と、種の範囲の帯）

データフロー:

```mermaid
flowchart TD
    CK["ec.minimal_paths(arm, seed)<br/>ckpt のパス"] --> TV["time_vectors(arm, seed)<br/>model.time_input(arange(96)) → (96, H)"]
    TV --> RS["relative_steps(vectors)<br/>step (95,)"]
    RS --> CS["collect_steps()<br/>steps[(cell, time_enc)] (種, 95)"]
    CS --> TAB["step_table(steps)<br/>stage1_time_vector_steps.csv"]
    CS --> TOP["top_boundaries(steps) / daily_mean_summary(steps)<br/>表示だけ"]
    CS --> FIG["plot_steps(steps)<br/>figures/stage1_time_vector_steps.png"]
```

使い方:
    .venv/bin/python src/models/LSTM_Aggregate/plot_time_vectors.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/LSTM_Aggregate -> repo root
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

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


# arm の定義（MINIMAL_MODULES・minimal_paths）と時刻符号の色・ラベルの出所
ec: Any = _load("lstm_eval_curves", Path(__file__).resolve().parent / "eval_curves.py")
cur: Any = ec.cmp.cur
NUM_SLOTS: int = ec.lm.NUM_SLOTS                  # 96

TIME_ENCS_WITH_VECTORS: tuple[str, ...] = ("fixed", "learned")   # none は時刻のベクトルを持たない
DEVICE = "cpu"
TOP_K = 5                                         # 表示する境目の数（セル × 時刻符号ごと）


def time_vectors(arm: str, seed: int) -> FloatArr:
    """arm と種の最良の ckpt から、96 スロットの時刻のベクトルを取り出す

    Args:
        arm: ec.MINIMAL_MODULES のキー（time_enc が fixed か learned のもの）
        seed: 学習の乱数の種

    Returns:
        時刻のベクトル, (96, HIDDEN)

    Raises:
        ValueError: 時刻のベクトルを持たない arm（time_enc = none）のとき
    """
    spec = ec.MINIMAL_MODULES[arm]
    _, ckpt_file = ec.minimal_paths(arm, seed)
    model = spec.module.load_model(ckpt_file, DEVICE)
    if model.time_input is None:
        raise ValueError(f"{arm}（time_enc={spec.time_enc}）は時刻のベクトルを持たない")
    with torch.no_grad():
        vectors = model.time_input(torch.arange(NUM_SLOTS))
    return np.asarray(vectors.numpy(), dtype=np.float64)


def relative_steps(vectors: FloatArr) -> FloatArr:
    """隣り合うスロットの時刻のベクトルの距離を、1 日の中でのばらつきで割る

    Args:
        vectors: 時刻のベクトル, (96, H)

    Returns:
        step, (95,)。step[s] はスロット s と s+1 の境目の値
    """
    spread = float(np.sqrt(((vectors - vectors.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return np.asarray(np.linalg.norm(np.diff(vectors, axis=0), axis=1) / spread, dtype=np.float64)


def boundary_hours() -> FloatArr:
    """境目の時刻（04:00 起点の 0〜28 時間軸）, -> (95,)。境目 s はスロット s+1 の開始"""
    return np.asarray(cur.SLOT_START_HOUR + np.arange(1, NUM_SLOTS) * cur.SLOT_MINUTES / 60.0, dtype=np.float64)


StepKey = tuple[str, str]                          # (cell, time_enc)


def collect_steps() -> dict[StepKey, FloatArr]:
    """セル × 時刻符号ごとに、種ごとの step を集める

    Returns:
        (cell, time_enc) → step, (種の数, 95)。種の並びは ec.MINIMAL_SEEDS
    """
    steps: dict[StepKey, FloatArr] = {}
    for cell in ec.CELL_MODULES:
        for time_enc in TIME_ENCS_WITH_VECTORS:
            arm = ec.time_arm(cell, time_enc)
            steps[(cell, time_enc)] = np.stack([relative_steps(time_vectors(arm, seed))
                                                for seed in ec.MINIMAL_SEEDS])
    return steps


def step_table(steps: dict[StepKey, FloatArr]) -> pd.DataFrame:
    """collect_steps の戻り値を縦長の表にする

    Returns:
        列 cell / time_enc / seed / boundary（スロット s+1 の番号）/ hour（境目の時刻）/ step
    """
    hours = boundary_hours()
    rows = [{"cell": cell, "time_enc": time_enc, "seed": seed, "boundary": s + 1, "hour": float(hours[s]),
             "step": float(values[i, s])}
            for (cell, time_enc), values in steps.items()
            for i, seed in enumerate(ec.MINIMAL_SEEDS)
            for s in range(values.shape[1])]
    return pd.DataFrame(rows)


def top_boundaries(steps: dict[StepKey, FloatArr], k: int = TOP_K) -> pd.DataFrame:
    """セル × 時刻符号ごとに、種平均の step が大きい境目を k 個

    Returns:
        列 cell / time_enc / rank / boundary / clock（境目の時刻の表示）/ step_mean / step_min / step_max
    """
    hours = boundary_hours()
    rows = []
    for (cell, time_enc), values in steps.items():
        mean = values.mean(axis=0)                                                  # (95,)
        for rank, s in enumerate(np.argsort(-mean)[:k], start=1):
            rows.append({"cell": cell, "time_enc": time_enc, "rank": rank, "boundary": int(s) + 1,
                         "clock": clock_label(float(hours[s])), "step_mean": float(mean[s]),
                         "step_min": float(values[:, s].min()), "step_max": float(values[:, s].max())})
    return pd.DataFrame(rows)


def daily_mean_summary(steps: dict[StepKey, FloatArr]) -> pd.DataFrame:
    """step の 1 日の平均を種ごとに取り、その種の平均・最小・最大

    Returns:
        列 cell / time_enc / mean / min / max
    """
    rows = []
    for (cell, time_enc), values in steps.items():
        per_seed = values.mean(axis=1)                                              # (種,)
        rows.append({"cell": cell, "time_enc": time_enc, "mean": float(per_seed.mean()),
                     "min": float(per_seed.min()), "max": float(per_seed.max())})
    return pd.DataFrame(rows)


def clock_label(hour: float) -> str:
    """0〜28 時間軸の値 → 時計の表示（12.25 → "12:15"、27.75 → "3:45"）"""
    minutes = int(round(hour * 60))
    return f"{(minutes // 60) % 24}:{minutes % 60:02d}"


def plot_steps(steps: dict[StepKey, FloatArr], out: Path) -> None:
    """セルごとの step（種平均の線と、種の最小〜最大の帯）。色は時刻符号の種類（eval_curves と同じ）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = ec.cmp.rd._figure_module()
    fm.setup_fonts()
    hours = boundary_hours()
    cells = tuple(ec.CELL_MODULES)
    fig, axes = plt.subplots(1, len(cells), figsize=(7.2 * len(cells), 4.2), sharey=True, squeeze=False)
    for ax, cell in zip(axes[0], cells):
        for time_enc in TIME_ENCS_WITH_VECTORS:
            values = steps[(cell, time_enc)]                                       # (種, 95)
            color = ec.TIME_ENC_COLORS[time_enc]
            ax.fill_between(hours, values.min(axis=0), values.max(axis=0), color=color, alpha=0.18, lw=0)
            ax.plot(hours, values.mean(axis=0), color=color, lw=1.8, label=ec.TIME_ENC_LABELS[time_enc],
                    solid_capstyle="round", solid_joinstyle="round")
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_title(ec.CELL_LABELS[cell], fontsize=12)
        ax.set_xlabel("境目の時刻", fontsize=10)
    axes[0][0].set_ylabel("隣のスロットとの距離（相対値）", fontsize=10)   # 定義はモジュールの docstring の step
    fig.suptitle("時刻のベクトルの隣り合うスロット間の変化", fontsize=14, y=0.99)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.93), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[time_vectors] 図: {out}")


def main() -> None:
    """表と図を出力する"""
    steps = collect_steps()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step_table(steps).to_csv(OUT_DIR / "stage1_time_vector_steps.csv", index=False)
    ec.cmp._show("step の 1 日の平均（種の平均・最小・最大）", daily_mean_summary(steps))
    ec.cmp._show(f"step の大きい境目（種平均の上位 {TOP_K}）", top_boundaries(steps))
    plot_steps(steps, FIG_DIR / "stage1_time_vector_steps.png")


if __name__ == "__main__":
    main()
