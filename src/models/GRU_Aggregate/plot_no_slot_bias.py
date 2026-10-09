"""
plot_no_slot_bias.py
====================
slot_bias ありの GRU（gru）と、slot_bias なしの GRU（gru_no_slot_bias、計画書 §12）を並べる。
どちらも学習後の補正なし・g = 1.0・種 42〜46。

    出力                                                 中身
    figures/stage1_no_slot_bias_curves.png                12 活動の時刻別行動者率（米国加重）。ATUS 実と 2 つの GRU
    figures/stage1_no_slot_bias_teacher_gap.png           実データの履歴で予測した率 − 行動者率（学習分割、12 活動 × 96 スロット）
    data/processed/aggregates/stage1_gru_no_slot_bias_train.csv
                                                          種ごとの学習の結果（最良 epoch・val）、上の差の最大と平均、
                                                          曲線の MSE（米国加重）
    data/processed/aggregates/stage1_gru_no_slot_bias_minute.csv
                                                          時刻の分（:00 / :15 / :30 / :45）ごとの、そのスロットへの切替の率
                                                          （ATUS 実と生成、米国加重）と、上の差の絶対値の平均（2 つの GRU）

★曲線は種平均の線と、種の最小〜最大の薄い帯。28 群は ATUS の群シェア pi_atus でまとめる（Stage 1 の評価と同じ）。
★「実データの履歴で予測した率」は gm.teacher_forced_rates（計画書 §3.2 の一致の左辺）。slot_bias があれば
  学習が止まった点で 96 × 12 のセルごとに行動者率と一致し、なければ 96 スロットの平均だけが一致する（§12.2）。
★ATUS の日誌は切替が :00 と :30 に集まる（記入の丸め）。slot_bias はスロットごとの値を持てるので、この
  スロットごとのぎざぎざを teacher forcing で写せる。分ごとの表はその違いを測る（minute_table）。
★読み込み・米国加重のまとめ方・図の体裁は stage1_gru_compare と共有する（写し書きしない）。

データフロー:

```mermaid
flowchart TD
    SET["compare.setup()<br/>real（ATUS 実）, pi_atus (28,)"] --> AT["compare.us_weighted_slot_rates(compare.agr.group_rates(*real), pi_atus)<br/>atus (12, 96)"]
    GP["compare.load_pool(arm, seed)<br/>arm ∈ ARMS, seed ∈ SEEDS"] --> SC["seed_curves<br/>curves[arm] (5, 12, 96)"]
    SET --> SC
    CK["gm.load_model(compare.ckpt_file(arm, seed))"] --> TG["teacher_gap(model, train_part, real_train)<br/>gaps[arm] (5, 12, 96)"]
    SPL["gm.load_split() → train_part<br/>sm.population_rates → real_train (12, 96)"] --> TG
    AT --> F1["plot_curves(atus, curves)<br/>stage1_no_slot_bias_curves.png"]
    SC --> F1
    TG --> F2["plot_teacher_gap(gaps)<br/>stage1_no_slot_bias_teacher_gap.png"]
    TG --> TAB["train_table → SUMMARY_CSV"]
    SC --> TAB
    AT --> TAB
    TG --> MIN["minute_table(real, gaps) → MINUTE_CSV"]
    GP --> MIN
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/plot_no_slot_bias.py

    前提: 2 つの GRU の ckpt と g = 1.0 のプール（種 42〜46）
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42 --no-slot-bias   # 種 43〜46 も
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/GRU_Aggregate -> repo root
FIG_DIR = Path(__file__).resolve().parent / "figures"
SUMMARY_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "stage1_gru_no_slot_bias_train.csv"
MINUTE_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "stage1_gru_no_slot_bias_minute.csv"

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
gm: Any = compare.gm

ARMS: tuple[str, ...] = ("gru", "gru_no_slot_bias")      # compare の arm 名
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
ARM_LEGEND: dict[str, str] = {"gru": "GRU slot_bias あり", "gru_no_slot_bias": "GRU slot_bias なし"}
LABEL_ATUS = "ATUS"
COLOR_ATUS = "#1a1a19"
BAND_ALPHA = 0.14                                        # 種の範囲の帯の濃さ（compare.plot_curves と同じ）
# 分ごとの表で、切替の率を並べる arm（2 つの GRU に、採用候補と DDPM を参考に足す）
MINUTE_ARMS: tuple[str, ...] = ("gru", "gru_no_slot_bias", "gru_calg125", "ddpm_tf96")
MINUTES: tuple[int, ...] = (0, 15, 30, 45)
# スロット s の開始時刻の分（04:00 起点の 15 分刻み）
SLOT_MINUTE: IntArr = np.asarray((15 * np.arange(gm.NUM_SLOTS)) % 60, dtype=np.int64)


def seed_curves(arm: str, pi_atus: FloatArr) -> FloatArr:
    """arm の種ごとの米国加重の時刻別行動者率, -> (種の数, 12, 96)"""
    return np.stack([compare.us_weighted_slot_rates(compare.cur.pool_to_slot_rates(compare.load_pool(arm, s)),
                                                    pi_atus) for s in SEEDS])


def teacher_gap(model: Any, train_part: Any, real_train: FloatArr) -> FloatArr:
    """実データの履歴で予測した率 − 行動者率（学習分割）, -> (12, 96)

    Args:
        model: 学習済みの GRUScheduler
        train_part: gm.load_split() の学習分割
        real_train: 学習分割の行動者率 sm.population_rates, (12, 96)
    """
    tf = gm.teacher_forced_rates(model, train_part.sched, train_part.cond_idx, train_part.weight)
    return np.asarray(tf - real_train, dtype=np.float64)


def arm_gaps(arm: str) -> FloatArr:
    """arm の種ごとの teacher_gap, -> (種の数, 12, 96)"""
    train_part, _ = gm.load_split()
    real_train = gm.sm.population_rates(train_part.sched, train_part.weight).numpy().astype(np.float64)
    return np.stack([teacher_gap(gm.load_model(compare.ckpt_file(arm, s)), train_part, real_train) for s in SEEDS])


def train_table(curves: dict[str, FloatArr], gaps: dict[str, FloatArr], atus: FloatArr) -> pd.DataFrame:
    """種ごとの学習の結果・teacher_gap の最大と平均・曲線の MSE

    Returns:
        列 arm / seed / best_epoch / best_val / tf_gap_max / tf_gap_mean / curve_mse_us
    """
    rows = []
    for arm in ARMS:
        for k, seed in enumerate(SEEDS):
            cfg = torch.load(compare.ckpt_file(arm, seed), map_location="cpu")["config"]
            gap = np.abs(gaps[arm][k])
            rows.append({"arm": arm, "seed": seed, "best_epoch": int(cfg["best_epoch"]),
                         "best_val": float(cfg["best_val"]), "tf_gap_max": float(gap.max()),
                         "tf_gap_mean": float(gap.mean()),
                         "curve_mse_us": float(np.mean((atus - curves[arm][k]) ** 2))})
    return pd.DataFrame(rows)


def switch_rate_by_minute(sched: IntArr, weight: FloatArr) -> dict[int, float]:
    """スロット s（s ≥ 1）へ切り替わる人の加重の割合を、s の開始の分ごとに平均する

    Args:
        sched: スケジュール, (N, 96)
        weight: 個票の重み, (N,)

    Returns:
        分（0 / 15 / 30 / 45）→ 切替の率。04:00（s = 0）は前のスロットが無いので含めない
    """
    switched = sched[:, 1:] != sched[:, :-1]                                        # (N, 95)
    per_slot = np.einsum("n,ns->s", weight / weight.sum(), switched)
    minute = SLOT_MINUTE[1:]
    return {mm: float(per_slot[minute == mm].mean()) for mm in MINUTES}


def minute_table(real: tuple[IntArr, IntArr, FloatArr], gaps: dict[str, FloatArr]) -> pd.DataFrame:
    """分ごとの切替の率（ATUS 実と MINUTE_ARMS、生成は米国加重）と、2 つの GRU の |teacher_gap| の平均

    Note:
        ★生成プールは群一様なので、群ごとに ATUS の TUFINLWGT の和へ重みを付け直す（sel.im.group_reweight）

    Returns:
        列 source / seed / quantity（switch_rate か abs_tf_gap）/ min00 / min15 / min30 / min45。
        seed = −1 は ATUS 実、または種の平均
    """
    sched_r, d_r, w_r = real
    rows: list[dict[str, Any]] = [{"source": "atus", "seed": -1, "quantity": "switch_rate",
                                   **{f"min{mm:02d}": v for mm, v in switch_rate_by_minute(sched_r, w_r).items()}}]
    for arm in MINUTE_ARMS:
        for seed in compare.ARM_SEEDS[arm]:
            pool = compare.load_pool(arm, seed)
            n_d, m, n_s = pool.shape
            gen_d = np.repeat(np.arange(n_d), m)
            w_gen = np.asarray(compare.sel.im.group_reweight(gen_d, w_r, d_r, gm.D_GROUPS), dtype=np.float64)
            rate = switch_rate_by_minute(pool.reshape(n_d * m, n_s), w_gen)
            rows.append({"source": arm, "seed": seed, "quantity": "switch_rate",
                         **{f"min{mm:02d}": v for mm, v in rate.items()}})
    for arm in ARMS:
        for k, seed in enumerate(SEEDS):
            gap = np.abs(gaps[arm][k])                                              # (12, 96)
            rows.append({"source": arm, "seed": seed, "quantity": "abs_tf_gap",
                         **{f"min{mm:02d}": float(gap[:, SLOT_MINUTE == mm].mean()) for mm in MINUTES}})
    cols = [f"min{mm:02d}" for mm in MINUTES]
    per_seed = [r for r in rows if r["seed"] >= 0]
    for source, quantity in dict.fromkeys((r["source"], r["quantity"]) for r in per_seed):
        sub = [r for r in per_seed if r["source"] == source and r["quantity"] == quantity]
        rows.append({"source": source, "seed": -1, "quantity": quantity,
                     **{c: float(np.mean([r[c] for r in sub])) for c in cols}})
    return pd.DataFrame(rows)


def _panel_axes(fig_title: str, y_label: str) -> tuple[Any, Any, Any]:
    """4×3 の小倍数の枠と figure_module を作る"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = compare.rd._figure_module()
    fm.setup_fonts()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(compare.cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.set_title(f"{compare.cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel(y_label, fontsize=10)
    fig.suptitle(fig_title, fontsize=15, y=0.995)
    return fig, axes, fm


def _finish(fig: Any, axes: Any, out: Path) -> None:
    """凡例を上に 1 行で置き、保存する"""
    import matplotlib.pyplot as plt
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[no_slot_bias] 図: {out}")


def plot_curves(atus: FloatArr, curves: dict[str, FloatArr], out: Path) -> None:
    """12 活動の時刻別行動者率（ATUS 実・2 つの GRU）

    Args:
        atus: ATUS 実（米国加重）, (12, 96)
        curves: arm → 種ごとの曲線, (種の数, 12, 96)
        out: 出力 PNG
    """
    fig, axes, fm = _panel_axes("12 活動の時刻別行動者率", "行動者率")
    hours = compare.cur.slot_hours()
    for c in range(len(compare.cur.ACT_NAMES)):
        ax = axes[c // 3][c % 3]
        for arm in ARMS:
            st = curves[arm][:, c]
            ax.fill_between(hours, st.min(axis=0), st.max(axis=0), color=compare.ARM_COLOR[arm],
                            alpha=BAND_ALPHA, lw=0)
        ax.plot(hours, atus[c], color=COLOR_ATUS, lw=2.6, label=LABEL_ATUS, solid_capstyle="round")
        for arm in ARMS:
            ax.plot(hours, curves[arm][:, c].mean(axis=0), color=compare.ARM_COLOR[arm], lw=1.6,
                    label=ARM_LEGEND[arm], solid_capstyle="round", solid_joinstyle="round")
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_ylim(bottom=0.0)
    _finish(fig, axes, out)


def plot_teacher_gap(gaps: dict[str, FloatArr], out: Path) -> None:
    """実データの履歴で予測した率 − 行動者率（学習分割）。0 の線が一致

    Args:
        gaps: arm → 種ごとの差, (種の数, 12, 96)
        out: 出力 PNG
    """
    fig, axes, fm = _panel_axes("実データの履歴で予測した率と行動者率の差", "予測 − 実")
    hours = compare.cur.slot_hours()
    for c in range(len(compare.cur.ACT_NAMES)):
        ax = axes[c // 3][c % 3]
        ax.axhline(0.0, color=COLOR_ATUS, lw=1.2)
        for arm in ARMS:
            st = gaps[arm][:, c]
            ax.fill_between(hours, st.min(axis=0), st.max(axis=0), color=compare.ARM_COLOR[arm],
                            alpha=BAND_ALPHA, lw=0)
            ax.plot(hours, st.mean(axis=0), color=compare.ARM_COLOR[arm], lw=1.6, label=ARM_LEGEND[arm],
                    solid_capstyle="round", solid_joinstyle="round")
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        lim = 1.1 * max(float(np.abs(gaps[arm][:, c]).max()) for arm in ARMS)
        ax.set_ylim(-lim, lim)
    _finish(fig, axes, out)


def main() -> None:
    """CLI"""
    real, pi_atus, _ = compare.setup()
    atus = compare.us_weighted_slot_rates(compare.agr.group_rates(*real), pi_atus)
    curves = {arm: seed_curves(arm, pi_atus) for arm in ARMS}
    gaps = {arm: arm_gaps(arm) for arm in ARMS}
    table = train_table(curves, gaps, atus)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))
        print(table.groupby("arm", sort=False)[["best_val", "tf_gap_max", "tf_gap_mean", "curve_mse_us"]]
              .agg(["mean", "min", "max"]).to_string())
    minute = minute_table(real, gaps)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(minute[minute["seed"] == -1].to_string(index=False))
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUMMARY_CSV, index=False)
    minute.to_csv(MINUTE_CSV, index=False)
    print(f"[no_slot_bias] 表: {SUMMARY_CSV}, {MINUTE_CSV}")
    plot_curves(atus, curves, FIG_DIR / "stage1_no_slot_bias_curves.png")
    plot_teacher_gap(gaps, FIG_DIR / "stage1_no_slot_bias_teacher_gap.png")


if __name__ == "__main__":
    main()
