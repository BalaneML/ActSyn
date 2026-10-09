"""
eval_curves.py
==============
LSTM 最低限と GRU 最低限（GRU_Minimal。LSTM とセルだけが違う GRU）の Stage 1 の時刻別行動者率を、
活動ごとの平均誤差・MAE・最大誤差（pt）と Top-5 の 15 分区間で並べる。GRU H = 128 と DDPM Transformer 型は参考
（米国加重。stage1_gru_compare.py の Q4 と同じ測り方）。
2026-10-06 から、最低限の 2 つに時刻符号（fixed / learned）を足した 4 つの arm も並べる。

★比べるもの（表は curves のすべて、図は PLOT_ARMS と TIME_PLOT_ARMS だけ）:
    lstm              LSTM 最低限（時刻符号・slot_bias・CFG なし、重みなしの損失、g = 1.0）  種 MINIMAL_SEEDS
    gru_minimal       GRU 最低限（LSTM 最低限とセルだけが違う）                              種 MINIMAL_SEEDS
    {lstm,gru_minimal}_time_fixed    最低限 ＋ 固定の時刻符号（Transformer 型 96 次元 φ ＋ Linear）  種 MINIMAL_SEEDS
    {lstm,gru_minimal}_time_learned  最低限 ＋ 学習型の時刻符号（Embedding(96, H)）                 種 MINIMAL_SEEDS
    {lstm,gru_minimal}_time_learned{size_tag}  学習型で層数 {1,2} × 幅 {64,128} × weight decay {0.01,0.1,1} の
                                               12 構成（GRID_SIZES。1 層・64・1 は lstm_time_learned と同じ）  種 MINIMAL_SEEDS
    gru               GRU H = 128（補正前、g = 1.0、重み付きの損失）                         種 42〜46
    gru_no_slot_bias  GRU H = 128 slot_bias なし（g = 1.0）。表だけ                          種 42〜46
    gru_calg125       GRU H = 128 補正後（g = 1.25 で補正・生成。採用候補）。表だけ           種 42〜46
    ddpm_tf96         DDPM Transformer 型（g = 1.25）。表だけ                                種 42〜46
  ★床は arm の種の本数ごとに作る（どの arm も種 5 本なので、同じ床になる）
  ★Top-k は k = TOP_K = 5（隣り合う区間も別に数える）。sre.TOP_K（= 3、GRU・Stage 2 の報告の値）は変えない
  ★図の丸印は、LSTM 最低限と GRU 最低限の Top-5 だけ（GRU H = 128 は線だけ）

作る表（data/processed/aggregates/）:
    stage1_lstm_curve_errors.csv  arm × 活動。平均誤差・MAE・最大誤差（種平均の曲線）と床（sre.activity_error_table）
    stage1_lstm_curve_topk.csv    arm × 活動 × 5。Top-5 の 15 分区間（sre.topk_error_table、k = 5）
    stage1_lstm_curve_seed.csv    arm × 種 × 活動。種ごとの曲線の平均誤差・MAE・最大 |誤差|（sre.seed_error_table）。
                                  eval_vs_gru.py が判定 D1（セルの差）と D2（時刻符号の効果）に使う
    stage1_lstm_curve_spots.csv   最低限の構成の arm × SPOTS。前回、時刻符号なしの 2 つが大きく外した 15 分区間の誤差
                                  （種平均・最小・最大）と、その区間の床（spot_table）

作る図（figures/）:
    stage1_lstm_curve_errors.png              PLOT_ARMS（LSTM 最低限・GRU 最低限・GRU H = 128）
    stage1_time_enc_curve_errors_{セル}.png   TIME_PLOT_ARMS[セル]（時刻符号なし・固定・学習型）。セルは lstm / gru_minimal
      ★色は時刻符号の種類ごとに固定し、2 枚で同じにする（TIME_ENC_COLORS）
    stage1_width_curve_errors_{セル}.png      WIDTH_PLOT_ARMS[セル]（学習型の H = 64・wd = 1 と H = 128・wd = 0.01）
    stage1_wd_curve_errors_{セル}.png         WD_PLOT_ARMS[セル]（学習型の H = 128 で wd = 0.01・0.1・1.0。青の濃淡）
    stage1_rate_curves_h64_wd0.01.png         RATE_PLOT_ARMS（1 層・H = 64・wd = 0.01 の LSTM と GRU）の時刻別行動者率を
                                              ATUS 実と並べる（誤差ではなく値そのもの）

データフロー:

```mermaid
flowchart TD
    LP["minimal_curves(arm)<br/>MINIMAL_MODULES[arm] = MinimalArm(module, time_enc, hidden, weight_decay, num_layers)<br/>minimal_paths(arm, seed)"] --> CV["curves[arm]<br/>種ごとの (12, 96)"]
    GP["cmp.load_pool(arm, seed)<br/>GRU H = 128・DDPM"] --> CV
    REAL["cmp.setup()<br/>ATUS 実"] --> RC["real_curve (12, 96)"]
    CV --> TAB["cmp.curve_error_tables<br/>errors / floors"]
    RC --> TAB
    TAB --> TS["topk_and_seed_tables<br/>topk（k = 5）/ seed_table"]
    CV --> TS
    TAB --> FIG["cmp.plot_curve_errors<br/>figures/stage1_lstm_curve_errors.png"]
    TS --> FIG
    TAB --> TFIG["cmp.plot_curve_errors（TIME_PLOT_ARMS）<br/>figures/stage1_time_enc_curve_errors_{セル}.png"]
    TS --> TFIG
    TAB --> WFIG["cmp.plot_curve_errors（WIDTH_PLOT_ARMS / WD_PLOT_ARMS）<br/>figures/stage1_{width,wd}_curve_errors_{セル}.png"]
    TS --> WFIG
    CV --> SP["spot_table<br/>stage1_lstm_curve_spots.csv"]
    TAB --> SP
```

使い方:
    .venv/bin/python src/models/LSTM_Aggregate/eval_curves.py

出力: data/processed/aggregates/stage1_lstm_curve_{errors,topk,seed,spots}.csv と
      src/models/LSTM_Aggregate/figures/stage1_lstm_curve_errors.png・stage1_time_enc_curve_errors_{lstm,gru_minimal}.png・
      stage1_{width,wd}_curve_errors_{lstm,gru_minimal}.png・stage1_rate_curves_h64_wd0.01.png
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/LSTM_Aggregate -> repo root
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

FloatArr = npt.NDArray[np.float64]
Floors = dict[str, dict[str, FloatArr]]           # 段の名前（low / high）→ sre.floor_errors の戻り値


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage1_gru_compare.py と同じ規則）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cmp: Any = _load("stage1_gru_compare", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_gru_compare.py")
lm: Any = _load("lstm_aggregate_model", Path(__file__).resolve().parent / "model.py")
gmin: Any = _load("gru_minimal_model", REPO_ROOT / "src" / "models" / "GRU_Minimal" / "model.py")
sre: Any = cmp.sre

MINIMAL_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
# 最低限の構成の 2 つのセル → 保存先（pool_path・ckpt_path）と load_model を持つモジュール
CELL_MODULES: dict[str, Any] = {"lstm": lm, "gru_minimal": gmin}
TIME_ENCS: tuple[str, ...] = ("none", "fixed", "learned")


def time_arm(cell: str, time_enc: str) -> str:
    """セルと時刻符号の種類から arm 名を作る（none はセルの名前のまま。例: lstm_time_learned）"""
    return cell if time_enc == "none" else f"{cell}_time_{time_enc}"


@dataclass(frozen=True)
class MinimalArm:
    """最低限の構成の arm の中身（保存先を決める引数）

    Attributes:
        module: 保存先（pool_path・ckpt_path）と load_model を持つモジュール（lm か gmin）
        time_enc: 時刻符号の種類 none / fixed / learned
        hidden: 隠れ状態の幅 H
        weight_decay: 学習の weight decay
        num_layers: 再帰の層数
    """
    module: Any
    time_enc: str
    hidden: int = lm.HIDDEN
    weight_decay: float = lm.WEIGHT_DECAY
    num_layers: int = lm.NUM_LAYERS


# 構造と weight decay の組み合わせ（2026-10-06 ユーザー指示。時刻符号はどれも学習型）。
#   層数 × 幅 × weight decay の 12 構成。(1, 64, 1.0) は採用した構成（time_arm(cell, "learned") と同じ arm）
GRID_LAYERS: tuple[int, ...] = (1, 2)
GRID_HIDDENS: tuple[int, ...] = (64, 128)
GRID_WEIGHT_DECAYS: tuple[float, ...] = (0.01, 0.1, 1.0)
GridSize = tuple[int, int, float]                 # (層数, 幅, weight decay)
GRID_SIZES: tuple[GridSize, ...] = tuple((n, h, wd) for n in GRID_LAYERS for h in GRID_HIDDENS
                                         for wd in GRID_WEIGHT_DECAYS)


def grid_arm(cell: str, num_layers: int, hidden: int, weight_decay: float) -> str:
    """学習型の時刻符号で層数・幅・weight decay を変えた arm 名

    Note:
        名前の印は lm.size_tag と同じ。採用した構成（1 層・H = 64・wd = 1.0）は印が空で、time_arm(cell, "learned")
        と同じ名前になる（例: lstm_time_learned、lstm_time_learned_h128_wd0.01、gru_minimal_time_learned_h64_l2_wd1）

    Args:
        cell: セル（CELL_MODULES のキー）
        num_layers: 再帰の層数
        hidden: 隠れ状態の幅 H
        weight_decay: weight decay

    Returns:
        arm 名
    """
    return f"{time_arm(cell, 'learned')}{lm.size_tag(hidden, num_layers, weight_decay)}"


def width_arm(cell: str, hidden: int, weight_decay: float) -> str:
    """1 層で幅と weight decay を変えた arm 名（grid_arm の層数 1 の略記。例: lstm_time_learned_h128_wd0.01）"""
    return grid_arm(cell, lm.NUM_LAYERS, hidden, weight_decay)


# 最低限の構成の arm → MinimalArm。eval_vs_gru.py・plot_time_vectors.py・grid_summary.py もこれを使う
# （arm の定義の唯一の出所）。H = 64・wd = 1.0 の 3 種類の時刻符号と、GRID_SIZES の学習型
MINIMAL_MODULES: dict[str, MinimalArm] = {
    **{time_arm(cell, enc): MinimalArm(mod, enc) for cell, mod in CELL_MODULES.items() for enc in TIME_ENCS},
    **{grid_arm(cell, n, h, wd): MinimalArm(mod, "learned", h, wd, n)
       for cell, mod in CELL_MODULES.items() for n, h, wd in GRID_SIZES}}
COMPARE_ARMS: tuple[str, ...] = ("gru", "gru_no_slot_bias", "gru_calg125", "ddpm_tf96")   # stage1_gru_compare の arm 名
TOP_K = 5                                         # 2026-10-06 ユーザー指示（隣り合う区間も別に数える）
PLOT_ARMS: tuple[str, ...] = ("lstm", "gru_minimal", "gru")
TOPK_MARK_ARMS: tuple[str, ...] = ("lstm", "gru_minimal")
# ★gru_minimal の赤は、lstm の青緑・gru の青との 3 色で validate_palette の全ペアの検査を通る
#   （CVD ΔE 13.1・通常 16.2・コントラスト 3:1 以上）。ほかの arm の色とも重ならない
COLORS: dict[str, str] = {**cmp.ARM_COLOR, "gru_minimal": "#e34948"}
LABELS: dict[str, str] = {**cmp.ARM_LABELS, "lstm": "LSTM 最低限", "gru_minimal": "GRU 最低限",
                          "gru": "GRU H=128（補正前）"}

# 時刻符号の比較の図（セルごとに 1 枚）。色は時刻符号の種類で決め、2 枚で同じにする
TIME_PLOT_ARMS: dict[str, tuple[str, ...]] = {cell: tuple(time_arm(cell, enc) for enc in TIME_ENCS)
                                               for cell in CELL_MODULES}
# ★none の青緑と fixed の橙は、DDPM の図の「時刻符号なし」「Transformer 型」と同じ色（同じ意味に同じ色）。
#   learned の紫を足した 3 色は validate_palette --pairs all を通る（CVD ΔE 9.2・通常 27.6）。
#   青緑はコントラスト 3:1 未満の WARN なので、数値は表（stage1_lstm_curve_*.csv と報告）で渡す
TIME_ENC_COLORS: dict[str, str] = {"none": "#1baf7a", "fixed": "#eb6834", "learned": "#4a3aa7"}
TIME_ENC_LABELS: dict[str, str] = {"none": "時刻符号なし", "fixed": "固定（Transformer 型）",
                                   "learned": "学習型（埋め込み）"}
CELL_LABELS: dict[str, str] = {"lstm": "LSTM", "gru_minimal": "GRU"}

# 幅と weight decay の比較の図（セルごとに 1 枚）。学習型の H = 64・wd = 1 と H = 128・wd = 0.01 を並べる
WIDTH_PLOT_ARMS: dict[str, tuple[str, ...]] = {
    cell: (time_arm(cell, "learned"), width_arm(cell, 128, 0.01)) for cell in CELL_MODULES}
# ★H = 64 は時刻符号の図と同じ紫、H = 128 は青。2 色で validate_palette --pairs all を通る（CVD ΔE 13.0・通常 16.3）
WIDTH_COLORS: tuple[str, ...] = ("#4a3aa7", "#2a78d6")
# H = 128 で weight decay だけを変えた図（セルごとに 1 枚）。weight decay は順序のある量なので、青の濃淡で塗る
WD_PLOT_WEIGHT_DECAYS: tuple[float, ...] = (0.01, 0.1, 1.0)
WD_PLOT_ARMS: dict[str, tuple[str, ...]] = {
    cell: tuple(width_arm(cell, 128, wd) for wd in WD_PLOT_WEIGHT_DECAYS) for cell in CELL_MODULES}
# ★既定パレットの青の段 450・550・700（wd が大きいほど濃い）。validate_palette --ordinal を通る。
#   wd = 0.01 の段 450 は、幅の図の H = 128 と同じ色。紫（H = 64）とは区別できない段があるので、H = 64 は載せない
WD_COLORS: tuple[str, ...] = ("#2a78d6", "#1c5cab", "#0d366b")
# 時刻別行動者率そのものの図（ATUS 実と並べる。2026-10-07 ユーザー指示）。1 層・H = 64・wd = 0.01 の LSTM と GRU
RATE_PLOT_ARMS: tuple[str, ...] = tuple(grid_arm(cell, 1, 64, 0.01) for cell in CELL_MODULES)
# ★LSTM の青緑と GRU の赤は、これまでの LSTM 最低限・GRU 最低限の図と同じ色。2 色で validate_palette --pairs all を
#   通る（CVD ΔE 13.1・通常 29.9）。ATUS 実は黒の太線
RATE_COLORS: tuple[str, ...] = ("#11a3a3", "#e34948")

# 前回の報告（Stage1_lstm_results.md §1・§3.2）で、時刻符号なしの 2 つが大きく外した 15 分区間。
# (活動, 区間の開始時刻)。結果を見る前に固定（2026-10-06）
SPOTS: tuple[tuple[str, str], ...] = (
    ("MEALS", "12:00"), ("WORK", "12:00"), ("TRAVEL", "17:00"),
    ("SLEEP_PERSONAL", "07:30"), ("SLEEP_PERSONAL", "22:00"), ("SLEEP_PERSONAL", "23:00"),
    ("LEISURE_SOCIAL", "22:00"), ("LEISURE_SOCIAL", "23:00"))


def minimal_curves(arm: str, pi_atus: FloatArr) -> list[FloatArr]:
    """最低限の構成の arm（MINIMAL_MODULES のキー）の、種ごとの米国加重の時刻別行動者率

    Note:
        ★ckpt より古い生成プールは、取り違えとして止める（cmp.rep.check_fresh）

    Args:
        arm: MINIMAL_MODULES のキー
        pi_atus: 米国加重の群の重み, (28,)

    Returns:
        種ごとの時刻別行動者率 (12, 96) の並び（MINIMAL_SEEDS の順）

    Raises:
        FileNotFoundError: 生成プールが無いとき
    """
    out = []
    for seed in MINIMAL_SEEDS:
        pool_file, ckpt_file = minimal_paths(arm, seed)
        if not pool_file.exists():
            raise FileNotFoundError(f"{arm} の生成プールが無い (seed={seed}): {pool_file}")
        cmp.rep.check_fresh(pool_file, ckpt_file)
        pool = np.asarray(cmp.cur.load_sample_pool(pool_file), dtype=np.int64)
        out.append(cmp.us_weighted_slot_rates(np.asarray(cmp.cur.pool_to_slot_rates(pool), dtype=np.float64),
                                              pi_atus))
    return out


def minimal_paths(arm: str, seed: int) -> tuple[Path, Path]:
    """最低限の構成の arm と種の (生成プールのパス, 最良の ckpt のパス)

    Args:
        arm: MINIMAL_MODULES のキー
        seed: 学習の乱数の種

    Returns:
        (pool_path, ckpt_path)。損失は既定（重みなし）
    """
    a = MINIMAL_MODULES[arm]
    size = {"time_enc": a.time_enc, "hidden": a.hidden, "num_layers": a.num_layers, "weight_decay": a.weight_decay}
    return Path(a.module.pool_path(seed, **size)), Path(a.module.ckpt_path(seed, **size))


def arm_seeds(arm: str) -> tuple[int, ...]:
    """arm の種の並び（curves[arm] の並びと同じ）"""
    return MINIMAL_SEEDS if arm in MINIMAL_MODULES else tuple(cmp.ARM_SEEDS[arm])


def plot_rate_curves(curves: dict[str, list[FloatArr]], real_curve: FloatArr, arms: tuple[str, ...],
                     colors: dict[str, str], labels: dict[str, str], out: Path) -> None:
    """12 活動の時刻別行動者率（米国加重）を、ATUS 実と arm ごとに並べる

    Note:
        ★線は種平均、薄い帯は種の最小〜最大。活動ごとに y 軸を独立させる（cmp.plot_curves と同じ見た目）

    Args:
        curves: arm → 種ごとの時刻別行動者率 (12, 96) の並び
        real_curve: ATUS 実の時刻別行動者率, (12, 96)
        arms: 描く arm（凡例の順）
        colors: arm → 線の色
        labels: arm → 凡例の名前
        out: 保存先
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = cmp.rd._figure_module()
    fm.setup_fonts()
    hours = cmp.cur.slot_hours()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(cmp.cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.plot(hours, real_curve[c], color=fm.COLOR_ATUS, lw=2.6, label=fm.LABEL_ATUS, solid_capstyle="round")
        for arm in arms:
            stack = np.asarray(curves[arm], dtype=np.float64)                         # (種, 12, 96)
            ax.fill_between(hours, stack[:, c].min(axis=0), stack[:, c].max(axis=0), color=colors[arm],
                            alpha=0.14, lw=0)
            ax.plot(hours, stack[:, c].mean(axis=0), color=colors[arm], lw=1.6, label=labels[arm],
                    solid_capstyle="round", solid_joinstyle="round")
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{cmp.cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)
    fig.suptitle("12 活動の時刻別行動者率", fontsize=15, y=0.995)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[eval_curves] 図: {out}")


def clock_to_slot(clock: str) -> int:
    """15 分区間の開始時刻 → スロットの番号（04:00 起点。"12:00" → 32、"03:45" → 95）"""
    hour, minute = (int(x) for x in clock.split(":"))
    minutes = (hour * 60 + minute - int(cmp.cur.SLOT_START_HOUR) * 60) % (24 * 60)
    return minutes // int(cmp.cur.SLOT_MINUTES)


def spot_table(curves: dict[str, list[FloatArr]], real_curve: FloatArr, floors: dict[int, Floors]) -> pd.DataFrame:
    """SPOTS の 15 分区間の誤差（生成 − ATUS 実、pt）を、最低限の構成の arm で並べる

    Args:
        curves: arm → 種ごとの時刻別行動者率 (12, 96) の並び
        real_curve: ATUS 実の時刻別行動者率, (12, 96)
        floors: 種の本数 → 2 段の床（cmp.curve_error_tables の 3 つ目の戻り値）

    Returns:
        列 arm / activity / clock / slot / real（ATUS 実の行動者率）/ err_mean / err_min / err_max（種の誤差）/
        floor_low / floor_high（その区間の |誤差| の床、pt）
    """
    rows = []
    for arm in MINIMAL_MODULES:
        err = sre.error_pt(real_curve, np.asarray(curves[arm], dtype=np.float64))        # (種, 12, 96)
        lo, hi = floors[len(curves[arm])]["low"]["slot_pt"], floors[len(curves[arm])]["high"]["slot_pt"]
        for act, clock in SPOTS:
            c, s = cmp.cur.ACT_NAMES.index(act), clock_to_slot(clock)
            rows.append({"arm": arm, "activity": act, "clock": clock, "slot": s, "real": float(real_curve[c, s]),
                         "err_mean": float(err[:, c, s].mean()), "err_min": float(err[:, c, s].min()),
                         "err_max": float(err[:, c, s].max()),
                         "floor_low": float(lo[c, s]), "floor_high": float(hi[c, s])})
    return pd.DataFrame(rows)


def topk_and_seed_tables(curves: dict[str, list[FloatArr]], real_curve: FloatArr,
                         floors: dict[int, Floors]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """arm ごとの Top-k（k = TOP_K）の表と、種ごとの誤差の表

    Args:
        curves: arm → 種ごとの時刻別行動者率 (12, 96) の並び
        real_curve: ATUS 実の時刻別行動者率, (12, 96)
        floors: 種の本数 → 2 段の床（cmp.curve_error_tables の 3 つ目の戻り値）

    Returns:
        (列 arm + sre.topk_error_table の列, 列 arm + sre.seed_error_table の列)
    """
    topk_rows, seed_rows = [], []
    for arm, cs in curves.items():
        stack = np.asarray(cs, dtype=np.float64)
        lo, hi = floors[len(cs)]["low"], floors[len(cs)]["high"]
        topk_rows.append(sre.topk_error_table(real_curve, stack, cmp.cur.ACT_NAMES, lo, hi, k=TOP_K).assign(arm=arm))
        seed_rows.append(sre.seed_error_table(real_curve, stack, cmp.cur.ACT_NAMES, arm_seeds(arm)).assign(arm=arm))

    def _arm_first(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(columns=["arm", *[c for c in df.columns if c != "arm"]])
    return _arm_first(pd.concat(topk_rows, ignore_index=True)), _arm_first(pd.concat(seed_rows, ignore_index=True))


def main() -> None:
    """表と図を出力する"""
    real, pi_atus, _ = cmp.setup()
    real_curve = cmp.us_weighted_slot_rates(cmp.agr.group_rates(*real), pi_atus)
    curves: dict[str, list[FloatArr]] = {arm: minimal_curves(arm, pi_atus) for arm in MINIMAL_MODULES}
    for arm in COMPARE_ARMS:
        curves[arm] = [cmp.us_weighted_slot_rates(cmp.cur.pool_to_slot_rates(cmp.load_pool(arm, s)), pi_atus)
                       for s in cmp.ARM_SEEDS[arm]]
    errors, _, floors = cmp.curve_error_tables(curves, real_curve, real, pi_atus)   # 2 つ目（Top-3）は使わない
    topk, seed_table = topk_and_seed_tables(curves, real_curve, floors)
    spots = spot_table(curves, real_curve, floors)
    cmp._show("Stage 1 活動ごとの平均誤差・MAE・最大誤差（pt、種平均の曲線）と床", errors)
    cmp._show(f"Stage 1 活動ごとの Top-{TOP_K} の 15 分区間（pt）",
              topk[topk["arm"].isin(TOPK_MARK_ARMS)])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUT_DIR / "stage1_lstm_curve_errors.csv", index=False)
    topk.to_csv(OUT_DIR / "stage1_lstm_curve_topk.csv", index=False)
    seed_table.to_csv(OUT_DIR / "stage1_lstm_curve_seed.csv", index=False)
    spots.to_csv(OUT_DIR / "stage1_lstm_curve_spots.csv", index=False)
    cmp._show("前回大きく外した 15 分区間の誤差（pt、種平均・最小・最大）",
              spots.pivot_table(index=["activity", "clock"], columns="arm", values="err_mean", sort=False)
              .reset_index())
    cmp.plot_curve_errors(curves, real_curve, topk[topk["arm"].isin(TOPK_MARK_ARMS)], floors,
                          FIG_DIR / "stage1_lstm_curve_errors.png", arms=PLOT_ARMS,
                          band_seeds=len(MINIMAL_SEEDS), colors=COLORS, labels=LABELS)
    for cell, arms in TIME_PLOT_ARMS.items():
        # 凡例でセルが分かるよう、ラベルの頭にセルの名前を付ける（図のタイトルは共通の関数が固定で書くため）
        colors = {time_arm(cell, enc): TIME_ENC_COLORS[enc] for enc in TIME_ENCS}
        labels = {time_arm(cell, enc): f"{CELL_LABELS[cell]} {TIME_ENC_LABELS[enc]}" for enc in TIME_ENCS}
        cmp.plot_curve_errors(curves, real_curve, topk[topk["arm"].isin(arms)], floors,
                              FIG_DIR / f"stage1_time_enc_curve_errors_{cell}.png", arms=arms,
                              band_seeds=len(MINIMAL_SEEDS), colors=colors, labels=labels)
    rate_labels = {arm: f"{CELL_LABELS[cell]} 1 層・H=64・wd=0.01（種 {len(MINIMAL_SEEDS)} 本）"
                   for arm, cell in zip(RATE_PLOT_ARMS, CELL_MODULES)}
    plot_rate_curves(curves, real_curve, RATE_PLOT_ARMS, dict(zip(RATE_PLOT_ARMS, RATE_COLORS, strict=True)),
                     rate_labels, FIG_DIR / "stage1_rate_curves_h64_wd0.01.png")
    for name, plot_arms, palette in (("width", WIDTH_PLOT_ARMS, WIDTH_COLORS), ("wd", WD_PLOT_ARMS, WD_COLORS)):
        for cell, arms in plot_arms.items():
            colors = dict(zip(arms, palette, strict=True))
            labels = {arm: f"{CELL_LABELS[cell]} 学習型 H={MINIMAL_MODULES[arm].hidden}・"
                           f"wd={MINIMAL_MODULES[arm].weight_decay:g}" for arm in arms}
            cmp.plot_curve_errors(curves, real_curve, topk[topk["arm"].isin(arms)], floors,
                                  FIG_DIR / f"stage1_{name}_curve_errors_{cell}.png", arms=arms,
                                  band_seeds=len(MINIMAL_SEEDS), colors=colors, labels=labels)


if __name__ == "__main__":
    main()
