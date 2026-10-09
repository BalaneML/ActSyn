"""
stage1_gru_compare.py
=====================
GRU_Aggregate（再帰型＋交差エントロピー）と DDPM の Stage 1 を、米国加重で並べて判定する
（計画書 src/models/GRU_Aggregate/docs/plan.md §4）。

★評価は米国加重: 生成の群別の値を ATUS の TUFINLWGT の群構成（group_weights("atus")）で平均する。
  Stage 1 の目的は元分布（ATUS）の再現なので、日本人口加重は使わない。

比べるもの（ARM_SEEDS）:

    gru           本計画のモデル（g = 1.0）                       種 42〜46
    gru_no_slot_bias  slot_bias なしの構造（計画書 §12、補正なし、g = 1.0）  種 42〜46
    gru_cal       gru の slot_bias を学習後に g = 1.0 で補正し、g = 1.0 で生成（計画書 §9）   種 42〜46
    gru_calg125   gru の slot_bias を学習後に g = 1.25 で補正し、g = 1.25 で生成（計画書 §9.5） 種 42〜46
    ddpm_tf96     clock_tf96（学習時のプール、g = 1.25）          種 42〜46
    ddpm_noclock  時刻符号なしの DDPM（ガードレールの外側の基準）  種 42〜44

部（--part）と問い:

    totals      Q1 5 活動の総量の比（生成 / ATUS 実）と種間 sd → 判定 C1・C2
    trajectory  Q1 途中の ckpt の小プール（64 人/群）の総量の比。DDPM（clock_tf96_traj）と同じ大きさ
    teacher     Q2 実データの履歴で条件付けた総量（teacher_forced_rates）と、自分で生成した総量の差
    guard       Q3 switch_emd / bigram_jsd / single_slot の差 / wrap_closure の差・暗記 → 判定 C3
    curves      Q4 12 活動の時刻別行動者率、‖実 − 生成‖²、代表点の値
                活動ごとの MAE・最大誤差（pt）とその時刻、活動ごとの Top-3 の 15 分区間（eval/slot_rate_errors.py）
                床は 2 段（mse の床と同じ考え方）。どちらも種の本数ぶんの完全なモデルのプールを平均した曲線の 95% 分位
                  下限側: ATUS 実からプールを引く（生成プールの有限さだけ）
                  上限側: ATUS の回答者を復元抽出してからプールを引く（ATUS の標本誤差 + プール）
    mse         MSE を 6 通りの粒度で並べる（判定には使わない）
                curve_mse_us / curve_mse_jp : 28 群を米国加重 / 日本人口加重で 1 本にした曲線 vs ATUS 実
                group_mse_atus              : 28 群ごとの値 vs ATUS 実（全セル 28 × 12 × 96 の一様平均）
                group_mse_atus_us / _jp     : 同じ全セルの 2 乗誤差を、群の重み（米国加重 / 日本人口加重）で平均
                rate_mse_astar              : 28 群ごとの値 vs 日本の教師 A*（論文の rate_mse と同じ定義）
                床: 完全なモデルでも出る MSE は floor_pool 〜 floor_real + floor_pool の間
    groups      群別（判定には使わない）。28 群ごとの MSE vs ATUS 実と群ごとの床、群の分離（separation_ratio）
    cfg         gru_cal の CFG の強さ g を CFG_SWEEP で振り、群別・総量・切替の指標を並べる（判定には使わない）
                ★gru_cal の補正は g = 1.0 で行ったので、g ≠ 1 では総量が補正からずれうる

判定（計画書 §4.3。結果を見る前に固定）。候補 CANDIDATES（gru / gru_no_slot_bias / gru_cal / gru_calg125）の
それぞれにかける:

    C1  5 活動のうち C_MIN_ACTS 以上で、候補の総量の比の種間 sd ≤ ddpm_tf96 の種間 sd × C1_SD_RATIO
    C2  5 活動のうち C_MIN_ACTS 以上で、|候補の総量の比の種平均 − 1| ≤ 床
        床 = ATUS の回答者を復元抽出したときの比の sd × FLOOR_Z（米国加重で作り直す）
    C3  4 指標それぞれで「候補の種の最大値 ≤ ddpm_noclock の種の最大値」、かつ暗記の判定が 0 本
        ★計画書の「ddpm_noclock の種の最大値を超えない」を、候補のすべての種に課す形で読む

データフロー:

```mermaid
flowchart TD
    GP["gm.pool_path(seed)<br/>rep.pool_csv(arm, seed)"] --> LP["load_pool<br/>pool (28, M, 96)"]
    LP --> PP["rd.pool_people<br/>sched / d / w"]
    REAL["rd.model_order_people<br/>ATUS 実 (sched, d, w)"] --> GW["rd.group_weights('atus')<br/>pi_atus (28,)"]
    PP --> TOT["totals_long / totals_summary<br/>rd.profile_table → 比 → C1・C2"]
    GW --> TOT
    REAL --> FL["rd.bootstrap_floor<br/>床"]
    FL --> TOT
    EC["gm.ckpt_path(seed) の _ep{epoch:04d}.pt"] --> EPP["make_epoch_pools<br/>64 人/群"]
    EPP --> TRJ["trajectory_long / trajectory_summary"]
    CK["gm.load_model"] --> TF["gm.teacher_forced_rates<br/>(12, 96)"]
    TF --> TCH["teacher_table（Q2）"]
    LP --> TCH
    LP --> GR["guard_long<br/>sel.guardrails / sm.memorization_report → C3"]
    LP --> CV["us_weighted_slot_rates<br/>(12, 96) → curve_table（Q4）"]
    CV --> SE["curve_error_tables<br/>sre.activity_error_table / sre.topk_error_table"]
    REAL --> PMC["perfect_model_curves<br/>floor_curves (N_FLOOR_POOL, 12, 96)<br/>下限側・上限側"]
    PMC --> SE
    LP --> MS["mse_values<br/>6 通りの MSE"]
    REAL --> MF["mse_floors<br/>floor_real / floor_pool"]
    MF --> MS
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_gru_compare.py --part all
    .venv/bin/python src/eval/diagnostics/stage1_gru_compare.py --part totals
    .venv/bin/python src/eval/diagnostics/stage1_gru_compare.py --part curves --gru-seeds 42   # 途中経過

出力: data/processed/aggregates/stage1_gru_{totals_long,totals,trajectory_long,trajectory,teacher,guard_long,
      guard,curves,curve_errors,curve_topk,mse_long,mse,groups_long,groups,cfg_long,cfg}{tag}.csv と
      src/models/GRU_Aggregate/figures/stage1_gru_{totals,trajectory,curves,curve_errors,groups,cfg}{tag}.png
      {tag} は --tag の値（既定は空）
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
FIG_DIR = REPO_ROOT / "src" / "models" / "GRU_Aggregate" / "figures"
# 出力のファイル名の接尾辞（--tag）。既定の空は H = 384 の報告（Stage1_gru_results.md）と同じ名前
# ★gm の既定のパスは H = 128（2026-09-30 に切替。H = 384 は outputs/archive/gru_h384/）。
#   H = 128 は --tag _h128 で回す。タグ無しで回すと、H = 384 の報告の表と図を H = 128 の値で上書きする
OUT_TAG = ""

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
People = tuple[IntArr, IntArr, FloatArr]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage2_select.py と同じ規則）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rd: Any = _load("gru_cmp_rare_diagnosis", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_rare_diagnosis.py")
gm: Any = _load("gru_aggregate_model", REPO_ROOT / "src" / "models" / "GRU_Aggregate" / "model.py")
sre: Any = _load("slot_rate_errors", REPO_ROOT / "src" / "eval" / "slot_rate_errors.py")
rep: Any = rd.rep
cur: Any = rd.cur
sm: Any = rd.sm
agr: Any = rd.agr
sel: Any = rep.sel
st: Any = cur.st

FOCUS_ACTS: tuple[str, ...] = rd.FOCUS_ACTS
FLOOR_Z: float = rd.FLOOR_Z

# 比べるもの
ARM_SEEDS: dict[str, tuple[int, ...]] = {
    "gru": (42, 43, 44, 45, 46),
    "gru_no_slot_bias": (42, 43, 44, 45, 46),
    "gru_cal": (42, 43, 44, 45, 46),
    "gru_calg125": (42, 43, 44, 45, 46),
    "ddpm_tf96": (42, 43, 44, 45, 46),
    "ddpm_noclock": (42, 43, 44),
}
DDPM_ARMS: dict[str, str] = {"ddpm_tf96": "clock_tf96", "ddpm_noclock": "noclock"}
ARM_LABELS: dict[str, str] = {
    "gru": "GRU（g=1.0）",
    "gru_g1.25": "GRU（g=1.25）",
    "gru_no_slot_bias": "GRU slot_bias なし（g=1.0）",
    "gru_cal": "GRU 補正後（g=1.0）",
    "gru_calg125": "GRU 補正後（g=1.25）",
    "ddpm_tf96": "DDPM Transformer 型（g=1.25）",
    "ddpm_noclock": "DDPM 時刻符号なし（g=1.25）",
    "lstm": "LSTM（g=1.0）",
}
# 図の横軸の短い名前
SHORT_LABELS: dict[str, str] = {"gru": "GRU", "gru_no_slot_bias": "GRU\nslot_bias\nなし",
                                "gru_cal": "GRU\n補正後\ng=1.0", "gru_calg125": "GRU\n補正後\ng=1.25",
                                "ddpm_tf96": "DDPM\nTransformer\n型",
                                "ddpm_noclock": "DDPM\n時刻符号\nなし"}
# ★色は arm ごとに固定する（図によって系列の数が違っても同じ arm は同じ色）。
#   gru / ddpm_tf96 / ddpm_noclock は stage1_ablation_curves.ARM_COLORS の 1〜3 番目と同じ。
#   gru_cal の赤紫は、この 4 色の並びで validate_palette（色覚の検査）を通った色
#   ★5 色目は検査を通る色が無かったので、GRU 補正後は g によらず同じ赤紫にし、印で分ける
#     （ARM_HOLLOW の arm は白抜きの印。色 = モデルの種類、印 = 補正の g）
#   ★gru_no_slot_bias の紫は、gru の青との 2 色で validate_palette の全検査を通る（CVD ΔE 13.0・通常 16.3）。
#     赤紫とは色覚の検査で近いので、両方を載せる図では横軸の位置と目盛りの名前で区別する
ARM_COLOR: dict[str, str] = {"gru": "#2a78d6", "gru_no_slot_bias": "#4a3aa7",
                             "gru_cal": "#b5179e", "gru_calg125": "#b5179e",
                             "ddpm_tf96": "#eb6834", "ddpm_noclock": "#1baf7a", "lstm": "#11a3a3"}
#   ★lstm の青緑は、gru の青・ddpm_tf96 の橙との 3 色で validate_palette の全検査を通る（CVD ΔE 15.1）。
#     LSTM_Aggregate/eval_curves.py の図で使う
ARM_HOLLOW: frozenset[str] = frozenset({"gru_cal"})
# GRU 補正後の arm → (補正に使った g, 生成の g)
GRU_CALIB: dict[str, tuple[float, float]] = {"gru_cal": (1.0, 1.0), "gru_calg125": (1.25, 1.25)}
# 判定をかける候補
CANDIDATES: tuple[str, ...] = ("gru", "gru_no_slot_bias", "gru_cal", "gru_calg125")
# CFG の比較（種 42 のみ。判定には使わない）
GRU_CFG_COMPARE = 1.25

# 判定（計画書 §4.3）
C1_SD_RATIO = 0.5
C_MIN_ACTS = 4
GUARD_METRICS: tuple[str, ...] = ("switch_emd", "bigram_jsd", "gap_single_slot_ratio", "gap_wrap_closure_rate")

# 途中の ckpt（Q1 の軌跡）
EPOCH_POOL_N: int = rd.EPOCH_POOL_N              # 64 人/群（DDPM の H4 と同じ）
DDPM_TRAJ_ARM: str = rd.H4_ARM                   # clock_tf96_traj
DDPM_BEST_EPOCHS: dict[int, int] = {42: 769, 43: 784, 44: 737}   # 学習ログの最良 epoch（診断の報告 §4）
DDPM_WINDOW: int = rd.H4_WINDOW                  # 最良 epoch ± 200
GRU_WINDOW: int = gm.EARLY_STOP_PATIENCE         # 最良 epoch ± 30（早期終了が見た範囲）
# MSE の床: 完全なモデルの生成プール（ATUS 実から群ごとに POOL_N 本を引く）を作る回数
N_FLOOR_POOL = 50
MSE_METRICS: tuple[str, ...] = ("curve_mse_us", "curve_mse_jp", "group_mse_atus", "group_mse_atus_us",
                                "group_mse_atus_jp", "rate_mse_astar")
# Q4 の誤差の表と図に並べる arm（plot_curves と同じ）
CURVE_ERROR_ARMS: tuple[str, ...] = ("gru", "gru_calg125", "ddpm_tf96")
FLOOR_BAND_COLOR = "#6f6e69"                     # 床の帯の灰色（plot_stage1_curves.COLOR_STULA と同じ）
# 群別と CFG の強さ（判定には使わない）
GROUP_FLOOR_SIMS = 50                            # 群ごとの床の見積もりの反復回数
CFG_SWEEP: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
DDPM_TRAIN_G = 1.25                              # DDPM の学習時のプールの CFG の強さ
DDPM_G1_SEEDS: tuple[int, ...] = (42, 43, 44)    # DDPM の g = 1.0 のプールがある種（H7 で作成）
# 群ごとの表と図に並べる (arm, g)
GROUP_ARMS: tuple[tuple[str, float], ...] = (("gru", 1.0), ("gru_no_slot_bias", 1.0), ("gru_cal", 1.0),
                                             ("gru_calg125", 1.25), ("ddpm_tf96", 1.25))
# 群ごとの図に並べる (arm, g)（gru_no_slot_bias は表だけ。紫と赤紫が隣り合うため）
GROUP_PLOT_ARMS: tuple[tuple[str, float], ...] = (("gru", 1.0), ("gru_cal", 1.0), ("gru_calg125", 1.25),
                                                  ("ddpm_tf96", 1.25))
AGE_LABELS: tuple[str, ...] = ("15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")
# 小プールの生成の乱数だけによる揺れを測る種の数（GRU 種 42 の最良の ckpt で引き直す）
NOISE_DRAWS = 8


# ============================================================
# 読み込みと米国加重
# ============================================================
def pool_csv(arm: str, seed: int) -> Path:
    """arm と種の生成プール CSV（存在は確かめない）"""
    if arm == "gru":
        return gm.pool_path(seed)
    if arm == "gru_no_slot_bias":
        return gm.pool_path(seed, use_slot_bias=False)
    if arm == "gru_g1.25":
        return gm.pool_path(seed, GRU_CFG_COMPARE)
    if arm in GRU_CALIB:
        calib_g, sample_g = GRU_CALIB[arm]
        return gm.pool_path(seed, sample_g, calib_guidance=calib_g)
    return rep.pool_csv(DDPM_ARMS[arm], seed)


def ckpt_file(arm: str, seed: int) -> Path:
    """arm と種の最良の ckpt"""
    if arm in GRU_CALIB:
        return gm.ckpt_path(seed, calib_guidance=GRU_CALIB[arm][0])
    if arm == "gru_no_slot_bias":
        return gm.ckpt_path(seed, use_slot_bias=False)
    if arm.startswith("gru"):
        return gm.ckpt_path(seed)
    return rep.ckpt_path(DDPM_ARMS[arm], seed)


def load_pool(arm: str, seed: int) -> IntArr:
    """生成プールを読む（ckpt より古いプールは取り違えとして止める）, -> (28, M, 96)"""
    path = pool_csv(arm, seed)
    if not path.exists():
        raise FileNotFoundError(f"生成プールが無い ({arm}, seed={seed}): {path}")
    rep.check_fresh(path, ckpt_file(arm, seed))
    return np.asarray(cur.load_sample_pool(path), dtype=np.int64)


def us_weighted_slot_rates(rates: FloatArr, pi_d: FloatArr) -> FloatArr:
    """群別の時刻別行動者率を、群の重み pi_d で 1 本へ畳む, (28, 12, 96) -> (12, 96)

    Note:
        ★個票の無い群（NaN）は除いて pi_d を正規化し直す。ATUS 平日は 28 群とも個票がある

    Args:
        rates: 群別の時刻別行動者率, (28, 12, 96)。stage2_curves.pool_to_slot_rates か agr.group_rates
        pi_d: 群の重み, (28,)。米国加重は rd.group_weights("atus", ...)

    Returns:
        加重平均の時刻別行動者率, (12, 96)
    """
    ok = ~np.isnan(rates).any(axis=(1, 2))
    w = pi_d[ok] / pi_d[ok].sum()
    return np.asarray(np.einsum("d,dcs->cs", w, rates[ok]), dtype=np.float64)


def setup() -> tuple[People, FloatArr, pd.DataFrame]:
    """ATUS 実（model.load_data の行順）、米国加重の群の重み、ATUS 実の profile_table"""
    real: People = rd.model_order_people()
    pi_atus = np.asarray(rd.group_weights("atus", {}, real[1], real[2]), dtype=np.float64)
    return real, pi_atus, rd.profile_table(*real, pi_atus)


def _rows(df: pd.DataFrame, **eq: Any) -> pd.DataFrame:
    """列 = 値 の条件をすべて満たす行"""
    return rd._rows(df, **eq)


def _count(df: pd.DataFrame, col: str) -> int:
    """真偽値の列で True の数"""
    return int(np.count_nonzero(df[col].to_numpy(dtype=bool)))


def _show(title: str, df: pd.DataFrame) -> None:
    """表を小数 4 桁で表示する"""
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.float_format", "{:.4f}".format):
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))


# ============================================================
# Q1 総量（C1・C2）
# ============================================================
def totals_long(pi_atus: FloatArr, real_prof: pd.DataFrame) -> pd.DataFrame:
    """(arm, 種, 活動) ごとの総量と、その比（生成 / ATUS 実）

    Returns:
        列 arm / seed / activity / level / ratio / doer_share / slots_per_doer。
        gru_g1.25（種 42）はプールがあるときだけ入れる（判定には使わない）
    """
    runs = [(arm, seed) for arm, seeds in ARM_SEEDS.items() for seed in seeds]
    if pool_csv("gru_g1.25", 42).exists():
        runs.append(("gru_g1.25", 42))
    rows = []
    for arm, seed in runs:
        prof = rd.profile_table(*rd.pool_people(load_pool(arm, seed)), pi_atus)
        for a in FOCUS_ACTS:
            rows.append({"arm": arm, "seed": seed, "activity": a, "level": prof.loc[a, "level"],
                         "ratio": prof.loc[a, "level"] / real_prof.loc[a, "level"],
                         "doer_share": prof.loc[a, "doer_share"],
                         "slots_per_doer": prof.loc[a, "slots_per_doer"]})
    return pd.DataFrame(rows)


def totals_summary(long: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    """活動ごとに arm の種平均・種間 sd と、C1・C2 の活動ごとの判定

    Args:
        long: totals_long の戻り値
        floor: rd.bootstrap_floor の戻り値（米国加重）

    Returns:
        行 = 活動。{arm}_mean / {arm}_sd、floor（= sd × FLOOR_Z）、候補ごとの {候補}_c1_pass / {候補}_c2_pass
    """
    rows = []
    for a in FOCUS_ACTS:
        row: dict[str, Any] = {"activity": a}
        for arm in ARM_SEEDS:
            r = _rows(long, arm=arm, activity=a)["ratio"].to_numpy(dtype=np.float64)
            row[f"{arm}_mean"] = float(r.mean())
            row[f"{arm}_sd"] = float(r.std(ddof=1))
            row[f"{arm}_by_seed"] = "/".join(f"{v:.2f}" for v in r)
        row["floor"] = FLOOR_Z * float(floor.loc[a, "level"])
        for cand in CANDIDATES:
            row[f"{cand}_c1_pass"] = row[f"{cand}_sd"] <= C1_SD_RATIO * row["ddpm_tf96_sd"]
            row[f"{cand}_c2_pass"] = abs(row[f"{cand}_mean"] - 1.0) <= row["floor"]
        rows.append(row)
    return pd.DataFrame(rows)


def judge_totals(summary: pd.DataFrame, cand: str) -> dict[str, bool]:
    """候補 cand の C1・C2 の判定（5 活動のうち C_MIN_ACTS 以上で合格）"""
    return {"C1": _count(summary, f"{cand}_c1_pass") >= C_MIN_ACTS,
            "C2": _count(summary, f"{cand}_c2_pass") >= C_MIN_ACTS}


def plot_totals(long: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    """活動ごとに arm の総量の比を種の点で並べる（灰色の帯は床）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    arms = list(ARM_SEEDS)
    fig, axes = plt.subplots(1, len(FOCUS_ACTS), figsize=(22, 4.4))
    for ax, a in zip(axes, FOCUS_ACTS):
        lim = float(_rows(summary, activity=a)["floor"].iloc[0])
        ax.axhspan(1 - lim, 1 + lim, color="#d9d9d6", alpha=0.6, lw=0)
        ax.axhline(1.0, color=fm.COLOR_ATUS, lw=1.2)
        for k, arm in enumerate(arms):
            r = _rows(long, arm=arm, activity=a)["ratio"].to_numpy(dtype=np.float64)
            ax.plot(np.full(len(r), k), r, "o", color=ARM_COLOR[arm], ms=7, alpha=0.85,
                    markerfacecolor="white" if arm in ARM_HOLLOW else ARM_COLOR[arm],
                    markeredgecolor=ARM_COLOR[arm] if arm in ARM_HOLLOW else "white",
                    markeredgewidth=1.4 if arm in ARM_HOLLOW else 0.8, label=ARM_LABELS[arm])
            ax.plot([k - 0.25, k + 0.25], [r.mean()] * 2, color=ARM_COLOR[arm], lw=2.2)
        ax.set_xticks(range(len(arms)), [SHORT_LABELS[a] for a in arms], fontsize=7.5)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_title(f"{cur.ACT_JA[a]}（{a}）", fontsize=11)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("総量の比（生成 / 実）", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(labels),
               frameon=False, fontsize=10)
    fig.suptitle("少ない活動の総量（米国加重）", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gru_compare] 図: {out}")


# ============================================================
# Q1 途中の ckpt の軌跡
# ============================================================
def gru_epoch_ckpts(seed: int) -> dict[int, Path]:
    """GRU の途中の ckpt, epoch → パス"""
    best = gm.ckpt_path(seed)
    found: dict[int, Path] = {}
    for p in best.parent.glob(f"{best.stem}_ep[0-9][0-9][0-9][0-9].pt"):
        found[int(p.stem.rsplit("_ep", 1)[1])] = p
    return dict(sorted(found.items()))


def gru_epoch_pool_path(seed: int, epoch: int | None, gen_seed: int = gm.POOL_SEED) -> Path:
    """途中の ckpt（epoch=None なら最良の ckpt）から作った小プールの CSV のパス"""
    tag = "best" if epoch is None else f"ep{epoch:04d}"
    rng = "" if gen_seed == gm.POOL_SEED else f"_r{gen_seed}"
    stem = gm.GEN_SAVE_PATH.stem
    return gm.GEN_SAVE_PATH.with_name(f"{stem}{gm.run_suffix(seed)}_{tag}_n{EPOCH_POOL_N}{rng}.csv")


def make_small_pool(ckpt: Path, out: Path, gen_seed: int = gm.POOL_SEED) -> None:
    """ckpt から小プール（EPOCH_POOL_N 人/群、g = 1.0）を作る。ckpt より新しいものがあれば作らない"""
    if out.exists() and out.stat().st_mtime > ckpt.stat().st_mtime:
        return
    model = gm.load_model(ckpt)
    sm.write_pool_csv(gm.group_pool(model, EPOCH_POOL_N, gm.GUIDANCE_SCALE, seed=gen_seed), out)


def make_epoch_pools() -> None:
    """GRU の全種の途中の ckpt・最良の ckpt の小プールと、生成の乱数だけの揺れを測る小プールを作る"""
    for seed in ARM_SEEDS["gru"]:
        ckpts = gru_epoch_ckpts(seed)
        if not ckpts:
            raise FileNotFoundError(f"GRU の途中の ckpt が無い (seed={seed})")
        for epoch, path in ckpts.items():
            make_small_pool(path, gru_epoch_pool_path(seed, epoch))
        make_small_pool(gm.ckpt_path(seed), gru_epoch_pool_path(seed, None))
    for k in range(NOISE_DRAWS):
        make_small_pool(gm.ckpt_path(42), gru_epoch_pool_path(42, None, gm.POOL_SEED + 1 + k),
                        gm.POOL_SEED + 1 + k)


def _small_ratio(path: Path, pi_atus: FloatArr, real_prof: pd.DataFrame) -> dict[str, float]:
    """小プールの 5 活動の総量の比"""
    prof = rd.profile_table(*rd.pool_people(cur.load_sample_pool(path)), pi_atus)
    return {a: float(prof.loc[a, "level"] / real_prof.loc[a, "level"]) for a in FOCUS_ACTS}


def trajectory_long(pi_atus: FloatArr, real_prof: pd.DataFrame) -> pd.DataFrame:
    """GRU と DDPM（clock_tf96_traj）の途中の ckpt の小プールの総量の比

    Returns:
        列 model / seed / epoch / activity / ratio / is_best。is_best=True は最良の ckpt の小プールで、
        epoch は最良 epoch
    """
    gen = rd._gen()
    rows = []
    for seed in ARM_SEEDS["gru"]:
        best_epoch = int(torch.load(gm.ckpt_path(seed), map_location="cpu")["config"]["best_epoch"])
        pools = {e: gru_epoch_pool_path(seed, e) for e in gru_epoch_ckpts(seed)}
        for epoch, path in [*pools.items(), (None, gru_epoch_pool_path(seed, None))]:
            for a, r in _small_ratio(path, pi_atus, real_prof).items():
                rows.append({"model": "gru", "seed": seed, "epoch": best_epoch if epoch is None else epoch,
                             "activity": a, "ratio": r, "is_best": epoch is None})
    for seed, best_epoch in DDPM_BEST_EPOCHS.items():
        pools = gen.epoch_pools(DDPM_TRAJ_ARM, seed, EPOCH_POOL_N)
        if not pools:
            raise FileNotFoundError(f"DDPM の途中の ckpt の小プールが無い (seed={seed})")
        best = gen.epoch_pool_path(DDPM_TRAJ_ARM, seed, None, EPOCH_POOL_N)
        for epoch, path in [*pools.items(), (None, best)]:
            for a, r in _small_ratio(path, pi_atus, real_prof).items():
                rows.append({"model": "ddpm_tf96", "seed": seed, "epoch": best_epoch if epoch is None else epoch,
                             "activity": a, "ratio": r, "is_best": epoch is None})
    return pd.DataFrame(rows)


def trajectory_summary(long: pd.DataFrame, pi_atus: FloatArr, real_prof: pd.DataFrame) -> pd.DataFrame:
    """活動ごとに、最良 epoch 前後の種内 sd・最良の ckpt の種間 sd・生成の乱数だけの sd

    Note:
        ★窓は GRU が最良 epoch ± GRU_WINDOW（早期終了の patience）、DDPM が ± DDPM_WINDOW（診断の H4 と同じ）
        ★noise_sd は GRU 種 42 の最良の ckpt から、生成の乱数だけを変えて NOISE_DRAWS 本作った小プールの sd。
          種内 sd がこれと同程度なら、途中の ckpt の間で総量は動いていない

    Returns:
        列 activity / model / within_sd / between_sd / noise_sd
    """
    noise = [_small_ratio(gru_epoch_pool_path(42, None, gm.POOL_SEED + 1 + k), pi_atus, real_prof)
             for k in range(NOISE_DRAWS)]
    rows = []
    for a in FOCUS_ACTS:
        noise_sd = float(np.std([n[a] for n in noise], ddof=1))
        for model, window in (("gru", GRU_WINDOW), ("ddpm_tf96", DDPM_WINDOW)):
            g = _rows(long, model=model, activity=a)
            within = []
            for seed in sorted(set(int(s) for s in g["seed"])):
                gs = _rows(g, seed=seed)
                best_epoch = int(_rows(gs, is_best=True)["epoch"].iloc[0])
                mid = _rows(gs, is_best=False)
                near = np.abs(mid["epoch"].to_numpy() - best_epoch) <= window
                within.append(float(np.std(mid["ratio"].to_numpy()[near], ddof=1)))
            best_ratio = _rows(g, is_best=True)["ratio"].to_numpy(dtype=np.float64)
            rows.append({"activity": a, "model": model, "window": window,
                         "within_sd": float(np.mean(within)),
                         "within_sd_by_seed": "/".join(f"{v:.2f}" for v in within),
                         "between_sd": float(best_ratio.std(ddof=1)), "noise_sd": noise_sd})
    return pd.DataFrame(rows)


def plot_trajectory(long: pd.DataFrame, out: Path) -> None:
    """GRU の epoch と総量の比（種ごとの線、丸は最良の ckpt）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    fig, axes = plt.subplots(1, len(FOCUS_ACTS), figsize=(17, 3.9))
    for ax, a in zip(axes, FOCUS_ACTS):
        ax.axhline(1.0, color=fm.COLOR_ATUS, lw=1.2)
        for color, seed in zip(fm.ARM_COLORS, ARM_SEEDS["gru"]):
            g = _rows(long, model="gru", activity=a, seed=seed, is_best=False).sort_values("epoch")
            ax.plot(g["epoch"], g["ratio"], color=color, lw=1.6, label=f"種 {seed}")
            b = _rows(long, model="gru", activity=a, seed=seed, is_best=True)
            ax.plot(b["epoch"].to_numpy(), b["ratio"].to_numpy(), "o", color=color, ms=8,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set_title(f"{cur.ACT_JA[a]}（{a}）", fontsize=11)
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("総量の比（生成 / 実）", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(labels),
               frameon=False, fontsize=10)
    fig.suptitle("GRU の学習の途中の ckpt", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gru_compare] 図: {out}")


# ============================================================
# Q2 teacher forcing と自分で生成した総量
# ============================================================
def teacher_table(pi_atus: FloatArr) -> pd.DataFrame:
    """候補（CANDIDATES）の種ごとに、5 活動の総量を 4 通りで並べる

    Note:
        ★tf_train は §3.2 の一致の左辺（学習分割の履歴で条件付けた予測確率の加重平均）で、real_train と
          一致するはず。generated − tf_train が「自分の出力で履歴を作ったことによるずれ」（Q2）
        ★gru_cal / gru_calg125 は slot_bias を生成の総量に合わせたので、tf_train は real_train からずれる

    Returns:
        列 arm / seed / activity / real_train / tf_train / real_val / tf_val / generated / real_all /
        gen_minus_tf（generated − tf_train）/ gen_over_tf
    """
    train_part, val_part = gm.load_split()
    real_train = sm.population_rates(train_part.sched, train_part.weight).numpy().astype(np.float64)
    real_val = sm.population_rates(val_part.sched, val_part.weight).numpy().astype(np.float64)
    sched_r, d_r, w_r = rd.model_order_people()
    real_all = us_weighted_slot_rates(agr.group_rates(sched_r, d_r, w_r), pi_atus)
    rows = []
    for arm, seed in [(a, s) for a in CANDIDATES for s in ARM_SEEDS[a]]:
        model = gm.load_model(ckpt_file(arm, seed))
        tf_train = gm.teacher_forced_rates(model, train_part.sched, train_part.cond_idx, train_part.weight)
        tf_val = gm.teacher_forced_rates(model, val_part.sched, val_part.cond_idx, val_part.weight)
        generated = us_weighted_slot_rates(cur.pool_to_slot_rates(load_pool(arm, seed)), pi_atus)
        for a in FOCUS_ACTS:
            c = cur.ACT_NAMES.index(a)
            row = {"arm": arm, "seed": seed, "activity": a,
                   **{k: float(v[c].mean()) for k, v in (("real_train", real_train), ("tf_train", tf_train),
                                                         ("real_val", real_val), ("tf_val", tf_val),
                                                         ("generated", generated), ("real_all", real_all))}}
            row["gen_minus_tf"] = row["generated"] - row["tf_train"]
            row["gen_over_tf"] = row["generated"] / row["tf_train"]
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Q3 系列の妥当さ（C3）
# ============================================================
def guard_long(real: People) -> pd.DataFrame:
    """(arm, 種) ごとの GUARD_METRICS・切替回数・暗記

    Returns:
        列 arm / seed / switch_emd / bigram_jsd / gap_single_slot_ratio / gap_wrap_closure_rate /
        switch_mean / dcr_gap / memorized
    """
    sched_r, d_r, w_r = real
    frag_real = sel.im.fragmentation_summary(sched_r, w_r)
    rows = []
    for arm, seeds in ARM_SEEDS.items():
        for seed in seeds:
            pool = load_pool(arm, seed)
            d, m, s = pool.shape
            gen = pool.reshape(d * m, s)
            gen_d = np.repeat(np.arange(d), m)
            g = sel.guardrails(gen, gen_d, sched_r, d_r, w_r)
            with contextlib.redirect_stdout(io.StringIO()):        # memorization_report は表を print する
                mem = sm.memorization_report(gen, sched_r)
            rows.append({"arm": arm, "seed": seed,
                         "switch_emd": float(g["switch_emd"]), "bigram_jsd": float(g["bigram_jsd"]),
                         "gap_single_slot_ratio": abs(float(g["single_slot_ratio"]) - float(frag_real["single_slot_ratio"])),
                         "gap_wrap_closure_rate": abs(float(g["wrap_closure_rate"]) - float(frag_real["wrap_closure_rate"])),
                         "switch_mean": float(g["switch_mean"]),
                         "dcr_gap": float(mem["DCR_gap(holdout-train)"]), "memorized": bool(mem["memorized"])})
    return pd.DataFrame(rows)


def guard_summary(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    """指標ごとに arm の種の最大値を並べ、候補ごとに C3 を判定する

    Returns:
        (行 = 指標。{arm}_max / {arm}_mean / {候補}_pass（{候補}_max ≤ ddpm_noclock_max）,
         候補 → C3 の判定)
    """
    rows = []
    for k in GUARD_METRICS:
        row: dict[str, Any] = {"metric": k}
        for arm in ARM_SEEDS:
            v = _rows(long, arm=arm)[k].to_numpy(dtype=np.float64)
            row[f"{arm}_mean"] = float(v.mean())
            row[f"{arm}_max"] = float(v.max())
        for cand in CANDIDATES:
            row[f"{cand}_pass"] = row[f"{cand}_max"] <= row["ddpm_noclock_max"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    c3 = {cand: _count(summary, f"{cand}_pass") == len(GUARD_METRICS)
          and _count(_rows(long, arm=cand), "memorized") == 0 for cand in CANDIDATES}
    return summary, c3


# ============================================================
# Q4 12 活動の時刻別行動者率
# ============================================================
def arm_curves(pi_atus: FloatArr) -> dict[str, list[FloatArr]]:
    """arm → 種ごとの米国加重の時刻別行動者率 (12, 96) の並び"""
    return {arm: [us_weighted_slot_rates(cur.pool_to_slot_rates(load_pool(arm, s)), pi_atus) for s in seeds]
            for arm, seeds in ARM_SEEDS.items()}


def curve_table(curves: dict[str, list[FloatArr]], real_curve: FloatArr) -> pd.DataFrame:
    """活動ごとの ‖実 − 生成‖²（種平均と種の範囲）と、代表点の値

    Returns:
        列 activity / {arm}_err_mean / {arm}_err_by_seed。代表点（cur.KEY_SLOTS）は activity が
        "MEALS@12:00" の形の行で、値は種平均の行動者率（atus 列が実データ）
    """
    rows = []
    for c, a in enumerate(cur.ACT_NAMES):
        row: dict[str, Any] = {"activity": a}
        for arm, cs in curves.items():
            err = [float(((real_curve[c] - cv[c]) ** 2).sum()) for cv in cs]
            row[f"{arm}_err_mean"] = float(np.mean(err))
            row[f"{arm}_err_by_seed"] = "/".join(f"{v:.4f}" for v in err)
        rows.append(row)
    for act, hour in cur.KEY_SLOTS:
        c = cur.ACT_NAMES.index(act)
        s = int(round((hour - cur.SLOT_START_HOUR) * 60 / cur.SLOT_MINUTES))
        row = {"activity": f"{act}@{int(hour):02d}:00", "atus": float(real_curve[c, s])}
        for arm, cs in curves.items():
            row[f"{arm}_value_mean"] = float(np.mean([cv[c, s] for cv in cs]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(curves: dict[str, list[FloatArr]], real_curve: FloatArr, out: Path) -> None:
    """12 活動の米国加重の時刻別行動者率（実・GRU・GRU 補正後 g = 1.25・DDPM Transformer 型）

    Note:
        ★線は種平均、薄い帯は種の最小〜最大（種が 2 本以上のときだけ）。凡例に種の本数を出す
        ★活動ごとに y 軸を独立させる（睡眠は 1.0 近く、ボランティアは 0.01 未満）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    arms = [a for a in ("gru", "gru_calg125", "ddpm_tf96") if a in curves]
    stacks = [np.asarray(curves[a], dtype=np.float64) for a in arms]            # (種, 12, 96)
    labels = [f"{ARM_LABELS[a]}・種 {len(st)} 本" for a, st in zip(arms, stacks)]
    hours = cur.slot_hours()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.plot(hours, real_curve[c], color=fm.COLOR_ATUS, lw=2.6, label=fm.LABEL_ATUS, solid_capstyle="round")
        for arm, st, label in zip(arms, stacks, labels):
            if len(st) >= 2:
                ax.fill_between(hours, st[:, c].min(axis=0), st[:, c].max(axis=0),
                                color=ARM_COLOR[arm], alpha=0.14, lw=0)
            ax.plot(hours, st[:, c].mean(axis=0), color=ARM_COLOR[arm], lw=1.6, label=label,
                    solid_capstyle="round", solid_joinstyle="round")
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)
    fig.suptitle("12 活動の時刻別行動者率", fontsize=15, y=0.995)
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[gru_compare] 図: {out}")


def perfect_model_curves(real: People, pi_atus: FloatArr, n_seeds: int, resample_atus: bool,
                         n_draws: int = N_FLOOR_POOL, seed: int = 0) -> FloatArr:
    """完全なモデルの米国加重の時刻別行動者率を n_draws 回作る（Q4 の誤差の床）

    Note:
        ★1 回ぶん = 群ごとに gm.POOL_N 本を群内の TUFINLWGT で引いたプールを n_seeds 個作り、曲線を平均したもの
        ★resample_atus = False: ATUS 実からそのまま引く（下限側。mse_floors の floor_pool と同じ）
          resample_atus = True: 回ごとに ATUS の回答者を復元抽出してから引く（上限側。floor_real + floor_pool）
        ★モデルの誤差は種平均の曲線で測るので、床も同じ本数の種を平均する

    Args:
        real: ATUS 実 (sched, d, w)
        pi_atus: 群の重み, (28,)
        n_seeds: 平均する種の本数（比べる arm の種の本数）
        resample_atus: True なら回ごとに ATUS を復元抽出する
        n_draws: 作る回数
        seed: 乱数の種

    Returns:
        floor_curves, (n_draws, 12, 96)
    """
    sched, d, w = real
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        i = rng.integers(0, len(sched), len(sched)) if resample_atus else np.arange(len(sched))
        sched_b, d_b, w_b = sched[i], d[i], w[i]
        by_d = [np.flatnonzero(d_b == g) for g in range(sm.D_GROUPS)]
        seed_curves = []
        for _ in range(n_seeds):
            pool = np.stack([sched_b[rng.choice(ix, gm.POOL_N, p=w_b[ix] / w_b[ix].sum())] for ix in by_d])
            seed_curves.append(us_weighted_slot_rates(np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64),
                                                      pi_atus))
        out.append(np.mean(seed_curves, axis=0))
    return np.asarray(out, dtype=np.float64)


# 床の段の名前 → perfect_model_curves の resample_atus
FLOOR_LEVELS: dict[str, bool] = {"low": False, "high": True}
Floors = dict[str, dict[str, FloatArr]]          # 段の名前（low / high）→ sre.floor_errors の戻り値


def curve_error_tables(curves: dict[str, list[FloatArr]], real_curve: FloatArr, real: People,
                       pi_atus: FloatArr) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Floors]]:
    """arm ごとに活動ごとの誤差の表（MAE・最大誤差）と Top-k の表を作る

    Note:
        ★床は arm の種の本数ごとに作る（種 5 本と 3 本では種平均の雑音が違う）

    Returns:
        (列 arm + sre.activity_error_table の列, 列 arm + sre.topk_error_table の列,
         種の本数 → 2 段の床)
    """
    floors: dict[int, Floors] = {}
    act_rows, topk_rows = [], []
    for arm, cs in curves.items():
        n = len(cs)
        if n not in floors:
            floors[n] = {level: sre.floor_errors(real_curve, perfect_model_curves(real, pi_atus, n, resample))
                         for level, resample in FLOOR_LEVELS.items()}
        stack = np.asarray(cs, dtype=np.float64)
        lo, hi = floors[n]["low"], floors[n]["high"]
        act_rows.append(sre.activity_error_table(real_curve, stack, cur.ACT_NAMES, lo, hi).assign(arm=arm))
        topk_rows.append(sre.topk_error_table(real_curve, stack, cur.ACT_NAMES, lo, hi).assign(arm=arm))

    def _arm_first(df: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, df[["arm", *[c for c in df.columns if c != "arm"]]])
    return _arm_first(pd.concat(act_rows, ignore_index=True)), _arm_first(pd.concat(topk_rows, ignore_index=True)), floors


def plot_curve_errors(curves: dict[str, list[FloatArr]], real_curve: FloatArr, topk: pd.DataFrame,
                      floors: dict[int, Floors], out: Path, arms: tuple[str, ...] = CURVE_ERROR_ARMS,
                      band_seeds: int | None = None, colors: dict[str, str] | None = None,
                      labels: dict[str, str] | None = None, ref_name: str = "実") -> None:
    """12 活動の誤差（生成 − 実、pt）の曲線に、床の帯と Top-k の印を重ねる

    Note:
        ★線は種平均の曲線の誤差。灰色の帯は 15 分区間ごとの |誤差| の床（±、種 band_seeds 本の床。
          None なら floors のうち種の本数が最大のもの）。濃い帯 = 下限側、薄い帯 = 上限側
        ★丸印は arm ごとの Top-k の区間
        ★活動ごとに y 軸を独立させる
        ★colors / labels を渡さなければ ARM_COLOR / ARM_LABELS を使う。ref_name は縦軸の「生成 − {ref_name}」
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    colors = ARM_COLOR if colors is None else colors
    labels = ARM_LABELS if labels is None else labels
    arms = tuple(a for a in arms if a in curves)
    n_band = max(floors) if band_seeds is None else band_seeds
    band_lo, band_hi = floors[n_band]["low"]["slot_pt"], floors[n_band]["high"]["slot_pt"]   # (12, 96)
    hours = cur.slot_hours()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for c, name in enumerate(cur.ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.fill_between(hours, -band_hi[c], band_hi[c], color=FLOOR_BAND_COLOR, alpha=0.12, lw=0,
                        label=f"床の上限側（種 {n_band} 本）")
        ax.fill_between(hours, -band_lo[c], band_lo[c], color=FLOOR_BAND_COLOR, alpha=0.30, lw=0,
                        label=f"床の下限側（種 {n_band} 本）")
        ax.axhline(0.0, color=fm.COLOR_ATUS, lw=0.8)
        for arm in arms:
            err = sre.error_pt(real_curve, np.asarray(curves[arm], dtype=np.float64)).mean(axis=0)
            ax.plot(hours, err[c], color=colors[arm], lw=1.6, label=labels[arm],
                    solid_capstyle="round", solid_joinstyle="round")
            slots = _rows(topk, arm=arm, activity=name)["slot"].to_numpy(dtype=np.int64)
            ax.scatter(hours[slots], err[c, slots], s=36, color=colors[arm], edgecolors="white",
                       linewidths=1.2, zorder=3)
        fm.style_axis(ax, 4.0, 28.0, 4.0)
        ax.set_title(f"{cur.ACT_JA[name]}（{name}）", fontsize=11)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel(f"生成 − {ref_name}（pt）", fontsize=10)
    fig.suptitle("12 活動の時刻別行動者率の誤差", fontsize=15, y=0.995)
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[gru_compare] 図: {out}")


# ============================================================
# MSE を 6 通りの粒度で（判定には使わない）
# ============================================================
def mse_refs(real: People, pi_atus: FloatArr, tgt: dict) -> dict[str, FloatArr]:
    """ATUS 実の群別の値と、米国加重・日本人口加重の曲線（mse_values の基準）"""
    group = np.asarray(agr.group_rates(*real), dtype=np.float64)                # (28, 12, 96)
    return {"group": group, "us": us_weighted_slot_rates(group, pi_atus),
            "jp": np.asarray(cur.weighted_slot_rates(group, tgt), dtype=np.float64)}


def group_weighted_mse(err2: FloatArr, pi_d: FloatArr) -> float:
    """全セルの 2 乗誤差を、群の中で一様に平均してから群の重み pi_d で平均する

    Note:
        ★pi_d が一様なら group_mse_atus（全セルの一様平均）と一致する（どの群も 12 × 96 セル）
        ★NaN だけの群は除いて pi_d を正規化し直す（us_weighted_slot_rates と同じ扱い）

    Args:
        err2: セルごとの 2 乗誤差, (28, 12, 96)
        pi_d: 群の重み, (28,)

    Returns:
        群の重みで平均した MSE
    """
    per_group = np.nanmean(err2, axis=(1, 2))                                     # (28,)
    ok = ~np.isnan(per_group)
    return float(np.sum(pi_d[ok] * per_group[ok]) / pi_d[ok].sum())


def mse_values(rates: FloatArr, refs: dict[str, FloatArr], pi_atus: FloatArr, tgt: dict) -> dict[str, float]:
    """群別の時刻別行動者率 (28, 12, 96) の MSE を 6 通りで返す

    Note:
        ★group_mse_atus_us / _jp の重みは、curve_mse_us / _jp で群を 1 本に畳む重みと同じ
          （米国加重 pi_atus、日本人口加重 tgt["pop"] の比）
        ★rate_mse_astar は stage2_targets.eval_against と同じ（公表のあるセルの単純平均、mask_12act）。
          日米の生活の違いを含むので、Stage 1 の良し悪しの判定には使わない

    Args:
        rates: 群別の時刻別行動者率, (28, 12, 96)
        refs: mse_refs の戻り値
        pi_atus: 米国加重の群の重み, (28,)
        tgt: stage2_targets.load_stula_targets の戻り値

    Returns:
        MSE_METRICS の各値
    """
    err2 = (refs["group"] - rates) ** 2                                           # (28, 12, 96)
    pop = np.asarray(tgt["pop"], dtype=np.float64).reshape(sm.D_GROUPS)
    return {
        "curve_mse_us": float(np.mean((refs["us"] - us_weighted_slot_rates(rates, pi_atus)) ** 2)),
        "curve_mse_jp": float(np.mean((refs["jp"] - cur.weighted_slot_rates(rates, tgt)) ** 2)),
        "group_mse_atus": float(np.nanmean(err2)),
        "group_mse_atus_us": group_weighted_mse(err2, pi_atus),
        "group_mse_atus_jp": group_weighted_mse(err2, pop / pop.sum()),
        "rate_mse_astar": float(st.eval_against(rates.reshape(sm.D_GROUPS, -1), tgt, st.mask_12act())["rate_mse"]),
    }


def mse_floors(real: People, pi_atus: FloatArr, tgt: dict, refs: dict[str, FloatArr],
               seed: int = 0) -> pd.DataFrame:
    """完全なモデルでも出る MSE の 2 つの成分（rate_mse_astar 以外）

    Note:
        ★floor_real: ATUS の回答者を復元抽出して群別の値を作り直したときの MSE（実データ側の標本誤差）
        ★floor_pool: ATUS 実から群ごとに gm.POOL_N 本を群内の TUFINLWGT で引いたプール（= 完全なモデルの
          生成）の MSE（生成プールが有限であることによる揺れ）
        ★完全なモデルの MSE は floor_pool（学習データの揺れまで写したとき）〜 floor_real + floor_pool の間

    Returns:
        列 arm（floor_real / floor_pool）/ seed / MSE_METRICS（rate_mse_astar は NaN）
    """
    sched, d, w = real
    rng = np.random.default_rng(seed)
    by_d = [np.flatnonzero(d == g) for g in range(sm.D_GROUPS)]
    rows = []
    for b in range(rd.N_BOOT):
        i = rng.integers(0, len(sched), len(sched))
        rates = np.asarray(agr.group_rates(sched[i], d[i], w[i]), dtype=np.float64)
        rows.append({"arm": "floor_real", "seed": b, **mse_values(rates, refs, pi_atus, tgt)})
    for b in range(N_FLOOR_POOL):
        pool = np.stack([sched[rng.choice(ix, gm.POOL_N, p=w[ix] / w[ix].sum())] for ix in by_d])
        rates = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
        rows.append({"arm": "floor_pool", "seed": b, **mse_values(rates, refs, pi_atus, tgt)})
    out = pd.DataFrame(rows)
    out["rate_mse_astar"] = np.nan
    return out


def mse_long(real: People, pi_atus: FloatArr, tgt: dict) -> pd.DataFrame:
    """(arm, 種) ごとの 6 通りの MSE と、床・ATUS 実そのものの行

    Returns:
        列 arm / seed / MSE_METRICS。arm = atus_real の行は ATUS 実そのもの（rate_mse_astar だけが意味を持つ）
    """
    refs = mse_refs(real, pi_atus, tgt)
    rows = []
    for arm, seeds in ARM_SEEDS.items():
        for seed in seeds:
            rates = np.asarray(cur.pool_to_slot_rates(load_pool(arm, seed)), dtype=np.float64)
            rows.append({"arm": arm, "seed": seed, **mse_values(rates, refs, pi_atus, tgt)})
    rows.append({"arm": "atus_real", "seed": -1, **mse_values(refs["group"], refs, pi_atus, tgt)})
    return pd.concat([pd.DataFrame(rows), mse_floors(real, pi_atus, tgt, refs)], ignore_index=True)


def mse_summary(long: pd.DataFrame) -> pd.DataFrame:
    """arm ごとに MSE_METRICS の平均・最小・最大（行 = arm × 指標）"""
    rows = []
    for arm in dict.fromkeys(str(a) for a in long["arm"]):
        g = _rows(long, arm=arm)
        for k in MSE_METRICS:
            v = g[k].to_numpy(dtype=np.float64)
            if np.isnan(v).all():
                continue
            rows.append({"arm": arm, "metric": k, "n": len(v), "mean": float(np.nanmean(v)),
                         "min": float(np.nanmin(v)), "max": float(np.nanmax(v))})
    return pd.DataFrame(rows)


# ============================================================
# 群別と CFG の強さ（判定には使わない）
# ============================================================
def group_label(d: int) -> str:
    """群 d の名前（例: 男25-34有業）。sm.cond_grid の (性, 年齢 7 区分, 就業)"""
    g, a, e = (int(v) for v in sm.cond_grid()[d])
    return f"{'男' if g == 0 else '女'}{AGE_LABELS[a]}{'有業' if e == 1 else '無業'}"


def weighted_rates(sched: IntArr, w: FloatArr) -> FloatArr:
    """個票の加重の時刻別行動者率, (N, 96) -> (12, 96)"""
    onehot = sched[:, None, :] == np.arange(sm.NUM_ACT)[None, :, None]            # (N, 12, 96)
    return np.asarray(np.einsum("n,ncs->cs", w / w.sum(), onehot), dtype=np.float64)


def group_floor(real: People, n_sims: int = GROUP_FLOOR_SIMS, seed: int = 0) -> FloatArr:
    """群ごとの MSE の床 (28,) = 実データ側の標本誤差 + 生成プール（gm.POOL_N 本）の揺れ

    Note:
        ★群の中で回答者を復元抽出した群別の値（人数は群の人数のまま）と、群の中から TUFINLWGT で
          gm.POOL_N 本を引いたプール（完全なモデルの生成）の値を、それぞれ ATUS 実の群別の値と比べた
          MSE の平均の和。完全なモデルでも、ATUS 実と比べればこの程度の MSE が出る
        ★回答者の少ない群ほど床は大きい（ATUS 平日の群の人数は 17〜329 人）

    Returns:
        群ごとの床, (28,)
    """
    sched, d, w = real
    atus = np.asarray(agr.group_rates(sched, d, w), dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.zeros(sm.D_GROUPS, dtype=np.float64)
    for g in range(sm.D_GROUPS):
        ix = np.flatnonzero(d == g)
        acc = 0.0
        for _ in range(n_sims):
            i = rng.choice(ix, len(ix), replace=True)
            acc += float(np.mean((weighted_rates(sched[i], w[i]) - atus[g]) ** 2))
            j = rng.choice(ix, gm.POOL_N, p=w[ix] / w[ix].sum())
            acc += float(np.mean((weighted_rates(sched[j], np.ones(len(j))) - atus[g]) ** 2))
        out[g] = acc / n_sims
    return out


def separation_reference(real: People, n_sims: int = 20, seed: int = 0) -> FloatArr:
    """完全なモデル（ATUS 実から群ごとに gm.POOL_N 本を TUFINLWGT で引いたプール）の separation_ratio, (n_sims,)

    Note:
        ★ATUS の群の人数は少ないので、群の間の距離には標本の揺れが乗る。完全なモデルのプールも同じ回答者から
          引くので同じ揺れを持ち、比はほぼ 1 になる（2026-09-27 の実測で 0.99〜1.06）。生成の比はこの値と比べる
    """
    sched, d, w = real
    rng = np.random.default_rng(seed)
    by_d = [np.flatnonzero(d == g) for g in range(sm.D_GROUPS)]
    gen_d = np.repeat(np.arange(sm.D_GROUPS), gm.POOL_N)
    w_gen = sel.im.group_reweight(gen_d, w, d, sm.D_GROUPS)
    out = []
    for _ in range(n_sims):
        pool = np.stack([sched[rng.choice(ix, gm.POOL_N, p=w[ix] / w[ix].sum())] for ix in by_d])
        out.append(float(sel.cd.separation_summary(sched, pool.reshape(-1, sm.NUM_SLOTS), d, gen_d, sm.NUM_ACT,
                                                   sm.D_GROUPS, w, w_gen)["separation_ratio"]))
    return np.asarray(out, dtype=np.float64)


def pool_csv_g(arm: str, seed: int, g: float) -> Path:
    """arm・種・CFG の強さ g の生成プール CSV（存在は確かめない）"""
    if arm == "gru":
        return gm.pool_path(seed, g)
    if arm == "gru_no_slot_bias":
        return gm.pool_path(seed, g, use_slot_bias=False)
    if arm in GRU_CALIB:
        return gm.pool_path(seed, g, calib_guidance=GRU_CALIB[arm][0])
    label = DDPM_ARMS[arm] if g == DDPM_TRAIN_G else f"{DDPM_ARMS[arm]}@g{g:g}"
    return rep.pool_csv(label, seed)


def load_pool_g(arm: str, seed: int, g: float) -> IntArr:
    """pool_csv_g のプールを読む（ckpt より古いプールは取り違えとして止める）, -> (28, M, 96)"""
    path = pool_csv_g(arm, seed, g)
    if not path.exists():
        raise FileNotFoundError(f"生成プールが無い ({arm}, seed={seed}, g={g:g}): {path}")
    rep.check_fresh(path, ckpt_file(arm, seed))
    return np.asarray(cur.load_sample_pool(path), dtype=np.int64)


def group_runs() -> list[tuple[str, float, int]]:
    """群別・CFG の部で読む (arm, g, 種) の並び"""
    runs = [("gru", g, s) for g in (1.0, GRU_CFG_COMPARE) for s in ARM_SEEDS["gru"]]
    runs += [("gru_no_slot_bias", g, s) for g in (1.0, GRU_CFG_COMPARE) for s in ARM_SEEDS["gru_no_slot_bias"]]
    runs += [("gru_cal", g, s) for g in CFG_SWEEP for s in ARM_SEEDS["gru_cal"]]
    runs += [("gru_calg125", 1.25, s) for s in ARM_SEEDS["gru_calg125"]]
    runs += [("ddpm_tf96", DDPM_TRAIN_G, s) for s in ARM_SEEDS["ddpm_tf96"]]
    runs += [("ddpm_tf96", 1.0, s) for s in DDPM_G1_SEEDS]
    return runs


def make_cfg_pools() -> None:
    """gru_cal の g ≠ 1.0 のプール（28 群 × gm.POOL_N 本、種 gm.POOL_SEED）を作る。ckpt より新しければ作らない"""
    for g in CFG_SWEEP:
        if g == gm.GUIDANCE_SCALE:
            continue
        for seed in ARM_SEEDS["gru_cal"]:
            ckpt = gm.ckpt_path(seed, calib_guidance=1.0)
            out = gm.pool_path(seed, g, calib_guidance=1.0)
            if out.exists() and out.stat().st_mtime > ckpt.stat().st_mtime:
                continue
            gm.write_pool(gm.load_model(ckpt), out, g)


def group_long(real: People, pi_atus: FloatArr, real_prof: pd.DataFrame,
               floor: FloatArr) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(arm, g, 種) ごとの群別の MSE と、群別・総量・切替の要約

    Returns:
        (群ごとの縦持ち: arm / g / seed / group / mse / floor / ratio,
         (arm, g, 種) ごと: group_mse_mean / ratio_median / separation_ratio / spearman_pairs /
         curve_mse_us / rare_abs_dev（5 活動の |総量の比 − 1| の平均）/ switch_emd)
    """
    sched_r, d_r, w_r = real
    atus = np.asarray(agr.group_rates(sched_r, d_r, w_r), dtype=np.float64)
    ref_us = us_weighted_slot_rates(atus, pi_atus)
    rows, runs = [], []
    for arm, g, seed in group_runs():
        pool = load_pool_g(arm, seed, g)
        rates = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
        mse = ((rates - atus) ** 2).mean(axis=(1, 2))                            # (28,)
        for dd in range(sm.D_GROUPS):
            rows.append({"arm": arm, "g": g, "seed": seed, "group": dd, "mse": float(mse[dd]),
                         "floor": float(floor[dd]), "ratio": float(mse[dd] / floor[dd])})
        n_d, m, n_s = pool.shape
        gen = pool.reshape(n_d * m, n_s)
        gen_d = np.repeat(np.arange(n_d), m)
        w_gen = sel.im.group_reweight(gen_d, w_r, d_r, sm.D_GROUPS)
        sep = sel.cd.separation_summary(sched_r, gen, d_r, gen_d, sm.NUM_ACT, sm.D_GROUPS, w_r, w_gen)
        prof = rd.profile_table(*rd.pool_people(pool), pi_atus)
        ratio = prof["level"].to_numpy(dtype=np.float64) / real_prof["level"].to_numpy(dtype=np.float64)
        runs.append({"arm": arm, "g": g, "seed": seed, "group_mse_mean": float(mse.mean()),
                     "ratio_median": float(np.median(mse / floor)),
                     "separation_ratio": float(sep["separation_ratio"]),
                     "spearman_pairs": float(sep["spearman_pairs"]),
                     "curve_mse_us": float(np.mean((ref_us - us_weighted_slot_rates(rates, pi_atus)) ** 2)),
                     "rare_abs_dev": float(np.mean(np.abs(ratio - 1.0))),
                     "switch_emd": float(sel.im.switch_dist_compare(sched_r, gen, w_r, w_gen)["emd"])})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def _by_g(df: pd.DataFrame, metric: str) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
    """列 g の値ごとに metric の (g, 平均, 最小, 最大)"""
    g_all = df["g"].to_numpy(dtype=np.float64)
    gs = np.unique(g_all)
    vals = [df[metric].to_numpy(dtype=np.float64)[g_all == g] for g in gs]
    return (gs, np.array([v.mean() for v in vals]), np.array([v.min() for v in vals]),
            np.array([v.max() for v in vals]))


def group_table(long: pd.DataFrame, real: People, floor: FloatArr) -> pd.DataFrame:
    """群ごとに、ATUS の人数・床・主な arm の MSE（種平均）と床との比

    Returns:
        行 = 群。列 group / label / n_atus / floor / {arm@g}_mse / {arm@g}_ratio
    """
    n_atus = np.bincount(real[1], minlength=sm.D_GROUPS)
    out = pd.DataFrame({"group": np.arange(sm.D_GROUPS), "label": [group_label(dd) for dd in range(sm.D_GROUPS)],
                        "n_atus": n_atus, "floor": floor})
    for arm, g in GROUP_ARMS:
        sub = _rows(long, arm=arm, g=g)
        idx = sub["group"].to_numpy(dtype=np.int64)
        m = (np.bincount(idx, weights=sub["mse"].to_numpy(dtype=np.float64), minlength=sm.D_GROUPS)
             / np.bincount(idx, minlength=sm.D_GROUPS))
        out[f"{arm}@g{g:g}_mse"] = m
        out[f"{arm}@g{g:g}_ratio"] = m / floor
    return out


def cfg_table(runs: pd.DataFrame) -> pd.DataFrame:
    """(arm, g) ごとに要約の種平均・最小・最大"""
    metrics = ["group_mse_mean", "ratio_median", "separation_ratio", "spearman_pairs", "curve_mse_us",
               "rare_abs_dev", "switch_emd"]
    agg = runs.groupby(["arm", "g"], sort=False)[metrics].agg(["mean", "min", "max"])
    agg.columns = [f"{m}_{k}" for m, k in agg.columns]
    return agg.reset_index()


def plot_groups(table: pd.DataFrame, out: Path) -> None:
    """群ごとの MSE / 床（種平均）。横軸は群、灰色の線は床（比 = 1）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    fig, ax = plt.subplots(figsize=(15, 5.2))
    xs = np.arange(len(table))
    ax.axhline(1.0, color="#8f8f8b", lw=1.2)
    for k, (arm, g) in enumerate(GROUP_PLOT_ARMS):
        ax.plot(xs + (k - 1.5) * 0.18, table[f"{arm}@g{g:g}_ratio"], "o", color=ARM_COLOR[arm], ms=7,
                markerfacecolor="white" if arm in ARM_HOLLOW else ARM_COLOR[arm],
                markeredgecolor=ARM_COLOR[arm] if arm in ARM_HOLLOW else "white",
                markeredgewidth=1.4 if arm in ARM_HOLLOW else 0.8, label=ARM_LABELS[arm])
    ax.set_xticks(xs, [f"{lab}（{n}）" for lab, n in zip(table["label"], table["n_atus"])],
                  rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("群ごとの MSE / 床", fontsize=10)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(GROUP_PLOT_ARMS), frameon=False, fontsize=10)
    fig.suptitle("群ごとの MSE", y=1.02, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gru_compare] 図: {out}")


def plot_cfg(runs: pd.DataFrame, sep_ref: FloatArr, out: Path) -> None:
    """CFG の強さ g と指標（gru_cal は種平均の線と種の範囲、DDPM Transformer 型は種平均の点）

    Args:
        runs: group_long の 2 つ目の戻り値
        sep_ref: separation_reference の戻り値（群の分離のパネルに灰色の帯で描く）
        out: 保存先
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    panels = [("group_mse_mean", "群ごとの MSE（28 群の平均）"), ("separation_ratio", "群の分離（生成 / 実）"),
              ("rare_abs_dev", "少ない活動の |総量の比 − 1|"), ("switch_emd", "切替回数の分布の距離")]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.9))
    for ax, (metric, title) in zip(axes, panels):
        gs, mean, lo, hi = _by_g(_rows(runs, arm="gru_cal"), metric)
        ax.fill_between(gs, lo, hi, color=ARM_COLOR["gru_cal"], alpha=0.18, lw=0)
        ax.plot(gs, mean, "-o", color=ARM_COLOR["gru_cal"], lw=2, ms=6, markerfacecolor="white",
                markeredgewidth=1.4, label="GRU 補正後（g=1.0 で補正）")
        g125 = _rows(runs, arm="gru_calg125")[metric].to_numpy(dtype=np.float64)
        ax.plot([1.25], [g125.mean()], "D", color=ARM_COLOR["gru_calg125"], ms=8, markeredgecolor="white",
                label="GRU 補正後（g=1.25 で補正）")
        gd, mean_d, _, _ = _by_g(_rows(runs, arm="ddpm_tf96"), metric)
        ax.plot(gd, mean_d, "s", color=ARM_COLOR["ddpm_tf96"], ms=7, label="DDPM Transformer 型")
        if metric == "separation_ratio":
            ax.axhspan(float(sep_ref.min()), float(sep_ref.max()), color="#d9d9d6", alpha=0.8, lw=0,
                       label="完全なモデル")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("CFG の強さ g", fontsize=9)
        ax.set_xticks(list(CFG_SWEEP))
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(labels),
               frameon=False, fontsize=10)
    fig.suptitle("CFG の強さ", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gru_compare] 図: {out}")


# ============================================================
# 部の実行
# ============================================================
def run_totals() -> None:
    """Q1 総量と C1・C2"""
    real, pi_atus, real_prof = setup()
    floor = rd.bootstrap_floor(real, pi_atus)
    long = totals_long(pi_atus, real_prof)
    summary = totals_summary(long, floor)
    verdicts = {cand: judge_totals(summary, cand) for cand in CANDIDATES}
    _show("ATUS 実（米国加重）", real_prof.reset_index(names="activity"))
    _show("Q1 総量の比（生成 / 実）の種平均・種間 sd と C1・C2", summary)
    g125 = _rows(long, arm="gru_g1.25")
    if len(g125):
        _show("参考: GRU 種 42 の g = 1.25（判定に使わない）", g125)
    for cand, verdict in verdicts.items():
        print(f"\n[{cand}] C1（sd ≤ DDPM × {C1_SD_RATIO}、{_count(summary, f'{cand}_c1_pass')}/5 活動）: "
              f"{'合格' if verdict['C1'] else '不合格'}")
        print(f"[{cand}] C2（|比 − 1| ≤ 床、{_count(summary, f'{cand}_c2_pass')}/5 活動）: "
              f"{'合格' if verdict['C2'] else '不合格'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_csv("totals_long"), index=False)
    summary.to_csv(out_csv("totals"), index=False)
    plot_totals(long, summary, out_fig("totals"))


def run_trajectory() -> None:
    """Q1 途中の ckpt の軌跡"""
    _, pi_atus, real_prof = setup()
    make_epoch_pools()
    long = trajectory_long(pi_atus, real_prof)
    summary = trajectory_summary(long, pi_atus, real_prof)
    _show("Q1 途中の ckpt の総量の比（64 人/群）", summary)
    long.to_csv(out_csv("trajectory_long"), index=False)
    summary.to_csv(out_csv("trajectory"), index=False)
    plot_trajectory(long, out_fig("trajectory"))


def run_teacher() -> None:
    """Q2 teacher forcing と自分で生成した総量"""
    _, pi_atus, _ = setup()
    table = teacher_table(pi_atus)
    _show("Q2 総量: 実データの履歴で条件付けた予測（tf）と自分で生成した値", table)
    agg = table.groupby(["arm", "activity"], sort=False)[["gen_minus_tf", "gen_over_tf"]].agg(["mean", "std"])
    print(agg.to_string())
    table.to_csv(out_csv("teacher"), index=False)


def run_guard() -> None:
    """Q3 系列の妥当さと C3"""
    real, _, _ = setup()
    long = guard_long(real)
    summary, c3 = guard_summary(long)
    _show("Q3 ガードレール（種ごと）", long)
    _show("Q3 種の最大値と C3", summary)
    for cand, ok in c3.items():
        print(f"[{cand}] C3（4 指標で候補の最大 ≤ noclock の最大、暗記 0 本）: {'合格' if ok else '不合格'}")
    long.to_csv(out_csv("guard_long"), index=False)
    summary.to_csv(out_csv("guard"), index=False)


def run_curves() -> None:
    """Q4 12 活動の時刻別行動者率"""
    real, pi_atus, _ = setup()
    real_curve = us_weighted_slot_rates(agr.group_rates(*real), pi_atus)
    curves = arm_curves(pi_atus)
    table = curve_table(curves, real_curve)
    _show("Q4 ‖実 − 生成‖²（活動ごと）と代表点", table)
    table.to_csv(out_csv("curves"), index=False)
    plot_curves(curves, real_curve, out_fig("curves"))
    errors, topk, floors = curve_error_tables(curves, real_curve, real, pi_atus)
    _show("Q4 活動ごとの MAE・最大誤差（pt、種平均の曲線）と床", errors)
    _show(f"Q4 活動ごとの Top-{sre.TOP_K} の 15 分区間（pt）", topk)
    errors.to_csv(out_csv("curve_errors"), index=False)
    topk.to_csv(out_csv("curve_topk"), index=False)
    plot_curve_errors(curves, real_curve, topk, floors, out_fig("curve_errors"))


def run_mse() -> None:
    """MSE を 6 通りの粒度で並べる"""
    real, pi_atus, _ = setup()
    tgt = cur.st.load_stula_targets()
    long = mse_long(real, pi_atus, tgt)
    summary = mse_summary(long)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3e}".format):
        print("\n=== MSE（6 通りの粒度）===")
        print(summary.to_string(index=False))
    for k in (m for m in MSE_METRICS if m != "rate_mse_astar"):          # 床は ATUS 実と比べる指標だけ
        lo = float(_rows(summary, arm="floor_pool", metric=k)["mean"].iloc[0])
        hi = lo + float(_rows(summary, arm="floor_real", metric=k)["mean"].iloc[0])
        print(f"完全なモデルの {k}: {lo:.3e} 〜 {hi:.3e}")
    long.to_csv(out_csv("mse_long"), index=False)
    summary.to_csv(out_csv("mse"), index=False)


def run_groups_and_cfg() -> None:
    """群別と CFG の強さ（groups と cfg は同じ読み込みを共有するので 1 回で両方を出す）"""
    real, pi_atus, real_prof = setup()
    make_cfg_pools()
    floor = group_floor(real)
    long, runs = group_long(real, pi_atus, real_prof, floor)
    table = group_table(long, real, floor)
    cfg = cfg_table(runs)
    sep_ref = separation_reference(real)
    with pd.option_context("display.width", 240, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print("\n=== 群ごとの MSE（種平均）と床 ===")
        print(table.to_string(index=False))
        print("\n=== CFG の強さ（(arm, g) ごとの種平均・最小・最大）===")
        print(cfg.to_string(index=False))
    print(f"完全なモデルの separation_ratio: 平均 {sep_ref.mean():.3f}（{sep_ref.min():.3f}〜{sep_ref.max():.3f}）")
    long.to_csv(out_csv("groups_long"), index=False)
    table.to_csv(out_csv("groups"), index=False)
    runs.to_csv(out_csv("cfg_long"), index=False)
    cfg.to_csv(out_csv("cfg"), index=False)
    plot_groups(table, out_fig("groups"))
    plot_cfg(runs, sep_ref, out_fig("cfg"))


def out_csv(name: str) -> Path:
    """表の出力先 data/processed/aggregates/stage1_gru_{name}{OUT_TAG}.csv"""
    return OUT_DIR / f"stage1_gru_{name}{OUT_TAG}.csv"


def out_fig(name: str) -> Path:
    """図の出力先 src/models/GRU_Aggregate/figures/stage1_gru_{name}{OUT_TAG}.png"""
    return FIG_DIR / f"stage1_gru_{name}{OUT_TAG}.png"


PARTS = {"totals": run_totals, "trajectory": run_trajectory, "teacher": run_teacher,
         "guard": run_guard, "curves": run_curves, "mse": run_mse,
         "groups": run_groups_and_cfg}


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GRU_Aggregate と DDPM の Stage 1 の比較（米国加重）")
    ap.add_argument("--part", choices=[*PARTS, "all"], required=True)
    ap.add_argument("--gru-seeds", type=int, nargs="+", default=None,
                    help="GRU の種を絞る（学習が揃う前の途中経過用。判定 C1〜C3 は既定の 5 本で出す）")
    ap.add_argument("--tag", default="", help="出力の表と図のファイル名の接尾辞（例: _h256）。既定は付けない")
    args = ap.parse_args()
    global OUT_TAG
    OUT_TAG = args.tag
    if args.gru_seeds is not None:
        ARM_SEEDS["gru"] = tuple(args.gru_seeds)
    for name, fn in PARTS.items():
        if args.part in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
