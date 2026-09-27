"""
stage1_gru_compare.py
=====================
GRU_Aggregate（再帰型＋交差エントロピー）と DDPM の Stage 1 を、米国加重で並べて判定する
（計画書 src/models/GRU_Aggregate/docs/plan.md §4）。

★評価は米国加重: 生成の群別の値を ATUS の TUFINLWGT の群構成（group_weights("atus")）で平均する。
  Stage 1 の目的は元分布（ATUS）の再現なので、日本人口加重は使わない。

比べるもの（ARM_SEEDS）:

    gru           本計画のモデル（g = 1.0）                       種 42〜46
    gru_cal       gru の slot_bias を学習後に補正（計画書 §9）   種 42〜46
    ddpm_tf96     clock_tf96（学習時のプール、g = 1.25）          種 42〜46
    ddpm_noclock  時刻符号なしの DDPM（ガードレールの外側の基準）  種 42〜44

部（--part）と問い:

    totals      Q1 5 活動の総量の比（生成 / ATUS 実）と種間 sd → 判定 C1・C2
    trajectory  Q1 途中の ckpt の小プール（64 人/群）の総量の比。DDPM（clock_tf96_traj）と同じ大きさ
    teacher     Q2 実データの履歴で条件付けた総量（teacher_forced_rates）と、自分で生成した総量の差
    guard       Q3 switch_emd / bigram_jsd / single_slot の差 / wrap_closure の差・暗記 → 判定 C3
    curves      Q4 12 活動の時刻別行動者率、‖実 − 生成‖²、代表点の値
    mse         MSE を 4 通りの粒度で並べる（判定には使わない）
                curve_mse_us / curve_mse_jp : 28 群を米国加重 / 日本人口加重で 1 本にした曲線 vs ATUS 実
                group_mse_atus              : 28 群ごとの値 vs ATUS 実（セルの単純平均）
                rate_mse_astar              : 28 群ごとの値 vs 日本の教師 A*（論文の rate_mse と同じ定義）
                床: 完全なモデルでも出る MSE は floor_pool 〜 floor_real + floor_pool の間

判定（計画書 §4.3。結果を見る前に固定）。候補 CANDIDATES（gru / gru_cal）のそれぞれにかける:

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
    LP --> MS["mse_values<br/>4 通りの MSE"]
    REAL --> MF["mse_floors<br/>floor_real / floor_pool"]
    MF --> MS
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_gru_compare.py --part all
    .venv/bin/python src/eval/diagnostics/stage1_gru_compare.py --part totals
    .venv/bin/python src/eval/diagnostics/stage1_gru_compare.py --part curves --gru-seeds 42   # 途中経過

出力: data/processed/aggregates/stage1_gru_{totals_long,totals,trajectory_long,trajectory,teacher,guard_long,
      guard,curves,mse_long,mse}.csv と src/models/GRU_Aggregate/figures/stage1_gru_{totals,trajectory,curves}.png
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
FIG_DIR = REPO_ROOT / "src" / "models" / "GRU_Aggregate" / "figures"

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
    "gru_cal": (42, 43, 44, 45, 46),
    "ddpm_tf96": (42, 43, 44, 45, 46),
    "ddpm_noclock": (42, 43, 44),
}
DDPM_ARMS: dict[str, str] = {"ddpm_tf96": "clock_tf96", "ddpm_noclock": "noclock"}
ARM_LABELS: dict[str, str] = {
    "gru": "GRU（g=1.0）",
    "gru_g1.25": "GRU（g=1.25）",
    "gru_cal": "GRU 補正後（g=1.0）",
    "ddpm_tf96": "DDPM Transformer 型（g=1.25）",
    "ddpm_noclock": "DDPM 時刻符号なし（g=1.25）",
}
# 図の横軸の短い名前
SHORT_LABELS: dict[str, str] = {"gru": "GRU", "gru_cal": "GRU\n補正後", "ddpm_tf96": "DDPM\nTransformer\n型",
                                "ddpm_noclock": "DDPM\n時刻符号\nなし"}
# ★色は arm ごとに固定する（図によって系列の数が違っても同じ arm は同じ色）。
#   gru / ddpm_tf96 / ddpm_noclock は stage1_ablation_curves.ARM_COLORS の 1〜3 番目と同じ。
#   gru_cal の赤紫は、この 4 色の並びで validate_palette（色覚の検査）を通った色
ARM_COLOR: dict[str, str] = {"gru": "#2a78d6", "gru_cal": "#b5179e", "ddpm_tf96": "#eb6834",
                             "ddpm_noclock": "#1baf7a"}
# 判定をかける候補
CANDIDATES: tuple[str, ...] = ("gru", "gru_cal")
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
MSE_METRICS: tuple[str, ...] = ("curve_mse_us", "curve_mse_jp", "group_mse_atus", "rate_mse_astar")
# 小プールの生成の乱数だけによる揺れを測る種の数（GRU 種 42 の最良の ckpt で引き直す）
NOISE_DRAWS = 8


# ============================================================
# 読み込みと米国加重
# ============================================================
def pool_csv(arm: str, seed: int) -> Path:
    """arm と種の生成プール CSV（存在は確かめない）"""
    if arm == "gru":
        return gm.pool_path(seed)
    if arm == "gru_g1.25":
        return gm.pool_path(seed, GRU_CFG_COMPARE)
    if arm == "gru_cal":
        return gm.pool_path(seed, calibrated=True)
    return rep.pool_csv(DDPM_ARMS[arm], seed)


def ckpt_file(arm: str, seed: int) -> Path:
    """arm と種の最良の ckpt"""
    if arm.startswith("gru"):
        return gm.ckpt_path(seed, calibrated=arm == "gru_cal")
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
                    markeredgecolor="white", markeredgewidth=0.8, label=ARM_LABELS[arm])
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
    """候補（gru / gru_cal）の種ごとに、5 活動の総量を 4 通りで並べる

    Note:
        ★tf_train は §3.2 の一致の左辺（学習分割の履歴で条件付けた予測確率の加重平均）で、real_train と
          一致するはず。generated − tf_train が「自分の出力で履歴を作ったことによるずれ」（Q2）
        ★gru_cal は slot_bias を生成の総量に合わせたので、tf_train は real_train からずれる

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
    """12 活動の米国加重の時刻別行動者率（実・GRU・GRU 補正後・DDPM Transformer 型）

    Note:
        ★線は種平均、薄い帯は種の最小〜最大（種が 2 本以上のときだけ）。凡例に種の本数を出す
        ★活動ごとに y 軸を独立させる（睡眠は 1.0 近く、ボランティアは 0.01 未満）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = rd._figure_module()
    fm.setup_fonts()
    arms = [a for a in ("gru", "gru_cal", "ddpm_tf96") if a in curves]
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


# ============================================================
# MSE を 4 通りの粒度で（判定には使わない）
# ============================================================
def mse_refs(real: People, pi_atus: FloatArr, tgt: dict) -> dict[str, FloatArr]:
    """ATUS 実の群別の値と、米国加重・日本人口加重の曲線（mse_values の基準）"""
    group = np.asarray(agr.group_rates(*real), dtype=np.float64)                # (28, 12, 96)
    return {"group": group, "us": us_weighted_slot_rates(group, pi_atus),
            "jp": np.asarray(cur.weighted_slot_rates(group, tgt), dtype=np.float64)}


def mse_values(rates: FloatArr, refs: dict[str, FloatArr], pi_atus: FloatArr, tgt: dict) -> dict[str, float]:
    """群別の時刻別行動者率 (28, 12, 96) の MSE を 4 通りで返す

    Note:
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
    return {
        "curve_mse_us": float(np.mean((refs["us"] - us_weighted_slot_rates(rates, pi_atus)) ** 2)),
        "curve_mse_jp": float(np.mean((refs["jp"] - cur.weighted_slot_rates(rates, tgt)) ** 2)),
        "group_mse_atus": float(np.nanmean((refs["group"] - rates) ** 2)),
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
    """(arm, 種) ごとの 4 通りの MSE と、床・ATUS 実そのものの行

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
    long.to_csv(OUT_DIR / "stage1_gru_totals_long.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_gru_totals.csv", index=False)
    plot_totals(long, summary, FIG_DIR / "stage1_gru_totals.png")


def run_trajectory() -> None:
    """Q1 途中の ckpt の軌跡"""
    _, pi_atus, real_prof = setup()
    make_epoch_pools()
    long = trajectory_long(pi_atus, real_prof)
    summary = trajectory_summary(long, pi_atus, real_prof)
    _show("Q1 途中の ckpt の総量の比（64 人/群）", summary)
    long.to_csv(OUT_DIR / "stage1_gru_trajectory_long.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_gru_trajectory.csv", index=False)
    plot_trajectory(long, FIG_DIR / "stage1_gru_trajectory.png")


def run_teacher() -> None:
    """Q2 teacher forcing と自分で生成した総量"""
    _, pi_atus, _ = setup()
    table = teacher_table(pi_atus)
    _show("Q2 総量: 実データの履歴で条件付けた予測（tf）と自分で生成した値", table)
    agg = table.groupby(["arm", "activity"], sort=False)[["gen_minus_tf", "gen_over_tf"]].agg(["mean", "std"])
    print(agg.to_string())
    table.to_csv(OUT_DIR / "stage1_gru_teacher.csv", index=False)


def run_guard() -> None:
    """Q3 系列の妥当さと C3"""
    real, _, _ = setup()
    long = guard_long(real)
    summary, c3 = guard_summary(long)
    _show("Q3 ガードレール（種ごと）", long)
    _show("Q3 種の最大値と C3", summary)
    for cand, ok in c3.items():
        print(f"[{cand}] C3（4 指標で候補の最大 ≤ noclock の最大、暗記 0 本）: {'合格' if ok else '不合格'}")
    long.to_csv(OUT_DIR / "stage1_gru_guard_long.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_gru_guard.csv", index=False)


def run_curves() -> None:
    """Q4 12 活動の時刻別行動者率"""
    real, pi_atus, _ = setup()
    real_curve = us_weighted_slot_rates(agr.group_rates(*real), pi_atus)
    curves = arm_curves(pi_atus)
    table = curve_table(curves, real_curve)
    _show("Q4 ‖実 − 生成‖²（活動ごと）と代表点", table)
    table.to_csv(OUT_DIR / "stage1_gru_curves.csv", index=False)
    plot_curves(curves, real_curve, FIG_DIR / "stage1_gru_curves.png")


def run_mse() -> None:
    """MSE を 4 通りの粒度で並べる"""
    real, pi_atus, _ = setup()
    tgt = cur.st.load_stula_targets()
    long = mse_long(real, pi_atus, tgt)
    summary = mse_summary(long)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3e}".format):
        print("\n=== MSE（4 通りの粒度）===")
        print(summary.to_string(index=False))
    for k in MSE_METRICS[:3]:
        lo = float(_rows(summary, arm="floor_pool", metric=k)["mean"].iloc[0])
        hi = lo + float(_rows(summary, arm="floor_real", metric=k)["mean"].iloc[0])
        print(f"完全なモデルの {k}: {lo:.3e} 〜 {hi:.3e}")
    long.to_csv(OUT_DIR / "stage1_gru_mse_long.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_gru_mse.csv", index=False)


PARTS = {"totals": run_totals, "trajectory": run_trajectory, "teacher": run_teacher,
         "guard": run_guard, "curves": run_curves, "mse": run_mse}


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GRU_Aggregate と DDPM の Stage 1 の比較（米国加重）")
    ap.add_argument("--part", choices=[*PARTS, "all"], required=True)
    ap.add_argument("--gru-seeds", type=int, nargs="+", default=None,
                    help="GRU の種を絞る（学習が揃う前の途中経過用。判定 C1〜C3 は既定の 5 本で出す）")
    args = ap.parse_args()
    if args.gru_seeds is not None:
        ARM_SEEDS["gru"] = tuple(args.gru_seeds)
    for name, fn in PARTS.items():
        if args.part in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
