"""
stage1_clock_replicates.py
==========================
Stage 1 のアブレーション（arm）を、学習の種の反復で読む。

arm どうしは構造や損失だけでなく学習の乱数列も違う（追加した層の初期化が乱数を消費するため）。
1 本ずつの比較では「arm の効果」と「種のばらつき」を分けられない。そこで各 arm を同じ種の組
（既定 42 / 43 / 44）で学習し、指標ごとに arm の平均・標準偏差と、基準の arm と完全に
分離したかを出す。

arm の中身と保存先の名前は src/models/DDPM_Aggregate_Simple/stage1_arms.py の ARMS が唯一の出所。

    arm ラベル       生成プール（model.sanity_check が書く CSV, 群一様 28 群 × 256 本）
    ARM              ddpm_simple_pretrain_samples{arm_suffix(ARM, seed)}.csv
    ARM@gS           ddpm_simple_pretrain_samples{arm_suffix(ARM, seed)}_g{S}.csv
                     （stage1_guidance_pool.py が CFG の強さ S で作り直したプール）

指標はどれも Stage 1 の目標である **ATUS 実データ**との距離として読む（日本の教師 A* ではない）:

    key_*        代表スロットの値（日本人口加重）。stage2_curves.KEY_SLOTS と同じ点
    key_*_employed / _nonemployed
                 同じ点を有業 / 無業の群だけで人口加重した値（群ごとの時刻の形を見る）
    gap_*        key_* と ATUS 実データの差の絶対値。小さいほど良い
    err_*        ‖ATUS実 − 生成‖²（日本人口加重の 12×96 曲線）を周期の帯域で分けた値
    guardrail    断片化など系列の妥当性。stage2_select.guardrails と同じ定義
    dcr_gap      暗記チェック（model.memorization_report）。memorized=1 は床の外（上）
    params       チェックポイントのパラメータ数

★判定について。各 arm 3 本しかないので検定の検出力は低い。ここでは
  「2 つの arm の値の範囲が重ならない（完全分離）」を最も強い読みとし、3 対 3 の完全分離が
  偶然に起きる確率は片側 1/20 = 0.05 である。平均差だけを根拠にしないこと。

★段の勝者（--judge）の規則は計画書で事前に固定したもの:
    1. ガードレール（GUARD_METRICS）のどれかで、候補の 3 本が全て基準の 3 本より悪い
       （完全分離で悪化）か、暗記の疑い（memorized）が 1 本でもあれば失格
    2. 失格でない arm（基準を含む）の中で、主指標（PRIMARY_METRICS）の平均の順位和が最小の arm
    3. 順位和が並んだらパラメータの少ない arm

データフロー:

```mermaid
flowchart TD
    ARMS["stage1_arms.ARMS / arm_suffix<br/>(arm, seed) → プールと ckpt の名前"] --> CSV
    CSV["生成プール CSV"] --> LP["stage2_curves.load_sample_pool<br/>pool (28, 256, 96)"]
    LP --> PR["pool_to_slot_rates<br/>rates (28, 12, 96)"]
    PR --> CUR["weighted_slot_rates<br/>curve (12, 96) 日本人口加重"]
    PR --> EMP["employment_slot_value<br/>有業 / 無業の key_*"]
    ATUS["atus_real_curve / group_rates<br/>ATUS 実"] --> REF["ref (ATUS 実の値)"]
    CUR --> KEY["key_* / err_*"]
    LP --> GR["stage2_select.guardrails"]
    LP --> MEM["model.memorization_report<br/>dcr_gap / memorized"]
    KEY --> LONG["縦持ち (arm, seed, metric, value)"]
    EMP --> LONG
    GR --> LONG
    MEM --> LONG
    REF --> GAP["add_gaps: gap_* = |値 − ATUS 実|"]
    LONG --> GAP --> SUM["summarize<br/>arm の平均・sd・基準との完全分離"]
    SUM --> JUDGE["judge<br/>ガードレール → 主指標の順位和 → パラメータ数"]
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_clock_replicates.py
    .venv/bin/python src/eval/diagnostics/stage1_clock_replicates.py \\
        --arms noclock clock clock_h12 clock_h48 clock_tf96 --base clock --judge --tag stage1
    .venv/bin/python src/eval/diagnostics/stage1_clock_replicates.py \\
        --arms clock@g1.25 clock@g1 --base clock@g1.25 --tag guidance
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
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

# 報告するガードレール。stage2_select.guardrails の戻り値のうち系列の形を見るもの
GUARDRAIL_KEYS = ("switch_mean", "single_slot_ratio", "wrap_closure_rate", "bigram_jsd",
                  "switch_emd", "travel_single_rate", "night_intrusion_rate")
# 有業 / 無業に分けて読む代表点 (活動名, スロット開始時刻)
EMPLOYMENT_KEY_SLOTS: list[tuple[str, float]] = [("WORK", 12.0), ("MEALS", 12.0)]
# gap_* を作る値（ATUS 実データとの差の絶対値を取る）
GAP_SOURCES = ("switch_mean", "single_slot_ratio", "wrap_closure_rate")

# 段の勝者の判定（計画書で事前に固定）。どれも小さいほど良い
PRIMARY_METRICS = ("gap_MEALS_12:00", "gap_WORK_12:00", "err_period_under_3.4h",
                   "gap_WORK_12:00_employed", "gap_WORK_12:00_nonemployed")
GUARD_METRICS = ("switch_emd", "bigram_jsd", "gap_single_slot_ratio", "gap_wrap_closure_rate")

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage2_select.py と同じ規則）。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cur: Any = _load("replicates_stage2_curves", SIMPLE_DIR / "stage2_curves.py")
sel: Any = _load("simple_stage2_select", SIMPLE_DIR / "stage2_select.py")
sm: Any = sel.sm
arms: Any = _load("simple_stage1_arms", SIMPLE_DIR / "stage1_arms.py")
agr: Any = _load("curves_atus_group_rates", REPO_ROOT / "src" / "eval" / "atus_group_rates.py")


def parse_label(label: str) -> tuple[str, float | None]:
    """arm ラベル "ARM" / "ARM@gS" を (arm 名, guidance) に分ける

    Args:
        label: arm ラベル。@gS は stage1_guidance_pool.py が CFG の強さ S で作り直したプール

    Returns:
        (arm 名, guidance)。@gS が無ければ guidance は None（学習時の sanity_check のプール）

    Raises:
        KeyError: 未知の arm 名
        ValueError: @ の後ろが gS の形でない
    """
    name, _, tag = label.partition("@")
    if name not in arms.ARMS:
        raise KeyError(f"未知の arm: {name}（既知: {sorted(arms.ARMS)}）")
    if not tag:
        return name, None
    if not tag.startswith("g"):
        raise ValueError(f"arm ラベルの @ の後ろは gS（例 @g1）: {label}")
    return name, float(tag[1:])


def pool_csv(label: str, seed: int) -> Path:
    """(arm ラベル, seed) の生成プール CSV のパスを返す（存在は確かめない）"""
    name, guidance = parse_label(label)
    tag = "" if guidance is None else f"_g{guidance:g}"
    stem = sm.GEN_SAVE_PATH.stem
    return sm.GEN_SAVE_PATH.with_name(f"{stem}{arms.arm_suffix(name, seed)}{tag}.csv")


def ckpt_path(label: str, seed: int) -> Path:
    """(arm ラベル, seed) の Stage 1 チェックポイントのパスを返す（存在は確かめない）"""
    name, _ = parse_label(label)
    stem = sm.MODEL_SAVE_PATH.stem
    return sm.MODEL_SAVE_PATH.with_name(f"{stem}{arms.arm_suffix(name, seed)}.pt")


def check_fresh(pool: Path, ckpt: Path) -> None:
    """生成プールが ckpt より古くないことを確かめる（取り違えたプールで評価しない）

    Note:
        ★学習時のプールは ckpt を保存した直後に書かれる。ckpt の方が新しければ、
          プールは前の学習の残りである。scp は -p で mtime を保って持ち帰ること

    Raises:
        RuntimeError: ckpt の方がプールより新しいとき
    """
    if ckpt.exists() and ckpt.stat().st_mtime > pool.stat().st_mtime + 1.0:
        raise RuntimeError(f"生成プールが ckpt より古い（取り違えの疑い）: {pool.name} < {ckpt.name}")


def count_params(ckpt: Path) -> float:
    """チェックポイントのパラメータ数（ckpt が無ければ NaN）"""
    if not ckpt.exists():
        return float("nan")
    state = torch.load(ckpt, map_location="cpu")["model"]
    return float(sum(v.numel() for v in state.values()))


def employment_slot_value(rates: FloatArr, pi_d: FloatArr, act: str, hour: float,
                          employed: int) -> float:
    """有業 (1) / 無業 (0) の群だけを人口加重して、1 点の行動者率を返す

    Args:
        rates: 群別の時刻別行動者率, (28, 12, 96)
        pi_d: 群の人口の割合, (28,)
        act: 活動名 (stage2_curves.ACT_NAMES)
        hour: スロットの開始時刻（04:00 起点の時刻）
        employed: 1 なら有業の群、0 なら無業の群

    Returns:
        人口加重した行動者率
    """
    c = cur.ACT_NAMES.index(act)
    s = int(round((hour - cur.SLOT_START_HOUR) * 60 / cur.SLOT_MINUTES))
    d = np.arange(len(pi_d))
    sel_d = (d % sm.N_E) == employed           # d = g·(N_A·N_E) + a·N_E + e
    w = pi_d[sel_d] / pi_d[sel_d].sum()
    return float((w * rates[sel_d, c, s]).sum())


def employment_metrics(rates: FloatArr, pi_d: FloatArr) -> dict[str, float]:
    """EMPLOYMENT_KEY_SLOTS の各点を有業 / 無業に分けた値"""
    out: dict[str, float] = {}
    for act, hour in EMPLOYMENT_KEY_SLOTS:
        time = f"{int(hour):02d}:{int(hour % 1 * 60):02d}"
        for employed, tag in ((1, "employed"), (0, "nonemployed")):
            out[f"key_{act}_{time}_{tag}"] = employment_slot_value(rates, pi_d, act, hour, employed)
    return out


def replicate_metrics(pool: npt.NDArray[np.int64], atus: FloatArr, tgt: dict, pi_d: FloatArr,
                      sched_real: npt.NDArray[np.int64], d_real: npt.NDArray[np.int64],
                      w_real: FloatArr) -> dict[str, float]:
    """1 本の生成プールから key_* / err_* / ガードレール / 暗記チェックを測る。

    Args:
        pool: 群別サンプルプール, dtype=int64, (28, M, 96)
        atus: ATUS 実データの日本人口加重曲線, (12, 96)
        tgt: stage2_targets.load_stula_targets の戻り値（人口重み）
        pi_d: 群の人口の割合, (28,)
        sched_real: 実 ATUS 平日のスケジュール, (N, 96)
        d_real: 実 ATUS の群インデックス, (N,)
        w_real: 実 ATUS の調査ウェイト, (N,)

    Returns:
        指標名 -> 値
    """
    rates = cur.pool_to_slot_rates(pool)
    curve = cur.weighted_slot_rates(rates, tgt)
    out: dict[str, float] = {}

    key = cur.key_slot_table({"gen": curve})
    for _, row in key.iterrows():
        out[f"key_{row['activity']}_{row['time']}"] = float(row["gen"])
    out.update(employment_metrics(rates, pi_d))

    resid = atus - curve
    out["err_all"] = float((resid ** 2).sum())
    for band, (lo, hi) in cur.BANDS.items():
        out[f"err_{band}"] = float((cur.band_component(resid, lo, hi) ** 2).sum())
    meals = cur.ACT_NAMES.index("MEALS")
    out["err_MEALS"] = float((resid[meals] ** 2).sum())

    d, m, s = pool.shape
    gen = pool.reshape(d * m, s)
    gen_d = np.repeat(np.arange(d), m)
    g = sel.guardrails(gen, gen_d, sched_real, d_real, w_real)
    out.update({k: float(g[k]) for k in GUARDRAIL_KEYS})

    with contextlib.redirect_stdout(io.StringIO()):       # memorization_report は表を print する
        mem = sm.memorization_report(gen, sched_real)
    out["dcr_gap"] = float(mem["DCR_gap(holdout-train)"])
    out["memorized"] = float(bool(mem["memorized"]))
    return out


def reference_values(atus: FloatArr, pi_d: FloatArr, sched_real: npt.NDArray[np.int64],
                     w_real: FloatArr) -> dict[str, float]:
    """ATUS 実データでの値（key_* と断片化の指標）を返す"""
    ref: dict[str, float] = {}
    for _, row in cur.key_slot_table({"atus": atus}).iterrows():
        ref[f"key_{row['activity']}_{row['time']}"] = float(row["atus"])
    sched_w, groups_w, w_w = agr.load_atus_weekday()
    ref.update(employment_metrics(agr.group_rates(sched_w, groups_w, w_w), pi_d))
    frag = sel.im.fragmentation_summary(sched_real, w_real)
    ref.update({k: float(frag[k]) for k in GAP_SOURCES})
    return ref


def add_gaps(long: pd.DataFrame, ref: dict[str, float]) -> pd.DataFrame:
    """key_* と GAP_SOURCES について gap_* = |値 − ATUS 実| の行を足す

    Args:
        long: 列 arm / seed / metric / value の縦持ち
        ref: ATUS 実データでの値

    Returns:
        gap_* の行を足した縦持ち
    """
    keys = [k for k in ref if k.startswith("key_") or k in GAP_SOURCES]
    src = long.loc[long["metric"].isin(keys).to_numpy()]
    names = [str(m) for m in src["metric"]]
    gaps = pd.DataFrame({
        "arm": src["arm"].to_numpy(), "seed": src["seed"].to_numpy(),
        "metric": ["gap_" + m.removeprefix("key_") for m in names],
        "value": np.abs(src["value"].to_numpy(dtype=np.float64) - np.array([ref[m] for m in names])),
        "pool": src["pool"].to_numpy(),
    })
    return pd.concat([long, gaps], ignore_index=True)


def summarize(long: pd.DataFrame, base: str) -> pd.DataFrame:
    """指標 × arm ごとに平均・sd と、基準 arm との差・完全分離の有無を返す。

    Args:
        long: 列 arm / seed / metric / value の縦持ち
        base: 基準の arm ラベル

    Returns:
        列 metric / arm / n / mean / sd / diff_vs_base / separated_vs_base。
        separated_vs_base は 2 つの arm の値の範囲が重ならないとき True（各 2 本以上のときだけ判定）
    """
    rows: list[dict[str, Any]] = []
    for metric, grp in long.groupby("metric", sort=False):
        b = grp.loc[grp["arm"] == base, "value"].to_numpy(dtype=np.float64)
        for arm, sub in grp.groupby("arm", sort=False):
            a = sub["value"].to_numpy(dtype=np.float64)
            rows.append({
                "metric": metric, "arm": arm, "n": len(a),
                "mean": a.mean(), "sd": a.std(ddof=1) if len(a) > 1 else np.nan,
                "diff_vs_base": a.mean() - b.mean() if len(b) else np.nan,
                "separated_vs_base": bool(arm != base and len(a) >= 2 and len(b) >= 2
                                          and (a.max() < b.min() or b.max() < a.min())),
            })
    return pd.DataFrame(rows)


def judge(long: pd.DataFrame, base: str, candidates: list[str]) -> pd.DataFrame:
    """段の勝者を事前に固定した規則で決める（モジュールの docstring の 1〜3）。

    Args:
        long: gap_* を足した縦持ち
        base: 基準の arm ラベル（ガードレールの比較相手。自身も勝者の候補に入る）
        candidates: 候補の arm ラベル

    Returns:
        arm ごとの行。列 arm / guard_ok / guard_fail / rank_sum / params / winner
    """
    def values(arm: str, metric: str) -> FloatArr:
        sel_rows = (long["arm"] == arm) & (long["metric"] == metric)
        return long.loc[sel_rows, "value"].to_numpy(dtype=np.float64)

    labels = [base, *candidates]
    guard_ok: list[bool] = []
    guard_fail: list[str] = []
    for arm in labels:
        fails = [m for m in GUARD_METRICS
                 if arm != base and len(values(arm, m)) >= 2
                 and values(arm, m).min() > values(base, m).max()]
        if values(arm, "memorized").max(initial=0.0) > 0.0:
            fails.append("memorized")
        guard_ok.append(not fails)
        guard_fail.append(",".join(fails))
    params = [float(np.nanmean(values(a, "params"))) for a in labels]
    # 主指標ごとに、失格でない arm の平均へ順位（小さいほど良い、同値は平均順位）を付けて足す
    means = {m: np.array([values(a, m).mean() for a in labels]) for m in PRIMARY_METRICS}
    ok_idx = [i for i, ok in enumerate(guard_ok) if ok]
    rank_sum = np.full(len(labels), np.nan)
    rank_sum[ok_idx] = 0.0
    for m in PRIMARY_METRICS:
        ranks = pd.Series(means[m][ok_idx]).rank(method="average").to_numpy(dtype=np.float64)
        rank_sum[ok_idx] += ranks
    best = min(ok_idx, key=lambda i: (rank_sum[i], params[i]))
    table = pd.DataFrame({"arm": labels, "guard_ok": guard_ok, "guard_fail": guard_fail,
                          "params": params, "rank_sum": rank_sum,
                          "winner": [i == best for i in range(len(labels))]})
    for m in PRIMARY_METRICS:
        table[f"mean_{m}"] = means[m]
    return table


def mean_sd_table(summ: pd.DataFrame, ref: dict[str, float], labels: list[str],
                  metrics: list[str]) -> pd.DataFrame:
    """表示用: 行 = 指標、列 = ATUS 実と各 arm の "平均±sd"（基準と完全分離なら * を付ける）"""
    out = pd.DataFrame(index=metrics)
    out["atus_real"] = [f"{ref[m]:.4f}" if m in ref else "" for m in metrics]
    for arm in labels:
        cells = []
        for m in metrics:
            r = summ[(summ["metric"] == m) & (summ["arm"] == arm)]
            if r.empty:
                cells.append("")
                continue
            row = r.iloc[0]
            star = "*" if row["separated_vs_base"] else ""
            cells.append(f"{row['mean']:.4f}±{row['sd']:.4f}{star}")
        out[arm] = cells
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["noclock", "clock"],
                    help="arm ラベル（ARM か ARM@gS）。既定は時刻符号なし / あり")
    ap.add_argument("--base", default=None, help="基準の arm ラベル。既定は --arms の先頭")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--judge", action="store_true",
                    help="段の勝者を事前に固定した規則で決める（候補は基準と noclock 以外の arm）")
    ap.add_argument("--tag", default="clock",
                    help="出力 CSV の名前 stage1_replicates_{tag}_{long,summary,judge}.csv")
    args = ap.parse_args()
    base = args.base or args.arms[0]
    if base not in args.arms:
        raise ValueError(f"--base {base} が --arms に無い")

    tgt = cur.st.load_stula_targets()
    atus = cur.atus_real_curve(tgt)
    pop = np.asarray(tgt["pop"], dtype=np.float64).reshape(-1)
    pi_d = pop / pop.sum()
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)

    records: list[dict[str, Any]] = []
    for label in args.arms:
        for seed in args.seeds:
            path = pool_csv(label, seed)
            if not path.exists():
                raise FileNotFoundError(f"生成プールが無い ({label}, seed={seed}): {path}")
            ckpt = ckpt_path(label, seed)
            if parse_label(label)[1] is None:
                check_fresh(path, ckpt)
            metrics = replicate_metrics(cur.load_sample_pool(path), atus, tgt, pi_d,
                                        sched_real, d_real, w_real)
            metrics["params"] = count_params(ckpt)
            records += [{"arm": label, "seed": seed, "metric": k, "value": v,
                         "pool": path.name} for k, v in metrics.items()]
            print(f"[replicates] {label:22s} seed={seed}  <- {path.name}", flush=True)

    ref = reference_values(atus, pi_d, sched_real, w_real)
    long = add_gaps(pd.DataFrame(records), ref)
    summ = summarize(long, base)

    show = [m for m in dict.fromkeys(long["metric"]) if m.startswith(("key_", "err_period", "gap_"))
            or m in GUARDRAIL_KEYS or m in ("err_all", "err_MEALS", "dcr_gap", "memorized", "params")]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 200, "display.max_colwidth", 40):
        print(f"\n=== arm の平均±sd（* = 基準 {base} と完全分離）===")
        print(mean_sd_table(summ, ref, args.arms, show).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_DIR / f"stage1_replicates_{args.tag}_long.csv", index=False)
    summ.to_csv(OUT_DIR / f"stage1_replicates_{args.tag}_summary.csv", index=False)
    print(f"\n[replicates] 書いた: stage1_replicates_{args.tag}_{{long,summary}}.csv")

    if args.judge:
        candidates = [a for a in args.arms if a not in (base, "noclock")]
        table = judge(long, base, candidates)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(f"\n=== 段の判定（基準 {base}）===")
            print(table.round(5).to_string(index=False))
        table.to_csv(OUT_DIR / f"stage1_replicates_{args.tag}_judge.csv", index=False)
        print(f"[replicates] 勝者: {table.loc[table['winner'], 'arm'].iloc[0]}")


if __name__ == "__main__":
    main()
