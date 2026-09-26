"""
stage1_clock_replicates.py
==========================
Stage 1 の時刻符号アブレーションを、学習の種の反復で読む。

時刻符号つき（`model.py --clock`）と時刻符号なしの Stage 1 は、時刻符号の有無だけでなく
学習の乱数列も違う（時刻符号の nn.Linear が初期化で乱数を消費するため）。1 本ずつの比較では
「時刻符号の効果」と「種のばらつき」を分けられない。そこで両群を同じ種の組
（既定 42 / 43 / 44）で学習し、指標ごとに群の平均・標準偏差と、2 群が完全に分離したかを出す。

    arm      seed  生成プール（model.sanity_check が書く CSV, 群一様 28 群 × 256 本）
    noclock  42    ddpm_simple_pretrain_samples_20260819.csv   本編（kernel k=3 の再学習と一致済み）
    noclock  S     ddpm_simple_pretrain_samples_s{S}.csv
    clock    42    ddpm_simple_pretrain_samples_clock.csv
    clock    S     ddpm_simple_pretrain_samples_clock_s{S}.csv

指標は 3 種類で、どれも Stage 1 の目標である **ATUS 実データ**との距離として読む
（日本の教師 A* ではない）:

    key_*     代表スロットの値（日本人口加重）。stage2_curves.KEY_SLOTS と同じ点
    err_*     ‖ATUS実 − 生成‖²（日本人口加重の 12×96 曲線）を周期の帯域で分けた値。
              帯域は stage2_curves.BANDS。小さいほど良い
    guardrail 断片化など系列の妥当性。stage2_select.guardrails と同じ定義
              （実 ATUS 平日の群構成へ重み付け）

★判定について。各群 3 本しかないので検定の検出力は低い。ここでは
  「2 群の値の範囲が重ならない（完全分離）」を最も強い読みとし、3 対 3 の完全分離が
  偶然に起きる確率は片側 1/20 = 0.05 である。平均差だけを根拠にしないこと。

データフロー:

```mermaid
flowchart TD
    CSV["生成プール CSV<br/>(arm, seed) ごとに 1 本"] --> LP["stage2_curves.load_sample_pool<br/>pool (28, 256, 96)"]
    LP --> PR["pool_to_slot_rates → weighted_slot_rates<br/>curve (12, 96) 日本人口加重"]
    ATUS["atus_real_curve<br/>ATUS 実 (12, 96) 日本人口加重"] --> ERR
    PR --> KEY["key_slot_table<br/>key_*"]
    PR --> ERR["band_component(ATUS − curve)<br/>err_* 帯域別の二乗和"]
    LP --> GR["stage2_select.guardrails<br/>switch_mean / single_slot_ratio / wrap_closure_rate / bigram_jsd …"]
    KEY --> LONG["縦持ち (arm, seed, metric, value)"]
    ERR --> LONG
    GR --> LONG
    LONG --> SUM["summarize<br/>群の平均・sd・完全分離"]
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_clock_replicates.py
    .venv/bin/python src/eval/diagnostics/stage1_clock_replicates.py --seeds 42 43
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"
GEN_DIR = REPO_ROOT / "outputs" / "generated"
OUT_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "stage1_clock_replicates.csv"

# 本編の時刻符号なし・種42。kernel スイープの k=3 がこのプールの指標を完全再現している
BASELINE_SEED = 42
BASELINE_NOCLOCK_CSV = GEN_DIR / "ddpm_simple_pretrain_samples_20260819.csv"

# 報告するガードレール。stage2_select.guardrails の戻り値のうち系列の形を見るもの
GUARDRAIL_KEYS = ("switch_mean", "single_slot_ratio", "wrap_closure_rate", "bigram_jsd",
                  "switch_emd", "travel_single_rate", "night_intrusion_rate")

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


def pool_csv(arm: str, seed: int) -> Path:
    """(arm, seed) の生成プール CSV のパスを返す。model.py の保存先の規則と同じ。

    Args:
        arm: "noclock" か "clock"
        seed: 学習の種

    Returns:
        生成プール CSV のパス（存在するかは確かめない）
    """
    if arm == "noclock" and seed == BASELINE_SEED:
        return BASELINE_NOCLOCK_CSV
    suffix = ("_clock" if arm == "clock" else "") + ("" if seed == BASELINE_SEED else f"_s{seed}")
    return GEN_DIR / f"ddpm_simple_pretrain_samples{suffix}.csv"


def replicate_metrics(pool: npt.NDArray[np.int64], atus: FloatArr, tgt: dict,
                      sched_real: npt.NDArray[np.int64], d_real: npt.NDArray[np.int64],
                      w_real: FloatArr) -> dict[str, float]:
    """1 本の生成プールから key_* / err_* / ガードレールを測る。

    Args:
        pool: 群別サンプルプール, dtype=int64, (28, M, 96)
        atus: ATUS 実データの日本人口加重曲線, (12, 96)
        tgt: stage2_targets.load_stula_targets の戻り値（人口重み）
        sched_real: 実 ATUS 平日のスケジュール, (N, 96)
        d_real: 実 ATUS の群インデックス, (N,)
        w_real: 実 ATUS の調査ウェイト, (N,)

    Returns:
        指標名 -> 値
    """
    curve = cur.weighted_slot_rates(cur.pool_to_slot_rates(pool), tgt)
    out: dict[str, float] = {}

    key = cur.key_slot_table({"gen": curve})
    for _, row in key.iterrows():
        out[f"key_{row['activity']}_{row['time']}"] = float(row["gen"])

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
    return out


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """指標ごとに群の平均・sd、平均差、完全分離の有無を返す。

    Args:
        long: 列 arm / seed / metric / value の縦持ち

    Returns:
        指標ごとの行。separated は 2 群の値の範囲が重ならないとき True
    """
    rows: list[dict[str, Any]] = []
    for metric, grp in long.groupby("metric", sort=False):
        a = grp.loc[grp["arm"] == "noclock", "value"].to_numpy(dtype=np.float64)
        b = grp.loc[grp["arm"] == "clock", "value"].to_numpy(dtype=np.float64)
        rows.append({
            "metric": metric,
            "noclock_mean": a.mean(), "noclock_sd": a.std(ddof=1) if len(a) > 1 else np.nan,
            "clock_mean": b.mean(), "clock_sd": b.std(ddof=1) if len(b) > 1 else np.nan,
            "diff": b.mean() - a.mean(),
            # 各群 2 本以上のときだけ判定する（1 本ずつなら範囲は必ず重ならない）
            "separated": bool(len(a) >= 2 and len(b) >= 2
                              and (a.max() < b.min() or b.max() < a.min())),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    tgt = cur.st.load_stula_targets()
    atus = cur.atus_real_curve(tgt)
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)

    records: list[dict[str, Any]] = []
    for arm in ("noclock", "clock"):
        for seed in args.seeds:
            path = pool_csv(arm, seed)
            if not path.exists():
                raise FileNotFoundError(f"生成プールが無い ({arm}, seed={seed}): {path}")
            metrics = replicate_metrics(cur.load_sample_pool(path), atus, tgt,
                                        sched_real, d_real, w_real)
            records += [{"arm": arm, "seed": seed, "metric": k, "value": v,
                         "pool": path.name} for k, v in metrics.items()]
            print(f"[replicates] {arm:8s} seed={seed}  <- {path.name}")
    long = pd.DataFrame(records)

    # 目標値（ATUS 実データ）を並べる。key_* は曲線、ガードレールは実データそのもの
    ref: dict[str, float] = {}
    for _, row in cur.key_slot_table({"atus": atus}).iterrows():
        ref[f"key_{row['activity']}_{row['time']}"] = float(row["atus"])
    frag = sel.im.fragmentation_summary(sched_real, w_real)
    ref.update({k: float(frag[k]) for k in ("switch_mean", "single_slot_ratio", "wrap_closure_rate")})

    wide = long.pivot_table(index="metric", columns=["arm", "seed"], values="value", sort=False)
    summ = summarize(long).set_index("metric")
    summ.insert(0, "atus_real", pd.Series(ref))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== 反復ごとの値 ===")
        print(wide.round(4).to_string())
        print("\n=== 群の要約（separated = 2 群の範囲が重ならない）===")
        print(summ.round(4).to_string())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_csv, index=False)
    print(f"\n[replicates] 縦持ちを書いた: {args.out_csv}")


if __name__ == "__main__":
    main()
