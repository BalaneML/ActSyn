"""
stage2_clock_compare.py
=======================
時刻符号なし／時刻符号つきの Stage 1 から始めた Stage 2 を、同じ物差しで並べる（関門 B）。

関門 B の問い: 時刻符号つきの Stage 1 から始めると、Stage 2 は日本の教師 A* の
**鋭い時刻構造**（周期 8 時間以下の成分。12:00 の昼食・7:00 の朝食・8:00 の通勤）を
時刻符号なしより多く埋めるか。そのとき系列の妥当性（ガードレール）を壊していないか。

★各 run は**自分の Stage 1 の zero-shot** を起点に測る。時刻符号なしの step200 は時刻符号なしの
  zero-shot から、時刻符号つきの step200 は時刻符号つきの zero-shot から、どれだけ A* へ寄ったか。
  起点を混ぜると「Stage 1 の差」と「Stage 2 で動いた量」が区別できない。

    run                 起点（zero-shot）                                  微調整後
    noclock_lam0.003    ddpm_simple_pretrain_common12_weekday_20260819     stage2_lam0.003/step200
    clock_lam0.003      ddpm_simple_pretrain_common12_weekday_clock        stage2_lam0.003_clock/step200
    clock_lam0 など      --run で足す

測る量（すべて n=1000・pool_seed=12345 のプール、stage2_select.dump_rates の .npz）:

    band_*        stage2_curves.band_closure_table の「埋めた割合」。日本人口加重、教師 A* への残差
    key_*         代表スロットの値（日本人口加重）。起点と微調整後の両方
    rate_rmse /   stage2_targets.eval_against（12act, 群の素平均）。dev_rmse が転移の判定量
    dev_rmse      （[[stage2-rate-mae-can-be-gamed]]：rate_mae は平滑化で下がる）
    guardrail     stage2_select.guardrails と memorization_guardrail（実 ATUS 平日の群構成へ重み付け）

★プールの取り違えを防ぐ。.npz の meta（step, n, pool_seed）が期待と一致しなければ止める。
  .npz の ckpt 欄はファイル名だけ（stage2_step200.pt）で run を区別できないので、
  run とプールの対応はこのスクリプトの引数が唯一の出所である。

データフロー:

```mermaid
flowchart TD
    ZS["起点 .npz<br/>rates (28, 12*96), pool (28, 1000, 96)"] --> CHK["check_meta<br/>step / n / pool_seed"]
    FT["微調整後 .npz"] --> CHK
    CHK --> W["stage2_curves.weighted_slot_rates<br/>curve (12, 96) 日本人口加重"]
    W --> BAND["band_closure_table(curves, teacher, base)<br/>band_*"]
    W --> KEY["key_slot_table<br/>key_*"]
    CHK --> EV["stage2_targets.eval_against<br/>rate_rmse / dev_rmse"]
    CHK --> GR["stage2_select.guardrails<br/>+ memorization_guardrail"]
    BAND --> OUT["縦持ち (run, stage, metric, value)"]
    KEY --> OUT
    EV --> OUT
    GR --> OUT
    W --> FIG["plot_curves<br/>教師・ATUS実・各 run の起点と微調整後"]
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage2_clock_compare.py
    .venv/bin/python src/eval/diagnostics/stage2_clock_compare.py \\
        --run clock_lam0_s300=outputs/generated/ddpm_simple_pretrain_common12_weekday_clock_rates.npz,outputs/generated/stage2_lam0_clock/stage2_step300_rates.npz,300
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"
GEN_DIR = REPO_ROOT / "outputs" / "generated"
OUT_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "stage2_clock_compare.csv"
OUT_FIG = REPO_ROOT / "outputs" / "figures" / "stage2_clock_compare.png"

# λ 掃引・LGO と同じプール設定。違う .npz を渡したら check_meta で止める
EXPECTED_N = 1000
EXPECTED_POOL_SEED = 12345

# 既定の比較（run 名 -> (起点 .npz, 微調整後 .npz, 微調整後の step)）
DEFAULT_RUNS: dict[str, tuple[Path, Path, int]] = {
    "noclock_lam0.003_s200": (GEN_DIR / "ddpm_simple_pretrain_common12_weekday_20260819_rates.npz",
                              GEN_DIR / "stage2_step200_rates.npz", 200),
    "clock_lam0.003_s200": (GEN_DIR / "ddpm_simple_pretrain_common12_weekday_clock_rates.npz",
                            GEN_DIR / "stage2_lam0.003_clock" / "stage2_step200_rates.npz", 200),
}

# 報告するガードレール（stage2_select.guardrails / memorization_guardrail の戻り値のキー）
GUARDRAIL_KEYS = ("switch_mean", "single_slot_ratio", "wrap_closure_rate", "bigram_jsd",
                  "switch_emd", "night_intrusion_rate", "travel_single_rate",
                  "var_ratio_switches", "dcr_gap", "exact_copy_rate")

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


cur: Any = _load("clockcmp_stage2_curves", SIMPLE_DIR / "stage2_curves.py")
sel: Any = _load("simple_stage2_select", SIMPLE_DIR / "stage2_select.py")
st: Any = sel.st
sm: Any = sel.sm


def load_pool_npz(path: Path, expected_step: int) -> dict[str, np.ndarray]:
    """dump_rates の .npz を読み、meta が期待どおりかを確かめる。

    Args:
        path: stage2_select.dump_rates が書いた .npz
        expected_step: 期待する step。zero-shot（Stage 1 の重み）なら 0

    Returns:
        pool (28, M, 96) / rates / rates_a / rates_b (28, 12*96)

    Raises:
        ValueError: meta の step / n / pool_seed が期待と違うとき
    """
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in ("pool", "rates", "rates_a", "rates_b", "meta")}
    step, n, pool_seed = (int(v) for v in out["meta"])
    if (step, n, pool_seed) != (expected_step, EXPECTED_N, EXPECTED_POOL_SEED):
        raise ValueError(f"{path} の meta (step, n, pool_seed) = {(step, n, pool_seed)} が期待 "
                         f"{(expected_step, EXPECTED_N, EXPECTED_POOL_SEED)} と違う。プールの取り違えを疑うこと")
    return out


def pool_scores(npz: dict[str, np.ndarray], tgt: dict, sched_real: np.ndarray,
                d_real: np.ndarray, w_real: np.ndarray) -> dict[str, float]:
    """1 本のプールの eval_against（12act）とガードレールを返す。

    Args:
        npz: load_pool_npz の戻り値
        tgt: stage2_targets.load_stula_targets の戻り値
        sched_real / d_real / w_real: sm.load_data と sm.cond_to_d の戻り値

    Returns:
        指標名 -> 値
    """
    ev = st.eval_against(npz["rates"], tgt, st.mask_12act(),
                         mu_hat_split=(npz["rates_a"], npz["rates_b"]))
    out = {k: float(ev[k]) for k in ("rate_mae", "rate_rmse", "rate_mse_split", "dev_rmse")}
    pool = npz["pool"].astype(np.int64)
    d, m, s = pool.shape
    gen = pool.reshape(d * m, s)
    gen_d = np.repeat(np.arange(d), m)
    g = sel.guardrails(gen, gen_d, sched_real, d_real, w_real)
    g.update(sel.memorization_guardrail(gen, sched_real, seed=EXPECTED_POOL_SEED))
    out.update({k: float(g[k]) for k in GUARDRAIL_KEYS})
    return out


def parse_run(spec: str) -> tuple[str, tuple[Path, Path, int]]:
    """`name=ZS_NPZ,FT_NPZ,STEP` を分解する。"""
    name, rest = spec.split("=", 1)
    zs, ft, step = rest.split(",")
    return name, (Path(zs), Path(ft), int(step))


# run 名の接頭辞 -> 凡例に出す構造の呼び名
ARCH_LABELS = {"clock": "時刻符号つき", "noclock": "時刻符号なし"}


def legend_labels(run: str) -> tuple[str, str]:
    """run 名から、図の凡例に出す (起点, 微調整後) の文言を作る。

    Stage 1 だけで学習したモデルを Pre-trained、Stage 2 まで学習したモデルを Fine-tuned と呼ぶ。
    CSV の run 列は run 名のまま残し、凡例だけをこの文言にする。

    Args:
        run: `{clock|noclock}_lam{λ}_s{step}` 形式の run 名（例: clock_lam0_s300）

    Returns:
        (起点の凡例, 微調整後の凡例)。
        例: ("時刻符号つき Pre-trained", "時刻符号つき Fine-tuned (λ=0, step300)")。
        ARCH_LABELS に無い接頭辞はそのまま使い、形式に合わない run 名は微調整後の凡例に run 名をそのまま使う
    """
    prefix = run.split("_lam")[0]
    arch = ARCH_LABELS.get(prefix, prefix)
    m = re.fullmatch(r"[a-z]+_lam(?P<lam>[0-9.]+)_s(?P<step>[0-9]+)", run)
    if m is None:
        return f"{arch} Pre-trained", run
    return (f"{arch} Pre-trained",
            f"{arch} Fine-tuned (λ={m['lam']}, step{m['step']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="NAME=ZS_NPZ,FT_NPZ,STEP",
                    help="比較に足す run。既定の 2 本（時刻符号なし/時刻符号つきの λ=0.003 step200）に加わる")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-fig", type=Path, default=OUT_FIG)
    args = ap.parse_args()

    runs = dict(DEFAULT_RUNS)
    runs.update(parse_run(s) for s in args.run)

    tgt = st.load_stula_targets()
    teacher = cur.weighted_slot_rates(np.asarray(tgt["group_rates_tbl"], dtype=np.float64), tgt)
    atus = cur.atus_real_curve(tgt)
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)

    records: list[dict[str, Any]] = []
    fig_curves: dict[str, FloatArr] = {}
    score_cache: dict[Path, dict[str, float]] = {}
    for run, (zs_path, ft_path, step) in runs.items():
        zs, ft = load_pool_npz(zs_path, 0), load_pool_npz(ft_path, step)
        print(f"[compare] {run}: {zs_path.name} -> {ft_path.relative_to(REPO_ROOT) if ft_path.is_absolute() else ft_path}")
        c_zs = cur.weighted_slot_rates(cur.load_rates_npz(zs_path), tgt)
        c_ft = cur.weighted_slot_rates(cur.load_rates_npz(ft_path), tgt)
        zs_label, ft_label = legend_labels(run)
        fig_curves.setdefault(zs_label, c_zs)
        fig_curves[ft_label] = c_ft

        bands = cur.band_closure_table({"zs": c_zs, "ft": c_ft}, teacher, "zs")
        for _, row in bands[bands.activity == "ALL"].iterrows():
            records.append({"run": run, "stage": "ft", "metric": f"band_closed_{row['band']}",
                            "value": float(row["closed"])})
        meals = bands[(bands.activity == "MEALS") & (bands.band == "all")]
        records.append({"run": run, "stage": "ft", "metric": "band_closed_MEALS_all",
                        "value": float(meals["closed"].iloc[0])})

        for stage, curve in (("zs", c_zs), ("ft", c_ft)):
            for _, row in cur.key_slot_table({"v": curve}).iterrows():
                records.append({"run": run, "stage": stage,
                                "metric": f"key_{row['activity']}_{row['time']}", "value": float(row["v"])})
        for stage, path, npz in (("zs", zs_path, zs), ("ft", ft_path, ft)):
            if path not in score_cache:
                score_cache[path] = pool_scores(npz, tgt, sched_real, d_real, w_real)
            records += [{"run": run, "stage": stage, "metric": k, "value": v}
                        for k, v in score_cache[path].items()]
    long = pd.DataFrame(records)

    teacher_key = {f"key_{r['activity']}_{r['time']}": float(r["v"])
                   for _, r in cur.key_slot_table({"v": teacher}).iterrows()}
    wide = long.pivot_table(index="metric", columns=["run", "stage"], values="value", sort=False)
    wide.insert(0, "teacher", pd.Series(teacher_key))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== run ごとの値（zs = 自分の Stage 1 の zero-shot, ft = 微調整後）===")
        print(wide.round(4).to_string())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_csv, index=False)
    print(f"\n[compare] 縦持ちを書いた: {args.out_csv}")
    # λ と step は凡例（run 名）に出る。--run で足した run もあるので題には書かない
    cur.plot_curves(teacher, fig_curves, args.out_fig,
                    "人口加重平均の時刻別行動者率（時刻符号なし vs 時刻符号つき）",
                    refs={cur.ATUS_REAL_LABEL: atus})
    print(f"[compare] 図を書いた: {args.out_fig}")


if __name__ == "__main__":
    main()
