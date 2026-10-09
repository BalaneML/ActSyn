"""
sweep_width_decay.py
====================
隠れ層の幅 H と weight decay の掃引（計画書 §10）。種 42 だけを学習し、最良の epoch の val を比べる。

★採用中の ckpt（outputs/checkpoints/gru_aggregate*.pt）は上書きしない。保存先は SWEEP_DIR。

```mermaid
flowchart TD
    GRID["HIDDENS × WEIGHT_DECAYS"] --> TR["gm.train(seed=gm.SEED, hidden, weight_decay)<br/>save_path = sweep_ckpt_path(hidden, weight_decay)"]
    TR --> CK["gru_sweep/gru_h{H}_wd{wd}.pt<br/>config: best_val, best_epoch, history, calib_gap_*"]
    CK --> SUM["summarize() → SUMMARY_CSV"]
    SUM --> CH["choose(table)<br/>val 最小と TIE 未満の差の組から、H・wd の小さい方"]
```

使い方:
    # 1 つの幅で weight decay を順に回す（別の幅は別のプロセスで並べて回せる）
    .venv/bin/python src/models/GRU_Aggregate/sweep_width_decay.py --hidden 256 --weight-decay 0 0.1 1 10
    # 回し終えた ckpt から表を作り、§10.3 の規則で選ぶ
    .venv/bin/python src/models/GRU_Aggregate/sweep_width_decay.py --summary
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名の model.py を取り違えないため）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm: Any = _load("gru_aggregate_model", Path(__file__).resolve().parent / "model.py")

# 計画書 §10.2・§10.3（結果を見る前に固定）
HIDDENS: tuple[int, ...] = (128, 256)   # 512 はユーザーの指示で外した（2026-09-29、結果を見る前）
WEIGHT_DECAYS: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
TIE = 1e-3                           # H = 384・wd = 0 の種 42〜46 の val の幅
SWEEP_DIR = REPO_ROOT / "outputs" / "checkpoints" / "gru_sweep"
SUMMARY_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "stage1_gru_width_decay_sweep.csv"


def sweep_ckpt_path(hidden: int, weight_decay: float) -> Path:
    """掃引の 1 本の ckpt のパス（例: gru_h256_wd0.1.pt）"""
    return SWEEP_DIR / f"gru_h{hidden}_wd{weight_decay:g}.pt"


def run_one(hidden: int, weight_decay: float) -> None:
    """種 42 を 1 本学習して sweep_ckpt_path に保存する（途中の ckpt と生成プールは作らない）"""
    print(f"[sweep] hidden={hidden} weight_decay={weight_decay:g}", flush=True)
    gm.train(seed=gm.SEED, save_path=sweep_ckpt_path(hidden, weight_decay), save_every=0,
             hidden=hidden, weight_decay=weight_decay)


def summarize() -> pd.DataFrame:
    """回し終えた ckpt の config から表を作る（無い組は飛ばす）

    Returns:
        1 行 1 本の表。列は hidden・weight_decay・params・best_epoch・best_val・train_at_best・
        stopped_epoch・val_at_stop・calib_gap_max・calib_gap_mean・sec_per_epoch
    """
    rows: list[dict[str, Any]] = []
    for hidden in HIDDENS:
        for wd in WEIGHT_DECAYS:
            path = sweep_ckpt_path(hidden, wd)
            if not path.exists():
                continue
            ckpt = torch.load(path, map_location="cpu")
            cfg, hist = ckpt["config"], ckpt["config"]["history"]
            best = int(cfg["best_epoch"])
            rows.append({
                "hidden": hidden, "weight_decay": wd,
                "params": sum(v.numel() for v in ckpt["model"].values()),
                "best_epoch": best, "best_val": float(cfg["best_val"]),
                "train_at_best": float(hist["train"][best - 1]),
                "stopped_epoch": int(cfg["stopped_epoch"]), "val_at_stop": float(hist["val"][-1]),
                "calib_gap_max": float(cfg["calib_gap_max"]), "calib_gap_mean": float(cfg["calib_gap_mean"]),
                "sec_per_epoch": sum(hist["sec"]) / len(hist["sec"]),
            })
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> dict[str, Any]:
    """§10.3 の規則で 1 組を選ぶ: val 最小との差が TIE 未満の組のうち、H が小さく、次に wd が小さいもの"""
    rows: list[dict[str, Any]] = [{str(k): v for k, v in r.items()} for r in table.to_dict("records")]
    best = min(float(r["best_val"]) for r in rows)
    tied = [r for r in rows if float(r["best_val"]) - best < TIE]
    return min(tied, key=lambda r: (int(r["hidden"]), float(r["weight_decay"])))


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GRU_Aggregate: 隠れ層の幅と weight decay の掃引（計画書 §10）")
    ap.add_argument("--hidden", type=int, choices=HIDDENS)
    ap.add_argument("--weight-decay", type=float, nargs="+", default=list(WEIGHT_DECAYS))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        table = summarize()
        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(SUMMARY_CSV, index=False)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
            print(table.to_string(index=False))
        picked = choose(table)
        print(f"[sweep] 選んだ組: hidden={int(picked['hidden'])} weight_decay={picked['weight_decay']:g} "
              f"(val {picked['best_val']:.4f})  → {SUMMARY_CSV}")
        return
    if args.hidden is None:
        ap.error("--hidden か --summary を指定する")
    for wd in args.weight_decay:
        run_one(args.hidden, wd)


if __name__ == "__main__":
    main()
