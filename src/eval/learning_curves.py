"""
learning_curves.py
==================
最良の ckpt の config["history"]（epoch ごとの train / val 損失）を読み、学習曲線の表と図を作る。

★対象（MODELS）: 1 スロットあたりの交差エントロピーを history に残す再帰型のモデル
    gru          GRU_Aggregate（H = 128・wd = 1、既定のパス）                 種 42〜46
    gru_h384     GRU_Aggregate（H = 384・wd = 0、outputs/archive/gru_h384/）  種 42〜46
    lstm         LSTM_Aggregate（最低限の構成、重みなしの損失）               種 42〜46
    gru_minimal  GRU_Minimal（LSTM と同じ条件の GRU、重みなしの損失）         種 42〜46
  ★DDPM の ε-MSE は単位が違うので対象にしない
  ★lstm・gru_minimal の train / val は重みなしの交差エントロピー、gru・gru_h384 は TUFINLWGT の重み付き。
    損失の定義が違うので、この 2 組の値を比べない（同じ尺度の val は LSTM_Aggregate/eval_vs_gru.py が測る）
  ★学習後に slot_bias を補正した ckpt（_cal・_calg1.25）は学習をやり直していないので history を持たない。
    補正前の ckpt の曲線がそのまま補正後のモデルの学習曲線

★train と val は同じ条件の値ではない:
    train  epoch の途中で更新され続ける重みで測ったバッチ損失の加重平均
    val    epoch 末の重みで測った損失
    GRU_Aggregate はさらに、train だけ確率 P_UNCOND = 0.1 で条件を落とした行を含む（CFG のため）。
    train − val の差の一部はこの測り方の違いから来るので、過学習の判断は val の推移（最良 epoch の後に
    上がるか）で行う

データフロー:

```mermaid
flowchart TD
    CK["ckpt_paths(model)<br/>(seed, path) の並び"] --> LH["load_history<br/>config['history'] / best_epoch / stopped_epoch"]
    LH --> LONG["curves_long<br/>列 model / seed / epoch / train / val"]
    LONG --> SUM["curves_summary<br/>種ごとの最良 epoch・val・train"]
    LONG --> FIG["plot_learning_curves<br/>figures/learning_curves{_model}.png"]
```

使い方:
    .venv/bin/python src/eval/learning_curves.py --models gru lstm gru_minimal
    .venv/bin/python src/eval/learning_curves.py --models gru_h384

出力: data/processed/aggregates/learning_curves_{model}.csv（long）・learning_curves_{model}_summary.csv と、
      モデルのフォルダの figures/learning_curves_{model}.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
MODEL_DIR = REPO_ROOT / "src" / "models"
H384_ARCHIVE = REPO_ROOT / "outputs" / "archive" / "gru_h384" / "checkpoints"


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


@dataclass(frozen=True)
class ModelSpec:
    """学習曲線を描くモデルの設定

    Attributes:
        label: 表示と図の題に出す名前
        folder: src/models の下のフォルダ名（図の置き場所 figures/ を決める）
        seeds: 学習の種
        title: 図の題。None なら「{label} の学習曲線（種 n 本）」
        zoom: True なら「全 epoch」と「縦軸を拡大」の 2 枚組、False なら全 epoch の 1 枚だけ（小見出しなし）
        time_enc: LSTM_Aggregate・GRU_Minimal の ckpt の時刻符号（ckpt_path の time_enc）
        hidden: 同じく隠れ状態の幅（ckpt_path の hidden）
        weight_decay: 同じく weight decay（ckpt_path の weight_decay）
    """
    label: str
    folder: str
    seeds: tuple[int, ...]
    title: str | None = None
    zoom: bool = True
    time_enc: str = "none"
    hidden: int = 64
    weight_decay: float = 1.0


MODELS: dict[str, ModelSpec] = {
    "gru": ModelSpec("GRU（H = 128・wd = 1）", "GRU_Aggregate", (42, 43, 44, 45, 46)),
    "gru_h384": ModelSpec("GRU（H = 384・wd = 0）", "GRU_Aggregate", (42, 43, 44, 45, 46)),
    "lstm": ModelSpec("LSTM 最低限（H = 64・1 層）", "LSTM_Aggregate", (42, 43, 44, 45, 46)),
    "gru_minimal": ModelSpec("GRU 最低限（H = 64・1 層）", "GRU_Minimal", (42, 43, 44, 45, 46)),
    # 学習型の時刻符号・1 層・H = 64・weight decay 0.01（2026-10-07 ユーザー指示。題もユーザー指定）
    "lstm_h64_wd0.01": ModelSpec("LSTM（学習型の時刻符号・H = 64・1 層・wd = 0.01）", "LSTM_Aggregate",
                                 (42, 43, 44, 45, 46), title="LSTM, input=64, hidden=64, num_layers=1の学習曲線 (5シード)",
                                 zoom=False, time_enc="learned", weight_decay=0.01),
    "gru_minimal_h64_wd0.01": ModelSpec("GRU（学習型の時刻符号・H = 64・1 層・wd = 0.01）", "GRU_Minimal",
                                        (42, 43, 44, 45, 46), title="GRU, input=64, hidden=64, num_layers=1の学習曲線 (5シード)",
                                        zoom=False, time_enc="learned", weight_decay=0.01),
}
# 系列の色（stage1_gru_compare.ARM_COLOR の gru と ddpm_tf96。2 色で色覚の検査を通った組）
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"


def ckpt_paths(model: str) -> list[tuple[int, Path]]:
    """モデルの種ごとの最良の ckpt のパス"""
    spec = MODELS[model]
    if model == "gru":
        gm = _load("gru_aggregate_model", MODEL_DIR / "GRU_Aggregate" / "model.py")
        return [(s, Path(gm.ckpt_path(s))) for s in spec.seeds]
    if model == "gru_h384":
        gm = _load("gru_aggregate_model", MODEL_DIR / "GRU_Aggregate" / "model.py")
        return [(s, H384_ARCHIVE / f"gru_aggregate{gm.run_suffix(s)}.pt") for s in spec.seeds]
    # ★LSTM_Aggregate・GRU_Minimal は spec の time_enc・hidden・weight_decay を明示して ckpt を選ぶ。
    #   lstm・gru_minimal は時刻符号なし（time_enc="none"）。2026-10-06 からモデル側の既定は学習型の時刻符号なので、
    #   既定に任せると _time_learned の ckpt を読んでしまう
    size = {"time_enc": spec.time_enc, "hidden": spec.hidden, "weight_decay": spec.weight_decay}
    if spec.folder == "GRU_Minimal":
        gmin = _load("gru_minimal_model", MODEL_DIR / "GRU_Minimal" / "model.py")
        return [(s, Path(gmin.ckpt_path(s, **size))) for s in spec.seeds]
    lm = _load("lstm_aggregate_model", MODEL_DIR / "LSTM_Aggregate" / "model.py")
    return [(s, Path(lm.ckpt_path(s, **size))) for s in spec.seeds]


def load_history(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """ckpt の config から学習曲線と最良 epoch を読む

    Args:
        path: 最良の ckpt（{"model", "config"}）

    Returns:
        (列 epoch / train / val / sec, {"best_epoch", "best_val", "stopped_epoch"})

    Raises:
        KeyError: config に history が無い（補正した ckpt など）
    """
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    if "history" not in config:
        raise KeyError(f"config に history が無い（学習後に補正した ckpt では？）: {path}")
    hist = pd.DataFrame(config["history"])
    meta = {k: float(config[k]) for k in ("best_epoch", "best_val", "stopped_epoch")}
    return hist, meta


def curves_long(model: str) -> pd.DataFrame:
    """種ごとの学習曲線を縦に積む

    Returns:
        列 model / seed / epoch / train / val / sec / best_epoch / stopped_epoch
    """
    frames = []
    for seed, path in ckpt_paths(model):
        hist, meta = load_history(path)
        frames.append(hist.assign(model=model, seed=seed, best_epoch=int(meta["best_epoch"]),
                                  stopped_epoch=int(meta["stopped_epoch"])))
    long = pd.concat(frames, ignore_index=True)
    long["epoch"] = long["epoch"].astype(int)
    return cast(pd.DataFrame, long[["model", "seed", "epoch", "train", "val", "sec", "best_epoch", "stopped_epoch"]])


def curves_summary(long: pd.DataFrame, max_epochs: int | None = None) -> pd.DataFrame:
    """種ごとに最良 epoch の値と、最良の後の val の上がり幅

    Args:
        long: curves_long の戻り値
        max_epochs: 学習の上限の epoch。渡すと、上限まで回った種（早期終了しなかった種）に印を付ける

    Returns:
        列 model / seed / best_epoch / stopped_epoch / best_val / train_at_best / val_minus_train_at_best /
        val_rise_after_best（最後の val − 最良の val。正なら最良の後に val が上がった）/ hit_max_epochs
    """
    rows = []
    for key, g in long.groupby(["model", "seed"], sort=False):
        model, seed = cast(tuple[str, int], key)
        g = g.sort_values("epoch")
        best = g[g["epoch"] == g["best_epoch"].iloc[0]].iloc[0]
        stopped = int(g["stopped_epoch"].iloc[0])
        rows.append({"model": model, "seed": int(seed), "best_epoch": int(best["epoch"]), "stopped_epoch": stopped,
                     "best_val": float(best["val"]), "train_at_best": float(best["train"]),
                     "val_minus_train_at_best": float(best["val"] - best["train"]),
                     "val_rise_after_best": float(g["val"].iloc[-1] - best["val"]),
                     "hit_max_epochs": bool(max_epochs is not None and stopped >= max_epochs)})
    return pd.DataFrame(rows)


def max_epochs_of(model: str) -> int:
    """学習の上限の epoch（モデルのフォルダの model.EPOCHS）"""
    folder = MODELS[model].folder
    return int(_load(f"{folder.lower()}_model", MODEL_DIR / folder / "model.py").EPOCHS)


def plot_learning_curves(long: pd.DataFrame, model: str, out: Path) -> None:
    """train（青）と val（橙）の学習曲線。種ごとに 1 本、最良 epoch に丸印

    Note:
        ★spec.zoom が True なら、左は全 epoch、右は最良 epoch の前後（縦軸を val の近くに寄せる）。
          False なら全 epoch の 1 枚だけで、小見出しを付けない
        ★図の題は spec.title（None なら「{label} の学習曲線（種 n 本）」）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rd = _load("lc_rare_diagnosis", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_rare_diagnosis.py")
    rd._figure_module().setup_fonts()
    spec = MODELS[model]
    zooms = (False, True) if spec.zoom else (False,)
    fig, axes = plt.subplots(1, len(zooms), figsize=(6.5 * len(zooms), 4.6), squeeze=False)
    seeds = list(dict.fromkeys(long["seed"]))
    for ax, zoom in zip(axes[0], zooms):
        for i, seed in enumerate(seeds):
            g = cast(pd.DataFrame, long[long["seed"] == seed]).sort_values("epoch")
            first = i == 0
            ax.plot(g["epoch"], g["train"], color=COLOR_TRAIN, lw=1.4, alpha=0.85, label="train" if first else None)
            ax.plot(g["epoch"], g["val"], color=COLOR_VAL, lw=1.4, alpha=0.85, label="val" if first else None)
            b = g[g["epoch"] == g["best_epoch"]]
            ax.scatter(b["epoch"], b["val"], s=40, color=COLOR_VAL, edgecolors="white", linewidths=1.2, zorder=3,
                       label="最良 epoch" if first else None)
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_ylabel("交差エントロピー（nats / スロット）" if spec.zoom else "交差エントロピー", fontsize=10)
        ax.grid(alpha=0.3, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if zoom:
            vals = long["val"].to_numpy(dtype=np.float64)
            best = np.asarray(long.groupby("seed")["val"].min(), dtype=np.float64)
            train_min = np.nanmin(long["train"].to_numpy(dtype=np.float64))
            lo, hi = float(min(best.min(), train_min)), float(np.quantile(vals, 0.5))
            pad = (hi - lo) * 0.15
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title("縦軸を拡大", fontsize=11)
        elif spec.zoom:
            ax.set_title("全 epoch", fontsize=11)
    fig.suptitle(spec.title or f"{spec.label} の学習曲線（種 {len(seeds)} 本）", fontsize=14)
    axes[0][0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[learning_curves] 図: {out}")


def run(model: str) -> None:
    """1 モデルの表と図を出力する"""
    long = curves_long(model)
    summary = curves_summary(long, max_epochs_of(model))
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        print(f"\n=== {MODELS[model].label} ===")
        print(summary.to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_DIR / f"learning_curves_{model}.csv", index=False)
    summary.to_csv(OUT_DIR / f"learning_curves_{model}_summary.csv", index=False)
    plot_learning_curves(long, model, MODEL_DIR / MODELS[model].folder / "figures" / f"learning_curves_{model}.png")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="ckpt の history から学習曲線の表と図を作る")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=["gru", "lstm"])
    args = ap.parse_args()
    for model in args.models:
        run(model)


if __name__ == "__main__":
    main()
