"""
eval_stage2_curves.py
=====================
GRU_Aggregate の Stage 2 の時刻別行動者率を、日本の教師 A* に対する活動ごとの MAE・最大誤差（pt）と
Top-3 の 15 分区間で並べる（日本人口加重の全国の曲線。stage1_gru_compare.py の Q4 と同じ表）。

★比べるもの（ARMS）:
    gru_pretrained  Stage 1 のまま（補正後 g = 1.25、Pre-trained）   種 42〜46
    gru_shift       δ を足す版（stage2.py、run = all、Fine-tuned）    種 42〜46
    gru_finetune    重みを更新する版（stage2_finetune.py、run = all） 種 42〜46
    ddpm_s2         DDPM Fine-tuned（λ = 0.003、step 200）            1 本
  ★率は plot_stage2_curves.stage2_rates（評価と同じ乱数・群あたり s2.EVAL_N 本）から読む
  ★Fine-tuned は教師 A* の 28 群すべてで合わせた run（all）なので、教師との誤差は学習に使った値との誤差。
    教師に使っていない群での誤差は LGO（fold）の評価で見る

★床（教師の個票が無いので、群ごとのセルの分散から雑音を引いて作る）:
    下限側  生成プールの有限さだけ。群のセルに分散 p(1 − p) / (EVAL_N × 種の本数) の正規雑音（p = A*）
    上限側  さらに教師の標本誤差 tgt["group_rates_var"]（design effect を無視した下限値）を足す
  どちらも cur.weighted_slot_rates で全国の曲線に畳んでから、sre.floor_errors で同じ統計量の 95% 分位を取る。
  ★雑音は 15 分区間の間で独立に引く。実際の生成は同じ人が隣の区間にも続くので区間の間で相関があり、
    独立に引くと最大 |誤差| の床は実際より大きめに出る（床の外と判定するには厳しめ = 安全側）

データフロー:

```mermaid
flowchart TD
    RT["p2c.stage2_rates(seed, run, method)<br/>(28, 12, 96)"] --> W["cur.weighted_slot_rates<br/>curves[arm] (12, 96)"]
    DD["cur.load_rates_npz(s2.DDPM_STEP200_RATES)"] --> W
    TGT["tgt['group_rates_tbl'] = A*"] --> TN["teacher_curve (12, 96)"]
    TGT --> FC["noise_floor_curves<br/>下限側・上限側"]
    VAR["tgt['group_rates_var']"] --> FC
    W --> TAB["sre.activity_error_table / sre.topk_error_table"]
    TN --> TAB
    FC --> TAB
    TAB --> FIG["cmp.plot_curve_errors<br/>figures/stage2_gru_curve_errors{tag}.png"]
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/eval_stage2_curves.py --tag _h128

出力: data/processed/aggregates/stage2_gru_curve_{errors,topk}{tag}.csv と
      src/models/GRU_Aggregate/figures/stage2_gru_curve_errors{tag}.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/GRU_Aggregate -> repo root
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

FloatArr = npt.NDArray[np.float64]


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


p2c: Any = _load("gru_plot_stage2_curves", Path(__file__).resolve().parent / "plot_stage2_curves.py")
s2: Any = p2c.s2
cur: Any = p2c.cur
cmp: Any = p2c.cmp
sre: Any = cmp.sre

# arm → (run, method)。ddpm_s2 は別に読む
GRU_ARMS: dict[str, tuple[str, str]] = {"gru_pretrained": ("zeroshot", "shift"), "gru_shift": ("all", "shift"),
                                        "gru_finetune": ("all", "finetune")}
ARM_LABELS: dict[str, str] = {"gru_pretrained": "GRU Pre-trained", "gru_shift": p2c.LABEL_SHIFT,
                              "gru_finetune": p2c.LABEL_FT, "ddpm_s2": "DDPM Fine-tuned（λ=0.003, step200）"}
# 色は plot_stage2_curves と同じ対応（δ を足す版 = 赤紫、重みを更新する版 = 青、DDPM = 橙）
ARM_COLOR: dict[str, str] = {"gru_shift": p2c.COLOR_GRU, "gru_finetune": p2c.COLOR_GRU_FT,
                             "ddpm_s2": p2c.COLOR_DDPM}
# ★図には Fine-tuned の 3 つだけを描く（Pre-trained は表だけ。4 本目の色が灰色の床の帯と紛れるため）
PLOT_ARMS: tuple[str, ...] = ("gru_shift", "gru_finetune", "ddpm_s2")
N_FLOOR_DRAWS = 50


def arm_curves(tgt: dict) -> dict[str, list[FloatArr]]:
    """arm → 種ごとの全国の時刻別行動者率 (12, 96) の並び"""
    curves: dict[str, list[FloatArr]] = {
        arm: [cur.weighted_slot_rates(p2c.stage2_rates(s, run, method), tgt) for s in s2.SEEDS]
        for arm, (run, method) in GRU_ARMS.items()}
    curves["ddpm_s2"] = [cur.weighted_slot_rates(cur.load_rates_npz(s2.DDPM_STEP200_RATES), tgt)]
    return curves


def noise_floor_curves(tgt: dict, n_seeds: int, with_teacher: bool, n_draws: int = N_FLOOR_DRAWS,
                       seed: int = 0) -> FloatArr:
    """教師 A* に正規雑音を足した群別の率を全国の曲線に畳む（完全なモデルの曲線）

    Args:
        tgt: stage2_targets.load_stula_targets の戻り値
        n_seeds: 平均する種の本数（生成プールの分散を 1 / n_seeds にする）
        with_teacher: True なら教師の標本誤差の分散も足す（上限側）
        n_draws: 作る回数
        seed: 乱数の種

    Returns:
        floor_curves, (n_draws, 12, 96)
    """
    p = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)                     # (28, 12, 96)
    var = p * (1.0 - p) / (s2.EVAL_N * n_seeds)
    if with_teacher:
        var = var + np.asarray(tgt["group_rates_var"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(var)
    return np.asarray([cur.weighted_slot_rates(p + rng.normal(0.0, 1.0, size=p.shape) * sd, tgt)
                       for _ in range(n_draws)], dtype=np.float64)


def error_tables(curves: dict[str, list[FloatArr]], teacher: FloatArr,
                 tgt: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Any]]:
    """arm ごとの活動ごとの誤差の表と Top-k の表（床は種の本数ごと）

    Returns:
        (列 arm + sre.activity_error_table の列, 列 arm + sre.topk_error_table の列,
         種の本数 → {"low", "high"} の床)
    """
    floors: dict[int, Any] = {}
    act_rows, topk_rows = [], []
    for arm, cs in curves.items():
        n = len(cs)
        if n not in floors:
            floors[n] = {level: sre.floor_errors(teacher, noise_floor_curves(tgt, n, with_teacher))
                         for level, with_teacher in (("low", False), ("high", True))}
        stack = np.asarray(cs, dtype=np.float64)
        lo, hi = floors[n]["low"], floors[n]["high"]
        act_rows.append(sre.activity_error_table(teacher, stack, cur.ACT_NAMES, lo, hi).assign(arm=arm))
        topk_rows.append(sre.topk_error_table(teacher, stack, cur.ACT_NAMES, lo, hi).assign(arm=arm))

    def _arm_first(df: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, df[["arm", *[c for c in df.columns if c != "arm"]]])
    return _arm_first(pd.concat(act_rows, ignore_index=True)), _arm_first(pd.concat(topk_rows, ignore_index=True)), floors


def main() -> None:
    """表と図を出力する"""
    ap = argparse.ArgumentParser(description="GRU Stage 2 の時刻別行動者率の誤差（教師 A* に対して）")
    ap.add_argument("--tag", default="", help="出力のファイル名の接尾辞（例: _h128）。既定は付けない")
    args = ap.parse_args()
    tgt = s2.st.load_stula_targets()
    teacher = cur.weighted_slot_rates(np.asarray(tgt["group_rates_tbl"], dtype=np.float64), tgt)
    curves = arm_curves(tgt)
    errors, topk, floors = error_tables(curves, teacher, tgt)
    cmp._show("Stage 2 活動ごとの MAE・最大誤差（pt、教師 A*、種平均の曲線）と床", errors)
    cmp._show(f"Stage 2 活動ごとの Top-{sre.TOP_K} の 15 分区間（pt）", topk)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUT_DIR / f"stage2_gru_curve_errors{args.tag}.csv", index=False)
    topk.to_csv(OUT_DIR / f"stage2_gru_curve_topk{args.tag}.csv", index=False)
    cmp.plot_curve_errors(curves, teacher, topk, floors, FIG_DIR / f"stage2_gru_curve_errors{args.tag}.png",
                          arms=PLOT_ARMS, band_seeds=len(s2.SEEDS), colors=ARM_COLOR, labels=ARM_LABELS,
                          ref_name="教師")


if __name__ == "__main__":
    main()
