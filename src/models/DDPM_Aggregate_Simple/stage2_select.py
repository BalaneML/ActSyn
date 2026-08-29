"""
stage2_select.py
================
Stage 2 の事後チェックポイント選択（Stage2_design.md §8.4, §9.3, §9.4）

Stage 2 は早期終了を使わず固定ステップ予算で回し切る。学習中に「良くなったか」を
判定できないのは、目的関数（集計適合）と守りたいもの（個票の構造）が別物だからである。
そこで保存済みチェックポイントを学習後に2軸で並べ、λ パレート曲線を描いて選ぶ。

    軸1 教師適合    28群表への rate_mae / dev_rmse、per-activity mae と相対誤差
                    ★循環している。28群すべてを教師にした条件では「集計にどこまで
                      合わせられるか」の上限として読む値であって、汎化の主張ではない。
                      --holdout-groups を使った場合だけ held-out 側が非循環になる
    軸2 ガードレール travel_single_rate / night_intrusion（feasibility）、
                    fragmentation_summary / switch EMD / bigram_jsd（individual_metrics）、
                    separation_ratio（conditioning）、OTHER_X シェア
                    ★非循環。集計損失が一切見ていない量なので独立した情報を持つ

★基準は zero-shot 値であって実データ値ではない。Stage 1 の時点でガードレール全指標が
  既にノイズ床の外にあるので（§9.2）、「Stage 2 はガードレールを壊さない」という主張は
  使えない。使えるのは「悪化させない」「改善する」の2つ。

★出力 CSV は数値の出所を列で機械的に区別する（§9.4）。このリポジトリは出所の
  取り違えを2回起こしている。

    teacher_groups  損失に使った群数（28 or それ未満）
    eval_kind       in-teacher / held-out
    reference       teacher / atus       （何と比べた値か）
    mask            11act / 12act        （OTHER_X を含むか）

使い方:
    .venv/bin/python3 src/models/DDPM_Aggregate_Simple/stage2_select.py \\
        --ckpt-dir outputs/checkpoints/stage2 --n 2000
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    """sys.modules に一意名で載せる。既に同じファイルが同じ名前で入っていれば使い回す。

    ★使い回しが要点。同名で読み直すと sys.modules のエントリは置き換わるが、
      先に読んだ側が掴んでいるモジュールオブジェクトは別のまま残る。すると
      「model.T_STEPS を差し替えたのに、こちらから呼ぶ生成は 1000 ステップのまま」
      のような、例外を出さずに黙って重くなる／数値が変わる食い違いが起きる。
    """
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sm: Any = _load("simple_model", HERE / "model.py")
ck: Any = _load("simple_stage2_checkpoint", HERE / "stage2_checkpoint.py")
st: Any = _load("simple_stage2_targets", HERE / "stage2_targets.py")
im: Any = _load("select_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")
fe: Any = _load("select_feasibility", REPO_ROOT / "src" / "eval" / "feasibility.py")
cd: Any = _load("select_conditioning", REPO_ROOT / "src" / "eval" / "conditioning.py")

OUT_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "stage2_checkpoint_selection.csv"
DEFAULT_N = 2000

# 軸2 で追いかける指標と、その zero-shot 実測値（Stage2_design.md §9.2）。
# ★基準は実データ値ではなく zero-shot 値。Stage 1 の時点で全指標がノイズ床の外に
#   あるので、「壊さない」ではなく「悪化させない／改善する」で主張を立てる。
# ★bigram_jsd と switch_emd は「重み付き・対角除く・行平均」の値。重み無しだと
#   0.0059 / 0.7765 になり、過去の記録に両方が混在しているので必ず揃える。
ZERO_SHOT_GUARDRAILS = {
    "travel_single_rate":   0.2229,
    "night_intrusion_rate": 0.0444,
    "switch_mean":          13.427,
    "single_slot_ratio":    0.2563,
    "wrap_closure_rate":    0.0950,
    "bigram_jsd":           0.0051,
    "switch_emd":           0.8515,
    "travel_odd_rate":      0.4724,
    "separation_ratio":     1.107,
    "other_x_share":        0.0136,
}
GUARDRAIL_KEYS = tuple(ZERO_SHOT_GUARDRAILS)


def guardrails(gen: np.ndarray, gen_d: np.ndarray, sched_real: np.ndarray,
               d_real: np.ndarray, w_real: np.ndarray) -> dict[str, float]:
    """軸2 のガードレール一式。生成側は実 ATUS 平日の群構成へ重み付けして測る。

    ★重みの出所は im.group_reweight ただ一つ。プールは群一様なので、非加重の人数比を
      使うと総変動距離で 0.139 ずれる。
    ★bigram_jsd と switch_emd は「重み付き・対角除く・行平均」で測る。重み無しだと
      別の値になり（0.0059 / 0.7765）、過去の記録に両方が混在しているので揃える。
    """
    w_gen = im.group_reweight(gen_d, w_real, d_real, sm.D_GROUPS)
    feas = fe.feasibility_summary(gen, w_gen)
    frag = im.fragmentation_summary(gen, w_gen)
    # ★キー名で絞り込まずに1つずつ明示的に引く。フィルタで書くと上流のキー名が
    #   変わったときに黙って指標が落ちる（feasibility_summary は night_intrusion_rate を
    #   night_intrusion というキーで返す）。存在しなければ KeyError で落ちる方がよい
    out: dict[str, float] = {
        "travel_single_rate":   float(feas["travel_single_rate"]),
        "travel_odd_rate":      float(feas["travel_odd_rate"]),
        "night_intrusion_rate": float(feas["night_intrusion"]),
        "switch_mean":          float(frag["switch_mean"]),
        "single_slot_ratio":    float(frag["single_slot_ratio"]),
        "wrap_closure_rate":    float(frag["wrap_closure_rate"]),
        "bigram_jsd":  float(im.bigram_jsd(sched_real, gen, sm.NUM_ACT, w_real, w_gen)),
        "switch_emd":  float(im.switch_dist_compare(sched_real, gen, w_real, w_gen)["emd"]),
        "separation_ratio": float(cd.separation_summary(
            sched_real, gen, d_real, gen_d, sm.NUM_ACT, sm.D_GROUPS,
            w_real, w_gen)["separation_ratio"]),
        # OTHER_X シェア（群等重み。§6 修正3 の基準に揃える）
        "other_x_share": float((gen == int(st.Common.OTHER_X)).mean()),
    }
    # zero-shot からの変化。λ パレート曲線の縦軸はこちらで、実データからの乖離ではない
    for k, base in ZERO_SHOT_GUARDRAILS.items():
        out[f"{k}_vs_zeroshot"] = out[k] / base if base else float("nan")
    return out


def evaluate_ckpt(path: Path, tgt: dict, sched_real: np.ndarray, d_real: np.ndarray,
                  w_real: np.ndarray, n: int, device: str) -> list[dict]:
    """1チェックポイントを2軸で測り、11act / 12act の2行を返す。"""
    model = sm.UNet1D().to(device)
    step, config = ck.load_ckpt(path, model, map_location=device)
    holdout = list(config.get("holdout", []))
    teacher_mask = np.ones(sm.D_GROUPS, dtype=bool)
    for d in holdout:
        teacher_mask[d] = False

    pool = sm.group_pool(model, n, verbose=False)                  # (28, n, 96)
    gen = pool.reshape(-1, sm.NUM_SLOTS)
    gen_d = np.repeat(np.arange(sm.D_GROUPS), n)

    rates = sm.pool_to_rates(pool)                                 # (28, 12*96) act-major
    guard = guardrails(gen, gen_d, sched_real, d_real, w_real)

    rows: list[dict] = []
    for mask_c, mask_name in ((st.mask_12act(), "12act"), (st.mask_11act(), "11act")):
        # 教師群と held-out 群を分けて測る。28群すべてが教師なら held-out 行は出ない
        for kind, sel in (("in-teacher", teacher_mask), ("held-out", ~teacher_mask)):
            if not sel.any():
                continue
            sub_tgt = {"group_rates_tbl": tgt["group_rates_tbl"][sel],
                       "pop": tgt["pop"].reshape(sm.D_GROUPS)[sel]}
            # ★採点の定義は stage2_targets.eval_against ただ一つ。群数は教師テンソルから
            #   読むので、教師群と held-out 群を同じ関数で測れる
            scores = st.eval_against(rates[sel], sub_tgt, mask_c)
            scores.pop("mask")          # mask は下の行で明示的に持たせる
            rows.append({"ckpt": path.name, "step": step,
                         "teacher_groups": int(teacher_mask.sum()),
                         "eval_kind": kind, "reference": "teacher", "mask": mask_name,
                         "n_per_group": n, **config, **scores, **guard})
    return rows


def run(ckpt_dir: Path, n: int = DEFAULT_N, out_csv: Path = OUT_CSV,
        device: str | None = None) -> pd.DataFrame:
    dev = device or sm.DEVICE
    paths = sorted([p for p in ckpt_dir.glob("stage2_step*.pt")],
                   key=lambda p: int(p.stem.removeprefix("stage2_step")))
    if not paths:
        raise SystemExit(f"ERROR: チェックポイントが無い: {ckpt_dir}")

    tgt = st.load_stula_targets()
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)

    rows: list[dict] = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name} を評価中 (28群 × {n} 本を生成) ...")
        rows.extend(evaluate_ckpt(path, tgt, sched_real, d_real, w_real, n, dev))
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n書き出し: {out_csv}")

    cols = ["step", "eval_kind", "rate_mae", "dev_rmse",
            "travel_single_rate", "switch_mean", "bigram_jsd", "other_x_share"]
    show = cast(pd.DataFrame, df[df["mask"] == "12act"])[cols]
    print("\n--- 軸1 教師適合 × 軸2 ガードレール（12act）---")
    print(show.round(5).to_string(index=False))
    print("\n★軸1 は teacher_groups=28 のとき循環している（学習目的そのもの）。"
          "\n  ガードレールの基準は実データ値ではなく zero-shot 値（§9.2）。")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2 の事後チェックポイント選択（教師適合 × ガードレール）")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "checkpoints" / "stage2")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="群あたり生成本数。評価用なので学習時の n と揃える必要はない")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    run(args.ckpt_dir, args.n, args.out_csv)


if __name__ == "__main__":
    main()
