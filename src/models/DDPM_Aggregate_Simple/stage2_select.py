"""
stage2_select.py
================
Stage 2 の事後チェックポイント選択（Stage2_design.md §8.4, §9.4, §9.8）

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
  既にノイズ床の外にあるので（§9.5）、「Stage 2 はガードレールを壊さない」という主張は
  使えない。使えるのは「悪化させない」「改善する」の2つ。

★出力 CSV は数値の出所を列で機械的に区別する（§9.8）。このリポジトリは出所の
  取り違えを2回起こしている。

    teacher_groups  損失に使った群数（28 or それ未満）
    eval_kind       in-teacher / held-out
    reference       teacher / atus       （何と比べた値か）
    mask            11act / 12act        （OTHER_X を含むか）
    pool_seed       プール生成の乱数種    （どの乱数列で測った値か）
    stage1_ckpt     出発点の Stage 1 重み （config 由来。どの重みから微調整したか）

★全 ckpt のプールは共通乱数（pool_seed）で作る。ck.load_ckpt は学習時の torch RNG を
  復元する副作用を持つので、seed を置き直さないと ckpt ごとに別の乱数列で生成される。
  n=2000 でもセル当たりの MC 標準偏差は最大 0.0112 あり、rate_mae の水準 0.0288 と
  同じ桁になるため、ckpt 間の差がモデル差か乱数差か区別できなくなる。

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

# 全チェックポイントのプールを同じ乱数列で作るための種（common random numbers）。
# ★固定が必須である。ck.load_ckpt は学習時の torch RNG を復元する副作用を持つので、
#   何もしないと ckpt ごとに別の乱数列で生成することになる。n=2000 でもセル当たりの
#   MC 標準偏差は σ=√(A*(1−A*)/n) で最大 0.0112 あり、rate_mae の実測水準 0.0288 と
#   同じ桁になる。共通乱数でなければ ckpt 間の差がモデル差か乱数差か区別できない。
DEFAULT_POOL_SEED = 12345

# 軸2 で追いかける指標と、その zero-shot 実測値（Stage2_design.md §9.5）。
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


def memorization_guardrail(gen: np.ndarray, sched_real: np.ndarray,
                           seed: int = 0) -> dict[str, float]:
    """暗記の診断。Stage 2 は ATUS 学習分割の上でさらに約23エポック回るので要る。

    Note:
        ★参照集合を同数に間引くことが要点。DCR_gap は「holdout への最近傍距離 −
        train への最近傍距離」だが、最近傍距離は参照集合が大きいほど小さくなる。
        ATUS 平日は train 3,363 / val 373 で 9 倍違うため、間引かないと
        **暗記が無くても gap が +4.7 出る**（過去に一度この交絡で誤読している）。
        両方を min(len(train), len(holdout)) まで同じ seed で間引いて比べる。
        ★train / holdout の分割は sm.split_indices ただ一つ。学習側と同じ規則で
        再現しないと「学習に使っていない個票」という前提が静かに壊れる。

    Args:
        gen: 生成スケジュール, dtype=int64, (M, 96)
        sched_real: 実 ATUS 平日のスケジュール全体, dtype=int64, (N, 96)
            学習側と同じ並びであること（sm.load_data の戻り値そのまま）
        seed: 参照集合の間引きに使う seed, default=0

    Returns:
        dict[str, float]
            dcr_train / dcr_holdout: 最近傍距離の平均（同数に間引いた参照集合に対して）
            dcr_gap: holdout − train。正で大きいほど暗記寄り
            exact_copy_rate: 学習個票と完全一致した生成の割合
    """
    train_idx, val_idx = sm.split_indices(len(sched_real))
    k = min(len(train_idx), len(val_idx))
    rng = np.random.default_rng(seed)
    tr = sched_real[rng.choice(train_idx, size=k, replace=False)]
    ho = sched_real[rng.choice(val_idx, size=k, replace=False)]
    m = im.memorization(gen, tr, ho, seed=seed)
    return {
        "dcr_train": float(m["DCR_mean[train]"]),
        "dcr_holdout": float(m["DCR_mean[holdout]"]),
        "dcr_gap": float(m["DCR_gap(holdout-train)"]),
        "exact_copy_rate": float(m["exact_copy_rate[train]"]),
        "n_ref_per_side": float(k),
    }


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
                  w_real: np.ndarray, n: int, device: str,
                  pool_seed: int = DEFAULT_POOL_SEED) -> list[dict]:
    """1チェックポイントを2軸で測り、11act / 12act の2行を返す。

    Args:
        path: 評価するチェックポイント（stage2_step*.pt）
        tgt: load_stula_targets の戻り値。28群ぶんの教師 A* と人口
        sched_real: 実 ATUS 平日のスケジュール, dtype=int64, (N, 96)
        d_real: 実 ATUS の群インデックス, dtype=int64, (N,)
        w_real: 実 ATUS の調査ウェイト, dtype=float64, (N,)
        n: 群あたりの生成本数 M。rate_mse_split が群内で二分するので偶数であること
        device: モデルを載せるデバイス
        pool_seed: プール生成の乱数種, default=DEFAULT_POOL_SEED

    Returns:
        1 ckpt ぶんの行 list[dict]。mask（12act/11act）× eval_kind（in-teacher/held-out）
        の組み合わせぶんだけ返る

    Note:
        ★torch.manual_seed を ck.load_ckpt の **後** に置くこと。load_ckpt は学習時の
        RNG を復元する副作用を持つので、先に seed を置くと上書きされてしまう。
    """
    # ★生成の前に落とす。1 ckpt の生成は 28群 × n 本で数十秒かかるので、
    #   払ってから弾くと掃引の本数ぶん無駄になる
    if n % 2 != 0:
        raise ValueError(f"split-batch 不偏推定には n が偶数である必要がある: {n}")

    model = sm.UNet1D().to(device)
    step, config = ck.load_ckpt(path, model, map_location=device)
    holdout = list(config.get("holdout", []))
    teacher_mask = np.ones(sm.D_GROUPS, dtype=bool)
    for d in holdout:
        teacher_mask[d] = False

    # 全 ckpt を同じ乱数列で生成する（common random numbers）
    torch.manual_seed(pool_seed)
    pool = sm.group_pool(model, n, verbose=False)                  # (28, n, 96)
    gen = pool.reshape(-1, sm.NUM_SLOTS)
    gen_d = np.repeat(np.arange(sm.D_GROUPS), n)

    rates = sm.pool_to_rates(pool)                                 # (28, 12*96) act-major
    # ★split-batch 不偏推定の材料（§9.4）。群の内側で前半・後半に割るので、
    #   2つの平均は独立で、かつ群をまたがない。群をまたいで割ると別の群の平均に
    #   なり、交差項が bias² を推定しなくなる（stage2_loss.group_rates_split と同じ理屈）
    half = n // 2
    rates_a = sm.pool_to_rates(pool[:, :half])                     # (28, 12*96)
    rates_b = sm.pool_to_rates(pool[:, half:])                     # (28, 12*96)
    guard = guardrails(gen, gen_d, sched_real, d_real, w_real)
    # ★暗記チェックは zero-shot 基準を持たない（ZERO_SHOT_GUARDRAILS に入れていない）。
    #   Stage 1 の実測が無いので比を出すと出所不明の数字になる。生の値で並べ、
    #   ckpt 間で dcr_gap が上がっていくかどうかを見る
    guard.update(memorization_guardrail(gen, sched_real, seed=pool_seed))

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
            scores = st.eval_against(rates[sel], sub_tgt, mask_c,
                                     (rates_a[sel], rates_b[sel]))
            scores.pop("mask")          # mask は下の行で明示的に持たせる
            rows.append({"ckpt": path.name, "step": step,
                         "teacher_groups": int(teacher_mask.sum()),
                         "eval_kind": kind, "reference": "teacher", "mask": mask_name,
                         "n_per_group": n, "pool_seed": pool_seed,
                         **config, **scores, **guard})
    return rows


def run(ckpt_dir: Path, n: int = DEFAULT_N, out_csv: Path = OUT_CSV,
        device: str | None = None, pool_seed: int = DEFAULT_POOL_SEED) -> pd.DataFrame:
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
        print(f"[{i}/{len(paths)}] {path.name} を評価中 "
              f"(28群 × {n} 本を生成, pool_seed={pool_seed}) ...")
        rows.extend(evaluate_ckpt(path, tgt, sched_real, d_real, w_real, n, dev, pool_seed))
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
          "\n  ガードレールの基準は実データ値ではなく zero-shot 値（§9.5）。"
          f"\n  全 ckpt は共通乱数 pool_seed={pool_seed} で生成してある"
          "（ckpt 間の差から生成の MC ノイズを除くため）。")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2 の事後チェックポイント選択（教師適合 × ガードレール）")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "checkpoints" / "stage2")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="群あたり生成本数。評価用なので学習時の n と揃える必要はない")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED,
                    help="プール生成の乱数種。全 ckpt に同じ値を使う（common random numbers）。"
                        "別のシードで測り直したいときだけ変える")
    args = ap.parse_args()
    run(args.ckpt_dir, args.n, args.out_csv, pool_seed=args.pool_seed)


if __name__ == "__main__":
    main()
