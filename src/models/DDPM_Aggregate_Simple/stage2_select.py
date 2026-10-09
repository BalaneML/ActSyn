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
from typing import Any

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
sp: Any = _load("select_schedule_plausibility",
                REPO_ROOT / "src" / "eval" / "schedule_plausibility.py")
# 非教師の公表統計（生活時間編・平均時刻編）。§9.7 / 実装項目 17
pb: Any = _load("select_stage2_published", HERE / "stage2_published.py")

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

# 軸2 の指標名 -> (statistic, weight_basis)。§9.8 が CSV に必ず持たせろと定めた2列。
#
# ★statistic 列が循環と非循環を機械的に分ける。教師（社会生活基本調査の時刻別
#   行動者率）が縛るのは slot_rate だけで、残りは教師から導出できない（§9.2）。
#   この区別を列に持たせないと、同じ「行動者率」という語が循環した値と非循環な値の
#   両方を指してしまう。
#
# ★設計書 §9.8 が例示する5値（slot_rate / daily_participation / derived_clock /
#   sequence / plausibility）に diversity / memorization / conditioning を足してある。
#   多様性（§9.5(b)）・暗記（§9.5(c)）・条件付け（§9.4）はいずれも5値のどれにも
#   収まらず、まとめると「非循環」としか言えなくなるため分けた。
#   daily_participation と derived_clock は実装項目 17（§9.7）が入ったときに使う。
#
# ★weight_basis 列が重みの取り違えを止める。同じ指標でも重み次第で値が変わる
#   （例: WORK の日次行動者率は非加重 0.4698 / TUFINLWGT 加重 0.5097）。
#     atus_comp : im.group_reweight で実 ATUS 平日の群構成へ重み付けして測った行
#     stula_pop : 日本の公表人口で重み付けして測った行（軸1 の dev_* など）
#     none      : 重みを使わない行
GUARDRAIL_META: dict[str, tuple[str, str]] = {
    # feasibility（§9.5(a)）
    "travel_single_rate":     ("plausibility", "atus_comp"),
    "travel_odd_rate":        ("plausibility", "atus_comp"),
    "night_intrusion_rate":   ("plausibility", "atus_comp"),
    # 断片化と系列（§9.5(a)）
    "switch_mean":            ("sequence", "atus_comp"),
    "single_slot_ratio":      ("sequence", "atus_comp"),
    "wrap_closure_rate":      ("sequence", "atus_comp"),
    "bigram_jsd":             ("sequence", "atus_comp"),
    "switch_emd":             ("sequence", "atus_comp"),
    # 条件付け（§9.4）
    "separation_ratio":       ("conditioning", "atus_comp"),
    # 活動シェア。教師と同種の統計量だが、比べる先は zero-shot であって教師ではない
    "other_x_share":          ("slot_rate", "none"),
    # 妥当性 12 指標（§9.6 / 実装項目 13）
    "sleep_holder_rate":      ("plausibility", "atus_comp"),
    "main_sleep_nocturnal":   ("plausibility", "atus_comp"),
    "main_sleep_share_ok":    ("plausibility", "atus_comp"),
    "main_sleep_share_mean":  ("plausibility", "atus_comp"),
    "main_sleep_mean_min":    ("plausibility", "atus_comp"),
    "work_holder_rate":       ("plausibility", "atus_comp"),
    "work_block_daytime":     ("plausibility", "atus_comp"),
    "work_span_daytime":      ("plausibility", "atus_comp"),
    "work_block_mean_min":    ("plausibility", "atus_comp"),
    "work_span_mean_min":     ("plausibility", "atus_comp"),
    "meals_count_ok":         ("plausibility", "atus_comp"),
    "meals_count_mean":       ("plausibility", "atus_comp"),
    # 多様性（§9.5(b)）。pairwise と group_dispersion は重みを取らないので none
    "pairwise_hamming_mean":  ("diversity", "none"),
    "pairwise_hamming_std":   ("diversity", "none"),
    "pairwise_hamming_iqr":   ("diversity", "none"),
    "group_hamming_mean":     ("diversity", "none"),
    "var_ratio_median":       ("diversity", "atus_comp"),
    "var_ratio_switches":     ("diversity", "atus_comp"),
    # 暗記（§9.5(c)）
    "dcr_train":              ("memorization", "none"),
    "dcr_holdout":            ("memorization", "none"),
    "dcr_gap":                ("memorization", "none"),
    "exact_copy_rate":        ("memorization", "none"),
    "n_ref_per_side":         ("memorization", "none"),
}


# 軸2 のうち**日本の公表値**と比べる行（§9.7 / 実装項目 17）。
# ★`reference = stula_published` として `atus` の行と分ける。同じ「行動者率」という語が
#   時刻別（教師・循環）と日次（公表・非循環）の両方を指すので、列で分けないと読めない。
# ★weight_basis は `stula_pop`。日本の公表値と比べる行なので日本の人口で重み付けする
#   （§9.7）。`atus_comp` で比べると群構成の差が行動の差に化ける。
PUBLISHED_META: dict[str, tuple[str, str]] = {
    # 日次行動者率（§9.7 の測る量 1）。1 対 1 の 7 活動と区間の 5 活動を分けて持つ
    "exact_mae":            ("daily_participation", "stula_pop"),
    "exact_max_abs":        ("daily_participation", "stula_pop"),
    "exact_n":              ("daily_participation", "none"),
    "union_outside_rate":   ("daily_participation", "stula_pop"),
    "union_max_gap":        ("daily_participation", "stula_pop"),
    "union_n":              ("daily_participation", "none"),
    # 派生時刻（§9.7 の測る量 2）。就寝は 0〜36 時の軸
    "wake_mae":             ("derived_clock", "stula_pop"),
    "wake_bias":            ("derived_clock", "stula_pop"),
    "wake_max_abs":         ("derived_clock", "stula_pop"),
    "wake_undef_gap":       ("derived_clock", "stula_pop"),
    "wake_n_groups":        ("derived_clock", "none"),
    "bed_mae":              ("derived_clock", "stula_pop"),
    "bed_bias":             ("derived_clock", "stula_pop"),
    "bed_max_abs":          ("derived_clock", "stula_pop"),
    "bed_undef_gap":        ("derived_clock", "stula_pop"),
    "bed_n_groups":         ("derived_clock", "none"),
}


def published_metrics(pool: np.ndarray) -> dict[str, float]:
    """生成プールを日本の公表値へ突き合わせる（§9.7）。

    Note:
        ★公表 CSV が無ければ空を返す。これらは評価の付加情報であって学習には要らず、
          配っていない計算機でも `stage2_select` が通る必要がある（LGO の eval は
          SQUID で走る）。黙って消えると気づけないので stderr に出す。
        ★教師とは無関係な量なので、教師群と held-out 群で分けない。全 28 群が
          そのまま非循環である（§9.2）。

    Args:
        pool: 群別サンプルプール, dtype=int64, (D, M, 96)。値は common12 ラベル

    Returns:
        指標の dict。公表 CSV が無ければ空
    """
    out: dict[str, float] = {}
    try:
        pub = pb.load_participation()
        out.update(pb.eval_participation(pb.participation_from_pool(pool), pub))
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: 日次行動者率の突合を飛ばす: {exc}", file=sys.stderr)
    try:
        mt = pb.load_mean_times()
        out.update(pb.eval_derived_times(pb.derived_times_from_pool(pool), mt))
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: 派生時刻の突合を飛ばす: {exc}", file=sys.stderr)
    return out


def meta_of(metric: str) -> tuple[str, str]:
    """軸2 の指標名から (statistic, weight_basis) を引く。

    ★未登録なら KeyError で落とす。既定値を返すと、指標を足したときに分類を
      忘れても黙って通り、循環／非循環の区別が壊れた CSV が出てしまう。

    Args:
        metric: 指標名。GUARDRAIL_META のキー

    Returns:
        (statistic, weight_basis)

    Raises:
        KeyError: GUARDRAIL_META に登録が無い指標
    """
    if metric not in GUARDRAIL_META:
        raise KeyError(
            f"{metric} が GUARDRAIL_META に無い。指標を足したら statistic と "
            f"weight_basis の分類も同時に決めること（§9.8）")
    return GUARDRAIL_META[metric]


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
               d_real: np.ndarray, w_real: np.ndarray,
               seed: int = 0) -> dict[str, float]:
    """軸2 のガードレール一式。生成側は実 ATUS 平日の群構成へ重み付けして測る。

    ★重みの出所は im.group_reweight ただ一つ。プールは群一様なので、非加重の人数比を
      使うと総変動距離で 0.139 ずれる。
    ★bigram_jsd と switch_emd は「重み付き・対角除く・行平均」で測る。重み無しだと
      別の値になり（0.0059 / 0.7765）、過去の記録に両方が混在しているので揃える。
    ★多様性（§9.5(b)）は報酬微調整で最も壊れやすい軸なので必ず 1 本入れる。DRaFT は
      報酬を上げ続けると出力が似通うことを実測しており、本リポジトリでも指数傾けで
      2,000 本のプールが実効 115 本（5.76%）まで痩せた。
    ★妥当性（§9.6）の 12 指標は sp.plausibility_summary をそのまま使う。窓・閾値は
      ATUS 実測から置いた値で、日本の実測ではない（Limitations に書く）。

    Args:
        gen: 生成スケジュール, dtype=int64, (M, 96)
        gen_d: 生成の群インデックス, dtype=int64, (M,)
        sched_real: 実 ATUS 平日のスケジュール, dtype=int64, (N, 96)
        d_real: 実 ATUS の群インデックス, dtype=int64, (N,)
        w_real: 実 ATUS の調査ウェイト, dtype=float64, (N,)
        seed: 多様性指標のペア標本抽出に使う seed, default=0

    Returns:
        指標名 -> 値。ZERO_SHOT_GUARDRAILS に基準がある指標は
        「指標名_vs_zeroshot」も併せて返す
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

    # --- 妥当性 12 指標（§9.6 / 実装項目 13）------------------------------
    # 教師は時刻別行動者率しか縛らないので、この層は教師から導出できない＝非循環。
    out.update({k: float(v) for k, v in
                sp.plausibility_summary(gen, w_gen).items()})

    # --- 多様性（§9.5(b) / 実装項目 14）-----------------------------------
    # ★pairwise は平均だけ見ても real / gen が 1% しか違わず判別できない。
    #   分布の広さ（std / iqr）が狭まったかどうかが多様性の崩壊を捉える。
    pair = im.pairwise_distance_dist(gen, seed=seed)
    out["pairwise_hamming_mean"] = float(pair["mean"])
    out["pairwise_hamming_std"] = float(pair["std"])
    out["pairwise_hamming_iqr"] = float(pair["iqr"])
    # 群内のばらつき。条件付けが強すぎて群が潰れていないかを見る（群で平均する）
    out["group_hamming_mean"] = float(
        im.group_dispersion(gen, gen_d, sm.D_GROUPS, seed=seed)["hamming_mean"].mean())
    # 個人別要約の散らばり。主役は var_ratio（生成の分散 / 実データの分散）。
    # ★平均ではなく中央値で縮約する。n_unique_act のように実データ側の分散が
    #   ほぼ 0 になる quantity があり、そこで var_ratio が 1e+26 まで発散して
    #   平均を支配してしまうため。switches は解釈しやすいので個別にも出す。
    cmp_disp = im.compare_dispersion(sched_real, gen, sm.NUM_ACT, sm.ACT_NAMES,
                                     w_real, w_gen)
    out["var_ratio_median"] = float(cmp_disp["var_ratio"].median())
    sw = cmp_disp.loc[cmp_disp["quantity"] == "switches", "var_ratio"]
    out["var_ratio_switches"] = float(sw.iloc[0]) if len(sw) else float("nan")

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
        1 ckpt ぶんの行 list[dict]。**1 指標 1 行の縦持ち**で、値は metric / value /
        vs_zeroshot に入り、出所は statistic / weight_basis / eval_kind / reference /
        mask が表す（§9.8）。横持ちだと循環した値（教師適合）と非循環な値
        （ガードレール）が同じ 1 行に混ざり、後から区別できない

    Note:
        ★torch.manual_seed を ck.load_ckpt の **後** に置くこと。load_ckpt は学習時の
        RNG を復元する副作用を持つので、先に seed を置くと上書きされてしまう。
    """
    # ★構造（時刻符号の有無）はチェックポイントの重みから決める。sm.UNet1D() 固定だと
    #   時刻符号つきの世代で load_state_dict が Unexpected key で落ちる
    model = sm.build_unet_for_ckpt(path).to(device)
    step, config = ck.load_ckpt(path, model, map_location=device)
    holdout = list(config.get("holdout", []))
    teacher_mask = np.ones(sm.D_GROUPS, dtype=bool)
    for d in holdout:
        teacher_mask[d] = False

    base = {"ckpt": path.name, "step": step,
            "teacher_groups": int(teacher_mask.sum()),
            "n_per_group": n, "pool_seed": pool_seed, **config}
    return evaluate_model(model, tgt, sched_real, d_real, w_real, n,
                          teacher_mask, base, pool_seed)


def evaluate_zeroshot(stage1_ckpt: Path, tgt: dict, sched_real: np.ndarray,
                      d_real: np.ndarray, w_real: np.ndarray, n: int, device: str,
                      pool_seed: int = DEFAULT_POOL_SEED) -> list[dict]:
    """微調整前の Stage 1 重みを同じ経路で測り、step=0 の基準線にする。

    ★なぜ定数 ZERO_SHOT_GUARDRAILS では足りないか。あちらに載っているのは
      feasibility と断片化の 10 指標だけで、妥当性 12（§9.6）と多様性 6（§9.5b）
      には zero-shot 実測が無い。§9.5 は「壊さない」ではなく「悪化させない／
      改善する」で主張を立てるとしており、多様性はパレート曲線の縦軸に必ず
      1 本入れると決めているので、基準線が無いと判定できない。
    ★さらに定数は n=256（28群 × 256 = 7,168 本）で測った値なので、評価既定の
      n=2000 の行とそのまま比べられない。同じ n・同じ pool_seed で測り直した
      この行が要る。
    ★Stage 1 ckpt は ck.load_ckpt が期待する形式（step / config / RNG を持つ）では
      ないので、sm.load_pretrained で読む。load_pretrained は torch の RNG を
      触らないが、evaluate_model が生成の直前に manual_seed を置くので、
      Stage 2 の世代と同じ乱数列になる（common random numbers が成立する）。

    Args:
        stage1_ckpt: Stage 1 の重み（ddpm_simple_pretrain_common12_weekday_*.pt）
        tgt: load_stula_targets の戻り値
        sched_real: 実 ATUS 平日のスケジュール, dtype=int64, (N, 96)
        d_real: 実 ATUS の群インデックス, dtype=int64, (N,)
        w_real: 実 ATUS の調査ウェイト, dtype=float64, (N,)
        n: 群あたりの生成本数 M。偶数であること
        device: モデルを載せるデバイス（load_pretrained は sm.DEVICE を使う）
        pool_seed: プール生成の乱数種, default=DEFAULT_POOL_SEED

    Returns:
        step=0 の行 list[dict]。Stage 2 の世代と同じ縦持ち形式なので、
        同じ CSV に並べてパレート曲線の起点にできる
    """
    model = sm.load_pretrained(stage1_ckpt).to(device)
    # zero-shot は教師を一度も見ていないが、軸1 の採点は 28 群すべてに対して行う
    # （§9.4 の zero-shot 基準線がまさにこの値）。したがって held-out 行は出ない
    teacher_mask = np.ones(sm.D_GROUPS, dtype=bool)
    base = {"ckpt": stage1_ckpt.name, "step": 0,
            "teacher_groups": int(teacher_mask.sum()),
            "n_per_group": n, "pool_seed": pool_seed,
            "holdout": [], "stage1_ckpt": stage1_ckpt.name, "lam": float("nan")}
    return evaluate_model(model, tgt, sched_real, d_real, w_real, n,
                          teacher_mask, base, pool_seed)


def make_pool(model: Any, n: int, pool_seed: int = DEFAULT_POOL_SEED
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """モデルから群別プールを作り、時刻別行動者率とその前半・後半を返す。

    evaluate_model と、教師を抜いた基準線を測る stage2_lgo.zeroshot_fold_rows の
    共通部分。**同じ関数を通すことが要点**で、経路が分かれると乱数列や二分の
    仕方がずれ、基準線と評価値の差がモデルの差でなくなる。

    Args:
        model: 生成に使うモデル
        n: 群あたりの生成本数 M。偶数であること
        pool_seed: プール生成の乱数種, default=DEFAULT_POOL_SEED

    Returns:
        (pool, rates, rates_a, rates_b)。pool は (D, M, 96) の common12 ラベル、
        rates は (D, 12*96) の時刻別行動者率、rates_a / rates_b は群内で
        前半 M/2 本・後半 M/2 本から作った同形の率

    Raises:
        ValueError: n が奇数（rate_mse_split が群内で二分できない）

    Note:
        ★torch.manual_seed は生成の直前に置く。呼び出し側の ck.load_ckpt は
        学習時の RNG を復元する副作用を持つので、先に置くと上書きされる。
    """
    # ★生成の前に落とす。1 モデルの生成は 28群 × n 本で数十秒かかるので、
    #   払ってから弾くと掃引の本数ぶん無駄になる
    if n % 2 != 0:
        raise ValueError(f"split-batch 不偏推定には n が偶数である必要がある: {n}")

    # ★経路でモードが変わらないようにする。sm.load_pretrained は eval() を呼ぶが
    #   sm.UNet1D() は train のままなので、揃えておかないと将来 dropout を足した
    #   ときに zero-shot 基準線とだけ値がずれる
    model.eval()

    # 全 ckpt を同じ乱数列で生成する（common random numbers）
    torch.manual_seed(pool_seed)
    pool = sm.group_pool(model, n, verbose=False)                  # (28, n, 96)
    rates = sm.pool_to_rates(pool)                                 # (28, 12*96) act-major
    # ★split-batch 不偏推定の材料（§9.4）。群の内側で前半・後半に割るので、
    #   2つの平均は独立で、かつ群をまたがない。群をまたいで割ると別の群の平均に
    #   なり、交差項が bias² を推定しなくなる（stage2_loss.group_rates_split と同じ理屈）
    half = n // 2
    rates_a = sm.pool_to_rates(pool[:, :half])                     # (28, 12*96)
    rates_b = sm.pool_to_rates(pool[:, half:])                     # (28, 12*96)
    return pool, rates, rates_a, rates_b


def teacher_fit_rows(rates: np.ndarray, rates_a: np.ndarray, rates_b: np.ndarray,
                     tgt: dict, teacher_mask: np.ndarray, base: dict) -> list[dict]:
    """軸1（教師適合）の行を作る。in-teacher と held-out を分けて測る。

    ★採点の定義をここ一箇所に閉じ込める。Stage 2 の世代・zero-shot 基準線・
      LGO の基準線がすべてこの関数を通るので、「held-out での改善」が
      定義の違いで出てしまう余地が無い。

    Args:
        rates: 生成の時刻別行動者率, (D, 12*96)
        rates_a: 群内前半から作った率, (D, 12*96)
        rates_b: 群内後半から作った率, (D, 12*96)
        tgt: load_stula_targets の戻り値。28群ぶんの教師 A* と人口
        teacher_mask: 教師に使った群が True, dtype=bool, (28,)
        base: 全行に付ける識別列（ckpt / step / holdout など）

    Returns:
        軸1 の行 list[dict]。28群すべてが教師なら held-out 行は出ない
    """
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
            scores.pop("mask")          # mask は行の列として明示的に持たせる
            for metric, value in scores.items():
                rows.append({**base, "eval_kind": kind, "reference": "teacher",
                             "statistic": "slot_rate", "mask": mask_name,
                             "weight_basis": "stula_pop", "metric": metric,
                             "value": float(value), "vs_zeroshot": float("nan")})
    return rows


def evaluate_model(model: Any, tgt: dict, sched_real: np.ndarray,
                   d_real: np.ndarray, w_real: np.ndarray, n: int,
                   teacher_mask: np.ndarray, base: dict,
                   pool_seed: int = DEFAULT_POOL_SEED) -> list[dict]:
    """モデルからプールを作り、2軸で測って縦持ちの行を返す。

    evaluate_ckpt（Stage 2 の世代）と evaluate_zeroshot（Stage 1 の基準線）の
    共通部分。**同じ関数を通すことが要点**で、経路が分かれると基準線と評価値が
    別の定義・別の乱数で作られ、比較が成り立たなくなる。

    Args:
        model: 生成に使うモデル
        tgt: load_stula_targets の戻り値
        sched_real: 実 ATUS 平日のスケジュール, dtype=int64, (N, 96)
        d_real: 実 ATUS の群インデックス, dtype=int64, (N,)
        w_real: 実 ATUS の調査ウェイト, dtype=float64, (N,)
        n: 群あたりの生成本数 M。偶数であること
        teacher_mask: 教師に使った群が True, dtype=bool, (28,)
        base: 全行に付ける識別列（ckpt / step / teacher_groups / config など）
        pool_seed: プール生成の乱数種, default=DEFAULT_POOL_SEED

    Returns:
        1 モデルぶんの行 list[dict]（1 指標 1 行の縦持ち）

    Raises:
        ValueError: n が奇数（rate_mse_split が群内で二分できない）

    Note:
        ★torch.manual_seed はこの関数の中、生成の直前に置く。呼び出し側の
        ck.load_ckpt は学習時の RNG を復元する副作用を持つので、先に置くと
        上書きされて ckpt ごとに別の乱数列になる。
    """
    pool, rates, rates_a, rates_b = make_pool(model, n, pool_seed)
    gen = pool.reshape(-1, sm.NUM_SLOTS)
    gen_d = np.repeat(np.arange(sm.D_GROUPS), n)

    guard = guardrails(gen, gen_d, sched_real, d_real, w_real, seed=pool_seed)
    # ★暗記チェックは zero-shot 基準を持たない（ZERO_SHOT_GUARDRAILS に入れていない）。
    #   Stage 1 の実測が無いので比を出すと出所不明の数字になる。生の値で並べ、
    #   ckpt 間で dcr_gap が上がっていくかどうかを見る
    guard.update(memorization_guardrail(gen, sched_real, seed=pool_seed))

    # --- 軸1: 教師適合。statistic=slot_rate は教師と同じ統計量＝循環している ---
    rows = teacher_fit_rows(rates, rates_a, rates_b, tgt, teacher_mask, base)

    # --- 軸2: ガードレール。教師が縛らない量＝非循環 ---
    # ★群で分けずにプール全体で測るので eval_kind は "all"、mask は 12act 固定。
    #   横持ちのときは mask × eval_kind の各行へ同じ値を複製していたが、縦持ちなら
    #   1 指標 1 行で重複しない
    for metric, value in guard.items():
        if metric.endswith("_vs_zeroshot"):
            continue                    # 生値の行の vs_zeroshot 列として載せる
        statistic, weight_basis = meta_of(metric)
        rows.append({**base, "eval_kind": "all", "reference": "atus",
                     "statistic": statistic, "mask": "12act",
                     "weight_basis": weight_basis, "metric": metric,
                     "value": float(value),
                     "vs_zeroshot": float(guard.get(f"{metric}_vs_zeroshot",
                                                    float("nan")))})

    # --- 軸2 の続き: 日本の公表値との突合（§9.7）。reference で atus の行と分ける ---
    # ★vs_zeroshot は入れない。zero-shot の実測は step=0 の行として同じ CSV に並ぶので、
    #   そちらと比べる（妥当性・多様性・暗記と同じ扱い）。
    for metric, value in published_metrics(pool).items():
        statistic, weight_basis = PUBLISHED_META[metric]
        rows.append({**base, "eval_kind": "all", "reference": "stula_published",
                     "statistic": statistic, "mask": "12act",
                     "weight_basis": weight_basis, "metric": metric,
                     "value": float(value), "vs_zeroshot": float("nan")})
    return rows


# 端末に出す要約表の列。CSV には全指標が入っているので、ここは「まず見る」ものだけ。
SUMMARY_AXIS1 = ("rate_mse_split", "rate_mae", "dev_rmse")
SUMMARY_AXIS2 = ("switch_mean", "travel_single_rate", "bigram_jsd",
                 "pairwise_hamming_std", "var_ratio_median", "dcr_gap")
# 日本の公表値との差（§9.7）。★`bed_bias` を先頭に置く。米 ATUS と日本の就寝は
# 28 群すべてで同じ向きに 0.81 時ずれており（§9.7.1）、この軸で最も判定力が高い。
SUMMARY_AXIS3 = ("bed_bias", "bed_mae", "wake_bias", "wake_mae",
                 "exact_mae", "union_outside_rate")


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """縦持ちの結果表から、端末に出す軸1・軸2 の要約表を作る。

    ★軸1 と軸2 を別の表にする。軸1 は eval_kind（in-teacher / held-out）で行が
      分かれるのに対し、軸2 は群で分けずプール全体で測る（eval_kind="all"）。
      1つの表に混ぜると pivot が意味の違う行を平均してしまう。

    Args:
        df: evaluate_ckpt が返した行の DataFrame（1 指標 1 行の縦持ち）

    Returns:
        (軸1 の表, 軸2 の表)。軸1 は index=(step, eval_kind)、軸2 は index=step、
        いずれも columns は SUMMARY_AXIS1 / SUMMARY_AXIS2 の順に並ぶ。
        該当する指標が 1 つも無ければ空の DataFrame を返す
    """
    ax1_src = df[(df["mask"] == "12act") & (df["statistic"] == "slot_rate")
                 & (df["metric"].isin(SUMMARY_AXIS1))]
    ax2_src = df[df["metric"].isin(SUMMARY_AXIS2)]
    ax1 = (ax1_src.pivot_table(index=["step", "eval_kind"], columns="metric",
                               values="value")
           .reindex(columns=[m for m in SUMMARY_AXIS1])
           if len(ax1_src) else pd.DataFrame())
    ax2 = (ax2_src.pivot_table(index="step", columns="metric", values="value")
           .reindex(columns=[m for m in SUMMARY_AXIS2])
           if len(ax2_src) else pd.DataFrame())
    return ax1, ax2


def summarize_published(df: pd.DataFrame) -> pd.DataFrame:
    """日本の公表値との差の要約表（§9.7）。

    Note:
        ★`summarize` とは別の関数にしてある。あちらは軸1（循環）と軸2（実データ基準）
          の 2 枚を返す契約で、呼び出し側とテストがその形に依存している。
        ★`reference` で絞る。指標名だけで拾うと、将来 `atus` 側に同名の指標が
          増えたときに黙って混ざる。

    Args:
        df: evaluate_ckpt が返した行の DataFrame（1 指標 1 行の縦持ち）

    Returns:
        index=step、columns=SUMMARY_AXIS3 の表。該当行が無ければ空
    """
    src = df[(df["reference"] == "stula_published")
             & (df["metric"].isin(SUMMARY_AXIS3))]
    if not len(src):
        return pd.DataFrame()
    return (src.pivot_table(index="step", columns="metric", values="value")
            .reindex(columns=[m for m in SUMMARY_AXIS3]))


def run(ckpt_dir: Path, n: int = DEFAULT_N, out_csv: Path = OUT_CSV,
        device: str | None = None, pool_seed: int = DEFAULT_POOL_SEED,
        stage1_ckpt: Path | None = None) -> pd.DataFrame:
    """ckpt_dir の全世代を 2 軸で採点し、縦持ちの CSV に落とす。

    Args:
        ckpt_dir: stage2_step*.pt が入っているディレクトリ
        n: 群あたりの生成本数。偶数であること, default=DEFAULT_N
        out_csv: 出力先, default=OUT_CSV
        device: モデルを載せるデバイス。None なら sm.DEVICE
        pool_seed: 全世代に共通の乱数種, default=DEFAULT_POOL_SEED
        stage1_ckpt: 微調整前の Stage 1 重み。渡すと step=0 の zero-shot 基準線を
            同じ n・同じ pool_seed で測って先頭に入れる。妥当性と多様性は
            ZERO_SHOT_GUARDRAILS に定数が無いので、これが無いと「悪化させて
            いないか」を判定できない, default=None

    Returns:
        全世代ぶんの行を連結した DataFrame（1 指標 1 行の縦持ち）
    """
    dev = device or sm.DEVICE
    paths = sorted([p for p in ckpt_dir.glob("stage2_step*.pt")],
                   key=lambda p: int(p.stem.removeprefix("stage2_step")))
    if not paths:
        raise SystemExit(f"ERROR: チェックポイントが無い: {ckpt_dir}")

    tgt = st.load_stula_targets()
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)

    rows: list[dict] = []
    # ★基準線を先に測る。Stage 2 の世代と同じ evaluate_model を通すので、
    #   定義も乱数列も揃う（別経路で測ると比較が成り立たない）
    if stage1_ckpt is not None:
        print(f"[0/{len(paths)}] zero-shot 基準線 {stage1_ckpt.name} を評価中 "
              f"(28群 × {n} 本を生成, pool_seed={pool_seed}) ...")
        rows.extend(evaluate_zeroshot(stage1_ckpt, tgt, sched_real, d_real,
                                      w_real, n, dev, pool_seed))
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name} を評価中 "
              f"(28群 × {n} 本を生成, pool_seed={pool_seed}) ...")
        rows.extend(evaluate_ckpt(path, tgt, sched_real, d_real, w_real, n, dev, pool_seed))
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n書き出し: {out_csv}")

    ax1_tbl, ax2_tbl = summarize(df)
    print("\n--- 軸1 教師適合（12act）★statistic=slot_rate は教師と同じ統計量 ---")
    print(ax1_tbl.round(6).to_string())
    print("\n--- 軸2 ガードレール（非循環）---")
    print(ax2_tbl.round(5).to_string())

    ax3_tbl = summarize_published(df)
    if len(ax3_tbl):
        print("\n--- 軸2 日本の公表値との差（非循環・§9.7）"
              " 単位は時 / 率 ---")
        print(ax3_tbl.round(4).to_string())
        print("  ★bed_bias が主。米 ATUS は日本より 0.81 時 早寝で、"
              "28 群すべて同じ向きである（§9.7.1）。"
              "\n    0 へ近づけば日本の就寝時刻へ転移したことになる。"
              "step=0 の行が出発点。")
    else:
        print("\n--- 軸2 日本の公表値との差: 行なし ---"
              "\n  data/processed/stula の timeuse_participation.csv /"
              " meantime_*.csv が無い。"
              "\n  parse_timeuse.py と parse_mean_time.py を流すこと（§9.7）")

    print("\n★軸1 は teacher_groups=28 のとき循環している（学習目的そのもの）。"
          "\n  非循環の証拠は --holdout-groups で群を抜いた held-out 行から取る。"
          "\n  パレート曲線の横軸は rate_mse_split（生成側の MC 雑音を抜いた二乗誤差、§9.4）。"
          "\n  ガードレールの基準は実データ値ではなく zero-shot 値で、vs_zeroshot 列に入る"
          "\n  （多様性・妥当性・暗記は zero-shot 実測が無いので生値のまま並ぶ）。"
          f"\n  全 ckpt は共通乱数 pool_seed={pool_seed} で生成してある"
          "（ckpt 間の差から生成の MC ノイズを除くため）。")
    return df


def dump_rates(ckpt: Path, out_npz: Path, n: int = DEFAULT_N,
               pool_seed: int = DEFAULT_POOL_SEED,
               device: str | None = None) -> dict[str, Any]:
    """1 チェックポイントの生成プールと時刻別行動者率を .npz へ保存する。

    **指標ではなく素材を残すための関数である。**`run` が書く CSV はスカラーの指標
    だけなので、後から別の重み付け（人口加重など）や別の統計量で測り直したくなると
    生成をやり直すしかない。プールを一度落としておけば、以後は CPU だけで済む。

    Note:
        ★`make_pool` を通す。`run` と同じ経路・同じ乱数なので、ここで落とした
          `rates` から計算した指標は CSV の値と一致する。別経路で生成すると
          common random numbers が崩れ、CSV と突き合わせられなくなる。
        ★`pool` は int8 で保存する。common12 は 0..11 なので情報は落ちない。
          int64 のままだと 28×1000×96 で 21.5 MB になるところが 2.7 MB で済む。
        ★Stage 1 の重みも読める。`step` キーの有無で Stage 2 の世代と区別する
          （`stage2_checkpoint.save_ckpt` は step / config を必ず入れる）。

    Args:
        ckpt: Stage 2 の世代（stage2_step*.pt）か Stage 1 の重み
        out_npz: 保存先。親ディレクトリが無ければ作る
        n: 群あたりの生成本数 M。偶数であること, default=DEFAULT_N
        pool_seed: プール生成の乱数種, default=DEFAULT_POOL_SEED
        device: モデルを載せるデバイス。None なら model.DEVICE, default=None

    Returns:
        保存した内容の要約 dict。`step` / `holdout` / `n` / `pool_seed` / `shape`
    """
    dev = device or sm.DEVICE
    raw = torch.load(ckpt, map_location=dev, weights_only=False)
    model = sm.UNet1D(arch=sm.arch_spec_from_ckpt(raw)).to(dev)
    model.load_state_dict(raw["model"])
    step = int(raw.get("step", 0))
    config: dict[str, Any] = dict(raw.get("config", {}))
    holdout = [int(d) for d in config.get("holdout", [])]

    pool, rates, rates_a, rates_b = make_pool(model, n, pool_seed)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        pool=pool.astype(np.int8),
        rates=rates, rates_a=rates_a, rates_b=rates_b,
        holdout=np.asarray(holdout, dtype=np.int64),
        meta=np.asarray([step, n, pool_seed], dtype=np.int64),
        ckpt=np.asarray(ckpt.name))
    info = {"step": step, "holdout": holdout, "n": n, "pool_seed": pool_seed,
            "shape": list(pool.shape), "out": str(out_npz),
            "bytes": out_npz.stat().st_size}
    print(f"[dump] {ckpt.name} step={step} holdout={holdout} "
          f"n={n} seed={pool_seed} -> {out_npz} ({info['bytes'] / 1e6:.1f} MB)")
    return info


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
    ap.add_argument("--stage1-ckpt", type=Path, default=None,
                    help="微調整前の Stage 1 重み。渡すと step=0 の zero-shot 基準線を "
                        "同じ n・同じ pool_seed で測って先頭に入れる。妥当性12 と "
                        "多様性6 は ZERO_SHOT_GUARDRAILS に定数が無いので、"
                        "これが無いと悪化したかを判定できない（§9.5）")
    ap.add_argument("--dump-rates", type=Path, default=None, metavar="CKPT",
                    help="指標を測らず、この ckpt の生成プールと時刻別行動者率を "
                        "--dump-out の .npz へ保存する。後から別の重み付けで測り直す "
                        "ための素材（stage2_curves.py --rates が読む）")
    ap.add_argument("--dump-out", type=Path, default=None,
                    help="--dump-rates の保存先 .npz。既定は outputs/generated/<ckpt名>_rates.npz")
    args = ap.parse_args()
    if args.dump_rates is not None:
        out = args.dump_out or (REPO_ROOT / "outputs" / "generated"
                                / f"{args.dump_rates.stem}_rates.npz")
        dump_rates(args.dump_rates, out, args.n, args.pool_seed)
        return
    run(args.ckpt_dir, args.n, args.out_csv, pool_seed=args.pool_seed,
        stage1_ckpt=args.stage1_ckpt)


if __name__ == "__main__":
    main()
