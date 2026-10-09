"""
stage2.py
=========
GRU_Aggregate の Stage 2: 重みを固定し、日本へのずれ δ（属性の足し算）を logits に足して、
生成した群別の時刻別行動者率が教師 A* に合うよう δ だけを反復で補正する。
計画書: src/models/GRU_Aggregate/docs/stage2_plan.md

    logits_jp[d, s, :] = logits_gru(a_<s, c_d, s) + delta[d, s, :]
    delta[d, s, c]     = base[s, c] + sex[g_d, s, c] + age[a_d, s, c] + emp[e_d, s, c]
    （基準の水準 = 男・15-24・無業 は 0。1 セルあたり 9 個の係数）

実行（計画書 §4）:

    E0  --zero-shot      δ = 0。28 群と、LGO の各 fold の held-out 群で採点する（起点）
    E1  （引数なし）     28 群すべてを教師にして補正する
    E2  --fold K         fold K の 4 群を教師から外して補正する（held-out 群は δ の推定に入れない）
    --judge              E0〜E2 の CSV から判定 J1〜J6 を出す

採点は DDPM の Stage 2 と同じ関数（stage2_select.teacher_fit_rows / guardrails / memorization_guardrail /
published_metrics）を通し、同じ縦持ちの形式で CSV に書く。全国の曲線の MSE（national_curve_mse）だけ足す。

データフロー:

```mermaid
flowchart TD
    CK["gm.load_model(gm.ckpt_path(seed, calib_guidance=CALIB_G))<br/>重みは固定"] --> GP
    SH["shift = JapanShift.zeros()"] --> GP["gm.group_pool(model, FIT_POOL_N, GUIDANCE,<br/>group_bias=shift.to_group_bias())"]
    GP --> GEN["cur.pool_to_slot_rates(pool)<br/>gen (28, 12, 96)"]
    TGT["tgt['group_rates_tbl']<br/>A* (28, 12, 96)"] --> LR
    GEN --> LR["log_ratio = log((A* + EPS) / (gen + EPS))"]
    TM["teacher_mask (28,)"] --> FIT
    LR --> FIT["fit_additive(log_ratio, teacher_mask)<br/>(s, c) ごとに 9 係数の最小二乗"]
    FIT --> UP["shift = shift.added(fitted, STEP)"]
    UP --> GP
    UP --> EV["evaluate(model, shift, ...)<br/>EVAL_N 本/群、種 EVAL_SEED"]
    EV --> CSV["stage2_gru{接尾辞}_{run}.csv"]
    CSV --> JD["judge → J1〜J6"]
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/stage2.py --seed 42 --zero-shot
    .venv/bin/python src/models/GRU_Aggregate/stage2.py --seed 42
    .venv/bin/python src/models/GRU_Aggregate/stage2.py --seed 42 --fold 3
    .venv/bin/python src/models/GRU_Aggregate/stage2.py --judge
    .venv/bin/python src/models/GRU_Aggregate/stage2.py --smoke
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
SHIFT_DIR = REPO_ROOT / "outputs" / "checkpoints"

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名ファイルの取り違えを防ぐ）"""
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
sm: Any = gm.sm
sel: Any = _load("gru_s2_stage2_select", SIMPLE_DIR / "stage2_select.py")
cur: Any = _load("gru_s2_stage2_curves", SIMPLE_DIR / "stage2_curves.py")
lgo: Any = _load("gru_s2_stage2_lgo", SIMPLE_DIR / "stage2_lgo.py")
st: Any = sel.st

# ============================================================
# 設定（計画書 §3.2・§5。結果を見る前に固定）
# ============================================================
GUIDANCE = 1.25                  # 生成の CFG の強さ（Stage 1 の採用の候補 gru_calg125 と同じ）
CALIB_G = 1.25                   # 読み込む Stage 1 の ckpt（_calg1.25）
ITERS = 10                       # 補正の回数（途中で止めない）
STEP = 0.5                       # 更新の幅（Stage 1 の補正で 1.0 は振動、0.5 は収束）
FIT_POOL_N = 1024                # 補正の 1 回のプールの群あたりの本数
FIT_SEED = 30000                 # 補正の反復 k のプールの種は FIT_SEED + k
EPS = 1e-4                       # 行動者率 0 のセルで log を発散させない値
EVAL_N: int = sel.DEFAULT_N      # 評価のプールの群あたりの本数（2000、DDPM と同じ）
EVAL_SEED: int = sel.DEFAULT_POOL_SEED   # 評価のプールの種（12345、DDPM と同じ）
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
LGO_SEED = 42                    # E2 の種（DDPM の LGO の報告と同じく fold ごとに 1 本）
N_COEF = 1 + (sm.N_G - 1) + (sm.N_A - 1) + (sm.N_E - 1)     # 9

# 判定（計画書 §6）
J2_MIN_FOLDS = 6
J3_RATIO = 1.2
# J3 で「実 ATUS との距離」を測る指標。bigram_jsd と switch_emd は実 ATUS との距離そのもの
J3_DISTANCE_METRICS: tuple[str, ...] = ("bigram_jsd", "switch_emd")
J3_GAP_METRICS: tuple[str, ...] = ("switch_mean", "single_slot_ratio", "wrap_closure_rate",
                                    "night_intrusion_rate", "pairwise_hamming_std")
# DDPM の Stage 2（λ = 0.003・step 200）の出力（計画書 §6 の注）
DDPM_SELECTION_CSV = OUT_DIR / "stage2_lam0.003_selection.csv"
DDPM_FOLD_CSV = "stage2_lam0.003_fold{k}_selection.csv"
DDPM_STEP200_RATES = REPO_ROOT / "outputs" / "generated" / "stage2_step200_rates.npz"
DDPM_STEP = 200
# Stage 2 の方法 → 保存先の接尾辞。"shift" は δ を足す版（本ファイル）、"finetune" は重みを更新する版
# （stage2_finetune.py）、"finetune_v1" はその最初の版（リハーサルを g = 1.25 で引いた。結果の保存のみ）。
# E0（zero-shot）は Stage 1 のモデルそのものなので、どの方法でも "shift" の CSV を使う
METHOD_TAG: dict[str, str] = {"shift": "", "finetune": "_ft", "finetune_v1": "_ftv1"}


# ============================================================
# 日本へのずれ δ
# ============================================================
@dataclass(frozen=True)
class JapanShift:
    """属性の足し算で持つ日本へのずれ δ（基準の水準は 0）

    Attributes:
        base: 共通の成分, (96, 12)
        sex: 性の効果, (N_G, 96, 12)。男（0）は 0
        age: 年齢の効果, (N_A, 96, 12)。15-24（0）は 0
        emp: 就業の効果, (N_E, 96, 12)。無業（0）は 0
    """
    base: FloatArr
    sex: FloatArr
    age: FloatArr
    emp: FloatArr

    @staticmethod
    def zeros() -> "JapanShift":
        """δ = 0"""
        z = (sm.NUM_SLOTS, sm.NUM_ACT)
        return JapanShift(np.zeros(z), np.zeros((sm.N_G, *z)), np.zeros((sm.N_A, *z)), np.zeros((sm.N_E, *z)))

    def to_group_bias(self) -> FloatArr:
        """群ごとの δ, -> (28, 96, 12)。行 d は sm.cond_grid()[d] の属性の係数の和"""
        grid = sm.cond_grid()
        return np.asarray(self.base[None] + self.sex[grid[:, 0]] + self.age[grid[:, 1]] + self.emp[grid[:, 2]],
                          dtype=np.float64)

    def added(self, other: "JapanShift", scale: float) -> "JapanShift":
        """self + scale · other"""
        return JapanShift(self.base + scale * other.base, self.sex + scale * other.sex,
                          self.age + scale * other.age, self.emp + scale * other.emp)

    def save(self, path: Path, history: list[dict[str, float]]) -> None:
        """npz に保存する（反復の記録 history も一緒に）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, base=self.base, sex=self.sex, age=self.age, emp=self.emp,
                            history=np.array(pd.DataFrame(history).to_json()))


def design_matrix() -> FloatArr:
    """28 群の属性の設計行列 (28, N_COEF)。列は [1, 女, 年齢 1..6, 有業]（基準の水準は列を持たない）"""
    grid = sm.cond_grid()
    cols = [np.ones(sm.D_GROUPS), (grid[:, 0] == 1).astype(np.float64)]
    cols += [(grid[:, 1] == a).astype(np.float64) for a in range(1, sm.N_A)]
    cols += [(grid[:, 2] == 1).astype(np.float64)]
    return np.stack(cols, axis=1)


def coef_to_shift(coef: FloatArr) -> JapanShift:
    """設計行列の係数 (N_COEF, 96, 12) を JapanShift へ並べ直す"""
    z = np.zeros_like(coef[0])
    return JapanShift(base=coef[0], sex=np.stack([z, coef[1]]),
                      age=np.stack([z, *coef[2:2 + sm.N_A - 1]]), emp=np.stack([z, coef[1 + sm.N_A]]))


def fit_additive(log_ratio: FloatArr, use: BoolArr) -> JapanShift:
    """群ごとの log 比を、属性の足し算で最小二乗に当てはめる

    Note:
        ★(スロット, 活動) のセルごとに、use が True で A* が公表されている群だけで解く。
          held-out 群（use = False）は一度も入らない
        ★群を等しく扱う（重みなし）。主な指標 rate_mse_split がセルの単純平均だから
        ★使える群が少なく係数が決まらないセルでは、最小ノルム解（np.linalg.lstsq）になる

    Args:
        log_ratio: 群別の log((A* + EPS) / (gen + EPS)), (28, 12, 96)。非公表セルは NaN
        use: 当てはめに使う群, dtype=bool, (28,)

    Returns:
        当てはめた δ の増分
    """
    x = design_matrix()
    coef = np.zeros((N_COEF, sm.NUM_SLOTS, sm.NUM_ACT), dtype=np.float64)
    for c in range(sm.NUM_ACT):
        for s in range(sm.NUM_SLOTS):
            y = log_ratio[:, c, s]
            rows = use & ~np.isnan(y)
            if rows.any():
                coef[:, s, c] = np.linalg.lstsq(x[rows], y[rows], rcond=None)[0]
    return coef_to_shift(coef)


# ============================================================
# 補正の反復と評価
# ============================================================
def teacher_rate_mse(gen: FloatArr, tgt: dict, teacher_mask: BoolArr) -> float:
    """教師群・公表セルでの (gen − A*)² の平均（反復の記録用。生成の雑音を含む）"""
    a_star = tgt["group_rates_tbl"]
    m = teacher_mask[:, None, None] & ~np.isnan(a_star)
    return float(np.mean((gen - a_star)[m] ** 2))


def calibrate_to_teacher(model: Any, tgt: dict, teacher_mask: BoolArr, iters: int = ITERS,
                         n_per_group: int = FIT_POOL_N, seed: int = FIT_SEED,
                         verbose: bool = True) -> tuple[JapanShift, list[dict[str, float]]]:
    """生成した群別の時刻別行動者率が A* に合うよう、δ だけを反復で補正する（model は変えない）

    Args:
        model: Stage 1 の GRU（重みは固定）
        tgt: stage2_targets.load_stula_targets の戻り値
        teacher_mask: 教師に使う群, dtype=bool, (28,)
        iters: 補正の回数, default=ITERS=10
        n_per_group: 1 回のプールの群あたりの本数, default=FIT_POOL_N=1024
        seed: 反復 0 のプールの種, default=FIT_SEED=30000
        verbose: 反復ごとに print するか

    Returns:
        (補正した δ, 反復ごとの記録 [{"iter", "teacher_rate_mse"（補正の前に測った値）}])
    """
    a_star = tgt["group_rates_tbl"]
    shift = JapanShift.zeros()
    history: list[dict[str, float]] = []
    for k in range(iters):
        pool = gm.group_pool(model, n_per_group, GUIDANCE, seed=seed + k, group_bias=shift.to_group_bias())
        gen = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)          # (28, 12, 96)
        history.append({"iter": float(k), "teacher_rate_mse": teacher_rate_mse(gen, tgt, teacher_mask)})
        if verbose:
            print(f"stage2 iter {k} | teacher rate_mse {history[-1]['teacher_rate_mse']:.4e}", flush=True)
        log_ratio = np.log((a_star + EPS) / (gen + EPS))
        shift = shift.added(fit_additive(log_ratio, teacher_mask), STEP)
    return shift, history


def national_curve_mse(rates: FloatArr, tgt: dict) -> float:
    """全国の曲線の MSE: 28 群を日本人口で加重した曲線と、同じ加重をした教師の曲線（12 × 96 の平均）"""
    teacher = cur.weighted_slot_rates(tgt["group_rates_tbl"], tgt)
    return float(np.mean((cur.weighted_slot_rates(rates, tgt) - teacher) ** 2))


def fold_masks(tgt: dict) -> dict[str, BoolArr]:
    """LGO の fold ごとの教師の群（fold K の 4 群が False）。分割は stage2_lgo.stratified_folds（DDPM と同じ）"""
    out: dict[str, BoolArr] = {}
    for k, held in enumerate(lgo.stratified_folds(tgt["pop"])):
        mask = np.ones(sm.D_GROUPS, dtype=bool)
        mask[held] = False
        out[f"fold{k}"] = mask
    return out


def evaluate(model: Any, shift: JapanShift | None, tgt: dict, masks: dict[str, BoolArr], base: dict[str, Any],
             n_per_group: int = EVAL_N, full: bool = True) -> list[dict[str, Any]]:
    """評価のプールを 1 つ作り、3 つの軸で採点した縦持ちの行を返す（stage2_select と同じ形式）

    Args:
        model: Stage 1 の GRU
        shift: 日本へのずれ δ。None なら zero-shot
        tgt: 教師
        masks: 名前 → 教師の群 (28,)。名前ごとに in-teacher / held-out の行を作る（列 mask_name）
        base: 全行に付ける識別列
        n_per_group: 評価のプールの群あたりの本数（偶数）, default=EVAL_N=2000
        full: False なら軸 1 だけ（--smoke 用）

    Returns:
        行 list[dict]
    """
    if n_per_group % 2 != 0:
        raise ValueError(f"rate_mse_split には偶数の本数が要る: {n_per_group}")
    bias = None if shift is None else shift.to_group_bias()
    pool = gm.group_pool(model, n_per_group, GUIDANCE, seed=EVAL_SEED, group_bias=bias)
    return score_pool(pool, tgt, masks, base, full)


def score_pool(pool: npt.NDArray[np.int64], tgt: dict, masks: dict[str, BoolArr], base: dict[str, Any],
               full: bool = True) -> list[dict[str, Any]]:
    """評価のプールを 3 つの軸で採点した縦持ちの行を返す（stage2_select と同じ形式）

    Note:
        ★モデルに依存しない。LSTM_Aggregate/stage2_agg.py も自分のプールをここへ渡して、同じ形式の CSV を書く

    Args:
        pool: 評価のプール, dtype=int64, (28, n, 96)。n は偶数（rate_mse_split で半分ずつに分ける）
        tgt: 教師
        masks: 名前 → 教師の群 (28,)。名前ごとに in-teacher / held-out の行を作る（列 mask_name）
        base: 全行に付ける識別列
        full: False なら軸 1 だけ（--smoke 用）

    Returns:
        行 list[dict]

    Raises:
        ValueError: 群あたりの本数が奇数のとき
    """
    n_per_group = pool.shape[1]
    if n_per_group % 2 != 0:
        raise ValueError(f"rate_mse_split には偶数の本数が要る: {n_per_group}")
    half = n_per_group // 2
    rates, rates_a, rates_b = (sm.pool_to_rates(p) for p in (pool, pool[:, :half], pool[:, half:]))
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        rows += sel.teacher_fit_rows(rates, rates_a, rates_b, tgt, mask,
                                     {**base, "mask_name": name, "holdout": str(np.flatnonzero(~mask).tolist())})
    slot = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
    rows.append({**base, "mask_name": "all", "eval_kind": "all", "reference": "teacher",
                 "statistic": "national_curve", "mask": "12act", "weight_basis": "stula_pop",
                 "metric": "national_curve_mse", "value": national_curve_mse(slot, tgt)})
    if not full:
        return rows

    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)
    gen = pool.reshape(-1, sm.NUM_SLOTS)
    gen_d = np.repeat(np.arange(sm.D_GROUPS), n_per_group)
    guard = sel.guardrails(gen, gen_d, sched_real, d_real, w_real, seed=EVAL_SEED)
    guard.update(sel.memorization_guardrail(gen, sched_real, seed=EVAL_SEED))
    with contextlib.redirect_stdout(io.StringIO()):                            # 表を print する
        guard["memorized"] = float(bool(sm.memorization_report(gen, sched_real)["memorized"]))
    for metric, value in guard.items():
        if metric.endswith("_vs_zeroshot"):
            continue
        statistic, weight_basis = ("memorization", "atus_comp") if metric == "memorized" else sel.meta_of(metric)
        rows.append({**base, "mask_name": "all", "eval_kind": "all", "reference": "atus", "statistic": statistic,
                     "mask": "12act", "weight_basis": weight_basis, "metric": metric, "value": float(value)})
    for metric, value in sel.published_metrics(pool).items():
        statistic, weight_basis = sel.PUBLISHED_META[metric]
        rows.append({**base, "mask_name": "all", "eval_kind": "all", "reference": "stula_published",
                     "statistic": statistic, "mask": "12act", "weight_basis": weight_basis,
                     "metric": metric, "value": float(value)})
    return rows


# ============================================================
# 保存先と実行
# ============================================================
def run_name(zero_shot: bool, fold: int | None) -> str:
    """実行の名前: zeroshot（E0）/ all（E1）/ fold{K}（E2）"""
    if zero_shot:
        return "zeroshot"
    return "all" if fold is None else f"fold{fold}"


def csv_path(seed: int, run: str, method: str = "shift") -> Path:
    """評価の CSV のパス

    Args:
        seed: 学習の種
        run: zeroshot / all / fold{K}
        method: METHOD_TAG のキー（"shift" = δ を足す版、"finetune" = 重みを更新する版）
    """
    return OUT_DIR / f"stage2_gru{METHOD_TAG[method]}{gm.run_suffix(seed)}_{run}.csv"


def shift_path(seed: int, run: str) -> Path:
    """補正した δ の npz のパス"""
    return SHIFT_DIR / f"gru_stage2{gm.run_suffix(seed)}_{run}.npz"


def run(seed: int, fold: int | None, zero_shot: bool) -> None:
    """E0 / E1 / E2 を 1 本回して CSV（と δ）を書く"""
    tgt = st.load_stula_targets()
    model = gm.load_model(gm.ckpt_path(seed, calib_guidance=CALIB_G))
    folds = fold_masks(tgt)
    name = run_name(zero_shot, fold)
    base: dict[str, Any] = {"model": "gru_calg125", "seed": seed, "run": name, "iters": 0 if zero_shot else ITERS,
                            "guidance_scale": GUIDANCE, "n_per_group": EVAL_N, "pool_seed": EVAL_SEED}
    all_mask = np.ones(sm.D_GROUPS, dtype=bool)
    if zero_shot:
        shift, masks = None, {"all": all_mask, **folds}
    else:
        teacher = all_mask if fold is None else folds[f"fold{fold}"]
        shift, history = calibrate_to_teacher(model, tgt, teacher)
        shift.save(shift_path(seed, name), history)
        masks = {name: teacher}
    rows = evaluate(model, shift, tgt, masks, base)
    out = csv_path(seed, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[stage2] 書いた: {out}")


def smoke() -> None:
    """反復 1 回・群あたり 4 本で、補正と軸 1 の採点が通ることだけを確かめる（何も保存しない）"""
    tgt = st.load_stula_targets()
    model = gm.load_model(gm.ckpt_path(LGO_SEED, calib_guidance=CALIB_G))
    teacher = fold_masks(tgt)["fold0"]
    shift, history = calibrate_to_teacher(model, tgt, teacher, iters=1, n_per_group=4)
    rows = evaluate(model, shift, tgt, {"fold0": teacher}, {"run": "smoke"}, n_per_group=4, full=False)
    kinds = {r["eval_kind"] for r in rows}
    assert {"in-teacher", "held-out", "all"} <= kinds and len(history) == 1
    print(f"smoke: OK ({len(rows)} 行)")


# ============================================================
# 判定 J1〜J6（計画書 §6）
# ============================================================
def _value(df: pd.DataFrame, metric: str, **eq: Any) -> float:
    """列 = 値 の条件と metric で 1 行に絞って value を返す（1 行でなければ止める）"""
    mask = df["metric"].to_numpy() == metric
    for col, val in eq.items():
        mask &= df[col].to_numpy() == val
    hit = df.loc[mask, "value"].to_numpy(dtype=np.float64)
    if len(hit) != 1:
        raise ValueError(f"{metric} {eq} が {len(hit)} 行ある（1 行のはず）")
    return float(hit[0])


def ddpm_references(tgt: dict) -> dict[str, Any]:
    """DDPM の Stage 2（λ = 0.003・step 200）の比較値。計画書 §6 の注の出所から読む"""
    df = pd.read_csv(DDPM_SELECTION_CSV)
    step = df["step"].to_numpy() == DDPM_STEP
    sub = df.loc[step]
    ref: dict[str, Any] = {
        "rate_mse_split": _value(sub, "rate_mse_split", eval_kind="in-teacher", mask="12act"),
        "dev_rmse": _value(sub, "dev_rmse", eval_kind="in-teacher", mask="12act"),
        "national_curve_mse": national_curve_mse(cur.load_rates_npz(DDPM_STEP200_RATES), tgt),
        "heldout": {},
    }
    for k in range(lgo.N_FOLDS):
        f = pd.read_csv(OUT_DIR / DDPM_FOLD_CSV.format(k=k))
        f = f.loc[f["step"].to_numpy() == DDPM_STEP]
        ref["heldout"][k] = _value(f, "rate_mse_split", eval_kind="held-out", mask="12act")
    return ref


def judge(method: str = "shift", path_of: Callable[[int, str], Path] | None = None,
          out_path: Path | None = None, label: str = "gru") -> pd.DataFrame:
    """E0〜E2 の CSV から J1〜J6 を判定して表を返す（CSV にも書く）

    Note:
        ★LSTM_Aggregate/stage2_agg.py は path_of・out_path・label を渡して、自分の CSV を同じ規則で判定する

    Args:
        method: METHOD_TAG のキー。E1・E2 の CSV をこの方法のものから読む（E0 は共通）。path_of を渡すと使わない
        path_of: (種, 実行の名前 zeroshot / all / fold{K}) → 評価の CSV のパス。None なら csv_path（GRU）
        out_path: 判定の表の保存先。None なら stage2_gru{METHOD_TAG[method]}_judge.csv
        label: 判定するモデルの値を入れる列の名前, default="gru"

    Returns:
        判定の表（列 item・{label}・zero_shot・ddpm・pass）
    """
    def gru_path(seed: int, run: str) -> Path:
        return csv_path(seed, run, "shift" if run == "zeroshot" else method)

    resolve = gru_path if path_of is None else path_of
    tgt = st.load_stula_targets()
    ref = ddpm_references(tgt)
    zs = {s: pd.read_csv(resolve(s, "zeroshot")) for s in SEEDS}
    e1 = {s: pd.read_csv(resolve(s, "all")) for s in SEEDS}
    e2 = {k: pd.read_csv(resolve(LGO_SEED, f"fold{k}")) for k in range(lgo.N_FOLDS)}
    rows: list[dict[str, Any]] = []

    # J1・J2: held-out の rate_mse_split（種 42）
    better_zs, better_ddpm = 0, 0
    for k in range(lgo.N_FOLDS):
        g = _value(e2[k], "rate_mse_split", eval_kind="held-out", mask="12act", mask_name=f"fold{k}")
        z = _value(zs[LGO_SEED], "rate_mse_split", eval_kind="held-out", mask="12act", mask_name=f"fold{k}")
        better_zs += g < z
        better_ddpm += g < ref["heldout"][k]
        rows.append({"item": f"J1/J2 fold{k}", label: g, "zero_shot": z, "ddpm": ref["heldout"][k]})
    rows.append({"item": "J1 held-out が zero-shot より改善した fold 数", label: better_zs,
                 "pass": better_zs == lgo.N_FOLDS})
    rows.append({"item": "J2 held-out が DDPM より小さい fold 数", label: better_ddpm,
                 "pass": better_ddpm >= J2_MIN_FOLDS})

    # J3: 実 ATUS との距離（E1 と E0 の 5 本の中央値）
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)
    real_guard = sel.guardrails(sched_real, d_real, sched_real, d_real, w_real, seed=EVAL_SEED)

    def med(runs: dict[int, pd.DataFrame], metric: str, **eq: Any) -> float:
        return float(np.median([_value(df, metric, **eq) for df in runs.values()]))

    j3_ok = True
    for metric in (*J3_DISTANCE_METRICS, *J3_GAP_METRICS):
        def dist(runs: dict[int, pd.DataFrame]) -> float:
            vals = [_value(df, metric, eval_kind="all", reference="atus") for df in runs.values()]
            if metric in J3_DISTANCE_METRICS:
                return float(np.median(vals))
            return float(np.median([abs(v - float(real_guard[metric])) for v in vals]))
        g, z = dist(e1), dist(zs)
        ok = g <= J3_RATIO * z
        j3_ok &= ok
        rows.append({"item": f"J3 {metric}（実 ATUS との距離）", label: g, "zero_shot": z, "pass": ok})
    copies = sum(_value(df, "exact_copy_rate", eval_kind="all", reference="atus") > 0 for df in e1.values())
    memorized = sum(_value(df, "memorized", eval_kind="all", reference="atus") > 0 for df in e1.values())
    rows.append({"item": "J3 exact copy・暗記の本数", label: copies + memorized, "pass": copies + memorized == 0})
    rows.append({"item": "J3", "pass": j3_ok and copies + memorized == 0})

    # J4: 公表表（起床・就寝・日次行動者率）
    j4_ok = True
    for metric in ("bed_bias", "wake_bias", "exact_mae"):
        g = med(e1, metric, reference="stula_published")
        z = med(zs, metric, reference="stula_published")
        ok = abs(g) < abs(z)
        j4_ok &= ok
        rows.append({"item": f"J4 |{metric}|", label: g, "zero_shot": z, "pass": ok})
    rows.append({"item": "J4", "pass": j4_ok})

    # J5・J6: 28 群すべて（循環）
    g5 = med(e1, "national_curve_mse", statistic="national_curve")
    rows.append({"item": "J5 全国の曲線の MSE", label: g5, "zero_shot": med(zs, "national_curve_mse",
                 statistic="national_curve"), "ddpm": ref["national_curve_mse"],
                 "pass": g5 < ref["national_curve_mse"]})
    g6 = med(e1, "rate_mse_split", eval_kind="in-teacher", mask="12act")
    g6d = med(e1, "dev_rmse", eval_kind="in-teacher", mask="12act")
    rows.append({"item": "J6 28 群の rate_mse_split", label: g6,
                 "zero_shot": med(zs, "rate_mse_split", eval_kind="in-teacher", mask="12act", mask_name="all"),
                 "ddpm": ref["rate_mse_split"], "pass": g6 < ref["rate_mse_split"]})
    rows.append({"item": "J6 28 群の dev_rmse", label: g6d,
                 "zero_shot": med(zs, "dev_rmse", eval_kind="in-teacher", mask="12act", mask_name="all"),
                 "ddpm": ref["dev_rmse"], "pass": g6d < ref["dev_rmse"]})
    out = pd.DataFrame(rows)
    out.to_csv(out_path or OUT_DIR / f"stage2_gru{METHOD_TAG[method]}_judge.csv", index=False)
    return out


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GRU_Aggregate の Stage 2（日本へのずれ δ の補正）")
    ap.add_argument("--seed", type=int, default=LGO_SEED)
    ap.add_argument("--fold", type=int, default=None, help="教師から外す LGO の fold（0〜6）。省略で 28 群すべて")
    ap.add_argument("--zero-shot", action="store_true", help="δ = 0 で採点する（E0）")
    ap.add_argument("--judge", action="store_true", help="E0〜E2 の CSV から J1〜J6 を出す")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.zero_shot and args.fold is not None:
        ap.error("--zero-shot は --fold と併用できない（E0 は全 fold の held-out 群で採点する）")
    if args.fold is not None and not 0 <= args.fold < lgo.N_FOLDS:
        ap.error(f"--fold は 0〜{lgo.N_FOLDS - 1}: {args.fold}")
    if args.smoke:
        smoke()
    elif args.judge:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
            print(judge().to_string(index=False))
    else:
        run(args.seed, args.fold, args.zero_shot)


if __name__ == "__main__":
    main()
