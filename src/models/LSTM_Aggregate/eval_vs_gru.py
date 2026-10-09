"""
eval_vs_gru.py
==============
LSTM 最低限と GRU 最低限（GRU_Minimal。LSTM とセルだけが違う GRU）の Stage 1 を、系列・総量・群・多様性・
val の交差エントロピーで並べ、判定 C1〜C3 と、セルの差の判定 D1・時刻符号の効果の判定 D2 をかける（米国加重）。
GRU H = 128 の 3 つは参考として同じ関数で測る。
時刻別行動者率の誤差の表は eval_curves.py が作る。D1・D2 の (a) はその種ごとの表を読むので、eval_curves.py を先に回す。

★比べるもの（ARM_SEEDS。最低限の構成の arm の定義は eval_curves.py の MINIMAL_MODULES）:
    lstm              LSTM 最低限（重みなしの損失）                                  種 MINIMAL_SEEDS
    gru_minimal       GRU 最低限（LSTM 最低限とセルだけが違う）                       種 MINIMAL_SEEDS
    {lstm,gru_minimal}_time_fixed    最低限 ＋ 固定の時刻符号（Transformer 型 φ ＋ Linear）  種 MINIMAL_SEEDS
    {lstm,gru_minimal}_time_learned  最低限 ＋ 学習型の時刻符号（Embedding(96, H)）         種 MINIMAL_SEEDS
    {lstm,gru_minimal}_time_learned{size_tag}  学習型で層数 {1,2} × 幅 {64,128} × wd {0.01,0.1,1} の 12 構成
                                               （ec.GRID_SIZES）  種 MINIMAL_SEEDS
    gru               GRU H = 128 補正前（g = 1.0、重み付きの損失）                   種 42〜46
    gru_no_slot_bias  GRU H = 128 slot_bias なし（g = 1.0）                           種 42〜46
    gru_calg125       GRU H = 128 補正後（g = 1.25 で補正・生成。採用候補）            種 42〜46
    ddpm_tf96         DDPM Transformer 型（g = 1.25）。C1 の基準（種間 sd）にだけ使う   種 42〜46
    ddpm_noclock      時刻符号なし DDPM（g = 1.25）。C3 の基準（種の最大）にだけ使う    種 42〜44

★1 つの生成プールで測るもの（pool_metrics。定義は stage1_gru_compare.py の guard_long・totals_long・group_long と同じ）:
    系列     switch_mean / single_slot_ratio / wrap_closure_rate / night_intrusion_rate、
             C3 の 4 指標（cmp.GUARD_METRICS）、暗記（dcr_gap / memorized）
    総量     5 活動（rd.FOCUS_ACTS）の総量の比 ratio_{活動} = 生成 / ATUS 実
    曲線     curve_mse_us（全体平均。判定・報告の本文には使わない）、meals_1200（12:00 の食事の行動者率）、
             share_work / share_housework（1 日のシェア）
    群       group_mse_atus、group_ratio_median（群ごとの MSE / 床 の中央値）、separation_ratio
    多様性   pairwise_hamming_mean / std / iqr、group_hamming_mean、var_ratio_median / var_ratio_switches
  ★多様性のうち pairwise_* と group_hamming_mean は組を重みなしで引く。生成プールは 28 群が同数なので、
    arm どうしは比べられるが、ATUS 実の値（標本の群構成）とは群の構成が違う。var_ratio_* は重み付き
★ckpt で測るもの（val_metrics）: val 373 人の交差エントロピーを、重み付き・重みなしの両方で測る。DDPM は NaN。
  あわせて、直前の活動を続ける予測確率の不足（val_stay_gap）と予測分布のエントロピー（val_entropy）を測る
  ★学習の損失は、最低限の 2 つが重みなし、GRU H = 128 が重み付き。両方の尺度で測れば、どの組も同じ尺度で比べられる

★判定 C1〜C3（judge_table。stage1_gru_compare と同じ定義・定数）:
    C1  5 活動のうち C_MIN_ACTS 以上で、総量の比の種間 sd ≤ ddpm_tf96 の種間 sd × C1_SD_RATIO
    C2  5 活動のうち C_MIN_ACTS 以上で、|総量の比の種平均 − 1| ≤ 床（rd.FLOOR_Z × ATUS の標本誤差の sd）
    C3  GUARD_METRICS の種の最大がどれも ddpm_noclock の種の最大以下で、暗記の判定が 0 本
★セルの差 D1（cell_table。結果を見る前に固定）: lstm と gru_minimal の種 5 本の範囲が重ならないことを「分離」と呼ぶ
    (a) 12 活動 × 3 つの量（平均誤差・MAE・最大 |誤差|、pt）。eval_curves.py の stage1_lstm_curve_seed.csv から
    (b) val の交差エントロピー（重みなし）・切替の回数・15 分で終わるエピソードの割合・昼食の山・群の分離
    better は種平均が目標に近い方（誤差と val は 0 に近い方、系列と昼食の山は ATUS 実、群の分離は完全なモデル）
★時刻符号の効果 D2（time_table。結果を見る前に固定）: セルごとに none↔fixed・none↔learned・fixed↔learned の 3 組を、
  D1 と同じ 41 項目・同じ「分離」と better で比べる
★事前の予測 P1〜P3（prediction_table。結果を見る前に固定。2026-10-06）:
    P1  fixed も learned も、12 活動の平均の MAE（種ごと）が none より小さい
    P2  12:00 の食事の |生成 − ATUS 実| は learned < fixed（1 スロット幅の段差を fixed の φ は線形には作れない）
    P3  switch_emd は learned > fixed（隣のスロットどうしの滑らかさの制約が無いぶん、切り替えが増える）
  判定は、種 5 本の範囲が分離して予測の向きなら「当たり」、分離して逆向きなら「外れ」、重なれば「分離せず」
★幅と weight decay の効果 D3（width_table。結果を見る前に固定）: セルごとに WIDTH_PAIRS の (幅, wd) の組を、
  D1 と同じ 41 項目・同じ「分離」と better で比べる。組は、両方を変えた H64・wd1 ↔ H128・wd0.01、
  幅だけの H64・wd1 ↔ H128・wd1、weight decay だけの H128・wd1 ↔ H128・wd0.1 と H128・wd1 ↔ H128・wd0.01
★事前の予測 P4〜P6（width_prediction_table。結果を見る前に固定。2026-10-06。根拠は GRU H = 128・2 層の掃引で
  wd = 0.1 が wd = 0 とほぼ同じに振る舞い、最良 epoch 49・最良の val が wd = 1 より 0.0056 高かったこと）:
    P4  最良 epoch は H = 128・wd = 0.01 の方が早い（過学習が早く始まる）
    P5  val の交差エントロピー（重みなし）は H = 128・wd = 0.01 の方が高い（下がらない）
    P6  最良の後の val の上がり幅（打ち切り時の val − 最良の val）は H = 128・wd = 0.01 の方が大きい
★事前の予測 P7〜P9（同じ関数。H = 128 の weight decay 0.01・0.1・1.0 の間。結果を見る前に固定。2026-10-06。
  根拠は同じ掃引で wd = 1 の val が最良・最良 epoch が wd = 0.1 の約 2 倍だったことと、
  H = 128・wd = 0.01 の生成が H = 64・wd = 1.0 より細切れでなかったこと）:
    P7  val の交差エントロピー（重みなし）は wd = 1.0 が最も低い（wd = 0.01 より、wd = 0.1 より低い）
    P8  最良 epoch は wd = 0.1 の方が wd = 1.0 より早い
    P9  15 分で終わるエピソードの割合は wd = 0.01 の方が wd = 1.0 より小さい（weight decay が強いと細切れになる）
★要因の効果 D4（factor_table・factor_summary。結果を見る前に固定。2026-10-06）: 層数 1→2・幅 64→128・
  weight decay 1→0.1・1→0.01・0.1→0.01 のそれぞれを、ほかの 2 つの要因をそろえた組（factor_pairs）で比べる。
  指標は FACTOR_METRICS。種 5 本の範囲が重ならなければ「分離」とし、向き（増・減）を数える
★事前の予測 P10・P11（width_prediction_table。結果を見る前に固定。2026-10-06。根拠は P7・P9 が H = 128・1 層で
  当たったこと）: 層数 × 幅の 4 つの組のどれでも
    P10 val の交差エントロピーは weight decay 1.0 の方が 0.01 より低い
    P11 15 分で終わるエピソードの割合は weight decay 0.01 の方が 1.0 より小さい
★学習の記録（training_table）: ckpt の config の history から、最良 epoch・打ち切り epoch・最良の val・
  最良の後の val の上がり幅・最良 epoch での train との差
★学習の目標のずれ（target_shift_table）: 群の中で重みなしにした ATUS 実の時刻別行動者率（最低限の 2 つが
  学習する目標）と、重み付きの値（評価の基準）の差。活動ごとの平均誤差・MAE・最大 |誤差|（pt）

データフロー:

```mermaid
flowchart TD
    SETUP["cmp.setup() / gm.load_split()"] --> REAL["load_real_data<br/>real: RealData"]
    LP["load_arm_pool(arm, seed)<br/>pool (28, M, 96)"] --> PM["pool_metrics(pool, real)"]
    LM["load_arm_model(arm, seed)"] --> VC["val_metrics(model, real.val_part)<br/>交差エントロピー・続ける確率の不足・エントロピー"]
    REAL --> PM
    REAL --> VC
    PM --> LONG["metrics_long → long<br/>stage1_lstm_vs_gru_long.csv"]
    VC --> LONG
    LONG --> REF["reference_values(real, long)<br/>stage1_lstm_vs_gru_ref.csv"]
    REAL --> REF
    LONG --> JT["judge_table(long, ref)<br/>stage1_lstm_vs_gru_judge.csv"]
    REF --> JT
    SEED["stage1_lstm_curve_seed.csv<br/>（eval_curves.py）"] --> CT["cell_table(long, seed_table, ref)<br/>stage1_lstm_vs_gru_cell.csv"]
    LONG --> CT
    REF --> CT
    SEED --> TT["time_table(long, seed_table, ref)<br/>stage1_lstm_vs_gru_time.csv"]
    LONG --> TT
    REF --> TT
    SEED --> PT["prediction_table(long, seed_table, ref)<br/>stage1_lstm_vs_gru_time_pred.csv"]
    LONG --> PT
    REF --> PT
    CKPT["ckpt の config['history']"] --> TRN["training_table()<br/>stage1_lstm_vs_gru_training.csv"]
    SEED --> WT["width_table(long, seed_table, ref)<br/>stage1_lstm_vs_gru_width.csv"]
    LONG --> WT
    TRN --> WP["width_prediction_table(long, training)<br/>P4〜P11 → stage1_lstm_vs_gru_width_pred.csv"]
    TRN --> SM["seed_metric_table(long, seed_table, training)"]
    LONG --> SM
    SEED --> SM
    SM --> FT["factor_table → factor_summary<br/>stage1_lstm_vs_gru_factor{,_summary}.csv"]
    LONG --> WP
    REAL --> TS["target_shift_table(real)<br/>stage1_lstm_vs_gru_target_shift.csv"]
```

使い方:
    .venv/bin/python src/models/LSTM_Aggregate/eval_curves.py     # 先に回す（D1・D2 の (a) の表を作る）
    .venv/bin/python src/models/LSTM_Aggregate/eval_vs_gru.py

出力: data/processed/aggregates/stage1_lstm_vs_gru_{long,ref,judge,cell,time,time_pred,training,width,width_pred,
      factor,factor_summary,target_shift}.csv
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/LSTM_Aggregate -> repo root
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
SEED_TABLE_CSV = OUT_DIR / "stage1_lstm_curve_seed.csv"   # eval_curves.py の出力

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
People = tuple[IntArr, IntArr, FloatArr]          # (sched, d, w)


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


cmp: Any = _load("stage1_gru_compare", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_gru_compare.py")
# 最低限の構成の arm の定義（MINIMAL_MODULES・minimal_paths）の出所
ec: Any = _load("lstm_eval_curves", Path(__file__).resolve().parent / "eval_curves.py")
sel: Any = cmp.sel
rd: Any = cmp.rd
cur: Any = cmp.cur
sm: Any = cmp.sm
agr: Any = cmp.agr
gm: Any = cmp.gm
sre: Any = cmp.sre

MINIMAL_SEEDS: tuple[int, ...] = ec.MINIMAL_SEEDS
# 最低限の構成の arm → ec.MinimalArm（保存先と load_model を持つモジュール・時刻符号・幅・weight decay）
MINIMAL_MODULES: dict[str, Any] = ec.MINIMAL_MODULES
C1_BASELINE_ARM = "ddpm_tf96"
C3_BASELINE_ARM = "ddpm_noclock"
ARM_SEEDS: dict[str, tuple[int, ...]] = {
    **{arm: MINIMAL_SEEDS for arm in MINIMAL_MODULES},
    "gru": cmp.ARM_SEEDS["gru"],
    "gru_no_slot_bias": cmp.ARM_SEEDS["gru_no_slot_bias"],
    "gru_calg125": cmp.ARM_SEEDS["gru_calg125"],
    C1_BASELINE_ARM: cmp.ARM_SEEDS[C1_BASELINE_ARM],
    C3_BASELINE_ARM: cmp.ARM_SEEDS[C3_BASELINE_ARM],
}
JUDGE_ARMS: tuple[str, ...] = (*MINIMAL_MODULES, "gru", "gru_no_slot_bias", "gru_calg125")
CELL_ARMS: tuple[str, str] = ("lstm", "gru_minimal")
# D2 で比べる時刻符号の組（セルごと）
TIME_PAIRS: tuple[tuple[str, str], ...] = (("none", "fixed"), ("none", "learned"), ("fixed", "learned"))
# D3 で比べる (幅, weight decay) の組（セルごと。時刻符号はどれも学習型・1 層）
Size = tuple[int, float]
GridSize = tuple[int, int, float]     # (層数, 幅, weight decay)。ec.GRID_SIZES の 1 つ
WIDTH_PAIRS: tuple[tuple[Size, Size], ...] = (
    ((64, 1.0), (128, 0.01)),         # 幅と weight decay を同時に変えた（最初の比較）
    ((64, 1.0), (128, 1.0)),          # 幅だけ（weight decay 1.0）
    ((128, 1.0), (128, 0.1)),         # weight decay だけ（H = 128）
    ((128, 1.0), (128, 0.01)))        # weight decay だけ（H = 128）
P4_P6_SIZE: Size = (128, 0.01)        # 予測 P4〜P6 をかける組（最初の比較のときに固定した）
# D4（要因の効果）で比べる指標（arm × 種の値。seed_metric_table が集める）
FACTOR_METRICS: tuple[str, ...] = ("val_ce_unweighted", "mae_pt_mean12", "single_slot_ratio", "switch_emd",
                                   "separation_ratio", "val_entropy", "best_epoch")
CELL_CURVE_QUANTITIES: tuple[str, ...] = ("bias_pt", "mae_pt", "max_abs_pt")
CELL_METRICS: tuple[str, ...] = ("val_ce_unweighted", "switch_mean", "single_slot_ratio", "meals_1200",
                                 "separation_ratio")
VAL_DEVICE = "cpu"                                # val の交差エントロピーは CPU で測る（決定性のため）
NOON_SLOT: int = int(round((12.0 - cur.SLOT_START_HOUR) * 60 / cur.SLOT_MINUTES))   # 32 = 12:00 の 15 分区間
MEALS: int = cur.ACT_NAMES.index("MEALS")
WORK: int = cur.ACT_NAMES.index("WORK")
HOUSEWORK: int = cur.ACT_NAMES.index("HOUSEWORK")


@dataclass(frozen=True)
class RealData:
    """ATUS 実の側の量（どの arm でも同じなので 1 回だけ作る）

    Attributes:
        people: ATUS 実 (sched, d, w)。model.load_data の行順
        pi_atus: 米国加重の群の重み, (28,)
        profile: ATUS 実の rd.profile_table（5 活動の総量）
        group_rates: 群別の時刻別行動者率, (28, 12, 96)
        curve_us: 米国加重の時刻別行動者率, (12, 96)
        group_floor: 群ごとの MSE の床, (28,)。cmp.group_floor
        fragmentation: im.fragmentation_summary（切替・1 スロットのエピソード・wrap_closure）
        val_part: gm.load_split の val 分割（373 人。cond_idx, sched, weight）
    """
    people: People
    pi_atus: FloatArr
    profile: pd.DataFrame
    group_rates: FloatArr
    curve_us: FloatArr
    group_floor: FloatArr
    fragmentation: dict[str, Any]
    val_part: Any


def load_real_data() -> RealData:
    """ATUS 実の側の量を作る"""
    people, pi_atus, profile = cmp.setup()
    sched_r, d_r, w_r = people
    group_rates = np.asarray(agr.group_rates(sched_r, d_r, w_r), dtype=np.float64)
    return RealData(people=people, pi_atus=pi_atus, profile=profile, group_rates=group_rates,
                    curve_us=cmp.us_weighted_slot_rates(group_rates, pi_atus),
                    group_floor=np.asarray(cmp.group_floor(people), dtype=np.float64),
                    fragmentation=dict(sel.im.fragmentation_summary(sched_r, w_r)),
                    val_part=gm.load_split()[1])


def load_arm_pool(arm: str, seed: int) -> IntArr:
    """arm と種の生成プール, -> (28, M, 96)。ckpt より古いプールは取り違えとして止める

    Raises:
        FileNotFoundError: 最低限の arm の生成プールが無いとき（GRU・DDPM は cmp.load_pool が投げる）
    """
    if arm not in MINIMAL_MODULES:
        return np.asarray(cmp.load_pool(arm, seed), dtype=np.int64)
    pool_file, ckpt_file = ec.minimal_paths(arm, seed)
    if not pool_file.exists():
        raise FileNotFoundError(f"{arm} の生成プールが無い (seed={seed}): {pool_file}")
    cmp.rep.check_fresh(pool_file, ckpt_file)
    return np.asarray(cur.load_sample_pool(pool_file), dtype=np.int64)


def load_arm_model(arm: str, seed: int) -> Any | None:
    """arm と種の最良の ckpt を VAL_DEVICE に eval モードで読む。DDPM の arm は None"""
    if arm in MINIMAL_MODULES:
        _, ckpt_file = ec.minimal_paths(arm, seed)
        return MINIMAL_MODULES[arm].module.load_model(ckpt_file, VAL_DEVICE)
    if arm.startswith("gru"):
        return gm.load_model(cmp.ckpt_file(arm, seed), VAL_DEVICE)
    return None


@torch.no_grad()
def val_metrics(model: Any, val_part: Any) -> dict[str, float]:
    """val 分割で、実データの履歴を入れたときの予測（teacher forcing）を測る

    Note:
        ★forward(a_prev, cond_idx) を持つモデルならどれでもよい（LSTMScheduler・GRUMinimalScheduler・
          GRUScheduler。GRUScheduler は条件付きの logits で、生成の g = 1.0 と同じ）
        ★交差エントロピーの重み付きは gm.weighted_ce と、重みなしは lm.batch_loss（既定）と同じ式
        ★val_stay_gap は、直前と同じ活動を続ける予測確率 p(a_s = a_{s−1}) の平均（s = 1〜95）から、
          実データで続く割合を引いた値。負なら続ける確率を低めに予測していて、生成で切り替えが増える。
          1 日の切り替えの機会は 95 回なので、生成の切り替えの回数は約 −95 × val_stay_gap だけ増える見込み
        ★val_entropy は予測分布のエントロピーの平均（nats）。大きいほど予測が平ら
        ★val_stay_gap・val_entropy は重みなし（人とスロットの単純平均）

    Args:
        model: VAL_DEVICE に載った eval モードのモデル
        val_part: gm.load_split の 2 つ目（cond_idx, sched, weight）

    Returns:
        {"val_ce_weighted", "val_ce_unweighted", "val_stay_gap", "val_entropy"}。交差エントロピーは 1 スロットあたりの nats
    """
    cond_idx = torch.as_tensor(val_part.cond_idx, dtype=torch.long, device=VAL_DEVICE)
    sched = torch.as_tensor(val_part.sched, dtype=torch.long, device=VAL_DEVICE)
    weight = torch.as_tensor(val_part.weight, dtype=torch.float64, device=VAL_DEVICE)
    logits = model(gm.shift_right(sched), cond_idx)                            # (N, 96, NUM_ACT)
    nll = F.cross_entropy(logits.reshape(-1, sm.NUM_ACT), sched.reshape(-1),
                          reduction="none").view_as(sched)                       # (N, 96)
    per_person = nll.mean(dim=1).double()                                        # (N,)
    prob = torch.softmax(logits.double(), dim=-1)                                # (N, 96, NUM_ACT)
    prev = sched[:, :-1]                                                         # 直前の活動, (N, 95)
    p_stay = prob[:, 1:, :].gather(-1, prev[..., None])[..., 0]                  # (N, 95)
    real_stay = (sched[:, 1:] == prev).double()
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)                  # (N, 96)
    return {"val_ce_weighted": float((weight * per_person).sum() / weight.sum()),
            "val_ce_unweighted": float(per_person.mean()),
            "val_stay_gap": float(p_stay.mean() - real_stay.mean()),
            "val_entropy": float(entropy.mean())}


def pool_metrics(pool: IntArr, real: RealData) -> dict[str, Any]:
    """1 つの生成プールの指標（米国加重）

    Note:
        1. sel.guardrails は、群が同数の生成プールを ATUS の群構成へ重み付けして測る（im.group_reweight）
        2. gap_* は |生成 − ATUS 実|、ratio_* は生成 / ATUS 実（stage1_gru_compare と同じ定義）
        3. group_ratio_median の床は群ごとの「完全なモデルでも出る MSE」（cmp.group_floor）

    Args:
        pool: 群別サンプルプール, dtype=int64, (28, M, 96)
        real: load_real_data の戻り値

    Returns:
        指標名 → 値（memorized だけ bool）
    """
    sched_r, d_r, w_r = real.people
    n_d, m, n_s = pool.shape
    gen = pool.reshape(n_d * m, n_s)
    gen_d = np.repeat(np.arange(n_d), m)
    g = sel.guardrails(gen, gen_d, sched_r, d_r, w_r)
    with contextlib.redirect_stdout(io.StringIO()):        # memorization_report は表を print する
        mem = sm.memorization_report(gen, sched_r)
    rates = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)              # (28, 12, 96)
    curve_us = cmp.us_weighted_slot_rates(rates, real.pi_atus)                     # (12, 96)
    group_mse = ((rates - real.group_rates) ** 2).mean(axis=(1, 2))                 # (28,)
    profile = rd.profile_table(*rd.pool_people(pool), real.pi_atus)
    frag = real.fragmentation
    out: dict[str, Any] = {
        "switch_mean": float(g["switch_mean"]),
        "single_slot_ratio": float(g["single_slot_ratio"]),
        "wrap_closure_rate": float(g["wrap_closure_rate"]),
        "night_intrusion_rate": float(g["night_intrusion_rate"]),
        "switch_emd": float(g["switch_emd"]),
        "bigram_jsd": float(g["bigram_jsd"]),
        "gap_single_slot_ratio": abs(float(g["single_slot_ratio"]) - float(frag["single_slot_ratio"])),
        "gap_wrap_closure_rate": abs(float(g["wrap_closure_rate"]) - float(frag["wrap_closure_rate"])),
        "dcr_gap": float(mem["DCR_gap(holdout-train)"]),
        "memorized": bool(mem["memorized"]),
        "curve_mse_us": float(np.mean((real.curve_us - curve_us) ** 2)),
        "meals_1200": float(curve_us[MEALS, NOON_SLOT]),
        "share_work": float(curve_us[WORK].mean()),
        "share_housework": float(curve_us[HOUSEWORK].mean()),
        "group_mse_atus": float(np.nanmean((real.group_rates - rates) ** 2)),
        "group_ratio_median": float(np.median(group_mse / real.group_floor)),
        "separation_ratio": float(g["separation_ratio"]),
        "pairwise_hamming_mean": float(g["pairwise_hamming_mean"]),
        "pairwise_hamming_std": float(g["pairwise_hamming_std"]),
        "pairwise_hamming_iqr": float(g["pairwise_hamming_iqr"]),
        "group_hamming_mean": float(g["group_hamming_mean"]),
        "var_ratio_median": float(g["var_ratio_median"]),
        "var_ratio_switches": float(g["var_ratio_switches"]),
    }
    for a in rd.FOCUS_ACTS:
        out[f"ratio_{a}"] = float(profile.loc[a, "level"] / real.profile.loc[a, "level"])
    return out


def metrics_long(real: RealData) -> pd.DataFrame:
    """(arm, 種) ごとの pool_metrics と val の交差エントロピー, 行 = arm × 種"""
    rows = []
    for arm, seeds in ARM_SEEDS.items():
        for seed in seeds:
            row: dict[str, Any] = {"arm": arm, "seed": seed, **pool_metrics(load_arm_pool(arm, seed), real)}
            model = load_arm_model(arm, seed)
            if model is not None:
                row.update(val_metrics(model, real.val_part))
            else:                                                                # DDPM は teacher forcing の予測を持たない
                row.update({k: float("nan") for k in ("val_ce_weighted", "val_ce_unweighted", "val_stay_gap",
                                                      "val_entropy")})
            rows.append(row)
    return pd.DataFrame(rows)


def reference_values(real: RealData, long: pd.DataFrame) -> pd.DataFrame:
    """比べるときの参照値（ATUS 実・床・C3 の基準）

    Args:
        real: load_real_data の戻り値
        long: metrics_long の戻り値（C3 の基準 = C3_BASELINE_ARM の種の最大を取る）

    Returns:
        列 name / value / note
    """
    sched_r, d_r, w_r = real.people
    frag = real.fragmentation
    pair = sel.im.pairwise_distance_dist(sched_r, seed=0)
    group_disp = sel.im.group_dispersion(sched_r, d_r, sm.D_GROUPS, seed=0)
    night = sel.fe.feasibility_summary(sched_r, w_r)["night_intrusion"]
    sep = np.asarray(cmp.separation_reference(real.people), dtype=np.float64)
    floor = rd.bootstrap_floor(real.people, real.pi_atus)
    rows: list[tuple[str, float, str]] = [
        ("atus_switch_mean", float(frag["switch_mean"]), "ATUS 実（米国加重）"),
        ("atus_single_slot_ratio", float(frag["single_slot_ratio"]), "ATUS 実（米国加重）"),
        ("atus_wrap_closure_rate", float(frag["wrap_closure_rate"]), "ATUS 実（米国加重）"),
        ("atus_night_intrusion_rate", float(night), "ATUS 実（米国加重）"),
        ("atus_meals_1200", float(real.curve_us[MEALS, NOON_SLOT]), "ATUS 実（米国加重）"),
        ("atus_share_work", float(real.curve_us[WORK].mean()), "ATUS 実（米国加重）"),
        ("atus_share_housework", float(real.curve_us[HOUSEWORK].mean()), "ATUS 実（米国加重）"),
        ("atus_pairwise_hamming_mean", float(pair["mean"]), "ATUS 実（組は重みなし・標本の群構成）"),
        ("atus_pairwise_hamming_std", float(pair["std"]), "ATUS 実（組は重みなし・標本の群構成）"),
        ("atus_pairwise_hamming_iqr", float(pair["iqr"]), "ATUS 実（組は重みなし・標本の群構成）"),
        ("atus_group_hamming_mean", float(group_disp["hamming_mean"].mean()), "ATUS 実（組は重みなし）"),
        ("separation_perfect_mean", float(sep.mean()), "完全なモデルの separation_ratio（20 回）"),
        ("separation_perfect_min", float(sep.min()), "同上"),
        ("separation_perfect_max", float(sep.max()), "同上"),
    ]
    rows += [(f"floor_ratio_{a}", rd.FLOOR_Z * float(floor.loc[a, "level"]), "C2 の床（|総量の比 − 1| の上限）")
             for a in rd.FOCUS_ACTS]
    base = cmp._rows(long, arm=C3_BASELINE_ARM)
    rows += [(f"c3_{k}", float(base[k].max()), f"C3 の基準（{C3_BASELINE_ARM} の種の最大）")
             for k in cmp.GUARD_METRICS]
    return pd.DataFrame(rows, columns=["name", "value", "note"])


def judge_table(long: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """JUDGE_ARMS の判定 C1〜C3（stage1_gru_compare の totals_summary・judge_totals・guard_summary と同じ定義）

    Args:
        long: metrics_long の戻り値
        ref: reference_values の戻り値（C2 の床と C3 の基準）

    Returns:
        行 = arm。列 n_seeds / ratio_mean_{活動} / ratio_sd_{活動} / c1_count / c1_pass / c2_count / c2_pass /
        c3_{指標}_max / c3_{指標}_pass / memorized（暗記の判定の本数）/ c3_pass
    """
    base = dict(zip(ref["name"], ref["value"]))
    c1_ref = cmp._rows(long, arm=C1_BASELINE_ARM)
    rows = []
    for arm in JUDGE_ARMS:
        g = cmp._rows(long, arm=arm)
        row: dict[str, Any] = {"arm": arm, "n_seeds": len(g)}
        c1_count = c2_count = 0
        for a in rd.FOCUS_ACTS:
            r = g[f"ratio_{a}"].to_numpy(dtype=np.float64)
            sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
            sd_ref = float(c1_ref[f"ratio_{a}"].to_numpy(dtype=np.float64).std(ddof=1))
            row[f"ratio_mean_{a}"], row[f"ratio_sd_{a}"] = float(r.mean()), sd
            c1_count += int(sd <= cmp.C1_SD_RATIO * sd_ref)
            c2_count += int(abs(float(r.mean()) - 1.0) <= base[f"floor_ratio_{a}"])
        row.update({"c1_count": c1_count, "c1_pass": c1_count >= cmp.C_MIN_ACTS,
                    "c2_count": c2_count, "c2_pass": c2_count >= cmp.C_MIN_ACTS})
        guard_pass = []
        for k in cmp.GUARD_METRICS:
            worst = float(g[k].max())
            row[f"c3_{k}_max"], row[f"c3_{k}_pass"] = worst, worst <= base[f"c3_{k}"]
            guard_pass.append(worst <= base[f"c3_{k}"])
        row["memorized"] = int(g["memorized"].sum())
        row["c3_pass"] = all(guard_pass) and row["memorized"] == 0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_ranges(part: str, item: str, quantity: str, values_a: FloatArr, values_b: FloatArr,
                   target: float) -> dict[str, Any]:
    """CELL_ARMS の 2 つの種の範囲を比べる 1 行（D1）

    Args:
        part: "a"（時刻別行動者率）か "b"（系列と群）
        item: 活動名か指標名
        quantity: 量の名前（a は bias_pt / mae_pt / max_abs_pt、b は指標名）
        values_a: CELL_ARMS[0] の種ごとの値
        values_b: CELL_ARMS[1] の種ごとの値
        target: 良い値（種平均がこれに近い方を better とする）

    Returns:
        列 part / item / quantity / {arm}_min / {arm}_max / {arm}_mean / separated / better
    """
    arm_a, arm_b = CELL_ARMS
    return {"part": part, "item": item, "quantity": quantity,
            f"{arm_a}_min": float(values_a.min()), f"{arm_a}_max": float(values_a.max()),
            f"{arm_a}_mean": float(values_a.mean()),
            f"{arm_b}_min": float(values_b.min()), f"{arm_b}_max": float(values_b.max()),
            f"{arm_b}_mean": float(values_b.mean()),
            "separated": is_separated(values_a, values_b),
            "better": closer_arm(arm_a, values_a, arm_b, values_b, target)}


def is_separated(values_a: FloatArr, values_b: FloatArr) -> bool:
    """2 つの arm の種ごとの値の範囲 [min, max] が重ならないか（D1・D2 の「分離」）"""
    return bool(values_a.max() < values_b.min() or values_b.max() < values_a.min())


def closer_arm(arm_a: str, values_a: FloatArr, arm_b: str, values_b: FloatArr, target: float) -> str:
    """種平均が target に近い方の arm（D1・D2 の better）。同じ距離なら arm_b"""
    dist_a, dist_b = abs(float(values_a.mean()) - target), abs(float(values_b.mean()) - target)
    return arm_a if dist_a < dist_b else arm_b


def metric_targets(ref: pd.DataFrame) -> dict[str, float]:
    """D1・D2 の (b) の指標ごとの目標（better を決める値）

    Args:
        ref: reference_values の戻り値

    Returns:
        CELL_METRICS の指標名 → 目標の値
    """
    base = dict(zip(ref["name"], ref["value"]))
    return {"val_ce_unweighted": 0.0, "switch_mean": base["atus_switch_mean"],
            "single_slot_ratio": base["atus_single_slot_ratio"], "meals_1200": base["atus_meals_1200"],
            "separation_ratio": base["separation_perfect_mean"]}


def cell_table(long: pd.DataFrame, seed_table: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """セルの差の判定 D1 の表（(a) 時刻別行動者率、(b) 系列と群）

    Args:
        long: metrics_long の戻り値
        seed_table: eval_curves.py の stage1_lstm_curve_seed.csv（arm × 種 × 活動）
        ref: reference_values の戻り値（(b) の目標）

    Returns:
        compare_ranges の行を並べた表
    """
    arm_a, arm_b = CELL_ARMS
    rows = []
    for act in cur.ACT_NAMES:
        for q in CELL_CURVE_QUANTITIES:
            va = cmp._rows(seed_table, arm=arm_a, activity=act)[q].to_numpy(dtype=np.float64)
            vb = cmp._rows(seed_table, arm=arm_b, activity=act)[q].to_numpy(dtype=np.float64)
            rows.append(compare_ranges("a", act, q, va, vb, target=0.0))
    targets = metric_targets(ref)
    for m in CELL_METRICS:
        va = cmp._rows(long, arm=arm_a)[m].to_numpy(dtype=np.float64)
        vb = cmp._rows(long, arm=arm_b)[m].to_numpy(dtype=np.float64)
        rows.append(compare_ranges("b", m, m, va, vb, target=targets[m]))
    return pd.DataFrame(rows)


def compare_pair(arm_a: str, values_a: FloatArr, arm_b: str, values_b: FloatArr, target: float) -> dict[str, Any]:
    """2 つの arm の種の範囲を比べる 1 行の値の部分（D2。列の名前は arm によらない）

    Returns:
        列 arm_a / a_min / a_max / a_mean / arm_b / b_min / b_max / b_mean / separated / better
    """
    return {"arm_a": arm_a, "a_min": float(values_a.min()), "a_max": float(values_a.max()),
            "a_mean": float(values_a.mean()),
            "arm_b": arm_b, "b_min": float(values_b.min()), "b_max": float(values_b.max()),
            "b_mean": float(values_b.mean()),
            "separated": is_separated(values_a, values_b),
            "better": closer_arm(arm_a, values_a, arm_b, values_b, target)}


def time_table(long: pd.DataFrame, seed_table: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """時刻符号の効果の判定 D2 の表（セル × TIME_PAIRS × D1 と同じ 41 項目）

    Args:
        long: metrics_long の戻り値
        seed_table: eval_curves.py の stage1_lstm_curve_seed.csv（arm × 種 × 活動）
        ref: reference_values の戻り値（(b) の目標）

    Returns:
        列 cell / pair（例: none-learned）/ part / item / quantity と compare_pair の列
    """
    targets = metric_targets(ref)
    rows = []
    for cell in ec.CELL_MODULES:
        for enc_a, enc_b in TIME_PAIRS:
            head = {"cell": cell, "pair": f"{enc_a}-{enc_b}"}
            rows += compare_items(head, ec.time_arm(cell, enc_a), ec.time_arm(cell, enc_b), long, seed_table, targets)
    return pd.DataFrame(rows)


def compare_items(head: dict[str, Any], arm_a: str, arm_b: str, long: pd.DataFrame, seed_table: pd.DataFrame,
                  targets: dict[str, float]) -> list[dict[str, Any]]:
    """2 つの arm を D1 と同じ 41 項目（(a) 12 活動 × 3 つの量、(b) CELL_METRICS）で比べる（D2・D3 で共有）

    Args:
        head: 各行の頭に付ける列（cell・pair など）
        arm_a: 比べる arm（列 arm_a）
        arm_b: 比べる arm（列 arm_b）
        long: metrics_long の戻り値
        seed_table: eval_curves.py の stage1_lstm_curve_seed.csv
        targets: metric_targets の戻り値（(b) の目標）

    Returns:
        41 行。列 head の列 / part / item / quantity と compare_pair の列
    """
    rows = []
    for act in cur.ACT_NAMES:
        for q in CELL_CURVE_QUANTITIES:
            va = cmp._rows(seed_table, arm=arm_a, activity=act)[q].to_numpy(dtype=np.float64)
            vb = cmp._rows(seed_table, arm=arm_b, activity=act)[q].to_numpy(dtype=np.float64)
            rows.append({**head, "part": "a", "item": act, "quantity": q,
                         **compare_pair(arm_a, va, arm_b, vb, target=0.0)})
    for m in CELL_METRICS:
        va = cmp._rows(long, arm=arm_a)[m].to_numpy(dtype=np.float64)
        vb = cmp._rows(long, arm=arm_b)[m].to_numpy(dtype=np.float64)
        rows.append({**head, "part": "b", "item": m, "quantity": m,
                     **compare_pair(arm_a, va, arm_b, vb, target=targets[m])})
    return rows


def width_table(long: pd.DataFrame, seed_table: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """幅と weight decay の効果の判定 D3 の表（セル × WIDTH_PAIRS × D1 と同じ 41 項目）

    Note:
        arm_a・arm_b は WIDTH_PAIRS の (幅, weight decay) の学習型の arm（ec.width_arm）

    Args:
        long: metrics_long の戻り値
        seed_table: eval_curves.py の stage1_lstm_curve_seed.csv
        ref: reference_values の戻り値

    Returns:
        列 cell / pair（例: h64_wd1-h128_wd0.01）/ part / item / quantity と compare_pair の列
    """
    targets = metric_targets(ref)
    rows = []
    for cell in ec.CELL_MODULES:
        for (h_a, wd_a), (h_b, wd_b) in WIDTH_PAIRS:
            head = {"cell": cell, "pair": f"h{h_a}_wd{wd_a:g}-h{h_b}_wd{wd_b:g}"}
            rows += compare_items(head, ec.width_arm(cell, h_a, wd_a), ec.width_arm(cell, h_b, wd_b),
                                  long, seed_table, targets)
    return pd.DataFrame(rows)


def training_table() -> pd.DataFrame:
    """最低限の構成の arm の学習の記録（ckpt の config の history から）

    Note:
        ★val は学習中（学習のデバイス）に測った値で、val_metrics（CPU で測り直す）とは別。
          train は epoch の途中で更新され続ける重みで測ったバッチ損失の平均なので、val と同じ条件の値ではない

    Returns:
        列 arm / seed / best_epoch / stopped_epoch / best_val / val_at_stop / train_at_best /
        rise_after_best（val_at_stop − best_val。最良の後に val が上がった幅）/ gap_at_best（best_val − train_at_best）
    """
    rows = []
    for arm in MINIMAL_MODULES:
        for seed in ARM_SEEDS[arm]:
            _, ckpt_file = ec.minimal_paths(arm, seed)
            cfg = torch.load(ckpt_file, map_location="cpu")["config"]
            val, train = cfg["history"]["val"], cfg["history"]["train"]
            best, stop = int(cfg["best_epoch"]), int(cfg["stopped_epoch"])
            rows.append({"arm": arm, "seed": seed, "best_epoch": best, "stopped_epoch": stop,
                         "best_val": float(val[best - 1]), "val_at_stop": float(val[stop - 1]),
                         "train_at_best": float(train[best - 1])})
    table = pd.DataFrame(rows)
    table["rise_after_best"] = table["val_at_stop"] - table["best_val"]
    table["gap_at_best"] = table["best_val"] - table["train_at_best"]
    return table


def judge_direction(small: FloatArr, large: FloatArr) -> str:
    """「small の方が小さい」という予測を、種の範囲で判定する

    Returns:
        "当たり"（small の最大 < large の最小）/ "外れ"（large の最大 < small の最小）/ "分離せず"
    """
    if small.max() < large.min():
        return "当たり"
    if large.max() < small.min():
        return "外れ"
    return "分離せず"


def prediction_table(long: pd.DataFrame, seed_table: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """事前の予測 P1〜P3 の判定（定義はモジュールの docstring）

    Note:
        P1 の量は、種ごとに 12 活動の MAE（mae_pt）を平均した値。P2 の量は、種ごとの |meals_1200 − ATUS 実|

    Args:
        long: metrics_long の戻り値
        seed_table: eval_curves.py の stage1_lstm_curve_seed.csv
        ref: reference_values の戻り値（ATUS 実の 12:00 の食事）

    Returns:
        列 cell / prediction / quantity / arm_small / small_min / small_max / small_mean（小さいと予測した arm）/
        arm_large / large_min / large_max / large_mean / verdict
    """
    atus_meals = float(dict(zip(ref["name"], ref["value"]))["atus_meals_1200"])

    def mae_mean(arm: str) -> FloatArr:
        g = cmp._rows(seed_table, arm=arm)
        return g.groupby("seed")["mae_pt"].mean().reindex(list(ARM_SEEDS[arm])).to_numpy(dtype=np.float64)

    def meals_gap(arm: str) -> FloatArr:
        return np.abs(cmp._rows(long, arm=arm)["meals_1200"].to_numpy(dtype=np.float64) - atus_meals)

    def switch_emd(arm: str) -> FloatArr:
        return cmp._rows(long, arm=arm)["switch_emd"].to_numpy(dtype=np.float64)

    rows = []
    for cell in ec.CELL_MODULES:
        none, fixed, learned = (ec.time_arm(cell, enc) for enc in ("none", "fixed", "learned"))
        checks = [("P1", "mae_pt_mean12", mae_mean, fixed, none),
                  ("P1", "mae_pt_mean12", mae_mean, learned, none),
                  ("P2", "meals_1200_abs_gap", meals_gap, learned, fixed),
                  ("P3", "switch_emd", switch_emd, fixed, learned)]
        for name, quantity, measure, arm_small, arm_large in checks:
            rows.append(prediction_row(cell, name, quantity, arm_small, measure(arm_small),
                                       arm_large, measure(arm_large)))
    return pd.DataFrame(rows)


def seed_metric_table(long: pd.DataFrame, seed_table: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """D4・P10・P11 で使う指標を、最低限の構成の arm × 種で 1 つの表にまとめる

    Note:
        mae_pt_mean12 は、種ごとの曲線の 12 活動の MAE（mae_pt）の平均。best_epoch は training_table から

    Returns:
        列 arm / seed と FACTOR_METRICS の列
    """
    mae = seed_table.groupby(["arm", "seed"], as_index=False).agg(mae_pt_mean12=("mae_pt", "mean"))
    cols = ["arm", "seed", *[m for m in FACTOR_METRICS if m in long.columns]]
    table = long.loc[long["arm"].isin(list(MINIMAL_MODULES)), cols]
    table = table.merge(mae, on=["arm", "seed"], how="left")
    return table.merge(training[["arm", "seed", "best_epoch"]], on=["arm", "seed"], how="left")


def factor_pairs() -> list[tuple[str, str, GridSize, GridSize]]:
    """D4 で比べる組（要因の変化, そろえた要因, 変える前, 変えた後）。どれも学習型の時刻符号

    Returns:
        層数 1→2（幅 × weight decay の 6 組）、幅 64→128（層数 × weight decay の 6 組）、
        weight decay 1→0.1・1→0.01・0.1→0.01（層数 × 幅の 4 組ずつ）
    """
    pairs: list[tuple[str, str, GridSize, GridSize]] = []
    for h in ec.GRID_HIDDENS:
        for wd in ec.GRID_WEIGHT_DECAYS:
            pairs.append(("層数 1→2", f"H{h}・wd{wd:g}", (1, h, wd), (2, h, wd)))
    for n in ec.GRID_LAYERS:
        for wd in ec.GRID_WEIGHT_DECAYS:
            pairs.append(("幅 64→128", f"{n} 層・wd{wd:g}", (n, 64, wd), (n, 128, wd)))
    for wd_a, wd_b in ((1.0, 0.1), (1.0, 0.01), (0.1, 0.01)):
        for n in ec.GRID_LAYERS:
            for h in ec.GRID_HIDDENS:
                pairs.append((f"wd {wd_a:g}→{wd_b:g}", f"{n} 層・H{h}", (n, h, wd_a), (n, h, wd_b)))
    return pairs


def factor_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """要因の効果の判定 D4 の表（セル × factor_pairs × FACTOR_METRICS）

    Args:
        metrics: seed_metric_table の戻り値

    Returns:
        列 cell / change / fixed / metric / arm_a / arm_b / a_mean / b_mean / diff（b − a）/
        separated（種の範囲が重ならない）/ direction（分離したときの向き: 増 / 減。分離しなければ —）
    """
    rows = []
    for cell in ec.CELL_MODULES:
        for change, fixed, size_a, size_b in factor_pairs():
            arm_a, arm_b = ec.grid_arm(cell, *size_a), ec.grid_arm(cell, *size_b)
            for m in FACTOR_METRICS:
                va = cmp._rows(metrics, arm=arm_a)[m].to_numpy(dtype=np.float64)
                vb = cmp._rows(metrics, arm=arm_b)[m].to_numpy(dtype=np.float64)
                separated = is_separated(va, vb)
                diff = float(vb.mean() - va.mean())
                rows.append({"cell": cell, "change": change, "fixed": fixed, "metric": m,
                             "arm_a": arm_a, "arm_b": arm_b, "a_mean": float(va.mean()), "b_mean": float(vb.mean()),
                             "diff": diff, "separated": separated,
                             "direction": ("増" if diff > 0 else "減") if separated else "—"})
    return pd.DataFrame(rows)


def factor_summary(table: pd.DataFrame) -> pd.DataFrame:
    """D4 の表を、セル × 要因の変化 × 指標ごとにまとめる

    Returns:
        列 cell / change / metric / n_pairs / n_up（分離して増えた組）/ n_down（分離して減った組）/ mean_diff
    """
    return (table.groupby(["cell", "change", "metric"], sort=False)
            .agg(n_pairs=("diff", "size"), n_up=("direction", lambda d: int((d == "増").sum())),
                 n_down=("direction", lambda d: int((d == "減").sum())), mean_diff=("diff", "mean"))
            .reset_index())


def prediction_row(cell: str, name: str, quantity: str, arm_small: str, small: FloatArr,
                   arm_large: str, large: FloatArr) -> dict[str, Any]:
    """「arm_small の方が小さい」という予測の 1 行（prediction_table・width_prediction_table で共有）

    Returns:
        列 cell / prediction / quantity / arm_small / small_min / small_max / small_mean /
        arm_large / large_min / large_max / large_mean / verdict（judge_direction）
    """
    return {"cell": cell, "prediction": name, "quantity": quantity,
            "arm_small": arm_small, "small_min": float(small.min()), "small_max": float(small.max()),
            "small_mean": float(small.mean()),
            "arm_large": arm_large, "large_min": float(large.min()), "large_max": float(large.max()),
            "large_mean": float(large.mean()),
            "verdict": judge_direction(small, large)}


def width_prediction_table(long: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """事前の予測 P4〜P11 の判定（定義はモジュールの docstring）

    Note:
        P4〜P6 は学習型の H = 64・wd = 1.0 と P4_P6_SIZE（H = 128・wd = 0.01）。
        P7〜P9 は H = 128 の weight decay 0.01・0.1・1.0 の間。
        P10・P11 は、層数 × 幅の 4 つの組ごとに weight decay 1.0 と 0.01 の間

    Args:
        long: metrics_long の戻り値（val の交差エントロピー・15 分で終わる割合）
        training: training_table の戻り値（最良 epoch・最良の後の val の上がり幅）

    Returns:
        prediction_row の列
    """
    def per_seed(table: pd.DataFrame, arm: str, column: str) -> FloatArr:
        return cmp._rows(table, arm=arm)[column].to_numpy(dtype=np.float64)

    def check(cell: str, name: str, table: pd.DataFrame, column: str, arm_small: str, arm_large: str) -> dict[str, Any]:
        return prediction_row(cell, name, column, arm_small, per_seed(table, arm_small, column),
                              arm_large, per_seed(table, arm_large, column))

    rows = []
    for cell in ec.CELL_MODULES:
        base = ec.time_arm(cell, "learned")
        wide = ec.width_arm(cell, *P4_P6_SIZE)
        h128 = {wd: ec.width_arm(cell, 128, wd) for wd in (0.01, 0.1, 1.0)}
        rows += [
            check(cell, "P4", training, "best_epoch", wide, base),
            check(cell, "P5", long, "val_ce_unweighted", base, wide),
            check(cell, "P6", training, "rise_after_best", base, wide),
            check(cell, "P7", long, "val_ce_unweighted", h128[1.0], h128[0.01]),
            check(cell, "P7", long, "val_ce_unweighted", h128[1.0], h128[0.1]),
            check(cell, "P8", training, "best_epoch", h128[0.1], h128[1.0]),
            check(cell, "P9", long, "single_slot_ratio", h128[0.01], h128[1.0])]
        for n in ec.GRID_LAYERS:
            for h in ec.GRID_HIDDENS:
                strong, weak = ec.grid_arm(cell, n, h, 1.0), ec.grid_arm(cell, n, h, 0.01)
                rows += [check(cell, "P10", long, "val_ce_unweighted", strong, weak),
                         check(cell, "P11", long, "single_slot_ratio", weak, strong)]
    return pd.DataFrame(rows)


def target_shift_table(real: RealData) -> pd.DataFrame:
    """学習の目標のずれ: 群の中で重みなしにした ATUS 実の曲線 − 重み付きの曲線（どちらも群の間は米国加重）

    Note:
        ★最低限の 2 つは重みなしで学習するので、完全に学習できても重みなしの曲線に近づく。この差の分は、
          評価（重み付きの曲線）で誤差として出うる

    Returns:
        列 activity / bias_pt / mae_pt / max_abs_pt / max_err_pt / max_time（sre.seed_error_table の列から seed を除いたもの）
    """
    sched_r, d_r, w_r = real.people
    unweighted = np.asarray(agr.group_rates(sched_r, d_r, np.ones_like(w_r)), dtype=np.float64)
    curve_unweighted = cmp.us_weighted_slot_rates(unweighted, real.pi_atus)        # (12, 96)
    return sre.seed_error_table(real.curve_us, curve_unweighted[None], cur.ACT_NAMES, [0]).drop(columns="seed")


def print_summary(long: pd.DataFrame, judge: pd.DataFrame, cell: pd.DataFrame, shift: pd.DataFrame,
                  time_effect: pd.DataFrame, pred: pd.DataFrame, width: pd.DataFrame, width_pred: pd.DataFrame,
                  training: pd.DataFrame) -> None:
    """主な表を表示する"""
    metrics: list[str] = ["val_ce_weighted", "val_ce_unweighted", "val_stay_gap", "val_entropy", "switch_mean",
                          "single_slot_ratio", "night_intrusion_rate", "meals_1200", "separation_ratio",
                          "group_ratio_median"]
    summary = long.groupby("arm", sort=False)[metrics].agg(["mean", "min", "max"]).T
    cmp._show("arm ごとの主な指標（種平均・最小・最大）", summary.reset_index())
    cmp._show("判定 C1〜C3", judge[["arm", "n_seeds", "c1_count", "c1_pass", "c2_count", "c2_pass",
                                     *[f"c3_{k}_pass" for k in cmp.GUARD_METRICS], "memorized", "c3_pass"]])
    sep = cell[cell["separated"]]
    cmp._show(f"D1 セルの差：分離した項目（{len(sep)} / {len(cell)}）", sep)
    counts = (time_effect.groupby(["cell", "pair"], sort=False)
              .agg(n_items=("separated", "size"), n_separated=("separated", "sum")).reset_index())
    cmp._show("D2 時刻符号の効果：分離した項目の数", counts)
    cmp._show("事前の予測 P1〜P3", pred)
    cmp._show("学習の記録（種の最小・最大）",
              training.groupby("arm", sort=False)[["best_epoch", "stopped_epoch", "best_val", "rise_after_best",
                                                     "gap_at_best"]].agg(["min", "max"]).T.reset_index())
    sep_w = width[width["separated"]]
    cmp._show(f"D3 幅と weight decay の効果：分離した項目（{len(sep_w)} / {len(width)}）", sep_w)
    cmp._show("事前の予測 P4〜P11", width_pred)
    cmp._show("学習の目標のずれ（重みなし − 重み付き、pt）", shift)


def main() -> None:
    """表を出力する"""
    if not SEED_TABLE_CSV.exists():
        raise FileNotFoundError(f"先に eval_curves.py を回す（D1・D2 の (a) に使う）: {SEED_TABLE_CSV}")
    real = load_real_data()
    long = metrics_long(real)
    ref = reference_values(real, long)
    judge = judge_table(long, ref)
    seed_table = pd.read_csv(SEED_TABLE_CSV)
    cell = cell_table(long, seed_table, ref)
    time_effect = time_table(long, seed_table, ref)
    pred = prediction_table(long, seed_table, ref)
    training = training_table()
    width = width_table(long, seed_table, ref)
    width_pred = width_prediction_table(long, training)
    factor = factor_table(seed_metric_table(long, seed_table, training))
    factor_sum = factor_summary(factor)
    shift = target_shift_table(real)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_DIR / "stage1_lstm_vs_gru_long.csv", index=False)
    ref.to_csv(OUT_DIR / "stage1_lstm_vs_gru_ref.csv", index=False)
    judge.to_csv(OUT_DIR / "stage1_lstm_vs_gru_judge.csv", index=False)
    cell.to_csv(OUT_DIR / "stage1_lstm_vs_gru_cell.csv", index=False)
    time_effect.to_csv(OUT_DIR / "stage1_lstm_vs_gru_time.csv", index=False)
    pred.to_csv(OUT_DIR / "stage1_lstm_vs_gru_time_pred.csv", index=False)
    training.to_csv(OUT_DIR / "stage1_lstm_vs_gru_training.csv", index=False)
    width.to_csv(OUT_DIR / "stage1_lstm_vs_gru_width.csv", index=False)
    width_pred.to_csv(OUT_DIR / "stage1_lstm_vs_gru_width_pred.csv", index=False)
    factor.to_csv(OUT_DIR / "stage1_lstm_vs_gru_factor.csv", index=False)
    factor_sum.to_csv(OUT_DIR / "stage1_lstm_vs_gru_factor_summary.csv", index=False)
    cmp._show("D4 要因の効果（分離して増えた組 / 減った組）", factor_sum)
    shift.to_csv(OUT_DIR / "stage1_lstm_vs_gru_target_shift.csv", index=False)
    print_summary(long, judge, cell, shift, time_effect, pred, width, width_pred, training)


if __name__ == "__main__":
    main()
