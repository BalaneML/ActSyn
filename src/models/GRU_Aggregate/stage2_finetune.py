"""
stage2_finetune.py
==================
GRU_Aggregate の Stage 2（重みを更新する版）: δ を logits に足す代わりに、GRU の重みそのものを
日本の教師 A* へ微調整する。評価と判定は stage2.py（δ を足す版）と同じ関数・同じ設定を使う
（E1・E2 の評価は stage2.evaluate、判定 J1〜J6 は stage2.judge(method="finetune")、E0 は共通）。

方法（傾けたリハーサル）。反復 k = 0..ITERS−1 ごとに:

    1. 採点用のプールを引く（群あたり FIT_POOL_N 本、g = GUIDANCE = 1.25、種 FIT_SEED + k。δ 版と同じ）
       → gen_g。教師への rate_mse はこれで測る
    2. リハーサル用のプールを g = REHEARSAL_GUIDANCE = 1.0 で引く（種 REHEARSAL_SEED + k）→ pool, gen_1
    3. 群ごとに、リハーサルの目標を置く（STEP は δ 版と同じ 0.5。スロットごとに和 1 へ正規化）:
           log target[d, :, s] = log gen_1[d, :, s] + STEP · (log A*[d, :, s] − log gen_g[d, :, s])
       g = 1.25 の生成と A* のずれの半分だけ、g = 1.0 の分布を動かす。g = 1.25 の生成が A* に一致すれば傾けは 0
    4. 群ごとに、pool の各系列 i に重み w_i を付け、重み付きの行動者率を target に合わせる
       （96 スロットの周辺への反復比例フィッティング。解は w_i ∝ exp(Σ_s η[d, s, x_is]) の形 = 指数傾け）
    5. 教師群の系列を w で重み付けし（群ごとに和を揃える）、teacher forcing の重み付き交差エントロピーで
       重みを FT_EPOCHS 回まわす（gm.run_epoch。条件を落とす確率 P_UNCOND も Stage 1 と同じ）

★リハーサルを g = 1.0 で引く理由。最尤の学習が合わせるのは条件付きの分布 p_θ(x | d) そのもの（g = 1.0）である。
  g = 1.25 で引いた系列で学習すると、CFG で強めた群の違いを条件付きの分布が覚え、次の生成でさらに g = 1.25 を
  掛けるので、反復ごとに群の違いが膨らむ。最初の版（v1: 採点とリハーサルを同じ g = 1.25 のプールで行った）の
  種 42・E1 では、群の分離 separation_ratio が 1.02 → 1.81 になり、g = 1.0 で採点しても 1.60 だった
  （Stage 1 のモデルは g = 1.0 で 0.80）。v1 の結果は stage2.METHOD_TAG の "finetune_v1"（_ftv1）に残してある

★なぜ δ と違う結果になり得るか。
  指数傾けした分布 p*(x) ∝ p(x)·exp(Σ_s η[s, x_s]) は、行動者率の制約を満たす分布のうち、元のモデルに KL で
  最も近い。これを自己回帰に分解すると
      p*(x_s | x_<s) ∝ p(x_s | x_<s) · exp(η[s, x_s]) · V_{s+1}(x_≤s)
  で、V_{s+1} は「この先のスロットで傾けがどれだけ効くか」の期待値（先読みの項）である。δ 版は V を持たず、
  各スロットの logits を履歴によらず同じだけ動かす。先読みの項を落とすと遷移と活動の長さが動く、というのが
  Stage 2 の J3（[docs/Stage2_results.md](docs/Stage2_results.md) §5.1）への仮説で、この版はそれを確かめる。
★教師から外した群（E2 の held-out）の系列はリハーサルに入れない。外した群は、条件の埋め込みなど
  群の間で共有する重みを通してだけ変わる（δ 版の「属性の足し算」に当たる仕組み）。

データフロー:

```mermaid
flowchart TD
    CK["gm.load_model(gm.ckpt_path(seed, calib_guidance=CALIB_G))"] --> GG
    CK --> GP
    GG["gm.group_pool(model, FIT_POOL_N, GUIDANCE, seed=FIT_SEED + k)<br/>g = 1.25"] --> GENG["gen_g (28, 12, 96)<br/>teacher_rate_mse を測る"]
    GP["gm.group_pool(model, FIT_POOL_N, REHEARSAL_GUIDANCE, seed=REHEARSAL_SEED + k)<br/>g = 1.0 → pool (28, n, 96)"] --> GEN1["gen_1 (28, 12, 96)"]
    TGT["tgt['group_rates_tbl']<br/>A* (28, 12, 96)"] --> ST
    GENG --> ST["rehearsal_target(gen_1, gen_g, A*, STEP)<br/>target (28, 12, 96)"]
    GEN1 --> ST
    ST --> RK["rake_weights(pool, target)<br/>w (28, n)"]
    GP --> RK
    RK --> RP["rehearsal_part(pool, w, teacher_mask)<br/>gm.SplitPart"]
    TM["teacher_mask (28,)"] --> RP
    RP --> FT["gm.run_epoch(model, loader, optimizer)<br/>FT_EPOCHS 回"]
    FT --> GG
    FT --> GP
    FT --> EV["s2.evaluate(model, None, ...)<br/>δ 版と同じ採点"]
    EV --> CSV["stage2_gru_ft{接尾辞}_{run}.csv"]
    CSV --> JD["s2.judge(method='finetune') → J1〜J6"]
```

使い方:
    .venv/bin/python src/models/GRU_Aggregate/stage2_finetune.py --seed 42            # E1
    .venv/bin/python src/models/GRU_Aggregate/stage2_finetune.py --seed 42 --fold 3   # E2
    .venv/bin/python src/models/GRU_Aggregate/stage2_finetune.py --judge
    .venv/bin/python src/models/GRU_Aggregate/stage2_finetune.py --smoke
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch


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


s2: Any = _load("gru_stage2", Path(__file__).resolve().parent / "stage2.py")
gm: Any = s2.gm
sm: Any = s2.sm
cur: Any = s2.cur
st: Any = s2.st

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]

# ============================================================
# 設定（結果を見る前に固定）
# ============================================================
# δ 版と同じもの: 反復の回数・歩幅・プール・種・CFG
ITERS: int = s2.ITERS                # 10
STEP: float = s2.STEP                # 0.5
FIT_POOL_N: int = s2.FIT_POOL_N      # 1024
FIT_SEED: int = s2.FIT_SEED          # 30000
EPS: float = s2.EPS                  # 1e-4
# この版だけのもの
REHEARSAL_GUIDANCE = 1.0             # リハーサルのプールの CFG（最尤が合わせる条件付きの分布。★モジュールの説明）
REHEARSAL_SEED = 40000               # 反復 k のリハーサルのプールの種は REHEARSAL_SEED + k（採点用の FIT_SEED とは別）
FT_LR = 1e-4                         # 微調整の学習率（Stage 1 の LR = 1e-3 の 1/10）
FT_EPOCHS = 1                        # 反復 1 回あたり、リハーサルのプールを何周するか
FT_WEIGHT_DECAY = 0.0                # AdamW の減衰は重みを 0 へ引く（Stage 1 の重みへではない）ので使わない
RAKE_ITERS = 30                      # 反復比例フィッティングの周回数（1 周 = 96 スロットを 1 回ずつ）
MODEL_NAME = "gru_calg125_ft"


# ============================================================
# 傾けの重み
# ============================================================
def rehearsal_target(gen_1: FloatArr, gen_g: FloatArr, a_star: FloatArr, step: float = STEP) -> FloatArr:
    """リハーサルの目標: g = 1.0 の率を、g = 1.25 の率と A* の log 比の step 倍だけ動かす, -> (..., 12, 96)

        log target = log gen_1 + step · (log A* − log gen_g)   （スロットごとに和 1 へ正規化）

    Args:
        gen_1: リハーサルのプール（g = 1.0）の行動者率, (..., 12, 96)。各スロットで 12 活動の和が 1
        gen_g: 採点用のプール（g = 1.25）の行動者率, (..., 12, 96)
        a_star: 教師 A*, (..., 12, 96)
        step: 歩幅。0 で gen_1 のまま。gen_g = A* なら step によらず gen_1 のまま（傾けない）

    Returns:
        目標の行動者率, (..., 12, 96)
    """
    log_t = np.log(gen_1 + EPS) + step * (np.log(a_star + EPS) - np.log(gen_g + EPS))
    t = np.exp(log_t - log_t.max(axis=-2, keepdims=True))
    return t / t.sum(axis=-2, keepdims=True)


def rake_weights(pool: IntArr, target: FloatArr, iters: int = RAKE_ITERS) -> FloatArr:
    """群ごとに、重み付きの行動者率が target に合う系列の重みを反復比例フィッティングで求める

    Note:
        ★スロット s ごとに w_i ← w_i · target[s, x_is] / (今の重み付きの率)[s, x_is] を 96 スロット順に掛ける。
          掛けた因子の積なので、解は w_i ∝ exp(Σ_s η[s, x_is]) の形（指数傾け）になる
        ★プールに 1 本も無い活動のセルは合わせられない（因子 1 のまま）。その分は次の反復のプールに任せる
        ★各スロットの更新の後に、群ごとに平均 1 へ正規化する（数値の桁を保つだけで、解は変わらない）

    Args:
        pool: 群別プール, dtype=int64, (D, n, 96)
        target: 目標の行動者率, (D, 12, 96)。各スロットで和 1
        iters: 周回数, default=RAKE_ITERS=30

    Returns:
        系列の重み, (D, n)。群ごとに平均 1
    """
    n_d, n, n_s = pool.shape
    rows = np.broadcast_to(np.arange(n_d)[:, None], (n_d, n))
    w = np.ones((n_d, n), dtype=np.float64)
    for _ in range(iters):
        for s in range(n_s):
            x = pool[:, :, s]
            rate = np.zeros((n_d, sm.NUM_ACT), dtype=np.float64)
            np.add.at(rate, (rows, x), w)
            rate /= w.sum(axis=1, keepdims=True)
            factor = np.divide(target[:, :, s], rate, out=np.ones_like(rate), where=rate > 0)
            w *= factor[rows, x]
            w /= w.mean(axis=1, keepdims=True)
    return w


def weighted_slot_rates(pool: IntArr, w: FloatArr) -> FloatArr:
    """重み付きの群別の時刻別行動者率, -> (D, 12, 96)"""
    n_d, n, n_s = pool.shape
    rows = np.broadcast_to(np.arange(n_d)[:, None, None], pool.shape)
    slots = np.broadcast_to(np.arange(n_s)[None, None, :], pool.shape)
    out = np.zeros((n_d, sm.NUM_ACT, n_s), dtype=np.float64)
    np.add.at(out, (rows, pool, slots), np.broadcast_to(w[:, :, None], pool.shape))
    return out / w.sum(axis=1)[:, None, None]


def ess_fraction(w: FloatArr) -> FloatArr:
    """群ごとの実効標本数の割合 (Σw)² / (n·Σw²), -> (D,)。1 = 重みが一様、小さいほど少数の系列に偏る"""
    return w.sum(axis=1) ** 2 / (w.shape[1] * (w ** 2).sum(axis=1))


def rehearsal_part(pool: IntArr, w: FloatArr, teacher_mask: BoolArr) -> Any:
    """教師群の系列だけを、群ごとに重みの和を 1 に揃えてリハーサルの個票にする

    Note:
        ★群の重みを揃えるのは δ 版の最小二乗（群を等しく扱う）と合わせるため
        ★held-out 群（teacher_mask = False）の系列は入れない

    Args:
        pool: 群別プール, (D, n, 96)
        w: 系列の重み, (D, n)
        teacher_mask: 教師に使う群, (D,)

    Returns:
        gm.SplitPart（cond_idx (N, 3)、sched (N, 96)、weight (N,)）。N = 教師群の数 × n
    """
    groups = np.flatnonzero(teacher_mask)
    n = pool.shape[1]
    grid = sm.cond_grid()
    cond = np.repeat(grid[groups], n, axis=0).astype(np.int64)
    sched = pool[groups].reshape(-1, sm.NUM_SLOTS).astype(np.int64)
    weight = (w[groups] / w[groups].sum(axis=1, keepdims=True)).reshape(-1)
    return gm.SplitPart(cond, sched, weight)


# ============================================================
# 微調整の反復
# ============================================================
def finetune_to_teacher(model: Any, tgt: dict, teacher_mask: BoolArr, seed: int, iters: int = ITERS,
                        n_per_group: int = FIT_POOL_N, fit_seed: int = FIT_SEED,
                        rehearsal_seed: int = REHEARSAL_SEED, lr: float = FT_LR,
                        epochs: int = FT_EPOCHS, verbose: bool = True) -> list[dict[str, float]]:
    """傾けたリハーサルで model の重みを A* へ微調整する（model をその場で書き換える）

    Args:
        model: Stage 1 の GRU（gru_calg125）。この関数の中で重みが変わる
        tgt: stage2_targets.load_stula_targets の戻り値
        teacher_mask: 教師に使う群, dtype=bool, (28,)
        seed: 微調整の乱数の種（並び順・条件を落とす行）
        iters: 反復の回数, default=ITERS=10
        n_per_group: 1 回のプールの群あたりの本数, default=FIT_POOL_N=1024
        fit_seed: 反復 0 の採点用のプール（g = 1.25）の種, default=FIT_SEED=30000
        rehearsal_seed: 反復 0 のリハーサルのプール（g = 1.0）の種, default=REHEARSAL_SEED=40000
        lr: 学習率, default=FT_LR=1e-4
        epochs: 反復 1 回あたりの周回数, default=FT_EPOCHS=1
        verbose: 反復ごとに print するか

    Returns:
        反復ごとの記録 [{"iter", "teacher_rate_mse"（更新の前に g = 1.25 で測った値）, "ess_mean", "ess_min",
        "rake_mse"（重み付きの率と目標の差の 2 乗平均）, "ft_loss"}]
    """
    torch.manual_seed(seed)
    a_star = tgt["group_rates_tbl"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=FT_WEIGHT_DECAY)
    history: list[dict[str, float]] = []
    for k in range(iters):
        pool_g = gm.group_pool(model, n_per_group, s2.GUIDANCE, seed=fit_seed + k)
        gen_g = np.asarray(cur.pool_to_slot_rates(pool_g), dtype=np.float64)      # (28, 12, 96)
        pool = gm.group_pool(model, n_per_group, REHEARSAL_GUIDANCE, seed=rehearsal_seed + k)
        gen_1 = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
        target = rehearsal_target(gen_1, gen_g, a_star)
        w = rake_weights(pool, target)
        ess = ess_fraction(w)[teacher_mask]
        rake_mse = float(np.mean((weighted_slot_rates(pool, w) - target)[teacher_mask] ** 2))
        loader = gm.make_loader(rehearsal_part(pool, w, teacher_mask), shuffle=True)
        loss = float("nan")
        for _ in range(epochs):
            loss = gm.run_epoch(model, loader, optimizer)
        history.append({"iter": float(k), "teacher_rate_mse": s2.teacher_rate_mse(gen_g, tgt, teacher_mask),
                        "ess_mean": float(ess.mean()), "ess_min": float(ess.min()),
                        "rake_mse": rake_mse, "ft_loss": loss})
        if verbose:
            h = history[-1]
            print(f"finetune iter {k} | teacher rate_mse {h['teacher_rate_mse']:.4e} | ess mean {h['ess_mean']:.3f} "
                  f"min {h['ess_min']:.3f} | rake_mse {h['rake_mse']:.2e} | loss {h['ft_loss']:.5f}", flush=True)
    model.eval()
    return history


# ============================================================
# 保存先と実行
# ============================================================
def ft_ckpt_path(seed: int, run: str) -> Path:
    """微調整した重みの ckpt のパス"""
    return s2.SHIFT_DIR / f"gru_stage2ft{gm.run_suffix(seed)}_{run}.pt"


def run(seed: int, fold: int | None) -> None:
    """E1 / E2 を 1 本回して、ckpt と評価の CSV を書く（E0 は stage2.py --zero-shot と共通）"""
    tgt = st.load_stula_targets()
    model = gm.load_model(gm.ckpt_path(seed, calib_guidance=s2.CALIB_G))
    name = s2.run_name(False, fold)
    teacher = np.ones(sm.D_GROUPS, dtype=bool) if fold is None else s2.fold_masks(tgt)[f"fold{fold}"]
    history = finetune_to_teacher(model, tgt, teacher, seed)
    gm.save_ckpt(model, ft_ckpt_path(seed, name), seed, extra={
        "stage2": "finetune", "version": 2, "run": name, "iters": ITERS, "step": STEP, "lr": FT_LR,
        "epochs": FT_EPOCHS, "fit_pool_n": FIT_POOL_N, "fit_seed": FIT_SEED, "guidance": s2.GUIDANCE,
        "rehearsal_guidance": REHEARSAL_GUIDANCE, "rehearsal_seed": REHEARSAL_SEED, "history": history})
    base: dict[str, Any] = {"model": MODEL_NAME, "seed": seed, "run": name, "iters": ITERS,
                            "guidance_scale": s2.GUIDANCE, "n_per_group": s2.EVAL_N, "pool_seed": s2.EVAL_SEED}
    rows = s2.evaluate(model, None, tgt, {name: teacher}, base)
    out = s2.csv_path(seed, name, "finetune")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[stage2 finetune] 書いた: {out}")


def smoke() -> None:
    """反復 1 回・群あたり 4 本で、微調整と軸 1 の採点が通ることだけを確かめる（何も保存しない）"""
    tgt = st.load_stula_targets()
    model = gm.load_model(gm.ckpt_path(s2.LGO_SEED, calib_guidance=s2.CALIB_G))
    teacher = s2.fold_masks(tgt)["fold0"]
    history = finetune_to_teacher(model, tgt, teacher, s2.LGO_SEED, iters=1, n_per_group=4)
    rows = s2.evaluate(model, None, tgt, {"fold0": teacher}, {"run": "smoke"}, n_per_group=4, full=False)
    kinds = {r["eval_kind"] for r in rows}
    assert {"in-teacher", "held-out", "all"} <= kinds and len(history) == 1
    print(f"smoke: OK ({len(rows)} 行)")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GRU_Aggregate の Stage 2（重みを更新する版・傾けたリハーサル）")
    ap.add_argument("--seed", type=int, default=s2.LGO_SEED)
    ap.add_argument("--fold", type=int, default=None, help="教師から外す LGO の fold（0〜6）。省略で 28 群すべて")
    ap.add_argument("--judge", action="store_true", help="E0〜E2 の CSV から J1〜J6 を出す")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.fold is not None and not 0 <= args.fold < s2.lgo.N_FOLDS:
        ap.error(f"--fold は 0〜{s2.lgo.N_FOLDS - 1}: {args.fold}")
    if args.smoke:
        smoke()
    elif args.judge:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
            print(s2.judge(method="finetune").to_string(index=False))
    else:
        run(args.seed, args.fold)


if __name__ == "__main__":
    main()
