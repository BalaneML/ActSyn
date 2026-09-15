"""
stage1_overfit_check.py
=======================
Stage 1（ATUS 事前学習）が過学習しているかを、3つの独立な経路で測る。

「過学習している」という指摘に答えるには、train と holdout の値を並べるだけでは
足りない。参照集合の大きさが違えば、過学習がゼロでも差が出るからである
（DCR は参照集合が大きいほど小さくなる）。そこで本スクリプトはどの経路でも
**train を holdout と同数へ間引いた帰無帯（床）**を作り、その外に出たときだけ
過学習と呼ぶ。

    経路 A  eps  保持個票の ε-MSE。学習目標そのものを、条件付き・t 層化格子・
                 固定ノイズで低分散に測り直す。学習中の val 損失は 1 エポックあたり
                 373 個票 × ランダムな 1 つの t しか引かないので分散が大きく、
                 「どのエポックが最良か」の判定には使えない
    経路 B  mem  暗記。生成プールから train / holdout への最近傍ハミング距離 (DCR)。
                 参照集合を同数へ揃えてから比べる
    経路 C  share 集計（活動別時間シェア）。生成プールを実データの群構成へ
                 重み付けしたうえで train / holdout と比べる

判定は片側である。過学習なら「train 側でだけ良い」ので、
    A: holdout の ε-MSE が床より**上**
    B: holdout との DCR が床より**上**（＝ train にだけ近い）
    C: holdout との Σ|Δ| が床より**上**
のときだけ過学習の証拠になる。床の内なら「差の証拠なし」であって、
「差が無いことの証明」ではない。

使い方:
    .venv/bin/python3 src/eval/diagnostics/stage1_overfit_check.py --check all
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


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


sm: Any = _load("simple_model", SIMPLE_DIR / "model.py")
im: Any = _load("ofchk_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")

DEFAULT_CKPT = REPO_ROOT / "outputs" / "checkpoints" / "ddpm_simple_pretrain_common12_weekday_20260819.pt"
DEFAULT_POOL = REPO_ROOT / "outputs" / "generated" / "ddpm_simple_pretrain_samples_20260819.csv"

# t 層化格子。0..999 を 20 刻みで 50 点。各点でノイズを固定するので、
# train と holdout は同じ (t, ε) の組で評価される
T_GRID = list(range(10, sm.T_STEPS, 20))
T_BANDS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]


def load_real() -> tuple[IntArr, IntArr, FloatArr, IntArr, IntArr]:
    """ATUS 平日と、学習時と同一の train / holdout 分割を返す。

    returns: (cond_idx, sched, weight, train_idx, holdout_idx)
    """
    cond_idx, sched, weight, _ = sm.load_data(sm.DATA_PATH)
    train_idx, holdout_idx = sm.split_indices(len(sched))
    return cond_idx, sched, weight, train_idx, holdout_idx


# ============================================================
# 経路 A: 保持個票の ε-MSE
# ============================================================
@torch.no_grad()
def per_sample_eps_mse(ckpt: Path, cond_idx: IntArr, sched: IntArr,
                       batch_size: int = 512, verbose: bool = True) -> FloatArr:
    """個票 × t 格子の ε-MSE 行列 (N, len(T_GRID)) を返す。

    ★条件付き（CFG の条件 dropout を掛けない）で測る。学習中の val 損失は
      P_UNCOND=0.1 で条件を落とした行を含むので、そのままでは意味が混ざる。
    ★ノイズは t ごとに固定する。train と holdout の差が「引いたノイズの違い」で
      説明されないようにするため。
    """
    device = sm.DEVICE
    net = sm.load_pretrained(ckpt)
    diffusion = sm.Diffusion()

    cond = torch.as_tensor(cond_idx, dtype=torch.long, device=device)
    x0 = sm.sched_to_x0(torch.as_tensor(sched, dtype=torch.long, device=device))
    n = x0.size(0)
    out = np.zeros((n, len(T_GRID)), dtype=np.float64)

    for j, t_val in enumerate(T_GRID):
        gen = torch.Generator(device="cpu").manual_seed(1000 + t_val)
        eps_all = torch.randn(x0.shape, generator=gen).to(device)
        for s in range(0, n, batch_size):
            sl = slice(s, min(s + batch_size, n))
            t = torch.full((x0[sl].size(0),), t_val, device=device, dtype=torch.long)
            x_t = diffusion.q_sample(x0[sl], t, eps_all[sl])
            eps_hat = net(x_t, t, cond[sl])
            out[sl, j] = ((eps_hat - eps_all[sl]) ** 2).mean(dim=(1, 2)).cpu().numpy()
        if verbose:
            print(f"  t={t_val:4d} done", flush=True)
    return out


def _size_matched_band(values: FloatArr, n_draw: int, n_boot: int,
                       seed: int = 0) -> tuple[float, float, float]:
    """values（train 側の個票値）から n_draw 本を非復元抽出した平均の (lo, median, hi)。"""
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(values, size=n_draw, replace=False).mean()
                      for _ in range(n_boot)])
    lo, md, hi = np.percentile(draws, [2.5, 50.0, 97.5])
    return float(lo), float(md), float(hi)


def eps_check(ckpt: Path, n_boot: int = 2000) -> None:
    cond_idx, sched, _, train_idx, holdout_idx = load_real()
    print(f"device={sm.DEVICE}  N={len(sched)}  "
          f"train={len(train_idx)}  holdout={len(holdout_idx)}  ckpt={ckpt.name}")
    per = per_sample_eps_mse(ckpt, cond_idx, sched)

    n_ho = len(holdout_idx)
    ind = per.mean(axis=1)  # t 格子で等重み平均した個票ごとの ε-MSE
    tr, ho = ind[train_idx], ind[holdout_idx]
    lo, md, hi = _size_matched_band(tr, n_ho, n_boot)
    print("\n=== A. 保持個票 ε-MSE（条件付き・t 層化格子 50 点・固定ノイズ） ===")
    print(f"train 全体({len(tr)})       {tr.mean():.6f}")
    print(f"holdout({n_ho})             {ho.mean():.6f}   gap {ho.mean() - tr.mean():+.6f}")
    print(f"床: train を {n_ho} 本へ間引き  [{lo:.6f}, {hi:.6f}]  中央 {md:.6f}")
    print(f"判定: holdout は床の{'外（過学習の兆候）' if ho.mean() > hi else '内（過学習の証拠なし）'}")

    print("\n--- t 帯別（帯ごとに床を取り直す） ---")
    rng_seed = 1
    for a, b in T_BANDS:
        cols = [j for j, t_val in enumerate(T_GRID) if a <= t_val < b]
        band = per[:, cols].mean(axis=1)
        tr_b, ho_b = band[train_idx], band[holdout_idx]
        lo_b, _, hi_b = _size_matched_band(tr_b, n_ho, n_boot // 2, seed=rng_seed)
        print(f"t[{a:4d},{b:4d})  train {tr_b.mean():.6f}  holdout {ho_b.mean():.6f}  "
              f"gap {ho_b.mean() - tr_b.mean():+.6f}  "
              f"床[{lo_b:.6f}, {hi_b:.6f}]  {'外' if ho_b.mean() > hi_b else '内'}")


# ============================================================
# 経路 B: 暗記（DCR）
# ============================================================
def _dcr(gen: IntArr, ref: IntArr) -> float:
    """生成プール全行から参照集合への最近傍ハミング距離の平均。"""
    return float(im.nn_distances(gen, ref, k=1, sample=len(gen), seed=0)[:, 0].mean())


def mem_check(pool: Path, n_sub: int = 5, n_band: int = 8) -> None:
    gen = pd.read_csv(pool)[[f"s{i}" for i in range(sm.NUM_SLOTS)]].to_numpy(np.int64)
    _, sched, _, train_idx, holdout_idx = load_real()
    train, holdout = sched[train_idx], sched[holdout_idx]
    n_ho = len(holdout)

    print("\n=== B. 暗記（DCR: 最近傍ハミング距離、大きいほど遠い） ===")
    print(f"pool={pool.name}  gen={gen.shape}")
    dcr_ho = _dcr(gen, holdout)
    dcr_tr_full = _dcr(gen, train)
    print(f"DCR(gen→train 全体 {len(train)})  {dcr_tr_full:.4f}")
    print(f"DCR(gen→holdout {n_ho})           {dcr_ho:.4f}")
    print(f"素の gap {dcr_ho - dcr_tr_full:+.4f}  ← 参照集合サイズの交絡が入っており読めない")

    rng = np.random.default_rng(0)
    subs = [_dcr(gen, train[rng.choice(len(train), n_ho, replace=False)])
            for _ in range(n_sub)]
    gap = dcr_ho - float(np.mean(subs))
    print(f"DCR(gen→train を {n_ho} 本へ間引き) {n_sub} 回平均 {np.mean(subs):.4f}")
    print(f"サイズを揃えた gap {gap:+.4f}")

    # 床: train 内の互いに素な同数部分集合どうしの gap
    gaps = []
    for _ in range(n_band):
        perm = rng.permutation(len(train))
        gaps.append(_dcr(gen, train[perm[:n_ho]]) - _dcr(gen, train[perm[n_ho:2 * n_ho]]))
    lo, hi = float(min(gaps)), float(max(gaps))
    print(f"床（train 内の互いに素な同数対 {n_band} 本）: [{lo:+.4f}, {hi:+.4f}]")
    print(f"判定: {'床の上（暗記の兆候）' if gap > hi else '床の内（暗記の証拠なし）'}")

    keys = set(map(tuple, train))
    exact = [i for i, row in enumerate(gen) if tuple(row) in keys]
    n_const_real = int(sum(len(np.unique(row)) == 1 for row in sched))
    print(f"exact copy {len(exact)}/{len(gen)}")
    for i in exact[:5]:
        vals, cnt = np.unique(gen[i], return_counts=True)
        desc = ", ".join(f"{sm.ACT_NAMES[v]}×{c}" for v, c in zip(vals.tolist(), cnt.tolist()))
        print(f"  row {i}: {desc}")
    print(f"参考: 実データ中の定数日記（96 スロット同一活動）は {n_const_real}/{len(sched)} 本")


# ============================================================
# 経路 C: 集計（活動別時間シェア）
# ============================================================
def share_check(pool: Path, n_boot: int = 500) -> None:
    df = pd.read_csv(pool)
    gen = df[[f"s{i}" for i in range(sm.NUM_SLOTS)]].to_numpy(np.int64)
    gen_d = df["group_d"].to_numpy(np.int64)
    cond_idx, sched, weight, train_idx, holdout_idx = load_real()
    d_real = sm.cond_to_d(cond_idx)

    # 生成プールは群一様なので、実データ（train 側）の群構成へ重み付けしてから比べる。
    # 非加重の人数比を使うと総変動距離で 0.139 ずれる（im.group_reweight の注記）
    w_gen = im.group_reweight(gen_d, weight[train_idx], d_real[train_idx], sm.D_GROUPS)
    share_gen = im.time_share(gen, sm.NUM_ACT, w_gen)

    def sum_abs_diff(idx: IntArr) -> float:
        return float(np.abs(share_gen - im.time_share(sched[idx], sm.NUM_ACT,
                                                      weight[idx])).sum())

    n_ho = len(holdout_idx)
    v_tr, v_ho = sum_abs_diff(train_idx), sum_abs_diff(holdout_idx)
    rng = np.random.default_rng(0)
    draws = np.array([sum_abs_diff(rng.choice(train_idx, n_ho, replace=False))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print("\n=== C. 集計（活動別時間シェア Σ|Δ|、12 活動合計） ===")
    print(f"gen vs train 全体   {v_tr:.4f}")
    print(f"gen vs holdout      {v_ho:.4f}")
    print(f"床: train を {n_ho} 本へ間引き {n_boot} 回  [{lo:.4f}, {hi:.4f}]  "
          f"中央 {np.median(draws):.4f}")
    print(f"判定: holdout は床の{'外（過学習の兆候）' if v_ho > hi else '内（過学習の証拠なし）'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", choices=["all", "eps", "mem", "share"], default="all")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    args = ap.parse_args()

    if args.check in ("all", "eps"):
        eps_check(args.ckpt)
    if args.check in ("all", "mem"):
        mem_check(args.pool)
    if args.check in ("all", "share"):
        share_check(args.pool)


if __name__ == "__main__":
    main()
