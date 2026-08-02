"""
指数傾け (Stage2) の単体テスト。

japan_match_experiment.exponential_tilt は凸双対を L-BFGS で解く。導出を1箇所でも
間違えると「収束はするが別の問題を解いている」状態になり、下流の評価指標だけを見ても
気づけない。ここで検証するのは3点:

    1. 勾配の同一性   : ∂(−g)/∂λ = (再構成周辺 − 教師)。双対の定義そのもの
    2. 往復 (round-trip): 既知の傾け θ* が作る周辺を教師として与えたら、
                          最適化後の再構成周辺がその教師に一致すること (ridge=0)
    3. 退化の非発生   : 一様重み (θ=0) が最適になる教師 = 提案分布そのものの周辺

    uv run python src/models/DDPM_Aggregate/test_tilting.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
# CVAE_Aggregate にも同名の japan_match_experiment.py があり、素の
# `import japan_match_experiment` は sys.path の順序次第で別モデルを掴む
# (pyrightconfig.json の extraPaths では CVAE 版が先に見つかる)。
# 完全修飾で参照して、実行時も型チェック時も DDPM 版に固定する。
sys.path.insert(0, str(REPO_ROOT))
from src.models.DDPM_Aggregate import japan_match_experiment as ex  # noqa: E402

D, M, T, C = ex.D_GROUPS, 400, ex.NUM_SLOTS, ex.NUM_ACT
N_G, N_A, N_E = ex.N_G, ex.N_A, ex.N_E
dev = ex.DEVICE


def synth_targets(pool: np.ndarray, theta_star: torch.Tensor | None, mask_c: np.ndarray) -> dict:
    """既知の傾け θ* をプールに適用して得られる周辺を「公表教師」として作る。

    θ*=None なら一様重み（= 提案分布そのものの周辺）を教師にする。
    人口 pop はランダムだが固定シードで、π(e|g,a) / π(a|g,e) の混合を実際に効かせる。
    """
    rng = np.random.default_rng(0)
    pop = rng.uniform(500, 5000, size=(N_G, N_A, N_E))
    pi_e = pop / pop.sum(axis=2, keepdims=True)
    pi_a = pop / pop.sum(axis=1, keepdims=True)

    idx = torch.as_tensor(pool, dtype=torch.long, device=dev)
    if theta_star is None:
        w = torch.full((D, M), 1.0 / M, device=dev)
    else:
        d_ar = torch.arange(D, device=dev)[:, None, None]
        t_ar = torch.arange(T, device=dev)[None, None, :]
        w = torch.softmax(theta_star[d_ar, idx, t_ar].sum(-1), dim=1)
    mu = torch.zeros(D, C, T, device=dev).scatter_add_(1, idx, w[:, :, None].expand(D, M, T))
    mu5 = mu.view(N_G, N_A, N_E, C, T)
    pe = torch.as_tensor(pi_e, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(pi_a, dtype=torch.float32, device=dev)
    mga = (mu5 * pe[..., None, None]).sum(dim=2).cpu().numpy().astype(np.float64)
    mge = (mu5 * pa[..., None, None]).sum(dim=1).cpu().numpy().astype(np.float64)
    # 未公表 NaN も混ぜて、マスク経路まで含めて検証する
    mga = mga.copy(); mge = mge.copy()
    mga[0, 0, int(np.flatnonzero(mask_c)[0]), :4] = np.nan
    return {"margin_ga": mga, "margin_ge": mge, "pi_e": pi_e, "pop": pop,
            "group_rates_tbl": np.zeros((D, C, T))}


def main():
    rng = np.random.default_rng(42)
    # 群ごとに異なる「もっともらしい」プール: 群依存に活動の出やすさを変えた1次マルコフ
    pool = np.zeros((D, M, T), dtype=np.int64)
    for d in range(D):
        base = rng.dirichlet(np.ones(C) * 0.7)
        x = rng.choice(C, size=M, p=base)
        for t in range(T):
            pool[:, :, t][d] = x
            stay = rng.random(M) < 0.85          # 85% は同じ活動を継続 = 現実的な持続性
            newx = rng.choice(C, size=M, p=base)
            x = np.where(stay, x, newx)

    mask_c = np.array([c not in {C - 1} for c in range(C)])   # 最終カテゴリを OTHER_X 相当で除外

    # ---- 1 & 2: 既知の θ* からの往復 ----
    # σ は控えめに。96スロット合計でスコアの散らばりが σ·√96 になるため、大きくすると
    # 重みが e^±5 級に散って ESS が数本まで落ち、「解けない問題」を解かせることになる
    theta_star = torch.as_tensor(
        rng.normal(0, 0.06, size=(D, C, T)), dtype=torch.float32, device=dev)
    theta_star[:, C - 1, :] = 0.0                              # 除外カテゴリは傾けない
    tgt = synth_targets(pool, theta_star, mask_c)

    # ridge は極小だが 0 にはしない: 公表2表の線形従属で双対に平坦方向が残るため
    # (exponential_tilt の docstring 参照)。ここでの狙いは θ* の復元であって
    # 正則化ではないので、識別性が回復する最小限に留める
    w, diag = ex.exponential_tilt(pool, tgt, mask_c, ridge=1e-10, iters=200, tag="roundtrip")
    print(f"\n[往復] 残差 MAE {diag['resid_mae']:.3e}  max {diag['resid_max']:.3e}  "
          f"ESS中央値 {diag['ess_median']:.0f}/{M}")
    assert diag["resid_mae"] < 1e-6, f"往復に失敗: 残差MAE {diag['resid_mae']:.3e}"
    assert diag["resid_max"] < 1e-4, f"往復に失敗: 残差max {diag['resid_max']:.3e}"
    assert np.allclose(w.sum(axis=1), 1.0), "重みが正規化されていない"

    # ---- 3: 教師 = 提案分布そのもの -> 一様重みが最適 (θ=0, KL=0) ----
    tgt0 = synth_targets(pool, None, mask_c)
    w0, diag0 = ex.exponential_tilt(pool, tgt0, mask_c, ridge=1e-10, iters=200, tag="identity")
    print(f"[恒等] 残差 MAE {diag0['resid_mae']:.3e}  KL平均 {diag0['kl_mean']:.3e}  "
          f"ESS中央値 {diag0['ess_median']:.0f}/{M}")
    assert diag0["kl_mean"] < 1e-3, f"退化: 教師が提案分布と同じなのに KL={diag0['kl_mean']:.3e}"
    assert abs(w0.max() - 1.0 / M) < 1e-4, "一様重みになっていない"

    # ---- 4: ridge を上げると KL(=傾けの強さ) が単調に減る ----
    kls = []
    for r in [1e-10, 1e-3, 1e-2, 1e-1]:
        _, dg = ex.exponential_tilt(pool, tgt, mask_c, ridge=r, iters=200, tag=f"ridge{r:g}")
        kls.append(dg["kl_mean"])
    print(f"[ridge] KL平均の推移 {['%.4f' % k for k in kls]}")
    assert all(kls[i] >= kls[i + 1] - 1e-6 for i in range(len(kls) - 1)), \
        f"ridge を上げても傾けが弱まっていない: {kls}"

    print("\ntest_tilting: OK")


if __name__ == "__main__":
    main()
