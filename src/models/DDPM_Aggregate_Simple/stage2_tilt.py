"""
stage2_tilt.py
==============
Stage 2 のリハーサル項に使う ATUS 個票を、群ごとに教師 A* へ傾ける（指数傾け）

なぜ要るか:
    リハーサル項（ATUS 個票の ε-MSE）は系列の自然さを守るが、米国の行動者率まで守ってしまう。
    集計損失 L_agg は日本の A* へ引くので、2 つの勾配は逆を向き（cos = −0.27）、
    L_agg は釣り合い点で止まる（λ=0.003 で 300〜600 step の rate_mse_split が平ら）。
    個票に重みを付けて群ごとの行動者率を A* に近づけると、リハーサルと L_agg の目標が揃う。

重み（群 d の学習個票 i）:
    w_i = B_d · b̃_i · exp(η_d · φ_i) / Z_d,   b̃_i = b_i / B_d,   B_d = Σ_{j∈d} b_j
    φ_i = スケジュールの one-hot (12×96 = 1152)、b_i = TUFINLWGT
    η_d = argmin_η  log Σ_{i∈d} b̃_i exp(η·φ_i) − η·A*_d + (ρ/2)‖η‖²     （凸）

    η_d[c,s] は「日本の方が多い（活動 c・時刻 s）には正、少ないには負」になる。
    個票が実際にしている (c,s) の η を全部足し、その exp を重みに掛ける。
    ρ→0 の解は「A* を満たす分布のうち ATUS に KL で最も近いもの」（I-射影）で、
    行動者率以外（系列のつながり）はなるべく変えない。

Note:
    ★群ごとの η だけを使う。群共通の η を足すと「28 群の誤差の和」を合わせにいき、
      群ごとには行き過ぎと不足が打ち消し合う（試算で rate_mse 0.0038 → 0.0043〜0.0073 に悪化）。
    ★群の重みの合計 B_d は変えない。群の構成比（米国）は従来どおりで、群の中だけを傾ける。
    ★傾けない群（LGO の held-out）は w = b のまま。その群の A* を読まないため。
    ★試算（学習分割 3,363 人、ρ=0.1）: A* との rate_mse 0.00380 → 0.00140、
      ESS/n の中央値 0.36・最小 0.19（最小の群で 9.5 人）。ρ<=0.03 は 0.00138 で頭打ち
      （ATUS 個票の範囲の限界）。

データフロー:

```mermaid
flowchart LR
    IN["sched (N,96) / groups (N,) / weight (N,)<br/>ATUS 学習分割"] --> PHI["phi = one_hot(sched) (n_d, 1152)<br/>log_b = log(b / B_d)"]
    AS["a_star (28,12,96)"] --> FIT
    PHI --> FIT["fit_group_eta(phi, log_b, target, rho)<br/>L-BFGS, float64"]
    FIT --> P["p = softmax(log_b + phi @ eta)<br/>w = B_d * p"]
    P --> OUT["TiltResult(weights (N,), eta (28,1152), table)"]
```

使い方:
    res = fit_tilt(sched_tr, groups_tr, weight_tr, a_star, tilt_groups, rho=0.1)
    weight[train_idx] = res.weights
"""
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

NUM_ACT = 12
NUM_SLOTS = 96
D_GROUPS = 28
DEFAULT_RHO = 0.1
# L-BFGS の設定。試算では全 28 群で勾配ノルム 1e-5 以下まで収束した
LBFGS_MAX_ITER = 2000
LBFGS_HISTORY = 50

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


@dataclass
class TiltResult:
    """傾けた重みと、その診断

    Attributes:
        weights: 傾けた重み, dtype=float64, (N,)。群ごとの合計は元の weight と同じ
        eta: 群ごとの η, dtype=float64, (D_GROUPS, NUM_ACT*NUM_SLOTS)。傾けない群は 0
        table: 群ごとの診断。列は d / tilted / n / ess_before / ess_after /
            mse_before / mse_after / grad_norm（mse は A*_d との 1152 セルの平均二乗誤差）
    """
    weights: FloatArr
    eta: FloatArr
    table: pd.DataFrame


def one_hot_flat(sched: IntArr) -> torch.Tensor:
    """スケジュールを活動優先の one-hot に平らにする, (n, 96) -> (n, 12*96), float64

    ★並びは c*96 + s（活動優先）。a_star[d].reshape(-1) と同じ並びにする
    """
    oh = torch.nn.functional.one_hot(torch.as_tensor(sched, dtype=torch.long), NUM_ACT)  # (n,96,12)
    return oh.permute(0, 2, 1).reshape(len(sched), NUM_ACT * NUM_SLOTS).to(torch.float64)


def fit_group_eta(phi: torch.Tensor, log_b: torch.Tensor, target: torch.Tensor,
                  rho: float) -> tuple[torch.Tensor, float]:
    """1 群の η を解く: argmin_η log Σ_i exp(log_b_i + η·φ_i) − η·target + (ρ/2)‖η‖²

    Args:
        phi: 群の個票の one-hot, dtype=float64, (n_d, 1152)
        log_b: 群の中で正規化した元の重みの対数 log(b_i / B_d), dtype=float64, (n_d,)
        target: 教師 A*_d, dtype=float64, (1152,)
        rho: リッジの強さ, 正の値

    Returns:
        (eta, grad_norm)。eta は dtype=float64, (1152,)、grad_norm は解での目的関数の勾配ノルム
    """
    eta = torch.zeros(phi.size(1), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([eta], lr=1.0, max_iter=LBFGS_MAX_ITER, history_size=LBFGS_HISTORY,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-11,
                            tolerance_change=1e-15)

    def objective() -> torch.Tensor:
        return (torch.logsumexp(log_b + phi @ eta, dim=0) - eta @ target
                + 0.5 * rho * eta.pow(2).sum())

    def closure() -> torch.Tensor:
        opt.zero_grad()
        f = objective()
        f.backward()
        return f

    opt.step(closure)
    (grad,) = torch.autograd.grad(objective(), eta)
    return eta.detach(), float(grad.norm())


def _ess(p: torch.Tensor) -> float:
    """正規化済みの重み p の実効標本数 (Σp)² / Σp² = 1 / Σp²"""
    return float(1.0 / p.pow(2).sum())


def fit_tilt(sched: IntArr, groups: IntArr, weight: FloatArr, a_star: FloatArr,
             tilt_groups: list[int] | IntArr, rho: float = DEFAULT_RHO) -> TiltResult:
    """ATUS 個票の重みを、群ごとに教師 A* へ傾ける

    Note:
        1. tilt_groups に無い群（LGO の held-out）は w = b のまま。A* を読まない
        2. rho=inf は傾けない（w = b、eta = 0）。従来のリハーサルと一致する
        3. 群の重みの合計は変えない。群の中の配分だけが変わる

    Args:
        sched: ATUS 個票のスケジュール (インデックス表現), dtype=int64, (N, 96)。学習分割だけを渡す
        groups: 個票の群インデックス d, dtype=int64, (N,)
        weight: 元の重み TUFINLWGT, dtype=float64, (N,), 正の値
        a_star: 教師 A*, dtype=float64, (D_GROUPS, NUM_ACT, NUM_SLOTS), 値域 [0, 1]
        tilt_groups: 傾ける群（ふつうは教師群）のインデックス
        rho: リッジの強さ, 正の値か inf, default=DEFAULT_RHO=0.1

    Returns:
        TiltResult

    Raises:
        ValueError: rho が正でも inf でもない場合、または a_star に NaN がある場合
    """
    if not (math.isinf(rho) or rho > 0.0):
        raise ValueError(f"rho は正か inf でなければならない: {rho}")
    a_flat = np.asarray(a_star, dtype=np.float64).reshape(D_GROUPS, NUM_ACT * NUM_SLOTS)
    if np.isnan(a_flat).any():
        raise ValueError("a_star に NaN がある。非公表セルは傾けに使えない")
    groups = np.asarray(groups, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    out = weight.copy()
    eta_all = np.zeros_like(a_flat)
    tilt_set = {int(d) for d in tilt_groups}
    rows: list[dict[str, float | int | bool]] = []

    for d in range(D_GROUPS):
        idx = np.flatnonzero(groups == d)
        if idx.size == 0:
            continue
        phi = one_hot_flat(sched[idx])
        b = torch.as_tensor(weight[idx], dtype=torch.float64)
        b_total = float(b.sum())
        p0 = b / b_total
        target = torch.as_tensor(a_flat[d])
        tilted = d in tilt_set and not math.isinf(rho)
        grad_norm = 0.0
        p1 = p0
        if tilted:
            eta, grad_norm = fit_group_eta(phi, torch.log(p0), target, rho)
            eta_all[d] = eta.numpy()
            with torch.no_grad():
                p1 = torch.softmax(torch.log(p0) + phi @ eta, dim=0)
            out[idx] = (p1 * b_total).numpy()
        rows.append({
            "d": d, "tilted": tilted, "n": int(idx.size),
            "ess_before": _ess(p0), "ess_after": _ess(p1),
            "mse_before": float(((p0 @ phi - target) ** 2).mean()),
            "mse_after": float(((p1 @ phi - target) ** 2).mean()),
            "grad_norm": grad_norm,
        })
    return TiltResult(weights=out, eta=eta_all, table=pd.DataFrame(rows))


def summarize(table: pd.DataFrame) -> dict[str, float]:
    """傾けた群の診断を 1 行に要約する（学習ログと ckpt の config に載せる）

    Args:
        table: TiltResult.table

    Returns:
        tilt_mse_before / tilt_mse_after: 傾けた群での A* との rate_mse（群の平均）
        tilt_ess_ratio_median / tilt_ess_ratio_min: 傾けた後の ESS / n
        tilt_ess_min: 傾けた後の ESS の最小（人数）
        tilt_grad_norm_max: η の解での勾配ノルムの最大（収束の確認）
    """
    t = table[table["tilted"]]
    if t.empty:
        return {}
    ratio = t["ess_after"] / t["n"]
    return {"tilt_mse_before": float(t["mse_before"].mean()),
            "tilt_mse_after": float(t["mse_after"].mean()),
            "tilt_ess_ratio_median": float(ratio.median()),
            "tilt_ess_ratio_min": float(ratio.min()),
            "tilt_ess_min": float(t["ess_after"].min()),
            "tilt_grad_norm_max": float(t["grad_norm"].max())}
