"""
stage2_gradient_probe.py
========================
末尾 K ステップ打ち切り（DRaFT-K、Stage2_design.md §4.3）が何によって正当化されるかを
3つの経路で測る。

設計の初版は「1段あたりの倍率が 1 未満なので、古いステップからの勾配は指数的に
潰れて寄与しない」と書いていた。本スクリプトはその主張を検算するために書かれ、
**主張が誤りであること**を示した。打ち切りが正しい理由は「古い項が消えるから」では
なく「古い項が勾配の向きをほとんど変えないから」である。

    経路 jac    解析。逆過程1ステップのヤコビアンのうち、網（denoiser）の反応と
                clamp を無視した「算数だけの部分」を閉じた形で出す。答えは
                1/√α_t で、これは必ず 1 より大きい。つまり DDPM の逆過程に
                縮小は組み込まれていない
    経路 prop   実測。同じ雑音 z を使い回して x_t を少しずらし、x_0 がどれだけ
                動くかを差分で測る。‖Δx_0‖/‖Δx_t‖ が ∂x_0/∂x_t のランダム方向に
                沿った大きさになる
    経路 depth  実測。K を伸ばしたときの ∂L/∂θ のノルムと、最も深い K との
                コサイン類似度。打ち切りが勾配の大きさと向きのどちらを削るのかを
                切り分ける

★経路 depth は全 K で同一の x_0 を見る。1本の逆過程を流して各 K の x_K を控え、
  そこから同じ z で末尾だけ回し直すため、∂L/∂x_0 は K によらず共通になる。
  こうしないと「K を変えたら別のサンプルが出た」だけの差を勾配の差と誤認する。

使い方:
    .venv/bin/python3 src/eval/diagnostics/stage2_gradient_probe.py --check all
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage1_overfit_check.py と同じ規則）。"""
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
sl: Any = _load("simple_stage2_loss", SIMPLE_DIR / "stage2_loss.py")
st: Any = _load("simple_stage2_targets", SIMPLE_DIR / "stage2_targets.py")

DEFAULT_CKPT = REPO_ROOT / "outputs" / "checkpoints" / "ddpm_simple_pretrain_common12_weekday_20260819.pt"

# 摂動を差し込む t。0 に近い側を細かく取る（末尾ほど打ち切りの判断に効くため）
PROBE_T = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
# 勾配を比べる K。2倍刻みにして「1段深くするたびの増分」ではなく
# 「深さを倍にするたびの増分」で見る
PROBE_K = [1, 2, 4, 8, 16, 32]


def draw_noise(batch: int, device: torch.device | str) -> dict[int, torch.Tensor]:
    """全ステップぶんの雑音 {ti: z} を先に引く。

    摂動あり・無しの2本の軌跡、および K の異なる末尾で同じ z を使うために必要。
    ti=0 は post_var[0] = 0 で雑音を使わないのでキーを作らない。
    """
    return {ti: torch.randn(batch, sm.IN_CH, sm.NUM_SLOTS, device=device)
            for ti in range(1, sm.T_STEPS)}


# ============================================================
# 経路 jac: 1段あたりの倍率を閉じた形で出す
# ============================================================
def jac_check() -> None:
    """網の反応と clamp を無視した1段あたりの倍率が 1/√α_t であることを示す。

    逆過程1ステップは x_{t-1} = post_coef_x0 · clamp(x0_hat) + post_coef_xt · x_t（＋雑音）で、
    x0_hat = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t である。∂ε̂/∂x_t = 0 と置き clamp も効かないとすると

        ∂x_{t-1}/∂x_t = post_coef_x0 / √ᾱ_t + post_coef_xt

    になる。√ᾱ_t = √ᾱ_{t-1}·√α_t を使って通分すると分子が 1−ᾱ_t になり、約分して 1/√α_t。
    α_t = 1 − β_t < 1 なので、これは必ず 1 より大きい。
    """
    diff = sm.Diffusion(device="cpu")
    j_closed = diff.post_coef_x0 / diff.sqrt_acp + diff.post_coef_xt
    j_ident = 1.0 / diff.alphas.sqrt()
    err = (j_closed - j_ident).abs()

    print("=== jac. 1段あたりの倍率（網の反応と clamp を無視した部分） ===")
    # ★ti=0 だけ誤差が大きいのは post_coef_x0[0] が float32 で 0.99983406 になるため
    #   （1−ᾱ_0 を引き算で作る桁落ち。model.py:680-684 に記録）。恒等式の不成立ではない
    print(f"|post_coef_x0/√ᾱ_t + post_coef_xt − 1/√α_t| は "
          f"ti=0 で {float(err[0]):.3e}（float32 の桁落ち）、"
          f"ti≥1 で最大 {float(err[1:].max()):.3e}  → 恒等式 1/√α_t が成り立つ")
    print(f"{'t':>5} {'ᾱ_t':>10} {'1/√α_t':>10} {'√(1-ᾱ)/√ᾱ':>11}")
    for ti in [0, 1, 10, 100, 500, 999]:
        nsr = float(diff.sqrt_1m_acp[ti] / diff.sqrt_acp[ti])
        print(f"{ti:5d} {float(diff.acp[ti]):10.6f} {float(j_ident[ti]):10.6f} {nsr:11.2f}")
    total = float(1.0 / diff.acp[-1].sqrt())
    print(f"全 {sm.T_STEPS} 段の積 = 1/√ᾱ_T = {total:.1f} 倍"
          "  → 縮小ではなく増幅。指数的な勾配消失は起きない")


# ============================================================
# 経路 prop: x_t の摂動が x_0 まで届く倍率を差分で測る
# ============================================================
@torch.no_grad()
def prop_check(ckpt: Path, batch: int = 8, rel: float = 1e-3, seed: int = 0,
               n_dir: int = 5) -> None:
    """‖Δx_0‖/‖Δx_t‖ を各 t で測る。

    args:
        rel:   摂動の大きさ。‖Δx_t‖ = rel · ‖x_t‖ に揃える。
               大きすぎると clamp と argmax の折れ目をまたいで線形近似から外れ、
               小さすぎると float32 の丸めに埋もれる
        n_dir: 1つの t で試すランダム方向の本数。倍率は方向によって数倍ばらつくので、
               1本だけ引いて t 依存を論じてはいけない
    """
    torch.manual_seed(seed)
    model = sm.load_pretrained(ckpt).to(sm.DEVICE).eval()
    diff = sm.Diffusion(device=sm.DEVICE)
    cond = torch.as_tensor(sm.cond_grid()[:batch], device=sm.DEVICE)
    zs = draw_noise(batch, sm.DEVICE)

    def run_tail(x: torch.Tensor, t_from: int) -> torch.Tensor:
        """x（= x_{t_from}）から ti = t_from−1 … 0 まで進めて x_0 を返す。"""
        for ti in reversed(range(t_from)):
            x = diff._reverse_step(model, x, ti, cond, sm.GUIDANCE_SCALE, zs.get(ti))
        return x

    # 基準の1本を流し、摂動を差し込む t の状態を控える
    x = torch.randn(batch, sm.IN_CH, sm.NUM_SLOTS, device=sm.DEVICE)
    state: dict[int, torch.Tensor] = {}
    for ti in reversed(range(sm.T_STEPS)):
        x = diff._reverse_step(model, x, ti, cond, sm.GUIDANCE_SCALE, zs.get(ti))
        if ti in PROBE_T or ti == 0:
            state[ti] = x.clone()
    act_ref = state[0].argmax(dim=1)                       # (batch, 96) 実際に出る活動

    print(f"\n=== prop. x_t を rel={rel:g} だけずらしたとき x_0 が動く倍率 "
          f"(batch={batch}, 方向 {n_dir} 本, seed={seed}) ===")
    print(f"{'t':>5} {'中央値':>10} {'最小':>10} {'最大':>10} {'argmax が変わった総数':>22}")
    for t in PROBE_T:
        gains: list[float] = []
        flips = 0
        for _ in range(n_dir):
            u = torch.randn_like(state[t])
            delta = u * (rel * state[t].norm() / u.norm())
            x0_pert = run_tail(state[t] + delta, t)
            gains.append(float((x0_pert - state[0]).norm() / delta.norm()))
            flips += int((x0_pert.argmax(dim=1) != act_ref).sum())
        g = torch.tensor(gains)
        print(f"{t:5d} {float(g.median()):10.3f} {float(g.min()):10.3f} "
              f"{float(g.max()):10.3f} {flips:22d}")
    print("→ t について系統的に減衰しない。指数的な消失は起きていない")


# ============================================================
# 経路 depth: K を伸ばすと ∂L/∂θ の大きさと向きがどう変わるか
# ============================================================
def depth_check(ckpt: Path, d_sub: int = 2, n: int = 8, seed: int = 0,
                reps: int = 3) -> None:
    """K ごとの ‖∂L/∂θ‖ と、最も深い K の勾配とのコサイン類似度を出す。

    損失は Stage 2 本番と同じ split-batch の重み付き二乗誤差
    （stage2_loss.agg_loss_from_rates）を使う。

    args:
        reps: 別々の逆過程で繰り返す回数。cos は打ち切りの正当化を支える数字なので
              1本だけで論じない
    """
    model = sm.load_pretrained(ckpt).to(sm.DEVICE).eval()
    diff = sm.Diffusion(device=sm.DEVICE)
    grid = torch.as_tensor(sm.cond_grid(), device=sm.DEVICE)
    d_pick = torch.arange(d_sub, device=sm.DEVICE)
    cond = grid[d_pick].repeat_interleave(n, dim=0)        # (d_sub*n, 3) 群優先
    batch = cond.size(0)

    q = sl.teacher_tensor(st.load_stula_targets(), sm.DEVICE)[d_pick]
    omega = sl.chi2_weights(q, float("inf"))

    def one_rep(rep_seed: int) -> dict[int, torch.Tensor]:
        """1本の逆過程に対し、K ごとの ∂L/∂θ を返す。"""
        torch.manual_seed(rep_seed)
        zs = draw_noise(batch, sm.DEVICE)
        # 1本流して各 K の x_K を控える。以降どの K も同じ x_0 に行き着くので、
        # ∂L/∂x_0 は K によらず共通になる
        state: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            x = torch.randn(batch, sm.IN_CH, sm.NUM_SLOTS, device=sm.DEVICE)
            for ti in reversed(range(sm.T_STEPS)):
                x = diff._reverse_step(model, x, ti, cond, sm.GUIDANCE_SCALE, zs.get(ti))
                if ti in PROBE_K:
                    state[ti] = x.clone()

        out: dict[int, torch.Tensor] = {}
        for k in PROBE_K:
            model.zero_grad(set_to_none=True)
            zs_tail = {ti: zs[ti] for ti in range(1, k)}
            x0 = diff._sample_tail(model, state[k], k, cond, sm.GUIDANCE_SCALE, zs_tail)
            y = sm.straight_through(x0)
            a_A, a_B = sl.group_rates_split(y, n)
            sl.agg_loss_from_rates(a_A, a_B, q, omega).backward()
            out[k] = torch.cat([p.grad.flatten() for p in model.parameters()
                                if p.grad is not None])
        return out

    runs = [one_rep(seed + r) for r in range(reps)]
    deepest = PROBE_K[-1]
    print(f"\n=== depth. 末尾 K ステップだけ勾配を流したときの ∂L/∂θ "
          f"(d_sub={d_sub}, n={n}, {reps} 本平均, seed={seed}) ===")
    print(f"{'K':>4} {'‖∂L/∂θ‖':>12} {'K=1 の何倍':>11} "
          f"{f'K={deepest} との cos（最小〜最大）':>30}")
    for k in PROBE_K:
        norms = torch.tensor([float(r[k].norm()) for r in runs])
        ratios = torch.tensor([float(r[k].norm() / r[PROBE_K[0]].norm()) for r in runs])
        cosines = torch.tensor([float(F.cosine_similarity(r[k], r[deepest], dim=0))
                                for r in runs])
        print(f"{k:4d} {float(norms.mean()):12.4e} {float(ratios.mean()):11.3f} "
              f"{float(cosines.mean()):11.4f} ({float(cosines.min()):.4f}〜"
              f"{float(cosines.max()):.4f})")
    print("→ 大きさは増え続ける（古い項は消えていない）が、向きは K=1 で既にほぼ確定")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", choices=["all", "jac", "prop", "depth"], default="all")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.check in ("all", "jac"):
        jac_check()
    if args.check in ("all", "prop"):
        prop_check(args.ckpt, seed=args.seed)
    if args.check in ("all", "depth"):
        depth_check(args.ckpt, seed=args.seed)


if __name__ == "__main__":
    main()
