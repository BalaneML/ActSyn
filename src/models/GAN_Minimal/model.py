"""
model.py
========
GAN_Minimal: 単純な MLP の条件付き WGAN-GP。共通の 2 段階学習の比較対象の 1 つ（2026-10-29 ユーザー決定）

生成器は VAE_Minimal のデコーダと同じ形で、z（16 次元）・条件・時刻埋め込み[s] から各スロットの 12 活動の logits を出す。
生成物は、スロットごとに straight-through Gumbel-softmax で引いた硬い one-hot（前向きは厳密な標本）にして critic に渡す。

決定事項（2026-10-29）:
    - 損失は WGAN-GP（Gulrajani et al. 2017）。critic 5 回に生成器 1 回、勾配ペナルティの重み 10
    - 最適化は WGAN-GP の標準: Adam、学習率 1e-4、β = (0, 0.9)（生成器・critic とも）。勾配のノルムの上限は掛けない
    - 1,000 epoch で固定。最終 epoch と val 最良（critic による Wasserstein 推定が最小）の 2 つの ckpt を残す
    - 大きさ: 生成器を LSTM（42,196）に合わせる（VAE のデコーダと同じ 42,170）。critic は数えない
    - 条件は COND_SPEC、時刻は学習型の埋め込み Embedding(96, ·)。critic も自分の条件の埋め込みを持つ

★1 epoch の定義: 生成器の更新 14 回（= 学習分割 3,363 人を 256 人ずつ 1 巡する回数。ほかのモデルの 1 epoch の更新回数と
  同じ）。生成器の更新の前に critic を 5 回更新し、そのたびに実データのミニバッチを 1 つ取る（実データは 1 epoch で約 5 巡）

学習の流れ（1 回の生成器の更新）:

```mermaid
flowchart TD
    RS["real_batches()<br/>ATUS の学習分割のミニバッチ"] --> CR["critic_step × N_CRITIC（5 回）<br/>real = one_hot(sched)、fake = generate_hard(gen, cond).detach()<br/>loss_d = D(fake) − D(real) + GP_WEIGHT · gradient_penalty"]
    CR --> GS["generator_step<br/>cond = 直前の実データの条件<br/>loss_g = −D(generate_hard(gen, cond))"]
    GS --> VAL["epoch の終わりに val_wasserstein（雑音を固定）<br/>最小なら val 最良として保存"]
```

使い方:
    .venv/bin/python src/models/GAN_Minimal/model.py --smoke
    .venv/bin/python src/models/GAN_Minimal/model.py --seed 42 --threads 4

出力:
    outputs/checkpoints/gan_minimal{構成}_final{接尾辞}.pt     最終 epoch（生成器・critic・critic の optimizer）
    outputs/checkpoints/gan_minimal{構成}_valbest{接尾辞}.pt   val 最良
        {構成} = stage1_tag() = _z16_h222、{接尾辞} は種 42 なら空、それ以外は _s{seed}
"""
import argparse
import copy
import importlib.util
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/GAN_Minimal -> repo root


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同じファイルが既に載っていればそれを返す）"""
    cached = sys.modules.get(name)
    cached_file = getattr(cached, "__file__", None)
    if cached is not None and cached_file is not None and Path(cached_file).resolve() == path.resolve():
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 分割・活動の引き方は LSTM と同じ関数を使う。straight-through Gumbel は共通の Stage 2 のものを使う
lm: Any = _load("lstm_aggregate_model", REPO_ROOT / "src" / "models" / "LSTM_Aggregate" / "model.py")
s2c: Any = _load("stage2_common", REPO_ROOT / "src" / "models" / "Stage2_Common" / "stage2_common.py")
gm: Any = lm.gm
sm: Any = lm.sm

IntArr = npt.NDArray[np.int64]
Variant = Literal["final", "valbest"]
VARIANTS: tuple[Variant, ...] = ("final", "valbest")

# ============================================================
# 1. 設定
# ============================================================
MODEL_SAVE_PATH = REPO_ROOT / "outputs" / "checkpoints" / "gan_minimal.pt"

NUM_SLOTS: int = sm.NUM_SLOTS        # 96
NUM_ACT: int = sm.NUM_ACT            # 12
D_GROUPS: int = sm.D_GROUPS          # 28

# 生成器（VAE_Minimal のデコーダと同じ形・同じ幅）
Z_DIM = 16
HIDDEN = 222
COND_DIM = 64
TIME_DIM = 64
# critic
CRITIC_HIDDEN = 256
CRITIC_COND_DIM = 64
LEAKY_SLOPE = 0.2

# WGAN-GP の標準
LR = 1e-4
BETAS = (0.0, 0.9)
N_CRITIC = 5
GP_WEIGHT = 10.0
TAU = 1.0                            # 生成物を硬い one-hot にする straight-through Gumbel の温度
EPOCHS = 1000
G_STEPS_PER_EPOCH = 14               # 3,363 人を 256 人ずつ 1 巡する回数（ほかのモデルの 1 epoch の更新回数と同じ）
SEED: int = lm.SEED                  # 42
VAL_NOISE_SEED = 0                   # val の Wasserstein 推定の雑音の種（毎回同じにして val を決定的にする）
GEN_BATCH: int = lm.GEN_BATCH        # 1024
DEVICE = "cpu"


# ============================================================
# 2. モデル
# ============================================================
class GANGenerator(nn.Module):
    """生成器 p(x_s | z, c, s)。VAE_Minimal のデコーダと同じ形（全スロット共有の MLP 1 層）

    Attributes:
        z_dim: 潜在変数の次元
        hidden: 中間層の幅
    """

    def __init__(self, z_dim: int = Z_DIM, hidden: int = HIDDEN, cond_dim: int = COND_DIM,
                 time_dim: int = TIME_DIM) -> None:
        """部品を作る

        Args:
            z_dim: 潜在変数の次元, default=Z_DIM=16
            hidden: 中間層の幅, default=HIDDEN=222
            cond_dim: 条件の埋め込みの幅, default=COND_DIM=64
            time_dim: 時刻埋め込みの幅, default=TIME_DIM=64
        """
        super().__init__()
        self.z_dim, self.hidden = z_dim, hidden
        self.cond_embeds = nn.ModuleList([nn.Embedding(card, dim) for _, card, dim, _ in sm.COND_SPEC])
        self.cond_proj = nn.Linear(sum(sm.EMB_DIMS), cond_dim)
        self.time_embed = nn.Embedding(NUM_SLOTS, time_dim)
        self.dec_hidden = nn.Linear(z_dim + cond_dim + time_dim, hidden)
        self.dec_out = nn.Linear(hidden, NUM_ACT)

    def forward(self, z: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        """各スロットの活動の logits

        Args:
            z: 潜在変数, (B, Z_DIM)
            cond_idx: 条件インデックス, dtype=int64, (B, 3)

        Returns:
            logits, (B, 96, NUM_ACT)
        """
        batch = z.size(0)
        c = self.cond_proj(torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1))
        inp = torch.cat([z[:, None, :].expand(batch, NUM_SLOTS, -1),
                         c[:, None, :].expand(batch, NUM_SLOTS, -1),
                         self.time_embed.weight[None, :, :].expand(batch, -1, -1)], dim=2)
        return self.dec_out(F.relu(self.dec_hidden(inp)))


class Critic(nn.Module):
    """critic D(x, c)。[one-hot を平らにした 1,152 次元; 条件の埋め込み] → MLP 2 層（LeakyReLU）→ 実数 1 つ"""

    def __init__(self, hidden: int = CRITIC_HIDDEN, cond_dim: int = CRITIC_COND_DIM) -> None:
        """部品を作る

        Args:
            hidden: 中間層の幅, default=CRITIC_HIDDEN=256
            cond_dim: 条件の埋め込みの幅, default=CRITIC_COND_DIM=64
        """
        super().__init__()
        self.cond_embeds = nn.ModuleList([nn.Embedding(card, dim) for _, card, dim, _ in sm.COND_SPEC])
        self.cond_proj = nn.Linear(sum(sm.EMB_DIMS), cond_dim)
        self.net = nn.Sequential(
            nn.Linear(NUM_SLOTS * NUM_ACT + cond_dim, hidden), nn.LeakyReLU(LEAKY_SLOPE),
            nn.Linear(hidden, hidden), nn.LeakyReLU(LEAKY_SLOPE),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        """critic の値

        Args:
            x: one-hot の 1 日, (B, 96, NUM_ACT)（勾配ペナルティでは実データと生成物の混合）
            cond_idx: 条件インデックス, dtype=int64, (B, 3)

        Returns:
            critic の値, (B,)
        """
        c = self.cond_proj(torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1))
        return self.net(torch.cat([x.reshape(x.size(0), -1), c], dim=1)).squeeze(1)


def stage1_tag(z_dim: int = Z_DIM, hidden: int = HIDDEN) -> str:
    """Pre-trained の構成の印（例: _z16_h222）"""
    return f"_z{z_dim}_h{hidden}"


# ============================================================
# 3. 生成と損失
# ============================================================
def generate_hard(gen: GANGenerator, cond_idx: torch.Tensor, generator: torch.Generator | None,
                  tau: float = TAU) -> torch.Tensor:
    """z ~ N(0, I) から 1 日を作り、スロットごとに straight-through Gumbel で硬い one-hot にする

    Note:
        z と Gumbel 雑音はどちらも CPU で引く（generator が None なら大域の CPU の乱数）

    Args:
        gen: 生成器
        cond_idx: 条件インデックス, dtype=int64, (B, 3)。生成器と同じデバイス
        generator: CPU の torch.Generator。None なら大域の乱数
        tau: Gumbel-softmax の温度, default=TAU=1.0

    Returns:
        勾配の通る one-hot, (B, 96, NUM_ACT)
    """
    dev = cond_idx.device
    z = torch.randn((cond_idx.size(0), gen.z_dim), generator=generator).to(dev)
    logits = gen(z, cond_idx)                                                  # (B, 96, 12)
    if generator is None:
        # straight_through_gumbel は generator を要求するので、大域の乱数から種を 1 つ引いて作る
        generator = torch.Generator().manual_seed(int(torch.randint(0, 2 ** 62, (1,)).item()))
    y, _ = s2c.straight_through_gumbel(logits, generator, tau)
    return y


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
    """WGAN-GP の勾配ペナルティ E[(‖∇_x̂ D(x̂)‖₂ − 1)²]。x̂ は実データと生成物を一様乱数 ε で混ぜた点

    Note:
        ε は CPU の大域の乱数で引く

    Args:
        critic: critic
        real: 実データの one-hot, (B, 96, 12)
        fake: 生成物の one-hot（勾配を切ったもの）, (B, 96, 12)
        cond_idx: 条件インデックス, dtype=int64, (B, 3)

    Returns:
        ペナルティ（スカラー）
    """
    eps = torch.rand(real.size(0), 1, 1).to(real.device)
    mixed = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    score = critic(mixed, cond_idx)
    grad, = torch.autograd.grad(score.sum(), mixed, create_graph=True)
    return (grad.reshape(grad.size(0), -1).norm(dim=1) - 1.0).pow(2).mean()


def critic_step(gen: GANGenerator, critic: Critic, opt_d: torch.optim.Optimizer, cond_idx: torch.Tensor,
                sched: torch.Tensor) -> dict[str, float]:
    """critic を 1 回更新する: loss_d = mean D(fake) − mean D(real) + GP_WEIGHT · GP

    Args:
        gen: 生成器（更新しない）
        critic: critic
        opt_d: critic の optimizer
        cond_idx: 実データの条件, dtype=int64, (B, 3)
        sched: 実データの活動, dtype=int64, (B, 96)

    Returns:
        記録 {"loss_d", "w_est", "gp"}。w_est = mean D(real) − mean D(fake)（Wasserstein 推定）
    """
    for p in critic.parameters():
        p.requires_grad_(True)
    real = F.one_hot(sched, NUM_ACT).float()
    with torch.no_grad():
        fake = generate_hard(gen, cond_idx, None)
    d_real, d_fake = critic(real, cond_idx).mean(), critic(fake, cond_idx).mean()
    gp = gradient_penalty(critic, real, fake, cond_idx)
    loss_d = d_fake - d_real + GP_WEIGHT * gp
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    return {"loss_d": loss_d.item(), "w_est": (d_real - d_fake).item(), "gp": gp.item()}


def generator_loss(gen: GANGenerator, critic: Critic, cond_idx: torch.Tensor) -> torch.Tensor:
    """生成器の損失 −mean D(G(z, c))。critic には勾配を溜めない（requires_grad を切る）

    Args:
        gen: 生成器
        critic: critic（更新しない）
        cond_idx: 条件インデックス, dtype=int64, (B, 3)

    Returns:
        損失（スカラー）。生成器のパラメータに勾配が届く
    """
    for p in critic.parameters():
        p.requires_grad_(False)
    return -critic(generate_hard(gen, cond_idx, None), cond_idx).mean()


def val_wasserstein(gen: GANGenerator, critic: Critic, loader: DataLoader) -> float:
    """val 分割の Wasserstein 推定 mean D(real) − mean D(fake)（人数の加重平均）。雑音を VAL_NOISE_SEED で固定

    Args:
        gen: 生成器
        critic: critic
        loader: val 分割の DataLoader（shuffle なし）

    Returns:
        Wasserstein 推定（小さいほど生成物が実データに近いと critic が判断している）
    """
    gen.eval()
    critic.eval()
    dev = next(gen.parameters()).device
    noise = torch.Generator().manual_seed(VAL_NOISE_SEED)
    total, count = 0.0, 0.0
    with torch.no_grad():
        for cond_idx, sched, _ in loader:
            cond_idx, sched = cond_idx.to(dev), sched.to(dev)
            real = F.one_hot(sched, NUM_ACT).float()
            fake = generate_hard(gen, cond_idx, noise)
            w = (critic(real, cond_idx).mean() - critic(fake, cond_idx).mean()).item()
            total += w * sched.size(0)
            count += sched.size(0)
    return total / count


# ============================================================
# 4. 学習と読み込み
# ============================================================
def make_optimizer(params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
    """WGAN-GP の標準の Adam（学習率 1e-4、β = (0, 0.9)）"""
    return torch.optim.Adam(params, lr=LR, betas=BETAS)


def real_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """学習分割のミニバッチを epoch をまたいで返し続ける"""
    while True:
        yield from loader


def train(epochs: int = EPOCHS, seed: int = SEED, device: str = DEVICE,
          save: bool = True) -> tuple[GANGenerator, Critic, dict[str, list[float]]]:
    """WGAN-GP を固定の epoch 数だけ学習し、最終 epoch と val 最良の 2 つの ckpt を保存する

    Args:
        epochs: epoch 数（固定。早期終了なし）, default=EPOCHS=1000
        seed: 学習の乱数の種, default=SEED=42
        device: 学習するデバイス, default=DEVICE="cpu"
        save: False なら何も保存しない

    Returns:
        (最終 epoch の生成器, 最終 epoch の critic, 学習曲線 {"epoch", "loss_d", "w_est", "gp", "loss_g", "val_w", "sec"})

    Raises:
        ValueError: epochs < 1 のとき
    """
    if epochs < 1:
        raise ValueError(f"epochs は 1 以上: {epochs}")
    torch.manual_seed(seed)
    train_part, val_part = gm.load_split()
    gen, critic = GANGenerator().to(device), Critic().to(device)
    opt_g, opt_d = make_optimizer(gen.parameters()), make_optimizer(critic.parameters())
    reals = real_batches(gm.make_loader(train_part, shuffle=True))
    val_loader = gm.make_loader(val_part, shuffle=False)
    n_gen = sum(p.numel() for p in gen.parameters())
    print(f"device={device}  train={len(train_part.sched)}  val={len(val_part.sched)}  生成器 {n_gen:,}  "
          f"critic {sum(p.numel() for p in critic.parameters()):,}", flush=True)

    history: dict[str, list[float]] = {k: [] for k in ("epoch", "loss_d", "w_est", "gp", "loss_g", "val_w", "sec")}
    best_val, best_epoch = math.inf, 0
    best: dict[str, Any] | None = None
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        gen.train()
        critic.train()
        logs: list[dict[str, float]] = []
        for _ in range(G_STEPS_PER_EPOCH):
            cond_idx = torch.empty(0)
            for _ in range(N_CRITIC):
                cond_idx, sched, _ = next(reals)
                cond_idx, sched = cond_idx.to(device), sched.to(device)
                log = critic_step(gen, critic, opt_d, cond_idx, sched)
            loss_g = generator_loss(gen, critic, cond_idx)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            logs.append({**log, "loss_g": loss_g.item()})
        val_w = val_wasserstein(gen, critic, val_loader)
        sec = time.perf_counter() - t0
        mean = {k: float(np.mean([r[k] for r in logs])) for k in ("loss_d", "w_est", "gp", "loss_g")}
        for key, val in (("epoch", ep), *mean.items(), ("val_w", val_w), ("sec", sec)):
            history[key].append(float(val))
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:4d} | loss_d {mean['loss_d']:+.4f} | W {mean['w_est']:.4f} | gp {mean['gp']:.4f} | "
                  f"loss_g {mean['loss_g']:+.4f} | val W {val_w:.4f} | {sec:.2f}s", flush=True)
        if val_w < best_val:
            best_val, best_epoch = val_w, ep
            best = snapshot(gen, critic, opt_d)
    print(f"val 最良: epoch {best_epoch}（val W {best_val:.4f}）", flush=True)
    if save:
        assert best is not None
        extra = {"epochs": epochs, "history": history, "best_epoch": best_epoch, "best_val": best_val}
        save_ckpt(snapshot(gen, critic, opt_d), ckpt_path(seed, "final"), seed, {**extra, "epoch": epochs})
        save_ckpt(best, ckpt_path(seed, "valbest"), seed, {**extra, "epoch": best_epoch})
        print(f"saved {ckpt_path(seed, 'final')} と {ckpt_path(seed, 'valbest')}")
    gen.eval()
    critic.eval()
    return gen, critic, history


def snapshot(gen: GANGenerator, critic: Critic, opt_d: torch.optim.Optimizer) -> dict[str, Any]:
    """生成器・critic・critic の optimizer の状態の複製（Stage 2 で critic の学習を続けるため optimizer も残す）"""
    return {"generator": copy.deepcopy(gen.state_dict()), "critic": copy.deepcopy(critic.state_dict()),
            "critic_optimizer": copy.deepcopy(opt_d.state_dict())}


def save_ckpt(state: dict[str, Any], path: Path, seed: int, extra: dict[str, Any] | None = None) -> None:
    """snapshot の状態と出所の記録（config）を保存する"""
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"z_dim": Z_DIM, "hidden": HIDDEN, "n_critic": N_CRITIC, "gp_weight": GP_WEIGHT,
                              "lr": LR, "betas": BETAS, "seed": seed}
    if extra is not None:
        config.update(extra)
    torch.save({**state, "config": config}, path)


def load_ckpt(path: Path, device: str = DEVICE) -> tuple[GANGenerator, Critic, dict[str, Any]]:
    """save_ckpt の逆。(生成器, critic, critic の optimizer の state_dict) を返す（どちらも eval モード）"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    gen, critic = GANGenerator().to(device), Critic().to(device)
    gen.load_state_dict(ckpt["generator"])
    critic.load_state_dict(ckpt["critic"])
    return gen.eval(), critic.eval(), ckpt["critic_optimizer"]


def ckpt_path(seed: int, variant: Variant) -> Path:
    """ckpt のパス（例: 種 42 の最終 epoch は gan_minimal_z16_h222_final.pt）

    Raises:
        ValueError: variant が VARIANTS に無いとき
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant は {VARIANTS} のどれか: {variant}")
    stem = f"{MODEL_SAVE_PATH.stem}{stage1_tag()}_{variant}{lm.run_suffix(seed)}"
    return MODEL_SAVE_PATH.with_name(f"{stem}{MODEL_SAVE_PATH.suffix}")


# ============================================================
# 5. 評価のプール
# ============================================================
@torch.no_grad()
def group_pool(gen: GANGenerator, n_per_group: int, seed: int) -> IntArr:
    """群別サンプルプール (28, n_per_group, 96)。z ~ N(0, I) → スロットごとに独立に温度 1 で引く（VAE と同じ引き方）"""
    gen.eval()
    dev = next(gen.parameters()).device
    flat = torch.as_tensor(sm.cond_grid(), dtype=torch.long).repeat_interleave(n_per_group, dim=0)
    g = torch.Generator().manual_seed(seed)
    outs = []
    for i in range(0, flat.size(0), GEN_BATCH):
        cond = flat[i:i + GEN_BATCH].to(dev)
        z = torch.randn((cond.size(0), gen.z_dim), generator=g).to(dev)
        logits = gen(z, cond)
        outs.append(gm.draw_categorical(logits.reshape(-1, NUM_ACT), g).reshape(cond.size(0), NUM_SLOTS))
    return torch.cat(outs).numpy().astype(np.int64).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


# ============================================================
# 6. CLI
# ============================================================
def smoke(device: str = DEVICE) -> None:
    """2 epoch 学習し、生成の形を確かめる（何も保存しない）"""
    gen, _, history = train(epochs=2, device=device, save=False)
    assert len(history["val_w"]) == 2 and all(math.isfinite(v) for v in history["val_w"])
    pool = group_pool(gen, 2, seed=0)
    assert pool.shape == (D_GROUPS, 2, NUM_SLOTS) and 0 <= pool.min() and pool.max() < NUM_ACT
    print(f"smoke: OK（生成器 {sum(p.numel() for p in gen.parameters()):,}）")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GAN_Minimal: 単純な MLP の条件付き WGAN-GP の Stage 1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default=DEVICE)
    ap.add_argument("--threads", type=int, default=0, help="CPU の計算のスレッド数。0 なら torch の既定")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.smoke:
        smoke(args.device)
        return
    print(f"[config] seed={args.seed} epochs={args.epochs} device={args.device}")
    train(epochs=args.epochs, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
