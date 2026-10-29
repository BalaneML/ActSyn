"""
stage2_adapter.py
=================
GAN_Minimal（WGAN-GP）を共通の Stage 2（Stage2_Common/stage2_common.py）で動かすための adapter

Stage 1 の 2 つの ckpt（最終 epoch と val 最良）を、別のモデル名として登録する:
    gan_minimal_final    → gan_minimal_z16_h222_final.pt から始める
    gan_minimal_valbest  → gan_minimal_z16_h222_valbest.pt から始める

    sample_relaxed   = gg.generate_hard（z ~ N(0, I) → スロットごとに straight-through Gumbel。z も Gumbel 雑音も
                       渡された generator から引く）を (B, 12, 96) に並べ替えたもの
    rehearsal_loss   = gg.generator_loss（−mean D(G(z, c))。Stage 1 の生成器の損失。z と雑音は CPU の大域の乱数）
    auxiliary_step   = gg.critic_step を N_CRITIC = 5 回（Stage 2 の間も critic を ATUS で学習し続ける。
                       実データはリハーサルと同じ並びから取る）
    extra_state      = critic の重みと critic の optimizer（再開に要る）
    val_loss         = gg.val_wasserstein（雑音を固定した val の Wasserstein 推定。Stage 1 の val と同じ定義）
    group_pool       = gg.group_pool（z ~ N(0, I) → スロットごとに独立に温度 1 で引く）

```mermaid
flowchart LR
    B["build(name, seed, device)"] --> LC["gg.load_ckpt(gg.ckpt_path(seed, variant), device)<br/>gen, critic, critic_optimizer"]
    LC --> GA["GANAdapter(gen, critic, critic_optimizer, variant, seed)"]
    GA -- "auxiliary_step(next_batch)" --> CS["gg.critic_step × N_CRITIC"]
    GA -- "sample_relaxed(cond_idx, generator, tau)" --> GH["gg.generate_hard(gen, cond_idx, generator, tau)<br/>(B, 96, 12) → (B, 12, 96)"]
    GA -- "rehearsal_loss(cond_idx, sched, weight)" --> GL["gg.generator_loss(gen, critic, cond_idx)"]
```
"""
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
COMMON_PATH = HERE.parents[0] / "Stage2_Common" / "stage2_common.py"


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


s2c: Any = _load("stage2_common", COMMON_PATH)
gg: Any = _load("gan_minimal_model", HERE / "model.py")

IntArr = npt.NDArray[np.int64]
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
NAME_HEAD = "gan_minimal"
NAMES: tuple[str, ...] = tuple(f"{NAME_HEAD}_{v}" for v in gg.VARIANTS)      # gan_minimal_final / gan_minimal_valbest


class GANAdapter(s2c.Stage2Adapter):
    """GAN_Minimal の Stage2Adapter。更新するのは生成器だけで、critic は auxiliary_step で自分の optimizer が更新する

    Attributes:
        name: gan_minimal_final / gan_minimal_valbest
        model: 生成器 gg.GANGenerator
        critic: gg.Critic
        opt_d: critic の Adam（Stage 1 の状態から続ける）
        seed: Pre-trained の種
        variant: final / valbest
    """

    def __init__(self, gen: Any, critic: Any, critic_optimizer: dict[str, Any], variant: str, seed: int) -> None:
        """
        Args:
            gen: Pre-trained の生成器
            critic: Pre-trained の critic
            critic_optimizer: Stage 1 の critic の optimizer の state_dict
            variant: final / valbest
            seed: Pre-trained の種
        """
        self.name = f"{NAME_HEAD}_{variant}"
        self.model = gen
        self.critic = critic
        self.opt_d = gg.make_optimizer(critic.parameters())
        self.opt_d.load_state_dict(critic_optimizer)
        self.variant = variant
        self.seed = seed

    def stage1_tag(self) -> str:
        """Pre-trained の構成の印（_z16_h222）。どちらの ckpt から始めたかは name に入っている"""
        return gg.stage1_tag()

    def sample_relaxed(self, cond_idx: torch.Tensor, generator: torch.Generator, tau: float) -> torch.Tensor:
        """z ~ N(0, I) から 96 スロットを独立に引く。前向きは硬い one-hot, (B, 12, 96)"""
        return gg.generate_hard(self.model, cond_idx, generator, tau).permute(0, 2, 1).contiguous()

    def rehearsal_loss(self, cond_idx: torch.Tensor, sched: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """生成器の損失 −mean D(G(z, c))（Stage 1 と同じ。sched・weight は使わない）"""
        del sched, weight
        return gg.generator_loss(self.model, self.critic, cond_idx)

    def auxiliary_step(self, next_batch: Callable[[], Batch]) -> dict[str, float]:
        """critic を N_CRITIC 回更新する（実データはリハーサルと同じ並びから取る）

        Returns:
            最後の更新の記録 {"critic_loss_d", "critic_w_est", "critic_gp"}
        """
        dev = next(self.model.parameters()).device
        self.critic.train()
        log: dict[str, float] = {}
        for _ in range(gg.N_CRITIC):
            cond_idx, sched, _ = next_batch()
            log = gg.critic_step(self.model, self.critic, self.opt_d, cond_idx.to(dev), sched.to(dev))
        return {f"critic_{k}": v for k, v in log.items()}

    def extra_state_dict(self) -> dict[str, Any]:
        """critic の重みと optimizer（再開に要る）"""
        return {"critic": self.critic.state_dict(), "critic_optimizer": self.opt_d.state_dict()}

    def load_extra_state_dict(self, state: dict[str, Any]) -> None:
        """extra_state_dict の逆"""
        self.critic.load_state_dict(state["critic"])
        self.opt_d.load_state_dict(state["critic_optimizer"])

    def val_loss(self, loader: DataLoader) -> float:
        """val の Wasserstein 推定（雑音を固定。Stage 1 の val と同じ定義）"""
        return gg.val_wasserstein(self.model, self.critic, loader)

    def group_pool(self, n_per_group: int, seed: int) -> IntArr:
        """評価のプール (28, n_per_group, 96)"""
        return gg.group_pool(self.model, n_per_group, seed)

    def config(self) -> dict[str, Any]:
        """ckpt の config に書く情報（Pre-trained の ckpt の名前と critic の更新回数）"""
        return {"stage1_ckpt": gg.ckpt_path(self.seed, self.variant).name, "n_critic": gg.N_CRITIC}


def build(name: str, seed: int, device: str) -> GANAdapter:
    """Pre-trained を載せた adapter を作る

    Raises:
        ValueError: name が NAMES に無いとき
        FileNotFoundError: Pre-trained の ckpt が無いとき
    """
    if name not in NAMES:
        raise ValueError(f"name は {NAMES} のどれか: {name}")
    variant = name.removeprefix(f"{NAME_HEAD}_")
    path = gg.ckpt_path(seed, variant)
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に src/models/GAN_Minimal/model.py --seed {seed} で Stage 1 を学習する")
    gen, critic, critic_optimizer = gg.load_ckpt(path, device)
    return GANAdapter(gen, critic, critic_optimizer, variant, seed)
