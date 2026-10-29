"""
stage2_adapter.py
=================
VAE_Minimal を共通の Stage 2（Stage2_Common/stage2_common.py）で動かすための adapter

    sample_relaxed   = z ~ N(0, I) → decode → スロットごとに straight-through Gumbel（s2c.straight_through_gumbel）
                       ★z も Gumbel 雑音も、渡された generator から引く（再開のときに同じ標本を引き直すため）
    rehearsal_loss   = vm.elbo（1 人あたりの負の ELBO、重みなし。Stage 1 と同じ損失。正規化しない）
    val_loss         = vm.val_loss（雑音の乱数を固定した決定的な値。Stage 1 の早期終了と同じ定義）
    group_pool       = vm.group_pool（評価のプール。z ~ N(0, I)、スロットごとに温度 1 で引く）
    Pre-trained      = vm.load_model(vm.ckpt_path(seed))

集計の勾配は、デコーダ・条件・時刻埋め込みにだけ届く（生成にエンコーダを使わないため）。
エンコーダはリハーサル（ELBO）の勾配で動く。

```mermaid
flowchart LR
    B["build(name, seed, device)"] --> LM["vm.load_model(vm.ckpt_path(seed), device)<br/>model"]
    LM --> VA["VAEAdapter(model, seed)"]
    VA -- "sample_relaxed(cond_idx, generator, tau)" --> PL["vm.prior_logits(model, cond_idx, generator)<br/>logits (B, 96, 12)"]
    PL --> ST["s2c.straight_through_gumbel(logits, generator, tau)<br/>y (B, 96, 12) → permute → (B, 12, 96)"]
    VA -- "rehearsal_loss(cond_idx, sched, weight)" --> EL["vm.elbo(model, cond_idx, sched)"]
    VA -- "group_pool(n_per_group, seed)" --> GP["vm.group_pool(model, n_per_group, seed)"]
```
"""
import importlib.util
import sys
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
vm: Any = _load("vae_minimal_model", HERE / "model.py")

IntArr = npt.NDArray[np.int64]
NAME = "vae_minimal"


class VAEAdapter(s2c.Stage2Adapter):
    """VAE_Minimal の Stage2Adapter

    Attributes:
        name: "vae_minimal"
        model: vm.VAEMinimal
        seed: Pre-trained の種
    """

    def __init__(self, model: Any, seed: int) -> None:
        """
        Args:
            model: Pre-trained の VAEMinimal
            seed: Pre-trained の種
        """
        self.name = NAME
        self.model = model
        self.seed = seed

    def stage1_tag(self) -> str:
        """Pre-trained の構成の印（_z16_h222_wd0.01）"""
        return vm.stage1_tag()

    def sample_relaxed(self, cond_idx: torch.Tensor, generator: torch.Generator, tau: float) -> torch.Tensor:
        """z ~ N(0, I) から 96 スロットを独立に引く。前向きは硬い one-hot, (B, 12, 96)"""
        logits = vm.prior_logits(self.model, cond_idx, generator)                # (B, 96, 12)
        y, _ = s2c.straight_through_gumbel(logits, generator, tau)               # (B, 96, 12)
        return y.permute(0, 2, 1).contiguous()

    def rehearsal_loss(self, cond_idx: torch.Tensor, sched: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """1 人あたりの負の ELBO（重みなし。weight は使わない）"""
        del weight
        loss, _, _ = vm.elbo(self.model, cond_idx, sched)
        return loss

    def val_loss(self, loader: DataLoader) -> float:
        """val 分割の負の ELBO（雑音の乱数を固定。Stage 1 の早期終了と同じ定義）"""
        value, _, _ = vm.val_loss(self.model, loader)
        return value

    def group_pool(self, n_per_group: int, seed: int) -> IntArr:
        """評価のプール (28, n_per_group, 96)"""
        return vm.group_pool(self.model, n_per_group, seed)

    def config(self) -> dict[str, Any]:
        """ckpt の config に書く情報（Pre-trained の ckpt の名前）"""
        return {"stage1_ckpt": vm.ckpt_path(self.seed).name}


def build(name: str, seed: int, device: str) -> VAEAdapter:
    """Pre-trained を載せた adapter を作る

    Args:
        name: "vae_minimal"
        seed: Pre-trained の種
        device: 載せる計算場所

    Returns:
        adapter

    Raises:
        ValueError: name が NAME でないとき
        FileNotFoundError: Pre-trained の ckpt が無いとき
    """
    if name != NAME:
        raise ValueError(f"name は {NAME}: {name}")
    path = vm.ckpt_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に src/models/VAE_Minimal/model.py --seed {seed} で Stage 1 を学習する")
    return VAEAdapter(vm.load_model(path, device), seed)
