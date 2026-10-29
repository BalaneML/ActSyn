"""
stage2_adapter.py
=================
DDPM_Minimal を共通の Stage 2（Stage2_Common/stage2_common.py）で動かすための adapter

    sample_relaxed   = dm.sample_relaxed_chain（逆過程 1,000 ステップ。末尾 K_TAIL = 1 ステップだけ勾配を通し、
                       x0 を straight-through で one-hot にする。雑音は渡された generator から作った計算場所の generator で引く）
    rehearsal_loss   = dm.eps_mse（ε の二乗誤差、重みなし。t と ε は CPU の大域の乱数）
    val_loss         = dm.val_loss（t と ε を固定した決定的な値。Stage 1 の早期終了と同じ定義）
    group_pool       = dm.group_pool（評価のプール。逆過程 1,000 ステップ → argmax、CFG なし）
    Pre-trained      = dm.load_model(dm.ckpt_path(seed))

```mermaid
flowchart LR
    B["build(name, seed, device)"] --> LM["dm.load_model(dm.ckpt_path(seed), device)<br/>model"]
    LM --> DA["DDPMAdapter(model, seed, device)<br/>diffusion = dm.base.Diffusion(device)"]
    DA -- "sample_relaxed(cond_idx, generator, tau)" --> SC["dm.sample_relaxed_chain(model, diffusion, cond_idx, generator, tau, dm.K_TAIL)<br/>y (B, 12, 96)"]
    DA -- "rehearsal_loss(cond_idx, sched, weight)" --> EM["dm.eps_mse(model, diffusion, cond_idx, sched)"]
    DA -- "group_pool(n_per_group, seed)" --> GP["dm.group_pool(model, diffusion, n_per_group, seed)"]
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
dm: Any = _load("ddpm_minimal_model", HERE / "model.py")

IntArr = npt.NDArray[np.int64]
NAME = "ddpm_minimal"


class DDPMAdapter(s2c.Stage2Adapter):
    """DDPM_Minimal の Stage2Adapter

    Attributes:
        name: "ddpm_minimal"
        model: dm.DDPMMinimal
        seed: Pre-trained の種
        diffusion: dm.base.Diffusion（拡散のスケジュール。model と同じ計算場所）
    """

    def __init__(self, model: Any, seed: int, device: str) -> None:
        """
        Args:
            model: Pre-trained の DDPMMinimal
            seed: Pre-trained の種
            device: model を載せた計算場所
        """
        self.name = NAME
        self.model = model
        self.seed = seed
        self.diffusion = dm.base.Diffusion(device)

    def stage1_tag(self) -> str:
        """Pre-trained の構成の印（_ch8_wd0.01）"""
        return dm.stage1_tag()

    def sample_relaxed(self, cond_idx: torch.Tensor, generator: torch.Generator, tau: float) -> torch.Tensor:
        """逆過程で 1 日を生成し、末尾 K_TAIL ステップだけ勾配を通す。前向きは argmax の one-hot, (B, 12, 96)"""
        return dm.sample_relaxed_chain(self.model, self.diffusion, cond_idx, generator, tau, dm.K_TAIL)

    def rehearsal_loss(self, cond_idx: torch.Tensor, sched: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """ε の二乗誤差（重みなし。weight は使わない）"""
        del weight
        return dm.eps_mse(self.model, self.diffusion, cond_idx, sched)

    def val_loss(self, loader: DataLoader) -> float:
        """val 分割の ε の二乗誤差（t と ε を固定。Stage 1 の早期終了と同じ定義）"""
        return dm.val_loss(self.model, self.diffusion, loader)

    def group_pool(self, n_per_group: int, seed: int) -> IntArr:
        """評価のプール (28, n_per_group, 96)"""
        return dm.group_pool(self.model, self.diffusion, n_per_group, seed)

    def config(self) -> dict[str, Any]:
        """ckpt の config に書く情報（Pre-trained の ckpt の名前と末尾のステップ数）"""
        return {"stage1_ckpt": dm.ckpt_path(self.seed).name, "k_tail": dm.K_TAIL}


def build(name: str, seed: int, device: str) -> DDPMAdapter:
    """Pre-trained を載せた adapter を作る

    Raises:
        ValueError: name が NAME でないとき
        FileNotFoundError: Pre-trained の ckpt が無いとき
    """
    if name != NAME:
        raise ValueError(f"name は {NAME}: {name}")
    path = dm.ckpt_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に src/models/DDPM_Minimal/model.py --seed {seed} で Stage 1 を学習する")
    return DDPMAdapter(dm.load_model(path, device), seed, device)
