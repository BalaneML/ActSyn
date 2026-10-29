"""
stage2_adapter.py
=================
LSTM_Aggregate（と、セルだけが違う GRU_Minimal）を、共通の Stage 2（Stage2_Common/stage2_common.py）で動かすための adapter

共通の Stage 2 が求める操作（Stage2Adapter）を、既存の LSTM の Stage 2（stage2_agg.py）の関数で実装する:

    sample_relaxed   = stage2_agg.sample_soft（96 スロットを順に straight-through Gumbel で引く。履歴の勾配は切らない）
    rehearsal_loss   = lm.batch_loss（teacher forcing の交差エントロピー、重みなし。Stage 1 と同じ損失）
    group_pool       = lm.group_pool（評価のプール。温度 1 の祖先サンプリング）
    Pre-trained      = stage2_agg.load_pretrained（H = 64・1 層・weight decay 0.01・学習型の時刻符号）

```mermaid
flowchart LR
    B["build(name, seed, device)"] --> LP["s2.load_pretrained(seed, name, device)<br/>model"]
    LP --> RA["RecurrentAdapter(name, model, seed)"]
    RA -- "sample_relaxed(cond_idx, generator, tau)" --> SS["s2.sample_soft(model, cond_idx, generator, tau, BPTT)<br/>y (B, 12, 96)"]
    RA -- "rehearsal_loss(cond_idx, sched, weight)" --> BL["lm.batch_loss(model, cond_idx, sched, weight)"]
    RA -- "group_pool(n_per_group, seed)" --> GP["lm.group_pool(model, n_per_group, seed)<br/>(28, n_per_group, 96)"]
```
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

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
s2: Any = _load("lstm_stage2_agg", HERE / "stage2_agg.py")
lm: Any = s2.lm

IntArr = npt.NDArray[np.int64]
NAMES: tuple[str, ...] = tuple(s2.CELLS)          # ("lstm", "gru_minimal")
BPTT = 0                                         # 履歴の勾配を切らない（96 スロットすべてを逆伝播する）


class RecurrentAdapter(s2c.Stage2Adapter):
    """LSTM か GRU_Minimal の Stage2Adapter

    Attributes:
        name: lstm / gru_minimal
        model: lm.LSTMScheduler か GRUMinimalScheduler
        seed: Pre-trained の種
    """

    def __init__(self, name: str, model: Any, seed: int) -> None:
        """
        Args:
            name: lstm / gru_minimal
            model: Pre-trained のモデル
            seed: Pre-trained の種
        """
        self.name = name
        self.model = model
        self.seed = seed

    def stage1_tag(self) -> str:
        """Pre-trained の構成の印（_time_learned_h64_wd0.01）"""
        return s2.stage1_tag()

    def sample_relaxed(self, cond_idx: torch.Tensor, generator: torch.Generator, tau: float) -> torch.Tensor:
        """96 スロットを順に引く。勾配は 96 スロットの履歴を通る（stage2_agg.sample_soft）"""
        y, _, _ = s2.sample_soft(self.model, cond_idx, generator, tau, BPTT)
        return y

    def rehearsal_loss(self, cond_idx: torch.Tensor, sched: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """teacher forcing の交差エントロピー（重みなし、1 スロットあたりの nats）"""
        return lm.batch_loss(self.model, cond_idx, sched, weight)

    def group_pool(self, n_per_group: int, seed: int) -> IntArr:
        """評価のプール (28, n_per_group, 96)（温度 1 の祖先サンプリング）"""
        return lm.group_pool(self.model, n_per_group, seed=seed)

    def config(self) -> dict[str, Any]:
        """ckpt の config に書く情報（Pre-trained の ckpt の名前と BPTT）"""
        return {"stage1_ckpt": s2.stage1_ckpt(self.seed, self.name).name, "bptt": BPTT}


def build(name: str, seed: int, device: str) -> RecurrentAdapter:
    """Pre-trained を載せた adapter を作る

    Args:
        name: lstm / gru_minimal
        seed: Pre-trained の種
        device: 載せる計算場所

    Returns:
        adapter

    Raises:
        ValueError: name が NAMES に無いとき
    """
    if name not in NAMES:
        raise ValueError(f"name は {NAMES} のどれか: {name}")
    return RecurrentAdapter(name, s2.load_pretrained(seed, name, device), seed)
