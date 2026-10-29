"""
stage2_common.py
================
モデルによらない共通の Stage 2（2 段階学習の 2 段目）。2026-10-29 のユーザー決定で新設

Stage 1（各モデルの通常の学習、ATUS）で作った Pre-trained を、次の損失で微調整する:

    L = L_agg + λ · L_reh
    L_agg = mean ω (Â − A*)²            生成した 1 日の硬い one-hot y を群ごとに平均した Â と、日本の教師 A* の素の MSE（ω = 1）
    L_reh = Stage 1 と同じ損失            ATUS の学習分割のミニバッチ（重みなし）

モデルごとに違う部分は Stage2Adapter（モデルごとに 1 つ実装する）に閉じ込め、学習のループは全モデルで同じにする。

全モデルで同じにする約束事（2026-10-23〜29 のユーザー決定）:
    1. 集計に使う y は硬い one-hot（前向きはモデル自身の標本、後ろ向きは連続な近似。カテゴリの logits を出すモデルは
       straight-through Gumbel-softmax、τ = 1）
    2. L_agg は素の MSE（ω = 1）。step 数 1,000・d_sub 7・n 256・AdamW（weight decay 0）・clip 1.0・学習率 1e-3 は 1 つ
       （パラメータの群分けはしない）
    3. リハーサルは重みなし。評価は米国加重（採点の関数 gs2.score_pool の側）
    4. 判定に使う ckpt は最終 step
    5. λ は未決（リハーサル損失の単位と一緒に決める）。既定の 0.01 は LSTM の本番の値で、暫定

★再開（--resume）は、止めずに回した場合とビット単位で同じ結果になるように作る。乱数は 4 つある:
    1. numpy の rng（群の抽出 d_pick）
    2. Gumbel 雑音の CPU の torch.Generator（gumbel）
    3. torch の大域の乱数（DataLoader の shuffle。iterator を作るたびに 1 つ、epoch の最初のバッチで 1 つ消費する）
    4. モデル固有の乱数（GAN の critic など。adapter.extra_state_dict に入れる）
  3 は epoch の途中の位置を DataLoader から取り出せないので、RehearsalStream が「epoch の iterator を作る直前の乱数の
  状態」と「その epoch で取ったバッチ数」を持ち、再開のときに同じ並びを作り直す

★既存の LSTM の Stage 2（LSTM_Aggregate/stage2_agg.py）と、乱数の消費の順序を含めて同じ計算をする。
  lr を 3 群とも 1e-3 にした本番の設定（_lam0.01_mse_lr0.001_steps1000_cpu）と、重みがビット単位で一致することを
  test_stage2_common.py で確かめる

データフロー（1 更新）:

```mermaid
flowchart TD
    LA0["load_adapter(name, seed, device)<br/>Pre-trained を載せた adapter"] --> AD["adapter.model"]
    RNG["rng（numpy）"] --> DP["d_pick_t (d_sub,)"]
    DP --> CD["cond = grid[d_pick_t].repeat_interleave(settings.n)<br/>(d_sub·n, 3)"]
    CD --> SR["adapter.sample_relaxed(cond, gumbel, settings.tau)<br/>y (d_sub·n, 12, 96)"]
    AD --> SR
    SR --> GR["sl.group_rates(y, settings.n)<br/>a_hat (d_sub, 12, 96)"]
    AST["a_star_all[d_pick_t]<br/>a_star (d_sub, 12, 96)"] --> LG
    GR --> LG["sl.agg_mse_from_rates(a_hat, a_star, omega)<br/>l_agg"]
    LG -- "l_agg.backward()" --> GRAD["θ.grad = g_agg"]
    STR["stream.next_batch()<br/>(b_cond, b_sched, b_weight)"] --> RH["adapter.rehearsal_loss(b_cond, b_sched, b_weight)<br/>l_reh"]
    RH -- "ft.add_rehearsal_grad(params, settings.lam * l_reh)" --> GRAD
    GRAD --> OPT["clip_grad_norm_(params, settings.grad_clip) → optimizer.step()<br/>AdamW・学習率 1 つ"]
    OPT --> AD
    AD -- "settings.save_every ごと" --> SAVE["save_checkpoint(...)<br/>stage2_step{S}.pt・history.csv"]
```

使い方:
    .venv/bin/python src/models/Stage2_Common/stage2_common.py --smoke
    .venv/bin/python src/models/Stage2_Common/stage2_common.py --model lstm --seed 42 --zero-shot
    .venv/bin/python src/models/Stage2_Common/stage2_common.py --model lstm --seed 42 --device cpu --threads 1
    .venv/bin/python src/models/Stage2_Common/stage2_common.py --model lstm --seed 42 --device cpu --threads 1 --resume

    フラグ:
        --model M        モデル（MODEL_ADAPTERS のキー: lstm / gru_minimal）
        --seed S         種（Pre-trained の ckpt と Stage 2 の乱数）
        --zero-shot      学習せず Pre-trained を採点する（E0）
        --lam V          リハーサルの重み λ（既定 0.01、暫定）
        --steps N        更新回数（既定 1,000）
        --resume         保存済みの最新の ckpt から続ける。無ければ最初から。
                         ★--resume なしで ckpt が既にあるときは止まる（上書きしない）
        --device D       cpu / mps / cuda
        --threads T      CPU の計算のスレッド数（0 は torch の既定）

出力（{stem} = stage2_common_{モデル}{Stage 1 の印}{変種}_seed{種}_{実行}、E0 は変種なし）:
    outputs/checkpoints/{stem}/stage2_step{S}.pt     save_every ごとの ckpt（再開に必要なものをすべて含む）
    outputs/checkpoints/{stem}/history.csv           1 更新ごとの記録（ckpt を保存するたびに書き直す）
    data/processed/aggregates/{stem}.csv             採点（GRU・LSTM の Stage 2 と同じ縦持ち）
    outputs/generated/{stem}_rates.npz               評価のプールの時刻別行動者率 (28, 12, 96)
        {変種} = _lam{λ}[_steps{N}][_dsub{K}][_n{N}][_tau{τ}][_lr{値}][_wd{値}][_clip{値}][_val{K}]_{device}
        （λ と計算場所は常に、それ以外は既定と違うときだけ書く）
"""
import argparse
import dataclasses
import importlib.util
import math
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"
GRU_DIR = REPO_ROOT / "src" / "models" / "GRU_Aggregate"
LSTM_DIR = REPO_ROOT / "src" / "models" / "LSTM_Aggregate"

# ★adapter のファイルはこのモジュールを MODULE_NAME で読む。CLI で __main__ として動かしたときも同じモジュールを
#   指すように登録しておく（別の名前で 2 回読むと Stage2Adapter のクラスが 2 つでき、isinstance が外れる）
MODULE_NAME = "stage2_common"
sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名ファイルの取り違えを防ぐ。同じファイルが既に載っていればそれを返す）"""
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


# 名前は LSTM_Aggregate/stage2_agg.py と同じにする（同じモジュールを 2 回読まない）
sm: Any = _load("gru_ddpm_simple_model", SIMPLE_DIR / "model.py")
gs2: Any = _load("gru_aggregate_stage2", GRU_DIR / "stage2.py")      # 採点（score_pool）・教師の読み込み
st: Any = gs2.st
cur: Any = gs2.cur
gm: Any = gs2.gm                                                     # ATUS の分割と DataLoader
sl: Any = _load("simple_stage2_loss", SIMPLE_DIR / "stage2_loss.py")
ck: Any = _load("simple_stage2_checkpoint", SIMPLE_DIR / "stage2_checkpoint.py")
ft: Any = _load("simple_stage2_finetune", SIMPLE_DIR / "stage2_finetune.py")

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]              # (cond_idx (B, 3), sched (B, 96), weight (B,))

# モデル → adapter のファイル。各ファイルは build(name, seed, device) -> Stage2Adapter を持つ
MODEL_ADAPTERS: dict[str, Path] = {
    "lstm": LSTM_DIR / "stage2_adapter.py",
    "gru_minimal": LSTM_DIR / "stage2_adapter.py",
}

# 採点（GRU・LSTM の Stage 2 と同じ）
EVAL_N: int = gs2.EVAL_N            # 2000
EVAL_SEED: int = gs2.EVAL_SEED      # 12345
SEED = 42

DEVICES: tuple[str, ...] = ("cpu", "mps", "cuda")
CKPT_ROOT = REPO_ROOT / "outputs" / "checkpoints"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
RATES_DIR = REPO_ROOT / "outputs" / "generated"
STEM_HEAD = "stage2_common"


# ============================================================
# 0. 設定
# ============================================================
@dataclasses.dataclass(frozen=True)
class Stage2Settings:
    """全モデル共通の Stage 2 の設定。既定値が 2026-10-29 に決めた本番の値

    Attributes:
        steps: 更新回数（早期終了なし。判定は最終 step）
        d_sub: 1 更新で使う群の数
        n: 群あたりの生成本数
        lam: リハーサルの重み λ。★暫定（リハーサル損失の単位と一緒に決める）
        tau: Gumbel-softmax の温度
        lr: 学習率（全パラメータで 1 つ）
        weight_decay: AdamW の weight decay
        grad_clip: 勾配のノルムの上限（L_agg と λ·L_reh を足した後に 1 回だけ掛ける）
        val_every: ATUS の val の損失を測る間隔（0 は測らない）。★結果に効く（RehearsalStream の Note）
        save_every: ckpt を保存する間隔（結果には効かない）
        log_every: 標準出力に 1 行書く間隔（結果には効かない）
    """
    steps: int = 1000
    d_sub: int = 7
    n: int = 256
    lam: float = 0.01
    tau: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_every: int = 10
    save_every: int = 25
    log_every: int = 10

    def variant_tag(self, device: str) -> str:
        """保存名の変種の印（例: _lam0.01_cpu、_lam0.003_steps300_cpu）

        Note:
            1. λ と計算場所は常に書く。計算場所が違うと浮動小数の丸めが違い、同じ種でも学習の経過がずれる
            2. それ以外は、結果に効く設定のうち既定と違うものだけを書く（save_every・log_every は書かない）

        Args:
            device: 学習した計算場所 cpu / mps / cuda

        Returns:
            印の文字列

        Raises:
            ValueError: device が DEVICES に無いとき
        """
        if device not in DEVICES:
            raise ValueError(f"device は {DEVICES} のどれか: {device}")
        default = Stage2Settings()
        tag = f"_lam{self.lam:g}"
        for field, label in (("steps", "steps"), ("d_sub", "dsub"), ("n", "n"), ("tau", "tau"), ("lr", "lr"),
                             ("weight_decay", "wd"), ("grad_clip", "clip"), ("val_every", "val")):
            value = getattr(self, field)
            if value != getattr(default, field):
                tag += f"_{label}{value:g}"
        return f"{tag}_{device}"


# ============================================================
# 1. モデルごとに実装する操作
# ============================================================
class Stage2Adapter(ABC):
    """共通の Stage 2 がモデルに求める操作。モデルごとに 1 つ実装する

    Attributes:
        name: モデルの名前（保存名の頭。MODEL_ADAPTERS のキー）
        model: 学習する nn.Module（Pre-trained を載せたもの）
    """
    name: str
    model: nn.Module

    @abstractmethod
    def stage1_tag(self) -> str:
        """Pre-trained の構成の印（保存名に入れる。例: _time_learned_h64_wd0.01）"""

    @abstractmethod
    def sample_relaxed(self, cond_idx: torch.Tensor, generator: torch.Generator, tau: float) -> torch.Tensor:
        """勾配の通る 1 日を生成する

        Note:
            1. ★前向きの値は、モデル自身の生成と同じ分布からの標本の厳密な one-hot（硬い one-hot）
            2. 後ろ向きは連続な近似（カテゴリの logits を出すモデルは straight_through_gumbel）
            3. 標本の乱数はすべて generator から引く（再開のときに同じ標本を引き直すため）

        Args:
            cond_idx: 条件インデックス, dtype=int64, (B, 3)。モデルと同じデバイス
            generator: 標本の乱数の CPU の torch.Generator
            tau: Gumbel-softmax の温度

        Returns:
            y: 勾配の通る one-hot, (B, NUM_ACT, NUM_SLOTS)
        """

    @abstractmethod
    def rehearsal_loss(self, cond_idx: torch.Tensor, sched: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Stage 1 と同じ損失（重みなし）を ATUS のミニバッチで計算する

        Args:
            cond_idx: 条件インデックス, dtype=int64, (B, 3)
            sched: 実データの活動, dtype=int64, (B, 96)
            weight: TUFINLWGT, (B,)。重みなしの損失では使わない

        Returns:
            backward できるスカラー
        """

    @abstractmethod
    def group_pool(self, n_per_group: int, seed: int) -> IntArr:
        """評価のプールをモデル自身の生成で作る

        Args:
            n_per_group: 群あたりの本数
            seed: 生成の乱数の種

        Returns:
            群別サンプルプール, dtype=int64, (D_GROUPS, n_per_group, NUM_SLOTS)。行 d は sm.cond_grid()[d] の条件
        """

    def config(self) -> dict[str, Any]:
        """ckpt の config に書く、モデル固有の情報（Pre-trained の ckpt の名前など）"""
        return {}

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Stage 2 で更新するパラメータ。既定は Stage 1 で学習したもの全部（= model の全パラメータ）"""
        return list(self.model.parameters())

    def auxiliary_step(self, next_batch: Callable[[], Batch]) -> dict[str, float]:
        """生成器の更新の前に、モデル固有の更新をする（WGAN-GP の critic など）。既定は何もしない

        Args:
            next_batch: ATUS の学習分割の次のミニバッチを返す関数（リハーサルと同じ並び）

        Returns:
            記録に足す値
        """
        del next_batch
        return {}

    def extra_state_dict(self) -> dict[str, Any]:
        """再開に要る、model と optimizer 以外の状態（critic・その optimizer・乱数など）。既定は空"""
        return {}

    def load_extra_state_dict(self, state: dict[str, Any]) -> None:
        """extra_state_dict の逆。既定は何もしない"""
        del state

    def val_loss(self, loader: DataLoader) -> float:
        """ATUS の val 分割での rehearsal_loss の平均（人数の加重平均）

        Note:
            LSTM の lm.run_epoch（評価のとき）と同じ足し方: バッチの損失 × 人数を Python の float で足し、人数の合計で割る

        Args:
            loader: ATUS の val 分割の DataLoader（shuffle なし）

        Returns:
            1 人あたりの損失
        """
        self.model.eval()
        dev = next(self.model.parameters()).device
        total, count = 0.0, 0.0
        with torch.no_grad():
            for cond_idx, sched, weight in loader:
                loss = self.rehearsal_loss(cond_idx.to(dev), sched.to(dev), weight.to(dev))
                w = float(sched.size(0))
                total += loss.item() * w
                count += w
        return total / count


def load_adapter(name: str, seed: int, device: str) -> Stage2Adapter:
    """MODEL_ADAPTERS のファイルを読み、Pre-trained を載せた adapter を作る

    Args:
        name: モデルの名前（MODEL_ADAPTERS のキー）
        seed: Pre-trained の種
        device: 載せる計算場所

    Returns:
        adapter

    Raises:
        ValueError: name が MODEL_ADAPTERS に無いとき
    """
    if name not in MODEL_ADAPTERS:
        raise ValueError(f"モデルは {tuple(MODEL_ADAPTERS)} のどれか: {name}")
    path = MODEL_ADAPTERS[name]
    mod = _load(f"stage2_adapter_{path.parent.name.lower()}", path)
    return mod.build(name, seed, device)


# ============================================================
# 2. 微分できる生成（カテゴリの logits を出すモデルに共通）
# ============================================================
def straight_through_gumbel(logits: torch.Tensor, generator: torch.Generator,
                            tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(logits) から 1 つ引き、勾配の通る one-hot を返す（straight-through Gumbel-softmax）

    Note:
        1. 前向きは argmax(logits + g) の厳密な one-hot。g は標準 Gumbel 雑音なので、softmax(logits) からの厳密な標本
        2. 後ろ向きは softmax((logits + g) / τ) の微分。y = hard + (soft − soft.detach()) で、後ろの項は前向きで 0
        3. 雑音は CPU の generator で float64 で引く（計算場所によらず同じ標本）。LSTM_Aggregate/stage2_agg.py の
           同名の関数と同じ計算（test_stage2_common.py で一致を確かめる）

    Args:
        logits: (..., K)
        generator: CPU の torch.Generator
        tau: 温度

    Returns:
        (y: 勾配の通る one-hot (..., K), a: 引いたカテゴリ, dtype=int64, (...))。どちらも logits と同じデバイス
    """
    u = torch.rand(logits.shape, generator=generator, dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
    gumbel = (-torch.log(-torch.log(u))).to(logits)
    z = logits + gumbel
    a = z.argmax(dim=-1)
    soft = F.softmax(z / tau, dim=-1)
    hard = F.one_hot(a, logits.shape[-1]).to(soft)
    return hard + (soft - soft.detach()), a


# ============================================================
# 3. リハーサルのミニバッチ（再開できる形）
# ============================================================
class RehearsalStream:
    """ATUS の学習分割のミニバッチを、epoch をまたいで返し続ける

    Note:
        1. 並びは `while True: yield from loader` と同じ（LSTM の Stage 2 の atus_batches と同じ乱数の消費）
        2. ★DataLoader（shuffle あり、generator なし）は大域の乱数を、iterator を作るときに 1 つ（_base_seed）、
           epoch の最初のバッチを取るときに 1 つ（RandomSampler の種）消費する。val の DataLoader も iterator を作るたびに
           1 つ消費するので、val を測る間隔を変えるとミニバッチの並びが変わる
        3. 再開のため、epoch の iterator を作る直前の大域の乱数の状態 epoch_rng と、その epoch で取ったバッチ数 consumed
           を持つ。load_state_dict は epoch_rng から iterator を作り直して consumed 個読み飛ばす
    """

    def __init__(self, loader: DataLoader) -> None:
        """
        Args:
            loader: ATUS の学習分割の DataLoader（shuffle あり）
        """
        self.loader = loader
        self._it: Iterator[Batch] | None = None
        self._epoch_rng: torch.Tensor | None = None
        self._consumed = 0

    def next_batch(self) -> Batch:
        """次のミニバッチ。epoch が尽きたら次の epoch の iterator を作る"""
        while True:
            if self._it is None:
                self._epoch_rng = torch.get_rng_state()
                self._it = iter(self.loader)
                self._consumed = 0
            try:
                batch = next(self._it)
            except StopIteration:
                self._it = None
                continue
            self._consumed += 1
            return batch

    def state_dict(self) -> dict[str, Any]:
        """再開に要る状態（epoch_rng・consumed）"""
        return {"epoch_rng": self._epoch_rng, "consumed": self._consumed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """state_dict の時点の並びへ戻す

        Note:
            ★iterator を作り直す分だけ大域の乱数が進む。呼んだ後で、大域の乱数を保存時の状態へ戻すこと（load_checkpoint）

        Args:
            state: state_dict の戻り値
        """
        self._it, self._epoch_rng, self._consumed = None, None, 0
        if state["epoch_rng"] is None:                       # まだ 1 つも取っていない
            return
        torch.set_rng_state(state["epoch_rng"])
        self._epoch_rng = state["epoch_rng"]
        self._it = iter(self.loader)
        for _ in range(int(state["consumed"])):
            next(self._it)
            self._consumed += 1


# ============================================================
# 4. ckpt
# ============================================================
def save_checkpoint(path: Path, adapter: Stage2Adapter, optimizer: torch.optim.Optimizer, step: int,
                    config: dict[str, Any], rng: np.random.Generator, gumbel: torch.Generator,
                    stream: RehearsalStream, history: list[dict[str, float]]) -> None:
    """再開に要るものをすべて 1 つのファイルに atomic に書き、history.csv も書き直す

    Args:
        path: 保存先（ck.ckpt_path で作る）
        adapter: 学習中の adapter
        optimizer: AdamW（1 次・2 次のモーメントを含む）
        step: 完了した更新回数（再開はこの次から）
        config: 設定（Stage2Settings と adapter.config など）
        rng: 群の抽出の numpy の Generator
        gumbel: 標本の乱数の CPU の torch.Generator
        stream: リハーサルのミニバッチ
        history: ここまでの 1 更新ごとの記録
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": adapter.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": rng.bit_generator.state,
        "gumbel_rng": gumbel.get_state(),
        "stream": stream.state_dict(),
        "history": history,
        "extra": adapter.extra_state_dict(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)                                         # 途中で落ちても既存のファイルは壊れない
    pd.DataFrame(history).to_csv(path.parent / "history.csv", index=False)


def load_checkpoint(path: Path, adapter: Stage2Adapter, optimizer: torch.optim.Optimizer,
                    rng: np.random.Generator, gumbel: torch.Generator, stream: RehearsalStream,
                    config: dict[str, Any]) -> tuple[int, list[dict[str, float]]]:
    """save_checkpoint の逆。モデル・optimizer・4 つの乱数・ミニバッチの位置・記録を戻す

    Note:
        1. ★大域の乱数は最後に戻す（stream.load_state_dict が iterator を作り直す分だけ進めるため）
        2. ★いったん CPU に読む。model.load_state_dict と optimizer.load_state_dict が、重みと 1 次・2 次のモーメントを
           パラメータの計算場所へ移す。AdamW の step のカウンタは CPU のまま残り、最初から回したときと同じ置き場所になる
           （計算場所へ直接読むと、step のカウンタまで GPU に載る）

    Args:
        path: 読む ckpt
        adapter: 書き込む先（Pre-trained を載せたもの。形が同じなら中身は問わない）
        optimizer: 書き込む先
        rng: 群の抽出の numpy の Generator（in-place で戻す）
        gumbel: 標本の乱数の CPU の torch.Generator（in-place で戻す）
        stream: リハーサルのミニバッチ（in-place で戻す）
        config: 今回の設定。保存時の設定と違えば止める

    Returns:
        (完了した更新回数, ここまでの記録)

    Raises:
        ValueError: 保存時の設定と今回の設定が違うとき
    """
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != config:
        diff = {k: (saved["config"].get(k), config.get(k)) for k in set(saved["config"]) | set(config)
                if saved["config"].get(k) != config.get(k)}
        raise ValueError(f"{path} の設定が今回の設定と違う（保存時, 今回）: {diff}")
    adapter.model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    adapter.load_extra_state_dict(saved["extra"])
    rng.bit_generator.state = saved["numpy_rng"]
    gumbel.set_state(saved["gumbel_rng"])
    stream.load_state_dict(saved["stream"])
    torch.set_rng_state(saved["torch_rng"])
    cuda_rng = saved["cuda_rng"]
    if cuda_rng is not None and torch.cuda.is_available() and len(cuda_rng) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_rng)
    return int(saved["step"]), list(saved["history"])


def load_weights(path: Path, adapter: Stage2Adapter) -> None:
    """Stage 2 の ckpt の重みだけを adapter.model に書く（採点用。CPU に読んでから、パラメータの計算場所へ写す）"""
    adapter.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])


# ============================================================
# 5. 学習
# ============================================================
def grad_norm(params: list[nn.Parameter]) -> float:
    """いま p.grad に載っている勾配全体の L2 ノルム（CPU の float64 で積む）"""
    sq = sum(float(p.grad.detach().cpu().double().pow(2).sum()) for p in params if p.grad is not None)
    return math.sqrt(sq)


def train_stage2(adapter: Stage2Adapter, tgt: dict, teacher_mask: BoolArr, seed: int,
                 settings: Stage2Settings = Stage2Settings(), out_dir: Path | None = None,
                 resume: bool = False) -> list[dict[str, float]]:
    """固定の step 数だけ微調整する（adapter.model を in-place で更新する）

    Note:
        1. 乱数は seed から作る（モジュールの docstring の 4 つ）。resume で ckpt から戻したときは、その状態から続ける
        2. 教師から外した群（teacher_mask = False）は L_agg に入らない。リハーサルは全群を使う
        3. 勾配のクリップは L_agg と λ·L_reh を足した後に 1 回だけ掛ける。記録の clip_norm はクリップ前のノルム
        4. 1 更新の中の順序（乱数の消費の順序）は LSTM_Aggregate/stage2_agg.py の train_stage2 と同じ

    Args:
        adapter: Pre-trained を載せた adapter（学習するデバイスに載せたもの）
        tgt: 教師（st.load_stula_targets）
        teacher_mask: L_agg に使う群, (28,)
        seed: 乱数の種
        settings: Stage 2 の設定, default=Stage2Settings()（本番の値）
        out_dir: ckpt と history.csv の保存先。None なら何も保存しない
        resume: True なら out_dir の最新の ckpt から続ける（無ければ最初から）

    Returns:
        1 更新ごとの記録 list[dict]（再開したときは最初の step からの分を含む）

    Raises:
        ValueError: 教師群が無いとき、out_dir なしで resume を指定したとき
        FileExistsError: resume なしで out_dir に ckpt が既にあるとき
    """
    teacher_groups = np.flatnonzero(teacher_mask)
    if len(teacher_groups) == 0:
        raise ValueError("教師群が空")
    if resume and out_dir is None:
        raise ValueError("resume には out_dir が要る")
    latest = ck.latest_ckpt(out_dir) if out_dir is not None else None
    if latest is not None and not resume:
        raise FileExistsError(f"{latest} が既にある。続けるなら resume、やり直すなら {out_dir} を消してから")

    dev = next(adapter.model.parameters()).device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gumbel = torch.Generator().manual_seed(seed)

    a_star_all = sl.teacher_tensor(tgt, dev)                                   # (28, 12, 96)
    omega_all = torch.ones_like(a_star_all)                                    # ω = 1（素の MSE）
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long, device=dev)     # (28, 3)
    params = adapter.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=settings.lr, weight_decay=settings.weight_decay)

    train_part, val_part = gm.load_split()
    stream = RehearsalStream(gm.make_loader(train_part, shuffle=True))
    val_loader = gm.make_loader(val_part, shuffle=False)

    config: dict[str, Any] = {"model": adapter.name, "seed": seed, **dataclasses.asdict(settings),
                              **adapter.config(), "device": str(dev),
                              "holdout": np.flatnonzero(~teacher_mask).tolist()}
    start, history = 0, []
    if latest is not None:
        start, history = load_checkpoint(latest, adapter, optimizer, rng, gumbel, stream, config)
        print(f"[stage2] {latest} から再開（step {start} まで完了）", flush=True)
    print(f"[stage2] model={adapter.name} steps={settings.steps} d_sub={settings.d_sub} n={settings.n} "
          f"lam={settings.lam:g} tau={settings.tau:g} lr={settings.lr:g} wd={settings.weight_decay:g} "
          f"teacher_groups={len(teacher_groups)}/28 device={dev} threads={torch.get_num_threads()}", flush=True)

    for step in range(start + 1, settings.steps + 1):
        t0 = time.perf_counter()
        d_pick = np.sort(rng.choice(teacher_groups, size=min(settings.d_sub, len(teacher_groups)), replace=False))
        d_pick_t = torch.as_tensor(d_pick, device=dev)
        cond = grid[d_pick_t].repeat_interleave(settings.n, dim=0)              # (d_sub·n, 3) 群優先
        a_star, omega = a_star_all[d_pick_t], omega_all[d_pick_t]
        adapter.model.train()
        aux = adapter.auxiliary_step(stream.next_batch)

        # ---- 集計側: 生成を通して逆伝播する ----
        optimizer.zero_grad(set_to_none=True)
        y = adapter.sample_relaxed(cond, gumbel, settings.tau)                  # (d_sub·n, 12, 96)
        a_hat = sl.group_rates(y, settings.n)                                   # (d_sub, 12, 96)
        l_agg = sl.agg_mse_from_rates(a_hat, a_star, omega)
        l_agg.backward()
        agg_gnorm = grad_norm(params)

        # ---- リハーサル側: Stage 1 と同じ損失 ----
        b_cond, b_sched, b_weight = stream.next_batch()
        l_reh = adapter.rehearsal_loss(b_cond.to(dev), b_sched.to(dev), b_weight.to(dev))
        tug = ft.add_rehearsal_grad(params, settings.lam * l_reh)
        total_gnorm = grad_norm(params)
        clip_norm = float(nn.utils.clip_grad_norm_(params, settings.grad_clip))
        optimizer.step()

        with torch.no_grad():
            log: dict[str, float] = {"step": step, "L_agg": float(l_agg.detach()), "L_reh": float(l_reh.detach()),
                                     "lam": settings.lam, "rate_mae": float((a_hat - a_star).abs().mean()),
                                     "rate_mse": float(((a_hat - a_star) ** 2).mean()), "clip_norm": clip_norm,
                                     "agg_gnorm": agg_gnorm, "total_gnorm": total_gnorm, **tug, **aux,
                                     "sec": time.perf_counter() - t0}
        if settings.val_every > 0 and (step % settings.val_every == 0 or step == 1):
            log["L_reh_val"] = adapter.val_loss(val_loader)
        history.append(log)
        if step % settings.log_every == 0 or step == 1:
            val_txt = f"  val={log['L_reh_val']:.5f}" if "L_reh_val" in log else ""
            print(f"  step {step:4d}/{settings.steps}  L_agg={log['L_agg']:+.6f}  rate_mse={log['rate_mse']:.6f}  "
                  f"L_reh={log['L_reh']:.5f}{val_txt}  |g_agg|={agg_gnorm:.1e}  tug_cos={log['tug_cos']:+.3f}  "
                  f"tug_ratio={log['tug_ratio']:.2f}  clip={clip_norm:.2f}  {log['sec']:.1f}s", flush=True)
        if out_dir is not None and (step % settings.save_every == 0 or step == settings.steps):
            save_checkpoint(ck.ckpt_path(out_dir, step), adapter, optimizer, step, config, rng, gumbel, stream,
                            history)
    adapter.model.eval()
    return history


# ============================================================
# 6. 採点
# ============================================================
def slot_jsd(rates: FloatArr, a_star: FloatArr, groups: BoolArr) -> float:
    """時刻別行動者率の JSD（群 d・スロット s ごとに 12 活動の分布の JSD を取り、群とスロットで平均する）

    Note:
        LSTM_Aggregate/stage2_agg.py の slot_jsd と同じ定義（自然対数、0 log 0 = 0）

    Args:
        rates: 生成の時刻別行動者率, (28, 12, 96)
        a_star: 教師 A*, (28, 12, 96)
        groups: 平均に使う群, (28,)

    Returns:
        平均の JSD
    """
    g, a = rates[groups], a_star[groups]
    m = 0.5 * (g + a)

    def kl(p: FloatArr) -> FloatArr:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, p * np.log(p / m), 0.0).sum(axis=1)                 # (群, 96)

    return float((0.5 * kl(g) + 0.5 * kl(a)).mean())


def evaluate(adapter: Stage2Adapter, tgt: dict, masks: dict[str, BoolArr], base: dict[str, Any],
             n_per_group: int = EVAL_N, full: bool = True) -> tuple[list[dict[str, Any]], FloatArr]:
    """評価のプールをモデル自身の生成で作り、GRU・LSTM の Stage 2 と同じ軸で採点する

    Args:
        adapter: 採点するモデルの adapter
        tgt: 教師
        masks: 名前 → 教師の群 (28,)
        base: 全行に付ける識別列
        n_per_group: 評価のプールの群あたりの本数, default=EVAL_N=2000
        full: False なら軸 1 だけ（--smoke 用）

    Returns:
        (採点の行 list[dict], 評価のプールの時刻別行動者率 (28, 12, 96))
    """
    adapter.model.eval()
    pool = adapter.group_pool(n_per_group, EVAL_SEED)
    rows = gs2.score_pool(pool, tgt, masks, base, full)
    rates = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
    a_star = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    for name, mask in masks.items():
        for kind, groups in (("in-teacher", mask), ("held-out", ~mask)):
            if groups.any():
                rows.append({**base, "mask_name": name, "eval_kind": kind, "reference": "teacher",
                             "statistic": "slot_rate", "mask": "12act", "weight_basis": "none",
                             "metric": "rate_jsd", "value": slot_jsd(rates, a_star, groups)})
    return rows, rates


# ============================================================
# 7. 保存先と実行
# ============================================================
def stem(adapter: Stage2Adapter, variant: str, seed: int, run_: str) -> str:
    """保存先の共通の名前（例: stage2_common_lstm_time_learned_h64_wd0.01_lam0.01_cpu_seed42_all）"""
    return f"{STEM_HEAD}_{adapter.name}{adapter.stage1_tag()}{variant}_seed{seed}_{run_}"


def run_dir(adapter: Stage2Adapter, variant: str, seed: int, run_: str) -> Path:
    """ckpt と history.csv を置くディレクトリ"""
    return CKPT_ROOT / stem(adapter, variant, seed, run_)


def csv_path(adapter: Stage2Adapter, variant: str, seed: int, run_: str) -> Path:
    """採点の CSV のパス"""
    return OUT_DIR / f"{stem(adapter, variant, seed, run_)}.csv"


def rates_path(adapter: Stage2Adapter, variant: str, seed: int, run_: str) -> Path:
    """評価のプールの時刻別行動者率の npz のパス"""
    return RATES_DIR / f"{stem(adapter, variant, seed, run_)}_rates.npz"


def run(name: str, seed: int, zero_shot: bool, device: str, settings: Stage2Settings = Stage2Settings(),
        resume: bool = False) -> Path:
    """E0（Pre-trained の採点）か E1（28 群すべてを教師にして微調整し、最終 step を採点）を 1 本回す

    Args:
        name: モデルの名前（MODEL_ADAPTERS のキー）
        seed: 種
        zero_shot: True なら学習せず Pre-trained を採点する（E0）
        device: 学習と採点の計算場所
        settings: Stage 2 の設定, default=Stage2Settings()
        resume: True なら保存済みの最新の ckpt から続ける

    Returns:
        書いた採点の CSV のパス
    """
    tgt = st.load_stula_targets()
    adapter = load_adapter(name, seed, device)
    every = np.ones(sm.D_GROUPS, dtype=bool)
    if zero_shot:
        run_, variant, step = "zeroshot", "", 0
    else:
        run_, variant, step = "all", settings.variant_tag(device), settings.steps
        train_stage2(adapter, tgt, every, seed, settings, run_dir(adapter, variant, seed, run_), resume)
    base: dict[str, Any] = {"model": f"{adapter.name}{adapter.stage1_tag()}", "seed": seed, "run": run_,
                            "lam": float("nan") if zero_shot else settings.lam, "tau": settings.tau, "step": step,
                            "n_per_group": EVAL_N, "pool_seed": EVAL_SEED, "device": device}
    rows, rates = evaluate(adapter, tgt, {"all": every}, base)
    out = csv_path(adapter, variant, seed, run_)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    rp = rates_path(adapter, variant, seed, run_)
    rp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rp, rates=rates)
    print(f"[stage2] 書いた: {out}\n[stage2] 書いた: {rp}", flush=True)
    return out


def smoke(name: str, device: str) -> None:
    """2 更新・2 群・群あたり 4 本で、学習と軸 1 の採点が通ることだけを確かめる（何も保存しない）"""
    tgt = st.load_stula_targets()
    adapter = load_adapter(name, SEED, device)
    before = {k: v.detach().clone() for k, v in adapter.model.state_dict().items()}
    history = train_stage2(adapter, tgt, np.ones(sm.D_GROUPS, dtype=bool), SEED,
                           Stage2Settings(steps=2, d_sub=2, n=4, val_every=1))
    moved = [k for k, v in adapter.model.state_dict().items() if not torch.equal(v, before[k])]
    rows, rates = evaluate(adapter, tgt, {"all": np.ones(sm.D_GROUPS, dtype=bool)}, {"run": "smoke"},
                           n_per_group=4, full=False)
    assert len(history) == 2 and all(math.isfinite(h["L_agg"]) for h in history)
    assert rates.shape == (sm.D_GROUPS, sm.NUM_ACT, sm.NUM_SLOTS)
    print(f"smoke: OK（{name}、{device}、{len(rows)} 行、更新されたパラメータ {len(moved)}/{len(before)}）")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="モデルによらない共通の Stage 2")
    ap.add_argument("--model", choices=tuple(MODEL_ADAPTERS), default="lstm")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--zero-shot", action="store_true", help="学習せず Pre-trained を採点する（E0）")
    ap.add_argument("--lam", type=float, default=Stage2Settings.lam, help="リハーサルの重み λ（暫定の既定 0.01）")
    ap.add_argument("--steps", type=int, default=Stage2Settings.steps)
    ap.add_argument("--resume", action="store_true", help="保存済みの最新の ckpt から続ける")
    ap.add_argument("--device", choices=DEVICES, default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="CPU の計算のスレッド数。0 なら torch の既定")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        ap.error("--device mps: この計算機では MPS が使えない")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda: この計算機では CUDA が使えない")
    if args.threads < 0:
        ap.error(f"--threads は 0 以上: {args.threads}")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.smoke:
        smoke(args.model, args.device)
        return
    run(args.model, args.seed, args.zero_shot, args.device, Stage2Settings(lam=args.lam, steps=args.steps),
        args.resume)


if __name__ == "__main__":
    main()
