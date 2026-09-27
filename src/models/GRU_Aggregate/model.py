"""
model.py
========
再帰型（GRU）＋交差エントロピーの Stage 1（GRU_Aggregate）

1 日の 96 スロットを 04:00 から順に 1 つずつ生成する条件付きの自己回帰モデル。
各スロットの活動の確率を、TUFINLWGT で重み付けした交差エントロピーで学習する（teacher forcing）。
計画書: src/models/GRU_Aggregate/docs/plan.md

★データ・分割・条件の符号化・時刻符号 φ・生成プールの CSV 形式は DDPM_Aggregate_Simple/model.py から
  import する（写し書きしない）。分割がずれると、DDPM との比較や暗記チェックの参照集合が別物になる。

総量を縛る仕組み（計画書 §3.2）:
    L = − Σ_i w_i Σ_s log p_θ(a_{i,s} | a_{i,<s}, c_i, s) / (96 · Σ_i w_i)
    ∂L/∂slot_bias[s, c] = Σ_i w_i (p_θ[i, s, c] − 1[a_{i,s} = c]) / (96 · Σ_i w_i)
    → 勾配が 0 の点では Σ_i w_i p_θ[i, s, c] = Σ_i w_i 1[a_{i,s} = c]（全スロット s・全活動 c）
    ★厳密には、学習で条件を落とした行（確率 P_UNCOND）の予測を含めた混合で成り立つ。
      学習後の実測は teacher_forced_gap で出す（ckpt の config では calib_gap_max / calib_gap_mean）

学習後の slot_bias の補正（計画書 §9、--calibrate）:
    交差エントロピーが縛るのは「実データの履歴で予測した総量」だけで、自分で生成した総量は種ごとに揺れる。
    そこで学習後に、生成した総量 gen が学習分割の行動者率 target に合うよう slot_bias だけを反復で動かす:
        slot_bias += CALIB_STEP · log((target + CALIB_EPS) / (gen + CALIB_EPS)).T

構造:

```mermaid
flowchart LR
    PREV["a_prev (B, S)<br/>直前の活動（s=0 は BOS=12）"] --> EA["act_embed<br/>Embedding(13, HIDDEN)"]
    PHI["phi (96, 96)<br/>sm.time_features(TIME_ARCH).T"] --> ET["time_proj<br/>Linear(96, HIDDEN)"]
    COND["cond_idx (B, 3)<br/>drop_mask (B,)"] --> EC["embed_cond<br/>cond_embeds → cond_proj<br/>落とした行は cond_null"]
    EA --> X["x (B, S, HIDDEN) = 3 つの和"]
    ET --> X
    EC --> X
    X --> GRU["gru<br/>GRU(HIDDEN, HIDDEN, NUM_LAYERS)"]
    GRU --> OUT["out_proj(h) + slot_bias<br/>logits (B, S, 12)"]
```

学習と生成:

```mermaid
flowchart TD
    LD["load_split<br/>sm.load_data / sm.split_indices"] --> TR["train<br/>batch_loss = weighted_ce（teacher forcing）"]
    TR --> CK["ckpt_path(seed)<br/>+ sm.epoch_ckpt_path(ckpt, epoch)"]
    CK --> GP["group_pool(model, POOL_N, guidance_scale)<br/>pool (28, 256, 96)"]
    GP --> CSV["pool_path(seed, guidance_scale)<br/>sm.write_pool_csv"]
    CK --> TF["teacher_forced_rates<br/>予測確率の加重平均 (12, 96)"]
    CK --> CAL["calibrate_slot_bias（--calibrate）<br/>group_pool → generated_rates → bias_update"]
    CAL --> CCK["ckpt_path(seed, calib_guidance=g)<br/>pool_path(seed, g, calib_guidance=g)"]
```

使い方:
    # 動作確認（2 epoch 学習・群あたり 2 本だけ生成。何も保存しない）
    .venv/bin/python src/models/GRU_Aggregate/model.py --smoke

    # 学習して生成プールを書く
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42

    # 保存済みの ckpt から CFG の強さを変えたプールだけを作る
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42 --pool-only --guidance 1.25

    # 保存済みの ckpt の slot_bias を CFG の強さ g で補正し、補正した ckpt と生成プール（同じ g）を書く
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42 --calibrate                   # g = 1.0 → _cal
    .venv/bin/python src/models/GRU_Aggregate/model.py --seed 42 --calibrate --guidance 1.25   # → _calg1.25

    フラグ:
        --seed S       : 学習の乱数の種（既定 42）。分割は変えない。42 以外は保存先に _s{S}
        --epochs N     : 学習 epoch の上限（既定 EPOCHS=300）
        --save-every N : N epoch ごとに途中の ckpt を {ckpt の stem}_ep{epoch:04d}.pt へ保存（既定 5、0 で保存しない）
        --guidance G   : 生成プールの CFG の強さ（既定 1.0 = CFG なし）。既定以外は保存先に _g{G}
        --no-pool      : 学習後の生成プールを作らない
        --pool-only    : 学習せず、保存済みの ckpt から生成プールだけを作る
        --calibrate    : 学習せず、保存済みの ckpt の slot_bias を --guidance の g で補正して、補正した ckpt と
                         生成プール（同じ g）を書く。接尾辞は g = 1.0 なら _cal、それ以外は _calg{G}
        --smoke        : 短時間の動作確認

出力:
    outputs/checkpoints/gru_aggregate{接尾辞}.pt               最良の ckpt（config に学習曲線 history）
    outputs/checkpoints/gru_aggregate{接尾辞}_ep{epoch:04d}.pt 途中の ckpt
    outputs/generated/gru_aggregate_samples{接尾辞}.csv        生成プール（28 群 × 256 本）
    outputs/checkpoints/gru_aggregate{接尾辞}{補正}.pt              slot_bias を補正した ckpt（config に補正の記録）
    outputs/generated/gru_aggregate_samples{接尾辞}{補正}{_gG}.csv  補正した ckpt の生成プール
        {補正} = _cal（g = 1.0 で補正）/ _calg1.25（g = 1.25 で補正）。{_gG} は生成の g（1.0 なら付けない）
"""
import argparse
import copy
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/GRU_Aggregate -> repo root


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名の model.py を取り違えないため）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# データ・分割・条件・時刻符号・CSV 形式の唯一の出所
sm: Any = _load("gru_ddpm_simple_model", REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "model.py")

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]

# ============================================================
# 1. 設定
# ============================================================
MODEL_SAVE_PATH = REPO_ROOT / "outputs" / "checkpoints" / "gru_aggregate.pt"
GEN_SAVE_PATH = REPO_ROOT / "outputs" / "generated" / "gru_aggregate_samples.csv"

NUM_SLOTS: int = sm.NUM_SLOTS        # 96（04:00 起点の 15 分）
NUM_ACT: int = sm.NUM_ACT            # 12（共通 12 分類）
D_GROUPS: int = sm.D_GROUPS          # 28
BOS = NUM_ACT                        # s = 0 の直前の活動の代わりに入れる記号

# 構造。H=384 でパラメータ 182.9 万（比べる DDPM clock_tf96 は 187.7 万）
HIDDEN = 384
NUM_LAYERS = 2
# 生成するスロットの時刻符号。DDPM の clock_tf96 と同じ Transformer 型 96 次元の φ
TIME_ARCH: Any = sm.ArchSpec(clock_kind="transformer")
# slot_bias の初期値 log(行動者率) の下限（行動者率 0 のセルで −inf にしないため）
RATE_FLOOR = 1e-6

# CFG（学習で条件を落とす確率は DDPM と同じ）
P_UNCOND: float = sm.P_UNCOND        # 0.1
# ★既定は CFG なし。診断で CFG が少ない活動の総量をずらすと分かったため（計画書 §3.3）
GUIDANCE_SCALE = 1.0

# 学習
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
BATCH_SIZE = 256
EPOCHS = 300
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 1e-4          # DDPM と同じ
SAVE_EVERY = 5
SEED: int = sm.SEED                  # 42。学習の乱数の既定値（分割は sm.split_indices が固定）

# 生成
POOL_N = 256                         # 群あたりの本数（DDPM の学習時プールと同じ）
POOL_SEED = 12345                    # DDPM の共通乱数プールと同じ値
GEN_BATCH = 1024

# 学習後の slot_bias の補正（計画書 §9。結果を見る前に固定）
CALIB_ITERS = 8                      # 反復の回数（最後の補正の後にもう 1 回測る）
CALIB_POOL_N = 1024                  # 1 回の反復のプールの群あたりの本数
CALIB_SEED = 20000                   # 反復 k のプールの種は CALIB_SEED + k（評価の POOL_SEED とは別）
# ★更新の幅。1.0 は履歴に依らない分布なら 1 回で目標に一致する幅だが、GRU では振動した（計画書 §9.3）。
#   1 回が長い活動では slot_bias の変化が「始める」と「続ける」の両方に効き、総量が約 2 倍動くため
CALIB_STEP = 0.5
CALIB_EPS = 1e-4                     # 行動者率 0 のセルで log を発散させない値
CALIB_TAG = "_cal"                   # 補正した ckpt・プールの接尾辞（g ≠ 1.0 で補正したものは _calg{G}）

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


# ============================================================
# 2. データ
# ============================================================
class SplitPart(NamedTuple):
    """学習分割または評価分割の個票"""
    cond_idx: IntArr     # 条件インデックス, (N, 3)。列は sm.COND_SPEC の順
    sched: IntArr        # 活動スケジュール, (N, 96)。値域 [0, 12)
    weight: FloatArr     # 調査ウェイト TUFINLWGT, (N,)


def load_split() -> tuple[SplitPart, SplitPart]:
    """ATUS 2024 平日を DDPM と同じ規則で学習 / 評価に分ける

    Returns:
        (train 3,363 人, val 373 人)
    """
    cond_idx, sched, weight, _ = sm.load_data()
    train_idx, val_idx = sm.split_indices(len(sched))
    return (SplitPart(cond_idx[train_idx], sched[train_idx], weight[train_idx]),
            SplitPart(cond_idx[val_idx], sched[val_idx], weight[val_idx]))


def make_loader(part: SplitPart, shuffle: bool) -> DataLoader:
    """(cond_idx, sched, weight) を返す DataLoader。全員を毎 epoch 使い、重みは損失に掛ける

    Note:
        ★DDPM は TUFINLWGT に比例した復元抽出でミニバッチを作る。ここでは全員を使い、
          損失を重みで掛ける。学習の目標は同じ分布で、少ない活動の行動者を毎 epoch 必ず見る
    """
    ds = TensorDataset(torch.as_tensor(part.cond_idx, dtype=torch.long),
                       torch.as_tensor(part.sched, dtype=torch.long),
                       torch.as_tensor(part.weight, dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def shift_right(sched: torch.Tensor) -> torch.Tensor:
    """各スロットの入力にする直前の活動 a_prev を作る, (B, S) -> (B, S)

    a_prev[:, 0] = BOS, a_prev[:, s] = sched[:, s−1]
    """
    bos = torch.full_like(sched[:, :1], BOS)
    return torch.cat([bos, sched[:, :-1]], dim=1)


# ============================================================
# 3. モデル
# ============================================================
class GRUScheduler(nn.Module):
    """条件付きの自己回帰モデル p_θ(a_s | a_<s, c, s)

    Note:
        1. 生成するスロットの時刻 φ[s] を、そのステップだけの入力として足す（time_proj）
        2. slot_bias（96 × 12）が総量を縛る仕組みの本体。φ の線形写像では各スロットの任意の値を
           作れない（Transformer 型 96 次元でも独立な成分は 34 個）ので、自由なバイアスを持つ
        3. out_proj は零初期化する。学習前の予測は softmax(slot_bias) = 行動者率になる（init_slot_bias）

    Attributes:
        hidden: 隠れ層の幅 H
        num_layers: GRU の層数
        phi: 時刻符号 φ, (96 スロット, 96 次元)。保存しない buffer（sm.time_features から毎回作る）
    """

    phi: torch.Tensor

    def __init__(self, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS) -> None:
        """部品を作る

        Args:
            hidden: 隠れ層の幅 H, default=HIDDEN=384
            num_layers: GRU の層数, default=NUM_LAYERS=2
        """
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.act_embed = nn.Embedding(NUM_ACT + 1, hidden)               # 12 活動 + BOS
        self.register_buffer("phi", sm.time_features(TIME_ARCH).T.contiguous(), persistent=False)
        self.time_proj = nn.Linear(TIME_ARCH.clock_dim, hidden)
        self.cond_embeds = nn.ModuleList([nn.Embedding(card, dim) for _, card, dim, _ in sm.COND_SPEC])
        self.cond_proj = nn.Linear(sum(sm.EMB_DIMS), hidden)
        self.cond_null = nn.Parameter(torch.zeros(hidden))
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        # 定数成分は slot_bias が持つので bias は付けない
        self.out_proj = nn.Linear(hidden, NUM_ACT, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.slot_bias = nn.Parameter(torch.zeros(NUM_SLOTS, NUM_ACT))

    @torch.no_grad()
    def init_slot_bias(self, rates: torch.Tensor) -> None:
        """slot_bias を log(行動者率) で初期化する（学習の初期を速めるため。§3.2 の一致には影響しない）

        Args:
            rates: スロットごとの行動者率, (NUM_ACT, NUM_SLOTS)。sm.population_rates の戻り値
        """
        self.slot_bias.copy_(rates.T.clamp_min(RATE_FLOOR).log().to(self.slot_bias))

    def embed_cond(self, cond_idx: torch.Tensor | None, batch: int,
                   drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        """条件（性・年齢・就業）を埋め込む, -> (B, HIDDEN)

        Args:
            cond_idx: 条件インデックス, dtype=int64, (B, 3)。None ならバッチ全体を cond_null にする
            batch: バッチの大きさ B（cond_idx=None では形から取れないため）
            drop_mask: True の行だけ cond_null に差し替える, dtype=bool, (B,)。学習で P_UNCOND の確率で立てる

        Returns:
            条件埋め込み, (B, HIDDEN)
        """
        if cond_idx is None:
            return self.cond_null.expand(batch, -1)
        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)   # (B, 16)
        c = self.cond_proj(c)
        if drop_mask is not None:
            c = torch.where(drop_mask[:, None], self.cond_null.expand_as(c), c)
        return c

    def forward(self, a_prev: torch.Tensor, cond_idx: torch.Tensor | None,
                drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        """直前の活動の列から、各スロットの活動の logits を一度に出す（学習・teacher forcing 用）

        Args:
            a_prev: 直前の活動, dtype=int64, (B, S)。shift_right(sched) の形。値域 [0, 13)
            cond_idx: 条件インデックス, (B, 3)。None なら条件なし
            drop_mask: 条件を落とす行, (B,)

        Returns:
            logits, (B, S, NUM_ACT)。スロット s の logits は a_prev[:, :s+1] だけに依存する
        """
        batch, length = a_prev.shape
        x = (self.act_embed(a_prev)
             + self.time_proj(self.phi[:length])[None]
             + self.embed_cond(cond_idx, batch, drop_mask)[:, None, :])     # (B, S, H)
        h, _ = self.gru(x)
        return self.out_proj(h) + self.slot_bias[:length]

    def step(self, a_prev: torch.Tensor, s: int, c_emb: torch.Tensor,
             state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """1 スロット分だけ進める（生成用）。forward の s 列目と同じ値を返す

        Args:
            a_prev: 直前の活動, dtype=int64, (B,)。s = 0 では BOS
            s: 生成するスロット, 0..95
            c_emb: embed_cond の出力, (B, HIDDEN)
            state: GRU の隠れ状態, (NUM_LAYERS, B, HIDDEN)。s = 0 では None

        Returns:
            (スロット s の logits (B, NUM_ACT), 次の隠れ状態)
        """
        x = self.act_embed(a_prev) + self.time_proj(self.phi[s]) + c_emb          # (B, H)
        h, next_state = self.gru(x[:, None, :], state)
        return self.out_proj(h[:, 0]) + self.slot_bias[s], next_state


# ============================================================
# 4. 損失と学習
# ============================================================
def weighted_ce(logits: torch.Tensor, sched: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """重み付き交差エントロピー − Σ_i w_i Σ_s log p[i, s, a_{i,s}] / (S · Σ_i w_i)

    Args:
        logits: (B, S, NUM_ACT)
        sched: 正解の活動, dtype=int64, (B, S)
        weight: 個票の重み, (B,)

    Returns:
        スカラーの損失（1 スロットあたりの nats）
    """
    nll = F.cross_entropy(logits.reshape(-1, NUM_ACT), sched.reshape(-1),
                          reduction="none").view_as(sched)                   # (B, S)
    return (weight[:, None] * nll).sum() / (sched.size(1) * weight.sum())


def batch_loss(model: GRUScheduler, cond_idx: torch.Tensor, sched: torch.Tensor,
               weight: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
    """teacher forcing の損失（各スロットの入力は実データの直前の活動）"""
    return weighted_ce(model(shift_right(sched), cond_idx, drop_mask), sched, weight)


def run_epoch(model: GRUScheduler, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None = None) -> float:
    """1 epoch の学習（optimizer あり）または評価（なし）を行い、重み付き交差エントロピーを返す

    Note:
        1. 学習では確率 P_UNCOND で条件を落とす。評価では落とさない（条件付きの損失を測る）
        2. 乱数（条件を落とす行）は CPU で引く。デバイスによらず同じ行を落とすため
        3. 戻り値はバッチの損失を Σw で重み付けした平均 = 分割全体の重み付き交差エントロピー

    Returns:
        重み付き交差エントロピー（1 スロットあたりの nats）
    """
    is_train = optimizer is not None
    model.train(is_train)
    dev = next(model.parameters()).device
    total, wsum = 0.0, 0.0
    with torch.set_grad_enabled(is_train):
        for cond_idx, sched, weight in loader:
            drop_mask = (torch.rand(sched.size(0)) < P_UNCOND).to(dev) if is_train else None
            cond_idx, sched, weight = cond_idx.to(dev), sched.to(dev), weight.to(dev)
            loss = batch_loss(model, cond_idx, sched, weight, drop_mask)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            w = float(weight.sum())
            total += loss.item() * w
            wsum += w
    return total / wsum


def train(epochs: int = EPOCHS, seed: int = SEED, save_path: Path | None = MODEL_SAVE_PATH,
          save_every: int = SAVE_EVERY, device: str = DEVICE) -> tuple[GRUScheduler, dict[str, list[float]]]:
    """学習し、val の重み付き交差エントロピーが最良の重みのモデルを返す

    Args:
        epochs: 学習 epoch の上限, default=EPOCHS=300
        seed: 学習の乱数の種（初期値・並び順・条件を落とす行）, default=SEED=42。分割は変えない
        save_path: 最良の ckpt の保存先。None なら何も保存しない
        save_every: 正なら N epoch ごとに sm.epoch_ckpt_path(save_path, ep) へ保存する, default=SAVE_EVERY=5
        device: 学習するデバイス, default=DEVICE

    Returns:
        (最良の重みを戻したモデル, 学習曲線 {"epoch", "train", "val", "sec"})

    Raises:
        ValueError: epochs < 1、save_every < 0、または save_every > 0 で save_path が None のとき
    """
    if epochs < 1:
        raise ValueError(f"epochs は 1 以上: {epochs}")
    if save_every < 0:
        raise ValueError(f"save_every は 0 以上: {save_every}")
    if save_every > 0 and save_path is None:
        raise ValueError("save_every > 0 には save_path が要る（途中の ckpt の保存先を作るため）")

    torch.manual_seed(seed)
    train_part, val_part = load_split()
    model = GRUScheduler().to(device)
    model.init_slot_bias(sm.population_rates(train_part.sched, train_part.weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train_part, shuffle=True)
    val_loader = make_loader(val_part, shuffle=False)
    print(f"device={device}  train={len(train_part.sched)}  val={len(val_part.sched)}  "
          f"params={count_params(model):,}", flush=True)

    history: dict[str, list[float]] = {"epoch": [], "train": [], "val": [], "sec": []}
    best_val, best_epoch, no_improve = math.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    ep = 0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, optimizer)
        va = run_epoch(model, val_loader)
        sec = time.perf_counter() - t0
        for key, val in (("epoch", ep), ("train", tr), ("val", va), ("sec", sec)):
            history[key].append(float(val))
        if ep == 1 or ep % 5 == 0:
            print(f"epoch {ep:4d} | train {tr:.5f} | val {va:.5f} | {sec:.1f}s", flush=True)
        if save_path is not None and save_every > 0 and ep % save_every == 0:
            save_ckpt(model, sm.epoch_ckpt_path(save_path, ep), seed, epoch=ep)
        if va < best_val - EARLY_STOP_MIN_DELTA:
            best_val, best_epoch, no_improve = va, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {ep} (best {best_val:.5f} @ epoch {best_epoch})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    gap = teacher_forced_gap(model, train_part)
    print(f"restored best: epoch {best_epoch} (val {best_val:.5f}) | "
          f"calibration gap on train: max {gap['max']:.2e}, mean {gap['mean']:.2e}", flush=True)
    if save_path is not None:
        save_ckpt(model, save_path, seed, extra={
            "best_epoch": best_epoch, "best_val": best_val, "stopped_epoch": ep,
            "calib_gap_max": gap["max"], "calib_gap_mean": gap["mean"], "history": history})
        print(f"saved model to {save_path}")
    return model, history


def count_params(model: nn.Module) -> int:
    """学習するパラメータの数"""
    return sum(p.numel() for p in model.parameters())


def save_ckpt(model: GRUScheduler, path: Path, seed: int, epoch: int | None = None,
              extra: dict[str, Any] | None = None) -> None:
    """重みと出所の記録（config）を {"model", "config"} で保存する

    Args:
        model: 保存するモデル
        path: 保存先
        seed: 学習の乱数の種
        epoch: 途中の ckpt ならその epoch。None なら書かない（最良の ckpt）
        extra: config に足す記録（最良の ckpt の best_epoch・学習曲線など）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"hidden": model.hidden, "num_layers": model.num_layers,
                              "time_encoding": TIME_ARCH.clock_kind, "seed": seed}
    if epoch is not None:
        config["epoch"] = epoch
    if extra is not None:
        config.update(extra)
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_model(path: Path, device: str = DEVICE) -> GRUScheduler:
    """save_ckpt で保存した ckpt から、同じ構造のモデルを eval モードで戻す"""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = GRUScheduler(hidden=cfg["hidden"], num_layers=cfg["num_layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ============================================================
# 5. 生成と teacher forcing の行動者率
# ============================================================
def guided_logits(l_cond: torch.Tensor, l_uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """logits にかける CFG: l_u + g·(l_c − l_u)。g = 1 で条件付きの logits そのもの"""
    return l_uncond + guidance_scale * (l_cond - l_uncond)


def draw_categorical(logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """softmax(logits) から 1 つずつ引く（逆関数法。乱数は CPU で引く）

    Note:
        ★乱数を CPU の generator で引くので、同じ確率なら学習したデバイスによらず同じ活動を引く

    Args:
        logits: (B, NUM_ACT)
        generator: CPU の torch.Generator。None なら大域の乱数

    Returns:
        引いた活動, dtype=int64, (B,)。CPU 上
    """
    p = torch.softmax(logits.float(), dim=-1).cpu().double()
    u = torch.rand(p.size(0), generator=generator, dtype=torch.float64)
    return (p.cumsum(dim=-1) < u[:, None]).sum(dim=-1).clamp_max(NUM_ACT - 1)


@torch.no_grad()
def sample(model: GRUScheduler, cond_idx: torch.Tensor, guidance_scale: float = GUIDANCE_SCALE,
           generator: torch.Generator | None = None, extra_bias: torch.Tensor | None = None) -> torch.Tensor:
    """s = 0..95 の順に 1 スロットずつ引いて 1 日を作る（温度 1.0）

    Note:
        ★extra_bias は CFG の後の logits に 1 回だけ足す。条件付き・条件なしの両方に足すのと同じ
          （(l_u + δ) + g·((l_c + δ) − (l_u + δ)) = l_u + g·(l_c − l_u) + δ）

    Args:
        model: 学習済みのモデル
        cond_idx: 条件インデックス, dtype=int64, (B, 3)
        guidance_scale: logits の CFG の強さ g, default=GUIDANCE_SCALE=1.0。
            1.0 では条件なしの経路を計算しない
        generator: CPU の torch.Generator, default=None
        extra_bias: 行ごと・スロットごとに logits へ足す値, (B, 96, 12)。None なら足さない（Stage 2 の δ）

    Returns:
        生成スケジュール, dtype=int64, (B, 96)。CPU 上、値域 [0, 12)
    """
    dev = next(model.parameters()).device
    batch = cond_idx.size(0)
    out = torch.empty((batch, NUM_SLOTS), dtype=torch.long)
    with sm.eval_mode(model):
        c_emb = model.embed_cond(cond_idx.to(dev), batch)
        u_emb = model.embed_cond(None, batch) if guidance_scale != 1.0 else None
        a_prev = torch.full((batch,), BOS, dtype=torch.long, device=dev)
        state_c: torch.Tensor | None = None
        state_u: torch.Tensor | None = None
        for s in range(NUM_SLOTS):
            logits, state_c = model.step(a_prev, s, c_emb, state_c)
            if u_emb is not None:
                l_uncond, state_u = model.step(a_prev, s, u_emb, state_u)
                logits = guided_logits(logits, l_uncond, guidance_scale)
            if extra_bias is not None:
                logits = logits + extra_bias[:, s].to(logits)
            a = draw_categorical(logits, generator)
            out[:, s] = a
            a_prev = a.to(dev)
    return out


def group_pool(model: GRUScheduler, n_per_group: int = POOL_N, guidance_scale: float = GUIDANCE_SCALE,
               seed: int = POOL_SEED, group_bias: FloatArr | None = None) -> IntArr:
    """群別サンプルプールを作る, -> (28, M, 96)。行 d は sm.cond_grid()[d] の条件

    Args:
        model: 学習済みのモデル
        n_per_group: 群あたりの本数 M, default=POOL_N=256
        guidance_scale: CFG の強さ, default=GUIDANCE_SCALE=1.0
        seed: 生成の乱数の種, default=POOL_SEED=12345
        group_bias: 群ごとに logits へ足す値, (D_GROUPS, 96, 12)。None なら足さない（Stage 2 の δ）

    Returns:
        群別サンプルプール, dtype=int64, (D_GROUPS, M, NUM_SLOTS)
    """
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    flat = grid.repeat_interleave(n_per_group, dim=0)                          # (28·M, 3)
    flat_d = np.repeat(np.arange(D_GROUPS), n_per_group)                        # (28·M,)
    gen = torch.Generator().manual_seed(seed)
    outs = []
    for i in range(0, flat.size(0), GEN_BATCH):
        bias = None if group_bias is None else torch.as_tensor(
            group_bias[flat_d[i:i + GEN_BATCH]], dtype=torch.float32)        # (B, 96, 12)
        outs.append(sample(model, flat[i:i + GEN_BATCH], guidance_scale, gen, bias))
    return torch.cat(outs).numpy().astype(np.int64).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


@torch.no_grad()
def teacher_forced_rates(model: GRUScheduler, sched: IntArr, cond_idx: IntArr, weight: FloatArr,
                         batch: int = GEN_BATCH) -> FloatArr:
    """実データの履歴で条件付けた予測確率の加重平均 Σ_i w_i p_θ[i, s, c] / Σ_i w_i

    §3.2 の一致の左辺。学習分割で、学習データの行動者率（sm.population_rates）と比べる。
    自分で生成した総量との差が計画書の Q2

    Args:
        model: 学習済みのモデル
        sched: 実データのスケジュール, (N, 96)
        cond_idx: 条件インデックス, (N, 3)
        weight: 個票の重み, (N,)
        batch: 一度に通す人数, default=GEN_BATCH

    Returns:
        予測確率の加重平均, dtype=float64, (NUM_ACT, NUM_SLOTS)
    """
    dev = next(model.parameters()).device
    acc = np.zeros((NUM_SLOTS, NUM_ACT), dtype=np.float64)
    with sm.eval_mode(model):
        for i in range(0, len(sched), batch):
            s = torch.as_tensor(sched[i:i + batch], dtype=torch.long, device=dev)
            c = torch.as_tensor(cond_idx[i:i + batch], dtype=torch.long, device=dev)
            p = torch.softmax(model(shift_right(s), c), dim=-1).cpu().numpy().astype(np.float64)
            acc += np.einsum("n,nsc->sc", weight[i:i + batch], p)
    return (acc / weight.sum()).T


def teacher_forced_gap(model: GRUScheduler, part: SplitPart) -> dict[str, float]:
    """|teacher_forced_rates − 行動者率| の最大と平均（§3.2 の一致がどこまで成り立っているか）"""
    tf = teacher_forced_rates(model, part.sched, part.cond_idx, part.weight)
    real = sm.population_rates(part.sched, part.weight).numpy().astype(np.float64)
    diff = np.abs(tf - real)
    return {"max": float(diff.max()), "mean": float(diff.mean())}


# ============================================================
# 6. 学習後の slot_bias の補正（計画書 §9）
# ============================================================
def split_group_weights(part: SplitPart) -> FloatArr:
    """分割の TUFINLWGT を群ごとに足して正規化した群の重み, -> (28,)"""
    d = sm.cond_to_d(part.cond_idx)
    tot = np.bincount(d, weights=part.weight, minlength=D_GROUPS).astype(np.float64)
    return tot / tot.sum()


def generated_rates(pool: IntArr, pi_d: FloatArr) -> FloatArr:
    """群別プールの時刻別行動者率を、群の重み pi_d で 1 本にする, (28, M, 96) -> (12, 96)

    Args:
        pool: 群別サンプルプール, (D_GROUPS, M, NUM_SLOTS)
        pi_d: 群の重み, (D_GROUPS,)。和が 1

    Returns:
        加重平均の時刻別行動者率, dtype=float64, (NUM_ACT, NUM_SLOTS)
    """
    rates = (pool[..., None] == np.arange(NUM_ACT)).mean(axis=1)            # (28, 96, 12)
    return np.asarray(np.einsum("d,dsc->cs", pi_d, rates), dtype=np.float64)


def bias_update(gen: FloatArr, target: FloatArr, step: float = CALIB_STEP,
                eps: float = CALIB_EPS) -> FloatArr:
    """slot_bias に足す量 step · log((target + eps) / (gen + eps)), (12, 96) -> (96, 12)

    Note:
        ★logits が履歴に依らない分布（softmax(slot_bias) そのもの）なら、step = 1・eps = 0 の 1 回で
          softmax の出力が target に一致する。GRU は履歴に依るので、反復して近づける

    Args:
        gen: 生成した時刻別行動者率, (NUM_ACT, NUM_SLOTS)
        target: 目標の時刻別行動者率, (NUM_ACT, NUM_SLOTS)
        step: 更新の幅, default=CALIB_STEP=0.5
        eps: 分母と分子に足す値, default=CALIB_EPS=1e-4

    Returns:
        slot_bias と同じ形の更新量, (NUM_SLOTS, NUM_ACT)
    """
    return np.asarray(step * np.log((target + eps) / (gen + eps)), dtype=np.float64).T


def calibrate_slot_bias(model: GRUScheduler, target: FloatArr, pi_d: FloatArr,
                        guidance_scale: float = GUIDANCE_SCALE, iters: int = CALIB_ITERS,
                        n_per_group: int = CALIB_POOL_N, seed: int = CALIB_SEED,
                        verbose: bool = True) -> list[dict[str, Any]]:
    """生成した時刻別行動者率が target に合うよう、slot_bias だけを反復で動かす（model を書き換える）

    Note:
        1. 反復 k では、種 seed + k のプール（CFG の強さ guidance_scale）を作り、generated_rates で 1 本にしてから
           bias_update を足す。補正した総量が一致するのは、同じ g で生成したときだけ
        2. 最後の補正の後にもう 1 回プールを作って測る（戻り値の最後の要素。slot_bias は変えない）

    Args:
        model: 学習済みのモデル
        target: 目標の時刻別行動者率, (NUM_ACT, NUM_SLOTS)。学習分割の sm.population_rates
        pi_d: 生成側の群の重み, (D_GROUPS,)。目標と同じ群構成（split_group_weights(学習分割)）
        guidance_scale: 補正のプールを作る CFG の強さ g, default=GUIDANCE_SCALE=1.0
        iters: 補正の回数, default=CALIB_ITERS=8
        n_per_group: 1 回のプールの群あたりの本数, default=CALIB_POOL_N=1024
        seed: 1 回目のプールの種, default=CALIB_SEED=20000
        verbose: 反復ごとに差を print するか

    Returns:
        反復ごとの記録 [{"iter", "max_abs_gap", "mean_abs_gap", "total_ratio"（12 活動の 生成 / 目標）}]。
        長さは iters + 1
    """
    history: list[dict[str, Any]] = []
    for k in range(iters + 1):
        gen = generated_rates(group_pool(model, n_per_group, guidance_scale, seed=seed + k), pi_d)
        gap = np.abs(gen - target)
        ratio = gen.mean(axis=1) / target.mean(axis=1)
        history.append({"iter": k, "max_abs_gap": float(gap.max()), "mean_abs_gap": float(gap.mean()),
                        "total_ratio": [float(v) for v in ratio]})
        if verbose:
            print(f"calib {k} | max |gen - target| {gap.max():.4f} | mean {gap.mean():.5f} | total ratio "
                  + " ".join(f"{sm.ACT_NAMES[c][:4]}={ratio[c]:.3f}" for c in range(NUM_ACT)), flush=True)
        if k == iters:
            break
        with torch.no_grad():
            model.slot_bias.add_(torch.as_tensor(bias_update(gen, target)).to(model.slot_bias))
    return history


def run_calibration(seed: int, guidance_scale: float = GUIDANCE_SCALE) -> None:
    """保存済みの最良の ckpt の slot_bias を CFG の強さ guidance_scale で補正し、補正した ckpt と
    同じ g の生成プールを書く（保存先は ckpt_path / pool_path の calib_guidance=guidance_scale）"""
    source = ckpt_path(seed)
    model = load_model(source)
    train_part, _ = load_split()
    target = sm.population_rates(train_part.sched, train_part.weight).numpy().astype(np.float64)
    history = calibrate_slot_bias(model, target, split_group_weights(train_part), guidance_scale)
    out = ckpt_path(seed, calib_guidance=guidance_scale)
    save_ckpt(model, out, seed, extra={"calibrated_from": source.name, "calibration": {
        "guidance_scale": guidance_scale, "iters": CALIB_ITERS, "pool_n": CALIB_POOL_N, "seed": CALIB_SEED,
        "step": CALIB_STEP, "eps": CALIB_EPS, "history": history}})
    print(f"saved calibrated model to {out}")
    write_pool(model, pool_path(seed, guidance_scale, calib_guidance=guidance_scale), guidance_scale)


# ============================================================
# 7. 保存先
# ============================================================
def run_suffix(seed: int) -> str:
    """保存先の接尾辞。種 42 は空、それ以外は _s{seed}（DDPM と同じ規則）"""
    return "" if seed == SEED else f"_s{seed}"


def calib_tag(calib_guidance: float | None) -> str:
    """補正の接尾辞。None（補正なし）は空、g = 1.0 で補正したものは _cal、それ以外は _calg{G}"""
    if calib_guidance is None:
        return ""
    return CALIB_TAG if calib_guidance == GUIDANCE_SCALE else f"{CALIB_TAG}g{calib_guidance:g}"


def ckpt_path(seed: int, calib_guidance: float | None = None) -> Path:
    """最良の ckpt のパス。calib_guidance を渡すと、その g で slot_bias を補正した ckpt"""
    cal = calib_tag(calib_guidance)
    return MODEL_SAVE_PATH.with_name(f"{MODEL_SAVE_PATH.stem}{run_suffix(seed)}{cal}{MODEL_SAVE_PATH.suffix}")


def pool_path(seed: int, guidance_scale: float = GUIDANCE_SCALE, calib_guidance: float | None = None) -> Path:
    """生成プールの CSV のパス

    Args:
        seed: 学習の種
        guidance_scale: 生成の CFG の強さ。既定以外なら _g{G} を付ける
        calib_guidance: 補正した ckpt のプールなら、補正に使った g（calib_tag の接尾辞を付ける）
    """
    cal = calib_tag(calib_guidance)
    tag = "" if guidance_scale == GUIDANCE_SCALE else f"_g{guidance_scale:g}"
    return GEN_SAVE_PATH.with_name(f"{GEN_SAVE_PATH.stem}{run_suffix(seed)}{cal}{tag}{GEN_SAVE_PATH.suffix}")


def write_pool(model: GRUScheduler, path: Path, guidance_scale: float) -> None:
    """群別サンプルプール（28 群 × POOL_N 本）を作って CSV に書く"""
    t0 = time.perf_counter()
    pool = group_pool(model, POOL_N, guidance_scale)
    sm.write_pool_csv(pool, path)
    print(f"saved pool ({D_GROUPS}×{POOL_N}, g={guidance_scale:g}, "
          f"{time.perf_counter() - t0:.1f}s) to {path}", flush=True)


def smoke() -> None:
    """2 epoch 学習し、生成と teacher forcing の行動者率の形を確かめる（何も保存しない）"""
    model, history = train(epochs=2, save_path=None, save_every=0)
    assert len(history["val"]) == 2 and all(math.isfinite(v) for v in history["val"])
    for g in (GUIDANCE_SCALE, 1.25):
        pool = group_pool(model, 2, g)
        assert pool.shape == (D_GROUPS, 2, NUM_SLOTS) and 0 <= pool.min() and pool.max() < NUM_ACT
    _, val_part = load_split()
    tf = teacher_forced_rates(model, val_part.sched, val_part.cond_idx, val_part.weight)
    assert tf.shape == (NUM_ACT, NUM_SLOTS) and np.allclose(tf.sum(axis=0), 1.0)
    print(f"smoke: OK (params={count_params(model):,})")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="GRU_Aggregate: 再帰型＋交差エントロピーの Stage 1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY)
    ap.add_argument("--guidance", type=float, default=GUIDANCE_SCALE)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.no_pool and args.pool_only:
        ap.error("--no-pool と --pool-only は併用できない")
    if args.calibrate and (args.pool_only or args.no_pool):
        ap.error("--calibrate は --pool-only / --no-pool と併用できない")

    ckpt = ckpt_path(args.seed)
    pool = pool_path(args.seed, args.guidance)
    if args.calibrate:                        # 補正は読み込む ckpt と書く ckpt・プールが別
        pool = pool_path(args.seed, args.guidance, calib_guidance=args.guidance)
        print(f"[config] calibrated ckpt={ckpt_path(args.seed, calib_guidance=args.guidance).name}")
    print(f"[config] seed={args.seed} epochs={args.epochs} save_every={args.save_every} "
          f"guidance={args.guidance:g}")
    print(f"[config] ckpt={ckpt.name}  pool={pool.name}")
    if args.smoke:
        smoke()
        return
    if args.calibrate:
        run_calibration(args.seed, args.guidance)
        return
    if args.pool_only:
        model = load_model(ckpt)
    else:
        model, _ = train(epochs=args.epochs, seed=args.seed, save_path=ckpt, save_every=args.save_every)
    if not args.no_pool:
        write_pool(model, pool, args.guidance)


if __name__ == "__main__":
    main()
