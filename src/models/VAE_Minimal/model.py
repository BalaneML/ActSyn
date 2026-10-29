"""
model.py
========
VAE_Minimal: 単純な MLP の条件付き VAE（CVAE）。共通の 2 段階学習の比較対象の 1 つ（2026-10-29 ユーザー決定）

1 日の 96 スロット × 12 活動を、潜在変数 z（16 次元）と条件（性・年齢・就業）から生成する。
デコーダは全スロットで共有する MLP で、スロット s の活動の logits を [z; 条件; 時刻埋め込み[s]] から出す。
同じ z のもとでスロットどうしは独立に引くので、日誌は細切れになりやすい（以前の CVAE で切り替え 48.1 回、実データ 12.64 回）。

決定事項（2026-10-29）:
    - 構造: 単純な MLP（エンコーダ・デコーダとも 1 層）、β = 1、学習型の時刻埋め込み Embedding(96, ·)
    - 条件: COND_SPEC（属性ごとの埋め込み → 連結 → 線形）。重みはこのモデルが学習する
    - 大きさ: 生成側（デコーダ・条件・時刻埋め込み）を LSTM（42,196）に合わせる → HIDDEN = 222 で 42,170
    - 損失: 1 人あたりの ELBO（96 スロットの交差エントロピーの和 + KL）。正規化しない。重みなし
    - Stage 1 の学習設定は LSTM と同じ（AdamW lr 1e-3・weight decay 0.01（2 次元以上）・clip 1.0・batch 256・
      最大 1,000 epoch・val が 30 epoch 改善しなければ停止・val 最良を採用）。値は LSTM のモジュールから取る

構造:

```mermaid
flowchart LR
    COND["cond_idx (B, 3)"] --> EC["embed_cond<br/>cond_embeds → cond_proj<br/>c_emb (B, COND_DIM)"]
    SCHED["sched (B, 96)"] --> OH["one_hot → flatten<br/>x (B, 96·12)"]
    OH --> ENC["encode(sched, c_emb)<br/>enc_hidden → ReLU → enc_out<br/>mu, logvar (B, Z_DIM)"]
    EC --> ENC
    ENC --> Z["z = mu + exp(logvar/2)·eps<br/>（生成時は z ~ N(0, I)）"]
    Z --> DEC["decode(z, c_emb)<br/>各スロット s で [z; c_emb; time_embed[s]]<br/>dec_hidden → ReLU → dec_out<br/>logits (B, 96, 12)"]
    EC --> DEC
    TE["time_embed<br/>Embedding(96, TIME_DIM)"] --> DEC
```

学習と生成:

```mermaid
flowchart TD
    TR["train(epochs, seed, save_path, device)<br/>run_epoch → elbo（重みなし）<br/>val_loss（雑音の乱数を固定）で早期終了"] --> CK["save_ckpt(model, ckpt_path(seed), seed, extra)"]
    CK --> GP["group_pool(model, n_per_group, seed)<br/>z ~ N(0, I) → decode → gm.draw_categorical<br/>pool (28, n_per_group, 96)"]
```

使い方:
    .venv/bin/python src/models/VAE_Minimal/model.py --smoke
    .venv/bin/python src/models/VAE_Minimal/model.py --seed 42 --threads 4

出力:
    outputs/checkpoints/vae_minimal{構成}{接尾辞}.pt   val 最良の ckpt（config に学習曲線 history）
        {構成} = stage1_tag() = _z16_h222_wd0.01、{接尾辞} は種 42 なら空、それ以外は _s{seed}
"""
import argparse
import copy
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/VAE_Minimal -> repo root


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


# Stage 1 の学習設定・分割・生成の引き方は LSTM と同じ値と関数を使う（写し書きしない）
lm: Any = _load("lstm_aggregate_model", REPO_ROOT / "src" / "models" / "LSTM_Aggregate" / "model.py")
gm: Any = lm.gm
sm: Any = lm.sm

IntArr = npt.NDArray[np.int64]

# ============================================================
# 1. 設定
# ============================================================
MODEL_SAVE_PATH = REPO_ROOT / "outputs" / "checkpoints" / "vae_minimal.pt"

NUM_SLOTS: int = sm.NUM_SLOTS        # 96
NUM_ACT: int = sm.NUM_ACT            # 12
D_GROUPS: int = sm.D_GROUPS          # 28

# 構造（2026-10-29 決定。生成側を LSTM の 42,196 に合わせて HIDDEN を決めた）
Z_DIM = 16
HIDDEN = 222                         # エンコーダとデコーダの中間層の幅（同じ値）
COND_DIM = 64                        # 条件の埋め込みの幅（LSTM の HIDDEN と同じ）
TIME_DIM = 64                        # 時刻埋め込みの幅（LSTM の HIDDEN と同じ）
BETA = 1.0                           # KL の重み

# 学習（LSTM と同じ。weight decay は Stage 2 が読み込む LSTM の Pre-trained と同じ 0.01）
LR: float = lm.LR                                    # 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP: float = lm.GRAD_CLIP                      # 1.0
EPOCHS: int = lm.EPOCHS                              # 1000
EARLY_STOP_PATIENCE: int = lm.EARLY_STOP_PATIENCE    # 30
EARLY_STOP_MIN_DELTA: float = lm.EARLY_STOP_MIN_DELTA  # 1e-4
SEED: int = lm.SEED                                  # 42
# val の ELBO の雑音 eps の乱数の種。毎回同じ雑音にして、val を決定的にする（大域の乱数も進めない）
VAL_NOISE_SEED = 0

GEN_BATCH: int = lm.GEN_BATCH        # 1024
DEVICE = "cpu"                       # 小さな MLP なので CPU が速い


# ============================================================
# 2. モデル
# ============================================================
class VAEMinimal(nn.Module):
    """単純な MLP の条件付き VAE: q(z | x, c) と p(x | z, c) = Π_s p(x_s | z, c, s)

    Note:
        1. デコーダは全スロットで共有する MLP。スロット s の入力は [z; c_emb; time_embed[s]]
        2. 生成側（generator_parameters）は cond_embeds・cond_proj・time_embed・dec_hidden・dec_out。
           エンコーダ（enc_hidden・enc_out）は学習にだけ使う
        3. 初期値はすべて PyTorch の既定

    Attributes:
        z_dim: 潜在変数の次元
        hidden: 中間層の幅（エンコーダ・デコーダ共通）
        cond_dim: 条件の埋め込みの幅
        time_dim: 時刻埋め込みの幅
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
        self.z_dim, self.hidden, self.cond_dim, self.time_dim = z_dim, hidden, cond_dim, time_dim
        self.cond_embeds = nn.ModuleList([nn.Embedding(card, dim) for _, card, dim, _ in sm.COND_SPEC])
        self.cond_proj = nn.Linear(sum(sm.EMB_DIMS), cond_dim)
        self.time_embed = nn.Embedding(NUM_SLOTS, time_dim)
        self.enc_hidden = nn.Linear(NUM_SLOTS * NUM_ACT + cond_dim, hidden)
        self.enc_out = nn.Linear(hidden, 2 * z_dim)
        self.dec_hidden = nn.Linear(z_dim + cond_dim + time_dim, hidden)
        self.dec_out = nn.Linear(hidden, NUM_ACT)

    def embed_cond(self, cond_idx: torch.Tensor) -> torch.Tensor:
        """属性（性・年齢・就業）を埋め込む, (B, 3) -> (B, COND_DIM)

        Args:
            cond_idx: 条件インデックス, dtype=int64, (B, 3)。列は sm.COND_SPEC の順

        Returns:
            c_emb: 条件の埋め込み, (B, COND_DIM)
        """
        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)   # (B, 16)
        return self.cond_proj(c)

    def encode(self, sched: torch.Tensor, c_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | x, c) の平均と対数分散

        Args:
            sched: 実データの活動, dtype=int64, (B, 96)
            c_emb: embed_cond の出力, (B, COND_DIM)

        Returns:
            (mu (B, Z_DIM), logvar (B, Z_DIM))
        """
        x = F.one_hot(sched, NUM_ACT).float().reshape(sched.size(0), NUM_SLOTS * NUM_ACT)   # (B, 1152)
        h = F.relu(self.enc_hidden(torch.cat([x, c_emb], dim=1)))
        mu, logvar = self.enc_out(h).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """各スロットの活動の logits。全スロットで共有する MLP に [z; c_emb; time_embed[s]] を通す

        Args:
            z: 潜在変数, (B, Z_DIM)
            c_emb: embed_cond の出力, (B, COND_DIM)

        Returns:
            logits, (B, 96, NUM_ACT)
        """
        batch = z.size(0)
        t = self.time_embed.weight                                              # (96, TIME_DIM)
        inp = torch.cat([z[:, None, :].expand(batch, NUM_SLOTS, -1),
                         c_emb[:, None, :].expand(batch, NUM_SLOTS, -1),
                         t[None, :, :].expand(batch, -1, -1)], dim=2)          # (B, 96, Z+C+T)
        return self.dec_out(F.relu(self.dec_hidden(inp)))

    def generator_parameters(self) -> list[nn.Parameter]:
        """生成に使うパラメータ（大きさを LSTM と比べる対象）"""
        parts: list[nn.Module] = [self.cond_embeds, self.cond_proj, self.time_embed, self.dec_hidden, self.dec_out]
        return [p for m in parts for p in m.parameters()]


def stage1_tag(z_dim: int = Z_DIM, hidden: int = HIDDEN, weight_decay: float = WEIGHT_DECAY) -> str:
    """Pre-trained の構成の印（例: _z16_h222_wd0.01）"""
    return f"_z{z_dim}_h{hidden}_wd{weight_decay:g}"


# ============================================================
# 3. 損失
# ============================================================
def elbo(model: VAEMinimal, cond_idx: torch.Tensor, sched: torch.Tensor,
         generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1 人あたりの負の ELBO（再構成の交差エントロピーの 96 スロットの和 + β·KL）のバッチ平均

    Note:
        1. 重みなし（TUFINLWGT を使わない）。正規化しない（スロットあたりではなく 1 人あたり）
        2. 再パラメータ化の雑音 eps は CPU で引いてからモデルの計算場所へ移す（計算場所によらず同じ値。
           generator が None なら大域の CPU の乱数）

    Args:
        model: VAEMinimal
        cond_idx: 条件インデックス, dtype=int64, (B, 3)
        sched: 実データの活動, dtype=int64, (B, 96)
        generator: eps の CPU の torch.Generator, default=None（大域の乱数）

    Returns:
        (loss: 負の ELBO のバッチ平均, recon: 再構成のバッチ平均, kl: KL のバッチ平均)。loss だけが勾配を持つ前提で使う
    """
    c_emb = model.embed_cond(cond_idx)
    mu, logvar = model.encode(sched, c_emb)
    eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype).to(mu.device)
    z = mu + torch.exp(0.5 * logvar) * eps
    logits = model.decode(z, c_emb)                                             # (B, 96, 12)
    recon = F.cross_entropy(logits.reshape(-1, NUM_ACT), sched.reshape(-1),
                            reduction="none").reshape(sched.size(0), NUM_SLOTS).sum(dim=1)   # (B,)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)            # (B,)
    loss = (recon + BETA * kl).mean()
    return loss, recon.mean().detach(), kl.mean().detach()


def val_loss(model: VAEMinimal, loader: DataLoader) -> tuple[float, float, float]:
    """val 分割の負の ELBO（人数の加重平均）。eps の乱数を毎回 VAL_NOISE_SEED から引くので決定的

    Args:
        model: VAEMinimal
        loader: val 分割の DataLoader（shuffle なし）

    Returns:
        (負の ELBO, 再構成, KL)。いずれも 1 人あたり
    """
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(VAL_NOISE_SEED)
    total, rec_sum, kl_sum, count = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for cond_idx, sched, _ in loader:
            loss, recon, kl = elbo(model, cond_idx.to(dev), sched.to(dev), gen)
            w = float(sched.size(0))
            total += loss.item() * w
            rec_sum += recon.item() * w
            kl_sum += kl.item() * w
            count += w
    return total / count, rec_sum / count, kl_sum / count


# ============================================================
# 4. 学習と読み込み
# ============================================================
def train(epochs: int = EPOCHS, seed: int = SEED, save_path: Path | None = MODEL_SAVE_PATH,
          device: str = DEVICE) -> tuple[VAEMinimal, dict[str, list[float]]]:
    """学習し、val の負の ELBO が最良の重みのモデルを返す（手順は lm.train と同じ）

    Note:
        1. モデルは torch.manual_seed(seed) の後に作るので、初期値も種で決まる。分割は gm.load_split で固定
        2. 早期終了・最良の重みの復元・clip・weight decay の掛け方は LSTM と同じ

    Args:
        epochs: 学習 epoch の上限, default=EPOCHS=1000
        seed: 学習の乱数の種, default=SEED=42
        save_path: 最良の ckpt の保存先。None なら何も保存しない
        device: 学習するデバイス, default=DEVICE="cpu"

    Returns:
        (最良の重みを戻したモデル, 学習曲線 {"epoch", "train", "val", "recon_val", "kl_val", "sec"})

    Raises:
        ValueError: epochs < 1 のとき
    """
    if epochs < 1:
        raise ValueError(f"epochs は 1 以上: {epochs}")
    torch.manual_seed(seed)
    train_part, val_part = gm.load_split()
    model = VAEMinimal().to(device)
    optimizer = torch.optim.AdamW(lm.param_groups(model, WEIGHT_DECAY), lr=LR)
    train_loader = gm.make_loader(train_part, shuffle=True)
    val_loader = gm.make_loader(val_part, shuffle=False)
    n_gen = sum(p.numel() for p in model.generator_parameters())
    print(f"device={device}  train={len(train_part.sched)}  val={len(val_part.sched)}  z={model.z_dim}  "
          f"hidden={model.hidden}  params={lm.count_params(model):,}（生成側 {n_gen:,}）", flush=True)

    history: dict[str, list[float]] = {"epoch": [], "train": [], "val": [], "recon_val": [], "kl_val": [], "sec": []}
    best_val, best_epoch, no_improve = math.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    ep = 0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, count = 0.0, 0.0
        for cond_idx, sched, _ in train_loader:
            loss, _, _ = elbo(model, cond_idx.to(device), sched.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item() * sched.size(0)
            count += sched.size(0)
        va, rec, kl = val_loss(model, val_loader)
        sec = time.perf_counter() - t0
        for key, val in (("epoch", ep), ("train", total / count), ("val", va), ("recon_val", rec), ("kl_val", kl),
                         ("sec", sec)):
            history[key].append(float(val))
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:4d} | train {total / count:.4f} | val {va:.4f}（再構成 {rec:.4f}・KL {kl:.4f}）| "
                  f"{sec:.2f}s", flush=True)
        if va < best_val - EARLY_STOP_MIN_DELTA:
            best_val, best_epoch, no_improve = va, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {ep} (best {best_val:.4f} @ epoch {best_epoch})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    print(f"restored best: epoch {best_epoch} (val {best_val:.4f})", flush=True)
    if save_path is not None:
        save_ckpt(model, save_path, seed, extra={"best_epoch": best_epoch, "best_val": best_val,
                                                  "stopped_epoch": ep, "history": history})
        print(f"saved model to {save_path}")
    return model, history


def save_ckpt(model: VAEMinimal, path: Path, seed: int, extra: dict[str, Any] | None = None) -> None:
    """重みと出所の記録（config）を {"model", "config"} で保存する

    Args:
        model: 保存するモデル
        path: 保存先
        seed: 学習の乱数の種
        extra: config に足す記録（best_epoch・学習曲線など）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"z_dim": model.z_dim, "hidden": model.hidden, "cond_dim": model.cond_dim,
                              "time_dim": model.time_dim, "beta": BETA, "weight_decay": WEIGHT_DECAY, "seed": seed}
    if extra is not None:
        config.update(extra)
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_model(path: Path, device: str = DEVICE) -> VAEMinimal:
    """save_ckpt で保存した ckpt から、同じ構造のモデルを eval モードで戻す

    Args:
        path: ckpt のパス
        device: 載せるデバイス, default=DEVICE="cpu"

    Returns:
        eval モードのモデル
    """
    cfg = torch.load(path, map_location="cpu", weights_only=False)
    arch = cfg["config"]
    model = VAEMinimal(z_dim=arch["z_dim"], hidden=arch["hidden"], cond_dim=arch["cond_dim"],
                       time_dim=arch["time_dim"]).to(device)
    model.load_state_dict(cfg["model"])
    return model.eval()


def ckpt_path(seed: int) -> Path:
    """最良の ckpt のパス（例: 種 42 は vae_minimal_z16_h222_wd0.01.pt、種 43 は vae_minimal_z16_h222_wd0.01_s43.pt）"""
    return MODEL_SAVE_PATH.with_name(f"{MODEL_SAVE_PATH.stem}{stage1_tag()}{lm.run_suffix(seed)}{MODEL_SAVE_PATH.suffix}")


# ============================================================
# 5. 生成
# ============================================================
def prior_logits(model: VAEMinimal, cond_idx: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """z ~ N(0, I) を generator で引き、各スロットの logits を作る（生成の前半。勾配は通す）

    Args:
        model: VAEMinimal
        cond_idx: 条件インデックス, dtype=int64, (B, 3)。モデルと同じデバイス
        generator: z の CPU の torch.Generator

    Returns:
        logits, (B, 96, NUM_ACT)
    """
    dev = next(model.parameters()).device
    z = torch.randn((cond_idx.size(0), model.z_dim), generator=generator).to(dev)
    return model.decode(z, model.embed_cond(cond_idx))


@torch.no_grad()
def group_pool(model: VAEMinimal, n_per_group: int, seed: int) -> IntArr:
    """群別サンプルプールを作る, -> (28, n_per_group, 96)。行 d は sm.cond_grid()[d] の条件

    Note:
        z と各スロットの活動を同じ CPU の generator から順に引く（z ~ N(0, I) → スロットごとに独立に温度 1 で引く）

    Args:
        model: VAEMinimal
        n_per_group: 群あたりの本数
        seed: 生成の乱数の種

    Returns:
        群別サンプルプール, dtype=int64, (D_GROUPS, n_per_group, NUM_SLOTS)
    """
    model.eval()
    dev = next(model.parameters()).device
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    flat = grid.repeat_interleave(n_per_group, dim=0)                          # (28·M, 3)
    gen = torch.Generator().manual_seed(seed)
    outs = []
    for i in range(0, flat.size(0), GEN_BATCH):
        cond = flat[i:i + GEN_BATCH].to(dev)
        logits = prior_logits(model, cond, gen)                                 # (b, 96, 12)
        acts = gm.draw_categorical(logits.reshape(-1, NUM_ACT), gen)            # (b·96,)
        outs.append(acts.reshape(cond.size(0), NUM_SLOTS))
    return torch.cat(outs).numpy().astype(np.int64).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


# ============================================================
# 6. CLI
# ============================================================
def smoke(device: str = DEVICE) -> None:
    """2 epoch 学習し、生成の形を確かめる（何も保存しない）"""
    model, history = train(epochs=2, save_path=None, device=device)
    assert len(history["val"]) == 2 and all(math.isfinite(v) for v in history["val"])
    pool = group_pool(model, 2, seed=0)
    assert pool.shape == (D_GROUPS, 2, NUM_SLOTS) and 0 <= pool.min() and pool.max() < NUM_ACT
    print(f"smoke: OK (params={lm.count_params(model):,}、生成側 {sum(p.numel() for p in model.generator_parameters()):,})")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="VAE_Minimal: 単純な MLP の条件付き VAE の Stage 1")
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
    path = ckpt_path(args.seed)
    print(f"[config] seed={args.seed} epochs={args.epochs} device={args.device} ckpt={path.name}")
    train(epochs=args.epochs, seed=args.seed, save_path=path, device=args.device)


if __name__ == "__main__":
    main()
