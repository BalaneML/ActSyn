"""
model.py
========
DDPM_Minimal: 既存の DDPM（DDPM_Aggregate_Simple の UNet1D）を小さく・補正なしにしたもの。
共通の 2 段階学習の比較対象の 1 つ（2026-10-29 ユーザー決定）

DDPM_Aggregate_Simple からの変更点（それ以外の denoiser・拡散のスケジュール・逆過程の式は同じ）:
    1. 幅 BASE_CH = 64 → 32（2026-10-30 決定。525,556 + 時刻埋め込み 3,072 = 528,628）
    2. CFG なし（条件を落とす学習をせず、生成は guidance 1.0）。dropout 0.1 → 0
    3. 学習型の時刻埋め込み slot_embed = Embedding(96, BASE_CH) を、入口の畳み込みの出力に足す
       （既存の UNet1D の既定にはスロットの時刻を表す経路が無い）
    4. ミニバッチは一様（TUFINLWGT の重み付き抽出をしない）。Stage 1 の学習設定は LSTM と同じ
       （AdamW lr 1e-3・weight decay 0.01（2 次元以上）・clip 1.0・batch 256・最大 1,000 epoch・
       val が 30 epoch 改善しなければ停止・val 最良を採用）

★幅は 12（= NUM_ACT）以上が要る。最初に使った幅 8（2026-10-29）は、入口 Conv1d(12 → 8) と出口 Conv1d(8 → 12) で
  12 × 96 = 1,152 個の値を 8 × 96 = 768 個に絞るので、ε の二乗誤差がどう学習しても 1/3 を下回れず、生成が崩れた
  （Stage2_Common/docs/Stage2_common_results.md §5）。GroupNorm(8, ·) のため 8 の倍数で、16 が最小
★次に使った幅 16（2026-10-29）は下限が無いが、Pre-trained の活動の割合が ATUS から大きく外れた（総変動距離 0.42。
  Stage2_Common/docs/Stage1_ddpm_minimal_traj_results.md）。幅 32・64 を種 42〜44 で比べ、結果を見る前に決めた
  選び方（総変動距離の範囲が重なれば狭い幅）で 32 にした（Stage1_ddpm_minimal_seeds_results.md）
★幅・dropout は DDPM_Aggregate_Simple/model.py のモジュールの定数で決まる（--kernel と同じ仕組み）。
  別名で読み込んだ写し（base）の定数だけを書き換えるので、ほかの実験が使う同じファイルには影響しない。
  幅は set_base_ch（CLI は --base-ch）で変える。モデルを作る前・ckpt を読む前に呼ぶ

★乱数の置き場所:
    - Stage 1 の t と ε、Stage 2 のリハーサルの t と ε は CPU の大域の乱数で引く（計算場所によらず同じ値。
      共通の Stage 2 の ckpt が状態を保存するので再開も厳密）
    - 逆過程の雑音（x_T と各ステップの z）は量が多い（1,792 本 × 12 × 96 × 1,000 ステップ）ので計算場所の
      generator で引く。Stage 2 では渡された CPU の generator から種を 1 つ引いて計算場所の generator を作るので、
      再開しても同じ標本になる。評価のプールは計算場所ごとに違う標本になる（CPU と GPU のプールは共有しない）

構造（DDPM_Aggregate_Simple の UNet1D に slot_embed を足したもの）:

```mermaid
flowchart LR
    XT["x_t (B, 12, 96)"] --> IC["in_conv"]
    SE["slot_embed.weight.T<br/>(BASE_CH, 96)"] --> ADD["h = in_conv(x_t) + slot_embed"]
    IC --> ADD
    T["t (B,)"] --> EMB["emb = timestep_embedding(t)<br/>+ embed_cond(cond_idx)"]
    C["cond_idx (B, 3)"] --> EMB
    ADD --> UNET["既存の下り・中間・上り経路<br/>（ResBlock1D に emb を注入）"]
    EMB --> UNET
    UNET --> EPS["eps_hat (B, 12, 96)"]
```

学習と生成:

```mermaid
flowchart TD
    SB["set_base_ch(base_ch)<br/>base.BASE_CH を書き換える（--base-ch。既定 32）"] --> TR
    TR["train(epochs, seed, save_path, device, lr, save_every)<br/>eps_mse（t・ε は CPU の大域の乱数）<br/>val_loss（t・ε を固定）で早期終了"] --> CK["save_ckpt(model, ckpt_path(seed, lr, run_tag), seed, extra)"]
    TR -. "save_every > 0（診断用）" .-> EP["save_ckpt(model, epoch_ckpt_path(save_path, ep), seed, extra)"]
    CK --> GP["group_pool(model, diffusion, n_per_group, seed)<br/>reverse_chain(model, diffusion, cond_idx, gen) → argmax<br/>pool (28, n_per_group, 96)"]
    CK --> SR["sample_relaxed_chain(model, diffusion, cond_idx, generator, tau, k)<br/>t = 999..k は勾配なし、t = k−1..0 は勾配あり<br/>base.straight_through(x0, tau) → y (B, 12, 96)"]
```

使い方:
    .venv/bin/python src/models/DDPM_Minimal/model.py --smoke
    .venv/bin/python src/models/DDPM_Minimal/model.py --measure --device cuda   # SQUID で所要時間を測る
    .venv/bin/python src/models/DDPM_Minimal/model.py --seed 42 --device cuda
    # 診断用（Stage2_common_results.md §7）：50 epoch ごとに途中の ckpt も残す。本番の ckpt と別名にする
    .venv/bin/python src/models/DDPM_Minimal/model.py --seed 42 --device cuda --save-every 50 --run-tag _traj
    .venv/bin/python src/models/DDPM_Minimal/model.py --seed 42 --device cuda --save-every 50 --lr 2e-4
    # 幅を変える（既定は 32）。保存名は _ch{幅}_wd0.01 になる
    .venv/bin/python src/models/DDPM_Minimal/model.py --base-ch 64 --seed 42 --device cuda --save-every 100
    .venv/bin/python src/models/DDPM_Minimal/model.py --base-ch 64 --measure --device cuda

出力:
    outputs/checkpoints/ddpm_minimal{構成}{run_tag}{接尾辞}.pt   val 最良の ckpt（config に学習曲線 history）
        {構成} = stage1_tag() = _ch{幅}_wd0.01（幅の既定は 32）。学習率が既定（1e-3）以外なら _lr{学習率} が付く
        （例 _ch32_wd0.01_lr0.0002）
        {run_tag} は --run-tag（既定は空）、{接尾辞} は種 42 なら空、それ以外は _s{seed}
    outputs/checkpoints/ddpm_minimal{構成}{run_tag}{接尾辞}_ep{epoch:04d}.pt   --save-every N のときの途中の ckpt
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

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/DDPM_Minimal -> repo root
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"


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


# Stage 1 の学習設定・分割は LSTM と同じ値と関数を使う
lm: Any = _load("lstm_aggregate_model", REPO_ROOT / "src" / "models" / "LSTM_Aggregate" / "model.py")
gm: Any = lm.gm
sm: Any = lm.sm

# ============================================================
# 1. 設定
# ============================================================
BASE_CH = 32                         # 2026-10-30 決定（16 から変更。NUM_ACT 以上で、GroupNorm(8, ·) のため 8 の倍数）
DROPOUT = 0.0                        # dropout を使わない（全モデル共通の決定）
# ★DDPM_Aggregate_Simple を別名で読み込んだ写し。この写しの定数だけを書き換える
base: Any = _load("ddpm_minimal_base_model", SIMPLE_DIR / "model.py")
base.BASE_CH = BASE_CH
base.DROPOUT = DROPOUT

MODEL_SAVE_PATH = REPO_ROOT / "outputs" / "checkpoints" / "ddpm_minimal.pt"
NUM_SLOTS: int = sm.NUM_SLOTS        # 96
NUM_ACT: int = sm.NUM_ACT            # 12
D_GROUPS: int = sm.D_GROUPS          # 28
T_STEPS: int = base.T_STEPS          # 1000
K_TAIL = 1                           # Stage 2 で勾配を保持する末尾のステップ数（既存の DDPM の既定）
GUIDANCE = 1.0                       # CFG なし

LR: float = lm.LR                                    # 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP: float = lm.GRAD_CLIP                      # 1.0
EPOCHS: int = lm.EPOCHS                              # 1000
EARLY_STOP_PATIENCE: int = lm.EARLY_STOP_PATIENCE    # 30
EARLY_STOP_MIN_DELTA: float = lm.EARLY_STOP_MIN_DELTA  # 1e-4
SEED: int = lm.SEED                                  # 42
VAL_NOISE_SEED = 0                   # val の t と ε の乱数の種（毎回同じにして val を決定的にする）
POOL_BATCH = 8192                    # 評価のプールを作るときの 1 回の本数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. モデル
# ============================================================
class DDPMMinimal(base.UNet1D):
    """幅 base.BASE_CH（既定 32）・CFG なし・dropout なしの UNet1D に、学習型の時刻埋め込みを足した ε 予測ネットワーク

    Note:
        1. 部品は base.UNet1D（時刻符号なしの既定の構造 ArchSpec()）と同じ。幅と dropout は base の定数で決まる
        2. slot_embed（Embedding(96, BASE_CH)）を入口の畳み込みの出力にスロットごとに足す。
           LSTM が入力に時刻埋め込みを足すのと同じ考え方
        3. null_emb（CFG の条件なしの埋め込み）は親の部品として残るが、使わない（勾配は来ない）
    """

    def __init__(self) -> None:
        """部品を作る（親の UNet1D の後に slot_embed を作る）

        Raises:
            ValueError: 幅 base.BASE_CH が入力のチャネル数 NUM_ACT より小さいとき（ε の二乗誤差に下限ができる）
        """
        if base.BASE_CH < NUM_ACT:
            raise ValueError(f"BASE_CH={base.BASE_CH} が入力の {NUM_ACT} チャネルより小さい。入口と出口の畳み込みで"
                             f"値が絞られ、ε の二乗誤差に 1 − BASE_CH/{NUM_ACT} の下限ができる")
        super().__init__(arch=base.ArchSpec())
        self.slot_embed = nn.Embedding(NUM_SLOTS, base.BASE_CH)

    def _encode(self, x_t: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """下り経路（base.UNet1D._encode と同じ。入口に slot_embed を足すところだけ違う）

        Args:
            x_t: ノイズ付き活動スケジュール, (B, 12, 96)
            emb: 拡散ステップ埋め込み + 条件埋め込み, (B, 256)

        Returns:
            (h1 (B, BASE_CH, 96), h2 (B, 2·BASE_CH, 48), h3 (B, 2·BASE_CH, 24))
        """
        h = self.in_conv(x_t) + self.slot_embed.weight.T[None]                  # (B, BASE_CH, 96)
        h1 = self.d1b(self.d1a(h, emb), emb)
        if self.attn1 is not None:
            h1 = self.attn1(h1)
        h2 = self.attn2(self.d2b(self.d2a(self.ds1(h1), emb), emb))
        h3 = self.attn3(self.d3b(self.d3a(self.ds2(h2), emb), emb))
        return h1, h2, h3


def set_base_ch(base_ch: int) -> None:
    """denoiser の幅を変える（写し base の定数 base.BASE_CH を書き換える）

    Note:
        DDPMMinimal・stage1_tag・ckpt_path・save_ckpt・load_model は、呼んだ時点の base.BASE_CH を使う。
        ほかの実験が使う DDPM_Aggregate_Simple のモジュールは変わらない

    Args:
        base_ch: 幅。NUM_ACT（12）以上で、GroupNorm(8, ·) のため 8 の倍数

    Raises:
        ValueError: 幅が NUM_ACT より小さいか、8 の倍数でないとき
    """
    if base_ch < NUM_ACT or base_ch % 8 != 0:
        raise ValueError(f"幅 {base_ch} は使えない。{NUM_ACT} 以上の 8 の倍数にする（GroupNorm(8, ·)）")
    base.BASE_CH = base_ch


def stage1_tag(base_ch: int | None = None, weight_decay: float = WEIGHT_DECAY, lr: float = LR) -> str:
    """Pre-trained の構成の印（例: _ch32_wd0.01。学習率が既定の LR 以外なら _ch32_wd0.01_lr0.0002）

    Args:
        base_ch: 幅, default=None（いまの base.BASE_CH。set_base_ch で変わる）
        weight_decay: weight decay, default=WEIGHT_DECAY=0.01
        lr: 学習率, default=LR=1e-3
    """
    ch = base.BASE_CH if base_ch is None else base_ch
    lr_tag = "" if lr == LR else f"_lr{lr:g}"
    return f"_ch{ch}_wd{weight_decay:g}{lr_tag}"


def count_params(model: nn.Module) -> int:
    """パラメータの数"""
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 3. 損失
# ============================================================
def eps_mse(model: DDPMMinimal, diffusion: Any, cond_idx: torch.Tensor, sched: torch.Tensor,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """ε 予測の二乗誤差（DDPM の標準の損失。重みなし・CFG の条件落としなし）

    Note:
        t ~ U{0, T−1} と ε ~ N(0, I) は CPU で引いてから計算場所へ移す（generator が None なら大域の CPU の乱数）

    Args:
        model: DDPMMinimal
        diffusion: base.Diffusion（スケジュールを持つ）
        cond_idx: 条件インデックス, dtype=int64, (B, 3)
        sched: 実データの活動, dtype=int64, (B, 96)
        generator: t と ε の CPU の torch.Generator, default=None（大域の乱数）

    Returns:
        バッチの二乗誤差の平均（スカラー）
    """
    dev = sched.device
    x0 = base.sched_to_x0(sched)                                               # (B, 12, 96) ∈ {0, 1}
    t = torch.randint(0, T_STEPS, (x0.size(0),), generator=generator).to(dev)
    eps = torch.randn(x0.shape, generator=generator).to(dev)
    eps_hat = model(diffusion.q_sample(x0, t, eps), t, cond_idx)
    return F.mse_loss(eps_hat, eps)


def val_loss(model: DDPMMinimal, diffusion: Any, loader: DataLoader) -> float:
    """val 分割の ε の二乗誤差（人数の加重平均）。t と ε を毎回 VAL_NOISE_SEED から引くので決定的

    Args:
        model: DDPMMinimal
        diffusion: base.Diffusion
        loader: val 分割の DataLoader（shuffle なし）

    Returns:
        1 人あたりの二乗誤差
    """
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(VAL_NOISE_SEED)
    total, count = 0.0, 0.0
    with torch.no_grad():
        for cond_idx, sched, _ in loader:
            loss = eps_mse(model, diffusion, cond_idx.to(dev), sched.to(dev), gen)
            total += loss.item() * sched.size(0)
            count += sched.size(0)
    return total / count


# ============================================================
# 4. 学習と読み込み
# ============================================================
def train(epochs: int = EPOCHS, seed: int = SEED, save_path: Path | None = MODEL_SAVE_PATH,
          device: str = DEVICE, lr: float = LR,
          save_every: int = 0) -> tuple[DDPMMinimal, dict[str, list[float]]]:
    """学習し、val の ε の二乗誤差が最良の重みのモデルを返す（手順は lm.train と同じ）

    Note:
        save_every > 0 のときの途中の ckpt の保存は、乱数を使わないので学習の結果を変えない

    Args:
        epochs: 学習 epoch の上限, default=EPOCHS=1000
        seed: 学習の乱数の種, default=SEED=42
        save_path: 最良の ckpt の保存先。None なら何も保存しない
        device: 学習するデバイス, default=DEVICE
        lr: AdamW の学習率, default=LR=1e-3
        save_every: 正なら N epoch ごとにその時点の重みを epoch_ckpt_path(save_path, ep) へ保存する（診断用）, default=0

    Returns:
        (最良の重みを戻したモデル, 学習曲線 {"epoch", "train", "val", "sec"})

    Raises:
        ValueError: epochs < 1、save_every < 0、または save_every > 0 なのに save_path が None のとき
    """
    if epochs < 1:
        raise ValueError(f"epochs は 1 以上: {epochs}")
    if save_every < 0:
        raise ValueError(f"save_every は 0 以上: {save_every}")
    if save_every > 0 and save_path is None:
        raise ValueError("save_every > 0 には save_path が要る（途中の ckpt の保存先を作るため）")
    torch.manual_seed(seed)
    train_part, val_part = gm.load_split()
    model = DDPMMinimal().to(device)
    diffusion = base.Diffusion(device)
    optimizer = torch.optim.AdamW(lm.param_groups(model, WEIGHT_DECAY), lr=lr)
    train_loader = gm.make_loader(train_part, shuffle=True)
    val_loader = gm.make_loader(val_part, shuffle=False)
    print(f"device={device}  train={len(train_part.sched)}  val={len(val_part.sched)}  base_ch={base.BASE_CH}  "
          f"params={count_params(model):,}", flush=True)

    history: dict[str, list[float]] = {"epoch": [], "train": [], "val": [], "sec": []}
    best_val, best_epoch, no_improve = math.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    ep = 0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, count = 0.0, 0.0
        for cond_idx, sched, _ in train_loader:
            loss = eps_mse(model, diffusion, cond_idx.to(device), sched.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item() * sched.size(0)
            count += sched.size(0)
        va = val_loss(model, diffusion, val_loader)
        sec = time.perf_counter() - t0
        for key, val in (("epoch", ep), ("train", total / count), ("val", va), ("sec", sec)):
            history[key].append(float(val))
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:4d} | train {total / count:.5f} | val {va:.5f} | {sec:.2f}s", flush=True)
        if save_every > 0 and ep % save_every == 0:
            assert save_path is not None
            save_ckpt(model, epoch_ckpt_path(save_path, ep), seed, extra={"epoch": ep, "val": va, "lr": lr})
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
    model.eval()
    print(f"restored best: epoch {best_epoch} (val {best_val:.5f})", flush=True)
    if save_path is not None:
        save_ckpt(model, save_path, seed, extra={"best_epoch": best_epoch, "best_val": best_val,
                                                  "stopped_epoch": ep, "lr": lr, "history": history})
        print(f"saved model to {save_path}")
    return model, history


def epoch_ckpt_path(save_path: Path, epoch: int) -> Path:
    """途中の epoch の ckpt の保存先 {save_path の stem}_ep{epoch:04d}.pt（DDPM_Aggregate_Simple と同じ形）"""
    return save_path.with_name(f"{save_path.stem}_ep{epoch:04d}{save_path.suffix}")


def save_ckpt(model: DDPMMinimal, path: Path, seed: int, extra: dict[str, Any] | None = None) -> None:
    """重みと出所の記録（config）を {"model", "config"} で保存する

    Args:
        model: 保存するモデル
        path: 保存先
        seed: 学習の乱数の種
        extra: config に足す記録（best_epoch・学習曲線など）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"base_ch": base.BASE_CH, "dropout": base.DROPOUT, "weight_decay": WEIGHT_DECAY,
                              "seed": seed}
    if extra is not None:
        config.update(extra)
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_model(path: Path, device: str = DEVICE) -> DDPMMinimal:
    """save_ckpt で保存した ckpt から eval モードのモデルを戻す

    Raises:
        ValueError: ckpt の幅が base.BASE_CH と違うとき（幅の違う ckpt は、先に set_base_ch で幅を合わせる）
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["config"]["base_ch"] != base.BASE_CH:
        raise ValueError(f"{path} の幅 {ckpt['config']['base_ch']} が BASE_CH={base.BASE_CH} と違う")
    model = DDPMMinimal().to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def ckpt_path(seed: int, lr: float = LR, run_tag: str = "") -> Path:
    """最良の ckpt のパス（幅は呼んだ時点の base.BASE_CH）

    例: 種 42 は ddpm_minimal_ch32_wd0.01.pt、種 43 は ddpm_minimal_ch32_wd0.01_s43.pt、
        学習率 2e-4 は ddpm_minimal_ch32_wd0.01_lr0.0002.pt、run_tag="_traj" は ddpm_minimal_ch32_wd0.01_traj.pt、
        set_base_ch(64) の後の種 42 は ddpm_minimal_ch64_wd0.01.pt

    Args:
        seed: 学習の乱数の種
        lr: 学習率, default=LR=1e-3（既定以外は stage1_tag に _lr{lr} が付く）
        run_tag: 本番の ckpt と分けるための印（例 "_traj"）, default=""
    """
    name = f"{MODEL_SAVE_PATH.stem}{stage1_tag(lr=lr)}{run_tag}{lm.run_suffix(seed)}{MODEL_SAVE_PATH.suffix}"
    return MODEL_SAVE_PATH.with_name(name)


# ============================================================
# 5. 生成
# ============================================================
def device_generator(generator: torch.Generator, device: torch.device) -> torch.Generator:
    """CPU の generator から種を 1 つ引き、計算場所の generator を作る（逆過程の雑音用）"""
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator).item())
    return torch.Generator(device=device).manual_seed(seed)


def sample_relaxed_chain(model: DDPMMinimal, diffusion: Any, cond_idx: torch.Tensor, generator: torch.Generator,
                         tau: float, k: int = K_TAIL) -> torch.Tensor:
    """逆過程で 1 日を生成し、末尾 k ステップだけ勾配を通して straight-through の one-hot を返す（Stage 2 用）

    Note:
        1. x_T と各ステップの雑音 z は、generator から作った計算場所の generator で引く
        2. t = 999..k は勾配なし、t = k−1..0 は勾配あり（既存の DDPM の Stage 2 と同じ打ち切り）
        3. 最後の x_0 は clamp された x0 の予測。前向きは argmax の one-hot（評価の生成と同じ）、
           後ろ向きは softmax(x0 / τ) の微分（base.straight_through）

    Args:
        model: DDPMMinimal
        diffusion: base.Diffusion
        cond_idx: 条件インデックス, dtype=int64, (B, 3)。モデルと同じデバイス
        generator: CPU の torch.Generator
        tau: straight-through の softmax の温度
        k: 勾配を保持する末尾のステップ数, default=K_TAIL=1

    Returns:
        y: 勾配の通る one-hot, (B, 12, 96)
    """
    dev = cond_idx.device
    gen = device_generator(generator, dev)
    shape = (cond_idx.size(0), NUM_ACT, NUM_SLOTS)
    with torch.no_grad():
        x = torch.randn(shape, generator=gen, device=dev)                       # x_T
        for ti in reversed(range(k, T_STEPS)):
            x = diffusion._reverse_step(model, x, ti, cond_idx, GUIDANCE, torch.randn(shape, generator=gen, device=dev))
    x = x.detach()
    for ti in reversed(range(k)):
        z = None if ti == 0 else torch.randn(shape, generator=gen, device=dev)
        x = diffusion._reverse_step(model, x, ti, cond_idx, GUIDANCE, z)
    return base.straight_through(x, tau)


@torch.no_grad()
def reverse_chain(model: DDPMMinimal, diffusion: Any, cond_idx: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """逆過程 1,000 ステップを勾配なしで回し、最後の x を返す（group_pool と診断の trajectory_pools.py で共有）

    Note:
        最後の x は ti = 0 の事後平均で、clamp された x0 の予測（値は [0, 1]）。argmax を取る前の連続値

    Args:
        model: DDPMMinimal
        diffusion: base.Diffusion
        cond_idx: 条件インデックス, dtype=int64, (B, 3)。モデルと同じデバイス
        gen: 計算場所の torch.Generator（x_T と各ステップの雑音 z をこの順に引く）

    Returns:
        x: 最後の x, dtype=float32, (B, 12, 96)
    """
    dev = cond_idx.device
    shape = (cond_idx.size(0), NUM_ACT, NUM_SLOTS)
    x = torch.randn(shape, generator=gen, device=dev)
    for ti in reversed(range(T_STEPS)):
        z = None if ti == 0 else torch.randn(shape, generator=gen, device=dev)
        x = diffusion._reverse_step(model, x, ti, cond_idx, GUIDANCE, z)
    return x


@torch.no_grad()
def group_pool(model: DDPMMinimal, diffusion: Any, n_per_group: int, seed: int,
               batch: int = POOL_BATCH) -> npt.NDArray[np.int64]:
    """群別サンプルプールを作る, -> (28, n_per_group, 96)。逆過程 1,000 ステップ → argmax（CFG なし）

    Note:
        雑音は計算場所の generator（種 seed）で引く。同じ計算場所・同じ種なら同じプールになる

    Args:
        model: DDPMMinimal
        diffusion: base.Diffusion
        n_per_group: 群あたりの本数
        seed: 生成の乱数の種
        batch: 1 回に生成する本数, default=POOL_BATCH=8192

    Returns:
        群別サンプルプール, dtype=int64, (D_GROUPS, n_per_group, NUM_SLOTS)
    """
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    flat = torch.as_tensor(sm.cond_grid(), dtype=torch.long).repeat_interleave(n_per_group, dim=0)
    outs = []
    for i in range(0, flat.size(0), batch):
        cond = flat[i:i + batch].to(dev)
        outs.append(reverse_chain(model, diffusion, cond, gen).argmax(dim=1).cpu())
    return torch.cat(outs).numpy().astype(np.int64).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


# ============================================================
# 6. CLI
# ============================================================
def smoke(device: str = DEVICE) -> None:
    """2 epoch 学習し、生成と微分できる生成の形を確かめる（逆過程は短くせず、本数を減らす。何も保存しない）"""
    model, history = train(epochs=2, save_path=None, device=device)
    assert len(history["val"]) == 2 and all(math.isfinite(v) for v in history["val"])
    diffusion = base.Diffusion(device)
    cond = torch.as_tensor(sm.cond_grid(), dtype=torch.long)[:2].to(device)
    y = sample_relaxed_chain(model, diffusion, cond, torch.Generator().manual_seed(0), 1.0)
    assert y.shape == (2, NUM_ACT, NUM_SLOTS)
    pool = group_pool(model, diffusion, 1, seed=0)
    assert pool.shape == (D_GROUPS, 1, NUM_SLOTS) and 0 <= pool.min() and pool.max() < NUM_ACT
    print(f"smoke: OK (params={count_params(model):,})")


def measure(device: str = DEVICE) -> None:
    """SQUID の見積もり用に、Stage 1 の 1 epoch・Stage 2 の 1 更新・評価のプールの所要時間を測る（学習結果は捨てる）"""
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    model, history = train(epochs=3, save_path=None, device=device)
    epoch_sec = float(np.median(history["sec"]))
    diffusion = base.Diffusion(device)
    cond = torch.as_tensor(sm.cond_grid(), dtype=torch.long)[:7].repeat_interleave(256, dim=0).to(device)
    gen = torch.Generator().manual_seed(0)
    times = []
    for _ in range(2):
        sync()
        t0 = time.perf_counter()
        y = sample_relaxed_chain(model, diffusion, cond, gen, 1.0)
        (y * torch.randn_like(y)).sum().backward()
        sync()
        times.append(time.perf_counter() - t0)
    update_sec = min(times)
    sync()
    t0 = time.perf_counter()
    group_pool(model, diffusion, POOL_BATCH // D_GROUPS, seed=0)
    sync()
    pool_sec = (time.perf_counter() - t0) * (2000 * D_GROUPS) / (POOL_BATCH // D_GROUPS * D_GROUPS)
    print(f"[measure] device={device}  base_ch={base.BASE_CH}  params={count_params(model):,}  "
          f"Stage 1: 1 epoch {epoch_sec:.2f} 秒（1,000 epoch で {epoch_sec * 1000 / 3600:.2f} 時間）")
    print(f"[measure] Stage 2: 1 更新 {update_sec:.1f} 秒（1,000 更新で {update_sec * 1000 / 3600:.2f} 時間。リハーサルと val は含まない）")
    print(f"[measure] 評価のプール 56,000 本: {pool_sec / 60:.1f} 分（E0・E1 で 2 回）")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="DDPM_Minimal: CFG なしの DDPM の Stage 1（幅の既定は 32）")
    ap.add_argument("--base-ch", type=int, default=BASE_CH, help="denoiser の幅。保存名の _ch{幅} になる")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default=DEVICE)
    ap.add_argument("--threads", type=int, default=0, help="CPU の計算のスレッド数。0 なら torch の既定")
    ap.add_argument("--lr", type=float, default=LR, help="AdamW の学習率。既定以外は保存名に _lr{学習率} が付く")
    ap.add_argument("--save-every", type=int, default=0,
                    help="N epoch ごとに途中の ckpt を {保存名の stem}_ep{epoch:04d}.pt へ保存する（診断用。0 なら保存しない）")
    ap.add_argument("--run-tag", default="", help="保存名に足す印（例 _traj）。本番の ckpt を上書きしないために使う")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure", action="store_true", help="所要時間だけを測る（何も保存しない）")
    args = ap.parse_args()
    set_base_ch(args.base_ch)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.smoke:
        smoke(args.device)
        return
    if args.measure:
        measure(args.device)
        return
    path = ckpt_path(args.seed, lr=args.lr, run_tag=args.run_tag)
    print(f"[config] base_ch={base.BASE_CH} seed={args.seed} epochs={args.epochs} device={args.device} "
          f"lr={args.lr:g} save_every={args.save_every} ckpt={path.name}")
    train(epochs=args.epochs, seed=args.seed, save_path=path, device=args.device, lr=args.lr,
          save_every=args.save_every)


if __name__ == "__main__":
    main()
