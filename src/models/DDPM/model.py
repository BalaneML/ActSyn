"""
model.py
================
活動スケジュール生成のための条件付きDDPM (ATUS 2024 time-use diary)

設計方針:
    - スケジュール表現: 
        96スロット × 17活動 (ATUS Lexicon 第1層) の one-hot を
        そのまま連続値として拡散 (x0 = 2*onehot - 1 ∈ {-1,+1}, 形状 (B,17,96))
      * Tang, Lu & Feng (2025, arXiv:2508.09164) の「離散one-hot→連続表現→
        Gaussian DDPM→argmaxデコード」の枠組み。
        K=17と小さいため学習可能
        埋め込みは使わない (入力Conv1dが実質の埋め込み層。将来 nn.Embedding +
        un-embedding head に差し替える場合は in_conv/out_conv を置換する)
    - denoiser: 
        小型 1D-UNet (時間軸96を 1D畳み込み + self-attention で処理)
    - 条件: 
        COND_SPEC で定義した属性を nn.Embedding し timestep埋め込みに加算注入
        属性の追加・削除は COND_SPEC の1箇所の編集で完結する
    - Classifier-Free Guidance:
        学習時に条件を確率 P_UNCOND で null埋め込みに
        置換し、サンプリング時に guidance scale w で誘導 (w=1.0 で無誘導と等価)
    - 損失: 
        標準の ε 予測 MSE (Ho et al. 2020)
    - サンプリング: 
        ancestral DDPM。毎ステップ x0_hat を clamp(-1,1) で安定化
        (DDIM による高速化は将来項目; バッファは共用可能)
    - EMA (decay 0.999) の重みを評価・保存に使用

このファイル単体で実行すると、atus2024_weighted_dataset.csv で学習し、
学習後に生成サニティチェックを行う。
"""
import copy
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wandb

# ============================================================
# 1. 設定（ハイパーパラメータ）
# ============================================================
# 実行ディレクトリに依存しないよう、スクリプト位置からリポジトリルートを解決
REPO_ROOT   = Path(__file__).resolve().parents[3]   # src/models/DDPM -> repo root
DATA_PATH   = REPO_ROOT / 'data' / 'processed' / 'atus2024' / 'atus2024_weighted_dataset.csv'
MODEL_SAVE_PATH = REPO_ROOT / 'outputs' / 'checkpoints' / 'ddpm_atus.pt'
GEN_SAVE_PATH   = REPO_ROOT / 'outputs' / 'generated' / 'ddpm_atus_schedules.csv'

# 活動スケジュール
NUM_SLOTS   = 96      # 15分刻み × 96 = 24時間（04:00開始）
NUM_ACT     = 17      # ATUS tier1: 01..16 + 18(Traveling)
ACT_NAMES   = [
                "PERSONAL_CARE",
                "HOUSEHOLD",
                "CARE_HH_MEMBERS",
                "CARE_NONHH",
                "WORK",
                "EDUCATION",
                "CONSUMER_PURCHASES",
                "PROF_PERSONAL_SVC",
                "HOUSEHOLD_SVC",
                "GOVERNMENT_SVC",
                "EATING_DRINKING",
                "SOCIAL_LEISURE",
                "SPORTS_EXERCISE",
                "RELIGIOUS",
                "VOLUNTEER",
                "TELEPHONE",
                "TRAVEL",
]
# TODO: learnable embedding に差し替える
IN_CH       = NUM_ACT  # 拡散空間のチャネル数

# 条件属性: (CSV列名, カテゴリ数, 埋め込み次元, インデックス化関数)
# 曜日条件を追加する例: ("day_of_week", 7, 4, lambda v: v - 1)
COND_SPEC = [
    ("age",    11, 8, lambda v: np.minimum(v // 10, 10)),  # 10歳刻み 0..10
    ("gender",  2, 4, lambda v: v),                        # 0=male, 1=female
]
COND_COLS  = [name for name, _, _, _ in COND_SPEC]
COND_CARD  = [card for _, card, _, _ in COND_SPEC]
EMB_DIMS   = [dim for _, _, dim, _ in COND_SPEC]

# データフィルタ: None=全日 / 'weekday'=月-金 / 'weekend'=土日 (day_of_week 1=Sun..7=Sat)
DAY_FILTER  = None

# DDPM
T_STEPS     = 1000
BETA_START  = 1e-4
BETA_END    = 0.02    # linear schedule (Ho et al. 2020 / Tang et al. 2025 準拠)

# Denoiser (1D-UNet)
BASE_CH     = 64
DROPOUT     = 0.1     # 小データ(~7.4k)の過学習対策
ATTN_HEADS  = 4
TIME_EMB_DIM = 256    # timestep/条件埋め込みの共通次元

# Classifier-Free Guidance
P_UNCOND       = 0.1   # 学習時に条件を null に落とす確率
GUIDANCE_SCALE = 2.0   # サンプリング時の誘導強度 (1.0 で無誘導と等価)

# 学習
BATCH_SIZE  = 256
EPOCHS      = 1500    # ~7.4k×0.9/256 ≈ 27 step/epoch → 約40k steps
LR          = 2e-4    # Ho et al. 2020 の標準値。学習が不安定なら 1e-4 に下げる
EMA_DECAY   = 0.999
VAL_RATIO   = 0.1
SEED        = 42
USE_WEIGHTED_SAMPLER = True   # TUFINLWGT による重み付きサンプリング
WEIGHT_COL  = "TUFINLWGT"
# Early stopping: val ε-MSE は t/ε の再サンプリングでエポック間ノイズが大きいため
# patience は長めに取る (0 以下で無効化)
EARLY_STOP_PATIENCE  = 100   # val loss が改善しないまま許容するエポック数
EARLY_STOP_MIN_DELTA = 1e-4  # これ未満の改善は「改善なし」とみなす
GEN_BATCH   = 1024    # 生成時のチャンクサイズ (T_STEPS回のforwardを回すため分割)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


# ============================================================
# 2. データ整形
# ============================================================
def load_data(path: str | Path) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64],
                                          npt.NDArray[np.float64], pd.DataFrame]:
    """CSV を読み、条件インデックス・活動グリッド・重み・生条件列を返す。"""
    df = pd.read_csv(path)

    if DAY_FILTER == 'weekday':
        df = df[~df["day_of_week"].isin([1, 7])].reset_index(drop=True)
    elif DAY_FILTER == 'weekend':
        df = df[df["day_of_week"].isin([1, 7])].reset_index(drop=True)

    # 条件属性を 0始まりのカテゴリインデックスに変換 (COND_SPEC から導出)
    cond_idx = np.stack(
        [fn(df[name].to_numpy()) for name, _, _, fn in COND_SPEC], axis=1
    ).astype(np.int64)

    # 活動グリッド (N, 96)、各値 0..16
    scols = [f"s{i}" for i in range(NUM_SLOTS)]
    schedules = df[scols].to_numpy().astype(np.int64)

    weights = df[WEIGHT_COL].to_numpy().astype(np.float64)
    return cond_idx, schedules, weights, df[COND_COLS]


class ScheduleDataset(Dataset):
    def __init__(self, cond_idx, schedules):
        self.cond_idx  = torch.as_tensor(cond_idx,  dtype=torch.long)
        self.schedules = torch.as_tensor(schedules, dtype=torch.long)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, i):
        return self.cond_idx[i], self.schedules[i]


def make_loaders(cond_idx, sched, weight):
    g = torch.Generator().manual_seed(SEED)
    n = len(sched)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * VAL_RATIO)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = ScheduleDataset(cond_idx[train_idx], sched[train_idx])
    val_ds   = ScheduleDataset(cond_idx[val_idx],   sched[val_idx])

    if USE_WEIGHTED_SAMPLER:
        w = torch.as_tensor(weight[train_idx], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)  # type: ignore
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def sched_to_x0(sched: torch.Tensor) -> torch.Tensor:
    """スケジュール (B,96) int -> 拡散空間 (B,17,96) ∈ {-1,+1}"""
    onehot = F.one_hot(sched, NUM_ACT).float().permute(0, 2, 1)  # (B,17,96)
    return 2.0 * onehot - 1.0


# ============================================================
# 3. Denoiser（1D-UNet）
# ============================================================
def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """sinusoidal timestep embedding (B,) -> (B, dim)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock1D(nn.Module):
    """GroupNorm→SiLU→Conv1d ×2 + timestep/条件埋め込みの加算注入 + skip"""
    def __init__(self, c_in: int, c_out: int, emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock1D(nn.Module):
    """時間軸(スロット間)の self-attention + residual"""
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, ATTN_HEADS, batch_first=True)

    def forward(self, x):
        h = self.norm(x).permute(0, 2, 1)          # (B,C,L) -> (B,L,C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1)


class UNet1D(nn.Module):
    """ε予測ネットワーク: (B,17,96) + t + cond -> (B,17,96)
    解像度 96 -> 48 -> 24、チャネル 64 -> 128 -> 128 の小型UNet (~2Mパラメータ)
    ※ val loss が発散する場合はダウンサンプル無しの ResConv×6 + Attn×2 (C=96) に縮小する
    """
    def __init__(self):
        super().__init__()
        c1, c2 = BASE_CH, BASE_CH * 2   # 64, 128

        # timestep埋め込み: sinusoidal(128) -> MLP -> 256
        self.time_mlp = nn.Sequential(
            nn.Linear(128, TIME_EMB_DIM), nn.SiLU(),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
        )
        # 条件埋め込み: 属性ごとに Embedding し連結 -> 256 (CVAE_Embedding と同パターン)
        self.cond_embeds = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(COND_CARD, EMB_DIMS)
        ])
        self.cond_proj = nn.Linear(sum(EMB_DIMS), TIME_EMB_DIM)
        self.null_emb = nn.Parameter(torch.zeros(TIME_EMB_DIM))  # CFG の無条件埋め込み

        self.in_conv = nn.Conv1d(IN_CH, c1, 3, padding=1)
        # down: L96@64 -> L48@128 -> L24@128
        self.d1a, self.d1b = ResBlock1D(c1, c1), ResBlock1D(c1, c1)
        self.ds1 = nn.Conv1d(c1, c1, 3, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock1D(c1, c2), ResBlock1D(c2, c2)
        self.attn2 = AttnBlock1D(c2)
        self.ds2 = nn.Conv1d(c2, c2, 3, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock1D(c2, c2), ResBlock1D(c2, c2)
        self.attn3 = AttnBlock1D(c2)
        # mid
        self.m1, self.m_attn, self.m2 = ResBlock1D(c2, c2), AttnBlock1D(c2), ResBlock1D(c2, c2)
        # up: skip-concat ミラー
        self.u3 = ResBlock1D(c2 + c2, c2)
        self.us2 = nn.Conv1d(c2, c2, 3, padding=1)   # nearest upsample 後の整形
        self.u2 = ResBlock1D(c2 + c2, c2)
        self.u2_attn = AttnBlock1D(c2)
        self.us1 = nn.Conv1d(c2, c1, 3, padding=1)
        self.u1 = ResBlock1D(c1 + c1, c1)
        # out (最終convはzero-init: 学習初期の出力を0にして安定化)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv1d(c1, IN_CH, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        out_bias = self.out_conv.bias
        if out_bias is not None:
            nn.init.zeros_(out_bias)

    # 条件インデックス (B,K) -> 埋め込み (B, TIME_EMB_DIM)。None なら null埋め込み
    def embed_cond(self, cond_idx, batch: int, drop_mask=None):
        if cond_idx is None:
            return self.null_emb.expand(batch, -1)
        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)
        c = self.cond_proj(c)
        if drop_mask is not None:  # 学習時のCFG dropout: mask行を null に置換
            c = torch.where(drop_mask[:, None], self.null_emb.expand_as(c), c)
        return c

    def forward(self, x_t, t, cond_idx=None, drop_mask=None):
        emb = self.time_mlp(timestep_embedding(t)) \
            + self.embed_cond(cond_idx, x_t.size(0), drop_mask)

        h1 = self.d1b(self.d1a(self.in_conv(x_t), emb), emb)        # (B,64,96)
        h2 = self.attn2(self.d2b(self.d2a(self.ds1(h1), emb), emb)) # (B,128,48)
        h3 = self.attn3(self.d3b(self.d3a(self.ds2(h2), emb), emb)) # (B,128,24)

        m = self.m2(self.m_attn(self.m1(h3, emb)), emb)             # (B,128,24)

        u = self.u3(torch.cat([m, h3], dim=1), emb)                 # (B,128,24)
        u = self.us2(F.interpolate(u, scale_factor=2, mode='nearest'))
        u = self.u2_attn(self.u2(torch.cat([u, h2], dim=1), emb))   # (B,128,48)
        u = self.us1(F.interpolate(u, scale_factor=2, mode='nearest'))
        u = self.u1(torch.cat([u, h1], dim=1), emb)                 # (B,64,96)
        return self.out_conv(F.silu(self.out_norm(u)))              # (B,17,96)


# ============================================================
# 4. Diffusion（forward過程・損失・サンプリング）
# ============================================================
class Diffusion:
    """β schedule と派生バッファを事前計算し、q_sample / loss / sample を提供"""
    def __init__(self, device=DEVICE):
        betas = torch.linspace(BETA_START, BETA_END, T_STEPS, device=device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)                          # ᾱ_t
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.sqrt_acp = acp.sqrt()
        self.sqrt_1m_acp = (1.0 - acp).sqrt()
        # posterior q(x_{t-1} | x_t, x_0)
        self.post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.post_coef_x0 = betas * acp_prev.sqrt() / (1.0 - acp)
        self.post_coef_xt = (1.0 - acp_prev) * alphas.sqrt() / (1.0 - acp)

    def q_sample(self, x0, t, eps):
        """x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε"""
        return (self.sqrt_acp[t][:, None, None] * x0
                + self.sqrt_1m_acp[t][:, None, None] * eps)

    def loss(self, model, sched, cond_idx):
        """標準 ε 予測 MSE + CFG 条件dropout"""
        x0 = sched_to_x0(sched)
        t = torch.randint(0, T_STEPS, (x0.size(0),), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        drop_mask = torch.rand(x0.size(0), device=x0.device) < P_UNCOND
        eps_hat = model(x_t, t, cond_idx, drop_mask)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, model, cond_idx, guidance_scale=GUIDANCE_SCALE, verbose=False):
        """ancestral DDPM + CFG。cond_idx (M,K) -> スケジュール (M,96) int
        毎ステップ x0_hat を clamp(-1,1) して安定化し、最後に channel argmax でデコード"""
        model.eval()
        m = cond_idx.size(0)
        x = torch.randn(m, IN_CH, NUM_SLOTS, device=cond_idx.device)
        for ti in reversed(range(T_STEPS)):
            t = torch.full((m,), ti, device=x.device, dtype=torch.long)
            eps_c = model(x, t, cond_idx)
            if guidance_scale != 1.0:
                eps_u = model(x, t, None)   # null埋め込みで無条件予測
                eps_hat = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps_hat = eps_c
            x0_hat = (x - self.sqrt_1m_acp[ti] * eps_hat) / self.sqrt_acp[ti]
            x0_hat.clamp_(-1.0, 1.0)
            mean = self.post_coef_x0[ti] * x0_hat + self.post_coef_xt[ti] * x
            if ti > 0:
                x = mean + self.post_var[ti].sqrt() * torch.randn_like(x)
            else:
                x = mean
            if verbose and ti % 200 == 0:
                print(f"  sampling t={ti}")
        return x.argmax(dim=1)   # (M,96)


# ============================================================
# 5. EMA（指数移動平均; DDPM標準プラクティス）
# ============================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ============================================================
# 6. 学習
# ============================================================
def run_epoch(model, diffusion, loader, optimizer=None, ema=None):
    """1エポック分の学習または評価を実行し、平均 ε-MSE を返す。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    sum_loss, n_samples = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for cond_idx, sched in loader:
            cond_idx = cond_idx.to(DEVICE)
            sched    = sched.to(DEVICE)
            loss = diffusion.loss(model, sched, cond_idx)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            bs = sched.size(0)
            sum_loss += loss.item() * bs
            n_samples += bs
    return sum_loss / n_samples


def train():
    run = wandb.init(
        project='domain-transfer-ddpm',
        config={
            "t_steps": T_STEPS, "beta_start": BETA_START, "beta_end": BETA_END,
            "base_ch": BASE_CH, "dropout": DROPOUT, "time_emb_dim": TIME_EMB_DIM,
            "cond_spec": [(n, c, d) for n, c, d, _ in COND_SPEC],
            "p_uncond": P_UNCOND, "guidance_scale": GUIDANCE_SCALE,
            "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
            "ema_decay": EMA_DECAY, "weighted_sampler": USE_WEIGHTED_SAMPLER,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "day_filter": DAY_FILTER, "data": DATA_PATH.name,
        }
    )

    torch.manual_seed(SEED)
    cond_idx, sched, weight, _ = load_data(DATA_PATH)
    train_loader, val_loader = make_loaders(cond_idx, sched, weight)

    model = UNet1D().to(DEVICE)
    diffusion = Diffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    ema = EMA(model)
    print(f"device={DEVICE}  params={sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    best_state = None   # ベスト時点の {model, ema, epoch} スナップショット
    epochs_no_improve = 0
    ep = 0
    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(model, diffusion, train_loader, optimizer, ema)
        va = run_epoch(model, diffusion, val_loader)
        print(f"epoch {ep:4d} | train {tr:.4f} | val {va:.4f}")
        wandb.log({"epoch": ep, "train/loss": tr, "val/loss": va})

        if va < best_val - EARLY_STOP_MIN_DELTA:
            best_val = va
            epochs_no_improve = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "ema": copy.deepcopy(ema.shadow),
                "epoch": ep,
            }
        else:
            epochs_no_improve += 1
            if 0 < EARLY_STOP_PATIENCE <= epochs_no_improve:
                best_ep = best_state["epoch"] if best_state else ep
                print(f"early stopping at epoch {ep} "
                      f"(no val improvement for {epochs_no_improve} epochs; "
                      f"best {best_val:.4f} @ epoch {best_ep})")
                break

    # ベスト時点の重みへ巻き戻してから保存・評価する
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        ema.shadow = best_state["ema"]
        print(f"restored best checkpoint: epoch {best_state['epoch']} (val {best_val:.4f})")

    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.summary["best_epoch"] = best_state["epoch"] if best_state else ep
        run.summary["stopped_epoch"] = ep

    # 生の重みと EMA 重みの両方を保存し、評価には EMA を使用
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": ema.shadow}, MODEL_SAVE_PATH)
    print(f"saved model (raw + ema) to {MODEL_SAVE_PATH}")
    wandb.save(str(MODEL_SAVE_PATH))
    wandb.finish()

    ema.copy_to(model)   # 以降 (生成・評価) は EMA 重み
    return model


# ============================================================
# 7. 生成 + サニティチェック
# ============================================================
@torch.no_grad()
def generate(model, cond_idx, guidance_scale=GUIDANCE_SCALE) -> np.ndarray:
    """条件インデックス (M,K) からスケジュール (M,96) を生成 (チャンク分割)"""
    diffusion = Diffusion()
    ci = torch.as_tensor(cond_idx, dtype=torch.long).to(DEVICE)
    outs = []
    for i in range(0, len(ci), GEN_BATCH):
        outs.append(diffusion.sample(model, ci[i:i + GEN_BATCH], guidance_scale).cpu().numpy())
        print(f"  generated {min(i + GEN_BATCH, len(ci))}/{len(ci)}")
    return np.concatenate(outs, axis=0)


def participation_curves(sched: np.ndarray) -> np.ndarray:
    """(N,96) -> (17,96) スロット別行為者率 (各活動 a について P(act=a | slot))"""
    curves = np.stack([(sched == a).mean(axis=0) for a in range(NUM_ACT)])
    return curves


def sanity_check(model):
    cond_idx, sched, _, cond_raw = load_data(DATA_PATH)

    gen = generate(model, cond_idx)   # 実データ全員の条件で1本ずつ生成

    # 生成CSV保存 (条件列 + s0..s95; eval notebook がそのまま読める形)
    GEN_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(
        [cond_raw.reset_index(drop=True),
         pd.DataFrame(gen, columns=[f"s{i}" for i in range(NUM_SLOTS)])],
        axis=1,
    )
    out.to_csv(GEN_SAVE_PATH, index=False)
    print(f"saved generated schedules to {GEN_SAVE_PATH}")

    def n_switches(arr):  # 1日あたりの状態遷移回数（連続性の粗い指標）
        return (arr[:, 1:] != arr[:, :-1]).sum(axis=1).mean()

    print("\n--- 生成サニティチェック ---")
    print(f"値域: {gen.min()}..{gen.max()}  形状: {gen.shape}")
    print(f"平均遷移回数 real={n_switches(sched):.2f}  gen={n_switches(gen):.2f}")

    # 夜間帯 (22:00-04:00 = s72..s95) の PERSONAL_CARE(睡眠含む) 比率
    night = slice(72, 96)
    print(f"夜間PERSONAL_CARE比率 real={(sched[:, night]==0).mean():.3f}  "
          f"gen={(gen[:, night]==0).mean():.3f}")

    # 17クラス時間シェア + 参加率曲線の JSD (base2; eval notebook と同じ流儀)
    real_share = np.bincount(sched.ravel(), minlength=NUM_ACT) / sched.size
    gen_share  = np.bincount(gen.ravel(),  minlength=NUM_ACT) / gen.size
    real_c, gen_c = participation_curves(sched), participation_curves(gen)
    print(f"\n{'activity':20s} {'real':>7s} {'gen':>7s} {'timecurve_JSD':>14s}")
    for a in np.argsort(-real_share):
        p, q = real_c[a], gen_c[a]
        jsd = jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2 if p.sum() > 0 and q.sum() > 0 else np.nan
        print(f"{ACT_NAMES[a]:20s} {real_share[a]:7.1%} {gen_share[a]:7.1%} {jsd:14.4f}")


if __name__ == "__main__":
    # 形状スモークテスト (学習前に必ず通す)
    _m = UNet1D().to(DEVICE)
    _d = Diffusion()
    _x = torch.randn(4, IN_CH, NUM_SLOTS, device=DEVICE)
    _t = torch.randint(0, T_STEPS, (4,), device=DEVICE)
    _c = torch.zeros(4, len(COND_SPEC), dtype=torch.long, device=DEVICE)
    assert _m(_x, _t, _c).shape == _x.shape
    assert _m(_x, _t, None).shape == _x.shape          # 無条件 (CFG) 経路
    _x0 = sched_to_x0(torch.zeros(4, NUM_SLOTS, dtype=torch.long, device=DEVICE))
    _t0 = torch.zeros(4, dtype=torch.long, device=DEVICE)
    assert (_d.q_sample(_x0, _t0, torch.zeros_like(_x0)) - _x0).abs().max() < 1e-4  # t=0でほぼ恒等
    del _m, _d, _x, _t, _c, _x0, _t0
    print("shape smoke test: OK")

    model = train()
    sanity_check(model)
