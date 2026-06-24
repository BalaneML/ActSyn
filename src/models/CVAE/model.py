"""
model.py
================
活動スケジュール生成のためのConditional VAE (ベースライン)

設計方針:
    - スケジュール表現  : 96スロット × 10状態の固定グリッドを one-hot 平坦化（960次元）
    - 条件表現: 5属性を one-hot 連結
    - encoder/decoder: 2層 MLP
    - decoder: スロットごとに独立な softmax
    - 損失: 96スロットの cross-entropy + β*KL
    - 補正なし: class weight / 遷移損失 / KL warmup / dropout は入れない

このファイル単体で実行すると、weighted_dataset.csv で学習し、
学習後に生成サニティチェックを行う。
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wandb

# ============================================================
# 1. 設定（ハイパーパラメータ）
# ============================================================
DATA_PATH   = '../../../data/processed/weighted_dataset.csv'
MODEL_SAVE_PATH = './cvae.pt'

# 活動スケジュール
NUM_SLOTS   = 96      # 15分刻み × 96 = 24時間（04:00開始）
NUM_ACT     = 10      # 活動状態 HOME=0 ... TRAVEL=9
X_DIM       = NUM_SLOTS * NUM_ACT   # 960

# 条件5属性のカテゴリ数（one-hot の次元）
AGE_BINS    = 11      # min(age // 10, 10) で 0..10 の11区分
NUM_GENDER  = 2
NUM_MEMBER  = 10      # 1..10 を 0..9 にシフト
NUM_ROLE    = 8       # role_household_type 0..7
NUM_WORKER  = 2
COND_DIM    = AGE_BINS + NUM_GENDER + NUM_MEMBER + NUM_ROLE + NUM_WORKER  # = 33

HIDDEN_DIM  = 512
Z_DIM       = 64
BETA        = 0.5

# 学習
BATCH_SIZE  = 1024
EPOCHS      = 500
LR          = 1e-3
VAL_RATIO   = 0.1
SEED        = 42
USE_WEIGHTED_SAMPLER = True   # WTPERFIN による重み付きサンプリング

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# ============================================================
# 2. データ整形
# ============================================================
def load_data(path: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """CSV を読み、条件インデックス・活動グリッド・重みを返す。"""
    df = pd.read_csv(path)

    # 条件属性を 0始まりのカテゴリインデックスに変換
    age    = np.minimum(df["age"].to_numpy() // 10, AGE_BINS - 1)   # 0..10
    gender = df["gender"].to_numpy()                                # 0,1
    member = df["num_member"].to_numpy() - 1                        # 1..10 -> 0..9
    role   = df["role_household_type"].to_numpy()                   # 0..7
    worker = df["worker_status"].to_numpy()                         # 0,1
    cond_idx = np.stack([age, gender, member, role, worker], axis=1).astype(np.int64)

    # 活動グリッド (N, 96)、各値 0..9
    scols = [f"s{i}" for i in range(NUM_SLOTS)]
    schedules = df[scols].to_numpy().astype(np.int64)

    # 重み
    weights = df["WTPERFIN"].to_numpy().astype(np.float64)
    return cond_idx, schedules, weights


def cond_to_onehot(cond_idx: torch.Tensor) -> torch.Tensor:
    """条件インデックス (B,5) を one-hot 連結 (B, COND_DIM) に変換。"""
    age, gender, member, role, worker = cond_idx.unbind(dim=1)
    parts = [
        F.one_hot(age,    AGE_BINS),
        F.one_hot(gender, NUM_GENDER),
        F.one_hot(member, NUM_MEMBER),
        F.one_hot(role,   NUM_ROLE),
        F.one_hot(worker, NUM_WORKER),
    ]
    return torch.cat(parts, dim=1).float()


class ScheduleDataset(Dataset):
    def __init__(self, cond_idx, schedules):
        self.cond_idx = torch.as_tensor(cond_idx, dtype=torch.long)
        self.schedules = torch.as_tensor(schedules,    dtype=torch.long)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, i):
        return self.cond_idx[i], self.schedules[i]


# ============================================================
# 3. モデル（最小構成の CVAE）
# ============================================================
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: [x_onehot(960) + cond(33)] -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(X_DIM + COND_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(HIDDEN_DIM, Z_DIM)  # μ
        self.fc_logvar = nn.Linear(HIDDEN_DIM, Z_DIM)  # logσ^2

        # decoder: [z(32) + cond(33)] -> hidden -> 96*10 ロジット
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM + COND_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, X_DIM),
        )

    # エンコーダー
    def encode(self, x_onehot, cond):
        h = self.encoder(torch.cat([x_onehot, cond], dim=1))  # [x_onehot(960) + cond(33)] -> [h(256)]
        return self.fc_mu(h), self.fc_logvar(h)  # [h(256)] -> [μ(32)], [h(256)] -> [logσ^2(32)]

    # 再パラメータ化トリック
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # logσ² → σ
        eps = torch.randn_like(std)  # N(0, I) からサンプリング:（ε ～ N(0, I)）
        return mu + eps * std  # z = μ + σ·ε

    # デコーダー
    def decode(self, z, cond):
        logits = self.decoder(torch.cat([z, cond], dim=1))  # [z(32) + cond(33)] -> hidden -> [logits(96*10)]
        return logits.view(-1, NUM_SLOTS, NUM_ACT)   # (B, 96, 10)

    def forward(self, sched, cond):
        x_onehot = F.one_hot(sched, NUM_ACT).float().view(sched.size(0), -1)
        mu, logvar = self.encode(x_onehot, cond)  # encoder input: [act, cond]
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, cond)  # decoder input: [z, cond]
        return logits, mu, logvar


# ============================================================
# 4. 損失
# ============================================================
def cvae_loss(logits, sched, mu, logvar, beta=BETA):
    # 再構成: 96スロットの cross-entropy（1サンプルあたり96スロット合計、バッチ平均）
    recon = F.cross_entropy(
        logits.reshape(-1, NUM_ACT), sched.reshape(-1), reduction="sum"
    ) / sched.size(0)
    # 正則化: KL( q(z|x,y) || N(0,I) )（バッチ平均）
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / sched.size(0)
    return recon + beta * kl, recon, kl


# ============================================================
# 5. 学習
# ============================================================
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


def run_epoch(model, loader, optimizer=None):
    """
    1エポック分の学習または評価を実行し、平均損失を返す。
    optimizer を渡すと学習モード、None だと評価モードになる。
    """
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    sum_loss, sum_recon, sum_kl, n_samples = 0.0, 0.0, 0.0, 0

    # 学習時のみ勾配を有効化
    with torch.set_grad_enabled(is_train):
        for cond_idx, sched in loader:
            cond_idx = cond_idx.to(DEVICE)
            sched    = sched.to(DEVICE)
            cond     = cond_to_onehot(cond_idx)

            logits, mu, logvar = model(sched, cond)
            loss, recon, kl = cvae_loss(logits, sched, mu, logvar)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # バッチ平均損失 × バッチサイズ で「合計」に戻して累積
            batch_size = sched.size(0)
            sum_loss  += loss.item()  * batch_size
            sum_recon += recon.item() * batch_size
            sum_kl    += kl.item()    * batch_size
            n_samples += batch_size

    # サンプル数で割ってエポック平均にする
    return sum_loss / n_samples, sum_recon / n_samples, sum_kl / n_samples


def train():
    wandb.init(
        project='domain-transfer-cvae',
        config={
            "z_dim": Z_DIM,
            "hidden_dim": HIDDEN_DIM,
            "beta": BETA,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "weighted_sampler": USE_WEIGHTED_SAMPLER,
        }
    )

    torch.manual_seed(SEED)
    cond_idx, sched, weight = load_data(DATA_PATH)
    train_loader, val_loader = make_loaders(cond_idx, sched, weight)

    model = CVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"device={DEVICE}  params={sum(p.numel() for p in model.parameters()):,}")

    # 学習ループ
    for ep in range(1, EPOCHS + 1):
        tr, tr_recon, tr_kl = run_epoch(model, train_loader, optimizer)
        va, va_recon, va_kl = run_epoch(model, val_loader)
        print(f"epoch {ep:3d} | train {tr:7.3f} (recon {tr_recon:6.3f}, kl {tr_kl:6.3f}) "
                f"| val {va:7.3f} (recon {va_recon:6.3f}, kl {va_kl:6.3f})")
        wandb.log({
            "epoch": ep,
            "train/loss":  tr, "train/recon": tr_recon, "train/kl": tr_kl,
            "val/loss":    va, "val/recon":   va_recon, "val/kl":   va_kl,
        })
        
    # 保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"saved model to {MODEL_SAVE_PATH}")
    wandb.save(str(MODEL_SAVE_PATH))
    wandb.finish()
    return model


# ============================================================
# 6. 生成サニティチェック
# ============================================================
@torch.no_grad()
def generate(model, cond_idx, sample=True):
    """条件インデックス (M,5) を与えてスケジュール (M,96) を生成。
    sample=True: スロットごとに確率サンプリング / False: argmax"""
    model.eval()
    cond = cond_to_onehot(torch.as_tensor(cond_idx, dtype=torch.long).to(DEVICE))
    z = torch.randn(cond.size(0), Z_DIM, device=DEVICE)
    logits = model.decode(z, cond)                 # (M, 96, 10)
    probs = F.softmax(logits, dim=-1)
    if sample:
        m = torch.distributions.Categorical(probs=probs)
        return m.sample().cpu().numpy()            # (M, 96)
    return probs.argmax(dim=-1).cpu().numpy()


def sanity_check(model):
    cond_idx, sched, _ = load_data(DATA_PATH)
    real_home = (sched == 0).mean()

    gen_s = generate(model, cond_idx, sample=True)    # 各個人の条件で1本ずつ生成
    gen_a = generate(model, cond_idx, sample=False)

    def n_switches(arr):  # 1日あたりの状態遷移回数（連続性の粗い指標）
        return (arr[:, 1:] != arr[:, :-1]).sum(axis=1).mean()

    print("\n--- 生成サニティチェック ---")
    print(f"値域(sample): {gen_s.min()}..{gen_s.max()}  形状: {gen_s.shape}")
    print(f"HOME比率   real={real_home:.3f}  gen(sample)={(gen_s==0).mean():.3f}  "
            f"gen(argmax)={(gen_a==0).mean():.3f}")
    print(f"平均遷移回数 real={n_switches(sched):.2f}  "
            f"gen(sample)={n_switches(gen_s):.2f}  gen(argmax)={n_switches(gen_a):.2f}")


if __name__ == "__main__":
    model = train()
    sanity_check(model)