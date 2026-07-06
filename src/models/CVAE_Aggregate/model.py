"""
model.py
================
集計マッチ学習のための CVAE（AggCVAE）

CVAE / CVAE_Embedding との差分（★が本モデルの新規部分）:
    - encoder      : ★無条件（条件を入力しない）。個票にデモグラベルが無い設定のため。
    - decoder 条件 : ★単一の群インデックス d を nn.Embedding で埋め込み
                     （CVAE の one-hot 連結 / CVAE_Embedding の5属性 Embedding に対応）。
                     ★埋め込みは null トークン（index = n_demo）を1つ持ち、
                     Stage1 のラベル無し学習では全個票を null に固定する。
    - 損失         : cvae_loss は CVAE と同一（96スロット cross-entropy + β*KL）。
    - 学習         : ★2段階。
                     Stage 1 (train_elbo)      : ELBO 学習（null 固定=無条件 / 真ラベル=skyline）。
                     Stage 2 (aggregate_match) : 群埋め込み+デコーダのみを集計マッチ損失
                                                 MSE(P μ̂, ν) で微調整。個票ラベル不使用。
    - 生成         : 群レベルの期待行動者率 μ̂(d)（group_rates）が主目的。
                     個票サンプリングは sample_schedules（CVAE の generate に対応）。

このファイルは単体実行しない（データ読み込みは実験ごとに異なるため実験スクリプトが担当）:
    uv run python src/models/CVAE_Aggregate/aggmatch_experiment.py    (D2-b シミュレーション)
    uv run python src/models/CVAE_Aggregate/japan_match_experiment.py (社基調への実転移)
"""
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. 設定（ハイパーパラメータ）
# ============================================================
# データ・チェックポイントのパスは実験スクリプト側で持つ（このモデルはデータセット非依存）

# 活動スケジュール
NUM_SLOTS   = 96      # 15分刻み × 96 = 24時間
# 活動状態数 n_act と群数 n_demo はデータセット依存のため AggCVAE のコンストラクタ引数

# モデル
HIDDEN_DIM  = 512
Z_DIM       = 64
DEMO_EMB    = 16      # ★群埋め込みの次元（CVAE_Embedding の COND_EMB_DIM に対応）
BETA        = 0.5

# Stage 1: ELBO 学習（pretrain / skyline）
PRETRAIN_EPOCHS = 200
BATCH_SIZE      = 1024
LR_PRETRAIN     = 1e-3

# Stage 2: 集計マッチ微調整 ★
AGG_STEPS   = 600
S_TRAIN     = 128     # 群あたり z サンプル数（μ̂ の MC 平均）
LR_EMB      = 1e-2    # 埋め込みはランダム初期化から動かすので高め
LR_DECODER  = 3e-4

EVAL_S      = 2048    # 評価時の群あたり z サンプル数
SEED        = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# ============================================================
# 2. データ整形
# ============================================================
# データ読み込みは実験ごとに異なるため、実験スクリプト側が担当する
# （CVAE/model.py の load_data / ScheduleDataset に相当する処理がここに無いのは意図的）:
#   - aggmatch_experiment.py : shakicho_sim.derive_strata による疑似層別データ
#   - japan_match_experiment.py : ATUS 共通12分類データ + 社基調公表集計
# モデルは活動状態数 n_act・群数 n_demo をコンストラクタで受けるデータセット非依存設計。


# ============================================================
# 3. モデル（AggCVAE: 無条件 encoder + 群埋め込み条件付き decoder）
# ============================================================
class AggCVAE(nn.Module):
    """demo index が null トークン (index = n_demo) のとき無条件（Stage1 / ラベル無し学習用）"""

    def __init__(self, n_act: int, n_demo: int):
        super().__init__()
        self.n_act = n_act
        x_dim = NUM_SLOTS * n_act

        # encoder: x_onehot(96*n_act) -> hidden -> (mu, logvar)
        # ★CVAE との差分: 条件を入力しない（個票にデモグラベルが無い設定）
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(HIDDEN_DIM, Z_DIM)  # μ
        self.fc_logvar = nn.Linear(HIDDEN_DIM, Z_DIM)  # logσ^2

        # ★CVAE との差分: 群埋め込み（+1 = null トークン）
        self.demo_emb  = nn.Embedding(n_demo + 1, DEMO_EMB)

        # decoder: [z(64) + demo_emb(16)] -> hidden -> 96*n_act ロジット
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM + DEMO_EMB, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, x_dim),
        )

    # 群インデックス (B,) -> 埋め込み (B, DEMO_EMB)（CVAE_Embedding の embed_cond に対応）
    def embed_demo(self, demo_idx: torch.Tensor) -> torch.Tensor:
        return self.demo_emb(demo_idx)

    # エンコーダー（★条件なし）
    def encode(self, x_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_onehot)               # [x_onehot(96*n_act)] -> [h(512)]
        return self.fc_mu(h), self.fc_logvar(h)  # [h(512)] -> [μ(64)], [logσ^2(64)]

    # 再パラメータ化トリック
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)  # logσ² → σ
        eps = torch.randn_like(std)    # N(0, I) からサンプリング（ε ～ N(0, I)）
        return mu + eps * std          # z = μ + σ·ε

    # デコーダー（e は埋め込み済み群ベクトル）
    def decode(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(torch.cat([z, e], dim=1))  # [z(64) + demo_emb(16)] -> [logits(96*n_act)]
        return logits.view(-1, NUM_SLOTS, self.n_act)    # (B, 96, n_act)

    def decode_probs(self, z: torch.Tensor, demo_idx: torch.Tensor) -> torch.Tensor:
        """(B,Z),(B,) -> スロット別活動確率 (B,96,n_act)"""
        logits = self.decode(z, self.embed_demo(demo_idx))
        return F.softmax(logits, dim=-1)

    def forward(self, sched: torch.Tensor, demo_idx: torch.Tensor):
        x_onehot = F.one_hot(sched, self.n_act).float().view(sched.size(0), -1)
        mu, logvar = self.encode(x_onehot)          # encoder input: [act] のみ（★条件なし）
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, self.embed_demo(demo_idx))  # decoder input: [z, demo_emb]
        return logits, mu, logvar


# ============================================================
# 4. 損失
# ============================================================
def cvae_loss(logits: torch.Tensor, sched: torch.Tensor, mu: torch.Tensor,
              logvar: torch.Tensor, beta: float = BETA) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 再構成: 96スロットの cross-entropy（1サンプルあたり96スロット合計、バッチ平均）
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), sched.reshape(-1), reduction="sum"
    ) / sched.size(0)
    # 正則化: KL( q(z|x) || N(0,I) )（バッチ平均）
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / sched.size(0)
    return recon + beta * kl, recon, kl


# ============================================================
# 5. 学習
# ============================================================
def train_elbo(model: AggCVAE, S_t: torch.Tensor, demo_t: torch.Tensor,
               w: npt.NDArray[np.float64], epochs: int, tag: str) -> None:
    """Stage 1: 重み付きサンプラで ELBO 学習（CVAE の train() に対応）。
    demo_t = null トークンで無条件 pretrain / 真ラベルで skyline。"""
    opt = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN)
    n = len(S_t)
    prob = torch.as_tensor(w / w.sum())
    for ep in range(1, epochs + 1):
        idx = torch.multinomial(prob, n, replacement=True)
        model.train()
        ep_loss = 0.0
        for b in range(0, n, BATCH_SIZE):
            bi = idx[b:b + BATCH_SIZE]
            sched, demo = S_t[bi].to(DEVICE), demo_t[bi].to(DEVICE)
            logits, mu, logvar = model(sched, demo)
            loss, _, _ = cvae_loss(logits, sched, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(bi)
        if ep % 50 == 0 or ep == 1:
            print(f"  [{tag}] epoch {ep:3d}  loss {ep_loss / n:8.3f}")


def aggregate_match(model: AggCVAE, P: npt.NDArray[np.float64],
                    nu: npt.NDArray[np.float64], n_act: int, tag: str) -> None:
    """★Stage 2（CVAE に無い新規部分）: 群埋め込み+デコーダを集計マッチ損失で微調整。
    μ̂(d) = E_z[softmax(dec(z, e_d))]（S_TRAIN サンプルの MC 平均; 微分可能）,
    ν̂ = P μ̂,  loss = MSE(ν̂, ν)。個票の群ラベルは一切見ない。"""
    P_t  = torch.as_tensor(P, dtype=torch.float32, device=DEVICE)
    nu_t = torch.as_tensor(nu, dtype=torch.float32, device=DEVICE).view(len(nu), n_act, NUM_SLOTS)
    D = P.shape[1]
    opt = torch.optim.Adam([
        {"params": model.demo_emb.parameters(), "lr": LR_EMB},
        {"params": model.decoder.parameters(),  "lr": LR_DECODER},
    ])
    model.train()
    for step in range(1, AGG_STEPS + 1):
        z  = torch.randn(D * S_TRAIN, Z_DIM, device=DEVICE)
        di = torch.arange(D, device=DEVICE).repeat_interleave(S_TRAIN)
        probs = model.decode_probs(z, di).view(D, S_TRAIN, NUM_SLOTS, model.n_act)
        mu_hat = probs.mean(dim=1).permute(0, 2, 1)            # (D, n_act, 96)
        nu_hat = torch.einsum("gd,daf->gaf", P_t, mu_hat)      # (G, n_act, 96)
        loss = F.mse_loss(nu_hat, nu_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  [{tag}] step {step:4d}  agg-mse {loss.item():.6f}")


# ============================================================
# 6. 生成（CVAE の generate / sanity_check に対応; 主目的は群レベル期待率 μ̂(d)）
# ============================================================
@torch.no_grad()
def group_rates(model: AggCVAE, n_demo: int, n_samples: int = EVAL_S) -> np.ndarray:
    """μ̂(d): 群ごとの期待スロット別活動確率 (D, 96*n_act)。離散化せず確率の平均。"""
    model.eval()
    out = []
    for d in range(n_demo):
        z = torch.randn(n_samples, Z_DIM, device=DEVICE)
        di = torch.full((n_samples,), d, dtype=torch.long, device=DEVICE)
        probs = model.decode_probs(z, di).mean(dim=0)         # (96, n_act)
        out.append(probs.T.reshape(-1).cpu().numpy())          # act-major (a*96+t) に転置
    return np.stack(out)


@torch.no_grad()
def group_rates_null(model: AggCVAE, n_demo: int, n_samples: int = EVAL_S) -> np.ndarray:
    """null トークン (index = n_demo) での期待行動者率 (1, F)。uncond 対照の評価用。"""
    model.eval()
    z = torch.randn(n_samples, Z_DIM, device=DEVICE)
    di = torch.full((n_samples,), n_demo, dtype=torch.long, device=DEVICE)
    probs = model.decode_probs(z, di).mean(dim=0)
    return probs.T.reshape(1, -1).cpu().numpy()


@torch.no_grad()
def sample_schedules(model: AggCVAE, d: int, n: int) -> np.ndarray:
    """群 d の離散スケジュール (n,96) をスロット別カテゴリカルサンプリングで生成"""
    model.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    di = torch.full((n,), d, dtype=torch.long, device=DEVICE)
    probs = model.decode_probs(z, di)
    return torch.distributions.Categorical(probs=probs).sample().cpu().numpy()
