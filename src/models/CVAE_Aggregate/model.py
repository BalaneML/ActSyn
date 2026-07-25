"""
model.py
================
集計マッチ学習のための CVAE（AggCVAE）

CVAE / CVAE_Embedding との差分（★が本モデルの新規部分）:
    - encoder     : ★無条件; 個票にデモグラベルが無い設定のため
    - decoder     : ★単一の群インデックス d を nn.Embedding で埋め込み
                    ★埋め込みは null トークンを1つ持ち、
                    Stage1 のラベル無し学習では全個票を null に固定する
    - 損失         : cvae_loss は CVAE と同一（96スロット cross-entropy + β*KL）
    - 学習         : ★2段階
                    Stage 1 (train_elbo)      : ELBO 学習（null 固定=無条件 / 真ラベル=skyline）
                    Stage 2 (aggregate_match) : 群埋め込み+デコーダのみを集計マッチ損失
                                                MSE(P μ̂, ν) で微調整; 個票ラベル不使用
    - 生成         : 群レベルの期待行動者率 μ̂(d)（group_rates）が主目的
                    個票サンプリングは sample_schedules

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
# 活動スケジュール
# 活動状態数 n_act と群数 n_demo はデータセット依存のため AggCVAE のコンストラクタ引数
NUM_SLOTS   = 96      # 15分刻み × 96 = 24時間

# モデル
HIDDEN_DIM  = 512  
Z_DIM       = 64      
DEMO_EMB    = 16      # ★群埋め込みの次元
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
# モデルは活動状態数 n_act・群数 n_demo をコンストラクタで受けるデータセット非依存設計


# ============================================================
# 3. モデル（AggCVAE: 無条件 encoder + 群埋め込み条件付き decoder）
# ============================================================
class AggCVAE(nn.Module):
    """Aggregate Conditional VAE
    args:
        n_act : 活動数
        n_demo: 群数 + 1ダミー (例: 年齢7層 × 性別2 -> n_demo = 14 + 1 = 15)
                demo index が null の時，無条件を意味する (ラベルなし学習用)
                Decoderは条件 (群埋め込み) を入力に取る設計 
                    -> 余分に確保した15番目を「無条件条件」として入力する
    """

    def __init__(self, n_act: int, n_demo: int):
        super().__init__()
        self.n_act = n_act
        x_dim = NUM_SLOTS * n_act

        # ★CVAE との差分: 条件を入力しない
        # ENCODER: x_onehot(96*n_act) -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(HIDDEN_DIM, Z_DIM)  # μ
        self.fc_logvar = nn.Linear(HIDDEN_DIM, Z_DIM)  # logσ^2

        # ★CVAE との差分: 群埋め込み（+1 = null トークン; ガイド §2.2）
        # 埋め込みテーブルの構造:
        #   行 0..n_demo-1 : 実在する群（性2×年齢7×就業2 = 28群）
        #   行 n_demo      : null トークン =「条件なし」を表す特別な群
        #                    Stage1 では全個票の条件をこの行に固定することで、
        #                    実質無条件の生成器として学習する
        self.demo_emb  = nn.Embedding(n_demo + 1, DEMO_EMB)

        # decoder: [z(64) + demo_emb(16)] -> hidden -> 96*n_act ロジット
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM + DEMO_EMB, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, x_dim),
        )

    # 群インデックス (B,) -> 埋め込み (B, DEMO_EMB)
    def embed_demo(self, demo_idx: torch.Tensor) -> torch.Tensor:
        return self.demo_emb(demo_idx)

    # エンコーダー q( z | x )
    def encode(self, x_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_onehot)               # [x_onehot(96*n_act)] -> [h(512)]
        return self.fc_mu(h), self.fc_logvar(h)  # [h(512)] -> [μ(64)], [logσ^2(64)]

    # 再パラメータ化トリック
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)  # logσ² → σ
        eps = torch.randn_like(std)    # N(0, I) からサンプリング（ε ～ N(0, I)）
        return mu + eps * std          # z = μ + ε·σ

    # デコーダー（e は埋め込み済み群ベクトル） p( x | z, e )
    def decode(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(torch.cat([z, e], dim=1))  # [z(64) + demo_emb(16)] -> [logits(96*n_act)]
        return logits.view(-1, NUM_SLOTS, self.n_act)    # (B, 96, n_act)

    def decode_probs(self, z: torch.Tensor, demo_idx: torch.Tensor) -> torch.Tensor:
        """各スロットにおける活動カテゴリの確率分布を計算する
        (B,Z),(B,) -> スロット別活動確率 (B,96,n_act)
        """
        logits = self.decode(z, self.embed_demo(demo_idx))
        return F.softmax(logits, dim=-1)

    def forward(self, sched: torch.Tensor, demo_idx: torch.Tensor):
        x_onehot = F.one_hot(sched, self.n_act).float().view(sched.size(0), -1)  # 活動系列をonehot化
        mu, logvar = self.encode(x_onehot)  # encoder: [x_onehot(96*n_act)] -> [μ(64)], [logσ^2(64)
        z = self.reparameterize(mu, logvar)  # 再構成トリック
        logits = self.decode(z, self.embed_demo(demo_idx))  # decoder: [z, demo_emb] -> [96, n_act]
        return logits, mu, logvar


# ============================================================
# 3.5 微分可能な系列統計量（提案①: 複数統計量教師; docs/proposals.md）
# ============================================================
def participation_probs(probs: torch.Tensor) -> torch.Tensor:
    """日次参加率 P(いずれかのスロットで活動a | z) = 1 − Π_t (1 − p_t(a))

    z 条件付きでスロット独立（decode_probs のカテゴリカル仮定）の下での閉形式。
    時刻別行動者率（1次周辺）からは導出できない、エピソードの個人内集中度を
    拘束する統計量。積は log1p の和で数値安定に計算する。

    probs: (..., NUM_SLOTS, n_act) -> (..., n_act)
    """
    # log Π_t(1−p_t) = Σ_t log(1−p_t) の形で計算する（ガイド §5）:
    #   ・1未満の数を96個直接掛けるとアンダーフローするので log の和にする
    #   ・log(1-p) でなく log1p(-p) を使うのは p が小さいときの桁落ち防止
    #   ・clamp は p_t=1 ちょうどのとき log(0)=−∞ になるのを回避
    log_none = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=-2)
    return 1.0 - torch.exp(log_none)  # 余事象: 1 −「全スロットで活動 a をしない確率」


# ============================================================
# 4. 損失
# ============================================================
def cvae_loss( logits: torch.Tensor,
                sched: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor, 
                beta: float = BETA
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    再構成:
        96スロットの cross-entropy（1サンプルあたり96スロット合計、バッチ平均）
        logits: デコーダ出力
        sched : 正解
    '''
    # reduction="sum" で全サンプル×全スロットの CE を合計してから
    # sched.size(0)（バッチ人数）で割る =「1人あたり96スロット分の合計CE」。
    # スロット平均（さらに /96）にしないのは、KL とのスケール比（β の意味）を
    # ELBO の定義 log p(x|z) = Σ_t log p(x_t|z) のまま保つため
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), sched.reshape(-1), reduction="sum"
    ) / sched.size(0)

    '''
    正則化: 
        KL( q(z|x) || N(0,I) )（バッチ平均）
    '''
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / sched.size(0)
    return recon + beta * kl, recon, kl


# ============================================================
# 5. 学習
# ============================================================
def train_elbo( model : AggCVAE,
                S_t   : torch.Tensor,
                demo_t: torch.Tensor,
                w     : npt.NDArray[np.float64],
                epochs: int,
                tag   : str
                ) -> None:
    """
    Stage 1: 重み付きサンプラで ELBO 学習（CVAE の train() に対応）
    demo_t = null トークンで無条件 pretrain / 真ラベルで skyline (天井性能)

    args:
        model : AggCVAE
        S_t   : 個票のスケジュール系列 (N, NUM_SLOTS)
        demo_t: 各個票に紐づく群インデックス (N,) / 無条件null or 群条件
        w     : サンプル重み (N,)
        epochs: 学習エポック数
        tag   : ログ出力用
    """
    opt = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN)
    n = len(S_t)
    prob = torch.as_tensor(w / w.sum())  # 調査ウェイト w を選択確率に正規化
    for ep in range(1, epochs + 1):
        # 重み付き復元抽出で毎エポック n 本を引き直す（weighted bootstrap）。
        # ウェイトの大きい個票ほど頻繁に提示される = 重み付き ELBO の近似
        idx = torch.multinomial(prob, n, replacement=True)
        model.train()
        ep_loss = 0.0
        for b in range(0, n, BATCH_SIZE):
            bi = idx[b:b + BATCH_SIZE]
            sched, demo = S_t[bi].to(DEVICE), demo_t[bi].to(DEVICE)
            logits, mu, logvar = model(sched, demo)
            loss, _, _ = cvae_loss(logits, sched, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(bi)
        if ep % 50 == 0 or ep == 1:
            print(f"  [{tag}] epoch {ep:3d}  loss {ep_loss / n:8.3f}")


def aggregate_match(model: AggCVAE,
                    P      : npt.NDArray[np.float64],
                    nu     : npt.NDArray[np.float64],
                    n_act  : int,
                    tag    : str,
                    part_nu: npt.NDArray[np.float64] | None = None,
                    part_w : float = 0.0,
                    steps  : int = AGG_STEPS
                    ) -> None:
    """
    ★Stage 2（CVAE に無い新規部分）: 群埋め込み+デコーダを集計マッチ損失で微調整
    μ̂(d) = E_z[softmax(dec(z, e_d))]（S_TRAIN サンプルの MC 平均; 微分可能）,
    ν̂ = P μ̂,  loss = MSE(ν̂, ν)。個票の群ラベルは一切見ない

    args:
        model  : AggCVAE
        P      : 構成行列[セルg, 群d]，セルgの人口のうち，群dの人が占める割合，各行の和=1
        nu     : 複数のセルgの人たちを平均した時刻別行動者率ベクトル
        n_act  : 活動の種類数
        tag    : ログ出力用
        part_nu: セル別・活動別の日次参加率 (G, n_act)。生活時間編「行動者率」に対応
                （提案①: 参加率は個人指示関数の期待値なのでセル集約が線形 ν_part = P·part(d)）
        part_w : 参加率マッチ項の重み。0 なら従来の周辺マッチのみ（既存呼び出しは不変）
        steps  : マッチ反復数（既定 AGG_STEPS）
    """
    P_t  = torch.as_tensor(P, dtype=torch.float32, device=DEVICE)
    nu_t = torch.as_tensor(nu, dtype=torch.float32, device=DEVICE).view(len(nu), n_act, NUM_SLOTS)
    part_t = None
    if part_nu is not None and part_w > 0:
        part_t = torch.as_tensor(part_nu, dtype=torch.float32, device=DEVICE)  # (G, n_act)
    D = P.shape[1]
    # ★encoder は optimizer に渡さない（凍結相当）。学習するのは以下の2つだけ:
    #   demo_emb : ランダム初期化のまま Stage1 を素通りしている（null しか使われて
    #              いない）ので、ゼロから大きく動かす必要がある → 高めの LR
    #   decoder  : Stage1 で覚えた「スケジュールらしさ」を壊さない程度の微調整 → 低め LR
    opt = torch.optim.Adam([
        {"params": model.demo_emb.parameters(), "lr": LR_EMB},
        {"params": model.decoder.parameters(),  "lr": LR_DECODER},
    ])
    model.train()
    for step in range(1, steps + 1):
        # z を事前分布 N(0,I) から全群まとめて D*S_TRAIN 本引く (D*S, 64)。
        # z はパラメータに依存しないので再パラメータ化トリックは不要（Stage1 との違い）
        z  = torch.randn(D * S_TRAIN, Z_DIM, device=DEVICE)
        # di = [0,0,...,0, 1,1,...,1, ...] 各群 S_TRAIN 本ずつの群番号 (D*S,)
        di = torch.arange(D, device=DEVICE).repeat_interleave(S_TRAIN)
        # 群×サンプル別のスロット活動確率 (D, S, 96, n_act)
        probs = model.decode_probs(z, di).view(D, S_TRAIN, NUM_SLOTS, model.n_act)
        # S 本の MC 平均 = μ̂(d)。離散サンプリングせず「確率のまま」平均するので
        # 勾配が softmax → decoder → 群埋め込みへ素直に流れる（微分可能性の核心）
        mu_hat = probs.mean(dim=1).permute(0, 2, 1)            # (D, n_act, 96) 教師の act-major 並びに転置
        # ν̂(g) = Σ_d P[g,d]·μ̂(d) セルごとの群加重平均（g=セル d=群 a=活動 f=時刻スロット）
        nu_hat = torch.einsum("gd,daf->gaf", P_t, mu_hat)      # (G, n_act, 96)
        loss = F.mse_loss(nu_hat, nu_t)
        if part_t is not None:
            # 参加率も個人指示関数の期待値なのでセル集約が線形 → 同じ P がそのまま使える
            part_hat = participation_probs(probs).mean(dim=1)  # (D, n_act)
            loss = loss + part_w * F.mse_loss(P_t @ part_hat, part_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
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
def group_participation(model: AggCVAE, n_demo: int,
                        n_samples: int = EVAL_S) -> tuple[np.ndarray, np.ndarray]:
    """群ごとの (日次参加率 (D, n_act), 行動者平均スロット数 (D, n_act))。

    行動者平均継続時間は恒等式 E[スロット数] = E[スロット数 | 参加]·P(参加) より
    dur = E_z[Σ_t p_t] / E_z[1 − Π_t(1 − p_t)] で計算（×15分で分に換算可能）。
    """
    model.eval()
    parts, durs = [], []
    for d in range(n_demo):
        z = torch.randn(n_samples, Z_DIM, device=DEVICE)
        di = torch.full((n_samples,), d, dtype=torch.long, device=DEVICE)
        probs = model.decode_probs(z, di)                      # (n, 96, n_act)
        part = participation_probs(probs).mean(dim=0)          # 日次参加率 P(参加) (n_act,)
        tot  = probs.sum(dim=-2).mean(dim=0)                   # 1人あたり期待スロット数 E[数] (n_act,)
        parts.append(part.cpu().numpy())
        # 恒等式 E[数] = E[数|参加]·P(参加) の変形: E[数]/P(参加) =「参加した人だけ」の
        # 平均スロット数。clamp は参加率ほぼ0の活動でのゼロ除算防止
        durs.append((tot / part.clamp(min=1e-6)).cpu().numpy())
    return np.stack(parts), np.stack(durs)


@torch.no_grad()
def sample_schedules(model: AggCVAE, d: int, n: int) -> np.ndarray:
    """群 d の離散スケジュール (n,96) をスロット別カテゴリカルサンプリングで生成"""
    model.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    di = torch.full((n,), d, dtype=torch.long, device=DEVICE)
    probs = model.decode_probs(z, di)
    return torch.distributions.Categorical(probs=probs).sample().cpu().numpy()
