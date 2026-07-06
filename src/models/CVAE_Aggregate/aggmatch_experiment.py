"""
D2-b 集計マッチ腕: セル集計のみから CVAE のデモグラ条件付けを学習する

目的 (実験デザイン D2-b, 計画ファイル参照):
    recovery_experiment.py の線形復元をベースラインに、非線形生成器 (CVAE) の
    「集計マッチ微調整」でデモグラ条件付けを個票ラベルなしで獲得できるかを検証する。
    ATLAS のシナリオを再現: 個票スケジュールはあるがデモグラベルが無い。
    教師は セル集計 ν(g) と構成行列 P のみ。個票の群ラベルは評価にだけ使う。

2段階学習:
    Stage 1 (pretrain)  : デモグラ条件を null トークンに固定した無条件 ELBO 学習。
                          ラベル不使用。スペック非依存なので1回だけ学習し使い回す。
    Stage 2 (agg-match) : 群埋め込み + デコーダを、集計マッチ損失で微調整。
                          μ̂(d) = E_z[softmax(dec(z, e_d))] (S サンプルの MC 平均; 微分可能),
                          ν̂ = P μ̂,  loss = MSE(ν̂, ν)。個票ラベルは一切見ない。

比較する腕:
    linear    : recovery_experiment.recover (lstsq) — 線形ベースライン
    aggmatch  : 上記2段階 — 本命
    placebo   : P の行をシャッフルして aggmatch — 負の対照 (機構主張の反証条件)
    uncond    : Stage1 のみ (条件付けなし; 全群に同じ分布) — プール分布だけで
                rate_rmse がどこまで下がるかの床。条件付けの寄与を分離する対照
    skyline   : 真の群ラベルで条件付き学習した同一アーキ CVAE — 生成器族の上限

評価 (個票 ground truth):
    rate_rmse  : 群別時刻別行動者率 μ̂(d) と μ*(d) の RMSE (linear と同一定義で比較可能)。
                 ※ プール分布でもかなり下がる (群間の共通構造が大きい) ため、
                   条件付け能力の指標としては dev_rmse を主に見ること。
    dev_rmse   : 群偏差 (μ(d) − μ̄; μ̄=人口平均) 部分だけの RMSE。
                 「群の違い」をどれだけ復元したかの直接指標。uncond は偏差ゼロ予測
                 に相当し、dev_rmse = 真の群偏差の RMS (判別ゼロの床) になる。
    bigram_jsd : 群ごとの活動遷移 bigram 分布 (17×17) の JSD 平均。
                 教師は1次周辺 (時刻別率) のみなので、これは統計量ホールドアウト
                 (D2-c #3: 周辺で学習→高次依存で評価) に相当する。
                 「条件外の依存が転移・保存されるか」(Novikov/仮説2) の直接テスト。

出力:
    data/processed/aggregates/<データセットstem>/aggmatch_experiment.csv
    outputs/checkpoints/aggmatch_pretrain.pt (Stage 1 チェックポイント; 再実行時に再利用)

使い方:
    uv run python src/models/CVAE_Aggregate/aggmatch_experiment.py [--reuse-pretrain]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/CVAE_Aggregate -> repo root
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "aggregates"))
from shakicho_sim import CELL_SPECS, SLOT_COLS, OUT_DIR, DEFAULT_DATASET, derive_strata  # noqa: E402
from recovery_experiment import one_hot_schedules, weighted_mean, recover  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "aggmatch_pretrain.pt"

# ===== 実験設定 =====
DEMO_COLS  = ["gender", "age_class"]   # 群 d = 性×年齢7区分 (D=14; recovery の sex_x_age7)
EVAL_CELL_SPECS = ["state", "state_x_daytype", "dow_x_emp_x_sex", "daytype_x_emp"]
                                        # σ_min 良 / 最良(セル極小) / 中 / 悪 の4点

NUM_SLOTS  = 96
HIDDEN_DIM = 512
Z_DIM      = 64
DEMO_EMB   = 16
BETA       = 0.5

PRETRAIN_EPOCHS = 200
BATCH_SIZE      = 1024
LR_PRETRAIN     = 1e-3

AGG_STEPS   = 600
S_TRAIN     = 128     # Stage2 の群あたり z サンプル数 (MC平均)
LR_EMB      = 1e-2    # 埋め込みはランダム初期化から動かすので高め
LR_DECODER  = 3e-4

EVAL_S      = 2048    # 評価時の群あたり z サンプル数
BIGRAM_N    = 2000    # bigram 評価用の離散サンプル数/群
SEED        = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


# ============================================================
# モデル: 無条件 encoder + 群埋め込み条件付き decoder の CVAE
# ============================================================
class AggCVAE(nn.Module):
    """demo index D が null トークン (Stage1 / ラベル無し学習用)"""

    def __init__(self, n_act: int, n_demo: int):
        super().__init__()
        self.n_act = n_act
        x_dim = NUM_SLOTS * n_act
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(HIDDEN_DIM, Z_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, Z_DIM)

        self.demo_emb  = nn.Embedding(n_demo + 1, DEMO_EMB)  # +1 = null
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM + DEMO_EMB, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, x_dim),
        )

    def decode_probs(self, z: torch.Tensor, demo_idx: torch.Tensor) -> torch.Tensor:
        """(B,Z),(B,) -> スロット別活動確率 (B,96,n_act)"""
        e = self.demo_emb(demo_idx)
        logits = self.decoder(torch.cat([z, e], dim=1)).view(-1, NUM_SLOTS, self.n_act)
        return F.softmax(logits, dim=-1)

    def forward(self, sched: torch.Tensor, demo_idx: torch.Tensor):
        x = F.one_hot(sched, self.n_act).float().view(sched.size(0), -1)
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        e = self.demo_emb(demo_idx)
        logits = self.decoder(torch.cat([z, e], dim=1)).view(-1, NUM_SLOTS, self.n_act)
        return logits, mu, logvar


def elbo_loss(logits, sched, mu, logvar):
    recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), sched.reshape(-1),
                            reduction="sum") / sched.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / sched.size(0)
    return recon + BETA * kl


def train_elbo(model: AggCVAE, S_t: torch.Tensor, demo_t: torch.Tensor, w: np.ndarray,
               epochs: int, tag: str):
    """重み付きサンプラで ELBO 学習 (demo_t = null で無条件 / 真ラベルで skyline)"""
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
            loss = elbo_loss(logits, sched, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(bi)
        if ep % 50 == 0 or ep == 1:
            print(f"  [{tag}] epoch {ep:3d}  loss {ep_loss / n:8.3f}")


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
def sample_schedules(model: AggCVAE, d: int, n: int) -> np.ndarray:
    """群 d の離散スケジュール (n,96) をスロット別カテゴリカルサンプリングで生成"""
    model.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    di = torch.full((n,), d, dtype=torch.long, device=DEVICE)
    probs = model.decode_probs(z, di)
    return torch.distributions.Categorical(probs=probs).sample().cpu().numpy()


def aggregate_match(model: AggCVAE, P: np.ndarray, nu: np.ndarray, n_act: int, tag: str):
    """Stage 2: 群埋め込み+デコーダを集計マッチ損失で微調整 (個票ラベル不使用)"""
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


def bigram_dist(scheds: np.ndarray, n_act: int, w: np.ndarray | None = None) -> np.ndarray:
    """活動遷移 bigram 分布 (n_act*n_act,)。w があれば個人重み付き。"""
    a, b = scheds[:, :-1].ravel(), scheds[:, 1:].ravel()
    ww = np.ones(len(scheds)) if w is None else w
    ww = np.repeat(ww, NUM_SLOTS - 1)
    counts = np.zeros(n_act * n_act)
    np.add.at(counts, a * n_act + b, ww)
    return counts / counts.sum()


def main():
    ap = argparse.ArgumentParser(description="D2-b: 集計マッチで CVAE の条件付けを学習")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--reuse-pretrain", action="store_true",
                    help="既存の Stage1 チェックポイントを再利用")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    df = derive_strata(pd.read_csv(args.dataset))
    n_act = int(df[SLOT_COLS].to_numpy().max()) + 1
    X = one_hot_schedules(df, n_act)                     # (N, n_act*96) act-major
    w = df["TUFINLWGT"].to_numpy(dtype=np.float64)
    S_np = df[SLOT_COLS].to_numpy(dtype=np.int64)
    S_t  = torch.as_tensor(S_np)
    out_dir = OUT_DIR / args.dataset.stem
    print(f"dataset: {args.dataset.name}  N={len(df)}  n_act={n_act}  device={DEVICE}")

    # --- 群定義と ground truth (評価専用) ---
    demo_key  = df[DEMO_COLS].astype(str).agg("_".join, axis=1)
    demo_cats = sorted(demo_key.unique())
    D = len(demo_cats)
    demo_of   = demo_key.map({d: i for i, d in enumerate(demo_cats)}).to_numpy()
    demo_idx  = {i: np.flatnonzero(demo_of == i) for i in range(D)}
    mu_star   = np.stack([weighted_mean(X, w, demo_idx[i]) for i in range(D)])
    bigram_star = [bigram_dist(S_np[demo_idx[i]], n_act, w[demo_idx[i]]) for i in range(D)]
    pi = np.array([w[demo_idx[i]].sum() for i in range(D)])
    pi /= pi.sum()                       # 群の人口シェア (偏差の重心に使用)
    dev_star = mu_star - pi @ mu_star    # 真の群偏差

    def rate_and_dev_rmse(mu_hat: np.ndarray) -> tuple[float, float]:
        dev_hat = mu_hat - pi @ mu_hat
        return (float(np.sqrt(((mu_hat - mu_star) ** 2).mean())),
                float(np.sqrt(((dev_hat - dev_star) ** 2).mean())))

    def evaluate(model: AggCVAE, cond: bool = True) -> tuple[float, float, float]:
        """cond=False は uncond 対照: 全群に null トークンの分布を使う"""
        if cond:
            mu_hat = group_rates(model, D)
        else:
            mu_hat = np.repeat(group_rates_null(model), D, axis=0)
        rmse, dev = rate_and_dev_rmse(mu_hat)
        jsds = [jensenshannon(bigram_dist(sample_schedules(model, i if cond else D, BIGRAM_N),
                                          n_act),
                              bigram_star[i], base=2) ** 2
                for i in range(D)]
        return rmse, dev, float(np.mean(jsds))

    @torch.no_grad()
    def group_rates_null(model: AggCVAE) -> np.ndarray:
        """null トークン (index D) での期待行動者率 (1, F)"""
        model.eval()
        z = torch.randn(EVAL_S, Z_DIM, device=DEVICE)
        di = torch.full((EVAL_S,), D, dtype=torch.long, device=DEVICE)
        probs = model.decode_probs(z, di).mean(dim=0)
        return probs.T.reshape(1, -1).cpu().numpy()

    # --- Stage 1: 無条件 pretrain (スペック非依存; 1回だけ) ---
    null_t = torch.full((len(df),), D, dtype=torch.long)
    pre = AggCVAE(n_act, D).to(DEVICE)
    if args.reuse_pretrain and CKPT_PATH.exists():
        pre.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Stage1: 再利用 {CKPT_PATH}")
    else:
        print("Stage1: 無条件 pretrain ...")
        train_elbo(pre, S_t, null_t, w, PRETRAIN_EPOCHS, "pretrain")
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pre.state_dict(), CKPT_PATH)

    # --- uncond 対照: pretrain のみ (条件付けなし; 判別ゼロの床) ---
    unc_rmse, unc_dev, unc_jsd = evaluate(pre, cond=False)

    # --- skyline: 真ラベルで条件付き学習 (セル層別に依らない上限; 1回だけ) ---
    print("skyline: 真ラベル教師あり学習 ...")
    sky = AggCVAE(n_act, D).to(DEVICE)
    train_elbo(sky, S_t, torch.as_tensor(demo_of), w, PRETRAIN_EPOCHS, "skyline")
    sky_rmse, sky_dev, sky_jsd = evaluate(sky)

    # --- セル層別ごとに linear / aggmatch / placebo ---
    records = []
    for cell_name in EVAL_CELL_SPECS:
        cell_cols = CELL_SPECS[cell_name]
        cell_key  = df[cell_cols].astype(str).agg("_".join, axis=1)
        cell_cats = sorted(cell_key.unique())
        cell_idx  = {g: np.flatnonzero((cell_key == g).to_numpy()) for g in cell_cats}
        G = len(cell_cats)
        P = np.zeros((G, D))
        for gi, g in enumerate(cell_cats):
            for di in range(D):
                inter = np.intersect1d(cell_idx[g], demo_idx[di], assume_unique=True)
                P[gi, di] = w[inter].sum()
            P[gi] /= P[gi].sum()
        sigma_min = float(np.linalg.svd(P, compute_uv=False)[-1]) if G >= D else 0.0
        nu = np.stack([weighted_mean(X, w, cell_idx[g]) for g in cell_cats])
        print(f"\n=== cell_spec={cell_name}  G={G} D={D}  σ_min={sigma_min:.4f} ===")

        arms = {}
        # linear ベースライン (μ̂ = lstsq; bigram は個票モデルが無いので出せない)
        arms["linear"] = (*rate_and_dev_rmse(recover(P, nu)), np.nan)
        # aggmatch (pretrain から fork して微調整)
        m = AggCVAE(n_act, D).to(DEVICE); m.load_state_dict(pre.state_dict())
        aggregate_match(m, P, nu, n_act, f"{cell_name}/aggmatch")
        arms["aggmatch"] = evaluate(m)
        # placebo (P 行シャッフル)
        m = AggCVAE(n_act, D).to(DEVICE); m.load_state_dict(pre.state_dict())
        aggregate_match(m, P[rng.permutation(G)], nu, n_act, f"{cell_name}/placebo")
        arms["placebo"] = evaluate(m)

        for arm, (rmse, dev, jsd) in arms.items():
            records.append({"cell_spec": cell_name, "G": G, "D": D, "sigma_min": sigma_min,
                            "arm": arm, "rate_rmse": rmse, "dev_rmse": dev, "bigram_jsd": jsd})

    rec_base = {"cell_spec": "(none)", "G": 0, "D": D, "sigma_min": np.nan}
    records.append({**rec_base, "arm": "uncond",
                    "rate_rmse": unc_rmse, "dev_rmse": unc_dev, "bigram_jsd": unc_jsd})
    records.append({**rec_base, "arm": "skyline",
                    "rate_rmse": sky_rmse, "dev_rmse": sky_dev, "bigram_jsd": sky_jsd})

    res = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "aggmatch_experiment.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n=== 結果 (dev_rmse: 群偏差=条件付け能力 / bigram_jsd: 条件外依存=統計量ホールドアウト) ===")
        print(res.to_string(index=False))
    print(f"uncond の dev_rmse = 判別ゼロの床 ({unc_dev:.4f})。これを下回って初めて「集計が条件付けを教えた」と言える。")
    print(f"\nsaved -> {out_dir / 'aggmatch_experiment.csv'}")


if __name__ == "__main__":
    main()
