"""
model.py
================
集計マッチ転移のための条件付きDDPM（AggDDPM）

DDPM/model.py との差分（★が本モデルの新規部分）:
    - 活動分類  : ★ATUS 17分類 → 共通12分類 (crosswalk_atus_stula.Common)
    - スコープ  : ★平日のみ (day_of_week 2..6)。CVAE_Aggregate/japan_match_experiment と同一
    - 条件      : ★性2 × 年齢7区分 × 就業2 = 28群。群インデックス d の定義は
                    CVAE_Aggregate/japan_match_experiment.d_index と厳密に一致させる
                    （属性ごとに Embedding して連結するので、28通りの one-hot より
                    統計的強度を共有できる。群 d との対応は cond_grid で相互変換）
    - サンプリング: ★DDIM を追加。Stage2 の指数傾けは群あたり数千本の提案サンプルを
                    必要とするため、ancestral の T=1000 ステップでは実用にならない
    - 集計出力  : ★group_pool / group_rates を追加。group_rates の返り値は
                    CVAE_Aggregate.model.group_rates と同一形式 (D, n_act*96) act-major で、
                    japan_match_experiment.eval_against にそのまま渡せる

なぜ CVAE でなく DDPM か（断片化の実測; docs/proposals.md 提案③の生成器側）:
    CVAE_Aggregate は p(x|z) = Π_t p(x_t|z) と分解するため、z で説明しきれない
    残差エントロピーが時間的に無相関なノイズとして出る。ATUS平日・共通12分類で
    平均切替回数 実 12.64 に対し生成 48.1 (3.8倍)、日次参加率も最大8倍過大。
    同一リポジトリの DDPM (17分類・全曜日) は切替 実 12.49 / 生成 12.71 (1.02倍)、
    参加率 0.354 / 0.351 で、系列構造を保つことが実測済み。

    代償は「μ̂(d) が閉形式で微分可能でない」こと。逆過程を勾配が通らないため
    CVAE_Aggregate の Stage2（μ̂ への勾配降下）は使えない。代わりに
    japan_match_experiment.py が凍結モデルのサンプルプールへの指数傾け
    （最大エントロピー・凸問題）で集計マッチする。

使い方:
    # Stage1: ATUS 平日・共通12分類・28群で条件付き pretrain
    uv run python src/models/DDPM_Aggregate/model.py

    # 短時間の動作確認 (学習 5 エポック・DDIM 10 ステップ)
    uv run python src/models/DDPM_Aggregate/model.py --smoke

    フラグ:
        --epochs N   : 学習エポック数を上書き
        --no-wandb   : wandb ログを無効化
        --smoke      : 形状・整合の確認だけを短時間で回す

出力:
    outputs/checkpoints/ddpm_japan_pretrain_common12_weekday.pt   Stage1 (raw + EMA)
    outputs/generated/ddpm_japan_pretrain_samples.csv             サニティ用の生成個票
"""
import argparse
import copy
import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/DDPM_Aggregate -> repo root
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, NUM_COMMON  # noqa: E402


def _load_module(name: str, path: Path):
    """sys.modules に一意名で直接載せる（japan_match_experiment.py の _load と同じ様式）。

    このリポジトリは同名ファイル (model.py など) をフラットに持つので、
    通常の import は sys.path の順序次第で静かに別モジュールを掴む。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 個票指標の唯一の出所。断片化統計はここへ委譲する（下の fragmentation_stats 参照）
im = _load_module("agg_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")

# ============================================================
# 1. 設定（ハイパーパラメータ）
# ============================================================
DATA_PATH       = REPO_ROOT / 'data' / 'processed' / 'atus2024' / 'atus2024_stula_common12_dataset.csv'
MODEL_SAVE_PATH = REPO_ROOT / 'outputs' / 'checkpoints' / 'ddpm_japan_pretrain_common12_weekday.pt'
GEN_SAVE_PATH   = REPO_ROOT / 'outputs' / 'generated' / 'ddpm_japan_pretrain_samples.csv'

# 活動スケジュール
NUM_SLOTS   = 96                 # 15分刻み × 96 = 24時間（04:00開始）
NUM_ACT     = NUM_COMMON         # ★共通12分類（OTHER_X を含む; 除外は教師・評価側の責務）
ACT_NAMES   = [c.name for c in Common]
IN_CH       = NUM_ACT            # 拡散空間のチャネル数

# ★群定義: d = g*(N_A*N_E) + a*N_E + e。CVAE_Aggregate/japan_match_experiment と同一
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E       # 28

# 条件属性: (CSV列名, カテゴリ数, 埋め込み次元, インデックス化関数)
# ★就業(telfs)を追加。1,2 = 就業/休業中 -> 有業。社基調の有業/無業に対応付ける
#   （japan_match_experiment.py の emp と同一定義）
COND_SPEC = [
    ("gender", N_G, 4, lambda v: v.astype(np.int64)),                       # 0=男 1=女
    ("age",    N_A, 8, lambda v: np.clip((v - 15) // 10, 0, 6)),            # 15歳起点10歳刻み7区分
    ("telfs",  N_E, 4, lambda v: np.isin(v, [1, 2]).astype(np.int64)),      # 0=無業 1=有業
]
COND_COLS  = [name for name, _, _, _ in COND_SPEC]
COND_CARD  = [card for _, card, _, _ in COND_SPEC]  # 条件属性のカテゴリ数 (gender->2, age->7, telfs->2)
EMB_DIMS   = [dim for _, _, dim, _ in COND_SPEC]

DAY_FILTER  = 'weekday'   # ★平日固定（スコープ=平日; 土日への拡張は daytype 条件化として将来課題）

# DDPM
T_STEPS     = 1000
BETA_START  = 1e-4        # 0.0001 (t=1) -> .. -> t=T:0.02 (t=T, 1000)
BETA_END    = 0.02        # linear schedule (Ho et al. 2020 / Tang et al. 2025 準拠)

# Denoiser (1D-UNet)
BASE_CH     = 64
DROPOUT     = 0.1         # 小データ(平日 ~3.7k)の過学習対策
ATTN_HEADS  = 4
TIME_EMB_DIM = 256

# Classifier-Free Guidance
P_UNCOND       = 0.1
# ★2.0 (DDPM/model.py の既定) から 1.25 へ。就業を条件に加えたことで CFG が WORK を
#   強く増幅するようになった。ancestral・M=128 での WORK の時間シェア:
#       CFG 1.0 → 0.156 / 1.25 → 0.167 / 1.5 → 0.255   (実 ATUS 平日 = 0.168)
#   全活動のシェア誤差合計 Σ|Δ| も 1.25 が最小 (0.044; 1.0 で 0.061, 1.5 で 0.174)
GUIDANCE_SCALE = 1.25

# 学習
BATCH_SIZE  = 256
# ★平日のみで N=3,736 -> 3,736×0.9/256 ≈ 13 step/epoch。DDPM/model.py の総ステップ数
#   (約40k) に揃えるため、エポック数を 1500 -> 3000 に倍化する
EPOCHS      = 1000
LR          = 2e-4  # 0.0002
EMA_DECAY   = 0.999
VAL_RATIO   = 0.1
SEED        = 42
USE_WEIGHTED_SAMPLER = True
WEIGHT_COL  = "TUFINLWGT"
# ★patience もステップ換算で DDPM/model.py と揃える (100ep×27step ≈ 200ep×13step)
EARLY_STOP_PATIENCE  = 200
EARLY_STOP_MIN_DELTA = 1e-4
GEN_BATCH   = 1024

# ★プール生成のサンプラ: 'ancestral'（既定）か 'ddim'
#
#   DDIM は速いが、この離散one-hot拡散では argmax デコードの品質を著しく落とす。
#   実測（M=128, 実 ATUS 平日の平均切替 12.64 に対する生成の平均切替）:
#       DDIM 50 η=0  → 17.70      DDIM 200 η=0 → 22.57   (決定的; ステップを増やすほど悪化)
#       DDIM 50 η=1  → 13.85      DDIM 100 η=1 → 14.96   DDIM 250 η=1 → 17.73
#       ancestral 1000 → 12.87  ← 実データとの比 1.02。DDPM/model.py の実測と同水準
#   決定的 DDIM (η=0) は条件付き平均へ収束するため x0 が「ぼやけ」、argmax が
#   スロットごとにちらついて断片化する。ステップを増やすと ODE に忠実になる分
#   かえって悪化する（近似誤差でなく、ぼやけそのものが原因）。η=1 で改善するが
#   ancestral には届かない。Route B の前提は「個票が実データ水準」なので既定は ancestral。
#   プールは npz にキャッシュされる一回限りのコストなので、速度より忠実度を採る。
SAMPLER     = 'ancestral'
DDIM_STEPS  = 50          # SAMPLER='ddim' 時のみ使用（反復実験用の高速経路）
DDIM_ETA    = 1.0         # 同上。η=0 は上記のとおり品質が落ちるので使わない

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


# ============================================================
# 2. 群インデックスと条件の相互変換 ★
# ============================================================
def d_index(g: int, a: int, e: int) -> int:
    """
    (性, 年齢7区分, 就業) -> 群インデックス d
    """
    return g * (N_A * N_E) + a * N_E + e


def cond_grid() -> npt.NDArray[np.int64]:
    """
    全28群の条件インデックス (D, 3)
    行 d が群 d に対応する

    COND_SPEC の並び (gender, age, telfs) と一致させる
    """
    grid = np.zeros((D_GROUPS, len(COND_SPEC)), dtype=np.int64)
    for g in range(N_G):
        for a in range(N_A):
            for e in range(N_E):
                grid[d_index(g, a, e)] = (g, a, e)
    return grid


# ============================================================
# 3. データ整形
# ============================================================
def load_data(path: str | Path = DATA_PATH):
    """
    CSV を読み、(条件インデックス, スケジュール, 調査ウェイト, 生条件列) を返す
    """
    df: pd.DataFrame = pd.read_csv(path)

    # ★平日 (月..金)。japan_match_experiment と同じ between(2,6) で揃える
    # .loc[bool Series] は DataFrame を返す（df[...] は Series との union になる）
    if DAY_FILTER == 'weekday':
        df = df.loc[df["day_of_week"].between(2, 6)].reset_index(drop=True)
    elif DAY_FILTER == 'weekend':
        df = df.loc[~df["day_of_week"].between(2, 6)].reset_index(drop=True)

    cond_idx = np.stack(
        [fn(df[name].to_numpy()) for name, _, _, fn in COND_SPEC], axis=1
    ).astype(np.int64)

    scols = [f"s{i}" for i in range(NUM_SLOTS)]
    schedules = df[scols].to_numpy().astype(np.int64)
    weights = df[WEIGHT_COL].to_numpy().astype(np.float64)
    return cond_idx, schedules, weights, df[COND_COLS]


def cond_to_d(cond_idx: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """
    条件インデックス (N,3) -> 群インデックス (N,)
    """
    return (cond_idx[:, 0] * (N_A * N_E) + cond_idx[:, 1] * N_E + cond_idx[:, 2])


class ScheduleDataset(Dataset):
    def __init__(self, cond_idx, schedules):
        self.cond_idx  = torch.as_tensor(cond_idx,  dtype=torch.long)
        self.schedules = torch.as_tensor(schedules, dtype=torch.long)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, i):
        return self.cond_idx[i], self.schedules[i]


def split_indices(n: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    学習/評価 - ホールドアウトの行インデックス (train_idx, val_idx)

    make_loaders から抽出
    暗記チェック（individual_metrics.memorization）が
    「生成物が学習集合にだけ近いか」を測るには、学習に使った行と使わなかった行を
    同じ規則で再現する必要がある。分割規則を2箇所に書くと静かにずれるので、
    唯一の出所をここに置く。乱数の使い方は抽出前と同一。
    """
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * VAL_RATIO)
    return perm[n_val:], perm[:n_val]


def make_loaders(cond_idx, sched, weight):
    train_idx, val_idx = split_indices(len(sched))

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
    """
    活動スケジュール (index表現) を onehot~{-1,+1} に変換
    スケジュール (B,96) int -> 拡散空間 (B,12,96) ∈ {-1,+1}
    """
    onehot = F.one_hot(sched, NUM_ACT).float().permute(0, 2, 1)
    return 2.0 * onehot - 1.0


# ============================================================
# 4. Denoiser（1D-UNet; DDPM/model.py と同構造・チャネル数のみ 17->12）
# ============================================================
def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """
    sinusoidal timestep embedding (B,) -> (B, dim)
    dim=128
    """
    half = dim // 2  # sin, cosのために, dimを2分割
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B,128)


class ResBlock1D(nn.Module):
    """
    GroupNorm→SiLU→Conv1d ×2 + timestep/条件埋め込みの加算注入 + skip
    """
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
    """
    時間軸(スロット間)の self-attention + residual
    """
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, ATTN_HEADS, batch_first=True)

    def forward(self, x):
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1)


class UNet1D(nn.Module):
    """
    ε予測ネットワーク: (B,12,96) + 拡散ステップt + 条件cond -> (B,12,96)
    ノイズεの shape は予測するデータ (活動系列) の shape と同じ
    """
    def __init__(self):
        super().__init__()
        c1, c2 = BASE_CH, BASE_CH * 2

        # sinusoidal ベクトル (B,128) を (B,256) に持ち上げる MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(128, TIME_EMB_DIM), 
            nn.SiLU(),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
        )

        # Condition Embedding
        self.cond_embeds = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(COND_CARD, EMB_DIMS)
        ])
        self.cond_proj = nn.Linear(sum(EMB_DIMS), TIME_EMB_DIM)
        self.null_emb = nn.Parameter(torch.zeros(TIME_EMB_DIM))

        # Down h1
        self.in_conv = nn.Conv1d(IN_CH, c1, 3, padding=1)
        self.d1a, self.d1b = ResBlock1D(c1, c1), ResBlock1D(c1, c1)
        self.ds1 = nn.Conv1d(c1, c1, 3, stride=2, padding=1)

        # Down h2
        self.d2a, self.d2b = ResBlock1D(c1, c2), ResBlock1D(c2, c2)
        self.attn2 = AttnBlock1D(c2)
        self.ds2 = nn.Conv1d(c2, c2, 3, stride=2, padding=1)

        # Down h3
        self.d3a, self.d3b = ResBlock1D(c2, c2), ResBlock1D(c2, c2)
        self.attn3 = AttnBlock1D(c2)

        # Bottleneck (middle)
        self.m1, self.m_attn, self.m2 = ResBlock1D(c2, c2), AttnBlock1D(c2), ResBlock1D(c2, c2)

        # Up with h3
        self.u3 = ResBlock1D(c2 + c2, c2)

        # Up with h2
        self.us2 = nn.Conv1d(c2, c2, 3, padding=1)
        self.u2 = ResBlock1D(c2 + c2, c2)
        self.u2_attn = AttnBlock1D(c2)

        # Up with h1
        self.us1 = nn.Conv1d(c2, c1, 3, padding=1)
        self.u1 = ResBlock1D(c1 + c1, c1)

        # 最終出力層
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv1d(c1, IN_CH, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        out_bias = self.out_conv.bias
        if out_bias is not None:
            nn.init.zeros_(out_bias)

    @property
    def in_channels(self) -> int:
        """
        拡散空間のチャネル数
        バックボーン実装に依らない共通の入口

        clock_diagnostics が純ノイズ x_T を作るのに使う。実装内部の層名
        (in_conv 等) に触らせないための薄い契約。
        """
        return IN_CH

    def embed_cond(self, cond_idx, batch: int, drop_mask=None):
        """
        条件 (性・年齢・就業) を256次元のベクトルにまとめる
        """
        if cond_idx is None:
            return self.null_emb.expand(batch, -1)

        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)  # (B,4)⊕(B,8)⊕(B,4)->(B,16)
        c = self.cond_proj(c)  # (B,16) -> (B,256)

        if drop_mask is not None:
            c = torch.where(drop_mask[:, None], self.null_emb.expand_as(c), c)
        return c  # (B, 256)

    def _encode(self, x_t, emb):
        """
        下り経路の中間特徴
        forward と features の共通部分（分岐させない）

        """
        h1 = self.d1b(self.d1a(self.in_conv(x_t), emb), emb)
        h2 = self.attn2(self.d2b(self.d2a(self.ds1(h1), emb), emb))
        h3 = self.attn3(self.d3b(self.d3a(self.ds2(h2), emb), emb))
        return h1, h2, h3

    def features(self, x_t, t, cond_idx=None) -> dict[str, torch.Tensor]:
        """
        中間特徴 {h1:(B,64,96), h2:(B,128,48), h3:(B,128,24)}
        """
        emb = self.time_mlp(timestep_embedding(t)) + self.embed_cond(cond_idx, x_t.size(0))
        h1, h2, h3 = self._encode(x_t, emb)
        return {"h1": h1, "h2": h2, "h3": h3}

    def forward(self, x_t, t, cond_idx=None, drop_mask=None):
        """
        UNetのforward
        ε_θ(x_t, t, c) -> ノイズを予測する
        """
        # Embedding (拡散ステップt + 社会属性条件cond)
        emb = (
            self.time_mlp(timestep_embedding(t)) 
            + self.embed_cond(cond_idx, x_t.size(0), drop_mask)
        )  # (B, 256)

        # Down
        h1, h2, h3 = self._encode(x_t, emb)

        # BottleNeck (middle)
        m = self.m2(self.m_attn(self.m1(h3, emb)), emb)  # Res1D -> Attn -> Res1D

        # Up
        u = self.u3(torch.cat([m, h3], dim=1), emb)
        u = self.us2(F.interpolate(u, scale_factor=2, mode='nearest'))
        u = self.u2_attn(self.u2(torch.cat([u, h2], dim=1), emb))
        u = self.us1(F.interpolate(u, scale_factor=2, mode='nearest'))
        u = self.u1(torch.cat([u, h1], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(u)))


# ============================================================
# 5. Diffusion（forward過程・損失・サンプリング）
# ============================================================
class Diffusion:
    """
    β schedule と派生バッファを事前計算し、q_sample / loss / sample / ddim_sample
    """
    def __init__(self, device=DEVICE):
        """
        (1000,)ベクトル
        args:
            betas: 拡散ステップtにおいてのノイズの強さ
            alphas: 1-betas
            acp: alphaの累積積
            acp_prev: acpの1つずらした
            sprt_acp: √{\bar(α)}
            sqrt_1m_acp: √{1-\bar(α)}
            post_var: 1ステップ前のvar
            post_coef_x0: 
            post_coef_xt:
        """
        betas = torch.linspace(BETA_START, BETA_END, T_STEPS, device=device)  # 0.0001 (t=1) -> .. -> t=T:0.02 (t=T, 1000)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)  # 累積積
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
        self.device = device
        self.betas = betas
        self.alphas = alphas
        self.acp = acp  # \bar(α)
        self.sqrt_acp = acp.sqrt()  # √{\bar(α)}
        self.sqrt_1m_acp = (1.0 - acp).sqrt()  # √{1-\bar(α)}
        self.post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.post_coef_x0 = betas * acp_prev.sqrt() / (1.0 - acp)
        self.post_coef_xt = (1.0 - acp_prev) * alphas.sqrt() / (1.0 - acp)

    def q_sample(self, x0, t, eps):
        """
        x0 から任意のtステップ先の x_t を求める
        x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
        """
        return (self.sqrt_acp[t][:, None, None] * x0
                + self.sqrt_1m_acp[t][:, None, None] * eps)

    def loss(self, model, sched, cond_idx):
        """
        標準 ε 予測 MSE + CFG 条件dropout
        """
        x0 = sched_to_x0(sched)  # (B,96)->(B,12,96)∈{-1,1} 活動系列をonehotにする
        t = torch.randint(0, T_STEPS, (x0.size(0),), device=x0.device)  # t~U{1,T}, T=1000
        eps = torch.randn_like(x0)  # eps~N(0,I), (B,12,96)
        x_t = self.q_sample(x0, t, eps)  # q(x_t|x0), x0からx_tを求める

        drop_mask = torch.rand(x0.size(0), device=x0.device) < P_UNCOND  # CFGの条件付け用

        eps_hat = model(x_t, t, cond_idx, drop_mask)  # 追加されたノイズを予測(ノイズ付きデータx_t, 拡散ステップt, 条件cond_idx)
        return F.mse_loss(eps_hat, eps)  # ノイズ間のMSE

    def _eps(self, model, x, t_scalar, cond_idx, guidance_scale):
        """
        CFG 込みの ε 予測。t_scalar は int
        """
        t = torch.full((x.size(0),), t_scalar, device=x.device, dtype=torch.long)
        eps_c = model(x, t, cond_idx)
        if guidance_scale == 1.0:
            return eps_c
        eps_u = model(x, t, None)
        return eps_u + guidance_scale * (eps_c - eps_u)

    @torch.no_grad()
    def sample(self, model, cond_idx, guidance_scale=GUIDANCE_SCALE, verbose=False):
        """
        ancestral DDPM + CFG
        cond_idx (M,K) -> スケジュール (M,96) int
        """
        model.eval()
        m = cond_idx.size(0)
        
        x = torch.randn(m, IN_CH, NUM_SLOTS, device=cond_idx.device)  # x_T~N(0,I)
        for ti in reversed(range(T_STEPS)):
            eps_hat = self._eps(model, x, ti, cond_idx, guidance_scale)
            x0_hat = (x - self.sqrt_1m_acp[ti] * eps_hat) / self.sqrt_acp[ti]
            x0_hat.clamp_(-1.0, 1.0)
            mean = self.post_coef_x0[ti] * x0_hat + self.post_coef_xt[ti] * x
            x = mean + self.post_var[ti].sqrt() * torch.randn_like(x) if ti > 0 else mean
            if verbose and ti % 200 == 0:
                print(f"  sampling t={ti}")
        return x.argmax(dim=1)

    @torch.no_grad()
    def ddim_sample(self, model, cond_idx, guidance_scale=GUIDANCE_SCALE,
                    steps: int = DDIM_STEPS, eta: float = DDIM_ETA):
        """★DDIM サンプリング (Song et al. 2021)。ancestral の T=1000 を steps 本に間引く。

        Stage2 の指数傾けは群あたり数千本の提案サンプルを要求するため、
        ancestral では現実的な時間で回らない（1本あたり 1000×2 forward）。
        eta=0 なら x_T が決まれば決定的だが、x_T が乱数なので個票の多様性は保たれる。
        eta=1 で ancestral 相当の確率的な逆過程になる。
        """
        model.eval()
        m = cond_idx.size(0)
        # t の間引き列 (昇順) と、その1つ前 (t_prev; 先頭は -1 = x0 に対応)
        ts = np.linspace(0, T_STEPS - 1, steps).round().astype(int)[::-1]
        x = torch.randn(m, IN_CH, NUM_SLOTS, device=cond_idx.device)
        for i, ti in enumerate(ts):
            eps_hat = self._eps(model, x, int(ti), cond_idx, guidance_scale)
            a_t = self.acp[int(ti)]
            x0_hat = ((x - (1 - a_t).sqrt() * eps_hat) / a_t.sqrt()).clamp_(-1.0, 1.0)
            if i == len(ts) - 1:
                x = x0_hat
                break
            a_prev = self.acp[int(ts[i + 1])]
            # σ_t: eta=0 で 0（決定的）, eta=1 で ancestral の後方分散に一致
            sigma = eta * (((1 - a_prev) / (1 - a_t)) * (1 - a_t / a_prev)).sqrt()
            # x_prev = √ᾱ_prev·x0_hat + 「x_t 方向」の残り + σ·z
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps_hat
            x = a_prev.sqrt() * x0_hat + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x.argmax(dim=1)


# ============================================================
# 6. EMA
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
# 7. 学習
# ============================================================
def run_epoch(model, diffusion: Diffusion, loader, optimizer=None, ema=None):
    """
    1エポック分の学習または評価を実行し、平均 ε-MSE を返す。
    """
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


def train(epochs: int = EPOCHS,
            use_wandb: bool = True,
            save_path: Path | None = MODEL_SAVE_PATH,
            model_factory: Callable[[], nn.Module] = UNet1D,
            backbone: str = "unet1d",
            extra_config: dict | None = None
        ):
    """
    save_path=None なら保存しない（--smoke が本番チェックポイントを潰さないため）

    ★ model_factory / backbone / extra_config は DDPM_Aggregate_Tang が同じ学習手順を
        別バックボーンで再利用するための口。既定値は現行のままなので挙動は変わらない。
        学習ループを共有することで、2つのバックボーンの差が「構造の差」だけに帰着する。
    """
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project='domain-transfer-ddpm-agg',
            config={
                "backbone": backbone,
                **(extra_config or {}),
                "t_steps": T_STEPS, "beta_start": BETA_START, "beta_end": BETA_END,
                "base_ch": BASE_CH, "dropout": DROPOUT, "time_emb_dim": TIME_EMB_DIM,
                "cond_spec": [(n, c, d) for n, c, d, _ in COND_SPEC],
                "p_uncond": P_UNCOND, "guidance_scale": GUIDANCE_SCALE,
                "batch_size": BATCH_SIZE, "lr": LR, "epochs": epochs,
                "ema_decay": EMA_DECAY, "weighted_sampler": USE_WEIGHTED_SAMPLER,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
                "day_filter": DAY_FILTER, "num_act": NUM_ACT, "d_groups": D_GROUPS,
                "data": DATA_PATH.name,
            }
        )

    torch.manual_seed(SEED)
    cond_idx, sched, weight, _ = load_data(DATA_PATH)
    train_loader, val_loader = make_loaders(cond_idx, sched, weight)

    model = model_factory().to(DEVICE)  # UNet
    diffusion = Diffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    ema = EMA(model)
    print(f"device={DEVICE}  N={len(sched)}  params={sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    ep = 0
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, diffusion, train_loader, optimizer, ema)
        va = run_epoch(model, diffusion, val_loader)
        if ep % 25 == 0 or ep == 1:
            print(f"epoch {ep:4d} | train {tr:.4f} | val {va:.4f}")
        if run is not None:
            run.log({"epoch": ep, "train/loss": tr, "val/loss": va})

        if va < best_val - EARLY_STOP_MIN_DELTA:
            best_val = va
            epochs_no_improve = 0
            best_state = {"model": copy.deepcopy(model.state_dict()),
                            "ema": copy.deepcopy(ema.shadow), "epoch": ep}
        else:
            epochs_no_improve += 1
            if 0 < EARLY_STOP_PATIENCE <= epochs_no_improve:
                best_ep = best_state["epoch"] if best_state else ep
                print(f"early stopping at epoch {ep} "
                        f"(no val improvement for {epochs_no_improve} epochs; "
                        f"best {best_val:.4f} @ epoch {best_ep})")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        ema.shadow = best_state["ema"]
        print(f"restored best checkpoint: epoch {best_state['epoch']} (val {best_val:.4f})")

    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.summary["best_epoch"] = best_state["epoch"] if best_state else ep
        run.summary["stopped_epoch"] = ep

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.shadow}, save_path)
        print(f"saved model (raw + ema) to {save_path}")
    if run is not None:
        run.finish()

    ema.copy_to(model)   # 以降 (生成・評価) は EMA 重み
    return model


def load_pretrained(path: Path = MODEL_SAVE_PATH, use_ema: bool = True,
                    model_factory: Callable[[], nn.Module] = UNet1D) -> nn.Module:
    """保存済み Stage1 を読み込む（既定は EMA 重み）。"""
    ckpt = torch.load(path, map_location=DEVICE)
    model = model_factory().to(DEVICE)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    model.eval()
    return model


# ============================================================
# 8. 群レベル生成 ★（Stage2 の提案分布 / 評価用）
# ============================================================
@torch.no_grad()
def group_pool(model, n_per_group: int, guidance_scale: float = GUIDANCE_SCALE,
                sampler: str = SAMPLER, ddim_steps: int = DDIM_STEPS, eta: float = DDIM_ETA,
                verbose: bool = True) -> npt.NDArray[np.int64]:
    """★群別サンプルプール (D, M, 96)。行 d は cond_grid()[d] の条件で生成。

    Stage2 の指数傾けはこのプールを提案分布 p_d として重み付けするので、
    M が小さいと傾け後の有効サンプル数 (ESS) が枯れる。M は数千を想定。
    sampler の選択根拠は SAMPLER の定義箇所を参照（既定 'ancestral'）。
    """
    diffusion = Diffusion()
    grid = torch.as_tensor(cond_grid(), device=DEVICE)
    # (群, サンプル) を平坦化してからチャンクする。群ごとに切ると端数バッチが増えて
    # 逆過程 (1000ステップ) の呼び出し効率が落ちるため
    flat = grid.repeat_interleave(n_per_group, dim=0)              # (D*M, 3)
    total = flat.size(0)
    outs = []
    for i in range(0, total, GEN_BATCH):
        ci = flat[i:i + GEN_BATCH].contiguous()
        s = (diffusion.sample(model, ci, guidance_scale) if sampler == 'ancestral'
                else diffusion.ddim_sample(model, ci, guidance_scale, ddim_steps, eta))
        outs.append(s.cpu().numpy())
        if verbose:
            print(f"  pooled {min(i + GEN_BATCH, total)}/{total} ({sampler})", flush=True)
    return np.concatenate(outs, axis=0).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


def pool_to_rates(pool: npt.NDArray[np.int64],
                    weights: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """★サンプルプール -> 群別期待行動者率 (D, n_act*96) act-major。

    CVAE_Aggregate.model.group_rates と同一形式（インデックス a*96+t）で返すので、
    japan_match_experiment.eval_against にそのまま渡せる。

    weights: (D, M) の非負重み（指数傾けの結果）。None なら一様（= zero-shot）。
    """
    D, M, T = pool.shape
    onehot = np.eye(NUM_ACT, dtype=np.float64)[pool]        # (D,M,96,n_act)
    if weights is None:
        rates = onehot.mean(axis=1)                          # (D,96,n_act)
    else:
        w = weights / weights.sum(axis=1, keepdims=True)
        rates = np.einsum("dm,dmta->dta", w, onehot)
    return rates.transpose(0, 2, 1).reshape(D, NUM_ACT * T)  # act-major


# ============================================================
# 9. サニティチェック ★（断片化が実データ水準か = Route B の前提条件）
# ============================================================
def fragmentation_stats(sched: npt.NDArray[np.int64],
                        w: npt.NDArray[np.float64] | None = None) -> dict:
    """切替回数・エピソード長・日次参加率。individual_metrics へ委譲する。

    ★ 旧実装はエピソード長を sched[:3000] で打ち切っていた。56,000行のプール
        (28群 × M=2000) に使うと **d=0,1（男性15-24歳）の2群しか見ない**ため、
        japan_match_experiment の「1スロットep比率」が zero-shot と tilted で
        同値になっていた。打ち切りは撤廃した。

    ★ 返り値のキーは呼び出し側 (sanity_check / japan_match_experiment) との
        互換のため変えない。wrap_closure_rate は新規に追加した分。

    ★ エピソード長系 (ep_len_*, single_slot_ratio) は個票を重み付けしない。
        individual_metrics.episode_lengths が全個票のエピソードを連結するため
        （重み付き分位の定義が一意でない）。切替回数と参加率は重み付き。
    """
    s = im.fragmentation_summary(sched, w)
    return {
        "switches": s["switch_mean"],
        "ep_len_median": s["ep_len_median"],
        "ep_len_mean": s["ep_len_mean"],
        "single_slot_ratio": s["single_slot_ratio"],
        "wrap_closure_rate": s["wrap_closure_rate"],
        "participation": im.participation(sched, NUM_ACT, w),
    }


def memorization_report(gen: npt.NDArray[np.int64], sched_real: npt.NDArray[np.int64],
                        sample: int = 2000, seed: int = 0, n_null: int = 5) -> dict:
    """★生成個票が「学習個票のコピー」になっていないかを表で出す。

    ★ なぜ必要か:
        val loss の early stopping は「平均的に効いている過学習」しか止めない。
        小データの拡散モデルは学習個票をそのまま再生する形で暗記しうる
        （CVAE_Aggregate の decoder で実際に起きた故障モード）。集計指標は
        暗記したモデルでこそ良く見えるので、集計だけを見ていると気づけない。

    ★ 読み方:
        判定は「gen が train **にだけ** 近いか」。train と holdout に同じだけ近いなら、
        それはデータ分布に近いだけで暗記ではない。見るべきは DCR_gap が 0 付近か
        どうかであって、DCR の絶対値ではない。

    ★ 参照集合のサイズを揃える（これをしないと指標が読めない）:
        最近傍距離は参照集合が大きいほど自然に小さくなる。train は holdout の
        約9倍あるので、素で比べると暗記が無くても DCR_gap が正に出る。
        実測: 実ホールドアウト個票（暗記があり得ない）の DCR は
                train全体(N=3363) に対し 21.83、サイズを揃えた train(N=373) に対し 26.54。
                サイズ差だけで 4.7 スロットずれる。
        よって DCR/NNDR は train を holdout と同数に間引いてから比べる。

    ★ 床（帰無帯）を併記する:
        「gap が 0 でない」だけでは暗記の証拠にならない。train から互いに素な
        同数の部分集合 A, B を取って gap を測ると、暗記が原理的にありえない状況での
        ばらつき（＝床）が得られる。実測 gap がこの幅に収まっていれば暗記なし。
        この考え方は individual_metrics.null_band と同じ。

    ★ exact/near copy は train **全体** に対して測る。「モデルが見た個票のどれかを
        そのまま出したか」が問いなので、ここは間引いてはいけない。

    ★ 学習/ホールドアウトの分割は split_indices（学習時と同一）から取る。
    """
    train_idx, val_idx = split_indices(len(sched_real))
    train, holdout = sched_real[train_idx], sched_real[val_idx]
    n_ref = len(holdout)
    rng = np.random.default_rng(seed)

    # サイズを揃えた比較（DCR/NNDR の gap 判定用）
    train_sub = train[rng.choice(len(train), n_ref, replace=False)]
    m = im.memorization(gen, train_sub, holdout, sample=sample, seed=seed)
    gap = m["DCR_gap(holdout-train)"]

    # 床: train 内の互いに素な同数部分集合 A, B での gap のばらつき
    nulls = []
    for i in range(n_null):
        pick = rng.choice(len(train), 2 * n_ref, replace=False)
        a, b = train[pick[:n_ref]], train[pick[n_ref:]]
        d_a = im.nn_distances(gen, a, k=1, sample=sample, seed=seed)[:, 0]
        d_b = im.nn_distances(gen, b, k=1, sample=sample, seed=seed)[:, 0]
        nulls.append(float(d_b.mean() - d_a.mean()))
    lo, hi = min(nulls), max(nulls)

    # exact/near copy は train 全体に対して（間引かない）
    full = im.memorization(gen, train, sample=sample, seed=seed)

    print(f"\n--- 暗記チェック (train N={len(train)} / holdout N={n_ref}; "
            f"DCR系は train を N={n_ref} に間引いて比較) ---")
    print(f"{'指標':<28}{'train':>10}{'holdout':>10}")
    for key in ["DCR_mean", "DCR_p05", "DCR_median", "NNDR_mean", "NNDR_p05"]:
        print(f"{key:<28}{m[f'{key}[train]']:>10.4f}{m[f'{key}[holdout]']:>10.4f}")
    print(f"{'exact_copy_rate[train全体]':<28}{full['exact_copy_rate[train]']:>10.4f}")
    print(f"{'near_copy_rate(<=2)[train全体]':<28}{full['near_copy_rate(<=2)[train]']:>10.4f}")
    # ★判定は片側。gap が床より「上」= train にだけ近い = 暗記。
    #   下に外れるのは train より holdout に近いという意味で、暗記ではない
    #   （生成物がホールドアウト個票そのものである場合など）。
    if gap > hi:
        verdict = "★床の外（上）＝暗記の疑い"
    elif gap < lo:
        verdict = "床の外（下）＝train より holdout に近い。暗記ではない"
    else:
        verdict = "暗記なし"
    print(f"{'DCR_gap(holdout-train)':<28}{gap:>10.4f}   床[{lo:+.4f}, {hi:+.4f}]  {verdict}")

    m.update({"DCR_gap_null_lo": lo, "DCR_gap_null_hi": hi,
                "memorized": gap > hi,
                "exact_copy_rate[train_full]": full["exact_copy_rate[train]"],
                "near_copy_rate(<=2)[train_full]": full["near_copy_rate(<=2)[train]"]})
    return m


def sanity_check(model, n_per_group: int = 256, sampler: str = SAMPLER,
                ddim_steps: int = DDIM_STEPS, save_path: Path | None = GEN_SAVE_PATH):
    """実 ATUS 平日と同一群構成で生成し、断片化と活動シェアを比較する。

    ★ save_path=None なら CSV を書かない。--smoke（5エポック学習・DDIM 10ステップ）が
        本番の生成CSVを潰さないため。train() の save_path と同じ考え方。
    """
    cond_idx, sched_real, w_real, _ = load_data(DATA_PATH)
    d_real = cond_to_d(cond_idx)

    print(f"\n群別サンプルプール生成 (D={D_GROUPS} × M={n_per_group}, sampler={sampler}) ...")
    # ancestral は1バッチあたり 1000ステップ×2(CFG) の前向き計算で数分かかる。
    # 無言で待たせないようバッチ進捗を出す
    pool = group_pool(model, n_per_group, sampler=sampler, ddim_steps=ddim_steps, verbose=True)
    gen = pool.reshape(-1, NUM_SLOTS)

    # 実データの群構成に合わせた生成側の重み。プールは群一様なので、群別の
    # 「調査ウェイト加重シェア ÷ 群内本数」を各行へ配る。
    # ★ 非加重の人数比 (np.bincount(d_real)/N) ではない。両者は ATUS 平日で
    #   総変動距離 0.139 ずれ、ここで使い分けると学習ログのサニティと
    #   clock_diagnostics の数値が食い違う。重みの出所は group_reweight ただ一つ。
    gen_d = np.repeat(np.arange(D_GROUPS), n_per_group)
    w_gen = im.group_reweight(gen_d, w_real, d_real, D_GROUPS)

    r = fragmentation_stats(sched_real, w_real)
    g = fragmentation_stats(gen, w_gen)

    print("\n--- 断片化サニティ (実 ATUS 平日 vs AggDDPM 生成) ---")
    print(f"{'指標':<22}{'real':>10}{'gen':>10}{'比':>8}")
    for k in ["switches", "ep_len_median", "ep_len_mean", "single_slot_ratio",
                "wrap_closure_rate"]:
        print(f"{k:<22}{r[k]:>10.3f}{g[k]:>10.3f}{g[k] / max(r[k], 1e-9):>8.2f}")
    print(f"{'mean participation':<22}{r['participation'].mean():>10.3f}"
            f"{g['participation'].mean():>10.3f}"
            f"{g['participation'].mean() / r['participation'].mean():>8.2f}")

    print(f"\n{'activity':<18}{'part_real':>10}{'part_gen':>10}{'share_real':>12}{'share_gen':>11}")
    sr = np.array([(w_real / w_real.sum())[:, None].repeat(NUM_SLOTS, 1)[sched_real == a].sum()
                    for a in range(NUM_ACT)])
    sg = np.array([w_gen[:, None].repeat(NUM_SLOTS, 1)[gen == a].sum() for a in range(NUM_ACT)])
    sr, sg = sr / NUM_SLOTS, sg / NUM_SLOTS
    for a in range(NUM_ACT):
        print(f"{ACT_NAMES[a]:<18}{r['participation'][a]:>10.3f}{g['participation'][a]:>10.3f}"
                f"{sr[a]:>12.4f}{sg[a]:>11.4f}")

    memorization_report(gen, sched_real)

    if save_path is None:
        print("\n(save_path=None のため生成CSVは書かない)")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid = cond_grid()
    meta = pd.DataFrame(np.repeat(grid, n_per_group, axis=0), columns=["gender", "age7", "employment"])
    meta.insert(0, "group_d", gen_d)   # w_gen と同一の群割り当て（ずれ得ない）
    # ★サンプラをCSVに残す。どの逆過程で作った個票かが後から判別できないと、
    #   断片化の数値（ancestral 12.87 vs DDIM 17.70）を取り違える
    meta.insert(1, "sampler", sampler if sampler == 'ancestral' else f"ddim{ddim_steps}_eta{DDIM_ETA}")
    pd.concat([meta, pd.DataFrame(gen, columns=[f"s{i}" for i in range(NUM_SLOTS)])],
                axis=1).to_csv(save_path, index=False)
    print(f"\nsaved generated schedules to {save_path}")


# ============================================================
# 10. スモークテスト
# ============================================================
def smoke_test():
    """学習前に必ず通す形状・整合チェック"""
    m = UNet1D().to(DEVICE)
    d = Diffusion()
    x = torch.randn(4, IN_CH, NUM_SLOTS, device=DEVICE)
    t = torch.randint(0, T_STEPS, (4,), device=DEVICE)
    c = torch.zeros(4, len(COND_SPEC), dtype=torch.long, device=DEVICE)
    assert m(x, t, c).shape == x.shape
    assert m(x, t, None).shape == x.shape                     # 無条件 (CFG) 経路
    x0 = sched_to_x0(torch.zeros(4, NUM_SLOTS, dtype=torch.long, device=DEVICE))
    t0 = torch.zeros(4, dtype=torch.long, device=DEVICE)
    assert (d.q_sample(x0, t0, torch.zeros_like(x0)) - x0).abs().max() < 1e-4

    # ★群インデックスの往復: cond_grid の行 d が d に戻ること
    grid = cond_grid()
    assert (cond_to_d(grid) == np.arange(D_GROUPS)).all(), "cond_grid と d_index の対応がずれている"

    # ★実データの条件が全群を覆い、d_index が japan_match と一致すること
    cond_idx, sched, _, _ = load_data(DATA_PATH)
    assert cond_idx[:, 0].max() < N_G and cond_idx[:, 1].max() < N_A and cond_idx[:, 2].max() < N_E
    assert sched.min() >= 0 and sched.max() < NUM_ACT

    # ★pool_to_rates が act-major (a*96+t) で、各スロットの活動確率が和1になること
    fake = np.random.default_rng(0).integers(0, NUM_ACT, size=(D_GROUPS, 32, NUM_SLOTS))
    rates = pool_to_rates(fake).reshape(D_GROUPS, NUM_ACT, NUM_SLOTS)
    assert np.allclose(rates.sum(axis=1), 1.0), "pool_to_rates の正規化が壊れている"
    # 重み付き版も同じ形になること
    w = np.random.default_rng(1).random((D_GROUPS, 32))
    assert np.allclose(pool_to_rates(fake, w).reshape(D_GROUPS, NUM_ACT, NUM_SLOTS).sum(1), 1.0)

    # ★fragmentation_stats が先頭3000行で打ち切られていないこと。
    #   先頭3000行を「終日1活動」、後続1000行を「毎スロット切替」にすると、
    #   打ち切りがあれば single_slot_ratio は 0 のままになる
    trunc = np.zeros((4000, NUM_SLOTS), dtype=np.int64)
    trunc[3000:] = np.arange(NUM_SLOTS) % 2
    fs = fragmentation_stats(trunc)
    assert fs["single_slot_ratio"] > 0.9, \
        f"fragmentation_stats が打ち切られている (single_slot_ratio={fs['single_slot_ratio']})"
    assert abs(fs["switches"] - 0.25 * (NUM_SLOTS - 1)) < 1e-9   # 1000/4000 行 × 95境界

    # ★split_indices が全行を過不足なく2分すること（暗記チェックが学習集合と
    #   ホールドアウトを取り違えると、判定の向きが逆になる）
    tr_i, va_i = split_indices(len(sched))
    assert len(tr_i) + len(va_i) == len(sched)
    assert set(tr_i.tolist()).isdisjoint(va_i.tolist())
    assert len(va_i) == int(len(sched) * VAL_RATIO)

    assert m.in_channels == IN_CH

    # ★DDIM が ancestral と同じ形・値域を返すこと
    ci = torch.as_tensor(grid[:4], device=DEVICE)
    s_ddim = d.ddim_sample(m, ci, guidance_scale=1.0, steps=5)
    assert s_ddim.shape == (4, NUM_SLOTS) and int(s_ddim.max()) < NUM_ACT
    print(f"smoke test: OK  (N={len(sched)}, D={D_GROUPS}, n_act={NUM_ACT}, "
            f"params={sum(p.numel() for p in m.parameters()):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AggDDPM: ATUS平日・共通12分類・28群の条件付き pretrain")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="短時間の動作確認のみ")
    args = ap.parse_args()

    smoke_test()
    if args.smoke:
        model = train(epochs=5, use_wandb=False, save_path=None)
        sanity_check(model, n_per_group=16, sampler='ddim', ddim_steps=10, save_path=None)
    else:
        model = train(epochs=args.epochs, use_wandb=not args.no_wandb)
        sanity_check(model)
