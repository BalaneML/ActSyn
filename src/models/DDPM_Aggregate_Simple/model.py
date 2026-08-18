"""
model.py
================
集計マッチ転移のための条件付きDDPM（AggDDPM）— 簡素化版

DDPM_Aggregate/model.py から意図的に4点を削った独立実装である。
このファイルは DDPM_Aggregate を import しない（完全に自己完結）。共通部分を写している
ぶん、DDPM_Aggregate 側を直してもここには伝播しない。両方を直す必要がある。

DDPM_Aggregate/model.py との差分（★が本モデルの変更点）:
    1. データ表現 : ★one-hot を {-1,+1} でなく {0,1} のまま拡散空間に載せる
                    sched_to_x0 と sample の clamp 範囲が対応して変わる
    2. サンプリング: ★DDIM を持たない。ancestral のみ
                    SAMPLER / DDIM_STEPS / DDIM_ETA と、それに付随する引数を削除
    3. 時刻埋め込み: ★sinusoidal を MLP で持ち上げず、256次元を直接足す
                    time_mlp (98,816 params) を削除
    4. 学習       : ★EMA を持たない。学習後の重みをそのまま生成に使う

変更1の測定済みの代償（採用前に確認した数値。設計判断の記録として残す）:
    正解チャネルと不正解チャネルの差（分離幅）は {-1,+1} で 2、{0,1} で 1 になる。
    分離/ノイズ比 = gap·√(ᾱ_t/(1-ᾱ_t)) が 1 を下回る境界は
        {-1,+1}: t = 395（全1000ステップの39.5%で1スロット単独判別が可能）
        {0,1}  : t = 258（同 25.8%）
    判別可能な区間が 4 割から 2.6 割に減るので、断片化（本研究の主指標）が
    悪化しうる。一方 12分類 one-hot の要素平均は {-1,+1} で -0.833、{0,1} で +0.083 で、
    事前分布 N(0,I) との平均のずれは {0,1} の方が小さい。両者は一長一短であり、
    どちらが良いかは実測で決める。比較相手は DDPM_Aggregate（同一の学習ループ）。

変更4について:
    EMA は拡散モデルで広く使われる。削除の影響は未測定である。
    なお DDPM_Aggregate では val 損失を raw 重みで測って early stopping し、
    生成には EMA 重みを使うという不整合があった。本実装では生成に使う重みが
    そのまま val で選ばれるので、選択指標と生成重みは一致する。

out_conv のゼロ初期化は残してある（DDPM_Aggregate と同じ）。
100エポック×2シードの A/B で、既定初期化より val ε-MSE が良かったため:
    epoch   5: ゼロ初期化 0.643 / 既定 0.335   ← 立ち上がりは不利
    epoch 100: ゼロ初期化 0.043 / 既定 0.046   ← 収束側で逆転（差 +0.0031 ± 0.0008）

使い方:
    # Stage1: ATUS 平日・共通12分類・28群で条件付き pretrain
    uv run python src/models/DDPM_Aggregate_Simple/model.py

    # 短時間の動作確認（学習 5 エポック・群あたり 2 本だけ生成）
    uv run python src/models/DDPM_Aggregate_Simple/model.py --smoke

    フラグ:
        --epochs N   : 学習エポック数を上書き
        --no-wandb   : wandb ログを無効化
        --smoke      : 形状・整合の確認だけを短時間で回す

出力:
    outputs/checkpoints/ddpm_simple_pretrain_common12_weekday.pt   Stage1 の重み
    outputs/generated/ddpm_simple_pretrain_samples.csv             サニティ用の生成個票
"""
import argparse
import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/DDPM_Aggregate_Simple -> repo root
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


# 個票指標の唯一の出所。断片化統計はここへ委譲する（下の fragmentation_stats 参照）。
# src/eval は全モデル共通の評価ライブラリなので、自己完結の対象外とする
im = _load_module("simple_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")

# ============================================================
# 1. 設定（ハイパーパラメータ）
# ============================================================
DATA_PATH       = REPO_ROOT / 'data' / 'processed' / 'atus2024' / 'atus2024_stula_common12_dataset.csv'
MODEL_SAVE_PATH = REPO_ROOT / 'outputs' / 'checkpoints' / 'ddpm_simple_pretrain_common12_weekday.pt'
GEN_SAVE_PATH   = REPO_ROOT / 'outputs' / 'generated' / 'ddpm_simple_pretrain_samples.csv'

# 活動スケジュール
NUM_SLOTS   = 96                 # 15分刻み × 96 = 24時間（04:00開始）
NUM_ACT     = NUM_COMMON         # 共通12分類（OTHER_X を含む; 除外は教師・評価側の責務）
ACT_NAMES   = [c.name for c in Common]
IN_CH       = NUM_ACT            # 拡散空間のチャネル数

# 群定義: d = g*(N_A*N_E) + a*N_E + e。CVAE_Aggregate/japan_match_experiment と同一
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E       # 28

# 条件属性: (CSV列名, カテゴリ数, 埋め込み次元, インデックス化関数)
COND_SPEC = [
    ("gender", N_G, 4, lambda v: v.astype(np.int64)),                       # 0=男 1=女
    ("age",    N_A, 8, lambda v: np.clip((v - 15) // 10, 0, 6)),            # 15歳起点10歳刻み7区分
    ("telfs",  N_E, 4, lambda v: np.isin(v, [1, 2]).astype(np.int64)),      # 0=無業 1=有業
]
COND_COLS  = [name for name, _, _, _ in COND_SPEC]
COND_CARD  = [card for _, card, _, _ in COND_SPEC]  # (gender->2, age->7, telfs->2)
EMB_DIMS   = [dim for _, _, dim, _ in COND_SPEC]

DAY_FILTER  = 'weekday'   # 平日固定（土日への拡張は daytype 条件化として将来課題）

# DDPM
T_STEPS     = 1000
BETA_START  = 1e-4        # 0.0001 (t=1) -> .. -> t=T:0.02 (t=T, 1000)
BETA_END    = 0.02        # linear schedule (Ho et al. 2020 / Tang et al. 2025 準拠)

# Denoiser (1D-UNet)
BASE_CH     = 64
DROPOUT     = 0.1         # 小データ(平日 ~3.7k)の過学習対策
ATTN_HEADS  = 4
# ★時刻埋め込みの次元。sinusoidal をこの次元で直接作り、MLP を通さずに足す。
#   条件埋め込み (cond_proj) の出力次元と null_emb の次元もこれに揃う
TIME_EMB_DIM = 256

# Classifier-Free Guidance
P_UNCOND       = 0.1
# 2.0 (DDPM/model.py の既定) から 1.25 へ。DDPM_Aggregate と同値に揃えてある。
#   ★この値の根拠は DDPM_Aggregate 側で測られたもので、2つの数値が別々の重み付けで
#     取られていることが判明している（WORK シェアは非加重の人数比、Σ|Δ| は調査ウェイト加重）。
#     調査ウェイト加重で測り直すと s=1.25 の WORK シェアは 0.185（実 0.168）で、
#     「実データにぴったり合う」という根拠は成立しない。Σ|Δ| の側（1.0 で 0.061、
#     1.25 で 0.044、1.5 で 0.174）は M=128 のMCノイズ sd 0.0035 の外で有効。
#     本モデルではデータ表現が変わるので、いずれ再掃引が必要になる
GUIDANCE_SCALE = 1.25

# 学習
BATCH_SIZE  = 256
# 平日のみで N=3,736 -> 3,736×0.9/256 ≈ 13 step/epoch
EPOCHS      = 1000
LR          = 2e-4  # 0.0002
VAL_RATIO   = 0.1
SEED        = 42
USE_WEIGHTED_SAMPLER = True
WEIGHT_COL  = "TUFINLWGT"
EARLY_STOP_PATIENCE  = 200
EARLY_STOP_MIN_DELTA = 1e-4
GEN_BATCH   = 1024

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


# ============================================================
# 2. 群インデックスと条件の相互変換
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

    # 平日 (月..金)。japan_match_experiment と同じ between(2,6) で揃える
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

    暗記チェック（individual_metrics.memorization）が
    「生成物が学習集合にだけ近いか」を測るには、学習に使った行と使わなかった行を
    同じ規則で再現する必要がある。分割規則を2箇所に書くと静かにずれるので、
    唯一の出所をここに置く。
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
    活動スケジュール (index表現) を onehot~{0,1} に変換
    スケジュール (B,96) int -> 拡散空間 (B,12,96) ∈ {0,1}

    ★DDPM_Aggregate は 2*onehot-1 で {-1,+1} に写す。本実装は写さない。
      値域が変わるので、逆過程の clamp も [0,1] に揃える（Diffusion.sample 参照）。
      デコードは argmax なので、この変更でも復元規則は変わらない。
    """
    return F.one_hot(sched, NUM_ACT).float().permute(0, 2, 1)


# ============================================================
# 4. Denoiser（1D-UNet）
# ============================================================
def timestep_embedding(t: torch.Tensor, dim: int = TIME_EMB_DIM) -> torch.Tensor:
    """
    sinusoidal timestep embedding (B,) -> (B, dim)

    ★DDPM_Aggregate は 128次元で作ってから MLP (Linear-SiLU-Linear) で 256次元へ
      持ち上げるが、本実装は最初から 256次元で作って直接足す。
      各 ResBlock1D の emb_proj が線形写像なので、時刻条件は
      「256次元フーリエ基底の線形読み出し」として残る。失うのは全ブロックで
      共有される非線形処理の分だけ。削減は 98,816 params。
    """
    half = dim // 2  # sin, cosのために, dimを2分割
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, dim)


class ResBlock1D(nn.Module):
    """
    GroupNorm→SiLU→Conv1d ×2 + timestep/条件埋め込みの加算注入 + skip

    Norm→Act→Conv の順序（pre-activation）は Ho et al. 2020 の resnet_block と同じ。
    残差経路が純粋な恒等写像になり、出口のゼロ初期化が成立する前提でもある。
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

    ★DDPM_Aggregate との差は time_mlp を持たないことだけ。
      それ以外の層構成・チャネル数・attention の位置は同一に保つ
      （比較したときの差が「時刻埋め込みの処理」だけに帰着するようにするため）。
    """
    def __init__(self):
        super().__init__()
        c1, c2 = BASE_CH, BASE_CH * 2

        # ★time_mlp は持たない。timestep_embedding(t) を直接 emb に足す

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

        # 最終出力層。ゼロ初期化により学習開始時の ε̂ が恒等的に 0 になる。
        # ε̂=0 は E[ε]=0 より「最適な定数予測器」であり、A/B で収束後の val が良かった
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

        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)  # (B,16)
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
        emb = timestep_embedding(t) + self.embed_cond(cond_idx, x_t.size(0))
        h1, h2, h3 = self._encode(x_t, emb)
        return {"h1": h1, "h2": h2, "h3": h3}

    def forward(self, x_t, t, cond_idx=None, drop_mask=None):
        """
        UNetのforward
        ε_θ(x_t, t, c) -> ノイズを予測する
        """
        # Embedding (拡散ステップt + 社会属性条件cond)
        # ★sinusoidal をそのまま足す（MLP を通さない）
        emb = timestep_embedding(t) + self.embed_cond(cond_idx, x_t.size(0), drop_mask)  # (B, 256)

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
    β schedule と派生バッファを事前計算し、q_sample / loss / sample を提供する

    ★DDPM_Aggregate との差は2点:
        - sample の clamp が [0,1]（データ表現が {0,1} なので）
        - ddim_sample を持たない
    """
    def __init__(self, device=DEVICE):
        """
        (1000,)ベクトル
        args:
            betas: 拡散ステップtにおいてのノイズの強さ
            alphas: 1-betas
            acp: alphaの累積積
            acp_prev: acpの1つずらした
            sqrt_acp: √{\\bar(α)}
            sqrt_1m_acp: √{1-\\bar(α)}
            post_var: 1ステップ前のvar
            post_coef_x0 / post_coef_xt: 後方平均の係数
        """
        betas = torch.linspace(BETA_START, BETA_END, T_STEPS, device=device)
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

        ★データ表現が {0,1} に変わっても ε ~ N(0,I) は変わらないので、
          損失の形も out_conv のゼロ初期化の意味も変わらない
        """
        x0 = sched_to_x0(sched)  # (B,96)->(B,12,96)∈{0,1}
        t = torch.randint(0, T_STEPS, (x0.size(0),), device=x0.device)  # t~U{0,T-1}
        eps = torch.randn_like(x0)  # eps~N(0,I), (B,12,96)
        x_t = self.q_sample(x0, t, eps)  # q(x_t|x0)

        drop_mask = torch.rand(x0.size(0), device=x0.device) < P_UNCOND  # CFGの条件dropout

        eps_hat = model(x_t, t, cond_idx, drop_mask)
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

        ★clamp は [0,1]。データ表現 {0,1} に合わせてある
          （DDPM_Aggregate は表現が {-1,+1} なので [-1,1]）
        """
        model.eval()
        m = cond_idx.size(0)

        x = torch.randn(m, IN_CH, NUM_SLOTS, device=cond_idx.device)  # x_T~N(0,I)
        for ti in reversed(range(T_STEPS)):
            eps_hat = self._eps(model, x, ti, cond_idx, guidance_scale)
            x0_hat = (x - self.sqrt_1m_acp[ti] * eps_hat) / self.sqrt_acp[ti]
            x0_hat.clamp_(0.0, 1.0)
            mean = self.post_coef_x0[ti] * x0_hat + self.post_coef_xt[ti] * x
            x = mean + self.post_var[ti].sqrt() * torch.randn_like(x) if ti > 0 else mean
            if verbose and ti % 200 == 0:
                print(f"  sampling t={ti}")
        return x.argmax(dim=1)


# ============================================================
# 6. 学習
# ============================================================
def run_epoch(model, diffusion: Diffusion, loader, optimizer=None):
    """
    1エポック分の学習または評価を実行し、平均 ε-MSE を返す。

    ★EMA を持たないので ema 引数は無い
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
            bs = sched.size(0)
            sum_loss += loss.item() * bs
            n_samples += bs
    return sum_loss / n_samples


def train(epochs: int = EPOCHS,
            use_wandb: bool = True,
            save_path: Path | None = MODEL_SAVE_PATH):
    """
    save_path=None なら保存しない（--smoke が本番チェックポイントを潰さないため）

    ★EMA が無いので、val 損失で選ばれた重みがそのまま生成に使われる。
      DDPM_Aggregate にあった「val は raw・生成は EMA」という不整合は無い
    """
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project='domain-transfer-ddpm-agg',
            config={
                "backbone": "unet1d_simple",
                "x0_encoding": "onehot01",       # ★{-1,+1} でなく {0,1}
                "time_embedding": "sinusoidal_direct",  # ★MLP を通さない
                "sampler": "ancestral",          # ★DDIM は持たない
                "ema": False,                    # ★EMA を持たない
                "t_steps": T_STEPS, "beta_start": BETA_START, "beta_end": BETA_END,
                "base_ch": BASE_CH, "dropout": DROPOUT, "time_emb_dim": TIME_EMB_DIM,
                "cond_spec": [(n, c, d) for n, c, d, _ in COND_SPEC],
                "p_uncond": P_UNCOND, "guidance_scale": GUIDANCE_SCALE,
                "batch_size": BATCH_SIZE, "lr": LR, "epochs": epochs,
                "weighted_sampler": USE_WEIGHTED_SAMPLER,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
                "day_filter": DAY_FILTER, "num_act": NUM_ACT, "d_groups": D_GROUPS,
                "data": DATA_PATH.name,
            }
        )

    torch.manual_seed(SEED)
    cond_idx, sched, weight, _ = load_data(DATA_PATH)
    train_loader, val_loader = make_loaders(cond_idx, sched, weight)

    model = UNet1D().to(DEVICE)
    diffusion = Diffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"device={DEVICE}  N={len(sched)}  params={sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    ep = 0
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, diffusion, train_loader, optimizer)
        va = run_epoch(model, diffusion, val_loader)
        if ep % 25 == 0 or ep == 1:
            print(f"epoch {ep:4d} | train {tr:.4f} | val {va:.4f}", flush=True)
        if run is not None:
            run.log({"epoch": ep, "train/loss": tr, "val/loss": va})

        if va < best_val - EARLY_STOP_MIN_DELTA:
            best_val = va
            epochs_no_improve = 0
            best_state = {"model": copy.deepcopy(model.state_dict()), "epoch": ep}
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
        print(f"restored best checkpoint: epoch {best_state['epoch']} (val {best_val:.4f})")

    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.summary["best_epoch"] = best_state["epoch"] if best_state else ep
        run.summary["stopped_epoch"] = ep

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, save_path)
        print(f"saved model to {save_path}")
    if run is not None:
        run.finish()

    return model


def load_pretrained(path: Path = MODEL_SAVE_PATH) -> nn.Module:
    """保存済み Stage1 を読み込む。

    ★EMA が無いので use_ema 引数も無い。チェックポイントのキーは "model" のみ
    """
    ckpt = torch.load(path, map_location=DEVICE)
    model = UNet1D().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ============================================================
# 7. 群レベル生成（Stage2 の提案分布 / 評価用）
# ============================================================
@torch.no_grad()
def group_pool(model, n_per_group: int, guidance_scale: float = GUIDANCE_SCALE,
                verbose: bool = True) -> npt.NDArray[np.int64]:
    """群別サンプルプール (D, M, 96)。行 d は cond_grid()[d] の条件で生成。

    Stage2 の指数傾けはこのプールを提案分布 p_d として重み付けするので、
    M が小さいと傾け後の有効サンプル数 (ESS) が枯れる。M は数千を想定。

    ★sampler / ddim_steps / eta 引数は持たない（ancestral のみ）
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
        outs.append(diffusion.sample(model, ci, guidance_scale).cpu().numpy())
        if verbose:
            print(f"  pooled {min(i + GEN_BATCH, total)}/{total}", flush=True)
    return np.concatenate(outs, axis=0).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


def pool_to_rates(pool: npt.NDArray[np.int64],
                    weights: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """サンプルプール -> 群別期待行動者率 (D, n_act*96) act-major。

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
# 8. サニティチェック（断片化が実データ水準か = Route B の前提条件）
# ============================================================
def fragmentation_stats(sched: npt.NDArray[np.int64],
                        w: npt.NDArray[np.float64] | None = None) -> dict:
    """切替回数・エピソード長・日次参加率。individual_metrics へ委譲する。

    エピソード長系 (ep_len_*, single_slot_ratio) は個票を重み付けしない。
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
    """生成個票が「学習個票のコピー」になっていないかを表で出す。

    なぜ必要か:
        val loss の early stopping は「平均的に効いている過学習」しか止めない。
        小データの拡散モデルは学習個票をそのまま再生する形で暗記しうる。
        集計指標は暗記したモデルでこそ良く見えるので、集計だけを見ていると気づけない。

    読み方:
        判定は「gen が train **にだけ** 近いか」。train と holdout に同じだけ近いなら、
        それはデータ分布に近いだけで暗記ではない。見るべきは DCR_gap が 0 付近か
        どうかであって、DCR の絶対値ではない。

    参照集合のサイズを揃える（これをしないと指標が読めない）:
        最近傍距離は参照集合が大きいほど自然に小さくなる。train は holdout の
        約9倍あるので、素で比べると暗記が無くても DCR_gap が正に出る。
        よって DCR/NNDR は train を holdout と同数に間引いてから比べる。

    床（帰無帯）を併記する:
        train から互いに素な同数の部分集合 A, B を取って gap を測ると、
        暗記が原理的にありえない状況でのばらつき（＝床）が得られる。
        実測 gap がこの幅に収まっていれば暗記なし。

    exact/near copy は train 全体に対して測る。「モデルが見た個票のどれかを
    そのまま出したか」が問いなので、ここは間引いてはいけない。
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
    # 判定は片側。gap が床より「上」= train にだけ近い = 暗記。
    # 下に外れるのは train より holdout に近いという意味で、暗記ではない
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


def sanity_check(model, n_per_group: int = 256, save_path: Path | None = GEN_SAVE_PATH,
                 with_memorization: bool = True):
    """実 ATUS 平日と同一群構成で生成し、断片化と活動シェアを比較する。

    save_path=None なら CSV を書かない（--smoke が本番の生成CSVを潰さないため）。
    with_memorization=False なら暗記チェックを飛ばす（--smoke でプールが小さいとき用）。
    """
    cond_idx, sched_real, w_real, _ = load_data(DATA_PATH)
    d_real = cond_to_d(cond_idx)

    print(f"\n群別サンプルプール生成 (D={D_GROUPS} × M={n_per_group}, ancestral) ...")
    # ancestral は1バッチあたり 1000ステップ×2(CFG) の前向き計算で数分かかる。
    # 無言で待たせないようバッチ進捗を出す
    pool = group_pool(model, n_per_group, verbose=True)
    gen = pool.reshape(-1, NUM_SLOTS)

    # 実データの群構成に合わせた生成側の重み。プールは群一様なので、群別の
    # 「調査ウェイト加重シェア ÷ 群内本数」を各行へ配る。
    # ★ 非加重の人数比 (np.bincount(d_real)/N) ではない。両者は ATUS 平日で
    #   総変動距離 0.139 ずれる。重みの出所は group_reweight ただ一つ。
    gen_d = np.repeat(np.arange(D_GROUPS), n_per_group)
    w_gen = im.group_reweight(gen_d, w_real, d_real, D_GROUPS)

    r = fragmentation_stats(sched_real, w_real)
    g = fragmentation_stats(gen, w_gen)

    print("\n--- 断片化サニティ (実 ATUS 平日 vs AggDDPM-Simple 生成) ---")
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
    # DDPM_Aggregate の同じ表と直接比べられるよう、シェア誤差の合計も出す
    print(f"\n{'Σ|Δ| (活動シェア誤差の合計)':<28}{np.abs(sg - sr).sum():>10.4f}")

    if with_memorization:
        memorization_report(gen, sched_real)

    if save_path is None:
        print("\n(save_path=None のため生成CSVは書かない)")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid = cond_grid()
    meta = pd.DataFrame(np.repeat(grid, n_per_group, axis=0), columns=["gender", "age7", "employment"])
    meta.insert(0, "group_d", gen_d)   # w_gen と同一の群割り当て（ずれ得ない）
    # サンプラ列は clock_diagnostics が読むので残す。本実装では常に ancestral
    meta.insert(1, "sampler", "ancestral")
    pd.concat([meta, pd.DataFrame(gen, columns=[f"s{i}" for i in range(NUM_SLOTS)])],
                axis=1).to_csv(save_path, index=False)
    print(f"\nsaved generated schedules to {save_path}")


# ============================================================
# 9. スモークテスト
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

    # ★データ表現が {0,1} であること（{-1,+1} に戻っていないことの検出）
    assert float(x0.min()) == 0.0 and float(x0.max()) == 1.0, \
        f"sched_to_x0 の値域が {{0,1}} でない (min={float(x0.min())}, max={float(x0.max())})"
    assert torch.allclose(x0.sum(dim=1), torch.ones_like(x0.sum(dim=1))), \
        "各スロットの one-hot の和が 1 でない"

    # ★時刻埋め込みが TIME_EMB_DIM 次元で直接出ること（MLP を挟んでいないこと）
    assert timestep_embedding(t).shape == (4, TIME_EMB_DIM)
    assert not hasattr(m, "time_mlp"), "time_mlp が残っている（簡素化版の前提が崩れている）"

    # ★DDIM を持たないこと
    assert not hasattr(d, "ddim_sample"), "ddim_sample が残っている"

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

    # ★fragmentation_stats が先頭3000行で打ち切られていないこと
    trunc = np.zeros((4000, NUM_SLOTS), dtype=np.int64)
    trunc[3000:] = np.arange(NUM_SLOTS) % 2
    fs = fragmentation_stats(trunc)
    assert fs["single_slot_ratio"] > 0.9, \
        f"fragmentation_stats が打ち切られている (single_slot_ratio={fs['single_slot_ratio']})"
    assert abs(fs["switches"] - 0.25 * (NUM_SLOTS - 1)) < 1e-9   # 1000/4000 行 × 95境界

    # ★split_indices が全行を過不足なく2分すること
    tr_i, va_i = split_indices(len(sched))
    assert len(tr_i) + len(va_i) == len(sched)
    assert set(tr_i.tolist()).isdisjoint(va_i.tolist())
    assert len(va_i) == int(len(sched) * VAL_RATIO)

    assert m.in_channels == IN_CH

    # ★逆過程の1ステップが有限で、値域が壊れないこと。
    #   DDIM が無くなったので、全1000ステップを回さずにここだけを検証する
    #   （full ancestral は --smoke の sanity_check 側で小さく回す）
    ci = torch.as_tensor(grid[:4], device=DEVICE)
    xt = torch.randn(4, IN_CH, NUM_SLOTS, device=DEVICE)
    with torch.no_grad():
        eps_hat = d._eps(m, xt, T_STEPS - 1, ci, GUIDANCE_SCALE)
        x0_hat = (xt - d.sqrt_1m_acp[-1] * eps_hat) / d.sqrt_acp[-1]
        x0_hat.clamp_(0.0, 1.0)
        mean = d.post_coef_x0[-1] * x0_hat + d.post_coef_xt[-1] * xt
    assert eps_hat.shape == xt.shape and torch.isfinite(eps_hat).all()
    assert float(x0_hat.min()) >= 0.0 and float(x0_hat.max()) <= 1.0
    assert torch.isfinite(mean).all()
    assert int(mean.argmax(dim=1).max()) < NUM_ACT

    print(f"smoke test: OK  (N={len(sched)}, D={D_GROUPS}, n_act={NUM_ACT}, "
            f"params={sum(p.numel() for p in m.parameters()):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AggDDPM-Simple: ATUS平日・共通12分類・28群の条件付き pretrain（簡素化版）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="短時間の動作確認のみ")
    args = ap.parse_args()

    smoke_test()
    if args.smoke:
        model = train(epochs=5, use_wandb=False, save_path=None)
        # DDIM が無いので生成は 1000 ステップ固定。群あたり 2 本に絞って回す。
        # 暗記チェックは参照集合に対してプールが小さすぎるので飛ばす
        sanity_check(model, n_per_group=2, save_path=None, with_memorization=False)
    else:
        model = train(epochs=args.epochs, use_wandb=not args.no_wandb)
        sanity_check(model)
