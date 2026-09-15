"""
model.py
================
集計マッチ転移のための条件付きDDPM（AggDDPM）


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
import contextlib
import copy
import importlib.util
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, overload

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
# 畳み込みの受容野。既定 3 が本編の設定で、--kernel で 1/5/7 に振れる。
# 「隣接スロットを見る畳み込み」が断片化の一致にどれだけ効いているかを測るための軸。
# 1 にすると畳み込み経路から局所混合が完全に消え、スロット間の情報は
# attention（48/24 解像度の3ブロック）と最近傍アップサンプルだけを通る。
KERNEL_SIZE = 3
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
    (性2, 年齢7区分, 就業2) -> 群28インデックス d
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

    値域が変わるので、逆過程の clamp も [0,1] に揃える（Diffusion.sample 参照）。
    デコードは argmax なので、この変更でも復元規則は変わらない。
    """
    return F.one_hot(sched, NUM_ACT).float().permute(0, 2, 1)


# ============================================================
# 4. Denoiser（1D-UNet）
# ============================================================
def timestep_embedding(t: torch.Tensor, dim: int = TIME_EMB_DIM) -> torch.Tensor:
    """拡散ステップtをsinusoidal埋め込みベクトルへ符号化する, (B,) -> (B, dim)

    Note:
        1. 近いステップは近いベクトル
        2. 異なるステップは異なるベクトル
        3. スカラーtからdim次元ベクトルへ広げる

    Args:
        t: 拡散ステップ数, dtype=int64, 値域[0, T_STEPS], (B,)
        dim: 出力次元数, default=TIME_EMB_DIM=256

    Returns:
        sinusoidal埋め込み, dtype=float32, (B, dim)
    """
    half = dim // 2  # sin, cosのために, dimを2分割
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, dim)


class ResBlock1D(nn.Module):
    """条件埋め込みを注入する1D残差ブロック (pre-activation ResNet)

    Note:
        1. GroupNorm -> SiLU -> Conv1d の pre-activation 構成を2段重ね, 入力を残差加算する
        2. emb を emb_proj で c_out 次元へ落とし、チャネル毎バイアスとして時間軸一様に加算する
        3. 時間長Lは変えない (padding = KERNEL_SIZE // 2)
    """
    def __init__(self, c_in: int, c_out: int, emb_dim: int = TIME_EMB_DIM):
        """残差ブロックの層を構築する。

        Args:
            c_in: 入力チャネル数, GroupNorm(8, c_in) のため8の倍数
            c_out: 出力チャネル数, 8の倍数, c_in と異なるとき skip は 1x1 conv になる
            emb_dim: 条件埋め込みの次元, default=TIME_EMB_DIM=256
        """
        super().__init__()
        k, pad = KERNEL_SIZE, KERNEL_SIZE // 2
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad)
        self.emb_proj = nn.Linear(emb_dim, c_out)

        self.norm2 = nn.GroupNorm(8, c_out)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad)

        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """残差ブロック, (B, c_in, L) -> (B, c_out, L)

        Args:
            x: 入力特徴, dtype=float32, (B, c_in, L)
                Lは呼び出し位置で NUM_SLOTS = 96 / 48 / 24 のいずれか
            emb: 拡散ステップ埋め込みと条件埋め込みの和,
                dtype=float32, (B, emb_dim) = (B, 256)

        Returns:
            出力特徴, dtype=float32, (B, c_out, L)。Lは入力と同じ
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock1D(nn.Module):
    """時間軸(スロット間)の self-attention + residual
    畳み込みが届かない遠いスロット同士を直接結ぶ
    """
    def __init__(self, ch: int):
        """attention層を構築する。

        Args:
            ch: 入出力チャネル数, GroupNorm(8, ch) のため8の倍数
        """
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, ATTN_HEADS, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """self-attentionを1回かける, (B, C, L) -> (B, C, L)

        Args:
            x: 入力特徴, dtype=float32, (B, C, L)

        Returns:
            出力特徴, dtype=float32, (B, C, L)
        """
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        return h.permute(0, 2, 1) + x


class UNet1D(nn.Module):
    """ε予測ネットワーク: (B,12,96) + 拡散ステップt + 条件cond -> (B,12,96)"""
    def __init__(self):
        super().__init__()
        c1, c2 = BASE_CH, BASE_CH * 2

        # Condition Embedding
        self.cond_embeds = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(COND_CARD, EMB_DIMS)
        ])
        self.cond_proj = nn.Linear(sum(EMB_DIMS), TIME_EMB_DIM)
        self.null_emb = nn.Parameter(torch.zeros(TIME_EMB_DIM))

        k, pad = KERNEL_SIZE, KERNEL_SIZE // 2

        # Down h1
        self.in_conv = nn.Conv1d(IN_CH, c1, k, padding=pad)
        self.d1a, self.d1b = ResBlock1D(c1, c1), ResBlock1D(c1, c1)
        self.ds1 = nn.Conv1d(c1, c1, k, stride=2, padding=pad)

        # Down h2
        self.d2a, self.d2b = ResBlock1D(c1, c2), ResBlock1D(c2, c2)
        self.attn2 = AttnBlock1D(c2)
        self.ds2 = nn.Conv1d(c2, c2, k, stride=2, padding=pad)

        # Down h3
        self.d3a, self.d3b = ResBlock1D(c2, c2), ResBlock1D(c2, c2)
        self.attn3 = AttnBlock1D(c2)

        # Bottleneck (middle)
        self.m1, self.m_attn, self.m2 = ResBlock1D(c2, c2), AttnBlock1D(c2), ResBlock1D(c2, c2)

        # Up with h3
        self.u3 = ResBlock1D(c2 + c2, c2)

        # Up with h2
        self.us2 = nn.Conv1d(c2, c2, k, padding=pad)
        self.u2 = ResBlock1D(c2 + c2, c2)
        self.u2_attn = AttnBlock1D(c2)

        # Up with h1
        self.us1 = nn.Conv1d(c2, c1, k, padding=pad)
        self.u1 = ResBlock1D(c1 + c1, c1)

        # 最終出力層
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv1d(c1, IN_CH, k, padding=pad)
        nn.init.zeros_(self.out_conv.weight)
        out_bias = self.out_conv.bias
        if out_bias is not None:
            nn.init.zeros_(out_bias)

    @property
    def in_channels(self) -> int:
        """拡散空間のチャネル数。バックボーン実装に依らない共通の入口。

        clock_diagnostics が純ノイズ x_T を作るのに使う。実装内部の層名
        (in_conv 等) に触らせないための薄い契約。
        """
        return IN_CH

    def embed_cond(self, cond_idx: torch.Tensor | None, batch: int,
                    drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        """条件 (性・年齢・就業) を256次元の条件埋め込みへ, (B,3) -> (B,256)

        Note:
            1. 属性ごとに別の Embedding 表を引き、連結して cond_proj で混ぜる
                (埋め込み段階では属性は独立、Linear で初めて属性間の相互作用が入る)
            2. 条件なしは学習可能な null_emb 1本で表す
                (cond_idx=None はバッチ全体、drop_mask は行単位)
            3. timestep_embedding(t) と同じ TIME_EMB_DIM 次元で出し、加算できる形にする

        Args:
            cond_idx: 社会属性の条件インデックス, dtype=int64, (B, 3)
                列は COND_SPEC の順で [gender, age, telfs]
                gender: {0=男, 1=女}, age: {0,..,6} (15歳起点10歳階級7区分),
                telfs : {0=無業, 1=有業}
                None のときバッチ全体を無条件 (null_emb) にする
            batch: バッチサイズ B。cond_idx=None では形状を取れないため明示的に受け取る
            drop_mask: CFGの条件dropoutマスク, dtype=bool, (B,)
                True の行だけ条件埋め込みを null_emb に差し替える
                学習時は P_UNCOND=0.1 で立てる (GaussianDiffusion.loss)

        Returns:
            条件埋め込み c, dtype=float32, (B, TIME_EMB_DIM) = (B, 256)
        """
        if cond_idx is None:
            return self.null_emb.expand(batch, -1)

        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)  # (B,16)
        c = self.cond_proj(c)  # (B,16) -> (B,256)

        if drop_mask is not None:
            c = torch.where(drop_mask[:, None], self.null_emb.expand_as(c), c)
        return c  # (B, 256)

    def _encode(self,x_t: torch.Tensor,
                emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """UNet1Dの下り経路を1回通し, 3つの解像度の中間特徴 (hidden) を返す, (B, 12, 96) -> 96/48/24

        Note:
            1. CNN1Dで時間軸を96 -> 48 -> 24と半減させる, チャネルは12 -> 64 (h1) -> 128 (h2, h3)
            2. 各ResBlock1Dへembを渡し, 拡散ステップと条件を全段に入力
            3. forward と features の共通部分（分岐させない）

        Args:
            x_t: ノイズ付き活動スケジュール, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
            emb: 拡散ステップ埋め込み + 条件埋め込み (和), dtype=float32, (B, TIME_EMB_DIM) = (B, 256)

        Returns:
            中間特徴のタプル (h1, h2, h3), dtype=float32
                h1: (B, BASE_CH,   NUM_SLOTS)    = (B,  64, 96)  ds1 の手前
                h2: (B, BASE_CH*2, NUM_SLOTS//2) = (B, 128, 48)  ds2 の手前
                h3: (B, BASE_CH*2, NUM_SLOTS//4) = (B, 128, 24)  Bottleneck への入力
        """
        h1 = self.d1b(self.d1a(self.in_conv(x_t), emb), emb)
        h2 = self.attn2(self.d2b(self.d2a(self.ds1(h1), emb), emb))
        h3 = self.attn3(self.d3b(self.d3a(self.ds2(h2), emb), emb))
        return h1, h2, h3

    def features(self, x_t: torch.Tensor, t: torch.Tensor,
                cond_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """下り経路の中間特徴を名前付きで返す（診断用）。

        eval/clock_diagnostics.py の B4 診断が、実装内部の層名に触らずに
        中間特徴を読むための入口。

        Args:
            x_t: ノイズ付き活動スケジュール, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
            t: 拡散ステップ数, dtype=int64, (B,)
            cond_idx: 社会属性の条件インデックス, dtype=int64, (B, 3)
                None のときバッチ全体を無条件にする

        Returns:
            中間特徴 {h1: (B,64,96), h2: (B,128,48), h3: (B,128,24)}, dtype=float32
        """
        emb = timestep_embedding(t) + self.embed_cond(cond_idx, x_t.size(0))
        h1, h2, h3 = self._encode(x_t, emb)
        return {"h1": h1, "h2": h2, "h3": h3}

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond_idx: torch.Tensor | None=None,
                drop_mask: torch.Tensor | None=None) -> torch.Tensor:
        """ノイズを予測する ε_θ(x_t, t, c), (B, 12, 96) -> (B, 12, 96)

        Args:
            x_t: ノイズ付き活動スケジュール, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
            t: 拡散ステップ数, dtype=int64, 値域[0, T_STEPS], (B,)
            cond_idx: 社会属性 (gender, age, telfs) の条件インデックス, dtype=int64, (B, 3)
                None のときバッチ全体を無条件にする
            drop_mask: CFGの条件dropoutマスク, dtype=bool, (B,)
                Trueの行だけ条件埋め込みをnull_embに差し替える

        Returns:
            予測ノイズ ε_θ, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
        """
        # Embedding (拡散ステップt + 社会属性条件cond)
        num_batch = x_t.size(0)
        emb = timestep_embedding(t) + self.embed_cond(cond_idx, num_batch, drop_mask)  # emb(B, 256) = time_emb + cond_emb

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
@contextlib.contextmanager
def eval_mode(model: nn.Module) -> Iterator[nn.Module]:
    """model を eval に切り替え、抜けるときに元のモードへ戻す。

    Stage 2 は1回のパラメータ更新の中で2つのモードを行き来する:
        集計側   (sample_differentiable) : eval。評価時と同じ生成器を微分する
        リハーサル側 (Diffusion.loss)      : train。Dropout を効かせた Stage 1 と同じ損失
    切り替え忘れは例外を出さずに静かに数値を変える（zero-shot 基準線は eval で
    測ってある）ので、元へ戻す責任を呼び出し側に持たせない。
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


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

    def _eps(self, model: UNet1D, x, t_scalar, cond_idx, guidance_scale):
        """
        CFG 込みの ε 予測。t_scalar は int
        """
        t = torch.full((x.size(0),), t_scalar, device=x.device, dtype=torch.long)
        eps_c = model(x, t, cond_idx)
        if guidance_scale == 1.0:
            return eps_c
        eps_u = model(x,  t, None)
        return eps_u + guidance_scale * (eps_c - eps_u)

    # 逆過程1ステップの式は、以前は sample の中と smoke_test の中に二重に書かれていた。
    # Stage 2（Stage2_design.md §4.3）が微分可能な逆過程を要求するので、3箇所目を
    # 作らずに済むよう1つの関数へ括り出してある。呼び出し元は sample /
    # _sample_head / _sample_tail / smoke_test の4つ。
    @overload
    def _reverse_step(self, model: UNet1D, x: torch.Tensor, ti: int, cond_idx,
                      guidance_scale: float, z: torch.Tensor | None = ...,
                      return_aux: Literal[False] = ...) -> torch.Tensor: ...

    @overload
    def _reverse_step(self, model: UNet1D, x: torch.Tensor, ti: int, cond_idx,
                      guidance_scale: float, z: torch.Tensor | None,
                      return_aux: Literal[True]) -> tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor]: ...

    def _reverse_step(self, model: UNet1D, x: torch.Tensor, ti: int, cond_idx,
                      guidance_scale: float, z: torch.Tensor | None = None,
                      return_aux: bool = False):
        """
        ancestral DDPM + CFG の逆過程を1ステップ進める。x_t -> x_{t-1}

        args:
            z         : そのステップで加える雑音 (B,12,96)。None なら新しく引く。
                        ★2パス勾配蓄積は1パス目と同じ z を再注入する必要があるので
                          引数で受け取れる形にしてある。randn_like 固定にすると2パス化できない
            return_aux: True なら (x_next, eps_hat, x0_hat, mean) を返す。
                        smoke_test の assert が中間量を見ているため

        ★clamp は [0,1]。データ表現 {0,1} に合わせてある
          （DDPM_Aggregate は表現が {-1,+1} なので [-1,1]）
        ★in-place の clamp_ ではなく clamp を使う。勾配を流す区間で in-place 演算を
          挟むと autograd がエラーを出す（sample 側の数値は変わらない）
        ★ti == 0 では post_var[0] = 0 なので z を引かない。乱数の消費順が
          切り出し前と同じになり、同一シードで sample の出力が1ビットも変わらない
        """
        eps_hat = self._eps(model, x, ti, cond_idx, guidance_scale)
        x0_hat = (x - self.sqrt_1m_acp[ti] * eps_hat) / self.sqrt_acp[ti]
        x0_hat = x0_hat.clamp(0.0, 1.0)
        mean = self.post_coef_x0[ti] * x0_hat + self.post_coef_xt[ti] * x
        if ti == 0:
            x_next = mean          # post_var[0] = 0。ここが最終出力
        else:
            if z is None:
                z = torch.randn_like(x)
            x_next = mean + self.post_var[ti].sqrt() * z
        return (x_next, eps_hat, x0_hat, mean) if return_aux else x_next

    @torch.no_grad()
    def sample(self, model: UNet1D, cond_idx, guidance_scale=GUIDANCE_SCALE, verbose=False):
        """
        ancestral DDPM + CFG
        cond_idx (M,K) -> スケジュール (M,96) int

        ★@torch.no_grad() はこのメソッドに付けたまま残すこと。_reverse_step 側へ移すと
          sample_differentiable から呼んでも勾配が一切流れなくなり、しかも例外を出さない
        """
        model.eval()
        m = cond_idx.size(0)

        x = torch.randn(m, IN_CH, NUM_SLOTS, device=cond_idx.device)  # x_T~N(0,I)
        for ti in reversed(range(T_STEPS)):
            x = self._reverse_step(model, x, ti, cond_idx, guidance_scale)
            if verbose and ti % 200 == 0:
                print(f"  sampling t={ti}")
        return x.argmax(dim=1)

    # --------------------------------------------------------
    # Stage 2: 末尾 K ステップだけ勾配を保持する逆過程（Stage2_design.md §4.3）
    #
    # 全 1000 ステップの計算グラフは B=28 でも 130.6 GB になり保持できない。
    # よって末尾 K ステップで打ち切る。
    #
    # ★捨てた項はゼロではない。「古いステップの勾配は指数的に消えるから捨ててよい」
    #   という説明は誤りで、逆過程1段の倍率はむしろ 1/√α_t > 1 である。
    #   捨ててよい根拠は「消えるから」ではなく「残した項と向きがほぼ同じだから」で、
    #   K=1 の勾配は K=32 の勾配と cos ≈ 0.96 で一致する。
    #   実測は src/eval/diagnostics/stage2_gradient_probe.py、議論は docs/Stage2_design.md §4.2 ②。
    #
    # head / tail に分けてあるのは2パス勾配蓄積のため。1パス目で x_K を保存すれば
    # 前段（999ステップ）は1回で済み、2パス目は tail だけを回せばよい。
    # --------------------------------------------------------
    def _sample_head(self, model: UNet1D, cond_idx, K: int,
                     guidance_scale=GUIDANCE_SCALE) -> torch.Tensor:
        """x_T ~ N(0,I) から t = 999 → K まで進めて x_K を返す。グラフを作らない。

        通常のサンプリングと計算内容は完全に同じで、中間活性を保存しないだけ。
        返り値は detach 済みで、ここでグラフが切れる。
        """
        with eval_mode(model), torch.no_grad():
            x = torch.randn(cond_idx.size(0), IN_CH, NUM_SLOTS, device=cond_idx.device)
            for ti in reversed(range(K, T_STEPS)):
                x = self._reverse_step(model, x, ti, cond_idx, guidance_scale)
        return x.detach()

    def _sample_tail(self, model: UNet1D, x_K: torch.Tensor, K: int, cond_idx,
                     guidance_scale=GUIDANCE_SCALE,
                     zs: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """x_K から t = K-1 → 0 まで進めて x_0 を返す。★ここだけ計算グラフが作られる。

        args:
            zs: 末尾 K 区間で使う雑音 {ti: z}。None なら新しく引く。
                2パス目では1パス目と同じものを渡す（別のサンプルを見ないため）。
                ti=0 は雑音を使わないので、キー 0 は無くてよい
        """
        x = x_K
        with eval_mode(model):
            for ti in reversed(range(K)):
                x = self._reverse_step(model, x, ti, cond_idx, guidance_scale,
                                       None if zs is None else zs.get(ti))
        return x

    def sample_differentiable(self, model: UNet1D, cond_idx, K: int,
                              guidance_scale=GUIDANCE_SCALE,
                              zs: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """打ち切り逆伝播つきサンプリング。(B,12,96) の連続値を返す。

        ★@torch.no_grad() を付けないこと。付けると勾配が流れないまま例外も出ない。
        ★返り値は sample と違って argmax していない。t=0 の逆過程出力そのもの。
          離散化は straight_through が行う
        ★ti=0 の返り値は x0_hat と厳密には一致しない。post_coef_xt[0] と post_var[0] は
          厳密に 0 だが、post_coef_x0[0] は実数では 1 でも float32 では 0.99983406 になる
          （1-acp[0] を引き算で作るときの桁落ち。相対 1.7e-4）。つまり返るのは
          x0_hat の 0.99983 倍である。softmax も argmax も正のスケールに対して
          ほぼ不変なので下流への影響は無いが、「厳密に一致する」と書かないこと
        ★K=0 なら全ステップが no_grad になり、返り値は勾配を持たない
        """
        x_K = self._sample_head(model, cond_idx, K, guidance_scale)
        return self._sample_tail(model, x_K, K, cond_idx, guidance_scale, zs)


def straight_through(x0: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """連続値 (B,12,96) を微分可能に one-hot 化する（Stage2_design.md §2.8）。

    なぜ必要か:
        評価の pool_to_rates は argmax → one-hot → 平均で率を作る。一方 sample_differentiable
        が返す x0 は各要素が [0,1] にクリップされただけで、活動チャネル方向の和は1にならない
        （ある時刻で和が 2.0 になる値が普通に出る）。生の x0 の平均を教師に合わせると
        評価とは別の量を最適化することになる。
        argmax は階段関数で微分がほぼ至るところ 0 なので、前向きは argmax のまま使い、
        後ろ向きだけ softmax の微分で置き換える。

    args:
        tau: softmax の温度。既定 1 で運用し掃引しない。勾配の大きさは tau について
             単調でなく tau≈0.3 で最大になるが、tau を下げると p が1チャネルに集中して
             代理が argmax に近づく（置き換えた意味が薄れる）。勾配が効かないときの
             予備のつまみとして下げる場合も 0.15 を下回らせない

    ★dim=1 は12活動の軸であって時刻軸ではない。テンソルは (B, IN_CH, NUM_SLOTS) = (B,12,96)
      なので、この softmax は各時刻スロットで12活動に対して正規化する。dim=2 にかけると
      「各活動が1日のどこか1スロットで起きる」という別物の制約になる
    """
    p = F.softmax(x0, dim=1) if tau == 1.0 else F.softmax(x0 / tau, dim=1)
    # ★F.one_hot は新しい軸を末尾に足す。p.argmax(dim=1) が (B,96) なので
    #   F.one_hot は (B,96,12) を返す。permute で (B,12,96) に戻さないと形が合わない
    oh = F.one_hot(p.argmax(dim=1), NUM_ACT).permute(0, 2, 1).float()
    # ★括弧が必須。oh + p - p.detach() は左から評価されて (oh + p) - p.detach() になり、
    #   float32 の丸めで前向きの値が one-hot から 6.0e-08 ずれる。括弧を付ければ厳密に一致する
    return oh + (p - p.detach())


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
    ★デバイスはモジュール定数 DEVICE ではなく model の実デバイスから取る。
      Stage 2 の事後選択はチェックポイントを任意のデバイスへ載せて評価するので、
      両者が食い違うと "Placeholder storage has not been allocated" で落ちる。
    """
    dev = next(model.parameters()).device
    diffusion = Diffusion(device=dev)
    grid = torch.as_tensor(cond_grid(), device=dev)
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
        # ★式を書き写さず _reverse_step を呼ぶ。書き写すと clamp などの修正が
        #   片方にしか入らない事故が起きる。中間量は return_aux で受け取る
        _, eps_hat, x0_hat, mean = d._reverse_step(
            m, xt, T_STEPS - 1, ci, GUIDANCE_SCALE, None, True)
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
    ap.add_argument("--kernel", type=int, default=None, choices=[1, 3, 5, 7],
                    help="畳み込みの受容野。省略すると本編の設定 (3) で"
                         "既定の保存先に書く。明示するとアブレーション扱いになり、"
                         "保存先に _k{K} が付くので本編の成果物とは混ざらない")
    args = ap.parse_args()

    if args.kernel is not None:
        # ★モデル構築より前に差し替える。UNet1D/ResBlock1D は __init__ で
        #   モジュール変数 KERNEL_SIZE を読むため、ここで決めた値が全層に効く。
        KERNEL_SIZE = args.kernel
        # --kernel を明示した実行は、値が 3 でもアブレーションとして別名に隔離する。
        # スイープ一式を同じ規則で並べられるようにするため。
        suffix = f"_k{args.kernel}"
        MODEL_SAVE_PATH = MODEL_SAVE_PATH.with_name(
            f"{MODEL_SAVE_PATH.stem}{suffix}{MODEL_SAVE_PATH.suffix}")
        GEN_SAVE_PATH = GEN_SAVE_PATH.with_name(
            f"{GEN_SAVE_PATH.stem}{suffix}{GEN_SAVE_PATH.suffix}")
    print(f"[config] kernel_size={KERNEL_SIZE}")
    print(f"[config] ckpt={MODEL_SAVE_PATH.name}")
    print(f"[config] gen ={GEN_SAVE_PATH.name}")

    smoke_test()
    if args.smoke:
        model = train(epochs=5, use_wandb=False, save_path=None)
        # DDIM が無いので生成は 1000 ステップ固定。群あたり 2 本に絞って回す。
        # 暗記チェックは参照集合に対してプールが小さすぎるので飛ばす
        sanity_check(model, n_per_group=2, save_path=None, with_memorization=False)
    else:
        # ★保存先は明示的に渡す。train/sanity_check の既定引数は定義時に
        #   束縛済みで、上の再代入では差し替わらないため。
        model = train(epochs=args.epochs, use_wandb=not args.no_wandb,
                      save_path=MODEL_SAVE_PATH)
        sanity_check(model, save_path=GEN_SAVE_PATH)
