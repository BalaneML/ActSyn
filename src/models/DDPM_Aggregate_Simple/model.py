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
        --kernel K   : 畳み込みの受容野（アブレーション。保存先に _k{K}）
        --clock      : 全 ResBlock1D に 24 時間の時計を足す（アブレーション。保存先に _clock）
        --seed S     : 学習の乱数の種。分割は変えない（反復実験。保存先に _s{S}）

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
# ★24 時間の時計（--clock のときだけ使う）。スロット s の位相 2πs/96 のフーリエ特徴を
#   調和次数 k=1..CLOCK_HARMONICS（周期 24h, 12h, 8h, 6h）で作り、各 ResBlock1D へ
#   スロットごとに違う値のバイアスとして足す。条件は emb_proj で全スロット同じ値として
#   足されるので、時計が無いと「何時に」を表す経路が無い（clock_diagnostics の B4）。
#   周期 24h の関数なので左端 04:00 と右端の翌 04:00 がつながる（活動日は環）
CLOCK_HARMONICS = 4
CLOCK_DIM = 2 * CLOCK_HARMONICS

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
# ★2 つの用途がある。split_indices の学習/評価の分割（常にこの値で固定）と、
#   学習の乱数（初期値・ミニバッチ・拡散の t と ε）の既定値。--seed が変えるのは後者だけで、
#   分割は変えない。分割まで変えると Stage 2 の val や暗記チェックの参照集合が別物になる
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


def clock_features(num_slots: int = NUM_SLOTS,
                   harmonics: int = CLOCK_HARMONICS) -> torch.Tensor:
    """時刻スロットを 24 時間周期のフーリエ特徴 φ へ符号化する, -> (2*harmonics, num_slots)

    φ[2(k−1), s] = cos(2πks / num_slots),  φ[2(k−1)+1, s] = sin(2πks / num_slots),  k = 1..harmonics

    Note:
        1. 拡散ステップの timestep_embedding とは別物。こちらは「1 日のうちの何時か」を表す
        2. 学習パラメータを持たない固定の特徴。学習するのは ResBlock1D.clock_proj だけ
        3. k=1 の cos/sin の組だけで 96 スロットすべてが別の点になる（円周上の 96 点）。
           k=2..4 は 12h・8h・6h 周期で、昼食の 1 時間のような狭い山を線形に作りやすくする

    Args:
        num_slots: 1 日のスロット数, default=NUM_SLOTS=96
        harmonics: 調和次数の上限 k, default=CLOCK_HARMONICS=4

    Returns:
        フーリエ特徴 φ, dtype=float32, (2*harmonics, num_slots)
    """
    s = torch.arange(num_slots, dtype=torch.float32)
    k = torch.arange(1, harmonics + 1, dtype=torch.float32)
    angle = 2.0 * math.pi * k[:, None] * s[None, :] / num_slots          # (H, S)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1).reshape(2 * harmonics, num_slots)


class ResBlock1D(nn.Module):
    """条件埋め込みを注入する1D残差ブロック (pre-activation ResNet)

    Note:
        1. GroupNorm -> SiLU -> Conv1d の pre-activation 構成を2段重ね, 入力を残差加算する
        2. emb を emb_proj で c_out 次元へ落とし、チャネル毎バイアスとして時間軸一様に加算する
        3. 時間長Lは変えない (padding = KERNEL_SIZE // 2)
        4. clock=True のときだけ、時計 φ を clock_proj で c_out 次元へ落とし、
           スロットごとに違う値のバイアスとして 2. と同じ位置に加算する
    """
    # clock=True のときだけ register_buffer で作る。型チェッカに Tensor と伝えるための宣言
    clock_phi: torch.Tensor

    def __init__(self, c_in: int, c_out: int, emb_dim: int = TIME_EMB_DIM,
                 clock: bool = False):
        """残差ブロックの層を構築する。

        Args:
            c_in: 入力チャネル数, GroupNorm(8, c_in) のため8の倍数
            c_out: 出力チャネル数, 8の倍数, c_in と異なるとき skip は 1x1 conv になる
            emb_dim: 条件埋め込みの次元, default=TIME_EMB_DIM=256
            clock: 24 時間の時計を足すか, default=False (従来の構造)
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

        # ★clock=False では nn.Linear を作らないので、乱数の消費も層の初期値も従来と同じ。
        #   clock=True では nn.Linear の初期化が乱数を消費するため、後続ブロックの初期値は
        #   時計なしのモデルと一致しない（新しく学習するモデルなので問題にしない）。
        # ★零初期化。学習前の出力は時計なしの構造と一致する（test_backbone で検証）
        self.clock_proj: nn.Linear | None = None
        if clock:
            self.register_buffer("clock_phi", clock_features(), persistent=False)
            self.clock_proj = nn.Linear(CLOCK_DIM, c_out)
            nn.init.zeros_(self.clock_proj.weight)
            nn.init.zeros_(self.clock_proj.bias)

    def clock_bias(self, length: int) -> torch.Tensor:
        """時計のバイアスを解像度 length で返す, -> (1, c_out, length)

        Note:
            ★φ は 96 スロットで作ってあり、stride = 96 // length で間引く。
              ds1/ds2（stride 2, padding = KERNEL_SIZE // 2）の出力位置 j は入力位置 2j を
              中心に畳み込むので、48 解像度の j はスロット 2j、24 解像度の j はスロット 4j にあたる。

        Args:
            length: 特徴の時間長 L。NUM_SLOTS を割り切る値 (96 / 48 / 24)

        Returns:
            スロットごとのバイアス, dtype=float32, (1, c_out, length)

        Raises:
            RuntimeError: 時計を持たないブロックで呼んだとき
            ValueError: length が NUM_SLOTS を割り切らないとき
        """
        if self.clock_proj is None:
            raise RuntimeError("clock=False の ResBlock1D には時計が無い")
        if NUM_SLOTS % length != 0:
            raise ValueError(f"時間長 {length} が NUM_SLOTS={NUM_SLOTS} を割り切らない")
        phi = self.clock_phi[:, ::NUM_SLOTS // length]          # (CLOCK_DIM, L)
        return self.clock_proj(phi.T).T[None]                    # (1, c_out, L)

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
        h = h + self.emb_proj(emb)[:, :, None]           # 全スロットで同じ値
        if self.clock_proj is not None:
            h = h + self.clock_bias(h.size(-1))          # スロットごとに違う値
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
    def __init__(self, clock: bool = False):
        """UNet1D の層を構築する。

        Args:
            clock: 全 ResBlock1D (11 個) に 24 時間の時計を足すか, default=False (従来の構造)。
                True でも零初期化なので、学習前の出力は False と一致する
        """
        super().__init__()
        self.clock = clock
        c1, c2 = BASE_CH, BASE_CH * 2

        def res_block(c_in: int, c_out: int) -> ResBlock1D:
            return ResBlock1D(c_in, c_out, clock=clock)

        # Condition Embedding
        self.cond_embeds = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(COND_CARD, EMB_DIMS)
        ])
        self.cond_proj = nn.Linear(sum(EMB_DIMS), TIME_EMB_DIM)
        self.null_emb = nn.Parameter(torch.zeros(TIME_EMB_DIM))

        k, pad = KERNEL_SIZE, KERNEL_SIZE // 2

        # Down h1
        self.in_conv = nn.Conv1d(IN_CH, c1, k, padding=pad)
        self.d1a, self.d1b = res_block(c1, c1), res_block(c1, c1)
        self.ds1 = nn.Conv1d(c1, c1, k, stride=2, padding=pad)

        # Down h2
        self.d2a, self.d2b = res_block(c1, c2), res_block(c2, c2)
        self.attn2 = AttnBlock1D(c2)
        self.ds2 = nn.Conv1d(c2, c2, k, stride=2, padding=pad)

        # Down h3
        self.d3a, self.d3b = res_block(c2, c2), res_block(c2, c2)
        self.attn3 = AttnBlock1D(c2)

        # Bottleneck (middle)
        self.m1, self.m_attn, self.m2 = res_block(c2, c2), AttnBlock1D(c2), res_block(c2, c2)

        # Up with h3
        self.u3 = res_block(c2 + c2, c2)

        # Up with h2
        self.us2 = nn.Conv1d(c2, c2, k, padding=pad)
        self.u2 = res_block(c2 + c2, c2)
        self.u2_attn = AttnBlock1D(c2)

        # Up with h1
        self.us1 = nn.Conv1d(c2, c1, k, padding=pad)
        self.u1 = res_block(c1 + c1, c1)

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
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


class Diffusion:
    """Stage1とStage2を実装"""
    def __init__(self, device=DEVICE):
        """β schedule と, そこから導かれるバッファを事前計算する

        Note:
            以下は全て (T_STEPS,) = (1000,) の1次元テンソル, dtype=float32
            拡散ステップ t でインデックスして使う

            1. betas: ノイズの強さ β_t, BETA_START=0.0001 から BETA_END=0.02 への線形スケジュール
            2. alphas: α_t = 1 - β_t
            3. acp: ᾱ_t = Π α_s (alphas の累積積), acp[0]=0.99990, acp[999]=4.04e-5
            4. sqrt_acp: √ᾱ_t, q_sample の x0 側の係数
            5. sqrt_1m_acp: √(1-ᾱ_t), q_sample の eps 側の係数
            6. post_var: 事後分散 σ²_t = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)
            7. post_coef_x0: 事後平均の x0_hat 側の係数 β_t·√ᾱ_{t-1}/(1-ᾱ_t)
            8. post_coef_xt: 事後平均の x_t 側の係数 (1-ᾱ_{t-1})·√α_t/(1-ᾱ_t)

            6〜8 は事後分布 q(x_{t-1}|x_t, x0) の閉形式で, _reverse_step だけが使う

        Args:
            device: バッファを置くデバイス, default=DEVICE
        """
        betas = torch.linspace(BETA_START, BETA_END, T_STEPS, device=device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
        self.device = device
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.sqrt_acp = acp.sqrt()
        self.sqrt_1m_acp = (1.0 - acp).sqrt()
        self.post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.post_coef_x0 = betas * acp_prev.sqrt() / (1.0 - acp)
        self.post_coef_xt = (1.0 - acp_prev) * alphas.sqrt() / (1.0 - acp)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """x0 から任意のtステップ先の x_t を求める (前向き拡散過程)

        Note:
            x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε

        Args:
            x0: 拡散対象の活動スケジュール, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96), 値域{0,1}
            t: 拡散ステップ数, dtype=int64, 値域[0, T_STEPS-1], (B,)
            eps: 乗せるノイズ, dtype=float32, (B, 12, 96), ~N(0, I), MSE教師

        Returns:
            ノイズ付きスケジュール x_t, dtype=float32, (B, 12, 96)
        """
        return (self.sqrt_acp[t][:, None, None] * x0
                + self.sqrt_1m_acp[t][:, None, None] * eps)

    def loss(self, model: UNet1D, sched: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        """Stage1学習の目的関数, 標準的な ε予測MSEに CFGを組み込んだもの

        Note:

        Args:
            model: UNet1D
            sched: 活動スケジュール (インデックス表現), dtype=int64, (B, 96)
            cond_idx: 条件インデックス, dtype=int64, (B, 3)

        Returns:
            バッチ損失, dtype=float32, (スカラ)
        """
        x0 = sched_to_x0(sched)  # (B,96)->(B,12,96)∈{0,1}
        t = torch.randint(0, T_STEPS, (x0.size(0),), device=x0.device)  # t~U{0,T-1}
        eps = torch.randn_like(x0)  # eps~N(0,I), (B,12,96)
        x_t = self.q_sample(x0, t, eps)  # q(x_t|x0)

        drop_mask = torch.rand(x0.size(0), device=x0.device) < P_UNCOND  # CFGの条件dropout

        eps_hat = model(x_t, t, cond_idx, drop_mask)
        return F.mse_loss(eps_hat, eps)  # ノイズ間のMSE

    def _eps(self, model: UNet1D, x: torch.Tensor,
            t_scalar: int, cond_idx: torch.Tensor | None, guidance_scale: float) -> torch.Tensor:
        """Classifier-Free Guidance (CFG) を適用した ε予測

        Note:
            1. tからt-1のノイズ予測のみを行う

        Args:
            model: UNet1D
            x: ノイズ付き活動スケジュール, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
            t_scalar: 拡散ステップ数, 値域[0, T_STEPS-1]
            cond_idx: 条件インデックス, dtype=int64, (B, 3)
            guidance_scale: CFGの強さ

        Returns:
            CFG適用後の予測ノイズ, dtype=float32, (B, 12, 96)
        """
        t = torch.full((x.size(0),), t_scalar, device=x.device, dtype=torch.long)
        eps_c = model(x, t, cond_idx)
        if guidance_scale == 1.0:
            return eps_c
        eps_u = model(x,  t, None)
        return eps_u + guidance_scale * (eps_c - eps_u)

    # 逆過程1ステップの式は、以前は sample の中と smoke_test の中に二重に書かれていた。
    # Stage 2（Stage2_design.md §4.3）が微分可能な逆過程を要求するので、3箇所目を
    # 作らずに済むよう1つの関数へ括り出してある。
    # 呼び出し元は sample / _sample_head / _sample_tail / smoke_test の4つ。
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
        """逆過程を1ステップ進める, x_t -> x_{t-1}

        Args:
            model: UNet1D, denoiser
            x: 現在の状態x_t, dtype=float32, (B, 12, 96)
            ti: 拡散ステップ, 値域[0, T_STEPS-1], スカラー
            guidance_scale: CFGの強さ, スカラー
            z: そのステップで加える雑音, デフォルトでtorch.randn_likeで引く, dtype=float32, (B, 12, 96)
            return_aux: Trueなら中間量を返す, bool, smoke_testのため

        Returns:
            return_aux=False: 1ステップ進めた x_{t-1}, dtype=float32, (B, 12, 96)
            return_aux=True: (x_next, eps_hat, x0_hat, mean)の4タプル, それぞれdtype=float32, (B, 12, 96)
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
    def sample(self, model: UNet1D, cond_idx: torch.Tensor,
                guidance_scale: float=GUIDANCE_SCALE, verbose: bool=False) -> torch.Tensor:
        """M本の条件に基づいた活動スケジュールをサンプリングする
        
        Args:
            model: UNet1D, εを予測する, denoiser
            cond_idx: 条件インデックス, dtype=int64, (M, 3), Mは生成本数
            guidance_scale: CFGの強さ s, default=GUIDANCE_SCALE=1.25
            verbose: Trueなら 200ステップごとに進捗を表示する

        Returns:
            活動スケジュール (インデックス表現), dtype=int64, (M, NUM_SLOTS) = (M, 96)
            値域[0, NUM_ACT-1]
        """
        model.eval()
        m = cond_idx.size(0)  # 生成本数

        x = torch.randn(m, IN_CH, NUM_SLOTS, device=cond_idx.device)  # x_T~N(0,I)
        for ti in reversed(range(T_STEPS)): # T_STEPS-1..0
            x = self._reverse_step(model, x, ti, cond_idx, guidance_scale)
            if verbose and ti % 200 == 0:
                print(f"  sampling t={ti}")
        return x.argmax(dim=1)  # 微分不可 -> argmaxのため

    # --------------------------------------------------------
    # Stage 2: 末尾 K ステップだけ勾配を保持する逆過程
    #
    # 全 1000 ステップの計算グラフは B=28 でも 130.6 GB になり保持できない。
    # よって末尾 K ステップで打ち切る
    #
    # head / tail に分けてあるのは2パス勾配蓄積
    # --------------------------------------------------------

    def _sample_head(self, model: UNet1D, cond_idx: torch.Tensor,
                    K: int, guidance_scale: float=GUIDANCE_SCALE) -> torch.Tensor:
        """x_T ~ N(0,I) から t = 999 → K まで進めて x_K を返す, 計算グラフを作らない

        Args:
            model: UNet1D, ε予測する, denoiser
            cond_idx: 条件インデックス, dtype=int64, (M, 3), Mは生成本数
            K: 勾配を保持する末尾ステップ数
            guidance_scale: CFGの強さ, default=GUIDANCE_SCALE=1.25

        Returns:
            残り Kステップ地点での状態 x_K, dtype=float32, (M, IN_CH, NUM_SLOTS) = (M, 12, 96)
        """
        with eval_mode(model), torch.no_grad():
            x = torch.randn(cond_idx.size(0), IN_CH, NUM_SLOTS, device=cond_idx.device)  # x_T, (M, 12, 96)
            for ti in reversed(range(K, T_STEPS)):  # ti = T_STEPS-1...K
                x = self._reverse_step(model, x, ti, cond_idx, guidance_scale)  # x_t -> x_{t-1}
        return x.detach()  # 計算グラフを切る

    def _sample_tail(self, model: UNet1D, x_K: torch.Tensor, K: int, 
                    cond_idx: torch.Tensor, guidance_scale: float=GUIDANCE_SCALE,
                    zs: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """x_K から t = K-1 → 0 まで進めて x_0 を返す, ここだけ計算グラフが作られる

        Args:
            model: UNet1D, εを予測する denoiser
            x_K: 残りKステップ地点の状態, dtype=float32, (M, IN_CH, NUM_SLOTS) = (M, 12, 96)
            K: 勾配を保持する末尾ステップ数, ti = K-1 .. 0
            cond_idx: 条件インデックス, dtype=int64, (M, 3)
            guidance_scale: CFGの強さs, default=GUIDANCE_SCALE=1.25
            zs: 末尾 K 区間で使う雑音 {ti: z}
                2パス目では1パス目と同じものを渡す（別のサンプルを見ないため）
                ti=0 は雑音を使わないので、キー 0 は無くてよい

        Returns:
            逆過程 t=0 の出力 x_0, dtype=float32, (M, IN_CH, NUM_SLOTS) = (M, 12, 96)
            argmaxしていない連続値, 関数外で離散化する
        """
        x = x_K
        with eval_mode(model):
            for ti in reversed(range(K)):
                x = self._reverse_step(model, x, ti, cond_idx, guidance_scale,
                                        None if zs is None else zs.get(ti))
        return x

    def sample_differentiable(self, model: UNet1D, cond_idx: torch.Tensor,
                                K: int, guidance_scale: float=GUIDANCE_SCALE,
                                zs: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """Kステップ打ち切り逆伝播つきサンプリング, (B,12,96) の連続値を返す

        Args:
            model: εを予測, denoiser
            cond_idx: 条件インデックス, dtype=int64, (B, 3)
            K: 勾配を保持する末尾ステップ
            guidance_scale: CFGの強さs, default=GUIDANCE_SCALE=1.25
            zs: 末尾K区間で使う雑音 {ti: z}

        Returns:
            逆過程 t=0 の出力 x_0, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
            argmaxしない, 離散化は関数外で行う
        """
        x_K = self._sample_head(model, cond_idx, K, guidance_scale)
        return self._sample_tail(model, x_K, K, cond_idx, guidance_scale, zs)


def straight_through(x0: torch.Tensor, tau: float=1.0) -> torch.Tensor:
    """連続値 (B,12,96) を微分可能に one-hot 化

    Args:
        x0: 離散化前の活動スケジュール, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
        tau: softmax の温度。既定 1 で運用し掃引しない。勾配の大きさは tau について
            単調でなく tau≈0.3 で最大になるが、tau を下げると p が1チャネルに集中して
            代理が argmax に近づく（置き換えた意味が薄れる）。勾配が効かないときの
            予備のつまみとして下げる場合も 0.15 を下回らせない

    Returns:
        微分可能な one-hot, dtype=float32, (B, 12, 96)
        前向きは厳密なone-hot, 後ろ向きはsoftmax(x0/tau) の微分が流れる
    """
    p = F.softmax(x0 / tau, dim=1)
    oh = F.one_hot(p.argmax(dim=1), NUM_ACT).permute(0, 2, 1).float()
    return oh + (p - p.detach())


# ============================================================
# 6. 学習
# ============================================================
def run_epoch(model: UNet1D, diffusion: Diffusion,loader: DataLoader,
            optimizer: torch.optim.Optimizer | None=None) -> float:
    """1エポック分の学習または評価を実行し, サンプル加重平均の ε-MSE を返す

    optimizerを渡せば学習, 渡さなければ評価として動く

    Args:
        model: ノイズ予測器ε_θ
        diffusion: βスケジュールを持つDiffusion
        loader:
            (cond_idx, sched)をyieldするDataLoader
            cond_idx: 条件インデックス, dtype=int64, (B, 3)
            sched: 活動スケジュール (インデックス表現), dtype=int64, (B, 96)
        optimizer: 学習時の最適化器, Noneなら評価モード, default=None

    Returns:
        エポック平均の ε-MSE, float
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
            save_path: Path | None = MODEL_SAVE_PATH,
            clock: bool = False,
            seed: int = SEED) -> UNet1D:
    """Stage1の学習を実行, val 損失が最良だった重みのモデルを返す

    ATUS実個票を教師に, 条件付きノイズ予測器 ε_θ(x_t, t, c)を学習

    Args:
        epochs: 学習エポック数の上限, default=EPOCHS=1000
        use_wandb: wandbへハイパラと学習曲線を記録するか, default=True
        save_path: チェックポイントの保存先, Noneなら保存しない
        clock: UNet1D に 24 時間の時計を足すか, default=False
        seed: 学習の乱数の種（初期値・ミニバッチ・t・ε）, default=SEED=42。
            学習/評価の分割は split_indices が SEED で固定するので、この値では変わらない

    Returns:
        best_stateを復元済みのUNet1D, 必ずしも最終エポックのおもみではない
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
                "kernel_size": KERNEL_SIZE,
                "clock": clock, "clock_harmonics": CLOCK_HARMONICS if clock else 0,
                "seed": seed,
            }
        )

    torch.manual_seed(seed)
    cond_idx, sched, weight, _ = load_data(DATA_PATH)
    train_loader, val_loader = make_loaders(cond_idx, sched, weight)

    model = UNet1D(clock=clock).to(DEVICE)
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
        # config は出所の記録。構造の判定には使わない（state_has_clock が重みのキーで決める）
        torch.save({"model": model.state_dict(),
                    "config": {"kernel_size": KERNEL_SIZE, "clock": clock, "seed": seed}}, save_path)
        print(f"saved model to {save_path}")
    if run is not None:
        run.finish()

    return model


def state_has_clock(state: dict[str, torch.Tensor]) -> bool:
    """重みの state_dict が時計つきの UNet1D のものかを返す

    Note:
        ★構造の判定は重みのキーだけで行う。config を持たない古いチェックポイント
          （20260819 版など）も、Stage 2 の世代も同じ規則で読めるようにするため。

    Args:
        state: UNet1D の state_dict

    Returns:
        clock_proj の重みを持てば True
    """
    return any(".clock_proj." in k for k in state)


def build_unet_for_ckpt(path: Path) -> UNet1D:
    """チェックポイントの重みに合う構造の UNet1D を組む（重みはまだ読まない）

    stage2_checkpoint.load_ckpt のように「組んだモデルへ後から読む」呼び出し側のための入口。
    Stage 1 の重みと Stage 2 の世代（stage2_step*.pt）のどちらも受け付ける。

    Args:
        path: キー "model" に state_dict を持つチェックポイント

    Returns:
        CPU 上の UNet1D。時計の有無は state_has_clock で決める
    """
    # ★weights_only=False。Stage 2 の世代は RNG 状態と config を含む。自分で書いたファイルだけを読む
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return UNet1D(clock=state_has_clock(ckpt["model"]))


def load_pretrained(path: Path = MODEL_SAVE_PATH) -> nn.Module:
    """保存済み Stage1 を読み込む。

    ★EMA が無いので use_ema 引数も無い。重みはキー "model"、出所の記録はキー "config"
      （20260819 版など古いチェックポイントには config が無い）
    ★時計の有無は重みのキーから決める（state_has_clock）
    """
    ckpt = torch.load(path, map_location=DEVICE)
    model = UNet1D(clock=state_has_clock(ckpt["model"])).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ============================================================
# 7. 群レベル生成（評価・チェックポイント選択用）
# ============================================================
@torch.no_grad()
def group_pool(model, n_per_group: int, guidance_scale: float = GUIDANCE_SCALE,
                verbose: bool = True) -> npt.NDArray[np.int64]:
    """群別サンプルプール (D, M, 96) を生成する。行 d は cond_grid()[d] の条件。

    Stage 1 の sanity_check と、Stage 2 のチェックポイント事後選択
    (stage2_select.evaluate_ckpt) の両方が、ここで作ったプールを
    pool_to_rates に通して群別行動者率を測る。

    Args:
        model: 学習済み UNet1D。Stage 1 でも Stage 2 のチェックポイントでもよい
        n_per_group: 群あたりの生成本数 M。群別行動者率の推定分散が
            チェックポイント間の差より小さくなる必要があるので、
            選択用途では数千を想定 (stage2_select.DEFAULT_N = 2000)
        guidance_scale: CFG のスケール, default=GUIDANCE_SCALE=1.25
        verbose: 生成の進捗を print するか, default=True

    Returns:
        群別サンプルプール, dtype=int64, (D_GROUPS, n_per_group, NUM_SLOTS)
        = (28, M, 96)。値域 [0, NUM_ACT)

    Note:
        Stage 2 の学習ループはこの関数を使わない (@torch.no_grad() なので勾配が
        通らない)。学習側は Diffusion.sample_differentiable を使う。
        本関数は評価と選択の専用である。

        (群, サンプル) を平坦化してからチャンクする。群ごとに切ると端数バッチが
        増えて、逆過程 (1000ステップ) の呼び出し効率が落ちるため。

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
    """サンプルプールを群別期待行動者率へ集計する, (D, M, 96) -> (D, n_act*96)

    Args:
        pool: 群別サンプルプール, dtype=int64, (D_GROUPS, M, NUM_SLOTS)
        weights: (D, M) の非負重み。None なら一様平均。
            無印 DDPM_Aggregate の指数傾けと API を揃えるために残してある。
            Simple 版の本番経路 (stage2_select / stage2_targets / sanity_check)
            は全て None で呼ぶ, default=None

    Returns:
        群別期待行動者率, dtype=float64, (D_GROUPS, NUM_ACT * NUM_SLOTS)
        act-major（インデックス a*96+t）。各スロットで活動方向の和が 1

    Note:
        CVAE_Aggregate.model.group_rates と同一形式で返すので、
        japan_match_experiment.eval_against にそのまま渡せる。

        Stage 2 の損失が straight-through の前向きを群平均した量と、この関数の
        出力が float64 で厳密一致することを test_stage2 が検査している
        （学習で下げる量と評価で測る量を同じにするため）。
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
    ap.add_argument("--clock", action="store_true",
                    help="全 ResBlock1D に 24 時間の時計（clock_proj）を足す。"
                         "アブレーション扱いで、保存先に _clock が付く")
    ap.add_argument("--seed", type=int, default=None,
                    help="学習の乱数の種（既定 SEED=42）。学習/評価の分割は変えない。"
                         "明示すると反復実験扱いで、保存先に _s{seed} が付く")
    args = ap.parse_args()
    seed = SEED if args.seed is None else args.seed

    suffix = ""
    if args.kernel is not None:
        # ★モデル構築より前に差し替える。UNet1D/ResBlock1D は __init__ で
        #   モジュール変数 KERNEL_SIZE を読むため、ここで決めた値が全層に効く。
        KERNEL_SIZE = args.kernel
        # --kernel を明示した実行は、値が 3 でもアブレーションとして別名に隔離する。
        # スイープ一式を同じ規則で並べられるようにするため。
        suffix += f"_k{args.kernel}"
    if args.clock:
        suffix += "_clock"
    if args.seed is not None:
        suffix += f"_s{args.seed}"
    if suffix:
        MODEL_SAVE_PATH = MODEL_SAVE_PATH.with_name(
            f"{MODEL_SAVE_PATH.stem}{suffix}{MODEL_SAVE_PATH.suffix}")
        GEN_SAVE_PATH = GEN_SAVE_PATH.with_name(
            f"{GEN_SAVE_PATH.stem}{suffix}{GEN_SAVE_PATH.suffix}")
    print(f"[config] kernel_size={KERNEL_SIZE}")
    print(f"[config] clock={args.clock}")
    print(f"[config] seed={seed}")
    print(f"[config] ckpt={MODEL_SAVE_PATH.name}")
    print(f"[config] gen ={GEN_SAVE_PATH.name}")

    smoke_test()
    if args.smoke:
        model = train(epochs=5, use_wandb=False, save_path=None, clock=args.clock, seed=seed)
        # DDIM が無いので生成は 1000 ステップ固定。群あたり 2 本に絞って回す。
        # 暗記チェックは参照集合に対してプールが小さすぎるので飛ばす
        sanity_check(model, n_per_group=2, save_path=None, with_memorization=False)
    else:
        # ★保存先は明示的に渡す。train/sanity_check の既定引数は定義時に
        #   束縛済みで、上の再代入では差し替わらないため。
        model = train(epochs=args.epochs, use_wandb=not args.no_wandb,
                      save_path=MODEL_SAVE_PATH, clock=args.clock, seed=seed)
        sanity_check(model, save_path=GEN_SAVE_PATH)
