"""
stage2_agg.py
=============
LSTM_Aggregate の Stage 2: 生成プールの時刻別行動者率と教師 A* の距離を、生成の連鎖（96 スロット）を通して
逆伝播し、LSTM の重みを更新する。DDPM_Aggregate_Simple/stage2_finetune.py の LSTM 版（2026-10-08 ユーザー決定）

1 更新でやること:
    1. 教師群から D_SUB 群を一様に抜き、群あたり N 本を生成する
    2. 各スロットの活動は Gumbel-max で引く（前向きは p_θ からの厳密な標本）。後ろ向きは
       softmax((logits + g) / τ) の微分を流す（straight-through Gumbel-softmax）
    3. 引いた活動の one-hot y を、次のスロットの入力 y @ act_embed.weight にする
       → L_agg の勾配は、そのスロットの確率だけでなく、96 スロットの履歴を通って届く（方法 (c)）
    4. y を群ごとに半分ずつ平均して ā_A, ā_B → L_agg = mean ω (ā_A − A*)(ā_B − A*)
       （split-batch。DDPM と同じ stage2_loss の関数。多様性への罰を含まない）
    5. L_reh = ATUS 学習分割の交差エントロピー（Stage 1 と同じ損失。重みなし）
    6. L_agg と λ·L_reh の勾配を別々に得て綱引きを測り（DDPM の add_rehearsal_grad）、足して AdamW を 1 歩

DDPM 版との対応:
    DDPM の straight-through（前向き argmax(x_0)、後ろ向き softmax(x_0/τ)）  → Gumbel-max ＋ softmax((l+g)/τ)
    DDPM の末尾 K ステップの逆伝播（K = 1）                                  → 96 スロットすべて（BPTT = 0）。
                                                                              --bptt K で K スロットごとに履歴を切れる
    DDPM のリハーサル（ε-MSE）                                              → 交差エントロピー（Stage 1 の損失）
    DDPM の層別学習率（cond / emb / clock / conv）                          → cond / time / rest
    CFG（g = 1.25）                                                         → なし（LSTM は CFG を持たない）

★設定は結果を見る前に固定する（下の「設定」）。判定 J1〜J6 に使う step は最終 step（STEPS）で、
  途中の ckpt は学習曲線を見るためだけに残す（DDPM のように結果を見て step を選ばない）

データフロー:

```mermaid
flowchart TD
    CK["lm.load_model(stage1_ckpt(seed))<br/>Pre-trained（H=64・wd=0.01・学習型の時刻符号）"] --> M["model"]
    M --> SS["sample_soft(model, cond, gumbel)<br/>y (B, 12, 96)・acts (B, 96)<br/>B = d_sub × n"]
    SS --> SP["sl.group_rates_split(y, n)<br/>a_A, a_B (d_sub, 12, 96)"]
    TGT["sl.teacher_tensor(tgt)<br/>a_star_all (28, 12, 96)"] --> LA
    SP --> LA["sl.agg_loss_from_rates(a_A, a_B, a_star, omega)<br/>l_agg"]
    LA -- "l_agg.backward()" --> G["θ.grad = g_agg"]
    ATUS["gm.make_loader(train_part)"] --> RH["lm.batch_loss → l_reh"]
    RH -- "ft.add_rehearsal_grad(params, lam · l_reh)" --> G
    G --> OPT["clip_grad_norm_ → AdamW（cond / time / rest）"]
    OPT --> M
    M -- "SAVE_EVERY ごと" --> SV["ck.save_ckpt(run_dir / stage2_step{S}.pt)"]
    SV --> EV["evaluate(model, tgt, masks)<br/>lm.group_pool(model, EVAL_N, EVAL_SEED) → gs2.score_pool"]
    EV --> CSV["csv_path(seed, run, variant)"]
    CSV --> JD["judge(variant) → gs2.judge → J1〜J6"]
```

実行（GRU の Stage 2 と同じ E0〜E2）:

    E0  --zero-shot      Pre-trained のまま採点する（28 群と、LGO の各 fold の held-out 群）
    E1  （引数なし）     28 群すべてを教師にして微調整し、最終 step を採点する
    E2  --fold K         fold K の 4 群を教師から外して微調整する（held-out 群は L_agg に入れない）
    --judge              E0〜E2 の CSV から判定 J1〜J6 を出す（GRU の stage2.judge と同じ規則）

使い方:
    .venv/bin/python src/models/LSTM_Aggregate/stage2_agg.py --smoke
    .venv/bin/python src/models/LSTM_Aggregate/stage2_agg.py --seed 42 --zero-shot
    .venv/bin/python src/models/LSTM_Aggregate/stage2_agg.py --seed 42 --lam 0.01
    .venv/bin/python src/models/LSTM_Aggregate/stage2_agg.py --seed 42 --lam 0.01 --fold 3
    .venv/bin/python src/models/LSTM_Aggregate/stage2_agg.py --judge --lam 0.01

    フラグ:
        --lam V          リハーサルの重み λ（既定 LAM = 0.01）
        --tau V          Gumbel-softmax の温度 τ（既定 1.0）。既定以外は保存先に _tau{V}
        --bptt K         K スロットごとに履歴の勾配を切る。0 は切らない（既定）。既定以外は保存先に _bptt{K}
        --steps / --d-sub / --n / --lr-cond / --lr-time / --lr-rest  学習の設定（既定は下の定数）
        --eval-step S    学習せず、保存済みの step S の ckpt を採点する

出力:
    outputs/checkpoints/stage2_lstm{Stage1}{変種}{種}_{実行}/stage2_step{S}.pt   SAVE_EVERY ごとの ckpt
    outputs/checkpoints/stage2_lstm{Stage1}{変種}{種}_{実行}/history.csv        1 更新ごとの記録
    data/processed/aggregates/stage2_lstm{Stage1}{変種}{種}_{実行}.csv           採点（GRU と同じ縦持ち）
    outputs/generated/stage2_lstm{Stage1}{変種}{種}_{実行}_rates.npz             評価のプールの時刻別行動者率 (28, 12, 96)
        {Stage1} = _time_learned_h64_wd0.01、{変種} = _lam{λ}[_tau{τ}][_bptt{K}]（E0 は空）、
        {種} は 42 なら空・それ以外は _s{seed}、{実行} = zeroshot / all / fold{K}
"""
import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"
GRU_DIR = REPO_ROOT / "src" / "models" / "GRU_Aggregate"


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名ファイルの取り違えを防ぐ）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lm: Any = _load("lstm_aggregate_model", Path(__file__).resolve().parent / "model.py")
gm: Any = lm.gm
sm: Any = lm.sm
# 採点・判定は GRU の Stage 2 と同じ関数（score_pool・judge・fold_masks）
gs2: Any = _load("gru_aggregate_stage2", GRU_DIR / "stage2.py")
st: Any = gs2.st
cur: Any = gs2.cur
lgo: Any = gs2.lgo
# 損失・ckpt・綱引きの測り方は DDPM の Stage 2 と同じ関数
sl: Any = _load("simple_stage2_loss", SIMPLE_DIR / "stage2_loss.py")
ck: Any = _load("simple_stage2_checkpoint", SIMPLE_DIR / "stage2_checkpoint.py")
ft: Any = _load("simple_stage2_finetune", SIMPLE_DIR / "stage2_finetune.py")

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]

# ============================================================
# 設定（2026-10-08。結果を見る前に固定）
# ============================================================
# Stage 1 の構成（ユーザー指定）: H = 64・1 層・weight decay 0.01・学習型の時刻符号
STAGE1_TIME_ENC: Any = "learned"
STAGE1_HIDDEN = 64
STAGE1_NUM_LAYERS = 1
STAGE1_WEIGHT_DECAY = 0.01

# 学習（DDPM の Stage 2 の既定に合わせる）
STEPS = 300                      # 固定の更新回数（早期終了なし）。判定に使うのは最終 step
D_SUB = 7                        # 1 更新で使う群の数
N = 256                          # 群あたりの生成本数（偶数。split-batch で半分ずつに分ける）
LAM = 0.01                       # リハーサルの重み λ（ユーザー指定の出発点）
TAU = 1.0                        # Gumbel-softmax の温度（DDPM の straight-through と同じく 1 で運用）
BPTT = 0                         # 履歴の勾配を切る間隔（スロット）。0 は切らない＝96 スロットすべて
EPS = float("inf")               # 損失の重み ω の床。inf は素の MSE（DDPM の主 A）
# 層別学習率。DDPM と同じ考え方: 群ごとに違う値を持つ cond と、スロットごとに違う値を持つ time を速く、
# 全員・全スロットに共通の部品（活動の埋め込み・LSTM・出力層）を遅く動かす
LR_COND = 1e-4                   # cond_embeds・cond_proj
LR_TIME = 1e-4                   # time_input（学習型の時刻符号）
LR_REST = 1e-5                   # act_embed・lstm・out_proj
GRAD_CLIP: float = lm.GRAD_CLIP  # 1.0。Stage 1 と同じ（96 スロットの逆伝播で勾配が跳ねたときの保険）
SAVE_EVERY = 25
VAL_EVERY = 10                   # ATUS の val の交差エントロピーを測る間隔（リハーサルの過学習の監視）
LOG_EVERY = 10

# 採点（GRU・DDPM の Stage 2 と同じ）
EVAL_N: int = gs2.EVAL_N         # 2000
EVAL_SEED: int = gs2.EVAL_SEED   # 12345
SEEDS: tuple[int, ...] = gs2.SEEDS
LGO_SEED: int = gs2.LGO_SEED

DEVICE: str = lm.DEVICE
CKPT_ROOT = REPO_ROOT / "outputs" / "checkpoints"
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
RATES_DIR = REPO_ROOT / "outputs" / "generated"
# 層別学習率の群。名前の先頭で判定する（上から順。当たらなければ rest）
PARAM_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {"cond": ("cond_embeds", "cond_proj"), "time": ("time_input",)}


# ============================================================
# 1. 微分できる生成
# ============================================================
def straight_through_gumbel(logits: torch.Tensor, generator: torch.Generator,
                            tau: float = TAU) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(logits) から 1 つ引き、勾配の通る one-hot を返す（straight-through Gumbel-softmax）

    Note:
        1. 前向きは argmax(logits + g) の厳密な one-hot。g は標準 Gumbel 雑音なので、これは
           softmax(logits) からの厳密な標本（Gumbel-max）
        2. 後ろ向きは softmax((logits + g) / τ) の微分。値は y = hard + (soft − soft.detach()) で、
           soft − soft.detach() は前向きでちょうど 0
        3. ★雑音は CPU の generator で引く（デバイスによらず同じ標本になる。gm.draw_categorical と同じ考え方）

    Args:
        logits: (B, NUM_ACT)
        generator: CPU の torch.Generator
        tau: 温度, default=TAU=1.0

    Returns:
        (y: 勾配の通る one-hot (B, NUM_ACT), a: 引いた活動, dtype=int64, (B,)。どちらも logits と同じデバイス)
    """
    u = torch.rand(logits.shape, generator=generator, dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
    gumbel = (-torch.log(-torch.log(u))).to(logits)
    z = logits + gumbel
    a = z.argmax(dim=-1)
    soft = F.softmax(z / tau, dim=-1)
    hard = F.one_hot(a, sm.NUM_ACT).to(soft)
    return hard + (soft - soft.detach()), a


def sample_soft(model: Any, cond_idx: torch.Tensor, generator: torch.Generator, tau: float = TAU,
                bptt: int = BPTT) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """s = 0..95 の順に引き、勾配が 96 スロットの履歴を通る 1 日を作る

    Note:
        1. スロット s の入力は y_{s−1} @ act_embed.weight[:12]（s = 0 は BOS の埋め込み）。
           y が厳密な one-hot なので、前向きの値は model.step（act_embed(a_prev)）と同じ
        2. bptt > 0 なら、s が bptt の倍数のスロットで LSTM の状態と入力の勾配を切る
           （そこより前の履歴へは勾配が戻らない。切り詰めた BPTT）

    Args:
        model: lm.LSTMScheduler（step_embedded を持つ）
        cond_idx: 条件インデックス, dtype=int64, (B, 3)。モデルと同じデバイス
        generator: Gumbel 雑音の CPU の torch.Generator
        tau: 温度, default=TAU=1.0
        bptt: 履歴の勾配を切る間隔（スロット）。0 は切らない, default=BPTT=0

    Returns:
        y: 勾配の通る one-hot, (B, NUM_ACT, NUM_SLOTS)。sl.group_rates_split の入力の形
        acts: 引いた活動, dtype=int64, (B, NUM_SLOTS)
        logits: 各スロットの logits, (B, NUM_SLOTS, NUM_ACT)（検算用）
    """
    batch = cond_idx.size(0)
    c_emb = model.embed_cond(cond_idx)                                          # (B, H)
    act_table = model.act_embed.weight[:sm.NUM_ACT]                             # (12, H)。BOS の行は除く
    e_prev = model.act_embed.weight[lm.BOS].expand(batch, -1)                   # (B, H)
    state: Any = None
    ys, acts, logits_all = [], [], []
    for s in range(sm.NUM_SLOTS):
        if bptt > 0 and s > 0 and s % bptt == 0:
            e_prev = e_prev.detach()
            state = (state[0].detach(), state[1].detach())
        logits, state = model.step_embedded(e_prev, c_emb, state, s)
        y, a = straight_through_gumbel(logits, generator, tau)
        ys.append(y)
        acts.append(a)
        logits_all.append(logits)
        e_prev = y @ act_table                                                  # (B, H)
    return torch.stack(ys, dim=2), torch.stack(acts, dim=1), torch.stack(logits_all, dim=1)


# ============================================================
# 2. 最適化
# ============================================================
def split_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """パラメータを層別学習率の群 cond / time / rest に分ける

    Note:
        time は時刻符号が none のモデルでは空なので、dict から除く

    Args:
        model: lm.LSTMScheduler

    Returns:
        {群名: パラメータの list}。並びは cond, time, rest

    Raises:
        ValueError: cond か rest が空のとき
    """
    groups: dict[str, list[nn.Parameter]] = {"cond": [], "time": [], "rest": []}
    for name, p in model.named_parameters():
        hit = next((g for g, prefixes in PARAM_GROUP_PREFIXES.items() if name.startswith(prefixes)), "rest")
        groups[hit].append(p)
    if not groups["time"]:
        del groups["time"]
    empty = [k for k, v in groups.items() if not v]
    if empty:
        raise ValueError(f"層別学習率の分割に失敗した。空の群: {empty}")
    return groups


def build_optimizer(model: nn.Module, lr_cond: float = LR_COND, lr_time: float = LR_TIME,
                    lr_rest: float = LR_REST) -> torch.optim.Optimizer:
    """層別学習率の AdamW（weight decay 0。DDPM の Stage 2 と同じ）。各 param_group は "name" を持つ

    Args:
        model: lm.LSTMScheduler
        lr_cond: cond_embeds・cond_proj の学習率, default=LR_COND=1e-4
        lr_time: time_input の学習率, default=LR_TIME=1e-4
        lr_rest: act_embed・lstm・out_proj の学習率, default=LR_REST=1e-5

    Returns:
        AdamW。ft.grad_norms が群名を読めるように param_groups[i]["name"] を持つ
    """
    lrs = {"cond": lr_cond, "time": lr_time, "rest": lr_rest}
    return torch.optim.AdamW([{"params": params, "lr": lrs[name], "name": name}
                              for name, params in split_param_groups(model).items()], weight_decay=0.0)


# ============================================================
# 3. 保存先
# ============================================================
def stage1_tag() -> str:
    """Stage 1 の構成の印（_time_learned_h64_wd0.01）"""
    return lm.time_tag(STAGE1_TIME_ENC) + lm.size_tag(STAGE1_HIDDEN, STAGE1_NUM_LAYERS, STAGE1_WEIGHT_DECAY)


def stage1_ckpt(seed: int) -> Path:
    """Pre-trained の ckpt のパス"""
    return lm.ckpt_path(seed, time_enc=STAGE1_TIME_ENC, hidden=STAGE1_HIDDEN, num_layers=STAGE1_NUM_LAYERS,
                        weight_decay=STAGE1_WEIGHT_DECAY)


def variant_tag(lam: float, tau: float = TAU, bptt: int = BPTT, steps: int = STEPS, lr_cond: float = LR_COND,
                lr_time: float = LR_TIME, lr_rest: float = LR_REST) -> str:
    """Stage 2 の設定の印（例: _lam0.01、_lam0.01_tau0.5_bptt8、_lam0.01_lr0.001_steps1000）

    Note:
        1. τ・bptt・学習率・step 数は既定と違うときだけ書く（既定の結果の名前を変えないため）
        2. 学習率は 3 群が同じなら _lr{値}、違えば _lr{cond}-{time}-{rest}
        3. ★学習率と step 数を印に入れないと、設定を変えた実行が既定の ckpt・CSV を上書きする

    Args:
        lam: リハーサルの重み λ
        tau: Gumbel-softmax の温度, default=TAU
        bptt: 履歴の勾配を切る間隔, default=BPTT
        steps: 更新回数, default=STEPS
        lr_cond: cond の学習率, default=LR_COND
        lr_time: time の学習率, default=LR_TIME
        lr_rest: rest の学習率, default=LR_REST

    Returns:
        印の文字列
    """
    tag = f"_lam{lam:g}"
    if tau != TAU:
        tag += f"_tau{tau:g}"
    if bptt != BPTT:
        tag += f"_bptt{bptt}"
    if (lr_cond, lr_time, lr_rest) != (LR_COND, LR_TIME, LR_REST):
        same = lr_cond == lr_time == lr_rest
        tag += f"_lr{lr_cond:g}" if same else f"_lr{lr_cond:g}-{lr_time:g}-{lr_rest:g}"
    if steps != STEPS:
        tag += f"_steps{steps}"
    return tag


def run_name(zero_shot: bool, fold: int | None) -> str:
    """実行の名前: zeroshot（E0）/ all（E1）/ fold{K}（E2）"""
    return gs2.run_name(zero_shot, fold)


def stem(seed: int, run: str, variant: str) -> str:
    """保存先の共通の名前。E0 は variant を空にする"""
    return f"stage2_lstm{stage1_tag()}{variant}{lm.run_suffix(seed)}_{run}"


def run_dir(seed: int, run: str, variant: str) -> Path:
    """ckpt と学習の記録を置くディレクトリ"""
    return CKPT_ROOT / stem(seed, run, variant)


def csv_path(seed: int, run: str, variant: str) -> Path:
    """採点の CSV のパス。E0（run = zeroshot）は variant によらず同じファイル"""
    return OUT_DIR / f"{stem(seed, run, '' if run == 'zeroshot' else variant)}.csv"


def rates_path(seed: int, run: str, variant: str) -> Path:
    """評価のプールの時刻別行動者率の npz のパス"""
    return RATES_DIR / f"{stem(seed, run, '' if run == 'zeroshot' else variant)}_rates.npz"


# ============================================================
# 4. 学習
# ============================================================
def teacher_mask_of(tgt: dict, fold: int | None) -> BoolArr:
    """L_agg に使う群 (28,)。fold を渡すと、その fold の 4 群を外す（GRU・DDPM と同じ LGO の分割）"""
    if fold is None:
        return np.ones(sm.D_GROUPS, dtype=bool)
    return gs2.fold_masks(tgt)[f"fold{fold}"]


def train_stage2(model: Any, tgt: dict, teacher_mask: BoolArr, seed: int, lam: float = LAM,
                 steps: int = STEPS, d_sub: int = D_SUB, n: int = N, tau: float = TAU, bptt: int = BPTT,
                 lr_cond: float = LR_COND, lr_time: float = LR_TIME, lr_rest: float = LR_REST,
                 out_dir: Path | None = None, save_every: int = SAVE_EVERY,
                 val_every: int = VAL_EVERY) -> list[dict[str, float]]:
    """固定の step 数だけ微調整する（model を in-place で更新する）

    Note:
        1. 乱数は 3 つ: numpy（群の抽出 d_pick）、Gumbel 雑音の CPU generator、torch の大域の乱数
           （ATUS のミニバッチの並び）。どれも seed から作る
        2. ★held-out 群（teacher_mask = False）は L_agg に一度も入らない。リハーサル（ATUS）は全群を使う
           （DDPM の Stage 2 と同じ）
        3. 勾配のクリップは L_agg と λ·L_reh を足した後に 1 回だけ掛ける。記録の clip_norm はクリップ前のノルム

    Args:
        model: Pre-trained の LSTM（学習するデバイスに載せたもの）
        tgt: 教師（st.load_stula_targets）
        teacher_mask: L_agg に使う群, (28,)
        seed: 乱数の種
        lam: リハーサルの重み λ, default=LAM=0.01
        steps: 更新回数, default=STEPS=300
        d_sub: 1 更新で使う群の数, default=D_SUB=7
        n: 群あたりの生成本数（偶数）, default=N=256
        tau: Gumbel-softmax の温度, default=TAU=1.0
        bptt: 履歴の勾配を切る間隔（0 は切らない）, default=BPTT=0
        lr_cond: cond の学習率, default=LR_COND=1e-4
        lr_time: time の学習率, default=LR_TIME=1e-4
        lr_rest: rest の学習率, default=LR_REST=1e-5
        out_dir: ckpt と history.csv の保存先。None なら何も保存しない
        save_every: ckpt を保存する間隔, default=SAVE_EVERY=25
        val_every: ATUS の val の交差エントロピーを測る間隔。0 で測らない, default=VAL_EVERY=10

    Returns:
        1 更新ごとの記録 list[dict]

    Raises:
        ValueError: n が奇数のとき、教師群が無いとき
    """
    if n % 2 != 0:
        raise ValueError(f"split-batch には n が偶数である必要がある: {n}")
    teacher_groups = np.flatnonzero(teacher_mask)
    if len(teacher_groups) == 0:
        raise ValueError("教師群が空")
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gumbel = torch.Generator().manual_seed(seed)

    a_star_all = sl.teacher_tensor(tgt, dev)                                   # (28, 12, 96)
    omega_all = sl.chi2_weights(a_star_all, EPS)
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long, device=dev)     # (28, 3)
    optimizer = build_optimizer(model, lr_cond, lr_time, lr_rest)
    params = [p for g in optimizer.param_groups for p in g["params"]]

    train_part, val_part = gm.load_split()
    train_loader = gm.make_loader(train_part, shuffle=True)
    val_loader = gm.make_loader(val_part, shuffle=False)

    def atus_batches() -> Any:
        while True:
            yield from train_loader

    atus = atus_batches()
    config: dict[str, Any] = {"stage1_ckpt": stage1_ckpt(seed).name, "seed": seed, "lam": lam, "steps": steps,
                              "d_sub": d_sub, "n": n, "tau": tau, "bptt": bptt, "eps": EPS,
                              "lr_cond": lr_cond, "lr_time": lr_time, "lr_rest": lr_rest, "grad_clip": GRAD_CLIP,
                              "holdout": np.flatnonzero(~teacher_mask).tolist()}
    print(f"[stage2] steps={steps} d_sub={d_sub} n={n} lam={lam:g} tau={tau:g} bptt={bptt} "
          f"teacher_groups={len(teacher_groups)}/28 lr cond={lr_cond:g} time={lr_time:g} rest={lr_rest:g} "
          f"device={dev}", flush=True)

    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        d_pick = np.sort(rng.choice(teacher_groups, size=min(d_sub, len(teacher_groups)), replace=False))
        d_pick_t = torch.as_tensor(d_pick, device=dev)
        cond = grid[d_pick_t].repeat_interleave(n, dim=0)                       # (d_sub·n, 3) 群優先
        a_star, omega = a_star_all[d_pick_t], omega_all[d_pick_t]

        # ---- 集計側: 96 スロットの生成を通して逆伝播する ----
        optimizer.zero_grad(set_to_none=True)
        y, _, _ = sample_soft(model, cond, gumbel, tau, bptt)
        a_A, a_B = sl.group_rates_split(y, n)
        l_agg = sl.agg_loss_from_rates(a_A, a_B, a_star, omega)
        l_agg.backward()
        agg_gnorm = ft.grad_norms(optimizer, "agg")

        # ---- リハーサル側: ATUS の交差エントロピー（Stage 1 と同じ損失）----
        b_cond, b_sched, b_weight = next(atus)
        l_reh = lm.batch_loss(model, b_cond.to(dev), b_sched.to(dev), b_weight.to(dev))
        tug = ft.add_rehearsal_grad(params, lam * l_reh)
        total_gnorm = ft.grad_norms(optimizer, "total")
        clip_norm = float(nn.utils.clip_grad_norm_(params, GRAD_CLIP))
        optimizer.step()

        with torch.no_grad():
            a_full = 0.5 * (a_A + a_B)
            log: dict[str, float] = {"step": step, "L_agg": float(l_agg.detach()), "L_reh": float(l_reh.detach()),
                                     "lam": lam, "rate_mae": float((a_full - a_star).abs().mean()),
                                     "rate_mse": float(((a_full - a_star) ** 2).mean()),
                                     "clip_norm": clip_norm, **agg_gnorm, **total_gnorm, **tug,
                                     "sec": time.perf_counter() - t0}
        if val_every > 0 and (step % val_every == 0 or step == 1):
            log["L_reh_val"] = lm.run_epoch(model, val_loader)
        history.append(log)
        if step % LOG_EVERY == 0 or step == 1:
            val_txt = f"  val={log['L_reh_val']:.5f}" if "L_reh_val" in log else ""
            gn = "/".join(f"{log[f'agg_gnorm_{g['name']}']:.1e}" for g in optimizer.param_groups)
            print(f"  step {step:4d}/{steps}  L_agg={log['L_agg']:+.6f}  rate_mse={log['rate_mse']:.6f}  "
                  f"L_reh={log['L_reh']:.5f}{val_txt}  |g_agg| cond/time/rest={gn}  "
                  f"tug_cos={log['tug_cos']:+.3f}  tug_ratio={log['tug_ratio']:.2f}  "
                  f"clip={clip_norm:.2f}  {log['sec']:.1f}s", flush=True)
        if out_dir is not None and (step % save_every == 0 or step == steps):
            ck.save_ckpt(ck.ckpt_path(out_dir, step), model, optimizer, step, config, np_rng=rng)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return history


# ============================================================
# 5. 採点
# ============================================================
def evaluate(model: Any, tgt: dict, masks: dict[str, BoolArr], base: dict[str, Any],
             n_per_group: int = EVAL_N, full: bool = True) -> tuple[list[dict[str, Any]], FloatArr]:
    """評価のプールを作り、GRU の Stage 2 と同じ 3 つの軸で採点する

    Args:
        model: 採点する LSTM
        tgt: 教師
        masks: 名前 → 教師の群 (28,)
        base: 全行に付ける識別列
        n_per_group: 評価のプールの群あたりの本数（偶数）, default=EVAL_N=2000
        full: False なら軸 1 だけ（--smoke 用）

    Returns:
        (採点の行 list[dict], 評価のプールの時刻別行動者率 (28, 12, 96))
    """
    pool = lm.group_pool(model, n_per_group, seed=EVAL_SEED)
    rows = gs2.score_pool(pool, tgt, masks, base, full)
    rates = np.asarray(cur.pool_to_slot_rates(pool), dtype=np.float64)
    return rows + jsd_rows(rates, tgt, masks, base), rates


def slot_jsd(rates: FloatArr, a_star: FloatArr, groups: BoolArr) -> float:
    """時刻別行動者率の JSD: 群 d・スロット s ごとに 12 活動の分布の JSD を取り、群とスロットで平均する

        JSD[d, s] = ½ KL(g[d,·,s] ‖ m) + ½ KL(A*[d,·,s] ‖ m),   m = ½ (g[d,·,s] + A*[d,·,s])

    Note:
        1. 自然対数（値域 [0, ln 2]）。損失側の sl.jsd_loss と同じ定義
        2. 0 log 0 = 0 として扱う（行動者率 0 のセルを落とす）。m > 0 なので KL は有限
        3. ★生成の率は評価のプール（群あたり 2000 本）の標本なので、標本の揺らぎの分だけ正に偏る
           （rate_mse_split のような偏りを消す版ではない）

    Args:
        rates: 生成の時刻別行動者率, (28, 12, 96)
        a_star: 教師 A*, (28, 12, 96)
        groups: 平均に使う群, (28,)

    Returns:
        平均の JSD
    """
    g, a = rates[groups], a_star[groups]
    m = 0.5 * (g + a)

    def kl(p: FloatArr) -> FloatArr:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, p * np.log(p / m), 0.0).sum(axis=1)                 # (群, 96)

    return float((0.5 * kl(g) + 0.5 * kl(a)).mean())


def jsd_rows(rates: FloatArr, tgt: dict, masks: dict[str, BoolArr], base: dict[str, Any]) -> list[dict[str, Any]]:
    """slot_jsd を採点の CSV と同じ縦持ちの行にする（教師の群 in-teacher と外した群 held-out）

    Args:
        rates: 評価のプールの時刻別行動者率, (28, 12, 96)
        tgt: 教師
        masks: 名前 → 教師の群 (28,)
        base: 全行に付ける識別列

    Returns:
        行 list[dict]（metric = rate_jsd）
    """
    a_star = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        for kind, groups in (("in-teacher", mask), ("held-out", ~mask)):
            if groups.any():
                rows.append({**base, "mask_name": name, "eval_kind": kind, "reference": "teacher",
                             "statistic": "slot_rate", "mask": "12act", "weight_basis": "none",
                             "metric": "rate_jsd", "value": slot_jsd(rates, a_star, groups)})
    return rows


def load_stage2(path: Path, device: str = DEVICE) -> Any:
    """Stage 2 の ckpt（ck.save_ckpt の形式）から LSTM を eval モードで戻す"""
    model = lm.LSTMScheduler(hidden=STAGE1_HIDDEN, num_layers=STAGE1_NUM_LAYERS, time_enc=STAGE1_TIME_ENC).to(device)
    ck.load_ckpt(path, model, map_location=device)
    return model.eval()


def write_scores(rows: list[dict[str, Any]], rates: FloatArr, seed: int, run: str, variant: str) -> None:
    """採点の CSV と時刻別行動者率の npz を書く"""
    out = csv_path(seed, run, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    rp = rates_path(seed, run, variant)
    rp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rp, rates=rates)
    print(f"[stage2] 書いた: {out}\n[stage2] 書いた: {rp}", flush=True)


def run(seed: int, fold: int | None, zero_shot: bool, lam: float = LAM, tau: float = TAU, bptt: int = BPTT,
        steps: int = STEPS, d_sub: int = D_SUB, n: int = N, lr_cond: float = LR_COND, lr_time: float = LR_TIME,
        lr_rest: float = LR_REST, eval_step: int | None = None) -> None:
    """E0 / E1 / E2 を 1 本回して採点の CSV を書く

    Args:
        seed: 種（Pre-trained の ckpt と Stage 2 の乱数）
        fold: E2 で外す fold（0〜6）。None なら 28 群すべてが教師
        zero_shot: True なら学習せず Pre-trained を採点する（E0）
        lam: リハーサルの重み λ
        tau: Gumbel-softmax の温度
        bptt: 履歴の勾配を切る間隔（0 は切らない）
        steps: 更新回数
        d_sub: 1 更新で使う群の数
        n: 群あたりの生成本数
        lr_cond: cond の学習率
        lr_time: time の学習率
        lr_rest: rest の学習率
        eval_step: 学習せず、保存済みのこの step の ckpt を採点する。None なら学習して最終 step を採点する
    """
    tgt = st.load_stula_targets()
    name = run_name(zero_shot, fold)
    variant = "" if zero_shot else variant_tag(lam, tau, bptt, steps, lr_cond, lr_time, lr_rest)
    folds = gs2.fold_masks(tgt)
    if zero_shot:
        model = lm.load_model(stage1_ckpt(seed))
        masks: dict[str, BoolArr] = {"all": np.ones(sm.D_GROUPS, dtype=bool), **folds}
        step = 0
    else:
        teacher = teacher_mask_of(tgt, fold)
        masks = {name: teacher}
        out_dir = run_dir(seed, name, variant)
        if eval_step is None:
            model = lm.load_model(stage1_ckpt(seed))
            train_stage2(model, tgt, teacher, seed, lam, steps, d_sub, n, tau, bptt, lr_cond, lr_time, lr_rest,
                         out_dir=out_dir)
            step = steps
        else:
            model = load_stage2(ck.ckpt_path(out_dir, eval_step))
            step = eval_step
    model.eval()
    base: dict[str, Any] = {"model": f"lstm{stage1_tag()}", "seed": seed, "run": name, "lam": lam if not zero_shot
                            else float("nan"), "tau": tau, "bptt": bptt, "step": step,
                            "n_per_group": EVAL_N, "pool_seed": EVAL_SEED}
    rows, rates = evaluate(model, tgt, masks, base)
    suffix = "" if eval_step is None or eval_step == steps else f"_step{eval_step}"
    write_scores(rows, rates, seed, name + suffix, variant)


def reference_rows(variant: str) -> list[dict[str, Any]]:
    """判定表に足す MAE と JSD の参考行（合否は付けない）。J1 と同じ held-out 群、J6 と同じ 28 群

    Note:
        1. rate_mae は時刻別行動者率の差の絶対値の平均（12 活動 × 96 スロット × 群）。値は採点の CSV にある
        2. rate_jsd は slot_jsd。保存済みの行動者率の npz から計算する（JSD を足す前の CSV にも使えるように）
        3. ★判定の規則（J1〜J6）は GRU と同じ MSE のまま変えない。MAE・JSD は並べて読むためだけ

    Args:
        variant: variant_tag の印

    Returns:
        行 list[dict]（列 item・lstm・zero_shot。pass は空）
    """
    tgt = st.load_stula_targets()
    a_star = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    folds = gs2.fold_masks(tgt)
    all_groups = np.ones(sm.D_GROUPS, dtype=bool)

    def rates_of(seed: int, run_: str) -> FloatArr:
        return np.asarray(np.load(rates_path(seed, run_, variant))["rates"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    zs42 = pd.read_csv(csv_path(LGO_SEED, "zeroshot", variant))
    zs42_rates = rates_of(LGO_SEED, "zeroshot")
    for k in range(lgo.N_FOLDS):
        run_ = f"fold{k}"
        e2 = pd.read_csv(csv_path(LGO_SEED, run_, variant))
        held = ~folds[run_]
        rows.append({"item": f"MAE（参考）held-out {run_}",
                     "lstm": gs2._value(e2, "rate_mae", eval_kind="held-out", mask="12act", mask_name=run_),
                     "zero_shot": gs2._value(zs42, "rate_mae", eval_kind="held-out", mask="12act", mask_name=run_)})
        rows.append({"item": f"JSD（参考）held-out {run_}", "lstm": slot_jsd(rates_of(LGO_SEED, run_), a_star, held),
                     "zero_shot": slot_jsd(zs42_rates, a_star, held)})
    e1 = [pd.read_csv(csv_path(s, "all", variant)) for s in SEEDS]
    zs = [pd.read_csv(csv_path(s, "zeroshot", variant)) for s in SEEDS]
    rows.append({"item": "MAE（参考）28 群の rate_mae（5 種の中央値）",
                 "lstm": float(np.median([gs2._value(d, "rate_mae", eval_kind="in-teacher", mask="12act")
                                          for d in e1])),
                 "zero_shot": float(np.median([gs2._value(d, "rate_mae", eval_kind="in-teacher", mask="12act",
                                                          mask_name="all") for d in zs]))})
    rows.append({"item": "JSD（参考）28 群の rate_jsd（5 種の中央値）",
                 "lstm": float(np.median([slot_jsd(rates_of(s, "all"), a_star, all_groups) for s in SEEDS])),
                 "zero_shot": float(np.median([slot_jsd(rates_of(s, "zeroshot"), a_star, all_groups)
                                               for s in SEEDS]))})
    return rows


def judge(lam: float = LAM, tau: float = TAU, bptt: int = BPTT, steps: int = STEPS, lr_cond: float = LR_COND,
          lr_time: float = LR_TIME, lr_rest: float = LR_REST) -> pd.DataFrame:
    """E0〜E2 の CSV から J1〜J6 を判定する（GRU の stage2.judge と同じ規則・同じ比較先の DDPM の値）

    Note:
        判定の後に MAE・JSD の参考行（reference_rows）を足して、同じ CSV に書き直す
    """
    variant = variant_tag(lam, tau, bptt, steps, lr_cond, lr_time, lr_rest)
    out = OUT_DIR / f"stage2_lstm{stage1_tag()}{variant}_judge.csv"
    table = gs2.judge(path_of=lambda seed, run_: csv_path(seed, run_, variant), out_path=out, label="lstm")
    table = pd.concat([table, pd.DataFrame(reference_rows(variant))], ignore_index=True)
    table.to_csv(out, index=False)
    return table


def smoke() -> None:
    """2 更新・群あたり 4 本で学習と軸 1 の採点が通ることだけを確かめる（何も保存しない）"""
    tgt = st.load_stula_targets()
    model = lm.load_model(stage1_ckpt(LGO_SEED))
    teacher = teacher_mask_of(tgt, 0)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = train_stage2(model, tgt, teacher, LGO_SEED, steps=2, d_sub=2, n=4, val_every=1)
    moved = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    rows, rates = evaluate(model, tgt, {"fold0": teacher}, {"run": "smoke"}, n_per_group=4, full=False)
    assert len(history) == 2 and all(math.isfinite(h["L_agg"]) for h in history)
    assert {"in-teacher", "held-out", "all"} <= {r["eval_kind"] for r in rows}
    assert rates.shape == (sm.D_GROUPS, sm.NUM_ACT, sm.NUM_SLOTS)
    print(f"smoke: OK（{len(rows)} 行、更新されたパラメータ {len(moved)}/{len(before)}）")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="LSTM_Aggregate の Stage 2（生成の連鎖を通して L_agg を逆伝播する）")
    ap.add_argument("--seed", type=int, default=LGO_SEED)
    ap.add_argument("--fold", type=int, default=None, help="教師から外す LGO の fold（0〜6）。省略で 28 群すべて")
    ap.add_argument("--zero-shot", action="store_true", help="Pre-trained のまま採点する（E0）")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--bptt", type=int, default=BPTT)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--d-sub", type=int, default=D_SUB)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--lr-cond", type=float, default=LR_COND)
    ap.add_argument("--lr-time", type=float, default=LR_TIME)
    ap.add_argument("--lr-rest", type=float, default=LR_REST)
    ap.add_argument("--eval-step", type=int, default=None, help="学習せず、保存済みの step の ckpt を採点する")
    ap.add_argument("--judge", action="store_true", help="E0〜E2 の CSV から J1〜J6 を出す")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.zero_shot and (args.fold is not None or args.eval_step is not None):
        ap.error("--zero-shot は --fold・--eval-step と併用できない")
    if args.fold is not None and not 0 <= args.fold < lgo.N_FOLDS:
        ap.error(f"--fold は 0〜{lgo.N_FOLDS - 1}: {args.fold}")
    if args.smoke:
        smoke()
    elif args.judge:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
            print(judge(args.lam, args.tau, args.bptt, args.steps, args.lr_cond, args.lr_time,
                        args.lr_rest).to_string(index=False))
    else:
        run(args.seed, args.fold, args.zero_shot, args.lam, args.tau, args.bptt, args.steps, args.d_sub, args.n,
            args.lr_cond, args.lr_time, args.lr_rest, args.eval_step)


if __name__ == "__main__":
    main()
