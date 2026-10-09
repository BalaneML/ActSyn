"""
model.py
========
再帰型（LSTM）＋交差エントロピーの Stage 1（LSTM_Aggregate）。最低限の構成 ＋ 学習型の時刻符号

1 日の 96 スロットを 04:00 から順に 1 つずつ生成する条件付きの自己回帰モデル。
入力は直前の活動と属性（性・年齢・就業）だけにした土台に、活動スケジュールに合わせた部品を 1 つずつ足していく。
足した部品: 学習型の時刻符号（2026-10-06 採用、既定。--time-enc none で足す前の最低限の構成に戻る）
解説と参考資料: src/models/LSTM_Aggregate/docs/lstm_primer.md
時刻符号の比較の報告: src/models/LSTM_Aggregate/docs/Stage1_time_encoding_results.md

GRU_Aggregate/model.py の GRUScheduler との違い（入れていない部品）:
    - 時刻符号: 既定は learned（2026-10-06 採用）。切替は TimeInput（--time-enc）:
          none    : 時刻符号なし。スロットの位置は LSTM が BOS から数えて状態 (h, c) に持つしかない
          fixed   : GRU と同じ Transformer 型 96 次元の固定の φ ＋ time_proj = Linear(96, HIDDEN)
          learned : time_embed = Embedding(96, HIDDEN)。スロットごとに自由なベクトルを学習する（既定）
      ★GRU は fixed。比較の結果（val・12:00 の食事の段差）で learned を採った。報告は上の時刻符号の比較の報告
      ★fixed の φ（bias の列を足した 96 × 97）は、float64 では満階数だが特異値が急に小さくなる
        （最大 53.3、34 番目で最大の 1e-8 倍。float32 での数値的な階数は 27）。重みのノルムを 3 以下に抑えると、
        12:00 の 1 スロットだけを持ち上げる形の最良の線形近似は「ピーク 0.26・±15 分 0.23・±30 分 0.14」の
        幅約 1.5 時間の山になる（完全に作るにはノルム約 9×10⁸ が要る）。learned はノルム 1 でどの形も作れる
        （ランク ≤ HIDDEN）。追加パラメータは fixed 6,208・learned 6,144（H = 64）
    - slot_bias（96 × 12）: なし。活動ごとの定数は out_proj の bias（12）が持つ
    - CFG（cond_null と、学習で条件を落とす P_UNCOND）: なし。生成は条件付きの logits そのもの
    - 学習後の slot_bias の補正: なし
    - 再帰の層: GRU 2 層・H=128 → LSTM 1 層・H=64
    - 初期値: すべて PyTorch の既定（GRU は out_proj を零、slot_bias を log(行動者率) で初期化）
    - 損失: 重みなしの交差エントロピー（2026-10-06 から。GRU は TUFINLWGT の重み付き。--weighted-loss で旧設定）

★次は GRU_Aggregate/model.py（gm）から import する（写し書きしない）:
  データの分割とミニバッチ（load_split・make_loader）、直前の活動の作り方（shift_right・BOS）、
  重み付き交差エントロピー（weighted_ce。--weighted-loss のときだけ）、活動の引き方（draw_categorical）。
  データ・条件の符号化・生成プールの CSV 形式の出所は、GRU と同じ DDPM_Aggregate_Simple/model.py（sm = gm.sm）

★GRU_Minimal/model.py（LSTM と同じ条件の GRU）は、train に GRUMinimalScheduler を渡し、この学習と生成の関数を
  そのまま使う。引数の型は LSTMScheduler と書いてあるが、forward・step・embed_cond・hidden・num_layers が
  同じ形なので GRUMinimalScheduler も渡せる（違いはセルだけ）

損失（teacher forcing。N は人数、w_i は TUFINLWGT）:
    重みなし（既定）             L = − Σ_i Σ_s log p_θ(a_{i,s} | a_{i,<s}, c_i) / (96 · N)
    重み付き（--weighted-loss）  L = − Σ_i w_i Σ_s log p_θ(a_{i,s} | a_{i,<s}, c_i) / (96 · Σ_i w_i)

構造:

```mermaid
flowchart LR
    PREV["a_prev (B, S)<br/>直前の活動（s=0 は BOS=12）"] --> EA["act_embed<br/>Embedding(13, HIDDEN)"]
    COND["cond_idx (B, 3)"] --> EC["embed_cond<br/>cond_embeds → cond_proj"]
    SLOT["slots = arange(S)<br/>生成するスロットの番号"] --> TI["time_input(slots) (S, HIDDEN)<br/>fixed: time_proj(phi[slots])<br/>learned: time_embed(slots)<br/>none: 作らない・足さない"]
    EA --> X["x (B, S, HIDDEN) = 和"]
    EC --> X
    TI --> X
    X --> LSTM["lstm<br/>LSTM(HIDDEN, HIDDEN, NUM_LAYERS)"]
    LSTM --> OUT["out_proj(h)<br/>logits (B, S, 12)"]
```

    ★time_input は __init__ の最後に作る。種が同じなら、ほかの部品の初期値は none / fixed / learned で一致する

学習と生成:

```mermaid
flowchart TD
    LD["gm.load_split<br/>train_part 3,363 人 / val_part 373 人"] --> TR["train(build_model=LSTMScheduler, time_enc,<br/>hidden, num_layers, weight_decay)<br/>batch_loss: F.cross_entropy（重みなし）<br/>--weighted-loss なら gm.weighted_ce"]
    TR --> CK["save_ckpt(model, ckpt_path(seed, weighted, time_enc,<br/>hidden, num_layers, weight_decay))"]
    CK --> GP["group_pool(model, POOL_N)<br/>sample → model.step(a_prev, c_emb, state, s)<br/>pool (28, 256, 96)"]
    GP --> CSV["write_pool → sm.write_pool_csv<br/>pool_path(seed, weighted, time_enc, hidden, num_layers, weight_decay)"]
```

使い方:
    # 動作確認（2 epoch 学習・群あたり 2 本だけ生成。何も保存しない）
    .venv/bin/python src/models/LSTM_Aggregate/model.py --smoke

    # 学習して生成プールを書く（既定は学習型の時刻符号。保存先に _time_learned）
    .venv/bin/python src/models/LSTM_Aggregate/model.py --seed 42

    # 保存済みの ckpt から生成プールだけを作る
    .venv/bin/python src/models/LSTM_Aggregate/model.py --seed 42 --pool-only

    # 時刻符号を足す前の最低限の構成（2026-10-06 までの LSTM 最低限）を学習する
    .venv/bin/python src/models/LSTM_Aggregate/model.py --seed 42 --time-enc none

    # 幅と weight decay を変えて学習する（保存先に _h128_wd0.01）
    .venv/bin/python src/models/LSTM_Aggregate/model.py --seed 42 --hidden 128 --weight-decay 0.01

    フラグ:
        --seed S         : 学習の乱数の種（既定 42）。分割は変えない。42 以外は保存先に _s{S}
        --epochs N       : 学習 epoch の上限（既定 EPOCHS=1000）
        --weighted-loss  : TUFINLWGT の重み付き交差エントロピーで学習する（2026-10-02 の旧設定）。保存先に _weighted
        --time-enc K     : 時刻符号の種類 none / fixed / learned（既定 learned）。none 以外は保存先に _time_{K}
        --hidden H       : 隠れ状態の幅（既定 HIDDEN=64）
        --num-layers L   : 再帰の層数（既定 NUM_LAYERS=1）
        --weight-decay W : 重み行列に掛ける weight decay（既定 WEIGHT_DECAY=1.0）。
                           3 つのどれかが既定と違えば、保存先に _h{H}[_l{L}]_wd{W}（size_tag）
        --no-pool        : 学習後の生成プールを作らない
        --pool-only      : 学習せず、保存済みの ckpt から生成プールだけを作る
        --smoke          : 短時間の動作確認

出力:
    outputs/checkpoints/lstm_aggregate{損失}{時刻}{構造}{接尾辞}.pt          最良の ckpt（config に学習曲線 history・損失・
                                                                            時刻符号・weight decay）
    outputs/generated/lstm_aggregate_samples{損失}{時刻}{構造}{接尾辞}.csv   生成プール（28 群 × 256 本）
        {損失} は重みなしなら空、--weighted-loss なら _weighted。{時刻} は none なら空、それ以外は _time_fixed /
        _time_learned（既定の learned でも付ける。名前で時刻符号が分かるように）。{構造} は既定の幅・層数・weight decay
        なら空、それ以外は size_tag（例: _h128_wd0.01）。{接尾辞} は種 42 なら空、それ以外は _s{seed}
        ★2026-10-02 の重み付き・種 42 の成果物は outputs/archive/lstm_weighted/ に退避した
"""
import argparse
import copy
import importlib.util
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/LSTM_Aggregate -> repo root


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同名の model.py を取り違えないため）"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 分割・ミニバッチ・重み付きの損失・活動の引き方は GRU と同じ関数を使う
gm: Any = _load("lstm_gru_model", REPO_ROOT / "src" / "models" / "GRU_Aggregate" / "model.py")
# データ・条件の符号化・CSV 形式の唯一の出所。GRU が読み込んだものを使う（同じファイルを 2 回読み込まない）
sm: Any = gm.sm

IntArr = npt.NDArray[np.int64]
LSTMState = tuple[torch.Tensor, torch.Tensor]   # (h, c)。どちらも (NUM_LAYERS, B, HIDDEN)

# ============================================================
# 1. 設定
# ============================================================
MODEL_SAVE_PATH = REPO_ROOT / "outputs" / "checkpoints" / "lstm_aggregate.pt"
GEN_SAVE_PATH = REPO_ROOT / "outputs" / "generated" / "lstm_aggregate_samples.csv"

NUM_SLOTS: int = sm.NUM_SLOTS        # 96（04:00 起点の 15 分）
NUM_ACT: int = sm.NUM_ACT            # 12（共通 12 分類）
D_GROUPS: int = sm.D_GROUPS          # 28
BOS: int = gm.BOS                    # 12。s = 0 の直前の活動の代わりに入れる記号

# 構造（2026-10-02 ユーザー指定）。パラメータ 36,052（GRU の H=128・2 層は 217,288）
HIDDEN = 64
NUM_LAYERS = 1

# 学習。★値は GRU（H=128・2 層）の掃引で選んだもの（GRU の計画書 §10）。H=64・1 層に合わせた調整はしていない
LR = 1e-3
WEIGHT_DECAY = 1.0                   # 重み行列だけに掛ける（1 次元の bias には掛けない。param_groups）
GRAD_CLIP = 1.0
# 2026-10-06 に 300 → 1000。旧 LSTM は最良 epoch 290 で上限の直前だった（GRU_Minimal との比較で上限に切られないように）
EPOCHS = 1000
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 1e-4
SEED: int = sm.SEED                  # 42。学習の乱数の既定値（分割は gm.load_split が固定）
# 損失（2026-10-06 ユーザー指示「重み付き損失 → シンプル損失」）。既定は重みなしの交差エントロピー
WEIGHTED_LOSS = False                # True で TUFINLWGT の重み付き（2026-10-02 の旧設定、--weighted-loss）
WEIGHTED_TAG = "_weighted"           # 重み付きで学習した ckpt・プールの保存先に付ける印

# 時刻符号（2026-10-06 追加）。既定は learned（同日、固定・なしとの比較の後にユーザーが採用）。
# none は時刻符号を入れる前の最低限の構成で、それまでの結果をそのまま再現する
TimeEncoding = Literal["none", "fixed", "learned"]
TIME_ENC_KINDS: tuple[str, ...] = get_args(TimeEncoding)
TIME_ENC: TimeEncoding = "learned"
# fixed の φ。GRU_Aggregate と同じ Transformer 型 96 次元（DDPM の clock_tf96 と同じ）
FIXED_TIME_ARCH: Any = sm.ArchSpec(clock_kind="transformer")
# ミニバッチは gm.make_loader（GRU と同じ 256 人。全員を毎 epoch 使う。TUFINLWGT は重み付きのときだけ損失に掛ける）

# 生成（GRU と同じ値）
POOL_N = 256                         # 群あたりの本数
POOL_SEED = 12345                    # DDPM・GRU の共通乱数プールと同じ値
GEN_BATCH = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


# ============================================================
# 2. モデル
# ============================================================
def check_time_enc(time_enc: str) -> None:
    """時刻符号の種類が TIME_ENC_KINDS のどれかであることを確かめる

    Raises:
        ValueError: 未知の種類のとき
    """
    if time_enc not in TIME_ENC_KINDS:
        raise ValueError(f"time_enc は {TIME_ENC_KINDS} のいずれか: {time_enc}")


class TimeInput(nn.Module):
    """生成するスロットの番号を、入力に足す時刻のベクトルにする, (S,) -> (S, HIDDEN)

    Note:
        1. fixed  : time_proj(phi[slots])。phi は sm.time_features の固定の φ（Transformer 型 96 次元）で、
                    学習するのは time_proj だけ。GRU_Aggregate の time_proj と同じ形
        2. learned: time_embed(slots)。スロットごとのベクトルをそのまま学習する
        3. ★learned は「96 次元の完全な基底（one-hot）＋ Linear（bias なし）」と同じ。
           fixed の φ も数値の上では満階数だが、特異値が急に小さくなるので、現実的な大きさの重みでは
           幅約 1.5 時間より細い形を作れない（数字はモジュールの docstring）。違いは作れる形の制限の強さ
        4. none の部品は作らない（モデル側で time_input = None にする）

    Attributes:
        kind: 時刻符号の種類, "fixed" か "learned"
        phi: 固定の時刻符号 φ, (96 スロット, 96 次元)。fixed のときだけ持つ、保存しない buffer
        time_proj: φ を隠れ層の幅へ写す Linear(96, HIDDEN)。fixed のときだけ持つ
        time_embed: スロットごとの埋め込み Embedding(96, HIDDEN)。learned のときだけ持つ
    """

    phi: torch.Tensor
    time_proj: nn.Linear
    time_embed: nn.Embedding

    def __init__(self, kind: TimeEncoding, hidden: int) -> None:
        """部品を作る（初期値は PyTorch の既定）

        Args:
            kind: 時刻符号の種類, "fixed" か "learned"
            hidden: 隠れ層の幅 H

        Raises:
            ValueError: kind が fixed でも learned でもないとき
        """
        super().__init__()
        self.kind = kind
        if kind == "fixed":
            self.register_buffer("phi", sm.time_features(FIXED_TIME_ARCH).T.contiguous(), persistent=False)
            self.time_proj = nn.Linear(FIXED_TIME_ARCH.clock_dim, hidden)
        elif kind == "learned":
            self.time_embed = nn.Embedding(NUM_SLOTS, hidden)
        else:
            raise ValueError(f"TimeInput の kind は fixed か learned: {kind}")

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """スロットの番号から時刻のベクトルを作る

        Args:
            slots: スロットの番号, dtype=int64, 形は任意（(S,) または 0 次元）。値域 [0, 96)

        Returns:
            時刻のベクトル, (*slots.shape, HIDDEN)
        """
        if self.kind == "fixed":
            return self.time_proj(self.phi[slots])
        return self.time_embed(slots)


class LSTMScheduler(nn.Module):
    """条件付きの自己回帰モデル p_θ(a_s | a_<s, c[, s])（最低限の構成）

    Note:
        1. 各スロットの入力は、直前の活動の埋め込みと属性の埋め込みの和。time_enc が none 以外なら
           生成するスロット s の時刻のベクトル time_input(s) も足す
        2. ★none では時刻を入力しないので、スロット s の位置は LSTM が BOS から数えて状態 (h, c) に持つしかない
        3. 初期値はすべて PyTorch の既定。nn.LSTM は一様分布 U(−1/√H, 1/√H) なので、
           学習前の忘却ゲートは σ(≈0) ≈ 0.5（解説は docs/lstm_primer.md §2）
        4. time_input は __init__ の最後に作る。種が同じなら、ほかの部品の初期値は 3 種類で一致する。
           none では作らないので、パラメータ数と state_dict のキーは時刻符号を入れる前と同じ

    Attributes:
        hidden: 隠れ状態の幅 H
        num_layers: LSTM の層数
        time_enc: 時刻符号の種類 none / fixed / learned
        time_input: 時刻のベクトルを作る部品。none では None
    """

    def __init__(self, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS,
                 time_enc: TimeEncoding = TIME_ENC) -> None:
        """部品を作る

        Args:
            hidden: 隠れ状態の幅 H, default=HIDDEN=64
            num_layers: LSTM の層数, default=NUM_LAYERS=1
            time_enc: 時刻符号の種類, default=TIME_ENC="learned"

        Raises:
            ValueError: 未知の time_enc のとき
        """
        check_time_enc(time_enc)
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.time_enc: TimeEncoding = time_enc
        self.act_embed = nn.Embedding(NUM_ACT + 1, hidden)               # 12 活動 + BOS
        self.cond_embeds = nn.ModuleList([nn.Embedding(card, dim) for _, card, dim, _ in sm.COND_SPEC])
        self.cond_proj = nn.Linear(sum(sm.EMB_DIMS), hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden, NUM_ACT)
        # ★最後に作る（ほかの部品の初期値を none と同じ乱数にするため）
        self.time_input: TimeInput | None = None if time_enc == "none" else TimeInput(time_enc, hidden)

    def add_time(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """入力 x に生成するスロットの時刻のベクトルを足す。none ならそのまま返す

        Args:
            x: 時刻を足す前の入力, (B, S, HIDDEN) または (B, HIDDEN)
            slots: スロットの番号, dtype=int64。x が (B, S, H) なら (S,)、(B, H) なら 0 次元

        Returns:
            x と同じ形の入力
        """
        if self.time_input is None:
            return x
        return x + self.time_input(slots)

    def embed_cond(self, cond_idx: torch.Tensor) -> torch.Tensor:
        """属性（性・年齢・就業）を埋め込む, (B, 3) -> (B, HIDDEN)

        Args:
            cond_idx: 条件インデックス, dtype=int64, (B, 3)。列は sm.COND_SPEC の順

        Returns:
            属性の埋め込み, (B, HIDDEN)
        """
        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)   # (B, 16)
        return self.cond_proj(c)

    def forward(self, a_prev: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        """直前の活動の列から、各スロットの活動の logits を一度に出す（学習・teacher forcing 用）

        Args:
            a_prev: 直前の活動, dtype=int64, (B, S)。gm.shift_right(sched) の形。値域 [0, 13)
            cond_idx: 条件インデックス, dtype=int64, (B, 3)

        Returns:
            logits, (B, S, NUM_ACT)。スロット s の logits は a_prev[:, :s+1] だけに依存する
        """
        x = self.act_embed(a_prev) + self.embed_cond(cond_idx)[:, None, :]     # (B, S, H)
        x = self.add_time(x, torch.arange(a_prev.size(1), device=a_prev.device))
        h, _ = self.lstm(x)                                                     # 最上層の h_t, (B, S, H)
        return self.out_proj(h)

    def step(self, a_prev: torch.Tensor, c_emb: torch.Tensor,
             state: LSTMState | None, slot: int) -> tuple[torch.Tensor, LSTMState]:
        """1 スロット分だけ進める（生成用）。forward の同じスロットと同じ値を返す

        Args:
            a_prev: 直前の活動, dtype=int64, (B,)。s = 0 では BOS
            c_emb: embed_cond の出力, (B, HIDDEN)
            state: LSTM の状態 (h, c)。どちらも (NUM_LAYERS, B, HIDDEN)。s = 0 では None（0 から始める）
            slot: 生成するスロットの番号 s, 値域 [0, 96)。time_enc が none なら使わない

        Returns:
            (このスロットの logits (B, NUM_ACT), 次の状態 (h, c))
        """
        return self.step_embedded(self.act_embed(a_prev), c_emb, state, slot)

    def step_embedded(self, e_prev: torch.Tensor, c_emb: torch.Tensor,
                      state: LSTMState | None, slot: int) -> tuple[torch.Tensor, LSTMState]:
        """直前の活動の埋め込みを受け取って 1 スロット分だけ進める（step の本体）

        Note:
            ★Stage 2（stage2_agg.py）は、直前の活動を one-hot y と埋め込み行列の積 y @ act_embed.weight で渡す。
              y が厳密な one-hot なら act_embed(a_prev) と同じ値になり、y に勾配があれば履歴を通して逆伝播できる

        Args:
            e_prev: 直前の活動の埋め込み, (B, HIDDEN)。step からは act_embed(a_prev)
            c_emb: embed_cond の出力, (B, HIDDEN)
            state: LSTM の状態 (h, c)。どちらも (NUM_LAYERS, B, HIDDEN)。s = 0 では None（0 から始める）
            slot: 生成するスロットの番号 s, 値域 [0, 96)。time_enc が none なら使わない

        Returns:
            (このスロットの logits (B, NUM_ACT), 次の状態 (h, c))
        """
        x = e_prev + c_emb                                                      # (B, H)
        x = self.add_time(x, torch.tensor(slot, device=e_prev.device))
        h, next_state = self.lstm(x[:, None, :], state)                         # h: (B, 1, H)
        return self.out_proj(h[:, 0]), next_state


# ============================================================
# 3. 損失と学習
# ============================================================
def batch_loss(model: LSTMScheduler, cond_idx: torch.Tensor, sched: torch.Tensor,
               weight: torch.Tensor, weighted: bool = WEIGHTED_LOSS) -> torch.Tensor:
    """teacher forcing の損失（各スロットの入力は実データの直前の活動）

    Args:
        model: 学習するモデル
        cond_idx: 条件インデックス, dtype=int64, (B, 3)
        sched: 正解の活動, dtype=int64, (B, 96)
        weight: 個票の重み TUFINLWGT, (B,)。weighted=False のときは使わない
        weighted: True なら TUFINLWGT の重み付き（gm.weighted_ce）、False なら重みなし, default=WEIGHTED_LOSS

    Returns:
        交差エントロピー（1 スロットあたりの nats）
    """
    logits = model(gm.shift_right(sched), cond_idx)                            # (B, 96, NUM_ACT)
    if weighted:
        return gm.weighted_ce(logits, sched, weight)
    return F.cross_entropy(logits.reshape(-1, NUM_ACT), sched.reshape(-1))     # B × 96 スロットの単純平均


def run_epoch(model: LSTMScheduler, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None,
              weighted: bool = WEIGHTED_LOSS) -> float:
    """1 epoch の学習（optimizer あり）または評価（なし）を行い、分割全体の交差エントロピーを返す

    Note:
        戻り値はバッチの損失の加重平均。バッチの重みは、weighted なら Σw、重みなしなら人数。
        どちらでも、分割全体で batch_loss を 1 回測った値と同じになる

    Args:
        model: モデル
        loader: gm.make_loader の DataLoader。(cond_idx, sched, weight) を返す
        optimizer: 渡すと学習する。None なら評価だけ
        weighted: batch_loss に渡す, default=WEIGHTED_LOSS

    Returns:
        交差エントロピー（1 スロットあたりの nats）
    """
    is_train = optimizer is not None
    model.train(is_train)
    dev = next(model.parameters()).device
    total, wsum = 0.0, 0.0
    with torch.set_grad_enabled(is_train):
        for cond_idx, sched, weight in loader:
            cond_idx, sched, weight = cond_idx.to(dev), sched.to(dev), weight.to(dev)
            loss = batch_loss(model, cond_idx, sched, weight, weighted)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            w = float(weight.sum()) if weighted else float(sched.size(0))
            total += loss.item() * w
            wsum += w
    return total / wsum


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW のパラメータ群。weight decay は重み行列（Linear・LSTM・Embedding）だけに掛ける

    Note:
        1 次元のパラメータ（bias）には掛けない（慣例どおり。GRU の param_groups と同じ範囲）

    Args:
        model: 学習するモデル
        weight_decay: 重み行列に掛ける AdamW の weight decay

    Returns:
        [{"params": 重み行列, "weight_decay": weight_decay}, {"params": 1 次元, "weight_decay": 0}]
    """
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def train(epochs: int = EPOCHS, seed: int = SEED, save_path: Path | None = MODEL_SAVE_PATH,
          device: str = DEVICE, weighted: bool = WEIGHTED_LOSS,
          build_model: Callable[..., nn.Module] = LSTMScheduler,
          time_enc: TimeEncoding = TIME_ENC, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS,
          weight_decay: float = WEIGHT_DECAY) -> tuple[Any, dict[str, list[float]]]:
    """学習し、val の交差エントロピーが最良の重みのモデルを返す

    Note:
        1. build_model は torch.manual_seed(seed) の後に呼ぶので、初期値も種で決まる
        2. ★GRU_Minimal は build_model=GRUMinimalScheduler を渡す（学習の手順を LSTM と共有する）
        3. val の交差エントロピーも、学習と同じ weighted で測る
        4. モデルは build_model(hidden=hidden, num_layers=num_layers, time_enc=time_enc) で作る
        5. config に weight_decay を残す（幅・層数は save_ckpt が残す）

    Args:
        epochs: 学習 epoch の上限, default=EPOCHS=1000
        seed: 学習の乱数の種（初期値・並び順）, default=SEED=42。分割は変えない
        save_path: 最良の ckpt の保存先。None なら何も保存しない
        device: 学習するデバイス, default=DEVICE
        weighted: 損失を TUFINLWGT で重み付けするか, default=WEIGHTED_LOSS=False
        build_model: キーワード引数 hidden・num_layers・time_enc を受け取ってモデルを作る関数, default=LSTMScheduler
        time_enc: 時刻符号の種類 none / fixed / learned, default=TIME_ENC="learned"
        hidden: 隠れ状態の幅 H, default=HIDDEN=64
        num_layers: 再帰の層数, default=NUM_LAYERS=1
        weight_decay: 重み行列に掛ける AdamW の weight decay, default=WEIGHT_DECAY=1.0

    Returns:
        (最良の重みを戻したモデル, 学習曲線 {"epoch", "train", "val", "sec"})

    Raises:
        ValueError: epochs < 1 のとき、未知の time_enc のとき
    """
    if epochs < 1:
        raise ValueError(f"epochs は 1 以上: {epochs}")
    check_time_enc(time_enc)

    torch.manual_seed(seed)
    train_part, val_part = gm.load_split()
    # ★LSTMScheduler と GRUMinimalScheduler のどちらも来る（forward・step・embed_cond・hidden・num_layers が同じ形）
    model: Any = build_model(hidden=hidden, num_layers=num_layers, time_enc=time_enc).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, weight_decay), lr=LR)
    train_loader = gm.make_loader(train_part, shuffle=True)
    val_loader = gm.make_loader(val_part, shuffle=False)
    print(f"device={device}  arch={type(model).__name__}  train={len(train_part.sched)}  val={len(val_part.sched)}  "
          f"hidden={model.hidden}  num_layers={model.num_layers}  weight_decay={weight_decay:g}  "
          f"weighted_loss={weighted}  time_enc={model.time_enc}  params={count_params(model):,}", flush=True)

    history: dict[str, list[float]] = {"epoch": [], "train": [], "val": [], "sec": []}
    best_val, best_epoch, no_improve = math.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    ep = 0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, optimizer, weighted)
        va = run_epoch(model, val_loader, weighted=weighted)
        sec = time.perf_counter() - t0
        for key, val in (("epoch", ep), ("train", tr), ("val", va), ("sec", sec)):
            history[key].append(float(val))
        if ep == 1 or ep % 5 == 0:
            print(f"epoch {ep:4d} | train {tr:.5f} | val {va:.5f} | {sec:.1f}s", flush=True)
        if va < best_val - EARLY_STOP_MIN_DELTA:
            best_val, best_epoch, no_improve = va, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {ep} (best {best_val:.5f} @ epoch {best_epoch})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"restored best: epoch {best_epoch} (val {best_val:.5f})", flush=True)
    if save_path is not None:
        save_ckpt(model, save_path, seed, extra={
            "arch": type(model).__name__, "weighted_loss": weighted, "weight_decay": weight_decay,
            "best_epoch": best_epoch, "best_val": best_val, "stopped_epoch": ep, "history": history})
        print(f"saved model to {save_path}")
    return model, history


def count_params(model: nn.Module) -> int:
    """学習するパラメータの数"""
    return sum(p.numel() for p in model.parameters())


def save_ckpt(model: LSTMScheduler, path: Path, seed: int, extra: dict[str, Any] | None = None) -> None:
    """重みと出所の記録（config）を {"model", "config"} で保存する

    Note:
        config の time_encoding に時刻符号の種類を残す（load_model が同じ構造を作るため）

    Args:
        model: 保存するモデル
        path: 保存先
        seed: 学習の乱数の種
        extra: config に足す記録（best_epoch・学習曲線など）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"hidden": model.hidden, "num_layers": model.num_layers,
                              "time_encoding": model.time_enc, "seed": seed}
    if extra is not None:
        config.update(extra)
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_model(path: Path, device: str = DEVICE) -> LSTMScheduler:
    """save_ckpt で保存した ckpt から、同じ構造のモデルを eval モードで戻す

    Args:
        path: ckpt のパス
        device: 載せるデバイス, default=DEVICE

    Note:
        時刻符号を入れる前の ckpt には time_encoding が無いので、none として読む

    Returns:
        eval モードのモデル
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = LSTMScheduler(hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                          time_enc=cfg.get("time_encoding", "none")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ============================================================
# 4. 生成
# ============================================================
@torch.no_grad()
def sample(model: LSTMScheduler, cond_idx: torch.Tensor,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """s = 0..95 の順に 1 スロットずつ引いて 1 日を作る（温度 1.0）

    Args:
        model: 学習済みのモデル
        cond_idx: 条件インデックス, dtype=int64, (B, 3)
        generator: CPU の torch.Generator, default=None（大域の乱数）

    Returns:
        生成スケジュール, dtype=int64, (B, 96)。CPU 上、値域 [0, 12)
    """
    dev = next(model.parameters()).device
    batch = cond_idx.size(0)
    out = torch.empty((batch, NUM_SLOTS), dtype=torch.long)
    with sm.eval_mode(model):
        c_emb = model.embed_cond(cond_idx.to(dev))
        a_prev = torch.full((batch,), BOS, dtype=torch.long, device=dev)
        state: LSTMState | None = None
        for s in range(NUM_SLOTS):
            logits, state = model.step(a_prev, c_emb, state, s)
            a = gm.draw_categorical(logits, generator)
            out[:, s] = a
            a_prev = a.to(dev)
    return out


def group_pool(model: LSTMScheduler, n_per_group: int = POOL_N, seed: int = POOL_SEED) -> IntArr:
    """群別サンプルプールを作る, -> (28, M, 96)。行 d は sm.cond_grid()[d] の条件

    Args:
        model: 学習済みのモデル
        n_per_group: 群あたりの本数 M, default=POOL_N=256
        seed: 生成の乱数の種, default=POOL_SEED=12345

    Returns:
        群別サンプルプール, dtype=int64, (D_GROUPS, M, NUM_SLOTS)
    """
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    flat = grid.repeat_interleave(n_per_group, dim=0)                          # (28·M, 3)
    gen = torch.Generator().manual_seed(seed)
    outs = [sample(model, flat[i:i + GEN_BATCH], gen) for i in range(0, flat.size(0), GEN_BATCH)]
    return torch.cat(outs).numpy().astype(np.int64).reshape(D_GROUPS, n_per_group, NUM_SLOTS)


# ============================================================
# 5. 保存先と CLI
# ============================================================
def run_suffix(seed: int) -> str:
    """保存先の接尾辞。種 42 は空、それ以外は _s{seed}（DDPM・GRU と同じ規則）"""
    return "" if seed == SEED else f"_s{seed}"


def loss_tag(weighted: bool) -> str:
    """保存先の損失の印。重みなしは空、重み付きは WEIGHTED_TAG"""
    return WEIGHTED_TAG if weighted else ""


def time_tag(time_enc: TimeEncoding) -> str:
    """保存先の時刻符号の印。none は空、それ以外は _time_{time_enc}"""
    check_time_enc(time_enc)
    return "" if time_enc == "none" else f"_time_{time_enc}"


def size_tag(hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> str:
    """保存先の構造と weight decay の印。既定（H = 64・1 層・wd = 1.0）は空、それ以外は _h{H}[_l{層数}]_wd{wd}

    Note:
        幅か weight decay の一方だけが既定と違っても両方を書く（例: _h128_wd1）。層数は既定と違うときだけ書く。
        GRU_Aggregate/sweep_width_decay.py の _h{H}_wd{wd} と同じ書き方

    Args:
        hidden: 隠れ状態の幅 H, default=HIDDEN=64
        num_layers: 再帰の層数, default=NUM_LAYERS=1
        weight_decay: AdamW の weight decay, default=WEIGHT_DECAY=1.0

    Returns:
        印（例: "_h128_wd0.01"）。既定なら ""
    """
    if (hidden, num_layers, weight_decay) == (HIDDEN, NUM_LAYERS, WEIGHT_DECAY):
        return ""
    layers = "" if num_layers == NUM_LAYERS else f"_l{num_layers}"
    return f"_h{hidden}{layers}_wd{weight_decay:g}"


def ckpt_path(seed: int, weighted: bool = WEIGHTED_LOSS, time_enc: TimeEncoding = TIME_ENC,
              hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> Path:
    """最良の ckpt のパス（例: 重み付き・時刻符号なし・種 43 は lstm_aggregate_weighted_s43.pt、
    learned・種 43 は lstm_aggregate_time_learned_s43.pt、learned・H = 128・wd = 0.01・種 42 は
    lstm_aggregate_time_learned_h128_wd0.01.pt）"""
    tags = f"{loss_tag(weighted)}{time_tag(time_enc)}{size_tag(hidden, num_layers, weight_decay)}{run_suffix(seed)}"
    return MODEL_SAVE_PATH.with_name(f"{MODEL_SAVE_PATH.stem}{tags}{MODEL_SAVE_PATH.suffix}")


def pool_path(seed: int, weighted: bool = WEIGHTED_LOSS, time_enc: TimeEncoding = TIME_ENC,
              hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> Path:
    """生成プールの CSV のパス（例: 重み付き・時刻符号なし・種 43 は lstm_aggregate_samples_weighted_s43.csv、
    fixed・種 42 は lstm_aggregate_samples_time_fixed.csv）。印の並びは ckpt_path と同じ"""
    tags = f"{loss_tag(weighted)}{time_tag(time_enc)}{size_tag(hidden, num_layers, weight_decay)}{run_suffix(seed)}"
    return GEN_SAVE_PATH.with_name(f"{GEN_SAVE_PATH.stem}{tags}{GEN_SAVE_PATH.suffix}")


def write_pool(model: LSTMScheduler, path: Path) -> None:
    """群別サンプルプール（28 群 × POOL_N 本）を作って CSV に書く"""
    t0 = time.perf_counter()
    pool = group_pool(model, POOL_N)
    sm.write_pool_csv(pool, path)
    print(f"saved pool ({D_GROUPS}×{POOL_N}, {time.perf_counter() - t0:.1f}s) to {path}", flush=True)


def smoke(weighted: bool = WEIGHTED_LOSS, time_enc: TimeEncoding = TIME_ENC, hidden: int = HIDDEN,
          num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> None:
    """2 epoch 学習し、生成の形を確かめる（何も保存しない）"""
    model, history = train(epochs=2, save_path=None, weighted=weighted, time_enc=time_enc, hidden=hidden,
                           num_layers=num_layers, weight_decay=weight_decay)
    assert len(history["val"]) == 2 and all(math.isfinite(v) for v in history["val"])
    pool = group_pool(model, 2)
    assert pool.shape == (D_GROUPS, 2, NUM_SLOTS) and 0 <= pool.min() and pool.max() < NUM_ACT
    print(f"smoke: OK (params={count_params(model):,})")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="LSTM_Aggregate: 再帰型（LSTM）＋交差エントロピーの Stage 1（最低限の構成）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--time-enc", choices=TIME_ENC_KINDS, default=TIME_ENC)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.no_pool and args.pool_only:
        ap.error("--no-pool と --pool-only は併用できない")

    size = {"hidden": args.hidden, "num_layers": args.num_layers, "weight_decay": args.weight_decay}
    ckpt = ckpt_path(args.seed, args.weighted_loss, args.time_enc, **size)
    pool = pool_path(args.seed, args.weighted_loss, args.time_enc, **size)
    print(f"[config] seed={args.seed} epochs={args.epochs} weighted_loss={args.weighted_loss}  "
          f"time_enc={args.time_enc}  hidden={args.hidden}  num_layers={args.num_layers}  "
          f"weight_decay={args.weight_decay:g}  ckpt={ckpt.name}  pool={pool.name}")
    if args.smoke:
        smoke(args.weighted_loss, args.time_enc, **size)
        return
    if args.pool_only:
        model = load_model(ckpt)
    else:
        model, _ = train(epochs=args.epochs, seed=args.seed, save_path=ckpt, weighted=args.weighted_loss,
                         time_enc=args.time_enc, **size)
    if not args.no_pool:
        write_pool(model, pool)


if __name__ == "__main__":
    main()
