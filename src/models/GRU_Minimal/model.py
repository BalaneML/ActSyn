"""
model.py
========
LSTM と同じ条件の GRU（GRU_Minimal）。最低限の構成 ＋ 学習型の時刻符号の Stage 1

LSTM_Aggregate/model.py の LSTMScheduler と、再帰の層（セル）だけが違う条件付きの自己回帰モデル。
1 日の 96 スロットを 04:00 から順に 1 つずつ生成する。
報告: src/models/LSTM_Aggregate/docs/Stage1_lstm_results.md（LSTM 最低限と GRU 最低限の比較）
      src/models/LSTM_Aggregate/docs/Stage1_time_encoding_results.md（時刻符号 none / fixed / learned の比較）

LSTMScheduler（lm）と同じもの:
    - 部品: act_embed・cond_embeds・cond_proj・out_proj（bias 12 個）。slot_bias・CFG・学習後の補正はない
    - 時刻符号: lm.TimeInput を使う（--time-enc none / fixed / learned、既定 learned。2026-10-06 採用）。
      __init__ の最後に作る
    - 幅と層: HIDDEN = 64・NUM_LAYERS = 1（lm から取る）
    - 初期値: PyTorch の既定
    - 学習: lm.train に build_model=GRUMinimalScheduler を渡す。損失（既定は重みなし）・最適化・早期終了・
      epoch の上限は LSTM と同じ関数と値になる
    - 生成: lm.group_pool・lm.write_pool（28 群 × 256 本、乱数 12345、温度 1.0）
違うもの:
    - 再帰の層: nn.LSTM → nn.GRU。状態は (h, c) → h だけ
    - パラメータ数: 36,052 → 27,732（再帰の層のゲートの重みが 4 組 → 3 組。none のとき。learned は各 +6,144）

GRU_Aggregate/model.py の GRUScheduler（GRU H = 128）との違い:
    slot_bias・CFG（cond_null）・学習後の補正がない。時刻符号は既定で learned（fixed が GRUScheduler と同じ φ）。
    H = 128・2 層 → 64・1 層。
    初期値は PyTorch の既定（GRUScheduler は out_proj を零、slot_bias を log(行動者率) で初期化）。
    損失は重みなし（GRUScheduler は TUFINLWGT の重み付き）

構造（LSTMScheduler の lstm を gru に替えたもの）:

```mermaid
flowchart LR
    PREV["a_prev (B, S)<br/>直前の活動（s=0 は BOS=12）"] --> EA["act_embed<br/>Embedding(13, HIDDEN)"]
    COND["cond_idx (B, 3)"] --> EC["embed_cond<br/>cond_embeds → cond_proj"]
    SLOT["slots = arange(S)"] --> TI["time_input(slots)（lm.TimeInput）<br/>fixed / learned。none は作らない"]
    EA --> X["x (B, S, HIDDEN) = 和"]
    EC --> X
    TI --> X
    X --> GRU["gru<br/>GRU(HIDDEN, HIDDEN, NUM_LAYERS)"]
    GRU --> OUT["out_proj(h)<br/>logits (B, S, 12)"]
```

学習と生成（関数はすべて LSTM_Aggregate/model.py の lm のもの）:

```mermaid
flowchart TD
    TR["train → lm.train(build_model=GRUMinimalScheduler, time_enc,<br/>hidden, num_layers, weight_decay)<br/>lm.batch_loss（既定は重みなし）"] --> CK["lm.save_ckpt(model, ckpt_path(seed, weighted, time_enc,<br/>hidden, num_layers, weight_decay))"]
    CK --> GP["lm.group_pool(model, lm.POOL_N)<br/>pool (28, 256, 96)"]
    GP --> CSV["lm.write_pool<br/>pool_path(seed, weighted, time_enc, hidden, num_layers, weight_decay)"]
```

使い方:
    # 動作確認（2 epoch 学習・群あたり 2 本だけ生成。何も保存しない）
    .venv/bin/python src/models/GRU_Minimal/model.py --smoke

    # 学習して生成プールを書く（既定は学習型の時刻符号。保存先に _time_learned）
    .venv/bin/python src/models/GRU_Minimal/model.py --seed 42

    # 保存済みの ckpt から生成プールだけを作る
    .venv/bin/python src/models/GRU_Minimal/model.py --seed 42 --pool-only

    # 時刻符号を足す前の最低限の構成（2026-10-06 までの GRU 最低限）を学習する
    .venv/bin/python src/models/GRU_Minimal/model.py --seed 42 --time-enc none

    フラグ（LSTM_Aggregate/model.py と同じ）:
        --seed S         : 学習の乱数の種（既定 42）。分割は変えない。42 以外は保存先に _s{S}
        --epochs N       : 学習 epoch の上限（既定 lm.EPOCHS=1000）
        --weighted-loss  : TUFINLWGT の重み付き交差エントロピーで学習する。保存先に _weighted
        --time-enc K     : 時刻符号の種類 none / fixed / learned（既定 learned）。none 以外は保存先に _time_{K}
        --hidden H・--num-layers L・--weight-decay W : 幅・層数・weight decay（既定 64・1・1.0）。
                           どれかが既定と違えば、保存先に _h{H}[_l{L}]_wd{W}（lm.size_tag）
        --no-pool        : 学習後の生成プールを作らない
        --pool-only      : 学習せず、保存済みの ckpt から生成プールだけを作る
        --smoke          : 短時間の動作確認

出力:
    outputs/checkpoints/gru_minimal{損失}{時刻}{構造}{接尾辞}.pt          最良の ckpt（config に学習曲線 history・損失・
                                                                      時刻符号・weight decay）
    outputs/generated/gru_minimal_samples{損失}{時刻}{構造}{接尾辞}.csv   生成プール（28 群 × 256 本）
        {損失} は重みなしなら空、--weighted-loss なら _weighted。{時刻} は none なら空、それ以外は _time_fixed /
        _time_learned。{構造} は既定の幅・層数・weight decay なら空、それ以外は lm.size_tag（例: _h128_wd0.01）。
        {接尾辞} は種 42 なら空、それ以外は _s{seed}
        ★GRU_Aggregate の gru_aggregate*.pt と取り違えないよう、名前の頭を gru_minimal にした
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/models/GRU_Minimal -> repo root


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


# 学習・生成の関数と値は LSTM のものを使う（条件をセル以外そろえるため。写し書きしない）
lm: Any = _load("lstm_aggregate_model", REPO_ROOT / "src" / "models" / "LSTM_Aggregate" / "model.py")
sm: Any = lm.sm

GRUState = torch.Tensor                            # h, (NUM_LAYERS, B, HIDDEN)

# ============================================================
# 1. 設定（構造・学習・生成の値はすべて lm から取る）
# ============================================================
MODEL_SAVE_PATH = REPO_ROOT / "outputs" / "checkpoints" / "gru_minimal.pt"
GEN_SAVE_PATH = REPO_ROOT / "outputs" / "generated" / "gru_minimal_samples.csv"

NUM_ACT: int = lm.NUM_ACT                          # 12
HIDDEN: int = lm.HIDDEN                            # 64
NUM_LAYERS: int = lm.NUM_LAYERS                    # 1
EPOCHS: int = lm.EPOCHS                            # 1000
SEED: int = lm.SEED                                # 42
WEIGHTED_LOSS: bool = lm.WEIGHTED_LOSS             # False（重みなし）
TIME_ENC: Any = lm.TIME_ENC                        # "learned"（学習型の時刻符号。2026-10-06 採用）
WEIGHT_DECAY: float = lm.WEIGHT_DECAY              # 1.0
DEVICE: str = lm.DEVICE


# ============================================================
# 2. モデル
# ============================================================
class GRUMinimalScheduler(nn.Module):
    """条件付きの自己回帰モデル p_θ(a_s | a_<s, c)。LSTMScheduler の再帰の層を GRU に替えたもの

    Note:
        1. 部品・その作る順・forward・step の形は LSTMScheduler と同じ。違いは self.gru（nn.GRU）だけ
        2. ★time_enc が none なら時刻を入力しないので、スロット s の位置は GRU が BOS から数えて状態 h に持つしかない
        3. 初期値はすべて PyTorch の既定。nn.GRU は一様分布 U(−1/√H, 1/√H)
        4. time_input（lm.TimeInput）は __init__ の最後に作る。none では作らない

    Attributes:
        hidden: 隠れ状態の幅 H
        num_layers: GRU の層数
        time_enc: 時刻符号の種類 none / fixed / learned
        time_input: 時刻のベクトルを作る部品。none では None
    """

    def __init__(self, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, time_enc: Any = TIME_ENC) -> None:
        """部品を作る（作る順は LSTMScheduler と同じ）

        Args:
            hidden: 隠れ状態の幅 H, default=HIDDEN=64
            num_layers: GRU の層数, default=NUM_LAYERS=1
            time_enc: 時刻符号の種類 none / fixed / learned, default=TIME_ENC="learned"

        Raises:
            ValueError: 未知の time_enc のとき
        """
        lm.check_time_enc(time_enc)
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.time_enc: str = time_enc
        self.act_embed = nn.Embedding(NUM_ACT + 1, hidden)               # 12 活動 + BOS
        self.cond_embeds = nn.ModuleList([nn.Embedding(card, dim) for _, card, dim, _ in sm.COND_SPEC])
        self.cond_proj = nn.Linear(sum(sm.EMB_DIMS), hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden, NUM_ACT)
        # ★最後に作る（ほかの部品の初期値を none と同じ乱数にするため）
        self.time_input: nn.Module | None = None if time_enc == "none" else lm.TimeInput(time_enc, hidden)

    def add_time(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """入力 x に生成するスロットの時刻のベクトルを足す。none ならそのまま返す（LSTMScheduler.add_time と同じ）

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
            a_prev: 直前の活動, dtype=int64, (B, S)。lm.gm.shift_right(sched) の形。値域 [0, 13)
            cond_idx: 条件インデックス, dtype=int64, (B, 3)

        Returns:
            logits, (B, S, NUM_ACT)。スロット s の logits は a_prev[:, :s+1] だけに依存する
        """
        x = self.act_embed(a_prev) + self.embed_cond(cond_idx)[:, None, :]     # (B, S, H)
        x = self.add_time(x, torch.arange(a_prev.size(1), device=a_prev.device))
        h, _ = self.gru(x)                                                      # 最上層の h_t, (B, S, H)
        return self.out_proj(h)

    def step(self, a_prev: torch.Tensor, c_emb: torch.Tensor,
             state: GRUState | None, slot: int) -> tuple[torch.Tensor, GRUState]:
        """1 スロット分だけ進める（生成用）。forward の同じスロットと同じ値を返す

        Args:
            a_prev: 直前の活動, dtype=int64, (B,)。s = 0 では BOS
            c_emb: embed_cond の出力, (B, HIDDEN)
            state: GRU の状態 h, (NUM_LAYERS, B, HIDDEN)。s = 0 では None（0 から始める）
            slot: 生成するスロットの番号 s, 値域 [0, 96)。time_enc が none なら使わない

        Returns:
            (このスロットの logits (B, NUM_ACT), 次の状態 h)
        """
        x = self.act_embed(a_prev) + c_emb                                      # (B, H)
        x = self.add_time(x, torch.tensor(slot, device=a_prev.device))
        h, next_state = self.gru(x[:, None, :], state)                          # h: (B, 1, H)
        return self.out_proj(h[:, 0]), next_state


# ============================================================
# 3. 学習と読み込み
# ============================================================
def train(epochs: int = EPOCHS, seed: int = SEED, save_path: Path | None = MODEL_SAVE_PATH,
          device: str = DEVICE, weighted: bool = WEIGHTED_LOSS, time_enc: Any = TIME_ENC,
          hidden: int = HIDDEN, num_layers: int = NUM_LAYERS,
          weight_decay: float = WEIGHT_DECAY) -> tuple[GRUMinimalScheduler, dict[str, list[float]]]:
    """LSTM と同じ手順で学習する（lm.train にこのモデルの作り方を渡す）

    Args:
        epochs: 学習 epoch の上限, default=EPOCHS=1000
        seed: 学習の乱数の種（初期値・並び順）, default=SEED=42。分割は変えない
        save_path: 最良の ckpt の保存先, default=MODEL_SAVE_PATH。None なら何も保存しない
        device: 学習するデバイス, default=DEVICE
        weighted: 損失を TUFINLWGT で重み付けするか, default=WEIGHTED_LOSS=False
        time_enc: 時刻符号の種類 none / fixed / learned, default=TIME_ENC="learned"
        hidden: 隠れ状態の幅 H, default=HIDDEN=64
        num_layers: GRU の層数, default=NUM_LAYERS=1
        weight_decay: 重み行列に掛ける AdamW の weight decay, default=WEIGHT_DECAY=1.0

    Returns:
        (最良の重みを戻したモデル, 学習曲線 {"epoch", "train", "val", "sec"})
    """
    model, history = lm.train(epochs=epochs, seed=seed, save_path=save_path, device=device,
                              weighted=weighted, build_model=GRUMinimalScheduler, time_enc=time_enc,
                              hidden=hidden, num_layers=num_layers, weight_decay=weight_decay)
    return model, history


def load_model(path: Path, device: str = DEVICE) -> GRUMinimalScheduler:
    """lm.save_ckpt で保存した ckpt から、同じ構造のモデルを eval モードで戻す

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
    model = GRUMinimalScheduler(hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                                time_enc=cfg.get("time_encoding", "none")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ============================================================
# 4. 保存先と CLI
# ============================================================
def _tags(seed: int, weighted: bool, time_enc: Any, hidden: int, num_layers: int, weight_decay: float) -> str:
    """保存先の印の並び（lm.ckpt_path と同じ: 損失・時刻符号・構造・種）"""
    return (f"{lm.loss_tag(weighted)}{lm.time_tag(time_enc)}{lm.size_tag(hidden, num_layers, weight_decay)}"
            f"{lm.run_suffix(seed)}")


def ckpt_path(seed: int, weighted: bool = WEIGHTED_LOSS, time_enc: Any = TIME_ENC, hidden: int = HIDDEN,
              num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> Path:
    """最良の ckpt のパス（例: 重み付き・時刻符号なし・種 43 は gru_minimal_weighted_s43.pt、
    learned・種 43 は gru_minimal_time_learned_s43.pt、learned・H = 128・wd = 0.01・種 42 は
    gru_minimal_time_learned_h128_wd0.01.pt）。規則は lm と同じ"""
    tags = _tags(seed, weighted, time_enc, hidden, num_layers, weight_decay)
    return MODEL_SAVE_PATH.with_name(f"{MODEL_SAVE_PATH.stem}{tags}{MODEL_SAVE_PATH.suffix}")


def pool_path(seed: int, weighted: bool = WEIGHTED_LOSS, time_enc: Any = TIME_ENC, hidden: int = HIDDEN,
              num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> Path:
    """生成プールの CSV のパス（例: 重み付き・時刻符号なし・種 43 は gru_minimal_samples_weighted_s43.csv、
    fixed・種 42 は gru_minimal_samples_time_fixed.csv）"""
    tags = _tags(seed, weighted, time_enc, hidden, num_layers, weight_decay)
    return GEN_SAVE_PATH.with_name(f"{GEN_SAVE_PATH.stem}{tags}{GEN_SAVE_PATH.suffix}")


def smoke(weighted: bool = WEIGHTED_LOSS, time_enc: Any = TIME_ENC, hidden: int = HIDDEN,
          num_layers: int = NUM_LAYERS, weight_decay: float = WEIGHT_DECAY) -> None:
    """2 epoch 学習し、生成の形を確かめる（何も保存しない）"""
    model, history = train(epochs=2, save_path=None, weighted=weighted, time_enc=time_enc, hidden=hidden,
                           num_layers=num_layers, weight_decay=weight_decay)
    assert len(history["val"]) == 2 and all(math.isfinite(v) for v in history["val"])
    pool = lm.group_pool(model, 2)
    assert pool.shape == (lm.D_GROUPS, 2, lm.NUM_SLOTS) and 0 <= pool.min() and pool.max() < NUM_ACT
    print(f"smoke: OK (params={lm.count_params(model):,})")


def main() -> None:
    """CLI（フラグは LSTM_Aggregate/model.py と同じ）"""
    ap = argparse.ArgumentParser(description="GRU_Minimal: LSTM と同じ条件の GRU（最低限の構成）の Stage 1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--time-enc", choices=lm.TIME_ENC_KINDS, default=TIME_ENC)
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
        lm.write_pool(model, pool)


if __name__ == "__main__":
    main()
