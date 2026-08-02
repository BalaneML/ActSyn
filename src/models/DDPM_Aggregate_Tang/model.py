"""
model.py (DDPM_Aggregate_Tang)
================
AggDDPM の denoiser を Tang et al. 2025 の構成に置き換えたバックボーン。

参照元:
    Min Tang, Peng Lu, Qing Feng (2025)
    "Generating Feasible and Diverse Synthetic Populations Using Diffusion Models"
    arXiv:2508.09164 — Fig.3 (ノイズ予測網) / Fig.4 (サブモジュール)
    LLM-wiki: [[papers/tang-2025-diffusion-popsynth]] / [[queries/ddpm-vs-other-diffusion]]

なぜこの論文を踏襲するか:
    - 同じドメイン（合成人口）・同じ離散データを DDPM で扱った先行研究で、
      「baseline はどの構成か」を論文に一行で書ける。
    - Tang 自身が「条件付き生成は未対応 (future work)」と明記しているので、
      我々の CFG + 属性条件付け + 集計教師 (Stage2) がそのまま差分になる。
    - Tang の構成には位置符号化がある。現行 AggDDPM には時刻スロットの位置符号が
      無く (src/eval/clock_diagnostics.py が診断中の問題)、踏襲すると同時に解消する。

Tang の構成 (Fig.3, D=13属性 / K=70語彙):
    one-hot(B,D,K) → FC(K,512) → 位置符号化 → double1dconv(D,64)
    → Down(64,128)→SA → Down(128,256)→SA → Down(256,256)→SA
    → double(256,512)→SA→double(512,512)→SA→double(512,256)→SA
    → Up(512,128)→SA → Up(256,64)→SA → Up(128,64)→SA
    → Conv1d(64,D) → FC(512,K)
    サブモジュール (Fig.4):
        double 1d conv (D,M) = Conv1d(D,M) → LayerNorm → GELU → Conv1d(M,M) → LayerNorm
        Down (D,M)  = MaxPool1d(2) → double(D,D) → double(D,M)  ＋ t 埋め込みを注入
        Up (D,M)    = Upsample(2) → concat(skip) → double(D,D) → double(D,M) ＋ 同上
        Self-Attention = LN → MultiheadAttention → residual → LN → FF(GELU) → residual

Tang からの意図的な差分（★が本モデルの変更点。論文の差分表はこの5点で書ける）:
    ★1 軸の取り方: Tang は conv/attention/プーリングを「埋め込み次元 512」に沿って
        走らせ、属性インデックス D をチャネル軸に置く。属性に順序が無いので妥当だが、
        96 スロットの時系列にそのまま当てると畳み込みが時間的局所性を使わなくなる。
        本実装は **チャネル＝埋め込み次元 / 長さ軸＝96 スロット** に転置する。
        （Tang 自身の論法「Kang の 2D 強制整形 → Tang の 1D 再設計」の延長）
    ★2 条件注入: Tang の「SiLU → FC(512,M) → repeat して加算」を adaLN
        (scale-shift; γ,β による正規化後の変調) に置換。加算バイアスより厳密に強い。
        注入箇所は Tang と同じ Down/Up の6点に限る（bottleneck には入れない）。
        条件は t 埋め込みと同じベクトルに合流させるので、機構は t と完全に共通。
    ★3 条件付け自体: Tang は無条件。性2×年齢7×就業2=28群を属性ごとの Embedding で
        条件付けし、classifier-free guidance を足す。
    ★4 幅: 既定 WIDTH_SCALE=0.5（Tang のチャネル比 64:128:256:256:512 は保つ）。
        Tang は N≈53,506 に対し我々は N=3,736 しかない。詳細は WIDTH_SCALE 参照。
    ★5 正則化: dropout 0.1 / EMA / early stopping を追加（Tang には無い）。
        少データでの過学習対策。学習ハイパラは DDPM_Aggregate と同一に揃える。

DDPM_Aggregate との共有:
    データ整形・拡散過程 (Diffusion)・学習ループ (run_epoch/train)・プール生成
    (group_pool)・サニティ (sanity_check) は DDPM_Aggregate/model.py をそのまま使う。
    2つのバックボーンが同一の分割・同一の手順で学習されるので、差は構造だけに帰着する。

使い方:
    uv run python src/models/DDPM_Aggregate_Tang/model.py
    uv run python src/models/DDPM_Aggregate_Tang/model.py --smoke
    uv run python src/models/DDPM_Aggregate_Tang/model.py --width-scale 1.0   # Tang 忠実

出力:
    outputs/checkpoints/ddpm_tang_w050_pretrain_common12_weekday.pt
    outputs/generated/ddpm_tang_w050_pretrain_samples.csv
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    """sys.modules に一意名で直接載せる（DDPM_Aggregate/model.py と同じ様式）。

    このリポジトリは同名ファイル (model.py) をフラットに持つので、通常の import は
    sys.path の順序次第で静かに別モデルを掴む。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 拡散過程・データ・学習ループの唯一の出所。ここに再実装しない
agg = _load_module("ddpm_agg_model", REPO_ROOT / "src" / "models" / "DDPM_Aggregate" / "model.py")

# ============================================================
# 1. 設定
# ============================================================
NUM_SLOTS = agg.NUM_SLOTS          # 96
NUM_ACT   = agg.NUM_ACT            # 共通12分類
IN_CH     = agg.IN_CH              # 拡散空間のチャネル数 = NUM_ACT
DEVICE    = agg.DEVICE

# Tang: fully connected (K, 512)。512 を 256 に落としても総 params は
# 12.66M → 12.09M でほぼ動かない（bottleneck と attention が支配的）ので Tang 値のまま
EMB_DIM   = 512
# Tang: double(D,64) → Down(64,128) → Down(128,256) → Down(256,256)
CH        = (64, 128, 256, 256)
BOTTLE_CH = 512

# ★調整つまみはこれ1本。Tang のチャネル比は保ったまま幅だけを縮める。
#   実測 params と「値/param」= 3,736個票 × 96スロット = 358,656 ÷ params:
#       width_scale  channels                    params      値/param
#       1.0 (Tang 忠実) (64,128,256,256,512)  13,122,644     0.027
#       0.7            (48, 88,176,176,360)    6,809,988     0.053
#       0.5 (既定)     (32, 64,128,128,256)    3,833,940     0.094
#       0.35           (24, 48, 88, 88,176)    2,168,420     0.165
#       現行 UNet1D                             1,857,940     0.193
#   参考: Tang 自身は 53,506×13属性 = 695k ÷ 12.7M = 0.055。
#   既定 0.5 は Tang より安全側で、実データ水準の断片化を出せている現行 1.86M の約2倍。
#   Chinchilla 則 (20 tokens/param) は「新鮮なデータ潤沢・実質1エポック」の
#   compute-optimal 配分の話で、データ固定・1000エポックの本設定には適用できない。
#   代わりに暗記を直接測る（agg.memorization_report）。
WIDTH_SCALE = 0.5

ATTN_HEADS = 4
DROPOUT    = 0.1                   # ★Tang には無い。少データ (N=3,736) の過学習対策

# 時刻位置符号。★Tang の positional encoding に対応する部分で、現行 AggDDPM に
# 欠けているもの。巡回PE（日跨ぎを環として扱う）は別実験として切り分け、
# ここでは踏襲の一貫性を優先して sinusoidal 絶対PE にする
POS_ENC_MAX_PERIOD = 10000.0


def _tag(width_scale: float) -> str:
    """チェックポイント名に埋める幅の識別子。幅違いが同じファイルを潰さないため。"""
    return f"w{int(round(width_scale * 100)):03d}"


def _paths(width_scale: float) -> tuple[Path, Path]:
    return (
        REPO_ROOT / 'outputs' / 'checkpoints'
        / f'ddpm_tang_{_tag(width_scale)}_pretrain_common12_weekday.pt',
        REPO_ROOT / 'outputs' / 'generated'
        / f'ddpm_tang_{_tag(width_scale)}_pretrain_samples.csv',
    )


MODEL_SAVE_PATH, GEN_SAVE_PATH = _paths(WIDTH_SCALE)


def _round8(v: float) -> int:
    """チャネル数を8の倍数へ。nn.MultiheadAttention が embed_dim % num_heads == 0 を要求する。"""
    return max(8, int(round(v / 8)) * 8)


def channels(width_scale: float) -> tuple[int, int, int, int, int]:
    """(c1, c2, c3, c4, bottleneck)。Tang のチャネル比を保ったまま width_scale 倍する。"""
    c1, c2, c3, c4 = (_round8(c * width_scale) for c in CH)
    return c1, c2, c3, c4, _round8(BOTTLE_CH * width_scale)


# ============================================================
# 2. サブモジュール（Tang Fig.4）
# ============================================================
def slot_positional_encoding(n_slots: int, dim: int) -> torch.Tensor:
    """時刻スロットの sinusoidal 絶対位置符号 (1, n_slots, dim)。

    ★ Tang の positional encoding に対応する。現行 AggDDPM にはこれが無く、
        「いまが朝か夜か」は Conv1d の境界パディングが階層を伝播する副産物としてしか
        入らなかった（src/eval/clock_diagnostics.py の問題設定）。
    """
    pos = torch.arange(n_slots, dtype=torch.float32)[:, None]
    half = dim // 2
    freqs = torch.exp(-math.log(POS_ENC_MAX_PERIOD) * torch.arange(half, dtype=torch.float32) / half)
    args = pos * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)[None, :, :]


class DoubleConv1D(nn.Module):
    """Tang Fig.4 左上: Conv1d → LayerNorm → GELU → Conv1d → LayerNorm。

    GroupNorm(1, C) は 1D における LayerNorm 相当（チャネル×長さ全体で正規化）。

    residual=True のとき GELU(x + f(x))。Tang の Down/Up は1本目を residual、
    2本目でチャネル数を変える構成（Fig.4 の double(D,D) → double(D,M)）。

    ★ emb_dim を与えると最後の正規化出力を adaLN で変調する:
          (γ, β) = Linear(emb_dim, 2*c_out)(SiLU(emb))
          h ← h · (1 + γ) + β
      Tang の「FC(512,M) の出力を加算」を scale-shift に置換したもの。
      Linear は零初期化するので、学習開始時点では恒等（=Tang の加算注入で
      バイアスが0の状態）から始まる。
    """

    def __init__(self, c_in: int, c_out: int, residual: bool = False,
                 emb_dim: int | None = None):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, c_out)
        self.dropout = nn.Dropout(DROPOUT)          # ★Tang には無い
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, c_out)

        self.mod: nn.Linear | None = None
        if emb_dim is not None:
            self.mod = nn.Linear(emb_dim, 2 * c_out)
            nn.init.zeros_(self.mod.weight)
            bias = self.mod.bias
            if bias is not None:
                nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        h = self.conv2(self.dropout(F.gelu(h)))
        h = self.norm2(h)
        if self.mod is not None:
            assert emb is not None, "emb_dim 付きの DoubleConv1D には emb が要る"
            gamma, beta = self.mod(F.silu(emb)).chunk(2, dim=1)
            h = h * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        return F.gelu(x + h) if self.residual else h


class Down1D(nn.Module):
    """Tang Fig.4 右上: MaxPool1d(2) → double(c_in,c_in) → double(c_in,c_out)。

    ★ 長さ軸は時間（96→48→24→12）。Tang は埋め込み次元を縮めていた（★1）。
    ★ t/条件の注入は2本目の double の adaLN で行う（★2）。
    """

    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv_a = DoubleConv1D(c_in, c_in, residual=True)
        self.conv_b = DoubleConv1D(c_in, c_out, emb_dim=emb_dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.conv_b(self.conv_a(self.pool(x)), emb)


class Up1D(nn.Module):
    """Tang Fig.4 下: Upsample(2) → concat(skip, dim=1) → double(c_cat,c_cat) → double(c_cat,c_out)。

    c_cat は連結後のチャネル数（アップサンプル側 + skip 側）。
    """

    def __init__(self, c_cat: int, c_out: int, emb_dim: int):
        super().__init__()
        self.conv_a = DoubleConv1D(c_cat, c_cat, residual=True)
        self.conv_b = DoubleConv1D(c_cat, c_out, emb_dim=emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv_b(self.conv_a(x), emb)


class SelfAttention1D(nn.Module):
    """Tang Fig.3 の Self-Attention。LN → MHA → residual → LN → FF(GELU) → residual。

    ★ 注意を張る軸は時間スロット（96/48/24/12）。入出力は (B, C, L)。
    """

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, ATTN_HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)                       # (B, L, C)
        n = self.norm1(h)
        a, _ = self.attn(n, n, n, need_weights=False)
        h = h + a
        h = h + self.ff(self.norm2(h))
        return h.permute(0, 2, 1)                    # (B, C, L)


# ============================================================
# 3. ノイズ予測網（Tang Fig.3）
# ============================================================
class TangUNet1D(nn.Module):
    """ε 予測ネットワーク: (B,12,96) + t + cond -> (B,12,96)。

    形（WIDTH_SCALE=0.5 の場合の実チャネル数）:
        (B,12,96) --permute--> (B,96,12) --Linear(12,512)--> (B,96,512)
              + slot PE      --permute--> (B,512,96)          ★時間軸に転置
        DoubleConv1D(512,32)                (B,32,96)  = h1   （stem は emb 注入なし）
        Down1D(32,64)   → SA                (B,64,48)  = h2
        Down1D(64,128)  → SA                (B,128,24) = h3
        Down1D(128,128) → SA                (B,128,12) = h4
        bot: DC(128,256)→SA→DC(256,256)→SA→DC(256,128)→SA     (B,128,12)
        Up1D(128+128,64) → SA               (B,64,24)
        Up1D(64+64,32)   → SA               (B,32,48)
        Up1D(32+32,32)   → SA               (B,32,96)
        Conv1d(32,512) --permute--> (B,96,512) --Linear(512,12)--> (B,12,96)
    """

    def __init__(self, width_scale: float | None = None):
        super().__init__()
        ws = WIDTH_SCALE if width_scale is None else width_scale
        self.width_scale = ws
        c1, c2, c3, c4, cb = channels(ws)

        # --- 入力表現（Tang: one-hot → FC(K,512) → positional encoding）---
        self.in_fc = nn.Linear(NUM_ACT, EMB_DIM)
        # 型を明示する。register_buffer だけだと nn.Module.__getattr__ 経由になり
        # 型チェッカが Tensor | Module と見て Tensor の演算を通さない
        self.pos_enc: torch.Tensor
        self.register_buffer("pos_enc", slot_positional_encoding(NUM_SLOTS, EMB_DIM))

        # --- t と条件の埋め込み。両者を1本のベクトルに合流させ、adaLN で注入する ---
        self.time_mlp = nn.Sequential(
            nn.Linear(128, EMB_DIM),
            nn.SiLU(),
            nn.Linear(EMB_DIM, EMB_DIM),
        )
        self.cond_embeds = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(agg.COND_CARD, agg.EMB_DIMS)
        ])
        self.cond_proj = nn.Linear(sum(agg.EMB_DIMS), EMB_DIM)
        self.null_emb = nn.Parameter(torch.zeros(EMB_DIM))

        # --- エンコーダ ---
        self.stem = DoubleConv1D(EMB_DIM, c1)
        self.down1, self.sa1 = Down1D(c1, c2, EMB_DIM), SelfAttention1D(c2)
        self.down2, self.sa2 = Down1D(c2, c3, EMB_DIM), SelfAttention1D(c3)
        self.down3, self.sa3 = Down1D(c3, c4, EMB_DIM), SelfAttention1D(c4)

        # --- bottleneck。★Tang Fig.3 は bottleneck に t を注入していないので合わせる
        #     （条件は down3 の出力 h4 を経由して届く） ---
        self.bot1, self.bsa1 = DoubleConv1D(c4, cb), SelfAttention1D(cb)
        self.bot2, self.bsa2 = DoubleConv1D(cb, cb), SelfAttention1D(cb)
        self.bot3, self.bsa3 = DoubleConv1D(cb, c4), SelfAttention1D(c4)

        # --- デコーダ。連結チャネル数は Tang Fig.3 の Up(512,128)/Up(256,64)/Up(128,64) と一致 ---
        self.up1, self.usa1 = Up1D(c4 + c3, c2, EMB_DIM), SelfAttention1D(c2)
        self.up2, self.usa2 = Up1D(c2 + c2, c1, EMB_DIM), SelfAttention1D(c1)
        self.up3, self.usa3 = Up1D(c1 + c1, c1, EMB_DIM), SelfAttention1D(c1)

        # --- 出力（Tang: Conv1d(64,D) → FC(512,K)）---
        self.out_conv = nn.Conv1d(c1, EMB_DIM, 3, padding=1)
        self.out_fc = nn.Linear(EMB_DIM, NUM_ACT)
        # 出口を零初期化して学習開始時の ε̂ を 0 にする（現行 UNet1D と同じ扱い）
        nn.init.zeros_(self.out_fc.weight)
        out_bias = self.out_fc.bias
        if out_bias is not None:
            nn.init.zeros_(out_bias)

    @property
    def in_channels(self) -> int:
        """拡散空間のチャネル数。clock_diagnostics が純ノイズ x_T を作るのに使う。"""
        return IN_CH

    def embed_cond(self, cond_idx, batch: int, drop_mask=None) -> torch.Tensor:
        """条件埋め込み (B, EMB_DIM)。DDPM_Aggregate.UNet1D.embed_cond と同一ロジック。

        属性ごとに Embedding して連結するので、28通りの one-hot より統計的強度を共有できる。
        cond_idx=None（CFG の無条件経路）と drop_mask=True の行は null_emb になる。
        """
        if cond_idx is None:
            return self.null_emb.expand(batch, -1)
        c = torch.cat([emb(cond_idx[:, i]) for i, emb in enumerate(self.cond_embeds)], dim=1)
        c = self.cond_proj(c)
        if drop_mask is not None:
            c = torch.where(drop_mask[:, None], self.null_emb.expand_as(c), c)
        return c

    def _embed_input(self, x_t: torch.Tensor) -> torch.Tensor:
        """(B,12,96) → (B,EMB_DIM,96)。★ここで時間軸に転置する（Tang からの差分1）。"""
        h = self.in_fc(x_t.permute(0, 2, 1))          # (B,96,EMB)
        h = h + self.pos_enc                          # ★時刻の位置符号
        return h.permute(0, 2, 1)                     # (B,EMB,96)

    def _encode(self, x_t: torch.Tensor, emb: torch.Tensor):
        """下り経路の中間特徴。forward と features の共通部分（分岐させない）。"""
        h1 = self.stem(self._embed_input(x_t))
        h2 = self.sa1(self.down1(h1, emb))
        h3 = self.sa2(self.down2(h2, emb))
        h4 = self.sa3(self.down3(h3, emb))
        return h1, h2, h3, h4

    def _emb(self, t: torch.Tensor, cond_idx, batch: int, drop_mask=None) -> torch.Tensor:
        return self.time_mlp(agg.timestep_embedding(t)) + self.embed_cond(cond_idx, batch, drop_mask)

    def features(self, x_t, t, cond_idx=None) -> dict[str, torch.Tensor]:
        """中間特徴 {h1:(B,c1,96), h2:(B,c2,48), h3:(B,c3,24)}。

        clock_diagnostics の位置プローブ用。キーと解像度は DDPM_Aggregate.UNet1D.features
        と揃えてあるので、同じ診断をそのまま両バックボーンに掛けられる。
        """
        emb = self._emb(t, cond_idx, x_t.size(0))
        h1, h2, h3, _ = self._encode(x_t, emb)
        return {"h1": h1, "h2": h2, "h3": h3}

    def forward(self, x_t, t, cond_idx=None, drop_mask=None) -> torch.Tensor:
        emb = self._emb(t, cond_idx, x_t.size(0), drop_mask)

        h1, h2, h3, h4 = self._encode(x_t, emb)

        b = self.bsa1(self.bot1(h4))
        b = self.bsa2(self.bot2(b))
        b = self.bsa3(self.bot3(b))

        u = self.usa1(self.up1(b, h3, emb))
        u = self.usa2(self.up2(u, h2, emb))
        u = self.usa3(self.up3(u, h1, emb))

        u = self.out_conv(u).permute(0, 2, 1)         # (B,96,EMB)
        return self.out_fc(u).permute(0, 2, 1)        # (B,12,96)


# ============================================================
# 4. 学習・読み込み・サニティ（学習ループ本体は DDPM_Aggregate と共有）
# ============================================================
def _factory():
    return TangUNet1D()


# ★既定パスを表す番兵。`save_path: Path | None = MODEL_SAVE_PATH` と書けないのは、
#   既定引数が def 時に1度だけ評価され、--width-scale による差し替えに追従しないため。
#   かつ None は DDPM_Aggregate と同じく「保存しない」の意味で残す必要がある
#   （--smoke が本番のチェックポイント・生成CSVを潰さないための約束）。
_DEFAULT_PATH = object()


def train(epochs: int = agg.EPOCHS, use_wandb: bool = True, save_path=_DEFAULT_PATH):
    """DDPM_Aggregate と同一の学習手順を TangUNet1D で回す。

    save_path: 省略時は MODEL_SAVE_PATH。None なら保存しない（agg.train と同じ意味）。
    """
    return agg.train(
        epochs=epochs,
        use_wandb=use_wandb,
        save_path=MODEL_SAVE_PATH if save_path is _DEFAULT_PATH else save_path,
        model_factory=_factory,
        backbone="tang",
        extra_config={
            "emb_dim": EMB_DIM, "ch": CH, "bottle_ch": BOTTLE_CH,
            "width_scale": WIDTH_SCALE, "attn_heads": ATTN_HEADS,
            "channels": channels(WIDTH_SCALE),
            "positional_encoding": "sinusoidal_absolute",
            "cond_injection": "adaln",
        },
    )


def load_pretrained(path: Path | None = None, use_ema: bool = True) -> nn.Module:
    """保存済み Stage1 を読み込む（既定は EMA 重み）。

    ★ 既定値を None にして本体で MODEL_SAVE_PATH を引くのは、--width-scale が
        モジュール変数を差し替えるため。既定引数は def 時に1度だけ評価されるので、
        `path: Path = MODEL_SAVE_PATH` と書くと幅を変えても旧パスを読みに行く。
    """
    return agg.load_pretrained(
        path=MODEL_SAVE_PATH if path is None else path,
        use_ema=use_ema, model_factory=_factory,
    )


def sanity_check(model, n_per_group: int = 256, sampler: str = agg.SAMPLER,
                 ddim_steps: int = agg.DDIM_STEPS, save_path=_DEFAULT_PATH):
    """実 ATUS 平日と同一群構成で生成し、断片化・活動シェア・暗記を比較する。

    save_path: 省略時は GEN_SAVE_PATH。None なら生成CSVを書かない。
    """
    return agg.sanity_check(
        model, n_per_group=n_per_group, sampler=sampler, ddim_steps=ddim_steps,
        save_path=GEN_SAVE_PATH if save_path is _DEFAULT_PATH else save_path,
    )


# ============================================================
# 5. スモークテスト
# ============================================================
def smoke_test():
    """学習前に必ず通す形状・整合チェック。"""
    m = TangUNet1D().to(DEVICE)
    d = agg.Diffusion()
    x = torch.randn(4, IN_CH, NUM_SLOTS, device=DEVICE)
    t = torch.randint(0, agg.T_STEPS, (4,), device=DEVICE)
    c = torch.zeros(4, len(agg.COND_SPEC), dtype=torch.long, device=DEVICE)

    assert m(x, t, c).shape == x.shape
    assert m(x, t, None).shape == x.shape                      # 無条件 (CFG) 経路

    # ★中間特徴のキーと解像度が DDPM_Aggregate.UNet1D と一致すること
    #   （clock_diagnostics が両バックボーンで同じ表を出せる条件）
    feats = m.features(x, t, c)
    assert set(feats) == {"h1", "h2", "h3"}
    assert [feats[k].shape[2] for k in ("h1", "h2", "h3")] == [96, 48, 24]

    # ★損失が有限で、逆過程が形を保つこと
    sched = torch.randint(0, NUM_ACT, (4, NUM_SLOTS), device=DEVICE)
    assert torch.isfinite(d.loss(m, sched, c))
    grid = torch.as_tensor(agg.cond_grid()[:4], device=DEVICE)
    s = d.ddim_sample(m, grid, guidance_scale=1.0, steps=5)
    assert s.shape == (4, NUM_SLOTS) and int(s.max()) < NUM_ACT

    c1, c2, c3, c4, cb = channels(WIDTH_SCALE)
    print(f"smoke test: OK  (backbone=tang, width_scale={WIDTH_SCALE}, "
          f"channels={(c1, c2, c3, c4, cb)}, "
          f"params={sum(p.numel() for p in m.parameters()):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AggDDPM (Tang et al. 2025 バックボーン): ATUS平日・共通12分類・28群の pretrain")
    ap.add_argument("--epochs", type=int, default=agg.EPOCHS)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="短時間の動作確認のみ")
    ap.add_argument("--width-scale", type=float, default=WIDTH_SCALE,
                    help="Tang のチャネル比を保ったまま幅を倍する係数 (1.0 = Tang 忠実)")
    args = ap.parse_args()

    # 幅を変えたらチェックポイント名も変える。幅違いが同じファイルを潰さないため
    WIDTH_SCALE = args.width_scale
    MODEL_SAVE_PATH, GEN_SAVE_PATH = _paths(WIDTH_SCALE)

    smoke_test()
    if args.smoke:
        model = agg.train(epochs=5, use_wandb=False, save_path=None,
                          model_factory=_factory, backbone="tang")
        sanity_check(model, n_per_group=16, sampler='ddim', ddim_steps=10, save_path=None)
    else:
        model = train(epochs=args.epochs, use_wandb=not args.no_wandb)
        sanity_check(model)
