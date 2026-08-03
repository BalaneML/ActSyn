"""
model.py (DDPM_Aggregate_DiT)
================
AggDDPM の denoiser を Diffusion Transformer (DiT) に置き換えたバックボーン。

参照元:
    William Peebles, Saining Xie (2023)
    "Scalable Diffusion Models with Transformers" (DiT)
    ICCV 2023, arXiv:2212.09748 — Fig.3 (ノイズ予測網と adaLN-Zero ブロック)
    LLM-wiki: [[2026-08-02]]（条件注入の整理）/ [[papers/li-2026-atlas]]
              / [[queries/ddpm-vs-other-diffusion]]

なぜ DiT を試すか:
    - 条件の型が DiT の想定と一致する。wiki の条件注入まとめ ([[2026-08-02]] 5節) は
        「クラスラベル・スカラー → adaLN-Zero (DiT) または AdaGN (U-Net) + CFG」と切り分ける。
        本タスクの条件は 性2×年齢7×就業2=28群 の離散ラベルと時刻 t だけで、
        空間整列を持たない。cross-attention や ControlNet の出番ではない。
    - 同ドメインの先行がある。ATLAS ([[papers/li-2026-atlas]]) は活動軌跡生成を
        BART autoencoder + 潜在 DiT・adaLN 条件注入で構成しており、
        「活動系列 × DiT × adaLN」は効くことが示されている。
    - U-Net 系2本 (現行 UNet1D / Tang) と帰納バイアスが直交する。畳み込みの局所性を
        一切使わず、96 スロット全体を最初の層から self-attention で見る。
        断片化（実 12.64 に対する生成の平均切替回数）が畳み込みの局所性に
        依存していたのかを切り分けられる。
    - DiT は位置符号を持つ。現行 AggDDPM に欠けている「時計」
        (src/eval/clock_diagnostics.py が診断中の問題) が Tang と同様に解消される。

DiT の構成 (Peebles & Xie Fig.3):
    latent → patchify → +位置符号 → DiT block × L → final layer → unpatchify
    DiT block (adaLN-Zero):
        c = t埋め込み + クラス埋め込み
        (shift1, scale1, gate1, shift2, scale2, gate2) = Linear(d, 6d)(SiLU(c))  ← 零初期化
        x ← x + gate1 · Attn( modulate(LN(x), shift1, scale1) )
        x ← x + gate2 · MLP ( modulate(LN(x), shift2, scale2) )
    final layer も同じ変調 (shift, scale) を持ち、出力 Linear は零初期化。

DiT からの意図的な差分（★が本モデルの変更点。論文の差分表はこの6点で書ける）:
    ★1 データ空間: DiT は VAE 潜在上の 2D パッチに拡散をかける (latent diffusion)。
        本実装は VAE を挟まず、活動 one-hot を ±1 に写した (12, 96) に直接拡散する。
        96スロット×12分類に潜在圧縮の利得は無い
        （[[queries/ddpm-vs-other-diffusion]]「Latent Diffusion: 属性13次元では不要」と同じ論法）。
    ★2 トークン化: 2D patchify → 時間軸 1D patchify。既定 PATCH=1（1トークン=15分1スロット）。
        Peebles は patch を小さくするほど良い (P=2 < 4 < 8) と報告しており、
        系列長 96 では計算制約が無いので下限の P=1 を既定にする。U-Net 2本の最高解像度
        96 と揃うので、比較のときに時間解像度の差が交絡しない。
    ★3 条件付け: DiT の単一クラス埋め込みを属性ごとの Embedding 3本
        (性/年齢/就業) に置換し、連結→射影して t 埋め込みに合流させる。
        28通りの one-hot より統計的強度を共有できる（現行 UNet1D / Tang と同一ロジック）。
        CFG のラベル dropout (P_UNCOND=0.1) は DiT と同じ。
    ★4 位置符号: DiT の frozen 2D sin-cos → 時刻スロットの 1D sin-cos。
        巡回PE（日跨ぎを環として扱う）は別実験として切り分け、ここでは
        DiT 踏襲の一貫性を優先して絶対PEにする（Tang バックボーンと同じ判断・同じ式）。
    ★5 正則化: dropout 0.1 と early stopping を追加（DiT には無い。EMA は DiT にもある）。
        DiT は ImageNet N=1.28M、我々は平日 N=3,736。
        学習ハイパラは DDPM_Aggregate と同一に揃える。
    ★6 規模: 既定は SIZE='xs'（深さ6・幅128）。DiT-S 忠実は --size s。詳細は SIZES の表。

DDPM_Aggregate との共有:
    データ整形・拡散過程 (Diffusion)・学習ループ (run_epoch/train)・プール生成
    (group_pool)・サニティ (sanity_check) は DDPM_Aggregate/model.py をそのまま使う。
    3つのバックボーンが同一の分割・同一の手順で学習されるので、差は構造だけに帰着する。

使い方:
    uv run python src/models/DDPM_Aggregate_DiT/model.py
    uv run python src/models/DDPM_Aggregate_DiT/model.py --smoke
    uv run python src/models/DDPM_Aggregate_DiT/model.py --size s --patch 2   # DiT-S/2 忠実

出力:
    outputs/checkpoints/ddpm_dit_xs_p1_pretrain_common12_weekday.pt
    outputs/generated/ddpm_dit_xs_p1_pretrain_samples.csv
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

# ★調整つまみは (SIZE, PATCH) の2本。
#   実測 params と「値/param」= 3,736個票 × 96スロット = 358,656 ÷ params (PATCH=1):
#       size  depth  hidden  heads      params    値/param   同容量の既存バックボーン
#       xs (既定) 6    128      4     1,852,628    0.194     現行 UNet1D (1,857,940) と差 0.3%
#       t         6    192      6     4,142,868    0.087     Tang w0.5 (3,833,940) と差 8%
#       s        12    384      6    32,429,268    0.011     DiT-S 忠実 (Peebles の最小構成)
#   （内訳: ブロックあたり 18d²+15d、周辺 3d²+175d+84）
#
#   既定を xs にする理由: 現行 UNet1D と params がほぼ一致するので、両者の差を
#   「容量の差」でなく「構造の差」に帰着できる。Transformer は畳み込みのような
#   局所性の帰納バイアスを持たない分データを食うが、N=3,736 では容量を増やす前に
#   暗記を直接測る（agg.memorization_report）のが先。
#   DiT-S 忠実 (32.4M) は 値/param=0.011 で暗記がほぼ確実なので、置けるようにはするが
#   比較の主役にはしない。
SIZES: dict[str, tuple[int, int, int]] = {
    # size: (depth, hidden, heads)
    "xs": (6, 128, 4),
    "t":  (6, 192, 6),
    "s":  (12, 384, 6),            # Peebles & Xie の DiT-S に一致
}
SIZE = "xs"

# ★1トークンがカバーする時間スロット数。既定 1 = 15分。
#   Peebles は P∈{2,4,8} で「小さいほど良い」を報告 (Fig.6)。系列長 96 では
#   attention の計算量が問題にならないので下限の 1 を既定にする。
#   P>1 は「1トークンが連続Pスロットを同時に復号する」ことになり、隣接スロットの
#   一貫性に帰納バイアスが入る。断片化（切替回数）への効果を見る実験軸として残す。
PATCH = 1

MLP_RATIO = 4.0                    # DiT 既定
DROPOUT   = 0.1                    # ★DiT には無い。少データ (N=3,736) の過学習対策

# 時刻位置符号。DiT の frozen sin-cos に対応する（★4）
POS_ENC_MAX_PERIOD = 10000.0


def _tag(size: str, patch: int) -> str:
    """チェックポイント名に埋める構成の識別子。構成違いが同じファイルを潰さないため。"""
    return f"{size}_p{patch}"


def _paths(size: str, patch: int) -> tuple[Path, Path]:
    return (
        REPO_ROOT / 'outputs' / 'checkpoints'
        / f'ddpm_dit_{_tag(size, patch)}_pretrain_common12_weekday.pt',
        REPO_ROOT / 'outputs' / 'generated'
        / f'ddpm_dit_{_tag(size, patch)}_pretrain_samples.csv',
    )


MODEL_SAVE_PATH, GEN_SAVE_PATH = _paths(SIZE, PATCH)


# ============================================================
# 2. サブモジュール（DiT Fig.3）
# ============================================================
def slot_positional_encoding(n_tokens: int, dim: int) -> torch.Tensor:
    """トークン位置の sinusoidal 絶対位置符号 (1, n_tokens, dim)。

    ★ DiT の frozen 2D sin-cos 位置符号に対応する 1D 版。式は
        DDPM_Aggregate_Tang.slot_positional_encoding と同一だが、バックボーン間に
        依存を作らないため各モジュールで閉じて持つ（3本の比較実験の独立性を優先。
        片方の PE を実験で触ったときにもう片方が黙って変わるのを避ける）。
    """
    pos = torch.arange(n_tokens, dtype=torch.float32)[:, None]
    half = dim // 2
    freqs = torch.exp(-math.log(POS_ENC_MAX_PERIOD) * torch.arange(half, dtype=torch.float32) / half)
    args = pos * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)[None, :, :]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN の変調: x·(1+scale) + shift。x は (B, L, C)、shift/scale は (B, C)。"""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    """DiT Fig.3 右: adaLN-Zero ブロック。

        (shift1, scale1, gate1, shift2, scale2, gate2) = Linear(d, 6d)(SiLU(c))
        x ← x + gate1 · Attn( modulate(LN(x), shift1, scale1) )
        x ← x + gate2 · MLP ( modulate(LN(x), shift2, scale2) )

    LayerNorm は elementwise_affine=False。γ,β は条件 c から作るので、層が固有の
    アフィン変換を別に持つと変調と役割が二重になる（DiT 実装と同じ扱い）。

    ★ Linear(d, 6d) を零初期化するので、学習開始時は gate=0、すなわち
      ブロック全体が恒等写像になる。残差の枝が閉じた状態から始まるので深くしても
      学習が壊れない（[[2026-08-02]] の「ControlNet の zero-initialization …
      DiT系にも zero-linear で同じ発想が移植されている」に対応する部分）。
      Tang の adaLN が「変調だけ恒等」なのに対し、こちらは「ブロックごと恒等」で強い。
    """

    def __init__(self, hidden: int, heads: int, mlp_ratio: float = MLP_RATIO):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=DROPOUT, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(approximate="tanh"),        # DiT 実装と同じ近似 GELU
            nn.Dropout(DROPOUT),                # ★DiT には無い
            nn.Linear(mlp_hidden, hidden),
        )
        self.ada_ln = nn.Linear(hidden, 6 * hidden)
        nn.init.zeros_(self.ada_ln.weight)
        ada_bias = self.ada_ln.bias
        if ada_bias is not None:
            nn.init.zeros_(ada_bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(F.silu(c)).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift1, scale1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1[:, None, :] * a
        h = modulate(self.norm2(x), shift2, scale2)
        return x + gate2[:, None, :] * self.mlp(h)


class FinalLayer(nn.Module):
    """DiT Fig.3 の final layer: adaLN(shift, scale) → LN → Linear(d, patch*C)。

    出力 Linear も零初期化するので、学習開始時の ε̂ は恒等的に 0
    （現行 UNet1D / Tang の出口零初期化と同じ扱い）。
    """

    def __init__(self, hidden: int, patch: int, out_ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ada_ln = nn.Linear(hidden, 2 * hidden)
        self.linear = nn.Linear(hidden, patch * out_ch)
        for layer in (self.ada_ln, self.linear):
            nn.init.zeros_(layer.weight)
            bias = layer.bias
            if bias is not None:
                nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(F.silu(c)).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ============================================================
# 3. ノイズ予測網（DiT Fig.3）
# ============================================================
class DiT1D(nn.Module):
    """ε 予測ネットワーク: (B,12,96) + t + cond -> (B,12,96)。

    形（SIZE='xs', PATCH=1 の場合）:
        (B,12,96) --permute--> (B,96,12) --Linear(12·P,128)--> (B,96,128)   patchify
              + 位置符号                                        (B,96,128)
        DiTBlock(128, 4 heads) × 6                              (B,96,128)
        FinalLayer --Linear(128,12·P)--> (B,96,12) --reshape/permute--> (B,12,96)

    U-Net 2本と違い解像度は最初から最後まで 96/PATCH で一定（等方的）。
    ダウンサンプルが無いので、細かい時間構造が段階的に潰れることが無い。
    """

    def __init__(self, size: str | None = None, patch: int | None = None):
        super().__init__()
        self.size = SIZE if size is None else size
        self.patch = PATCH if patch is None else patch
        if self.size not in SIZES:
            raise ValueError(f"未知の size: {self.size}（選べるのは {sorted(SIZES)}）")
        if self.patch < 1 or NUM_SLOTS % self.patch != 0:
            raise ValueError(f"patch は {NUM_SLOTS} の正の約数であること（指定 {self.patch}）")
        depth, hidden, heads = SIZES[self.size]
        self.depth, self.hidden, self.heads = depth, hidden, heads
        self.n_tokens = NUM_SLOTS // self.patch

        # --- 入力表現（DiT: patchify → +位置符号）---
        self.x_embedder = nn.Linear(self.patch * IN_CH, hidden)
        # 型を明示する。register_buffer だけだと nn.Module.__getattr__ 経由になり
        # 型チェッカが Tensor | Module と見て Tensor の演算を通さない
        self.pos_enc: torch.Tensor
        self.register_buffer("pos_enc", slot_positional_encoding(self.n_tokens, hidden))

        # --- t と条件の埋め込み。DiT と同じく1本のベクトル c に合流させる ---
        self.time_mlp = nn.Sequential(
            nn.Linear(128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.cond_embeds = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(agg.COND_CARD, agg.EMB_DIMS)
        ])
        self.cond_proj = nn.Linear(sum(agg.EMB_DIMS), hidden)
        self.null_emb = nn.Parameter(torch.zeros(hidden))

        # --- 本体 ---
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(depth)])
        self.final = FinalLayer(hidden, self.patch, IN_CH)

        # features() が返す3点のタップ位置（浅・中・深）。深さが変わっても等間隔に散る
        self.tap_idx = [max(0, round(depth * f) - 1) for f in (1 / 3, 2 / 3, 1.0)]

    @property
    def in_channels(self) -> int:
        """拡散空間のチャネル数。clock_diagnostics が純ノイズ x_T を作るのに使う。"""
        return IN_CH

    def embed_cond(self, cond_idx, batch: int, drop_mask=None) -> torch.Tensor:
        """条件埋め込み (B, hidden)。DDPM_Aggregate.UNet1D.embed_cond と同一ロジック。

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

    def _cond_vec(self, t: torch.Tensor, cond_idx, batch: int, drop_mask=None) -> torch.Tensor:
        """DiT の c = t埋め込み + ラベル埋め込み。adaLN-Zero へ入る唯一のベクトル。"""
        return self.time_mlp(agg.timestep_embedding(t)) + self.embed_cond(cond_idx, batch, drop_mask)

    def patchify(self, x_t: torch.Tensor) -> torch.Tensor:
        """(B, 12, 96) -> (B, n_tokens, patch*12)。連続する patch スロットを1トークンに畳む。"""
        b = x_t.size(0)
        return x_t.permute(0, 2, 1).reshape(b, self.n_tokens, self.patch * IN_CH)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, patch*12) -> (B, 12, 96)。patchify の逆。"""
        b = tokens.size(0)
        return tokens.reshape(b, NUM_SLOTS, IN_CH).permute(0, 2, 1)

    def _run_blocks(self, x_t: torch.Tensor, c: torch.Tensor,
                    taps: list[int] | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """トークン列を全ブロックに通す。forward と features の共通部分（分岐させない）。

        taps に指定したブロック番号の出力を (B, hidden, n_tokens) に転置して併せて返す。
        """
        h = self.x_embedder(self.patchify(x_t)) + self.pos_enc
        collected: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            h = blk(h, c)
            if taps is not None and i in taps:
                collected.append(h.permute(0, 2, 1))     # (B, hidden, n_tokens)
        return h, collected

    def features(self, x_t, t, cond_idx=None) -> dict[str, torch.Tensor]:
        """中間特徴 {h1, h2, h3} = 浅・中・深のブロック出力 (B, hidden, n_tokens)。

        clock_diagnostics の位置プローブ用。キーは現行 UNet1D / Tang と揃えてあるので
        同じ診断をそのまま掛けられるが、★解像度の意味が違う: DiT はダウンサンプルを
        持たない等方的な構造なので、3点とも同じ解像度 (96/PATCH) で深さだけが違う。
        U-Net 2本の h1/h2/h3 (96/48/24) は解像度も深さも違う。表を並べるときは
        position_probe が出す resolution 列で区別すること。
        """
        c = self._cond_vec(t, cond_idx, x_t.size(0))
        _, taps = self._run_blocks(x_t, c, taps=self.tap_idx)
        return {f"h{i + 1}": h for i, h in enumerate(taps)}

    def forward(self, x_t, t, cond_idx=None, drop_mask=None) -> torch.Tensor:
        c = self._cond_vec(t, cond_idx, x_t.size(0), drop_mask)
        h, _ = self._run_blocks(x_t, c)
        return self.unpatchify(self.final(h, c))


# ============================================================
# 4. 学習・読み込み・サニティ（学習ループ本体は DDPM_Aggregate と共有）
# ============================================================
def _factory():
    return DiT1D()


# ★既定パスを表す番兵。`save_path: Path | None = MODEL_SAVE_PATH` と書けないのは、
#   既定引数が def 時に1度だけ評価され、--size / --patch による差し替えに追従しないため。
#   かつ None は DDPM_Aggregate と同じく「保存しない」の意味で残す必要がある
#   （--smoke が本番のチェックポイント・生成CSVを潰さないための約束）。
_DEFAULT_PATH = object()


def train(epochs: int = agg.EPOCHS, use_wandb: bool = True, save_path=_DEFAULT_PATH):
    """DDPM_Aggregate と同一の学習手順を DiT1D で回す。

    save_path: 省略時は MODEL_SAVE_PATH。None なら保存しない（agg.train と同じ意味）。
    """
    depth, hidden, heads = SIZES[SIZE]
    return agg.train(
        epochs=epochs,
        use_wandb=use_wandb,
        save_path=MODEL_SAVE_PATH if save_path is _DEFAULT_PATH else save_path,
        model_factory=_factory,
        backbone="dit",
        extra_config={
            "size": SIZE, "patch": PATCH, "depth": depth, "hidden": hidden,
            "heads": heads, "mlp_ratio": MLP_RATIO, "n_tokens": NUM_SLOTS // PATCH,
            "positional_encoding": "sinusoidal_absolute",
            "cond_injection": "adaln_zero",
        },
    )


def load_pretrained(path: Path | None = None, use_ema: bool = True) -> nn.Module:
    """保存済み Stage1 を読み込む（既定は EMA 重み）。

    ★ 既定値を None にして本体で MODEL_SAVE_PATH を引くのは、--size / --patch が
        モジュール変数を差し替えるため。既定引数は def 時に1度だけ評価されるので、
        `path: Path = MODEL_SAVE_PATH` と書くと構成を変えても旧パスを読みに行く。
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
    m = DiT1D().to(DEVICE)
    d = agg.Diffusion()
    x = torch.randn(4, IN_CH, NUM_SLOTS, device=DEVICE)
    t = torch.randint(0, agg.T_STEPS, (4,), device=DEVICE)
    c = torch.zeros(4, len(agg.COND_SPEC), dtype=torch.long, device=DEVICE)

    assert m(x, t, c).shape == x.shape
    assert m(x, t, None).shape == x.shape                      # 無条件 (CFG) 経路

    # ★patchify/unpatchify が往復すること。時間順が入れ替わっても形は合うので、
    #   ここで押さえないと「時刻がずれた学習」が静かに走る
    assert torch.allclose(m.unpatchify(m.patchify(x)), x)

    # ★中間特徴のキーが DDPM_Aggregate.UNet1D と一致すること
    #   （clock_diagnostics が3バックボーンで同じ表を出せる条件）。
    #   解像度は DiT だけ等方的なので一致しない — features のドキュメント参照
    feats = m.features(x, t, c)
    assert set(feats) == {"h1", "h2", "h3"}
    assert all(feats[k].shape[2] == m.n_tokens for k in feats)

    # ★損失が有限で、逆過程が形を保つこと
    sched = torch.randint(0, NUM_ACT, (4, NUM_SLOTS), device=DEVICE)
    assert torch.isfinite(d.loss(m, sched, c))
    grid = torch.as_tensor(agg.cond_grid()[:4], device=DEVICE)
    s = d.ddim_sample(m, grid, guidance_scale=1.0, steps=5)
    assert s.shape == (4, NUM_SLOTS) and int(s.max()) < NUM_ACT

    print(f"smoke test: OK  (backbone=dit, size={m.size}, patch={m.patch}, "
          f"depth={m.depth}, hidden={m.hidden}, heads={m.heads}, tokens={m.n_tokens}, "
          f"params={sum(p.numel() for p in m.parameters()):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AggDDPM (DiT バックボーン): ATUS平日・共通12分類・28群の pretrain")
    ap.add_argument("--epochs", type=int, default=agg.EPOCHS)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="短時間の動作確認のみ")
    ap.add_argument("--size", type=str, default=SIZE, choices=sorted(SIZES),
                    help="モデル規模 (s = Peebles の DiT-S 忠実)")
    ap.add_argument("--patch", type=int, default=PATCH,
                    help="1トークンがカバーする時間スロット数 (96 の約数)")
    args = ap.parse_args()

    # 構成を変えたらチェックポイント名も変える。構成違いが同じファイルを潰さないため
    SIZE, PATCH = args.size, args.patch
    MODEL_SAVE_PATH, GEN_SAVE_PATH = _paths(SIZE, PATCH)

    smoke_test()
    if args.smoke:
        model = agg.train(epochs=5, use_wandb=False, save_path=None,
                          model_factory=_factory, backbone="dit")
        sanity_check(model, n_per_group=16, sampler='ddim', ddim_steps=10, save_path=None)
    else:
        model = train(epochs=args.epochs, use_wandb=not args.no_wandb)
        sanity_check(model)
