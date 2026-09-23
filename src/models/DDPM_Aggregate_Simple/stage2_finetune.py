"""
stage2_finetune.py
==================
Stage 2 の学習ループ

日本の集計表だけを教師にして拡散モデルのパラメータを更新する

1更新でやること:
    1. 教師群から |D_sub| 群を一様抽出し, 群あたり n 本を打ち切り逆伝播つきで生成
    2. straight-through で one-hot 化し, 群ごとに A/B へ二分して集計
    3. L_agg = split-batch 不偏推定の重み付き二乗誤差
    4. L_atus = ATUS 実個票のリハーサル項（Stage 1 と同じ ε-MSE）
    5. L_agg.backward() と (λ·L_atus).backward() を別々に呼んで AdamW を1歩

早期終了は使わない
固定ステップ予算で回し切り、M_save ごとに保存した
チェックポイントを学習後に (教師適合, ガードレール) の2軸で選ぶ

リハーサル項は「学習中のモデル」で計算する

使い方:
    # 動作確認（生成を短くして数更新だけ回す）
    .venv/bin/python3 src/models/DDPM_Aggregate_Simple/stage2_finetune.py --smoke

    # 本番（SQUID）
    .venv/bin/python3 src/models/DDPM_Aggregate_Simple/stage2_finetune.py \\
        --steps 300 --d-sub 7 --n 256 --K 1 --eps inf --lam auto

    主なフラグ:
        --steps N        固定ステップ予算（早期終了なし）
        --d-sub M        1更新で使う群数。メモリ制約は K×D_sub×n <= 6600（§4.5）
        --n N            群あたり生成本数。n=256 で教師セルの45.6%が識別可能（§4.6）
        --K N            勾配を保持する末尾ステップ数。DRaFT は K=1 でも機能すると報告
        --chunk N        1度に勾配を保持する個票数。0 で予算から自動。B 未満なら2パス蓄積
        --eps V          損失の重み床。inf=素のMSE（主A）、0.01=χ²（主B）
        --loss {sq,jsd}  jsd はアブレーション1点のみ。chunk を使えないので
                         メモリ判定が K×B になる（1パス固定）
        --lam V          リハーサル重み。auto なら最初の LAM_WARMUP_STEPS 更新の
                         |L_agg| / L_atus の中央値で決めて以降固定する
        --holdout-groups d1,d2,...  LGO。損失から外す群（生成と評価は常に全28群）
        --resume         ckpt-dir の最新チェックポイントから再開。
                         torch / numpy 両方の RNG と λ を引き継ぐ
        --lr-cond / --lr-emb / --lr-conv / --lr-clock  層別学習率（既定は LR_* 定数）。
                         --lr-clock は時計つきの Stage 1（model.py --clock）のときだけ効く

学習ログで最初に見る量:
    agg_gnorm_cond / _emb / _conv  集計側だけで θ に載った勾配の L2 ノルム（層別 LR の3群）
    agg_gnorm_clock                時計つきの Stage 1 のときだけ。0 に張り付くなら
                                   集計勾配は時計（スロットごとに違う値の経路）を使っていない
        ★L_agg・rate_mae・g_* は straight-through と clamp より上流の量なので、
          代理勾配が潰れて θ が全く動いていなくても正常値を出す
    total_gnorm_*                  リハーサル項を足した後のノルム。
        ★agg との比が「集計側が更新方向にどれだけ効いているか」そのもの。
          実測では λ=auto のとき λ‖g_atus‖/‖g_agg‖ = 22〜51 倍、さらに
          cos(g_agg, g_atus) = −0.27（逆向き）。λ を下げないと集計は通らない
    L_atus_val                     ATUS val 分割の ε-MSE（--val-every ごと）
        300 更新は学習分割の約23エポック相当。上がり始めたらリハーサルの過学習
    x0_floor_frac                  x_0 が下側 clamp に張り付いた要素の割合
        zero-shot の実測は 0.338。上がり続けるなら代理勾配が痩せている
"""
import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    """sys.modules に一意名で載せる。既に同じファイルが同じ名前で入っていれば使い回す。

    ★使い回しが要点。同名で読み直すと sys.modules のエントリは置き換わるが、
      先に読んだ側が掴んでいるモジュールオブジェクトは別のまま残る。すると
      「model.T_STEPS を差し替えたのに、こちらから呼ぶ生成は 1000 ステップのまま」
      のような、例外を出さずに黙って重くなる／数値が変わる食い違いが起きる。
    """
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sm: Any = _load("simple_model", HERE / "model.py")
ck: Any = _load("simple_stage2_checkpoint", HERE / "stage2_checkpoint.py")
st: Any = _load("simple_stage2_targets", HERE / "stage2_targets.py")
sl: Any = _load("simple_stage2_loss", HERE / "stage2_loss.py")

# 既定値。根拠は Stage2_design.md の対応する節
DEFAULT_STEPS = 300          # §4.7 の見積り（D_sub=7 なら実測3.2時間）
DEFAULT_D_SUB = 7            # §4.5 の逃げ道の第一手。K<=3 まで1パスで収まる
DEFAULT_N = 256              # §4.6 の分解能。教師セルの45.6%が識別可能になる
DEFAULT_K = 1                # §4.9 の通り DRaFT は K=1 でも機能すると報告している
DEFAULT_SAVE_EVERY = 25
# val の ε-MSE を測る間隔。リハーサル項が ATUS 側へ過学習していないかの監視。
# ★val 分割は 373 本（2 バッチ）なので、1回 0.1 秒程度で 1 更新 39 秒に対して無視できる。
#   固定 seed で毎回同じ (t, ε) を使うため、ステップ間の差はモデルの変化だけを反映する
DEFAULT_VAL_EVERY = 10
# 層別学習率（§8.3）。★2群ではなく3群に分ける。
# 旧実装は cond_embeds/cond_proj/null_emb/emb_proj を1つの「条件経路」群にしていたが、
# その 98.5%（312,512/317,192）は emb_proj である。emb_proj が受け取るのは
#     emb = timestep_embedding(t) + embed_cond(cond_idx)      （model.py:505）
# という **和** なので、emb_proj は条件専用ではなく拡散ステップの注入路でもある。
# K=1 では集計勾配が t=0 にしか届かないため、ここを conv の10倍で動かすと
# 「時刻応答を t=0 に特化させる」方向に働く。中間の学習率を与えて分ける。
LR_COND = 1e-4               # 群だけに効く純粋な条件パラメータ 4,680 (0.27%)
LR_EMB  = 2e-5               # emb_proj 312,512 (17.8%)。時刻と条件の共有注入路
LR_CONV = 1e-5               # conv/attention/GroupNorm 1,441,932 (82.0%)
# 時計つきの Stage 1（model.py の --clock）のときだけ作る群。clock_proj 10,944 (0.62%)。
# ★cond と同じ 1e-4 にする。時計は群によらず全員に共通の「何時に」を表す唯一の経路で、
#   日本の昼食が 12:00 に揃う・7:00 に朝食をとるといった全群共通の時刻構造はここを通る。
#   時計なしの λ=0.003 では周期 8h 以下の残差が 2% 台しか埋まらなかった（stage2_curves の
#   band_closure_table）。cond と同じく小さく低次元の経路なので、速く動かしても
#   拡散ステップ応答（emb）や畳み込み（conv）を壊しにくい
LR_CLOCK = 1e-4
MEMORY_BUDGET = 6600         # §4.5 K×chunk <= 約6,600（A100 40GB）

# λ=auto を決めるのに使う更新数。
# ★1更新だけで決めてはいけない。L_agg は split-batch 推定量なので、期待値が bias² でも
#   実現値は大きく振れ、負にもなりうる（stage2_loss.agg_loss_from_rates の Note）。
#   初回がたまたま 0 近傍に落ちると λ≈0 が全ステップ固定され、リハーサル項が
#   実質無効のまま予算を回し切る。中央値を採るのは平均より外れ値に強いため
LAM_WARMUP_STEPS = 5
STAGE1_CKPT = REPO_ROOT / "outputs" / "checkpoints" / "ddpm_simple_pretrain_common12_weekday_20260819.pt"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "stage2"

# 層別学習率の群分け（§8.3）。上から順に判定し、最初に当たった群へ入れる。
# ★COND_PATH_KEYS は「群ごとに違う値を持つ」パラメータだけ。ここを動かさないと
#   dev_*（人口平均からの群のズレ ＝ 条件付け能力）は改善しない。
#   日米差の 64.2%（二乗和）は dev 成分なので、ここが主役である。
# ★EMB_PATH_KEYS は時刻埋め込みと条件埋め込みの **和** を各 ResBlock へ注入する経路。
#   群による違いも通すが、同時に拡散ステップ応答そのものでもある。
COND_PATH_KEYS = ("cond_embeds", "cond_proj", "null_emb")
EMB_PATH_KEYS = ("emb_proj",)
# ★CLOCK_PATH_KEYS は時計つきモデルにだけある。時刻（1 日のうちの何時か）ごとに違う値を
#   足す経路で、群によらない。無いモデルでは clock 群を作らない（従来の 3 群のまま）
CLOCK_PATH_KEYS = ("clock_proj",)


# ============================================================
# 群のマスクと抽出
# ============================================================
def build_teacher_mask(holdout: list[int]) -> np.ndarray:
    """損失に使う群の bool マスク (28,), default（空）なら28群すべてが教師"""
    mask = np.ones(st.D_GROUPS, dtype=bool)
    for d in holdout:
        if not 0 <= d < st.D_GROUPS:
            raise ValueError(f"群インデックスが範囲外: {d}")
        mask[d] = False
    if not mask.any():
        raise ValueError("教師群が空になっている")
    return mask


def stratified_holdout(pop: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """人口シェアで層化して k 群を選ぶ

    群人口シェアは18.7倍の開きがあるので、無作為に選ぶと大きい群ばかり、あるいは
    小さい群ばかりになりうる。
    シェア順に k 個の帯へ分け、各帯から1つずつ引いて大小を混ぜる。
    """
    order = np.argsort(pop.reshape(st.D_GROUPS))
    rng = np.random.default_rng(seed)
    picked = [int(rng.choice(band)) for band in np.array_split(order, k)]
    return sorted(picked)


# ============================================================
# 学習
# ============================================================
PARAM_GROUP_NAMES = ("cond", "emb", "conv")
CLOCK_GROUP_NAME = "clock"


def split_param_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """パラメータを層別学習率の群へ分ける。時計なしは3群、時計つきは4群

    Note:
        ★名前の断片で判定する。上から cond -> emb -> clock -> conv の順に当てるので、
        COND_PATH_KEYS / EMB_PATH_KEYS / CLOCK_PATH_KEYS が重なっていない必要がある。
        ★戻り値の dict の並びが build_optimizer の param_groups の並びになる。
        時計なしモデルでは cond / emb / conv（PARAM_GROUP_NAMES と同じ）、
        時計つきモデルでは cond / emb / clock / conv。
        ★clock 群だけは空でもよい（時計なしモデル）。空なら dict から除く。

    Args:
        model: UNet1D

    Returns:
        {群名: パラメータの list}

    Raises:
        ValueError: cond / emb / conv のいずれかが空になった場合
    """
    groups: dict[str, list[torch.nn.Parameter]] = {
        "cond": [], "emb": [], CLOCK_GROUP_NAME: [], "conv": []}
    for name, p in model.named_parameters():
        if any(k in name for k in COND_PATH_KEYS):
            groups["cond"].append(p)
        elif any(k in name for k in EMB_PATH_KEYS):
            groups["emb"].append(p)
        elif any(k in name for k in CLOCK_PATH_KEYS):
            groups[CLOCK_GROUP_NAME].append(p)
        else:
            groups["conv"].append(p)
    if not groups[CLOCK_GROUP_NAME]:
        del groups[CLOCK_GROUP_NAME]
    empty = [k for k, v in groups.items() if not v]
    if empty:
        raise ValueError(f"層別 LR の分割に失敗した。空の群: {empty}")
    return groups


def build_optimizer(model: torch.nn.Module, lr_cond: float = LR_COND,
                    lr_emb: float = LR_EMB,
                    lr_conv: float = LR_CONV,
                    lr_clock: float = LR_CLOCK) -> torch.optim.Optimizer:
    """層別学習率つき AdamW, param_groups は split_param_groups の並びで、各群に "name" を持つ

    Note:
        UNet1D 1,759,124 params の内訳:
            cond     4,680 ( 0.27%) = cond_embeds 72 + cond_proj 4,352 + null_emb 256
            emb    312,512 (17.77%) = emb_proj ×11
            conv 1,441,932 (81.97%) = 畳み込み + attention + GroupNorm
        時計つき（--clock）は clock 10,944 (0.62%) = clock_proj ×11 が加わる。

        ★cond だけが「群ごとに違う値」を持つ。日米差の 64.2%（二乗和）は
        dev 成分（群ごとの描き分け）なので、そこを動かせるのはこの 4,680 だけである。
        ★emb は時刻埋め込みと条件埋め込みの **和** を注入する共有経路なので、
        速く動かすと拡散ステップ応答そのものが変わる。cond と conv の中間に置く。
        ★clock だけが「スロットごとに違う値」を持つ。全群共通の時刻構造を動かす経路。
        ★param_groups[i]["name"] に群名を入れる。grad_norms はこの名前で群を読むので、
          並びを組み替えても「conv のノルムを cond として報告する」壊れ方をしない。

    Args:
        model: Stage1 の重みを読んだUNet1D
        lr_cond: 群専用の条件パラメータの学習率, default=LR_COND=1e-4
        lr_emb: emb_proj（時刻・条件の共有注入路）の学習率, default=LR_EMB=2e-5
        lr_conv: conv/attention の学習率, default=LR_CONV=1e-5
        lr_clock: clock_proj（時計）の学習率。時計なしモデルでは使わない, default=LR_CLOCK=1e-4

    Returns:
        AdamW, weight_decay=0.0
    """
    groups = split_param_groups(model)
    lrs = {"cond": lr_cond, "emb": lr_emb, CLOCK_GROUP_NAME: lr_clock, "conv": lr_conv}
    return torch.optim.AdamW(
        [{"params": params, "lr": lrs[name], "name": name} for name, params in groups.items()],
        weight_decay=0.0)


def check_shapes(K: int, d_sub: int, n: int) -> None:
    """K / D_sub / n が学習ループの前提を満たすか, 生成を1回でも回す前にチェックする

    ★生成の後で落とさないことが要点。本番の1更新は A100 で約40秒かかるので、
    「n が奇数なので group_rates_split で ValueError」を生成の後に出すと、
    その40秒とジョブの枠が無駄になる。ここは全て引数だけで判定できる。

    Args:
        K: 勾配を保持する末尾ステップ数
        d_sub: 1更新で使う群数
        n: 群あたりの生成本数

    Raises:
        SystemExit: いずれかが前提を満たさないとき
    """
    if K < 1:
        raise SystemExit(
            f"ERROR: K は1以上でなければならない（K={K}）。\n"
            f"       K=0 は末尾の勾配区間が空なので集計側に勾配が1本も流れず、\n"
            f"       L_agg.backward() が 'does not require grad' で落ちる（§4.3）")
    if d_sub < 1:
        raise SystemExit(f"ERROR: D_sub は1以上でなければならない（d_sub={d_sub}）")
    if n < 2 or n % 2 != 0:
        raise SystemExit(
            f"ERROR: n は2以上の偶数でなければならない（n={n}）。\n"
            f"       split-batch 不偏推定が群内を A/B の半分ずつへ割るため（§7.5）")


def check_memory_budget(K: int, d_sub: int, n: int, chunk: int | None = None,
                        budget: int = MEMORY_BUDGET) -> None:
    """勾配を保持するピーク = K × chunk, 学習前にチェック"""
    if K < 1:
        raise SystemExit(f"ERROR: K は1以上でなければならない（K={K}）")
    load = K * (d_sub * n if chunk is None else chunk)
    if load <= budget:
        return
    what = f"K×D_sub×n = {K}×{d_sub}×{n}" if chunk is None else f"K×chunk = {K}×{chunk}"
    raise SystemExit(
        f"ERROR: {what} = {load:,} が予算 {budget:,} を超えている。\n"
        f"       chunk <= {budget // K} に下げるか、K を下げること（§4.5）")


def resolve_chunk(K: int, d_sub: int, n: int,
                    chunk: int | None = None,
                    budget: int = MEMORY_BUDGET) -> int:
    """1度に勾配を保持する個票数を決める

    ★B（欲しい本数）と chunk（一度に勾配を保持できる本数）は別の量である。
    B は統計的要請（n は §4.6 の分解能）、chunk はメモリ制約（K×chunk <= 約6,600）で決まる。
    chunk=None なら予算に収まる最大値を自動で選ぶ。2パス蓄積は勾配が一括計算と厳密に
    一致する（近似ではない）ので、自動で下げても学習の意味は変わらない。
    """
    if K < 1:
        raise SystemExit(f"ERROR: K は1以上でなければならない（K={K}）")
    total = d_sub * n
    if chunk is not None and chunk > 0:
        return min(chunk, total)
    auto = budget // K
    if auto <= 0:
        raise SystemExit(f"ERROR: K={K} が大きすぎて chunk を1本も取れない（予算 {budget:,}）")
    return min(total, auto)


def aggregate_step(diffusion: Any, 
                    model: torch.nn.Module,
                    cond: torch.Tensor,
                    K: int,
                    n: int,
                    a_star: torch.Tensor,
                    omega: torch.Tensor,
                    chunk: int,
                    loss_kind: str = "sq",
                    tau: float = 1.0) -> tuple[float, torch.Tensor, torch.Tensor, dict[str, float]]:
    """集計側の1更新分の勾配を θ.grad へ蓄積し、(L_agg の値, ā_A, ā_B, 診断) を返す。

    ★返り値の損失はスカラー値であって計算グラフを持たない。backward はこの関数の中で
      済ませてある。呼び出し側は先に optimizer.zero_grad() を済ませておくこと。

    ★4つ目の返り値は straight-through の入口の診断（_x0_diagnostics）。
      L_agg も g_diagnostics も straight-through と clamp より **上流** の量なので、
      代理勾配が潰れていても正常値を出す。潰れたことが見えるのはここと、
      呼び出し側が測る θ の勾配ノルムだけである。

    chunk >= B なら1パス（素直に autograd を通す）。chunk < B なら2パス勾配蓄積
    （gradient caching、§2.9）に切り替える。

    ★loss_kind="jsd" は chunk を無視して常に1パスで回る。JSD は群平均の非線形関数で
      split-batch が使えず、2パスの g（∂L/∂ā の解析形）を持たないためである。
      したがって jsd のピークは K×chunk ではなく K×B で決まる。予算判定は
      呼び出し側（run が check_memory_budget に total を渡す）で別に行うこと。

    2パスが要る理由は「損失がバッチ全体の関数で分解できないのに、バッチがメモリに
    入らない」から。普通の勾配蓄積（部分ごとに損失を計算して足す）は使えない。
    損失が非線形なので部分から全体を復元できず、n=4/chunk=2 の例では正しい損失 0.0025 に
    対して 0.1625 と65倍ずれる。

    2パスの原理は連鎖律を「定数の部分」と「分解できる部分」に割ること:

        ∂L/∂θ = (∂L/∂Ã) · (∂Ã/∂θ),   ∂Ã/∂θ = (1/n) Σ_i ∂y_i/∂θ

    1パス目で ∂L/∂Ã を数値 g に潰してしまえば、残りは y について線形なので
    チャンクの和へ分解できる。近似ではなく厳密に一致する。

    ★2パス目は1パス目と同じ乱数でなければ別のサンプルを見ることになる。末尾 K 区間の
      雑音 zs を1パス目で作って両方に渡す。x_K も1パス目のものを使い回すので、
      前段（T−K ステップ）は1回しか回らない。2回回るのは末尾 K だけである。

    計算コスト:
      追加分は「1パス目の末尾 K（no_grad）」＋「(チャンク数−1) 回ぶんの末尾 K と backward」で、
      前段の T−K ステップには依存しない。K=1・4分割なら数ステップ相当なので、T=1000 の
      1更新（実測155秒、§4.7）に対しては小さいはずである。
      ★ただし対象ハードウェア（A100）での実測はまだ無い。手元の MPS・小さい T では
        計測ノイズが支配的で数値を確定できなかった。学習ログの sec 列で確認すること。
    """
    total = cond.size(0)
    if loss_kind == "jsd" or chunk >= total:
        return _aggregate_step_one_pass(diffusion, model, cond, K, n, a_star, omega, loss_kind, tau)
    return _aggregate_step_two_pass(diffusion, model, cond, K, n, a_star, omega, chunk, tau)


def _draw_tail_noise(total: int, K: int, dev: torch.device | str) -> dict[int, torch.Tensor]:
    """末尾 K 区間で使う雑音 {ti: z}。ti=0 は雑音を使わないのでキーは 1..K-1"""
    return {ti: torch.randn(total, sm.IN_CH, sm.NUM_SLOTS, device=dev) for ti in range(1, K)}


def _x0_diagnostics(x0: torch.Tensor) -> dict[str, float]:
    """straight-through の入口 x_0 の状態を測る, 代理勾配が死んでいないかの唯一の観測点

    Note:
        逆過程の各段は x0_hat を clamp(0,1) してから使う（model._reverse_step）。
        clamp は飽和した要素の勾配を厳密に 0 にするので、飽和が増えるほど
        「その活動を増やせ／減らせ」という梃子が効かなくなる。
        ★実測（Stage1 重み・K=1・CFG=1.25）では clamp の 33.8pt が下側、0.7pt が上側で、
        飽和のほぼ全量が下側だった。argmax で選ばれたチャネルの被 clamp は 8.7% で、
        効いていないのは主に「今は選ばれていない活動を持ち上げる」向きである。
        ★下側だけを数えるのは、そちらが厳密に 0.0 として観測できるため。
        逆過程の最終段は x_0 = post_coef_x0[0] · clamp(x0_hat) で
        post_coef_x0[0] = 0.99983406（float32 の桁落ち）なので、上側の飽和は
        1.0 ではなく 0.99983406 に張り付き、閾値で数えると実装定数に依存してしまう。

    Args:
        x0: 逆過程 t=0 の出力, dtype=float32, (B, IN_CH, NUM_SLOTS) = (B, 12, 96)
            argmax 前の連続値。straight_through へ渡す直前のもの

    Returns:
        統計量の dict[str, float], wandb へ log.update() で流し込める
            x0_floor_frac: x_0 が厳密に 0 の要素の割合 ＝ 下側 clamp で勾配が
                切れている要素の割合。上がり続けるなら代理勾配が痩せている
            x0_max: x_0 の最大値。飽和の上限が 0.99983406 付近にあるかの確認用
            st_p_max_mean: softmax(x_0) の最大値の平均（τ=1 相当）。
                一様は 1/12 = 0.0833、設計書 §2.8 の最悪ケースは 0.198
    """
    with torch.no_grad():
        p = torch.softmax(x0, dim=1)
        return {
            "x0_floor_frac": float((x0 <= 0.0).float().mean()),
            "x0_max": float(x0.max()),
            "st_p_max_mean": float(p.max(dim=1).values.mean()),
        }


def _aggregate_step_one_pass(diffusion: Any,
                            model: torch.nn.Module,
                            cond: torch.Tensor,
                            K: int,
                            n: int,
                            a_star: torch.Tensor,
                            omega: torch.Tensor,
                            loss_kind: str,
                            tau: float) -> tuple[float, torch.Tensor, torch.Tensor, dict[str, float]]:
    """素直に autograd を通す版, chunk >= B のとき, および jsd のときに使う"""
    zs = _draw_tail_noise(cond.size(0), K, cond.device)
    x0 = diffusion.sample_differentiable(model, cond, K, zs=zs)
    diag = _x0_diagnostics(x0)
    y = sm.straight_through(x0, tau)
    if loss_kind == "jsd":
        l_agg = sl.jsd_loss(y, a_star, n)
        a_A, a_B = sl.group_rates_split(y.detach(), n)
    else:
        a_A, a_B = sl.group_rates_split(y, n)
        l_agg = sl.agg_loss_from_rates(a_A, a_B, a_star, omega)
    l_agg.backward()
    return float(l_agg.detach()), a_A.detach(), a_B.detach(), diag


def _aggregate_step_two_pass(diffusion: Any,
                            model: torch.nn.Module,
                            cond: torch.Tensor,
                            K: int,
                            n: int,
                            a_star: torch.Tensor,
                            omega: torch.Tensor,
                            chunk: int,
                            tau: float) -> tuple[float, torch.Tensor, torch.Tensor, dict[str, float]]:
    """2パス勾配蓄積"""
    total = cond.size(0)
    zs = _draw_tail_noise(total, K, cond.device)

    # ---- 1パス目: x_K まで進めて Ã と g を得る。グラフは作らない ----
    x_K = diffusion._sample_head(model, cond, K)
    with torch.no_grad():
        x0 = diffusion._sample_tail(model, x_K, K, cond, zs=zs)
        diag = _x0_diagnostics(x0)
        y = sm.straight_through(x0, tau)
        a_A, a_B = sl.group_rates_split(y, n)
        l_agg = float(sl.agg_loss_from_rates(a_A, a_B, a_star, omega))
        g_A, g_B = sl.loss_grad(a_A, a_B, a_star, omega)
        g_per = sl.per_sample_grad(g_A, g_B, n)          # (B,12,96)

    # ---- 2パス目: チャンクごとに末尾 K だけ再計算し、g を上流勾配として注入 ----
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        zs_c = {ti: z[start:end] for ti, z in zs.items()}
        x0_c = diffusion._sample_tail(model, x_K[start:end], K, cond[start:end], zs=zs_c)
        y_c = sm.straight_through(x0_c, tau)
        # ★backward(gradient=...) は「この値を上流から来た勾配とみなせ」という指示。
        #   底にある概念は VJP（ベクトル・ヤコビアン積）で、計算しているのは vᵀJ
        y_c.backward(gradient=g_per[start:end])
    return l_agg, a_A, a_B, diag


def val_epsilon_mse(diffusion: Any, model: torch.nn.Module, val_loader: Any,
                    dev: torch.device | str, seed: int = 0) -> float:
    """ATUS の val 分割で ε-MSE を測る, リハーサル項が過学習していないかの唯一の監視点

    Note:
        ★学習の乱数を汚さない。Diffusion.loss は t と ε を大域 RNG から引くので、
        素朴に呼ぶと学習側の乱数列がずれて --resume の再現性が壊れる。
        呼び出し前の RNG 状態を退避し、固定 seed で評価してから元へ戻す。
        ★毎回同じ (t, ε) で測る。t~U{0,999} を測るたびに引き直すと、
        ステップ間の差がモデルの変化か t の引きの差か分からなくなる。
        ★model のモードも呼び出し前へ戻す。Stage 2 は train と eval を
        1更新の中で行き来するので、ここで書き換えたままにすると集計側に dropout が乗る。

    Args:
        diffusion: Diffusion
        model: 評価するUNet1D
        val_loader: (cond_idx, sched) を yield する val 側 DataLoader
        dev: モデルのデバイス
        seed: t と ε を引く固定 seed, default=0

    Returns:
        val 分割のサンプル加重平均 ε-MSE, float
    """
    rng_state = torch.get_rng_state()
    was_training = model.training
    model.eval()
    try:
        torch.manual_seed(seed)
        total, n_samples = 0.0, 0
        with torch.no_grad():
            for cond_idx, sched in val_loader:
                sched = sched.to(dev)
                loss = diffusion.loss(model, sched, cond_idx.to(dev))
                total += float(loss) * sched.size(0)
                n_samples += sched.size(0)
    finally:
        model.train(was_training)
        torch.set_rng_state(rng_state)
    return total / max(n_samples, 1)


def grad_norms(optimizer: torch.optim.Optimizer, prefix: str) -> dict[str, float]:
    """いま θ.grad に載っている勾配の L2 ノルムを層別 LR の群ごとに測る

    Note:
        ★これが「Stage 2 が実際に θ を動かしているか」を見る唯一の量である。
        L_agg も rate_mae も g_diagnostics も straight-through と clamp より上流なので、
        代理勾配が潰れて θ が全く動いていなくても全て正常値を出す。
        3〜13時間の予算を空回りで使い切る事故は、この値が 0 に張り付くことでしか見えない。
        ★群名は build_optimizer が param_groups[i]["name"] に入れた値を読む（並び順に依存しない）。
        ★時計つきモデルでは <prefix>_gnorm_clock が加わる。agg_gnorm_clock が 0 に張り付くなら、
          集計勾配は時計の経路を使っていない。

    Args:
        optimizer: build_optimizer が作った AdamW。各 param_group が "name" キーを持つ
        prefix: 出力キーの接頭辞。集計側だけの勾配なら "agg"、両項を足した後なら "total"

    Returns:
        統計量の dict[str, float]
            <prefix>_gnorm_cond: 群専用の条件パラメータ（cond_embeds / cond_proj / null_emb）
            <prefix>_gnorm_emb: emb_proj（時刻・条件の共有注入路）
            <prefix>_gnorm_clock: clock_proj（時計）。時計つきモデルのときだけ
            <prefix>_gnorm_conv: conv・attention・GroupNorm

    Raises:
        KeyError: param_group に "name" が無い場合（build_optimizer 以外で作った optimizer）
    """
    out: dict[str, float] = {}
    for group in optimizer.param_groups:
        if "name" not in group:
            raise KeyError("param_group に 'name' が無い。build_optimizer で作った optimizer を渡すこと")
        name = group["name"]
        sq = sum(float(p.grad.detach().pow(2).sum())
                 for p in group["params"] if p.grad is not None)
        out[f"{prefix}_gnorm_{name}"] = math.sqrt(sq)
    return out


def run(steps: int = DEFAULT_STEPS,
        d_sub: int = DEFAULT_D_SUB,
        n: int = DEFAULT_N,
        K: int = DEFAULT_K,
        eps: float = float("inf"),
        loss_kind: str = "sq",
        lam: float | None = None,
        chunk: int | None = None,
        holdout: list[int] | None = None,
        save_every: int = DEFAULT_SAVE_EVERY,
        val_every: int = DEFAULT_VAL_EVERY,
        ckpt_dir: Path = CKPT_DIR,
        stage1_ckpt: Path = STAGE1_CKPT,
        resume: bool = False,
        seed: int = 42,
        use_wandb: bool = True,
        device: str | None = None,
        lr_cond: float = LR_COND,
        lr_emb: float = LR_EMB,
        lr_conv: float = LR_CONV,
        lr_clock: float = LR_CLOCK) -> torch.nn.Module:
    """Stage 2 を固定ステップ回す, 返すのは最終ステップのモデル

    lr_cond / lr_emb / lr_conv / lr_clock は build_optimizer の層別学習率。
    lr_clock は時計つきの Stage 1（model.py の --clock）のときだけ使う。
    """
    dev = device or sm.DEVICE
    check_shapes(K, d_sub, n)
    chunk = resolve_chunk(K, d_sub, n, chunk)
    check_memory_budget(K, d_sub, n, chunk)
    # ★jsd は chunk を無視して1パスで回る（aggregate_step の Note）。上の判定は
    #   K×chunk しか見ていないので、jsd のときだけ K×B で測り直す。これが無いと
    #   「両方のゲートを通ってから OOM」になり、5時間の枠を失う
    if loss_kind == "jsd":
        check_memory_budget(K, d_sub, n, chunk=d_sub * n)
    torch.manual_seed(seed)

    # ---- 教師 ----
    tgt = st.load_stula_targets()
    a_star_all = sl.teacher_tensor(tgt, dev)                  # (28,12,96)
    omega_all = sl.chi2_weights(a_star_all, eps)
    teacher_mask = build_teacher_mask(holdout or [])
    teacher_groups = np.flatnonzero(teacher_mask)

    # ---- モデル ----
    model = sm.load_pretrained(stage1_ckpt).to(dev)
    diffusion = sm.Diffusion(device=dev)
    optimizer = build_optimizer(model, lr_cond=lr_cond, lr_emb=lr_emb,
                                lr_conv=lr_conv, lr_clock=lr_clock)
    stage1_clock = bool(getattr(model, "clock", False))

    # ---- ATUS リハーサル用のイテレータ ----
    cond_idx, sched, weight, _ = sm.load_data()
    train_loader, val_loader = sm.make_loaders(cond_idx, sched, weight)

    def atus_batches():
        while True:
            yield from train_loader

    atus = atus_batches()

    # ---- 乱数源は2つ。どちらも再開時に復元する ----
    # torch : x_T ~ N(0,I) と末尾 K 区間の雑音 zs（load_ckpt が必ず戻す）
    # numpy : 群サブサンプリング d_pick（load_ckpt に rng を渡したときだけ戻る）
    grid = torch.as_tensor(sm.cond_grid(), device=dev)         # (28,3)
    rng = np.random.default_rng(seed)

    # ---- 再開 ----
    # ★λ も引き継ぐ。--lam auto のまま再開すると、再開後の1ステップから新しい λ を
    #   引き直してしまい、「同じ設定の別の実験」が元の run の続きとして記録される
    lam_auto = lam is None
    lam_samples: list[tuple[float, float]] = []
    start_step = 0
    if resume:
        latest = ck.latest_ckpt(ckpt_dir)
        if latest is not None:
            start_step, prev_config = ck.load_ckpt(latest, model, optimizer,
                                                    map_location=dev, np_rng=rng)
            print(f"[resume] {latest.name} から再開する (step={start_step})")
            prev_lam = prev_config.get("lam")
            if lam_auto and isinstance(prev_lam, (int, float)) and math.isfinite(prev_lam):
                lam, lam_auto = float(prev_lam), False
                print(f"[resume] lam を引き継ぐ: {lam:.4g}（再推定しない）")

    # ★どの設定・どの Stage1 重みから作られた ckpt かを重みと一緒に残す（§9.8）。
    #   stage2_select が **config をそのまま CSV 列にするので、ここに入れた分だけ
    #   事後選択の出所列が増える。過去に生成プールの取り違えを起こしている
    base_config: dict[str, Any] = {
        "d_sub": d_sub, "n": n, "K": K, "eps": eps, "loss": loss_kind,
        "chunk": chunk, "holdout": holdout or [], "seed": seed,
        "stage1_ckpt": stage1_ckpt.name,
        "lr_cond": lr_cond, "lr_emb": lr_emb, "lr_conv": lr_conv,
        # ★時計なしの Stage 1 では lr_clock を使わないので NaN で残す（CSV 列を揃えるため）
        "stage1_clock": stage1_clock,
        "lr_clock": lr_clock if stage1_clock else float("nan"),
        "guidance_scale": sm.GUIDANCE_SCALE,
    }

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(project="domain-transfer-ddpm-agg",
                                job_type="stage2",
                                config={**base_config, "steps": steps, "lam": lam})

    n_pass = 1 if chunk >= d_sub * n else 2
    print(f"[stage2] steps={steps} d_sub={d_sub} n={n} K={K} eps={eps} loss={loss_kind} "
            f"chunk={chunk} ({n_pass}パス) teacher_groups={len(teacher_groups)}/28 device={dev}")
    print(f"[stage2] stage1={stage1_ckpt.name} guidance={sm.GUIDANCE_SCALE} "
            f"lr cond={lr_cond:g} emb={lr_emb:g} conv={lr_conv:g} "
            f"clock={f'{lr_clock:g}' if stage1_clock else '（時計なし）'} "
            f"val_every={val_every}")

    for step in range(start_step + 1, steps + 1):
        t0 = time.time()
        # ---- 群サブサンプリング（一様抽出, 多数回の更新で各群が等しく現れる）----
        d_pick = np.sort(rng.choice(teacher_groups, size=min(d_sub, len(teacher_groups)),
                                    replace=False))
        d_pick_t = torch.as_tensor(d_pick, device=dev)
        cond = grid[d_pick_t].repeat_interleave(n, dim=0)       # (d_sub*n, 3) 群優先
        a_star = a_star_all[d_pick_t]
        omega = omega_all[d_pick_t]

        # ---- 集計側（eval モードで微分する。§12 未決 G）----
        # ★zero_grad を先に置く。aggregate_step は内部で backward まで済ませるため
        optimizer.zero_grad(set_to_none=True)
        l_agg, a_A, a_B, x0_diag = aggregate_step(diffusion, model, cond, K, n, a_star,
                                                    omega, chunk, loss_kind)
        # ★集計側だけの θ 勾配をここで測る。リハーサル側を足した後では
        #   「集計の信号が straight-through と clamp を抜けて来ているか」が分からない
        agg_gnorm = grad_norms(optimizer, "agg")

        # ---- リハーサル側（train モード。Stage 1 と同じ損失）----
        # ★集計側の backward が済んでから作る。2つの計算グラフを同時に持たないので
        #   ピークは max(集計側, リハーサル側) であって和にならない（§4.5）
        model.train()
        b_cond, b_sched = next(atus)
        l_atus = diffusion.loss(model, b_sched.to(dev), b_cond.to(dev))
        l_atus_num = float(l_atus.detach())

        # ---- λ を最初の LAM_WARMUP_STEPS 更新の中央値で決める ----
        # 1更新だけで決めると L_agg の振れ（split-batch 推定量なので負にもなる）が
        # そのまま全ステップの重みになる。warmup 中は暫定値で回し、揃ったら固定する
        if lam_auto:
            lam_samples.append((abs(l_agg), max(l_atus_num, 1e-12)))
            lam = float(np.median([a for a, _ in lam_samples])
                        / np.median([b for _, b in lam_samples]))
            if len(lam_samples) >= LAM_WARMUP_STEPS:
                lam_auto = False
                print(f"[stage2] lam=auto -> {lam:.4g} で確定"
                        f"（{LAM_WARMUP_STEPS} 更新の中央値。"
                        f"|L_agg| {min(a for a, _ in lam_samples):.4g}"
                        f"〜{max(a for a, _ in lam_samples):.4g}）")
            else:
                print(f"[stage2] lam=auto 暫定 {lam:.4g} "
                        f"(warmup {len(lam_samples)}/{LAM_WARMUP_STEPS})")

        assert lam is not None      # lam_auto の分岐か呼び出し側が必ず与えている
        (lam * l_atus).backward()
        total_gnorm = grad_norms(optimizer, "total")
        optimizer.step()

        # ---- 記録 ----
        with torch.no_grad():
            a_full = 0.5 * (a_A + a_B)          # 全 n 本の群平均（A半分とB半分の平均）
            log: dict[str, Any] = {
                    "step": step, "L_agg": l_agg,
                    "L_atus": l_atus_num, "lam": lam, "sec": time.time() - t0,
                    "rate_mae": float((a_full - a_star).abs().mean()),
                    "other_x_share": float(a_full[:, int(st.Common.OTHER_X)].mean()),
                    **agg_gnorm, **total_gnorm, **x0_diag}
            # ★g の診断は二次形式のときだけ。loss_grad は split-batch の二乗誤差の
            #   勾配なので、jsd で回しているときに混ぜると別の損失の勾配を報告することになる
            if loss_kind != "jsd":
                g_A, _ = sl.loss_grad(a_A, a_B, a_star, omega)
                log.update(sl.g_diagnostics(g_A, a_star, n, act_names=sm.ACT_NAMES))

        # ---- val の ε-MSE（リハーサル項の過学習監視）----
        # ★学習の乱数を汚さない実装になっている（val_epsilon_mse の Note）。
        #   300 更新は ATUS 学習分割の約23エポック相当なので、無監視では回さない
        if val_every > 0 and (step % val_every == 0 or step == start_step + 1):
            log["L_atus_val"] = val_epsilon_mse(diffusion, model, val_loader, dev)

        if wandb_run is not None:
            wandb_run.log(log)
        if step % 10 == 0 or step == start_step + 1:
            val_txt = (f"  val={log['L_atus_val']:.6f}" if "L_atus_val" in log else "")
            # 群の並びは param_groups と同じ（時計つきなら cond/emb/clock/conv）
            gnorm_txt = "/".join(f"{log[f'agg_gnorm_{g['name']}']:.1e}"
                                 for g in optimizer.param_groups)
            print(f"  step {step:4d}/{steps}  L_agg={log['L_agg']:+.6f}  "
                    f"L_atus={log['L_atus']:.6f}{val_txt}  rate_mae={log['rate_mae']:.5f}  "
                    f"OTHER_X={log['other_x_share']:.4f}  |g_agg|={gnorm_txt}  "
                    f"floor={log['x0_floor_frac']:.3f}  {log['sec']:.1f}s")

        if step % save_every == 0 or step == steps:
            ck.save_ckpt(ck.ckpt_path(ckpt_dir, step), model, optimizer, step,
                            {**base_config, "lam": lam}, np_rng=rng)
            print(f"  [ckpt] {ck.ckpt_path(ckpt_dir, step).name}")

    if wandb_run is not None:
        wandb_run.finish()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AggDDPM-Simple Stage 2: 公表集計表だけを教師にした微調整")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--d-sub", type=int, default=DEFAULT_D_SUB)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--eps", type=str, default="inf",
                    help="損失の重み床。inf=素のMSE（主A）、0.01=χ²（主B）")
    ap.add_argument("--loss", choices=["sq", "jsd"], default="sq",
                    help="jsd はアブレーション1点のみ（split-batch が使えない）")
    ap.add_argument("--chunk", type=int, default=0,
                    help="1度に勾配を保持する個票数。0 で予算から自動決定。"
                        "B 未満になると2パス勾配蓄積に切り替わる（勾配は一括計算と一致）")
    ap.add_argument("--lam", type=str, default="auto",
                    help=f"リハーサル重み。auto なら最初の {LAM_WARMUP_STEPS} 更新の "
                        "|L_agg|/L_atus の中央値で決めて以降固定する。"
                        "--resume では ckpt の値を引き継ぐ")
    ap.add_argument("--holdout-groups", type=str, default="",
                    help="LGO。損失から外す群をカンマ区切りで。'auto:4' で人口層化して4群選ぶ")
    ap.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    ap.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY,
                    help="ATUS val 分割の ε-MSE を測る間隔。0 で無効。"
                        "リハーサル項の過学習を見る唯一の監視点")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--stage1-ckpt", type=Path, default=STAGE1_CKPT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="生成を短くして数更新だけ回す動作確認")
    ap.add_argument("--lr-cond", type=float, default=LR_COND,
                    help="群専用の条件パラメータ（cond_embeds / cond_proj / null_emb）の学習率")
    ap.add_argument("--lr-emb", type=float, default=LR_EMB,
                    help="emb_proj（拡散ステップと条件の共有注入路）の学習率")
    ap.add_argument("--lr-conv", type=float, default=LR_CONV,
                    help="conv / attention / GroupNorm の学習率")
    ap.add_argument("--lr-clock", type=float, default=LR_CLOCK,
                    help="clock_proj（時計）の学習率。時計つきの Stage 1 のときだけ使う")
    args = ap.parse_args()
    lrs = {"lr_cond": args.lr_cond, "lr_emb": args.lr_emb,
           "lr_conv": args.lr_conv, "lr_clock": args.lr_clock}

    holdout: list[int] = []
    if args.holdout_groups.startswith("auto:"):
        holdout = stratified_holdout(st.load_stula_targets()["pop"],
                                    int(args.holdout_groups.split(":")[1]), args.seed)
        print(f"[stage2] 人口層化で外す群: {holdout}")
    elif args.holdout_groups:
        holdout = [int(x) for x in args.holdout_groups.split(",")]

    if args.smoke:
        # ★T_STEPS を短くしてから Diffusion を作る。sample_differentiable のループ範囲も
        #   モジュール変数を読むので、走り終わるまで差し替えたままにする
        sm.T_STEPS = 20
        run(steps=2,
            d_sub=2,
            n=4,
            K=args.K,
            eps=float(args.eps),
            loss_kind=args.loss,
            lam=None if args.lam == "auto" else float(args.lam),
            chunk=args.chunk or None,
            holdout=holdout,
            save_every=1,
            val_every=1,
            ckpt_dir=args.ckpt_dir / "smoke",
            stage1_ckpt=args.stage1_ckpt,
            seed=args.seed,
            use_wandb=False,
            **lrs)
        print("stage2 smoke: OK")
        return

    run(steps=args.steps,
        d_sub=args.d_sub,
        n=args.n,
        K=args.K,
        eps=float(args.eps),
        loss_kind=args.loss,
        lam=None if args.lam == "auto" else float(args.lam),
        chunk=args.chunk or None,
        holdout=holdout,
        save_every=args.save_every,
        val_every=args.val_every,
        ckpt_dir=args.ckpt_dir,
        stage1_ckpt=args.stage1_ckpt,
        resume=args.resume,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        **lrs)


if __name__ == "__main__":
    main()
