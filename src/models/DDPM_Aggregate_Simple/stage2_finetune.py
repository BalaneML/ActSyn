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
        --loss {sq,jsd}  jsd はアブレーション1点のみ
        --lam V          リハーサル重み。auto なら初回の L_agg/L_atus 比で決める
        --holdout-groups d1,d2,...  LGO。損失から外す群（生成と評価は常に全28群）
        --resume         ckpt-dir の最新チェックポイントから再開
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
LR_COND = 1e-4               # §8.3 条件経路。Stage 1 の 2e-4 から下げる
LR_CONV = 1e-5               # §8.3 conv/attention。破壊力が大きいのでさらに1桁下げる
MEMORY_BUDGET = 6600         # §4.5 K×chunk <= 約6,600（A100 40GB）
STAGE1_CKPT = REPO_ROOT / "outputs" / "checkpoints" / "ddpm_simple_pretrain_common12_weekday_20260819.pt"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "stage2"

# 条件経路とみなすパラメータ名の断片（§8.3 の層別学習率）。
# cond_embeds / cond_proj / null_emb は条件そのもの、emb_proj は時刻・条件埋め込みを
# 各 ResBlock へ注入する経路。畳み込みの中身は一切含まない
COND_PATH_KEYS = ("cond_embeds", "cond_proj", "null_emb", "emb_proj")


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
def build_optimizer(model: torch.nn.Module, lr_cond: float = LR_COND,
                    lr_conv: float = LR_CONV) -> torch.optim.Optimizer:
    """層別学習率つき AdamW

    Note:
        UNet1D 1,759,124 params:
            cond 317,192 (18.0%) = cond_embeds + cond_proj + null_emb + emb_proj×11
            conv 1,441,932 (82.0%) = 畳み込み66.6% + attention15.1% + GroupNorm0.3%

    Args:
        model: Stage1 の重みを読んだUNet1D
        lr_cond: 条件経路の学習率, default=LR_COND=1e-4
        lr_conv: conv/attention の学習率, default=LR_CONV=1e-5
    
    Returns:
        param groupを持つ AdamW, weight_decay=0.0
    """
    cond, conv = [], []
    for name, p in model.named_parameters():
        (cond if any(k in name for k in COND_PATH_KEYS) else conv).append(p)
    if not cond or not conv:
        raise ValueError(f"層別 LR の分割に失敗した (cond={len(cond)}, conv={len(conv)})")
    return torch.optim.AdamW([{"params": cond, "lr": lr_cond},
                                {"params": conv, "lr": lr_conv}], weight_decay=0.0)


def check_memory_budget(K: int, d_sub: int, n: int, chunk: int | None = None,
                        budget: int = MEMORY_BUDGET) -> None:
    """勾配を保持するピーク = K × chunk, 学習前にチェック"""
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
                    tau: float = 1.0) -> tuple[float, torch.Tensor, torch.Tensor]:
    """集計側の1更新分の勾配を θ.grad へ蓄積し、(L_agg の値, ā_A, ā_B) を返す。

    ★返り値の損失はスカラー値であって計算グラフを持たない。backward はこの関数の中で
      済ませてある。呼び出し側は先に optimizer.zero_grad() を済ませておくこと。

    chunk >= B なら1パス（素直に autograd を通す）。chunk < B なら2パス勾配蓄積
    （gradient caching、§2.9）に切り替える。

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


def _aggregate_step_one_pass(diffusion: Any,
                            model: torch.nn.Module,
                            cond: torch.Tensor,
                            K: int,
                            n: int,
                            a_star: torch.Tensor,
                            omega: torch.Tensor,
                            loss_kind: str,
                            tau: float) -> tuple[float, torch.Tensor, torch.Tensor]:
    """素直に autograd を通す版, chunk >= B のとき, および jsd のときに使う"""
    zs = _draw_tail_noise(cond.size(0), K, cond.device)
    x0 = diffusion.sample_differentiable(model, cond, K, zs=zs)
    y = sm.straight_through(x0, tau)
    if loss_kind == "jsd":
        l_agg = sl.jsd_loss(y, a_star, n)
        a_A, a_B = sl.group_rates_split(y.detach(), n)
    else:
        a_A, a_B = sl.group_rates_split(y, n)
        l_agg = sl.agg_loss_from_rates(a_A, a_B, a_star, omega)
    l_agg.backward()
    return float(l_agg.detach()), a_A.detach(), a_B.detach()


def _aggregate_step_two_pass(diffusion: Any,
                            model: torch.nn.Module,
                            cond: torch.Tensor,
                            K: int,
                            n: int,
                            a_star: torch.Tensor,
                            omega: torch.Tensor,
                            chunk: int,
                            tau: float) -> tuple[float, torch.Tensor, torch.Tensor]:
    """2パス勾配蓄積"""
    total = cond.size(0)
    zs = _draw_tail_noise(total, K, cond.device)

    # ---- 1パス目: x_K まで進めて Ã と g を得る。グラフは作らない ----
    x_K = diffusion._sample_head(model, cond, K)
    with torch.no_grad():
        y = sm.straight_through(diffusion._sample_tail(model, x_K, K, cond, zs=zs), tau)
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
    return l_agg, a_A, a_B


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
        ckpt_dir: Path = CKPT_DIR,
        stage1_ckpt: Path = STAGE1_CKPT,
        resume: bool = False,
        seed: int = 42,
        use_wandb: bool = True,
        device: str | None = None) -> torch.nn.Module:
    """Stage 2 を固定ステップ回す, 返すのは最終ステップのモデル"""
    dev = device or sm.DEVICE
    chunk = resolve_chunk(K, d_sub, n, chunk)
    check_memory_budget(K, d_sub, n, chunk)
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
    optimizer = build_optimizer(model)

    # ---- ATUS リハーサル用のイテレータ ----
    cond_idx, sched, weight, _ = sm.load_data()
    train_loader, _ = sm.make_loaders(cond_idx, sched, weight)

    def atus_batches():
        while True:
            yield from train_loader

    atus = atus_batches()

    # ---- 再開 ----
    start_step = 0
    if resume:
        latest = ck.latest_ckpt(ckpt_dir)
        if latest is not None:
            start_step, _ = ck.load_ckpt(latest, model, optimizer, map_location=dev)
            print(f"[resume] {latest.name} から再開する (step={start_step})")

    grid = torch.as_tensor(sm.cond_grid(), device=dev)         # (28,3)
    rng = np.random.default_rng(seed)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(project="domain-transfer-ddpm-agg",
                                job_type="stage2",
                                config={"steps": steps,
                                        "d_sub": d_sub,
                                        "n": n,
                                        "K": K,
                                        "eps": eps,
                                        "loss": loss_kind,
                                        "lam": lam,
                                        "chunk": chunk,
                                        "holdout": holdout or [],
                                        "lr_cond": LR_COND,
                                        "lr_conv": LR_CONV,
                                        "seed": seed})

    n_pass = 1 if chunk >= d_sub * n else 2
    print(f"[stage2] steps={steps} d_sub={d_sub} n={n} K={K} eps={eps} loss={loss_kind} "
            f"chunk={chunk} ({n_pass}パス) teacher_groups={len(teacher_groups)}/28 device={dev}")

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
        l_agg, a_A, a_B = aggregate_step(diffusion, model, cond, K, n, a_star, omega,
                                        chunk, loss_kind)

        # ---- リハーサル側（train モード。Stage 1 と同じ損失）----
        # ★集計側の backward が済んでから作る。2つの計算グラフを同時に持たないので
        #   ピークは max(集計側, リハーサル側) であって和にならない（§4.5）
        model.train()
        b_cond, b_sched = next(atus)
        l_atus = diffusion.loss(model, b_sched.to(dev), b_cond.to(dev))

        # ---- λ を初回の実測値で決める ----
        if lam is None:
            lam = abs(l_agg) / max(float(l_atus.detach()), 1e-12)
            print(f"[stage2] lam=auto -> {lam:.4g} "
                    f"(L_agg={l_agg:.4g} / L_atus={float(l_atus.detach()):.4g})")

        (lam * l_atus).backward()
        optimizer.step()

        # ---- 記録 ----
        with torch.no_grad():
            a_full = 0.5 * (a_A + a_B)          # 全 n 本の群平均（A半分とB半分の平均）
            log = {"step": step, "L_agg": l_agg,
                    "L_atus": float(l_atus.detach()), "lam": lam, "sec": time.time() - t0,
                    "rate_mae": float((a_full - a_star).abs().mean()),
                    "other_x_share": float(a_full[:, int(st.Common.OTHER_X)].mean())}
            # ★g の診断は二次形式のときだけ。loss_grad は split-batch の二乗誤差の
            #   勾配なので、jsd で回しているときに混ぜると別の損失の勾配を報告することになる
            if loss_kind != "jsd":
                g_A, _ = sl.loss_grad(a_A, a_B, a_star, omega)
                log.update(sl.g_diagnostics(g_A, a_star, n, act_names=sm.ACT_NAMES))
        if wandb_run is not None:
            wandb_run.log(log)
        if step % 10 == 0 or step == start_step + 1:
            print(f"  step {step:4d}/{steps}  L_agg={log['L_agg']:+.6f}  "
                    f"L_atus={log['L_atus']:.6f}  rate_mae={log['rate_mae']:.5f}  "
                    f"OTHER_X={log['other_x_share']:.4f}  {log['sec']:.1f}s")

        if step % save_every == 0 or step == steps:
            ck.save_ckpt(ck.ckpt_path(ckpt_dir, step), model, optimizer, step,
                            {"d_sub": d_sub, "n": n, "K": K, "eps": eps, "loss": loss_kind,
                            "lam": lam, "chunk": chunk, "holdout": holdout or [],
                            "seed": seed})
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
                    help="リハーサル重み。auto なら初回の L_agg/L_atus 比で決める")
    ap.add_argument("--holdout-groups", type=str, default="",
                    help="LGO。損失から外す群をカンマ区切りで。'auto:4' で人口層化して4群選ぶ")
    ap.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--stage1-ckpt", type=Path, default=STAGE1_CKPT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="生成を短くして数更新だけ回す動作確認")
    args = ap.parse_args()

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
            ckpt_dir=args.ckpt_dir / "smoke",
            stage1_ckpt=args.stage1_ckpt,
            seed=args.seed,
            use_wandb=False)
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
        ckpt_dir=args.ckpt_dir,
        stage1_ckpt=args.stage1_ckpt,
        resume=args.resume,
        seed=args.seed,
        use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
