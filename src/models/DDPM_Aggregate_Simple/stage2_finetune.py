"""
stage2_finetune.py
==================
Stage 2 の学習ループ（Stage2_design.md §8, §11.2）

**日本の集計表だけを教師にして拡散モデルのパラメータを更新する。日本の個票は使わない。**

1更新でやること:
    1. 教師群から |D_sub| 群を一様抽出し、群あたり n 本を打ち切り逆伝播つきで生成
    2. straight-through で one-hot 化し、群ごとに A/B へ二分して集計
    3. L_agg = split-batch 不偏推定の重み付き二乗誤差
    4. L_atus = ATUS 実個票のリハーサル項（Stage 1 と同じ ε-MSE）
    5. L_agg.backward() と (λ·L_atus).backward() を別々に呼んで AdamW を1歩

早期終了は使わない（§8.4）。固定ステップ予算で回し切り、M_save ごとに保存した
チェックポイントを学習後に (教師適合, ガードレール) の2軸で選ぶ。Stage 1 の
val ε-MSE 早期終了は目的関数が別物なので流用できない。

★リハーサル項は「学習中のモデル」で計算する。凍結モデルと比べるのは別物
  （関数空間アンカー λ_fn。§8.3 の補助アンカーで、初期は 0 固定）。
  失うのが怖いのは個票レベルの現実性（断片化・travel pairing・睡眠の連続性）で、
  それを教えているのは ATUS の実個票そのものだから、直接それを守る。

★λ は「米国事前分布への依存度のダイヤル」。単一点ではなくパレート曲線を報告する。
  初期値は勘で決めず、1回目の L_agg と L_atus の実測値を見て同程度になる値の
  周りを対数スケールで振る（--lam を明示しなければ最初の更新で自動的に合わせる）。

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
        --eps V          損失の重み床。inf=素のMSE（主A）、0.01=χ²（主B）
        --loss {sq,jsd}  jsd はアブレーション1点のみ
        --lam V          リハーサル重み。auto なら初回の L_agg/L_atus 比で決める
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
    """sys.modules に一意名で載せる（model.py:76 と同じ様式）。

    このリポジトリは model.py という同名ファイルを4つのモデルフォルダに持つので、
    通常の import は sys.path の順序次第で静かに別モジュールを掴む。
    """
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
# 学習
# ============================================================
def build_optimizer(model: torch.nn.Module,
                    lr_cond: float = LR_COND, lr_conv: float = LR_CONV
                    ) -> torch.optim.Optimizer:
    """層別学習率つき AdamW（§8.3）。

    §8.2 の実測で、教師とのズレは位相ではなく「量」が支配することが分かっている
    （必要な時刻シフトは全活動45分以内で、大半はシフトによる改善が0〜6%）。量を動かすのは
    条件経路なので、そちらを高い学習率にして conv/attention は低く抑える。
    畳み込みは全パラメータの66%を占め、系列構造を直接壊せる側である。
    """
    cond, conv = [], []
    for name, p in model.named_parameters():
        (cond if any(k in name for k in COND_PATH_KEYS) else conv).append(p)
    if not cond or not conv:
        raise ValueError(f"層別 LR の分割に失敗した (cond={len(cond)}, conv={len(conv)})")
    return torch.optim.AdamW([{"params": cond, "lr": lr_cond},
                              {"params": conv, "lr": lr_conv}], weight_decay=0.0)


def check_memory_budget(K: int, d_sub: int, n: int, budget: int = MEMORY_BUDGET) -> None:
    """1パス版のピークは K × D_sub × n で決まる（§4.5）。超過なら学習前に落とす。

    ★5時間回した後に OOM で失う事故を防ぐための事前チェック。逃げ道は優先順に
      (1) D_sub を下げる  (2) 2パス勾配蓄積で chunk を下げる  (3) 勾配チェックポイント
    """
    load = K * d_sub * n
    if load > budget:
        raise SystemExit(
            f"ERROR: K×D_sub×n = {K}×{d_sub}×{n} = {load:,} が予算 {budget:,} を超えている。\n"
            f"       D_sub <= {budget // (K * n)} に下げるか、K を下げること（§4.5）")


def run(steps: int = DEFAULT_STEPS, d_sub: int = DEFAULT_D_SUB, n: int = DEFAULT_N,
        K: int = DEFAULT_K, eps: float = float("inf"), loss_kind: str = "sq",
        lam: float | None = None,
        save_every: int = DEFAULT_SAVE_EVERY, ckpt_dir: Path = CKPT_DIR,
        stage1_ckpt: Path = STAGE1_CKPT, resume: bool = False,
        seed: int = 42, use_wandb: bool = True, device: str | None = None) -> torch.nn.Module:
    """Stage 2 を固定ステップ予算で回す。返すのは最終ステップのモデル。

    ★早期終了は入れない。何が起きているか分からないまま止まるのを避けるため、
      まず固定ステップで1本回し、L_agg と ATUS val ε-MSE の推移を見てから
      判定量と閾値を決める（§8.4）。
    ★まず28群すべてを教師にする。LGO（leave-groups-out）は teacher_mask として
      後から足せる形にしてある（§8.4、§10.2-8）。
    """
    dev = device or sm.DEVICE
    check_memory_budget(K, d_sub, n)
    torch.manual_seed(seed)

    # ---- 教師 ----
    tgt = st.load_stula_targets()
    q_all = sl.teacher_tensor(tgt, dev)                       # (28,12,96)
    omega_all = sl.chi2_weights(q_all, eps)
    teacher_groups = np.arange(st.D_GROUPS)

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
        wandb_run = wandb.init(project="domain-transfer-ddpm-agg", job_type="stage2",
                               config={"steps": steps, "d_sub": d_sub, "n": n, "K": K,
                                       "eps": eps, "loss": loss_kind, "lam": lam,
                                       "lr_cond": LR_COND, "lr_conv": LR_CONV,
                                       "seed": seed})

    print(f"[stage2] steps={steps} d_sub={d_sub} n={n} K={K} eps={eps} loss={loss_kind} "
          f"teacher_groups={len(teacher_groups)}/28 device={dev}")

    for step in range(start_step + 1, steps + 1):
        t0 = time.time()
        # ---- 群サブサンプリング（一様抽出。多数回の更新で各群が等しく現れる）----
        d_pick = np.sort(rng.choice(teacher_groups, size=min(d_sub, len(teacher_groups)),
                                    replace=False))
        d_pick_t = torch.as_tensor(d_pick, device=dev)
        cond = grid[d_pick_t].repeat_interleave(n, dim=0)       # (d_sub*n, 3) 群優先
        q = q_all[d_pick_t]
        omega = omega_all[d_pick_t]

        # ---- 集計側（eval モードで微分する。§12 未決 G）----
        x0 = diffusion.sample_differentiable(model, cond, K)
        y = sm.straight_through(x0)
        if loss_kind == "jsd":
            l_agg = sl.jsd_loss(y, q, n)
            a_A, a_B = sl.group_rates_split(y.detach(), n)
        else:
            a_A, a_B = sl.group_rates_split(y, n)
            l_agg = sl.agg_loss_from_rates(a_A, a_B, q, omega)

        # ---- リハーサル側（train モード。Stage 1 と同じ損失）----
        model.train()
        b_cond, b_sched = next(atus)
        l_atus = diffusion.loss(model, b_sched.to(dev), b_cond.to(dev))

        # ---- λ を初回の実測値で決める（§8.3「勘で決めない」）----
        if lam is None:
            lam = float(abs(l_agg.detach())) / max(float(l_atus.detach()), 1e-12)
            print(f"[stage2] lam=auto -> {lam:.4g} "
                  f"(L_agg={float(l_agg.detach()):.4g} / L_atus={float(l_atus.detach()):.4g})")

        # ---- 更新。★別々に backward してピークを max(両者) に抑える（§4.5）----
        optimizer.zero_grad(set_to_none=True)
        l_agg.backward()
        (lam * l_atus).backward()
        optimizer.step()

        # ---- 記録（生成済みの量から追加コストなしに取れるものだけ）----
        with torch.no_grad():
            a_full = y.detach().view(len(d_pick), n, sm.NUM_ACT, sm.NUM_SLOTS).mean(dim=1)
            log = {"step": step, "L_agg": float(l_agg.detach()),
                   "L_atus": float(l_atus.detach()), "lam": lam, "sec": time.time() - t0,
                   "rate_mae": float((a_full - q).abs().mean()),
                   "other_x_share": float(a_full[:, int(st.Common.OTHER_X)].mean())}
            # ★g の診断は二次形式のときだけ。loss_grad は split-batch の二乗誤差の
            #   勾配なので、jsd で回しているときに混ぜると別の損失の勾配を報告することになる
            if loss_kind != "jsd":
                g_A, _ = sl.loss_grad(a_A.detach(), a_B.detach(), q, omega)
                log.update(sl.g_diagnostics(g_A, q, n, act_names=sm.ACT_NAMES))
        if wandb_run is not None:
            wandb_run.log(log)
        if step % 10 == 0 or step == start_step + 1:
            print(f"  step {step:4d}/{steps}  L_agg={log['L_agg']:+.6f}  "
                  f"L_atus={log['L_atus']:.6f}  rate_mae={log['rate_mae']:.5f}  "
                  f"OTHER_X={log['other_x_share']:.4f}  {log['sec']:.1f}s")

        if step % save_every == 0 or step == steps:
            ck.save_ckpt(ck.ckpt_path(ckpt_dir, step), model, optimizer, step,
                         {"d_sub": d_sub, "n": n, "K": K, "eps": eps, "loss": loss_kind,
                          "lam": lam, "seed": seed})
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
    ap.add_argument("--lam", type=str, default="auto",
                    help="リハーサル重み。auto なら初回の L_agg/L_atus 比で決める")
    ap.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--stage1-ckpt", type=Path, default=STAGE1_CKPT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="生成を短くして数更新だけ回す動作確認")
    args = ap.parse_args()

    if args.smoke:
        # ★T_STEPS を短くしてから Diffusion を作る。sample_differentiable のループ範囲も
        #   モジュール変数を読むので、走り終わるまで差し替えたままにする
        sm.T_STEPS = 20
        run(steps=2, d_sub=2, n=4, K=args.K, eps=float(args.eps), loss_kind=args.loss,
            lam=None if args.lam == "auto" else float(args.lam),
            save_every=1, ckpt_dir=args.ckpt_dir / "smoke",
            stage1_ckpt=args.stage1_ckpt, seed=args.seed, use_wandb=False)
        print("stage2 smoke: OK")
        return

    run(steps=args.steps, d_sub=args.d_sub, n=args.n, K=args.K, eps=float(args.eps),
        loss_kind=args.loss, lam=None if args.lam == "auto" else float(args.lam),
        save_every=args.save_every, ckpt_dir=args.ckpt_dir,
        stage1_ckpt=args.stage1_ckpt, resume=args.resume, seed=args.seed,
        use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
