"""
stage1_rare_diagnosis_gen.py
============================
少ない活動の行動者率が外れる原因の診断のうち、GPU で生成を回す部分（SQUID で実行）。
集計と判定は stage1_rare_diagnosis.py（手元）が行う。計画: ~/.claude/plans/stage1-crispy-hoare.md

    --mode restore       H5・H6。ATUS 平日の実個票 x0 を雑音水準 t0 まで進めてから逆過程で戻す
                         （Diffusion.restore）。t0 ごとに argmax 後の個票と、argmax 前の連続値の
                         活動別の和を保存する。t0 が小さければエピソードの長さだけ、大きければ
                         「誰がどの活動をするか」まで作り直すので、どの t0 から実データとずれるかで
                         逆過程のどの段階がずれを作るかを切り分ける
    --mode epoch-pools   H4。model.py --save-every が残した途中の ckpt ごとに小さなプールを作る。
                         どれも同じ乱数の種（pool_seed）から作るので、差は重みの差だけになる

データフロー:

```mermaid
flowchart TD
    subgraph restore["--mode restore"]
        LD["model.load_data<br/>cond_idx (N, 3) / sched (N, 96)"] --> X0["sched_to_x0<br/>x0 (B, 12, 96)"]
        X0 --> RS["Diffusion.restore(model, x0, t0, cond_idx)<br/>q_sample → _sample_tail(K = t0 + 1)"]
        RS --> OUT["連続値 (B, 12, 96)"]
        OUT --> AM["argmax → sched_out (T0, N, 96)"]
        OUT --> SUM["raw_sum / soft_sum (T0, N, 12)"]
        AM --> NPZ["ddpm_simple_restore{接尾辞}.npz"]
        SUM --> NPZ
    end
    subgraph epoch["--mode epoch-pools"]
        EP["{ckpt の stem}_ep{epoch:04d}.pt"] --> GP["torch.manual_seed(pool_seed)<br/>group_pool(model, n_per_group)"]
        GP --> CSV["ddpm_simple_pretrain_samples{接尾辞}_ep{epoch:04d}_n{M}.csv"]
    end
```

使い方（SQUID では jobs/rare_diag_simple.sh から）:
    python src/eval/diagnostics/stage1_rare_diagnosis_gen.py --mode restore --arm clock_tf96 --seed 43
    python src/eval/diagnostics/stage1_rare_diagnosis_gen.py --mode epoch-pools --arm clock_tf96_traj \\
        --seed 43 --every 100
    手元で動作確認（小さな設定、出力先を分ける）:
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis_gen.py --mode restore --arm clock_tf96 \\
        --seed 42 --limit 16 --t0 0 3 --out /tmp/restore_smoke.npz
"""
import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"

# 部分ノイズ化の水準（計画で固定）。999 = T_STEPS − 1 はほぼ純粋な雑音からの生成
DEFAULT_T0: tuple[int, ...] = (10, 50, 100, 200, 300, 500, 700, 999)
# 雑音とプールの乱数の種。stage1_guidance_pool.DEFAULT_POOL_SEED と同じ値（arm の間で共通乱数）
DEFAULT_POOL_SEED = 12345
# epoch ごとのプールの大きさ。ボランティア（行動者 4.5%）でも 1 本あたり約 80 人の行動者が出る
DEFAULT_EPOCH_POOL_N = 64
EPOCH_CKPT_RE = re.compile(r"_ep(\d{4})\.pt$")


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage2_select.py と同じ規則）。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sm: Any = _load("simple_model", SIMPLE_DIR / "model.py")
arms: Any = _load("simple_stage1_arms", SIMPLE_DIR / "stage1_arms.py")


# ============================================================
# 保存先の名前（stage1_arms.arm_suffix が唯一の出所）
# ============================================================
def ckpt_path(arm: str, seed: int) -> Path:
    """(arm, seed) の最良の ckpt のパス（存在は確かめない）"""
    stem = sm.MODEL_SAVE_PATH.stem
    return sm.MODEL_SAVE_PATH.with_name(f"{stem}{arms.arm_suffix(arm, seed)}.pt")


def restore_path(arm: str, seed: int) -> Path:
    """(arm, seed) の部分ノイズ化の結果（npz）のパス"""
    return sm.GEN_SAVE_PATH.with_name(f"ddpm_simple_restore{arms.arm_suffix(arm, seed)}.npz")


def epoch_pool_path(arm: str, seed: int, epoch: int | None, n_per_group: int) -> Path:
    """途中の ckpt（epoch=None なら最良の ckpt）から作った小プールの CSV のパス"""
    tag = "best" if epoch is None else f"ep{epoch:04d}"
    stem = sm.GEN_SAVE_PATH.stem
    return sm.GEN_SAVE_PATH.with_name(f"{stem}{arms.arm_suffix(arm, seed)}_{tag}_n{n_per_group}.csv")


def epoch_ckpts(arm: str, seed: int) -> dict[int, Path]:
    """model.epoch_ckpt_path の規則で保存された途中の ckpt, epoch → パス"""
    best = ckpt_path(arm, seed)
    found: dict[int, Path] = {}
    for p in best.parent.glob(f"{best.stem}_ep[0-9][0-9][0-9][0-9].pt"):
        m = EPOCH_CKPT_RE.search(p.name)
        if m is not None:
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def is_fresh(out: Path, ckpt: Path) -> bool:
    """出力が ckpt より新しければ True（作り直さない）"""
    return out.exists() and out.stat().st_mtime > ckpt.stat().st_mtime


# ============================================================
# --mode restore（H5・H6）
# ============================================================
@torch.no_grad()
def restore_all(model: Any, cond_idx: np.ndarray, sched: np.ndarray, t0_list: list[int],
                guidance_scale: float, noise_seed: int) -> dict[str, np.ndarray]:
    """全個票を t0 ごとに雑音化して戻す

    Args:
        model: 学習済み UNet1D
        cond_idx: 条件インデックス, dtype=int64, (N, 3)
        sched: 実データのスケジュール, dtype=int64, (N, 96)
        t0_list: 雑音化する水準の並び
        guidance_scale: CFG の強さ
        noise_seed: t0 ごとに torch.manual_seed へ渡す種（arm の間で雑音を揃える）

    Returns:
        t0 (T0,), sched_out (T0, N, 96) uint8,
        raw_sum (T0, N, 12)  argmax 前の連続値を活動ごとにスロットで和をとったもの,
        soft_sum (T0, N, 12) 連続値を各スロットで和 1 に正規化してから和をとったもの
    """
    dev = next(model.parameters()).device
    diffusion = sm.Diffusion(device=dev)
    n = len(sched)
    sched_out = np.zeros((len(t0_list), n, sm.NUM_SLOTS), dtype=np.uint8)
    raw_sum = np.zeros((len(t0_list), n, sm.NUM_ACT), dtype=np.float32)
    soft_sum = np.zeros((len(t0_list), n, sm.NUM_ACT), dtype=np.float32)
    for k, t0 in enumerate(t0_list):
        start = time.time()
        torch.manual_seed(noise_seed)
        for i in range(0, n, sm.GEN_BATCH):
            s = torch.as_tensor(sched[i:i + sm.GEN_BATCH], device=dev)
            ci = torch.as_tensor(cond_idx[i:i + sm.GEN_BATCH], device=dev)
            out = diffusion.restore(model, sm.sched_to_x0(s), t0, ci, guidance_scale)   # (B, 12, 96)
            pos = out.clamp_min(0.0)
            soft = pos / pos.sum(dim=1, keepdim=True).clamp_min(1e-8)
            sched_out[k, i:i + len(s)] = out.argmax(dim=1).cpu().numpy().astype(np.uint8)
            raw_sum[k, i:i + len(s)] = out.sum(dim=2).cpu().numpy()
            soft_sum[k, i:i + len(s)] = soft.sum(dim=2).cpu().numpy()
        print(f"[restore] t0={t0:4d}  ({time.time() - start:.0f}s)", flush=True)
    return {"t0": np.asarray(t0_list, dtype=np.int64), "sched_out": sched_out,
            "raw_sum": raw_sum, "soft_sum": soft_sum}


def run_restore(args: argparse.Namespace) -> None:
    ckpt = ckpt_path(args.arm, args.seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"チェックポイントが無い ({args.arm}, seed={args.seed}): {ckpt}")
    out = Path(args.out) if args.out else restore_path(args.arm, args.seed)
    if args.out is None and is_fresh(out, ckpt) and not args.force:
        print(f"[restore] skip（ckpt より新しい）: {out.name}")
        return
    cond_idx, sched, _, _ = sm.load_data()
    if args.limit is not None:
        cond_idx, sched = cond_idx[:args.limit], sched[:args.limit]
    model = sm.load_pretrained(ckpt)
    res = restore_all(model, cond_idx, sched, args.t0, args.guidance, args.pool_seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t0=res["t0"], sched_out=res["sched_out"], raw_sum=res["raw_sum"],
                        soft_sum=res["soft_sum"], guidance=np.float64(args.guidance),
                        noise_seed=np.int64(args.pool_seed), n_rows=np.int64(len(sched)))
    print(f"[restore] {args.arm} seed={args.seed} -> {out}")


# ============================================================
# --mode epoch-pools（H4）
# ============================================================
def run_epoch_pools(args: argparse.Namespace) -> None:
    found = epoch_ckpts(args.arm, args.seed)
    if not found:
        raise FileNotFoundError(f"途中の ckpt が無い ({args.arm}, seed={args.seed})。"
                                "model.py --save-every で学習すること")
    wanted = args.epochs if args.epochs else [e for e in found if e % args.every == 0]
    missing = [e for e in wanted if e not in found]
    if missing:
        raise FileNotFoundError(f"途中の ckpt が無い epoch: {missing}（ある: {list(found)}）")
    targets: list[tuple[int | None, Path]] = [(e, found[e]) for e in wanted]
    if args.with_best:
        targets.append((None, ckpt_path(args.arm, args.seed)))
    for epoch, ckpt in targets:
        out = epoch_pool_path(args.arm, args.seed, epoch, args.n_per_group)
        if is_fresh(out, ckpt) and not args.force:
            print(f"[epoch-pools] skip（ckpt より新しい）: {out.name}")
            continue
        start = time.time()
        model = sm.load_pretrained(ckpt)
        torch.manual_seed(args.pool_seed)
        pool = sm.group_pool(model, args.n_per_group, guidance_scale=args.guidance, verbose=False)
        sm.write_pool_csv(pool, out)
        print(f"[epoch-pools] {args.arm} seed={args.seed} epoch={epoch} -> {out.name} "
              f"({time.time() - start:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("restore", "epoch-pools"), required=True)
    ap.add_argument("--arm", required=True, help="stage1_arms.ARMS のキー")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--guidance", type=float, default=sm.GUIDANCE_SCALE)
    ap.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED,
                    help="雑音（restore）とプール（epoch-pools）の乱数の種")
    ap.add_argument("--force", action="store_true", help="出力が ckpt より新しくても作り直す")
    # restore
    ap.add_argument("--t0", type=int, nargs="+", default=list(DEFAULT_T0), help="雑音化する水準")
    ap.add_argument("--limit", type=int, default=None, help="先頭の N 人だけ使う（動作確認用）")
    ap.add_argument("--out", default=None, help="restore の出力先（既定は outputs/generated の決まった名前）")
    # epoch-pools
    ap.add_argument("--every", type=int, default=100, help="この倍数の epoch の ckpt だけ使う")
    ap.add_argument("--epochs", type=int, nargs="*", default=None,
                    help="使う epoch を明示する（--every より優先。ジョブを分けるとき）")
    ap.add_argument("--with-best", action="store_true", help="最良の ckpt のプールも作る")
    ap.add_argument("--n-per-group", type=int, default=DEFAULT_EPOCH_POOL_N)
    args = ap.parse_args()
    if any(not 0 <= t < sm.T_STEPS for t in args.t0):
        ap.error(f"--t0 は [0, {sm.T_STEPS - 1}]: {args.t0}")
    if args.every <= 0:
        ap.error(f"--every は正: {args.every}")
    if args.mode == "restore":
        run_restore(args)
    else:
        run_epoch_pools(args)


if __name__ == "__main__":
    main()
