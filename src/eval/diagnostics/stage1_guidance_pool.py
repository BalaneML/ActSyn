"""
stage1_guidance_pool.py
=======================
Stage 1 の学習済みモデルから、CFG の強さ（guidance_scale）を変えて生成プールを作り直す（段 0）。

問い: 時刻符号つき Stage 1 の WORK 12:00 の過大（生成 0.370 / 実 0.303）は、
      学習の問題か、生成時の CFG（既定 1.25）の問題か。

設計:
    ★比べる 2 本は同じ重み・同じ乱数の種（pool_seed）から作る。学習時の sanity_check が
      書いたプールは学習の乱数状態を引き継いでいるので、guidance だけを変えた比較にならない。
      そこで 1.25 も含めて、ここで作り直す（共通乱数）。
    ★保存先は学習時のプール名に _g{scale} を付けた別名。学習時のプールは上書きしない。
    ★出力が ckpt より新しければ作り直さない（--force で作り直す）。

    arm と種から ckpt と出力の名前を引くのは stage1_arms.arm_suffix（唯一の出所）。

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_guidance_pool.py --arm clock --seeds 42 43 44
    .venv/bin/python src/eval/diagnostics/stage1_guidance_pool.py --arm clock --guidance 1.0 1.25 1.5

出力:
    outputs/generated/ddpm_simple_pretrain_samples{接尾辞}_g{scale}.csv
    評価は stage1_clock_replicates.py --arms clock@g1 clock@g1.25
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"

# プール生成の乱数の種。stage2_select の DEFAULT_POOL_SEED と同じ値にそろえる
DEFAULT_POOL_SEED = 12345
DEFAULT_N_PER_GROUP = 256


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


def ckpt_path(arm: str, seed: int) -> Path:
    """(arm, seed) の Stage 1 チェックポイントのパス（存在は確かめない）"""
    stem = sm.MODEL_SAVE_PATH.stem
    return sm.MODEL_SAVE_PATH.with_name(f"{stem}{arms.arm_suffix(arm, seed)}.pt")


def guidance_pool_path(arm: str, seed: int, scale: float) -> Path:
    """(arm, seed, guidance) の生成プール CSV のパス（存在は確かめない）"""
    stem = sm.GEN_SAVE_PATH.stem
    return sm.GEN_SAVE_PATH.with_name(f"{stem}{arms.arm_suffix(arm, seed)}_g{scale:g}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="clock", help="stage1_arms.ARMS のキー")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, sm.GUIDANCE_SCALE])
    ap.add_argument("--n-per-group", type=int, default=DEFAULT_N_PER_GROUP)
    ap.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED)
    ap.add_argument("--force", action="store_true", help="出力が ckpt より新しくても作り直す")
    args = ap.parse_args()

    for seed in args.seeds:
        ckpt = ckpt_path(args.arm, seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"チェックポイントが無い ({args.arm}, seed={seed}): {ckpt}")
        model = sm.load_pretrained(ckpt)
        for scale in args.guidance:
            out = guidance_pool_path(args.arm, seed, scale)
            if out.exists() and out.stat().st_mtime > ckpt.stat().st_mtime and not args.force:
                print(f"[guidance] skip（ckpt より新しい）: {out.name}")
                continue
            t0 = time.time()
            torch.manual_seed(args.pool_seed)
            pool = sm.group_pool(model, args.n_per_group, guidance_scale=scale, verbose=False)
            sm.write_pool_csv(pool, out)
            print(f"[guidance] {args.arm} seed={seed} s={scale:g} -> {out.name} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
