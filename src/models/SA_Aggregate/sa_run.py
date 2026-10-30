"""
sa_run.py
=========
比較 a（集計表だけから日誌を作る SA）を群ごとに回して保存する（SA_setup.md）

    .venv/bin/python src/models/SA_Aggregate/sa_run.py --variant rules --K 1000 --groups 0 5
    .venv/bin/python src/models/SA_Aggregate/sa_run.py --variant norules --K 1000 --workers 4

保存先（stem = sa_{variant}_K{K}_N{N}_seed{seed}）:
    outputs/generated/sa_aggregate/{stem}/group{d:02d}.npz   best_X (N, 96) int8・hist・秒数
    outputs/generated/sa_aggregate/{stem}/history_group{d:02d}.csv
    outputs/generated/sa_aggregate/{stem}_pool.npz           28 群がそろったときだけ。
                                                             pool (28, N, 96) は 12 分類 int64（gs2.score_pool に渡す形）、
                                                             pool20 は 20 分類 int8

```mermaid
flowchart LR
    R["sa_targets.load_rates20()<br/>rates20 (28, 20, 96)"] --> C["round_slot_counts(rates20[d], N)<br/>counts (20, 96)"]
    C --> X0["initial_solution(counts, rng)<br/>X (N, 96)"]
    P["sa_targets.load_participation20()<br/>part20 (28, 20)"] --> SNA["sna = round(part20[d] · N)"]
    X0 --> A["sa_core.anneal(...)"]
    SNA --> A
    A --> G["group{d:02d}.npz"]
    G --> POOL["{stem}_pool.npz<br/>STULA20_TO_COMMON で 12 分類へ"]
```

★既に保存した群があると止まる（取り違え防止）。続きから回すときは --skip-done を付ける。
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sa_core as core  # noqa: E402
import sa_targets as tg  # noqa: E402

OUT_ROOT = tg.REPO_ROOT / "outputs" / "generated" / "sa_aggregate"

# 論文どおりの冷却（SA_setup.md §5）。変えるのは冷却の間隔 K だけ
T_INIT = 1.0
T_CONV = 1e-4
COOLING_RATE = 0.999


def stem_of(variant: str, cooling_interval: int, n_people: int, seed: int) -> str:
    """保存名の stem"""
    return f"sa_{variant}_K{cooling_interval}_N{n_people}_seed{seed}"


def initial_solution(counts: npt.NDArray[np.int64], rng: np.random.Generator) -> npt.NDArray[np.int8]:
    """スロットごとに、統計どおりの人数の活動 ID をランダムに配る（SA_setup.md §4.1 手順 2）

    Args:
        counts: (20, 96) スロットごとの人数。各列の和が N
        rng: 乱数

    Returns:
        X (N, 96) int8
    """
    n_people = int(counts[:, 0].sum())
    X = np.empty((n_people, tg.NUM_SLOTS), dtype=np.int8)
    for s in range(tg.NUM_SLOTS):
        ids = np.repeat(np.arange(tg.NUM_ACT20, dtype=np.int8), counts[:, s])
        X[:, s] = rng.permutation(ids)
    return X


def group_seed(seed: int, d: int) -> int:
    """群ごとの乱数の種。初期解と anneal で同じ値を使う（使う乱数の系列は別）"""
    return seed * 1000 + d


def solve_group(d: int, variant: str, cooling_interval: int, n_people: int, seed: int,
                out_dir: Path) -> tuple[int, float, tuple[int, int, int]]:
    """群 d を解いて保存する

    Returns:
        (d, 秒数, 最良解の (f1, f2, f3))
    """
    rates20 = tg.load_rates20()
    part20 = tg.load_participation20()
    counts = tg.round_slot_counts(rates20[d], n_people)
    X0 = initial_solution(counts, np.random.default_rng(group_seed(seed, d)))
    sna = np.round(part20[d] * n_people).astype(np.int64)
    use_rules = variant == "rules"

    t0 = time.perf_counter()
    best_X, hist = core.anneal(X0.copy(), sna, core.F3_WEIGHT, core.RULE_MEMBER, core.RULE_MAIN,
                               core.RULE_CAP, core.RULE_RELEVANT, use_rules, cooling_interval,
                               T_INIT, T_CONV, COOLING_RATE, group_seed(seed, d))
    sec = time.perf_counter() - t0

    # 時刻別行動者率が初期解から変わっていないことを確かめる（近傍が構成上保つはずの量）
    for s in range(tg.NUM_SLOTS):
        got = np.bincount(best_X[:, s], minlength=tg.NUM_ACT20)
        if not np.array_equal(got, counts[:, s]):
            raise RuntimeError(f"群 {d} のスロット {s} で人数が変わった")
    f = core.objectives(best_X, sna, core.F3_WEIGHT, core.RULE_MEMBER, core.RULE_MAIN, core.RULE_CAP,
                        use_rules)
    np.savez_compressed(out_dir / f"group{d:02d}.npz", best_X=best_X, hist=hist, seconds=sec,
                        sna=sna, counts=counts)
    pd.DataFrame(hist, columns=core.HIST_COLS).to_csv(out_dir / f"history_group{d:02d}.csv", index=False)
    return d, sec, (int(f[0]), int(f[1]), int(f[2]))


def assemble_pool(out_dir: Path, n_people: int) -> Path | None:
    """28 群がそろっていれば、12 分類の評価のプールにまとめて保存する"""
    files = [out_dir / f"group{d:02d}.npz" for d in range(tg.D_GROUPS)]
    if not all(f.exists() for f in files):
        return None
    pool20 = np.stack([np.load(f)["best_X"] for f in files])          # (28, N, 96) int8
    pool = tg.STULA20_TO_COMMON[pool20.astype(np.int64)]              # (28, N, 96) int64
    path = out_dir.parent / f"{out_dir.name}_pool.npz"
    np.savez_compressed(path, pool=pool, pool20=pool20, n_people=n_people)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["rules", "norules"], required=True)
    ap.add_argument("--K", type=int, required=True, help="冷却の間隔（同じ温度で行う反復の回数）")
    ap.add_argument("--N", type=int, default=2000, help="群あたりの人数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groups", type=int, nargs="*", default=None, help="解く群（既定は 28 群すべて）")
    ap.add_argument("--workers", type=int, default=1, help="並列に解く群の数（プロセス）")
    ap.add_argument("--skip-done", action="store_true", help="保存済みの群を飛ばす")
    args = ap.parse_args()

    stem = stem_of(args.variant, args.K, args.N, args.seed)
    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = list(range(tg.D_GROUPS)) if args.groups is None else args.groups
    done = [d for d in groups if (out_dir / f"group{d:02d}.npz").exists()]
    if done and not args.skip_done:
        raise SystemExit(f"{out_dir} に保存済みの群がある: {done}。続きから回すなら --skip-done")
    todo = [d for d in groups if d not in done]
    n_iter = core.n_temperatures(T_INIT, T_CONV, COOLING_RATE) * args.K
    print(f"{stem}: 群 {todo}、1 群あたり {n_iter:,} 回")

    def report(res: tuple[int, float, tuple[int, int, int]]) -> None:
        d, sec, f = res
        print(f"  群 {d:2d}: {sec:8.1f} 秒  (f1, f2, f3) = {f}  1 回 {sec / n_iter * 1e6:.2f} µs", flush=True)

    if args.workers <= 1:
        for d in todo:
            report(solve_group(d, args.variant, args.K, args.N, args.seed, out_dir))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_group, d, args.variant, args.K, args.N, args.seed, out_dir) for d in todo]
            for fu in futs:
                report(fu.result())
    path = assemble_pool(out_dir, args.N)
    if path is not None:
        print(f"評価のプール: {path}")


if __name__ == "__main__":
    main()
