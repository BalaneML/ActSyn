"""
trajectory_pools.py
===================
DDPM_Minimal の Stage 1 の途中の ckpt（model.py --save-every）ごとに小さなプールを作り、ATUS の再現の度合いと、
逆過程の最後の x が 1 つの活動に決まっているかを並べる（Stage2_Common/docs/Stage2_common_results.md §6・§7 の案 A と B）

確かめる 2 つの候補（§6）:
    (a) 逆過程が 1 つの活動に収束しない → どの ckpt でも、最後の x が one-hot に近くない
    (b) 学習の途中で生成の総量が漂う → val がほぼ同じ ckpt どうしで、活動の割合が大きく動く

測るもの（ckpt ごと。群あたり n_per_group 本、生成の種は評価のプールと同じ sel.DEFAULT_POOL_SEED = 12345）:
    1. val の ε の二乗誤差（ckpt の config に記録した値）
    2. 12 活動の割合（ATUS の群構成で重み付け）と、ATUS 平日の実データ（調査ウェイト）からの総変動距離 tv_atus
    3. 採点と同じ sel.guardrails のうち GUARD_KEYS の 6 つ（切り替え回数・15 分で終わるエピソードの割合・
       bigram_jsd・睡眠・身の回りの用事を持つ日誌の割合・最長の睡眠ブロック・群の描き分け）
    4. 最後の x の形（final_x_stats）: one-hot に近いスロット・最大値で同点のスロット・2 番目に大きい値の平均
val 最良の ckpt だけは、生成の種を 1 つずらしたプールも作る（同じ ckpt での標本の揺れを見るため）

```mermaid
flowchart LR
    CL["ckpt_list(stem, ckpt_dir)<br/>{stem}_ep*.pt と {stem}.pt"] --> MC["measure_ckpt(path, real, diffusion, n_per_group, seed, device)"]
    MC --> PF["pool_and_final_x(model, diffusion, n_per_group, seed)"]
    PF --> RC["dm.reverse_chain(model, diffusion, cond, gen)<br/>x (B, 12, 96)"]
    RC --> FX["final_x_stats(x)"]
    RC --> AM["x.argmax(dim=1)<br/>pool (28, n_per_group, 96)"]
    AM --> PM["pool_metrics(pool, real)<br/>activity_shares・tv_atus・sel.guardrails"]
    FX --> CSV["rows → stage1_traj_{stem}.csv"]
    PM --> CSV
```

使い方:
    .venv/bin/python src/models/DDPM_Minimal/trajectory_pools.py --stem ddpm_minimal_ch16_wd0.01_traj --device cuda
    .venv/bin/python src/models/DDPM_Minimal/trajectory_pools.py --stem ddpm_minimal_ch32_wd0.01 --device cuda   # 幅は ckpt から読む
    .venv/bin/python src/models/DDPM_Minimal/trajectory_pools.py --smoke   # 手元の CPU で形だけ確かめる（何も残さない）

出力:
    data/processed/aggregates/stage1_traj_{stem}.csv   1 行 = 1 つの ckpt と生成の種
"""
import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（同じファイルが既に載っていればそれを返す）"""
    cached = sys.modules.get(name)
    cached_file = getattr(cached, "__file__", None)
    if cached is not None and cached_file is not None and Path(cached_file).resolve() == path.resolve():
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dm: Any = _load("ddpm_minimal_model", HERE / "model.py")
sel: Any = _load("ddpm_minimal_stage2_select", SIMPLE_DIR / "stage2_select.py")   # 採点と同じ guardrails
sm: Any = dm.sm

CKPT_DIR: Path = dm.MODEL_SAVE_PATH.parent
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"
N_PER_GROUP = 64                         # DDPM_Aggregate_Simple の H4（Stage1_rare_activity_diagnosis.md）と同じ
POOL_SEED: int = sel.DEFAULT_POOL_SEED   # 12345（評価のプールと同じ）
GUARD_KEYS = ("switch_mean", "single_slot_ratio", "bigram_jsd", "sleep_holder_rate", "main_sleep_mean_min",
              "separation_ratio")
ONE_HOT_MAX = 0.9                        # one-hot に近いスロット: 最大が 0.9 超で、
ONE_HOT_REST = 0.1                       # 0.1 を超える活動がその 1 つだけ

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


class RealData(NamedTuple):
    """ATUS 平日の実データ（採点の score_pool と同じ sm.load_data）"""
    sched: IntArr      # (N, 96)
    d: IntArr          # (N,) 群
    w: FloatArr        # (N,) 調査ウェイト（TUFINLWGT）
    shares: FloatArr   # (12,) 調査ウェイトで重み付けした活動の割合


def activity_shares(sched: IntArr, w: FloatArr) -> FloatArr:
    """1 人ごとの活動の割合（96 スロットのうち何割か）を重み w で平均する

    Args:
        sched: 活動, dtype=int64, (N, 96)
        w: 1 人ごとの重み, dtype=float64, (N,)

    Returns:
        活動の割合, dtype=float64, (12,)。和は 1
    """
    frac = np.stack([(sched == a).mean(axis=1) for a in range(sm.NUM_ACT)], axis=1)   # (N, 12)
    wn = w / w.sum()
    return (wn[:, None] * frac).sum(axis=0)


def load_real() -> RealData:
    """ATUS 平日の実データと、その活動の割合"""
    cond_idx, sched, w, _ = sm.load_data()
    sched_arr = np.asarray(sched, dtype=np.int64)
    w_arr = np.asarray(w, dtype=np.float64)
    d_arr = np.asarray(sm.cond_to_d(cond_idx), dtype=np.int64)
    return RealData(sched_arr, d_arr, w_arr, activity_shares(sched_arr, w_arr))


def final_x_stats(x: torch.Tensor) -> dict[str, float]:
    """逆過程の最後の x が、スロットごとに 1 つの活動に決まっているか

    Args:
        x: 最後の x（clamp された x0 の予測）, dtype=float32, (B, 12, 96)

    Returns:
        {"clean_onehot": 最大が ONE_HOT_MAX を超え、ONE_HOT_REST を超える活動がそれだけのスロットの割合,
         "tie_rate": 2 つ以上の活動が最大値で並ぶスロットの割合（argmax は番号の小さい活動を返す）,
         "second_mean": 2 番目に大きい値の平均}
    """
    top2 = x.topk(2, dim=1).values                                            # (B, 2, 96)
    mx = top2[:, 0]                                                           # (B, 96)
    clean = (mx > ONE_HOT_MAX) & ((x > ONE_HOT_REST).sum(dim=1) == 1)
    tie = (x == mx[:, None]).sum(dim=1) >= 2
    return {"clean_onehot": float(clean.float().mean()), "tie_rate": float(tie.float().mean()),
            "second_mean": float(top2[:, 1].mean())}


@torch.no_grad()
def pool_and_final_x(model: Any, diffusion: Any, n_per_group: int, seed: int,
                     batch: int = dm.POOL_BATCH) -> tuple[IntArr, dict[str, float]]:
    """dm.group_pool と同じ手順・同じ乱数でプールを作り、最後の x の形も測る

    Args:
        model: DDPMMinimal
        diffusion: base.Diffusion
        n_per_group: 群あたりの本数
        seed: 生成の乱数の種
        batch: 1 回に生成する本数, default=dm.POOL_BATCH=8192

    Returns:
        (プール, dtype=int64, (28, n_per_group, 96), final_x_stats を本数で重み付けした平均)
    """
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    flat = torch.as_tensor(sm.cond_grid(), dtype=torch.long).repeat_interleave(n_per_group, dim=0)
    outs: list[torch.Tensor] = []
    sums: dict[str, float] = {}
    for i in range(0, flat.size(0), batch):
        cond = flat[i:i + batch].to(dev)
        x = dm.reverse_chain(model, diffusion, cond, gen)
        for key, val in final_x_stats(x).items():
            sums[key] = sums.get(key, 0.0) + val * cond.size(0)
        outs.append(x.argmax(dim=1).cpu())
    pool = torch.cat(outs).numpy().astype(np.int64).reshape(sm.D_GROUPS, n_per_group, sm.NUM_SLOTS)
    return pool, {key: val / flat.size(0) for key, val in sums.items()}


def pool_metrics(pool: IntArr, real: RealData) -> dict[str, float]:
    """プールの活動の割合（ATUS の群構成で重み付け）・実データからの総変動距離・sel.guardrails の一部

    Args:
        pool: 群別サンプルプール, dtype=int64, (28, n_per_group, 96)
        real: ATUS 平日の実データ

    Returns:
        {"tv_atus", "share_{活動名}" × 12, GUARD_KEYS の 6 つ}
    """
    n_per_group = pool.shape[1]
    gen = pool.reshape(-1, sm.NUM_SLOTS)
    gen_d = np.repeat(np.arange(sm.D_GROUPS), n_per_group)
    w_gen = np.asarray(sel.im.group_reweight(gen_d, real.w, real.d, sm.D_GROUPS), dtype=np.float64)
    shares = activity_shares(gen, w_gen)
    guard = sel.guardrails(gen, gen_d, real.sched, real.d, real.w, seed=POOL_SEED)
    out: dict[str, float] = {"tv_atus": float(0.5 * np.abs(shares - real.shares).sum())}
    out.update({f"share_{name}": float(v) for name, v in zip(sm.ACT_NAMES, shares)})
    out.update({key: float(guard[key]) for key in GUARD_KEYS})
    return out


def ckpt_list(stem: str, ckpt_dir: Path = CKPT_DIR) -> list[Path]:
    """途中の ckpt {stem}_ep*.pt（epoch の順）と、あれば val 最良の {stem}.pt"""
    paths = sorted(ckpt_dir.glob(f"{stem}_ep*.pt"))
    best = ckpt_dir / f"{stem}.pt"
    return paths + ([best] if best.exists() else [])


def measure_ckpt(path: Path, real: RealData, diffusion: Any, n_per_group: int, seed: int,
                 device: str) -> dict[str, Any]:
    """1 つの ckpt から小さなプールを作って測る

    Note:
        ckpt の config の幅に合わせて dm.set_base_ch を呼ぶ（呼んだ後は dm の関数がこの幅を使う）

    Args:
        path: ckpt（途中の ckpt は config に epoch・val、val 最良の ckpt は best_epoch・best_val を持つ）
        real: ATUS 平日の実データ
        diffusion: base.Diffusion（device と同じ計算場所）
        n_per_group: 群あたりの本数
        seed: 生成の乱数の種
        device: 計算場所

    Returns:
        1 行分の記録（ckpt・base_ch・epoch・is_best・lr・val・pool_seed・n_per_group・pool_metrics・final_x_stats・sec）
    """
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    is_best = "best_epoch" in config
    dm.set_base_ch(int(config["base_ch"]))
    model = dm.load_model(path, device=device)
    t0 = time.perf_counter()
    pool, fx = pool_and_final_x(model, diffusion, n_per_group, seed)
    row: dict[str, Any] = {"ckpt": path.name, "base_ch": int(config["base_ch"]),
                           "epoch": int(config["best_epoch"] if is_best else config["epoch"]),
                           "is_best": is_best, "lr": float(config.get("lr", dm.LR)),
                           "val": float(config["best_val"] if is_best else config["val"]),
                           "pool_seed": seed, "n_per_group": n_per_group}
    row.update(pool_metrics(pool, real))
    row.update(fx)
    row["sec"] = time.perf_counter() - t0
    return row


def run_stem(stem: str, real: RealData, diffusion: Any, n_per_group: int, seed: int, device: str,
             ckpt_dir: Path = CKPT_DIR) -> pd.DataFrame:
    """stem の全 ckpt を測る。val 最良の ckpt は生成の種 seed と seed + 1 の 2 つで測る

    Raises:
        FileNotFoundError: stem の ckpt が 1 つも無いとき
    """
    paths = ckpt_list(stem, ckpt_dir)
    if not paths:
        raise FileNotFoundError(f"{ckpt_dir}/{stem}*.pt が無い")
    rows: list[dict[str, Any]] = []
    for path in paths:
        seeds = (seed, seed + 1) if path.name == f"{stem}.pt" else (seed,)
        for s in seeds:
            row = {"stem": stem, **measure_ckpt(path, real, diffusion, n_per_group, s, device)}
            rows.append(row)
            print(f"{row['ckpt']:48s} seed {s}  epoch {row['epoch']:4d}  val {row['val']:.5f}  "
                  f"tv_atus {row['tv_atus']:.3f}  sleep {row['share_SLEEP_PERSONAL']:.3f}  "
                  f"switch {row['switch_mean']:.1f}  clean {row['clean_onehot']:.3f}  tie {row['tie_rate']:.3f}  "
                  f"({row['sec']:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def smoke() -> None:
    """2 epoch を途中の ckpt つきで学習し、群あたり 2 本で 1 つの ckpt を測る（手元の CPU、何も残さない）"""
    real = load_real()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ddpm_minimal_smoke.pt"
        dm.train(epochs=2, seed=0, save_path=path, device="cpu", save_every=1)
        paths = ckpt_list(path.stem, Path(tmp))
        assert [p.name for p in paths] == ["ddpm_minimal_smoke_ep0001.pt", "ddpm_minimal_smoke_ep0002.pt",
                                           "ddpm_minimal_smoke.pt"], paths
        row = measure_ckpt(paths[1], real, dm.base.Diffusion("cpu"), 2, POOL_SEED, "cpu")
    assert row["epoch"] == 2 and abs(sum(row[f"share_{n}"] for n in sm.ACT_NAMES) - 1.0) < 1e-9
    assert all(np.isfinite(row[k]) for k in ("tv_atus", "clean_onehot", "tie_rate", "second_mean", *GUARD_KEYS))
    summary = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
    print(f"smoke: OK {summary}")


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description="DDPM_Minimal の Stage 1 の途中の ckpt ごとのプールの指標")
    ap.add_argument("--stem", action="append", default=[],
                    help="ckpt の stem（繰り返し可）。例 ddpm_minimal_ch16_wd0.01_traj")
    ap.add_argument("--n-per-group", type=int, default=N_PER_GROUP)
    ap.add_argument("--seed", type=int, default=POOL_SEED, help="生成の乱数の種（val 最良の ckpt は seed + 1 でも測る）")
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default=dm.DEVICE)
    ap.add_argument("--smoke", action="store_true", help="手元の CPU で形だけ確かめる（何も残さない）")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.stem:
        ap.error("--stem が要る")
    real = load_real()
    diffusion = dm.base.Diffusion(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in args.stem:
        table = run_stem(stem, real, diffusion, args.n_per_group, args.seed, args.device)
        out = OUT_DIR / f"stage1_traj_{stem}.csv"
        table.to_csv(out, index=False)
        print(f"[traj] 書いた: {out}", flush=True)


if __name__ == "__main__":
    main()
