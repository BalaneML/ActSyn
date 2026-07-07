"""
D2-d 複数統計量教師腕: 時刻別行動者率(1次周辺)に日次参加率を足すと何が変わるか

提案① (docs/proposals.md): 周辺マッチは条件外の遷移構造を侵食する (D2-b で bigram_jsd
悪化を実証済み)。日本の公表統計には時間帯編(時刻別行動者率=1次周辺)と独立に、
生活時間編の「行動者率(日次参加率)・行動者平均時間」というエピソード構造を拘束する
統計量がある。本実験は擬似日本 (ATUS + shakicho_sim のセル層別) で、
参加率を補助教師に加える効果を教師重み λ_part の用量反応として測る。

鍵となる性質:
    - 日次参加率は個人指示関数 1(いずれかのスロットで活動a) の期待値なので、
      セル集約が線形: ν_part(g) = Σ_d P[g,d]·part(d)。ATLAS の構成行列 P と
      σ_min 診断がそのまま適用できる (拡張されるのは特徴写像 φ の側)。
    - モデル側は z 条件付きスロット独立の閉形式 1 − Π_t(1 − p_t) で微分可能
      (model.participation_probs)。

腕 (セル層別ごと):
    part_w{λ}    : 周辺マッチ + λ·参加率マッチ。λ=0 は D2-b aggmatch の再現
    placebo_part : 参加率教師のセル行だけシャッフル (周辺教師は正しいまま)。
                   参加率「信号」が効くのか、単なる正則化なのかを分離する負の対照
    uncond       : Stage1 のみ (判別ゼロの床; 参加率指標にも床を与える)

評価 (個票 ground truth):
    rate_rmse / dev_rmse / bigram_jsd : D2-b と同一定義 (比較可能)
    part_rmse : 群別日次参加率 (D, n_act) の RMSE — 教師に使った統計量の群別復元
    dur_rmse  : 行動者平均スロット数の RMSE (参加率>1% の群×活動のみ) — 教師に
                入れていない継続時間構造の回復 (統計量ホールドアウト)

出力: data/processed/aggregates/<データセットstem>/multistat_experiment.csv
使い方: uv run python src/models/CVAE_Aggregate/multistat_experiment.py [--reuse-pretrain]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "aggregates"))
from shakicho_sim import CELL_SPECS, SLOT_COLS, OUT_DIR, DEFAULT_DATASET, derive_strata  # noqa: E402
from recovery_experiment import one_hot_schedules, weighted_mean  # noqa: E402
from model import (  # noqa: E402
    AggCVAE, train_elbo, aggregate_match, group_rates, group_rates_null,
    group_participation, participation_probs, sample_schedules,
    DEVICE, SEED, NUM_SLOTS, PRETRAIN_EPOCHS, EVAL_S, Z_DIM,
)

CKPT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "aggmatch_pretrain.pt"

# ===== 実験設定 =====
DEMO_COLS = ["gender", "age_class"]     # 群 d = 性×年齢7区分 (D=14; D2-a/b と同一)
EVAL_CELL_SPECS = ["state", "dow_x_emp_x_sex"]   # σ_min 良 / 中 の2点
PART_WEIGHTS = [0.0, 0.3, 1.0, 3.0]     # 用量反応: λ=0 が D2-b aggmatch の再現
PLACEBO_W = 1.0                          # placebo_part の教師重み
DUR_MASK_MIN = 0.01                      # dur_rmse を測る参加率の下限 (少人数の不安定回避)
BIGRAM_N = 2000


def bigram_dist(scheds: np.ndarray, n_act: int, w: np.ndarray | None = None) -> np.ndarray:
    """活動遷移 bigram 分布 (n_act*n_act,)。w があれば個人重み付き。(D2-b と同一定義)"""
    a, b = scheds[:, :-1].ravel(), scheds[:, 1:].ravel()
    ww = np.ones(len(scheds)) if w is None else w
    ww = np.repeat(ww, NUM_SLOTS - 1)
    counts = np.zeros(n_act * n_act)
    np.add.at(counts, a * n_act + b, ww)
    return counts / counts.sum()


def part_dur_stats(counts: np.ndarray, w: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行集合 idx の (日次参加率 (n_act,), 行動者平均スロット数 (n_act,))。
    counts: 個人×活動のスロット数 (N, n_act)。dur は参加者ゼロの活動で NaN。"""
    ww, c = w[idx], counts[idx]
    doer = (c > 0).astype(float)
    part = ww @ doer / ww.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        dur = (ww @ c) / (ww @ doer)
    return part, dur


def main():
    ap = argparse.ArgumentParser(description="D2-d: 参加率を補助教師に足した集計マッチの用量反応")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--reuse-pretrain", action="store_true",
                    help="既存の Stage1 チェックポイント (aggmatch_pretrain.pt) を再利用")
    ap.add_argument("--cells", nargs="+", default=EVAL_CELL_SPECS, choices=list(CELL_SPECS))
    ap.add_argument("--steps", type=int, default=None, help="マッチ反復数の上書き (スモークテスト用)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    match_kw = {} if args.steps is None else {"steps": args.steps}

    df = derive_strata(pd.read_csv(args.dataset))
    n_act = int(df[SLOT_COLS].to_numpy().max()) + 1
    X = one_hot_schedules(df, n_act)
    w = df["TUFINLWGT"].to_numpy(dtype=np.float64)
    S_np = df[SLOT_COLS].to_numpy(dtype=np.int64)
    S_t = torch.as_tensor(S_np)
    counts = (S_np[:, :, None] == np.arange(n_act)[None, None, :]).sum(axis=1)  # (N, n_act)
    out_dir = OUT_DIR / args.dataset.stem
    print(f"dataset: {args.dataset.name}  N={len(df)}  n_act={n_act}  device={DEVICE}")

    # --- 群定義と ground truth (評価専用) ---
    demo_key = df[DEMO_COLS].astype(str).agg("_".join, axis=1)
    demo_cats = sorted(demo_key.unique())
    D = len(demo_cats)
    demo_of = demo_key.map({d: i for i, d in enumerate(demo_cats)}).to_numpy()  # type: ignore
    demo_idx = {i: np.flatnonzero(demo_of == i) for i in range(D)}
    mu_star = np.stack([weighted_mean(X, w, demo_idx[i]) for i in range(D)])
    bigram_star = [bigram_dist(S_np[demo_idx[i]], n_act, w[demo_idx[i]]) for i in range(D)]
    pd_stats = [part_dur_stats(counts, w, demo_idx[i]) for i in range(D)]
    part_star = np.stack([p for p, _ in pd_stats])                     # (D, n_act)
    dur_star = np.stack([d_ for _, d_ in pd_stats])                    # (D, n_act) NaN あり
    dur_mask = (part_star > DUR_MASK_MIN) & ~np.isnan(dur_star)
    pi = np.array([w[demo_idx[i]].sum() for i in range(D)])
    pi /= pi.sum()
    dev_star = mu_star - pi @ mu_star

    def metrics(mu_hat: np.ndarray, part_hat: np.ndarray, dur_hat: np.ndarray,
                bigram_jsd: float) -> dict:
        dev_hat = mu_hat - pi @ mu_hat
        return {
            "rate_rmse": float(np.sqrt(((mu_hat - mu_star) ** 2).mean())),
            "dev_rmse": float(np.sqrt(((dev_hat - dev_star) ** 2).mean())),
            "bigram_jsd": bigram_jsd,
            "part_rmse": float(np.sqrt(((part_hat - part_star) ** 2).mean())),
            "dur_rmse": float(np.sqrt(((dur_hat - dur_star)[dur_mask] ** 2).mean())),
        }

    def evaluate(model: AggCVAE, cond: bool = True) -> dict:
        if cond:
            mu_hat = group_rates(model, D)
            part_hat, dur_hat = group_participation(model, D)
            jsd = float(np.mean([
                jensenshannon(bigram_dist(sample_schedules(model, i, BIGRAM_N), n_act),
                              bigram_star[i], base=2) ** 2 for i in range(D)]))
        else:   # uncond: 全群に null トークンの分布 (判別ゼロの床)
            mu_hat = np.repeat(group_rates_null(model, D), D, axis=0)
            with torch.no_grad():
                z = torch.randn(EVAL_S, Z_DIM, device=DEVICE)
                di = torch.full((EVAL_S,), D, dtype=torch.long, device=DEVICE)
                probs = model.decode_probs(z, di)
                part0 = participation_probs(probs).mean(dim=0)
                dur0 = (probs.sum(dim=-2).mean(dim=0) / part0.clamp(min=1e-6))
            part_hat = np.repeat(part0.cpu().numpy()[None], D, axis=0)
            dur_hat = np.repeat(dur0.cpu().numpy()[None], D, axis=0)
            sched0 = sample_schedules(model, D, BIGRAM_N)
            jsd = float(np.mean([
                jensenshannon(bigram_dist(sched0, n_act), bigram_star[i], base=2) ** 2
                for i in range(D)]))
        return metrics(mu_hat, part_hat, dur_hat, jsd)

    # --- Stage 1: 無条件 pretrain (D2-b と共通チェックポイント) ---
    null_t = torch.full((len(df),), D, dtype=torch.long)
    pre = AggCVAE(n_act, D).to(DEVICE)
    if args.reuse_pretrain and CKPT_PATH.exists():
        pre.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Stage1: 再利用 {CKPT_PATH}")
    else:
        print("Stage1: 無条件 pretrain ...")
        train_elbo(pre, S_t, null_t, w, PRETRAIN_EPOCHS, "pretrain")
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pre.state_dict(), CKPT_PATH)

    records = [{"cell_spec": "(none)", "G": 0, "sigma_min": np.nan, "arm": "uncond",
                "part_w": np.nan, **evaluate(pre, cond=False)}]

    # --- セル層別ごとに 用量反応 + placebo ---
    for cell_name in args.cells:
        cell_cols = CELL_SPECS[cell_name]
        cell_key = df[cell_cols].astype(str).agg("_".join, axis=1)
        cell_cats = sorted(cell_key.unique())
        cell_idx = {g: np.flatnonzero((cell_key == g).to_numpy()) for g in cell_cats}
        G = len(cell_cats)
        P = np.zeros((G, D))
        for gi, g in enumerate(cell_cats):
            for di in range(D):
                inter = np.intersect1d(cell_idx[g], demo_idx[di], assume_unique=True)
                P[gi, di] = w[inter].sum()
            P[gi] /= P[gi].sum()
        sigma_min = float(np.linalg.svd(P, compute_uv=False)[-1]) if G >= D else 0.0
        nu = np.stack([weighted_mean(X, w, cell_idx[g]) for g in cell_cats])
        part_nu = np.stack([part_dur_stats(counts, w, cell_idx[g])[0] for g in cell_cats])
        print(f"\n=== cell_spec={cell_name}  G={G} D={D}  σ_min={sigma_min:.4f} ===")

        def fork() -> AggCVAE:
            m = AggCVAE(n_act, D).to(DEVICE)
            m.load_state_dict(pre.state_dict())
            return m

        for lam in PART_WEIGHTS:
            m = fork()
            aggregate_match(m, P, nu, n_act, f"{cell_name}/part_w{lam:g}",
                            part_nu=part_nu, part_w=lam, **match_kw)
            records.append({"cell_spec": cell_name, "G": G, "sigma_min": sigma_min,
                            "arm": f"part_w{lam:g}", "part_w": lam, **evaluate(m)})
        # placebo: 参加率教師のセル行だけシャッフル (周辺教師は正しいまま)
        m = fork()
        aggregate_match(m, P, nu, n_act, f"{cell_name}/placebo_part",
                        part_nu=part_nu[rng.permutation(G)], part_w=PLACEBO_W, **match_kw)
        records.append({"cell_spec": cell_name, "G": G, "sigma_min": sigma_min,
                        "arm": "placebo_part", "part_w": PLACEBO_W, **evaluate(m)})

    res = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "multistat_experiment.csv"
    res.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n=== D2-d 結果 (part_w0 = D2-b aggmatch 再現 / dur_rmse = 教師外の継続時間構造) ===")
        print(res.to_string(index=False))
    print("\n判定: 参加率教師が本物の信号なら part_w>0 で part_rmse/dur_rmse/bigram_jsd が改善し、"
          "placebo_part では改善が消えるはず。dev_rmse が悪化しないかも併せて見る。")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
