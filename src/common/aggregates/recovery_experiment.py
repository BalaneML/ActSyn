"""
D2-a 復元実験: セル集計のみからの群分布復元と σ_min(P) 理論の検証

目的 (実験デザイン D2-a, 計画ファイル参照):
    ATLAS の設定 ν(g) = Σ_d p(d|g) μ(d) を線形逆問題として忠実に再現する。
    「公表されるのはセル集計 ν(g) と構成行列 P だけ」という制約下で
    群分布 μ(d) (デモグラ群ごとの時刻別行動者率) を最小二乗で復元し、
    個票 ground truth μ*(d) との誤差を測る。
    復元誤差が理論予測 (誤差増幅 1/σ_min(P)) に従うかを層別スイープ横断で検証する。

3つのノイズ条件 (誤差の分解):
    clean     : ν を個票から厳密に計算。残る誤差 = 「群分布がセル間で不変」という
                ATLAS 仮定の違反 (モデルバイアス)。daytype をセルに使う層別は
                群の行動がセル(曜日)ごとに本当に違うため、ここが大きく出るはず。
    rounded   : ν を社基調の公表形式 (% 0.1単位 = 小数第4位丸め... 実際は0.001) に丸め。
                公表丸めの測定誤差の感度 (計画ファイル「公表集計の測定誤差」)。
    bootstrap : セル内の回答者を復元抽出でリサンプルして ν を再計算 (N_REPS回)。
                有限標本ノイズ。このノイズの増幅こそ 1/σ_min の理論が予言する対象。
                P は「センサス由来で正確」という ATLAS の設定を保ち、ν のみ揺らす。
                ※ ノイズの素の大きさは 1/√(セルn) でセル層別ごとに違うため、
                  bootstrap 増幅分と 1/σ_min の相関は交絡で消える (実測で確認済み)。
    synthetic : ν に固定強度 (NOISE_SIGMA) の iid ガウスノイズを注入 (N_REPS回)。
                ノイズ規模を全層別で統一した「統制された用量反応」条件。
                理論予測: 増幅分 ≈ (1/σ_min) × ノイズ強度 — 関数形検証はこちらで行う。

負の対照 (placebo):
    P の行 (セルの構成比) をシャッフルして ν(clean) と食い違わせる。
    復元が大きく劣化しなければ「P が情報を運んでいる」という機構主張は成立しない。

出力:
    data/processed/aggregates/<データセットstem>/recovery_experiment.csv
        cell_spec, demo_spec, G, D, sigma_min, error_amp, min_cell_n,
        noise (clean/rounded/bootstrap/placebo), rep, rmse, mae
    + コンソールに要約表と error_amp↔ノイズ増分の Spearman 相関

使い方:
    uv run python src/common/aggregates/recovery_experiment.py [--dataset ...] [--reps 30]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shakicho_sim import (
    CELL_SPECS, DEMO_SPECS, SLOT_COLS, N_SLOTS, OUT_DIR, DEFAULT_DATASET, derive_strata,
)

ROUND_DECIMALS = 3    # 行動者率の公表丸め: % 0.1単位 = 割合で小数3桁
NOISE_SIGMA    = 0.01 # synthetic 条件の注入ノイズ標準偏差 (行動者率スケール)


def one_hot_schedules(df: pd.DataFrame, n_act: int) -> np.ndarray:
    """個票 -> 特徴行列 X[N, n_act*96] (X[i, a*96+t]=1 iff s_it=a)。μ,ν は X の重み付き平均。"""
    S = df[SLOT_COLS].to_numpy(dtype=np.int64)
    N = len(df)
    X = np.zeros((N, n_act * N_SLOTS), dtype=np.float32)
    X[np.arange(N)[:, None], S * N_SLOTS + np.arange(N_SLOTS)[None, :]] = 1.0
    return X


def weighted_mean(X: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """行集合 idx の重み付き平均 (行動者率ベクトル)"""
    ww = w[idx]
    return ww @ X[idx] / ww.sum()


def recover(P: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """ν = P M の最小二乗解 (行動者率なので [0,1] にクリップ)"""
    M, *_ = np.linalg.lstsq(P, nu, rcond=None)
    return np.clip(M, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="D2-a: 集計のみからの群分布復元実験")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--reps", type=int, default=30, help="bootstrap 反復数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = derive_strata(pd.read_csv(args.dataset))
    n_act = int(df[SLOT_COLS].to_numpy().max()) + 1
    X = one_hot_schedules(df, n_act)
    w = df["TUFINLWGT"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    out_dir = OUT_DIR / args.dataset.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {args.dataset.name}  N={len(df)}  活動クラス数={n_act}  F={X.shape[1]}")

    records = []
    for demo_name, demo_cols in DEMO_SPECS.items():
        # デモグラ群のキーと ground truth μ*(d)
        demo_key = df[demo_cols].astype(str).agg("_".join, axis=1)
        demo_cats = sorted(demo_key.unique())
        demo_idx = {d: np.flatnonzero((demo_key == d).to_numpy()) for d in demo_cats}
        mu_star = np.stack([weighted_mean(X, w, demo_idx[d]) for d in demo_cats])  # D×F

        for cell_name, cell_cols in CELL_SPECS.items():
            if not set(cell_cols) <= set(df.columns):
                continue
            cell_key = df[cell_cols].astype(str).agg("_".join, axis=1)
            cell_cats = sorted(cell_key.unique())
            cell_idx = {g: np.flatnonzero((cell_key == g).to_numpy()) for g in cell_cats}
            G, D = len(cell_cats), len(demo_cats)

            # 構成行列 P (センサス由来=正確とみなし個票の全重みから構成) と診断
            P = np.zeros((G, D))
            for gi, g in enumerate(cell_cats):
                wg = np.zeros(D)
                for di, d in enumerate(demo_cats):
                    inter = np.intersect1d(cell_idx[g], demo_idx[d], assume_unique=True)
                    wg[di] = w[inter].sum()
                P[gi] = wg / wg.sum()
            sv = np.linalg.svd(P, compute_uv=False)
            sigma_min = float(sv[-1]) if G >= D else 0.0
            base = {"cell_spec": cell_name, "demo_spec": demo_name, "G": G, "D": D,
                    "sigma_min": sigma_min,
                    "error_amp": np.inf if sigma_min == 0 else 1.0 / sigma_min,
                    "min_cell_n": min(len(v) for v in cell_idx.values())}

            # セル集計 ν (clean) と各ノイズ条件での復元
            nu = np.stack([weighted_mean(X, w, cell_idx[g]) for g in cell_cats])  # G×F

            def record(noise: str, rep: int, M_hat: np.ndarray):
                err = M_hat - mu_star
                records.append({**base, "noise": noise, "rep": rep,
                                "rmse": float(np.sqrt((err ** 2).mean())),
                                "mae": float(np.abs(err).mean())})

            record("clean",   0, recover(P, nu))
            record("rounded", 0, recover(P, np.round(nu, ROUND_DECIMALS)))
            record("placebo", 0, recover(P[rng.permutation(G)], nu))
            for rep in range(args.reps):
                nu_b = np.stack([
                    weighted_mean(X, w, rng.choice(cell_idx[g], size=len(cell_idx[g]), replace=True))
                    for g in cell_cats])
                record("bootstrap", rep, recover(P, nu_b))
                nu_s = np.clip(nu + rng.normal(0.0, NOISE_SIGMA, size=nu.shape), 0.0, 1.0)
                record("synthetic", rep, recover(P, nu_s))

    res = pd.DataFrame(records)
    res.to_csv(out_dir / "recovery_experiment.csv", index=False)

    # --- 要約: 層別×ノイズ条件の平均RMSE ---
    summary = (res.groupby(["cell_spec", "demo_spec", "G", "D", "sigma_min", "error_amp",
                            "min_cell_n", "noise"], observed=True)["rmse"]
                  .mean().unstack("noise"))
    summary["boot_excess"]  = summary["bootstrap"] - summary["clean"]  # 標本ノイズの増幅分 (交絡あり)
    summary["synth_excess"] = summary["synthetic"] - summary["clean"]  # 統制ノイズの増幅分 (理論検証用)
    summary = summary.reset_index().sort_values(["demo_spec", "error_amp"])
    with pd.option_context("display.width", 240, "display.float_format", "{:.4f}".format):
        print("\n=== 復元RMSE (clean=モデルバイアス / bootstrap=+標本ノイズ / synthetic=+統制ノイズ / placebo=負の対照) ===")
        print(summary.to_string(index=False))

    # --- 理論検証: error_amp とノイズ増幅分の相関 (full column rank の層別のみ) ---
    ok = summary[np.isfinite(summary["error_amp"])]
    for demo_name in ok["demo_spec"].unique():
        sub = ok[ok["demo_spec"] == demo_name]
        rho_b = sub[["error_amp", "boot_excess"]].corr(method="spearman").iloc[0, 1]
        rho_s = sub[["error_amp", "synth_excess"]].corr(method="spearman").iloc[0, 1]
        print(f"\nSpearman(1/σ_min, ノイズ増幅分) [{demo_name}, n={len(sub)}層別]: "
              f"bootstrap={rho_b:.3f} (交絡あり) / synthetic={rho_s:.3f} (統制済み; 理論予測は正の強相関)")
    print("\n注意: bootstrap のノイズ規模は 1/√(セルn) でセル層別ごとに異なり σ_min と交絡する。")
    print("      理論の関数形 (増幅分 ∝ 1/σ_min) の検証は synthetic 条件で行うこと。")
    print(f"saved -> {out_dir / 'recovery_experiment.csv'}")


if __name__ == "__main__":
    main()
