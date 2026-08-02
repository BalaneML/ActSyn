"""
実転移 (Route B): ATUS(米)条件付きDDPM pretrain → 社会生活基本調査(実公表集計，日)への
                  指数傾け (exponential tilting) による集計マッチ転移

CVAE_Aggregate/japan_match_experiment.py との差分（★が本スクリプトの新規部分）:
    - 生成器  : ★CVAE (分解デコーダ) → DDPM (1D-UNet + self-attention)
                CVAE は p(x|z)=Π_t p(x_t|z) と分解するため個票が断片化する
                (ATUS平日・共通12分類で 平均切替 実12.64 / 生成48.1 = 3.8倍)。
                DDPM は系列構造を保つ。代償は μ̂(d) が閉形式で微分可能でないこと
    - Stage2  : ★μ̂ への勾配降下 → 凍結モデルのサンプルプールへの指数傾け
                拡散の逆過程を勾配が通らないため、生成器は一切更新しない。
                代わりに「どの日記をどれだけ重く数えるか」だけを最適化する
    - アンカー: ★関数空間アンカー (重み ANCHOR_W の調整項) → 提案分布への KL そのもの
                最大エントロピー原理により、制約を満たす傾けのうち事前学習分布に
                KL 最小のものが一意に選ばれる。アンカーがハイパーパラメータでなく
                最適化問題の定義に組み込まれる
    - 最適化  : ★非凸の勾配降下 → 凸問題 (双対は滑らかな凹関数; L-BFGS で大域最適)

定式化:
    提案分布 p_d = 凍結 DDPM の群 d 条件付き分布。プールは群あたり M 本のサンプル。
    特徴量 φ(x)_{c,t} = 1(x_t = c)  (時刻別行動者率がそのまま E_q[φ])

        min_q  Σ_d KL(q_d ‖ p_d)
        s.t.   Σ_e π(e|g,a)·E_{q_(g,a,e)}[φ] = ν_ga   (公表 性×年齢, 14行)
               Σ_a π(a|g,e)·E_{q_(g,a,e)}[φ] = ν_ge   (公表 性×就業, 4行)

    ラグランジュ双対より最適解は指数傾けの形になる:

        q_d(x) ∝ p_d(x)·exp⟨θ_d, φ(x)⟩,
        θ_(g,a,e) = π(e|g,a)·λ_ga + π(a|g,e)·η_ge

    ★自由変数は 28群ぶんでなく「公表された18行ぶん」の乗数 (λ, η) だけ。
      未公表セル・比較不能活動 OTHER_X は λ,η のマスクで自動的に落ちる。
    双対関数 (最大化):
        g(λ,η) = −Σ_d log E_{p_d}[exp⟨θ_d,φ⟩] + ⟨λ,ν_ga⟩ + ⟨η,ν_ge⟩
    勾配は (教師 − 再構成周辺) なので、最大化がそのまま残差ゼロへ向かう。

    ★RIDGE > 0 は必須（省略可能な調整項ではない）:
      公表2表には構造的な線形従属がある。各 (g, 活動c, 時刻t) について
          Σ_a π(a|g)·ν_ga = Σ_{a,e} π(a,e|g)·μ = Σ_e π(e|g)·ν_ge
      が恒等的に成り立つため、18行×12活動×96スロットの制約のうち
      2×12×96 = 2,304 本が冗長。したがって ridge=0 では双対に平坦方向が残り、
      λ,η は一意に決まらない（q 自体は変わらないまま λ が発散し、
      数値的には softmax が飽和して ESS だけが枯れる）。
      ridge>0 は (a) 双対を狭義凸にして最小ノルム乗数を一意に選び、
      (b) 有限プール M でモーメントを厳密に追いすぎて θ が発散するのを防ぐ。
      CVAE 版の ANCHOR_W に相当する唯一のつまみだが、役割は正則化でなく識別性の回復。

教師/評価の分離: CVAE_Aggregate 版と同一 (統計量ホールドアウト)
    教師 = margin_ga (14行) + margin_ge (4行)
    評価 = 群別行動者率 (28群) — 学習には一切使わない
    ★load_stula_targets / eval_against / nan_renorm_pop_weights は
      CVAE_Aggregate/japan_match_experiment.py から import して唯一の出所を保つ

比較条件:
    zero-shot : 傾けなし (一様重み) — 米国DDPMの構造持ち込みだけで日本集計にどこまで合うか
    tilted    : 指数傾け (本命)
    shuffled  : 群対応をシャッフルしたプールに傾ける (負の対照)
    baseline  : 全群=日本の人口平均 (モデル不要の床)

使い方:
    # Stage1 の学習は model.py 側:
    uv run python src/models/DDPM_Aggregate/model.py

    # Stage2 (プール生成 → 指数傾け → 評価)
    uv run python src/models/DDPM_Aggregate/japan_match_experiment.py

    フラグ:
        --pool-size N   : 群あたりのプール本数 (既定 2000)
        --reuse-pool    : 保存済みプールを再利用 (Stage2 だけ回し直すとき)
        --ridge R       : L2 緩和の強さ
        --ridge-sweep   : 複数の RIDGE で ESS と残差のトレードオフを掃引
        --ddim-steps N  : プール生成の DDIM ステップ数

出力:
    data/processed/aggregates/ddpm_japan_match_experiment.csv   バリアント別の評価指標
    data/processed/aggregates/ddpm_japan_match_tilting.csv      傾けの診断 (ESS/残差)
    outputs/generated/ddpm_japan_pool.npz                       群別サンプルプール (再利用用)
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --- モジュールの明示ロード ---------------------------------------------------
# CVAE_Aggregate と DDPM_Aggregate は同名ファイル (model.py / japan_match_experiment.py)
# を持つフラット構成なので、`import` は sys.path の順序次第で静かに別モデルを掴む。
# sys.modules に一意な名前で直接載せて、順序への依存を断つ。
CVAE_AGG = REPO_ROOT / "src" / "models" / "CVAE_Aggregate"
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# CVAE 版 experiment は `from model import ...` を持つので、先に "model" を
# CVAE_Aggregate/model.py で確定させる（DDPM_Aggregate/model.py が掴まれるのを防ぐ）
_load("model", CVAE_AGG / "model.py")
jm = _load("cvae_agg_japan_match", CVAE_AGG / "japan_match_experiment.py")  # 教師読込・評価の唯一の出所
ddpm = _load("ddpm_agg_model", HERE / "model.py")

from crosswalk_atus_stula import Common, EXCLUDED  # noqa: E402
# ---------------------------------------------------------------------------

PROC_DIR  = REPO_ROOT / "data" / "processed"
OUT_CSV   = PROC_DIR / "aggregates" / "ddpm_japan_match_experiment.csv"
DIAG_CSV  = PROC_DIR / "aggregates" / "ddpm_japan_match_tilting.csv"
# ★系列構造の分布距離（教師に入っていない量での検証）。ノイズ床を併記する
SEQ_CSV   = PROC_DIR / "aggregates" / "ddpm_japan_match_sequence.csv"
POOL_PATH = REPO_ROOT / "outputs" / "generated" / "ddpm_japan_pool.npz"

N_G, N_A, N_E = ddpm.N_G, ddpm.N_A, ddpm.N_E
D_GROUPS = ddpm.D_GROUPS
NUM_ACT  = ddpm.NUM_ACT
NUM_SLOTS = ddpm.NUM_SLOTS

POOL_SIZE  = 2000     # 群あたりの提案サンプル本数
# λ,η への L2。★M=2000・DDIM プールでの掃引で内点最適だった値:
#   ridge   0.3     0.1     0.03    0.01    0.003   1e-5    0
#   dev_rmse 0.0293 0.0272  0.0239  0.0208  0.0219  0.0539  0.2063
#   ESS中央値 33.4%  13.8%   8.5%    5.8%    4.2%    1.1%    0.05%
# 教師への残差は ridge を下げるほど単調に減る (0.0111→0.0043) のに、28群ホールドアウト
# 評価は 0.01 より下で悪化する = 教師周辺への過適合。ESS がその手前で警告を出す
RIDGE      = 1e-2
LBFGS_ITER = 200
RESTARTS   = 8        # L-BFGS の履歴を張り直す回数（目的関数が動かなくなれば早期終了）
SEED       = 42

BREAK_CANDIDATES    = jm.BREAK_CANDIDATES
PRESERVE_CANDIDATES = jm.PRESERVE_CANDIDATES

DEVICE = ddpm.DEVICE


# ============================================================
# 1. サンプルプール（提案分布）
# ============================================================
def _pool_meta(sampler: str, ddim_steps: int) -> dict[str, str]:
    """プールの生成条件。npz に一緒に保存し、再利用時に照合する。

    ★ サンプラ種別が記録されていないと、そのプールが ancestral 由来か DDIM 由来かを
      後から判別できない。断片化は両者で大きく違う（平均切替 12.87 vs 17.70）ので、
      取り違えると数値の解釈を丸ごと誤る。
    """
    ckpt = ddpm.MODEL_SAVE_PATH
    return {
        "sampler": sampler,
        "ddim_steps": str(ddim_steps if sampler == "ddim" else ""),
        "eta": str(ddpm.DDIM_ETA if sampler == "ddim" else ""),
        "guidance_scale": str(ddpm.GUIDANCE_SCALE),
        "checkpoint": ckpt.name,
        "checkpoint_mtime": str(int(ckpt.stat().st_mtime)) if ckpt.exists() else "",
    }


def build_pool(n_per_group: int, sampler: str, ddim_steps: int, reuse: bool) -> np.ndarray:
    """群別サンプルプール (D, M, 96) を生成またはロードする。"""
    want = _pool_meta(sampler, ddim_steps)
    if reuse and POOL_PATH.exists():
        z = np.load(POOL_PATH)
        pool = z["pool"]
        got = {k: str(z[k]) for k in want if k in z.files}
        missing = [k for k in want if k not in z.files]
        diff = {k: (got[k], want[k]) for k in got if got[k] != want[k]}
        if missing or diff:
            # ★黙って使わない。合わないプールを再利用すると、結果が別条件のものになる
            print("保存プールのメタデータが要求と不一致 → 再生成")
            if missing:
                print(f"  記録なし: {missing}（メタデータ導入前に作られたプール）")
            for k, (g, w) in diff.items():
                print(f"  {k}: 保存={g!r} 要求={w!r}")
        elif pool.shape[0] == D_GROUPS and pool.shape[1] >= n_per_group:
            print(f"プール再利用: {POOL_PATH} {pool.shape} -> 先頭 {n_per_group} 本を使用")
            print(f"  メタデータ: {want}")
            return pool[:, :n_per_group]
        else:
            print(f"保存プールの形 {pool.shape} が要求 (D={D_GROUPS}, M={n_per_group}) と不一致 → 再生成")

    model = ddpm.load_pretrained()
    print(f"Stage1 読み込み: {ddpm.MODEL_SAVE_PATH.name} (EMA重み)")
    print(f"プール生成: D={D_GROUPS} × M={n_per_group} (sampler={sampler}) ...")
    pool = ddpm.group_pool(model, n_per_group, sampler=sampler, ddim_steps=ddim_steps)
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"pool": pool, **want}
    np.savez_compressed(POOL_PATH, **arrays)
    print(f"saved pool -> {POOL_PATH}  メタデータ: {want}")
    return pool


# ============================================================
# 2. Stage 2: 指数傾け ★
# ============================================================
def exponential_tilt(pool: np.ndarray,
                     tgt: dict,
                     mask_c: np.ndarray,
                     ridge: float = RIDGE,
                     iters: int = LBFGS_ITER,
                     tag: str = "tilted") -> tuple[np.ndarray, dict]:
    """★公表周辺へ指数傾けする。生成器は一切更新しない。

    args:
        pool  : (D, M, 96) 凍結 DDPM の群別サンプルプール = 提案分布 p_d
        tgt   : load_stula_targets の返り値 (margin_ga / margin_ge / pi_e / pop)
        mask_c: 比較不能活動 OTHER_X を落とすマスク (n_act,)
        ridge : λ,η への L2 罰則。0 なら制約を厳密に満たしにいく
    returns:
        (重み (D, M) 正規化済み, 診断 dict)
    """
    D, M, T = pool.shape
    # ★凸問題なので精度優先で CPU float64 に固定する。MPS は float64 非対応で、
    #   float32 だと L-BFGS が残差 1e-4 前後で頭打ちになる（率スケールで 0.01 ポイント）。
    #   規模は D×M×T ≈ 5M 要素と小さく、CPU でも数十秒で解ける
    dev, dt = "cpu", torch.float64

    mga = torch.as_tensor(tgt["margin_ga"], dtype=dt, device=dev)  # (2,7,12,96)
    mge = torch.as_tensor(tgt["margin_ge"], dtype=dt, device=dev)  # (2,2,12,96)
    pi_e = torch.as_tensor(tgt["pi_e"], dtype=dt, device=dev)      # π(e|g,a) (2,7,2)
    pop  = torch.as_tensor(tgt["pop"],  dtype=dt, device=dev)      # (2,7,2)
    pi_a = pop / pop.sum(dim=1, keepdim=True)                      # π(a|g,e) (2,7,2)
    mc   = torch.as_tensor(mask_c, device=dev)

    # マスク: 未公表 NaN と比較不能活動を λ,η の両方から落とす（CVAE 版の二段マスクと同じ）
    m_ga = (~torch.isnan(mga)) & mc[None, None, :, None]
    m_ge = (~torch.isnan(mge)) & mc[None, None, :, None]
    mga_f, mge_f = torch.nan_to_num(mga), torch.nan_to_num(mge)

    idx = torch.as_tensor(pool, dtype=torch.long, device=dev)                 # (D,M,T)
    d_ar = torch.arange(D, device=dev)[:, None, None]
    t_ar = torch.arange(T, device=dev)[None, None, :]

    lam = torch.zeros(N_G, N_A, NUM_ACT, T, dtype=dt, device=dev, requires_grad=True)
    eta = torch.zeros(N_G, N_E, NUM_ACT, T, dtype=dt, device=dev, requires_grad=True)

    def theta_of(lam_, eta_) -> torch.Tensor:
        """(λ,η) -> 群別の傾けベクトル θ_d (D, n_act, 96)"""
        lam_m = lam_ * m_ga                      # (G,A,C,T)
        eta_m = eta_ * m_ge                      # (G,E,C,T)
        # θ_(g,a,e) = π(e|g,a)·λ_ga + π(a|g,e)·η_ge
        th = (pi_e[..., None, None] * lam_m[:, :, None]      # (G,A,E,C,T)
              + pi_a[..., None, None] * eta_m[:, None, :])
        # (G,A,E) の C 順フラット化は d = g*(A*E)+a*E+e と一致する（d_index の定義）
        return th.reshape(D, NUM_ACT, T)

    def scores_of(theta: torch.Tensor) -> torch.Tensor:
        """s_{d,m} = ⟨θ_d, φ(x_dm)⟩ = Σ_t θ_d[x_dm_t, t]（one-hot を作らず gather で）"""
        return theta[d_ar, idx, t_ar].sum(-1)    # (D,M)

    def neg_dual() -> torch.Tensor:
        theta = theta_of(lam, eta)
        s = scores_of(theta)
        # −g(λ,η) = Σ_d logE_p[exp⟨θ,φ⟩] − ⟨λ,ν_ga⟩ − ⟨η,ν_ge⟩ + ridge/2·‖·‖²
        logmgf = (torch.logsumexp(s, dim=1) - np.log(M)).sum()
        lin = ((lam * m_ga) * mga_f).sum() + ((eta * m_ge) * mge_f).sum()
        reg = 0.5 * ridge * (((lam * m_ga) ** 2).sum() + ((eta * m_ge) ** 2).sum())
        return logmgf - lin + reg

    opt = torch.optim.LBFGS([lam, eta], max_iter=iters, history_size=50,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    n_eval = {"n": 0}

    def closure():
        opt.zero_grad()
        loss = neg_dual()
        loss.backward()
        n_eval["n"] += 1
        return loss

    # torch の LBFGS は履歴が飽和すると tolerance_change で早々に止まるので、
    # 履歴を張り直して (リスタート) 目的関数が動かなくなるまで繰り返す。
    # 目的関数が最良だった反復を保持して最後に戻す: ridge が極小のときは平坦方向が
    # ほぼ残り、行き過ぎた反復で ESS だけが落ちることがあるため
    prev, best = float("inf"), None
    for _ in range(RESTARTS):
        opt.step(closure)  # type: ignore[arg-type]
        with torch.no_grad():
            cur = float(neg_dual().detach())
        if not np.isfinite(cur):
            break
        if cur < prev:
            best = (lam.detach().clone(), eta.detach().clone())
        if abs(prev - cur) <= 1e-15 * max(1.0, abs(cur)):
            prev = min(prev, cur)
            break
        prev = cur
    if best is not None:
        with torch.no_grad():
            lam.copy_(best[0]); eta.copy_(best[1])
    opt.zero_grad()
    neg_dual().backward()
    grad_norm = float(torch.cat([
        (lam.grad if lam.grad is not None else torch.zeros_like(lam)).reshape(-1),
        (eta.grad if eta.grad is not None else torch.zeros_like(eta)).reshape(-1)]).norm())

    with torch.no_grad():
        theta = theta_of(lam, eta)
        s = scores_of(theta)
        w = torch.softmax(s, dim=1)                                  # (D,M) 正規化済み
        # 再構成周辺 μ̂ と残差
        mu = torch.zeros(D, NUM_ACT, T, dtype=dt, device=dev).scatter_add_(
            1, idx, w[:, :, None].expand(D, M, T))                   # (D,C,T)
        mu5 = mu.view(N_G, N_A, N_E, NUM_ACT, T)
        mu_ga = (mu5 * pi_e[..., None, None]).sum(dim=2)
        mu_ge = (mu5 * pi_a[..., None, None]).sum(dim=1)
        r_ga = (mu_ga - mga_f)[m_ga]
        r_ge = (mu_ge - mge_f)[m_ge]
        resid = torch.cat([r_ga, r_ge])
        ess = 1.0 / (w ** 2).sum(dim=1)                              # (D,) 有効サンプル数
        kl = (w * (torch.log(w.clamp(min=1e-30)) + np.log(M))).sum(dim=1)  # KL(q_d‖p_d)

    diag = {
        "tag": tag, "ridge": ridge, "pool_size": M, "lbfgs_evals": n_eval["n"],
        "grad_norm": grad_norm,
        "resid_mae": float(resid.abs().mean()), "resid_rmse": float((resid ** 2).mean().sqrt()),
        "resid_max": float(resid.abs().max()),
        "ess_min": float(ess.min()), "ess_median": float(ess.median()), "ess_mean": float(ess.mean()),
        "ess_frac_min": float(ess.min() / M), "ess_frac_median": float(ess.median() / M),
        "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
    }
    print(f"  [{tag}] ridge={ridge:g}  残差MAE {diag['resid_mae']:.5f}  "
          f"ESS中央値 {diag['ess_median']:.0f}/{M} ({diag['ess_frac_median']:.1%})  "
          f"ESS最小 {diag['ess_min']:.0f}  KL平均 {diag['kl_mean']:.3f} nats")
    return w.cpu().numpy().astype(np.float64), diag


# ============================================================
# 3. 実行
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="AggDDPM → 社基調 指数傾け転移 (Route B)")
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--reuse-pool", action="store_true")
    ap.add_argument("--ridge", type=float, default=RIDGE)
    ap.add_argument("--ridge-sweep", action="store_true",
                    help="複数の ridge で ESS と残差のトレードオフを掃引する")
    ap.add_argument("--sampler", choices=["ancestral", "ddim"], default=ddpm.SAMPLER,
                    help="プール生成のサンプラ。ddim は高速だが個票が断片化する (model.py の SAMPLER 参照)")
    ap.add_argument("--ddim-steps", type=int, default=ddpm.DDIM_STEPS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # --- 社基調ターゲット (平日) : CVAE 版と同一の読み込み経路 ---
    tgt = jm.load_stula_targets("timeband_weekday")
    mask_c = np.array([c not in EXCLUDED for c in Common])

    # --- 提案分布のプール ---
    pool = build_pool(args.pool_size, args.sampler, args.ddim_steps, args.reuse_pool)
    M = pool.shape[1]

    # --- バリアント ---
    records, diags = [], []

    # zero-shot: 傾けなし（一様重み）
    mu_zero = ddpm.pool_to_rates(pool)
    records.append({"variant": "zero-shot", "eval": "weekday_groups",
                    **jm.eval_against(mu_zero, tgt, mask_c)})

    ridges = [args.ridge]
    if args.ridge_sweep:
        ridges = [1e-2, 1e-3, 1e-4, 1e-5, 0.0]

    print("\nStage2: 指数傾け (最大エントロピー; 生成器は凍結) ...")
    w_main = None   # 後段の断片化チェックで使う「本命 ridge」の重み（掃引時に最後の値を掴まないよう明示）
    for r in ridges:
        w, diag = exponential_tilt(pool, tgt, mask_c, ridge=r,
                                   tag=f"tilted(ridge={r:g})" if args.ridge_sweep else "tilted")
        diags.append(diag)
        name = f"tilted_ridge{r:g}" if args.ridge_sweep else "tilted"
        records.append({"variant": name, "eval": "weekday_groups",
                        **jm.eval_against(ddpm.pool_to_rates(pool, w), tgt, mask_c)})
        if r == args.ridge:
            w_main = w
    if w_main is None:
        w_main = w

    # shuffled: 群対応を壊したプールに傾ける（負の対照）
    perm = rng.permutation(D_GROUPS)
    w_s, diag_s = exponential_tilt(pool[perm], tgt, mask_c, ridge=args.ridge, tag="shuffled")
    diags.append(diag_s)
    records.append({"variant": "shuffled", "eval": "weekday_groups",
                    **jm.eval_against(ddpm.pool_to_rates(pool[perm], w_s), tgt, mask_c)})

    # baseline: 全群=人口平均（モデル不要の床）
    grp_tbl, pop = tgt["group_rates_tbl"], tgt["pop"].reshape(D_GROUPS)
    W0 = jm.nan_renorm_pop_weights(grp_tbl, pop / pop.sum())
    baseline_mu = np.broadcast_to(np.nansum(grp_tbl * W0, axis=0), grp_tbl.shape)
    records.append({"variant": "baseline", "eval": "weekday_groups",
                    **jm.eval_against(baseline_mu, tgt, mask_c)})

    # --- 出力 ---
    res = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    pd.DataFrame(diags).to_csv(DIAG_CSV, index=False)

    main_cols = ["variant", "eval", "rate_mae", "rate_rmse", "dev_mae", "dev_rmse"] + \
                [f"{p}_{k}" for k in BREAK_CANDIDATES + PRESERVE_CANDIDATES for p in ("mae", "rmse")]
    act_names = [c.name for c in Common if mask_c[int(c)]]
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n=== 平日の群別行動者率(28群)=統計量ホールドアウト ===")
        print(res[main_cols].to_string(index=False))
        print("\n=== 活動別 RMSE (OTHER_X除外) ===")
        print(res[["variant"] + [f"rmse_{n}" for n in act_names]].to_string(index=False))
        print("\n=== 傾けの診断 (残差 vs 有効サンプル数) ===")
        print(pd.DataFrame(diags)[["tag", "ridge", "resid_mae", "resid_max",
                                    "ess_median", "ess_frac_median", "ess_min",
                                    "kl_mean"]].to_string(index=False))

    # --- 断片化が傾けで壊れていないことの確認（Route B の主張の核心）---
    print("\n=== 断片化 (傾け前後; 傾けは日単位の重み付けなので系列内部に触れない) ===")
    cond_idx, sched_real, w_real, _ = ddpm.load_data()
    r_stat = ddpm.fragmentation_stats(sched_real, w_real)
    counts = np.bincount(ddpm.cond_to_d(cond_idx), minlength=D_GROUPS).astype(float)
    pi_d = counts / counts.sum()
    flat = pool.reshape(-1, NUM_SLOTS)

    # ★系列構造の分布距離。教師（1次周辺）に入っていない量なので、集計が合っていても
    #   系列が壊れていないかの独立な検証になる。実データ自身の有限標本ゆらぎ（床）を
    #   併記する — 「実と値が違う」だけでは乖離の証拠にならないため
    im = ddpm.im
    jsd_floor = im.null_band_2sample(
        sched_real, lambda r, g: im.bigram_jsd(r, g, NUM_ACT),
        n_gen=len(flat), n_boot=40, seed=SEED, w=w_real)
    emd_floor = im.null_band_2sample(
        sched_real, lambda r, g: im.switch_dist_compare(r, g)["emd"],
        n_gen=len(flat), n_boot=40, seed=SEED, w=w_real)
    print(f"  ノイズ床(95%): bigram_jsd ≤ {jsd_floor[2]:.5f}   切替EMD ≤ {emd_floor[2]:.4f}")

    seq_rows = []
    for label, wts in [("zero-shot", None), ("tilted", w_main)]:
        ww = np.repeat(pi_d / M, M) if wts is None else (pi_d[:, None] * wts).reshape(-1)
        g_stat = ddpm.fragmentation_stats(flat, ww)
        jsd = im.bigram_jsd(sched_real, flat, NUM_ACT, w_real, ww)
        sw  = im.switch_dist_compare(sched_real, flat, w_real, ww)
        print(f"  {label:<10} 切替 {g_stat['switches']:.2f} (実 {r_stat['switches']:.2f})  "
              f"1スロットep比率 {g_stat['single_slot_ratio']:.1%} (実 {r_stat['single_slot_ratio']:.1%})  "
              f"平均参加率 {g_stat['participation'].mean():.3f} (実 {r_stat['participation'].mean():.3f})")
        print(f"  {'':<10} bigram_jsd {jsd:.5f} {'★床の外' if jsd > jsd_floor[2] else '床の内'}   "
              f"切替EMD {sw['emd']:.4f} {'★床の外' if sw['emd'] > emd_floor[2] else '床の内'}   "
              f"wrap_closure {g_stat['wrap_closure_rate']:.3f} (実 {r_stat['wrap_closure_rate']:.3f})")
        seq_rows.append({
            "variant": label, "switches": g_stat["switches"],
            "single_slot_ratio": g_stat["single_slot_ratio"],
            "wrap_closure_rate": g_stat["wrap_closure_rate"],
            "bigram_jsd": jsd, "bigram_jsd_floor_hi": jsd_floor[2],
            "switch_emd": sw["emd"], "switch_emd_floor_hi": emd_floor[2],
            "switch_ks": sw["ks"],
        })
    seq_rows.append({
        "variant": "real", "switches": r_stat["switches"],
        "single_slot_ratio": r_stat["single_slot_ratio"],
        "wrap_closure_rate": r_stat["wrap_closure_rate"],
        "bigram_jsd": 0.0, "bigram_jsd_floor_hi": jsd_floor[2],
        "switch_emd": 0.0, "switch_emd_floor_hi": emd_floor[2], "switch_ks": 0.0,
    })
    pd.DataFrame(seq_rows).to_csv(SEQ_CSV, index=False)

    print(f"\nsaved -> {OUT_CSV}")
    print(f"saved -> {DIAG_CSV}")
    print(f"saved -> {SEQ_CSV}")


if __name__ == "__main__":
    main()
