"""
conditioning.py
================
条件付き生成モデルの中心的主張「**群 d を与えると d らしい個票が出る**」の検証。

なぜ別モジュールか:
    individual_metrics.py / feasibility.py / clock_diagnostics.py は
    **28群をプールした後**の分布・構造・時刻を見る。プールしてしまうと
    「どの群にも同じ平均的な個票を出しているモデル」と
    「群ごとに正しく描き分けているモデル」が同じ点数になる。
    条件付けが効いているかは別の軸なので分ける。

★ 3つの見方（どれか一つでは足りない）:
    A. group_profile_comparison  群ごとの活動プロファイルが実データと合うか
       → 合わない群があれば、その群の条件付けが効いていない
    B. separation_summary        群と群の**差**が生成側で縮んでいないか
       → A が全群で緩く合っていても、差が縮んでいれば描き分けていない
    C. conditioning_accuracy     生成個票から指定した属性を当て返せるか
       → 実データで学習した分類器を生成個票に当て、実 holdout の精度と比べる

★ 「実データより精度が高い」も欠陥である:
    実個票には例外（働いていても WORK が無い日など）があり、分類器の精度は 1.0 にならない。
    生成側の精度が実 holdout を**上回る**のは、群ごとに型どおりの個票しか出していない
    （群内多様性の消失）ことを意味する。上振れ・下振れの両方を見る。

★ 床（ノイズ床）の考え方は individual_metrics と同じ:
    実データを取り直したときの揺れの外に出て初めて乖離と言う。
    群別の解析は1群あたりの標本が薄くなるので床が広い。床の広さも一緒に読む。

使い方:
    uv run python src/eval/test_conditioning.py       # 自己テスト
    cond = <importlib で src/eval/conditioning.py をロード>
    cond.group_profile_comparison(real, gen, d_real, d_gen, n_act, n_groups, w_real, w_gen)
    cond.conditioning_accuracy(real_tr, d_tr, real_va, d_va, gen, d_gen, attr_grid, n_act)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_SLOTS = 96
SLOT_MIN = 15


def load_module(name: str, path: Path):
    """sys.modules に一意名で直接載せる（同名 model.py の取り違えを防ぐ）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


im = load_module("cond_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")


# ============================================================
# A. 群ごとの活動プロファイル
# ============================================================
def group_profiles(sched: IntArr, groups: IntArr, n_act: int, n_groups: int,
                   w: FloatArr | None = None) -> tuple[FloatArr, IntArr]:
    """群ごとの活動時間シェア (n_groups, n_act) と群別本数 (n_groups,)。

    シェアは群内で正規化する（各行の和が1）。群内が0人の行は全て 0 になる。
    群間の構成比の違いは入らないので、**群ごとの中身だけ**を比べられる。
    """
    sched = np.asarray(sched)
    groups = np.asarray(groups)
    prof = np.zeros((n_groups, n_act), dtype=np.float64)
    counts = np.bincount(groups, minlength=n_groups).astype(np.int64)
    for d in range(n_groups):
        sel = groups == d
        if not sel.any():
            continue
        prof[d] = im.time_share(sched[sel], n_act, None if w is None else np.asarray(w)[sel])
    return prof, counts


def _jsd(p: FloatArr, q: FloatArr) -> float:
    """Jensen-Shannon divergence（自然対数, 0..ln2）。片方が全0なら nan。"""
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.nansum(np.where(p > 0, p * np.log(p / m), 0.0))
        kl_qm = np.nansum(np.where(q > 0, q * np.log(q / m), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def group_profile_comparison(real: IntArr, gen: IntArr, d_real: IntArr, d_gen: IntArr,
                             n_act: int, n_groups: int, group_labels: list[str],
                             w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                             n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """★A: 群ごとの活動プロファイル JSD と、その群の標本数で決まる床。

    床は「実データの**その群**を復元抽出して2つに分けたときの JSD」＝
    同じ群の実データ同士でも標本誤差でこれだけ離れる、という量。
    生成側の JSD がこの床を超えた群だけを「条件付けが効いていない群」と呼べる。

    ★ 床は群内の実標本数で決まるので、n_real が小さい群では広い。
      「床の内」を「合っている」と読んではいけない（検出力が無いだけかもしれない）。
    """
    pr, n_r = group_profiles(real, d_real, n_act, n_groups, w_real)
    pg, n_g = group_profiles(gen, d_gen, n_act, n_groups, w_gen)
    rng = np.random.default_rng(seed)
    real = np.asarray(real)
    d_real = np.asarray(d_real)

    rows = []
    for d in range(n_groups):
        obs = _jsd(pr[d], pg[d])
        idx = np.where(d_real == d)[0]
        floor_hi = float("nan")
        if len(idx) >= 4:
            vals = np.empty(n_boot, dtype=np.float64)
            half = len(idx) // 2
            sub_w = None if w_real is None else np.asarray(w_real)[idx]
            for b in range(n_boot):
                perm = rng.permutation(len(idx))
                a, c = idx[perm[:half]], idx[perm[half:]]
                wa = None if sub_w is None else sub_w[perm[:half]]
                wc = None if sub_w is None else sub_w[perm[half:]]
                vals[b] = _jsd(im.time_share(real[a], n_act, wa),
                               im.time_share(real[c], n_act, wc))
            floor_hi = float(np.quantile(vals, 0.95))
        rows.append({
            "group_d": d, "label": group_labels[d],
            "n_real": int(n_r[d]), "n_gen": int(n_g[d]),
            "jsd": obs, "jsd_floor_hi": floor_hi,
            "verdict": ("判定不能" if not np.isfinite(obs) or not np.isfinite(floor_hi)
                        else "★床の外" if obs > floor_hi else "床の内"),
        })
    return pd.DataFrame(rows)


# ============================================================
# B. 群と群の差が縮んでいないか
# ============================================================
def separation_summary(real: IntArr, gen: IntArr, d_real: IntArr, d_gen: IntArr,
                       n_act: int, n_groups: int,
                       w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                       min_n: int = 20) -> dict:
    """★B: 群間プロファイル距離（全ペアの JSD 平均）を実 / 生成で比べる。

    `separation_ratio` < 1 は生成側で**群の差が縮んでいる**＝条件付けが弱い。
    A（群ごとの一致）が緩くても B が縮んでいれば描き分けていない。

    min_n: 実データがこの人数未満の群はペアから外す（薄い群の推定誤差で
        群間距離が水増しされるため）。
    """
    pr, n_r = group_profiles(real, d_real, n_act, n_groups, w_real)
    pg, _ = group_profiles(gen, d_gen, n_act, n_groups, w_gen)
    use = [d for d in range(n_groups) if n_r[d] >= min_n and pg[d].sum() > 0]

    dr, dg = [], []
    for i, a in enumerate(use):
        for b in use[i + 1:]:
            dr.append(_jsd(pr[a], pr[b]))
            dg.append(_jsd(pg[a], pg[b]))
    dr_arr, dg_arr = np.asarray(dr), np.asarray(dg)
    return {
        "n_groups_used": len(use),
        "n_pairs": len(dr_arr),
        "sep_real": float(dr_arr.mean()),
        "sep_gen": float(dg_arr.mean()),
        "separation_ratio": float(dg_arr.mean() / dr_arr.mean()) if dr_arr.mean() > 0 else np.nan,
        # 群間の順位が保たれているか（縮んでいても順序が合っていれば描き分けはある）
        "spearman_pairs": float(pd.Series(dr_arr).corr(pd.Series(dg_arr), method="spearman")),
    }


def attribute_contrast(real: IntArr, gen: IntArr, d_real: IntArr, d_gen: IntArr,
                       attr_grid: IntArr, attr_col: int, n_act: int, act_names: list[str],
                       w_real: FloatArr | None = None, w_gen: FloatArr | None = None
                       ) -> pd.DataFrame:
    """★B': 属性を1つ切り替えたときの活動シェアの動きを実 / 生成で比べる。

    attr_grid: (n_groups, n_attr) の属性表（`ddpm.cond_grid()` を渡す）。
    attr_col: 見る属性の列（例: employment）。

    その属性の値ごとに全体シェアを出し、値間の差（コントラスト）を比べる。
    「モデルが employment を見ているか」を活動レベルの言葉で読める形にする。
    """
    attr_grid = np.asarray(attr_grid)
    a_real = attr_grid[np.asarray(d_real), attr_col]
    a_gen = attr_grid[np.asarray(d_gen), attr_col]
    values = np.unique(attr_grid[:, attr_col])
    if len(values) != 2:
        raise ValueError(f"2値属性のみ対応（列 {attr_col} は {len(values)} 値）")

    def share(sched, sel, w):
        return im.time_share(np.asarray(sched)[sel], n_act,
                             None if w is None else np.asarray(w)[sel])

    lo, hi = values[0], values[1]
    r_diff = (share(real, a_real == hi, w_real) - share(real, a_real == lo, w_real)) * NUM_SLOTS * SLOT_MIN
    g_diff = (share(gen, a_gen == hi, w_gen) - share(gen, a_gen == lo, w_gen)) * NUM_SLOTS * SLOT_MIN
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(r_diff) > 1e-9, g_diff / r_diff, np.nan)
    return pd.DataFrame({
        "activity": act_names,
        f"real Δ分/日 ({hi}-{lo})": r_diff,
        f"gen Δ分/日 ({hi}-{lo})": g_diff,
        "gen/real": ratio,
    })


def plot_group_jsd(table: pd.DataFrame, path=None, title: str = "") -> None:
    """群ごとの JSD と床を並べる（group_profile_comparison の結果を渡す）。

    棒 = 観測 JSD、線 = その群の床。棒が線を超えた群が「条件付けが効いていない群」。
    群を n_real 順に並べるので、床の広さが標本数で決まることも同時に見える。
    """
    import matplotlib.pyplot as plt

    t = table.sort_values("n_real").reset_index(drop=True)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    colors = ["tab:red" if v == "★床の外" else "tab:blue" for v in t["verdict"]]
    ax.bar(x, t["jsd"], color=colors, alpha=0.85, label="JSD (実 vs 生成)")
    ax.plot(x, t["jsd_floor_hi"], color="black", lw=1.4, ls="--", marker="_",
            label="床 (実データ同士の95%点)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(t["label"], t["n_real"])],
                       rotation=90, fontsize=7)
    ax.set_ylabel("プロファイル JSD")
    ax.set_title(title or "群ごとの活動プロファイル一致（赤 = 床の外）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    im._save(fig, path)
    plt.show()


# ============================================================
# C. 生成個票から属性を当て返せるか
# ============================================================
def condition_features(sched: IntArr, n_act: int) -> FloatArr:
    """個票 -> 分類器の特徴 (N, 2*n_act + 1)。

    活動別の1日合計分・活動別エピソード本数・切替回数。
    生スロット列 (96) を直接使わないのは、時刻のずれだけで属性が当たるのを避け、
    「何をどれだけしたか」で属性が読めるかを見たいため。
    """
    sched = np.asarray(sched)
    minutes = np.stack([(sched == a).sum(axis=1) for a in range(n_act)], axis=1) * SLOT_MIN
    eps = []
    for a in range(n_act):
        is_a = sched == a
        eps.append(is_a[:, 0].astype(np.int64) + (is_a[:, 1:] & ~is_a[:, :-1]).sum(axis=1))
    episodes = np.stack(eps, axis=1)
    switches = im.n_switches(sched)[:, None]
    return np.concatenate([minutes, episodes, switches], axis=1).astype(np.float64)


def conditioning_accuracy(real_train: IntArr, d_train: IntArr,
                          real_hold: IntArr, d_hold: IntArr,
                          gen: IntArr, d_gen: IntArr,
                          attr_grid: IntArr, attr_names: list[str], n_act: int,
                          seed: int = 0, max_iter: int = 2000) -> pd.DataFrame:
    """★C: 実 train で属性分類器を学習し、実 holdout と生成個票に当てる。

    返り値の読み方（`balanced` = balanced accuracy）:
      gen ≈ holdout       条件付けが実データ並みに効いている
      gen <  holdout      条件が効いていない（群を無視した個票）
      gen >  holdout      群ごとに型どおり＝群内多様性の消失（過剰な条件付け）

    ★ balanced accuracy を主に見る。生成プールは群一様でクラス比が実データと違うので、
      素の accuracy は比較できない。`chance` は「常に多数派」の balanced accuracy = 0.5。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    attr_grid = np.asarray(attr_grid)
    x_tr = condition_features(real_train, n_act)
    x_ho = condition_features(real_hold, n_act)
    x_gn = condition_features(gen, n_act)

    rows = []
    for col, name in enumerate(attr_names):
        y_tr = attr_grid[np.asarray(d_train), col]
        y_ho = attr_grid[np.asarray(d_hold), col]
        y_gn = attr_grid[np.asarray(d_gen), col]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed),
        )
        clf.fit(x_tr, y_tr)
        rows.append({
            "attribute": name,
            "n_class": int(len(np.unique(y_tr))),
            "balanced[train]": float(balanced_accuracy_score(y_tr, clf.predict(x_tr))),
            "balanced[holdout]": float(balanced_accuracy_score(y_ho, clf.predict(x_ho))),
            "balanced[gen]": float(balanced_accuracy_score(y_gn, clf.predict(x_gn))),
            "chance": float(1.0 / len(np.unique(y_tr))),
        })
    out = pd.DataFrame(rows)
    out["gen - holdout"] = out["balanced[gen]"] - out["balanced[holdout]"]
    return out
