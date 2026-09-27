"""
stage1_rare_diagnosis.py
========================
Stage 1 で少ない活動（買い物・介護・育児・移動・スポーツ・ボランティア）の行動者率が外れる原因を、
計画で事前に固定した仮説ごとに切り分ける（計画: ~/.claude/plans/stage1-crispy-hoare.md）。

用語（この診断の中で一意に使う）:

    総量            1 日の平均行動者率 = その活動のスロット数の平均 / 96
    総量の比        生成の総量 / ATUS 実データの総量（1 なら一致）
    行動者          その日に 1 スロットでもその活動をした人
    行動者の割合    行動者の人数の割合
    1 人あたりの長さ 行動者 1 人あたりのスロット数（= 総量 × 96 / 行動者の割合）
    エピソードの数  1 人あたりの連続区間の数（移動では 1 人あたりの移動の回数）
    長時間の行動者  その活動が LONG_DOER_SLOTS スロット（3 時間）以上の人の割合

群の中は個票のウェイト（ATUS 実は TUFINLWGT、生成は一様）で、群の間は group_weights の重みで平均する。
★群の重みは 28 群で 1 つの割合を使う。報告書の曲線（stage2_curves.weighted_slot_rates）は
  非公表セルごとに重みを正規化し直すので、総量は曲線の平均と小数 3 桁目で違いうる。

部（--part）と仮説:

    existing   段 A。既存のプールだけで読む（手元）
               H1 生成の乱数: 同じ ckpt の 2 プール（学習時・@g1.25）の差 vs 種の間の差
               H2 評価の加重: 総量の比を日本人口加重と ATUS ウェイトの群構成で出す
               H3 暗記: 行動者に絞った train までの距離（DCR）を、holdout の行動者の距離と比べる

データフロー:

```mermaid
flowchart TD
    ARMS["stage1_clock_replicates.pool_csv<br/>(arm, seed) → プール CSV"] --> LP["stage2_curves.load_sample_pool<br/>pool (28, M, 96)"]
    LP --> PP["pool_people<br/>sched (N, 96) / d (N,) / w (N,)"]
    REAL["atus_group_rates.load_atus_weekday<br/>sched / groups / w"] --> RP["ATUS 実の個票"]
    PP --> AP["activity_profile<br/>総量 / 行動者の割合 / 1 人あたりの長さ / ..."]
    RP --> AP
    GW["group_weights<br/>japan: tgt['pop'] / atus: TUFINLWGT の群和"] --> AP
    AP --> H1["h1_pool_pairs<br/>プール間の差 vs 種間 sd"]
    AP --> H2["h2_weighting<br/>日本人口加重 vs ATUS の群構成"]
    PP --> H3["h3_copies<br/>individual_metrics.memorization<br/>行動者だけの DCR"]
    SPLIT["model.split_indices<br/>train / holdout"] --> H3
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part existing
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "processed" / "aggregates"

# 診断する少ない活動（stage2_curves.ACT_NAMES の名前）
FOCUS_ACTS: tuple[str, ...] = ("SHOPPING", "CAREGIVING", "TRAVEL", "SPORTS", "VOLUNTEER")
# 長時間の行動者の閾値。12 スロット = 3 時間（介護・育児で実 6.5%）
LONG_DOER_SLOTS = 12
DIAG_SEEDS: tuple[int, ...] = (42, 43, 44)
# H1 の対: 学習時のプールと共通乱数のプール（@g1.25）が同じ ckpt から揃っている arm
H1_ARMS: tuple[str, ...] = ("clock", "clock_tf96")
# H1 の判定: プール間の差 ≥ 種間 sd × この値なら「生成の乱数」を支持（計画で固定）
H1_RATIO_THRESHOLD = 1.0 / 3.0

GroupWeightKind = Literal["japan", "atus"]
FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


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


rep: Any = _load("rare_diag_replicates",
                 REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_clock_replicates.py")
cur: Any = rep.cur
sm: Any = rep.sm
agr: Any = rep.agr
im: Any = rep.sel.im


# ============================================================
# 個票の読み込みと群の重み
# ============================================================
def pool_people(pool: IntArr) -> tuple[IntArr, IntArr, FloatArr]:
    """群別プールを個票の並びに展開する

    Args:
        pool: 群別サンプルプール, dtype=int64, (28, M, 96)

    Returns:
        (sched (28·M, 96), 群インデックス d (28·M,), 個票のウェイト w (28·M,) = 1)
    """
    n_d, m, s = pool.shape
    sched = np.asarray(pool.reshape(n_d * m, s), dtype=np.int64)
    d = np.repeat(np.arange(n_d, dtype=np.int64), m)
    return sched, d, np.ones(n_d * m, dtype=np.float64)


def load_pool_people(label: str, seed: int) -> tuple[IntArr, IntArr, FloatArr]:
    """(arm ラベル, seed) のプールを読んで個票に展開する（学習時のプールは mtime を確かめる）"""
    path = rep.pool_csv(label, seed)
    if not path.exists():
        raise FileNotFoundError(f"生成プールが無い ({label}, seed={seed}): {path}")
    if rep.parse_label(label)[1] is None:
        rep.check_fresh(path, rep.ckpt_path(label, seed))
    return pool_people(cur.load_sample_pool(path))


def group_weights(kind: GroupWeightKind, tgt: dict, d_real: IntArr, w_real: FloatArr) -> FloatArr:
    """群の間の重み (28,)

    Args:
        kind: "japan" は日本の人口 tgt["pop"]、"atus" は ATUS の TUFINLWGT の群ごとの和
        tgt: stage2_targets.load_stula_targets の戻り値
        d_real: ATUS 実の群インデックス, (N,)
        w_real: ATUS 実の調査ウェイト, (N,)

    Returns:
        和が 1 の群の重み, (28,)
    """
    if kind == "japan":
        pop = np.asarray(tgt["pop"], dtype=np.float64).reshape(-1)
        return pop / pop.sum()
    tot = np.bincount(d_real, weights=w_real, minlength=sm.D_GROUPS).astype(np.float64)
    return tot / tot.sum()


# ============================================================
# 活動ごとの個人単位の要約
# ============================================================
def activity_profile(sched: IntArr, d: IntArr, w: FloatArr, pi_d: FloatArr,
                     act: str) -> dict[str, float]:
    """1 活動の総量を、行動者の割合と 1 人あたりの長さに分ける

    Args:
        sched: 個票のスケジュール, dtype=int64, (N, 96)
        d: 群インデックス, (N,)
        w: 群の中の個票のウェイト, (N,)
        pi_d: 群の間の重み, (28,)。個票の無い群は除いて正規化し直す
        act: 活動名（stage2_curves.ACT_NAMES）

    Returns:
        level（総量）, doer_share（行動者の割合）, slots_per_doer（1 人あたりの長さ）,
        episodes_per_person（エピソードの数）, long_doer_share（長時間の行動者）
    """
    c = cur.ACT_NAMES.index(act)
    hit = sched == c                                                   # (N, 96)
    slots = hit.sum(axis=1)
    episodes = hit[:, 0].astype(np.int64) + (hit[:, 1:] & ~hit[:, :-1]).sum(axis=1)
    per_person = np.stack([slots / sm.NUM_SLOTS, slots > 0, episodes,
                           slots >= LONG_DOER_SLOTS], axis=1).astype(np.float64)   # (N, 4)

    means = np.zeros(4, dtype=np.float64)
    total = 0.0
    for g in range(len(pi_d)):
        idx = np.flatnonzero(d == g)
        if len(idx) == 0:
            continue
        wn = w[idx] / w[idx].sum()
        means += pi_d[g] * (wn @ per_person[idx])
        total += pi_d[g]
    level, doer, epi, long_ = means / total
    return {"level": float(level), "doer_share": float(doer),
            "slots_per_doer": float(level * sm.NUM_SLOTS / doer) if doer > 0 else float("nan"),
            "episodes_per_person": float(epi), "long_doer_share": float(long_)}


def profile_table(sched: IntArr, d: IntArr, w: FloatArr, pi_d: FloatArr,
                  acts: tuple[str, ...] = FOCUS_ACTS) -> pd.DataFrame:
    """activity_profile を活動ごとに並べた表（行 = 活動）"""
    return pd.DataFrame({a: activity_profile(sched, d, w, pi_d, a) for a in acts}).T


# ============================================================
# H1 生成の乱数
# ============================================================
def h1_pool_pairs(real_prof: pd.DataFrame, pi_d: FloatArr) -> pd.DataFrame:
    """同じ ckpt の 2 プールの総量の比の差を、種の間のばらつきと比べる

    Args:
        real_prof: ATUS 実の profile_table
        pi_d: 群の間の重み（日本人口）

    Returns:
        (arm, 活動) ごとの pool_diff（2 プールの総量の比の差の絶対値、種で平均）,
        seed_sd（2 プールを平均した総量の比の種間 sd）, ratio = pool_diff / seed_sd, supports_h1
    """
    rows = []
    for arm in H1_ARMS:
        per_seed: dict[str, list[tuple[float, float]]] = {a: [] for a in FOCUS_ACTS}
        for seed in DIAG_SEEDS:
            prof_t = profile_table(*load_pool_people(arm, seed), pi_d)
            prof_g = profile_table(*load_pool_people(f"{arm}@g1.25", seed), pi_d)
            for a in FOCUS_ACTS:
                base = real_prof.loc[a, "level"]
                per_seed[a].append((prof_t.loc[a, "level"] / base, prof_g.loc[a, "level"] / base))
        for a in FOCUS_ACTS:
            pairs = np.asarray(per_seed[a], dtype=np.float64)          # (種, 2)
            pool_diff = float(np.abs(pairs[:, 0] - pairs[:, 1]).mean())
            seed_sd = float(pairs.mean(axis=1).std(ddof=1))
            ratio = pool_diff / seed_sd if seed_sd > 0 else float("inf")
            rows.append({"arm": arm, "activity": a,
                         "ratio_train_pool": "/".join(f"{v:.2f}" for v in pairs[:, 0]),
                         "ratio_g1.25_pool": "/".join(f"{v:.2f}" for v in pairs[:, 1]),
                         "pool_diff": pool_diff, "seed_sd": seed_sd, "diff_over_sd": ratio,
                         "supports_h1": ratio >= H1_RATIO_THRESHOLD})
    return pd.DataFrame(rows)


# ============================================================
# H2 評価の加重
# ============================================================
def h2_weighting(real: tuple[IntArr, IntArr, FloatArr], weights: dict[str, FloatArr],
                 arms: tuple[str, ...] = H1_ARMS) -> pd.DataFrame:
    """総量の比を、群の重みを変えて並べる

    Args:
        real: ATUS 実の (sched, d, w)
        weights: 重みの名前 → 群の重み (28,)
        arms: 読む arm（学習時のプール）

    Returns:
        (arm, 活動, 重み) ごとの種別の総量の比と、|log 比| の種平均
    """
    real_prof = {k: profile_table(*real, pi) for k, pi in weights.items()}
    rows = []
    for arm in arms:
        people = [load_pool_people(arm, s) for s in DIAG_SEEDS]
        for k, pi in weights.items():
            profs = [profile_table(*p, pi) for p in people]
            for a in FOCUS_ACTS:
                r = np.asarray([p.loc[a, "level"] / real_prof[k].loc[a, "level"] for p in profs])
                rows.append({"arm": arm, "activity": a, "group_weight": k,
                             "ratio_by_seed": "/".join(f"{v:.2f}" for v in r),
                             "mean_abs_log_ratio": float(np.abs(np.log(r)).mean())})
    return pd.DataFrame(rows)


# ============================================================
# H3 暗記
# ============================================================
def distance_to_train(people: IntArr, ref: IntArr, seed: int = 0) -> tuple[float, float]:
    """people の各個票から ref の最近傍までのハミング距離（DCR）の (平均, 5% 点)

    Note:
        ★完全一致やハミング距離 2 以下の率は、96 スロットの 1 日では holdout でも 0 になり
          区別がつかない（2026-09-27 に確認）。そこで距離の分布そのものを比べる。
    """
    if len(people) == 0:
        return float("nan"), float("nan")
    m = im.memorization(people, ref, sample=2000, seed=seed)
    return float(m["DCR_mean[train]"]), float(m["DCR_p05[train]"])


def h3_copies(arms: tuple[str, ...] = H1_ARMS) -> pd.DataFrame:
    """行動者に絞った train までの距離を、holdout の行動者の距離（暗記が起きえない基準）と並べる

    Note:
        ★holdout の個票は学習に使っていないので、holdout の行動者から train までの距離は
          「データ分布に近いだけ」の基準になる。生成の行動者の距離がこれより短く、しかも
          総量の比の大きい種ほど短ければ、少数の行動者を再生して総量が決まっている疑いがある。

    Returns:
        (arm または holdout, 種, 活動) ごとの行動者数・DCR の平均・DCR の 5% 点
    """
    _, sched_real, _, _ = sm.load_data()
    train_idx, val_idx = sm.split_indices(len(sched_real))
    train, holdout = sched_real[train_idx], sched_real[val_idx]
    rows = []
    for a in FOCUS_ACTS:
        c = cur.ACT_NAMES.index(a)
        doers = holdout[(holdout == c).any(axis=1)]
        dcr, dcr05 = distance_to_train(doers, train)
        rows.append({"arm": "holdout", "seed": -1, "activity": a, "n_doers": len(doers),
                     "dcr_mean": dcr, "dcr_p05": dcr05})
    for arm in arms:
        for seed in DIAG_SEEDS:
            sched, _, _ = load_pool_people(arm, seed)
            for a in FOCUS_ACTS:
                c = cur.ACT_NAMES.index(a)
                doers = sched[(sched == c).any(axis=1)]
                dcr, dcr05 = distance_to_train(doers, train)
                rows.append({"arm": arm, "seed": seed, "activity": a, "n_doers": len(doers),
                             "dcr_mean": dcr, "dcr_p05": dcr05})
    return pd.DataFrame(rows)


# ============================================================
# 部の実行
# ============================================================
def run_existing() -> None:
    """段 A（H1〜H3）を集計して表示し、CSV に書く"""
    tgt = cur.st.load_stula_targets()
    sched_r, d_r, w_r = agr.load_atus_weekday()
    real: tuple[IntArr, IntArr, FloatArr] = (sched_r, d_r, w_r)
    weights = {"japan": group_weights("japan", tgt, d_r, w_r),
               "atus": group_weights("atus", tgt, d_r, w_r)}
    real_prof = profile_table(sched_r, d_r, w_r, weights["japan"])

    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print("=== ATUS 実（日本人口加重）===")
        print(real_prof.to_string())

        h1 = h1_pool_pairs(real_prof, weights["japan"])
        print(f"\n=== H1 生成の乱数（支持: プール間の差 ≥ 種間 sd × {H1_RATIO_THRESHOLD:.2f}）===")
        print(h1.to_string(index=False))

        h2 = h2_weighting(real, weights)
        print("\n=== H2 評価の加重（|log 総量の比| の種平均が atus で小さければ加重が拡大している）===")
        print(h2.pivot_table(index=["arm", "activity"], columns="group_weight",
                             values="mean_abs_log_ratio").to_string())

        h3 = h3_copies()
        print("\n=== H3 行動者に絞った train までの距離（holdout は暗記が起きえない基準）===")
        print(h3.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h1.to_csv(OUT_DIR / "stage1_rare_diag_h1.csv", index=False)
    h2.to_csv(OUT_DIR / "stage1_rare_diag_h2.csv", index=False)
    h3.to_csv(OUT_DIR / "stage1_rare_diag_h3.csv", index=False)
    print("\n[rare_diag] 書いた: stage1_rare_diag_{h1,h2,h3}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("existing",), required=True,
                    help="existing: 段 A（H1〜H3、既存プールのみ）")
    args = ap.parse_args()
    if args.part == "existing":
        run_existing()


if __name__ == "__main__":
    main()
