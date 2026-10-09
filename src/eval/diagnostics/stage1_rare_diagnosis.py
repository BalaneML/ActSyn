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
    restore    段 B。stage1_rare_diagnosis_gen --mode restore の結果を読む
               H5 実個票を t0 まで雑音化して戻したとき、床を超える最小の t0（t*）で段階を分ける
                  （t* ≤ 100: エピソードの長さ、t* ≥ 500: 誰が行動者か）
               H6 最大の t0 で argmax 前の連続値の総量と argmax 後の総量を比べる
    final      H6 の補足。t0 = 999 のスロットごとの連続値から、僅差で argmax を取れなかった分と
               全チャネルに薄く乗った値（漏れ）を分ける（連続値の和では両者を区別できないため）
    guidance   H7 CFG: clock_tf96 の guidance 1.0 と 1.25（同じ乱数）の差を、H1 のプール間の差と比べる
    trajectory 段 C。H4 ckpt の選び方: 途中の ckpt の小プールで、最良 epoch 前後の種内の揺れと
               種間の揺れを比べる（--best-epochs に学習ログの最良 epoch を渡す）
    seeds      H8 K=4 と Transformer 型を種 5 本ずつで比べる（長時間の行動者の割合など）

判定の閾値は計画で事前に固定したもの（H1_RATIO_THRESHOLD / FLOOR_Z / H5_T_* / H6_THRESHOLD / H4_WINDOW）。
床 = ATUS の回答者を復元抽出したときの比の sd × FLOOR_Z。

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
    NPZ["ddpm_simple_restore{接尾辞}.npz<br/>sched_out / soft_sum / raw_sum"] --> H5["h5_restore / h5_first_t0<br/>t0 ごとの比と t*"]
    NPZ --> H6["h6_decoding<br/>argmax 前後の総量"]
    FIN["ddpm_simple_restore{接尾辞}_final.npz<br/>x_final (N, 12, 96)"] --> H6B["h6_near_ties<br/>僅差の負け / 漏れ"]
    BOOT["bootstrap_floor<br/>復元抽出の sd"] --> H5
    EPP["gen.epoch_pools<br/>_ep{epoch:04d}_n64.csv"] --> H4["h4_trajectory<br/>種内 sd vs 種間 sd"]
    AP --> H7["h7_guidance<br/>@g1 vs @g1.25"]
    AP --> H8["h8_seeds<br/>種 5 本の完全分離"]
```

使い方:
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part existing
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part restore
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part final
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part guidance
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part trajectory \\
        --best-epochs 42:612 43:658 44:700
    .venv/bin/python src/eval/diagnostics/stage1_rare_diagnosis.py --part seeds
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

# 段 B・C（計画で固定した判定の閾値）
METRICS: tuple[str, ...] = ("level", "doer_share", "slots_per_doer", "episodes_per_person",
                            "long_doer_share")
RESTORE_ARMS: tuple[str, ...] = ("clock", "clock_tf96")
ARM_LABELS: dict[str, str] = {"clock": "倍音 K=4", "clock_tf96": "Transformer 型"}
# 床 = ATUS の回答者を復元抽出したときの比の sd × FLOOR_Z（約 95%）
FLOOR_Z = 2.0
N_BOOT = 200
# H5: 床を超える最小の t0（t*）がこれ以下なら「エピソードの長さ」、これ以上なら「誰が行動者か」の段階
H5_T_LOCAL = 100
H5_T_GLOBAL = 500
# H6: argmax 前の連続値の総量と argmax 後の総量の相対差がこれを超えたら離散化のずれ
H6_THRESHOLD = 0.10
# H6 の補足（final）: その活動の値がこれ以上なのに argmax を取れなかったスロットを「僅差の負け」、
# LEAK_VALUE 未満の値を「漏れ」と呼ぶ
H6_TIE_VALUE = 0.3
LEAK_VALUE = 0.1
# H4: 最良 epoch の前後この幅の epoch で種内の揺れを測る
H4_WINDOW = 200
H4_ARM = "clock_tf96_traj"
EPOCH_POOL_N = 64
# H8: 種を 5 本にして比べる
H8_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
FIG_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "docs" / "figures"

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


def _gen() -> Any:
    """stage1_rare_diagnosis_gen（保存先の名前の唯一の出所）を読む"""
    return _load("rare_diag_gen", REPO_ROOT / "src" / "eval" / "diagnostics" / "stage1_rare_diagnosis_gen.py")


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
def group_weighted_mean(values: FloatArr, d: IntArr, w: FloatArr, pi_d: FloatArr) -> FloatArr:
    """個人単位の値を、群の中は w、群の間は pi_d で平均する

    Args:
        values: 個人単位の値, (N, k)
        d: 群インデックス, (N,)
        w: 群の中の個票のウェイト, (N,)
        pi_d: 群の間の重み, (28,)。個票の無い群は除いて正規化し直す

    Returns:
        加重平均, (k,)
    """
    means = np.zeros(values.shape[1], dtype=np.float64)
    total = 0.0
    for g in range(len(pi_d)):
        idx = np.flatnonzero(d == g)
        if len(idx) == 0:
            continue
        wn = w[idx] / w[idx].sum()
        means += pi_d[g] * (wn @ values[idx])
        total += pi_d[g]
    return means / total


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
    level, doer, epi, long_ = group_weighted_mean(per_person, d, w, pi_d)
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
# 段 B・C の共通: 実データの並びと床
# ============================================================
def _rows(df: pd.DataFrame, **eq: Any) -> pd.DataFrame:
    """列 = 値 の条件をすべて満たす行（.loc と numpy の真偽値で絞り、型を DataFrame に保つ）"""
    mask = np.ones(len(df), dtype=bool)
    for col, val in eq.items():
        mask &= df[col].to_numpy() == val
    return df.loc[mask]


def model_order_people() -> tuple[IntArr, IntArr, FloatArr]:
    """model.load_data の行順の ATUS 実（restore の入力と同じ並び）, (sched, d, w)"""
    cond_idx, sched, w, _ = sm.load_data()
    return (np.asarray(sched, dtype=np.int64), np.asarray(sm.cond_to_d(cond_idx), dtype=np.int64),
            np.asarray(w, dtype=np.float64))


def bootstrap_floor(people: tuple[IntArr, IntArr, FloatArr], pi_d: FloatArr,
                    n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """ATUS の回答者を復元抽出したときの、各指標の比（作り直し / 元）の sd

    Returns:
        index = 活動, columns = METRICS の sd の表
    """
    sched, d, w = people
    base = profile_table(sched, d, w, pi_d)[list(METRICS)].to_numpy()
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        i = rng.integers(0, len(sched), len(sched))
        ratios.append(profile_table(sched[i], d[i], w[i], pi_d)[list(METRICS)].to_numpy() / base)
    return pd.DataFrame(np.std(np.asarray(ratios), axis=0, ddof=1),
                        index=list(FOCUS_ACTS), columns=list(METRICS))


# ============================================================
# H5 逆過程のどの段階でずれるか / H6 最後の離散化
# ============================================================
def load_restore(arm: str, seed: int, n_rows: int) -> dict[str, np.ndarray]:
    """stage1_rare_diagnosis_gen --mode restore の結果を読む（ckpt より古ければ止める）"""
    path = _gen().restore_path(arm, seed)
    if not path.exists():
        raise FileNotFoundError(f"部分ノイズ化の結果が無い ({arm}, seed={seed}): {path}")
    rep.check_fresh(path, rep.ckpt_path(arm, seed))
    with np.load(path) as z:
        res = {k: z[k] for k in z.files}
    if int(res["n_rows"]) != n_rows:
        raise ValueError(f"入力の行数が違う: {int(res['n_rows'])} != {n_rows}（--limit 付きの結果?）")
    return res


def h5_restore(people: tuple[IntArr, IntArr, FloatArr], pi_d: FloatArr) -> pd.DataFrame:
    """t0 ごとの各指標の比（復元した個票 / 入力の実個票）

    Returns:
        縦持ち (arm, seed, t0, activity, metric, ratio)
    """
    sched, d, w = people
    base = profile_table(sched, d, w, pi_d)
    rows = []
    for arm in RESTORE_ARMS:
        for seed in DIAG_SEEDS:
            res = load_restore(arm, seed, len(sched))
            for k, t0 in enumerate(res["t0"]):
                prof = profile_table(res["sched_out"][k].astype(np.int64), d, w, pi_d)
                for a in FOCUS_ACTS:
                    for m in METRICS:
                        rows.append({"arm": arm, "seed": seed, "t0": int(t0), "activity": a,
                                     "metric": m, "ratio": prof.loc[a, m] / base.loc[a, m]})
    return pd.DataFrame(rows)


def h5_first_t0(long: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    """床（FLOOR_Z × 復元抽出の sd）を超える最小の t0（t*）と、その段階

    Returns:
        (arm, seed, activity, metric) ごとの floor, t_star（超えなければ NaN）,
        ratio_at_max_t0, stage（"長さ" / "中間" / "行動者" / "床の内"）
    """
    rows = []
    for arm in RESTORE_ARMS:
        for seed in DIAG_SEEDS:
            for a in FOCUS_ACTS:
                for m in METRICS:
                    g = _rows(long, arm=arm, seed=seed, activity=a, metric=m).sort_values("t0")
                    rows.append(_first_t0_row(g, arm, seed, a, m, FLOOR_Z * float(floor.loc[a, m])))
    return pd.DataFrame(rows)


def _first_t0_row(g: pd.DataFrame, arm: str, seed: int, a: str, m: str, lim: float) -> dict[str, Any]:
    """h5_first_t0 の 1 行（g は t0 の昇順）"""
    t0 = g["t0"].to_numpy()
    ratio = g["ratio"].to_numpy()
    over = np.flatnonzero(np.abs(ratio - 1.0) > lim)
    t_star = float(t0[over[0]]) if len(over) else float("nan")
    if np.isnan(t_star):
        stage = "床の内"
    elif t_star <= H5_T_LOCAL:
        stage = "長さ"
    elif t_star >= H5_T_GLOBAL:
        stage = "行動者"
    else:
        stage = "中間"
    return {"arm": arm, "seed": seed, "activity": a, "metric": m, "floor": lim,
            "t_star": t_star, "ratio_at_max_t0": float(ratio[-1]), "stage": stage}


def h6_decoding(people: tuple[IntArr, IntArr, FloatArr], pi_d: FloatArr) -> pd.DataFrame:
    """最大の t0（ほぼ純粋な雑音からの生成）で、argmax 前後の総量を比べる

    Returns:
        (arm, seed, activity) ごとの argmax_level（argmax 後）, soft_level（各スロットで和 1 に
        正規化した連続値）, raw_level（連続値そのもの）, rel_diff = soft / argmax − 1, supports_h6
    """
    sched, d, w = people
    rows = []
    for arm in RESTORE_ARMS:
        for seed in DIAG_SEEDS:
            res = load_restore(arm, seed, len(sched))
            k = int(np.argmax(res["t0"]))
            out = res["sched_out"][k].astype(np.int64)
            for a in FOCUS_ACTS:
                c = cur.ACT_NAMES.index(a)
                per_person = np.stack([(out == c).sum(axis=1), res["soft_sum"][k][:, c],
                                       res["raw_sum"][k][:, c]], axis=1) / sm.NUM_SLOTS
                am, soft, raw = group_weighted_mean(per_person.astype(np.float64), d, w, pi_d)
                rel = soft / am - 1.0
                rows.append({"arm": arm, "seed": seed, "activity": a, "argmax_level": am,
                             "soft_level": soft, "raw_level": raw, "rel_diff": rel,
                             "supports_h6": abs(rel) > H6_THRESHOLD})
    return pd.DataFrame(rows)


def load_final(arm: str, seed: int, n_rows: int) -> FloatArr:
    """stage1_rare_diagnosis_gen --mode final-continuous の連続値, (N, 12, 96)"""
    path = _gen().final_continuous_path(arm, seed)
    if not path.exists():
        raise FileNotFoundError(f"t0 = 999 の連続値が無い ({arm}, seed={seed}): {path}")
    rep.check_fresh(path, rep.ckpt_path(arm, seed))
    with np.load(path) as z:
        if int(z["n_rows"]) != n_rows:
            raise ValueError(f"入力の行数が違う: {int(z['n_rows'])} != {n_rows}")
        return np.asarray(z["x_final"], dtype=np.float64)


def h6_near_ties(people: tuple[IntArr, IntArr, FloatArr], pi_d: FloatArr) -> pd.DataFrame:
    """argmax で失われる分を「僅差の負け」と「漏れ」に分ける（H6 の測り方の補足）

    Note:
        ★h6_decoding の連続値の和は、全チャネルに薄く乗った値（漏れ）も数えるので、
          argmax が少ない活動を落としているかを直接は測れない。ここではスロットごとの値から、
          その活動の値が H6_TIE_VALUE 以上なのに argmax を取れなかったスロット（僅差の負け）を数える。
          判定は「僅差の負け / argmax 後の総量 > H6_THRESHOLD」

    Returns:
        (arm, seed, activity) ごとの real_level（入力の実個票）, argmax_level, tie_loss_level,
        leak_level, tie_over_argmax, ratio_argmax（argmax / 実）, ratio_with_ties
        （(argmax + 僅差の負け) / 実）, supports_h6
    """
    sched, d, w = people
    rows = []
    for arm in RESTORE_ARMS:
        for seed in DIAG_SEEDS:
            x = load_final(arm, seed, len(sched))                  # (N, 12, 96)
            am = x.argmax(axis=1)                                  # (N, 96)
            for a in FOCUS_ACTS:
                c = cur.ACT_NAMES.index(a)
                xc = x[:, c, :]
                per_person = np.stack([
                    (sched == c).sum(axis=1),                              # 入力の実個票
                    (am == c).sum(axis=1),                                 # argmax 後
                    ((am != c) & (xc >= H6_TIE_VALUE)).sum(axis=1),        # 僅差の負け
                    np.where(xc < LEAK_VALUE, np.clip(xc, 0.0, None), 0.0).sum(axis=1),   # 漏れ
                ], axis=1).astype(np.float64) / sm.NUM_SLOTS
                real, won, lost, leak = group_weighted_mean(per_person, d, w, pi_d)
                rows.append({"arm": arm, "seed": seed, "activity": a, "real_level": real,
                             "argmax_level": won, "tie_loss_level": lost, "leak_level": leak,
                             "tie_over_argmax": lost / won, "ratio_argmax": won / real,
                             "ratio_with_ties": (won + lost) / real,
                             "supports_h6": lost / won > H6_THRESHOLD})
    return pd.DataFrame(rows)


# ============================================================
# H7 CFG
# ============================================================
def h7_guidance(real_prof: pd.DataFrame, pi_d: FloatArr, h1: pd.DataFrame) -> pd.DataFrame:
    """clock_tf96 の guidance 1.0 と 1.25（同じ乱数の種）の総量の比を比べる

    Note:
        ★基準の「プール間の差」は H1 の値（学習時のプールと @g1.25、乱数が違う 2 本）。
          g1 と g1.25 は同じ乱数なので、この基準は保守的（差が出にくい側）

    Returns:
        活動ごとの種別の比, guidance の差の平均, プール間の差, 実データへ寄ったか, supports_h7
    """
    rows = []
    for a in FOCUS_ACTS:
        r1, r125 = [], []
        for seed in DIAG_SEEDS:
            base = real_prof.loc[a, "level"]
            r1.append(profile_table(*load_pool_people("clock_tf96@g1", seed), pi_d).loc[a, "level"] / base)
            r125.append(profile_table(*load_pool_people("clock_tf96@g1.25", seed), pi_d).loc[a, "level"] / base)
        g1, g125 = np.asarray(r1), np.asarray(r125)
        diff = float(np.abs(g1 - g125).mean())
        pool_diff = float(_rows(h1, arm="clock_tf96", activity=a)["pool_diff"].to_numpy()[0])
        closer = float(np.abs(np.log(g1)).mean()) < float(np.abs(np.log(g125)).mean())
        rows.append({"activity": a, "ratio_g1": "/".join(f"{v:.2f}" for v in g1),
                     "ratio_g1.25": "/".join(f"{v:.2f}" for v in g125),
                     "guidance_diff": diff, "pool_diff": pool_diff, "g1_closer_to_real": closer,
                     "supports_h7": diff > pool_diff and closer})
    return pd.DataFrame(rows)


# ============================================================
# H4 ckpt の選び方（epoch の軌跡）
# ============================================================
def h4_trajectory(real_prof: pd.DataFrame, pi_d: FloatArr,
                  best_epochs: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """途中の ckpt の小プールで、総量の比を epoch に沿って並べる

    Note:
        ★途中の ckpt は SQUID にだけあるので、ここでは mtime のガードをかけられない。
          ガードは生成側（stage1_rare_diagnosis_gen の is_fresh）で行っている

    Args:
        real_prof: ATUS 実の profile_table
        pi_d: 群の間の重み
        best_epochs: 種 → 最良 epoch（学習ログの "restored best checkpoint: epoch N"）

    Returns:
        (縦持ち (seed, epoch, activity, is_best, ratio)。is_best=True の行は最良の ckpt で、
         epoch は最良 epoch。活動ごとの要約:
         within_sd = 最良 epoch ± H4_WINDOW の種内 sd の種平均,
         between_sd = 最良の ckpt の小プールの比の種間 sd, supports_h4)
    """
    gen = _gen()
    rows = []
    best_rows = []
    for seed in DIAG_SEEDS:
        pools = gen.epoch_pools(H4_ARM, seed, EPOCH_POOL_N)
        if not pools:
            raise FileNotFoundError(f"途中の ckpt の小プールが無い ({H4_ARM}, seed={seed})")
        for epoch, path in pools.items():
            prof = profile_table(*pool_people(cur.load_sample_pool(path)), pi_d)
            for a in FOCUS_ACTS:
                rows.append({"seed": seed, "epoch": epoch, "activity": a, "is_best": False,
                             "ratio": prof.loc[a, "level"] / real_prof.loc[a, "level"]})
        best = gen.epoch_pool_path(H4_ARM, seed, None, EPOCH_POOL_N)
        if not best.exists():
            raise FileNotFoundError(f"最良の ckpt の小プールが無い: {best}")
        prof = profile_table(*pool_people(cur.load_sample_pool(best)), pi_d)
        for a in FOCUS_ACTS:
            best_rows.append({"seed": seed, "epoch": best_epochs[seed], "activity": a, "is_best": True,
                              "ratio": prof.loc[a, "level"] / real_prof.loc[a, "level"]})
    long = pd.DataFrame(rows)
    best_df = pd.DataFrame(best_rows)
    summary = []
    for a in FOCUS_ACTS:
        within = []
        for seed in DIAG_SEEDS:
            g = _rows(long, activity=a, seed=seed)                 # 途中の ckpt だけ（最良は別表）
            near = np.abs(g["epoch"].to_numpy() - best_epochs[seed]) <= H4_WINDOW
            within.append(float(np.std(g["ratio"].to_numpy()[near], ddof=1)))
        best_ratio = _rows(best_df, activity=a)["ratio"].to_numpy()
        between = float(np.std(best_ratio, ddof=1))
        summary.append({"activity": a, "within_sd": float(np.mean(within)),
                        "within_sd_by_seed": "/".join(f"{v:.2f}" for v in within),
                        "between_sd": between,
                        "best_ratio_by_seed": "/".join(f"{v:.2f}" for v in best_ratio),
                        "supports_h4": float(np.mean(within)) >= between})
    return pd.concat([long, best_df], ignore_index=True), pd.DataFrame(summary)


# ============================================================
# H8 Transformer 型と長時間の介護（種 5 本）
# ============================================================
def h8_seeds(real_prof: pd.DataFrame, pi_d: FloatArr) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K=4 と Transformer 型を種 5 本ずつで比べる（学習時のプール）

    Returns:
        (縦持ち (arm, seed, activity, metric, value, ratio),
         (活動, 指標) ごとの 2 arm の範囲と完全分離の有無)
    """
    rows = []
    for arm in RESTORE_ARMS:
        for seed in H8_SEEDS:
            prof = profile_table(*load_pool_people(arm, seed), pi_d)
            for a in FOCUS_ACTS:
                for m in METRICS:
                    rows.append({"arm": arm, "seed": seed, "activity": a, "metric": m,
                                 "value": prof.loc[a, m], "ratio": prof.loc[a, m] / real_prof.loc[a, m]})
    long = pd.DataFrame(rows)
    summary = []
    for a in FOCUS_ACTS:
        for m in METRICS:
            summary.append(_separation_row(long, real_prof, a, m))
    return long, pd.DataFrame(summary)


def _separation_row(long: pd.DataFrame, real_prof: pd.DataFrame, a: str, m: str) -> dict[str, Any]:
    """h8_seeds の 1 行: 2 arm の範囲と完全分離"""
    k4 = _rows(long, arm="clock", activity=a, metric=m)["value"].to_numpy()
    tf = _rows(long, arm="clock_tf96", activity=a, metric=m)["value"].to_numpy()
    return {"activity": a, "metric": m, "real": float(real_prof.loc[a, m]),
            "clock_min": float(k4.min()), "clock_max": float(k4.max()),
            "tf96_min": float(tf.min()), "tf96_max": float(tf.max()),
            "separated": bool(k4.max() < tf.min() or tf.max() < k4.min())}


# ============================================================
# 図（タイトルは名前だけ、数値は表で渡す）
# ============================================================
def _figure_module() -> Any:
    """色と日本語フォントの設定を stage1_ablation_curves と共有する"""
    return _load("rare_diag_curves",
                 REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage1_ablation_curves.py")


def plot_restore(long: pd.DataFrame, floor: pd.DataFrame, out: Path) -> None:
    """t0 と総量の比の関係（arm ごとに種の平均と範囲、灰色の帯は床）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = _figure_module()
    fm.setup_fonts()
    lv = _rows(long, metric="level")
    t0s = sorted(int(t) for t in np.unique(lv["t0"].to_numpy()))
    xs = np.arange(len(t0s))
    fig, axes = plt.subplots(1, len(FOCUS_ACTS), figsize=(17, 3.9), sharey=False)
    for ax, a in zip(axes, FOCUS_ACTS):
        lim = FLOOR_Z * float(floor.loc[a, "level"])
        ax.axhspan(1 - lim, 1 + lim, color="#d9d9d6", alpha=0.6, lw=0)
        ax.axhline(1.0, color=fm.COLOR_ATUS, lw=1.2)
        for color, arm in zip(fm.ARM_COLORS, RESTORE_ARMS):
            g = _rows(lv, activity=a, arm=arm)
            piv = g.pivot_table(index="t0", columns="seed", values="ratio").reindex(t0s)
            ax.fill_between(xs, piv.min(axis=1), piv.max(axis=1), color=color, alpha=0.18, lw=0)
            ax.plot(xs, piv.mean(axis=1), color=color, lw=2, marker="o", ms=4, label=ARM_LABELS[arm])
        ax.set_xticks(xs, [str(t) for t in t0s], fontsize=8)
        ax.set_title(f"{cur.ACT_JA[a]}（{a}）", fontsize=11)
        ax.set_xlabel("雑音化の水準 t0", fontsize=9)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("総量の比（復元 / 実）", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(labels),
               frameon=False, fontsize=10)
    fig.suptitle("部分ノイズ化からの復元", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rare_diag] 図: {out}")


def plot_trajectory(long: pd.DataFrame, best_epochs: dict[int, int], out: Path) -> None:
    """epoch と総量の比の関係（種ごとの線は途中の ckpt、丸は最良の ckpt を最良 epoch の位置に）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = _figure_module()
    fm.setup_fonts()
    fig, axes = plt.subplots(1, len(FOCUS_ACTS), figsize=(17, 3.9))
    for ax, a in zip(axes, FOCUS_ACTS):
        ax.axhline(1.0, color=fm.COLOR_ATUS, lw=1.2)
        for color, seed in zip(fm.ARM_COLORS, DIAG_SEEDS):
            g = _rows(long, activity=a, seed=seed, is_best=False).sort_values("epoch")
            ax.plot(g["epoch"], g["ratio"], color=color, lw=1.8, label=f"種 {seed}")
            b = _rows(long, activity=a, seed=seed, is_best=True)
            ax.plot([best_epochs[seed]], b["ratio"].to_numpy(), "o", color=color, ms=8,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set_title(f"{cur.ACT_JA[a]}（{a}）", fontsize=11)
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("総量の比（生成 / 実）", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(labels),
               frameon=False, fontsize=10)
    fig.suptitle("学習の途中の ckpt", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rare_diag] 図: {out}")


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


def _japan_setup() -> tuple[FloatArr, pd.DataFrame]:
    """日本人口の群の重みと、ATUS 実（load_atus_weekday の並び）の profile_table"""
    tgt = cur.st.load_stula_targets()
    sched_r, d_r, w_r = agr.load_atus_weekday()
    pi_d = group_weights("japan", tgt, d_r, w_r)
    return pi_d, profile_table(sched_r, d_r, w_r, pi_d)


def _show(title: str, df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400,
                           "display.float_format", "{:.4f}".format):
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))


def run_restore() -> None:
    """段 B の H5・H6 を集計し、表と図を書く"""
    pi_d, _ = _japan_setup()
    people = model_order_people()
    floor = bootstrap_floor(people, pi_d)
    long = h5_restore(people, pi_d)
    first = h5_first_t0(long, floor)
    dec = h6_decoding(people, pi_d)
    _show(f"床（ATUS の復元抽出での比の sd × {FLOOR_Z:g}）",
          (floor * FLOOR_Z).reset_index(names="activity"))
    _show("H5 t0 ごとの総量の比（種の平均）",
          _rows(long, metric="level").pivot_table(index=["activity", "arm"], columns="t0",
                                                  values="ratio").reset_index())
    _show(f"H5 床を超える最小の t0（t* ≤ {H5_T_LOCAL}: 長さ、t* ≥ {H5_T_GLOBAL}: 行動者）",
          first.loc[first["metric"].isin(["level", "doer_share", "slots_per_doer",
                                          "long_doer_share"]).to_numpy()])
    _show(f"H6 argmax 前後の総量（支持: |相対差| > {H6_THRESHOLD:g}）", dec)
    long.to_csv(OUT_DIR / "stage1_rare_diag_h5_long.csv", index=False)
    first.to_csv(OUT_DIR / "stage1_rare_diag_h5_tstar.csv", index=False)
    dec.to_csv(OUT_DIR / "stage1_rare_diag_h6.csv", index=False)
    floor.to_csv(OUT_DIR / "stage1_rare_diag_floor.csv")
    plot_restore(long, floor, FIG_DIR / "stage1_rare_restore.png")
    print("[rare_diag] 書いた: stage1_rare_diag_{h5_long,h5_tstar,h6,floor}.csv")


def run_final() -> None:
    """H6 の補足（僅差の負けと漏れ）を集計する"""
    pi_d, _ = _japan_setup()
    ties = h6_near_ties(model_order_people(), pi_d)
    _show(f"H6 補足: 僅差の負け（値 ≥ {H6_TIE_VALUE:g} で argmax を取れない）と漏れ（値 < {LEAK_VALUE:g}）"
          f"（支持: 僅差の負け / argmax > {H6_THRESHOLD:g}）", ties)
    ties.to_csv(OUT_DIR / "stage1_rare_diag_h6_ties.csv", index=False)


def run_guidance() -> None:
    """H7 を集計する（段 A の H1 の CSV を基準に使う）"""
    pi_d, real_prof = _japan_setup()
    h1_path = OUT_DIR / "stage1_rare_diag_h1.csv"
    if not h1_path.exists():
        raise FileNotFoundError(f"先に --part existing を実行すること: {h1_path}")
    h7 = h7_guidance(real_prof, pi_d, pd.read_csv(h1_path))
    _show("H7 CFG（支持: guidance の差 > プール間の差 かつ g1 が実データに近い）", h7)
    h7.to_csv(OUT_DIR / "stage1_rare_diag_h7.csv", index=False)


def run_trajectory(best_epochs: dict[int, int]) -> None:
    """段 C の H4 を集計し、表と図を書く"""
    pi_d, real_prof = _japan_setup()
    long, summary = h4_trajectory(real_prof, pi_d, best_epochs)
    _show("H4 途中の ckpt ごとの総量の比",
          _rows(long, is_best=False).pivot_table(index=["activity", "seed"], columns="epoch",
                                                 values="ratio").reset_index())
    _show(f"H4 ckpt の選び方（支持: 最良 epoch ±{H4_WINDOW} の種内 sd ≥ 種間 sd）", summary)
    long.to_csv(OUT_DIR / "stage1_rare_diag_h4_long.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_rare_diag_h4.csv", index=False)
    plot_trajectory(long, best_epochs, FIG_DIR / "stage1_rare_trajectory.png")


def run_seeds() -> None:
    """段 C の H8 を集計する"""
    pi_d, real_prof = _japan_setup()
    long, summary = h8_seeds(real_prof, pi_d)
    _show("H8 K=4 と Transformer 型（種 5 本の範囲、separated = 完全分離）", summary)
    long.to_csv(OUT_DIR / "stage1_rare_diag_h8_long.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_rare_diag_h8.csv", index=False)


def parse_best_epochs(items: list[str]) -> dict[int, int]:
    """["42:612", ...] → {42: 612, ...}。DIAG_SEEDS が全て揃っていること"""
    out: dict[int, int] = {}
    for item in items:
        seed, _, epoch = item.partition(":")
        out[int(seed)] = int(epoch)
    missing = [s for s in DIAG_SEEDS if s not in out]
    if missing:
        raise ValueError(f"--best-epochs に無い種: {missing}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("existing", "restore", "final", "guidance", "trajectory", "seeds"),
                    required=True, help="集計する部（モジュールの説明を参照）")
    ap.add_argument("--best-epochs", nargs="+", default=None,
                    help="trajectory 用。種:最良 epoch（学習ログの restored best checkpoint）")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.part == "existing":
        run_existing()
    elif args.part == "restore":
        run_restore()
    elif args.part == "final":
        run_final()
    elif args.part == "guidance":
        run_guidance()
    elif args.part == "trajectory":
        if args.best_epochs is None:
            ap.error("--part trajectory には --best-epochs 42:E 43:E 44:E が要る")
        run_trajectory(parse_best_epochs(args.best_epochs))
    else:
        run_seeds()


if __name__ == "__main__":
    main()
