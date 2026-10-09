"""
stage2_teacher_fit.py
=====================
教師適合（軸1）を**曲線**と **MSE** で読む。教師 A*・ATUS実データ・Pre-trained・Fine-tuned の
4 本を、全体（28 群の STULA 人口加重平均）と 28 群それぞれについて描き、MSE を表にする。
ATUS実データ は参照として図にだけ載せ、MSE の表には入れない（群あたり 17〜329 人で、
標本の揺れ p(1−p)/n が n=1000 の生成と同列に比べられないため）。
★全体の図で 28 群をまとめる重みは系列で違う。教師A*・Pre-trained・Fine-tuned は STULA の
  人口シェア pi_d、ATUS実データ は ATUS の調査ウェイト TUFINLWGT だけ（米国の人口構成）を使う。

`stage2_curves.py` は全体の曲線だけを描き、群ごとの図と MSE の表を持たない。
このモジュールは同じ部品（`pool_to_slot_rates` 以外の読み込み・加重・時刻軸）を
`stage2_curves` から借りて、群ごとの図と MSE の表を足す。

入力は `stage2_select.dump_rates` が書く .npz（`jobs/dump_rates_stage2.sh`）か、
Stage 1 の学習直後に `model.sanity_check` が書く生成サンプル CSV（群あたり 256 本）である。
Pre-trained と Fine-tuned は**同じ n・同じ pool_seed** で生成したものを渡すこと。
n が違うと MSE に含まれる生成側の MC 分散 p(1−p)/n が揃わず、差がモデルの差でなくなる。
既定では止める。暫定で比べるときだけ `--allow-n-mismatch` を付け、`mse_split_*` 列を併せて読む。

処理の流れ:

```mermaid
flowchart TD
    TGT["st.load_stula_targets()<br/>tgt['group_rates_tbl'] = a_star (28,12,96)<br/>tgt['pop'] (2,7,2) → pi_d (28,)<br/>tgt['group_rates_var'] (28,12,96)"]
    NPZ_P["--pretrained .npz または .csv<br/>npz: rates / rates_a / rates_b (28, 12*96), ckpt, meta<br/>csv: group_d, s0..s95（--pretrained-ckpt で mtime 照合）"]
    NPZ_F["--finetuned .npz"]

    NPZ_P --> LP["load_pool<br/>.npz → load_pool_npz / .csv → load_pool_csv<br/>→ PoolRates"]
    NPZ_F --> LP
    LP --> NCHK{"n / pool_seed が揃っているか<br/>揃っていなければ --allow-n-mismatch が無い限り止める"}
    NCHK --> CHK
    CHK["check_provenance<br/>st.eval_against の rate_mse を<br/>--check-csv の記録値と照合"]
    TGT --> CHK

    ATUS["agr.load_atus_weekday()<br/>sched (3736,96) / groups / w = TUFINLWGT"]
    ATUS --> AR["load_atus_rates<br/>agr.group_rates(sched, groups, w) → atus_rates (28,12,96)<br/>agr.group_counts(groups, w) → atus_share (28,)"]
    AR --> AW["einsum(atus_share, atus_rates)<br/>ATUS実データ の全体曲線 (12,96)<br/>★pi_d ではなく ATUS 重みだけで畳む"]
    AW --> FIG0
    AR --> FIGD

    TGT --> W["cv.weighted_slot_rates<br/>全体曲線 (12,96)"]
    LP --> W
    W --> FIG0["plot_rates_figure<br/>全体の図 overall.png"]
    LP --> FIGD["plot_rates_figure<br/>群 d の図 group_dd.png (28枚)"]
    TGT --> FIGD

    LP --> MSE["mse_table<br/>群別 MSE_d (28,)<br/>全体 3 定義"]
    TGT --> FL["teacher_floor_by_group<br/>教師A*の標本誤差の床"]
    FL --> MSE
    MSE --> CSV["--out-csv"]
    FIG0 --> PDF["all_figures.pdf（29 ページ）"]
    FIGD --> PDF
```

MSE の定義（`mask_c` は採点する活動。既定は 12 活動すべて）:

    群 d の MSE      MSE_d        = mean_{c,s} (rates[d,c,s] − a_star[d,c,s])²
    全セル（群等重み） MSE_equal    = mean_d MSE_d            ＝ stage2_select の rate_mse
    全セル（人口加重） MSE_pop      = Σ_d pi_d · MSE_d
    全体曲線          MSE_curve    = mean_{c,s} (Σ_d pi_d rates[d,c,s] − Σ_d pi_d a_star[d,c,s])²

★MSE_d は生成側の MC 分散 mean_{c,s} p(1−p)/n を床として含む。これを除いた
  split-batch 版（`mse_split_*` 列）も併記する。定義は `st.eval_against` の
  `rate_mse_split` と同じで、群内の前半 n/2・後半 n/2 の誤差の積の平均である。

使い方:
    python src/models/DDPM_Aggregate_Simple/stage2_teacher_fit.py \\
        --pretrained outputs/generated/ddpm_simple_pretrain_common12_weekday_20260819_rates.npz \\
        --finetuned  outputs/generated/stage2_step200_rates.npz \\
        --out-dir    outputs/figures/teacher_fit \\
        --out-csv    data/processed/aggregates/stage2_teacher_fit_mse.csv

    # 暫定: Pre-trained に学習直後の 256 本 CSV を使う（Fine-tuned は n=1000）
    python src/models/DDPM_Aggregate_Simple/stage2_teacher_fit.py \\
        --pretrained outputs/generated/ddpm_simple_pretrain_samples_20260819.csv \\
        --pretrained-ckpt outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_20260819.pt \\
        --finetuned  outputs/generated/stage2_step200_rates.npz \\
        --allow-n-mismatch \\
        --out-dir    outputs/figures/teacher_fit_pretrained_n256 \\
        --out-csv    data/processed/aggregates/stage2_teacher_fit_mse_pretrained_n256.csv
"""
import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる。既に同じファイルが同じ名前で入っていれば使い回す。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ★torch を読まない。描画と採点は numpy だけで済む（stage2_curves と同じ方針）
st: Any = _load("curves_stage2_targets", HERE / "stage2_targets.py")
cv: Any = _load("fit_stage2_curves", HERE / "stage2_curves.py")
agr: Any = _load("fit_atus_group_rates", REPO_ROOT / "src" / "eval" / "atus_group_rates.py")

# 系列ラベル。教師A* / Pre-trained / Fine-tuned は図の凡例と表の列名の両方で、
# ATUS実データ は図の凡例だけで使う（MSE の表には載せない）
LABEL_TEACHER = "教師A*"
LABEL_PRETRAINED = "Pre-trained"
LABEL_FINETUNED = "Fine-tuned"
LABEL_ATUS = "ATUS実データ"          # stage2_curves.ATUS_REAL_LABEL と同じ語

# 系列の色。教師は正解なので黒の太線、生成 2 本は検証済みの 2 色（青・橙）
COLOR_TEACHER = "#1a1a19"
COLOR_PRETRAINED = "#2a78d6"
COLOR_FINETUNED = "#eb6834"
# ATUS実データ は配色の 3 番目（緑）の濃い段。明るい段 #1baf7a は白地とのコントラストが
# 2.74:1 で 3:1 に届かず、細線だと読みにくい。#199e70 は 3 色の全組で色覚検証を通る
COLOR_ATUS = "#199e70"

# 活動の英語名。src/viz/aggddpm_architecture.py の ACT_NAMES_EN と同じ語にする
ACT_EN: dict[str, str] = {
    "SLEEP_PERSONAL": "Sleep/personal",
    "MEALS": "Meals",
    "WORK": "Work",
    "SCHOOL": "School",
    "HOUSEWORK": "Housework",
    "CAREGIVING": "Care",
    "SHOPPING": "Shopping",
    "TRAVEL": "Travel",
    "LEISURE_SOCIAL": "Leisure/social",
    "SPORTS": "Sports",
    "VOLUNTEER": "Volunteer",
    "OTHER_X": "Other",
}

# 群の属性ラベル。添字は stage2_targets.d_index の (g, a, e) と同じ
GENDER_JA = ("男", "女")                                   # g: 0=男, 1=女
AGE7_JA = ("15〜24歳", "25〜34歳", "35〜44歳", "45〜54歳",
           "55〜64歳", "65〜74歳", "75歳以上")                # a: AGE15_TO_7 の 10 歳 7 区分
EMPLOYMENT_JA = ("無業者", "有業者")                        # e: 0=無業者, 1=有業者

DEFAULT_CHECK_CSV = REPO_ROOT / "data/processed/aggregates/stage2_lam0.003_selection.csv"


@dataclass(frozen=True)
class PoolRates:
    """1 モデルぶんの生成プールから作った時刻別行動者率。

    Attributes:
        label: 系列ラベル（LABEL_PRETRAINED か LABEL_FINETUNED）
        rates: 群別の時刻別行動者率, dtype=float64, (28, 12, 96)
        rates_a: 群内の前半 n/2 本から作った率, (28, 12, 96)
        rates_b: 群内の後半 n/2 本から作った率, (28, 12, 96)
        ckpt: 生成に使ったチェックポイントのファイル名
        step: Stage 2 の更新回数。Stage 1 の重みなら 0
        n: 群あたりの生成本数
        pool_seed: 生成の乱数種
    """
    label: str
    rates: npt.NDArray[np.float64]
    rates_a: npt.NDArray[np.float64]
    rates_b: npt.NDArray[np.float64]
    ckpt: str
    step: int
    n: int
    pool_seed: int


def load_pool_npz(path: Path, label: str) -> PoolRates:
    """`stage2_select.dump_rates` が書いた .npz を読む。

    Args:
        path: .npz のパス。`rates` / `rates_a` / `rates_b` / `meta` / `ckpt` を持つこと
        label: 系列ラベル

    Returns:
        読み込んだ時刻別行動者率と出所の情報

    Raises:
        ValueError: 必要なキーが無い場合
    """
    shape = (st.D_GROUPS, st.NUM_COMMON, st.NUM_SLOTS)
    with np.load(path) as z:
        missing = [k for k in ("rates", "rates_a", "rates_b", "meta", "ckpt") if k not in z]
        if missing:
            raise ValueError(f"{path} にキー {missing} が無い（dump_rates の出力ではない）")
        step, n, pool_seed = (int(v) for v in z["meta"])
        return PoolRates(
            label=label,
            rates=np.asarray(z["rates"], dtype=np.float64).reshape(shape),
            rates_a=np.asarray(z["rates_a"], dtype=np.float64).reshape(shape),
            rates_b=np.asarray(z["rates_b"], dtype=np.float64).reshape(shape),
            ckpt=str(z["ckpt"]), step=step, n=n, pool_seed=pool_seed)


def load_pool_csv(path: Path, label: str, ckpt_path: Path | None = None) -> PoolRates:
    """生成サンプル CSV（`group_d`, `s0`..`s95`）を読み、時刻別行動者率を作る。

    Stage 1 の学習直後に `model.sanity_check` が書く CSV（群あたり 256 本）を
    そのまま使うための入口である。.npz と違って由来の重みも乱数種も記録されていない。

    Note:
        ★`ckpt_path` を渡すと mtime を照合する。CSV が重みより古ければ、別の重みで
          作ったプールなので止める（2026-08-03 に古いプールを 3 日間使った前例がある）。
        ★乱数種は記録されていないので `pool_seed=-1` にする。`check_provenance` は
          対応行を見つけられず、照合は行われない。
        ★split-batch 用の前半・後半は、群の中でファイル上の並び順に二分する。
          群内のサンプルは互いに独立なので、どこで切っても 2 つの半分は独立である。

    Args:
        path: 生成サンプル CSV のパス
        label: 系列ラベル
        ckpt_path: 生成に使った重み。渡せば mtime を照合する, default=None

    Returns:
        読み込んだ時刻別行動者率と出所の情報。step は 0（Stage 1）とする

    Raises:
        ValueError: 群あたりの本数が奇数の場合、または CSV が重みより古い場合
    """
    if ckpt_path is not None and path.stat().st_mtime < ckpt_path.stat().st_mtime:
        raise ValueError(f"{path.name} が {ckpt_path.name} より古い。別の重みで作ったプールである")
    pool = cv.load_sample_pool(path)                                 # (28, M, 96)
    m = int(pool.shape[1])
    if m % 2 != 0:
        raise ValueError(f"split-batch には群あたりの本数が偶数である必要がある: {m}")
    half = m // 2
    return PoolRates(
        label=label,
        rates=cv.pool_to_slot_rates(pool),
        rates_a=cv.pool_to_slot_rates(pool[:, :half]),
        rates_b=cv.pool_to_slot_rates(pool[:, half:]),
        ckpt=ckpt_path.name if ckpt_path is not None else path.name,
        step=0, n=m, pool_seed=-1)


def load_pool(path: Path, label: str, ckpt_path: Path | None = None) -> PoolRates:
    """拡張子で .npz（dump_rates）と .csv（生成サンプル）を振り分けて読む"""
    if path.suffix == ".npz":
        return load_pool_npz(path, label)
    return load_pool_csv(path, label, ckpt_path)


def check_provenance(rec: PoolRates, tgt: dict, csv_path: Path) -> float | None:
    """rate_mse を事後選択 CSV の記録値と照合し、プールの出所を確かめる。

    Note:
        ★`dump_rates` と `stage2_select.run` は同じ `make_pool` を同じ乱数種で通るので、
          同じチェックポイントなら rate_mse は**厳密に一致する**。一致しなければ別の
          重みか別の乱数で作ったプールであり、そのまま描くと取り違えになる
          （2026-08-03 に古いプールを 3 日間使った前例がある）。
        ★CSV に対応行が無ければ照合できない。そのときは None を返して続ける。

    Args:
        rec: 照合するプール
        tgt: `st.load_stula_targets` の戻り値
        csv_path: `stage2_select` が書いた縦持ち CSV

    Returns:
        照合に使った記録値。対応行が無ければ None

    Raises:
        ValueError: 記録値と一致しない場合
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    hit = cast(pd.DataFrame, df[(df["ckpt"] == rec.ckpt) & (df["step"] == rec.step)
                                & (df["n_per_group"] == rec.n) & (df["pool_seed"] == rec.pool_seed)
                                & (df["eval_kind"] == "in-teacher") & (df["mask"] == "12act")
                                & (df["metric"] == "rate_mse")])
    if hit.empty:
        return None
    recorded = float(hit["value"].to_numpy()[0])
    flat = rec.rates.reshape(st.D_GROUPS, -1)
    got = float(st.eval_against(flat, tgt, st.mask_12act())["rate_mse"])
    if not np.isclose(got, recorded, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"{rec.label}: rate_mse {got:.9f} が記録値 {recorded:.9f}（{csv_path.name}）と一致しない。\n"
            f"       {rec.ckpt} step={rec.step} とは別のプールである")
    return recorded


def load_atus_rates() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ATUS 2024 平日の実データから群別の時刻別行動者率と、ATUS 重みによる群シェアを作る。

    Note:
        ★集計は `src/eval/atus_group_rates.py` が唯一の出所で、ここでは呼ぶだけにする。
          群への割り当て・12 分類・04:00 起点 96 スロット・平日の規約は教師A* と同じ。
        ★群の中は ATUS の調査ウェイト TUFINLWGT で加重する。群あたりの標本は 17〜329 人
          なので、小さい群の曲線は 1/n 刻みで粗い。
        ★群シェアも TUFINLWGT だけから作る（群ごとの重みの和 / 全体の和）。これで 28 群を
          畳むと、ATUS 平日の全回答者を TUFINLWGT で加重平均した曲線と一致する
          （米国の人口構成）。STULA の pi_d は使わない。
          `stage2_curves.atus_real_curve` は pi_d で畳むので、全体曲線はそちらと別の量になる。

    Returns:
        (atus_rates, atus_share)。atus_rates は群別の時刻別行動者率, dtype=float64,
        (D_GROUPS, NUM_COMMON, NUM_SLOTS)。atus_share は TUFINLWGT による群シェア,
        dtype=float64, (D_GROUPS,)、和は 1

    Raises:
        ValueError: 標本の無い群がある場合（その群の曲線が NaN になり描けない）
    """
    sched, groups, w = agr.load_atus_weekday()
    rates = np.asarray(agr.group_rates(sched, groups, w), dtype=np.float64)
    if np.isnan(rates).any():
        empty = sorted({int(d) for d in np.argwhere(np.isnan(rates))[:, 0]})
        raise ValueError(f"ATUS 平日に標本の無い群がある: {empty}")
    _, wsum, _ = agr.group_counts(groups, w)
    share = np.asarray(wsum, dtype=np.float64) / float(np.sum(wsum))
    return rates, share


def group_label(d: int) -> str:
    """群インデックス d → 「男，15〜24歳，有業者」の形のラベル"""
    g, rest = divmod(d, st.N_A * st.N_E)
    a, e = divmod(rest, st.N_E)
    return f"{GENDER_JA[g]}，{AGE7_JA[a]}，{EMPLOYMENT_JA[e]}"


def pop_shares(tgt: dict) -> npt.NDArray[np.float64]:
    """STULA の推定人口から群の人口シェア pi_d を作る, -> (28,)。和は 1"""
    pop = np.asarray(tgt["pop"], dtype=np.float64).reshape(st.D_GROUPS)
    return pop / pop.sum()


def mse_by_group(rates: npt.NDArray[np.float64], a_star: npt.NDArray[np.float64],
                 mask_c: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """群ごとの MSE, -> (28,)。MSE_d = mean_{c,s} (rates − a_star)²（c は mask_c の活動）"""
    err = (rates - a_star)[:, mask_c, :]
    return np.asarray((err ** 2).mean(axis=(1, 2)), dtype=np.float64)


def split_mse_by_group(rec: PoolRates, a_star: npt.NDArray[np.float64],
                       mask_c: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """群ごとの split-batch MSE, -> (28,)。生成側の MC 分散を除いた bias² の不偏推定

    mean_{c,s} (rates_a − a_star)(rates_b − a_star)。負になりうる（`st.eval_against` の
    rate_mse_split と同じ定義）。
    """
    prod = ((rec.rates_a - a_star) * (rec.rates_b - a_star))[:, mask_c, :]
    return np.asarray(prod.mean(axis=(1, 2)), dtype=np.float64)


def teacher_floor_by_group(tgt: dict, mask_c: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """教師A* 自身の標本誤差が作る MSE の床（群ごと）, -> (28,)

    `mean_{c,s} Var(a_star[d,c,s])`。モデルが真の率を当てても採点に残る量で、
    `st.teacher_floor` の rate_mse を群ごとに分けたもの（群で平均すると一致する）。
    """
    var = np.asarray(tgt["group_rates_var"], dtype=np.float64)[:, mask_c, :]
    return np.asarray(var.mean(axis=(1, 2)), dtype=np.float64)


def mse_table(recs: list[PoolRates], tgt: dict,
              mask_c: npt.NDArray[np.bool_]) -> pd.DataFrame:
    """群別 28 行と全体 3 行の MSE 表を作る。

    全体の 3 行の定義はモジュール docstring の MSE_equal / MSE_pop / MSE_curve。
    教師A* の列は MSE ではなく**教師の標本誤差の床**（同じ定義で取った期待値）である。

    Args:
        recs: Pre-trained と Fine-tuned のプール
        tgt: `st.load_stula_targets` の戻り値
        mask_c: 採点する活動, dtype=bool, (12,)

    Returns:
        列 scope / d / group / pop_share / mse_教師A*の床 / mse_<label> / mse_split_<label> /
        change_pct の DataFrame。change_pct は Pre-trained → Fine-tuned の MSE の変化率
    """
    a_star = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    pi = pop_shares(tgt)
    floor = teacher_floor_by_group(tgt, mask_c)
    var = np.asarray(tgt["group_rates_var"], dtype=np.float64)[:, mask_c, :]

    rows: list[dict[str, Any]] = []
    for d in range(st.D_GROUPS):
        rows.append({"scope": "group", "d": d, "group": group_label(d),
                     "pop_share": float(pi[d]), f"mse_{LABEL_TEACHER}の床": float(floor[d])})
    overall = [
        {"scope": "overall", "d": -1, "group": "全セル（群等重み）", "pop_share": 1.0,
         f"mse_{LABEL_TEACHER}の床": float(floor.mean())},
        {"scope": "overall", "d": -1, "group": "全セル（人口加重）", "pop_share": 1.0,
         f"mse_{LABEL_TEACHER}の床": float((pi * floor).sum())},
        # 全体曲線の床: Var(Σ_d pi_d a_star_d) = Σ_d pi_d² Var_d（層は互いに素な標本で独立）
        {"scope": "overall", "d": -1, "group": "全体曲線（人口加重平均の曲線）", "pop_share": 1.0,
         f"mse_{LABEL_TEACHER}の床": float((np.einsum("d,dcs->cs", pi ** 2, var)).mean())},
    ]

    teacher_curve = np.einsum("d,dcs->cs", pi, a_star)[mask_c]
    for rec in recs:
        mse_d = mse_by_group(rec.rates, a_star, mask_c)
        split_d = split_mse_by_group(rec, a_star, mask_c)
        for d in range(st.D_GROUPS):
            rows[d][f"mse_{rec.label}"] = float(mse_d[d])
            rows[d][f"mse_split_{rec.label}"] = float(split_d[d])
        overall[0][f"mse_{rec.label}"] = float(mse_d.mean())
        overall[0][f"mse_split_{rec.label}"] = float(split_d.mean())
        overall[1][f"mse_{rec.label}"] = float((pi * mse_d).sum())
        overall[1][f"mse_split_{rec.label}"] = float((pi * split_d).sum())
        gen_curve = np.einsum("d,dcs->cs", pi, rec.rates)[mask_c]
        gen_a = np.einsum("d,dcs->cs", pi, rec.rates_a)[mask_c]
        gen_b = np.einsum("d,dcs->cs", pi, rec.rates_b)[mask_c]
        overall[2][f"mse_{rec.label}"] = float(((gen_curve - teacher_curve) ** 2).mean())
        overall[2][f"mse_split_{rec.label}"] = float(
            ((gen_a - teacher_curve) * (gen_b - teacher_curve)).mean())

    out = pd.DataFrame(rows + overall)
    before, after = out[f"mse_{LABEL_PRETRAINED}"], out[f"mse_{LABEL_FINETUNED}"]
    out["change_pct"] = (after - before) / before * 100.0
    out.insert(0, "mask", "12act" if bool(mask_c.all()) else "11act")
    return out


def _hour_label(h: float) -> str:
    """04:00 起点の 0-36 時間軸の値 → 時計表示（28 → "4:00"）"""
    return f"{int(h) % 24}:00"


def plot_rates_figure(ax_grid: Any, curves: dict[str, npt.NDArray[np.float64]]) -> None:
    """12 活動の時刻別行動者率を 4x3 の小倍数の各軸へ描く。

    Note:
        ★活動ごとに y 軸を独立させる。睡眠は 1.0 近くまで行き、ボランティアは
          0.01 未満なので、共通軸にすると小さい活動が潰れる（stage2_curves と同じ）。
        ★curves の順に描く。教師A* を先に太線で描き、ATUS実データ・生成 2 本を上に
          細線で重ねる。重なっても教師が見える。凡例の並びもこの順になる。

    Args:
        ax_grid: plt.subplots(4, 3) が返す軸の 2 次元配列
        curves: 系列ラベル → 時刻別行動者率 (12, 96)。LABEL_TEACHER を含むこと
    """
    hours = cv.slot_hours()
    style = {LABEL_TEACHER: (COLOR_TEACHER, 2.6),
             LABEL_ATUS: (COLOR_ATUS, 1.5),
             LABEL_PRETRAINED: (COLOR_PRETRAINED, 1.5),
             LABEL_FINETUNED: (COLOR_FINETUNED, 1.5)}
    for c, name in enumerate(cv.ACT_NAMES):
        ax = ax_grid[c // 3][c % 3]
        for label, cur in curves.items():
            color, lw = style[label]
            ax.plot(hours, cur[c], color=color, lw=lw, label=label,
                    solid_capstyle="round", solid_joinstyle="round")
        ax.set_title(f"{cv.ACT_JA[name]}（{ACT_EN[name]}）", fontsize=11)
        ax.set_xlim(4.0, 28.0)
        ticks = np.arange(4, 29, 4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_hour_label(float(h)) for h in ticks])
        ax.set_ylim(bottom=0.0)
        ax.grid(color="#e4e3dd", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8.5)
        if c // 3 == 3:
            ax.set_xlabel("時刻", fontsize=10)
        if c % 3 == 0:
            ax.set_ylabel("行動者率", fontsize=10)


def render_figure(title: str, curves: dict[str, npt.NDArray[np.float64]]) -> Any:
    """1 枚の図（タイトル・凡例・4x3 の小倍数）を作って Figure を返す。

    Args:
        title: 図のタイトル
        curves: 系列ラベル → 時刻別行動者率 (12, 96)

    Returns:
        matplotlib の Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    plot_rates_figure(axes, curves)
    fig.suptitle(title, fontsize=16, y=0.993)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.968), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return fig


def _setup_fonts() -> None:
    """日本語フォントを設定する。無ければ例外で止める（文字化けした図を残さない）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    jp = [f for f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic",
                      "IPAexGothic", "Noto Sans CJK JP") if f in names]
    if not jp:
        raise RuntimeError("日本語フォントが見つからない。タイトルと活動名が文字化けするので止める")
    plt.rcParams["font.family"] = jp[0]
    plt.rcParams["axes.unicode_minus"] = False
    # ★PDF は TrueType で埋め込む。既定の Type 3 はグリフ名を ASCII で書くので、
    #   Hiragino の日本語グリフで UnicodeEncodeError になる
    plt.rcParams["pdf.fonttype"] = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", type=Path, required=True,
                    help="Stage 1 の重みから dump_rates で作った .npz")
    ap.add_argument("--finetuned", type=Path, required=True,
                    help="Stage 2 の世代から dump_rates で作った .npz")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/figures/teacher_fit")
    ap.add_argument("--out-csv", type=Path,
                    default=REPO_ROOT / "data/processed/aggregates/stage2_teacher_fit_mse.csv")
    ap.add_argument("--check-csv", type=Path, default=DEFAULT_CHECK_CSV,
                    help="rate_mse を照合する事後選択 CSV")
    ap.add_argument("--pretrained-ckpt", type=Path, default=None,
                    help="--pretrained が .csv のとき、生成に使った重み（mtime を照合する）")
    ap.add_argument("--allow-n-mismatch", action="store_true",
                    help="n / pool_seed が違っても止めない。MSE は生成側の MC 分散 p(1−p)/n の"
                         "ぶん n の小さい側が不利になるので、mse_split_* 列と併せて読むこと")
    args = ap.parse_args()

    tgt = st.load_stula_targets()
    a_star = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    recs = [load_pool(args.pretrained, LABEL_PRETRAINED, args.pretrained_ckpt),
            load_pool(args.finetuned, LABEL_FINETUNED)]

    # ★n と pool_seed が揃っていなければ MSE の MC 分散が揃わない。比較の前提なので既定では止める
    same_pool_setting = len({(r.n, r.pool_seed) for r in recs}) == 1
    if not same_pool_setting:
        detail = ", ".join(f"{r.label} n={r.n} seed={r.pool_seed}" for r in recs)
        if not args.allow_n_mismatch:
            raise ValueError(f"Pre-trained と Fine-tuned で n / pool_seed が違う: {detail}")
        print(f"[fit] ★n / pool_seed が揃っていない（--allow-n-mismatch）: {detail}")
    for r in recs:
        recorded = check_provenance(r, tgt, args.check_csv)
        status = f"記録値 {recorded:.6f} と一致" if recorded is not None else "照合する記録なし"
        print(f"[fit] {r.label:<12} {r.ckpt} step={r.step} n={r.n} seed={r.pool_seed}  {status}")

    # --- MSE の表（12 活動を主、11 活動を併記）---------------------------------
    tables = [mse_table(recs, tgt, st.mask_12act()), mse_table(recs, tgt, st.mask_11act())]
    table = pd.concat(tables, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"[fit] MSE の表を書いた: {args.out_csv}")

    with pd.option_context("display.width", 220, "display.max_rows", 40):
        cols = ["group", "pop_share", f"mse_{LABEL_TEACHER}の床",
                f"mse_{LABEL_PRETRAINED}", f"mse_{LABEL_FINETUNED}", "change_pct"]
        # ★表示用に文字列へ直してから出す。to_string の float_format / formatters は
        #   pandas のスタブと型が合わず、型チェックで警告になる
        disp = tables[0][cols].copy()
        for c in cols[2:5]:
            disp[c] = [f"{v:.3e}" for v in np.asarray(disp[c], dtype=np.float64)]
        disp["pop_share"] = [f"{v:.4f}" for v in np.asarray(disp["pop_share"], dtype=np.float64)]
        disp["change_pct"] = [f"{v:+.1f}" for v in np.asarray(disp["change_pct"], dtype=np.float64)]
        print(disp.to_string(index=False))

    # --- 図 ------------------------------------------------------------------
    _setup_fonts()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # ★図にはタイトルと凡例だけを載せる。MSE・人口シェア・n は表（--out-csv）で読む
    pages: list[tuple[str, dict[str, npt.NDArray[np.float64]], Path]] = []
    # ★全体曲線の群シェアは系列で違う。教師A*・生成は STULA の pi_d（cv.weighted_slot_rates）、
    #   ATUS実データ は TUFINLWGT だけから作った atus_share（米国の人口構成）。
    #   したがって ATUS実データ と他の線の差には、群の中の違いに加えて群構成の違いも入る
    atus_rates, atus_share = load_atus_rates()
    overall_curves = {LABEL_TEACHER: cv.weighted_slot_rates(a_star, tgt),
                      LABEL_ATUS: np.einsum("d,dcs->cs", atus_share, atus_rates),
                      **{r.label: cv.weighted_slot_rates(r.rates, tgt) for r in recs}}
    pages.append(("時刻別行動者率（全体：28群のSTULA人口加重平均）",
                  overall_curves, args.out_dir / "overall.png"))
    for d in range(st.D_GROUPS):
        pages.append((f"時刻別行動者率（{group_label(d)}）",
                      {LABEL_TEACHER: a_star[d], LABEL_ATUS: atus_rates[d],
                       **{r.label: r.rates[d] for r in recs}},
                      args.out_dir / f"group_{d:02d}.png"))

    pdf_path = args.out_dir / "all_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        for title, curves, png in pages:
            fig = render_figure(title, curves)
            fig.savefig(png, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[fit] 図を書いた: {args.out_dir}（png 29 枚 + {pdf_path.name}）")


if __name__ == "__main__":
    main()
