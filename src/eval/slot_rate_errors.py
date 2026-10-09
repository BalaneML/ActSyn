"""
slot_rate_errors.py
===================
時刻別行動者率の誤差を、活動ごと・15 分区間（スロット）ごとに見る表を作る。

★なぜ要るか:
    ‖実 − 生成‖²（96 スロットの和）や全セルの MSE は、値が小さく単位（行動者率）に戻せない。
    さらに時刻を潰すので「どの活動のどの時刻がずれているか」が分からない。
    平均は山を削っても下がる（rate_mae は平滑化で下がる）が、最大誤差は山を削ると上がるので、
    平均の弱点を補える。

★単位はポイント（pt）: 誤差 = (生成 − 実) × 100。符号付きで、正 = 生成が多い。

★作る表:
    activity_error_table  1 活動 1 行。平均誤差（符号付き pt）・MAE（pt）・最大誤差（符号付き pt）とその時刻・
                          実の値・種の範囲・床
    topk_error_table      活動ごとに |誤差| の大きい順に k 個（既定 TOP_K）の 15 分区間（隣り合う区間も別に数える）
    seed_error_table      種 × 活動 1 行。種ごとの曲線で測った平均誤差・MAE・最大 |誤差|（種の範囲を比べるため）

★平均誤差（bias_pt）: 96 区間の符号付きの誤差の平均 = 1 日のシェアの差 × 100。正 = その活動が 1 日を通して多すぎる。
  MAE は ± が打ち消し合わないので、山の時刻のずれ（前後に + と − が出る）も拾う。2 つを並べると、
  「総量のずれ」（平均誤差が大きい）と「時刻の形のずれ」（MAE だけが大きい）を見分けられる。

★種の扱い: 誤差は種平均の曲線で測る。最大誤差の時刻も種平均の曲線で決め、その時刻での種ごとの
  誤差の最小〜最大を併記する（種ごとに最大を取ると、種ごとに時刻が変わって表がまとまらない）。
  種ごとの曲線で測った値（seed_error_table）は、2 つのモデルの種の範囲が重なるかを比べるために使う。

★床: 96 区間の最大を取ると、完全なモデルでも雑音で正の値が出る。そこで「完全なモデルの曲線」
  floor_curves (B, 12, 96) を呼び出し側から受け取り、同じ統計量（最大 |誤差|・MAE・|平均誤差|）を B 回測って
  上側 FLOOR_QUANTILE 分位を床とする。床は 2 段で受け取る:
    floor_low   下限側。生成プールの有限さだけ（参照の曲線そのものを写した完全なモデル）
    floor_high  上限側。参照の標本誤差 + 生成プールの有限さ（参照の母集団を写した完全なモデル）
  floor_position は |誤差| と 2 段の床の位置関係（FLOOR_POSITIONS の 3 値）:
    below_low  下限側の床以下。雑音と区別できない
    between    2 つの床の間。参照の標本誤差の範囲なので、課題とは言い切れない
    above_high 上限側の床より大きい。完全なモデルでは出ない誤差（課題）
  ★Top-k の判定にも、その活動の最大 |誤差| の床を使う（96 区間から選んだことを床の側でも数えるため）
  ★floor_curves は、モデルと同じ本数の種を平均した曲線にすること（種平均で雑音が減るので）

★NaN: 実の値が NaN のセル（Stage 2 の教師 A* で公表の無いセル）は除いて測る。

データフロー:

```mermaid
flowchart TD
    REAL["real_curve (12, 96)"] --> ERR["error_pt<br/>err (S, 12, 96) pt"]
    SEED["seed_curves (S, 12, 96)"] --> ERR
    FC["floor_curves (B, 12, 96)<br/>下限側・上限側の 2 組"] --> FE["floor_errors<br/>floor_low / floor_high"]
    REAL --> FE
    ERR --> AT["activity_error_table<br/>1 活動 1 行"]
    ERR --> TK["topk_error_table<br/>活動 × k 行"]
    ERR --> ST["seed_error_table<br/>種 × 活動 1 行"]
    FE --> AT
    FE --> TK
```

使い方:
    .venv/bin/python src/eval/test_slot_rate_errors.py    # 自己テスト
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArr = npt.NDArray[np.float64]

SLOT_START_HOUR = 4          # スロット 0 は 04:00（stage2_curves.SLOT_START_HOUR と同じ）
SLOT_MINUTES = 15
TOP_K = 3
FLOOR_QUANTILE = 0.95
FLOOR_POSITIONS: tuple[str, ...] = ("below_low", "between", "above_high")
PT = 100.0                   # 行動者率 → ポイント


def slot_label(slot: int) -> str:
    """スロットの開始時刻を HH:MM で返す（24 時以降は 0 時に戻す。スロット 0 = 04:00）"""
    minutes = SLOT_START_HOUR * 60 + slot * SLOT_MINUTES
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def error_pt(real_curve: FloatArr, curves: FloatArr) -> FloatArr:
    """符号付きの誤差（pt）, (12, 96) と (..., 12, 96) -> (..., 12, 96)。正 = 生成が多い"""
    return (np.asarray(curves, dtype=np.float64) - real_curve) * PT


def _abs_max_slot(err: FloatArr) -> npt.NDArray[np.int64]:
    """活動ごとに |誤差| が最大のスロット（NaN は除く）, (12, 96) -> (12,)"""
    return np.asarray(np.nanargmax(np.abs(err), axis=-1), dtype=np.int64)


def floor_position(abs_err: float, low: float, high: float) -> str:
    """|誤差| と 2 段の床の位置関係（FLOOR_POSITIONS のどれか）"""
    if abs_err <= low:
        return FLOOR_POSITIONS[0]
    return FLOOR_POSITIONS[1] if abs_err <= high else FLOOR_POSITIONS[2]


def floor_errors(real_curve: FloatArr, floor_curves: FloatArr,
                 quantile: float = FLOOR_QUANTILE) -> dict[str, FloatArr]:
    """完全なモデルでも出る誤差の上側分位（床）

    Args:
        real_curve: 実の時刻別行動者率, (12, 96)
        floor_curves: 完全なモデルの曲線, (B, 12, 96)。モデルと同じ本数の種を平均したもの
        quantile: 床にする上側分位

    Returns:
        {"max_pt": (12,) 最大 |誤差| の床, "mae_pt": (12,) MAE の床, "bias_pt": (12,) |平均誤差| の床,
         "slot_pt": (12, 96) 区間ごとの |誤差| の床}
    """
    err = error_pt(real_curve, floor_curves)                                   # (B, 12, 96)
    abs_err = np.abs(err)
    return {"max_pt": np.asarray(np.quantile(np.nanmax(abs_err, axis=-1), quantile, axis=0), dtype=np.float64),
            "mae_pt": np.asarray(np.quantile(np.nanmean(abs_err, axis=-1), quantile, axis=0), dtype=np.float64),
            "bias_pt": np.asarray(np.quantile(np.abs(np.nanmean(err, axis=-1)), quantile, axis=0), dtype=np.float64),
            "slot_pt": np.asarray(np.quantile(abs_err, quantile, axis=0), dtype=np.float64)}


def activity_error_table(real_curve: FloatArr, seed_curves: FloatArr, act_names: Sequence[str],
                         floor_low: dict[str, FloatArr], floor_high: dict[str, FloatArr]) -> pd.DataFrame:
    """1 活動 1 行の誤差の表

    Args:
        real_curve: 実の時刻別行動者率, (12, 96)
        seed_curves: 種ごとの生成の時刻別行動者率, (S, 12, 96)
        act_names: 活動名, 長さ 12
        floor_low: 下限側の床（floor_errors の戻り値）
        floor_high: 上限側の床（floor_errors の戻り値）

    Returns:
        列 activity / n_seeds / bias_pt（平均誤差、符号付き）/ bias_floor_low_pt / bias_floor_high_pt / bias_position /
        mae_pt / mae_floor_low_pt / mae_floor_high_pt / mae_position /
        max_err_pt（符号付き）/ max_time / max_slot / real_at_max / gen_at_max /
        err_seed_min_pt / err_seed_max_pt / max_floor_low_pt / max_floor_high_pt / max_position
        ★bias_position は |平均誤差| と床の位置関係
    """
    err = error_pt(real_curve, seed_curves)                                    # (S, 12, 96)
    mean_err = err.mean(axis=0)                                                # 種平均の曲線の誤差
    gen_mean = np.asarray(seed_curves, dtype=np.float64).mean(axis=0)
    max_slot = _abs_max_slot(mean_err)
    rows = []
    for c, name in enumerate(act_names):
        s = int(max_slot[c])
        bias = float(np.nanmean(mean_err[c]))
        mae = float(np.nanmean(np.abs(mean_err[c])))
        bias_lo, bias_hi = float(floor_low["bias_pt"][c]), float(floor_high["bias_pt"][c])
        mae_lo, mae_hi = float(floor_low["mae_pt"][c]), float(floor_high["mae_pt"][c])
        max_lo, max_hi = float(floor_low["max_pt"][c]), float(floor_high["max_pt"][c])
        rows.append({"activity": name, "n_seeds": int(err.shape[0]),
                     "bias_pt": bias, "bias_floor_low_pt": bias_lo, "bias_floor_high_pt": bias_hi,
                     "bias_position": floor_position(abs(bias), bias_lo, bias_hi),
                     "mae_pt": mae, "mae_floor_low_pt": mae_lo, "mae_floor_high_pt": mae_hi,
                     "mae_position": floor_position(mae, mae_lo, mae_hi),
                     "max_err_pt": float(mean_err[c, s]), "max_time": slot_label(s), "max_slot": s,
                     "real_at_max": float(real_curve[c, s]), "gen_at_max": float(gen_mean[c, s]),
                     "err_seed_min_pt": float(err[:, c, s].min()), "err_seed_max_pt": float(err[:, c, s].max()),
                     "max_floor_low_pt": max_lo, "max_floor_high_pt": max_hi,
                     "max_position": floor_position(abs(float(mean_err[c, s])), max_lo, max_hi)})
    return pd.DataFrame(rows)


def topk_error_table(real_curve: FloatArr, seed_curves: FloatArr, act_names: Sequence[str],
                     floor_low: dict[str, FloatArr], floor_high: dict[str, FloatArr],
                     k: int = TOP_K) -> pd.DataFrame:
    """活動ごとに |誤差| の大きい順に k 個の 15 分区間

    Note:
        ★隣り合う区間も別に数える（12:00 と 12:15 が両方入りうる）
        ★position は、その活動の最大 |誤差| の床（floor_*["max_pt"]）と比べる

    Args:
        real_curve: 実の時刻別行動者率, (12, 96)
        seed_curves: 種ごとの生成の時刻別行動者率, (S, 12, 96)
        act_names: 活動名, 長さ 12
        floor_low: 下限側の床（floor_errors の戻り値）
        floor_high: 上限側の床（floor_errors の戻り値）
        k: 活動ごとに取る区間の数

    Returns:
        列 activity / rank（1 始まり）/ time / slot / real / gen / err_pt（符号付き）/
        err_seed_min_pt / err_seed_max_pt / max_floor_low_pt / max_floor_high_pt / position
    """
    err = error_pt(real_curve, seed_curves)
    mean_err = err.mean(axis=0)
    gen_mean = np.asarray(seed_curves, dtype=np.float64).mean(axis=0)
    rows = []
    for c, name in enumerate(act_names):
        abs_err = np.where(np.isnan(mean_err[c]), -np.inf, np.abs(mean_err[c]))   # NaN は最後に回す
        order = np.argsort(-abs_err, kind="stable")[:k]
        for rank, s in enumerate(int(v) for v in order):
            if not np.isfinite(abs_err[s]):
                break
            rows.append({"activity": name, "rank": rank + 1, "time": slot_label(s), "slot": s,
                         "real": float(real_curve[c, s]), "gen": float(gen_mean[c, s]),
                         "err_pt": float(mean_err[c, s]),
                         "err_seed_min_pt": float(err[:, c, s].min()), "err_seed_max_pt": float(err[:, c, s].max()),
                         "max_floor_low_pt": float(floor_low["max_pt"][c]),
                         "max_floor_high_pt": float(floor_high["max_pt"][c]),
                         "position": floor_position(float(abs_err[s]), float(floor_low["max_pt"][c]),
                                                    float(floor_high["max_pt"][c]))})
    return pd.DataFrame(rows)


def seed_error_table(real_curve: FloatArr, seed_curves: FloatArr, act_names: Sequence[str],
                     seeds: Sequence[int]) -> pd.DataFrame:
    """種ごとの曲線で測った、活動ごとの平均誤差・MAE・最大 |誤差|（種 × 活動 1 行）

    Note:
        ★activity_error_table は種平均の曲線で測る。こちらは種ごとの曲線で測るので、種平均の曲線より
          生成プールの雑音を多く含む。2 つのモデルの種の範囲が重なるか（分離するか）を比べるために使う
        ★最大 |誤差| の時刻は種ごとに決める（種ごとに違ってよい）

    Args:
        real_curve: 実の時刻別行動者率, (12, 96)
        seed_curves: 種ごとの生成の時刻別行動者率, (S, 12, 96)
        act_names: 活動名, 長さ 12
        seeds: seed_curves の各行の種, 長さ S

    Returns:
        列 seed / activity / bias_pt（符号付き）/ mae_pt / max_abs_pt / max_err_pt（符号付き）/ max_time

    Raises:
        ValueError: seeds の長さが seed_curves の種の数と違うとき
    """
    err = error_pt(real_curve, seed_curves)                                    # (S, 12, 96)
    if len(seeds) != err.shape[0]:
        raise ValueError(f"seeds の長さ {len(seeds)} が種の数 {err.shape[0]} と違う")
    rows = []
    for i, seed in enumerate(seeds):
        max_slot = _abs_max_slot(err[i])                                       # (12,)
        for c, name in enumerate(act_names):
            s = int(max_slot[c])
            rows.append({"seed": int(seed), "activity": name,
                         "bias_pt": float(np.nanmean(err[i, c])),
                         "mae_pt": float(np.nanmean(np.abs(err[i, c]))),
                         "max_abs_pt": float(abs(err[i, c, s])), "max_err_pt": float(err[i, c, s]),
                         "max_time": slot_label(s)})
    return pd.DataFrame(rows)
