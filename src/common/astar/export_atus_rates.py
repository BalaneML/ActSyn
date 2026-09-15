"""
export_atus_rates.py
====================
ATUS の群別・時刻別行動者率を CSV に書き出す（export_astar.py の対）

A* は日本の公表表から作った (28群, 12活動, 96スロット) の時刻別行動者率で、
export_astar.py が A_star_weekday.csv に書き出している。本スクリプトは
**まったく同じ量を米国側 (ATUS 個票) で測ったもの**を、同じ行の並び・同じ列名で書き出す。
2つの CSV は 336 行が 1 行ずつ対応するので、そのまま並べて突き合わせられる。

    A_star_weekday.csv           日本 (社会生活基本調査の公表表)   ← export_astar.py
    A_atus_weekday_*.csv         米国 (ATUS 個票)                  ← 本スクリプト

加重版と非加重版の2本を書き出す:
    weighted    調査ウェイト TUFINLWGT で加重。A* は母集団の率なので、対応させるならこちら
    unweighted  標本そのままの割合。重み付けの選択が結論を変えていないかの確認用

★率だけを見て A* とのズレを論じない。n_d (標本数) と n_eff (Kish の有効標本数) を
  同じ行に持たせてある。最小の群は n_d=17 / n_eff=11.7 で、表現できる率の最小刻みが
  0.086 ある。A* のセルは中央値 0.0112 なので、薄い群では標本誤差のほうが大きい。

★読み戻すときは float_precision="round_trip" を付けること（export_astar.py と同じ理由）:
      pd.read_csv(path, float_precision="round_trip")

使い方:
    .venv/bin/python3 src/common/astar/export_atus_rates.py
    .venv/bin/python3 src/common/astar/export_atus_rates.py --out-dir path/to/dir
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import atus_group_rates as ag  # noqa: E402
# 群ラベルと活動の日本語名は export_astar.py が持っている。ここに5つ目の複製を作らない
from export_astar import ACT_JA, AGE7, EMP, SEX  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "atus2024"
KEY_COLS = ["d", "sex", "age", "employment", "act_idx", "act_key", "act_ja"]
CNT_COLS = ["n_d", "n_eff", "wshare"]


def build_frame(rates: np.ndarray, n_d: np.ndarray, n_eff: np.ndarray,
                wshare: np.ndarray) -> pd.DataFrame:
    """(28,12,96) の率 + 群ごとの標本情報 -> 336行の wide 形式。

    行の並び (d 昇順 × Common 昇順) は export_astar.build_frame と同一にする。
    """
    rows: list[dict[str, object]] = []
    for d in range(ag.D_GROUPS):
        g, rem = divmod(d, ag.N_A * ag.N_E)
        a, e = divmod(rem, ag.N_E)
        for c in ag.Common:
            rec: dict[str, object] = {
                "d": d,
                "sex": SEX[g],
                "age": AGE7[a],
                "employment": EMP[e],
                "act_idx": int(c),
                "act_key": c.name,
                "act_ja": ACT_JA[c],
                "n_d": int(n_d[d]),
                "n_eff": float(n_eff[d]),
                "wshare": float(wshare[d]),
            }
            for j in range(ag.NUM_SLOTS):
                rec[f"s{j}"] = float(rates[d, int(c), j])
            rows.append(rec)
    return pd.DataFrame(rows)


def verify(path: Path, rates: np.ndarray, label: str) -> None:
    """書き出した CSV を読み戻し、元のテンソルと厳密一致することを確かめる。

    export_astar.verify と同じ契約（%.17g で書き、round_trip で読む）。
    加えて、1スロット1ラベルの帰結である「12活動の和 = 1」を検査する。
    """
    df = pd.read_csv(path, float_precision="round_trip")
    back = df[ag.SLOT_COLS].to_numpy(dtype=float).reshape(
        ag.D_GROUPS, ag.NUM_COMMON, ag.NUM_SLOTS)
    assert np.array_equal(np.nan_to_num(back, nan=-1.0),
                          np.nan_to_num(rates, nan=-1.0)), "CSV がテンソルと厳密一致しない"

    col_sum = np.nansum(back, axis=1)                       # (28, 96)
    gap = float(np.abs(col_sum - 1.0).max())
    assert gap < 1e-9, f"12活動の和が1でない (最大差 {gap:.3e})"
    n_nan = int(np.isnan(back).sum())
    n_zero = int((back == 0).sum())
    nd = df["n_d"].to_numpy(dtype=np.int64)
    print(f"  [{label}] 行数={len(df)}  標本なし(NaN)={n_nan}  "
          f"行動者ゼロ={n_zero} ({n_zero / back.size:.1%})")
    print(f"  [{label}] 12活動の和: 1 からの最大差 {gap:.3e}  "
          f"n_d min={int(nd.min())} max={int(nd.max())}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ATUS の群別・時刻別行動者率を A* と同じ形式で CSV に書き出す")
    ap.add_argument("--data", type=Path, default=ag.DATA_PATH, help="ATUS 共通12分類の個票 CSV")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="出力ディレクトリ")
    args = ap.parse_args()

    src: Path = args.data
    print(f"source: {src}")
    print(f"  sha256={hashlib.sha256(src.read_bytes()).hexdigest()[:16]}  "
          f"mtime={pd.Timestamp(src.stat().st_mtime, unit='s', tz='Asia/Tokyo'):%Y-%m-%d %H:%M}")

    sched, groups, w = ag.load_atus_weekday(src)
    n_d, wsum, n_eff = ag.group_counts(groups, w)
    print(f"  平日 N={len(sched)}  群={ag.D_GROUPS}  空の群={int((n_d == 0).sum())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for label, weights, eff in (("weighted", w, n_eff),
                                ("unweighted", None, n_d.astype(np.float64))):
        rates = ag.group_rates(sched, groups, weights)
        df = build_frame(rates, n_d, eff, wsum / wsum.sum())
        out = args.out_dir / f"A_atus_weekday_{label}.csv"
        df.to_csv(out, index=False, float_format="%.17g")
        verify(out, rates, label)
        print(f"  -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
