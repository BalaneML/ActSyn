"""
export_astar.py
===============
Stage 2 の教師テンソル A* を CSV に書き出す

load_stula_targets() はメモリ上に (28,12,96) を作るだけで、教師そのものはどこにも
残らない。中身を人が確認したり、実行間で同一性を照合したりするための書き出し口。

★これは「成果物」であって「キャッシュ」ではない。学習 (stage2_finetune.py) は
  従来どおり CSV から毎回組み直す。ここで書いたファイルを学習が読むようにすると、
  入力 CSV を更新したのに教師が古いまま、という取り違えが起こりうる。

出力形式は data/processed/stula/*.csv と同じ wide 形式に揃える:
    キー列 + pop_k + s0..s95   (336行 = 28群 × 12活動)
    率は教師テンソルと同じ [0,1]。s0 は 04:00-04:15。

★読み戻すときは float_precision="round_trip" を付けること。既定の "high" は
  正確丸めでないため、値の約73%が最下位1ビットずれる:
      pd.read_csv(path, float_precision="round_trip")

使い方:
    python src/common/astar/export_astar.py
    python src/common/astar/export_astar.py --out path/to/A_star.csv
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"

sys.path.insert(0, str(SIMPLE_DIR))
import stage2_targets as st  # noqa: E402

sys.path.insert(0, str(st.REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, STULA_TO_COMMON  # noqa: E402

DEFAULT_OUT = st.REPO_ROOT / "outputs" / "astar" / "A_star_weekday.csv"

# 群インデックス d = 性*14 + 年齢*2 + 就業 を人が読める形に戻すためのラベル。
# d_index() と同じ規則で展開するので、ここに別の並びを書いてはいけない。
SEX = ["男", "女"]
AGE7 = ["15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
EMP = ["無業", "有業"]      # e=0 が無業、e=1 が有業（load_stula_targets の写像と同じ）

ACT_JA = {
    Common.SLEEP_PERSONAL: "睡眠・身の回りの用事",
    Common.MEALS:          "食事",
    Common.WORK:           "仕事",
    Common.SCHOOL:         "学業",
    Common.HOUSEWORK:      "家事",
    Common.CAREGIVING:     "介護・看護／育児",
    Common.SHOPPING:       "買い物",
    Common.TRAVEL:         "移動・通勤通学",
    Common.LEISURE_SOCIAL: "余暇・交際",
    Common.SPORTS:         "スポーツ",
    Common.VOLUNTEER:      "ボランティア・社会参加",
    Common.OTHER_X:        "その他",
}


def stula_codes(c: Common) -> str:
    """共通12分類 c に写る社基調20分類のコード。クロスウォークから導出する（複製しない）"""
    return ",".join(sorted(code for code, com in STULA_TO_COMMON.items() if com == c))


def build_frame(name: str) -> pd.DataFrame:
    tgt = st.load_stula_targets(name)
    A = np.asarray(tgt["group_rates_tbl"])          # (28,12,96)
    pop = np.asarray(tgt["pop"]).reshape(-1)        # (28,)

    rows: list[dict[str, object]] = []
    for d in range(st.D_GROUPS):
        g, rem = divmod(d, st.N_A * st.N_E)
        a, e = divmod(rem, st.N_E)
        for c in Common:
            rec: dict[str, object] = {
                "d": d,
                "sex": SEX[g],
                "age": AGE7[a],
                "employment": EMP[e],
                "act_idx": int(c),
                "act_key": c.name,
                "act_ja": ACT_JA[c],
                "stula_codes": stula_codes(c),
                "pop_k": float(pop[d]),
            }
            for j in range(st.NUM_SLOTS):
                rec[f"s{j}"] = float(A[d, int(c), j])
            rows.append(rec)
    return pd.DataFrame(rows)


def verify(path: Path, name: str) -> None:
    """書き出した CSV を読み戻し、元のテンソルとビット単位で一致することを確かめる

    ★allclose ではなく array_equal で見る。厳密一致には書き出しと読み込みの両方に
      条件があり、どちらか一方でも欠けると値の約73%が最下位1ビットずれる（2e-16）:
        書き出し: float_format="%.17g"   （既定は有効16桁で1桁足りない）
        読み込み: float_precision="round_trip"（既定 "high" は正確丸めでない）
      数値としては無害だが、「この CSV は教師そのものか」の照合ができなくなる。
    """
    A = np.asarray(st.load_stula_targets(name)["group_rates_tbl"])
    df = pd.read_csv(path, float_precision="round_trip")
    back = df[st.SLOT_COLS].to_numpy(dtype=float).reshape(st.D_GROUPS, len(Common), st.NUM_SLOTS)
    same = np.array_equal(back, A) or np.array_equal(np.nan_to_num(back, nan=-1.0),
                                                     np.nan_to_num(A, nan=-1.0))
    assert same, "CSV がテンソルと厳密一致しない"

    col_sum = np.nansum(back, axis=1)
    n_nan = int(np.isnan(back).sum())
    n_zero = int((back == 0).sum())
    print(f"  行数={len(df)}  非公表(NaN)={n_nan}  行動者ゼロ={n_zero} ({n_zero / back.size:.1%})")
    print(f"  12活動の和: min={col_sum.min():.6f} max={col_sum.max():.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="教師テンソル A* を CSV に書き出す")
    ap.add_argument("--table", default=st.DEFAULT_TABLE, help="入力の timeband 表名")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力 CSV のパス")
    args = ap.parse_args()

    src = st.STULA_DIR / f"{args.table}.csv"
    print(f"source: {src}")
    print(f"  sha256={hashlib.sha256(src.read_bytes()).hexdigest()[:16]}  "
          f"mtime={pd.Timestamp(src.stat().st_mtime, unit='s', tz='Asia/Tokyo'):%Y-%m-%d %H:%M}")

    df = build_frame(args.table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.17g")
    verify(args.out, args.table)
    print(f"-> {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
