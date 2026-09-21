"""
stage2_lgo.py
=============
Stage 2 の LGO（leave-groups-out）ドライバ（Stage2_design.md §9.4 / 実装項目 18）

28 群を 4 群ずつ 7 分割し、全群がちょうど 1 回ずつ held-out になるように 7 本の
学習を回すための道具。**目的は軸1 の循環を断つこと**である。teacher_groups=28 の
ままだと教師適合は学習の目的関数そのものを測っており、値が改善しても主張に
ならない（§9.2）。教師から外した群での適合だけが非循環な証拠になる。

    分割   : 人口シェアで層化する。群人口シェアには 18.7 倍の開きがあり、
             無作為に切ると大きい群ばかり／小さい群ばかりの fold ができる（§8.4）
    決定性 : 乱数を使わない。fold の割り当てが実行ごとに変われば、どの群がどの
             fold で held-out だったかを後から再現できない
    運用   : 学習コストが 7 倍になるので、**まず λ を 1 本決めてから**回す（§9.4）

★stage2_finetune.stratified_holdout とは別物である。あちらは「1 回分の k 群を
  乱数で抽出する」もので、全群を覆わないし fold 間で重複もしうる。こちらは
  28 群の**分割**であり、全群が過不足なく 1 回ずつ held-out になる。

★なぜ 1 fold では足りないか。4 群を 1 回抜くだけだと「どの 4 群を抜いたか」で
  結果が動き、たまたま良かった／悪かった可能性を潰せない。7 fold なら 28 群
  すべてについて held-out での値が 1 つずつ揃い、その分布を報告できる。

使い方:
    # 7 fold の群割り当てを見る（stdout は機械可読、層化の効き具合は stderr）
    python src/models/DDPM_Aggregate_Simple/stage2_lgo.py --print-folds

    # SQUID へ 7 本投入する（λ は掃引で決めた値を入れる）
    bash jobs/submit_lgo.sh 0.01

    # 7 本の結果 CSV から held-out 群 28 個の分布を出す
    python src/models/DDPM_Aggregate_Simple/stage2_lgo.py \\
        --collect data/processed/aggregates/stage2_lam0.01_fold*_selection.csv
"""
import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
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


st: Any = _load("simple_stage2_targets", HERE / "stage2_targets.py")

# 28 = 7 × 4。設計書 §9.4 の決定（4 群ずつ 7 分割）。
N_FOLDS = 7


def stratified_folds(pop: np.ndarray, n_folds: int = N_FOLDS) -> list[list[int]]:
    """人口シェアで層化して 28 群を n_folds 個の fold へ分割する。

    全群がちょうど 1 回ずつ held-out になる（抽出ではなく分割）。

    Note:
        ★蛇行（serpentine）で配る。シェア降順に並べ、fold 0→6 の次は 6→0 と
          折り返しながら 1 つずつ配ると、各 fold が大・中・小をひと通り持つ。
          折り返さずに順番へ配ると fold 0 に大きい群ばかりが集まり、fold 間で
          held-out の人口シェアが大きく食い違う。
        ★乱数を使わない。同じ pop なら常に同じ分割になる。argsort も
          kind="stable" にして、人口が同値の群の順序まで決める。

    Args:
        pop: 群人口。(28,) に reshape できる形であればよい（load_stula_targets の
            戻り値は (2,7,2)）
        n_folds: 分割数。D_GROUPS が割り切れること, default=N_FOLDS=7

    Returns:
        fold ごとの群インデックスの list。各要素は昇順で、全体で 0..27 を
        過不足なく覆う

    Raises:
        ValueError: D_GROUPS が n_folds で割り切れない場合
    """
    if n_folds < 1 or st.D_GROUPS % n_folds != 0:
        raise ValueError(f"{st.D_GROUPS} 群を {n_folds} 個へ均等に分割できない")
    order = np.argsort(-np.asarray(pop, dtype=np.float64).reshape(st.D_GROUPS),
                       kind="stable")
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, d in enumerate(order):
        lap, pos = divmod(i, n_folds)
        folds[pos if lap % 2 == 0 else n_folds - 1 - pos].append(int(d))
    return [sorted(f) for f in folds]


def fold_shares(pop: np.ndarray, folds: list[list[int]]) -> np.ndarray:
    """fold ごとの held-out 人口シェア合計を返す, -> (n_folds,)

    層化が効いているかの確認に使う。値が揃っているほど、どの fold を抜いても
    教師に残る人口が同じということである。

    Args:
        pop: 群人口。(28,) に reshape できる形
        folds: stratified_folds の戻り値

    Returns:
        fold ごとのシェア合計, dtype=float64, (n_folds,)。全体で 1.0 になる
    """
    p = np.asarray(pop, dtype=np.float64).reshape(st.D_GROUPS)
    share = p / p.sum()
    return np.array([share[f].sum() for f in folds], dtype=np.float64)


def fold_floors(tgt: dict, folds: list[list[int]]) -> pd.DataFrame:
    """fold ごとの「教師自身の標本誤差による床」を出す（実装項目 16 × 18）。

    **LGO の 7 個の値を読むための物差しである。**fold 間のばらつきがここで出る床より
    小さければ、それは教師の標本誤差の揺らぎであってモデルの当たり外れではない。

    Note:
        ★fold ごとに床が違う。held-out 4 群の実回答者数はまちまちで、30 代の無業男性
          （n=162〜227）のような薄い層を含む fold は床が高くなる。全 fold 共通の
          1 本の床で読むと、薄い fold の値を過小に評価する。
        ★`stage2_select` の held-out 行と同じ部分集合の取り方に揃えてある。あちらは
          `group_rates_tbl[sel]` と `pop[sel]` で `sub_tgt` を作るので、こちらも
          `group_rates_var[sel]` を足した同じ形を `teacher_floor` へ渡す。
          dev_* は「その 4 群の中での人口平均からのズレ」なので、28 群全体で
          取った床とは別の値になる。

    Args:
        tgt: `stage2_targets.load_stula_targets` の戻り値。`group_rates_var` が要る
        folds: `stratified_folds` の戻り値

    Returns:
        fold ごとの床。列は fold / groups / pop_share / n_layer_min /
        rate_mse / rate_mae / dev_mse / dev_rmse
    """
    if tgt.get("group_rates_var") is None:
        raise ValueError(
            "group_rates_var が無いので fold ごとの床を出せない。\n"
            "       python src/common/preprocess/stula/parse_timeband.py を流すこと")
    pop = np.asarray(tgt["pop"]).reshape(st.D_GROUPS)
    share = pop / pop.sum()
    n15 = tgt["n_layer"]
    rows = []
    for i, f in enumerate(folds):
        sel = np.asarray(f, dtype=np.int64)
        sub = {"group_rates_tbl": tgt["group_rates_tbl"][sel],
               "group_rates_var": tgt["group_rates_var"][sel],
               "pop": pop[sel]}
        fl = st.teacher_floor(sub, st.mask_12act())
        # 群 d -> (性, 年齢7, 就業) を戻して、構成する 5 歳区分の最小標本数を見る
        n_min = min(n15[d // (st.N_A * st.N_E), a15, d % st.N_E]
                    for d in f
                    for a15, a7 in st.AGE15_TO_7.items()
                    if a7 == (d // st.N_E) % st.N_A)
        rows.append({"fold": i, "groups": ",".join(str(d) for d in f),
                     "pop_share": float(share[sel].sum()), "n_layer_min": int(n_min),
                     "rate_mse": fl["rate_mse"], "rate_mae": fl["rate_mae"],
                     "dev_mse": fl["dev_mse"], "dev_rmse": fl["dev_rmse"]})
    return pd.DataFrame(rows)


def collect_heldout(csv_paths: list[Path]) -> pd.DataFrame:
    """7 本の結果 CSV から held-out 行だけを集めて 1 枚にする。

    Note:
        ★同じ群が 2 回現れたら落とす。fold の指定を間違えて同じ群を 2 回
          held-out にすると、平均が静かにその群へ寄る。holdout 列（学習時の
          config そのもの）から復元して検査する。
        ★欠落は警告に留める。7 本のうち 1 本が失敗したときに、残りだけで
          傾向を見たい場面があるため。ただし「28 群を覆った」とは言えなく
          なるので、報告に使うなら 7 本揃えること。
        ★step は fold 間で揃っている前提を置かない。事後選択で fold ごとに
          別の step が選ばれうるので、step も列に残したまま返す。

    Args:
        csv_paths: stage2_select が書いた CSV のパス。fold ごとに 1 本

    Returns:
        held-out 行だけの DataFrame。列は元の CSV と同じで、fold の出所が
        分かるよう source_csv 列を足す

    Raises:
        ValueError: held-out 行が 1 行も無い CSV があった場合、または
            同じ群が複数の fold で held-out になっている場合
    """
    frames: list[pd.DataFrame] = []
    seen: list[int] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        held = cast(pd.DataFrame, df[df["eval_kind"] == "held-out"]).copy()
        if held.empty:
            raise ValueError(
                f"held-out 行が無い: {path}\n"
                f"       --holdout-groups を指定せずに学習した結果ではないか"
                f"（28 群すべてが教師だと held-out 行は出ない）")
        held["source_csv"] = path.name
        # holdout 列は学習時の config がそのまま入っている（CSV では "[0, 4, 7, 25]"）
        seen.extend(int(d) for d in ast.literal_eval(str(held["holdout"].iloc[0])))
        frames.append(held)

    dup = sorted({d for d in seen if seen.count(d) > 1})
    if dup:
        raise ValueError(
            f"同じ群が複数の fold で held-out になっている: {dup}\n"
            f"       fold は 28 群の分割でなければならない（stratified_folds を使うこと）")
    missing = sorted(set(range(st.D_GROUPS)) - set(seen))
    if missing:
        print(f"WARNING: held-out になっていない群がある: {missing}\n"
              f"         28 群を覆っていないので、報告に使うなら全 fold を揃えること",
              file=sys.stderr)
    return pd.concat(frames, ignore_index=True)


def heldout_distribution(held: pd.DataFrame,
                         metrics: tuple[str, ...] = ("rate_mse_split", "rate_mae",
                                                     "dev_rmse")) -> pd.DataFrame:
    """fold をまたいだ held-out の分布を出す。

    Note:
        ★分布は fold 単位（1 fold = held-out 4 群をまとめて採点した 1 個）なので、
          7 fold で 7 個になる。設計書 §9.4 の「held-out 群 28 個の rate_mae /
          dev_rmse」を群ごと 28 個と読むと実装できない。**dev_* が 1 群では
          定義できない**ためである（群偏差は人口平均からのズレなので、群が 1 つ
          だとその群自身が平均になり恒等的に 0 になる）。
          28 という数は「28 群すべてが 1 回ずつ held-out 側で評価される」ことを
          指すと解釈し、報告は「7 fold の held-out 評価（各 4 群、計 28 群を網羅）」
          とする。

    Args:
        held: collect_heldout の戻り値
        metrics: 集計する指標名, default=(rate_mse_split, rate_mae, dev_rmse)

    Returns:
        指標ごとに n / mean / std / min / median / max を持つ DataFrame
    """
    sub = held[(held["mask"] == "12act") & (held["metric"].isin(metrics))]
    agg = cast(pd.DataFrame,
               sub.groupby("metric")["value"].agg(["count", "mean", "std", "min",
                                                   "median", "max"]))
    return cast(pd.DataFrame, agg.reindex([m for m in metrics if m in agg.index]))


def zeroshot_fold_rows(stage1_ckpt: Path, tgt: dict, folds: list[list[int]],
                       n: int = 1000, pool_seed: int = 12345,
                       device: str | None = None) -> pd.DataFrame:
    """微調整前の重みを、fold ごとの held-out 群で測って step=0 の基準線にする。

    ★なぜ別に測る必要があるか。`stage2_select` が CSV の先頭に入れる zero-shot 行は
      `holdout=[]` で評価されるので **held-out 行が出ない**（`evaluate_zeroshot` が
      teacher_mask を全 True に固定しているため）。その結果、held-out での改善を
      読む起点が step=25 になる。step=25 は既に 25 ステップ微調整済みで、しかも
      in-teacher 側では step0 -> 25 でわずかに悪化する山がある。起点をそこに置くと
      「zero-shot から改善した」とは言えない。

    ★プールは 1 回だけ作って 7 通りに採点する。zero-shot のモデルは fold に依存せず、
      fold は「どの群を held-out と呼ぶか」を決めるだけだからである。実測でも、
      独立に走った 3 本の fold ジョブの step=0 行 134 指標がすべて完全一致している
      （同じ Stage 1 重み・同じ pool_seed）。したがって生成は 7 分の 1 で足りる。

    Args:
        stage1_ckpt: 微調整前の Stage 1 重み
        tgt: load_stula_targets の戻り値
        folds: stratified_folds の戻り値。fold ごとの held-out 群
        n: 群あたりの生成本数。fold の評価と揃えること, default=1000
        pool_seed: プール生成の乱数種。fold の評価と揃えること, default=12345
        device: モデルを載せるデバイス。None なら model.DEVICE

    Returns:
        fold ごとの held-out / in-teacher 行を縦に積んだ DataFrame。列は
        stage2_select が書く CSV と同じなので、そのまま突き合わせられる
    """
    # ★torch を使うのはこの関数だけ。--floors / --collect は numpy と pandas しか
    #   要らないので、モジュール先頭では読まない（フロントエンドで動かせなくなる）。
    sel: Any = _load("lgo_stage2_select", HERE / "stage2_select.py")
    sm: Any = _load("lgo_simple_model", HERE / "model.py")

    dev = device or sm.DEVICE
    model = sm.load_pretrained(stage1_ckpt).to(dev)
    print(f"zero-shot 基準線: {stage1_ckpt.name} で 28群 × {n} 本を生成 "
          f"(pool_seed={pool_seed}) ...")
    _, rates, rates_a, rates_b = sel.make_pool(model, n, pool_seed)

    frames: list[pd.DataFrame] = []
    for i, f in enumerate(folds):
        teacher_mask = np.ones(sm.D_GROUPS, dtype=bool)
        for d in f:
            teacher_mask[d] = False
        base = {"ckpt": stage1_ckpt.name, "step": 0,
                "teacher_groups": int(teacher_mask.sum()),
                "n_per_group": n, "pool_seed": pool_seed,
                "holdout": list(f), "stage1_ckpt": stage1_ckpt.name,
                "lam": float("nan"), "fold": i}
        # ★採点は stage2_select.teacher_fit_rows ただ一つ。fold 側の評価と同じ関数を
        #   通すので、基準線と評価値が別定義になる余地が無い
        frames.append(pd.DataFrame(
            sel.teacher_fit_rows(rates, rates_a, rates_b, tgt, teacher_mask, base)))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2 の LGO（4群 × 7 fold）ドライバ")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS,
                    help=f"分割数。28 を割り切れること, default={N_FOLDS}")
    ap.add_argument("--print-folds", action="store_true",
                    help="fold の群割り当てを stdout へ機械可読で出す"
                        "（`fold_id 群,群,...` の形式）。層化の効き具合は stderr")
    ap.add_argument("--collect", type=Path, nargs="+", default=None,
                    help="fold ごとの結果 CSV。held-out 行を束ねて分布を出す")
    ap.add_argument("--floors", action="store_true",
                    help="fold ごとの教師の床を出す。LGO の 7 個を読む物差しになる")
    ap.add_argument("--zeroshot-baseline", type=Path, default=None,
                    help="微調整前の Stage 1 重み。fold ごとの held-out 群で測り、"
                        "step=0 の基準線 CSV を書く。これが無いと held-out の改善を "
                        "step=25 起点でしか言えない")
    ap.add_argument("--n", type=int, default=1000,
                    help="--zeroshot-baseline の群あたり生成本数。fold の評価と"
                        "揃えること, default=1000")
    ap.add_argument("--pool-seed", type=int, default=12345,
                    help="--zeroshot-baseline の乱数種。fold の評価と揃えること,"
                        " default=12345")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="--zeroshot-baseline の出力先。既定は "
                        "data/processed/aggregates/stage2_lgo_zeroshot_baseline.csv")
    args = ap.parse_args()

    if args.collect:
        held = collect_heldout(list(args.collect))
        print(f"held-out 行: {len(held)} 行 / {held['source_csv'].nunique()} fold")
        print("\n--- fold をまたいだ held-out の分布（12act）---")
        print(heldout_distribution(held).to_string())
        print("\n★この値だけが非循環である（教師に使っていない群での適合）。"
              "\n  teacher_groups=28 の in-teacher 行は学習の目的関数そのものなので"
              "\n  改善しても主張にならない（§9.2）。")
        return

    tgt = st.load_stula_targets()
    folds = stratified_folds(tgt["pop"], args.n_folds)
    shares = fold_shares(tgt["pop"], folds)

    if args.floors:
        fl = fold_floors(tgt, folds)
        print("=== fold ごとの教師の床（12act、design effect 無視の下限）===")
        print(fl.to_string(index=False,
                           formatters={"rate_mse": "{:.3e}".format,
                                       "rate_mae": "{:.5f}".format,
                                       "dev_mse": "{:.3e}".format,
                                       "dev_rmse": "{:.5f}".format,
                                       "pop_share": "{:.4f}".format}))
        print(f"\nrate_mse の床: {fl['rate_mse'].min():.3e}〜{fl['rate_mse'].max():.3e} "
              f"(比 {fl['rate_mse'].max() / fl['rate_mse'].min():.2f}倍)")
        print("★fold 間の差がこの床より小さければ、モデルの当たり外れではなく"
              "\n  教師の標本誤差の揺らぎである。fold ごとに床が違う点に注意する。")
        return

    if args.zeroshot_baseline is not None:
        out = (args.out_csv or REPO_ROOT / "data" / "processed" / "aggregates"
               / "stage2_lgo_zeroshot_baseline.csv")
        df = zeroshot_fold_rows(args.zeroshot_baseline, tgt, folds,
                                n=args.n, pool_seed=args.pool_seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n書き出し: {out}  ({len(df)} 行)")
        held = df[(df.eval_kind == "held-out") & (df["mask"] == "12act")
                  & (df.metric.isin(["rate_mse_split", "dev_rmse"]))]
        piv = held.pivot_table(index="fold", columns="metric", values="value")
        print("\n--- fold ごとの held-out 基準線（zero-shot, 12act）---")
        print(piv.to_string(float_format="{:.4e}".format))
        print("\n★これが step=0 の起点である。fold の CSV の held-out 行"
              "（step>=25）をこの行と比べること。")
        return

    if args.print_folds:
        for i, f in enumerate(folds):
            print(f"{i} {','.join(str(d) for d in f)}")
        print(f"\n層化の確認（held-out 人口シェア、{args.n_folds} fold）:",
              file=sys.stderr)
        for i, (f, s) in enumerate(zip(folds, shares)):
            print(f"  fold {i}: {s:7.4f}  群 {f}", file=sys.stderr)
        print(f"  シェアの範囲 {shares.min():.4f}〜{shares.max():.4f} "
              f"(比 {shares.max() / shares.min():.2f}倍、"
              f"変動係数 {shares.std() / shares.mean():.3f})", file=sys.stderr)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
