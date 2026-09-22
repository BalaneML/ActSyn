"""
stage2_wandb_artifacts.py
=========================
LGO 7 fold の成果物を wandb の各 run へ artifact として添付する

設計: src/models/DDPM_Aggregate_Simple/docs/Stage2_evaluation_results.md §9

**何のためのモジュールか。** LGO の学習は GPU 25 時間かかり、成果物は SQUID の
`work` 領域にしか無い。`work` はバックアップ領域ではないので、消えたら作り直しになる。
学習曲線は既に wandb のオフライン run に記録されているので、**同じ run に重み・ログ・
評価結果を付けて 1 か所にまとめる**。

★このモジュールは**フロントエンドで動かす**。計算ノードは外部ネットワークへ出られない
  ので、wandb への送信はフロントエンドからしか行えない。PBS ジョブではない。

★`wandb sync` より**後**に実行すること。`resume="must"` は run が既にオンラインに
  存在することを要求する。順序は `jobs/upload_lgo_wandb.sh` が保証する。

データフロー:

```mermaid
flowchart TB
    subgraph SQUID["SQUID work 領域（フロントエンドから見える）"]
        OFF["$WORK/wandb/wandb/<br/>offline-run-*-RUN_ID<br/>★1階層深い"]
        CKPT["outputs/checkpoints/<br/>stage2_lam0.003_fold{i}/<br/>stage2_step200.pt"]
        TLOG["$WORK/logs/<br/>simple_stage2_0:{train_job}.sqd.log"]
        ELOG["$WORK/logs/<br/>stage2_select_0:{eval_job}.sqd.log"]
        CSV["data/processed/aggregates/<br/>stage2_lam0.003_fold{i}_selection.csv"]
    end
    OFF -->|"wandb sync（jobs/upload_lgo_wandb.sh）"| RUN["wandb run RUN_ID<br/>project=WANDB_PROJECT"]
    CKPT -->|"log_artifact type=model"| RUN
    TLOG --> LOGART["type=log"] --> RUN
    ELOG --> LOGART
    CSV -->|"log_artifact type=dataset"| RUN
```

★添付するのは `stage2_step200.pt` だけである。step200 は λ 掃引（28 群すべてが教師）
  で**事前に**決めた世代であり、held-out の値を見て選び直していない。選び直すと
  held-out が「使っていないデータ」でなくなり、非循環性の主張が成り立たなくなる。
  他の 11 世代は SQUID に残る。

使い方（SQUID のフロントエンド）:
    bash jobs/upload_lgo_wandb.sh --dry-run    # 何をどこへ送るか確認
    bash jobs/upload_lgo_wandb.sh              # 本番

    # python を直接呼ぶ場合（sync は済んでいること）
    singularity run ${SIF} python src/models/DDPM_Aggregate_Simple/stage2_wandb_artifacts.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

WANDB_PROJECT = "domain-transfer-ddpm-agg"      # stage2_finetune.run :612 と同じ
LAM = "0.003"                                   # 掃引で決めた λ
SEL_STEP = 200                                  # 掃引で事前に決めた世代

# fold -> (wandb run id, held-out 群, 学習ジョブ, 評価ジョブ)
#
# ★この表がコード側の唯一の出所である。run id は実機のログから拾った
#   （`grep -oE "offline-run-[0-9_]+-[a-z0-9]+"` を学習ログに当てる）。
#   held-out 群は学習時の config そのもので、ckpt 内にも同じ値が入っている。
FOLDS: dict[int, tuple[str, tuple[int, ...], str, str]] = {
    0: ("me2bbley", (0, 4, 7, 25), "1320205", "1325424"),
    1: ("37vkv5gu", (6, 10, 14, 26), "1320206", "1325425"),
    2: ("xkqwnbn9", (2, 11, 15, 21), "1320207", "1325426"),
    3: ("ro0kgfnz", (1, 5, 8, 17), "1320208", "1325427"),
    4: ("woe7z8a3", (9, 12, 22, 27), "1320209", "1325428"),
    5: ("tlnl6va8", (16, 19, 20, 23), "1320210", "1325429"),
    6: ("sml7viek", (3, 13, 18, 24), "1320211", "1325432"),
}

# zero-shot 基準線のジョブ。fold に属さないので、fold 0 の run へまとめて付ける
# （全 fold で共通の 1 本しか無く、どこか 1 か所に置くしかないため）
BASELINE_JOB = "1334854"
BASELINE_CSV = "stage2_lgo_zeroshot_baseline.csv"


def work_dir() -> Path:
    """SQUID の work 領域を返す。

    Note:
        ★`jobs/_common.sh` の `WORK` と同じ場所を指す必要がある。環境変数が
          無いときはグループ名から組み立てず、明示的に落とす。推測で別の場所を
          見に行くと「ファイルが無い」ではなく「別人の領域を見た」になりうる。

    Returns:
        work 領域のパス

    Raises:
        SystemExit: WORK が環境に無い場合
    """
    w = os.environ.get("WORK")
    if not w:
        raise SystemExit(
            "ERROR: 環境変数 WORK が無い。jobs/upload_lgo_wandb.sh 経由で実行すること\n"
            "       （直接動かすなら export WORK=/sqfs/work/<グループ>/<利用者番号>）")
    return Path(w)


def fold_files(fold: int, work: Path) -> dict[str, list[Path]]:
    """1 fold ぶんの添付候補を artifact 種別ごとに集める。

    Args:
        fold: fold 番号 0..6
        work: work 領域のパス

    Returns:
        artifact 種別 -> ファイルの list。存在しないものは含めない
    """
    _, _, train_job, eval_job = FOLDS[fold]
    ckpt = (REPO_ROOT / "outputs" / "checkpoints" / f"stage2_lam{LAM}_fold{fold}"
            / f"stage2_step{SEL_STEP}.pt")
    logs = [work / "logs" / f"simple_stage2_0:{train_job}.sqd.log",
            work / "logs" / f"stage2_select_0:{eval_job}.sqd.log"]
    csvs = [REPO_ROOT / "data" / "processed" / "aggregates"
            / f"stage2_lam{LAM}_fold{fold}_selection.csv"]
    # ★基準線は全 fold 共通の 1 本。fold 0 にだけ付ける
    if fold == 0:
        logs.append(work / "logs" / f"stage2_zsbase_0:{BASELINE_JOB}.sqd.log")
        csvs.append(REPO_ROOT / "data" / "processed" / "aggregates" / BASELINE_CSV)
    return {"model": [ckpt] if ckpt.exists() else [],
            "log": [p for p in logs if p.exists()],
            "dataset": [p for p in csvs if p.exists()]}


def describe(fold: int, kind: str) -> tuple[str, str]:
    """artifact の名前と説明を作る。

    Args:
        fold: fold 番号
        kind: artifact 種別（model / log / dataset）

    Returns:
        (artifact 名, 説明文)
    """
    _, held, train_job, eval_job = FOLDS[fold]
    g = ",".join(str(d) for d in held)
    return (f"stage2_lgo_fold{fold}_{kind}", {
        "model": (f"LGO fold{fold} の step{SEL_STEP} の重み（λ={LAM}）。"
                  f"held-out 群 {g} は教師に使っていない。"
                  f"step{SEL_STEP} は λ 掃引で事前に決めた世代で、"
                  f"held-out の値を見て選び直してはいない"),
        "log": (f"LGO fold{fold} のジョブログ。学習 {train_job} / 評価 {eval_job}"),
        "dataset": (f"LGO fold{fold} の事後選択の結果 CSV（縦持ち、2763 行）。"
                    f"eval_kind=held-out の行だけが非循環である"),
    }[kind])


def build_plan(work: Path) -> list[tuple[int, str, list[Path]]]:
    """送信内容を組み立てる。送信はしない。

    Args:
        work: work 領域のパス

    Returns:
        (fold, artifact 種別, ファイル) の list
    """
    plan: list[tuple[int, str, list[Path]]] = []
    for fold in sorted(FOLDS):
        files = fold_files(fold, work)
        for kind in ("model", "log", "dataset"):
            if files[kind]:
                plan.append((fold, kind, files[kind]))
            else:
                print(f"WARNING: fold{fold} の {kind} が 1 つも見つからない。飛ばす",
                      file=sys.stderr)
    return plan


def print_plan(plan: list[tuple[int, str, list[Path]]]) -> int:
    """送信内容を印字し、合計バイト数を返す。

    Args:
        plan: build_plan の戻り値

    Returns:
        合計バイト数
    """
    total = 0
    for fold, kind, files in plan:
        run_id, held, _, _ = FOLDS[fold]
        size = sum(p.stat().st_size for p in files)
        total += size
        name, _ = describe(fold, kind)
        print(f"  fold{fold} (run {run_id}, held-out {','.join(map(str, held))})")
        print(f"    {name:<32} {kind:<8} {len(files)} file  {size / 1e6:8.1f} MB")
        for p in files:
            print(f"      {p}")
    print(f"\n  合計 {total / 1e6:.1f} MB を "
          f"wandb project '{WANDB_PROJECT}' の {len(FOLDS)} run へ送る")
    return total


def upload(plan: list[tuple[int, str, list[Path]]]) -> int:
    """artifact を各 run へ添付する。

    Note:
        ★`resume="must"` を使う。run が無ければ例外で落ちる。`allow` にすると
          新しい run を勝手に作ってしまい、学習曲線と切り離された空の run が
          project に増える。

    Args:
        plan: build_plan の戻り値

    Returns:
        失敗した fold の数
    """
    import wandb

    by_fold: dict[int, list[tuple[str, list[Path]]]] = {}
    for fold, kind, files in plan:
        by_fold.setdefault(fold, []).append((kind, files))

    n_fail = 0
    for fold in sorted(by_fold):
        run_id, held, _, _ = FOLDS[fold]
        print(f"\n--- fold{fold} -> run {run_id} ---", flush=True)
        try:
            run = wandb.init(project=WANDB_PROJECT, id=run_id, resume="must")
        except Exception as exc:                        # noqa: BLE001
            print(f"ERROR: run {run_id} を開けない: {exc}\n"
                  f"       先に wandb sync したか確認すること", file=sys.stderr)
            n_fail += 1
            continue
        try:
            for kind, files in by_fold[fold]:
                name, desc = describe(fold, kind)
                art = wandb.Artifact(name=name, type=kind, description=desc,
                                     metadata={"fold": fold, "lam": float(LAM),
                                               "step": SEL_STEP,
                                               "holdout": list(held)})
                for p in files:
                    art.add_file(str(p))
                run.log_artifact(art)
                print(f"  {name} ({kind}, {len(files)} file) を添付", flush=True)
        finally:
            run.finish()
    return n_fail


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LGO 7 fold の成果物を wandb の各 run へ添付する")
    ap.add_argument("--dry-run", action="store_true",
                    help="送信せず、何をどの run へ送るかだけ印字する")
    args = ap.parse_args()

    work = work_dir()
    print(f"work    : {work}")
    print(f"repo    : {REPO_ROOT}")
    print(f"project : {WANDB_PROJECT}")
    print(f"λ={LAM}  添付する世代 = step{SEL_STEP}\n")

    plan = build_plan(work)
    print_plan(plan)

    if args.dry_run:
        print("\n--dry-run のため送信しない。"
              "内容を確認したら --dry-run を外して実行すること")
        return

    n_fail = upload(plan)
    print(f"\n失敗した fold: {n_fail}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
