#!/bin/bash
# LGO（4群 × 7 fold）の学習を 7 本まとめて投入する。
#
# ★これは PBS ジョブではない。フロントエンドで実行する投入スクリプトであり、
#   jobs/submit.sh と同じ層にある（qsub を 7 回呼ぶ）。
#
# 何をするか:
#   28 群を人口シェアで層化して 4 群 × 7 fold に分割し、fold ごとに 1 本ずつ
#   学習ジョブを投げる。全群がちょうど 1 回ずつ held-out になるので、7 本の
#   結果を束ねると 28 群すべてについて「教師に使っていない群での適合」が揃う。
#   これが軸1 の循環を断つ唯一の証拠である（Stage2_design.md §9.2 / §9.4）。
#
# 使い方:
#   bash jobs/submit_lgo.sh 0.01              # λ=0.01 で 7 fold
#   bash jobs/submit_lgo.sh 0.01 04:00:00     # 経過時間を明示する
#
# ★λ は先に掃引で決めること（§9.4 の運用）。学習コストが 7 倍（GPU 約 25 時間）に
#   なるので、λ が定まらないまま回すとそのぶんが丸ごと無駄になる。
#
# 出力（fold ごとに分かれる）:
#   outputs/checkpoints/stage2_lam${LAM}_fold${i}/stage2_step*.pt
#
# 7 本が終わったら、fold ごとに事後選択を回してから束ねる:
#   for i in 0 1 2 3 4 5 6; do
#       CKPT_DIR=${REPO}/outputs/checkpoints/stage2_lam0.01_fold${i} \
#           jobs/submit.sh -v CKPT_DIR jobs/eval_stage2_select.sh
#   done
#   python src/models/DDPM_Aggregate_Simple/stage2_lgo.py \
#       --collect data/processed/aggregates/stage2_lam0.01_fold*_selection.csv

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source jobs/_common.sh

LAM="${1:?λ を指定すること（例: bash jobs/submit_lgo.sh 0.01）。掃引で決めた値を使う}"
ELAPS="${2:-04:00:00}"

if [ ! -f "${SIF}" ]; then
    echo "ERROR: コンテナが無い: ${SIF}" >&2
    exit 1
fi

# fold の割り当てはコンテナ内の python で出す。
# ★フロントエンドの system python には numpy / pandas が入っていないので、
#   ここで直接 python を呼ぶと ModuleNotFoundError になる。
# stdout から機械可読な `fold_id 群,群,...` の行だけを拾い、層化の確認は
# stderr としてそのまま端末へ流す。
echo "=== fold の割り当て ==="
RAW="$(singularity run "${SIF}" python \
    src/models/DDPM_Aggregate_Simple/stage2_lgo.py --print-folds)"
status=$?
if [ ${status} -ne 0 ]; then
    echo "ERROR: fold を計算できなかった（exit ${status}）" >&2
    exit 1
fi

# ★NGC コンテナは起動バナー（"== PyTorch ==" やバージョン表示など約 40 行）を
#   **stdout** へ出す。素通しすると `fold` にバナー行、`HOLDOUT` に空文字が入り、
#   holdout 無しのジョブが何十本も qsub される（2026-09-18 に実機で確認）。
#   機械可読な `fold_id 群,群,...` の行だけを残すこと。
FOLDS="$(printf '%s\n' "${RAW}" | grep -E '^[0-9]+ [0-9]+(,[0-9]+)*$')"
N_FOLD_LINES="$(printf '%s\n' "${FOLDS}" | grep -c . || true)"
if [ "${N_FOLD_LINES}" -ne 7 ]; then
    echo "ERROR: fold 行が 7 本でない（${N_FOLD_LINES} 本）。1 本も投入していない。" >&2
    echo "--- stage2_lgo.py の stdout ---" >&2
    printf '%s\n' "${RAW}" >&2
    exit 1
fi

echo
echo "=== 投入（λ=${LAM}, elapstim_req=${ELAPS}）==="
n_ok=0
while read -r fold groups; do
    [ -z "${fold}" ] && continue
    CKPT_DIR="${REPO}/outputs/checkpoints/stage2_lam${LAM}_fold${fold}"
    printf 'fold %s (群 %-14s) -> ' "${fold}" "${groups}"
    if CKPT_DIR="${CKPT_DIR}" HOLDOUT="${groups}" LAM="${LAM}" \
            jobs/submit.sh -l "elapstim_req=${ELAPS}" -v HOLDOUT,LAM,CKPT_DIR \
                jobs/train_ddpm_simple_stage2.sh; then
        n_ok=$((n_ok + 1))
    fi
done <<< "${FOLDS}"

echo
echo "投入できたジョブ: ${n_ok} 本"
if [ "${n_ok}" -ne 7 ]; then
    echo "WARNING: 7 本揃っていない。28 群を覆えないので、足りないぶんを投げ直すこと" >&2
    exit 1
fi
echo "状態は qstat、予定開始は sstat で見る。"
