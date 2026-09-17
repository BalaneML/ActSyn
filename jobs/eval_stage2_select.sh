#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=05:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_stage2_select
#PBS -r n
#PBS -m e
#
# Stage 2 の事後チェックポイント選択（教師適合 × ガードレール）
# 設計: src/models/DDPM_Aggregate_Simple/docs/Stage2_design.md §8.4 / §9.8
#
# 何をするジョブか:
#   学習ジョブが残した stage2_step*.pt を 1 つずつ読み、群別に n 本ずつ生成して
#   (軸1 教師適合, 軸2 ガードレール) の 2 軸で採点し、1 本の CSV に落とす。
#   全 ckpt を共通乱数 pool_seed で生成するので、ckpt 間の差から生成の
#   モンテカルロ雑音が抜ける。
#
# ★これは学習とは別のジョブである。学習 3.2 時間のあとに、この生成コストが丸ごと乗る。
#
# elapstim_req=05:00:00 の根拠:
#   生成の実測は B=7,168 で 155 秒（§4.7）。1 ckpt = 28群 × n 本なので
#
#       所要 ≒ ckpt数 × 28 × n / 7168 × 155 秒
#
#   | SAVE_EVERY | ckpt数 | N=2000 | N=1000 |
#   |---|---|---|---|
#   | 25 | 12 | 4.0 時間 | 2.0 時間 |
#   | 50 |  6 | 2.0 時間 | 1.0 時間 |
#
#   ★λ を決める粗い掃引の段階では SAVE_EVERY=50 + N=1000 で十分である
#     （n=1000 の rate_mae の MC 床は 0.00384、zero-shot 比 13.7%。n=2000 なら 9.6%）。
#     選んだ近傍だけ N=2000 で測り直すこと。
#   既定は最も重い側（12 ckpt × 2000）でも収まる 5 時間にしてある。
#   CKPT_DIR の中身を数えて見積りを出すので、投入前にログ冒頭を見て切り詰めること。
#
# 出力:
#   data/processed/aggregates/stage2_checkpoint_selection.csv
#   logs/stage2_select_${PBS_JOBID}.log
#
# 投入手順:
#   jobs/submit.sh jobs/eval_stage2_select.sh
#
#   環境変数で上書きできる:
#     N=2000 POOL_SEED=12345 CKPT_DIR=... OUT_CSV=...
#   例: N=1000 jobs/submit.sh -v N jobs/eval_stage2_select.sh
#   λ 掃引の 1 本を採点する（CKPT_DIR は学習ジョブが出力したもの）:
#     CKPT_DIR=${REPO}/outputs/checkpoints/stage2_lam0.01 N=1000 \
#         jobs/submit.sh -v CKPT_DIR,N jobs/eval_stage2_select.sh

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

N="${N:-2000}"
POOL_SEED="${POOL_SEED:-12345}"
CKPT_DIR="${CKPT_DIR:-${REPO}/outputs/checkpoints/stage2}"
# ★出力も CKPT_DIR ごとに分ける。λ 掃引では ckpt ディレクトリが λ ごとに分かれるので、
#   固定名にすると後から回した λ が前の結果を上書きしてしまう。
OUT_CSV="${OUT_CSV:-${REPO}/data/processed/aggregates/$(basename "${CKPT_DIR}")_selection.csv}"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
TEACHER="${REPO}/data/processed/stula/timeband_weekday.csv"
LOG="${WORK}/logs/stage2_select_${PBS_JOBID:-manual}.log"

# --- 事前チェック 1: 入力 ------------------------------------------------
for f in "${SIF}" "${TEACHER}" "${DATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done
if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: チェックポイントのディレクトリが無い: ${CKPT_DIR}" >&2
    exit 1
fi

# --- 事前チェック 2: ★所要時間の見積り ------------------------------------
# 5 時間の枠に収まらない組み合わせを、生成を始める前に落とす。
# 途中で打ち切られると CSV が 1 行も書かれない（run は全 ckpt を回し終えてから書く）。
N_CKPT="$(find "${CKPT_DIR}" -maxdepth 1 -name 'stage2_step*.pt' | wc -l | tr -d ' ')"
if [ "${N_CKPT}" -eq 0 ]; then
    echo "ERROR: stage2_step*.pt が 1 つも無い: ${CKPT_DIR}" >&2
    echo "       先に jobs/train_ddpm_simple_stage2.sh を流すこと" >&2
    exit 1
fi
# 155 秒 / 7168 本 を基準に秒で見積もる（整数演算のため 1000 倍して計算）
EST_SEC=$(( N_CKPT * 28 * N * 155 / 7168 ))
LIMIT_SEC=$(( 5 * 3600 ))
echo "ckpt=${N_CKPT} 本  n=${N}  pool_seed=${POOL_SEED}"
echo "生成本数 = ${N_CKPT} × 28 × ${N} = $(( N_CKPT * 28 * N )) 本"
echo "所要見積り = $(( EST_SEC / 3600 ))h $(( (EST_SEC % 3600) / 60 ))m  (枠 5h)"
if [ "${EST_SEC}" -gt "${LIMIT_SEC}" ]; then
    echo "ERROR: 見積り $(( EST_SEC / 3600 ))h が枠 5h を超えている。" >&2
    echo "       逃げ道は (1) N を下げる（1000 なら半分。MC 床 0.00271 -> 0.00384）" >&2
    echo "                (2) 学習側の SAVE_EVERY を上げて ckpt 数を減らす" >&2
    echo "                (3) elapstim_req を伸ばす" >&2
    exit 1
fi

echo "log:  ${LOG}"
echo "out:  ${OUT_CSV}"
mkdir -p "$(dirname "${OUT_CSV}")"

# --- 実行環境の記録 ------------------------------------------------------
{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "ckpt_dir=${CKPT_DIR}  ckpt=${N_CKPT} 本"
    echo "n=${N} pool_seed=${POOL_SEED}"
    echo "est=$(( EST_SEC / 3600 ))h $(( (EST_SEC % 3600) / 60 ))m"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/stage2_select.py \
    --ckpt-dir "${CKPT_DIR}" --n "${N}" --out-csv "${OUT_CSV}" \
    --pool-seed "${POOL_SEED}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m  (次回の elapstim_req 見直しに使う)"

if [ -f "${OUT_CSV}" ]; then
    echo "書き出し: ${OUT_CSV}  ($(wc -l < "${OUT_CSV}" | tr -d ' ') 行)"
    echo "次に見る列:"
    echo "  軸1  rate_mae / dev_rmse   ★dev_rmse が改善しないまま rate_mae だけ下がるのは"
    echo "                              「さらに平滑化した」だけの可能性がある"
    echo "  軸2  *_vs_zeroshot          1 を超えたら zero-shot より悪化"
    echo "  暗記 dcr_gap                ckpt を追って上がるなら暗記寄り"
else
    echo "WARNING: CSV が書かれていない。ログ末尾を確認すること: ${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
