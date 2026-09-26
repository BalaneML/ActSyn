#!/bin/bash
#PBS -q DBG
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_guid
#PBS -r n
#PBS -m e
#
# Stage 1 の学習済みモデルから、CFG の強さを変えて生成プールを作り直す（アブレーション段 0）。
# 中身は src/eval/diagnostics/stage1_guidance_pool.py（同じ乱数の種で guidance だけを変える）。
#
# 投入:
#   for s in 42 43 44; do ARM=clock SEED=$s jobs/submit.sh -v ARM,SEED jobs/guidance_pool_simple.sh; done
#
# 出力（${REPO} 配下）:
#   outputs/generated/ddpm_simple_pretrain_samples{接尾辞}_g{scale}.csv（scale = 1 と 1.25）
#
# elapstim_req=00:10:00 の根拠:
#   学習ジョブの生成部分（7,168 本、guidance 1.25）が約 2 分。1.0 と 1.25 の 2 本で 5 分未満。
#   手元の MPS では 1 本 74 分かかったので、生成はこのジョブで行う。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

ARM="${ARM:?ARM を指定すること（stage1_arms.ARMS のキー）}"
SEED="${SEED:?SEED を指定すること}"
GUIDANCE="${GUIDANCE:-1.0 1.25}"
LOG="${WORK}/logs/simple_guidance_${ARM}_s${SEED}_${PBS_JOBID:-manual}.log"

{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "arm=${ARM}  seed=${SEED}  guidance=${GUIDANCE}"
} > "${LOG}" 2>&1

SECONDS=0
# shellcheck disable=SC2086  # GUIDANCE は空白区切りの値の並びとして展開する
run_gpu python src/eval/diagnostics/stage1_guidance_pool.py \
    --arm "${ARM}" --seeds "${SEED}" --guidance ${GUIDANCE} >> "${LOG}" 2>&1
status=$?
echo "elapsed: $((SECONDS / 60))m $((SECONDS % 60))s" | tee -a "${LOG}"
echo "exit status: ${status}"
exit ${status}
