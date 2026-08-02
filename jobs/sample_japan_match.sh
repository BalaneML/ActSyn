#!/bin/bash
#PBS -q SQUID-S
#PBS --group=<グループ名>
#PBS -l elapstim_req=02:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_japan_match
#PBS -r n
#
# 学習済み AggDDPM を使った日本側マッチング実験（サンプリング + 指数傾け）。
# train_ddpm.sh の完了後に投入する。
#
# 投入: qsub jobs/sample_japan_match.sh
#
# 学習より短時間で終わるため elapstim_req を短くしてポイント消費を抑える。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

LOG="${WORK}/logs/japan_match_${PBS_JOBID:-manual}.log"

echo "log: ${LOG}"

run_gpu python src/models/DDPM_Aggregate/japan_match_experiment.py \
    --reuse-pool \
    --ridge-sweep \
    > "${LOG}" 2>&1
status=$?

echo "exit status: ${status}"
exit ${status}
