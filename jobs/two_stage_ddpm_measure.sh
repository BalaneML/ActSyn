#!/bin/bash
#PBS -q DBG
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_2stage_ddpm_measure
#PBS -r n
#
# 共通の 2 段階学習の DDPM_Minimal（幅 16・CFG なし）の所要時間を測る（DBG キュー、最大 10 分）。
# 本番（jobs/two_stage_ddpm.sh）の要求経過時間は、ここで出た値から見積もる。
#
# 測るもの（src/models/DDPM_Minimal/model.py の measure）:
#   1. Stage 1 の 1 epoch（3 epoch 学習した中央値。学習結果は捨てる）
#   2. Stage 2 の 1 更新（逆過程 1,000 ステップ × 1,792 本、末尾 1 ステップの逆伝播。リハーサルと val は含まない）
#   3. 評価のプール 56,000 本（8,176 本で測って換算）
#
# 投入: jobs/submit.sh jobs/two_stage_ddpm_measure.sh

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

LOG="${WORK}/logs/two_stage_ddpm_measure_${PBS_JOBID:-manual}.log"
echo "log: ${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv
run_gpu python src/models/DDPM_Minimal/model.py --measure --device cuda 2>&1 | tee "${LOG}"
