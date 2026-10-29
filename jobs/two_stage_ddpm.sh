#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=04:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_2stage_ddpm
#PBS -r n
#PBS -m e
#
# 共通の 2 段階学習を DDPM_Minimal（幅 8・CFG なし）で回す: Stage 1 → E0（Pre-trained の採点）→ E1（Stage 2）。
#
# ★要求経過時間（上の elapstim_req）は仮の値。jobs/two_stage_ddpm_measure.sh の実測から見積もり、
#   ユーザーの承認を得てから投入時に上書きする:
#       SEED=42 jobs/submit.sh -v SEED -l elapstim_req=HH:MM:SS jobs/two_stage_ddpm.sh
#
# ★出力が既にある段は飛ばす。E1 は --resume で保存済みの最新の ckpt から続けるので、時間切れで止まっても
#   同じコマンドで再投入すれば続きから回る（25 step ごとに保存）。
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_minimal_ch8_wd0.01{_s種}.pt                                   Stage 1
#   data/processed/aggregates/stage2_common_ddpm_minimal_ch8_wd0.01_seed{種}_zeroshot.csv   E0
#   outputs/checkpoints/stage2_common_ddpm_minimal_ch8_wd0.01_lam0.01_cuda_seed{種}_all/    E1 の ckpt・history.csv
#   data/processed/aggregates/stage2_common_ddpm_minimal_ch8_wd0.01_lam0.01_cuda_seed{種}_all.csv  E1

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

SEED="${SEED:-42}"
SUFFIX=""
[ "${SEED}" != "42" ] && SUFFIX="_s${SEED}"
STAGE1="${REPO}/outputs/checkpoints/ddpm_minimal_ch8_wd0.01${SUFFIX}.pt"
E0_CSV="${REPO}/data/processed/aggregates/stage2_common_ddpm_minimal_ch8_wd0.01_seed${SEED}_zeroshot.csv"
E1_CSV="${REPO}/data/processed/aggregates/stage2_common_ddpm_minimal_ch8_wd0.01_lam0.01_cuda_seed${SEED}_all.csv"
LOG="${WORK}/logs/two_stage_ddpm_s${SEED}_${PBS_JOBID:-manual}.log"
echo "seed=${SEED}  log=${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv

set -o pipefail
{
    if [ -f "${STAGE1}" ]; then
        echo "=== Stage 1: ${STAGE1} があるので飛ばす ==="
    else
        echo "=== Stage 1 ==="
        run_gpu python src/models/DDPM_Minimal/model.py --seed "${SEED}" --device cuda || exit 1
    fi
    if [ -f "${E0_CSV}" ]; then
        echo "=== E0: ${E0_CSV} があるので飛ばす ==="
    else
        echo "=== E0（Pre-trained の採点）==="
        run_gpu python src/models/Stage2_Common/stage2_common.py --model ddpm_minimal --seed "${SEED}" \
            --zero-shot --device cuda || exit 1
    fi
    if [ -f "${E1_CSV}" ]; then
        echo "=== E1: ${E1_CSV} があるので飛ばす ==="
    else
        echo "=== E1（Stage 2、保存済みの ckpt があれば続きから）==="
        run_gpu python src/models/Stage2_Common/stage2_common.py --model ddpm_minimal --seed "${SEED}" \
            --device cuda --resume || exit 1
    fi
    echo "=== done ==="
} 2>&1 | tee "${LOG}"
