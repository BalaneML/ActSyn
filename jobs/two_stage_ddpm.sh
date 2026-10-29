#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=03:30:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_2stage_ddpm
#PBS -r n
#PBS -m e
#
# 共通の 2 段階学習を DDPM_Minimal（幅 16・CFG なし）で回す: Stage 1 → E0（Pre-trained の採点）→ E1（Stage 2）。
# 幅 8 の出力（ddpm_minimal_ch8_*）は別名なので、このスクリプトは読まない・上書きしない。
#
# elapstim_req=03:30:00 の根拠（2026-10-29、幅 16 を jobs/two_stage_ddpm_measure.sh で実測、A100、ジョブ 1632014）:
#   Stage 1 は 1 epoch 0.32 秒（最大 1,000 epoch で約 5 分）、Stage 2 は 1 更新 6.9 秒。幅 8 の本番では
#   リハーサルと val を含む 1 step が測定値の 1.09 倍だったので 7.6 秒とみて、1,000 step で 2.1 時間。
#   評価のプール 56,000 本は 3.8 分（E0・E1 で 2 回）。起動を含めて約 2.4 時間 × 安全率 1.5 ≒ 3.5 時間（ユーザー承認済み）
#
# 投入: SEED=42 jobs/submit.sh -v SEED jobs/two_stage_ddpm.sh
#
# ★出力が既にある段は飛ばす。E1 は --resume で保存済みの最新の ckpt から続けるので、時間切れで止まっても
#   同じコマンドで再投入すれば続きから回る（25 step ごとに保存）。
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_minimal_ch16_wd0.01{_s種}.pt                                   Stage 1
#   data/processed/aggregates/stage2_common_ddpm_minimal_ch16_wd0.01_seed{種}_zeroshot.csv   E0
#   outputs/checkpoints/stage2_common_ddpm_minimal_ch16_wd0.01_lam0.01_cuda_seed{種}_all/    E1 の ckpt・history.csv
#   data/processed/aggregates/stage2_common_ddpm_minimal_ch16_wd0.01_lam0.01_cuda_seed{種}_all.csv  E1

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

SEED="${SEED:-42}"
SUFFIX=""
[ "${SEED}" != "42" ] && SUFFIX="_s${SEED}"
TAG="ddpm_minimal_ch16_wd0.01"     # model.py の "ddpm_minimal" + stage1_tag()。BASE_CH を変えたらここも変える
STAGE1="${REPO}/outputs/checkpoints/${TAG}${SUFFIX}.pt"
E0_CSV="${REPO}/data/processed/aggregates/stage2_common_${TAG}_seed${SEED}_zeroshot.csv"
E1_CSV="${REPO}/data/processed/aggregates/stage2_common_${TAG}_lam0.01_cuda_seed${SEED}_all.csv"
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
