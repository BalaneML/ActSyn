#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=05:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_2stage_ddpm
#PBS -r n
#PBS -m e
#
# 共通の 2 段階学習を DDPM_Minimal（幅 32・CFG なし）で回す: Stage 1 → E0（Pre-trained の採点）→ E1（Stage 2）。
# 幅 32 は、幅 32・64 を種 42〜44 で比べて事前の選び方で決めた（2026-10-30。
# src/models/Stage2_Common/docs/Stage1_ddpm_minimal_seeds_results.md）。
# 幅 8・16 の出力（ddpm_minimal_ch8_*・ddpm_minimal_ch16_*）は別名なので、このスクリプトは読まない・上書きしない。
#
# elapstim_req=05:00:00 の根拠（2026-10-30、幅 32 を model.py --measure で実測、A100、ジョブ 1634006）:
#   種 42 の Stage 1 はジョブ 1634006 で保存済みなので回さない（ほかの種でも 1 epoch 0.32 秒、最大 1,000 epoch で約 5 分）。
#   Stage 2 は 1 更新 10.5 秒。幅 16 の本番（ジョブ 1632018）ではリハーサルと val を含む 1 step が測定値の
#   1.09 倍だったので 11.4 秒とみて、1,000 step で約 3.2 時間。評価のプール 56,000 本は 5.4 分（E0・E1 で 2 回）。
#   起動を含めて約 3.4 時間 × 安全率 1.5 ≒ 5 時間（ユーザー承認済み）
#
# 投入: SEED=42 jobs/submit.sh -v SEED jobs/two_stage_ddpm.sh
#       SEED=42 LR=1e-4 jobs/submit.sh -v SEED,LR jobs/two_stage_ddpm.sh   （Stage 2 の学習率を変えるとき）
#
# LR（既定 0.001）は E1 の学習率。既定と違えば E1 の保存名に _lr{値} が付く（stage2_common.py の variant_tag）。
# DDPM（幅 32）は 2026-11-01 に 1e-4 に決めた（ユーザー決定）。1e-3 だと、新しく作る AdamW の 1 回目の更新で
# Pre-trained が崩れる（src/models/Stage2_Common/docs/Stage2_common_results.md §7）。E0 は学習率によらないので共通。
#
# ★出力が既にある段は飛ばす。E1 は --resume で保存済みの最新の ckpt から続けるので、時間切れで止まっても
#   同じコマンドで再投入すれば続きから回る（25 step ごとに保存）。
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_minimal_ch32_wd0.01{_s種}.pt                                   Stage 1
#   data/processed/aggregates/stage2_common_ddpm_minimal_ch32_wd0.01_seed{種}_zeroshot.csv   E0
#   outputs/checkpoints/stage2_common_ddpm_minimal_ch32_wd0.01_lam0.01[_lr値]_cuda_seed{種}_all/    E1 の ckpt・history.csv
#   data/processed/aggregates/stage2_common_ddpm_minimal_ch32_wd0.01_lam0.01[_lr値]_cuda_seed{種}_all.csv  E1

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

SEED="${SEED:-42}"
LR="${LR:-0.001}"
SUFFIX=""
[ "${SEED}" != "42" ] && SUFFIX="_s${SEED}"
# 保存名の学習率の印は Python の f"{lr:g}" と同じ書き方（C の printf %g と一致する）。既定 0.001 なら付けない
LR_TAG=""
[ "$(printf '%g' "${LR}")" != "0.001" ] && LR_TAG="_lr$(printf '%g' "${LR}")"
TAG="ddpm_minimal_ch32_wd0.01"     # model.py の "ddpm_minimal" + stage1_tag()。BASE_CH を変えたらここも変える
STAGE1="${REPO}/outputs/checkpoints/${TAG}${SUFFIX}.pt"
E0_CSV="${REPO}/data/processed/aggregates/stage2_common_${TAG}_seed${SEED}_zeroshot.csv"
E1_CSV="${REPO}/data/processed/aggregates/stage2_common_${TAG}_lam0.01${LR_TAG}_cuda_seed${SEED}_all.csv"
LOG="${WORK}/logs/two_stage_ddpm_s${SEED}${LR_TAG}_${PBS_JOBID:-manual}.log"
echo "seed=${SEED}  lr=${LR}  log=${LOG}"
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
            --device cuda --lr "${LR}" --resume || exit 1
    fi
    echo "=== done ==="
} 2>&1 | tee "${LOG}"
