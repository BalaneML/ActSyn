#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=00:40:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_ddpm_min_seeds
#PBS -r n
#PBS -m e
#
# DDPM_Minimal の幅 32・64 の Stage 1 を種 43・44 でも回し、種 42 と合わせた 3 つの種で Stage 2 に使う幅を選ぶ
# （src/models/Stage2_Common/docs/Stage1_ddpm_minimal_width_results.md §5 の案 S。2026-10-30 ユーザー決定）。
# 選び方と予測は、結果を見る前に Stage2_common_setup.md §5 に書いた。幅と種のほかは共通の約束事のまま。
# 幅と種の組ごとに次の 2 つを回す:
#   1. Stage 1（途中の ckpt は残さない。保存名は ddpm_minimal_ch{幅}_wd0.01_s{種}）
#   2. val 最良の ckpt から、生成の種 12345・12346 で群あたり 64 本のプールを作って測る（trajectory_pools.py）
#
# elapstim_req=00:40:00 の根拠（2026-10-30、ジョブ 1634006 の実測、A100）:
#   Stage 1 は 1 epoch 0.32〜0.33 秒で、種 42 は 352〜385 epoch で止まった（2 分前後）。止まらなくても 1,000 epoch で 5.5 分。
#   プールは 1 つ 11 秒（幅 32）・20 秒（幅 64）。上限は 4 組 × 5.5 分 + プール 8 つで約 3 分 + 起動で、
#   約 27 分 × 安全率 1.5 ≒ 40 分（ユーザー承認済み）
#
# 投入: jobs/submit.sh jobs/ddpm_minimal_seeds.sh
#
# ★出力が既にある段は飛ばす。時間切れで止まっても、同じコマンドで再投入すれば残りの段から回る
# ★val 最良の ckpt は、その幅と種の本番と同じ名前になる（種を増やすときの Stage 2 の起点にそのまま使える）
#
# 出力（${REPO} 配下。幅 W = 32, 64、種 S = 43, 44）:
#   outputs/checkpoints/ddpm_minimal_chW_wd0.01_sS.pt                   1 の val 最良の ckpt
#   data/processed/aggregates/stage1_traj_ddpm_minimal_chW_wd0.01_sS.csv  2 の表（生成の種ごとに 1 行）

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

CKPT="${REPO}/outputs/checkpoints"
AGG="${REPO}/data/processed/aggregates"
LOG="${WORK}/logs/ddpm_minimal_seeds_${PBS_JOBID:-manual}.log"
echo "log=${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv

set -o pipefail
{
    for CH in 32 64; do
        for SEED in 43 44; do
            STEM="ddpm_minimal_ch${CH}_wd0.01_s${SEED}"     # model.py の "ddpm_minimal" + stage1_tag() + _s{種}（学習率 1e-3）
            if [ -f "${CKPT}/${STEM}.pt" ]; then
                echo "=== 幅 ${CH}・種 ${SEED} の 1: ${STEM}.pt があるので飛ばす ==="
            else
                echo "=== 幅 ${CH}・種 ${SEED} の 1: Stage 1 ==="
                run_gpu python src/models/DDPM_Minimal/model.py --base-ch "${CH}" --seed "${SEED}" --device cuda || exit 1
            fi
            if [ -f "${AGG}/stage1_traj_${STEM}.csv" ]; then
                echo "=== 幅 ${CH}・種 ${SEED} の 2: stage1_traj_${STEM}.csv があるので飛ばす ==="
            else
                echo "=== 幅 ${CH}・種 ${SEED} の 2: val 最良の ckpt のプール ==="
                run_gpu python src/models/DDPM_Minimal/trajectory_pools.py --stem "${STEM}" --device cuda || exit 1
            fi
        done
    done
    echo "=== done ==="
} 2>&1 | tee "${LOG}"
