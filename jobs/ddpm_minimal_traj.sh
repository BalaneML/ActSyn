#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=00:30:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_ddpm_min_traj
#PBS -r n
#PBS -m e
#
# DDPM_Minimal（幅 16）の Pre-trained が ATUS の活動の割合を再現しない原因を切り分ける
# （src/models/Stage2_Common/docs/Stage2_common_results.md §7 の案 A と B を 1 本で。2026-10-29 ユーザー決定）:
#   1. Stage 1 を本番と同じ設定（学習率 1e-3）で回し直し、50 epoch ごとに途中の ckpt を残す
#      （保存名に _traj を付けるので、E0・E1 に使った本番の ckpt は上書きしない）
#   2. Stage 1 を学習率 2e-4（既存の DDPM と同じ値）で回し、50 epoch ごとに途中の ckpt を残す（保存名 _lr0.0002）
#   3. 各 ckpt から群あたり 64 本のプールを作り、活動の割合・系列の指標・最後の x の one-hot らしさを測る
#      （trajectory_pools.py。本番の ckpt も同じ物差しで測り、E0 の値とのつながりを確かめる）
#
# elapstim_req=00:30:00 の根拠（2026-10-29、幅 16 の実測：ジョブ 1632014・1632018、A100）:
#   Stage 1 は 1 epoch 0.33 秒で、1 本あたり最大 1,000 epoch・約 5.5 分（2 本で約 11 分）。
#   プールは 1,792 本の逆過程 1,000 ステップを最大 42 回（1 回 10 秒以下とみて 7 分以下）、指標の計算は 1 回 0.2 秒。
#   起動を含めて約 20 分 × 安全率 1.5 ≒ 30 分
#
# 投入: jobs/submit.sh jobs/ddpm_minimal_traj.sh
#
# ★出力が既にある段は飛ばす。時間切れで止まっても、同じコマンドで再投入すれば残りの段から回る
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_minimal_ch16_wd0.01_traj.pt・…_traj_ep{0050,0100,…}.pt          1 の val 最良と途中の ckpt
#   outputs/checkpoints/ddpm_minimal_ch16_wd0.01_lr0.0002.pt・…_lr0.0002_ep{0050,0100,…}.pt  2 の val 最良と途中の ckpt
#   data/processed/aggregates/stage1_traj_ddpm_minimal_ch16_wd0.01{,_traj,_lr0.0002}.csv      3 の表（1 行 = 1 ckpt と生成の種）

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

CKPT="${REPO}/outputs/checkpoints"
AGG="${REPO}/data/processed/aggregates"
BASE="ddpm_minimal_ch16_wd0.01"     # model.py の "ddpm_minimal" + stage1_tag()（種 42・学習率 1e-3）
LOG="${WORK}/logs/ddpm_minimal_traj_${PBS_JOBID:-manual}.log"
echo "log=${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv

set -o pipefail
{
    if [ -f "${CKPT}/${BASE}_traj.pt" ]; then
        echo "=== 1: ${BASE}_traj.pt があるので飛ばす ==="
    else
        echo "=== 1: Stage 1（学習率 1e-3、50 epoch ごとに途中の ckpt）==="
        run_gpu python src/models/DDPM_Minimal/model.py --seed 42 --device cuda --save-every 50 --run-tag _traj || exit 1
    fi
    if [ -f "${CKPT}/${BASE}_lr0.0002.pt" ]; then
        echo "=== 2: ${BASE}_lr0.0002.pt があるので飛ばす ==="
    else
        echo "=== 2: Stage 1（学習率 2e-4、50 epoch ごとに途中の ckpt）==="
        run_gpu python src/models/DDPM_Minimal/model.py --seed 42 --device cuda --save-every 50 --lr 2e-4 || exit 1
    fi
    for STEM in "${BASE}" "${BASE}_traj" "${BASE}_lr0.0002"; do
        if [ -f "${AGG}/stage1_traj_${STEM}.csv" ]; then
            echo "=== 3: stage1_traj_${STEM}.csv があるので飛ばす ==="
        else
            echo "=== 3: ${STEM} の ckpt ごとのプール ==="
            run_gpu python src/models/DDPM_Minimal/trajectory_pools.py --stem "${STEM}" --device cuda || exit 1
        fi
    done
    echo "=== done ==="
} 2>&1 | tee "${LOG}"
