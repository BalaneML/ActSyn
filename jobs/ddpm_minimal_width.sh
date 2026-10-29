#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=01:45:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_ddpm_min_width
#PBS -r n
#PBS -m e
#
# DDPM_Minimal の幅を 32 と 64 に広げて、Pre-trained が ATUS の活動の割合を再現するかを確かめる
# （src/models/Stage2_Common/docs/Stage1_ddpm_minimal_traj_results.md §5 の案 W。2026-10-30 ユーザー決定）。
# 幅のほかは共通の約束事のまま（学習率 1e-3・CFG なし・dropout 0・重みなし・種 42）。幅ごとに次の 3 つを回す:
#   1. Stage 1。100 epoch ごとに途中の ckpt を残す（保存しても学習は変わらない。test_model.py の f）
#   2. 各 ckpt から群あたり 64 本のプールを作り、活動の割合・系列の指標・最後の x の one-hot らしさを測る
#      （trajectory_pools.py。幅 16 の ckpt と同じ物差し）
#   3. Stage 1 の 1 epoch・Stage 2 の 1 更新・評価のプールの所要時間を測る（model.py --measure。Stage 2 の見積もり用）
#
# elapstim_req=01:45:00 の根拠（2026-10-30）:
#   幅 16 の実測（A100、ジョブ 1633858）は、Stage 1 が 1 epoch 0.33 秒、プール 1 つが 7 秒。
#   手元の CPU で batch 1,792 の前向きを測ると、幅 16 に対して幅 32 は 2.3 倍、幅 64 は 8.3 倍かかった。
#   この比を GPU にもそのまま当てはめた上限で、Stage 1（1,000 epoch）は幅 32 で 13 分・幅 64 で 46 分、
#   プール 12 個は 3 分・12 分、--measure は 2 分・7 分。起動を含めて約 85 分 × 安全率 1.2 ≒ 1 時間 45 分（ユーザー承認済み）
#
# 投入: jobs/submit.sh jobs/ddpm_minimal_width.sh
#
# ★出力が既にある段は飛ばす。時間切れで止まっても、同じコマンドで再投入すれば残りの段から回る
#   （Stage 1 は途中から再開できないので、止まった幅の Stage 1 は最初からやり直しになる）
# ★val 最良の ckpt は、その幅の本番と同じ名前になる（設定が同じなので、幅を採用したら Stage 2 の起点にそのまま使える）
#
# 出力（${REPO} 配下。幅 W = 32, 64）:
#   outputs/checkpoints/ddpm_minimal_chW_wd0.01.pt・…_ep{0100,0200,…}.pt   1 の val 最良と途中の ckpt
#   data/processed/aggregates/stage1_traj_ddpm_minimal_chW_wd0.01.csv      2 の表（1 行 = 1 つの ckpt と生成の種）
#   ${WORK}/logs/ddpm_minimal_measure_chW.log                              3 の所要時間

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

CKPT="${REPO}/outputs/checkpoints"
AGG="${REPO}/data/processed/aggregates"
LOG="${WORK}/logs/ddpm_minimal_width_${PBS_JOBID:-manual}.log"
echo "log=${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv

set -o pipefail
{
    for CH in 32 64; do
        STEM="ddpm_minimal_ch${CH}_wd0.01"     # model.py の "ddpm_minimal" + stage1_tag()（種 42・学習率 1e-3）
        if [ -f "${CKPT}/${STEM}.pt" ]; then
            echo "=== 幅 ${CH} の 1: ${STEM}.pt があるので飛ばす ==="
        else
            echo "=== 幅 ${CH} の 1: Stage 1（100 epoch ごとに途中の ckpt）==="
            run_gpu python src/models/DDPM_Minimal/model.py --base-ch "${CH}" --seed 42 --device cuda --save-every 100 || exit 1
        fi
        if [ -f "${AGG}/stage1_traj_${STEM}.csv" ]; then
            echo "=== 幅 ${CH} の 2: stage1_traj_${STEM}.csv があるので飛ばす ==="
        else
            echo "=== 幅 ${CH} の 2: ckpt ごとのプール ==="
            run_gpu python src/models/DDPM_Minimal/trajectory_pools.py --stem "${STEM}" --device cuda || exit 1
        fi
        MEASURE="${WORK}/logs/ddpm_minimal_measure_ch${CH}.log"
        if grep -q "評価のプール" "${MEASURE}" 2>/dev/null; then
            echo "=== 幅 ${CH} の 3: ${MEASURE} があるので飛ばす ==="
        else
            echo "=== 幅 ${CH} の 3: 所要時間 ==="
            run_gpu python src/models/DDPM_Minimal/model.py --base-ch "${CH}" --measure --device cuda | tee "${MEASURE}" || exit 1
        fi
    done
    echo "=== done ==="
} 2>&1 | tee "${LOG}"
