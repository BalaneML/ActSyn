#!/bin/bash
#PBS -q DBG
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_arm
#PBS -r n
#PBS -m e
#
# Stage 1 アブレーションの 1 arm × 1 種を学習する（AggDDPM-Simple）。
#
# arm の中身（構造 ArchSpec・損失 rate_lam / rate_mode・保存先の接尾辞）は
# src/models/DDPM_Aggregate_Simple/stage1_arms.py の ARMS が唯一の出所。
# このスクリプトは名前を組み立てない。保存先の接尾辞もコンテナ内の Python で
# stage1_arms.arm_suffix から引く（shell と Python の二重管理で生成プールを取り違えないため）。
#
# 投入:
#   ARM=clock_h48 SEED=43 jobs/submit.sh -v ARM,SEED jobs/train_ddpm_simple_arm.sh
#   for s in 42 43 44; do ARM=clock_h48 SEED=$s jobs/submit.sh -v ARM,SEED jobs/train_ddpm_simple_arm.sh; done
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_simple_pretrain_common12_weekday{接尾辞}.pt
#   outputs/generated/ddpm_simple_pretrain_samples{接尾辞}.csv
#   接尾辞 = ARMS[ARM].suffix + (_s{SEED}、SEED != 42 のとき)
#
# elapstim_req=00:10:00（DBG の上限）の根拠:
#   時刻符号つき（K=4）の Stage 1 は学習と 7,168 本の生成で実測 8 分。
#   96 解像度の attention を足す arm は計算が増えるので、最初の 1 本で所要時間を確かめる。
#   10 分を超えて打ち切られたら SQUID-S（-q SQUID-S -l elapstim_req=01:30:00）へ回す。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

ARM="${ARM:?ARM を指定すること（stage1_arms.ARMS のキー）}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1000}"

case "${SEED}" in
    ''|*[!0-9]*) echo "ERROR: SEED は非負の整数（指定値: ${SEED}）" >&2; exit 1 ;;
esac

# 保存先の接尾辞を arm の表から引く。未知の arm はここで止まる
SUFFIX="$(singularity exec "${SIF}" python -c "
import sys
sys.path.insert(0, '${REPO}/src/models/DDPM_Aggregate_Simple')
import stage1_arms
print(stage1_arms.arm_suffix('${ARM}', ${SEED}))
")" || { echo "ERROR: arm_suffix を引けない（ARM=${ARM}）" >&2; exit 1; }

CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday${SUFFIX}.pt"
POOL="${REPO}/outputs/generated/ddpm_simple_pretrain_samples${SUFFIX}.csv"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/simple_arm_${ARM}_s${SEED}_${PBS_JOBID:-manual}.log"

echo "arm: ${ARM}  seed: ${SEED}  epochs: ${EPOCHS}  suffix: ${SUFFIX}"
echo "log:   ${LOG}"
echo "ckpt:  ${CKPT}"
echo "pool:  ${POOL}"

# --- 事前チェック --------------------------------------------------------
for f in "${SIF}" "${DATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done

# 本編の成果物を踏まないことを確認する（model.py --arm も noclock 種 42 の学習を禁止している）
BASE_CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday.pt"
BASE_STAMP=""
[ -f "${BASE_CKPT}" ] && BASE_STAMP="$(stat -c %Y "${BASE_CKPT}")"

# 再投入は過去の結果を退避してから（ckpt と生成プールの組を同じ時刻印で退避する）
STAMP="$(date +%Y%m%d_%H%M%S)"
for f in "${CKPT}" "${POOL}"; do
    if [ -f "${f}" ]; then
        mv "${f}" "${f%.*}_${STAMP}.${f##*.}"
        echo "backed up existing -> ${f%.*}_${STAMP}.${f##*.}"
    fi
done

# --- 実行環境の記録 ------------------------------------------------------
{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "arm=${ARM}  seed=${SEED}  epochs=${EPOCHS}  suffix=${SUFFIX}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/model.py \
    --arm "${ARM}" --seed "${SEED}" --epochs "${EPOCHS}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 60))m $((SECONDS % 60))s" | tee -a "${LOG}"

# 本編チェックポイントが触られていないことを確認
if [ -n "${BASE_STAMP}" ]; then
    NOW_STAMP="$(stat -c %Y "${BASE_CKPT}" 2>/dev/null || echo missing)"
    if [ "${BASE_STAMP}" != "${NOW_STAMP}" ]; then
        echo "ERROR: 本編チェックポイントが変更された。調査すること: ${BASE_CKPT}" >&2
        status=1
    fi
fi

for f in "${CKPT}" "${POOL}"; do
    if [ -f "${f}" ]; then
        echo "saved: ${f} ($(du -h "${f}" | cut -f1))"
    else
        echo "WARNING: not written: ${f}。ログ末尾を確認すること: ${LOG}" >&2
        status=1
    fi
done

echo "exit status: ${status}"
exit ${status}
