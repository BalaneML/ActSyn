#!/bin/bash
#PBS -q SQUID-S
#PBS --group=<グループ名>
#PBS -l elapstim_req=24:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_ddpm_train
#PBS -r n
#PBS -m e
#PBS -M <メールアドレス>
#
# AggDDPM の pretrain（ATUS平日・共通12分類・28群の条件付き学習）。
#
# 投入: qsub jobs/train_ddpm.sh
#
# elapstim_req は超過した瞬間にジョブが強制終了され、保存前のモデルは失われる。
# 見積り時間の 1.2〜1.5 倍を指定すること。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

# 環境変数で上書きできる（`EPOCHS=1000 qsub -v EPOCHS jobs/train_ddpm.sh`）。
# train_ddpm_tang.sh と同じ流儀に揃えてある。
EPOCHS="${EPOCHS:-300}"
LOG="${WORK}/logs/ddpm_train_${PBS_JOBID:-manual}.log"

echo "log: ${LOG}"

# 標準出力を明示的にファイルへ流す。既定ではジョブ終了後まで出力が書き出されず、
# 実行中の進捗が追えないため。
run_gpu python src/models/DDPM_Aggregate/model.py --epochs "${EPOCHS}" > "${LOG}" 2>&1
status=$?

echo "exit status: ${status}"
exit ${status}
