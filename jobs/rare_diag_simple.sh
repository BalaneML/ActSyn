#!/bin/bash
#PBS -q DBG
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_rare
#PBS -r n
#PBS -m e
#
# 少ない活動の行動者率が外れる原因の診断のうち、生成を回す部分（AggDDPM-Simple Stage 1）。
# 中身は src/eval/diagnostics/stage1_rare_diagnosis_gen.py。集計は手元の stage1_rare_diagnosis.py。
#
#   MODE=restore      H5・H6。ATUS 平日の実個票を雑音水準 t0 まで進めてから戻す
#   MODE=epoch-pools  H4。途中の ckpt（model.py --save-every）ごとに 64 人/群の小プールを作る
#   MODE=final-continuous  H6 の補足。t0 = 999 の argmax 前の連続値をスロットごとに保存する
#
# 投入:
#   for s in 42 43 44; do ARM=clock_tf96 SEED=$s MODE=restore \
#       jobs/submit.sh -v ARM,SEED,MODE jobs/rare_diag_simple.sh; done
#   ARM=clock_tf96_traj SEED=43 MODE=epoch-pools EVERY=100 \
#       jobs/submit.sh -v ARM,SEED,MODE,EVERY jobs/rare_diag_simple.sh
#   epoch を分けて投入する（10 分に収まらないとき）:
#   ARM=clock_tf96_traj SEED=43 MODE=epoch-pools EPOCHS_LIST="600 650 700" WITH_BEST=1 \
#       jobs/submit.sh -v ARM,SEED,MODE,EPOCHS_LIST,WITH_BEST jobs/rare_diag_simple.sh
#
# 出力（${REPO} 配下）:
#   restore:      outputs/generated/ddpm_simple_restore{接尾辞}.npz
#   final-continuous: outputs/generated/ddpm_simple_restore{接尾辞}_final.npz
#   epoch-pools:  outputs/generated/ddpm_simple_pretrain_samples{接尾辞}_ep{epoch:04d}_n64.csv
#                 （WITH_BEST=1 なら ..._best_n64.csv も）
#
# elapstim_req=00:10:00（DBG の上限）の根拠:
#   7,168 本 × 1000 ステップの生成が約 2 分（jobs/guidance_pool_simple.sh の実測）。
#   restore は 3,736 人 × Σ(t0+1) = 2,867 ステップで、その約 1.5 倍（約 3 分）。
#   epoch-pools は 1 ckpt あたり 1,792 本 × 1000 ステップ（約 0.5 分）で、10 本で約 5 分。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

ARM="${ARM:?ARM を指定すること（stage1_arms.ARMS のキー）}"
SEED="${SEED:?SEED を指定すること}"
MODE="${MODE:?MODE を指定すること（restore / epoch-pools / final-continuous）}"
EVERY="${EVERY:-100}"
EPOCHS_LIST="${EPOCHS_LIST:-}"
WITH_BEST="${WITH_BEST:-0}"
LOG="${WORK}/logs/simple_rare_${MODE}_${ARM}_s${SEED}_${PBS_JOBID:-manual}.log"

ARGS=(--mode "${MODE}" --arm "${ARM}" --seed "${SEED}")
case "${MODE}" in
    restore|final-continuous) ;;
    epoch-pools)
        if [ -n "${EPOCHS_LIST}" ]; then
            # shellcheck disable=SC2206  # EPOCHS_LIST は空白区切りの epoch の並びとして展開する
            ARGS+=(--epochs ${EPOCHS_LIST})
        else
            ARGS+=(--every "${EVERY}")
        fi
        [ "${WITH_BEST}" = "1" ] && ARGS+=(--with-best)
        ;;
    *) echo "ERROR: MODE は restore / epoch-pools / final-continuous（指定値: ${MODE}）" >&2; exit 1 ;;
esac

{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "args: ${ARGS[*]}"
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/eval/diagnostics/stage1_rare_diagnosis_gen.py "${ARGS[@]}" >> "${LOG}" 2>&1
status=$?
echo "elapsed: $((SECONDS / 60))m $((SECONDS % 60))s" | tee -a "${LOG}"
echo "exit status: ${status}"
exit ${status}
