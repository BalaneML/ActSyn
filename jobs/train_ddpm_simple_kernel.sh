#!/bin/bash
#PBS -q SQUID-S
#PBS --group=G16263
#PBS -l elapstim_req=01:30:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_kernel
#PBS -r n
#PBS -m e
#PBS -M yokoyama.jun@ist.osaka-u.ac.jp
#
# Conv の受容野アブレーション（AggDDPM-Simple の kernel_size スイープ）。
#
# 問い: 活動切り替え回数が実データとよく一致するのは、隣接スロットを見る
#       畳み込みの局所性のおかげか。
#
# 設計: jobs/train_ddpm_simple.sh と データ・分割・学習ループ・条件付け・CFG・
#       サンプラ・乱数シードをすべて同一にし、UNet1D の畳み込みカーネルだけを
#       1/3/5/7 に振る。kernel=1 では畳み込み経路から局所混合が完全に消え、
#       スロット間の情報は attention（48/24 解像度の3ブロック）と最近傍
#       アップサンプルだけを通る。
#
#       単発 A/B ではなくスイープにするのは、容量の交絡を切り分けるため。
#       畳み込みのパラメータ数は k に比例して増える:
#           k=1: 1,035,156 / k=3: 1,759,124 / k=5: 2,483,092 / k=7: 3,207,060
#       断片化の一致が容量で決まるなら k について単調に良くなるはずで、
#       頭打ち・反転するなら容量では説明できない。
#
# 出力（${REPO} 配下。本編の成果物とは _k{K} で必ず分離される）:
#   outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_k${KERNEL}.pt
#   outputs/generated/ddpm_simple_pretrain_samples_k${KERNEL}.csv
#
# 投入手順:
#   for k in 1 3 5 7; do KERNEL=$k qsub -v KERNEL jobs/train_ddpm_simple_kernel.sh; done
#
# elapstim_req=01:30:00 の根拠:
#   同一構成の本学習 (job 1120951) が実測 7 分。k=7 は畳み込みが約2倍になるが
#   attention とオーバヘッドが律速なので 15 分程度と見込み、6倍の余裕を取る。
#   （本編ジョブの 08:00:00 は初回の見込みで、実測を受けて切り詰めた値）

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

EPOCHS="${EPOCHS:-1000}"
KERNEL="${KERNEL:?KERNEL を指定すること（例: KERNEL=1 qsub -v KERNEL ...）}"

case "${KERNEL}" in
    1|3|5|7) ;;
    *) echo "ERROR: KERNEL は 1/3/5/7 のいずれか（指定値: ${KERNEL}）" >&2; exit 1 ;;
esac

CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_k${KERNEL}.pt"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/simple_kernel${KERNEL}_${PBS_JOBID:-manual}.log"

echo "kernel: ${KERNEL}"
echo "epochs: ${EPOCHS}"
echo "log:   ${LOG}"
echo "ckpt:  ${CKPT}"

# --- 事前チェック --------------------------------------------------------
for f in "${SIF}" "${DATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done

# 本編の成果物を絶対に踏まないことを確認する。--kernel を明示した実行は
# model.py 側で _k{K} 付きの保存先に切り替わるが、二重に守る。
BASE_CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday.pt"
BASE_STAMP=""
[ -f "${BASE_CKPT}" ] && BASE_STAMP="$(stat -c %Y "${BASE_CKPT}")"

# 同じ kernel の再投入は過去の結果を退避してから
if [ -f "${CKPT}" ]; then
    BACKUP="${CKPT%.pt}_$(date +%Y%m%d_%H%M%S).pt"
    mv "${CKPT}" "${BACKUP}"
    echo "backed up existing checkpoint -> ${BACKUP}"
fi

# --- 実行環境の記録 ------------------------------------------------------
{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "kernel=${KERNEL}  epochs=${EPOCHS}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/model.py \
    --epochs "${EPOCHS}" --kernel "${KERNEL}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m"

# 本編チェックポイントが触られていないことを確認
if [ -n "${BASE_STAMP}" ]; then
    NOW_STAMP="$(stat -c %Y "${BASE_CKPT}" 2>/dev/null || echo missing)"
    if [ "${BASE_STAMP}" != "${NOW_STAMP}" ]; then
        echo "ERROR: 本編チェックポイントが変更された。調査すること: ${BASE_CKPT}" >&2
        status=1
    else
        echo "本編チェックポイントは無変更（mtime ${BASE_STAMP}）"
    fi
fi

if [ -f "${CKPT}" ]; then
    echo "checkpoint saved: ${CKPT} ($(du -h "${CKPT}" | cut -f1))"
else
    echo "WARNING: checkpoint not written. ログ末尾を確認すること: ${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
