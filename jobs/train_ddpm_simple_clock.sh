#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=01:30:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_clock
#PBS -r n
#PBS -m e
#
# 時計アブレーション（AggDDPM-Simple の UNet1D に 24 時間の時計だけを足す）。
#
# 問い: Stage 1 が ATUS 自身の鋭い日内リズム（正午の食事ピーク・昼休みの仕事の凹み）を
#       半分に鈍らせるのは、「何時か」を表す経路がバックボーンに無いからか。
#       Stage 2 が日本の時刻構造（12:00 の昼食・7:00 の朝食・8:00 の通勤）を作れない
#       原因も同じか（stage2_curves の band_closure_table で周期 8h 以下の残差が 2% 台しか
#       埋まらなかった）。
#
# 設計: jobs/train_ddpm_simple.sh と データ・分割・学習ループ・条件付け・CFG・
#       サンプラ・乱数シード・kernel_size をすべて同一にし、--clock だけを足す。
#       --clock は 11 個の ResBlock1D それぞれに零初期化の clock_proj (Linear 8 -> c_out)
#       を足し、24h 周期のフーリエ特徴 φ (k=1..4) をスロットごとのバイアスとして加える
#       （+10,944 params、1,759,124 -> 1,770,068）。軸・条件の注入方法・幅は変えない。
#
# 出力（${REPO} 配下。本編の成果物とは _clock で必ず分離される）:
#   outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_clock.pt
#   outputs/generated/ddpm_simple_pretrain_samples_clock.csv
#
# 投入手順:
#   jobs/submit.sh jobs/train_ddpm_simple_clock.sh
#
# 学習後（手元へ持ち帰ってから。B4/B5 は forward だけなので手元の MPS で数分）:
#   scp squid:…/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_clock.pt outputs/checkpoints/
#   scp squid:…/outputs/generated/ddpm_simple_pretrain_samples_clock.csv outputs/generated/
#   .venv/bin/python src/eval/clock_diagnostics.py --model-dir src/models/DDPM_Aggregate_Simple \
#       --ckpt outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_clock.pt \
#       --gen outputs/generated/ddpm_simple_pretrain_samples_clock.csv --tag clock
#   基準（時計なし）は同じ手順を 20260819 版に掛けた
#   data/processed/aggregates/ddpm_clock_diagnostics_DDPM_Aggregate_Simple_20260819.csv
#
# elapstim_req=01:30:00 の根拠:
#   同一構成の本学習 (job 1120951) と kernel スイープ (k=3) が実測 7 分。
#   時計は 11 個の Linear(8, c_out) だけで計算量はほぼ変わらない。
#   train_ddpm_simple_kernel.sh と同じ枠を取る。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

EPOCHS="${EPOCHS:-1000}"

CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_clock.pt"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/simple_clock_${PBS_JOBID:-manual}.log"

echo "clock: on"
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

# 本編の成果物を絶対に踏まないことを確認する。--clock を明示した実行は
# model.py 側で _clock 付きの保存先に切り替わるが、二重に守る。
BASE_CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday.pt"
BASE_STAMP=""
[ -f "${BASE_CKPT}" ] && BASE_STAMP="$(stat -c %Y "${BASE_CKPT}")"

# 再投入は過去の結果を退避してから
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
    echo "clock=on  epochs=${EPOCHS}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/model.py \
    --epochs "${EPOCHS}" --clock >> "${LOG}" 2>&1
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
