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
# 学習の種による反復（時計の効果を種のばらつきから切り分ける）:
#   CLOCK=1 SEED=43 jobs/submit.sh -v CLOCK,SEED jobs/train_ddpm_simple_clock.sh   # 時計つき・種43
#   CLOCK=0 SEED=43 jobs/submit.sh -v CLOCK,SEED jobs/train_ddpm_simple_clock.sh   # 時計なし・種43
#   SEED は学習の乱数だけを変え、学習/評価の分割は変えない（model.py の --seed）。
#   保存先は model.py と同じ規則で _clock / _s{SEED} が付く。SEED 未指定は種42（本編と同じ）。
#   ★CLOCK=0 かつ SEED 未指定は本編の再学習になり保存先が本編と重なるので、このジョブでは弾く
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
CLOCK="${CLOCK:-1}"
SEED="${SEED:-}"
# 行動者率の偏りの項 L_rate（model.py の --rate-lam / --rate-gamma）。
# 未指定なら従来の損失。例:
#   RATE_LAM=1 jobs/submit.sh -q DBG -l elapstim_req=00:10:00 -v RATE_LAM jobs/train_ddpm_simple_clock.sh
RATE_LAM="${RATE_LAM:-}"
RATE_GAMMA="${RATE_GAMMA:-}"

case "${CLOCK}" in
    0|1) ;;
    *) echo "ERROR: CLOCK は 0 か 1（指定値: ${CLOCK}）" >&2; exit 1 ;;
esac
if [ "${CLOCK}" = "0" ] && [ -z "${SEED}" ]; then
    echo "ERROR: CLOCK=0 で SEED 未指定は本編の再学習になる（保存先が本編と重なる）。SEED を指定すること" >&2
    exit 1
fi

if [ -n "${RATE_GAMMA}" ] && [ -z "${RATE_LAM}" ]; then
    echo "ERROR: RATE_GAMMA は RATE_LAM と一緒に指定すること（λ=0 では効かない）" >&2
    exit 1
fi

# model.py の保存先の規則（_clock -> _rate{λ}[g{γ}] -> _s{SEED} の順）と同じ名前を組む。
# ★λ・γ の書式は Python の f"{x:g}" と printf '%g' で揃える（1.0 -> 1, 0.1 -> 0.1）
SUFFIX=""
ARGS=(--epochs "${EPOCHS}")
if [ "${CLOCK}" = "1" ]; then SUFFIX="${SUFFIX}_clock"; ARGS+=(--clock); fi
if [ -n "${RATE_LAM}" ]; then
    SUFFIX="${SUFFIX}_rate$(printf '%g' "${RATE_LAM}")"
    ARGS+=(--rate-lam "${RATE_LAM}")
    if [ -n "${RATE_GAMMA}" ]; then
        SUFFIX="${SUFFIX}g$(printf '%g' "${RATE_GAMMA}")"
        ARGS+=(--rate-gamma "${RATE_GAMMA}")
    fi
fi
if [ -n "${SEED}" ]; then SUFFIX="${SUFFIX}_s${SEED}"; ARGS+=(--seed "${SEED}"); fi

CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday${SUFFIX}.pt"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/simple${SUFFIX}_${PBS_JOBID:-manual}.log"

echo "clock: ${CLOCK}  seed: ${SEED:-42(既定)}  rate_lam: ${RATE_LAM:-0}  rate_gamma: ${RATE_GAMMA:-1(既定)}"
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
    echo "clock=${CLOCK}  seed=${SEED:-42}  epochs=${EPOCHS}  rate_lam=${RATE_LAM:-0}  rate_gamma=${RATE_GAMMA:-1}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/model.py "${ARGS[@]}" >> "${LOG}" 2>&1
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
