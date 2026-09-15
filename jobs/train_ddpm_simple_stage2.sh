#!/bin/bash
#PBS -q SQUID-S
#PBS --group=G16263
#PBS -l elapstim_req=06:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_stage2
#PBS -r n
#PBS -m e
#PBS -M yokoyama.jun@ist.osaka-u.ac.jp
#
# AggDDPM-Simple Stage 2: 公表集計表だけを教師にした微調整
# 設計: src/models/DDPM_Aggregate_Simple/docs/Stage2_design.md
#
# 何をするジョブか:
#   Stage 1（ATUS 個票で事前学習した重み）を出発点に、社会生活基本調査の公表集計表
#   第8-1表を教師にしてパラメータを更新する。日本側の個票は 1 本も使わない。
#   1更新 = 群あたり n 本を打ち切り逆伝播つきで生成し、群集計を教師に合わせる。
#
# elapstim_req=06:00:00 の根拠（§4.7）:
#   生成 1 回の実測は B=7,168 で 155 秒（job 1120951 のファイル mtime から算出。
#   ただし生成と評価の合計なので生成コストの上限として使っている）。
#   D_sub=7・n=256 なら B=1,792 で 1 更新 39 秒、300 更新で 3.2 時間。その約2倍。
#
#   ★この2倍のマージンは「1更新あたりの実測が無い初回に限った措置」である。
#     Stage 1 の「超過すると丸ごと消えるから過大側に振る」という理屈は Stage 2 では
#     使わない（下の定期チェックポイントで再開できるため）。SQUID のポイントは
#     要求経過時間に効く可能性があり、過大な要求はそのまま浪費になりうる。
#     elapsed 行を見た 2 回目以降は切り詰めること。
#
# 出力（${WORK} 配下）:
#   outputs/checkpoints/stage2/stage2_step{N}.pt   ★複数世代を残す（§8.4 の事後選択）
#   logs/simple_stage2_${PBS_JOBID}.log
#
# 投入手順:
#   qsub jobs/smoke.sh                    # まず DBG キューで 10 分の動作確認
#   qsub jobs/train_ddpm_simple_stage2.sh
#
#   環境変数で上書きできる:
#     STEPS=300 D_SUB=7 N=256 K=1 CHUNK=0 EPS=inf LAM=auto LOSS=sq HOLDOUT= RESUME=0
#     CHUNK=0 は予算からの自動決定。B 未満になると2パス勾配蓄積へ切り替わる
#     （勾配は一括計算と厳密に一致するので、下がっても学習の意味は変わらない）
#   例: EPS=0.01 qsub -v EPS jobs/train_ddpm_simple_stage2.sh     # 主B（χ²）
#       RESUME=1 qsub -v RESUME jobs/train_ddpm_simple_stage2.sh  # 途中から再開

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

STEPS="${STEPS:-300}"
D_SUB="${D_SUB:-7}"
N="${N:-256}"
K="${K:-1}"
CHUNK="${CHUNK:-0}"
EPS="${EPS:-inf}"
LOSS="${LOSS:-sq}"
LAM="${LAM:-auto}"
HOLDOUT="${HOLDOUT:-}"
RESUME="${RESUME:-0}"
SAVE_EVERY="${SAVE_EVERY:-25}"

STAGE1="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_20260819.pt"
TEACHER="${REPO}/data/processed/stula/timeband_weekday.csv"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
CKPT_DIR="${REPO}/outputs/checkpoints/stage2"
LOG="${WORK}/logs/simple_stage2_${PBS_JOBID:-manual}.log"

echo "steps=${STEPS} d_sub=${D_SUB} n=${N} K=${K} chunk=${CHUNK} eps=${EPS} loss=${LOSS} lam=${LAM}"
echo "holdout='${HOLDOUT}' resume=${RESUME}"
echo "log:  ${LOG}"
echo "ckpt: ${CKPT_DIR}"

# --- 事前チェック 1: 入力ファイル ---------------------------------------
for f in "${SIF}" "${STAGE1}" "${TEACHER}" "${DATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done

# --- 事前チェック 2: ★VRAM 予算 ----------------------------------------
# 1パス版のピークは K × D_sub × n で決まる（§4.5）。既存ジョブは nvidia-smi を
# ログに書くだけで検証していないが、Stage 2 は 5 時間級なので、超過なら即座に落とす。
# 5 時間回した後に OOM で失う事故を防ぐのが目的。
#
# ★ここは実機の VRAM から予算を引き直す粗いゲートである（A100 40GB なら 6733）。
#   「想定より小さい GPU を引いた」を捕まえるのが役目で、厳密な判定は Python 側の
#   stage2_finetune.check_memory_budget（MEMORY_BUDGET=6600 固定）が行う。
#   両方通らないと学習は始まらないので、実効的にはより厳しい 6600 が効く。
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if [ -z "${VRAM_MIB}" ]; then
    echo "ERROR: nvidia-smi から VRAM 容量を読めない" >&2
    exit 1
fi
# 予算 = (VRAM - 固定費 1.7GB) / (4.664 MB per K×sample) × 0.8（断片化の余裕2割）
BUDGET=$(( (VRAM_MIB - 1700) * 1000 / 4664 * 8 / 10 ))
TOTAL=$(( D_SUB * N ))
# CHUNK=0（自動）なら Python 側が予算に収まるところまで chunk を下げて2パスへ切り替える。
# ここではその「下がる先」で判定し、自動でも収まらない場合だけを弾く。
# CHUNK を明示したときはその値をそのまま判定する。
if [ "${CHUNK}" -gt 0 ]; then
    LOAD=$(( K * CHUNK ))
else
    LOAD=$(( K * TOTAL ))
    if [ "${LOAD}" -gt "${BUDGET}" ]; then
        LOAD=$(( K * (BUDGET / K) ))
        echo "note: B=${TOTAL} は1パスで収まらないので2パス勾配蓄積へ自動で切り替わる"
    fi
fi
echo "VRAM=${VRAM_MIB} MiB  予算 K×chunk <= ${BUDGET}  要求=${LOAD}  (B=${TOTAL})"
if [ "${LOAD}" -gt "${BUDGET}" ]; then
    echo "ERROR: K×chunk = ${LOAD} が予算 ${BUDGET} を超えている（K=${K} CHUNK=${CHUNK}）。" >&2
    echo "       逃げ道は優先順に (1) CHUNK を下げる／0 にして自動決定させる" >&2
    echo "                        (2) D_SUB を下げる  (3) 勾配チェックポイント" >&2
    exit 1
fi

# --- 過去の世代を退避 ----------------------------------------------------
# ★RESUME=1 のときは退避しない（最新世代から続きを回すため）
if [ "${RESUME}" != "1" ] && [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
    BACKUP="${CKPT_DIR}_$(date +%Y%m%d_%H%M%S)"
    mv "${CKPT_DIR}" "${BACKUP}"
    echo "backed up existing checkpoints -> ${BACKUP}"
fi
mkdir -p "${CKPT_DIR}"

# --- 実行環境の記録 ------------------------------------------------------
{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "stage1: ${STAGE1}"
    echo "steps=${STEPS} d_sub=${D_SUB} n=${N} K=${K} chunk=${CHUNK} eps=${EPS} loss=${LOSS} lam=${LAM}"
    echo "holdout='${HOLDOUT}' resume=${RESUME} save_every=${SAVE_EVERY}"
    echo "VRAM=${VRAM_MIB} MiB  budget=${BUDGET}  load=${LOAD}  B=${TOTAL}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

ARGS=(--steps "${STEPS}" --d-sub "${D_SUB}" --n "${N}" --K "${K}" --chunk "${CHUNK}"
      --eps "${EPS}" --loss "${LOSS}" --lam "${LAM}"
      --save-every "${SAVE_EVERY}" --ckpt-dir "${CKPT_DIR}" --stage1-ckpt "${STAGE1}")
[ -n "${HOLDOUT}" ] && ARGS+=(--holdout-groups "${HOLDOUT}")
[ "${RESUME}" = "1" ] && ARGS+=(--resume)

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/stage2_finetune.py "${ARGS[@]}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m"

N_CKPT="$(ls -1 "${CKPT_DIR}"/stage2_step*.pt 2>/dev/null | wc -l)"
if [ "${N_CKPT}" -gt 0 ]; then
    echo "checkpoints saved: ${N_CKPT} 世代 in ${CKPT_DIR}"
    echo "次は事後選択:"
    echo "  python src/models/DDPM_Aggregate_Simple/stage2_select.py --ckpt-dir ${CKPT_DIR}"
else
    echo "WARNING: checkpoint が1つも書かれていない。ログ末尾を確認すること: ${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
