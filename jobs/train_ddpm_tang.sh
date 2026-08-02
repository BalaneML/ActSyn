#!/bin/bash
#PBS -q SQUID-S
#PBS --group=<グループ名>
#PBS -l elapstim_req=08:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_tang_stage1
#PBS -r n
#PBS -m e
#PBS -M <メールアドレス>
#
# Stage1（pretrain）の本学習: AggDDPM の denoiser を Tang et al. 2025 バックボーン
# (src/models/DDPM_Aggregate_Tang/model.py) に置き換えた条件付き学習。
# データ・分割・学習ループは jobs/train_ddpm.sh と完全に同一で、denoiser だけが違う。
# したがって両ジョブの差は「構造の差」だけに帰着する（比較実験の前提）。
#
# 学習内容: ATUS 2024 平日・共通12分類・28群 (性2×年齢7×就業2) の条件付き DDPM。
#           early stopping (patience 200 epoch) で val 最良を復元してから保存する。
# 学習後:   sanity_check が 28群×256個票を ancestral 1000 ステップで生成し、
#           断片化・活動シェア・暗記を実 ATUS 平日と比較する。
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_tang_w050_pretrain_common12_weekday.pt  ← Stage2 が読む
#   outputs/generated/ddpm_tang_w050_pretrain_samples.csv
#
# 投入手順:
#   qsub jobs/smoke.sh                              # 先に DBG(10分) で通すこと
#   qsub jobs/train_ddpm_tang.sh                    # 本学習（既定: 1000ep, width 0.5）
#   EPOCHS=1500 qsub -v EPOCHS jobs/train_ddpm_tang.sh
#   WIDTH_SCALE=1.0 qsub -v WIDTH_SCALE jobs/train_ddpm_tang.sh   # Tang 忠実な幅
#   （WIDTH_SCALE を変えるとチェックポイント名も変わるので既定幅の結果は潰れない）
#
# elapstim_req=08:00:00 の根拠:
#   実測 (Apple M系 MPS, fp32, width_scale=0.5 / 3.83M params)
#       学習 1 step (B=256)      2.16 s   → 13 step/epoch + val 2 batch ≈ 30 s/epoch
#       推論 1 forward (B=1024)  3.11 s
#   これを積むと MPS で 学習 1000ep ≈ 8.3h、生成 7バッチ×1000ステップ×2(CFG) ≈ 12h。
#   A100 は本ワークロード（小さい conv と attention でオーバヘッド律速）で MPS の
#   おおむね 10 倍前後を見込み、学習 ≈ 1h + 生成 ≈ 1.2h ＝ 実行 2〜3h。
#   ここに squid_guide.md の安全率 1.2〜1.5 倍を掛けたうえで、速度比の見込み違いに
#   耐えるよう 8h を確保する。★初回実行後は下の "elapsed" 行を見て切り詰めること
#   （ポイントは要求経過時間に効くため、過大な要求はそのまま浪費になる）。
#
# ★ agg.train はチェックポイントを学習終了時に一度だけ保存する。elapstim を超過すると
#   保存前に強制終了され、その回の学習は丸ごと失われる（途中再開の口は無い）。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

EPOCHS="${EPOCHS:-1000}"
WIDTH_SCALE="${WIDTH_SCALE:-0.5}"

# model.py の _tag() と同じ規則 (0.5 -> w050)。bash に浮動小数演算が無いので awk で丸める
TAG="w$(awk -v w="${WIDTH_SCALE}" 'BEGIN { printf "%03d", int(w * 100 + 0.5) }')"
CKPT="${REPO}/outputs/checkpoints/ddpm_tang_${TAG}_pretrain_common12_weekday.pt"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/tang_stage1_${PBS_JOBID:-manual}.log"

echo "epochs: ${EPOCHS}  width_scale: ${WIDTH_SCALE}"
echo "log:   ${LOG}"
echo "ckpt:  ${CKPT}"

# --- 事前チェック --------------------------------------------------------
# GPU ポイントは CPU の約6倍。入力欠落で数分後に落ちるジョブを投入前に潰す。
for f in "${SIF}" "${DATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done

# 既存チェックポイントは退避する。再投入で過去の Stage1 を黙って上書きすると、
# それを使った Stage2・評価結果と対応が取れなくなる。
if [ -f "${CKPT}" ]; then
    BACKUP="${CKPT%.pt}_$(date +%Y%m%d_%H%M%S).pt"
    mv "${CKPT}" "${BACKUP}"
    echo "backed up existing checkpoint -> ${BACKUP}"
fi

# --- 実行環境の記録 ------------------------------------------------------
# 論文の数値をどのコード・どのGPUで出したかを後から辿れるようにする。
{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "epochs=${EPOCHS} width_scale=${WIDTH_SCALE}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

# 標準出力を明示的にファイルへ流す。既定ではジョブ終了後まで出力が書き出されず、
# 実行中の進捗が追えないため。
SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Tang/model.py \
    --epochs "${EPOCHS}" --width-scale "${WIDTH_SCALE}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m  (次回の elapstim_req 見直しに使う)"
if [ -f "${CKPT}" ]; then
    echo "checkpoint saved: ${CKPT} ($(du -h "${CKPT}" | cut -f1))"
else
    echo "WARNING: checkpoint not written. ログ末尾を確認すること: ${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
