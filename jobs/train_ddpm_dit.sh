#!/bin/bash
#PBS -q SQUID-S
#PBS --group=<グループ名>
#PBS -l elapstim_req=04:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_dit_stage1
#PBS -r n
#PBS -m e
#PBS -M <メールアドレス>
#
# Stage1（pretrain）の本学習: AggDDPM の denoiser を Diffusion Transformer
# (src/models/DDPM_Aggregate_DiT/model.py; Peebles & Xie 2023) に置き換えた条件付き学習。
# データ・分割・学習ループは jobs/train_ddpm.sh / jobs/train_ddpm_tang.sh と完全に同一で、
# denoiser だけが違う。したがって3ジョブの差は「構造の差」だけに帰着する（比較実験の前提）。
#
# 学習内容: ATUS 2024 平日・共通12分類・28群 (性2×年齢7×就業2) の条件付き DDPM。
#           early stopping (patience 200 epoch) で val 最良を復元してから保存する。
# 学習後:   sanity_check が 28群×256個票を ancestral 1000 ステップで生成し、
#           断片化・活動シェア・暗記を実 ATUS 平日と比較する。
#
# 既定構成 (SIZE=xs): depth 6 / hidden 128 / heads 4 / patch 1 → 1,852,628 params。
#   現行 UNet1D (1,857,940) と params が 0.3% しか違わないので、両者の差を
#   「容量の差」でなく「構造の差」に帰着できる。
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_dit_xs_p1_pretrain_common12_weekday.pt  ← Stage2 が読む
#   outputs/generated/ddpm_dit_xs_p1_pretrain_samples.csv
#
# 投入手順:
#   qsub jobs/smoke.sh                          # 先に DBG(10分) で通すこと
#   qsub jobs/train_ddpm_dit.sh                 # 本学習（既定: 1000ep, size=xs, patch=1）
#   EPOCHS=1500 qsub -v EPOCHS jobs/train_ddpm_dit.sh
#   SIZE=t qsub -v SIZE jobs/train_ddpm_dit.sh          # Tang w0.5 と同容量 (4.14M)
#   SIZE=s qsub -v SIZE jobs/train_ddpm_dit.sh          # DiT-S 忠実 (32.4M)
#   PATCH=2 qsub -v PATCH jobs/train_ddpm_dit.sh        # 1トークン=30分。断片化への効果を見る
#   （SIZE / PATCH を変えるとチェックポイント名も変わるので既定構成の結果は潰れない）
#
# elapstim_req=04:00:00 の根拠:
#   実測 (Apple M系 MPS, fp32, size=xs / 1.85M params)
#       学習 1 step (B=256)      0.737 s  → 13 step/epoch + val 2 batch ≈ 11 s/epoch
#       推論 1 forward (B=1024)  0.944 s
#   これを積むと MPS で 学習 1000ep ≈ 2.9h、生成 7バッチ×1000ステップ×2(CFG) ≈ 3.7h。
#   A100 は本ワークロードで MPS のおおむね 10 倍前後を見込み、学習 ≈ 20m + 生成 ≈ 22m。
#   ここに squid_guide.md の安全率 1.2〜1.5 倍を掛けても 1h 弱だが、速度比の見込み違いに
#   耐えるよう 4h を確保する（Tang バックボーンの 8h の半分。DiT の方が実測で約3倍速い:
#   Tang w0.5 は 2.16 s/step・3.11 s/forward）。
#   ★初回実行後は下の "elapsed" 行を見て切り詰めること
#   （ポイントは要求経過時間に効くため、過大な要求はそのまま浪費になる）。
#   ★SIZE=s (32.4M) は既定の 17 倍の params なので、この見積りは使えない。
#     elapstim_req を上げてから投入すること。
#
# ★ agg.train はチェックポイントを学習終了時に一度だけ保存する。elapstim を超過すると
#   保存前に強制終了され、その回の学習は丸ごと失われる（途中再開の口は無い）。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

EPOCHS="${EPOCHS:-1000}"
SIZE="${SIZE:-xs}"
PATCH="${PATCH:-1}"

# model.py の _tag() と同じ規則 (xs, 1 -> xs_p1)
TAG="${SIZE}_p${PATCH}"
CKPT="${REPO}/outputs/checkpoints/ddpm_dit_${TAG}_pretrain_common12_weekday.pt"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/dit_stage1_${PBS_JOBID:-manual}.log"

echo "epochs: ${EPOCHS}  size: ${SIZE}  patch: ${PATCH}"
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
    echo "epochs=${EPOCHS} size=${SIZE} patch=${PATCH}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

# 標準出力を明示的にファイルへ流す。既定ではジョブ終了後まで出力が書き出されず、
# 実行中の進捗が追えないため。
SECONDS=0
run_gpu python src/models/DDPM_Aggregate_DiT/model.py \
    --epochs "${EPOCHS}" --size "${SIZE}" --patch "${PATCH}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m  (次回の elapstim_req 見直しに使う)"
if [ -f "${CKPT}" ]; then
    echo "checkpoint saved: ${CKPT} ($(du -h "${CKPT}" | cut -f1))"
else
    echo "WARNING: checkpoint not written. ログ末尾を確認すること: ${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
