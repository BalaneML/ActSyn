#!/bin/bash
#PBS -q SQUID-S
#PBS --group=<グループ名>
#PBS -l elapstim_req=08:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_simple_stage1
#PBS -r n
#PBS -m e
#PBS -M <メールアドレス>
#
# Stage1（pretrain）の本学習: AggDDPM の簡素化版
# (src/models/DDPM_Aggregate_Simple/model.py)。
#
# jobs/train_ddpm.sh（原本 UNet1D）との差は次の4点だけで、データ・分割・
# 学習ループ・条件付け・CFG は同一である。したがって両ジョブの差は
# 「削った4点の効果」に帰着する（比較実験の前提）。
#   1. データ表現   : one-hot {-1,+1} -> {0,1}
#   2. サンプリング : DDIM を持たない（ancestral のみ）
#   3. 時刻埋め込み : sinusoidal を MLP で持ち上げず直接加算（-98,816 params）
#   4. 学習        : EMA を持たない
#
# 学習内容: ATUS 2024 平日・共通12分類・28群 (性2×年齢7×就業2) の条件付き DDPM。
#           early stopping (patience 200 epoch) で val 最良を復元してから保存する。
# 学習後:   sanity_check が 28群×256個票を ancestral 1000 ステップで生成し、
#           断片化・活動シェア・暗記を実 ATUS 平日と比較する。
#           Σ|Δ|（活動シェア誤差の合計）も出力するので、原本の同じ表と直接比べられる。
#
# 出力（${REPO} 配下）:
#   outputs/checkpoints/ddpm_simple_pretrain_common12_weekday.pt
#   outputs/generated/ddpm_simple_pretrain_samples.csv
#
# 投入手順:
#   qsub jobs/smoke.sh                     # 先に DBG(10分) で通すこと
#   qsub jobs/train_ddpm_simple.sh         # 本学習（既定: 1000ep）
#   EPOCHS=300 qsub -v EPOCHS jobs/train_ddpm_simple.sh
#
# elapstim_req=08:00:00 の根拠:
#   実測 (Apple M系 MPS, fp32, 1.76M params)
#       学習 100 epoch (13 step/epoch + val) = 640〜940 s  → 約 6.4〜9.4 s/epoch
#   これを積むと MPS で 学習 1000ep ≈ 2.6h。生成は 7バッチ×1000ステップ×2(CFG) で、
#   同規模の train_ddpm_tang.sh の実測から MPS で ≈ 12h 相当。
#   A100 は本ワークロード（小さい conv と attention でオーバヘッド律速）で MPS の
#   おおむね 10 倍前後を見込み、学習 ≈ 0.3h + 生成 ≈ 1.2h ＝ 実行 1.5〜2h。
#   ここに squid_guide.md の安全率 1.2〜1.5 倍を掛け、速度比の見込み違いに耐えるよう
#   8h を確保する。★初回実行後は下の "elapsed" 行を見て切り詰めること
#   （ポイントは要求経過時間に効くため、過大な要求はそのまま浪費になる）。
#
# ★ train はチェックポイントを学習終了時に一度だけ保存する。elapstim を超過すると
#   保存前に強制終了され、その回の学習は丸ごと失われる（途中再開の口は無い）。

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

EPOCHS="${EPOCHS:-1000}"

CKPT="${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday.pt"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOG="${WORK}/logs/simple_stage1_${PBS_JOBID:-manual}.log"

echo "epochs: ${EPOCHS}"
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
# それを使った評価結果と対応が取れなくなる。
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
    echo "epochs=${EPOCHS}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

# 標準出力を明示的にファイルへ流す。既定ではジョブ終了後まで出力が書き出されず、
# 実行中の進捗が追えないため。
SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/model.py \
    --epochs "${EPOCHS}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m  (次回の elapstim_req 見直しに使う)"
if [ -f "${CKPT}" ]; then
    echo "checkpoint saved: ${CKPT} ($(du -h "${CKPT}" | cut -f1))"
else
    echo "WARNING: checkpoint not written. ログ末尾を確認すること: ${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
