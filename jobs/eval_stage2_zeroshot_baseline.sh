#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=01:00:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_stage2_zsbase
#PBS -r n
#PBS -m e
#
# LGO の zero-shot 基準線: 微調整前の重みを fold ごとの held-out 群で測る
# 設計: src/models/DDPM_Aggregate_Simple/docs/Stage2_design.md §9.4
#
# ★何の穴を埋めるジョブか。
#   `stage2_select` が CSV の先頭に入れる step=0 の行は `holdout=[]` で評価される
#   （evaluate_zeroshot が teacher_mask を全 True に固定している）。そのため
#   **held-out 行が出ない**。結果として「教師に使っていない群で改善した」の起点が
#   step=25 になるが、step=25 は既に 25 ステップ微調整済みであり、しかも in-teacher
#   側では step0 -> 25 でわずかに悪化する山がある。起点をそこに置いたままでは
#   zero-shot からの改善を主張できない。
#
# ★なぜ 1 本で 7 fold 分が取れるか。
#   zero-shot のモデルは fold に依存しない。fold が決めるのは「どの群を held-out と
#   呼ぶか」だけである。実測でも、独立に走った 3 本の fold ジョブ（1325424 /
#   1325425 / 1325432）の step=0 行 134 指標がすべて完全一致している（同じ Stage 1
#   重み・同じ pool_seed=12345）。したがってプールは 1 回作れば足り、生成は
#   fold ごとに回す場合の 7 分の 1 で済む。
#
# データフロー:
#
# ```mermaid
# flowchart TD
#     S1["ddpm_simple_pretrain_*.pt<br/>（微調整前 = Stage 1）"]
#     S1 --> MP["stage2_select.make_pool<br/>torch.manual_seed(POOL_SEED)<br/>28群 × N 本"]
#     MP --> R["rates / rates_a / rates_b<br/>(28, 12*96)"]
#     R --> TF["stage2_select.teacher_fit_rows<br/>fold ごとに teacher_mask を変えて 7 回"]
#     TF --> CSV["stage2_lgo_zeroshot_baseline.csv<br/>step=0 の held-out 行 × 7 fold"]
#     CSV --> CMP["fold の *_selection.csv の<br/>held-out 行（step>=25）と比較"]
# ```
#
# ★採点は stage2_select.teacher_fit_rows ただ一つを通す。fold 側の評価と同じ関数
#   なので、基準線と評価値が別定義になる余地が無い。
#
# 所要:
#   生成は 28群 × N 本の 1 回だけ。155 秒 / 7168 本（§4.7）より N=1000 で約 10 分。
#   枠 1 時間はコンテナ起動と教師の読み込みを含めた余裕。
#
# 使い方:
#   jobs/submit.sh jobs/eval_stage2_zeroshot_baseline.sh
#   N=2000 jobs/submit.sh -v N jobs/eval_stage2_zeroshot_baseline.sh
#
# ★N と POOL_SEED は fold の評価と必ず揃えること。揃っていないと基準線との差が
#   モデルの差ではなく生成本数・乱数の差になる。既定は fold 側と同じ 1000 / 12345。
#
# 出力:
#   data/processed/aggregates/stage2_lgo_zeroshot_baseline.csv

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

N="${N:-1000}"
POOL_SEED="${POOL_SEED:-12345}"
STAGE1="${STAGE1:-${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_20260819.pt}"
OUT_CSV="${OUT_CSV:-${REPO}/data/processed/aggregates/stage2_lgo_zeroshot_baseline.csv}"
TEACHER="${REPO}/data/processed/stula/timeband_weekday.csv"
LOG="${WORK}/logs/stage2_zsbase_${PBS_JOBID:-manual}.log"

echo "=== LGO zero-shot 基準線  job=${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "stage1=${STAGE1}"
echo "n=${N}  pool_seed=${POOL_SEED}"

# --- 事前チェック: 入力 ---------------------------------------------------
for f in "${SIF}" "${TEACHER}" "${STAGE1}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done

# ★fold 側の評価と N / POOL_SEED が揃っているかを実物で確かめる。
#   揃っていない基準線は、差がモデルの差でなくなるので意味を成さない。
EXIST="$(ls "${REPO}"/data/processed/aggregates/stage2_lam*_fold*_selection.csv 2>/dev/null | head -1)"
if [ -n "${EXIST}" ]; then
    SEEN="$(awk -F, 'NR==2 {for(i=1;i<=NF;i++) h[i]=$i} NR==2 {print}' "${EXIST}" >/dev/null 2>&1; \
            head -2 "${EXIST}" | tail -1 | cut -d, -f4,5)"
    echo "fold 側の (n_per_group, pool_seed) = ${SEEN}   ← 今回の ${N},${POOL_SEED} と一致すべき"
    if [ "${SEEN}" != "${N},${POOL_SEED}" ]; then
        echo "ERROR: fold の評価と n / pool_seed が違う（${SEEN} vs ${N},${POOL_SEED}）。" >&2
        echo "       基準線との差がモデルの差でなくなるので止める。" >&2
        exit 1
    fi
else
    echo "WARNING: 比較対象の *_fold*_selection.csv がまだ無い。n / pool_seed の照合を飛ばす" >&2
fi

echo "log:  ${LOG}"
echo "out:  ${OUT_CSV}"
mkdir -p "$(dirname "${OUT_CSV}")"

{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "stage1=${STAGE1}  n=${N}  pool_seed=${POOL_SEED}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
run_gpu python src/models/DDPM_Aggregate_Simple/stage2_lgo.py \
    --zeroshot-baseline "${STAGE1}" --n "${N}" --pool-seed "${POOL_SEED}" \
    --out-csv "${OUT_CSV}" >> "${LOG}" 2>&1
status=$?

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m"

if [ -f "${OUT_CSV}" ]; then
    echo "書き出し: ${OUT_CSV}  ($(wc -l < "${OUT_CSV}" | tr -d ' ') 行)"
    tail -20 "${LOG}"
else
    echo "WARNING: CSV が書かれていない。ログ末尾を確認すること: ${LOG}" >&2
    tail -30 "${LOG}" >&2
fi

echo "exit status: ${status}"
exit ${status}
