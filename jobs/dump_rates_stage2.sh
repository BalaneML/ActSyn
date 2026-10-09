#!/bin/bash
#PBS -q SQUID-S
#PBS -l elapstim_req=00:30:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_dump_rates
#PBS -r n
#PBS -m e
#
# 生成プールと時刻別行動者率を .npz へ落とす（指標は測らない）
#
# ★何のためのジョブか。
#   `stage2_select` が書く CSV はスカラーの指標だけで、群ごと・スロットごとの値を
#   残していない。そのため「人口加重で測り直す」「曲線を描く」といった、あとから
#   出てくる要求のたびに 28群 × N 本の生成をやり直すことになる。
#   プールを一度落としておけば、以後はどんな重み付け・どんな統計量でも CPU だけで
#   計算できる。1 世代あたり数 MB しかない。
#
# ★N と POOL_SEED は評価 CSV と必ず揃えること。
#   揃っていれば、落とした rates から計算した指標は CSV の値と**厳密に一致する**
#   （`dump_rates` は `make_pool` を通るので生成経路も乱数も同じ）。これは
#   「この npz はその CSV と同じプールである」ことの検算になる。
#   既定 1000 / 12345 は λ 掃引・LGO の両方と同じ値である。
#
# データフロー:
#
# ```mermaid
# flowchart TD
#     CKPT["stage2_step200.pt<br/>（CKPTS で列挙）"]
#     CKPT --> DR["stage2_select.dump_rates<br/>make_pool(model, N, POOL_SEED)"]
#     DR --> POOL["pool (28, N, 96) int8"]
#     DR --> RATES["rates / rates_a / rates_b<br/>(28, 12*96)"]
#     POOL --> NPZ["outputs/generated/<ckpt名>_rates.npz"]
#     RATES --> NPZ
#     NPZ --> CURVES["stage2_curves.py --rates<br/>人口加重の時刻別行動者率"]
# ```
#
# 所要:
#   1 世代あたり生成 155 秒 / 7168 本（§4.7）より N=1000 で約 10 分。
#   枠 30 分はコンテナ起動と余裕を含む。世代を増やすときは枠も伸ばすこと。
#
# 使い方:
#   # λ 掃引（28群教師）の step200 だけ
#   jobs/submit.sh jobs/dump_rates_stage2.sh
#
#   # 微調整前（Stage 1）も同じ N・同じ seed で揃えたいとき（+ 約 10 分）
#   CKPTS="outputs/checkpoints/stage2_lam0.003/stage2_step200.pt outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_20260819.pt" \
#       jobs/submit.sh -l elapstim_req=01:00:00 -v CKPTS jobs/dump_rates_stage2.sh
#
# 出力:
#   outputs/generated/<ckpt名>_rates.npz
#
# 手元へ持ち帰る:
#   scp squid:~/…/outputs/generated/*_rates.npz outputs/generated/
#   .venv/bin/python src/models/DDPM_Aggregate_Simple/stage2_curves.py \
#       --samples outputs/generated/ddpm_simple_pretrain_samples_20260819.csv=zero-shot \
#       --rates   outputs/generated/stage2_step200_rates.npz=step200 \
#       --out outputs/figures/stage2_weighted_slot_rates.png

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

N="${N:-1000}"
POOL_SEED="${POOL_SEED:-12345}"
# 既定は λ 掃引（28群すべてが教師）の step200 ただ1本。
# ★LGO の fold ckpt ではない。fold の重みは held-out 群を教師から外して学習したもので、
#   「全28群で測った曲線」を描く対象ではない。
CKPTS="${CKPTS:-outputs/checkpoints/stage2_lam0.003/stage2_step200.pt}"
OUT_DIR="${OUT_DIR:-${REPO}/outputs/generated}"
LOG="${WORK}/logs/dump_rates_${PBS_JOBID:-manual}.log"

echo "=== 生成プールの書き出し  job=${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "n=${N}  pool_seed=${POOL_SEED}"
echo "out:  ${OUT_DIR}"
echo "log:  ${LOG}"

# --- 事前チェック: 入力が全部あるか。1 本でも欠けていたら生成前に止める ---------
if [ ! -f "${SIF}" ]; then
    echo "ERROR: コンテナが無い: ${SIF}" >&2
    exit 1
fi
missing=0
for c in ${CKPTS}; do
    p="${c}"
    case "${p}" in /*) ;; *) p="${REPO}/${p}" ;; esac
    if [ ! -f "${p}" ]; then
        echo "ERROR: ckpt が無い: ${p}" >&2
        missing=$((missing + 1))
    fi
done
if [ ${missing} -ne 0 ]; then
    echo "       ${missing} 本足りないので 1 本も生成せずに止める。" >&2
    echo "       置き場所: ls ${REPO}/outputs/checkpoints/" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "$(dirname "${LOG}")"
{
    echo "=== job ${PBS_JOBID:-manual}  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ckpts : ${CKPTS}"
    echo "n=${N}  pool_seed=${POOL_SEED}"
    nvidia-smi
    echo "==="
} > "${LOG}" 2>&1

SECONDS=0
status=0
for c in ${CKPTS}; do
    p="${c}"
    case "${p}" in /*) ;; *) p="${REPO}/${p}" ;; esac
    base="$(basename "${p}" .pt)"
    out="${OUT_DIR}/${base}_rates.npz"
    echo "--- ${base} -> ${out}"
    run_gpu python src/models/DDPM_Aggregate_Simple/stage2_select.py \
        --dump-rates "${p}" --dump-out "${out}" \
        --n "${N}" --pool-seed "${POOL_SEED}" >> "${LOG}" 2>&1
    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "ERROR: ${base} の書き出しが失敗した（exit ${rc}）" >&2
        status=${rc}
    fi
done

echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m"
echo "=== 書き出したファイル ==="
ls -la "${OUT_DIR}"/*_rates.npz 2>/dev/null || echo "（1 つも書けていない）"
tail -30 "${LOG}"

echo "exit status: ${status}"
exit ${status}
