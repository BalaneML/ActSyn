#!/bin/bash
#PBS -q SQUID-H
#PBS -l elapstim_req=04:00:00
#PBS -l cpunum_job=76
#PBS -l gpunum_job=8
#PBS -N dt_stage2_lgo
#PBS -r n
#PBS -m e
#
# LGO 7 fold の事後チェックポイント選択を、1 ノードの 8 GPU に並べて同時に流す。
# 設計: src/models/DDPM_Aggregate_Simple/docs/Stage2_design.md §8.4 / §9.2
#
# ★なぜ jobs/eval_stage2_select.sh を 7 回投げないのか。
#   SQUID-H は**ノード単位課金**である。1 GPU しか使わないジョブを 7 本投げると
#   1 ノード分 × 7 を払うことになる。7 fold を 1 ノードの別々の GPU に載せれば
#   支払いは 1 ノード分で済み、しかも壁時計時間は 1 fold ぶんに縮む。
#   実測 2h13m/fold として、7 本直列なら 15.5 時間、この形なら約 2.2 時間。
#
#   | 形 | ノード時間 | 消費係数 | ポイント |
#   |---|---|---|---|
#   | SQUID-S に 7 本   | 2.22 × 0.5 × 7 = 7.8 | 1.3762 | 10.7 |
#   | SQUID-H に 7 本   | 2.22 × 1.0 × 7 = 15.5 | 2.2934 | 35.6 |
#   | SQUID-H に 1 本（これ） | 2.22 × 1.0 × 1 = 2.2 | 2.2934 |  5.1 |
#
# ★★安いが速いとは限らない。待ち時間で選ぶなら、まず SQUID-S を試すこと。
#
#   2026-09-20 に両方を実際に投入して予定開始を比べた結果:
#
#       SQUID-H  1 本（8 GPU / 1 ノード）  -> 9/25 07:11
#       SQUID-S  7 本（1 GPU / 0.5 ノード）-> 9/20 15:17, 9/23 10:50, 9/25 04:35 …
#
#   **優先度キューの方が遅かった。** 効くのは待ち行列の本数ではなく、確保させる
#   資源の粒度である。SG1H は 1 ノードが丸ごと空くのを待つが、SG1S の 1 GPU
#   ジョブは部分的に空いたノードの隙間へ入れる。GPU 区画が埋まっている間は、
#   小さく刻んだ方が先に流れる（SG1H は 14 待ち・8 実行、SG1S は 509 待ち・8 実行
#   という「混雑度」だけを見ると逆の結論になるので注意）。
#
#   同じ日に確かめた、効かなかったもの:
#     - elapstim_req を 5h -> 3h に縮める。予定開始は 1 秒も動かない。
#
#   したがってこのスクリプトが有利なのは次のときだけである:
#     (1) ポイントを最優先で節約したい（S の 10.7 pt に対し 5.1 pt）
#     (2) GPU 区画が空いていて 1 ノードがすぐ取れる
#     (3) 7 fold の結果を**同時に**揃えたい（S では最初と最後が数日離れる）
#
# データフロー:
#
# ```mermaid
# flowchart TD
#     CKPT["outputs/checkpoints/<br/>stage2_lam${LAM}_fold${i}/<br/>stage2_step*.pt（12 世代）"]
#     subgraph NODE["SQUID-H 1 ノード（GPU 8 基 / CPU 76 コア）"]
#         G0["CUDA_VISIBLE_DEVICES=0<br/>fold 0"]
#         G1["CUDA_VISIBLE_DEVICES=1<br/>fold 1"]
#         GD["…"]
#         G6["CUDA_VISIBLE_DEVICES=6<br/>fold 6"]
#     end
#     CKPT --> G0 & G1 & GD & G6
#     G0 --> C0["stage2_lam${LAM}_fold0_selection.csv"]
#     G1 --> C1["stage2_lam${LAM}_fold1_selection.csv"]
#     G6 --> C6["stage2_lam${LAM}_fold6_selection.csv"]
#     C0 & C1 & C6 --> COL["stage2_lgo.py --collect<br/>（eval_kind=held-out 行のみを束ねる）"]
# ```
#
# ★GPU 7 基しか使わない（fold は 7 つ）。8 基目は空くが、ノード単位課金なので
#   支払いは変わらない。fold を増やす設計ではないので埋める意味がない。
#
# 使い方:
#   jobs/submit.sh jobs/eval_stage2_lgo_parallel.sh
#   LAM=0.003 N=1000 jobs/submit.sh -v LAM,N jobs/eval_stage2_lgo_parallel.sh
#
#   起動だけ DBG キュー（10 分枠）で確かめる:
#   SMOKE=1 jobs/submit.sh -q DBG -l elapstim_req=00:10:00 -v SMOKE \
#       jobs/eval_stage2_lgo_parallel.sh
#
# 出力:
#   data/processed/aggregates/stage2_lam${LAM}_fold${i}_selection.csv  （7 本）
#   ${WORK}/logs/stage2_lgo_${PBS_JOBID}_fold${i}.log                  （7 本）

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

LAM="${LAM:-0.003}"
N="${N:-1000}"
POOL_SEED="${POOL_SEED:-12345}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6}"
SMOKE="${SMOKE:-0}"
STAGE1="${STAGE1:-${REPO}/outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_20260819.pt}"
TEACHER="${REPO}/data/processed/stula/timeband_weekday.csv"
DATA="${REPO}/data/processed/atus2024/atus2024_stula_common12_dataset.csv"
LOGDIR="${WORK}/logs"
JOBID="${PBS_JOBID:-manual}"

N_FOLD="$(printf '%s\n' ${FOLDS} | grep -c .)"

echo "=== LGO 並列事後選択  job=${JOBID}  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "dirty : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "lam=${LAM}  n=${N}  pool_seed=${POOL_SEED}  folds='${FOLDS}' (${N_FOLD} 本)  smoke=${SMOKE}"

# --- 事前チェック 1: GPU がフォルド数だけ見えているか ----------------------
# ★ここで落とさないと、同じ GPU に 2 つ載って両方が遅くなるか OOM する。
N_GPU="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
echo "見えている GPU: ${N_GPU} 基"
if [ "${N_GPU}" -lt "${N_FOLD}" ]; then
    echo "ERROR: GPU が ${N_GPU} 基しかない。fold ${N_FOLD} 本を別々の GPU に載せられない。" >&2
    echo "       gpunum_job=8 のノード単位キュー（SQUID-H / DBG）で流すこと。" >&2
    exit 1
fi

# --- 事前チェック 2: 入力 -------------------------------------------------
for f in "${SIF}" "${TEACHER}" "${DATA}" "${STAGE1}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: not found: ${f}" >&2
        exit 1
    fi
done
for i in ${FOLDS}; do
    d="${REPO}/outputs/checkpoints/stage2_lam${LAM}_fold${i}"
    if [ ! -d "${d}" ]; then
        echo "ERROR: ckpt ディレクトリが無い: ${d}" >&2
        exit 1
    fi
    n_ckpt="$(find "${d}" -maxdepth 1 -name 'stage2_step*.pt' | wc -l | tr -d ' ')"
    if [ "${n_ckpt}" -eq 0 ]; then
        echo "ERROR: stage2_step*.pt が 1 つも無い: ${d}" >&2
        exit 1
    fi
    echo "  fold${i}: ckpt=${n_ckpt} 本"
done

# --- 事前チェック 3: 所要時間の見積り ---------------------------------------
# 155 秒 / 7168 本（§4.7）を基準にする。並列なので壁時計は 1 fold ぶん。
# ★CPU は 76 コアを ${N_FOLD} 本で分け合う。単独実行（38 コア）より CPU 律速の
#   区間が遅くなりうるので、枠は実測 2h13m に対して 4h を取ってある。
#   打ち切られると CSV が 1 行も書かれない（run は全 ckpt を回し終えてから書く）。
N_CKPT="$(find "${REPO}/outputs/checkpoints/stage2_lam${LAM}_fold$(printf '%s\n' ${FOLDS} | head -1)" \
    -maxdepth 1 -name 'stage2_step*.pt' | wc -l | tr -d ' ')"
N_EVAL=$(( N_CKPT + 1 ))
EST_SEC=$(( N_EVAL * 28 * N * 155 / 7168 ))
echo "所要見積り（1 fold ぶん = 壁時計）= $(( EST_SEC / 3600 ))h $(( (EST_SEC % 3600) / 60 ))m  (枠 4h)"

# --- CPU スレッドの配分 ----------------------------------------------------
# ★既定のまま流すと 1 プロセスが 76 コアぶんのスレッドを立て、7 本で 532 スレッドに
#   なって取り合いで遅くなる。fold ごとに等分する。
NCPU="$(nproc 2>/dev/null || echo 76)"
THREADS=$(( NCPU / N_FOLD ))
[ "${THREADS}" -lt 1 ] && THREADS=1
echo "CPU: ${NCPU} コア -> 1 fold あたり ${THREADS} スレッド"

mkdir -p "${LOGDIR}" "${REPO}/data/processed/aggregates"

# --- 起動 -----------------------------------------------------------------
SECONDS=0
pids=()
launched=()
gpu=0
for i in ${FOLDS}; do
    CKPT_DIR="${REPO}/outputs/checkpoints/stage2_lam${LAM}_fold${i}"
    OUT_CSV="${REPO}/data/processed/aggregates/stage2_lam${LAM}_fold${i}_selection.csv"
    LOG="${LOGDIR}/stage2_lgo_${JOBID}_fold${i}.log"

    # ★環境変数は**部分シェルの中だけ**で設定する。ループ変数 gpu を export して
    #   次の周回で書き換える形にすると、読み手に「子プロセスがどちらの値を見たか」が
    #   判断できない（bash は & の時点で fork するので実際は安全だが、それは
    #   読んで分かることではない）。部分シェルなら値が外へ漏れず、対応も一目で済む。
    # ★Singularity はホストの環境変数を引き継ぐが、確実を期して
    #   SINGULARITYENV_ 付きでも渡す。ここを取り違えると 7 本すべてが
    #   GPU 0 に載り、静かに 7 倍遅くなる（OOM で落ちればまだ気づける）。
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${gpu}"
        export OMP_NUM_THREADS="${THREADS}"
        export SINGULARITYENV_OMP_NUM_THREADS="${THREADS}"
        export MKL_NUM_THREADS="${THREADS}"
        export SINGULARITYENV_MKL_NUM_THREADS="${THREADS}"

        if [ "${SMOKE}" = "1" ]; then
            # 起動と GPU の割り当てだけを確かめる。生成は行わない。
            # ★device 名だけでは足りない。8 基とも同型なので名前は全部同じに出る。
            #   GPU が本当に別々かは UUID でしか確かめられない。
            run_gpu python -c "
import os, torch
ok = torch.cuda.is_available()
uuid = 'none'
if ok:
    try:
        uuid = str(torch.cuda.get_device_properties(0).uuid)[:8]
    except AttributeError:      # torch のバージョンによっては uuid を持たない
        uuid = 'unavailable'
print('fold=${i}',
      'CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)'),
      'cuda=' + str(ok),
      'device=' + (torch.cuda.get_device_name(0) if ok else 'none'),
      'uuid=' + uuid,
      'threads=' + str(torch.get_num_threads()), flush=True)
"
        else
            run_gpu python src/models/DDPM_Aggregate_Simple/stage2_select.py \
                --ckpt-dir "${CKPT_DIR}" --n "${N}" --out-csv "${OUT_CSV}" \
                --pool-seed "${POOL_SEED}" --stage1-ckpt "${STAGE1}"
        fi
    ) > "${LOG}" 2>&1 &
    pids+=("$!")
    launched+=("${i}")
    echo "  fold${i} -> GPU ${gpu}  pid=$!  log=${LOG}"
    gpu=$(( gpu + 1 ))
done

echo
echo "=== ${#pids[@]} 本を起動した。全部の終了を待つ ==="

n_fail=0
for k in "${!pids[@]}"; do
    i="${launched[$k]}"
    if wait "${pids[$k]}"; then
        echo "fold${i}: exit 0"
    else
        st=$?
        echo "fold${i}: ★exit ${st}  -> ${LOGDIR}/stage2_lgo_${JOBID}_fold${i}.log の末尾を見ること" >&2
        n_fail=$(( n_fail + 1 ))
    fi
done

echo
echo "elapsed: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m  (次回の elapstim_req 見直しに使う)"

# --- 結果の確認 -----------------------------------------------------------
if [ "${SMOKE}" = "1" ]; then
    echo "=== SMOKE: 各 fold が見た GPU ==="
    cat "${LOGDIR}"/stage2_lgo_"${JOBID}"_fold*.log
    echo "★fold ごとに別の device が出ていること。同じものが並んだら GPU 固定が効いていない。"
else
    echo "=== 書き出した CSV ==="
    n_csv=0
    for i in ${FOLDS}; do
        f="${REPO}/data/processed/aggregates/stage2_lam${LAM}_fold${i}_selection.csv"
        if [ -f "${f}" ]; then
            printf "  fold%s  %s 行  %s\n" "${i}" "$(wc -l < "${f}" | tr -d ' ')" "${f}"
            n_csv=$(( n_csv + 1 ))
        else
            printf "  fold%s  ★CSV が無い\n" "${i}" >&2
        fi
    done
    echo "揃った CSV: ${n_csv} / ${N_FOLD} 本"
    if [ "${n_csv}" -eq "${N_FOLD}" ]; then
        echo
        echo "次（フロントエンドで実行）:"
        echo "  singularity run ${SIF} python src/models/DDPM_Aggregate_Simple/stage2_lgo.py \\"
        echo "      --collect ${REPO}/data/processed/aggregates/stage2_lam${LAM}_fold*_selection.csv"
        echo "  singularity run ${SIF} python src/models/DDPM_Aggregate_Simple/stage2_lgo.py --floors"
        echo "★fold ごとに教師の床が 2.78 倍違う。共通の床で読まないこと（§9.4）。"
    fi
fi

echo "失敗した fold: ${n_fail} 本"
echo "exit status: ${n_fail}"
exit ${n_fail}
