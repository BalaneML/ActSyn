#!/bin/bash
# 全ジョブスクリプトから source される共通設定。
# 単体で実行するものではない。

set -u

# --- グループ名 -----------------------------------------------------------
# 利用者ごとに違うのでリポジトリには置かず、jobs/local.env（git 管理外）から読む。
# 初回のみ次を実行すること:
#     cp jobs/local.env.example jobs/local.env   # そのあと自分の値を書く
# 投入時は jobs/submit.sh が同じファイルを読み、qsub --group= へ渡す。
_ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/local.env"
if [ ! -f "${_ENV_FILE}" ]; then
    echo "ERROR: ${_ENV_FILE} が無い。cp jobs/local.env.example jobs/local.env" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${_ENV_FILE}"
GROUP="${GROUP:?GROUP が jobs/local.env に無い}"

# work 領域。home は 10GB 上限なので、コード・データ・出力・キャッシュはすべてここに置く。
WORK="/sqfs/work/${GROUP}/${USER}"
REPO="${WORK}/DomainTransfer_trial"
SIF="${WORK}/containers/domaintransfer.sif"

# --- Singularity ---------------------------------------------------------
# work 領域とジョブ投入ディレクトリをコンテナ内へ見せる。
#
# /sqfs/work はシンボリックリンクで、実体は /sqfs2/cmc/0/work にある。
# readlink -f で解決しないとバインドに失敗するが、解決先をそのままの位置へ
# マウントすると、コンテナ内の /sqfs/work/... は読み取り専用のままになる。
# 下の MPLCONFIGDIR / XDG_CACHE_HOME / WANDB_DIR はいずれも /sqfs/work/... 表記
# なので、それらが黙って /tmp へフォールバックする（＝ジョブ終了と同時に消える）。
# `解決先:元のパス` の形でマウントし直し、ホストと同じパスで書けるようにする。
export SINGULARITY_BIND="$(readlink -f "${WORK}"):${WORK},${PBS_O_WORKDIR:-${PWD}}"

# pull/build 時の一時ファイルが home を埋めないようにする。
export SINGULARITY_CACHEDIR="${WORK}/.cache/singularity"

# --- home への書き込みを避ける -------------------------------------------
# Singularity は既定で $HOME をバインドするため、これらを指定しないと
# matplotlib のフォントキャッシュや wandb の作業ファイルが home 10GB を圧迫する。
export MPLCONFIGDIR="${WORK}/.cache/matplotlib"
export XDG_CACHE_HOME="${WORK}/.cache"

# --- wandb ---------------------------------------------------------------
# 計算ノードは外部ネットワークへ接続できない。オフラインで記録し、
# ジョブ完了後にフロントエンドで `wandb sync` して送信する。
export WANDB_MODE=offline
export WANDB_DIR="${WORK}/wandb"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${WORK}/logs"

# --- 実行ヘルパ -----------------------------------------------------------
# GPU ジョブ用。--nv でホストの GPU デバイスとドライバをコンテナへ渡す。
# exec ではなく run を使う: NGC イメージのエントリポイントが CUDA の
# forward-compatibility を設定するため、これを飛ばすと古いドライバで初期化に失敗する。
run_gpu() {
    singularity run --nv "${SIF}" "$@"
}

# CPU のみのジョブ用。
run_cpu() {
    singularity run "${SIF}" "$@"
}
