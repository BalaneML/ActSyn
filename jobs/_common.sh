#!/bin/bash
# 全ジョブスクリプトから source される共通設定。
# 単体で実行するものではない。
#
# ★ 利用開始時に GROUP を自分のグループ名に書き換える（フロントエンドで `groups` で確認できる）。
#   各ジョブスクリプト冒頭の `#PBS --group=` も同じ値に揃える必要がある。
#   まとめて置換する場合:
#       sed -i 's/<グループ名>/実際のグループ名/' jobs/*.sh

set -u

GROUP="<グループ名>"

# work 領域。home は 10GB 上限なので、コード・データ・出力・キャッシュはすべてここに置く。
WORK="/sqfs/work/${GROUP}/${USER}"
REPO="${WORK}/DomainTransfer_trial"
SIF="${WORK}/containers/domaintransfer.sif"

# --- Singularity ---------------------------------------------------------
# work 領域とジョブ投入ディレクトリをコンテナ内へ見せる。
# readlink -f でシンボリックリンクを解決しておかないとバインドに失敗する。
export SINGULARITY_BIND="$(readlink -f "${WORK}"),${PBS_O_WORKDIR:-${PWD}}"

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
