#!/bin/bash
# ジョブ投入ラッパ。qsub の代わりにこれを使う。
#
# なぜ必要か:
#   グループ名 (--group) とメールアドレス (-M) は利用者ごとに違うのでリポジトリに
#   書けない。かといって jobs/_common.sh に置くこともできない。_common.sh が
#   source されるのはジョブが計算ノードで動き始めた後であり、--group はそれより前、
#   qsub を実行する時点で決まっていなければならないためである。
#   そこで投入側であるここが jobs/local.env（git 管理外）を読み、qsub の
#   コマンドライン引数として渡す。
#
# 使い方（qsub をそのまま置き換える）:
#   jobs/submit.sh jobs/smoke.sh
#   EPOCHS=300 jobs/submit.sh -v EPOCHS jobs/train_ddpm_simple.sh
#   for k in 1 3 5 7; do KERNEL=$k jobs/submit.sh -v KERNEL jobs/train_ddpm_simple_kernel.sh; done
#
# qsub のその他のオプションはすべてそのまま透過する。

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${HERE}/local.env"

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: ${ENV_FILE} が無い。テンプレートから作ること:" >&2
    echo "    cp jobs/local.env.example jobs/local.env" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"

GROUP="${GROUP:?GROUP が jobs/local.env に無い}"
MAIL="${MAIL:?MAIL が jobs/local.env に無い}"

# テンプレートをコピーしただけの状態を弾く。放置すると --group=<グループ名> が
# そのまま qsub に渡り、エラーの原因が分かりにくくなる。
case "${GROUP}${MAIL}" in
    *"<"*) echo "ERROR: ${ENV_FILE} の GROUP / MAIL が未設定のまま" >&2; exit 1 ;;
esac

exec qsub --group="${GROUP}" -M "${MAIL}" "$@"
