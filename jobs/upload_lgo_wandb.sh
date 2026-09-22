#!/bin/bash
# LGO 7 fold のオフライン run を wandb へ同期し、成果物を artifact として添付する。
#
# ★これは PBS ジョブではない。**フロントエンドで実行する**。
#   計算ノードは外部ネットワークへ出られないので、wandb への送信はここからしか行えない
#   （jobs/submit_lgo.sh と同じ層のスクリプトである）。
#
# なぜ必要か:
#   LGO の学習は GPU 25 時間かかり、成果物は SQUID の work 領域にしか無い。
#   work はバックアップ領域ではない。学習曲線は既にオフライン run に入っているので、
#   同じ run へ重み・ログ・評価 CSV を付けて 1 か所にまとめる。
#
# ★オフライン run の場所は **$WORK/wandb/wandb/** であり、1 階層深い。
#   `WANDB_DIR=$WORK/wandb` を渡すと wandb がその下にもう 1 つ `wandb/` を掘るためで、
#   docs/SQUID_guide.md L284 の `wandb sync $WORK/wandb/offline-run-*` は
#   何にもマッチせず、黙って成功したように見える。
#
# ★7 本だけを同期する。work には 23 本のオフライン run があり、そのうち 5 本は
#   同期済みである。`offline-run-*` を丸ごと渡すと他の実験の run まで project に
#   流れ込む。
#
# 使い方:
#   bash jobs/upload_lgo_wandb.sh --dry-run     # 何をどこへ送るか確認（同期もしない）
#   bash jobs/upload_lgo_wandb.sh               # 同期 -> 添付
#
# 送信されるもの（--dry-run で実物の一覧が出る）:
#   offline run 7 本（学習曲線）             約 24 MB
#   stage2_step200.pt × 7                    約 149 MB
#   ジョブログ 15 本                          約 1 MB
#   評価 CSV 8 本                             約 5 MB

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source jobs/_common.sh

DRY=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY=1
fi

# --- ★jobs/_common.sh の既定を 2 つ上書きする -------------------------------
#
# (1) WANDB_MODE。_common.sh:49 は `offline` を輸出する。計算ノードが外部へ出られない
#     ため学習ジョブにはそれが正しいが、**このスクリプトはフロントエンドで動く送信側**
#     である。offline のままだと wandb.init(resume="must") がオンラインの run を
#     開かずに**新しいオフライン run を掘る**。エラーにならず、成功したように見えて
#     何も送られない（2026-09-22 に実機で踏んだ）。
export WANDB_MODE=online
#
# (2) artifact の作業場所。wandb は送信前にファイルを**複製**して退避する。既定の
#     退避先は $HOME の下で、SQUID の home は 10GB 固定・拡張不可である。21MB の
#     ckpt を 7 本コピーした時点で `OSError: [Errno 122] Disk quota exceeded` になる。
#     XDG_CACHE_HOME を work に向けるだけでは足りない（wandb は専用の変数を見る）。
export WANDB_CACHE_DIR="${WORK}/.cache/wandb"
export WANDB_DATA_DIR="${WORK}/.cache/wandb/staging"
export WANDB_ARTIFACT_DIR="${WORK}/.cache/wandb/artifacts"
export WANDB_CONFIG_DIR="${WORK}/.config/wandb"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}" \
         "${WANDB_CONFIG_DIR}"
# ★コンテナへも確実に渡す。Singularity はホストの環境を引き継ぐが、
#   ここを取り違えると (1) と同じく「静かに何もしない」に戻る。
export SINGULARITYENV_WANDB_MODE="${WANDB_MODE}"
export SINGULARITYENV_WANDB_CACHE_DIR="${WANDB_CACHE_DIR}"
export SINGULARITYENV_WANDB_DATA_DIR="${WANDB_DATA_DIR}"
export SINGULARITYENV_WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR}"
export SINGULARITYENV_WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}"

# fold と wandb run id の対応。★python 側の FOLDS と同じ値でなければならない。
# ここは同期する run を選ぶためだけに使う（添付の対応付けは python 側が持つ）。
RUN_IDS="me2bbley 37vkv5gu xkqwnbn9 ro0kgfnz woe7z8a3 tlnl6va8 sml7viek"

# --- 事前チェック 1: 資格情報 ---------------------------------------------
# ★ここで落とさないと、wandb sync が対話プロンプトを出して固まる。
if ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: ${HOME}/.netrc に wandb の資格情報が無い。" >&2
    echo "       先に 'wandb login' を済ませること。" >&2
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "ERROR: コンテナが無い: ${SIF}" >&2
    exit 1
fi

# --- 事前チェック 1b: 退避先が本当に work にあるか ---------------------------
# ★home の残量ではなく**退避先そのもの**を確かめる。最初に踏んだのは
#   「home が満杯」ではなく「退避先が home だった」ことによる失敗である
#   （2026-09-22、`OSError: [Errno 122]`）。上の WANDB_DATA_DIR を向け直した後は、
#   home が上限に張り付いていても 21MB の ckpt を退避できることを実機で確認した。
#   したがって home の使用量は**止める理由にはならない**。警告に留める。
if ! touch "${WANDB_DATA_DIR}/.writetest" 2>/dev/null; then
    echo "ERROR: 退避先に書けない: ${WANDB_DATA_DIR}" >&2
    echo "       ここが work の下でないと、21MB の ckpt 7 本を複製する時点で" >&2
    echo "       home の 10GB 上限に当たる（拡張不可）。" >&2
    exit 1
fi
rm -f "${WANDB_DATA_DIR}/.writetest"
echo "退避先: ${WANDB_DATA_DIR}（work 配下・書き込み可）"

HOME_USED_GB=$(( $(du -sm "${HOME}" 2>/dev/null | cut -f1 || echo 0) / 1024 ))
if [ "${HOME_USED_GB}" -ge 10 ]; then
    echo "WARNING: home が上限 10GB に達している（約 ${HOME_USED_GB} GB）。" >&2
    echo "         退避先は work なので送信自体は通るが、いずれ別の場所で詰まる。" >&2
    du -sh "${HOME}"/* "${HOME}"/.[a-z]* 2>/dev/null | sort -rh | head -3 >&2
fi

# --- 事前チェック 2: オフライン run が 7 本そろっているか ---------------------
OFFLINE_ROOT="${WORK}/wandb/wandb"
DIRS=""
n_found=0
for rid in ${RUN_IDS}; do
    d="$(ls -d "${OFFLINE_ROOT}"/offline-run-*-"${rid}" 2>/dev/null | head -1)"
    if [ -z "${d}" ]; then
        echo "ERROR: run ${rid} のオフラインディレクトリが無い: ${OFFLINE_ROOT}" >&2
        exit 1
    fi
    DIRS="${DIRS} ${d}"
    n_found=$((n_found + 1))
done
if [ "${n_found}" -ne 7 ]; then
    echo "ERROR: オフライン run が 7 本そろっていない（${n_found} 本）" >&2
    exit 1
fi
echo "=== 同期するオフライン run（${n_found} 本）==="
for d in ${DIRS}; do
    printf '  %-72s %s\n' "$(basename "${d}")" "$(du -sh "${d}" | cut -f1)"
done

if [ "${DRY}" -eq 1 ]; then
    echo
    echo "=== --dry-run: 添付の内訳 ==="
    WORK="${WORK}" run_cpu python \
        src/models/DDPM_Aggregate_Simple/stage2_wandb_artifacts.py --dry-run
    echo
    echo "★--dry-run のため wandb へは何も送っていない。"
    exit 0
fi

# --- 同期 -----------------------------------------------------------------
# ★フロントエンドに wandb CLI が無いのでコンテナ経由で呼ぶ。
echo
echo "=== wandb sync ==="
# shellcheck disable=SC2086
run_cpu wandb sync ${DIRS}
status=$?
if [ ${status} -ne 0 ]; then
    echo "ERROR: wandb sync が失敗した（exit ${status}）。artifact の添付は行わない。" >&2
    echo "       添付は resume='must' で run がオンラインに在ることを要求する。" >&2
    exit 1
fi

# --- 添付 -----------------------------------------------------------------
echo
echo "=== artifact の添付 ==="
WORK="${WORK}" run_cpu python \
    src/models/DDPM_Aggregate_Simple/stage2_wandb_artifacts.py
status=$?

echo
if [ ${status} -eq 0 ]; then
    echo "完了。wandb の project 'domain-transfer-ddpm-agg' で確認すること。"
else
    echo "WARNING: 添付に失敗した fold がある（exit ${status}）" >&2
fi
exit ${status}
