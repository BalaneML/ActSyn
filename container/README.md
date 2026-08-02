# SQUID コンテナ実験環境

SQUID（大阪大学 D3センター）で本リポジトリの DDPM / CVAE を実行するための
Singularity コンテナ環境。ジョブスクリプトは [`jobs/`](../jobs/) にある。

SQUID 全般の使い方（ログイン、ファイルシステム、キュー、ポイント）は
`docs/squid_guide.md` を参照。ここではコンテナ固有の手順のみを扱う。

---

## 構成

| ファイル | 役割 |
|---|---|
| `container/domaintransfer.def` | イメージ定義。ベースは NGC PyTorch 25.02 |
| `container/requirements-container.txt` | コンテナへ追加する依存（バージョンは `uv.lock` に一致） |
| `jobs/_common.sh` | 全ジョブ共通の環境変数と実行ヘルパ |
| `jobs/smoke.sh` | DBG キュー（10分）での動作確認（両バックボーン） |
| `jobs/train_ddpm.sh` | AggDDPM Stage1 の学習（SQUID-S / GPU 1基 / 24時間） |
| `jobs/train_ddpm_tang.sh` | AggDDPM Stage1 を Tang バックボーンで学習（SQUID-S / GPU 1基 / 8時間） |
| `jobs/sample_japan_match.sh` | サンプリング + マッチング実験（SQUID-S / GPU 1基 / 2時間） |

### 設計上の判断

- **`pyproject.toml` と `uv.lock` は変更しない。**
  `src/` 配下は `sys.path.insert` とリポジトリルート相対パスで完結しており、
  `pip install -e .` を必要としない。したがって `requires-python = ">=3.14"` は
  コンテナ内（Python 3.12）でも問題にならない。
- **torch と numpy はコンテナ同梱版を使う。**
  NGC イメージの torch は A100 向けにビルド済みで、CUDA の forward-compatibility
  ライブラリを同梱する。pip で上書きするとこの利点が失われる。
- **その他の依存は `uv.lock` と同一バージョンに固定する。**
  ローカル環境と数値結果が食い違わないようにするため。

---

## 1. ベースイメージのタグが実機で使えるか確認する

`domaintransfer.def` の `From: nvcr.io/nvidia/pytorch:25.02-py3` は
CUDA 12.8 を含む。NVIDIA の要件はドライバ 570 以降だが、データセンタ GPU
（SQUID の A100 が該当）では R470 系 470.57 以降、R525 系 525.85 以降、
R535 系 535.86 以降でも forward-compatibility により動作する。

HPC フロントエンドから会話型ジョブで実機のドライバ版を確認する。

```bash
ssh <利用者番号>@squidhpc.hpc.cmc.osaka-u.ac.jp
qlogin -q INTG -l elapstim_req=00:10:00,gpunum_job=1 --group=<グループ名>

module load BaseGPU
nvidia-smi          # "Driver Version:" の値を控える
logout
```

上記の対応表に載っていないドライバ版だった場合は、
[PyTorch コンテナのリリースノート](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
で要件を満たすタグを探し、`domaintransfer.def` の `From:` 行だけを差し替える。
```bash
Driver Version: 590.48.01
```
---

## 2. イメージをビルドする

**HPC フロントエンド（`squidhpc`）でビルドできる。ただし作業領域と出力先は
ノードのローカルディスク（`/tmp`）に置く。**

work 領域（`/sqfs/work`）は singularity がビルド用に作るマウント名前空間の中から
見えない。`SINGULARITY_TMPDIR` に指定すると rootfs の展開で、出力先に指定すると
SIF の書き出しで失敗する（→「つまずきやすい点」）。
名前空間の外で動くダウンロードキャッシュ（`SINGULARITY_CACHEDIR`）だけは work 領域でよい。

```bash
ssh <利用者番号>@squidhpc.hpc.cmc.osaka-u.ac.jp

# work 領域を使う前に必要
newgrp <グループ名>

export WORK=/sqfs/work/<グループ名>/<利用者番号>

# blob のダウンロードキャッシュを work 領域へ逃がす。
# 既定では home（10GB 上限）に置かれ、NGC イメージのサイズで容易に溢れる。
# SINGULARITY_TMPDIR は設定しない（既定の /tmp を使う）。
export SINGULARITY_CACHEDIR=$WORK/.cache/singularity
mkdir -p "$SINGULARITY_CACHEDIR" "$WORK/containers"

# リポジトリを配置（フロントエンドは外部ネットワークへ接続できる）
cd $WORK
git clone https://github.com/BalaneML/DomainTransfer_trial.git
cd DomainTransfer_trial

# /tmp の空きを確認する。rootfs の展開に約 25GB、SIF に約 11GB 必要。
df -h /tmp

# def ファイル内の %files がリポジトリルート相対のため、ここで実行する
singularity build -F -f /tmp/domaintransfer.sif container/domaintransfer.def

# 完成後に work へ移す。cp は名前空間の外で動くので work へ書ける。
cp /tmp/domaintransfer.sif $WORK/containers/ && rm /tmp/domaintransfer.sif
ls -lh $WORK/containers/domaintransfer.sif
```

- `-f` は fakeroot ビルド。SQUID では root 権限なしでビルドするためこれが必要。
- `-F` は既存 SIF の上書き。`-f` とは別のオプションなので両方指定する。
- ベースイメージの pull から通して数十分、キャッシュ済みなら約 5 分。完成する SIF は約 11GB。
- `%post` の最後で torch / numpy / pandas の import と
  バージョン表示を行うので、ここで失敗すれば依存の衝突がジョブ投入前に検出できる。
- `%post` の冒頭に出る `15:4: not a valid test operator` は、NGC の `/etc/shinit_v2` を
  `/bin/sh` で source したことによる警告。ビルドは正常に継続するので無視してよい。
- ビルド後、work 領域のキャッシュ（`$WORK/.cache/singularity`、約 20GB）は削除してよい。

> 実績: 2026-08-02、`squidhpc3` でビルド成功（ドライバ 590.48.01）。

---

## 3. コードとデータを配置する

コードは §2 の `git clone` で配置済み。データは Git 管理外なので別途転送する。

```bash
# ローカル端末側で実行（data/ は約 553MB）
tar czf data.tar.gz data/processed data/raw/opened
scp data.tar.gz <利用者番号>@squidhpc.hpc.cmc.osaka-u.ac.jp:/sqfs/work/<グループ名>/<利用者番号>/DomainTransfer_trial/

# SQUID 側で展開
cd /sqfs/work/<グループ名>/<利用者番号>/DomainTransfer_trial
tar xzf data.tar.gz && rm data.tar.gz
```

パスワードとワンタイムパスワードの入力が転送ごとに必要なため、
`tar` でまとめて 1 回で送る。

> **データガバナンス**: `data/raw/protected/` の学外計算機への持ち出し可否は、
> データ提供元の利用規約に従って判断すること。上記の `tar` コマンドは
> `data/raw/opened` のみを対象にしている。

---

## 4. ジョブスクリプトのプレースホルダを埋める

`jobs/*.sh` には 2 種類のプレースホルダがある。

```bash
cd /sqfs/work/<グループ名>/<利用者番号>/DomainTransfer_trial
sed -i 's/<グループ名>/実際のグループ名/'       jobs/*.sh
sed -i 's/<メールアドレス>/実際のアドレス/'      jobs/train_ddpm.sh jobs/train_ddpm_tang.sh
```

グループ名はフロントエンドで `groups` を実行して確認する。

---

## 5. 動作確認してから本番を投入する

```bash
qsub jobs/smoke.sh              # DBG キュー、10分。GPU 可視性と --smoke を確認
qstat                           # QUE / RUN を確認
# 完了後、投入ディレクトリに出力された dt_smoke.o<ID> を確認する

qsub jobs/train_ddpm.sh         # Stage1 本学習（UNet1D バックボーン）
qsub jobs/train_ddpm_tang.sh    # Stage1 本学習（Tang バックボーン）
qsub jobs/sample_japan_match.sh # 学習完了後
```

実行中の進捗は `$WORK/logs/` 配下のログで追える
（ジョブスクリプト内で明示的にリダイレクトしているため）。

学習ログの送信はジョブ完了後にフロントエンドで行う。計算ノードは
外部ネットワークへ接続できないため、ジョブ中は `WANDB_MODE=offline` で記録される。

```bash
wandb sync $WORK/wandb/offline-run-*
```

---

## つまずきやすい点

| 症状 | 原因と対処 |
|---|---|
| コンテナ内で `torch.cuda.is_available()` が False | `--nv` が付いていない。`jobs/_common.sh` の `run_gpu` を使う |
| CUDA の初期化でエラー | `singularity exec` を使っている。`run` に変えて NGC のエントリポイント（CUDA 互換レイヤの設定）を通す |
| ローカルと結果が食い違う | ホストの `~/.local` のパッケージが混入している。`domaintransfer.def` の `PYTHONNOUSERSITE=1` が効いているか確認する |
| ビルドが `packer failed to pack: ... mkdir rootfs: no such file or directory` で失敗 | `SINGULARITY_TMPDIR` が work 領域を指している。ビルドの名前空間から `/sqfs/work` は見えない。`unset SINGULARITY_TMPDIR` して既定の `/tmp` を使う |
| `%post` は成功するのに SIF の書き出しが `permission denied` で失敗 | 出力先が work 領域。`/tmp` に出力してから `cp` で work へ移す |
| ビルドが容量不足で失敗 | `SINGULARITY_CACHEDIR` が home を指している。または `/tmp` の空きが 40GB 未満（`df -h /tmp`） |
| work 領域が `permission denied` で読み書きできない | `newgrp <グループ名>` を実行していない |
| ジョブが書き込みエラーで落ちる | 出力先が home（10GB 上限）。`jobs/_common.sh` が work 領域を指しているか確認する |
| ジョブ最後のコマンドが実行されない | ジョブスクリプト末尾に改行が無い |

---

## 参照元

- [コンテナの利用方法(SQUID)](https://www.hpc.cmc.osaka-u.ac.jp/system/manual/squid-use/singularity/)
- [基本的な利用方法（SQUID GPUノード）](https://www.hpc.cmc.osaka-u.ac.jp/system/manual/squid-use/gpu-use/)
- [PyTorch Release 25.02 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html)
- [PyTorch - NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
