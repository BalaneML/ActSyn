# ActSyn — 公表集計だけで行う活動スケジュール生成のドメイン転移

**日本の個票を一切使わずに、日本の活動スケジュール個票を生成する。**

米国 ATUS 2024 の個票（1人1日 = 15分刻み96スロットの活動系列）で条件付き生成モデルを
学習し、日本の社会生活基本調査（社基調）の**公表集計表だけ**を教師にして日本側へ転移する。
出力は「性 × 年齢7区分 × 就業 = 28群」それぞれについて、日本の集計値に合致し、
かつ1日の系列として自然な個票スケジュールの集合。

```
ATUS個票 (N=3,736, 平日) ──[Stage 1]──▶ 条件付き生成モデル ──▶ 群別スケジュール
                                              ▲
社基調 公表集計表 ────────[Stage 2]───────────┘   (日本の個票は使わない)
```

| | 教師 | 何を決めるか |
|---|---|---|
| **Stage 1** | 米国 ATUS の**個票** | 「群 d らしい1日のスケジュール」の生成器 |
| **Stage 2** | 日本社基調の**公表集計** | 日本の集計に合わせる（重みの傾け、またはパラメータの微調整） |

Stage 2 には2つの経路がある。**指数傾け**（生成器を凍結し、日記1本ごとの重みだけを凸最適化で
決める）と、**パラメータ微調整**（打ち切り逆伝播つき生成で集計損失を直接下げる）である。
前者は `DDPM_Aggregate/japan_match_experiment.py`、後者は `DDPM_Aggregate_Simple/stage2_*.py`。

詳細な理論と実測結果は [DDPM_Aggregate 読解ガイド](src/models/DDPM_Aggregate/DDPM_Aggregate_guide.md)
（拡散モデル入門者向け、アルゴリズム擬似コード付き）を参照。

---

## 環境

依存関係は [uv](https://docs.astral.sh/uv/) で管理する。

```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

HPC（大阪大学 SQUID）で回す場合は Singularity コンテナを使う。手順は
[container/README.md](container/README.md)、ジョブスクリプトは [jobs/](jobs/) にある。

`data/` `outputs/` `logs/` `notebooks/` `docs/` は Git 管理外（[.gitignore](.gitignore)）。
生データは各配布元から取得し、下記の前処理で `data/processed/` を再生成する。

---

## データの入手と前処理

### 1. ATUS 2024（米国 American Time Use Survey）— Stage 1 の個票

[BLS のデータページ](https://www.bls.gov/tus/data/datafiles-2024.htm) から zip を取得し、
展開した `.dat` を `data/raw/opened/ATUS2024/` に置く。

| ファイル | 内容 |
|---|---|
| `atusact_2024.dat` | Activity file（1行 = 1エピソード） |
| `atusresp_2024.dat` | Respondent file（調査ウェイト・曜日・就業状態） |
| `atusrost_2024.dat` | Roster file（年齢・性別） |

```bash
uv run python src/common/preprocess/atus/preprocess.py
```

→ `data/processed/atus2024/atus2024_weighted_dataset.csv`
（ATUS Lexicon 第1層の17分類、04:00起点96スロット、`TUFINLWGT` 付き）

### 2. 社会生活基本調査 令和3年 調査票A（日本）— Stage 2 の教師

e-Stat（統計コード 00200533「時間帯編」）から Excel を取得し、
`data/raw/opened/STULA2021/<statInfId>.xlsx` に置く。使うのは第8-1表
（曜日・男女・ふだんの就業状態・行動の種類・年齢・時刻区分別行動者率）。

```bash
uv run python src/common/preprocess/stula/parse_timeband.py
```

→ `data/processed/stula/timeband_weekday.csv`（0:00起点を04:00起点へ回転済み、
欠損マーカー `-`（=0.0）と `…`/`X`（=非公表 NaN）を分離）

### 3. クロスウォーク（ATUS 17分類 ↔ 社基調 20分類 → 共通12分類）

```bash
uv run python src/common/preprocess/stula/crosswalk_atus_stula.py
```

→ `data/processed/atus2024/atus2024_stula_common12_dataset.csv`（Stage 1 の入力）
→ `data/processed/stula/crosswalk_atus_stula.csv` / `.md`（論文貼付用の対応表）

共通12分類は `SLEEP_PERSONAL, MEALS, WORK, SCHOOL, HOUSEWORK, CAREGIVING, SHOPPING,
TRAVEL, LEISURE_SOCIAL, SPORTS, VOLUNTEER, OTHER_X`。両側で意味的に対応が取れないコードは
`OTHER_X` に集約し、集計マッチと評価から除外する（生成自体はされる）。

### 4. NHTS 2022（旧系列）

初期の CVAE 系モデルが使うトリップ由来10分類のデータセット。
[NHTS](https://nhts.ornl.gov/) から CSV を取得したうえで、
`src/common/preprocess/nhts/preprocess.py` → `merge_weight.py` の順に実行すると
`data/processed/weighted_dataset.csv` が生成される（パスはスクリプト内で直接指定）。

---

## モデル

`src/models/<name>/` が1モデル1フォルダ。各 `model.py` の冒頭 docstring に
「どのモデルからの差分か」と設計判断の根拠（実測値つき）が書いてある。

| フォルダ | データ / 分類 | 位置づけ |
|---|---|---|
| [CVAE](src/models/CVAE/model.py) | NHTS 10分類 | 最初のベースライン。条件は5属性 one-hot |
| [CVAE_Embedding](src/models/CVAE_Embedding/model.py) | NHTS 10分類 | 上に Embedding 層と early stopping を追加 |
| [CVAE_Aggregate](src/models/CVAE_Aggregate/model.py) | 共通12分類・平日 | 集計マッチ転移の初版（AggCVAE）。Stage 2 は μ̂ への勾配降下 |
| [DDPM](src/models/DDPM/model.py) | ATUS 17分類・全曜日 | 条件付き DDPM の素の実装 |
| [**DDPM_Aggregate**](src/models/DDPM_Aggregate/model.py) | 共通12分類・平日・28群 | **本線（AggDDPM）**。1D-UNet + self-attention + CFG |
| [DDPM_Aggregate_Tang](src/models/DDPM_Aggregate_Tang/model.py) | 同上 | denoiser を Tang et al. 2025 (arXiv:2508.09164) 構成に置換 |
| [DDPM_Aggregate_DiT](src/models/DDPM_Aggregate_DiT/model.py) | 同上 | denoiser を DiT (Peebles & Xie 2023) に置換 |
| [DDPM_Aggregate_Simple](src/models/DDPM_Aggregate_Simple/model.py) | 同上 | AggDDPM から4点を削った簡素版。Stage 2 パラメータ微調整の土台 |

### なぜ CVAE から DDPM へ移ったか

CVAE 系は decoder を $p(x \mid z) = \prod_t p(x_t \mid z)$ とスロットごとに分解するため、
$z$ で説明しきれない残差が時間的に無相関なノイズとして現れ、個票が激しく断片化した。
ATUS 平日・共通12分類での1日の平均活動切替回数は **実データ 12.64 に対し CVAE 生成 48.1（3.8倍）**。
DDPM は毎ステップ全スロットを同時に見るので系列構造が保たれる（ancestral で 12.87 = 1.02倍）。

代償として、拡散の逆過程には素直に勾配が通らない。Stage 2 が指数傾け（凸最適化）と
打ち切り逆伝播（DRaFT-K）の2経路に分かれるのはこのためである。

### AggDDPM の設計要点

- **スケジュール表現**: 96スロット × 12活動の one-hot を $\{-1,+1\}$ に写して連続拡散に載せ、
  復号は argmax（Tang et al. 2025 の枠組み）
- **条件**: 性2 × 年齢7区分 × 就業2 = 28群。28通りの one-hot ではなく属性ごとに
  Embedding して連結する（最小の群は日記が17本しかないため、統計的強度を共有する）
- **classifier-free guidance**: 学習時に確率 0.1 で条件を落とし、生成時に $w = 1.25$ で誘導
  （$w = 1.5$ では WORK の時間シェアが 0.255 と実データ 0.168 から大きく外れる）
- **EMA**（decay 0.999）の重みを生成に使い、early stopping は検証 ε-MSE で行う

---

## 実行

### Stage 1（ATUS 個票での事前学習）

```bash
# 形状・整合チェックだけ（数秒。学習前に必ず通す）
uv run python -c "import sys; sys.path.insert(0,'src/models/DDPM_Aggregate'); import model; model.smoke_test()"

# 短時間の動作確認（学習5エポック。チェックポイントは保存しない）
uv run python src/models/DDPM_Aggregate/model.py --smoke

# 本番学習
uv run python src/models/DDPM_Aggregate/model.py            # UNet1D（本線）
uv run python src/models/DDPM_Aggregate_Tang/model.py       # Tang バックボーン
uv run python src/models/DDPM_Aggregate_DiT/model.py        # DiT バックボーン
uv run python src/models/DDPM_Aggregate_Simple/model.py     # 簡素版
```

共通フラグ: `--epochs N`（エポック数の上書き）、`--no-wandb`（wandb 無効化）、`--smoke`。
バックボーン固有のフラグは各 `model.py` の docstring を参照
（DiT は `--size` / `--patch`、Tang は `--width-scale`、Simple は `--kernel`）。

出力は `outputs/checkpoints/*.pt` と `outputs/generated/*_pretrain_samples.csv`。

### Stage 2-A: 指数傾け（生成器を凍結）

提案分布（凍結モデルの群別サンプルプール）に対し、公表2表を制約とする最大エントロピー傾け
$q_d(x) \propto p_d(x)\exp\langle\theta_d, \phi(x)\rangle$ を凸双対で解く。
自由変数は28群ぶんではなく「公表された18行ぶん」の乗数だけ。

```bash
uv run python src/models/DDPM_Aggregate/japan_match_experiment.py

# プールを再利用して傾けだけ回し直す（速い）+ ridge 掃引
uv run python src/models/DDPM_Aggregate/japan_match_experiment.py --reuse-pool --ridge-sweep
```

出力は `data/processed/aggregates/ddpm_japan_match_experiment.csv`（バリアント別の評価指標）と
`ddpm_japan_match_tilting.csv`（ESS / 残差 / KL の診断）。

> `ridge > 0` は省略可能な調整項ではない。公表2表には構造的な線形従属があり、
> `ridge = 0` では双対に平坦方向が残って乗数が発散する（実測で dev_rmse が 0.0208 → 0.2063 に破綻）。

### Stage 2-B: パラメータ微調整（DRaFT-K）

生成の末尾 K ステップだけ勾配を保持して集計損失を直接下げる。ATUS 実個票のリハーサル項
（重み λ）を併走させて、個票レベルの現実性を守る。早期終了は使わず固定ステップ予算で
回し切り、保存したチェックポイントを学習後に（教師適合, ガードレール）の2軸で選ぶ。

```bash
# 動作確認
uv run python src/models/DDPM_Aggregate_Simple/stage2_finetune.py --smoke

# 本番
uv run python src/models/DDPM_Aggregate_Simple/stage2_finetune.py \
    --steps 300 --d-sub 7 --n 256 --K 1 --eps inf --lam auto

# 事後チェックポイント選択
uv run python src/models/DDPM_Aggregate_Simple/stage2_select.py \
    --ckpt-dir outputs/checkpoints/stage2 --n 2000
```

主なフラグ:

| フラグ | 意味 |
|---|---|
| `--steps N` | 固定ステップ予算（早期終了なし） |
| `--d-sub M` / `--n N` | 1更新で使う群数 / 群あたり生成本数。`K × d_sub × n <= 6600` |
| `--K N` | 勾配を保持する末尾ステップ数（DRaFT-K） |
| `--chunk N` | 1度に勾配を保持する個票数。0 で自動。B 未満なら2パス勾配蓄積へ切替 |
| `--eps V` | 損失の重み床。`inf` = 素の MSE（主A）、`0.01` = χ²（主B） |
| `--lam V` | リハーサル重み。`auto` で初回の L_agg / L_atus 比から決める |
| `--holdout-groups d1,d2,...` | leave-groups-out。損失から外す群（生成と評価は常に全28群） |
| `--resume` | 最新チェックポイントから再開 |

`--holdout-groups` を使うと held-out 側の評価が非循環になる。全28群を教師にした条件での
教師適合は「集計にどこまで合わせられるかの上限」であって、汎化の主張ではない。

---

## 評価

`src/eval/` はモデル非依存の評価モジュール。活動分類数と活動名は引数で受け取るので、
共通12分類・ATUS 17分類・NHTS 10分類のどれにも同じ関数を使える。

| モジュール | 測るもの |
|---|---|
| [individual_metrics.py](src/eval/individual_metrics.py) | 断片化（切替回数・エピソード長）、継続時間表、bigram JSD、切替分布 EMD、多様性、暗記チェック、ホールドアウト分割、可視化ヘルパ |
| [feasibility.py](src/eval/feasibility.py) | 構造的にありえるか（移動が往復で閉じているか、睡眠が細切れでないか） |
| [clock_diagnostics.py](src/eval/clock_diagnostics.py) | 「モデルは時刻を知っているか」の5診断（カーブ鈍化・初回開始時刻・日跨ぎ・位相プローブ・巡回シフト等変性） |

いずれも**実データ自身の有限標本ゆらぎ（ノイズ床）を併記する**。床の外に出て初めて乖離と言う。

```bash
uv run python src/eval/clock_diagnostics.py                                    # 既定は DDPM_Aggregate
uv run python src/eval/clock_diagnostics.py --model-dir src/models/DDPM_Aggregate_Tang
```

### 教師と評価の分離（統計量ホールドアウト）

| | 使うもの |
|---|---|
| **学習（Stage 2 の教師）** | 公表周辺2表（性×年齢 14行 + 性×就業 4行） |
| **評価（ホールドアウト）** | 公表**群別**行動者率（28群クロス表）— Stage 2 で一切使わない |

比較条件は4つ。主張が通るのは「`tilted` が `zero-shot` と `baseline` の**両方**を下回り、
かつ `shuffled` でその改善が消える」場合に限る。

| バリアント | 内容 | 役割 |
|---|---|---|
| `zero-shot` | 傾けなし（一様重み） | 米国モデルの構造持ち込みだけでどこまで合うか |
| `tilted` | 指数傾け | 本命 |
| `shuffled` | 群対応をシャッフルしたプールに傾ける | 負の対照 |
| `baseline` | 全群 = 日本の人口平均 | モデル不要の床 |

指標は `rate_mae` / `rate_rmse`（総合誤差）と `dev_mae` / `dev_rmse`（群偏差成分のみ =
**条件付け能力の主指標**）。全群に人口平均を返すだけでも `rate_*` はある程度下がるため、
群差を当てられているかは偏差成分でしか測れない。

---

## テスト

各テストは `main()` に素の assert と診断 print を並べた自己完結スクリプト。

```bash
uv run python src/eval/test_individual_metrics.py
uv run python src/eval/test_feasibility.py
uv run python src/eval/test_clock_diagnostics.py
uv run python src/models/DDPM_Aggregate/test_tilting.py             # 凸双対の導出検証
uv run python src/models/DDPM_Aggregate/test_memorization_report.py # 暗記診断のサイズ交絡
uv run python src/models/DDPM_Aggregate_Tang/test_backbone.py
uv run python src/models/DDPM_Aggregate_DiT/test_backbone.py
uv run python src/models/DDPM_Aggregate_Simple/test_backbone.py
uv run python src/models/DDPM_Aggregate_Simple/test_stage2.py
```

---

## リポジトリ構成

```
.
├── src/
│   ├── common/preprocess/
│   │   ├── atus/preprocess.py              ATUS 2024 → 96スロット × 17分類
│   │   ├── stula/parse_timeband.py         社基調 時間帯編 Excel → CSV（04:00起点へ回転）
│   │   ├── stula/crosswalk_atus_stula.py   ATUS ↔ 社基調 → 共通12分類
│   │   └── nhts/                           NHTS 2022（旧系列）
│   ├── models/
│   │   ├── CVAE/ CVAE_Embedding/           初期ベースライン（NHTS 10分類）
│   │   ├── CVAE_Aggregate/                 集計マッチ転移の初版
│   │   ├── DDPM/                           条件付き DDPM（ATUS 17分類）
│   │   ├── DDPM_Aggregate/                 ★本線。Stage1 + 指数傾け Stage2 + 読解ガイド
│   │   ├── DDPM_Aggregate_Tang/            Tang et al. 2025 バックボーン
│   │   ├── DDPM_Aggregate_DiT/             DiT バックボーン
│   │   └── DDPM_Aggregate_Simple/          簡素版 + Stage2 パラメータ微調整一式
│   └── eval/                               個票指標 / 実現可能性 / 時計診断（モデル非依存）
├── jobs/                                   SQUID (PBS) ジョブスクリプト
├── container/                              Singularity イメージ定義とビルド手順
├── data/                                   生データ・前処理結果（Git 管理外）
└── outputs/                                チェックポイント・生成結果（Git 管理外）
```

### 新しいバックボーンを足すとき

`DDPM_Aggregate_Tang` / `DDPM_Aggregate_DiT` は、データ整形・拡散過程・学習ループ・
プール生成・サニティチェックを `DDPM_Aggregate/model.py` からそのまま使い、
denoiser だけを差し替えている。同一の分割・同一の手順で学習されるので、
差が構造だけに帰着する。新しいバックボーンも同じ形にするのが最短である。

`DDPM_Aggregate_Simple` だけは例外で、`DDPM_Aggregate` を import しない完全に自己完結した
コピーである。共通部分を写しているため、**片方を直しても他方には伝播しない**。

`src/eval/clock_diagnostics.py` の `--model-dir` で診断対象を差し替える条件は、
`load_pretrained` / `Diffusion` / `cond_grid` / `features` / `in_channels` の契約を満たすこと。

---

## 参考文献

- Min Tang, Peng Lu, Qing Feng (2025). *Generating Feasible and Diverse Synthetic Populations
  Using Diffusion Models*. arXiv:2508.09164
- William Peebles, Saining Xie (2023). *Scalable Diffusion Models with Transformers* (DiT).
  ICCV 2023, arXiv:2212.09748
- Jonathan Ho, Ajay Jain, Pieter Abbeel (2020). *Denoising Diffusion Probabilistic Models*.
  NeurIPS 2020
