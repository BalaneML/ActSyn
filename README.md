# NHTS2022から活動生成

NHTS2022の個人属性を条件として、15分刻み96スロットの1日の活動スケジュールを生成する条件付きVAE (CVAE)。

## preparation
NHTS2022をダウンロードする
1. NHTSのWebサイトへアクセス  
[NHTS](https://nhts.ornl.gov/)
2. サイトからCSVファイルをダウンロード  
ダウンロードが完了したファイルの構成は以下のようになっている  
```bash
csv
|_Catation.pdf
|_hhv2pub.csv
|_ldtv2pub.csv
|_perv2pub.csv
|_tripv2pub.csv
|_vehv2pub.csv
```

## pre-processing
pathを指定して以下の2つのファイルを順番に実行する
1. ``src/preprocess/preprocess.py``
1. ``src/preprocess/merge_weight.py``

実行すると、学習に使う ``weighted_dataset.csv`` が生成される。各行が1個人で、属性5列 (age, gender, num_member, role_household_type, worker_status) と活動スケジュール96列 (s0〜s95)、個人重み (WTPERFIN) を持つ。

<!-- ## training
設定はYAMLファイルで管理する。``configs/baseline.yaml`` を編集するか、コピーして実験ごとの設定ファイルを作る。

```bash
# YAMLの設定で学習
python train.py --config configs/baseline.yaml

# 設定をベースに一部だけ上書きして実験
python train.py --config configs/baseline.yaml --beta 0.5 --out checkpoints/beta05.pt
python train.py --config configs/baseline.yaml --device cuda --epochs 500
```

設定の優先度は CLI引数 > YAML > デフォルト値。学習済みモデルは ``--out`` で指定したパスに、学習曲線は ``--history-csv`` で指定したパスに保存される。

主な設定項目は以下の通り。

| 項目 | 説明 | デフォルト |
| --- | --- | --- |
| ``latent_dim`` | 潜在変数zの次元 | 32 |
| ``hidden_dims`` | 隠れ層のサイズ (層数も可変) | [512, 512] |
| ``embedding_dim`` | condition埋め込みの次元 | 8 |
| ``beta`` | KL項の重み (β-VAE) | 1.0 |
| ``batch_size`` | バッチサイズ | 500 |
| ``epochs`` | エポック数 | 300 |

学習データは個人重み (WTPERFIN) による動的サンプリング (``WeightedRandomSampler``、復元抽出) で読み込まれ、母集団分布を反映する。

## generation
学習済みモデルに条件 (合成人口の属性) を与えてスケジュールを生成する。

```python
import pandas as pd
from src import Trainer, ScheduleGenerator

model, cfg = Trainer.load("checkpoints/baseline.pt")
generator = ScheduleGenerator(model, cfg)

population = pd.read_csv("synthetic_population.csv")
schedules = generator.generate(population)  # [N, 96]
```

入力する属性の列は ``age, gender, num_member, role_household_type, worker_status`` の順。学習時の年齢範囲外 (0〜4歳や100歳など) が来ても、内部で年齢ビンにクランプされるため破綻しない。

## project structure
```bash
.
├── configs/            # 実験設定 (YAML)
├── src/                # ライブラリ本体
│   ├── config.py       # 全設定 (CVAEConfig)
│   ├── condition.py    # condition埋め込み
│   ├── dataset.py      # データ読み込み
│   ├── model.py        # Encoder / Decoder / CVAE
│   ├── loss.py         # 損失関数
│   ├── trainer.py      # 学習ループ・保存/読み込み
│   ├── sampler.py      # 条件付き生成
│   └── preprocess/     # 前処理
├── train.py            # 学習スクリプト
├── checkpoints/        # 学習済みモデル (.pt)
└── logs/               # 学習曲線 (CSV)
```

## model
- 入力 (condition): 5つの個人属性をそれぞれ8次元に埋め込み、加算して統合
- 出力 (schedule): 各スロット10カテゴリの分類 (96スロット × 10カテゴリ)
- conditionはencoder/decoderの両方にconcatで注入 -->