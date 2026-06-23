# NHTS2022から活動生成

NHTS2022の個人属性を条件として、15分刻み96スロットの1日の活動スケジュールを生成する。
ベースモデルは条件付きVAE (CVAE)。複数の比較モデルを **1モデル1フォルダ** で管理し、
モデル名で切り替えて学習・生成できる構成になっている。

## environment
依存関係は [uv](https://docs.astral.sh/uv/) で管理している。コマンドは `uv run` 経由で実行する。

```bash
uv sync          # 依存関係をインストール
uv run python -c "import torch; print(torch.__version__)"   # 動作確認
```

## preparation
NHTS2022をダウンロードする。
1. NHTSのWebサイトへアクセス → [NHTS](https://nhts.ornl.gov/)
2. サイトからCSVファイルをダウンロード。構成は以下のようになっている。
```bash
csv
|_Citation.pdf
|_hhv2pub.csv
|_ldtv2pub.csv
|_perv2pub.csv
|_tripv2pub.csv
|_vehv2pub.csv
```

## pre-processing
pathを指定して以下の2つのファイルを順番に実行する。
1. `src/common/preprocess/preprocess.py`
2. `src/common/preprocess/merge_weight.py`

実行すると、学習に使う `weighted_dataset.csv` が生成される。各行が1個人で、属性5列
(age, gender, num_member, role_household_type, worker_status) と活動スケジュール96列
(s0〜s95)、個人重み (WTPERFIN) を持つ。

## training
設定はYAMLファイルで管理する。`model_configs/Baseline3.yaml` を編集するか、
コピーして実験ごとの設定ファイルを作る。

```bash
# YAMLの設定で学習 (モデルはデフォルトの cvae)
uv run python scripts/train.py --config model_configs/Baseline3.yaml

# 設定をベースに一部だけ上書きして実験
uv run python scripts/train.py --config model_configs/Baseline3.yaml --beta 0.5 --out outputs/checkpoints/beta05.pt
uv run python scripts/train.py --config model_configs/Baseline3.yaml --device cuda --epochs 500

# モデルを切り替えて学習 (レジストリに登録した名前を指定)
uv run python scripts/train.py --model cvae --config model_configs/Baseline3.yaml
```

設定の優先度は **CLI引数 > YAML > デフォルト値**。学習に使うモデルは `--model`
(または YAML の `model:` キー、未指定なら `cvae`) で選ぶ。学習済みモデルは `--out`
で指定したパスに、学習曲線は `--history-csv` で指定したパスに保存される。
チェックポイントにはモデル種別 (`model_name`) も埋め込まれるため、生成時に自動判別される。

主なCVAE設定項目は以下の通り。

| 項目 | 説明 | デフォルト |
| --- | --- | --- |
| `latent_dim` | 潜在変数zの次元 | 32 |
| `hidden_dims` | 隠れ層のサイズ (層数も可変) | [512, 512] |
| `embedding_dim` | condition埋め込みの次元 | 8 |
| `beta` | KL項の重み (β-VAE) | 1.0 |
| `batch_size` | バッチサイズ | 500 |
| `epochs` | エポック数 | 300 |

学習データは個人重み (WTPERFIN) による動的サンプリング (`WeightedRandomSampler`、復元抽出)
で読み込まれ、母集団分布を反映する。

## generation
学習済みモデルに条件 (合成人口の属性) を与えてスケジュールを生成する。

```bash
uv run python scripts/generate.py \
    --model outputs/checkpoints/baseline.pt \
    --population data/raw/SyntheticPopulation/Matsumoto_with_work.csv \
    --out outputs/generated/matsumoto_with_work.csv
```

モデル種別はチェックポイントから自動判別される (`--model-name` で明示的に上書き可能)。
属性も一緒に保存したい場合は `--with-condition` を付ける。

スクリプトを介さずコードから使う場合:

```python
import pandas as pd
from src import get_model_bundle

bundle = get_model_bundle("cvae")
model, cfg = bundle.trainer_cls.load("outputs/checkpoints/baseline.pt")
generator = bundle.generator_cls(model, cfg)

population = pd.read_csv("synthetic_population.csv")
schedules = generator.generate(population)  # [N, 96]
```

入力する属性の列は `age, gender, num_member, role_household_type, worker_status` の順。
学習時の年齢範囲外 (0〜4歳や100歳など) が来ても、内部で年齢ビンにクランプされるため破綻しない。

## project structure
```bash
.
├── model_configs/          # 実験設定 (YAML)
├── scripts/
│   ├── train.py            # 学習スクリプト (--model でモデル切替)
│   └── generate.py         # 生成スクリプト
├── src/
│   ├── __init__.py         # 後方互換の re-export
│   ├── common/             # モデル非依存の共通基盤
│   │   ├── base.py         # 共通インターフェース (Config/Model/Trainer/Generator の規約)
│   │   ├── dataset.py      # ScheduleDataset (データ読み込み)
│   │   ├── synthetic_population.py  # 合成人口 → condition 変換
│   │   └── preprocess/     # NHTS 前処理
│   └── models/             # モデル群 (1モデル1フォルダ)
│       ├── __init__.py     # MODEL_REGISTRY / get_model_bundle()
│       ├── registry.py     # ModelBundle 定義
│       └── cvae/           # ベースモデル: 条件付き VAE
│           ├── __init__.py # bundle を公開
│           ├── config.py   # 設定 (CVAEConfig)
│           ├── condition.py # condition 埋め込み
│           ├── model.py    # Encoder / Decoder / CVAE
│           ├── loss.py     # 損失関数
│           ├── trainer.py  # 学習ループ・保存/読み込み
│           └── sampler.py  # 条件付き生成
├── outputs/                # 学習済みモデル (.pt) / 生成結果
└── logs/                   # 学習曲線 (CSV)
```

## 比較モデルの追加方法
新しいモデルは `src/models/<name>/` を1フォルダ作って追加する。共通基盤
(`ScheduleDataset` や合成人口変換) と学習/生成スクリプトはそのまま再利用できる。

1. **フォルダを作る**: `src/models/<name>/` に最低限 `config.py` / `model.py` /
   `trainer.py` / `sampler.py` を実装する。CVAE をひな型にすると早い。
   - 各クラスは [src/common/base.py](src/common/base.py) の規約 (Protocol) を満たす。
     - `Config`: frozen dataclass。`num_slots`, `num_activities`, `device`, `seed`,
       `condition` (`column_order` を持つ) を備える。
     - `Model` (`nn.Module`): `generate(cond) -> schedule_idx [B, 96]` を実装。
       学習用の `forward` の返り値はモデル/loss 固有でよい。
     - `Trainer`: `__init__(model, config)`, `fit(dataset, verbose)`, `save(path)`,
       および classmethod/staticmethod の `load(path) -> (model, config)`。
       `save` の dict に `"model_name": "<name>"` を含めるとチェックポイント単体で種別判別できる。
     - `Generator`: `__init__(model, config)`, `generate(condition) -> np.ndarray`。
   - 共通データセットを使う場合は `from ...common.dataset import ScheduleDataset` を import する。

2. **bundle を公開する**: `src/models/<name>/__init__.py` で4クラスを束ねる。
   ```python
   from ..registry import ModelBundle
   from .config import MyConfig
   from .model import MyModel
   from .trainer import MyTrainer
   from .sampler import MyGenerator

   bundle = ModelBundle(
       config_cls=MyConfig,
       model_cls=MyModel,
       trainer_cls=MyTrainer,
       generator_cls=MyGenerator,
   )
   ```

3. **レジストリに登録する**: [src/models/__init__.py](src/models/__init__.py) の
   `MODEL_REGISTRY` に1行追加する。
   ```python
   from .mymodel import bundle as mymodel_bundle
   MODEL_REGISTRY = {"cvae": cvae_bundle, "mymodel": mymodel_bundle}
   ```

4. **学習・生成する**:
   ```bash
   uv run python scripts/train.py --model mymodel --config model_configs/MyModel.yaml
   ```

> 補足: 現状 `scripts/train.py` の CLI 上書き対象 (`field_names`) は CVAE のフィールドを
> 前提にしている。別の config 構造を持つモデルを足す場合は、YAML での設定指定を基本にするか、
> bundle 側の config フィールドから動的に組み立てる拡張を行う。

## model (CVAE)
- 入力 (condition): 5つの個人属性をそれぞれ8次元に埋め込み、加算して統合
- 出力 (schedule): 各スロット10カテゴリの分類 (96スロット × 10カテゴリ)
- conditionはencoder/decoderの両方にconcatで注入

## 評価指標 / 評価観点
詳細は [NOTE.md](NOTE.md) を参照。
- 指標: JSD, EMD
- 観点例: 時刻区分別/活動別の行動者率、属性群ごとの分布、異常スケジュール生成率、
  生成の多様性、学習データの暗記率、各活動の開始/終了/継続時間分布、1人1日の活動数分布
