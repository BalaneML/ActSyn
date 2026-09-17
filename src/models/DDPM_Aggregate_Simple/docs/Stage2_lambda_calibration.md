# Stage 2: λ の基準値 X の実測（2026-09-17）

**対象**: `src/models/DDPM_Aggregate_Simple/stage2_finetune.py` の `--lam auto`。
掃引の設計根拠は [Stage2_design.md](./Stage2_design.md) §12 D、勾配の綱引きの実測表は
[Stage2_implementation.md](./Stage2_implementation.md) §6.2 にある。
処理の流れは [図③](./pipeline/Stage2.md) を参照する。

---

## 1. 結論

```
X = 0.2843
```

`--lam auto` が返す値である。

### 掃引に使う λ

X を 0.03 倍した 0.008529 のような値は論文で説明しにくいため、**1-3-10 の対数刻みの
キリのいい値**を使う。X = 0.2843 は 0.3 と 5% 以内で一致するので、上端を `0.3` と
呼んでも `--lam auto` の推定値を指したことになる。

| λ | λ/X | `λ‖g_atus‖/‖g_agg‖` | 集計成分のシェア |
|---|---|---|---|
| `0` | 0 | 0 | 100%（実測） |
| `0.003` | 0.011 | 0.32 | 約 95% |
| **`0.01`** | 0.035 | 1.06 | **約 72%** |
| **`0.03`** | 0.106 | 3.17 | **約 33%** |
| `0.1` | 0.352 | 10.6 | 約 10% |
| `0.3` | 1.055 | 31.7 | 約 3% |

`λ‖g_atus‖/‖g_agg‖ = 30.0 × (λ/X)` である（§6.2 の `X` 行が 30.0）。
**シェアの列は §6.2 の実測 5 点を対数内挿した概算であり、実測値ではない。**

§6.2 が転換点とした シェア 86.6%（= 0.03X = 0.008529）は、**`0.003` と `0.01` の間に
入る**。この 2 点で挟めるので、転移が起きる境界は捉えられる。

論文での記述は次の形になる。

> λ を 0.003 から 0.3 まで 1-3-10 の対数刻みで振った。
> 上端 0.3 は `--lam auto` の推定値 0.284 に一致する。

## 2. 測定の条件

ジョブ 1316736（`DBG-G` キュー）。`elapsed 0h 3m`、`exit status: 0`。

```
steps=5 d_sub=7 n=256 K=1 eps=inf loss=sq chunk=1792 (1パス) teacher_groups=28/28
stage1=ddpm_simple_pretrain_common12_weekday_20260819.pt
guidance=1.25  lr cond=1e-4 emb=2e-5 conv=1e-5
device=cuda (NVIDIA A100-SXM4-40GB, VRAM 40960 MiB)
VRAM 予算 K×chunk <= 6733  要求=1792  (B=1792)
```

`steps` 以外は本番ジョブ `jobs/train_ddpm_simple_stage2.sh` の既定と同一である。

### なぜ 5 更新で足りるか

`LAM_WARMUP_STEPS = 5`（`stage2_finetune.py:123`）であり、λ は最初の 5 更新の
`|L_agg|` と `L_atus` それぞれの中央値の比として決まり、そこで固定される。
6 更新目以降は λ に影響しない。

### なぜ DBG キューで測った値が本番と一致するか

乱数源が 2 つとも固定されているためである。

- `torch.manual_seed(seed)` — `stage2_finetune.py:551`
- `np.random.default_rng(seed)` — `stage2_finetune.py:579`
- `--seed` の既定は 42 — `stage2_finetune.py:741`

`d_sub` / `n` / `K` / `eps` / `loss` を本番と揃えれば、最初の 5 更新は同じ乱数列を
辿る。したがって中央値から決まる λ も同じ値になる。

**この測り方は掃引の前段として再利用できる。** `SQUID-S`（`SG1S`）の待ちが
数日に達するときでも、`DBG`（10 分枠）は空いていることが多い。

```bash
STEPS=5 jobs/submit.sh -q DBG -l elapstim_req=00:10:00 -v STEPS \
        jobs/train_ddpm_simple_stage2.sh
```

`qsub` のコマンドライン引数がジョブスクリプト内の `#PBS` ディレクティブを上書きする。

## 3. warmup の推移

| 更新 | 暫定 λ |
|---|---|
| 1/5 | 0.288 |
| 2/5 | 0.2861 |
| 3/5 | 0.2843 |
| 4/5 | 0.2832 |
| **確定** | **0.2843** |

確定時の `|L_agg|` の範囲は 0.002542〜0.003571 である。振れ幅は 1.4 倍にとどまり、
中央値で決める設計が効いている（1 更新だけで決めると split-batch 推定量の振れが
そのまま全ステップの重みになる）。

## 4. 1 更新目の診断値

```
step 1/5  L_agg=+0.002661  L_atus=0.009239  val=0.009435  rate_mae=0.02718
          OTHER_X=0.0129  |g_agg|=2.6e-05/1.3e-04/4.8e-04  floor=0.333  43.4s
```

- `floor=0.333` — 下側 clamp の飽和率。1.0 に張り付いていないので代理勾配は潰れていない。
- `|g_agg|` — cond / emb / conv の順。3 群とも非ゼロで、`straight_through` と `clamp` を
  抜けて θ に届いている。この 2 つだけが空回りを検出できる量である（`L_agg` も
  `rate_mae` も clamp より上流なので、θ が動かなくても正常値を出す）。
- `val=0.009435` と `L_atus=0.009239` — リハーサル項は過学習していない。

## 5. 所要時間の実測

**1 更新 = 43.4 秒**（見積もりは 39 秒）。

| steps | 所要 | 備考 |
|---|---|---|
| 5 | 3 分 | この測定（初期化込み） |
| 300 | **3.6 時間** | 本番 |

`jobs/train_ddpm_simple_stage2.sh` の `elapstim_req=06:00:00` は
「1 更新あたりの実測が無い初回に限った措置」である。実測が出たので、以降は
**4 時間**（マージン 11%）で投入する。

```bash
jobs/submit.sh -l elapstim_req=04:00:00 jobs/train_ddpm_simple_stage2.sh
```

## 6. 掃引の並列投入

学習ジョブの保存先は λ ごとに分かれる（`jobs/train_ddpm_simple_stage2.sh`）。

```sh
CKPT_DIR="${CKPT_DIR:-${REPO}/outputs/checkpoints/stage2_lam${LAM}}"
```

`LAM=0.01` なら `outputs/checkpoints/stage2_lam0.01/` に保存される。事後選択の
出力も `CKPT_DIR` のベース名から導かれ、`data/processed/aggregates/` の下に
`stage2_lam0.01_selection.csv` として落ちる。

**これは 2026-09-17 に入れた修正である。** それ以前は学習側が
`outputs/checkpoints/stage2` 固定で、事後選択側だけが `CKPT_DIR` を受け付ける
非対称な状態だった。固定のままだと、複数の λ を同時に投入したときに同じ
ディレクトリへ書き合い、起動時の自動退避も互いに踏み合って結果が混ざる。

投入は次の形になる。

```bash
for L in 0 0.003 0.01 0.03 0.1 0.3; do
    LAM=$L jobs/submit.sh -l elapstim_req=04:00:00 -v LAM \
        jobs/train_ddpm_simple_stage2.sh
done
```

## 7. 記録

- wandb（オフライン）: `offline-run-20260917_160337-n9qwkbi7`
  同期手順は `WANDB_DIR` の下がもう 1 階層深い点に注意する
  （`$WORK/wandb/wandb/offline-run-*`）。
- ジョブログ: `$WORK/logs/simple_stage2_0:1316736.sqd.log`
- この測定で生成された `stage2_step5.pt` は破棄した（5 更新の重みに用途が無いため）。
- **`λ=auto`・300 更新のジョブ 1316163 は `qdel` した。** `SG1S` の予定開始が
  2 日後（`sstat` 表示 2026-09-19 15:53）で、かつ `elapstim_req` が実測前の
  6 時間のままだったためである。λ=X は掃引の端点として最終的に 1 本必要だが、
  「集計成分 3.4%、`L_agg` は横ばいか微増」と結果が予測できているので優先度は最低である。
  転移が起きるかは `0.003`〜`0.01` で決まるため、そちらを先に回す。
