# 計画書: 再帰型（GRU）＋交差エントロピーの Stage 1（GRU_Aggregate）

作成 2026-09-27。前提の診断は [Stage1_rare_activity_diagnosis.md](../../DDPM_Aggregate_Simple/docs/Stage1_rare_activity_diagnosis.md)。

## 1. 背景と問い

- **診断の結論：** DDPM（`clock_tf96`）では、少ない活動（買い物・介護・育児・移動・スポーツ・ボランティア）の総量が学習の途中で大きく動いた（途中の ckpt での総量の比の種内 sd 0.20〜0.75）。総量は逆過程の SNR ≈ 1 の帯で決まり、val の ε-MSE はその間ほとんど変わらない。損失が総量を縛っていない。
- **狙い：** 1 日の 96 スロットを 04:00 から順に 1 つずつ生成する再帰モデルに替え、各スロットの活動の確率を交差エントロピーで学習する。出力に「スロット × 活動」ごとのバイアスを持たせ、学習が止まった点で**予測確率の加重平均が学習データの行動者率と一致する**ようにする（§3.2）。
- **範囲：** Stage 1 だけ。Stage 2 への接続は、この比較の結果を見てから決める。

| # | 問い | 見るもの |
|---|---|---|
| Q1 | 少ない活動の総量は、種によって揺れなくなるか | 総量の比の種間 sd（種 5 本） |
| Q2 | 自分の出力で履歴を作って生成したとき、総量はずれるか | 実データの履歴で条件付けた総量と、自分で生成した総量の差 |
| Q3 | 1 日の系列として崩れないか | 断片化・04:00 の境目・遷移の分布 |
| Q4 | 時刻の形（昼の食事の山・仕事の谷）は DDPM と比べてどうか | 12 活動の時刻別行動者率 |

## 2. データ（DDPM と同じ。比較のため 1 行も変えない）

| 項目 | 値 | 取り出し方 |
|---|---|---|
| ファイル | `data/processed/atus2024/atus2024_stula_common12_dataset.csv` | `DDPM_Aggregate_Simple/model.py` の `load_data` |
| 範囲 | ATUS 2024 平日、3,736 人 | 同上（`DAY_FILTER = 'weekday'`） |
| 系列 | 04:00 起点の 15 分 × 96 スロット、共通 12 分類 | 同上 |
| 条件 | 性 × 年齢 7 区分 × 就業の 28 群（`COND_SPEC`） | 同上 |
| 分割 | train 3,363 / val 373 | `split_indices`（種 42 固定） |
| 重み | TUFINLWGT | **損失の重み**に使う（DDPM はミニバッチの抽出に使っている。§3.2） |

- データの読み込み・分割・条件の符号化は、`DDPM_Aggregate_Simple/model.py` から import する。写し書きしない（分割がずれると、比較や暗記チェックの参照集合が別物になる）。
- **評価は米国加重**：生成の群別の値を ATUS の TUFINLWGT の群構成で加重する（Stage 1 の目的は元分布の再現）。

## 3. 手法

### 3.1 構造

```mermaid
flowchart LR
    PREV["a_prev (B, 96)<br/>直前の活動（s=0 は BOS=12）"] --> EA["act_embed<br/>Embedding(13, H)"]
    PHI["phi (96, 96)<br/>time_features(ArchSpec(clock_kind='transformer'))"] --> ET["time_proj<br/>Linear(96, H)"]
    COND["cond_idx (B, 3)<br/>drop_mask (B,)"] --> EC["cond_embed<br/>Embedding × 3 → Linear(16, H)<br/>落とした行は cond_null"]
    EA --> SUM["x (B, 96, H) = 3 つの和"]
    ET --> SUM
    EC --> SUM
    SUM --> GRU["gru<br/>GRU(H, H, num_layers=2)"]
    GRU --> OUT["out_proj(h) + slot_bias<br/>logits (B, 96, 12)"]
    OUT --> SM["log_softmax → log p(a_s | a_<s, c, s)"]
```

| 部品 | 中身 | 理由 |
|---|---|---|
| `phi` | Transformer 型の時刻符号 96 次元（`clock_tf96` と同じ φ を import） | 生成するスロットの時刻を、そのステップだけの入力として明示する |
| `slot_bias` | 「スロット × 活動」ごとの自由なバイアス（96 × 12） | 総量を縛る仕組みの本体（§3.2）。φ の線形写像では各スロットの任意の値を作れない（独立な成分は 97 のうち 34） |
| `cond_null` | 条件を落とした行に使う学習可能なベクトル（確率 `P_UNCOND = 0.1`、DDPM と同じ） | 生成時の CFG のため |
| 隠れ層の幅 H | 384（パラメータ 182.9 万） | 比べる DDPM（`clock_tf96`、187.7 万）と同程度にそろえる |

`slot_bias` は、学習データの米国加重の行動者率の対数で初期化する（学習の初期を速めるため。§3.2 の一致の条件には影響しない）。

### 3.2 学習と「総量を縛る」仕組み

損失は、TUFINLWGT で重み付けした交差エントロピー（teacher forcing。各ステップの入力は実データの直前の活動）:

`L = − Σ_i w_i Σ_s log p_θ(a_{i,s} | a_{i,<s}, c_i, s) / (96 · Σ_i w_i)`

`slot_bias[s, c]` による微分は `Σ_i w_i (p_θ[i, s, c] − 1[a_{i,s} = c]) / (96 · Σ_i w_i)` なので、学習が止まった点では

`Σ_i w_i p_θ[i, s, c] = Σ_i w_i 1[a_{i,s} = c]`（全スロット s・全活動 c）

が成り立つ。つまり**実データの履歴で条件付けたとき**、予測確率の米国加重平均が学習データの行動者率と一致する。

- 一致するのは全体（米国加重）の行動者率で、群ごとではない。
- 自分の出力で履歴を作って生成したときの総量は、この一致の対象外。そのずれを Q2 で測る。
- DDPM は重みでミニバッチを引いているが、ここでは全員を毎 epoch 使い、損失を重みで掛ける。どちらも学習の目標は同じ分布で、損失を掛ける方が少ない活動の行動者を毎 epoch 必ず見る（ボランティアの行動者は学習分割に 165 人、重み付き抽出では実効 102 人）。

| 項目 | 値 |
|---|---|
| 最適化 | AdamW、学習率 1e-3、weight decay 0、勾配のノルムを 1.0 で切る |
| バッチ | 256 |
| 早期終了 | val の重み付き交差エントロピー、patience 30、最大 300 epoch |
| 途中の ckpt | 5 epoch ごとに保存（Q1 の軌跡用） |

### 3.3 生成

- s = 0..95 の順に softmax から 1 つずつ引く（温度 1.0 固定）。
- CFG は logits にかける: `logits = l_u + g·(l_c − l_u)`。**既定 g = 1.0**（CFG なし）。今回の診断で、CFG が少ない活動の総量をずらすと分かったため。g = 1.25 は比較用に 1 回だけ回す。
- 生成プール: 28 群 × 256 本、乱数の種 12345（DDPM の共通乱数プールと同じ）。形式は `write_pool_csv`（DDPM と同じ CSV）。

保存先:

| もの | パス |
|---|---|
| 最良の ckpt | `outputs/checkpoints/gru_aggregate{接尾辞}.pt` |
| 途中の ckpt | `outputs/checkpoints/gru_aggregate{接尾辞}_ep{epoch:04d}.pt` |
| 生成プール | `outputs/generated/gru_aggregate_samples{接尾辞}.csv` |

接尾辞は種 42 なら空、それ以外は `_s{seed}`（DDPM と同じ規則）。

## 4. 評価（米国加重、DDPM と同じ条件で並べる）

### 4.1 比べるもの

| 名前 | 中身 | 種 |
|---|---|---|
| `gru` | 本計画のモデル（g = 1.0） | 42〜46 |
| `ddpm_tf96` | `clock_tf96`（学習時のプール、g = 1.25） | 42〜46（既存） |
| `ddpm_noclock` | 時刻符号なしの DDPM（ガードレールの外側の基準） | 42〜44（既存） |

### 4.2 指標

| 問い | 指標 | 出所（再利用） |
|---|---|---|
| Q1 | 5 活動の総量の比（米国加重）と、その種間 sd | `stage1_rare_diagnosis.profile_table` + `group_weights("atus")` |
| Q1 | 途中の ckpt ごとの総量の比（64 人/群の小プール、DDPM の H4 と同じ大きさ） | 同上 |
| Q2 | 実データの履歴で条件付けた総量 − 自分で生成した総量 | 新設 `teacher_forced_rates` |
| Q3 | switch_emd / bigram_jsd / single_slot / wrap_closure、暗記（DCR の差） | `stage2_select.guardrails`（既に米国加重）/ `memorization_report` |
| Q4 | 12 活動の時刻別行動者率、‖実 − 生成‖²（活動ごと）、昼の食事と仕事の値 | 新設 `us_weighted_slot_rates` + `stage2_curves.pool_to_slot_rates` |
| 床 | ATUS の回答者の復元抽出での比の sd × 2（米国加重で作り直す） | `stage1_rare_diagnosis.bootstrap_floor` |

### 4.3 判定（事前に固定）

| # | 条件 | 合格の基準 |
|---|---|---|
| C1 | 総量の安定（Q1） | 5 活動のうち 4 以上で、`gru` の総量の比の種間 sd が `ddpm_tf96` の半分以下 |
| C2 | 総量の一致（Q1） | 5 活動のうち 4 以上で、`gru` の総量の比の種平均が床の内（\|比 − 1\| ≤ 床） |
| C3 | 系列の妥当さ（Q3） | switch_emd・bigram_jsd・single_slot の差・wrap_closure の差が、`ddpm_noclock` の種の最大値を超えない。暗記の判定が 0 本 |

- **採用の候補にする：** C1・C2・C3 をすべて満たす。
- **C3 で落ちた場合：** 悪化の程度を `ddpm_noclock` と並べて示し、次の手（1 ステップで 1 時間分を出す／粗 → 細の順で生成する）の判断を仰ぐ。
- Q2・Q4 は判定に使わず、数値と図で報告する。

**事前の予測**（結果の前に書いておく）:

- C1 は満たす（`slot_bias` の定常条件による）。
- Q2 のずれは、短いエピソードが多い移動で最大になる。
- C3 のうち、いちばん危ういのは switch_emd。

## 5. 実装

| ファイル | 中身 |
|---|---|
| `src/models/GRU_Aggregate/model.py`（新設） | 定数、`GRUScheduler`（nn.Module）、`train`、`sample`、`group_pool`、`teacher_forced_rates`、保存と読み込み、CLI（`--seed` `--epochs` `--save-every` `--guidance` `--no-pool` `--smoke`） |
| `src/models/GRU_Aggregate/test_model.py`（新設） | §7 の単体テスト |
| `src/eval/diagnostics/stage1_gru_compare.py`（新設） | §4 の評価。`us_weighted_slot_rates` を置く |
| `src/models/GRU_Aggregate/docs/`（新設） | この計画書、結果の報告書、図 |

import して再利用するもの（写し書きしない）:

- `DDPM_Aggregate_Simple/model.py`：`load_data` `split_indices` `cond_to_d` `cond_grid` `COND_SPEC` `P_UNCOND` `time_features` `ArchSpec` `write_pool_csv` `memorization_report` `NUM_SLOTS` `NUM_ACT`
- `src/eval/diagnostics/stage1_rare_diagnosis.py`：`profile_table` `group_weights` `bootstrap_floor` `pool_people`
- `DDPM_Aggregate_Simple/stage2_select.py`：`guardrails`
- `DDPM_Aggregate_Simple/stage2_curves.py`：`load_sample_pool` `pool_to_slot_rates` `ACT_NAMES`

データの流れ:

```mermaid
flowchart TD
    LD["load_data / split_indices<br/>cond_idx, sched, weight"] --> TR["train<br/>重み付き交差エントロピー・teacher forcing"]
    TR --> CK["gru_aggregate{接尾辞}.pt<br/>+ _ep{epoch:04d}.pt"]
    CK --> GP["group_pool(model, 256, guidance=1.0)<br/>pool (28, 256, 96)"]
    CK --> TF["teacher_forced_rates(model, sched, cond_idx)<br/>予測確率の米国加重平均 (12, 96)"]
    GP --> CSV["gru_aggregate_samples{接尾辞}.csv<br/>write_pool_csv"]
    CSV --> EV["stage1_gru_compare.py<br/>us_weighted_slot_rates / profile_table / guardrails"]
    TF --> EV
    DDPM["DDPM の既存プール<br/>clock_tf96 / noclock"] --> EV
    EV --> REP["表・図 → 判定 C1〜C3"]
```

## 6. 実行

- すべて手元（MPS / CPU）で回す。生成は 96 ステップなので SQUID は使わない見込み。1 本目で学習と生成の時間を測り、1 本 30 分を超えるなら SQUID の DBG に回す。
- 順序:
  1. 実装とテスト → `--smoke` で完走
  2. 種 42 を 1 本学習し、時間・学習曲線・生成の見た目を確認
  3. 種 43〜46
  4. g = 1.25 のプール（種 42 のみ）
  5. 評価と報告書
- 実装の単位ごとにコミットし、テストが通ってから `feat/stage1-clock-ablation` へ push する。

**止まる条件**（その時点で報告する）:

- 学習が発散する（val が 3 epoch 続けて悪化し、最良の 2 倍を超える）。
- 生成の見た目が明らかに崩れている（全員が同じ系列、切替が実データの 3 倍以上など）。
- C3 で落ちる。

## 7. 検証（`test_model.py`）

| # | 検証すること |
|---|---|
| a | 因果性：スロット s の logits が a_{≥s} に依存しない（未来の活動を書き換えても不変） |
| b | 損失が、手で書いた重み付き交差エントロピーと一致する |
| c | `slot_bias` の勾配が `Σ_i w_i (p − y) / (96 Σ w)` と一致する（総量を縛る仕組みの式を固定する） |
| d | 条件を落とした行は、条件なし（`cond_idx=None`）と一致する。g = 1.0 の CFG は条件付きの logits と一致する |
| e | 同じ種で生成が再現する。値域は [0, 12)。`write_pool_csv` → `load_sample_pool` で往復する |
| f | `teacher_forced_rates` が、予測確率の加重平均を手で計算したものと一致する |
| g | 保存した ckpt から同じ構造・同じ出力が戻る |

あわせて Pylance の警告をゼロにし、`--smoke` での完走を確認する。

## 8. この計画の外（結果を見て決める）

- Stage 2 への接続。各スロットの確率が微分できる形で出るので、argmax を通さずに集計の損失をかけられる見込み。
- 1 ステップで 1 時間分（4 スロット）を出す、または粗 → 細の順で生成する形への拡張（C3 で落ちた場合）。
- 休日の日誌を足す（平日に限定する方針の変更を伴う）。
