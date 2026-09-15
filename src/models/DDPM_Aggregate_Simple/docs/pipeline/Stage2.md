# Stage 2 の処理の流れ（集計表だけでの微調整）

**対象**: `src/models/DDPM_Aggregate_Simple/stage2_*.py`。
この 2 枚は [Stage2_implementation.md](../Stage2_implementation.md) §3 にあったアスキー図を
置き換えたものである。設計上の根拠は [Stage2_design.md](../Stage2_design.md)、
実装判断の根拠表は [Stage2_implementation.md](../Stage2_implementation.md) §3 に残してある。

**関連**: [Stage 1 の処理の流れ](./Stage1.md)

図中の識別子はすべてコード中の実名である。モジュール別名は `stage2_finetune.py` の
`_load()` が付けるものと同じ（`sm`=`model`, `st`=`stage2_targets`, `sl`=`stage2_loss`,
`ck`=`stage2_checkpoint`）。

---

## 図③: 1 更新の構造（`stage2_finetune.run` :301）

```mermaid
flowchart TB
    subgraph INIT["初期化 :314-364"]
        direction TB
        I1["model = sm.load_pretrained(STAGE1_CKPT) :327<br/>ddpm_simple_pretrain_common12_weekday_20260819.pt<br/>1,759,124 params すべて学習対象 (凍結しない)"]
        I2["optimizer = build_optimizer(model) :140<br/>名前が COND_PATH_KEYS を含む → 群 cond, LR_COND=1e-4<br/>それ以外 → 群 conv, LR_CONV=1e-5<br/>AdamW(weight_decay=0.0)"]
        I3["tgt = st.load_stula_targets() :320<br/>data/processed/stula/timeband_weekday.csv<br/>group_rates_tbl (28,12,96) と pop (2,7,2)"]
        I4["q_all = sl.teacher_tensor(tgt, dev) (28,12,96)<br/>omega_all = sl.chi2_weights(q_all, eps)<br/>eps=inf なら全て1 (主A の素の MSE)、有限なら 1/(q+eps) を平均1に正規化 (主B の χ²)"]
        I5["teacher_mask = build_teacher_mask(holdout) :108<br/>(28,) bool。held-out 群だけ False<br/>teacher_groups = flatnonzero(teacher_mask)"]
        I6["chunk = resolve_chunk(K, d_sub, n, chunk) :176<br/>check_memory_budget :159 で K*chunk が MEMORY_BUDGET=6600 以下か検査"]
        I7["atus = atus_batches()<br/>sm.make_loaders の train_loader を無限に回す"]
        I8["grid = sm.cond_grid() (28,3)"]
    end

    subgraph STEP["1 更新 (step = 1..steps、早期終了なし)"]
        direction TB

        P0["d_pick = rng.choice(teacher_groups, d_sub) :369<br/>cond = grid[d_pick].repeat_interleave(n) → (d_sub*n, 3) 群優先<br/>q = q_all[d_pick] / omega = omega_all[d_pick]"]
        P1["optimizer.zero_grad(set_to_none=True) :378"]

        subgraph AGG["集計側 aggregate_step :194 (eval モードで微分。backward まで内部で済ませる)"]
            direction TB
            BR{"loss_kind が jsd、または chunk が total 以上か"}

            subgraph ONE["1パス _aggregate_step_one_pass :250"]
                direction TB
                O1["zs = _draw_tail_noise(total, K, dev) :236"]
                O2["x0 = diffusion.sample_differentiable(model, cond, K, zs) :672"]
                O3["y = sm.straight_through(x0, tau=1.0) :691<br/>前向きは one-hot、後ろ向きは softmax の微分"]
                O4["a_A, a_B = sl.group_rates_split(y, n) :100<br/>群ごとに前半 n/2 と後半 n/2 へ二分"]
                O5["l_agg = sl.agg_loss_from_rates(a_A, a_B, q, omega) :118<br/>((a_A - q)*(a_B - q)*omega).mean() — 群 d・活動 c・時刻 s の3軸とも平均"]
                O6["l_agg.backward()"]
                O1 --> O2 --> O3 --> O4 --> O5 --> O6
            end

            subgraph TWO["2パス _aggregate_step_two_pass :268 (gradient caching)"]
                direction TB
                T1["zs = _draw_tail_noise(total, K, dev)<br/>★1パス側でも同じ順で引く。chunk で軌跡が変わらないため"]
                T2["x_K = diffusion._sample_head(model, cond, K) :642<br/>t = 999 から K まで no_grad で進める<br/>★2パス間で唯一キャッシュするテンソル"]
                T3["no_grad で _sample_tail → straight_through → y<br/>a_A, a_B = sl.group_rates_split(y, n)<br/>l_agg = float(sl.agg_loss_from_rates(...))"]
                T4["g_A, g_B = sl.loss_grad(a_A, a_B, q, omega) :171<br/>g_A = omega*(a_B - q)/scale, scale = |D_sub|*12*96 (交差ペア)<br/>g_per = sl.per_sample_grad(g_A, g_B, n) :188 → (B,12,96)"]
                T5["for start in range(0, total, chunk):<br/>x0_c = _sample_tail(model, x_K[start:end], K, cond[start:end], zs_c) :655<br/>末尾 K ステップだけ再計算 (ここだけグラフを作る)"]
                T6["y_c = sm.straight_through(x0_c, tau)<br/>y_c.backward(gradient=g_per[start:end])<br/>キャッシュした上流勾配を VJP として注入<br/>start を chunk ずつ進めて T5 と T6 を繰り返す"]
                T1 --> T2 --> T3 --> T4 --> T5 --> T6
            end

            BR -->|"yes"| ONE
            BR -->|"no"| TWO
        end

        subgraph REH["リハーサル側 (train モード。Stage 1 と同じ損失)"]
            direction TB
            H1["model.train() :385<br/>b_cond, b_sched = next(atus) — ATUS 個票 256 本"]
            H2["l_atus = diffusion.loss(model, b_sched, b_cond) :387<br/>ε-MSE + CFG 条件dropout (Stage1 図② と同一の関数)"]
            H3["初回のみ lam = abs(l_agg) / l_atus :391"]
            H4["(lam * l_atus).backward() :395"]
            H1 --> H2 --> H3 --> H4
        end

        UPD["optimizer.step() :396"]
        LOG["ログ :399-409<br/>a_full = 0.5*(a_A + a_B)<br/>rate_mae, other_x_share, sl.g_diagnostics(g_A, q, n)"]
        SV{"step が save_every(25) の倍数、または最終 step か"}
        CK["ck.save_ckpt(ck.ckpt_path(ckpt_dir, step), model, optimizer, step, config) :417<br/>outputs/checkpoints/stage2/stage2_step(step).pt<br/>config に holdout を入れて事後選択へ渡す"]

        NEXT["次の step へ (P0 に戻る)"]

        P0 --> P1 --> AGG
        AGG --> REH
        REH --> UPD --> LOG --> SV
        SV -->|"yes"| CK
        SV -->|"no"| NEXT
        CK --> NEXT
    end

    INIT --> P0
    NEXT --> POST["steps に達したら終了<br/>学習後: stage2_select.py → 図④"]
```

**要点**

- **`backward()` を 2 回に分けて呼ぶ**ので、集計側とリハーサル側の計算グラフを同時に持たない。
  ピークメモリは和ではなく `max(集計側, リハーサル側)` である（設計書 §4.5）。
- **`chunk` が `total` 未満なら 2 パス勾配蓄積へ自動で切り替わる**（`resolve_chunk`）。
  `x_K` を 1 パス目で保存するので前段（`T-K` ステップ）は 1 回しか回らず、2 回回るのは末尾 K だけ。
  勾配は一括計算と厳密に一致する（近似ではない。`test_two_pass_matches_full_batch` が
  最大差 1e-10 台で固定している）。
- 素朴な勾配蓄積が使えないのは、損失がバッチ全体の非線形関数だからである。
  部分ごとに損失を計算して足すと `n=4/chunk=2` で正解 0.0025 に対し 0.1625 と 65 倍ずれる。
- 教師は**教師ネットワークではなく公表集計表**（社会生活基本調査 令和3年 第8-1表）である。
  `teacher_mask` が制約するのは**損失だけ**で、生成と評価は常に 28 群すべてを対象にする。
- 設計判断（12 チャネルのまま再正規化しない／split-batch 不偏推定／群は等重み／早期終了なし）
  の根拠は [Stage2_implementation.md](../Stage2_implementation.md) §3 の表を参照する。

---

## 図④: 事後チェックポイント選択（`stage2_select.run` :170）

```mermaid
flowchart TB
    START["ckpt_dir.glob('stage2_step*.pt') :171<br/>step の昇順に並べる"]
    REAL["cond_idx, sched_real, w_real, _ = sm.load_data() :179<br/>d_real = sm.cond_to_d(cond_idx) :180"]
    TGT["tgt = st.load_stula_targets() :178"]

    subgraph EV["evaluate_ckpt(path, ...) :134 — 1 チェックポイントぶん"]
        direction TB
        C1["step, config = ck.load_ckpt(path, model) :136<br/>config['holdout'] から teacher_mask (28,) を復元"]
        C2["pool = sm.group_pool(model, n=2000) :142<br/>(28, 2000, 96)"]
        C3["gen = pool.reshape(-1, 96)<br/>gen_d = repeat(arange(28), 2000)<br/>rates = sm.pool_to_rates(pool) → (28, 12*96) act-major"]
        C1 --> C2 --> C3
    end

    subgraph AX1["軸1: 教師適合"]
        direction TB
        X1["mask_c を mask_12act / mask_11act の2通り<br/>sel を teacher_mask (in-teacher) / その否定 (held-out) の2通り"]
        X2["st.eval_against(rates[sel], sub_tgt, mask_c) :163<br/>n_cells, rate_mae, rate_mse, rate_rmse, max_abs_err, dev_mae, dev_rmse<br/>および活動別の mae_活動名 / rmse_活動名 / rel_活動名"]
        X1 --> X2
    end

    subgraph AX2["軸2: ガードレール guardrails() :98"]
        direction TB
        Y0["w_gen = im.group_reweight(gen_d, w_real, d_real, 28)<br/>プールは群一様なので実 ATUS 平日の群構成へ重み付ける"]
        Y1["fe.feasibility_summary → travel_single_rate, travel_odd_rate, night_intrusion"]
        Y2["im.fragmentation_summary → switch_mean, single_slot_ratio, wrap_closure_rate"]
        Y3["im.bigram_jsd と im.switch_dist_compare の emd<br/>いずれも重み付き・対角除く・行平均"]
        Y4["cd.separation_summary の separation_ratio<br/>および other_x_share"]
        Y5["各指標について 指標名_vs_zeroshot = 値 / ZERO_SHOT_GUARDRAILS の対応値<br/>★基準は実データ値ではなく zero-shot 値"]
        Y0 --> Y1 --> Y5
        Y0 --> Y2 --> Y5
        Y0 --> Y3 --> Y5
        Y0 --> Y4 --> Y5
    end

    ROWS["1 ckpt につき mask × eval_kind の行を生成<br/>ckpt, step, teacher_groups, eval_kind, reference, mask, n_per_group<br/>+ config + 軸1 スコア + 軸2 ガードレール"]
    OUT["OUT_CSV<br/>data/processed/aggregates/stage2_checkpoint_selection.csv"]

    START --> C1
    TGT --> X1
    REAL --> X1
    REAL --> Y0
    C3 --> X1
    C3 --> Y0
    X2 --> ROWS
    Y5 --> ROWS
    ROWS --> LOOP["paths の次の ckpt へ (evaluate_ckpt を繰り返す)"]
    LOOP --> OUT
```

**要点**

- **軸1 は `teacher_groups=28` のとき循環している**（学習の目的関数そのものを測っている）。
  非循環にするには `--holdout-groups` で群を抜いて学習し、`held-out` 行を読む。
- **軸2 の基準は実データ値ではなく zero-shot 値**（`ZERO_SHOT_GUARDRAILS` :83）。
  Stage 1 の時点で全指標がノイズ床の外にあるので、「壊さない」ではなく
  「悪化させない／改善する」で主張を立てる（設計書 §9.2）。
- 学習側に早期終了は無い。固定ステップ予算で回し切ってから、この 2 軸で事後に選ぶ。
- `bigram_jsd` と `switch_emd` は重み付きの値で固定する。重み無しだと別の値
  (0.0059 / 0.7765) になり、過去の記録に両方が混在している。
