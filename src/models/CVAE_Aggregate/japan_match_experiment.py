"""
実転移: ATUS(米)条件付きpretrain → 社会生活基本調査(実公表集計，日)への集計マッチ転移

目的:
    日本個票なしで、実公表集計だけからデモグラ条件付き日本活動スケジュール生成を学習し、
    教師と評価を厳密に分離して転移の成否を測る。D2-b の教訓を反映:
        - Stage1: 米国「真ラベル」での条件付き pretrain (米国の群偏差構造を持ち込む; 無条件nullは未使用)
        - Stage2: 群別率への直接マッチ (Pの逆問題を回避) + 関数空間アンカー

群定義:   d = 性(男/女) × 年齢7区分(10歳) × 就業(有業/無業) = 28群 
活動分類: 共通12分類, OTHER_X除外

スコープ = 平日 (曜日は条件化しない設計判断):
    ATUS は平日 diary (day_of_week 2..6) のみ使用し、pretrain・マッチ・評価の推定対象を
    「平日スケジュール生成器」に統一する (全曜日 pretrain だと anchor が平日/週末混合分布に
    係留してしまう)。土日への拡張は daytype (平日/土/日) の条件トークン化として将来課題。

教師/評価の分離 (統計量ホールドアウト; 全て平日・全国の実公表値):
    教師 = 公表の2つの「周辺」:
        margin_ga: 性2×年齢7 の時刻別行動者率 (就業=総数 の行)  [14行]
        margin_ge: 性2×就業2 の時刻別行動者率 (年齢=総数 の行)   [4行]
        モデル側は μ̂(d) を推定人口由来の π(e|g,a) / π(a|g,e) で混合して周辺を再構成しマッチ
    評価 = 公表の群別行動者率 (性2×年齢7×就業2 = 28群別の時刻別行動者率) [28行] — 学習には一切使わない

比較条件:
    zero-shot : 米国pretrainのまま (転移なし) — 米国構造の持ち込みだけで日本集計にどこまで合うか
    matched   : 周辺マッチ + アンカー (本命)
    no-anchor : 周辺マッチのみ (アンカーの寄与のablation)
    shuffled  : 周辺教師の群ラベルをシャッフルしてマッチ (負の対照)
    baseline  : 「全群=日本の人口平均」と予測する判別ゼロの参照条件 (モデル不要)。
                他バリアントと同じ全指標 (rate_mae/mse/rmse, dev_mae/rmse, 活動別MAE/RMSE) で
                結果CSVに1行として出力。稀な活動では baseline が MAE で有利になるので
                (「ほぼゼロと出す」だけで下がる)、MAE 単独でなく RMSE と併読すること

指標の読み分け:
    margin_mse : 教師(周辺2表)への当てはまり。Stage2 の学習目的関数そのもの (アンカー項を除く)
    rate_*     : 非教師(28群)への汎化。margin_mse と並べると汎化ギャップが読める
    rate_mse   : rate_rmse の二乗。順位情報は rate_rmse と同一で、損失(MSE)と単位を
                    揃えるためだけに置く。解釈・考察は rate_rmse / rate_mae 側で行うこと

アンカー: 関数空間アンカー（pretrain デコーダ出力への正則化）
    凍結CVAE(Stage1の後) と 集計マッチCVAE(Stage2中)の同一 (潜在変数z, 群d) でのデコーダ出力確率とのMSE
    率スケールとの整合が良く、ELBO係数の調整より安定

Stage1 の過学習診断 (個票ホールドアウト; --stage1-holdout / --holdout-only でのみ実行):
    上記の統計量ホールドアウトは「日本の集計」に対する評価であり、Stage1 が米国個票を
    過学習していないかは測れない。そこで ATUS 平日個票を群別層別で train/val に分割し、
    検証分割の重み付き ELBO の推移を記録する (run_stage1_holdout)。
    測るのは個票レベルの汎化のみ: 検証 recon が上昇に転じれば decoder が日記を記憶している。
    診断は本番とは別インスタンスで学習して捨てるため、チェックポイントにも Stage2 の結果にも
    影響しない (診断後に乱数列を張り直すので、既定実行と同一の結果になる)。
    注意: 下流で実際に使う量は z~N(0,I) 由来の群別行動者率 μ̂(d) であり、
    検証 ELBO は encoder→decoder の再構成経路の指標。μ̂ の汎化を直接見る指標ではない。

入力:
    data/processed/atus2024/atus2024_stula_common12_dataset.csv  米国個票 (全曜日; 本script が平日へ絞る)
    data/processed/stula/timeband_weekday.csv                    日本の公表集計 (平日・全国)

出力:
    data/processed/aggregates/japan_match_experiment.csv          バリアント別の評価指標 + コンソール要約
    outputs/checkpoints/japan_pretrain_common12_weekday.pt        Stage1 pretrain (--reuse-pretrain で再利用)
    outputs/checkpoints/japan_match_{variant}.pt                  バリアント別チェックポイント
    data/processed/aggregates/japan_match_stage1_holdout.csv      Stage1 診断の検証ELBO推移 (診断実行時のみ)

使い方:
    # 通常実行 (Stage1 pretrain から Stage2・評価まで通す)
    uv run python src/models/CVAE_Aggregate/japan_match_experiment.py

    # Stage1 を再利用して Stage2 から回す (Stage2 のみを変更して試すとき)
    uv run python src/models/CVAE_Aggregate/japan_match_experiment.py --reuse-pretrain

    # Stage1 の過学習診断のみ (Stage1本番/Stage2 は実行しない; 所要は通常実行と同程度)
    uv run python src/models/CVAE_Aggregate/japan_match_experiment.py --holdout-only

    # 診断してから通常実行 (診断は本番の結果を変えない)
    uv run python src/models/CVAE_Aggregate/japan_match_experiment.py --stage1-holdout

    フラグ:
        --reuse-pretrain : 既存の Stage1 チェックポイントがあれば読み込み、Stage1 を省略する
                            (ファイルが無い場合は通常どおり Stage1 を学習する)
        --stage1-holdout : Stage1 の個票ホールドアウト診断を実行してから本番へ進む
        --holdout-only   : 診断のみ実行して終了する (--stage1-holdout の指定は不要)

    主な調整箇所: MATCH_STEPS / ANCHOR_W / VAL_FRAC (本file),
                    PRETRAIN_EPOCHS / WEIGHT_DECAY (model.py; Stage1 の過学習を左右する2つ)
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, EXCLUDED, NUM_COMMON, stula_to_common  # noqa: E402
from model import (  # noqa: E402
    AggCVAE, train_elbo, group_rates, DEVICE, Z_DIM, NUM_SLOTS,
    PRETRAIN_EPOCHS, S_TRAIN, LR_EMB, LR_DECODER,
)

PROC_DIR   = REPO_ROOT / "data" / "processed"
ATUS_PATH  = PROC_DIR / "atus2024" / "atus2024_stula_common12_dataset.csv"
STULA_DIR  = PROC_DIR / "stula"
CKPT_PATH  = REPO_ROOT / "outputs" / "checkpoints" / "japan_pretrain_common12_weekday.pt"
OUT_CSV    = PROC_DIR / "aggregates" / "japan_match_experiment.csv"
HOLDOUT_CSV = PROC_DIR / "aggregates" / "japan_match_stage1_holdout.csv"

SLOT_COLS  = [f"s{j}" for j in range(NUM_SLOTS)]

# 群定義: (gender 0男/1女) × (age7) × (emp 0無業/1有業); d = g*14 + a*2 + e -> g in {0..27}
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E
AGE15_TO_7 = {i: min((i - 1) // 2, 6) for i in range(1, 16)}  # 社基調5歳15区分→10歳7区分

MATCH_STEPS = 600
ANCHOR_W    = 1.0     # 関数空間アンカーの重み (no-anchor バリアントでは0)
SEED        = 42
VAL_FRAC    = 0.2     # Stage1 個票ホールドアウトの検証割合 (--stage1-holdout 時のみ)
VAL_EVERY   = 10      # 検証 ELBO を記録するエポック間隔

# 事前登録リスト (米日差の報告軸)
BREAK_CANDIDATES    = ["WORK", "TRAVEL"]           # 破れ候補 (労働・通勤の時間帯構造)
PRESERVE_CANDIDATES = ["SLEEP_PERSONAL", "MEALS"]  # 保存候補


def d_index(g: int, a: int, e: int) -> int:
    return g * (N_A * N_E) + a * N_E + e


# ============================================================
# 社基調ターゲットの構築 (平日/土曜/日曜の各表から)
# ============================================================
def load_stula_targets(name: str) -> dict:
    """
    timeband CSV  ->  { group_rates_tbl (28,12,96), (demo, act, sche)
                        margin_ga (14,12,96),       (gender × age, act, sche)
                        margin_ge (4,12,96),        (gender × employ, act, sche)
                        pi_e (2,7,2: e|g,a),        (employ, age, gender)
                        pop (2,7,2)                 (employ, age, gender) 
                    } 率は[0,1], NaN=非公表
    """
    df = pd.read_csv(STULA_DIR / f"{name}.csv")  
    df = cast(pd.DataFrame, df[df["region"] == "00_全国"])
    com = stula_to_common(df)   # activity20 -> common12 に率を集約
    com["g"] = com["gender"].map({"1_男": 0, "2_女": 1})  # type: ignore
    com["e"] = com["employment"].map({"1_有業者": 1, "2_無業者": 0})  # type: ignore
    com["a15"] = com["age_class"].str[:2].map(lambda c: int(c) if isinstance(c, str) and c.isdigit() and c != "00" else np.nan)
    com["a7"] = com["a15"].map(AGE15_TO_7)  # type: ignore
    cidx = {c.name: int(c) for c in Common}
    com["ci"] = com["common"].map(cidx)  # type: ignore

    # 人口 (推定人口千人): 元dfの行動=総数行から。5歳15区分 (pop15) と10歳7区分 (pop) の両方を持つ
    pop_src = cast(pd.DataFrame, df[(df["activity"].str.startswith("00_"))])
    pop_src = pop_src.assign(
        g=pop_src["gender"].map({"1_男": 0, "2_女": 1}),  # type: ignore
        e=pop_src["employment"].map({"1_有業者": 1, "2_無業者": 0}),  # type: ignore
        a15=pop_src["age_class"].str[:2].map(lambda c: int(c) if isinstance(c, str) and c.isdigit() and c != "00" else np.nan),
    ).dropna(subset=["g", "e", "a15"])
    pop15 = np.zeros((N_G, 16, N_E))   # a15 は 1..15 (index 1..15 を使用)
    for (g, a15, e), grp in pop_src.groupby(["g", "a15", "e"]):  # type: ignore
        pop15[int(g), int(a15), int(e)] = grp["population_k"].iloc[0]  # type: ignore
    # 社基調の公表は5歳15区分だが、モデルの群定義は10歳7区分。
    # 5歳2区分ぶんの人口を足し合わせて7区分へ集約する（率でなく人数なので単純加算でよい）
    pop = np.zeros((N_G, N_A, N_E))
    for a15, a7 in AGE15_TO_7.items():
        pop[:, a7, :] += pop15[:, a15, :]
    # π(e|g,a): 性g・年齢a の人のうち就業状態 e の割合（e 軸で正規化）。
    # margin_match の「全確率の公式による周辺再構成」の混合重みになる
    pi_e = pop / pop.sum(axis=2, keepdims=True)   # π(e|g,a)

    def rates_block(sub: pd.DataFrame, keys: list[str], shape: tuple) -> np.ndarray:
        """keys でインデックスした率テンソル (…, 12, 96)。行の重みは wrow 列 (人口重み平均)。

        同じインデックスに公表行が複数落ちる場合（5歳2区分→10歳1区分の集約など）は
        wrow（人口）で加重平均する。率の集約は人数重みの加重平均でないと正しくない
        （人数の多い5歳区分の率が10歳区分の率をより強く決めるべき）。
        NaN（非公表）はスロット単位で分子 acc・分母 wsum の両方から外し、
        全行 NaN のスロットは 0/0 = NaN のまま返す（後段の m_ga/m_ge マスクで除外）。
        """
        wsum = np.zeros(shape + (NUM_COMMON, NUM_SLOTS))
        acc  = np.zeros(shape + (NUM_COMMON, NUM_SLOTS))
        for _, r in sub.iterrows():
            idx = tuple(int(r[k]) for k in keys) + (int(r["ci"]),)  # type: ignore
            vals = r[SLOT_COLS].to_numpy(dtype=float) / 100.0  # type: ignore
            m = ~np.isnan(vals)
            acc[idx][m]  += r["wrow"] * vals[m]
            wsum[idx][m] += r["wrow"]
        with np.errstate(invalid="ignore"):
            return acc / wsum

    # group_rates_tbl: 群別行動者率 (公表の性×年齢×就業=28群別; 5歳2区分を5歳人口重みで10歳へ集約)
    sub = com.dropna(subset=["g", "e", "a7"]).copy()
    sub["wrow"] = [pop15[int(g), int(a15), int(e)]
                    for g, a15, e in zip(sub["g"], sub["a15"], sub["e"])]
    group_rates_tbl = rates_block(sub, ["g", "a7", "e"], (N_G, N_A, N_E)).reshape(D_GROUPS, NUM_COMMON, NUM_SLOTS)
    # margin_ga: 就業=総数 の行 (性×年齢; 5歳→10歳は e 合計人口で重み付け)
    sub = cast(pd.DataFrame, com[(com["employment"] == "0_総数")]).dropna(subset=["g", "a7"]).copy()
    sub["wrow"] = [pop15[int(g), int(a15), :].sum() for g, a15 in zip(sub["g"], sub["a15"])]
    margin_ga = rates_block(sub, ["g", "a7"], (N_G, N_A))
    # margin_ge: 年齢=総数 の行 (性×就業; 1セル1行なので重みは1)
    sub = cast(pd.DataFrame, com[(com["age_class"] == "00_総数")]).dropna(subset=["g", "e"]).copy()
    sub["wrow"] = 1.0
    margin_ge = rates_block(sub, ["g", "e"], (N_G, N_E))
    return {"group_rates_tbl": group_rates_tbl, "margin_ga": margin_ga, "margin_ge": margin_ge,
            "pi_e": pi_e, "pop": pop}


# ============================================================
# Stage 2: 周辺マッチ + 関数空間アンカー (model.aggregate_match の周辺マッチ変種)
# ============================================================
def margin_match(   model         : AggCVAE,
                    pre           : AggCVAE,
                    tgt           : dict,
                    mask_c        : np.ndarray,
                    anchor_w      : float,
                    tag           : str,
                    shuffle_groups: np.ndarray | None = None
                ):
    """
    margin_ga/margin_ge にマッチさせる
    shuffle_groups は shuffled 用の群置換 (μ̂ 側に適用)
    model.aggregate_match (ここでは使わない) との違い: 
        P の逆問題を回避して公表周辺 (性×年齢, 性×就業) を
        π で混合再構成して直接マッチ + pretrain デコーダへの関数空間アンカー項。

    args:
        model         : 微調整するモデル
        pre           : Stage1までのPretrainモデル
        tgt           : 社基調の教師データ一式 (周辺2表，人口シェアπ，推定人口)
        mask_c        : 比較不能な活動 OHTER_Xを損失から外すマスク
        anchor_w      : アンカー重み; matchedは1.0, no-anchorでは0.0
        shuffle_groups: 群置換をして間違った対応で学習させる．0〜27 を並べ替えた1本の順列 (None: 通常の対応で学習)
    """
    mga = torch.as_tensor(tgt["margin_ga"], dtype=torch.float32, device=DEVICE)  # 教師1: 性×年齢の時刻別行動者率 (2,7,12,96), (sex, age, act, sched)
    mge = torch.as_tensor(tgt["margin_ge"], dtype=torch.float32, device=DEVICE)  # 教師2: 性×就業の時刻別行動者率 (2,2,12,96), (sex, employment, act, sched)
    pi_e = torch.as_tensor(tgt["pi_e"], dtype=torch.float32, device=DEVICE)      # π(就業|性,年齢): 性g・年齢a の人のうち、就業状態が e である人口シェア, (2,7,2), (sex, age, employment)
    pop  = torch.as_tensor(tgt["pop"], dtype=torch.float32, device=DEVICE)       # 28群の推定人口, (2,7,2)
    pi_a = pop / pop.sum(dim=1, keepdim=True)                                    # π(年齢|性,就業): 性g・就業e の人のうち、年齢が a である人口シェア
    mc   = torch.as_tensor(mask_c, device=DEVICE)                                # 共通12分類のうち、米日で比較不能として除外する OTHER_X
    m_ga, m_ge = ~torch.isnan(mga), ~torch.isnan(mge)                            # NaNマスク
    mga, mge = torch.nan_to_num(mga), torch.nan_to_num(mge)                      # NaNを0に置き換える．値としての0に意味はない

    # 学習速度の違いのため2つに分ける
    opt = torch.optim.Adam([
        {"params": model.demo_emb.parameters(), "lr": LR_EMB},      # 群埋め込みを学習する
        {"params": model.decoder.parameters(),  "lr": LR_DECODER},  # 群対応するスケジュール生成を学習する
    ])

    # shuffled に対しては壊れた群対応を渡す（通常時は恒等順列 = そのままの対応）。
    # 群番号を並べ替えると「教師の行 g と、モデルが生成する群 d」の対応が崩れるので、
    # 正しい対応があるから効く、という機構主張の反証条件になる
    perm = torch.arange(D_GROUPS, device=DEVICE) if shuffle_groups is None \
        else torch.as_tensor(shuffle_groups, device=DEVICE)

    # 学習モード
    model.train()

    # 集計マッチループ
    for step in range(1, MATCH_STEPS + 1):
        z  = torch.randn(D_GROUPS * S_TRAIN, Z_DIM, device=DEVICE)  # 潜在変数z (28*128=3584, 64); 事前分布 N(0,I) から引くだけなので再パラメータ化不要
        di = perm.repeat_interleave(S_TRAIN)  # (28*128=3584,), [0,0,...(128個)...,0, 1,1,...,1, 2,...], 最初の128本を群0,次の128本を群1,...,最後の128本を群27
        probs = model.decode_probs(z, di).view(D_GROUPS, S_TRAIN, NUM_SLOTS, NUM_COMMON)  # (28*128=3584,96,n_act) -> (28群, 128人, 96時刻, 12活動)

        # 各郡の期待行動者率: (28群, 128人, 96時刻, 12活動) --各群で平均--> (28, 96, 12) --並び替え(教師側に合わせる)--> (28, 12 96) --群を3軸に展開--> (2, 7, 2, 12, 96)
        mu = probs.mean(dim=1).permute(0, 2, 1).view(N_G, N_A, N_E, NUM_COMMON, NUM_SLOTS)

        # 周辺の再構成: π で混合（全確率の公式; ガイド §4）
        # 公表周辺 margin_ga は「性g・年齢a の全員（有業+無業を合わせた総数）」の
        # 行動者率。モデルは (g,a,e) 別の μ̂ しか持たないので、全確率の公式
        #     μ(g,a) = Σ_e π(e|g,a) · μ(g,a,e)
        # で就業をつぶした周辺を再構成する。具体例:
        #   40代男性の行動者率 = (有業割合)×(有業40代男性の率) + (無業割合)×(無業40代男性の率)
        # π は推定人口から計算した既知の人口シェアなので、これは定数重みの
        # 加重平均にすぎず微分可能（勾配は μ̂ 側に流れる）。
        #   pi_e (2,7,2): この性・年齢の中で、有業/無業は何割か → e 軸(dim=2)をつぶす
        #   pi_a (2,7,2): この性・就業の中で、各年齢は何割か   → a 軸(dim=1)をつぶす
        # [..., None, None] は π を μ (2,7,2,12,96) にブロードキャストするための
        # 軸合わせ（活動・時刻の2軸を追加）
        mu_ga = (mu * pi_e[..., None, None]).sum(dim=2)   # Σ_e π(e|g,a) μ -> (2,7,12,96)
        mu_ge = (mu * pi_a[..., None, None]).sum(dim=1)   # Σ_a π(a|g,e) μ -> (2,2,12,96)

        # マッチ損失: 周辺2表それぞれと MSE。
        # [..., mc, :] で比較不能活動 OTHER_X の列を落とし、
        # [m_ga...] で非公表 NaN のセルを損失から除外（マスクの二段がけ）
        loss = (F.mse_loss(mu_ga[..., mc, :][m_ga[..., mc, :]], mga[..., mc, :][m_ga[..., mc, :]])
                + F.mse_loss(mu_ge[..., mc, :][m_ge[..., mc, :]], mge[..., mc, :][m_ge[..., mc, :]]))
        if anchor_w > 0:   # 関数空間アンカー: pretrain デコーダとの出力距離（ガイド §4）
            # 周辺2表（14+4=18行）だけで28群×12活動×96スロットを拘束するのは
            # 劣決定（自由度が余る）。放置すると decoder が Stage1 で覚えた
            # 「スケジュールらしさ」を捨てて周辺だけ合わせる退化解に行きうる。
            # そこで同じ (z,d) に対する凍結 pretrain の出力確率との MSE を罰則に足す。
            # 「重み空間で動くな」でなく「出力（関数）空間で離れるな」という正則化なので
            # 率のスケールと直接整合する（ELBO 係数の調整より安定）
            with torch.no_grad():
                probs_pre = pre.decode_probs(z, di).view_as(probs)
            loss = loss + anchor_w * F.mse_loss(probs, probs_pre)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  [{tag}] step {step:4d}  loss {loss.item():.6f}")


# ============================================================
# Stage 1 の過学習診断 (個票ホールドアウト; --stage1-holdout でのみ実行)
# ============================================================
def stratified_split(
        demo_of: np.ndarray,
        rng: np.random.Generator,
        val_frac: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    群 d ごとに val_frac を検証側へ回す層別分割 -> (train index, val index)

    ATUS 平日の群別人数は最小17・最大329と偏りが大きく、単純なランダム分割では
    小群が検証側から消える。群内で分割して全28群が両側に残るようにする。
    """
    tr, va = [], []
    for d in np.unique(demo_of):
        idx = rng.permutation(np.flatnonzero(demo_of == d))
        n_va = max(1, int(round(len(idx) * val_frac)))
        va.append(idx[:n_va])
        tr.append(idx[n_va:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def run_stage1_holdout(
        S_t: torch.Tensor,
        demo_of: np.ndarray,
        w: np.ndarray,
        rng: np.random.Generator
        ) -> None:
    """
    Stage1 (米国条件付き pretrain) が個票を過学習していないかを診断する。

    本番の pretrain とは別のモデルを訓練分割だけで学習し、検証分割の重み付き ELBO の
    推移を記録する。本番モデル・チェックポイント・Stage2 の結果には一切触れない
    (本番は全個票で学習するので、ここで学習したモデルは診断後に捨てる)。

    ATUS の TUCASEID は 1行1人 (1人1日記) なので、行単位の分割がそのまま個人単位の
    ホールドアウトになる。診断するのは個票レベルの汎化のみ:
    検証 recon (96スロット合計 CE) が訓練 recon から乖離して上昇し始めれば、
    decoder が個々の日記を記憶している = PRETRAIN_EPOCHS が過剰という読み方になる。
    """
    idx_tr, idx_va = stratified_split(demo_of, rng, VAL_FRAC)
    demo_t = torch.as_tensor(demo_of)
    print(f"\nStage1 個票ホールドアウト: train={len(idx_tr)}  val={len(idx_va)}  "
            f"(群別層別, val_frac={VAL_FRAC})")

    diag = AggCVAE(NUM_COMMON, D_GROUPS).to(DEVICE)
    hist = train_elbo(diag, S_t[idx_tr], demo_t[idx_tr], w[idx_tr],
                        PRETRAIN_EPOCHS, "stage1-holdout",
                        val=(S_t[idx_va], demo_t[idx_va], w[idx_va]),
                        eval_every=VAL_EVERY)

    df = pd.DataFrame(hist)
    df["n_train"], df["n_val"] = len(idx_tr), len(idx_va)
    HOLDOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HOLDOUT_CSV, index=False)

    # 要約は DataFrame の行アクセス (Series) でなく履歴 dict から直に取る
    best = min(hist, key=lambda h: h["val_loss"])   # 検証損失が最小のエポック
    last = hist[-1]                                  # 最終エポック
    print("\n=== Stage1 個票ホールドアウト (重み付き ELBO; 1個票あたり96スロット合計) ===")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df[["epoch", "train_loss", "val_loss", "train_recon", "val_recon",
                    "train_kl", "val_kl"]].to_string(index=False))
    print(f"\nval_loss 最小   : epoch {best['epoch']}  val {best['val_loss']:.3f}")
    print(f"最終エポック    : epoch {last['epoch']}  "
            f"train {last['train_loss']:.3f}  val {last['val_loss']:.3f}  "
            f"gap {last['val_loss'] - last['train_loss']:.3f}")
    # 検証損失が最小点から悪化していれば、その差が過学習の大きさ
    print(f"val 悪化幅      : {last['val_loss'] - best['val_loss']:+.3f}  "
            f"(最小点 epoch {best['epoch']} → 最終 epoch {last['epoch']}; "
            f"正なら PRETRAIN_EPOCHS={PRETRAIN_EPOCHS} は過剰)")
    print(f"saved -> {HOLDOUT_CSV}")


# ============================================================
# 評価
# ============================================================
def nan_renorm_pop_weights(grp_tbl: np.ndarray, pi_d: np.ndarray) -> np.ndarray:
    """セル (活動,時刻) ごとに非公表 NaN の群を除外し再正規化した人口重み (28,12,96)。
    全群 NaN のセルは 0/0=NaN (評価マスク m で除外されるので害はない)"""
    w = np.where(np.isnan(grp_tbl), 0.0, pi_d[:, None, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return w / w.sum(axis=0)


def margin_fit_mse(mu_hat: np.ndarray, tgt: dict, mask_c: np.ndarray) -> float:
    """μ̂ (28, 12*96 act-major) の周辺2表への当てはまり = 学習目的関数の値。

    margin_match の損失 (L284-285) と同一定義:
        MSE(Σ_e π(e|g,a) μ̂, margin_ga) + MSE(Σ_a π(a|g,e) μ̂, margin_ge)
    アンカー項 (anchor_w * MSE(probs, probs_pre)) は含めない。これは正則化であって
    データ項ではなく、バリアント間で重みが異なる (matched=1.0, no-anchor=0.0) ため、
    足すと「教師への当てはまり」の比較にならないから。

    eval_against が非教師の28群 (統計量ホールドアウト) を測るのに対し、こちらは
    学習で最小化した教師そのものを測る。両者を並べて汎化ギャップを読む。

    shuffled バリアントの注意: 学習は置換された群対応で行われるが、ここでは
    eval_against と同じ μ̂ (恒等対応で生成) を渡す。したがって shuffled の値は
    「学習ログの loss」ではなく「正しい対応で測り直した当てはまり」であり、
    他バリアントと同じ土俵の量になる。
    """
    mu = mu_hat.reshape(N_G, N_A, N_E, NUM_COMMON, NUM_SLOTS)
    pop  = tgt["pop"]                                   # (2,7,2)
    pi_e = tgt["pi_e"]                                  # π(就業|性,年齢) (2,7,2)
    pi_a = pop / pop.sum(axis=1, keepdims=True)         # π(年齢|性,就業) (2,7,2)
    total = 0.0
    for mu_m, tgt_m in (((mu * pi_e[..., None, None]).sum(axis=2), tgt["margin_ga"]),
                        ((mu * pi_a[..., None, None]).sum(axis=1), tgt["margin_ge"])):
        # OTHER_X 列を落とし (mask_c)、非公表 NaN セルを除外 (学習時と同じ二段マスク)。
        # 活動軸は末尾から2番目なので、学習側の mga[..., mc, :] と同じく ... で位置を合わせる
        m = ~np.isnan(tgt_m[..., mask_c, :])
        e = (mu_m[..., mask_c, :] - tgt_m[..., mask_c, :])[m]
        total += float((e ** 2).mean())
    return total


def eval_against(mu_hat: np.ndarray, tgt: dict, mask_c: np.ndarray) -> dict:
    """μ̂ (28, 12*96 act-major) vs 公表の群別行動者率。OTHER_X と NaN を除外して MAE / MSE / RMSE を計算。

    MAE と RMSE を必ず併記する。教師セル (28群×11活動×96スロット) は率<0.01 のセルが
    約半数を占める強い偏りがあり、両者は別のものを測るため:
        MAE  = 平均何ポイントずれるか。解釈可能な主指標だが、
                「稀な活動をほぼゼロと出す」だけで下がる (多様性を犠牲にすると得をする)。
        RMSE = 二乗和が誤差上位セルに集中するため、実質「ピーク帯 (WORK 昼・SLEEP 深夜)
                がどれだけ合うか」の指標。裾に敏感な補助指標として残す。
        MSE  = RMSE の二乗であり順位情報は RMSE と同一。学習損失 (MSE) と単位を
                揃えて読むためだけに置く。解釈は RMSE 側で行うこと。
    """
    grp_tbl = tgt["group_rates_tbl"]                        # (28,12,96)
    mh = mu_hat.reshape(D_GROUPS, NUM_COMMON, NUM_SLOTS)
    pop = tgt["pop"].reshape(D_GROUPS)
    pi_d = pop / pop.sum()
    m = ~np.isnan(grp_tbl) & mask_c[None, :, None]
    err = (mh - grp_tbl)[m]
    mae  = float(np.abs(err).mean())
    mse  = float((err ** 2).mean())
    rmse = float(np.sqrt(mse))
    # 群偏差 (人口平均からの差): 条件付け能力。
    # 非公表 NaN の群をセルごとに除外して π を再正規化し、モデル側の平均も
    # 同じ群集合で取る (教師とモデルで比較対象の「人口平均」を揃える)
    W = nan_renorm_pop_weights(grp_tbl, pi_d)
    mean_t = np.nansum(grp_tbl * W, axis=0)
    mean_h = np.nansum(mh * W, axis=0)
    dev = ((mh - mean_h) - (grp_tbl - mean_t))[m]
    dev_mae  = float(np.abs(dev).mean())
    dev_rmse = float(np.sqrt((dev ** 2).mean()))
    per_act = {}
    for c in Common:
        if not mask_c[int(c)]:
            continue
        e = (mh - grp_tbl)[:, int(c)][m[:, int(c)]]
        per_act[f"mae_{c.name}"]  = float(np.abs(e).mean())
        per_act[f"rmse_{c.name}"] = float(np.sqrt((e ** 2).mean()))
    return {"rate_mae": mae, "rate_mse": mse, "rate_rmse": rmse,
            "dev_mae": dev_mae, "dev_rmse": dev_rmse, **per_act}


def main():
    # --- コマンドライン引数の受け取りと乱数シードの固定 ---
    ap = argparse.ArgumentParser(description="ATUS pretrain → 社基調 実集計マッチ転移")
    ap.add_argument("--reuse-pretrain", action="store_true")  # 事前学習モデルを使用する (Stage 2から始める) コマンド
    ap.add_argument("--stage1-holdout", action="store_true",  # Stage1 の個票ホールドアウト診断を追加実行する
                    help="Stage1 の過学習診断 (個票ホールドアウトの検証ELBO) を実行してから本番へ進む")
    ap.add_argument("--holdout-only", action="store_true",    # 診断だけ回して終了する
                    help="Stage1 ホールドアウト診断のみ実行し、Stage1本番/Stage2 は行わない")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # --- ATUS (共通12分類・平日のみ) と米国真ラベル (g2×a7×e2) ---
    df = pd.read_csv(ATUS_PATH)  # 102列 -> (ID, w, age, gender, day, telfs, s0..s95)
    # 平日 (月..金) のみ: スコープ=平日。boolean mask での絞り込みは戻り値が DataFrame と
    # 推論されないため、他の絞り込み箇所と同様に cast で型を明示する
    df = cast(pd.DataFrame, df[df["day_of_week"].between(2, 6)]).reset_index(drop=True)
    S_t = torch.as_tensor(df[SLOT_COLS].to_numpy(dtype=np.int64))  # (N, 96); dfからスロット部分を取り出し -> numpy tensor
    w   = df["TUFINLWGT"].to_numpy(dtype=np.float64)  # (7439,); dfかから重み部分を取り出し -> numpy tensor
    a7  = np.clip((df["age"].to_numpy() - 15) // 10, 0, 6)   # 15歳起点10歳刻みの7区分 (15-24→0, ..., 75+→6)
    emp = df["telfs"].isin([1, 2]).to_numpy().astype(int)    # ATUS 就業状態: 1,2(就業or休業中) = 有業（社基調の有業/無業に対応付け）
    demo_of = np.array([d_index(g, a, e) for g, a, e in zip(df["gender"], a7, emp)])  # 各人物に対応する群番号を割り当てている 
    print(f"ATUS: N={len(df)}  D={D_GROUPS}  device={DEVICE}")

    # --- Stage1 過学習診断 (任意; 本番モデルとは別インスタンスで捨て学習) ---
    if args.stage1_holdout or args.holdout_only:
        run_stage1_holdout(S_t, demo_of, w, rng)
        if args.holdout_only:
            return
        # 診断が torch/numpy の RNG を消費するため、以降の本番経路を
        # 診断なし実行と同一の乱数列に戻す (結果の再現性を保つ)
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)

    # --- 社基調ターゲット (平日) ---
    tgt_wd  = load_stula_targets("timeband_weekday")
    mask_c = np.array([c not in EXCLUDED for c in Common])   # OTHER_X（米日で対応が取れない残差カテゴリ）を教師・評価の両方から除外; 除外集合の定義はクロスウォーク側が唯一の出所
    # ベースライン: 「全群=人口平均」と予測する判別ゼロの参照条件 (モデル不要)
    # μ̂ を全群共通の人口平均にして他バリアントと同じ eval_against に通し、
    # 全指標 (rate_rmse / dev_rmse / 活動別RMSE) を結果CSVに載せる
    grp_tbl, pop = tgt_wd["group_rates_tbl"], tgt_wd["pop"].reshape(D_GROUPS)
    pi_d = pop / pop.sum()
    W0 = nan_renorm_pop_weights(grp_tbl, pi_d)
    baseline_mu = np.broadcast_to(np.nansum(grp_tbl * W0, axis=0), grp_tbl.shape)

    # --- Stage 1: 米国真ラベルで条件付き pretrain ---
    pre = AggCVAE(NUM_COMMON, D_GROUPS).to(DEVICE)
    if args.reuse_pretrain and CKPT_PATH.exists():
        pre.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Stage1 再利用: {CKPT_PATH}")
    else:
        print("Stage1: 米国真ラベルで条件付き pretrain ...")
        train_elbo(pre, S_t, torch.as_tensor(demo_of), w, PRETRAIN_EPOCHS, "US-pretrain")
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pre.state_dict(), CKPT_PATH)

    # --- バリアントの実行 ---
    def fork() -> AggCVAE:
        m_ = AggCVAE(NUM_COMMON, D_GROUPS).to(DEVICE)
        m_.load_state_dict(pre.state_dict())
        return m_

    # --- Stage 2 ---
    print("Stage2: 社会生活基本調査・周辺マッチング ...")

    # --- Stage 2: zero-shot --
    variants: dict[str, AggCVAE] = {"zero-shot": pre}

    # --- Stage 2: matched --
    m_ = fork()
    margin_match(m_, pre, tgt_wd, mask_c, ANCHOR_W, "matched")
    variants["matched"] = m_

    # --- Stage 2: no-anchor --
    m_ = fork()
    margin_match(m_, pre, tgt_wd, mask_c, 0.0, "no-anchor")
    variants["no-anchor"] = m_

    # --- Stage 2: shuffled --
    m_ = fork()
    margin_match(m_, pre, tgt_wd, mask_c, ANCHOR_W, "shuffled",
                shuffle_groups=rng.permutation(D_GROUPS))
    variants["shuffled"] = m_

    # --- バリアントごとのチェックポイント保存 (ノートブック等での追加分析用) ---
    for variant, m_ in variants.items():
        variant_path = CKPT_PATH.parent / f"japan_match_{variant}.pt"
        torch.save(m_.state_dict(), variant_path)
        print(f"saved variant checkpoint -> {variant_path}")

    # --- 評価: 平日の群別行動者率 (統計量ホールドアウト) ---
    # margin_mse (教師=周辺2表への当てはまり) と rate_* (非教師=28群) を同じ μ̂ から出し、
    # 1行で「教師にどれだけ合ったか」対「非教師にどれだけ汎化したか」を読めるようにする
    records = []
    for variant, model in variants.items():
        mu_hat = group_rates(model, D_GROUPS)
        records.append({"variant": variant, "eval": "weekday_groups",
                        "margin_mse": margin_fit_mse(mu_hat, tgt_wd, mask_c),
                        **eval_against(mu_hat, tgt_wd, mask_c)})
    records.append({"variant": "baseline", "eval": "weekday_groups",
                    "margin_mse": margin_fit_mse(baseline_mu, tgt_wd, mask_c),
                    **eval_against(baseline_mu, tgt_wd, mask_c)})
    res = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    main_cols = ["variant", "eval", "margin_mse",
                 "rate_mae", "rate_mse", "rate_rmse", "dev_mae", "dev_rmse"] + \
                [f"{p}_{k}" for k in BREAK_CANDIDATES + PRESERVE_CANDIDATES for p in ("mae", "rmse")]
    act_names = [c.name for c in Common if mask_c[int(c)]]
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        # margin_mse / rate_mse は率の二乗 (1e-4 オーダー) なので 4桁だと 0.0000 に潰れる。
        # 主表だけ6桁で出す
        print("\n=== 平日の群別行動者率(28群)=統計量ホールドアウト ===")
        print("    margin_mse = 教師(周辺2表)への当てはまり / rate_*・dev_* = 非教師(28群)への汎化")
        with pd.option_context("display.float_format", "{:.6f}".format):
            print(res[main_cols].to_string(index=False))
        print("\n=== 活動別 MAE (全活動分類; OTHER_X除外) ===")
        print(res[["variant"] + [f"mae_{n}" for n in act_names]].to_string(index=False))
        print("\n=== 活動別 RMSE (全活動分類; OTHER_X除外) ===")
        print(res[["variant"] + [f"rmse_{n}" for n in act_names]].to_string(index=False))
    print(f"\n事前登録: 破れ候補={BREAK_CANDIDATES} 保存候補={PRESERVE_CANDIDATES}")
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
