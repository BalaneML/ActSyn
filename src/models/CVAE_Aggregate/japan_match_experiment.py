"""
実転移腕: ATUS(米)条件付きpretrain → 社会生活基本調査(実公表集計)への集計マッチ転移

目的 (計画ファイル ステップ5-6):
    日本個票なしで、実公表集計だけからデモグラ条件付き日本活動スケジュール生成を学習し、
    教師と評価を厳密に分離して転移の成否を測る。D2-b の教訓を反映:
      - Stage1 は米国「真ラベル」での条件付き pretrain (米国の群偏差構造を持ち込む)
      - Stage2 は群別率への直接マッチ (Pの逆問題を回避) + 関数空間アンカー
      - uncond床・zero-shot・placebo・アンカー無し の対照を常設

群定義: d = 性(2) × 年齢7区分(10歳) × 就業(有業/無業) = 28群 (共通12分類, OTHER_X除外)

教師/評価の分離 (統計量ホールドアウト; 全て平日・全国の実公表値):
    教師 = 公表の2つの「周辺」:
        margin_ga: 性×年齢7 の時刻別行動者率 (就業=総数 の行)  [14行]
        margin_ge: 性×就業 の時刻別行動者率 (年齢=総数 の行)   [4行]
      モデル側は μ̂(d) を推定人口由来の π(e|g,a) / π(a|g,e) で混合して周辺を再構成しマッチ。
    評価 = 公表の3元クロス (性×年齢7×就業) の時刻別行動者率 [28行] — 学習には一切使わない。
    さらに土曜・日曜表への外挿誤差を副次評価として報告 (教師は平日のみ)。

腕:
    zero-shot : 米国pretrainのまま (転移なし) — 米国構造の持ち込みだけでどこまで合うか
    matched   : 周辺マッチ + アンカー (本命)
    no-anchor : 周辺マッチのみ (アンカーの寄与のablation)
    placebo   : 周辺教師の群ラベルをシャッフルしてマッチ (負の対照)
    (床)      : 「全群=日本の人口平均」と予測したときの dev_rmse (判別ゼロの床; モデル不要)

アンカー: 関数空間蒸留 — 同一 (z,d) での pretrain デコーダ出力確率との MSE。
    率スケールとの整合が良く、ELBO係数の調整より安定。

出力: data/processed/aggregates/japan_match_experiment.csv + コンソール要約
使い方: uv run python src/models/CVAE_Aggregate/japan_match_experiment.py [--reuse-pretrain]
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, NUM_COMMON, stula_to_common  # noqa: E402
from model import (  # noqa: E402
    AggCVAE, train_elbo, group_rates, DEVICE, Z_DIM, NUM_SLOTS,
    PRETRAIN_EPOCHS, S_TRAIN, LR_EMB, LR_DECODER,
)

PROC_DIR   = REPO_ROOT / "data" / "processed"
ATUS_PATH  = PROC_DIR / "atus2024_common12_dataset.csv"
STULA_DIR  = PROC_DIR / "stula"
CKPT_PATH  = REPO_ROOT / "outputs" / "checkpoints" / "japan_pretrain_common12.pt"
OUT_CSV    = PROC_DIR / "aggregates" / "japan_match_experiment.csv"

SLOT_COLS  = [f"s{j}" for j in range(NUM_SLOTS)]

# 群定義: (gender 0男/1女) × (age7) × (emp 0無業/1有業) — d = g*14 + a*2 + e
N_G, N_A, N_E = 2, 7, 2
D_GROUPS = N_G * N_A * N_E
AGE16_TO_7 = {i: min((i - 1) // 2, 6) for i in range(1, 16)}  # 社基調5歳16区分→10歳7区分

MATCH_STEPS = 600
ANCHOR_W    = 1.0     # 関数空間アンカーの重み (no-anchor 腕では0)
SEED        = 42

# 事前登録リスト (米日差の報告軸)
BREAK_CANDIDATES    = ["WORK", "TRAVEL"]           # 破れ候補 (労働・通勤の時間帯構造)
PRESERVE_CANDIDATES = ["SLEEP_PERSONAL", "MEALS"]  # 保存候補


def d_index(g: int, a: int, e: int) -> int:
    return g * (N_A * N_E) + a * N_E + e


# ============================================================
# 社基調ターゲットの構築 (平日/土曜/日曜の各表から)
# ============================================================
def load_stula_targets(name: str) -> dict:
    """timeband CSV -> {cross (28,12,96), margin_ga (14,12,96), margin_ge (4,12,96),
                        pi_e (2,7,2: e|g,a), pop (2,7,2)} 率は[0,1], NaN=非公表"""
    df = pd.read_csv(STULA_DIR / f"{name}.csv")
    df = cast(pd.DataFrame, df[df["region"] == "00_全国"])
    com = stula_to_common(df)   # activity20 -> common12 に率を集約
    com["g"] = com["gender"].map({"1_男": 0, "2_女": 1})  # type: ignore
    com["e"] = com["employment"].map({"1_有業者": 1, "2_無業者": 0})  # type: ignore
    com["a16"] = com["age_class"].str[:2].map(lambda c: int(c) if c.isdigit() and c != "00" else np.nan)
    com["a7"] = com["a16"].map(AGE16_TO_7)  # type: ignore
    cidx = {c.name: int(c) for c in Common}
    com["ci"] = com["common"].map(cidx)  # type: ignore

    # 人口 (推定人口千人): 元dfの行動=総数行から。5歳16区分 (pop16) と10歳7区分 (pop) の両方を持つ
    pop_src = cast(pd.DataFrame, df[(df["activity"].str.startswith("00_"))])
    pop_src = pop_src.assign(
        g=pop_src["gender"].map({"1_男": 0, "2_女": 1}),  # type: ignore
        e=pop_src["employment"].map({"1_有業者": 1, "2_無業者": 0}),  # type: ignore
        a16=pop_src["age_class"].str[:2].map(lambda c: int(c) if c.isdigit() and c != "00" else np.nan),
    ).dropna(subset=["g", "e", "a16"])
    pop16 = np.zeros((N_G, 16, N_E))   # a16 は 1..15 (index 1..15 を使用)
    for (g, a16, e), grp in pop_src.groupby(["g", "a16", "e"]):  # type: ignore
        pop16[int(g), int(a16), int(e)] = grp["population_k"].iloc[0]
    pop = np.zeros((N_G, N_A, N_E))
    for a16, a7 in AGE16_TO_7.items():
        pop[:, a7, :] += pop16[:, a16, :]
    pi_e = pop / pop.sum(axis=2, keepdims=True)   # π(e|g,a)

    def rates_block(sub: pd.DataFrame, keys: list[str], shape: tuple) -> np.ndarray:
        """keys でインデックスした率テンソル (…, 12, 96)。行の重みは wrow 列 (人口重み平均)。"""
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

    # cross: 性×年齢×就業 (公表3元クロス; 5歳2区分を5歳人口重みで10歳へ集約)
    sub = com.dropna(subset=["g", "e", "a7"]).copy()
    sub["wrow"] = [pop16[int(g), int(a16), int(e)]
                   for g, a16, e in zip(sub["g"], sub["a16"], sub["e"])]
    cross = rates_block(sub, ["g", "a7", "e"], (N_G, N_A, N_E)).reshape(D_GROUPS, NUM_COMMON, NUM_SLOTS)
    # margin_ga: 就業=総数 の行 (性×年齢; 5歳→10歳は e 合計人口で重み付け)
    sub = cast(pd.DataFrame, com[(com["employment"] == "0_総数")]).dropna(subset=["g", "a7"]).copy()
    sub["wrow"] = [pop16[int(g), int(a16), :].sum() for g, a16 in zip(sub["g"], sub["a16"])]
    margin_ga = rates_block(sub, ["g", "a7"], (N_G, N_A))
    # margin_ge: 年齢=総数 の行 (性×就業; 1セル1行なので重みは1)
    sub = cast(pd.DataFrame, com[(com["age_class"] == "00_総数")]).dropna(subset=["g", "e"]).copy()
    sub["wrow"] = 1.0
    margin_ge = rates_block(sub, ["g", "e"], (N_G, N_E))
    return {"cross": cross, "margin_ga": margin_ga, "margin_ge": margin_ge,
            "pi_e": pi_e, "pop": pop}


# ============================================================
# Stage 2: 周辺マッチ + 関数空間アンカー (model.aggregate_match の周辺マッチ変種)
# ============================================================
def margin_match(model: AggCVAE, pre: AggCVAE, tgt: dict, mask_c: np.ndarray,
                 anchor_w: float, tag: str, shuffle_groups: np.ndarray | None = None):
    """margin_ga/margin_ge にマッチ。shuffle_groups は placebo 用の群置換 (μ̂ 側に適用)。
    model.aggregate_match との違い: P の逆問題を回避して公表周辺 (性×年齢, 性×就業) を
    π で混合再構成して直接マッチ + pretrain デコーダへの関数空間アンカー項。"""
    mga = torch.as_tensor(tgt["margin_ga"], dtype=torch.float32, device=DEVICE)  # (2,7,12,96)
    mge = torch.as_tensor(tgt["margin_ge"], dtype=torch.float32, device=DEVICE)  # (2,2,12,96)
    pi_e = torch.as_tensor(tgt["pi_e"], dtype=torch.float32, device=DEVICE)      # (2,7,2)
    pop  = torch.as_tensor(tgt["pop"], dtype=torch.float32, device=DEVICE)
    pi_a = pop / pop.sum(dim=1, keepdim=True)                                    # π(a|g,e)
    mc   = torch.as_tensor(mask_c, device=DEVICE)
    m_ga, m_ge = ~torch.isnan(mga), ~torch.isnan(mge)
    mga, mge = torch.nan_to_num(mga), torch.nan_to_num(mge)

    opt = torch.optim.Adam([
        {"params": model.demo_emb.parameters(), "lr": LR_EMB},
        {"params": model.decoder.parameters(),  "lr": LR_DECODER},
    ])
    perm = torch.arange(D_GROUPS, device=DEVICE) if shuffle_groups is None \
        else torch.as_tensor(shuffle_groups, device=DEVICE)
    model.train()
    for step in range(1, MATCH_STEPS + 1):
        z  = torch.randn(D_GROUPS * S_TRAIN, Z_DIM, device=DEVICE)
        di = perm.repeat_interleave(S_TRAIN)
        probs = model.decode_probs(z, di).view(D_GROUPS, S_TRAIN, NUM_SLOTS, NUM_COMMON)
        mu = probs.mean(dim=1).permute(0, 2, 1).view(N_G, N_A, N_E, NUM_COMMON, NUM_SLOTS)
        # 周辺の再構成: π で混合
        mu_ga = (mu * pi_e[..., None, None]).sum(dim=2)   # Σ_e π(e|g,a) μ -> (2,7,12,96)
        mu_ge = (mu * pi_a[..., None, None]).sum(dim=1)   # Σ_a π(a|g,e) μ -> (2,2,12,96)
        loss = (F.mse_loss(mu_ga[..., mc, :][m_ga[..., mc, :]], mga[..., mc, :][m_ga[..., mc, :]])
                + F.mse_loss(mu_ge[..., mc, :][m_ge[..., mc, :]], mge[..., mc, :][m_ge[..., mc, :]]))
        if anchor_w > 0:   # 関数空間アンカー: pretrain デコーダとの出力距離
            with torch.no_grad():
                probs_pre = pre.decode_probs(z, di).view_as(probs)
            loss = loss + anchor_w * F.mse_loss(probs, probs_pre)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  [{tag}] step {step:4d}  loss {loss.item():.6f}")


# ============================================================
# 評価
# ============================================================
def eval_against(mu_hat: np.ndarray, tgt: dict, mask_c: np.ndarray) -> dict:
    """μ̂ (28, 12*96 act-major) vs 公表3元クロス。OTHER_X と NaN を除外して RMSE 等を計算"""
    cross = tgt["cross"]                                    # (28,12,96)
    mh = mu_hat.reshape(D_GROUPS, NUM_COMMON, NUM_SLOTS)
    pop = tgt["pop"].reshape(D_GROUPS)
    pi_d = pop / pop.sum()
    m = ~np.isnan(cross) & mask_c[None, :, None]
    rmse = float(np.sqrt(((mh - cross)[m] ** 2).mean()))
    # 群偏差 (人口平均からの差): 条件付け能力
    mean_t = np.nansum(cross * pi_d[:, None, None], axis=0)
    mean_h = (mh * pi_d[:, None, None]).sum(axis=0)
    dev_rmse = float(np.sqrt((((mh - mean_h) - (cross - mean_t))[m] ** 2).mean()))
    per_act = {c.name: float(np.sqrt(((mh - cross)[:, int(c)][m[:, int(c)]] ** 2).mean()))
               for c in Common if mask_c[int(c)]}
    return {"rate_rmse": rmse, "dev_rmse": dev_rmse, **{f"rmse_{k}": v for k, v in per_act.items()}}


def main():
    ap = argparse.ArgumentParser(description="ATUS pretrain → 社基調 実集計マッチ転移")
    ap.add_argument("--reuse-pretrain", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # --- ATUS (共通12分類) と米国真ラベル (g×a7×e) ---
    df = pd.read_csv(ATUS_PATH)
    S_t = torch.as_tensor(df[SLOT_COLS].to_numpy(dtype=np.int64))
    w   = df["TUFINLWGT"].to_numpy(dtype=np.float64)
    a7  = np.clip((df["age"].to_numpy() - 15) // 10, 0, 6)
    emp = df["telfs"].isin([1, 2]).to_numpy().astype(int)
    demo_of = np.array([d_index(g, a, e) for g, a, e in zip(df["gender"], a7, emp)])
    print(f"ATUS: N={len(df)}  D={D_GROUPS}  device={DEVICE}")

    # --- 社基調ターゲット ---
    tgt_wd  = load_stula_targets("timeband_weekday")
    tgt_sat = load_stula_targets("timeband_saturday")
    tgt_sun = load_stula_targets("timeband_sunday")
    mask_c = np.array([c not in (Common.OTHER_X,) for c in Common])   # ③除外
    # 床: 「全群=人口平均」の dev_rmse (モデル不要)
    cross, pop = tgt_wd["cross"], tgt_wd["pop"].reshape(D_GROUPS)
    pi_d = pop / pop.sum()
    m = ~np.isnan(cross) & mask_c[None, :, None]
    floor_dev = float(np.sqrt(((cross - np.nansum(cross * pi_d[:, None, None], axis=0))[m] ** 2).mean()))
    print(f"日本クロスの群偏差RMS (判別ゼロの床) = {floor_dev:.4f}")

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

    # --- 腕の実行 ---
    def fork() -> AggCVAE:
        m_ = AggCVAE(NUM_COMMON, D_GROUPS).to(DEVICE)
        m_.load_state_dict(pre.state_dict())
        return m_

    arms: dict[str, AggCVAE] = {"zero-shot": pre}
    m_ = fork(); margin_match(m_, pre, tgt_wd, mask_c, ANCHOR_W, "matched")
    arms["matched"] = m_
    m_ = fork(); margin_match(m_, pre, tgt_wd, mask_c, 0.0, "no-anchor")
    arms["no-anchor"] = m_
    m_ = fork(); margin_match(m_, pre, tgt_wd, mask_c, ANCHOR_W, "placebo",
                              shuffle_groups=rng.permutation(D_GROUPS))
    arms["placebo"] = m_

    # --- 評価: 平日クロス (ホールドアウト) + 土日 (外挿) ---
    records = []
    for arm, model in arms.items():
        mu_hat = group_rates(model, D_GROUPS)
        for tgt_name, tgt in [("weekday_cross", tgt_wd), ("saturday_extrap", tgt_sat),
                              ("sunday_extrap", tgt_sun)]:
            records.append({"arm": arm, "eval": tgt_name, **eval_against(mu_hat, tgt, mask_c)})
    res = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    main_cols = ["arm", "eval", "rate_rmse", "dev_rmse"] + \
                [f"rmse_{k}" for k in BREAK_CANDIDATES + PRESERVE_CANDIDATES]
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n=== 平日クロス(28群)=統計量ホールドアウト / 土日=外挿 ===")
        print(res[main_cols].to_string(index=False))
    print(f"\n床 (判別ゼロ dev_rmse): {floor_dev:.4f} / 事前登録: 破れ候補={BREAK_CANDIDATES} 保存候補={PRESERVE_CANDIDATES}")
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
