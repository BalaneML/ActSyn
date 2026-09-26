"""
clock_diagnostics.py
================
「生成モデルは時刻を知っているか」の診断。

★ なぜこの診断が要るか:
    AggDDPM (src/models/DDPM_Aggregate/model.py) のバックボーンには
    **時刻スロット軸の位置符号が存在しない**。
      - AttnBlock1D は nn.MultiheadAttention を素で呼ぶ（位置埋め込みの加算なし）
        → 時間軸の self-attention は置換不変
      - timestep_embedding は拡散ステップ t 用であって時刻スロット t 用ではない
    したがって「いまが朝か夜か」は Conv1d(padding=1) の境界効果が階層を伝播する
    副産物としてしか入らない。逆過程の初期 (t≈T, x_t がほぼ純ノイズ) は日内リズムの
    大枠を決める段階なので、ここで時刻符号が無いことは原理的に効きうる。

    この空白を埋める設計（学習型絶対PE / 巡回フーリエ時刻符号 / 巡回バックボーン）に
    進む前に、**欠如が実害かどうか**を再学習なしで測るのが本モジュール。

★ 5つの診断:
    B1 curve_comparison      時刻別行動者率カーブの鈍化（ピーク高さ・L1）
    B2 onset_comparison      個人の初回開始時刻の分布（集団カーブが合っていても
                             個人がばらついていれば時刻符号は無い）
    B3 wrap_comparison       日跨ぎ境界の破れ（活動日は「環」か「線分」か）
    B4 position_probe        中間特徴から時刻位相を線形復元できるか（モデル内部）
    B5 shift_equivariance    時間軸を巡回シフトしたときの予測のずれ（モデル内部）

    B1-B3 は個票の配列だけを見る（torch 不要）。B4-B5 は学習済みモデルを見る。

★ 「値が違う」だけでは乖離の証拠にならない:
    B1-B3 は individual_metrics.null_band / null_band_2sample で実データ自身の
    有限標本ゆらぎ（床）を併記する。床の外に出て初めて乖離と言う。

★ B4/B5 の絶対値は単独では読めない:
    どちらもモデル内部の量で、「いくつなら十分か」の外部基準が無い。そこで同じ
    モデル内の 2 つのヤードスティックと並べて比べる:
      cond_gap   条件（群）を変えたときの予測の動き = 使えている情報の大きさ
      noise_gap  入力ノイズを引き直したときの予測の動き = 出力の変動幅そのもの
    位置シフトへの反応が cond_gap より桁で小さければ、モデルは時刻をほとんど
    見ていない。

★ 診断対象は差し替えられる:
    --model-dir でモデルフォルダを渡す。DDPM_Aggregate_Tang は Tang et al. 2025 の
    構成に寄せたバックボーンで、こちらには時刻の位置符号がある。同じ診断を両方に
    掛けて B4/B5 を並べれば、位置符号の有無が実測でどう出るかを比較できる。
    切替の条件は load_pretrained / Diffusion / cond_grid / features / in_channels の
    契約を満たすこと。出力 CSV はフォルダ名で分かれる。

使い方:
    uv run python src/eval/test_clock_diagnostics.py    # 自己テスト
    uv run python src/eval/clock_diagnostics.py         # 本番診断 (DDPM_Aggregate)
    uv run python src/eval/clock_diagnostics.py --model-dir src/models/DDPM_Aggregate_Tang

    # 同じフォルダの別の重みを比べる（時刻符号アブレーション）。重みと生成 CSV は同じ組を渡す
    uv run python src/eval/clock_diagnostics.py --model-dir src/models/DDPM_Aggregate_Simple \\
        --ckpt outputs/checkpoints/ddpm_simple_pretrain_common12_weekday_clock.pt \\
        --gen outputs/generated/ddpm_simple_pretrain_samples_clock.csv --tag clock
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import wasserstein_distance

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_SLOTS = 96
OUT_CSV = REPO_ROOT / "data" / "processed" / "aggregates" / "ddpm_clock_diagnostics.csv"
# 診断対象の既定。--model-dir で DDPM_Aggregate_Tang などに差し替えられる
DEFAULT_MODEL_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate"

# B1 でピーク鈍化を見る主対象。日内リズムが鋭い活動ほど「時刻符号の欠如」が出やすい
PEAK_ACTS = ["TRAVEL", "MEALS", "WORK", "SLEEP_PERSONAL"]
# B5 の巡回シフト量（スロット）。1=15分, 4=1時間, 12=3時間, 24=6時間, 48=12時間
SHIFTS = (1, 4, 12, 24, 48)
# B4/B5 で使う拡散ステップ。999 はほぼ純ノイズ = 日内リズムの大枠を決める段階
NOISE_LEVELS = (999, 800, 500, 200)


def load_module(name: str, path: Path):
    """sys.modules に一意名で直接載せる（同名 model.py の取り違えを防ぐ）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


im = load_module("clockdiag_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")


# ============================================================
# B1. 時刻別行動者率カーブの鈍化
# ============================================================
def curve_stats(sched: IntArr, n_act: int, w: FloatArr | None = None) -> dict[str, FloatArr]:
    """時刻別行動者率カーブ (96, n_act) と、そのピーク高さ (n_act,)。

    「時刻符号が無ければ日内リズムのピークが平坦化する」という予測を測る土台。
    """
    curve = im.participation_by_slot(sched, n_act, w)
    return {"curve": curve, "peak": curve.max(axis=0)}


def curve_comparison(real: IntArr, gen: IntArr, n_act: int, act_names: list[str],
                     w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                     n_boot: int = 60, seed: int = 0) -> pd.DataFrame:
    """★B1: 活動別のピーク高さ比とカーブ L1 距離、それぞれのノイズ床。

    peak_ratio < 1 は生成側のピークが低い = 鈍化。床の外に出ていれば
    「時刻符号が無いせいでリズムがぼやけている」ことの直接の証拠になる。
    """
    cr = curve_stats(real, n_act, w_real)
    cg = curve_stats(gen, n_act, w_gen)

    rows = []
    for a in range(n_act):
        l1 = float(np.abs(cr["curve"][:, a] - cg["curve"][:, a]).sum())
        # ピーク高さの床は1標本（実データの推定精度）、L1 の床は2標本
        p_lo, _, p_hi = im.null_band(
            real, _peak_of(a, n_act), n_boot=n_boot, seed=seed, w=w_real)
        _, _, l1_hi = im.null_band_2sample(
            real, _l1_of(a, n_act), n_gen=len(gen), n_boot=n_boot, seed=seed, w=w_real)
        rows.append({
            "activity": act_names[a],
            "peak_real": cr["peak"][a], "peak_gen": cg["peak"][a],
            "peak_ratio": float(cg["peak"][a] / cr["peak"][a]) if cr["peak"][a] > 0 else np.nan,
            "peak_ci_lo": p_lo, "peak_ci_hi": p_hi,
            "peak_verdict": "床の内" if p_lo <= cg["peak"][a] <= p_hi else "★床の外",
            "curve_l1": l1, "curve_l1_floor_hi": l1_hi,
            "l1_verdict": "床の内" if l1 <= l1_hi else "★床の外",
        })
    return pd.DataFrame(rows)


def _peak_of(a: int, n_act: int) -> Callable[..., float]:
    """活動 a の時刻別行動者率カーブのピーク高さを返す stat_fn（null_band 用）。"""
    def stat(sched: IntArr, w: FloatArr | None = None) -> float:
        return float(im.participation_by_slot(sched, n_act, w)[:, a].max())
    return stat


def _l1_of(a: int, n_act: int) -> Callable[[IntArr, IntArr], float]:
    """活動 a のカーブ L1 距離を返す stat_fn（null_band_2sample 用）。"""
    def stat(r: IntArr, g: IntArr) -> float:
        cr = im.participation_by_slot(r, n_act)[:, a]
        cg = im.participation_by_slot(g, n_act)[:, a]
        return float(np.abs(cr - cg).sum())
    return stat


# ============================================================
# B2. 個人の初回開始時刻
# ============================================================
def first_onset(sched: IntArr, act: int) -> IntArr:
    """活動 act を1回以上する個票について、その最初のスロット (K,)。

    しない個票は除く。集団カーブが合っていても、個人の開始時刻が過分散なら
    「モデルが時刻を知らず、開始をランダムに置いている」ことになる。
    """
    sched = np.asarray(sched)
    has = (sched == act).any(axis=1)
    return np.argmax(sched[has] == act, axis=1).astype(np.int64)


def first_onset_with_weights(sched: IntArr, act: int, w: FloatArr | None = None
                             ) -> tuple[IntArr, FloatArr | None]:
    """first_onset に、行動者だけを残した重みを添えて返す (onsets, weights)。

    w=None なら weights も None。重み付き比較（群構成を揃えた比較）で、
    「その活動をした人」の部分集合に対応する重みを取り出すために要る。
    """
    sched = np.asarray(sched)
    has = (sched == act).any(axis=1)
    onsets = np.argmax(sched[has] == act, axis=1).astype(np.int64)
    return onsets, None if w is None else np.asarray(w, dtype=np.float64)[has]


def _weighted_std(x: IntArr, w: FloatArr | None) -> float:
    """重み付き標準偏差（母標準偏差, ddof=0）。w=None なら np.std と一致する。"""
    xf = np.asarray(x, dtype=np.float64)
    if w is None:
        return float(xf.std())
    mean = float(np.average(xf, weights=w))
    return float(np.sqrt(np.average((xf - mean) ** 2, weights=w)))


def onset_comparison(real: IntArr, gen: IntArr, n_act: int, act_names: list[str],
                     acts: list[str] | None = None,
                     w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                     n_boot: int = 60, seed: int = 0) -> pd.DataFrame:
    """★B2: 初回開始時刻の分布距離 (EMD) と標準偏差の比、ノイズ床つき。

    std_ratio > 1 は生成側の開始時刻が過分散 = 個人ごとに時刻がぶれている。

    w_real / w_gen を渡すと重み付き経験分布で EMD・中央値・標準偏差を測る。
    群構成を揃えた比較（実データ側は survey weight、生成側は群構成の再重み付け）
    をするときは必ず渡す。渡さないと両側とも素の標本になり、群構成が違う
    プール同士を比べることになる。
    ★ 床は null_band_2sample の契約どおり「w を再抽出確率として使い、抽出後は
      一様重みで評価」する。観測値（重み付き）とは推定量が違うが、どちらも
      同じ母集団量の一致推定量なので突き合わせて読める。
    """
    targets = acts if acts is not None else act_names
    rows = []
    for name in targets:
        if name not in act_names:
            continue
        a = act_names.index(name)
        orr, w_orr = first_onset_with_weights(real, a, w_real)
        og, w_og = first_onset_with_weights(gen, a, w_gen)
        if len(orr) < 30 or len(og) < 30:
            continue                      # 標本が薄すぎる活動は EMD が読めない
        emd = float(wasserstein_distance(orr.astype(float), og.astype(float),
                                         w_orr, w_og))
        _, _, emd_hi = im.null_band_2sample(
            real, _onset_emd_of(a), n_gen=len(gen), n_boot=n_boot, seed=seed, w=w_real)
        std_real = _weighted_std(orr, w_orr)
        std_gen = _weighted_std(og, w_og)
        rows.append({
            "activity": name, "n_real": len(orr), "n_gen": len(og),
            "onset_median_real": float(im.weighted_quantile(orr.astype(np.float64),
                                                            0.5, w_orr)),
            "onset_median_gen": float(im.weighted_quantile(og.astype(np.float64),
                                                           0.5, w_og)),
            "onset_std_real": std_real, "onset_std_gen": std_gen,
            "onset_std_ratio": float(std_gen / std_real) if std_real > 0 else np.nan,
            "onset_emd": emd, "onset_emd_floor_hi": emd_hi,
            "verdict": "床の内" if emd <= emd_hi else "★床の外",
        })
    return pd.DataFrame(rows)


def _onset_emd_of(a: int) -> Callable[[IntArr, IntArr], float]:
    """活動 a の初回開始時刻 EMD を返す stat_fn（null_band_2sample 用）。"""
    def stat(r: IntArr, g: IntArr) -> float:
        orr, og = first_onset(r, a), first_onset(g, a)
        if len(orr) == 0 or len(og) == 0:
            return float("nan")
        return float(wasserstein_distance(orr.astype(float), og.astype(float)))
    return stat


# ============================================================
# B3. 日跨ぎ境界の破れ
# ============================================================
def wrap_comparison(real: IntArr, gen: IntArr,
                    w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                    n_boot: int = 200, seed: int = 0) -> dict[str, Any]:
    """★B3: wrap_closure_rate（日記の始端と終端で活動が変わる割合）の比較。

    スケジュールは 04:00 起点で、slot 95 (03:45-04:00) と slot 0 (04:00-04:15) は
    同じ人の連続した睡眠の途中にあたる。実 ATUS 平日 val = 0.068。
    生成が大きく上回れば「活動日は環である」構造をモデルが持っていない証拠で、
    巡回パディング・巡回位置符号の根拠になる。
    """
    r = im.wrap_closure_rate(real, w_real)
    g = im.wrap_closure_rate(gen, w_gen)
    lo, _, hi = im.null_band(real, im.wrap_closure_rate, n_boot=n_boot, seed=seed, w=w_real)
    return {"wrap_real": r, "wrap_gen": g,
            "wrap_ratio": float(g / r) if r > 0 else np.nan,
            "ci_lo": lo, "ci_hi": hi,
            "verdict": "床の内" if lo <= g <= hi else "★床の外"}


# ============================================================
# B1 の可視化
# ============================================================
def plot_curves(real: IntArr, gen: IntArr, n_act: int, act_names: list[str],
                acts: list[str] | None = None,
                w_real: FloatArr | None = None, w_gen: FloatArr | None = None,
                path: Any = None, title: str = "") -> None:
    """時刻別行動者率カーブを実 vs 生成で重ねる（00:00 開始表示）。

    B1 の peak_ratio が何を測っているかを目で確認するための図。
    ピークが低く裾が持ち上がっていれば「時刻符号が無い」ことの視覚的な証拠になる。
    """
    import matplotlib.pyplot as plt

    targets = [a for a in (acts if acts is not None else act_names) if a in act_names]
    cr = im.participation_by_slot(real, n_act, w_real)
    cg = im.participation_by_slot(gen, n_act, w_gen)
    roll = im.roll_slots()

    n_col = 2
    n_row = int(np.ceil(len(targets) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(11, 2.6 * n_row), squeeze=False)
    for ax, name in zip(axes.ravel(), targets):
        a = act_names.index(name)
        x = np.arange(NUM_SLOTS)
        ax.plot(x, np.roll(cr[:, a], roll), label="real train", lw=1.6)
        ax.plot(x, np.roll(cg[:, a], roll), label="gen", lw=1.6, ls="--")
        ax.set_title(f"{name}  (peak比 {cg[:, a].max() / cr[:, a].max():.3f})", fontsize=10)
        ax.set_xticks(np.arange(0, NUM_SLOTS + 1, 16))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 4)])
        ax.set_xlim(0, NUM_SLOTS - 1)
        ax.set_ylabel("行動者率")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(title or "時刻別行動者率カーブ (real train vs gen)")
    fig.tight_layout()
    im._save(fig, path)
    plt.show()


# ============================================================
# B4. 位置プローブ（モデル内部）
# ============================================================
def _phase_targets(length: int) -> FloatArr:
    """位置 t の巡回位相 [sin(2πt/L), cos(2πt/L)] (L, 2)。

    絶対位置そのものでなく位相を当てさせるのは、96スロットが 24時間の巡回だから。
    分散が既知（各列 0.5）なので R² が解釈しやすいという実利もある。
    """
    t = np.arange(length, dtype=np.float64)
    ang = 2.0 * np.pi * t / length
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)


def ridge_probe_r2(feats: FloatArr, targets: FloatArr, n_train: int,
                   ridge: float = 1e-3) -> float:
    """特徴 (N, L, C) から位相 (L, 2) をリッジ線形回帰で当てたときの R²。

    N 方向で train/test に分ける。同じ位置の特徴を学習にも評価にも使うと
    R² が楽観的に出るため、評価は未見のサンプルで行う。
    """
    n, length, c = feats.shape
    y = np.broadcast_to(targets[None, :, :], (n, length, 2))
    xtr = feats[:n_train].reshape(-1, c)
    ytr = y[:n_train].reshape(-1, 2)
    xte = feats[n_train:].reshape(-1, c)
    yte = y[n_train:].reshape(-1, 2)

    # バイアス項を明示的に足す（正則化はバイアスに掛けない）
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    g = xtr.T @ xtr + ridge * len(xtr) * np.eye(c)
    beta = np.linalg.solve(g, xtr.T @ (ytr - ytr.mean(0)))
    pred = xte @ beta + ytr.mean(0)

    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def position_probe(model, diffusion, cond_idx, n_samples: int = 512,
                   t_step: int = 999, ridge: float = 1e-3,
                   boundary_width: int = 8, seed: int = 0) -> pd.DataFrame:
    """★B4: 学習済みバックボーンの中間特徴に時刻位相が載っているか。

    ★ 入力は純ノイズ (x_T ~ N(0,I))。内容から時刻を推定できない状況で
      「バックボーン自身がどれだけ位置を持っているか」だけを取り出すため。
      内容つきの x_t で測ると、活動そのものから時刻が読めてしまい交絡する。

    ★ 境界寄り / 内部 に分けて出す。Conv1d のゼロパディング由来の位置情報は
      境界に集中するので、「内部の R² がほぼ 0、境界だけ高い」なら
      位置情報の出所がパディングだと特定できる。
    """
    import torch

    torch.manual_seed(seed)
    device = cond_idx.device
    ci = cond_idx[torch.randint(0, cond_idx.size(0), (n_samples,), device=device)]
    x = torch.randn(n_samples, model.in_channels, NUM_SLOTS, device=device)
    t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
    with torch.no_grad():
        feats = model.features(x, t, ci)

    n_train = n_samples // 2
    rows = []
    for name, h in feats.items():
        f = h.permute(0, 2, 1).float().cpu().numpy().astype(np.float64)   # (N, L, C)
        length = f.shape[1]
        tg = _phase_targets(length)
        bw = max(1, boundary_width * length // NUM_SLOTS)
        edge = np.zeros(length, dtype=bool)
        edge[:bw] = True
        edge[-bw:] = True
        rows.append({
            "feature": name, "resolution": length, "channels": f.shape[2],
            "r2_all": ridge_probe_r2(f, tg, n_train, ridge),
            "r2_boundary": ridge_probe_r2(f[:, edge], tg[edge], n_train, ridge),
            "r2_interior": ridge_probe_r2(f[:, ~edge], tg[~edge], n_train, ridge),
        })
    return pd.DataFrame(rows)


# ============================================================
# B5. 巡回シフト等変性ギャップ（モデル内部）
# ============================================================
def shift_equivariance(model, diffusion, x0, cond_idx, shifts=SHIFTS,
                       t_step: int = 999, seed: int = 0) -> pd.DataFrame:
    """★B5: 時間軸を k スロット巡回シフトしたときの ε 予測のずれ。

        gap(k) = ‖ ε(roll(x, k)) − roll(ε(x), k) ‖ / ‖ ε(x) ‖

    位置情報を持たないモデルはシフト等変になり gap ≈ 0。実データの日内リズムは
    シフト等変でないので、時刻符号を持つモデルなら gap は大きくなる。
    ゼロパディングの境界効果があるため厳密な 0 にはならない。

    ★ 単独では読めないので、同じモデル内の 2 つのヤードスティックを併記する:
        cond_gap   条件（群）を入れ替えたときのずれ = 条件情報の効き
        noise_gap  ノイズを引き直したときのずれ     = 出力の変動幅そのもの
      gap ≪ cond_gap なら、モデルは群の違いには反応するが時刻には反応していない。
    """
    import torch

    torch.manual_seed(seed)
    t = torch.full((x0.size(0),), t_step, device=x0.device, dtype=torch.long)
    eps = torch.randn_like(x0)
    x = diffusion.q_sample(x0, t, eps)

    with torch.no_grad():
        base = model(x, t, cond_idx)
        denom = base.norm().item()
        # ヤードスティック1: 条件を1つずらす（群 d -> d+1）
        shifted_cond = torch.roll(cond_idx, shifts=1, dims=0)
        cond_gap = (model(x, t, shifted_cond) - base).norm().item() / denom
        # ヤードスティック2: ノイズを引き直す
        x2 = diffusion.q_sample(x0, t, torch.randn_like(x0))
        noise_gap = (model(x2, t, cond_idx) - base).norm().item() / denom

        rows = []
        for k in shifts:
            rolled = model(torch.roll(x, shifts=int(k), dims=2), t, cond_idx)
            gap = (rolled - torch.roll(base, shifts=int(k), dims=2)).norm().item() / denom
            rows.append({"shift_slots": int(k), "shift_hours": int(k) * 0.25,
                         "gap": gap, "cond_gap": cond_gap, "noise_gap": noise_gap,
                         "gap_over_cond": gap / cond_gap if cond_gap > 0 else np.nan})
    return pd.DataFrame(rows)


# ============================================================
# 本番診断
# ============================================================
def _load_generated(path: Path) -> tuple[IntArr, IntArr, str]:
    """生成CSVから (スケジュール, 群インデックス, サンプラ名) を読む。"""
    df = pd.read_csv(path)
    sched = df[[f"s{i}" for i in range(NUM_SLOTS)]].to_numpy().astype(np.int64)
    sampler = str(df["sampler"].iloc[0]) if "sampler" in df.columns else "不明(記録なし)"
    return sched, df["group_d"].to_numpy().astype(np.int64), sampler


def run(n_probe: int = 512, n_shift: int = 256, seed: int = 0,
        model_dir: Path = DEFAULT_MODEL_DIR, ckpt: Path | None = None,
        gen_path: Path | None = None, tag: str = "") -> dict[str, pd.DataFrame]:
    """実 ATUS 平日 vs 学習済み AggDDPM で B1-B5 を回し、結果を表で返す。

    model_dir: 診断対象のモデルフォルダ。既定は DDPM_Aggregate。
        DDPM_Aggregate_Tang を渡せば同じ診断を Tang バックボーンに掛けられる
        （両者は load_pretrained / Diffusion / cond_grid / features / in_channels の
        同じ契約を満たす）。出力 CSV はフォルダ名で分ける。
    ckpt: B4/B5 に使う重み。None ならモデルフォルダの既定（load_pretrained の既定引数）。
        同じフォルダの別の重み（DDPM_Aggregate_Simple の _clock 版や日付つきの版）を比べるときに渡す
    gen_path: B1-B3 に使う生成 CSV。None ならモデルフォルダの GEN_SAVE_PATH。
        ★ckpt と gen_path は同じ重みの組を渡すこと。片方だけ差し替えると
          B1-B3 と B4/B5 が別のモデルを測る（例外は出ない）
    tag: 出力 CSV 名の末尾に付ける識別子。ckpt を差し替えるときに結果を分ける
    """
    import torch

    ddpm = load_module("clockdiag_ddpm", model_dir / "model.py")
    suffix = f"_{model_dir.name}" + (f"_{tag}" if tag else "")
    out_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}{suffix}{OUT_CSV.suffix}")
    print(f"診断対象: {model_dir.name}/model.py")
    gen_file = Path(gen_path if gen_path is not None else ddpm.GEN_SAVE_PATH)
    print(f"  生成 CSV: {gen_file}")
    print(f"  重み    : {ckpt if ckpt is not None else '（load_pretrained の既定）'}")
    cond_idx, sched_real, w_real, _ = ddpm.load_data()
    gen, gen_d, sampler = _load_generated(gen_file)

    # 生成側は群一様に作られているので、実データの群構成に重みで合わせる
    d_real = ddpm.cond_to_d(cond_idx)
    w_gen = im.group_reweight(gen_d, w_real, d_real, ddpm.D_GROUPS)

    n_act = ddpm.NUM_ACT
    act_names = ddpm.ACT_NAMES
    print(f"実 ATUS 平日 N={len(sched_real)} / 生成 N={len(gen)} (sampler={sampler}, "
          f"{gen_file.name})")

    print("\n=== B1. 時刻別行動者率カーブ（ピーク鈍化）===")
    b1 = curve_comparison(sched_real, gen, n_act, act_names, w_real, w_gen, seed=seed)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(b1.to_string(index=False))

    print("\n=== B2. 初回開始時刻の分布（個人レベルの時刻のぶれ）===")
    b2 = onset_comparison(sched_real, gen, n_act, act_names,
                          w_real=w_real, w_gen=w_gen, seed=seed)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(b2.to_string(index=False))

    print("\n=== B3. 日跨ぎ境界（活動日は環か）===")
    b3d = wrap_comparison(sched_real, gen, w_real, w_gen, seed=seed)
    print(f"  wrap_closure_rate  実 {b3d['wrap_real']:.4f}  生成 {b3d['wrap_gen']:.4f}  "
          f"比 {b3d['wrap_ratio']:.2f}  床[{b3d['ci_lo']:.4f}, {b3d['ci_hi']:.4f}]  "
          f"{b3d['verdict']}")
    b3 = pd.DataFrame([b3d])

    model = ddpm.load_pretrained(ckpt) if ckpt is not None else ddpm.load_pretrained()
    diffusion = ddpm.Diffusion()
    grid = torch.as_tensor(ddpm.cond_grid(), device=ddpm.DEVICE)

    print("\n=== B4. 位置プローブ（純ノイズ入力・中間特徴から時刻位相を線形復元）===")
    b4 = pd.concat([position_probe(model, diffusion, grid, n_samples=n_probe,
                                   t_step=t, seed=seed).assign(t_step=t)
                    for t in NOISE_LEVELS], ignore_index=True)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(b4.to_string(index=False))

    print("\n=== B5. 巡回シフト等変性ギャップ ===")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sched_real), n_shift, replace=False)
    x0 = ddpm.sched_to_x0(torch.as_tensor(sched_real[idx], device=ddpm.DEVICE))
    ci = torch.as_tensor(cond_idx[idx], device=ddpm.DEVICE)
    b5 = pd.concat([shift_equivariance(model, diffusion, x0, ci, t_step=t,
                                       seed=seed).assign(t_step=t)
                    for t in NOISE_LEVELS], ignore_index=True)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(b5.to_string(index=False))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tagged = [df.assign(diagnostic=tag, backbone=model_dir.name) for tag, df in
              [("B1_curve", b1), ("B2_onset", b2), ("B3_wrap", b3),
               ("B4_probe", b4), ("B5_shift", b5)]]
    pd.concat(tagged, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")
    return {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="生成モデルは時刻を知っているかの診断 (B1-B5)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="診断対象のモデルフォルダ (既定: src/models/DDPM_Aggregate)")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="B4/B5 に使う重み (既定: モデルフォルダの load_pretrained の既定)")
    ap.add_argument("--gen", type=Path, default=None,
                    help="B1-B3 に使う生成 CSV (既定: モデルフォルダの GEN_SAVE_PATH)。--ckpt と同じ重みの生成を渡す")
    ap.add_argument("--tag", type=str, default="",
                    help="出力 CSV 名の末尾に付ける識別子 (例: clock, 20260819)")
    args = ap.parse_args()

    def _abs(p: Path | None) -> Path | None:
        return None if p is None else (p if p.is_absolute() else REPO_ROOT / p)

    model_dir = _abs(args.model_dir)
    assert model_dir is not None
    run(model_dir=model_dir, ckpt=_abs(args.ckpt), gen_path=_abs(args.gen), tag=args.tag)
