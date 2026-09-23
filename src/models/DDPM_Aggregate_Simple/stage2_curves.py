"""
stage2_curves.py
================
**人口加重平均をとった時刻別行動者率**の曲線を作り、教師 `A*` と生成を重ねて描く。

`stage2_select` が出す CSV はスカラーの指標だけで、群ごと・スロットごとの値を残して
いない。誤差が「いつ・どの活動で」出ているかを見るにはここで作り直す。

★重みについて。軸1 の指標（`eval_against` の `rate_*`）は**セルの素平均**であり、
  人口加重ではない。このモジュールが出す曲線は `tgt["pop"]` による**人口加重**で、
  「日本全体としての 1 日」を表す。両者は別の量なので、図と指標を突き合わせるときは
  どちらの重みかを必ず明示すること。重み自体は `stage2_targets.nan_renorm_pop_weights`
  を使い、非公表 NaN セルの扱いを `eval_against` と揃えてある。

    加重平均  A_bar[c,s] = Σ_d W[d,c,s] · A[d,c,s]      （Σ_d W[d,c,s] = 1）

処理の流れ:

```mermaid
flowchart TD
    ASTAR["load_stula_targets<br/>tgt['group_rates_tbl'] (28,12,96)<br/>tgt['pop'] (28,)"]
    CSV["生成サンプル CSV<br/>group_d, s0..s95"]
    NPZ["make_pool が保存した .npz<br/>rates (28, 12*96)"]

    CSV --> LP["load_sample_pool<br/>pool (28, M, 96)"]
    LP --> PR["pool_to_slot_rates<br/>rates (28,12,96)"]
    NPZ --> RS["load_rates_npz<br/>rates (28,12,96)"]

    ATUS["--atus-real<br/>atus_group_rates.load_atus_weekday<br/>→ group_rates (28,12,96)"]

    ASTAR --> W["nan_renorm_pop_weights<br/>W (28,12,96)"]
    W --> WA["weighted_slot_rates<br/>curve (12,96)"]
    PR --> WA
    RS --> WA
    ATUS --> WA
    ASTAR --> WA

    WA --> FIG["plot_curves<br/>4x3 の小倍数図<br/>refs は灰色の破線"]
    WA --> TBL["gap_table<br/>加重ギャップの大きい活動"]
    WA --> BAND["band_closure_table<br/>残差を周期の帯域 BANDS に分け<br/>base_label から埋めた割合を測る"]
    WA --> KEY["key_slot_table<br/>KEY_SLOTS の値を並べる"]
```

★帯域分解（`band_closure_table`）の読み方。
  残差 R0 = 教師 − base（既定は zero-shot）と、生成が動かした量 Δ = 生成 − base を
  時刻軸の rfft で周期の帯域に分け、帯域ごとに「埋めた割合」⟨Δ_b, R0_b⟩ / ‖R0_b‖² を出す。
  帯域は rfft の調和次数で互いに直交に切るので、4 帯域の ‖R0_b‖² の和は ‖R0‖² に一致する。
  「1 日平均（k=0）だけ埋まり、鋭い山（k≥3）が埋まらない」なら、モデルが時刻に依らない
  増減しか作れていない。埋めた割合は内積なので、生成側の MC 雑音（R0 と無相関）は
  期待値で 0 に落ちる。

使い方:
    # 教師 vs zero-shot（GPU 不要。手元のサンプル CSV から作る）
    python src/models/DDPM_Aggregate_Simple/stage2_curves.py \\
        --samples outputs/generated/ddpm_simple_pretrain_samples_20260819.csv=zero-shot \\
        --out outputs/figures/stage2_weighted_slot_rates.png

    # Stage 2 の世代を足す（--samples / --rates は何本でも並べられる）
    python src/models/DDPM_Aggregate_Simple/stage2_curves.py \\
        --samples outputs/generated/ddpm_simple_pretrain_samples_20260819.csv=zero-shot \\
        --rates outputs/generated/stage2_lam0.003_step200_rates.npz=step200 \\
        --out outputs/figures/stage2_weighted_slot_rates.png

    # ATUS 実データを重ね、帯域分解を CSV へ書く（Stage 1 の平滑化と日米差を分けて読む）
    python src/models/DDPM_Aggregate_Simple/stage2_curves.py \\
        --samples outputs/generated/ddpm_simple_pretrain_samples_20260819.csv=zero-shot \\
        --rates outputs/generated/stage2_step200_rates.npz=step200 \\
        --atus-real --out-bands data/processed/aggregates/stage2_band_closure.csv \\
        --out outputs/figures/stage2_weighted_slot_rates_atus.png
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる。既に同じファイルが同じ名前で入っていれば使い回す。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ★torch を読まない。model.py は torch を import するので、描画だけのために
#   読み込むと GPU の無い環境で無駄に重くなる。必要なのは教師と Common 列挙だけ
st: Any = _load("curves_stage2_targets", HERE / "stage2_targets.py")

SLOT_START_HOUR = 4          # スロット 0 は 04:00（model.NUM_SLOTS のコメントと同じ）
SLOT_MINUTES = 15
ACT_NAMES: list[str] = [c.name for c in st.Common]

# 日本語ラベル。図の軸に出すので A_star_weekday.csv の act_ja と同じ語にする
ACT_JA: dict[str, str] = {
    "SLEEP_PERSONAL": "睡眠・身の回りの用事",
    "MEALS": "食事",
    "WORK": "仕事",
    "SCHOOL": "学業",
    "HOUSEWORK": "家事",
    "CAREGIVING": "介護・育児",
    "SHOPPING": "買い物",
    "TRAVEL": "移動",
    "LEISURE_SOCIAL": "余暇・交際",
    "SPORTS": "スポーツ",
    "VOLUNTEER": "ボランティア",
    "OTHER_X": "その他",
}

# 時刻軸 rfft の調和次数 k の半開区間 [lo, hi)。周期は 24h / k
# k=0 は 1 日平均、k>=7 は周期 3.4 時間未満（昼食の 1 時間の山など）
BANDS: dict[str, tuple[int, int]] = {
    "daily_mean": (0, 1),
    "period_24h_12h": (1, 3),
    "period_8h_to_4h": (3, 7),
    "period_under_3.4h": (7, st.NUM_SLOTS // 2 + 1),
}

# 日本側で鋭く、Stage 2 が作るべき山の代表点 (活動名, スロット開始時刻)
KEY_SLOTS: list[tuple[str, float]] = [
    ("MEALS", 7.0), ("MEALS", 12.0), ("MEALS", 19.0), ("WORK", 12.0), ("TRAVEL", 8.0),
]

ATUS_REAL_LABEL = "ATUS実データ"


def load_sample_pool(path: Path) -> npt.NDArray[np.int64]:
    """生成サンプル CSV を群別プールへ畳む, -> (28, M, 96)

    Args:
        path: `group_d` 列と `s0`..`s95` 列を持つ CSV。`model.sanity_check` や
            Stage 1 の生成が書き出す形式

    Returns:
        群別サンプルプール, dtype=int64, (D_GROUPS, M, NUM_SLOTS)。
        M は 1 群あたりの本数

    Raises:
        ValueError: 群ごとの本数が揃っていない場合、または群が 28 揃っていない場合
    """
    df = pd.read_csv(path)
    scols = [f"s{i}" for i in range(st.NUM_SLOTS)]
    d = df["group_d"].to_numpy(dtype=np.int64)
    sched = df[scols].to_numpy(dtype=np.int64)
    counts = np.bincount(d, minlength=st.D_GROUPS)
    if len(counts) != st.D_GROUPS or not np.all(counts == counts[0]) or counts[0] == 0:
        raise ValueError(
            f"群ごとの本数が揃っていない: {counts.tolist()}\n"
            f"       群別プールへ畳めないので、生成をやり直すこと")
    order = np.argsort(d, kind="stable")
    return sched[order].reshape(st.D_GROUPS, int(counts[0]), st.NUM_SLOTS)


def pool_to_slot_rates(pool: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """群別プールを時刻別行動者率へ畳む, -> (28, 12, 96)

    各スロットで 12 活動の割合を足すと 1 になる（教師 `A*` と同じ統計量）。
    「1 日に 1 回以上その活動をしたか」の**日次**行動者率
    （`stage2_published.participation_from_pool`）とは別物である。

    Args:
        pool: 群別サンプルプール, dtype=int64, (D_GROUPS, M, NUM_SLOTS)

    Returns:
        時刻別行動者率, dtype=float64, (D_GROUPS, NUM_COMMON, NUM_SLOTS)
    """
    m = pool.shape[1]
    out = np.zeros((st.D_GROUPS, st.NUM_COMMON, st.NUM_SLOTS), dtype=np.float64)
    for c in range(st.NUM_COMMON):
        out[:, c, :] = (pool == c).sum(axis=1) / m
    return out


def load_rates_npz(path: Path) -> npt.NDArray[np.float64]:
    """`make_pool` が保存した .npz から時刻別行動者率を読む, -> (28, 12, 96)

    Args:
        path: `rates` 配列を持つ .npz。(28, 12*96) act-major か (28, 12, 96)

    Returns:
        時刻別行動者率, dtype=float64, (D_GROUPS, NUM_COMMON, NUM_SLOTS)

    Raises:
        ValueError: `rates` キーが無い場合
    """
    with np.load(path) as z:
        if "rates" not in z:
            raise ValueError(f"'rates' が無い: {path}（キー: {list(z.keys())}）")
        r = np.asarray(z["rates"], dtype=np.float64)
    return r.reshape(st.D_GROUPS, st.NUM_COMMON, st.NUM_SLOTS)


def weighted_slot_rates(rates: npt.NDArray[np.float64],
                        tgt: dict) -> npt.NDArray[np.float64]:
    """人口加重で 28 群を 1 本へ畳む, -> (12, 96)

    Note:
        ★重みは `eval_against` の `dev_*` と同じ `nan_renorm_pop_weights` を使う。
          セルごとに非公表 NaN の群を除いて再正規化するので、教師と生成で
          「人口平均」の定義が揃う。

    Args:
        rates: 時刻別行動者率, (D_GROUPS, NUM_COMMON, NUM_SLOTS)
        tgt: `stage2_targets.load_stula_targets` の戻り値

    Returns:
        人口加重平均の時刻別行動者率, dtype=float64, (NUM_COMMON, NUM_SLOTS)
    """
    grp_tbl = tgt["group_rates_tbl"]
    pop = np.asarray(tgt["pop"], dtype=np.float64).reshape(st.D_GROUPS)
    pi_d = pop / pop.sum()
    w = st.nan_renorm_pop_weights(grp_tbl, pi_d)              # (28,12,96)
    return np.asarray(np.nansum(rates * w, axis=0), dtype=np.float64)


def slot_hours() -> npt.NDArray[np.float64]:
    """スロット中心の時刻（04:00 起点の 0-36 時間軸）, -> (96,)"""
    return SLOT_START_HOUR + (np.arange(st.NUM_SLOTS) + 0.5) * SLOT_MINUTES / 60.0


def gap_table(curves: dict[str, npt.NDArray[np.float64]],
              teacher: npt.NDArray[np.float64]) -> pd.DataFrame:
    """活動ごとに、教師からの加重ギャップを要約する。

    Args:
        curves: ラベル -> 人口加重曲線 (NUM_COMMON, NUM_SLOTS)
        teacher: 教師の人口加重曲線, (NUM_COMMON, NUM_SLOTS)

    Returns:
        活動 × ラベルごとに mae / max_abs / peak_hour（最大ギャップの時刻）を持つ
        DataFrame。mae の降順
    """
    rows: list[dict[str, Any]] = []
    hours = slot_hours()
    for label, cur in curves.items():
        diff = cur - teacher
        for c, name in enumerate(ACT_NAMES):
            s = int(np.argmax(np.abs(diff[c])))
            rows.append({"activity": name, "label": label,
                         "mae": float(np.abs(diff[c]).mean()),
                         "max_abs": float(np.abs(diff[c]).max()),
                         "signed_at_max": float(diff[c][s]),
                         "peak_hour": float(hours[s]),
                         "teacher_mean": float(teacher[c].mean()),
                         "gen_mean": float(cur[c].mean())})
    return pd.DataFrame(rows).sort_values(["mae"], ascending=False).reset_index(drop=True)


def atus_real_curve(tgt: dict) -> npt.NDArray[np.float64]:
    """ATUS 平日の実データを教師と同じ人口加重で 1 本へ畳む, -> (12, 96)

    Note:
        ★群内は ATUS の調査ウェイト TUFINLWGT、群間は日本の人口 `tgt["pop"]` で重み付ける。
          教師・生成と同じ `weighted_slot_rates` を通すので、3 本の曲線の差は
          「群構成の違い」ではなく「群の中の時刻別行動者率の違い」だけになる。
        ★zero-shot との差が Stage 1 の再現誤差、教師との差が日米差にあたる。

    Args:
        tgt: `stage2_targets.load_stula_targets` の戻り値

    Returns:
        人口加重平均の時刻別行動者率, dtype=float64, (NUM_COMMON, NUM_SLOTS)
    """
    agr: Any = _load("curves_atus_group_rates", REPO_ROOT / "src" / "eval" / "atus_group_rates.py")
    sched, groups, w = agr.load_atus_weekday()
    return weighted_slot_rates(agr.group_rates(sched, groups, w), tgt)


def band_component(x: npt.NDArray[np.float64], lo: int, hi: int) -> npt.NDArray[np.float64]:
    """時刻軸（最後の軸）の rfft で調和次数 k ∈ [lo, hi) だけを残した成分を返す

    Args:
        x: 時刻軸を最後に持つ配列, (..., NUM_SLOTS)
        lo: 残す調和次数の下端（含む）
        hi: 残す調和次数の上端（含まない）

    Returns:
        x と同じ形の帯域成分, dtype=float64。BANDS の全帯域を足すと x に戻る
    """
    spec = np.fft.rfft(x, axis=-1)
    keep = np.zeros_like(spec)
    keep[..., lo:hi] = spec[..., lo:hi]
    return np.asarray(np.fft.irfft(keep, n=x.shape[-1], axis=-1), dtype=np.float64)


def band_closure_table(curves: dict[str, npt.NDArray[np.float64]],
                       teacher: npt.NDArray[np.float64],
                       base_label: str) -> pd.DataFrame:
    """残差を周期の帯域に分け、base_label から各生成が埋めた割合を測る

    R0 = 教師 − base, Δ = 生成 − base, R1 = 教師 − 生成 として、帯域 b ごとに

        residual_share = ‖R0_b‖² / ‖R0‖²       残差のうちその帯域にある割合
        closed         = ⟨Δ_b, R0_b⟩ / ‖R0_b‖²   その帯域の残差を埋めた割合（1 で完全, 負は逆向き）
        remaining      = ‖R1_b‖² / ‖R0_b‖²       その帯域に残った残差の割合

    Note:
        ★closed は内積なので、生成の MC 雑音（R0 と無相関）は期待値で 0 に落ちる。
          remaining は雑音の二乗を含むので、雑音の多い帯域（k が大きい側）では 1 を超えうる。
        ★activity="ALL" の行は 12 活動をまとめた値。活動ごとの行は ‖R0‖² がその活動の
          分だけなので、活動間で residual_share の分母が違う。

    Args:
        curves: ラベル -> 人口加重曲線 (NUM_COMMON, NUM_SLOTS)。base_label を含むこと
        teacher: 教師の人口加重曲線, (NUM_COMMON, NUM_SLOTS)
        base_label: 起点にする曲線のラベル（通常は "zero-shot"）

    Returns:
        列 label / activity / band / residual_share / closed / remaining の DataFrame。
        band="all" の行は帯域を分けない全体の値

    Raises:
        KeyError: base_label が curves に無い場合
    """
    base = curves[base_label]
    r0 = teacher - base
    selections: list[tuple[str, list[int]]] = [("ALL", list(range(st.NUM_COMMON)))]
    selections += [(name, [c]) for c, name in enumerate(ACT_NAMES)]
    band_items = [*BANDS.items(), ("all", (0, st.NUM_SLOTS // 2 + 1))]

    rows: list[dict[str, Any]] = []
    for label, cur in curves.items():
        if label == base_label:
            continue
        delta, r1 = cur - base, teacher - cur
        for act, idx in selections:
            total0 = float((r0[idx] ** 2).sum())
            for band, (lo, hi) in band_items:
                r0_b = band_component(r0[idx], lo, hi)
                e0 = float((r0_b ** 2).sum())
                if e0 == 0.0:
                    closed = remaining = float("nan")
                else:
                    closed = float((band_component(delta[idx], lo, hi) * r0_b).sum()) / e0
                    remaining = float((band_component(r1[idx], lo, hi) ** 2).sum()) / e0
                rows.append({"label": label, "activity": act, "band": band,
                             "residual_share": e0 / total0 if total0 > 0 else float("nan"),
                             "closed": closed, "remaining": remaining})
    return pd.DataFrame(rows)


def key_slot_table(curves: dict[str, npt.NDArray[np.float64]]) -> pd.DataFrame:
    """KEY_SLOTS の各点で曲線の値を横に並べる

    Args:
        curves: ラベル -> 人口加重曲線 (NUM_COMMON, NUM_SLOTS)。教師も含めて渡す

    Returns:
        行 = (activity, time)、列 = ラベルの DataFrame。time は "HH:MM"（スロットの開始時刻）
    """
    rows: list[dict[str, Any]] = []
    for act, hour in KEY_SLOTS:
        c = ACT_NAMES.index(act)
        s = int(round((hour - SLOT_START_HOUR) * 60 / SLOT_MINUTES))
        row: dict[str, Any] = {"activity": act, "time": f"{int(hour):02d}:{int(hour % 1 * 60):02d}"}
        row.update({label: float(cur[c, s]) for label, cur in curves.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(teacher: npt.NDArray[np.float64],
                curves: dict[str, npt.NDArray[np.float64]],
                out_path: Path, title: str = "",
                refs: dict[str, npt.NDArray[np.float64]] | None = None) -> None:
    """12 活動の人口加重時刻別行動者率を 4x3 の小倍数で描く。

    Note:
        ★活動ごとに y 軸を独立させる。睡眠は 1.0 近くまで行き、ボランティアは
          0.005 程度なので、共通軸にすると小さい活動が潰れて読めない。
        ★x 軸は 04:00 起点の 0-36 時間軸。24 時で折り返さない（深夜が左端へ
          飛んで曲線が割れるため）。
        ★教師を黒の太線、生成を細線にする。どれが正解かを一目で分かるようにする。
        ★refs（ATUS 実データなど、生成でも教師でもない参照）は灰色の破線にする。

    Args:
        teacher: 教師の人口加重曲線, (NUM_COMMON, NUM_SLOTS)
        curves: ラベル -> 生成の人口加重曲線, 各 (NUM_COMMON, NUM_SLOTS)
        out_path: 保存先 .png
        title: 図全体のタイトル, default=""
        refs: ラベル -> 参照の人口加重曲線, 各 (NUM_COMMON, NUM_SLOTS), default=None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 日本語フォント。無ければ英字名のままにして文字化けを避ける
    jp = [f for f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic",
                      "IPAexGothic", "Noto Sans CJK JP")
          if any(f == fn.name for fn in font_manager.fontManager.ttflist)]
    use_ja = bool(jp)
    if use_ja:
        plt.rcParams["font.family"] = jp[0]
    plt.rcParams["axes.unicode_minus"] = False

    hours = slot_hours()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
    # ★色は明示的に持つ。get_cmap(...).colors は Colormap 基底クラスに無く、
    #   型チェッカが通らない（ListedColormap にしかない属性）
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")

    for c, name in enumerate(ACT_NAMES):
        ax = axes[c // 3][c % 3]
        ax.plot(hours, teacher[c], color="black", lw=2.2, label="教師 A*" if use_ja else "teacher A*")
        for label, ref in (refs or {}).items():
            ax.plot(hours, ref[c], color="#7f7f7f", lw=1.4, ls="--", label=label)
        for k, (label, cur) in enumerate(curves.items()):
            ax.plot(hours, cur[c], color=colors[k % len(colors)], lw=1.4, label=label)
        ax.set_title(f"{ACT_JA[name] if use_ja else name}  ({name})", fontsize=10)
        ax.set_xlim(hours[0], hours[-1])
        ax.set_xticks(np.arange(4, 29, 4))
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(3):
        axes[3][j].set_xlabel("時刻（04:00 起点, 28 = 翌 04:00）" if use_ja
                              else "hour (from 04:00)", fontsize=9)
    for i in range(4):
        axes[i][0].set_ylabel("人口加重 行動者率" if use_ja
                              else "pop-weighted rate", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_spec(spec: str, kind: str) -> tuple[Path, str]:
    """`path=label` を分解する。`=label` が無ければファイル名の語幹をラベルにする。"""
    if "=" in spec:
        p, label = spec.rsplit("=", 1)
        return Path(p), label
    p = Path(spec)
    return p, f"{kind}:{p.stem}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", action="append", default=[], metavar="PATH[=LABEL]",
                    help="生成サンプル CSV（group_d, s0..s95）。複数指定可")
    ap.add_argument("--rates", action="append", default=[], metavar="PATH[=LABEL]",
                    help="make_pool が保存した .npz（rates キー）。複数指定可")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs/figures/stage2_weighted_slot_rates.png")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="加重曲線そのものを CSV へ出す（label, activity, slot, rate）")
    ap.add_argument("--title", type=str, default="人口加重平均の時刻別行動者率（全28群）")
    ap.add_argument("--atus-real", action="store_true",
                    help="ATUS 平日の実データを同じ人口加重で畳み、参照として重ねる")
    ap.add_argument("--base-label", type=str, default="zero-shot",
                    help="帯域分解で起点にする曲線のラベル。無ければ帯域分解を飛ばす")
    ap.add_argument("--out-bands", type=Path, default=None,
                    help="帯域分解の表を CSV へ出す（label, activity, band, residual_share, closed, remaining）")
    args = ap.parse_args()

    if not args.samples and not args.rates:
        ap.error("--samples か --rates を 1 つ以上指定すること")

    tgt = st.load_stula_targets()
    teacher = weighted_slot_rates(np.asarray(tgt["group_rates_tbl"], dtype=np.float64), tgt)

    refs: dict[str, npt.NDArray[np.float64]] = {}
    if args.atus_real:
        refs[ATUS_REAL_LABEL] = atus_real_curve(tgt)
        print(f"[curves] {ATUS_REAL_LABEL:<12} <- atus_group_rates（平日, TUFINLWGT）")

    curves: dict[str, npt.NDArray[np.float64]] = {}
    for spec in args.samples:
        path, label = _parse_spec(spec, "samples")
        rates = pool_to_slot_rates(load_sample_pool(path))
        curves[label] = weighted_slot_rates(rates, tgt)
        print(f"[curves] {label:<12} <- {path.name}")
    for spec in args.rates:
        path, label = _parse_spec(spec, "rates")
        curves[label] = weighted_slot_rates(load_rates_npz(path), tgt)
        print(f"[curves] {label:<12} <- {path.name}")

    # 各スロットで 12 活動の和が 1 になっているかの確認（教師と同じ統計量である証拠）
    for label, cur in [("teacher", teacher), *refs.items(), *curves.items()]:
        s = cur.sum(axis=0)
        print(f"[check] {label:<12} 各スロットの活動和: "
              f"min={s.min():.6f} max={s.max():.6f}")

    tbl = gap_table(curves, teacher)
    print("\n=== 教師からの加重ギャップ（mae 降順）===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tbl.to_string(index=False,
                            formatters={"mae": "{:.5f}".format,
                                        "max_abs": "{:.5f}".format,
                                        "signed_at_max": "{:+.5f}".format,
                                        "peak_hour": "{:.2f}".format,
                                        "teacher_mean": "{:.5f}".format,
                                        "gen_mean": "{:.5f}".format}))

    print("\n=== 代表スロットの値（人口加重, スロット開始時刻）===")
    key_tbl = key_slot_table({"teacher": teacher, **refs, **curves})
    print(key_tbl.round(3).to_string(index=False))

    if args.base_label in curves:
        bands = band_closure_table(curves, teacher, args.base_label)
        print(f"\n=== 帯域別に埋めた割合（起点 {args.base_label}, 12 活動まとめ）===")
        print(bands[bands.activity == "ALL"].round(3).to_string(index=False))
        print(f"\n=== 帯域別に埋めた割合 closed（活動別, 起点 {args.base_label}）===")
        by_act = bands[bands.activity != "ALL"].pivot_table(
            index=["label", "activity"], columns="band", values="closed", sort=False)
        print(by_act[[*BANDS, "all"]].round(3).to_string())
        if args.out_bands is not None:
            args.out_bands.parent.mkdir(parents=True, exist_ok=True)
            bands.to_csv(args.out_bands, index=False)
            print(f"\n[curves] 帯域分解を書いた: {args.out_bands}")
    else:
        print(f"\n[curves] 起点 '{args.base_label}' が曲線に無いので帯域分解を飛ばす")

    if args.out_csv is not None:
        rows = []
        for label, cur in [("teacher", teacher), *refs.items(), *curves.items()]:
            for c, name in enumerate(ACT_NAMES):
                for s in range(st.NUM_SLOTS):
                    rows.append({"label": label, "activity": name, "slot": s,
                                 "hour": float(slot_hours()[s]), "rate": float(cur[c, s])})
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\n[curves] 曲線を書いた: {args.out_csv}")

    plot_curves(teacher, curves, args.out, args.title, refs=refs)
    print(f"[curves] 図を書いた: {args.out}")


if __name__ == "__main__":
    main()
