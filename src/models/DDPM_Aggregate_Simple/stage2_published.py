"""
stage2_published.py
===================
**非教師**の公表統計（生活時間編）との突合（Stage2_design.md §9.7 / 実装項目 17）

★`stage2_targets.py` と混同しないこと。あちらは**教師** `A*`（時間帯編の時刻別行動者率）
  を作る。Stage 2 の損失はそれ自体なので、`A*` への適合は循環している（§9.2）。
  こちらが読むのは**日次行動者率**（生活時間編 第70-3表）で、**教師が縛らない量**である。
  日本の公表値に照らせる唯一の非循環な軸であり、`reference = stula_published` として
  軸2 に並ぶ（§9.8）。

```mermaid
flowchart TD
    X["原表 第70-3表<br/>行動者率(15歳以上)"] -->|parse_timeuse.py| C["timeuse_participation.csv<br/>1行1層 × a01..a20"]
    C -->|load_participation| P20["part20 (28, 20)<br/>社基調20分類の日次行動者率"]
    P20 -->|common12_bounds| B["lo, hi (28, 12)<br/>common12 の上下限"]
    POOL["生成プール (28, M, 96)<br/>common12 ラベル"] -->|participation_from_pool| G["part_gen (28, 12)"]
    B --> E["eval_participation<br/>exact 7 は誤差 / union 5 は区間内か"]
    G --> E
```

★**日次行動者率は分類の和で足せない。**

    P(∃s: x_s ∈ C) ≠ Σ_{c∈C} P(∃s: x_s = c)

同じ人が 1 日のうちに `12_テレビ` と `13_休養` の両方をすれば、和は二重に数える。
`crosswalk_atus_stula.stula_to_common` は単純和なので、**時刻別**行動者率には正しいが
**日次**行動者率には誤りである。common12 と 1 対 1 なのは 7 活動だけで、残る 5 活動は
上下限でしか言えない：

    max_c P_c  ≤  P_∪  ≤  min(1, Σ_c P_c)

下限は「最も多い 1 つの活動をした人は必ず和集合もした」、上限は「誰も重複しなかった」
場合である。生成側がこの区間の外に出たら**確実な不一致**であり、中に入っていても
一致の証明にはならない（区間が広いため）。7 活動の厳密な突合の方が強い証拠である。

使い方:
    pub = load_participation()                      # {"part20": (28,20), "pop": (28,), ...}
    lo, hi = common12_bounds(pub["part20"])
    part_gen = participation_from_pool(pool)        # (28, 12)
    rows = eval_participation(part_gen, pub)
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))
from crosswalk_atus_stula import Common, NUM_COMMON, STULA_TO_COMMON  # noqa: E402


def _load(name: str, path: Path):
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


st: Any = _load("simple_stage2_targets", HERE / "stage2_targets.py")
# 起床・就寝の規則は src/eval 側の実装が唯一のものである（§9.7）
sdt: Any = _load("stula_derived_times",
                 REPO_ROOT / "src" / "eval" / "stula_derived_times.py")

STULA_DIR = REPO_ROOT / "data" / "processed" / "stula"
DEFAULT_TABLE = "timeuse_participation"
N_STULA_ACT = 20
NUM_SLOTS = 96

# 第70-3表（生活時間編）の軸の値。時間帯編（`stage2_targets`）と符号が同じなのは
# 男女と就業だけで、年齢は**この表が既に 10 歳 7 区分**である点が違う（時間帯編は 5 歳 15 区分）。
DAYTYPE_WEEKDAY = "2_平日"
REGION_JAPAN = "00_全国"
HEALTH_TOTAL = "0_総数"

# ★平均時刻編は**軸の値の書き方が違う**。ここを共用にしてはいけない。
#   生活時間編 : 曜日 `2_平日`（`1_週全体` が先にある）/ 地域 `00_全国`
#   平均時刻編 : 曜日 `1_平日`（週全体が無い）  / 地域 `0_全国`（ゼロ 1 桁）
#   取り違えても例外は出ず、フィルタが 0 行になって「28 群が揃わない」で初めて気づく。
MT_DAYTYPE_WEEKDAY = "1_平日"
MT_REGION_JAPAN = "0_全国"
LIFE_STAGE_TOTAL = "0_総数"
MEANTIME_TABLES = {"wake": "meantime_wake", "bed": "meantime_bed"}
# 平均時刻の妥当な範囲（時）。起床は 0〜12 時、就寝は 0〜36 時の軸に載る
MEANTIME_RANGE = {"wake": (0.0, 12.0), "bed": (17.0, 36.0)}
GENDER_CODE = {"1_男": 0, "2_女": 1}
EMPLOY_CODE = {"1_有業者": 1, "2_無業者": 0}     # stage2_targets と同じ向き（有業=1）
# ★モデルの 7 区分とそのまま 1 対 1 で合う。畳み込みも人口加重も要らない（§9.7）
AGE7_CODE = {f"{i}_": i - 1 for i in range(1, 8)}

# common12 -> 社基調20分類のコード。1 つなら厳密、複数なら和集合で上下限になる
COMMON_TO_STULA: dict[int, list[str]] = {int(c): [] for c in Common}
for _code, _com in sorted(STULA_TO_COMMON.items()):
    COMMON_TO_STULA[int(_com)].append(_code)
# 1 対 1 で突き合わせられる common12（§9.7 の 7 活動）
EXACT_COMMON = tuple(c for c, codes in sorted(COMMON_TO_STULA.items()) if len(codes) == 1)
UNION_COMMON = tuple(c for c, codes in sorted(COMMON_TO_STULA.items()) if len(codes) > 1)


def load_participation(name: str = DEFAULT_TABLE,
                       daytype: str = DAYTYPE_WEEKDAY) -> dict:
    """生活時間編の日次行動者率を 28 群へ読む。

    Note:
        ★健康は `0_総数` で周辺化する。モデルは健康状態を条件に持たないので、
          健康別の行を使うと群の定義がずれる。`就業 × 健康` は完全交差（9×6）で
          総数行が全就業区分にあるため、素直に取れる。
        ★年齢の畳み込みが要らない。この表は既に 10 歳 7 区分で、`AGE15_TO_7` の像と
          1 対 1 である。時間帯編で必要だった人口加重の畳み込み（§3.3 P6）は不要。
        ★率は [0,1] へ直して返す。原表は % である。

    Args:
        name: csv ファイル名（拡張子なし）, default=DEFAULT_TABLE
        daytype: 曜日の値, default=DAYTYPE_WEEKDAY（"2_平日"）

    Returns:
        dict。part20 (28,20) 日次行動者率 [0,1] / pop (28,) 推定人口（千人） /
        n_layer (28,) 実回答者数 / daytype

    Raises:
        FileNotFoundError: csv が無い場合（`parse_timeuse.py` を流すこと）
        ValueError: 28 群が揃わない場合
    """
    path = STULA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が無い。"
            "python src/common/preprocess/stula/parse_timeuse.py を流すこと")
    df = pd.read_csv(path)
    sub = cast(pd.DataFrame, df[(df["daytype"] == daytype)
                                & (df["region"] == REGION_JAPAN)
                                & (df["health"] == HEALTH_TOTAL)
                                & (df["gender"].isin(GENDER_CODE))
                                & (df["employment"].isin(EMPLOY_CODE))]).copy()
    sub["a7"] = cast("pd.Series", sub["age_class"]).str[:2].map(AGE7_CODE)
    sub = cast(pd.DataFrame, sub.dropna(subset=["a7"]))
    if len(sub) != st.D_GROUPS:
        raise ValueError(
            f"28 群が揃わない: {len(sub)} 行（daytype={daytype}）\n"
            f"       軸の値が表と食い違っていないか確認すること")

    act_cols = [f"a{i:02d}" for i in range(1, N_STULA_ACT + 1)]
    # ★群インデックスは `st.d_index` に出させる。ここで g*14+a*2+e と書き下すと
    #   群定義が 2 箇所に散り、片方だけ直したときに黙ってずれる
    g_i = cast("pd.Series", sub["gender"]).map(GENDER_CODE).to_numpy(dtype=np.int64)
    e_i = cast("pd.Series", sub["employment"]).map(EMPLOY_CODE).to_numpy(dtype=np.int64)
    a_i = sub["a7"].to_numpy(dtype=np.int64)
    d_i = np.array([st.d_index(int(g), int(a), int(e))
                    for g, a, e in zip(g_i, a_i, e_i)], dtype=np.int64)
    if len(set(d_i.tolist())) != st.D_GROUPS:
        raise ValueError(f"群インデックスが重複している: {sorted(d_i.tolist())}")

    part20 = np.full((st.D_GROUPS, N_STULA_ACT), np.nan)
    pop = np.zeros(st.D_GROUPS)
    n_layer = np.zeros(st.D_GROUPS)
    part20[d_i] = sub[act_cols].to_numpy(dtype=np.float64) / 100.0
    pop[d_i] = sub[["population_k"]].to_numpy(dtype=np.float64).ravel()
    n_layer[d_i] = sub[["sample_size"]].to_numpy(dtype=np.float64).ravel()
    return {"part20": part20, "pop": pop, "n_layer": n_layer, "daytype": daytype}


def common12_bounds(part20: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """社基調20分類の日次行動者率 -> common12 の上下限, (28,20) -> ((28,12), (28,12))

    Note:
        ★1 対 1 の 7 活動では lo == hi になる。そこだけが厳密な突合であり、
          §9.7 で「実質的に独立な非循環の証拠」と位置づけているのもこの 7 つである。
        ★和集合の 5 活動は `max_c P_c ≤ P_∪ ≤ min(1, Σ_c P_c)`。下限は最も多い 1 つを
          した人が和集合にも入ることから、上限は誰も重複しない場合から出る。
          **区間の中に入っていても一致の証明にはならない。**外に出たときだけ
          「確実な不一致」と言える。

    Args:
        part20: 日次行動者率 [0,1], (D, 20)

    Returns:
        (lo, hi) の組。各 (D, 12), dtype=float64。NaN は伝播する
    """
    p = np.asarray(part20, dtype=np.float64)
    d = p.shape[0]
    lo = np.full((d, NUM_COMMON), np.nan)
    hi = np.full((d, NUM_COMMON), np.nan)
    for c, codes in sorted(COMMON_TO_STULA.items()):
        cols = [int(x) - 1 for x in codes]            # '01' -> 列 0
        block = p[:, cols]
        lo[:, c] = np.max(block, axis=1)
        hi[:, c] = np.minimum(1.0, np.sum(block, axis=1))
    return lo, hi


def participation_from_pool(pool: npt.NDArray[np.int64]) -> np.ndarray:
    """生成プール -> 群別の日次行動者率, (D, M, 96) -> (D, 12)

    Note:
        ★`sm.pool_to_rates`（時刻別行動者率）とは別物である。あちらは各スロットで
          活動方向の和が 1 になるが、こちらは「1 日に 1 回以上したか」なので
          和は 1 を大きく超える。混同すると §9.8 が防ごうとしている取り違えになる。

    Args:
        pool: 群別サンプルプール, dtype=int64, (D, M, NUM_SLOTS)。値は common12 ラベル

    Returns:
        群別の日次行動者率 [0,1], dtype=float64, (D, 12)
    """
    arr = np.asarray(pool)
    d, m, _ = arr.shape
    out = np.zeros((d, NUM_COMMON))
    for c in range(NUM_COMMON):
        out[:, c] = (arr == c).any(axis=2).sum(axis=1) / m
    return out


def eval_participation(part_gen: np.ndarray, pub: dict) -> dict:
    """生成の日次行動者率を公表値へ突き合わせる。

    Note:
        ★重みは `weight_basis = stula_pop`（§9.7）。日本の公表値と比べる行なので、
          群を日本の公表人口で重み付けする。ATUS の群構成で重み付けすると
          群構成の差が行動の差に化ける。
        ★厳密な 7 活動と区間の 5 活動を**分けて返す**。混ぜると「区間に入っている」
          だけの弱い証拠が、厳密一致と同じ重みで読まれてしまう。

    Args:
        part_gen: 生成の日次行動者率 [0,1], (D, 12)
        pub: `load_participation` の戻り値

    Returns:
        指標の dict。
        exact_mae / exact_max_abs / exact_n : 1 対 1 の 7 活動での誤差（人口加重）
        union_outside_rate / union_max_gap / union_n : 和集合 5 活動で区間の外に
            出た割合と、外れ幅の最大（人口加重、区間内なら 0）
    """
    lo, hi = common12_bounds(pub["part20"])
    g = np.asarray(part_gen, dtype=np.float64)
    pop = np.asarray(pub["pop"], dtype=np.float64)
    w = pop / pop.sum()

    ex = list(EXACT_COMMON)
    err = np.abs(g[:, ex] - lo[:, ex])                      # 1対1 では lo == hi
    m_ex = ~np.isnan(err)
    wex = np.broadcast_to(w[:, None], err.shape)
    out = {
        "exact_mae": float((err[m_ex] * wex[m_ex]).sum() / wex[m_ex].sum()),
        "exact_max_abs": float(np.nanmax(err)),
        "exact_n": int(m_ex.sum()),
    }

    un = list(UNION_COMMON)
    gap = np.maximum(lo[:, un] - g[:, un], g[:, un] - hi[:, un])   # 区間内なら負
    gap = np.maximum(gap, 0.0)
    m_un = ~np.isnan(gap)
    wun = np.broadcast_to(w[:, None], gap.shape)
    out["union_outside_rate"] = float(
        ((gap[m_un] > 0) * wun[m_un]).sum() / wun[m_un].sum())
    out["union_max_gap"] = float(np.nanmax(gap))
    out["union_n"] = int(m_un.sum())
    return out


# ============================================================
# 派生時刻（起床・就寝）— 平均時刻編との突合（§9.7 の測る量 2）
# ============================================================
def load_mean_times(daytype: str = MT_DAYTYPE_WEEKDAY) -> dict:
    """平均時刻編の起床・就寝を 28 群へ読む。

    Note:
        ★年齢は 5 歳 15 区分なので 7 区分へ畳む。畳み方は**行動者数での加重平均**で、
          人口加重ではない。公表の平均時刻は「時刻が定まった人」の平均なので、
          層の平均を束ねる重みも行動者数（＝推定人口 × 行動者率）でなければ一致しない。
        ★`行動者率` も 28 群へ畳んで返す。これは「起床・就寝の時刻が定まった人の割合」で、
          生成側の不詳率と比べる相手である。実測は 97〜100% 程度あり、生成側で
          不詳が多ければ平均が合っていても睡眠の作りが壊れていることになる。
        ★ライフステージは `0_総数` で周辺化する。モデルが条件に持たない軸である。

    Args:
        daytype: 曜日の値, default=MT_DAYTYPE_WEEKDAY（"1_平日"）。
            ★生活時間編の `2_平日` とは書き方が違う

    Returns:
        dict。wake_hour / bed_hour (28,) 平均時刻（時、就寝は 0〜36 時の軸）、
        wake_actor_rate / bed_actor_rate (28,) 時刻が定まった割合 [0,1]、
        pop (28,) 推定人口（千人）、daytype

    Raises:
        FileNotFoundError: csv が無い場合（`parse_mean_time.py` を流すこと）
        ValueError: 60 層（2性 × 2就業 × 5歳15区分）が揃わない場合、または
            平均時刻が想定の範囲を外れる場合
    """
    out: dict[str, Any] = {"daytype": daytype}
    pop_out = np.zeros(st.D_GROUPS)
    for kind, name in MEANTIME_TABLES.items():
        path = STULA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} が無い。"
                "python src/common/preprocess/stula/parse_mean_time.py を流すこと")
        df = pd.read_csv(path)
        sub = cast(pd.DataFrame, df[(df["daytype"] == daytype)
                                   & (df["region"] == MT_REGION_JAPAN)
                                   & (df["life_stage"] == LIFE_STAGE_TOTAL)
                                   & (df["gender"].isin(GENDER_CODE))
                                   & (df["employment"].isin(EMPLOY_CODE))]).copy()
        sub["a15"] = cast("pd.Series", sub["age_class"]).str[:2].map(
            lambda c: int(c) if str(c).isdigit() and str(c) != "00" else np.nan)
        sub = cast(pd.DataFrame, sub.dropna(subset=["a15"]))
        n_need = st.N_G * st.N_E * 15
        if len(sub) != n_need:
            raise ValueError(
                f"{name}: 層が {len(sub)} 行で {n_need} 行に足りない"
                f"（daytype={daytype} / region={MT_REGION_JAPAN}）\n"
                f"       軸の値の書き方が表と食い違っていないか確認すること")

        g_i = cast("pd.Series", sub["gender"]).map(GENDER_CODE).to_numpy(dtype=np.int64)
        e_i = cast("pd.Series", sub["employment"]).map(EMPLOY_CODE).to_numpy(dtype=np.int64)
        a7_i = np.array([st.AGE15_TO_7[int(a)] for a in sub["a15"].to_numpy()],
                        dtype=np.int64)
        d_i = np.array([st.d_index(int(g), int(a), int(e))
                        for g, a, e in zip(g_i, a7_i, e_i)], dtype=np.int64)

        pop15 = sub[["population_k"]].to_numpy(dtype=np.float64).ravel()
        rate15 = sub[["actor_rate"]].to_numpy(dtype=np.float64).ravel() / 100.0
        mean15 = sub[["mean_time_hour"]].to_numpy(dtype=np.float64).ravel()
        actors = pop15 * rate15
        # 行動者数で加重して 7 区分へ畳む。非公表（NaN）の層は分子・分母の両方から外す
        pub = ~np.isnan(mean15)
        num = np.zeros(st.D_GROUPS)
        den = np.zeros(st.D_GROUPS)
        pop_sum = np.zeros(st.D_GROUPS)
        act_sum = np.zeros(st.D_GROUPS)
        np.add.at(num, d_i, np.where(pub, actors * mean15, 0.0))
        np.add.at(den, d_i, np.where(pub, actors, 0.0))
        np.add.at(pop_sum, d_i, pop15)
        np.add.at(act_sum, d_i, actors)
        with np.errstate(invalid="ignore", divide="ignore"):
            hour = np.where(den > 0, num / den, np.nan)
            out[f"{kind}_actor_rate"] = np.where(pop_sum > 0, act_sum / pop_sum, np.nan)
        lo, hi = MEANTIME_RANGE[kind]
        bad = ~np.isnan(hour) & ((hour < lo) | (hour >= hi))
        if bad.any():
            raise ValueError(
                f"{name}: 畳んだ平均時刻が {lo}〜{hi} 時の外に {int(bad.sum())} 群ある")
        out[f"{kind}_hour"] = hour
        pop_out = pop_sum
    out["pop"] = pop_out
    return out


def derived_times_from_pool(pool: npt.NDArray[np.int64]) -> dict:
    """生成プール -> 群ごとの起床・就寝の平均時刻と不詳率。

    Note:
        ★規則の実装は `src/eval/stula_derived_times.py` ただ一つである。ここで
          条件を書き直すと、同じ規則の答えが 2 つになる。

    Args:
        pool: 群別サンプルプール, dtype=int64, (D, M, NUM_SLOTS)。値は common12 ラベル

    Returns:
        dict。wake_hour / bed_hour (D,) 平均時刻（時）、
        wake_undef_rate / bed_undef_rate (D,) 不詳の割合 [0,1]
    """
    arr = np.asarray(pool)
    d = arr.shape[0]
    out = {k: np.full(d, np.nan) for k in
           ("wake_hour", "bed_hour", "wake_undef_rate", "bed_undef_rate")}
    for i in range(d):
        s = sdt.derived_time_summary(arr[i])
        out["wake_hour"][i] = s["wake_mean"]
        out["bed_hour"][i] = s["bed_mean"]
        out["wake_undef_rate"][i] = s["wake_undef_rate"]
        out["bed_undef_rate"][i] = s["bed_undef_rate"]
    return out


def eval_derived_times(gen: dict, pub_mt: dict) -> dict:
    """生成の派生時刻を公表値へ突き合わせる。

    Note:
        ★重みは `weight_basis = stula_pop`（§9.7）。日本の公表値と比べる行なので、
          群を日本の公表人口で重み付けする。
        ★平均時刻の誤差と不詳率の差を**両方**返す。平均だけ合っていても、不詳
          （規則を満たす夜間睡眠が無い個票）が実測より多ければ睡眠の作りは壊れている。
        ★就寝は 0〜36 時の軸どうしで引く。片方だけ 24 時で折り返していると
          深夜の就寝で 24 時間ぶんの差が出るので、符号ではなく大きさで気づく。

    Args:
        gen: `derived_times_from_pool` の戻り値
        pub_mt: `load_mean_times` の戻り値

    Returns:
        指標の dict。wake_mae / wake_max_abs / wake_bias / bed_mae / bed_max_abs /
        bed_bias（いずれも時、bias は 生成 − 公表 の人口加重平均）と、
        wake_undef_gap / bed_undef_gap（生成の不詳率 − 公表の不詳率、人口加重）
    """
    pop = np.asarray(pub_mt["pop"], dtype=np.float64)
    w = pop / pop.sum()
    res: dict[str, float] = {}
    for kind in ("wake", "bed"):
        g = np.asarray(gen[f"{kind}_hour"], dtype=np.float64)
        q = np.asarray(pub_mt[f"{kind}_hour"], dtype=np.float64)
        m = ~np.isnan(g) & ~np.isnan(q)
        ww = w[m] / w[m].sum()
        diff = g[m] - q[m]
        res[f"{kind}_mae"] = float((ww * np.abs(diff)).sum())
        res[f"{kind}_max_abs"] = float(np.abs(diff).max()) if m.any() else float("nan")
        res[f"{kind}_bias"] = float((ww * diff).sum())
        # 公表の「不詳率」は 1 − 行動者率
        gu = np.asarray(gen[f"{kind}_undef_rate"], dtype=np.float64)
        qu = 1.0 - np.asarray(pub_mt[f"{kind}_actor_rate"], dtype=np.float64)
        mu = ~np.isnan(gu) & ~np.isnan(qu)
        wu = w[mu] / w[mu].sum()
        res[f"{kind}_undef_gap"] = float((wu * (gu[mu] - qu[mu])).sum())
        res[f"{kind}_n_groups"] = int(m.sum())
    return res
