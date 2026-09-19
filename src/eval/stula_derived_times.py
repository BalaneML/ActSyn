"""
stula_derived_times.py
======================
社会生活基本調査の規則で「起床時刻」「就寝時刻」を生成物から機械的に決める
（Stage2_design.md §9.7 / 実装項目 17、規則の出所は `[[articles/stula-2016-terminology]]`）

**なぜ要るか。**教師 `A*`（時刻別行動者率）への適合は循環している（§9.2）。派生時刻は
教師が縛らない量で、日本の公表値（平均時刻編）に照らせる。`im.first_onset` のような
「最初に現れたスロット」とは**別物**である点に注意する。あちらは任意の閾値で決まるが、
こちらは公表統計が実際に使っている操作的定義そのものである。

規則（`[[articles/stula-2016-terminology]]` 3節）:

| 時刻 | 規則 |
|---|---|
| 起床 | 0時以降12時前に始まり **60分を超えて** 続く最初の睡眠の終了時刻 |
| 就寝 | 17時以降 **36時（翌12時）前** に始まり60分を超えて続く睡眠の開始時刻。該当が2つ以上なら継続時間最長（同じなら早い方） |

- **連結規則**: 睡眠と睡眠の間の非睡眠が **30分以内なら睡眠が継続している** とみなす
- **日界**: 就寝時刻は **0〜36時** の座標で扱う（翌 1:00 は 25.0 であって 1.0 ではない）。
  これを守らないと、遅い就寝が平均を**下げて**しまい公表値と比べられない

★**座標系が公表統計と違う。**本リポジトリの表現は 04:00 起点・24時間**閉じ**の 96 スロットで、
公表統計は 0:00 起点・連続 2 日間（0〜48時）である。この差が規則の翻訳に 2 箇所効く。

```mermaid
flowchart TD
    A["生成 (N, 96)<br/>04:00 起点・円周"] --> B["睡眠マスク"]
    B --> C["30分以内の非睡眠を埋める<br/>bridge_short_gaps"]
    C --> D["円周 run へ分解<br/>_circular_runs"]
    D --> E["60分超の run だけ残す"]
    E --> F["終了が 0-12時 -> 最も早い<br/>= 起床"]
    E --> G["開始が 17-36時 -> 最長<br/>= 就寝"]
    F --> H["wake_hour [0,12)"]
    G --> I["bed_hour36 [17,36)"]
```

1. **円周の run を使う（線形ではない）。**04:00 起点だと夜間睡眠が必ず配列の端をまたぐ
   （23:00 就寝 = スロット 76、07:00 起床 = スロット 12）。線形に読むと夜間睡眠が
   「末尾 20 スロット」と「先頭 12 スロット」の 2 本に割れ、末尾側は 5 時間しか無いのに
   就寝側の判定に使われ、**60 分超の条件を満たしても継続時間最長の比較で不利になる**。
   円周で 1 本として扱えば実際の 8 時間として読める。

2. **起床は「0-12時に *始まる*」ではなく「0-12時に *終わる*」で判定する。**公表統計は
   日記が 0:00 で始まるので、その時点で既に眠っている人の睡眠は「0:00 に始まる」と
   記録され、条件を満たす。24 時間閉じの円周には日記の開始に当たる境界が無いので、
   同じ条件をそのまま当てると夜間睡眠（23:00 開始）が窓から外れ、**ほぼ全員が不詳**に
   なる。終了時刻で判定すれば、規則が取ろうとしている量（目覚めた時刻）がそのまま出る。

★**該当なしは NaN（不詳）** で返し、平均から除外する。規則どおりの扱いである。
**終日睡眠（96 スロット全部が睡眠）も不詳**にする。円周では開始も終了も実在せず、
残すと「04:00 起床・28:00 就寝」というもっともらしい値が出て、モデルが睡眠へ潰れた
事故を指標が隠してしまう。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    """sys.modules に一意名で載せる（src/eval の他モジュールと同じ作法）。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ★円周分解・スロット→時計・窓判定は schedule_plausibility のものをそのまま使う。
#   ここで書き直すと、04:00 起点の端またぎという同じ問題への答えが 2 つになり、
#   片方だけ直したときに黙ってずれる。`_circular_runs` は名前こそ私有だが、
#   このリポジトリで円周 run を返す唯一の実装である。
spl: Any = load_module("stula_dt_schedule_plausibility",
                       REPO_ROOT / "src" / "eval" / "schedule_plausibility.py")
NUM_SLOTS = spl.NUM_SLOTS
SLOT_MIN = spl.SLOT_MIN
SLEEP_PERSONAL = spl.SLEEP_PERSONAL
_circular_runs = spl._circular_runs
in_window = spl.in_window
slot_to_clock_min = spl.slot_to_clock_min

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]

# 60 分を「超えて」続くこと。15 分スロットなので 4 スロット超 = 5 スロット以上
MIN_SLEEP_MIN = 60
# 睡眠間の非睡眠がこの長さ以内なら睡眠が継続しているとみなす
MAX_GAP_MIN = 30
# 起床: 終了時刻がこの窓に入る睡眠（0:00-12:00）
WAKE_WINDOW = (0, 12 * 60)
# 就寝: 開始時刻がこの窓に入る睡眠（17:00-翌12:00 の 19 時間。日跨ぎ窓）
BED_WINDOW = (17 * 60, 12 * 60)
# 就寝時刻を載せる軸の折り返し。これより前の時刻は「翌日」として +24 する
BED_AXIS_WRAP_HOUR = 12


def bridge_short_gaps(sleep: BoolArr, max_gap_min: int = MAX_GAP_MIN) -> BoolArr:
    """睡眠と睡眠の間の短い非睡眠を睡眠で埋める（連結規則）。

    Note:
        ★円周で埋める。配列の端をまたぐ隙間（例: スロット 95 と 0 の間）も
          規則の対象である。
        ★睡眠が 1 スロットも無い行は触らない。その行の「非睡眠 run」は 96 スロット
          丸ごとの 1 本で、埋める相手の睡眠が存在しないためである。

    Args:
        sleep: 睡眠スロットの真偽値, dtype=bool, (N, NUM_SLOTS)
        max_gap_min: 埋める隙間の上限（分）, default=MAX_GAP_MIN=30

    Returns:
        隙間を埋めた睡眠マスク, dtype=bool, (N, NUM_SLOTS)。入力は変更しない
    """
    out = np.array(sleep, dtype=bool, copy=True)
    # ★np.asarray で包む。ndarray.any(axis=...) はスタブ上 np.bool を返すことに
    #   なっており、そのまま添字を取ると reportIndexIssue になる
    has_sleep = np.asarray(out.any(axis=1))
    row, start, length = _circular_runs(~out)
    sel = np.asarray((length * SLOT_MIN <= max_gap_min) & has_sleep[row])
    if not sel.any():
        return out
    r, s, ln = row[sel], start[sel], length[sel]
    # ★ragged な run をまとめてスロット添字へ展開する。run ごとに Python ループを
    #   回すと、プール (28群 × 2000本) で run が数十万本になり効かない
    offs = np.arange(int(ln.sum())) - np.repeat(np.cumsum(ln) - ln, ln)
    out[np.repeat(r, ln), (np.repeat(s, ln) + offs) % NUM_SLOTS] = True
    return out


def sleep_episodes(sched: IntArr, sleep_act: int = SLEEP_PERSONAL,
                   max_gap_min: int = MAX_GAP_MIN,
                   min_sleep_min: int = MIN_SLEEP_MIN
                   ) -> tuple[IntArr, IntArr, IntArr]:
    """規則を満たす睡眠エピソードを円周上で拾う。

    Args:
        sched: スケジュール, dtype=int64, (N, NUM_SLOTS)。値は共通12分類のラベル
        sleep_act: 睡眠の活動ラベル, default=SLEEP_PERSONAL
        max_gap_min: 連結する隙間の上限（分）, default=MAX_GAP_MIN
        min_sleep_min: 継続時間の下限（分）。**この値を超える**ものだけ残す,
            default=MIN_SLEEP_MIN

    Returns:
        (row, start, length) の3本, dtype=int64, (R,)。
        row が個票、start が開始スロット、length がスロット数
    """
    sleep = bridge_short_gaps(np.asarray(sched) == sleep_act, max_gap_min)
    row, start, length = _circular_runs(sleep)
    keep = length * SLOT_MIN > min_sleep_min          # 「超えて」なので等号を含めない
    # ★終日睡眠は落とす。円周では start も end も「配列の 0 番目」という便宜上の
    #   位置でしかなく、起床も就寝も実在しない。残すと 04:00 起床・28:00 就寝という
    #   もっともらしい値が出てしまい、**モデルが睡眠へ潰れた事故を指標が隠す**。
    #   規則の側でも「終了時刻」「開始時刻」が定義できないので不詳が正しい。
    keep &= length < NUM_SLOTS
    return row[keep], start[keep], length[keep]


def _first_per_row(row: IntArr, key: FloatArr, n: int) -> FloatArr:
    """行ごとに key が最小の値を返す, -> (n,)。該当が無い行は NaN。

    Args:
        row: 各要素の属する行, dtype=int64, (R,)
        key: 比較に使う値, dtype=float64, (R,)
        n: 行数

    Returns:
        行ごとの最小値, dtype=float64, (n,)。該当なしは NaN
    """
    out = np.full(n, np.inf)
    np.minimum.at(out, row, key)
    return np.where(np.isinf(out), np.nan, out)


def wake_time(sched: IntArr, **kw) -> FloatArr:
    """起床時刻（0時からの時間、[0,12)）, (N, 96) -> (N,)

    規則: 60 分を超えて続く睡眠のうち、**終了時刻が 0:00-12:00 に入る最初のもの**の
    終了時刻。該当が無ければ NaN（不詳）。

    Note:
        ★「0-12時に *始まる*」ではなく「0-12時に *終わる*」で判定する理由は
          モジュール docstring の 2 を参照。24 時間閉じの円周には日記の開始境界が
          無いため、開始時刻で判定するとほぼ全員が不詳になる。

    Args:
        sched: スケジュール, dtype=int64, (N, NUM_SLOTS)
        **kw: `sleep_episodes` へ渡す（sleep_act / max_gap_min / min_sleep_min）

    Returns:
        起床時刻（時）, dtype=float64, (N,)。範囲 [0,12)、不詳は NaN
    """
    n = np.asarray(sched).shape[0]
    row, start, length = sleep_episodes(sched, **kw)
    end_min = slot_to_clock_min((start + length) % NUM_SLOTS)   # 終端の次＝目覚めた時刻
    sel = in_window(end_min, WAKE_WINDOW)
    return _first_per_row(row[sel], end_min[sel] / 60.0, n)


def bed_time(sched: IntArr, **kw) -> FloatArr:
    """就寝時刻（1日目0時からの時間、[17,36)）, (N, 96) -> (N,)

    規則: 60 分を超えて続く睡眠のうち、**開始時刻が 17:00-翌12:00 に入るもの**の開始時刻。
    2 つ以上あれば継続時間が最長のもの、同じなら早い方。該当が無ければ NaN（不詳）。

    Note:
        ★戻り値は 0〜36 時の軸である。翌 1:00 の就寝は **25.0**（1.0 ではない）。
          公表統計がこの軸で平均を取るためで、24 時間で折り返すと遅い就寝が平均を
          下げてしまい比較が成り立たない。

    Args:
        sched: スケジュール, dtype=int64, (N, NUM_SLOTS)
        **kw: `sleep_episodes` へ渡す（sleep_act / max_gap_min / min_sleep_min）

    Returns:
        就寝時刻（時）, dtype=float64, (N,)。範囲 [17,36)、不詳は NaN
    """
    n = np.asarray(sched).shape[0]
    row, start, length = sleep_episodes(sched, **kw)
    start_min = slot_to_clock_min(start)
    sel = in_window(start_min, BED_WINDOW)
    row, start_min, length = row[sel], start_min[sel], length[sel]
    hour = start_min / 60.0
    hour36 = np.where(hour < BED_AXIS_WRAP_HOUR, hour + 24.0, hour)

    out = np.full(n, np.nan)
    if len(row) == 0:
        return out
    # 行ごとに「長さ最大、同着なら早い方」を取る。行 -> 長さ降順 -> 時刻昇順 に
    # 並べ替えて、各行の先頭を拾う
    order = np.lexsort((hour36, -length, row))
    r_sorted = row[order]
    _, first = np.unique(r_sorted, return_index=True)
    out[r_sorted[first]] = hour36[order][first]
    return out


def derived_time_summary(sched: IntArr, w: FloatArr | None = None) -> dict[str, float]:
    """起床・就寝の要約。公表値（平均時刻編）と突き合わせるための行を作る。

    Note:
        ★不詳（NaN）は平均から除外し、その割合を別に返す。規則どおりの扱いだが、
          **不詳率そのものが指標である**。生成物に「60 分を超える夜間睡眠が無い」
          個票が多ければ、平均が合っていても睡眠の作りが壊れている。

    Args:
        sched: スケジュール, dtype=int64, (N, NUM_SLOTS)
        w: 個票の重み, dtype=float64, (N,)。None なら一様, default=None

    Returns:
        wake_mean / wake_sd / wake_undef_rate / bed_mean / bed_sd / bed_undef_rate。
        時刻は時単位。就寝は 0〜36 時の軸
    """
    n = np.asarray(sched).shape[0]
    wn = np.ones(n) / n if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    out: dict[str, float] = {}
    for name, fn in (("wake", wake_time), ("bed", bed_time)):
        x = fn(sched)
        ok = ~np.isnan(x)
        ws = wn[ok].sum()
        if ws <= 0:
            out[f"{name}_mean"] = float("nan")
            out[f"{name}_sd"] = float("nan")
        else:
            mean = float((wn[ok] * x[ok]).sum() / ws)
            var = float((wn[ok] * (x[ok] - mean) ** 2).sum() / ws)
            out[f"{name}_mean"] = mean
            out[f"{name}_sd"] = float(np.sqrt(max(var, 0.0)))
        out[f"{name}_undef_rate"] = float(wn[~ok].sum())
    return out


def compare_derived_times(real: IntArr, gen: IntArr,
                          w_real: FloatArr | None = None,
                          w_gen: FloatArr | None = None) -> dict[str, float]:
    """実データと生成の派生時刻を並べ、差を出す。

    Args:
        real: 実データのスケジュール, dtype=int64, (N_real, NUM_SLOTS)
        gen: 生成のスケジュール, dtype=int64, (N_gen, NUM_SLOTS)
        w_real: 実データの重み, (N_real,), default=None
        w_gen: 生成の重み, (N_gen,), default=None

    Returns:
        `derived_time_summary` の各値を real_* / gen_* として持ち、
        平均の差を d_wake_mean / d_bed_mean（生成 − 実、時）で返す
    """
    r = derived_time_summary(real, w_real)
    g = derived_time_summary(gen, w_gen)
    out = {f"real_{k}": v for k, v in r.items()}
    out.update({f"gen_{k}": v for k, v in g.items()})
    out["d_wake_mean"] = g["wake_mean"] - r["wake_mean"]
    out["d_bed_mean"] = g["bed_mean"] - r["bed_mean"]
    out["d_wake_undef_rate"] = g["wake_undef_rate"] - r["wake_undef_rate"]
    out["d_bed_undef_rate"] = g["bed_undef_rate"] - r["bed_undef_rate"]
    return out
