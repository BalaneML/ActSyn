"""
test_atus_group_rates.py
========================
atus_group_rates.py のテスト（pytest は使わない。既存の規約どおり単体スクリプト）

固定しているのは次の5点:
    (a) 一様な重みを渡した加重版が非加重版と一致する（重み経路の健全性）
    (b) 各 (群, スロット) で12活動の和が 1（1スロット1ラベルの帰結）
    (c) 群の割り当てが stage2_targets.d_index と同じ規則である
    (d) to_act_major の並びが model.pool_to_rates と同じである
    (e) 活動ラベルの取り違えが無い（WORK のピークが昼に立つ）

使い方:
    .venv/bin/python3 src/eval/test_atus_group_rates.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
import atus_group_rates as ag  # noqa: E402

SIMPLE_DIR = REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple"


def _load(name: str, path: Path) -> ModuleType:
    """同名ファイルの取り違えを避けるためファイル直ロードする（repo 共通の作法）。"""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_csv(tmp: Path) -> Path:
    """28群それぞれ1人ずつ + 週末1人 の合成個票。群割り当ての検査に使う。"""
    rows: list[dict[str, object]] = []
    for g in range(ag.N_G):
        for a in range(ag.N_A):
            for e in range(ag.N_E):
                rows.append({
                    "TUCASEID": len(rows),
                    "TUFINLWGT": 1.0 + len(rows),          # 群ごとに違う重み
                    "age": 15 + 10 * a + 3,                # 区分の内側に入る年齢
                    "gender": g,
                    "day_of_week": 3,                      # 平日
                    "telfs": 1 if e == 1 else 3,           # 1|2 = 有業、それ以外 = 無業
                    **{f"s{j}": (len(rows) + j) % ag.NUM_COMMON for j in range(ag.NUM_SLOTS)},
                })
    rows.append({**rows[0], "TUCASEID": 999, "day_of_week": 1})   # 日曜。除外されるはず
    path = tmp / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_uniform_weight_matches_unweighted() -> None:
    """(a) 一様な重みでは加重版と非加重版が一致する。"""
    sched, groups, _ = ag.load_atus_weekday()
    unw = ag.group_rates(sched, groups)
    uni = ag.group_rates(sched, groups, np.ones(len(sched)))
    gap = float(np.abs(unw - uni).max())
    # ★厳密一致は要求しない。非加重は整数和を1回割る経路、加重は einsum 経路で
    #   加算順が違う。teacher_free_space が要求する厳密性は非加重経路の中だけの話
    assert gap < 1e-12, f"一様重みが非加重と一致しない (最大差 {gap:.3e})"
    print(f"test_uniform_weight_matches_unweighted: OK  (最大差 {gap:.3e})")


def test_channel_sum_is_one() -> None:
    """(b) 各 (群, スロット) で12活動の和が 1。加重・非加重の両方で確かめる。"""
    sched, groups, w = ag.load_atus_weekday()
    for label, weights in (("weighted", w), ("unweighted", None)):
        rates = ag.group_rates(sched, groups, weights)
        gap = float(np.abs(np.nansum(rates, axis=1) - 1.0).max())
        assert gap < 1e-9, f"{label}: 12活動の和が1でない (最大差 {gap:.3e})"
        assert float(np.nanmin(rates)) >= 0.0, f"{label}: 負の率がある"
    print("test_channel_sum_is_one: OK")


def test_group_index_matches_d_index() -> None:
    """(c) 群の割り当てが stage2_targets.d_index と同じ規則である。

    合成個票で28群すべてを1周し、曜日フィルタと空群の NaN もここで確かめる。
    """
    sys.path.insert(0, str(SIMPLE_DIR))
    st = _load("stage2_targets_for_test", SIMPLE_DIR / "stage2_targets.py")

    with tempfile.TemporaryDirectory() as tmp:
        sched, groups, w = ag.load_atus_weekday(_synthetic_csv(Path(tmp)))

    assert len(sched) == ag.D_GROUPS, f"平日フィルタが効いていない (N={len(sched)})"
    k = 0
    for g in range(ag.N_G):
        for a in range(ag.N_A):
            for e in range(ag.N_E):
                assert groups[k] == st.d_index(g, a, e), \
                    f"(g={g}, a={a}, e={e}) の群が d_index と違う"
                k += 1

    # 1人だけの群では率が 0/1 のどちらかしか取らない
    rates = ag.group_rates(sched, groups, w)
    assert set(np.unique(rates)) <= {0.0, 1.0}, "1人の群の率が 0/1 以外を取っている"

    # 空の群は NaN で埋まる
    partial = ag.group_rates(sched[:1], groups[:1], w[:1])
    assert not np.isnan(partial[0]).any(), "標本のある群が NaN になっている"
    assert np.isnan(partial[1:]).all(), "標本の無い群が NaN で埋まっていない"
    print("test_group_index_matches_d_index: OK")


def test_act_major_matches_pool_to_rates() -> None:
    """(d) to_act_major の並びが model.pool_to_rates と同じ（c*96 + s）。"""
    sm = _load("simple_model_for_test", SIMPLE_DIR / "model.py")
    rng = np.random.default_rng(0)
    n = 5
    pool = rng.integers(0, ag.NUM_COMMON, size=(ag.D_GROUPS, n, ag.NUM_SLOTS))

    expect = sm.pool_to_rates(pool)                                   # (28, 12*96)
    sched = pool.reshape(-1, ag.NUM_SLOTS)
    groups = np.repeat(np.arange(ag.D_GROUPS), n)
    got = ag.to_act_major(ag.group_rates(sched, groups))
    gap = float(np.abs(expect - got).max())
    assert gap < 1e-12, f"act-major の並びが pool_to_rates と違う (最大差 {gap:.3e})"
    print(f"test_act_major_matches_pool_to_rates: OK  (最大差 {gap:.3e})")


# crosswalk_atus_stula.md の「ATUS」列（平日・TUFINLWGT 加重の人時シェア, %）。
# あちらはエピソード単位で weight × 分を積んだ値で、こちらはスロット率を群集約した値。
# 経路が違うので、一致すれば「重み付け・群集約・活動ラベルの並び」が同時に裏づく
PUBLISHED_SHARE_PCT = {
    "SLEEP_PERSONAL": 40.02, "MEALS": 4.60, "WORK": 16.79, "SCHOOL": 2.08,
    "HOUSEWORK": 7.85, "CAREGIVING": 2.29, "SHOPPING": 1.09, "TRAVEL": 4.55,
    "LEISURE_SOCIAL": 17.41, "SPORTS": 1.34, "VOLUNTEER": 0.47, "OTHER_X": 1.50,
}


def _national_curve() -> np.ndarray:
    """群を人口（ウェイト）シェアで集約した全国の時刻別行動者率 (12, 96)。"""
    sched, groups, w = ag.load_atus_weekday()
    rates = ag.group_rates(sched, groups, w)
    _, wsum, _ = ag.group_counts(groups, w)
    return np.einsum("d,dcs->cs", wsum / wsum.sum(), rates)


def test_activity_labels() -> None:
    """(e) 活動ラベルの取り違えが無いこと。

    WORK=2 / SCHOOL=3 / HOUSEWORK=4 / CAREGIVING=5 の並びは一度取り違えている
    （Stage2_design.md §9.2）。ここでは2通りで確かめる:
        1. 全国の時間シェアが crosswalk_atus_stula.md の公表値と一致する（±0.1pt）
        2. ピークの時刻が常識と合う。s0 = 04:00 なので
           WORK は勤務時間帯 09:00-17:00（s20..s52）、SLEEP は深夜（s72..s8）
    """
    national = _national_curve()
    for c in ag.Common:
        share_pct = float(national[int(c)].sum()) / ag.NUM_SLOTS * 100.0
        want = PUBLISHED_SHARE_PCT[c.name]
        assert abs(share_pct - want) < 0.1, \
            f"{c.name}: 時間シェア {share_pct:.2f}% が公表値 {want:.2f}% と違う"

    work_peak = int(np.argmax(national[int(ag.Common.WORK)]))
    assert 20 <= work_peak <= 52, f"WORK のピークが勤務時間帯にない (スロット {work_peak})"
    sleep_peak = int(np.argmax(national[int(ag.Common.SLEEP_PERSONAL)]))
    assert sleep_peak >= 72 or sleep_peak <= 8, f"睡眠のピークが深夜にない (スロット {sleep_peak})"
    print(f"test_activity_labels: OK  (WORK peak s{work_peak}, SLEEP peak s{sleep_peak}, "
          f"12活動の時間シェアが公表値と ±0.1pt 以内)")


def main() -> None:
    test_uniform_weight_matches_unweighted()
    test_channel_sum_is_one()
    test_group_index_matches_d_index()
    test_act_major_matches_pool_to_rates()
    test_activity_labels()
    print("\nすべて OK")


if __name__ == "__main__":
    main()
