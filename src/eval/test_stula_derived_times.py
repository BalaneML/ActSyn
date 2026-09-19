"""
test_stula_derived_times.py
===========================
`stula_derived_times.py` の単体テスト（Stage2_design.md §9.7 / 実装項目 17）

規則の出所は `[[articles/stula-2016-terminology]]` 3節。ここで固定するのは
**規則の翻訳**であって実装の細部ではない。特に次の 3 つは、間違えても例外を出さず
数値だけが静かにずれるので、テストで釘を打っておく。

    (1) 60 分は「超えて」であり「以上」ではない
    (2) 就寝時刻は 0〜36 時の軸に載る（翌 1:00 は 25.0）
    (3) 夜間睡眠は 04:00 起点の配列で端をまたぐので、円周で 1 本として読む

実行: python src/eval/test_stula_derived_times.py
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dt: Any = _load("stula_derived_times", HERE / "stula_derived_times.py")

SLEEP = 0
AWAKE = 8           # LEISURE_SOCIAL。睡眠以外なら何でもよい
N_SLOTS = 96


def _slot_of(hour: float) -> int:
    """時計の時刻 -> 04:00 起点のスロット番号。"""
    return int(round((hour * 60 - 4 * 60) % (24 * 60) / 15)) % N_SLOTS


def _make(spans: list[tuple[float, float]], n: int = 1) -> np.ndarray:
    """時計時刻の区間 [開始, 終了) を睡眠にした (n, 96) を作る。日跨ぎ可。"""
    s = np.full((n, N_SLOTS), AWAKE, dtype=np.int64)
    for lo, hi in spans:
        a = _slot_of(lo)
        n_slot = int(round(((hi - lo) % 24) * 4))
        s[:, (a + np.arange(n_slot)) % N_SLOTS] = SLEEP
    return s


def test_typical_night() -> None:
    """典型: 23:00 就寝 -> 07:00 起床。配列の端をまたぐ 1 本として読めること。"""
    s = _make([(23.0, 7.0)])
    assert dt.wake_time(s)[0] == 7.0, dt.wake_time(s)
    assert dt.bed_time(s)[0] == 23.0, dt.bed_time(s)

    # ★円周で 1 本になっていること。線形に読むと末尾 20 スロットと先頭 28 スロットの
    #   2 本に割れる。その場合 bed 側に渡るのは 5 時間の断片になる
    row, start, length = dt.sleep_episodes(s)
    assert len(row) == 1, f"夜間睡眠が {len(row)} 本に割れている（円周で読めていない）"
    assert length[0] == 32, f"睡眠が 8 時間 (32 スロット) でない: {length[0]}"
    print(f"  (1) 23:00-07:00 が円周 1 本 (32 スロット) / wake=7.0 / bed=23.0: OK")


def test_bed_axis_is_36h() -> None:
    """★就寝が 0〜36 時の軸に載ること。翌 1:00 は 25.0 であって 1.0 ではない。"""
    s = _make([(1.0, 9.0)])                       # 深夜 1:00 就寝 -> 9:00 起床
    bed = dt.bed_time(s)[0]
    assert bed == 25.0, f"翌1:00 の就寝が 25.0 でない: {bed}"
    assert dt.wake_time(s)[0] == 9.0

    # 24 時間で折り返していたら、遅い就寝の平均が早い就寝より小さくなってしまう
    early, late = dt.bed_time(_make([(21.0, 5.0)]))[0], bed
    assert late > early, f"遅い就寝 {late} が早い就寝 {early} より小さい（軸の折り返し）"
    print(f"  (2) 翌1:00 の就寝 = {bed} / 21:00 の就寝 = {early}（順序が保たれる）: OK")


def test_sixty_minutes_is_strict() -> None:
    """★「60分を超えて」は厳密。ちょうど 60 分の睡眠は採らない。"""
    exact = _make([(2.0, 3.0)])                   # 2:00-3:00 = ちょうど 60 分
    assert np.isnan(dt.wake_time(exact)[0]), "ちょうど60分を採ってしまっている"
    over = _make([(2.0, 3.25)])                   # 75 分
    assert dt.wake_time(over)[0] == 3.25, dt.wake_time(over)
    print("  (3) 60分ちょうどは不詳、75分は採用: OK")


def test_gap_bridging() -> None:
    """★睡眠間の非睡眠 30 分以内は睡眠が継続しているとみなす。"""
    # 23:00-2:00 と 2:30-7:00。間の 30 分は埋まって 1 本 (8 時間) になる
    s = _make([(23.0, 2.0), (2.5, 7.0)])
    row, _, length = dt.sleep_episodes(s)
    assert len(row) == 1, f"30 分の隙間が埋まっていない: {len(row)} 本"
    assert length[0] == 32, f"連結後が 8 時間でない: {length[0]}"
    assert dt.wake_time(s)[0] == 7.0

    # 45 分の隙間は埋めない。2 本のままで、起床は「最も早い終了」＝ 2:00
    s2 = _make([(23.0, 2.0), (2.75, 7.0)])
    row2, _, _ = dt.sleep_episodes(s2)
    assert len(row2) == 2, f"45 分の隙間を埋めてしまった: {len(row2)} 本"
    assert dt.wake_time(s2)[0] == 2.0, dt.wake_time(s2)
    print("  (4) 30分の隙間は連結 (1本/8時間)、45分は連結しない (2本): OK")


def test_bed_picks_longest() -> None:
    """就寝は「該当が2つ以上なら継続時間最長」。昼寝ではなく夜間睡眠を採ること。"""
    # 18:00-19:30 の仮眠 (90分) と 23:00-07:00 の夜間睡眠 (8時間)
    s = _make([(18.0, 19.5), (23.0, 7.0)])
    row, _, length = dt.sleep_episodes(s)
    assert len(row) == 2, f"2 本にならない: {len(row)}"
    assert dt.bed_time(s)[0] == 23.0, \
        f"短い仮眠 18:00 を就寝に採ってしまった: {dt.bed_time(s)[0]}"
    # 起床は「最も早い終了」なので仮眠の 19:30 ではなく 07:00（窓 0-12時の外を除く）
    assert dt.wake_time(s)[0] == 7.0, dt.wake_time(s)
    print("  (5) 90分の仮眠と8時間の夜間睡眠 -> 就寝=23.0 / 起床=7.0: OK")


def test_undefined_cases() -> None:
    """不詳の扱い。睡眠なし・終日睡眠・窓外はすべて NaN。"""
    none = np.full((1, N_SLOTS), AWAKE, dtype=np.int64)
    assert np.isnan(dt.wake_time(none)[0]) and np.isnan(dt.bed_time(none)[0])

    # ★終日睡眠。円周では開始も終了も実在しないので不詳。残すと 04:00 起床・
    #   28:00 就寝というもっともらしい値が出て、睡眠へ潰れた事故を隠す
    allsleep = np.zeros((1, N_SLOTS), dtype=np.int64)
    assert np.isnan(dt.wake_time(allsleep)[0]), "終日睡眠に起床時刻が出ている"
    assert np.isnan(dt.bed_time(allsleep)[0]), "終日睡眠に就寝時刻が出ている"

    # 13:00-15:00 の昼寝だけ。起床窓 (0-12時) にも就寝窓 (17-36時) にも入らない
    nap = _make([(13.0, 15.0)])
    assert np.isnan(dt.wake_time(nap)[0]), "昼寝の終了を起床に採っている"
    assert np.isnan(dt.bed_time(nap)[0]), "昼寝の開始を就寝に採っている"
    print("  (6) 睡眠なし / 終日睡眠 / 昼寝のみ はすべて不詳: OK")


def test_summary_excludes_undefined() -> None:
    """要約が不詳を平均から外し、その割合を別に返すこと。"""
    s = np.concatenate([_make([(23.0, 7.0)], n=3),
                        np.full((1, N_SLOTS), AWAKE, dtype=np.int64)])
    r = dt.derived_time_summary(s)
    assert r["wake_mean"] == 7.0, r
    assert abs(r["wake_undef_rate"] - 0.25) < 1e-12, r
    assert r["wake_sd"] == 0.0
    print(f"  (7) 4 個票中 1 個不詳 -> wake_mean=7.0 / undef_rate=0.25: OK")


def test_on_real_atus() -> None:
    """実 ATUS 2024 で常識的な値が出ること（回帰の見張り）。"""
    import pandas as pd
    path = (REPO_ROOT / "data" / "processed" / "atus2024"
            / "atus2024_stula_common12_dataset.csv")
    if not path.exists():
        print(f"  (8) skip: {path.name} が無い")
        return
    df = pd.read_csv(path)
    sched = df[[f"s{j}" for j in range(N_SLOTS)]].to_numpy(dtype=np.int64)
    w = df["TUFINLWGT"].to_numpy(dtype=float) if "TUFINLWGT" in df.columns else None
    r = dt.derived_time_summary(sched, w)
    assert 5.5 < r["wake_mean"] < 9.0, f"起床の平均が常識外: {r['wake_mean']:.2f}"
    assert 21.0 < r["bed_mean"] < 25.0, f"就寝の平均が常識外: {r['bed_mean']:.2f}"
    assert r["wake_undef_rate"] < 0.10 and r["bed_undef_rate"] < 0.10, \
        f"不詳が多すぎる: wake {r['wake_undef_rate']:.3f} / bed {r['bed_undef_rate']:.3f}"
    print(f"  (8) 実ATUS N={len(df)}: 起床 {r['wake_mean']:.2f}h "
          f"(不詳 {r['wake_undef_rate']:.1%}) / 就寝 {r['bed_mean']:.2f}h "
          f"(不詳 {r['bed_undef_rate']:.1%}): OK")


def main() -> None:
    test_typical_night()
    test_bed_axis_is_36h()
    test_sixty_minutes_is_strict()
    test_gap_bridging()
    test_bed_picks_longest()
    test_undefined_cases()
    test_summary_excludes_undefined()
    test_on_real_atus()
    print("\ntest_stula_derived_times: OK")


if __name__ == "__main__":
    main()
