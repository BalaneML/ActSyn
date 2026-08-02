"""
test_feasibility.py
================
feasibility.py の検証。test_tilting.py と同じ形式（main() に素の assert ＋ 診断 print）。

重要な項目:
    (2) travel_pairing が「往復2本」「片道1本」を正しく数え分けること。
        Step 1 で見つかった最大の欠陥（片道だけの個票が3.1倍）を測る主指標なので、
        ここが狂うと結論が狂う。
    (3) night_intrusion が深夜帯（slot 80..95）だけを見ていること。
        04:00 開始のオフセットを間違えると 04:00-08:00 を深夜と誤認する。

使い方:
    uv run python src/eval/test_feasibility.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fe = _load("feasibility", HERE / "feasibility.py")
im = _load("individual_metrics", HERE / "individual_metrics.py")


def test_episode_count():
    """(1) エピソード本数の数え方。"""
    row = np.zeros(96, dtype=np.int64)
    assert fe.episode_count_by_person(row[None, :], 0).tolist() == [1]
    assert fe.episode_count_by_person(row[None, :], 7).tolist() == [0]

    # 先頭から始まるエピソードも数える
    row = np.zeros(96, dtype=np.int64); row[0:4] = 7
    assert fe.episode_count_by_person(row[None, :], 7).tolist() == [1]

    # 離れた2本
    row = np.zeros(96, dtype=np.int64); row[10:12] = 7; row[40:42] = 7
    assert fe.episode_count_by_person(row[None, :], 7).tolist() == [2]

    # 隣接して同じ活動なら1本
    row = np.zeros(96, dtype=np.int64); row[10:20] = 7
    assert fe.episode_count_by_person(row[None, :], 7).tolist() == [1]
    print("  (1) episode_count_by_person: OK")


def test_travel_pairing():
    """(2) ★往復2本 / 片道1本 を数え分けること。"""
    # 往復: 家 -> 移動 -> 仕事 -> 移動 -> 家
    trip = np.zeros(96, dtype=np.int64)
    trip[32:34] = 7; trip[34:60] = 2; trip[60:62] = 7
    # 片道: 移動が1本だけで戻らない
    one = np.zeros(96, dtype=np.int64)
    one[32:34] = 7; one[34:] = 2
    # 移動なし
    none = np.zeros(96, dtype=np.int64)

    sched = np.stack([trip, one, none])
    assert fe.episode_count_by_person(sched, 7).tolist() == [2, 1, 0]

    s = fe.travel_pairing_summary(sched)
    assert abs(s["travel_single_rate"] - 1 / 3) < 1e-12, s["travel_single_rate"]
    assert abs(s["travel_zero_rate"] - 1 / 3) < 1e-12
    assert abs(s["travel_odd_rate"] - 1 / 3) < 1e-12, "0本を奇数に数えてはいけない"
    assert abs(s["travel_participation"] - 2 / 3) < 1e-12
    assert abs(s["travel_actor_mean"] - 1.5) < 1e-12
    assert abs(s["travel_mean"] - 1.0) < 1e-12

    tab = fe.travel_pairing(sched)
    assert list(tab["n_travel"]) == ["0", "1", "2", "3", "4+"]
    assert abs(tab["share"].sum() - 1.0) < 1e-12
    assert abs(tab[tab.n_travel == "1"]["share"].iloc[0] - 1 / 3) < 1e-12

    # 重みを片道側に寄せると single_rate が上がること
    sw = fe.travel_pairing_summary(sched, np.array([0.1, 0.8, 0.1]))
    assert sw["travel_single_rate"] > s["travel_single_rate"]
    print("  (2) travel_pairing: OK  (往復2本/片道1本/移動なし を数え分け)")


def test_night_intrusion():
    """(3) ★深夜帯 (slot 80..95) だけを見ていること。"""
    # 深夜に睡眠を1スロットだけ MEALS で割る
    row = np.zeros(96, dtype=np.int64)
    row[85] = 1
    r = fe.night_intrusion(row[None, :])
    assert r["night_intrusion_rate"] == 1.0
    assert r["night_intrusion_per_person"] == 1.0

    # 同じ割り込みを 04:00-08:00 (slot 0..15) に置くと深夜帯には入らない
    row2 = np.zeros(96, dtype=np.int64)
    row2[5] = 1
    assert fe.night_intrusion(row2[None, :])["night_intrusion_rate"] == 0.0, \
        "04:00-08:00 の割り込みを深夜として数えている（16スロットのずれ）"

    # 長い割り込み（max_slots 超え）は数えない
    row3 = np.zeros(96, dtype=np.int64)
    row3[84:90] = 1                       # 6スロット = 90分
    assert fe.night_intrusion(row3[None, :])["night_intrusion_rate"] == 0.0

    # 前後が睡眠でなければ割り込みではない
    row4 = np.zeros(96, dtype=np.int64)
    row4[80:96] = 8                       # 深夜がまるごと LEISURE
    row4[85] = 1
    assert fe.night_intrusion(row4[None, :])["night_intrusion_rate"] == 0.0

    # 割り込み2回は per_person=2、rate=1
    row5 = np.zeros(96, dtype=np.int64)
    row5[83] = 1; row5[89] = 4
    r5 = fe.night_intrusion(row5[None, :])
    assert r5["night_intrusion_per_person"] == 2.0 and r5["night_intrusion_rate"] == 1.0

    # 深夜スロットの範囲そのもの
    ns = fe._night_slots()
    assert ns.tolist() == list(range(80, 96)), ns.tolist()
    print("  (3) night_intrusion: OK  (深夜帯 = slot 80..95 のみ)")


def test_summary():
    """(4) feasibility_summary の整合。"""
    trip = np.zeros(96, dtype=np.int64)
    trip[32:34] = 7; trip[34:60] = 2; trip[60:62] = 7
    one = np.zeros(96, dtype=np.int64)
    one[32:34] = 7; one[34:] = 2
    sched = np.stack([trip, one])

    s = fe.feasibility_summary(sched)
    assert abs(s["travel_odd"] - 0.5) < 1e-12
    assert s["wrap_mismatch"] == 0.5, "one は末尾 WORK / 先頭 SLEEP で不整合"
    assert s["no_sleep_at_night"] == 0.5, "one は深夜が WORK で睡眠なし"
    assert 0.0 <= s["ANY"] <= 1.0 and s["ANY"] >= s["travel_odd"]
    assert s[f"long_travel(>{fe.MAX_TRAVEL_MIN}min)"] == 0.0

    # 長い移動を検出すること
    lng = np.zeros(96, dtype=np.int64); lng[20:40] = 7       # 20スロット = 300分
    assert fe.feasibility_summary(lng[None, :])[f"long_travel(>{fe.MAX_TRAVEL_MIN}min)"] == 1.0

    # 全員が終日睡眠なら違反ゼロ
    allsleep = np.zeros((3, 96), dtype=np.int64)
    s2 = fe.feasibility_summary(allsleep)
    assert s2["ANY"] == 0.0, s2
    print("  (4) feasibility_summary: OK")


def main():
    print("feasibility のテスト")
    test_episode_count()
    test_travel_pairing()
    test_night_intrusion()
    test_summary()

    # 実 ATUS 平日 train の基準値（プラン作成時に実測）
    ddpm = _load("ddpm_agg_model", REPO_ROOT / "src" / "models" / "DDPM_Aggregate" / "model.py")
    cond_idx, sched, weight, _ = ddpm.load_data()
    tr_idx, va_idx = im.holdout_split(len(sched), ddpm.VAL_RATIO, ddpm.SEED)
    S = sched[tr_idx]

    tab = fe.travel_pairing(S)
    s = fe.travel_pairing_summary(S)
    print(f"\n--- 実 ATUS 平日 train (N={len(S)}) の基準値 ---")
    print("  TRAVEL 本数の分布: " + "  ".join(
        f"{r.n_travel}本 {r.share:.3f}" for r in tab.itertuples()))
    print(f"  片道のみ(1本) = {s['travel_single_rate']:.4f}  "
          f"参加率 = {s['travel_participation']:.4f}  "
          f"行動者平均本数 = {s['travel_actor_mean']:.3f}")
    ni = fe.night_intrusion(S)
    print(f"  睡眠への割り込み = {ni['night_intrusion_rate']:.4f} "
          f"({ni['night_intrusion_per_person']:.4f} 回/人)")

    # プランに記録した実測値と一致すること
    assert abs(s["travel_single_rate"] - 0.077) < 2e-3, s["travel_single_rate"]
    assert abs(float(tab[tab.n_travel == "2"]["share"].iloc[0]) - 0.332) < 2e-3
    assert abs(float(tab[tab.n_travel == "0"]["share"].iloc[0]) - 0.256) < 2e-3
    assert abs(s["travel_participation"] - 0.744) < 2e-3
    print("  基準値の再現: OK")

    print("\ntest_feasibility: OK")


if __name__ == "__main__":
    main()
