"""
test_individual_metrics.py
================
individual_metrics.py の形状・整合・定義の検証。

DDPM_Aggregate/test_tilting.py と同じ形式（main() に素の assert ＋ 診断 print）。

特に重要な2項目:
    (5) holdout_split が DDPM_Aggregate/model.py の make_loaders と完全一致すること。
        ずれると「学習に使っていない個票」という前提が静かに壊れる。
    (8) 時間帯の帯ラベル "00:00-02:00" に入る境界が t = 80..87 であること。
        16スロットずれると 04:00-06:00 帯を深夜帯と誤認する。

使い方:
    uv run python src/eval/test_individual_metrics.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    """フラット配置の model.py 群を名前衝突なしにロードする。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


im = _load("individual_metrics", HERE / "individual_metrics.py")


def test_sequence_decomposition():
    """(1)(2) 手で作った系列で系列分解が既知の値になること。"""
    # 96 = 10 + 20 + 66 の3エピソード、切替2回
    row = np.concatenate([np.zeros(10), np.ones(20), np.full(66, 2)]).astype(np.int64)
    sched = row[None, :]

    assert im.n_switches(sched).tolist() == [2]
    assert im.switch_matrix(sched).shape == (1, 95)
    assert im.switch_matrix(sched).sum() == 2

    segs = im.to_segments(row, [f"A{i}" for i in range(3)])
    assert segs == [("A0", 0, 10), ("A1", 10, 30), ("A2", 30, 96)], segs

    runs = im.episode_lengths(sched)
    assert sorted(runs.tolist()) == [10, 20, 66]

    by_act = im.episode_lengths_by_act(sched, 3)
    assert by_act[0].tolist() == [10] and by_act[1].tolist() == [20] and by_act[2].tolist() == [66]

    # 全個票について run 長の和が 96
    rng = np.random.default_rng(0)
    many = rng.integers(0, 5, size=(40, 96)).astype(np.int64)
    for i in range(len(many)):
        assert im.episode_lengths(many[i:i + 1]).sum() == 96
    # 活動別に分けても総和は保存される
    assert sum(len(x) for x in im.episode_lengths_by_act(many, 5)) == \
        len(im.episode_lengths(many))
    print("  (1)(2) 系列分解: OK")


def test_distribution_quantities():
    """(3)(4) 分布量の正規化と重みの扱い。"""
    rng = np.random.default_rng(1)
    n_act = 12
    sched = rng.integers(0, n_act, size=(200, 96)).astype(np.int64)
    w = rng.random(200) + 0.1

    ts = im.time_share(sched, n_act)
    assert abs(ts.sum() - 1.0) < 1e-12, ts.sum()
    assert abs(im.time_share(sched, n_act, w).sum() - 1.0) < 1e-12

    part = im.participation(sched, n_act)
    assert (part >= 0).all() and (part <= 1 + 1e-12).all()

    # 一様重みを明示的に渡した場合と w=None が一致すること
    uni = np.ones(200)
    assert np.allclose(im.time_share(sched, n_act), im.time_share(sched, n_act, uni))
    assert np.allclose(im.participation(sched, n_act), im.participation(sched, n_act, uni))

    # 時刻別行動者率は各スロットで和1 (model.py:711-717 の smoke と同じ性質)
    pbs = im.participation_by_slot(sched, n_act, w)
    assert pbs.shape == (96, n_act)
    assert np.allclose(pbs.sum(axis=1), 1.0), "participation_by_slot の正規化が壊れている"

    # pop_mean = time_share × 1440 の一貫性
    dt = im.duration_table(sched, n_act, [f"A{a}" for a in range(n_act)], w)
    assert np.allclose(dt["pop_mean_min"].to_numpy(), ts_w := im.time_share(sched, n_act, w) * 1440)
    print(f"  (3)(4) 分布量: OK  (time_share 和={ts.sum():.12f}, pop_mean 一致)")


def test_holdout_split_matches_make_loaders():
    """(5) holdout_split が make_loaders の分割と完全一致すること。"""
    ddpm = _load("ddpm_agg_model", REPO_ROOT / "src" / "models" / "DDPM_Aggregate" / "model.py")
    cond_idx, sched, weight, _ = ddpm.load_data()
    n = len(sched)

    tr_idx, va_idx = im.holdout_split(n, ddpm.VAL_RATIO, ddpm.SEED)

    # make_loaders が実際に作る Dataset の中身と突き合わせる
    train_loader, val_loader = ddpm.make_loaders(cond_idx, sched, weight)
    tr_ds = train_loader.dataset.schedules.numpy()
    va_ds = val_loader.dataset.schedules.numpy()

    assert len(tr_idx) == len(tr_ds) and len(va_idx) == len(va_ds), \
        f"分割サイズ不一致: {len(tr_idx)}/{len(tr_ds)}, {len(va_idx)}/{len(va_ds)}"
    assert (sched[tr_idx] == tr_ds).all(), "train 分割が make_loaders と一致しない"
    assert (sched[va_idx] == va_ds).all(), "val 分割が make_loaders と一致しない"
    assert len(np.intersect1d(tr_idx, va_idx)) == 0, "train と val が重複している"
    assert len(tr_idx) + len(va_idx) == n

    # numpy の RNG では別の並びになることの確認（この差が壊れると意味がなくなる）
    np_perm = np.random.default_rng(ddpm.SEED).permutation(n)
    assert not (np_perm[:len(va_idx)] == va_idx).all(), \
        "numpy と torch の並びが一致してしまっている（前提の確認が無意味になる）"

    print(f"  (5) holdout_split: OK  (N={n}, train={len(tr_idx)}, val={len(va_idx)})")
    return ddpm, cond_idx, sched, weight, tr_idx, va_idx


def test_against_fragmentation_stats(ddpm, sched):
    """(6) switch_mean が model.py の fragmentation_stats["switches"] と一致すること。

    ★ fragmentation_stats は individual_metrics へ委譲され、旧実装の
      sched[:3000] 打ち切りは撤廃された。3000行を超えるデータで一致することを
      確認するのがこのテストの主眼（打ち切りが復活したら ep_len 系がずれる）。
    """
    assert len(sched) > 3000, "打ち切り撤廃の確認には3000行を超えるデータが要る"
    sub = sched
    mine = im.fragmentation_summary(sub)
    theirs = ddpm.fragmentation_stats(sub)
    assert abs(mine["switch_mean"] - theirs["switches"]) < 1e-9, \
        f"{mine['switch_mean']} vs {theirs['switches']}"
    assert abs(mine["ep_len_mean"] - theirs["ep_len_mean"]) < 1e-9
    assert abs(mine["ep_len_median"] - theirs["ep_len_median"]) < 1e-9
    assert abs(mine["single_slot_ratio"] - theirs["single_slot_ratio"]) < 1e-9

    # 重み付きでも一致すること
    w = np.random.default_rng(2).random(len(sub)) + 0.1
    assert abs(im.switch_stats(sub, w)["mean"] - ddpm.fragmentation_stats(sub, w)["switches"]) < 1e-9

    # 参加率も model.py と一致すること
    assert np.allclose(im.participation(sub, ddpm.NUM_ACT, w),
                       ddpm.fragmentation_stats(sub, w)["participation"])
    print(f"  (6) fragmentation_stats と一致: OK  (switches={mine['switch_mean']:.6f})")


def test_band_accounting(sched):
    """(7) 帯分割で切替を落とさない／二重計上しないこと。"""
    sub = sched[:500]
    w = np.random.default_rng(3).random(len(sub)) + 0.1

    for band_hours in (1, 2, 3, 4, 6, 8, 12):
        b = im.switches_by_band(sub, w, band_hours)
        assert int(b["n_boundaries"].sum()) == 95, \
            f"band_hours={band_hours}: 境界本数の合計 {b['n_boundaries'].sum()} != 95"
        total = float((im._norm_w(len(sub), w) * im.n_switches(sub)).sum())
        assert abs(b["switches_per_person"].sum() - total) < 1e-9, \
            f"band_hours={band_hours}: 帯合計 {b['switches_per_person'].sum()} != 全体 {total}"
        assert len(b) == 24 // band_hours

    # 04:00 を含む帯だけ境界が1本少ない（t=0 は境界を持たない）
    b2 = im.switches_by_band(sub, w, 2)
    row_04 = b2[b2["label"] == "04:00-06:00"]
    assert int(row_04["n_boundaries"].iloc[0]) == 7, "04:00 を含む帯の境界本数が7でない"
    assert (b2[b2["label"] != "04:00-06:00"]["n_boundaries"] == 8).all()
    print("  (7) 帯の勘定: OK  (境界95本を保存, 04:00帯のみ7本)")


def test_night_band_slot_mapping(sched):
    """(8) ★深夜帯のスロット対応。"00:00-02:00" の境界が t = 80..87 であること。"""
    # slot と時刻の対応そのもの
    assert int(im.slot_clock_min(0)) == 4 * 60, "slot 0 が 04:00 でない"
    assert int(im.slot_clock_min(80)) == 0, "slot 80 が 00:00 でない"
    assert int(im.slot_clock_min(95)) == 3 * 60 + 45, "slot 95 が 03:45 でない"
    assert im.fmt_time(80) == "00:00" and im.fmt_time(0) == "04:00"
    assert im.roll_slots() == 16

    # 境界 t は slot t の開始時刻に起きる -> 00:00-02:00 帯は t = 80..87
    clock = im._boundary_clock_min()          # t = 1..95
    t_of = np.arange(1, im.NUM_SLOTS)
    in_band = t_of[(clock >= 0) & (clock < 2 * 60)]
    assert in_band.tolist() == list(range(80, 88)), \
        f"00:00-02:00 帯の境界が {in_band.tolist()}（期待 80..87）= スロット対応が16ずれている"

    # 00:00-04:00 は境界16本
    night_t = t_of[(clock >= 0) & (clock < 4 * 60)]
    assert night_t.tolist() == list(range(80, 96)), night_t.tolist()

    # 深夜帯だけに切替を仕込んだ系列で night_switches が拾えること
    row = np.zeros(96, dtype=np.int64)
    row[85] = 1                                # 境界 t=85 と t=86 の2回（01:15 付近）
    ns = im.night_switches(row[None, :])
    assert ns["n_boundaries"] == 16
    assert abs(ns["switches_per_person"] - 2.0) < 1e-12
    assert abs(ns["share_of_all_switches"] - 1.0) < 1e-12, "深夜以外を拾っている"

    # 逆に slot 0..15（04:00-08:00）に仕込んだ切替は深夜帯に入らないこと
    row2 = np.zeros(96, dtype=np.int64)
    row2[5] = 1
    assert im.night_switches(row2[None, :])["switches_per_person"] == 0.0, \
        "04:00-08:00 の切替を深夜帯として数えている"

    # 日跨ぎ境界は n_switches に入らず wrap_closure_rate で測る
    row3 = np.concatenate([np.zeros(48), np.ones(48)]).astype(np.int64)
    assert im.n_switches(row3[None, :]).tolist() == [1]
    assert im.wrap_closure_rate(row3[None, :]) == 1.0
    assert im.wrap_closure_rate(np.zeros((1, 96), dtype=np.int64)) == 0.0
    print("  (8) 深夜帯のスロット対応: OK  (00:00-02:00 = 境界 t=80..87)")


def test_duration_identities():
    """(9) duration_table の恒等式。"""
    rng = np.random.default_rng(4)
    n_act = 12
    names = [f"A{a}" for a in range(n_act)]
    for w_on in (False, True):
        sched = rng.integers(0, n_act, size=(150, 96)).astype(np.int64)
        w = rng.random(150) + 0.1 if w_on else None
        dt = im.duration_table(sched, n_act, names, w)

        # pop_mean = actor_mean × participation
        lhs = dt["pop_mean_min"].to_numpy()
        rhs = dt["actor_mean_min"].to_numpy() * dt["participation"].to_numpy()
        assert np.allclose(lhs, rhs, atol=1e-9), np.abs(lhs - rhs).max()

        # pop_mean = time_share × 1440
        assert np.allclose(lhs, im.time_share(sched, n_act, w) * 1440, atol=1e-9)

        # participation は participation() と一致
        assert np.allclose(dt["participation"].to_numpy(), im.participation(sched, n_act, w))

        # 全員平均時間の合計は1日 = 1440分
        assert abs(lhs.sum() - 1440.0) < 1e-6, lhs.sum()

        # actor_mean >= pop_mean（参加率 <= 1 なので）
        assert (dt["actor_mean_min"].to_numpy() >= lhs - 1e-9).all()

    # 手で作った系列: 活動1が 20 スロット連続 1本のみ
    row = np.zeros(96, dtype=np.int64); row[10:30] = 1
    dt = im.duration_table(row[None, :], 2, ["A", "B"])
    b = dt[dt.activity == "B"].iloc[0]
    assert b["ep_mean_min"] == 20 * 15 and b["actor_mean_min"] == 20 * 15
    assert b["pop_mean_min"] == 20 * 15 and b["participation"] == 1.0
    assert b["n_episodes_per_person"] == 1.0
    print("  (9) 継続時間の恒等式: OK  (pop_mean = actor_mean × part = share×1440)")


def test_weighted_quantile():
    """(10) weighted_quantile が一様重みで np.percentile と一致すること。"""
    rng = np.random.default_rng(5)
    for n in (7, 50, 373):
        x = rng.normal(size=n) * 10
        qs = [0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
        mine = np.asarray(im.weighted_quantile(x, qs))
        theirs = np.percentile(x, [q * 100 for q in qs])
        assert np.allclose(mine, theirs, atol=1e-9), np.abs(mine - theirs).max()

    # 重みを2倍にしても比が同じなら結果は変わらない（スケール不変）
    x = rng.normal(size=30)
    w = rng.random(30) + 0.1
    assert np.allclose(np.asarray(im.weighted_quantile(x, [0.25, 0.5, 0.75], w)),
                       np.asarray(im.weighted_quantile(x, [0.25, 0.5, 0.75], w * 3.7)))

    # 片側に重みを寄せるとその側へ動くこと。
    # 両端を 0/1 に張る規約（一様重みで np.percentile と一致させるため）なので、
    # 最小値と最大値は q=0 / q=1 に固定される。重みが効くのは内部の位置。
    x = np.arange(10.0)
    lo_heavy = np.array([10, 10, 10, 1, 1, 1, 1, 1, 1, 1.0])
    hi_heavy = lo_heavy[::-1].copy()
    m_uni = im.weighted_quantile(x, 0.5)
    m_lo = im.weighted_quantile(x, 0.5, lo_heavy)
    m_hi = im.weighted_quantile(x, 0.5, hi_heavy)
    assert m_lo < m_uni < m_hi, (m_lo, m_uni, m_hi)
    assert im.weighted_quantile(x, 0.75, lo_heavy) < im.weighted_quantile(x, 0.75, hi_heavy)
    # 端点は重みに関わらず min / max
    assert im.weighted_quantile(x, 0.0, lo_heavy) == 0.0
    assert im.weighted_quantile(x, 1.0, lo_heavy) == 9.0
    print(f"  (10) weighted_quantile: OK  (一様重みで np.percentile と一致, "
          f"median 低重み={m_lo:.2f} < 一様={m_uni:.2f} < 高重み={m_hi:.2f})")


def test_dist_compare_and_bootstrap():
    """(11) 同一入力で emd = ks = 0。bootstrap CI が平均を含むこと。"""
    rng = np.random.default_rng(6)
    sched = rng.integers(0, 12, size=(300, 96)).astype(np.int64)

    d = im.switch_dist_compare(sched, sched)
    assert abs(d["emd"]) < 1e-12 and abs(d["ks"]) < 1e-12, d

    # 明らかに断片化した系列とは距離が開くこと
    frag = rng.integers(0, 12, size=(300, 96)).astype(np.int64)
    smooth = np.repeat(rng.integers(0, 12, size=(300, 6)), 16, axis=1).astype(np.int64)
    d2 = im.switch_dist_compare(frag, smooth)
    assert d2["emd"] > 10.0 and d2["ks"] > 0.9, d2

    sw = im.n_switches(sched).astype(np.float64)
    lo, hi = im.bootstrap_ci(sw, n_boot=500, seed=0)
    assert lo < sw.mean() < hi, (lo, sw.mean(), hi)
    assert hi - lo < 5.0, "CI が広すぎる（実装がおかしい可能性）"
    print(f"  (11) 分布距離と bootstrap: OK  (同一入力 emd={d['emd']:.2e}, CI=[{lo:.2f},{hi:.2f}])")


def test_compare_tables(sched):
    """compare_frag / compare_activity が形を保つこと（描画は notebook 側で確認）。"""
    a, b = sched[:200], sched[200:400]
    cf = im.compare_frag(a, b)
    assert set(cf.columns) == {"metric", "real", "gen", "ratio"}
    assert (cf[cf.metric == "switch_mean"]["ratio"].iloc[0] > 0)

    names = [f"A{i}" for i in range(12)]
    ca = im.compare_activity(a, b, 12, names)
    assert len(ca) == 12
    assert abs(ca["share_real"].sum() - 1.0) < 1e-12
    assert abs(ca["share_gen"].sum() - 1.0) < 1e-12
    print("  compare_frag / compare_activity: OK")


def test_null_band(sched):
    """(12) ★ノイズ床。同一入力なら全指標が床の内に入ること。"""
    real = sched[:1500]
    stat = lambda s, w=None: im.switch_stats(s, w)["mean"]

    lo, md, hi = im.null_band(real, stat, n_boot=200, seed=0)
    obs = stat(real)
    assert lo < obs < hi, (lo, obs, hi)
    assert lo < md < hi

    # gen = real なら全行「床の内」
    stats = {
        "switch_mean": lambda s, w=None: im.switch_stats(s, w)["mean"],
        "switch_var": lambda s, w=None: im.switch_stats(s, w)["var"],
        "night": lambda s, w=None: im.night_switches(s, w)["switches_per_person"],
        "ep_len_mean": lambda s: float(im.episode_lengths(s).mean()),
    }
    df = im.compare_with_band(real, real, stats, n_boot=150, seed=0)
    assert set(df.columns) == {"metric", "real", "ci_lo", "ci_hi", "gen", "ratio", "verdict"}
    assert (df["verdict"] == "床の内").all(), df
    assert np.allclose(df["ratio"], 1.0)
    assert (df["ci_lo"] <= df["real"]).all() and (df["real"] <= df["ci_hi"]).all()

    # 明らかに違う入力なら床の外に出ること
    smooth = np.repeat(np.random.default_rng(0).integers(0, 12, size=(len(real), 6)),
                       16, axis=1).astype(np.int64)
    df2 = im.compare_with_band(real, smooth, stats, n_boot=150, seed=0)
    assert (df2[df2.metric == "switch_mean"]["verdict"] == "★床の外").all(), df2

    # 床は実データ側の N で決まる。gen の N を増やしても床は変わらない
    df3 = im.compare_with_band(real, np.concatenate([smooth] * 3), stats,
                               n_boot=150, seed=0)
    assert np.allclose(df2["ci_lo"], df3["ci_lo"]) and np.allclose(df2["ci_hi"], df3["ci_hi"])

    # (sched) だけを取る stat_fn も (sched, w) を取るものも扱えること
    d4 = im.compare_with_band(real, real, {"a": lambda s: float(len(s)),
                                           "b": lambda s, w=None: float(len(s))},
                              n_boot=20, seed=0)
    assert len(d4) == 2
    # --- 2標本統計量の床 ---
    n_act = 12
    jsd = lambda r, g: im.bigram_jsd(r, g, n_act)
    j_lo, j_md, j_hi = im.null_band_2sample(real, jsd, n_gen=len(real), n_boot=40, seed=0)
    assert 0.0 <= j_lo <= j_md <= j_hi
    assert j_md > 0, "同分布からの2標本でも有限標本ゆらぎで JSD は 0 より大きいはず"

    # 実データ自身との JSD は 0（=床の内）、明らかに違う構造は床の外
    assert im.bigram_jsd(real, real, n_act) <= j_hi
    assert im.bigram_jsd(real, smooth, n_act) > j_hi

    # gen を増やすと床は縮むが、実データ側の誤差が残るので 0 にはならない
    _, j_md_big, _ = im.null_band_2sample(real, jsd, n_gen=len(real) * 8,
                                          n_boot=40, seed=0)
    assert j_md_big < j_md and j_md_big > 0, (j_md, j_md_big)
    print(f"  (12) null_band / compare_with_band: OK  (switch_mean CI=[{lo:.3f},{hi:.3f}], "
          f"bigram_jsd 床 median={j_md:.5f})")


def test_dispersion(sched):
    """(13) ばらつき指標。分散が縮んだ入力を検出できること。"""
    n_act = 12
    names = [f"A{i}" for i in range(n_act)]
    real = sched[:1200]

    dp = im.dispersion_profile(real, n_act, names)
    assert len(dp) == 2 + 2 * n_act              # switches + n_unique_act + 各活動2種
    assert (dp["var"] >= 0).all()
    sw_row = dp[dp.quantity == "switches"].iloc[0]
    assert abs(sw_row["mean"] - im.switch_stats(real)["mean"]) < 1e-9
    assert abs(sw_row["var"] - im.switch_stats(real)["var"]) < 1e-9

    # minutes の合計が 1440 になること
    mins = dp[dp.quantity.str.startswith("minutes:")]["mean"].sum()
    assert abs(mins - 1440.0) < 1e-6, mins

    # 分散を人工的に縮めた入力: 全員を同じ個票の複製にすると var=0
    clone = np.repeat(real[:1], len(real), axis=0)
    cmp = im.compare_dispersion(real, clone, n_act, names)
    assert cmp[cmp.quantity == "switches"]["var_gen"].iloc[0] == 0.0
    assert cmp[cmp.quantity == "switches"]["var_ratio"].iloc[0] == 0.0

    # pairwise 距離: 複製集合は距離0、実データは正
    dr = im.pairwise_distance_dist(real, 5000, seed=0)
    dc = im.pairwise_distance_dist(clone, 5000, seed=0)
    assert dr["mean"] > 0.3 and dc["mean"] == 0.0
    assert dr["iqr"] > 0 and dc["iqr"] == 0.0
    assert dr["p05"] <= dr["median"] <= dr["p95"]

    # group_dispersion
    groups = np.arange(len(real)) % 4
    gd = im.group_dispersion(real, groups, 4, sample_pairs=500)
    assert len(gd) == 4 and (gd["n"] > 0).all()
    print(f"  (13) dispersion: OK  (実データ Hamming 平均={dr['mean']:.3f} "
          f"IQR={dr['iqr']:.3f}, 複製集合=0)")


def test_bigram(sched):
    """(14) ★bigram。定義（対角の扱い）で値が変わることを明示的に確認。"""
    n_act = 12
    real = sched[:1500]
    gen = sched[1500:3000]

    # ★bincount 版が、スロットごとの素朴なループと厳密に一致すること
    #   （_bigram_counts の高速化で遷移の勘定が壊れていないかの確認）
    w_chk = np.random.default_rng(7).random(len(real)) + 0.1
    wn = w_chk / w_chk.sum()
    naive = np.zeros((n_act, n_act))
    for t in range(1, real.shape[1]):
        np.add.at(naive, (real[:, t - 1], real[:, t]), wn)
    assert np.allclose(im._bigram_counts(real, n_act, w_chk, False), naive)

    M = im.bigram_matrix(real, n_act, exclude_diagonal=False)
    rs = M.sum(axis=1)
    assert np.allclose(rs[rs > 0], 1.0), "行和が1でない"
    assert (M >= 0).all()

    Md = im.bigram_matrix(real, n_act, exclude_diagonal=True)
    assert np.allclose(np.diag(Md), 0.0), "exclude_diagonal で対角が残っている"
    rsd = Md.sum(axis=1)
    assert np.allclose(rsd[rsd > 0], 1.0)

    # 同一入力なら 0
    for excl in (True, False):
        for avg in ("row", "joint"):
            v = im.bigram_jsd(real, real, n_act, exclude_diagonal=excl, average=avg)
            assert abs(v) < 1e-12, (excl, avg, v)

    # 対角の扱いで値が変わること（プラン記載の桁の違い）
    v_in = im.bigram_jsd(real, gen, n_act, exclude_diagonal=False)
    v_ex = im.bigram_jsd(real, gen, n_act, exclude_diagonal=True)
    assert v_ex > v_in, (v_in, v_ex)
    assert v_ex > 3 * v_in, f"対角込み {v_in:.5f} / 除く {v_ex:.5f} の差が想定より小さい"

    # 明らかに違う遷移構造なら大きくなること
    smooth = np.repeat(np.random.default_rng(1).integers(0, n_act, size=(1500, 6)),
                       16, axis=1).astype(np.int64)
    assert im.bigram_jsd(real, smooth, n_act) > v_ex

    # average の値チェック
    try:
        im.bigram_jsd(real, gen, n_act, average="bad")
        raise AssertionError("不正な average を受け付けている")
    except ValueError:
        pass
    print(f"  (14) bigram: OK  (対角込み={v_in:.5f} < 対角除く={v_ex:.5f})")


def test_memorization(sched):
    """(15) 最近傍距離と暗記率。"""
    train = sched[:1500]
    holdout = sched[1500:1900]

    # 自分自身への最近傍は距離0、第2近傍はそれ以上
    d = im.nn_distances(train[:100], train, k=2, sample=100, seed=0)
    assert d.shape == (100, 2)
    assert (d[:, 0] == 0).all(), "自分自身が最近傍にならない"
    assert (d[:, 1] >= d[:, 0]).all(), "第2近傍が第1近傍より小さい"

    # gen が train の完全コピーなら exact_copy_rate = 1
    m = im.memorization(train[:200], train, holdout, sample=200, seed=0)
    assert abs(m["exact_copy_rate[train]"] - 1.0) < 1e-12
    assert m["DCR_mean[train]"] == 0.0
    assert m["DCR_mean[holdout]"] > 0.0
    assert m["DCR_gap(holdout-train)"] > 0.0, "train に近いのに gap が正でない"

    # 無関係な乱数なら train にも holdout にも遠く、gap は小さい
    rnd = np.random.default_rng(2).integers(0, 12, size=(200, 96)).astype(np.int64)
    m2 = im.memorization(rnd, train, holdout, sample=200, seed=0)
    assert m2["exact_copy_rate[train]"] == 0.0
    assert m2["DCR_mean[train]"] > 50
    assert abs(m2["DCR_gap(holdout-train)"]) < 5, m2["DCR_gap(holdout-train)"]

    # NNDR は [0,1]
    for k in ("NNDR_mean[train]", "NNDR_p05[train]"):
        assert 0.0 <= m2[k] <= 1.0, (k, m2[k])
    print(f"  (15) memorization: OK  (完全コピー DCR=0/gap={m['DCR_gap(holdout-train)']:.1f}, "
          f"乱数 DCR={m2['DCR_mean[train]']:.1f})")


def test_group_reweight(sched):
    """(16) 群構成の再重み付け。"""
    rng = np.random.default_rng(0)
    d_ref = rng.integers(0, 5, 500)
    w_ref = rng.random(500) + 0.1
    d_gen = np.repeat(np.arange(5), 40)

    w = im.group_reweight(d_gen, w_ref, d_ref, 5)
    assert abs(w.sum() - 1.0) < 1e-12

    # 再重み付け後の群シェアが実データの重み付きシェアと一致すること
    wr = w_ref / w_ref.sum()
    for d in range(5):
        assert abs(w[d_gen == d].sum() - wr[d_ref == d].sum()) < 1e-12

    # 生成側に存在しない群は重み0で、和は1に満たなくなる（0除算しない）
    d_gen2 = np.repeat(np.arange(4), 40)
    w2 = im.group_reweight(d_gen2, w_ref, d_ref, 5)
    assert np.isfinite(w2).all() and w2.sum() < 1.0
    print("  (16) group_reweight: OK")


def main():
    print("individual_metrics のテスト")
    test_sequence_decomposition()
    test_distribution_quantities()
    ddpm, cond_idx, sched, weight, tr_idx, va_idx = test_holdout_split_matches_make_loaders()
    test_against_fragmentation_stats(ddpm, sched)
    test_band_accounting(sched)
    test_night_band_slot_mapping(sched)
    test_duration_identities()
    test_weighted_quantile()
    test_dist_compare_and_bootstrap()
    test_compare_tables(sched)
    test_null_band(sched)
    test_dispersion(sched)
    test_bigram(sched)
    test_memorization(sched)
    test_group_reweight(sched)

    # 実 val の実測値を再現するか（プラン作成時に測った基準値）
    S, W = sched[va_idx], weight[va_idx]
    st_u = im.switch_stats(S)
    st_w = im.switch_stats(S, W)
    ns = im.night_switches(S, W)
    dt = im.duration_table(S, ddpm.NUM_ACT, ddpm.ACT_NAMES, W)
    work = dt[dt.activity == "WORK"].iloc[0]
    print("\n--- 実 ATUS 平日 val (N=%d) の基準値 ---" % len(S))
    print(f"  切替 平均 非重み={st_u['mean']:.3f} 重み付き={st_w['mean']:.3f} "
          f"var={st_u['var']:.3f} median={st_u['median']:.1f} p95={st_u['p95']:.1f} max={st_u['max']:.0f}")
    print(f"  深夜 00:00-04:00 = {ns['switches_per_person']:.3f} 回/人 "
          f"(全切替の {100 * ns['share_of_all_switches']:.1f}%), 境界あたり {ns['switch_rate']:.4f}")
    print(f"  wrap_closure_rate = {im.wrap_closure_rate(S, W):.4f}")
    print(f"  WORK: ep_mean={work['ep_mean_min']:.1f} actor_mean={work['actor_mean_min']:.1f} "
          f"pop_mean={work['pop_mean_min']:.1f} part={work['participation']:.3f}")

    assert abs(st_u["mean"] - 12.780) < 5e-3, st_u["mean"]
    assert abs(st_w["mean"] - 12.619) < 5e-3, st_w["mean"]
    # var は母分散 (ddof=0)。np.var(ddof=1) の 22.930 とは N/(N-1) 倍違う
    assert abs(st_u["var"] - 22.869) < 5e-3, st_u["var"]
    assert abs(st_u["var"] * len(S) / (len(S) - 1) - 22.930) < 5e-3
    assert abs(st_u["std"] - 4.782) < 5e-3, st_u["std"]
    assert abs(st_u["cv"] - 0.374) < 5e-3, st_u["cv"]
    assert abs(im.wrap_closure_rate(S, W) - 0.0683) < 5e-4
    assert abs(ns["switches_per_person"] - 0.231) < 5e-3, ns["switches_per_person"]
    assert abs(work["ep_mean_min"] - 219.8) < 0.1, work["ep_mean_min"]
    assert abs(work["actor_mean_min"] - 470.9) < 0.1, work["actor_mean_min"]
    assert abs(work["pop_mean_min"] - 217.2) < 0.1, work["pop_mean_min"]
    print("  基準値の再現: OK")

    print("\ntest_individual_metrics: OK")


if __name__ == "__main__":
    main()
