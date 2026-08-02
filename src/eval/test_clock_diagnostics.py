"""
test_clock_diagnostics.py
================
clock_diagnostics.py の検証。

診断は「時計が無い」と結論しうるので、**指標そのものが時計に反応することを
先に示す**必要がある。各テストは「答えが分かっている合成データ」で
    - 時計を壊した入力 -> 床の外 / gap 大 / R² 高
    - 壊していない入力 -> 床の内 / gap ≈ 0 / R² ≈ 0
の両方向を確認する。片方向だけだと、常に「異常なし」を返す指標でも通ってしまう。

使い方:
    uv run python src/eval/test_clock_diagnostics.py
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


cd = _load("clock_diagnostics", HERE / "clock_diagnostics.py")
im = cd.im

NUM_SLOTS = cd.NUM_SLOTS
N_ACT = 4
ACT_NAMES = ["SLEEP_PERSONAL", "WORK", "MEALS", "TRAVEL"]


def make_clocked(n: int, seed: int = 0) -> np.ndarray:
    """時刻が固定された合成日記 (n, 96)。

    SLEEP(0) 00-20 / TRAVEL(3) 20-24 / WORK(1) 24-52 / MEALS(2) 52-56 /
    WORK(1) 56-72 / TRAVEL(3) 72-76 / SLEEP(0) 76-96。
    開始時刻は個票ごとに ±2 スロットだけ揺らす（実データ相当の弱いばらつき）。
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(NUM_SLOTS, dtype=np.int64)
    for a, s, e in [(3, 20, 24), (1, 24, 52), (2, 52, 56), (1, 56, 72), (3, 72, 76)]:
        base[s:e] = a
    out = np.stack([np.roll(base, int(k)) for k in rng.integers(-2, 3, n)])
    return out.astype(np.int64)


def make_declocked(sched: np.ndarray, seed: int = 0) -> np.ndarray:
    """各個票を大きく巡回シフトして時計だけを壊す。

    活動の構成・エピソード長・切替回数はそのまま。日内リズム（時刻との対応）
    だけが失われるので、時刻を見る指標だけが反応するはず。
    """
    rng = np.random.default_rng(seed)
    return np.stack([np.roll(r, int(k)) for r, k in
                     zip(sched, rng.integers(0, NUM_SLOTS, len(sched)))]).astype(np.int64)


def test_curve_comparison():
    """(1) B1: 時計を壊すとピークが鈍り、L1 が床の外に出ること。"""
    real = make_clocked(1200, seed=0)
    same = make_clocked(1200, seed=1)
    flat = make_declocked(real, seed=2)

    # 同じ生成過程どうしなら床の内
    ok = cd.curve_comparison(real, same, N_ACT, ACT_NAMES, n_boot=40, seed=0)
    assert (ok["l1_verdict"] == "床の内").all(), ok
    assert np.allclose(ok["peak_ratio"], 1.0, atol=0.15), ok

    # 時計を壊すとピークが鈍り、L1 は床の外
    bad = cd.curve_comparison(real, flat, N_ACT, ACT_NAMES, n_boot=40, seed=0)
    assert (bad["l1_verdict"] == "★床の外").all(), bad
    # 巡回シフトで平坦化するので、鋭いピークを持つ活動は必ず低くなる
    sharp = bad[bad.activity.isin(["WORK", "TRAVEL", "MEALS"])]
    assert (sharp["peak_ratio"] < 0.95).all(), sharp
    assert (bad["curve_l1"] > ok["curve_l1"]).all()

    # 切替回数はほぼ変わらないこと（B1 が断片化でなく時刻に反応している確認）。
    # 巡回シフトで切れ目が切替位置に当たると、その1回が日跨ぎ境界へ移って
    # 数えられなくなるため厳密一致にはならない（境界は t=1..95 の95本だけ）
    d_sw = abs(im.switch_stats(real)["mean"] - im.switch_stats(flat)["mean"])
    assert d_sw < 0.2, d_sw
    print(f"  (1) B1 curve: OK  (時計あり L1={ok['curve_l1'].mean():.4f} / "
          f"壊した後 L1={bad['curve_l1'].mean():.4f}, peak比={bad['peak_ratio'].min():.2f})")


def test_first_onset():
    """(2) B2: 初回開始時刻が手計算と合い、過分散が std_ratio と EMD に出ること。"""
    # WORK(1) が slot 24 から始まる個票 100 本 + WORK を全くしない 20 本
    sched = make_clocked(100, seed=0)
    none = np.zeros((20, NUM_SLOTS), dtype=np.int64)
    both = np.concatenate([sched, none])

    onset = cd.first_onset(both, 1)
    assert len(onset) == 100, "WORK をしない個票が除かれていない"
    assert abs(float(np.median(onset)) - 24.0) <= 2.0, float(np.median(onset))

    # 決め打ちの1本で厳密確認
    row = np.zeros((1, NUM_SLOTS), dtype=np.int64)
    row[0, 37:40] = 2
    assert cd.first_onset(row, 2).tolist() == [37]
    assert len(cd.first_onset(row, 3)) == 0

    real = make_clocked(1200, seed=0)
    flat = make_declocked(real, seed=2)
    ok = cd.onset_comparison(real, make_clocked(1200, seed=1), N_ACT, ACT_NAMES,
                             n_boot=40, seed=0)
    bad = cd.onset_comparison(real, flat, N_ACT, ACT_NAMES, n_boot=40, seed=0)
    assert (ok["verdict"] == "床の内").all(), ok
    assert (bad["verdict"] == "★床の外").all(), bad

    # SLEEP は必ず slot 0 始まりなので実データ側の分散が 0。比は定義できず nan になる
    sleep = bad[bad.activity == "SLEEP_PERSONAL"].iloc[0]
    assert sleep["onset_std_real"] == 0.0 and np.isnan(sleep["onset_std_ratio"])
    rest = bad[bad.activity != "SLEEP_PERSONAL"]
    assert (rest["onset_std_ratio"] > 3.0).all(), rest
    print(f"  (2) B2 onset: OK  (std比 時計あり={ok['onset_std_ratio'].mean():.2f} / "
          f"壊した後={rest['onset_std_ratio'].mean():.2f})")


def test_wrap():
    """(3) B3: 日跨ぎ境界の破れが検出されること。"""
    real = make_clocked(1200, seed=0)
    assert im.wrap_closure_rate(real) == 0.0, "合成データは始端も終端も SLEEP のはず"

    ok = cd.wrap_comparison(real, make_clocked(1200, seed=1), n_boot=100, seed=0)
    assert ok["verdict"] == "床の内", ok

    # 終端だけ別活動にして環を壊す
    broken = real.copy()
    broken[:, -1] = 1
    bad = cd.wrap_comparison(real, broken, n_boot=100, seed=0)
    assert bad["wrap_gen"] == 1.0 and bad["verdict"] == "★床の外", bad
    print(f"  (3) B3 wrap: OK  (実={ok['wrap_real']:.3f} 破壊後={bad['wrap_gen']:.3f})")


def test_ridge_probe():
    """(4) B4: 位置情報を含む特徴で R²≈1、含まない特徴で R²≈0 になること。"""
    rng = np.random.default_rng(0)
    length, n, c = 96, 200, 16
    tg = cd._phase_targets(length)
    assert tg.shape == (length, 2)
    assert abs(tg[:, 0].mean()) < 1e-9 and abs(tg[:, 1].mean()) < 1e-9

    # 位置情報あり: 位相を特徴に埋め込む（+ ノイズ）
    feats = rng.normal(size=(n, length, c))
    feats[:, :, 0] += 5.0 * tg[None, :, 0]
    feats[:, :, 1] += 5.0 * tg[None, :, 1]
    r2_pos = cd.ridge_probe_r2(feats, tg, n_train=n // 2)
    assert r2_pos > 0.9, r2_pos

    # 位置情報なし: 純ランダム特徴
    r2_rand = cd.ridge_probe_r2(rng.normal(size=(n, length, c)), tg, n_train=n // 2)
    assert abs(r2_rand) < 0.05, r2_rand

    # 位置に依らない定数オフセットは位置情報にならない
    flatf = rng.normal(size=(n, length, c)) + 3.0
    assert abs(cd.ridge_probe_r2(flatf, tg, n_train=n // 2)) < 0.05
    print(f"  (4) B4 probe: OK  (位置埋め込みあり R²={r2_pos:.3f} / なし R²={r2_rand:+.3f})")


def test_shift_equivariance():
    """(5) B5: シフト等変なモデルで gap≈0、時計を持つモデルで gap 大。"""
    import torch
    import torch.nn as nn

    ch, length, n = 4, NUM_SLOTS, 32

    class Diff:
        """q_sample だけを持つ最小スタブ（本物の Diffusion と同じ呼び出し形）。"""
        def q_sample(self, x0, t, eps):
            return 0.7 * x0 + 0.7 * eps

    class Equivariant(nn.Module):
        """巡回パディングの畳み込みのみ = 厳密にシフト等変。条件も時刻も見ない。"""
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, 3, padding=1, padding_mode="circular")

        def forward(self, x, t, cond_idx=None, drop_mask=None):
            return self.conv(x)

    class Clocked(Equivariant):
        """位置ごとの固定バイアスを足す = 時計を持つ。"""
        def __init__(self):
            super().__init__()
            self.pos = nn.Parameter(torch.randn(1, ch, length), requires_grad=False)

        def forward(self, x, t, cond_idx=None, drop_mask=None):
            return self.conv(x) + self.pos

    torch.manual_seed(0)
    x0 = torch.randn(n, ch, length)
    cond = torch.zeros(n, 3, dtype=torch.long)

    eq = cd.shift_equivariance(Equivariant().eval(), Diff(), x0, cond, seed=0)
    assert (eq["gap"] < 1e-5).all(), eq
    assert set(eq["shift_slots"]) == set(cd.SHIFTS)

    ck = cd.shift_equivariance(Clocked().eval(), Diff(), x0, cond, seed=0)
    assert (ck["gap"] > 0.05).all(), ck
    assert (ck["gap"] > eq["gap"] * 1000).all()

    # ヤードスティックが埋まっていること（この2モデルは条件を見ないので cond_gap=0）
    assert (eq["noise_gap"] > 0).all()
    assert np.isnan(eq["gap_over_cond"]).all(), "cond_gap=0 のとき比は nan にすべき"
    print(f"  (5) B5 shift: OK  (等変モデル gap={eq['gap'].max():.2e} / "
          f"時計つき gap={ck['gap'].min():.3f})")


def test_position_probe_integration():
    """(6) B4 の配線: features() を持つモデルに対して表が返ること。"""
    import torch
    import torch.nn as nn

    class Stub(nn.Module):
        """h1 に位置位相を、h2 に純ノイズを載せる。R² が両者を区別すること。"""
        def __init__(self):
            super().__init__()
            ang = 2 * np.pi * torch.arange(NUM_SLOTS) / NUM_SLOTS
            # 素の属性にする（register_buffer は不要。CPU のみで動かすスタブ）
            self.phase = torch.stack([ang.sin(), ang.cos()])[None].repeat(1, 4, 1)

        @property
        def in_channels(self) -> int:
            """position_probe が純ノイズ x_T を作るのに使う。バックボーン共通の契約。"""
            return 12

        def features(self, x, t, cond_idx=None):
            b = x.size(0)
            h1 = torch.randn(b, 8, NUM_SLOTS) + 5.0 * self.phase
            return {"h1": h1, "h2": torch.randn(b, 8, NUM_SLOTS // 2)}

    torch.manual_seed(0)
    cond = torch.zeros(4, 3, dtype=torch.long)
    df = cd.position_probe(Stub().eval(), None, cond, n_samples=64, t_step=999)
    assert list(df["feature"]) == ["h1", "h2"]
    assert df.loc[0, "r2_all"] > 0.9, df
    assert abs(df.loc[1, "r2_all"]) < 0.1, df
    assert df.loc[0, "resolution"] == NUM_SLOTS and df.loc[1, "resolution"] == NUM_SLOTS // 2
    print(f"  (6) B4 配線: OK  (位置あり h1 R²={df.loc[0, 'r2_all']:.3f} / "
          f"なし h2 R²={df.loc[1, 'r2_all']:+.3f})")


def main():
    print("clock_diagnostics のテスト")
    test_curve_comparison()
    test_first_onset()
    test_wrap()
    test_ridge_probe()
    test_shift_equivariance()
    test_position_probe_integration()
    print("\ntest_clock_diagnostics: OK")


if __name__ == "__main__":
    main()
