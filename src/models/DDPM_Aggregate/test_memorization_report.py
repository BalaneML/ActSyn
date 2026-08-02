"""
memorization_report（暗記チェック）の単体テスト。

この診断は「生成個票が学習個票のコピーになっていないか」を最近傍距離で測る。
素朴に組むと **参照集合のサイズ差で交絡する**: 最近傍距離は参照集合が大きいほど
自然に小さくなるので、train(N=3363) と holdout(N=373) を素で比べると、暗記が
無くても DCR_gap が正に出る。model.memorization_report はこれを避けるために
train を holdout と同数に間引いてから比べ、さらに train 内の互いに素な部分集合で
床（帰無帯）を作る。この構造が壊れると、指標は動くのに意味を失う。

検証するのは3点:

    1. サイズ交絡の存在  : 実ホールドアウト個票（暗記があり得ない）でも、
                          参照集合を間引かないと DCR がずれること。間引き処理が
                          必要である根拠そのもの
    2. 暗記の検出        : 学習個票をそのまま並べた生成物を、床の外（上）として
                          検出し、exact_copy_rate=1.0 を返すこと
    3. 偽陽性を出さない  : 学習データと無関係な一様乱数を「暗記」と言わないこと
                          （床の中に収まること）

    uv run python src/models/DDPM_Aggregate/test_memorization_report.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "common" / "preprocess" / "stula"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


agg = _load("memtest_ddpm_agg", REPO_ROOT / "src" / "models" / "DDPM_Aggregate" / "model.py")
im = _load("memtest_individual_metrics", REPO_ROOT / "src" / "eval" / "individual_metrics.py")

_, SCHED, _, _ = agg.load_data()
TRAIN_IDX, VAL_IDX = agg.split_indices(len(SCHED))
TRAIN, HOLDOUT = SCHED[TRAIN_IDX], SCHED[VAL_IDX]


def test_size_confound_exists():
    """(1) 参照集合のサイズ差だけで DCR が動くこと。間引きが要る根拠。

    「生成物」の位置に実ホールドアウト個票を置く。暗記は原理的にありえないので、
    ここで出る差はすべてサイズ由来。
    """
    rng = np.random.default_rng(0)
    sub = TRAIN[rng.choice(len(TRAIN), len(HOLDOUT), replace=False)]
    d_full = im.nn_distances(HOLDOUT, TRAIN, k=1, sample=2000)[:, 0].mean()
    d_sub = im.nn_distances(HOLDOUT, sub, k=1, sample=2000)[:, 0].mean()
    print(f"  (1) サイズ交絡: train全体(N={len(TRAIN)}) DCR={d_full:.3f}  "
          f"間引き(N={len(sub)}) DCR={d_sub:.3f}  差={d_sub - d_full:+.3f}")
    assert d_sub > d_full + 1.0, \
        "参照集合を間引いても DCR が動かない。この前提が崩れるとテスト2/3の解釈も変わる"
    print("  (1) サイズ交絡の存在: OK")


def test_detects_memorization():
    """(2) 学習個票のコピーを暗記として検出すること。"""
    rng = np.random.default_rng(1)
    gen = TRAIN[rng.choice(len(TRAIN), 2000)]
    m = agg.memorization_report(gen, SCHED)
    assert m["exact_copy_rate[train_full]"] == 1.0, m["exact_copy_rate[train_full]"]
    assert m["memorized"], \
        f"暗記を検出できていない (gap={m['DCR_gap(holdout-train)']:.4f} " \
        f"床[{m['DCR_gap_null_lo']:.4f}, {m['DCR_gap_null_hi']:.4f}])"
    print("  (2) 暗記の検出: OK")


def test_no_false_positive():
    """(3) 学習データと無関係な生成物を暗記と言わないこと。"""
    rng = np.random.default_rng(2)
    gen = rng.integers(0, agg.NUM_ACT, size=(2000, agg.NUM_SLOTS))
    m = agg.memorization_report(gen, SCHED)
    assert m["exact_copy_rate[train_full]"] == 0.0
    assert not m["memorized"], \
        f"無関係な生成物を暗記と誤判定 (gap={m['DCR_gap(holdout-train)']:.4f} " \
        f"床[{m['DCR_gap_null_lo']:.4f}, {m['DCR_gap_null_hi']:.4f}])"
    print("  (3) 偽陽性を出さない: OK")


if __name__ == "__main__":
    print(f"memorization_report のテスト (N={len(SCHED)}, "
          f"train={len(TRAIN)}, holdout={len(HOLDOUT)})")
    test_size_confound_exists()
    test_detects_memorization()
    test_no_false_positive()
    print("\ntest_memorization_report: OK")
