"""
LSTM_Aggregate の単体テスト

    a. 因果性      : スロット s の logits が a_{≥s} に依存しない。step を 96 回回すと forward と同じ logits になる
                     （既定の 1 層と、層を増やしたときのために 2 層でも確かめる。時刻符号 none / fixed / learned のすべて）
    b. 損失        : batch_loss が手で書いた交差エントロピーと一致する。既定は重みなしで、重みを変えても不変。
                     weighted=True は重み付きで、重みの定数倍で不変
    c. 属性        : cond_idx を変えると、最初と最後のスロットの logits が変わる
    d. 生成        : 同じ種で再現する。値域は [0, 12)。group_pool の行 d は cond_grid()[d] の条件。
                     write_pool_csv → load_sample_pool で往復する
    e. 保存と読み込み : 同じ構造・同じ出力が戻る（時刻符号の種類も戻る）。既定の構造（H = 64・1 層・learned）で
                     パラメータ数が 42,196（none 36,052・fixed 42,260）。保存先の名前（_weighted・_time_{K}。
                     既定の learned にも _time_learned が付く）
    f. weight decay の範囲 : param_groups は 1 次元のパラメータ（bias）に weight decay を掛けない
    g. epoch の平均 : 大きさの違うバッチに分けても、run_epoch は分割全体で 1 回測った batch_loss と一致する
                     （重みなしは人数、重み付きは Σw でバッチを平均する）
    h. 時刻符号    : none の state_dict のキーは時刻符号を入れる前と同じ。同じ種なら時刻以外の部品の初期値は
                     3 種類で一致する。fixed の φ は sm.time_features と一致する。時刻の部品に勾配が届く。
                     time_encoding の無い旧 ckpt は none として読む。未知の種類は ValueError
    i. 幅と weight decay : size_tag は既定で空、それ以外は _h{H}[_l{L}]_wd{W}。train に渡した幅・層数・weight decay が
                     モデルと config に届く（1 epoch）

    .venv/bin/python src/models/LSTM_Aggregate/test_model.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lm: Any = _load("lstm_aggregate_model", Path(__file__).resolve().parent / "model.py")
cur: Any = _load("lstm_test_stage2_curves",
                 REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage2_curves.py")
DEVICE = "cpu"          # 決定性のため CPU 固定
SMALL = 16              # テスト用の幅


def _model(seed: int = 0, num_layers: int = 1, time_enc: str = lm.TIME_ENC) -> Any:
    """小さな幅のモデル（初期値は PyTorch の既定。時刻符号の既定は採用した learned）"""
    torch.manual_seed(seed)
    return lm.LSTMScheduler(hidden=SMALL, num_layers=num_layers, time_enc=time_enc).to(DEVICE).eval()


def _inputs(batch: int = 6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(cond_idx (B, 3), sched (B, 96), weight (B,)) の乱数の入力"""
    g = torch.Generator().manual_seed(seed)
    grid = torch.as_tensor(lm.sm.cond_grid(), dtype=torch.long)
    cond = grid[torch.randint(0, lm.D_GROUPS, (batch,), generator=g)]
    sched = torch.randint(0, lm.NUM_ACT, (batch, lm.NUM_SLOTS), generator=g)
    weight = torch.rand(batch, generator=g) * 5000 + 100
    return cond, sched, weight


def test_causality() -> None:
    """a. スロット s の logits は a_{≥s} に依存しない。step の繰り返しは forward と一致する"""
    for time_enc in lm.TIME_ENC_KINDS:
        for num_layers in (1, 2):
            _check_causality(time_enc, num_layers)
    print(f"  a. 因果性・step と forward の一致（1 層・2 層 × {'/'.join(lm.TIME_ENC_KINDS)}）: OK")


def _check_causality(time_enc: str, num_layers: int) -> None:
    """test_causality の 1 つの構造の分

    Args:
        time_enc: 時刻符号の種類
        num_layers: LSTM の層数
    """
    m = _model(num_layers=num_layers, time_enc=time_enc)
    cond, sched, _ = _inputs()
    with torch.no_grad():
        logits = m(lm.gm.shift_right(sched), cond)
        k = 40
        changed = sched.clone()
        changed[:, k:] = (sched[:, k:] + 1) % lm.NUM_ACT
        logits2 = m(lm.gm.shift_right(changed), cond)
        assert torch.equal(logits[:, :k + 1], logits2[:, :k + 1]), f"{time_enc}: 未来の活動が logits に漏れている"
        assert not torch.allclose(logits[:, k + 1], logits2[:, k + 1]), f"{time_enc}: 直前の活動が届いていない"

        c_emb = m.embed_cond(cond)
        a_prev = lm.gm.shift_right(sched)
        state = None
        for s in range(lm.NUM_SLOTS):
            ls, state = m.step(a_prev[:, s], c_emb, state, s)
            assert torch.allclose(ls, logits[:, s], atol=1e-5), f"{time_enc}: step と forward がスロット {s} で違う"
    assert state is not None
    h, c = state
    assert h.shape == c.shape == (num_layers, cond.size(0), SMALL), f"状態の形が違う: {tuple(h.shape)}"


def test_loss() -> None:
    """b. batch_loss が手で書いた交差エントロピー（重みなし・重み付き）と一致する"""
    m = _model()
    cond, sched, weight = _inputs()
    with torch.no_grad():
        logits = m(lm.gm.shift_right(sched), cond)
        logp = F.log_softmax(logits, dim=-1).gather(-1, sched[..., None])[..., 0]    # (B, 96)

        # 既定（重みなし）: B × 96 スロットの単純平均。重みを変えても値は変わらない
        assert lm.WEIGHTED_LOSS is False
        plain = -logp.mean()
        got_plain = lm.batch_loss(m, cond, sched, weight)
        assert torch.allclose(got_plain, plain, rtol=1e-6), f"{float(got_plain)} != {float(plain)}"
        assert torch.allclose(lm.batch_loss(m, cond, sched, torch.rand_like(weight)), got_plain, rtol=1e-6)

        # 重み付き（--weighted-loss）: 重みの定数倍で不変（正規化が Σw で入っている）
        manual = -(weight[:, None] * logp).sum() / (lm.NUM_SLOTS * weight.sum())
        got = lm.batch_loss(m, cond, sched, weight, weighted=True)
        assert torch.allclose(got, manual, rtol=1e-6), f"{float(got)} != {float(manual)}"
        assert torch.allclose(lm.batch_loss(m, cond, sched, weight * 7.0, weighted=True), got, rtol=1e-6)
        assert not torch.allclose(got, got_plain), "重みが損失に効いていない"
    print(f"  b. 交差エントロピー（重みなし {float(got_plain):.4f}・重み付き {float(got):.4f} nats）: OK")


def test_condition() -> None:
    """c. 属性が最初と最後のスロットの logits に届く"""
    m = _model()
    cond, sched, _ = _inputs()
    other = cond.clone()
    other[:, 0] = 1 - other[:, 0]                       # 性だけを入れ替える（COND_SPEC の 0 列目、値は 0 か 1）
    a_prev = lm.gm.shift_right(sched)
    with torch.no_grad():
        l_cond = m(a_prev, cond)
        l_other = m(a_prev, other)
    assert not torch.allclose(l_cond[:, 0], l_other[:, 0]), "属性が s = 0 の logits に届いていない"
    assert not torch.allclose(l_cond[:, -1], l_other[:, -1]), "属性が s = 95 の logits に届いていない"
    print("  c. 属性が logits に届く: OK")


def test_sampling() -> None:
    """d. 再現性・値域・群別プールの並び・CSV の往復"""
    m = _model()
    cond, _, _ = _inputs(batch=8)
    a = lm.sample(m, cond, torch.Generator().manual_seed(3))
    b = lm.sample(m, cond, torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "同じ種で生成が再現しない"
    assert a.dtype == torch.long and a.shape == (8, lm.NUM_SLOTS)
    assert int(a.min()) >= 0 and int(a.max()) < lm.NUM_ACT

    pool = lm.group_pool(m, 2, seed=5)
    assert np.array_equal(pool, lm.group_pool(m, 2, seed=5))
    assert pool.shape == (lm.D_GROUPS, 2, lm.NUM_SLOTS) and pool.dtype == np.int64
    # 行 d は cond_grid()[d] の条件（28 × 2 = 56 本は GEN_BATCH 未満なので、1 回の sample と同じ乱数になる）
    flat = torch.as_tensor(lm.sm.cond_grid(), dtype=torch.long).repeat_interleave(2, dim=0)
    direct = lm.sample(m, flat, torch.Generator().manual_seed(5)).numpy()
    assert np.array_equal(pool.reshape(-1, lm.NUM_SLOTS), direct), "群の並びが cond_grid と違う"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pool.csv"
        lm.sm.write_pool_csv(pool, path)
        assert np.array_equal(cur.load_sample_pool(path), pool), "CSV の往復で壊れた"
    print("  d. 生成の再現性・値域・群別プールの並び・CSV の往復: OK")


def test_save_load() -> None:
    """e. 保存した ckpt から同じ構造・同じ出力が戻る。パラメータ数と保存先の名前"""
    m = _model(num_layers=2)
    cond, sched, _ = _inputs()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lstm.pt"
        lm.save_ckpt(m, path, seed=7, extra={"best_epoch": 3})
        ckpt = torch.load(path, map_location="cpu")
        assert set(ckpt) == {"model", "config"}
        cfg = ckpt["config"]
        assert (cfg["hidden"], cfg["num_layers"], cfg["seed"], cfg["best_epoch"]) == (SMALL, 2, 7, 3)
        back = lm.load_model(path, DEVICE)
    assert back.hidden == SMALL and back.num_layers == 2 and not back.training
    with torch.no_grad():
        assert torch.equal(back(lm.gm.shift_right(sched), cond), m(lm.gm.shift_right(sched), cond))

    # 時刻符号なし（H = 64・1 層）のパラメータ数。LSTM 本体は 4 ブロック × (H·H_in + H·H + 2H)
    full = lm.LSTMScheduler(time_enc="none")
    lstm_params = sum(p.numel() for p in full.lstm.parameters())
    assert lstm_params == 4 * (64 * 64 + 64 * 64 + 2 * 64) == 33_280, lstm_params
    assert lm.count_params(full) == 36_052, lm.count_params(full)
    assert lm.ckpt_path(42, time_enc="none").name == "lstm_aggregate.pt"
    assert lm.ckpt_path(43, time_enc="none").name == "lstm_aggregate_s43.pt"
    assert lm.pool_path(42, time_enc="none").name == "lstm_aggregate_samples.csv"
    assert lm.pool_path(43, time_enc="none").name == "lstm_aggregate_samples_s43.csv"
    assert lm.ckpt_path(43, weighted=True, time_enc="none").name == "lstm_aggregate_weighted_s43.pt"
    assert lm.pool_path(42, weighted=True, time_enc="none").name == "lstm_aggregate_samples_weighted.csv"

    # 既定は学習型の時刻符号（2026-10-06 採用）。既定の保存先にも _time_learned が付く
    assert lm.TIME_ENC == "learned" and lm.LSTMScheduler().time_enc == "learned"
    assert lm.ckpt_path(42).name == "lstm_aggregate_time_learned.pt"
    assert lm.pool_path(43).name == "lstm_aggregate_samples_time_learned_s43.csv"

    # 時刻符号: 種類が ckpt から戻り、出力も同じ。パラメータ数は fixed が +96·64+64、learned が +96·64
    for time_enc, n_params in (("fixed", 36_052 + 96 * 64 + 64), ("learned", 36_052 + 96 * 64)):
        mt = _model(time_enc=time_enc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lstm.pt"
            lm.save_ckpt(mt, path, seed=7)
            assert torch.load(path, map_location="cpu")["config"]["time_encoding"] == time_enc
            back = lm.load_model(path, DEVICE)
        assert back.time_enc == time_enc
        with torch.no_grad():
            assert torch.equal(back(lm.gm.shift_right(sched), cond), mt(lm.gm.shift_right(sched), cond))
        assert lm.count_params(lm.LSTMScheduler(time_enc=time_enc)) == n_params, time_enc
    assert lm.ckpt_path(42, time_enc="learned").name == "lstm_aggregate_time_learned.pt"
    assert lm.ckpt_path(43, weighted=True, time_enc="fixed").name == "lstm_aggregate_weighted_time_fixed_s43.pt"
    assert lm.pool_path(43, time_enc="fixed").name == "lstm_aggregate_samples_time_fixed_s43.csv"
    print("  e. 保存と読み込み・パラメータ数（none 36,052 / fixed 42,260 / learned 42,196）・保存先: OK")


def test_param_groups() -> None:
    """f. weight decay は重み行列だけに掛かり、1 次元のパラメータ（bias）には掛からない"""
    m = _model()
    decay, no_decay = lm.param_groups(m, 0.5)
    assert decay["weight_decay"] == 0.5 and no_decay["weight_decay"] == 0.0
    ids = {name: id(p) for name, p in m.named_parameters()}
    decay_ids = {id(p) for p in decay["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert decay_ids.isdisjoint(no_decay_ids) and decay_ids | no_decay_ids == set(ids.values()), "漏れか重複がある"
    for name in ("lstm.bias_ih_l0", "lstm.bias_hh_l0", "cond_proj.bias", "out_proj.bias"):
        assert ids[name] in no_decay_ids, f"{name} に weight decay が掛かっている"
    for name in ("lstm.weight_ih_l0", "lstm.weight_hh_l0", "act_embed.weight", "out_proj.weight"):
        assert ids[name] in decay_ids, f"{name} に weight decay が掛かっていない"

    # 時刻の部品: 重み行列（time_proj.weight・time_embed.weight）に掛かり、time_proj.bias には掛からない
    for time_enc, decayed, not_decayed in (("fixed", ["time_input.time_proj.weight"], ["time_input.time_proj.bias"]),
                                           ("learned", ["time_input.time_embed.weight"], [])):
        mt = _model(time_enc=time_enc)
        decay_t, no_decay_t = lm.param_groups(mt, 0.5)
        named = dict(mt.named_parameters())
        assert all(any(named[n] is p for p in decay_t["params"]) for n in decayed), time_enc
        assert all(any(named[n] is p for p in no_decay_t["params"]) for n in not_decayed), time_enc
    print("  f. weight decay の範囲（1 次元には掛けない。時刻の部品も同じ規則）: OK")


def test_time_encoding() -> None:
    """h. 時刻符号の部品の作り方・互換性・勾配"""
    models = {k: _model(seed=3, time_enc=k) for k in lm.TIME_ENC_KINDS}

    # none の state_dict のキーは、時刻の部品を持たない（時刻符号を入れる前と同じ）
    none_keys = set(models["none"].state_dict())
    assert models["none"].time_input is None and not any(k.startswith("time_input") for k in none_keys)
    assert set(models["fixed"].state_dict()) == none_keys | {"time_input.time_proj.weight", "time_input.time_proj.bias"}
    assert set(models["learned"].state_dict()) == none_keys | {"time_input.time_embed.weight"}

    # 同じ種なら、時刻以外の部品の初期値は 3 種類で一致する（time_input を最後に作るため）
    ref = models["none"].state_dict()
    for k in ("fixed", "learned"):
        sd = models[k].state_dict()
        assert all(torch.equal(sd[name], ref[name]) for name in none_keys), f"{k}: 時刻以外の初期値が none と違う"

    # fixed の φ は GRU_Aggregate と同じ Transformer 型 96 次元。保存しない buffer
    ti = models["fixed"].time_input
    assert torch.equal(ti.phi, lm.sm.time_features(lm.sm.ArchSpec(clock_kind="transformer")).T)
    assert ti.phi.shape == (lm.NUM_SLOTS, 96) and "time_input.phi" not in models["fixed"].state_dict()

    # 時刻の部品に勾配が届く。fixed・learned の時刻のベクトルは 96 スロットで互いに違う
    cond, sched, weight = _inputs()
    for k in ("fixed", "learned"):
        m = models[k].train()
        m.zero_grad()
        lm.batch_loss(m, cond, sched, weight).backward()
        grads = [p.grad for name, p in m.named_parameters() if name.startswith("time_input.")]
        assert grads and all(g is not None and float(g.abs().sum()) > 0 for g in grads), f"{k}: 勾配が届いていない"
        with torch.no_grad():
            vec = m.time_input(torch.arange(lm.NUM_SLOTS))                       # (96, H)
        assert vec.shape == (lm.NUM_SLOTS, SMALL)
        assert float(torch.cdist(vec, vec).add(torch.eye(lm.NUM_SLOTS) * 1e9).min()) > 0, f"{k}: 同じ時刻のベクトルがある"

    # 旧 ckpt（config に time_encoding が無い）は none として読む
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.pt"
        torch.save({"model": models["none"].state_dict(),
                    "config": {"hidden": SMALL, "num_layers": 1, "seed": 0}}, path)
        assert lm.load_model(path, DEVICE).time_enc == "none"

    # 未知の種類は ValueError
    for bad in ("sinusoidal", ""):
        try:
            lm.LSTMScheduler(hidden=SMALL, time_enc=bad)
        except ValueError:
            continue
        raise AssertionError(f"未知の time_enc={bad!r} が通った")
    print("  h. 時刻符号（キー・共通の初期値・φ・勾配・旧 ckpt・未知の種類）: OK")


def test_epoch_average() -> None:
    """g. バッチに分けて測った run_epoch が、全員を 1 回で測った batch_loss と一致する"""
    m = _model()
    cond, sched, weight = _inputs(batch=7)
    # 7 人を 3 人・3 人・1 人のバッチに分ける（大きさの違うバッチを平均する）
    loader = DataLoader(TensorDataset(cond, sched, weight), batch_size=3, shuffle=False)
    with torch.no_grad():
        for weighted in (False, True):
            whole = float(lm.batch_loss(m, cond, sched, weight, weighted))
            got = lm.run_epoch(m, loader, weighted=weighted)
            assert abs(got - whole) < 1e-6, f"weighted={weighted}: {got} != {whole}"
    print("  g. epoch の平均（重みなしは人数・重み付きは Σw）: OK")


def test_size_options() -> None:
    """i. 幅・層数・weight decay の保存先の印と、train から config までの受け渡し"""
    assert lm.size_tag() == ""
    assert lm.size_tag(128, 1, 0.01) == "_h128_wd0.01"
    assert lm.size_tag(128, 1, 1.0) == "_h128_wd1"
    assert lm.size_tag(64, 2, 1.0) == "_h64_l2_wd1"
    assert lm.ckpt_path(42, hidden=128, weight_decay=0.01).name == "lstm_aggregate_time_learned_h128_wd0.01.pt"
    assert lm.pool_path(43, hidden=128, weight_decay=0.01).name == "lstm_aggregate_samples_time_learned_h128_wd0.01_s43.csv"
    assert lm.ckpt_path(43, time_enc="none", hidden=128, weight_decay=0.01).name == "lstm_aggregate_h128_wd0.01_s43.pt"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lstm.pt"
        model, history = lm.train(epochs=1, save_path=path, device=DEVICE, hidden=SMALL, num_layers=1,
                                  weight_decay=0.01)
        cfg = torch.load(path, map_location="cpu")["config"]
        back = lm.load_model(path, DEVICE)
    assert len(history["val"]) == 1 and model.hidden == SMALL and back.hidden == SMALL
    assert (cfg["hidden"], cfg["num_layers"], cfg["weight_decay"], cfg["time_encoding"]) == (SMALL, 1, 0.01, lm.TIME_ENC)
    print("  i. 幅と weight decay（保存先の印・train から config まで）: OK")


if __name__ == "__main__":
    print("LSTM_Aggregate tests")
    test_causality()
    test_loss()
    test_condition()
    test_sampling()
    test_save_load()
    test_param_groups()
    test_epoch_average()
    test_time_encoding()
    test_size_options()
    print("test_model: OK")
