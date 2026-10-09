"""
GRU_Minimal の単体テスト（観点 a〜f は LSTM_Aggregate/test_model.py と同じ）

    a. 因果性      : スロット s の logits が a_{≥s} に依存しない。lm.sample が使う step を 96 回回すと forward と
                     同じ logits になる（既定の 1 層と、2 層でも確かめる。時刻符号 none / fixed / learned のすべて）
    b. 損失        : lm.batch_loss（既定は重みなし）が手で書いた交差エントロピーと一致する
    c. 属性        : cond_idx を変えると、最初と最後のスロットの logits が変わる
    d. 生成        : lm.sample・lm.group_pool で、同じ種なら再現する。値域は [0, 12)。group_pool の行 d は
                     cond_grid()[d] の条件。write_pool_csv → load_sample_pool で往復する
    e. 保存と読み込み : lm.save_ckpt → load_model で同じ構造・同じ出力が戻る（時刻符号の種類も戻る）。
                     既定の構造（H = 64・1 層・learned）でパラメータ数が 33,876（none 27,732・fixed 33,940）。保存先の名前
    f. weight decay の範囲 : lm.param_groups は 1 次元のパラメータ（bias）に weight decay を掛けない
    g. 条件が同じ  : 定数が lm と同じ値。state_dict のキーと形は、再帰の層（lstm.* と gru.*）を除いて
                     LSTMScheduler と同じ（時刻符号 3 種類のすべて）。lm.train に渡すと GRUMinimalScheduler が
                     学習され、時刻符号の種類・幅・weight decay が config に残る。同じ種なら時刻以外の初期値は 3 種類で一致する。
                     保存先の印は lm.size_tag と同じ（_h{H}_wd{W}）

    .venv/bin/python src/models/GRU_Minimal/test_model.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gmin: Any = _load("gru_minimal_model", Path(__file__).resolve().parent / "model.py")
lm: Any = gmin.lm
cur: Any = _load("gru_minimal_test_stage2_curves",
                 REPO_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage2_curves.py")
DEVICE = "cpu"          # 決定性のため CPU 固定
SMALL = 16              # テスト用の幅


def _model(seed: int = 0, num_layers: int = 1, time_enc: str = lm.TIME_ENC) -> Any:
    """小さな幅のモデル（初期値は PyTorch の既定。時刻符号の既定は採用した learned）"""
    torch.manual_seed(seed)
    return gmin.GRUMinimalScheduler(hidden=SMALL, num_layers=num_layers, time_enc=time_enc).to(DEVICE).eval()


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
        num_layers: GRU の層数
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
    assert isinstance(state, torch.Tensor), "GRU の状態は h だけ（LSTM のような組ではない）"
    assert state.shape == (num_layers, cond.size(0), SMALL), f"状態の形が違う: {tuple(state.shape)}"


def test_loss() -> None:
    """b. lm.batch_loss（既定は重みなし）が手で書いた交差エントロピーと一致する"""
    m = _model()
    cond, sched, weight = _inputs()
    with torch.no_grad():
        logits = m(lm.gm.shift_right(sched), cond)
        logp = F.log_softmax(logits, dim=-1).gather(-1, sched[..., None])[..., 0]    # (B, 96)
        got = lm.batch_loss(m, cond, sched, weight)
        assert torch.allclose(got, -logp.mean(), rtol=1e-6), f"{float(got)} != {float(-logp.mean())}"
        manual_w = -(weight[:, None] * logp).sum() / (lm.NUM_SLOTS * weight.sum())
        assert torch.allclose(lm.batch_loss(m, cond, sched, weight, weighted=True), manual_w, rtol=1e-6)
    print(f"  b. 交差エントロピー（重みなし {float(got):.4f} nats）: OK")


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
    """d. lm の生成の関数で、再現性・値域・群別プールの並び・CSV の往復"""
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
        path = Path(tmp) / "gru_minimal.pt"
        lm.save_ckpt(m, path, seed=7, extra={"best_epoch": 3})
        cfg = torch.load(path, map_location="cpu")["config"]
        assert (cfg["hidden"], cfg["num_layers"], cfg["seed"], cfg["best_epoch"]) == (SMALL, 2, 7, 3)
        back = gmin.load_model(path, DEVICE)
    assert isinstance(back, gmin.GRUMinimalScheduler) and not back.training
    assert back.hidden == SMALL and back.num_layers == 2
    with torch.no_grad():
        assert torch.equal(back(lm.gm.shift_right(sched), cond), m(lm.gm.shift_right(sched), cond))

    # 時刻符号なし（H = 64・1 層）のパラメータ数。GRU 本体は 3 ブロック × (H·H_in + H·H + 2H)
    full = gmin.GRUMinimalScheduler(time_enc="none")
    gru_params = sum(p.numel() for p in full.gru.parameters())
    assert gru_params == 3 * (64 * 64 + 64 * 64 + 2 * 64) == 24_960, gru_params
    assert lm.count_params(full) == 27_732, lm.count_params(full)
    assert gmin.ckpt_path(42, time_enc="none").name == "gru_minimal.pt"
    assert gmin.ckpt_path(43, time_enc="none").name == "gru_minimal_s43.pt"
    assert gmin.pool_path(42, time_enc="none").name == "gru_minimal_samples.csv"
    assert gmin.pool_path(43, weighted=True, time_enc="none").name == "gru_minimal_samples_weighted_s43.csv"
    # 既定は学習型の時刻符号（2026-10-06 採用）。既定の保存先にも _time_learned が付く
    assert gmin.GRUMinimalScheduler().time_enc == "learned"
    assert gmin.ckpt_path(42).name == "gru_minimal_time_learned.pt"

    # 時刻符号: 種類が ckpt から戻り、出力も同じ。パラメータ数は fixed が +96·64+64、learned が +96·64
    for time_enc, n_params in (("fixed", 27_732 + 96 * 64 + 64), ("learned", 27_732 + 96 * 64)):
        mt = _model(time_enc=time_enc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gru_minimal.pt"
            lm.save_ckpt(mt, path, seed=7)
            back = gmin.load_model(path, DEVICE)
        assert back.time_enc == time_enc
        with torch.no_grad():
            assert torch.equal(back(lm.gm.shift_right(sched), cond), mt(lm.gm.shift_right(sched), cond))
        assert lm.count_params(gmin.GRUMinimalScheduler(time_enc=time_enc)) == n_params, time_enc
    assert gmin.ckpt_path(43, time_enc="learned").name == "gru_minimal_time_learned_s43.pt"
    assert gmin.pool_path(42, time_enc="fixed").name == "gru_minimal_samples_time_fixed.csv"
    print("  e. 保存と読み込み・パラメータ数（none 27,732 / fixed 33,940 / learned 33,876）・保存先: OK")


def test_param_groups() -> None:
    """f. weight decay は重み行列だけに掛かり、1 次元のパラメータ（bias）には掛からない"""
    m = _model()
    decay, no_decay = lm.param_groups(m, 0.5)
    ids = {name: id(p) for name, p in m.named_parameters()}
    decay_ids = {id(p) for p in decay["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert decay_ids.isdisjoint(no_decay_ids) and decay_ids | no_decay_ids == set(ids.values()), "漏れか重複がある"
    for name in ("gru.bias_ih_l0", "gru.bias_hh_l0", "cond_proj.bias", "out_proj.bias"):
        assert ids[name] in no_decay_ids, f"{name} に weight decay が掛かっている"
    for name in ("gru.weight_ih_l0", "gru.weight_hh_l0", "act_embed.weight", "out_proj.weight"):
        assert ids[name] in decay_ids, f"{name} に weight decay が掛かっていない"
    print("  f. weight decay の範囲（1 次元には掛けない）: OK")


def test_same_conditions() -> None:
    """g. セル以外の条件が LSTM と同じ"""
    # 定数は lm から取っている（写し書きしていない）
    for name in ("NUM_ACT", "HIDDEN", "NUM_LAYERS", "EPOCHS", "SEED", "WEIGHTED_LOSS", "DEVICE", "TIME_ENC",
                 "WEIGHT_DECAY"):
        assert getattr(gmin, name) == getattr(lm, name), f"{name} が LSTM と違う"
    assert (gmin.HIDDEN, gmin.NUM_LAYERS, gmin.WEIGHTED_LOSS, gmin.TIME_ENC) == (64, 1, False, "learned")

    # state_dict のキーと形は、再帰の層を除いて同じ。再帰の層はキーの名前（lstm / gru）だけが違う
    for time_enc in lm.TIME_ENC_KINDS:
        lstm_sd = lm.LSTMScheduler(time_enc=time_enc).state_dict()
        gru_sd = gmin.GRUMinimalScheduler(time_enc=time_enc).state_dict()
        lstm_rest = {k: tuple(v.shape) for k, v in lstm_sd.items() if not k.startswith("lstm.")}
        gru_rest = {k: tuple(v.shape) for k, v in gru_sd.items() if not k.startswith("gru.")}
        assert lstm_rest == gru_rest, f"{time_enc}: 再帰の層のほかに違う部品がある"
        lstm_rnn = {k.removeprefix("lstm.") for k in lstm_sd if k.startswith("lstm.")}
        gru_rnn = {k.removeprefix("gru.") for k in gru_sd if k.startswith("gru.")}
        assert lstm_rnn == gru_rnn, f"再帰の層のパラメータの名前が違う: {lstm_rnn ^ gru_rnn}"

    # 同じ種なら、時刻以外の部品の初期値は 3 種類で一致する（time_input を最後に作るため）
    ref = _model(seed=3, time_enc="none").state_dict()
    for time_enc in ("fixed", "learned"):
        sd = _model(seed=3, time_enc=time_enc).state_dict()
        assert all(torch.equal(sd[k], v) for k, v in ref.items()), f"{time_enc}: 時刻以外の初期値が none と違う"

    # lm.train に渡すと GRUMinimalScheduler が学習され、損失の設定が config に残る（1 epoch、保存先は一時）
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gru_minimal.pt"
        model, history = gmin.train(epochs=1, save_path=path, device=DEVICE)
        cfg = torch.load(path, map_location="cpu")["config"]
    assert isinstance(model, gmin.GRUMinimalScheduler) and len(history["val"]) == 1
    assert cfg["arch"] == "GRUMinimalScheduler" and cfg["weighted_loss"] is False
    assert cfg["time_encoding"] == "learned" and cfg["weight_decay"] == lm.WEIGHT_DECAY   # 既定（採用した構成）

    # 明示した時刻符号の種類・幅・weight decay も lm.train を通ってモデルと config に届く
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gru_minimal_none.pt"
        model, _ = gmin.train(epochs=1, save_path=path, device=DEVICE, time_enc="none", hidden=SMALL,
                              weight_decay=0.01)
        cfg = torch.load(path, map_location="cpu")["config"]
    assert model.time_enc == "none" and cfg["time_encoding"] == "none"
    assert (model.hidden, cfg["hidden"], cfg["weight_decay"]) == (SMALL, SMALL, 0.01)
    assert gmin.ckpt_path(43, hidden=128, weight_decay=0.01).name == "gru_minimal_time_learned_h128_wd0.01_s43.pt"
    assert gmin.pool_path(42, hidden=128, weight_decay=0.01).name == "gru_minimal_samples_time_learned_h128_wd0.01.csv"
    print("  g. 条件が LSTM と同じ（定数・部品・学習の手順・時刻符号の初期値）: OK")


if __name__ == "__main__":
    print("GRU_Minimal tests")
    test_causality()
    test_loss()
    test_condition()
    test_sampling()
    test_save_load()
    test_param_groups()
    test_same_conditions()
    print("test_model: OK")
