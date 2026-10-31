"""
共通の Stage 2（stage2_common.py）の単体テスト

    a. straight-through Gumbel: 共通の関数が LSTM_Aggregate/stage2_agg.py の同名の関数と同じ値を返す
    b. 既存の LSTM の Stage 2 と同じ計算: lr を 3 群とも 1e-3 にした stage2_agg.train_stage2 と、重みがビット単位で一致し、
       損失の記録も一致する（LSTM と GRU_Minimal の両方。val を毎 step 測り、乱数の消費の順序まで確かめる）
    c. 再開: 途中の ckpt から再開した結果が、止めずに回した結果とビット単位で一致する
       （epoch の途中から再開する場合と、epoch をちょうど使い切った step から再開する場合）
    d. 上書きしない: resume なしで ckpt が既にあると止まる。設定が違う ckpt からは再開しない
    e. 生成の前向きは硬い one-hot（LSTM と GRU_Minimal）
    f. 保存名: 変種の印は λ と計算場所を常に書き、既定と違う設定だけを足す
    g. CLI の --lr: 設定に渡り、既定と違うときだけ保存名に _lr{値} が付く

    .venv/bin/python src/models/Stage2_Common/test_stage2_common.py
"""
import dataclasses
import importlib.util
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s2c: Any = _load("stage2_common", HERE / "stage2_common.py")
s2: Any = _load("lstm_stage2_agg", HERE.parent / "LSTM_Aggregate" / "stage2_agg.py")
sm: Any = s2c.sm
SEED = 42
TGT: dict = s2c.st.load_stula_targets()
EVERY = np.ones(sm.D_GROUPS, dtype=bool)
EXACT_KEYS = ("L_agg", "L_reh", "L_reh_val", "rate_mae", "rate_mse")      # ビット単位で一致すべき記録


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    """2 つのモデルの全パラメータ・バッファがビット単位で同じか"""
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _same_history(h1: list[dict[str, float]], h2: list[dict[str, float]], keys: tuple[str, ...]) -> bool:
    """記録の keys が全 step で同じ値か（どちらにも無い step は飛ばす）"""
    if len(h1) != len(h2):
        return False
    for r1, r2 in zip(h1, h2):
        for k in keys:
            if (k in r1) != (k in r2) or (k in r1 and r1[k] != r2[k]):
                return False
    return True


def test_gumbel() -> None:
    """a. 共通の straight_through_gumbel が LSTM の Stage 2 の同名の関数と同じ値を返す"""
    logits = torch.randn(64, sm.NUM_ACT, generator=torch.Generator().manual_seed(0))
    y1, a1 = s2c.straight_through_gumbel(logits, torch.Generator().manual_seed(7), 1.0)
    y2, a2 = s2.straight_through_gumbel(logits, torch.Generator().manual_seed(7), 1.0)
    assert torch.equal(y1, y2) and torch.equal(a1, a2)
    print("a. straight-through Gumbel が LSTM の Stage 2 と一致: OK")


def test_same_as_lstm_stage2(cell: str) -> None:
    """b. lr を 3 群とも 1e-3 にした既存の Stage 2 と、重み・損失の記録がビット単位で一致する"""
    steps, d_sub, n = 5, 2, 8
    old_model = s2.load_pretrained(SEED, cell, "cpu")
    old_hist = s2.train_stage2(old_model, TGT, EVERY, SEED, lam=0.01, steps=steps, d_sub=d_sub, n=n, tau=1.0,
                               bptt=0, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-3, weight_decay=0.0, out_dir=None,
                               val_every=1, cell=cell, agg_loss="mse")
    adapter = s2c.load_adapter(cell, SEED, "cpu")
    new_hist = s2c.train_stage2(adapter, TGT, EVERY, SEED,
                                s2c.Stage2Settings(steps=steps, d_sub=d_sub, n=n, lam=0.01, val_every=1))
    assert _same_weights(old_model, adapter.model), f"{cell}: 重みが既存の Stage 2 と違う"
    assert _same_history(old_hist, new_hist, EXACT_KEYS), f"{cell}: 損失の記録が既存の Stage 2 と違う"
    for r_old, r_new in zip(old_hist, new_hist):
        # パラメータの並びが違うので、ノルムと綱引きの量は足す順序の分だけ末尾の桁がずれうる（記録だけの量）
        for k in ("clip_norm", "tug_cos", "tug_ratio"):
            assert math.isclose(r_old[k], r_new[k], rel_tol=1e-6), (cell, k, r_old[k], r_new[k])
    print(f"b. 既存の LSTM の Stage 2 とビット単位で一致（{cell}、{steps} step、val 毎 step）: OK")


def test_resume_exact() -> None:
    """c. 途中の ckpt から再開した結果が、止めずに回した結果とビット単位で一致する"""
    settings = s2c.Stage2Settings(steps=16, d_sub=1, n=4, val_every=3, save_every=7)
    n_batches = len(s2c.gm.make_loader(s2c.gm.load_split()[0], shuffle=True))
    assert n_batches == 14, f"ATUS の学習分割のミニバッチ数が想定（14）と違う: {n_batches}"
    with tempfile.TemporaryDirectory() as tmp:
        full_dir = Path(tmp) / "full"
        full = s2c.load_adapter("lstm", SEED, "cpu")
        full_hist = s2c.train_stage2(full, TGT, EVERY, SEED, settings, out_dir=full_dir)
        # step 7 は epoch の途中、step 14 は epoch をちょうど使い切った位置（1 step に 1 バッチ）
        for stop in (7, 14):
            part_dir = Path(tmp) / f"from{stop}"
            part_dir.mkdir()
            shutil.copy(s2c.ck.ckpt_path(full_dir, stop), s2c.ck.ckpt_path(part_dir, stop))
            torch.manual_seed(999)                                # 再開が乱数を ckpt から戻すことを確かめる
            resumed = s2c.load_adapter("lstm", SEED, "cpu")
            hist = s2c.train_stage2(resumed, TGT, EVERY, SEED, settings, out_dir=part_dir, resume=True)
            assert _same_weights(full.model, resumed.model), f"step {stop} からの再開: 重みが違う"
            keys = tuple(k for k in full_hist[-1] if k != "sec")
            assert _same_history(full_hist, hist, keys), f"step {stop} からの再開: 記録が違う"
    print("c. 再開（epoch の途中・epoch の終わり）が止めずに回した結果とビット単位で一致: OK")


def test_refuse_overwrite() -> None:
    """d. resume なしで ckpt があると止まる。設定が違う ckpt からは再開しない"""
    settings = s2c.Stage2Settings(steps=2, d_sub=1, n=4, val_every=0, save_every=1)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        s2c.train_stage2(s2c.load_adapter("lstm", SEED, "cpu"), TGT, EVERY, SEED, settings, out_dir=out)
        try:
            s2c.train_stage2(s2c.load_adapter("lstm", SEED, "cpu"), TGT, EVERY, SEED, settings, out_dir=out)
            raise AssertionError("resume なしで上書きした")
        except FileExistsError:
            pass
        other = s2c.Stage2Settings(steps=3, d_sub=1, n=4, val_every=0, save_every=1)
        try:
            s2c.train_stage2(s2c.load_adapter("lstm", SEED, "cpu"), TGT, EVERY, SEED, other, out_dir=out, resume=True)
            raise AssertionError("設定が違う ckpt から再開した")
        except ValueError:
            pass
    print("d. 上書きしない・設定が違う ckpt から再開しない: OK")


def test_hard_one_hot(cell: str) -> None:
    """e. sample_relaxed の前向きは硬い one-hot で、勾配はモデルに届く"""
    adapter = s2c.load_adapter(cell, SEED, "cpu")
    cond = torch.as_tensor(sm.cond_grid(), dtype=torch.long)[:4]
    y = adapter.sample_relaxed(cond, torch.Generator().manual_seed(0), 1.0)
    assert y.shape == (4, sm.NUM_ACT, sm.NUM_SLOTS)
    assert torch.equal(y.detach(), y.detach().round()) and torch.equal(y.detach().sum(dim=1), torch.ones(4, sm.NUM_SLOTS))
    (y * torch.randn_like(y)).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in adapter.model.parameters())
    print(f"e. 生成の前向きは硬い one-hot（{cell}）: OK")


def test_variant_tag() -> None:
    """f. 変種の印"""
    assert s2c.Stage2Settings().variant_tag("cpu") == "_lam0.01_cpu"
    assert s2c.Stage2Settings(lam=0.003, steps=300).variant_tag("mps") == "_lam0.003_steps300_mps"
    assert s2c.Stage2Settings(lr=1e-4, val_every=5).variant_tag("cuda") == "_lam0.01_lr0.0001_val5_cuda"
    assert s2c.Stage2Settings(save_every=5, log_every=1).variant_tag("cpu") == "_lam0.01_cpu"   # 結果に効かない設定
    adapter = s2c.load_adapter("lstm", SEED, "cpu")
    assert s2c.stem(adapter, "", 42, "zeroshot") == "stage2_common_lstm_time_learned_h64_wd0.01_seed42_zeroshot"
    assert s2c.stem(adapter, "_lam0.01_cpu", 43, "all") == "stage2_common_lstm_time_learned_h64_wd0.01_lam0.01_cpu_seed43_all"
    print("f. 保存名: OK")


def test_cli_lr() -> None:
    """g. CLI の --lr"""
    ap = s2c.build_parser()
    default = s2c.settings_from_args(ap.parse_args(["--model", "ddpm_minimal"]))
    assert default == s2c.Stage2Settings() and default.variant_tag("cuda") == "_lam0.01_cuda"
    lowered = s2c.settings_from_args(ap.parse_args(["--model", "ddpm_minimal", "--lr", "1e-4"]))
    assert lowered.lr == 1e-4 and lowered.variant_tag("cuda") == "_lam0.01_lr0.0001_cuda"
    assert dataclasses.replace(lowered, lr=s2c.Stage2Settings().lr) == s2c.Stage2Settings()   # lr 以外は本番の既定
    print("g. CLI の --lr: OK")


if __name__ == "__main__":
    test_gumbel()
    for cell_ in ("lstm", "gru_minimal"):
        test_same_as_lstm_stage2(cell_)
        test_hard_one_hot(cell_)
    test_resume_exact()
    test_refuse_overwrite()
    test_variant_tag()
    test_cli_lr()
    print("all OK")
