"""
VAE_Minimal（model.py・stage2_adapter.py）の単体テスト

    a. 大きさ: 生成側のパラメータ数が LSTM（学習型の時刻符号・H = 64）の ±1% に入る
    b. デコーダは全スロット共有: 時刻埋め込みの行を入れ替えると、出力のスロットが同じように入れ替わる
    c. ELBO: 再構成と KL は 0 以上。val の値は決定的で、大域の乱数を進めない
    d. 生成: group_pool は同じ種で同じプールを返し、形と値域が正しい
    e. adapter: sample_relaxed の前向きは硬い one-hot。集計の勾配はデコーダ・条件・時刻埋め込みに届き、エンコーダには届かない
    f. 共通の Stage 2 で回せて、途中の ckpt からの再開が止めずに回した結果とビット単位で一致する
       （VAE はリハーサルの雑音に大域の乱数を使うので、LSTM とは別に確かめる）

    .venv/bin/python src/models/VAE_Minimal/test_model.py
"""
import importlib.util
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


s2c: Any = _load("stage2_common", HERE.parent / "Stage2_Common" / "stage2_common.py")
vm: Any = _load("vae_minimal_model", HERE / "model.py")
va: Any = _load("stage2_adapter_vae_minimal", HERE / "stage2_adapter.py")
lm: Any = vm.lm
sm: Any = vm.sm


def _model(seed: int = 0) -> Any:
    """学習していない VAEMinimal（CPU）"""
    torch.manual_seed(seed)
    return vm.VAEMinimal().eval()


def _batch(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """無作為な条件と日誌"""
    g = torch.Generator().manual_seed(seed)
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    cond = grid[torch.randint(0, sm.D_GROUPS, (n,), generator=g)]
    sched = torch.randint(0, sm.NUM_ACT, (n, sm.NUM_SLOTS), generator=g)
    return cond, sched


def test_size() -> None:
    """a. 生成側のパラメータ数が LSTM の ±1% に入る"""
    n_gen = sum(p.numel() for p in _model().generator_parameters())
    n_lstm = lm.count_params(lm.LSTMScheduler(hidden=64, num_layers=1, time_enc="learned"))
    assert n_lstm == 42_196 and n_gen == 42_170, (n_lstm, n_gen)
    assert abs(n_gen - n_lstm) / n_lstm < 0.01
    print(f"a. 生成側 {n_gen:,}（LSTM {n_lstm:,}）: OK")


def test_shared_decoder() -> None:
    """b. 時刻埋め込みの行を入れ替えると、出力のスロットが同じように入れ替わる"""
    model = _model()
    cond, _ = _batch(3)
    z = torch.randn(3, vm.Z_DIM, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        c_emb = model.embed_cond(cond)
        base = model.decode(z, c_emb)
        perm = torch.randperm(sm.NUM_SLOTS, generator=torch.Generator().manual_seed(2))
        model.time_embed.weight.copy_(model.time_embed.weight[perm])
        moved = model.decode(z, c_emb)
    assert base.shape == (3, sm.NUM_SLOTS, sm.NUM_ACT)
    assert torch.allclose(moved, base[:, perm], atol=1e-6)
    print("b. デコーダは全スロット共有: OK")


def test_elbo() -> None:
    """c. 再構成・KL が 0 以上。val は決定的で大域の乱数を進めない"""
    model = _model()
    cond, sched = _batch(32)
    loss, recon, kl = vm.elbo(model, cond, sched)
    assert recon.item() >= 0 and kl.item() >= 0 and abs(loss.item() - (recon + vm.BETA * kl).item()) < 1e-3
    train_part, val_part = vm.gm.load_split()
    loader = vm.gm.make_loader(val_part, shuffle=False)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    v1 = vm.val_loss(model, loader)
    torch.set_rng_state(state)
    v2 = vm.val_loss(model, loader)
    assert v1 == v2, (v1, v2)
    # val の DataLoader は iterator を作るときに大域の乱数を 1 つ使う（shuffle なしでも）。eps は使わない
    torch.set_rng_state(state)
    iter(loader)
    after_iter = torch.get_rng_state()
    torch.set_rng_state(state)
    vm.val_loss(model, loader)
    assert torch.equal(torch.get_rng_state(), after_iter), "val の雑音が大域の乱数を使っている"
    print("c. ELBO・val の決定性: OK")


def test_group_pool() -> None:
    """d. 同じ種で同じプール。形と値域"""
    model = _model()
    p1, p2 = vm.group_pool(model, 3, seed=5), vm.group_pool(model, 3, seed=5)
    assert p1.shape == (sm.D_GROUPS, 3, sm.NUM_SLOTS) and np.array_equal(p1, p2)
    assert 0 <= p1.min() and p1.max() < sm.NUM_ACT
    print("d. 生成のプール: OK")


def test_adapter_gradient() -> None:
    """e. 前向きは硬い one-hot。集計の勾配はデコーダ・条件・時刻埋め込みに届き、エンコーダには届かない"""
    adapter = va.VAEAdapter(_model(), seed=42)
    cond, _ = _batch(8)
    y = adapter.sample_relaxed(cond, torch.Generator().manual_seed(0), 1.0)
    assert y.shape == (8, sm.NUM_ACT, sm.NUM_SLOTS)
    assert torch.equal(y.detach(), y.detach().round()) and torch.equal(y.detach().sum(dim=1), torch.ones(8, sm.NUM_SLOTS))
    (y * torch.randn_like(y)).sum().backward()
    model = adapter.model
    for name, p in model.named_parameters():
        reached = p.grad is not None and float(p.grad.abs().sum()) > 0
        assert reached == (not name.startswith("enc_")), f"{name}: 勾配が届いた = {reached}"
    print("e. adapter の硬い one-hot と勾配の届く先: OK")


def test_common_loop_resume() -> None:
    """f. 共通の Stage 2 で回し、step 3 から再開した結果が止めずに回した結果とビット単位で一致する"""
    tgt = s2c.st.load_stula_targets()
    every = np.ones(sm.D_GROUPS, dtype=bool)
    settings = s2c.Stage2Settings(steps=6, d_sub=2, n=8, val_every=2, save_every=3)
    with tempfile.TemporaryDirectory() as tmp:
        full_dir, part_dir = Path(tmp) / "full", Path(tmp) / "part"
        full = va.VAEAdapter(_model(), seed=42)
        full_hist = s2c.train_stage2(full, tgt, every, 42, settings, out_dir=full_dir)
        part_dir.mkdir()
        shutil.copy(s2c.ck.ckpt_path(full_dir, 3), s2c.ck.ckpt_path(part_dir, 3))
        torch.manual_seed(999)
        resumed = va.VAEAdapter(_model(), seed=42)
        hist = s2c.train_stage2(resumed, tgt, every, 42, settings, out_dir=part_dir, resume=True)
        sa, sb = full.model.state_dict(), resumed.model.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa), "再開: 重みが違う"
        keys = [k for k in full_hist[-1] if k != "sec"]
        assert all(r1.get(k) == r2.get(k) for r1, r2 in zip(full_hist, hist) for k in keys), "再開: 記録が違う"
    print("f. 共通の Stage 2 で回せる・再開がビット単位で一致: OK")


if __name__ == "__main__":
    test_size()
    test_shared_decoder()
    test_elbo()
    test_group_pool()
    test_adapter_gradient()
    test_common_loop_resume()
    print("all OK")
