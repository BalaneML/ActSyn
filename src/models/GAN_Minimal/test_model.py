"""
GAN_Minimal（model.py・stage2_adapter.py）の単体テスト

    a. 大きさ: 生成器のパラメータ数が VAE のデコーダと同じ 42,170（LSTM 42,196 の ±1%）
    b. 生成: generate_hard の前向きは硬い one-hot。同じ generator の種なら同じ値
    c. 更新の分離: critic_step は critic だけを動かす。generator_loss の勾配は生成器にだけ届き、critic には溜まらない
    d. val: Wasserstein 推定は決定的で、雑音のために大域の乱数を進めない
    e. ckpt: 生成器・critic・critic の optimizer が保存して読み戻せる
    f. 共通の Stage 2 で回せて（critic も auxiliary_step で学習を続ける）、途中の ckpt からの再開が止めずに回した結果と
       ビット単位で一致する（critic の重みと optimizer も一致）

    .venv/bin/python src/models/GAN_Minimal/test_model.py
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
gg: Any = _load("gan_minimal_model", HERE / "model.py")
ga: Any = _load("stage2_adapter_gan_minimal", HERE / "stage2_adapter.py")
sm: Any = gg.sm


def _nets(seed: int = 0) -> tuple[Any, Any]:
    """学習していない生成器と critic（CPU）"""
    torch.manual_seed(seed)
    return gg.GANGenerator(), gg.Critic()


def _adapter(seed: int = 0) -> Any:
    """学習していない GANAdapter（critic の optimizer は新品）"""
    gen, critic = _nets(seed)
    return ga.GANAdapter(gen, critic, gg.make_optimizer(critic.parameters()).state_dict(), "final", 42)


def _cond(n: int) -> torch.Tensor:
    """群の条件を n 行"""
    return torch.as_tensor(sm.cond_grid(), dtype=torch.long)[torch.arange(n) % sm.D_GROUPS]


def test_size() -> None:
    """a. 生成器のパラメータ数"""
    gen, _ = _nets()
    assert sum(p.numel() for p in gen.parameters()) == 42_170
    print("a. 生成器 42,170: OK")


def test_generate_hard() -> None:
    """b. 硬い one-hot・同じ種で同じ値"""
    gen, _ = _nets()
    y1 = gg.generate_hard(gen, _cond(5), torch.Generator().manual_seed(1))
    y2 = gg.generate_hard(gen, _cond(5), torch.Generator().manual_seed(1))
    assert y1.shape == (5, sm.NUM_SLOTS, sm.NUM_ACT) and torch.equal(y1, y2)
    assert torch.equal(y1.detach(), y1.detach().round()) and torch.equal(y1.detach().sum(dim=2), torch.ones(5, sm.NUM_SLOTS))
    print("b. 生成は硬い one-hot・再現性: OK")


def test_update_separation() -> None:
    """c. critic_step は critic だけ、generator_loss の勾配は生成器だけ"""
    gen, critic = _nets()
    opt_d = gg.make_optimizer(critic.parameters())
    g_before = {k: v.clone() for k, v in gen.state_dict().items()}
    d_before = {k: v.clone() for k, v in critic.state_dict().items()}
    sched = torch.randint(0, sm.NUM_ACT, (8, sm.NUM_SLOTS), generator=torch.Generator().manual_seed(2))
    log = gg.critic_step(gen, critic, opt_d, _cond(8), sched)
    assert all(np.isfinite(v) for v in log.values()) and log["gp"] >= 0
    assert all(torch.equal(v, g_before[k]) for k, v in gen.state_dict().items()), "critic_step が生成器を動かした"
    assert any(not torch.equal(v, d_before[k]) for k, v in critic.state_dict().items()), "critic が動いていない"
    for p in critic.parameters():
        p.grad = None
    gg.generator_loss(gen, critic, _cond(8)).backward()
    assert all(p.grad is None for p in critic.parameters()), "生成器の損失の勾配が critic に溜まった"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in gen.parameters())
    print("c. 更新の分離: OK")


def test_val_deterministic() -> None:
    """d. val の Wasserstein 推定は決定的で、大域の乱数は DataLoader の分しか進めない"""
    gen, critic = _nets()
    _, val_part = gg.gm.load_split()
    loader = gg.gm.make_loader(val_part, shuffle=False)
    torch.manual_seed(5)
    state = torch.get_rng_state()
    v1 = gg.val_wasserstein(gen, critic, loader)
    torch.set_rng_state(state)
    v2 = gg.val_wasserstein(gen, critic, loader)
    assert v1 == v2
    torch.set_rng_state(state)
    iter(loader)
    after_iter = torch.get_rng_state()
    torch.set_rng_state(state)
    gg.val_wasserstein(gen, critic, loader)
    assert torch.equal(torch.get_rng_state(), after_iter)
    print("d. val の決定性: OK")


def test_ckpt_roundtrip() -> None:
    """e. 生成器・critic・critic の optimizer を保存して読み戻せる"""
    gen, critic = _nets()
    opt_d = gg.make_optimizer(critic.parameters())
    gg.critic_step(gen, critic, opt_d, _cond(4), torch.zeros(4, sm.NUM_SLOTS, dtype=torch.long))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "g.pt"
        gg.save_ckpt(gg.snapshot(gen, critic, opt_d), path, 42)
        gen2, critic2, opt_state = gg.load_ckpt(path)
    assert all(torch.equal(a, b) for a, b in zip(gen.state_dict().values(), gen2.state_dict().values()))
    assert all(torch.equal(a, b) for a, b in zip(critic.state_dict().values(), critic2.state_dict().values()))
    assert opt_state["state"].keys() == opt_d.state_dict()["state"].keys()
    print("e. ckpt の保存と読み戻し: OK")


def test_common_loop_resume() -> None:
    """f. 共通の Stage 2 で回し、step 2 から再開した結果が止めずに回した結果とビット単位で一致する"""
    tgt = s2c.st.load_stula_targets()
    every = np.ones(sm.D_GROUPS, dtype=bool)
    settings = s2c.Stage2Settings(steps=4, d_sub=1, n=4, val_every=2, save_every=2)
    with tempfile.TemporaryDirectory() as tmp:
        full_dir, part_dir = Path(tmp) / "full", Path(tmp) / "part"
        full = _adapter()
        full_hist = s2c.train_stage2(full, tgt, every, 42, settings, out_dir=full_dir)
        part_dir.mkdir()
        shutil.copy(s2c.ck.ckpt_path(full_dir, 2), s2c.ck.ckpt_path(part_dir, 2))
        torch.manual_seed(999)
        resumed = _adapter()
        hist = s2c.train_stage2(resumed, tgt, every, 42, settings, out_dir=part_dir, resume=True)
        for a, b in ((full.model, resumed.model), (full.critic, resumed.critic)):
            sa, sb = a.state_dict(), b.state_dict()
            assert all(torch.equal(sa[k], sb[k]) for k in sa), "再開: 重みが違う"
        oa, ob = full.opt_d.state_dict()["state"], resumed.opt_d.state_dict()["state"]
        assert all(torch.equal(oa[i]["exp_avg"], ob[i]["exp_avg"]) for i in oa), "再開: critic の optimizer が違う"
        keys = [k for k in full_hist[-1] if k != "sec"]
        assert all(r1.get(k) == r2.get(k) for r1, r2 in zip(full_hist, hist) for k in keys), "再開: 記録が違う"
        assert "critic_w_est" in full_hist[-1]
    print("f. 共通の Stage 2 で回せる（critic も学習）・再開がビット単位で一致: OK")


if __name__ == "__main__":
    test_size()
    test_generate_hard()
    test_update_separation()
    test_val_deterministic()
    test_ckpt_roundtrip()
    test_common_loop_resume()
    print("all OK")
