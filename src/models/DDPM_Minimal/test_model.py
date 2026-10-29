"""
DDPM_Minimal（model.py・stage2_adapter.py）の単体テスト（CPU で回る大きさ）

    a. 大きさと写しの分離: 幅 8・dropout 0 のパラメータ数 68,860。書き換えたのは別名の写し（dm.base）の定数だけで、
       ほかの実験が使う DDPM_Aggregate_Simple のモジュールは幅 64・dropout 0.1 のまま
    b. 時刻埋め込みを 0 にすると、同じ重みの既存の UNet1D（時刻符号なし）と同じ出力になる（_encode の書き換えの検算）
    c. 損失: val は決定的で、t と ε のために大域の乱数を進めない
    d. Stage 2 の生成: 前向きは硬い one-hot で、同じ雑音で勾配なしに回した逆過程の argmax と一致する（評価の生成と同じ標本）。
       勾配はモデルに届く
    e. 共通の Stage 2 で回せて、途中の ckpt からの再開が止めずに回した結果とビット単位で一致する

    .venv/bin/python src/models/DDPM_Minimal/test_model.py
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
REPO_ROOT = HERE.parents[2]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s2c: Any = _load("stage2_common", HERE.parent / "Stage2_Common" / "stage2_common.py")
dm: Any = _load("ddpm_minimal_model", HERE / "model.py")
da: Any = _load("stage2_adapter_ddpm_minimal", HERE / "stage2_adapter.py")
sm: Any = dm.sm


def _model(seed: int = 0) -> Any:
    """学習していない DDPMMinimal（CPU）"""
    torch.manual_seed(seed)
    return dm.DDPMMinimal().eval()


def test_size_and_isolation() -> None:
    """a. パラメータ数と、定数を書き換えたのが写しだけであること"""
    assert dm.count_params(_model()) == 68_860
    assert dm.base.BASE_CH == 8 and dm.base.DROPOUT == 0.0
    shared = s2c.sm                                         # ほかの実験と同じ名前で読み込まれた DDPM_Aggregate_Simple
    assert shared is not dm.base and shared.BASE_CH == 64 and shared.DROPOUT == 0.1
    print("a. 大きさ 68,860・写しの分離: OK")


def test_matches_base_unet() -> None:
    """b. 時刻埋め込みを 0 にすると既存の UNet1D と同じ出力"""
    model = _model()
    ref = dm.base.UNet1D(arch=dm.base.ArchSpec()).eval()
    ref.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("slot_embed")})
    with torch.no_grad():
        for p in model.parameters():                        # 零初期化の出力層を動かして、比較を意味のあるものにする
            p.add_(0.01 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
        ref.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("slot_embed")})
        model.slot_embed.weight.zero_()
        g = torch.Generator().manual_seed(3)
        x = torch.randn(4, sm.NUM_ACT, sm.NUM_SLOTS, generator=g)
        t = torch.randint(0, dm.T_STEPS, (4,), generator=g)
        cond = torch.as_tensor(sm.cond_grid(), dtype=torch.long)[:4]
        assert torch.allclose(model(x, t, cond), ref(x, t, cond), atol=1e-6)
    print("b. 時刻埋め込み 0 で既存の UNet1D と一致: OK")


def test_val_deterministic() -> None:
    """c. val は決定的で、大域の乱数は DataLoader の分しか進めない"""
    model = _model()
    diffusion = dm.base.Diffusion("cpu")
    _, val_part = dm.gm.load_split()
    loader = dm.gm.make_loader(val_part, shuffle=False)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    v1 = dm.val_loss(model, diffusion, loader)
    torch.set_rng_state(state)
    v2 = dm.val_loss(model, diffusion, loader)
    assert v1 == v2
    torch.set_rng_state(state)
    iter(loader)
    after_iter = torch.get_rng_state()
    torch.set_rng_state(state)
    dm.val_loss(model, diffusion, loader)
    assert torch.equal(torch.get_rng_state(), after_iter)
    print("c. val の決定性: OK")


def test_relaxed_matches_sampler() -> None:
    """d. Stage 2 の生成の前向きが、同じ雑音の勾配なしの逆過程の argmax と一致し、勾配がモデルに届く"""
    model = _model()
    diffusion = dm.base.Diffusion("cpu")
    cond = torch.as_tensor(sm.cond_grid(), dtype=torch.long)[:3]
    y = dm.sample_relaxed_chain(model, diffusion, cond, torch.Generator().manual_seed(9), 1.0)
    assert y.shape == (3, sm.NUM_ACT, sm.NUM_SLOTS)
    assert torch.equal(y.detach(), y.detach().round()) and torch.equal(y.detach().sum(dim=1), torch.ones(3, sm.NUM_SLOTS))
    gen = dm.device_generator(torch.Generator().manual_seed(9), torch.device("cpu"))
    shape = (3, sm.NUM_ACT, sm.NUM_SLOTS)
    with torch.no_grad():
        x = torch.randn(shape, generator=gen)
        for ti in reversed(range(dm.T_STEPS)):
            z = None if ti == 0 else torch.randn(shape, generator=gen)
            x = diffusion._reverse_step(model, x, ti, cond, dm.GUIDANCE, z)
    assert torch.equal(y.detach().argmax(dim=1), x.argmax(dim=1))
    (y * torch.randn_like(y)).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    print("d. Stage 2 の生成が評価の生成と同じ標本・勾配が届く: OK")


def test_common_loop_resume() -> None:
    """e. 共通の Stage 2 で回し、step 2 から再開した結果が止めずに回した結果とビット単位で一致する"""
    tgt = s2c.st.load_stula_targets()
    every = np.ones(sm.D_GROUPS, dtype=bool)
    settings = s2c.Stage2Settings(steps=4, d_sub=1, n=2, val_every=2, save_every=2)
    with tempfile.TemporaryDirectory() as tmp:
        full_dir, part_dir = Path(tmp) / "full", Path(tmp) / "part"
        full = da.DDPMAdapter(_model(), seed=42, device="cpu")
        full_hist = s2c.train_stage2(full, tgt, every, 42, settings, out_dir=full_dir)
        part_dir.mkdir()
        shutil.copy(s2c.ck.ckpt_path(full_dir, 2), s2c.ck.ckpt_path(part_dir, 2))
        torch.manual_seed(999)
        resumed = da.DDPMAdapter(_model(), seed=42, device="cpu")
        hist = s2c.train_stage2(resumed, tgt, every, 42, settings, out_dir=part_dir, resume=True)
        sa, sb = full.model.state_dict(), resumed.model.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa), "再開: 重みが違う"
        keys = [k for k in full_hist[-1] if k != "sec"]
        assert all(r1.get(k) == r2.get(k) for r1, r2 in zip(full_hist, hist) for k in keys), "再開: 記録が違う"
    print("e. 共通の Stage 2 で回せる・再開がビット単位で一致: OK")


if __name__ == "__main__":
    test_size_and_isolation()
    test_matches_base_unet()
    test_val_deterministic()
    test_relaxed_matches_sampler()
    test_common_loop_resume()
    print("all OK")
