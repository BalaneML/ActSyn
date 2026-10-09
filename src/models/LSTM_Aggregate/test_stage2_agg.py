"""
LSTM_Aggregate の Stage 2（stage2_agg.py）の単体テスト

    a. straight-through Gumbel: 前向きは argmax(logits + g) の厳密な one-hot。引いた頻度は softmax(logits) に一致する
    b. 微分できる生成: sample_soft の各スロットの logits は、引いた系列を teacher forcing に通した logits と一致する
       （ソフトな入力 y @ act_embed.weight が、前向きでは act_embed(a) と同じ値）。y の argmax は引いた活動
    c. 履歴を通る勾配: 既定（bptt = 0）では活動の埋め込み（BOS 以外の 12 行）に勾配が届き、
       bptt = 1（毎スロットで履歴を切る）では届かない。L_agg の勾配は cond・time・rest の 3 群すべてに届く
    d. 層別学習率の群: すべてのパラメータがちょうど 1 つの群に入る。時刻符号 none では time 群を作らない
    e. 保存先: Stage 1 の ckpt の名前、変種の印、E0 の CSV が変種によらないこと

    .venv/bin/python src/models/LSTM_Aggregate/test_stage2_agg.py
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s2: Any = _load("lstm_stage2_agg", Path(__file__).resolve().parent / "stage2_agg.py")
lm: Any = s2.lm
gm: Any = s2.gm
sm: Any = s2.sm
sl: Any = s2.sl
SMALL = 16              # テスト用の幅


def _model(seed: int = 0, time_enc: str = "learned") -> Any:
    """小さな幅のモデル（CPU）"""
    torch.manual_seed(seed)
    return lm.LSTMScheduler(hidden=SMALL, num_layers=1, time_enc=time_enc).eval()


def _cond(batch: int, seed: int = 0) -> torch.Tensor:
    """群の条件を batch 行（cond_grid から無作為に）"""
    g = torch.Generator().manual_seed(seed)
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    return grid[torch.randint(0, sm.D_GROUPS, (batch,), generator=g)]


def test_gumbel() -> None:
    """a. 前向きは厳密な one-hot、頻度は softmax に一致する"""
    logits = torch.tensor([[1.0, 0.0, -1.0, 2.0] + [-3.0] * (sm.NUM_ACT - 4)])
    big = logits.expand(40000, -1).contiguous().requires_grad_(True)
    y, a = s2.straight_through_gumbel(big, torch.Generator().manual_seed(1))
    assert torch.equal(y.detach(), F.one_hot(a, sm.NUM_ACT).float()), "前向きが one-hot でない"
    freq = y.detach().mean(dim=0)
    prob = F.softmax(logits[0], dim=-1)
    assert (freq - prob).abs().max() < 0.01, f"頻度が softmax と合わない: {freq} vs {prob}"
    # 後ろ向きは softmax の微分（勾配が 0 でない）
    (y[:, 0]).sum().backward()
    assert big.grad is not None and big.grad.abs().sum() > 0
    print("a. straight-through Gumbel: OK")


def test_sample_matches_teacher_forcing() -> None:
    """b. sample_soft の logits = teacher forcing の logits"""
    model = _model()
    cond = _cond(32)
    with torch.no_grad():
        y, acts, logits = s2.sample_soft(model, cond, torch.Generator().manual_seed(2))
        tf = model(gm.shift_right(acts), cond)
    assert y.shape == (32, sm.NUM_ACT, sm.NUM_SLOTS) and acts.shape == (32, sm.NUM_SLOTS)
    assert torch.equal(y.argmax(dim=1), acts)
    err = (logits - tf).abs().max().item()
    assert err < 1e-5, f"teacher forcing と食い違う: {err:.2e}"
    print(f"b. 微分できる生成 = teacher forcing: OK（最大差 {err:.1e}）")


def _act_grad(bptt: int) -> torch.Tensor:
    """群平均の行動者率の線形な損失を逆伝播し、活動の埋め込みの勾配を返す"""
    model = _model()
    cond = _cond(16)
    y, _, _ = s2.sample_soft(model, cond, torch.Generator().manual_seed(3), bptt=bptt)
    w = torch.randn(sm.NUM_ACT, sm.NUM_SLOTS, generator=torch.Generator().manual_seed(4))
    (y.mean(dim=0) * w).sum().backward()
    assert model.act_embed.weight.grad is not None
    return model.act_embed.weight.grad


def test_history_gradient() -> None:
    """c. 既定では履歴を通り、bptt = 1 では切れる。L_agg の勾配は 3 群すべてに届く"""
    full, cut = _act_grad(0), _act_grad(1)
    assert full[:sm.NUM_ACT].abs().sum() > 0, "既定で活動の埋め込みに勾配が届かない"
    assert cut[:sm.NUM_ACT].abs().sum() == 0, "bptt = 1 なのに活動の埋め込みに勾配が届く"
    assert full[lm.BOS].abs().sum() > 0 and cut[lm.BOS].abs().sum() > 0, "BOS の行に勾配が届かない"

    model = _model()
    n = 8
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    cond = grid[:2].repeat_interleave(n, dim=0)
    a_star = torch.rand(2, sm.NUM_ACT, sm.NUM_SLOTS, generator=torch.Generator().manual_seed(5))
    a_star = a_star / a_star.sum(dim=1, keepdim=True)
    optimizer = s2.build_optimizer(model)
    optimizer.zero_grad()
    y, _, _ = s2.sample_soft(model, cond, torch.Generator().manual_seed(6))
    a_A, a_B = sl.group_rates_split(y, n)
    sl.agg_loss_from_rates(a_A, a_B, a_star, torch.ones_like(a_star)).backward()
    norms = s2.ft.grad_norms(optimizer, "agg")
    assert set(norms) == {"agg_gnorm_cond", "agg_gnorm_time", "agg_gnorm_rest"}
    assert all(v > 0 for v in norms.values()), f"勾配が届かない群がある: {norms}"
    print("c. 履歴を通る勾配: OK")


def test_param_groups() -> None:
    """d. すべてのパラメータがちょうど 1 つの群に入る"""
    for time_enc, names in (("learned", ["cond", "time", "rest"]), ("none", ["cond", "rest"])):
        model = _model(time_enc=time_enc)
        groups = s2.split_param_groups(model)
        assert list(groups) == names, f"{time_enc}: 群が {list(groups)}"
        ids = [id(p) for ps in groups.values() for p in ps]
        assert len(ids) == len(set(ids)) == len(list(model.parameters())), f"{time_enc}: 重複か漏れがある"
    model = _model()
    groups = s2.split_param_groups(model)
    assert {id(p) for p in groups["time"]} == {id(p) for p in model.time_input.parameters()}
    assert {id(p) for p in groups["cond"]} == {id(p) for m in (model.cond_embeds, model.cond_proj)
                                               for p in m.parameters()}
    print("d. 層別学習率の群: OK")


def test_paths() -> None:
    """e. 保存先の名前"""
    assert s2.stage1_ckpt(42).name == "lstm_aggregate_time_learned_h64_wd0.01.pt"
    assert s2.stage1_ckpt(43).name == "lstm_aggregate_time_learned_h64_wd0.01_s43.pt"
    assert s2.variant_tag(0.01) == "_lam0.01"
    assert s2.variant_tag(0.003, tau=0.5, bptt=8) == "_lam0.003_tau0.5_bptt8"
    assert s2.variant_tag(0.01, steps=1000, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-3) == "_lam0.01_lr0.001_steps1000"
    assert s2.variant_tag(0.01, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-4) == "_lam0.01_lr0.001-0.001-0.0001"
    assert s2.csv_path(42, "zeroshot", "_lam0.01") == s2.csv_path(42, "zeroshot", "_lam0.1")
    assert s2.csv_path(43, "fold3", "_lam0.01").name == "stage2_lstm_time_learned_h64_wd0.01_lam0.01_s43_fold3.csv"
    assert s2.run_dir(42, "all", "_lam0.01").name == "stage2_lstm_time_learned_h64_wd0.01_lam0.01_all"
    print("e. 保存先: OK")


def test_slot_jsd() -> None:
    """f. 評価の JSD が損失側の sl.jsd_loss と一致し、同じ分布で 0 になる"""
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(sm.NUM_ACT), size=(sm.D_GROUPS, sm.NUM_SLOTS)).transpose(0, 2, 1)   # (28, 12, 96)
    g = rng.dirichlet(np.ones(sm.NUM_ACT), size=(sm.D_GROUPS, sm.NUM_SLOTS)).transpose(0, 2, 1)
    a[0, 3, :] = 0.0                                       # 行動者率 0 のセル（0 log 0 = 0）
    a[0] /= a[0].sum(axis=0, keepdims=True)
    every = np.ones(sm.D_GROUPS, dtype=bool)
    assert s2.slot_jsd(a, a, every) == 0.0
    ref = float(s2.sl.jsd_loss(torch.as_tensor(g, dtype=torch.float64), torch.as_tensor(a, dtype=torch.float64), 1))
    assert abs(s2.slot_jsd(g, a, every) - ref) < 1e-9
    assert 0.0 < s2.slot_jsd(g, a, every) < np.log(2.0)
    print("f. 時刻別行動者率の JSD: OK")


if __name__ == "__main__":
    test_gumbel()
    test_sample_matches_teacher_forcing()
    test_history_gradient()
    test_param_groups()
    test_paths()
    test_slot_jsd()
    print("all OK")
