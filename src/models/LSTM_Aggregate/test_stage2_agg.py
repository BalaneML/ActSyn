"""
LSTM_Aggregate の Stage 2（stage2_agg.py）の単体テスト

    a. straight-through Gumbel: 前向きは argmax(logits + g) の厳密な one-hot。引いた頻度は softmax(logits) に一致する
    b. 微分できる生成: sample_soft の各スロットの logits は、引いた系列を teacher forcing に通した logits と一致する
       （ソフトな入力 y @ act_embed.weight が、前向きでは act_embed(a) と同じ値）。y の argmax は引いた活動
    c. 履歴を通る勾配: 既定（bptt = 0）では活動の埋め込み（BOS 以外の 12 行）に勾配が届き、
       bptt = 1（毎スロットで履歴を切る）では届かない。L_agg の勾配は cond・time・rest の 3 群すべてに届く
    d. 層別学習率の群: すべてのパラメータがちょうど 1 つの群に入る。時刻符号 none では time 群を作らない。
       weight decay > 0 では 2 次元以上だけに減衰が掛かり、bias は {群}_bias（減衰 0）に入る
    e. 保存先: Stage 1 の ckpt の名前、変種の印、E0 の CSV が変種によらないこと。GRU は頭が stage2_gru_minimal。
       計算場所の印は mps だけ空で、cpu・cuda は変種の最後に付く
    f. 評価の JSD: 損失側の sl.jsd_loss と一致し、同じ分布で 0 になる
    b〜d は LSTM と GRU（--cell gru_minimal）の両方で確かめる

    .venv/bin/python src/models/LSTM_Aggregate/test_stage2_agg.py
"""
import importlib.util
import math
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


def _model(seed: int = 0, time_enc: str = "learned", cell: str = "lstm") -> Any:
    """小さな幅のモデル（CPU）"""
    torch.manual_seed(seed)
    return s2.CELL_CLASSES[cell](hidden=SMALL, num_layers=1, time_enc=time_enc).eval()


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


def test_sample_matches_teacher_forcing(cell: str) -> None:
    """b. sample_soft の logits = teacher forcing の logits"""
    model = _model(cell=cell)
    cond = _cond(32)
    with torch.no_grad():
        y, acts, logits = s2.sample_soft(model, cond, torch.Generator().manual_seed(2))
        tf = model(gm.shift_right(acts), cond)
    assert y.shape == (32, sm.NUM_ACT, sm.NUM_SLOTS) and acts.shape == (32, sm.NUM_SLOTS)
    assert torch.equal(y.argmax(dim=1), acts)
    err = (logits - tf).abs().max().item()
    assert err < 1e-5, f"teacher forcing と食い違う: {err:.2e}"
    print(f"b. 微分できる生成 = teacher forcing（{cell}）: OK（最大差 {err:.1e}）")


def _act_grad(bptt: int, cell: str) -> torch.Tensor:
    """群平均の行動者率の線形な損失を逆伝播し、活動の埋め込みの勾配を返す"""
    model = _model(cell=cell)
    cond = _cond(16)
    y, _, _ = s2.sample_soft(model, cond, torch.Generator().manual_seed(3), bptt=bptt)
    w = torch.randn(sm.NUM_ACT, sm.NUM_SLOTS, generator=torch.Generator().manual_seed(4))
    (y.mean(dim=0) * w).sum().backward()
    assert model.act_embed.weight.grad is not None
    return model.act_embed.weight.grad


def test_history_gradient(cell: str) -> None:
    """c. 既定では履歴を通り、bptt = 1 では切れる。L_agg の勾配は 3 群すべてに届く"""
    full, cut = _act_grad(0, cell), _act_grad(1, cell)
    assert full[:sm.NUM_ACT].abs().sum() > 0, "既定で活動の埋め込みに勾配が届かない"
    assert cut[:sm.NUM_ACT].abs().sum() == 0, "bptt = 1 なのに活動の埋め込みに勾配が届く"
    assert full[lm.BOS].abs().sum() > 0 and cut[lm.BOS].abs().sum() > 0, "BOS の行に勾配が届かない"

    model = _model(cell=cell)
    n = 8
    grid = torch.as_tensor(sm.cond_grid(), dtype=torch.long)
    cond = grid[:2].repeat_interleave(n, dim=0)
    a_star = torch.rand(2, sm.NUM_ACT, sm.NUM_SLOTS, generator=torch.Generator().manual_seed(5))
    a_star = a_star / a_star.sum(dim=1, keepdim=True)
    optimizer = s2.build_optimizer(model)
    optimizer.zero_grad()
    y, _, _ = s2.sample_soft(model, cond, torch.Generator().manual_seed(6))
    a_hat_1, a_hat_2 = sl.group_rates_split(y, n)
    sl.agg_loss_from_rates(a_hat_1, a_hat_2, a_star, torch.ones_like(a_star)).backward()
    norms = s2.ft.grad_norms(optimizer, "agg")
    assert set(norms) == {"agg_gnorm_cond", "agg_gnorm_time", "agg_gnorm_rest"}
    assert all(v > 0 for v in norms.values()), f"勾配が届かない群がある: {norms}"
    print(f"c. 履歴を通る勾配（{cell}）: OK")


def test_param_groups(cell: str) -> None:
    """d. すべてのパラメータがちょうど 1 つの群に入る"""
    for time_enc, names in (("learned", ["cond", "time", "rest"]), ("none", ["cond", "rest"])):
        model = _model(time_enc=time_enc, cell=cell)
        groups = s2.split_param_groups(model)
        assert list(groups) == names, f"{time_enc}: 群が {list(groups)}"
        ids = [id(p) for ps in groups.values() for p in ps]
        assert len(ids) == len(set(ids)) == len(list(model.parameters())), f"{time_enc}: 重複か漏れがある"
    model = _model(cell=cell)
    groups = s2.split_param_groups(model)
    assert {id(p) for p in groups["time"]} == {id(p) for p in model.time_input.parameters()}
    assert {id(p) for p in groups["cond"]} == {id(p) for m in (model.cond_embeds, model.cond_proj)
                                               for p in m.parameters()}
    optimizer = s2.build_optimizer(model, weight_decay=0.01)
    names = [g["name"] for g in optimizer.param_groups]
    assert names == ["cond", "cond_bias", "time", "rest", "rest_bias"], f"weight decay の群が {names}"
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())), "weight decay の群に重複か漏れがある"
    for g in optimizer.param_groups:
        decayed = g["weight_decay"] > 0
        assert all((p.ndim >= 2) == decayed for p in g["params"]), f"{g['name']}: 減衰の範囲が違う"
    assert [g["name"] for g in s2.build_optimizer(model).param_groups] == ["cond", "time", "rest"]
    print(f"d. 層別学習率の群（{cell}）: OK")


def test_paths() -> None:
    """e. 保存先の名前"""
    assert s2.stage1_ckpt(42).name == "lstm_aggregate_time_learned_h64_wd0.01.pt"
    assert s2.stage1_ckpt(43).name == "lstm_aggregate_time_learned_h64_wd0.01_s43.pt"
    # split は 2026-10-12 までの結果と同じ名前（印なし）、既定の mse は _mse
    assert s2.variant_tag(0.01) == "_lam0.01_mse"
    assert s2.variant_tag(0.01, agg_loss="split") == "_lam0.01"
    assert s2.variant_tag(0.003, tau=0.5, bptt=8, agg_loss="split") == "_lam0.003_tau0.5_bptt8"
    assert (s2.variant_tag(0.01, steps=1000, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-3, agg_loss="split")
            == "_lam0.01_lr0.001_steps1000")
    assert (s2.variant_tag(0.01, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-4, agg_loss="split")
            == "_lam0.01_lr0.001-0.001-0.0001")
    assert (s2.variant_tag(0.01, steps=1000, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-3, weight_decay=0.01,
                           agg_loss="split") == "_lam0.01_lr0.001_steps1000_s2wd0.01")
    assert (s2.variant_tag(0.01, steps=1000, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-3, weight_decay=0.01)
            == "_lam0.01_mse_lr0.001_steps1000_s2wd0.01")
    assert s2.csv_path(42, "zeroshot", "_lam0.01") == s2.csv_path(42, "zeroshot", "_lam0.1")
    assert s2.csv_path(43, "fold3", "_lam0.01").name == "stage2_lstm_time_learned_h64_wd0.01_lam0.01_s43_fold3.csv"
    assert s2.run_dir(42, "all", "_lam0.01").name == "stage2_lstm_time_learned_h64_wd0.01_lam0.01_all"
    # GRU: Stage 1 は GRU_Minimal の ckpt、保存先の頭は stage2_gru_minimal（GRU_Aggregate の stage2_gru_ と分ける）
    assert s2.stage1_ckpt(42, "gru_minimal").name == "gru_minimal_time_learned_h64_wd0.01.pt"
    assert s2.stage1_ckpt(44, "gru_minimal").name == "gru_minimal_time_learned_h64_wd0.01_s44.pt"
    assert (s2.csv_path(43, "fold3", "_lam0.01", "gru_minimal").name
            == "stage2_gru_minimal_time_learned_h64_wd0.01_lam0.01_s43_fold3.csv")
    assert s2.csv_path(42, "zeroshot", "_lam0.01", "gru_minimal") != s2.csv_path(42, "zeroshot", "_lam0.01")
    assert s2.device_tag("mps") == "" and s2.device_tag("cpu") == "_cpu" and s2.device_tag("cuda") == "_cuda"
    assert s2.variant_tag(0.01, device="mps") == "_lam0.01_mse"
    assert (s2.variant_tag(0.01, steps=1000, lr_cond=1e-3, lr_time=1e-3, lr_rest=1e-3, agg_loss="split", device="cpu")
            == "_lam0.01_lr0.001_steps1000_cpu")
    assert (s2.run_dir(43, "fold2", s2.variant_tag(0.01, device="cpu")).name
            == "stage2_lstm_time_learned_h64_wd0.01_lam0.01_mse_cpu_s43_fold2")
    try:
        s2.device_tag("tpu")
        raise AssertionError("未知の計算場所で例外が出ない")
    except ValueError:
        pass
    print("e. 保存先: OK")


def test_agg_mse() -> None:
    """g. 素の MSE の集計損失: 群平均と手計算の二乗誤差に一致し、split の半分平均の平均とも一致する"""
    gen = torch.Generator().manual_seed(7)
    d_sub, n = 3, 6
    acts = torch.randint(0, sm.NUM_ACT, (d_sub * n, sm.NUM_SLOTS), generator=gen)
    y = F.one_hot(acts, sm.NUM_ACT).permute(0, 2, 1).float()                  # (d_sub·n, 12, 96)
    a_star = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS, generator=gen)
    omega = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS, generator=gen)
    a_hat = sl.group_rates(y, n)
    a_hat_1, a_hat_2 = sl.group_rates_split(y, n)
    assert torch.allclose(a_hat, 0.5 * (a_hat_1 + a_hat_2))
    ref = sum(float(((y[d * n:(d + 1) * n].mean(dim=0) - a_star[d]) ** 2 * omega[d]).sum())
              for d in range(d_sub)) / a_star.numel()
    assert math.isclose(float(sl.agg_mse_from_rates(a_hat, a_star, omega)), ref, rel_tol=1e-5)
    assert float(sl.agg_mse_from_rates(a_hat, a_star, omega)) >= 0
    print("g. 素の MSE の集計損失: OK")


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
    for cell_ in s2.CELLS:
        test_sample_matches_teacher_forcing(cell_)
        test_history_gradient(cell_)
        test_param_groups(cell_)
    test_paths()
    test_slot_jsd()
    test_agg_mse()
    print("all OK")
