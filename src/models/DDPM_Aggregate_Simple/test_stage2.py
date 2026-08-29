"""
Stage 2 基盤の単体テスト（Stage2_design.md §10.3）。

対象は §10.2 の実装 1〜2:
    1. 定期チェックポイントと再開            stage2_checkpoint.py
    2. 逆過程1ステップの切り出し             model.Diffusion._reverse_step

検証する内容:
    (ckpt) save_ckpt -> load_ckpt の往復で model/optimizer/step/RNG が戻る。
           一時ファイルが残らない。latest_ckpt が step を数値順で選ぶ
    (f2)   _reverse_step が旧インライン式と厳密一致し、smoke_test が通る

★ 出口の零初期化について:
    UNet1D は out_conv を零初期化するので、そのままでは勾配の大きさが測れない。
    test_backbone.py と同じ _wake_up() で out_conv だけを小さな乱数で埋めてから測る。

    .venv/bin/python3 src/models/DDPM_Aggregate_Simple/test_stage2.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 動的ロードしたモジュールは型チェッカから中身が見えないので Any で受ける
sm: Any = _load("simple_model", HERE / "model.py")
ck: Any = _load("simple_stage2_checkpoint", HERE / "stage2_checkpoint.py")
DEVICE = "cpu"          # テストは決定性重視で CPU 固定


def _wake_up(model: Any, seed: int = 0) -> Any:
    """零初期化された out_conv を小さな乱数で埋める（勾配を測れるようにするため）。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.out_conv.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def _model(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return _wake_up(sm.UNet1D().to(DEVICE), seed).eval()


def _cond(n_groups: int = 3) -> torch.Tensor:
    return torch.as_tensor(sm.cond_grid()[:n_groups], device=DEVICE)


# ============================================================
# 1. チェックポイント
# ============================================================
def test_checkpoint_roundtrip() -> None:
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # optimizer に中身を持たせる（モーメントが空だと復元の検証にならない）
    loss = model(torch.randn(2, sm.IN_CH, sm.NUM_SLOTS),
                 torch.zeros(2, dtype=torch.long), _cond(2)).square().mean()
    loss.backward()
    opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        path = ck.ckpt_path(d, 250)
        assert path.name == "stage2_step250.pt"
        rng_at_save = torch.get_rng_state()   # ★save_ckpt が payload に入れるのはこの状態
        ck.save_ckpt(path, model, opt, 250, {"K": 1, "n": 256})
        assert path.exists()
        assert not path.with_suffix(".tmp").exists(), "一時ファイルが残っている"
        print("  (1) save_ckpt: 保存され .tmp が残らない: OK")

        # 別インスタンスへ復元する
        model2 = sm.UNet1D().to(DEVICE)              # ここで RNG が進む
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        torch.manual_seed(999)                       # RNG をさらにわざと動かしてから戻す
        step, cfg = ck.load_ckpt(path, model2, opt2)

        assert step == 250 and cfg == {"K": 1, "n": 256}
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2), f"重みが復元されていない: {k1}"
        # AdamW の1次・2次モーメントが戻っていること
        s1 = opt.state_dict()["state"]
        s2 = opt2.state_dict()["state"]
        assert set(s1) == set(s2) and len(s1) > 0
        for i in s1:
            assert torch.equal(s1[i]["exp_avg"], s2[i]["exp_avg"])
            assert torch.equal(s1[i]["exp_avg_sq"], s2[i]["exp_avg_sq"])
        assert torch.equal(torch.get_rng_state(), rng_at_save), "RNG が復元されていない"
        print("  (2) load_ckpt: 重み・AdamWモーメント・step・config・RNG が復元される: OK")

        # 上書き保存しても既存が壊れない（atomic な差し替え）
        ck.save_ckpt(path, model, opt, 250, {"K": 1, "n": 256})
        assert ck.load_ckpt(path, model2)[0] == 250

        # latest_ckpt は step を数値で比較する（文字列順だと 9 > 10 になる）
        for s in (9, 10, 100):
            ck.save_ckpt(ck.ckpt_path(d, s), model, opt, s)
        assert ck.latest_ckpt(d) == ck.ckpt_path(d, 250)
        assert ck.latest_ckpt(d / "nonexistent") is None
        print("  (3) latest_ckpt: step を数値順で選び、空なら None: OK")

    print("test_checkpoint_roundtrip: OK")


# ============================================================
# 2. 逆過程1ステップの切り出し
# ============================================================
def test_reverse_step_matches_inline() -> None:
    """(f2) _reverse_step が切り出し前のインライン式と厳密一致すること。"""
    model = _model()
    diff = sm.Diffusion(device=DEVICE)
    cond = _cond()
    torch.manual_seed(0)
    xt = torch.randn(cond.size(0), sm.IN_CH, sm.NUM_SLOTS, device=DEVICE)

    for ti in (sm.T_STEPS - 1, 500, 1, 0):
        with torch.no_grad():
            # 切り出し前の式をそのまま書く（clamp_ だけは非 in-place に直してある）
            eps_ref = diff._eps(model, xt, ti, cond, sm.GUIDANCE_SCALE)
            x0_ref = ((xt - diff.sqrt_1m_acp[ti] * eps_ref) / diff.sqrt_acp[ti]).clamp(0.0, 1.0)
            mean_ref = diff.post_coef_x0[ti] * x0_ref + diff.post_coef_xt[ti] * xt
            x_next, eps, x0_hat, mean = diff._reverse_step(
                model, xt, ti, cond, sm.GUIDANCE_SCALE, None, True)
        assert torch.equal(eps, eps_ref), f"eps_hat が一致しない (ti={ti})"
        assert torch.equal(x0_hat, x0_ref), f"x0_hat が一致しない (ti={ti})"
        assert torch.equal(mean, mean_ref), f"mean が一致しない (ti={ti})"
        # smoke_test が見ている4つの assert と同じもの
        assert eps.shape == xt.shape and torch.isfinite(eps).all()
        assert float(x0_hat.min()) >= 0.0 and float(x0_hat.max()) <= 1.0
        assert torch.isfinite(mean).all()
        assert int(mean.argmax(dim=1).max()) < sm.NUM_ACT
    print("  (1) return_aux の4返り値が旧インライン式と厳密一致: OK")

    # ti=0 は雑音を加えない（post_var[0] は厳密に 0）
    with torch.no_grad():
        x_next, _, x0_hat, mean = diff._reverse_step(
            model, xt, 0, cond, sm.GUIDANCE_SCALE, None, True)
    assert float(diff.post_var[0]) == 0.0 and float(diff.post_coef_xt[0]) == 0.0
    assert torch.equal(x_next, mean), "ti=0 で x_next が mean と一致しない"
    # ★x0_hat とは厳密には一致しない。post_coef_x0[0] は実数では 1 だが
    #   float32 では 0.99983406（1-acp[0] の桁落ち）。相対 1.7e-4 のスケール差が残る
    assert abs(float(diff.post_coef_x0[0]) - 1.0) < 2e-4
    rel = float((mean - x0_hat).abs().max() / x0_hat.abs().max())
    assert rel < 1e-3, f"ti=0 のスケール差が想定より大きい: {rel}"
    print(f"  (2) ti=0: post_var=0, x_next==mean, x0_hat との相対差 {rel:.2e}: OK")

    # 同じ z を渡せば同じ x_next になる（2パス蓄積の乱数再現性の前提）
    z = torch.randn_like(xt)
    with torch.no_grad():
        a = diff._reverse_step(model, xt, 500, cond, sm.GUIDANCE_SCALE, z)
        b = diff._reverse_step(model, xt, 500, cond, sm.GUIDANCE_SCALE, z)
    assert torch.equal(a, b), "同じ z を渡しても結果が変わる"
    print("  (3) z を渡すと決定的: OK")

    sm.smoke_test()   # ★重複除去したハンドコピーが通ること
    print("  (4) model.smoke_test() が _reverse_step 経由で通る: OK")
    print("test_reverse_step_matches_inline: OK")


def main() -> None:
    test_checkpoint_roundtrip()
    test_reverse_step_matches_inline()
    print("\ntest_stage2: OK")


if __name__ == "__main__":
    main()
