"""
Stage 2 基盤の単体テスト（Stage2_design.md §10.3）。

対象は §10.2 の実装 1:
    1. 定期チェックポイントと再開            stage2_checkpoint.py

検証する内容:
    (ckpt) save_ckpt -> load_ckpt の往復で model/optimizer/step/RNG が戻る。
           一時ファイルが残らない。latest_ckpt が step を数値順で選ぶ

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


def main() -> None:
    test_checkpoint_roundtrip()
    print("\ntest_stage2: OK")


if __name__ == "__main__":
    main()
