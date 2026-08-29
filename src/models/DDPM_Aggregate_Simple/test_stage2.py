"""
Stage 2 基盤の単体テスト（Stage2_design.md §10.3）。

対象は §10.2 の実装 1〜5:
    1. 定期チェックポイントと再開            stage2_checkpoint.py
    2. 逆過程1ステップの切り出し             model.Diffusion._reverse_step
    3. 打ち切り逆伝播つきサンプラ             model.Diffusion.sample_differentiable
    4. straight-through デコーダ             model.straight_through
    5. 教師 A* と28群表への採点              stage2_targets.py

検証する内容:
    (ckpt) save_ckpt -> load_ckpt の往復で model/optimizer/step/RNG が戻る。
           一時ファイルが残らない。latest_ckpt が step を数値順で選ぶ
    (f2)   _reverse_step が旧インライン式と厳密一致し、smoke_test が通る
    (f)    同一シードで sample_differentiable(K=0) が sample と一致
           ＝ 切り出しリファクタで生成の数値が1ビットも変わっていない
    (c)    K=0 では勾配が流れない
    (c2)   ★K>=1 では勾配が流れる。@torch.no_grad() の付け間違いは (c) だけでは
           検出できない（勾配が一切流れなくなっても (c) は通ってしまう）
    (a)    straight-through の前向きが厳密に one-hot で、群平均が pool_to_rates と一致
    (h)    stage2_targets.load_stula_targets が移植元 (CVAE_Aggregate) と厳密一致
    (k)    教師 A* の実測性質が設計書 §3.6 から動いていない
    (g)    A* が各 (群,時刻) で12チャネルの和 1
    (d_idx) stage2_targets.d_index が model.d_index と全28組で一致

★ 出口の零初期化について:
    UNet1D は out_conv を零初期化するので、そのままでは勾配の大きさが測れない。
    test_backbone.py と同じ _wake_up() で out_conv だけを小さな乱数で埋めてから測る。

★ (k) の期待値が食い違ったときは、テスト側を黙って合わせないこと。
    前処理か教師データが設計書の執筆時から動いた証拠であり、設計書に載っている
    実測値（勾配配分・必要な n・zero-shot 基準線）が全部無効になる。

    .venv/bin/python3 src/models/DDPM_Aggregate_Simple/test_stage2.py
"""
import contextlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
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
st: Any = _load("simple_stage2_targets", HERE / "stage2_targets.py")
DEVICE = "cpu"          # テストは決定性重視で CPU 固定
T_SHORT = 12            # 全1000ステップは重いので、逆過程の往復は短い T で見る


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


@contextlib.contextmanager
def _short_T():
    """T_STEPS を短くした Diffusion を貸し出す（本番の T=1000 は重いので）。

    ★sm.T_STEPS はバッファの構築だけでなく sample / _sample_head のループ範囲にも
      使われる。構築後に戻すと「バッファは12段なのにループは1000段」になって
      IndexError で落ちるので、使い終わるまで差し替えたままにする。
      test_backbone.test_reverse_process と同じ流儀。
    """
    original = sm.T_STEPS
    sm.T_STEPS = T_SHORT
    try:
        yield sm.Diffusion(device=DEVICE)
    finally:
        sm.T_STEPS = original


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


# ============================================================
# 3. 打ち切り逆伝播つきサンプラ
# ============================================================
def test_sample_differentiable() -> None:
    with _short_T() as diff:
        model = _model()
        cond = _cond()

        # (f) 同一シードで sample と一致する ＝ 切り出しで数値が変わっていない
        torch.manual_seed(7)
        ref = diff.sample(model, cond)
        torch.manual_seed(7)
        out0 = diff.sample_differentiable(model, cond, K=0)
        assert out0.shape == (cond.size(0), sm.IN_CH, sm.NUM_SLOTS)
        assert torch.equal(out0.argmax(dim=1), ref), "K=0 の生成が sample と一致しない"
        print("  (1) (f) K=0 の出力が sample と厳密一致: OK")

        # (c) K=0 では勾配が流れない
        assert not out0.requires_grad, "K=0 なのに勾配が流れている"
        print("  (2) (c) K=0 で requires_grad=False: OK")

        # (c2) K>=1 では勾配が流れる。★これが @torch.no_grad() の付け間違いを検出する
        for K in (1, 2):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(7)
            out = diff.sample_differentiable(model, cond, K=K)
            assert out.requires_grad, f"K={K} なのに勾配が流れていない"
            out.sum().backward()
            n_nonzero = sum(1 for p in model.parameters()
                            if p.grad is not None and float(p.grad.abs().max()) > 0)
            n_total = sum(1 for _ in model.parameters())
            assert n_nonzero > 0, f"K={K} で非ゼロ勾配のパラメータが1つも無い"
            print(f"  (3) (c2) K={K}: 非ゼロ勾配 {n_nonzero}/{n_total} パラメータ: OK")

        # モードの復元（リハーサル項が train を要求するため）
        model.train()
        diff.sample_differentiable(model, cond, K=1)
        assert model.training, "sample_differentiable の後に train モードへ戻っていない"
        model.eval()
        diff.sample_differentiable(model, cond, K=1)
        assert not model.training, "sample_differentiable の後に eval モードへ戻っていない"
        print("  (4) 呼び出し前のモードへ復元される: OK")

        # zs を渡すと決定的（2パス蓄積の前提。K=1 では ti=0 が雑音を使わないので K=3 で見る）
        K = 3
        zs = {ti: torch.randn(cond.size(0), sm.IN_CH, sm.NUM_SLOTS, device=DEVICE)
              for ti in range(1, K)}
        x_K = diff._sample_head(model, cond, K)
        a = diff._sample_tail(model, x_K, K, cond, zs=zs)
        b = diff._sample_tail(model, x_K, K, cond, zs=zs)
        assert torch.equal(a, b), "同じ x_K と zs で結果が変わる"
        assert not diff._sample_tail(model, x_K, K, cond).equal(a), \
            "zs を渡さなくても同じ結果になる（雑音が効いていない）"
        print("  (5) head/tail 分割: 同じ x_K と zs なら決定的: OK")
    print("test_sample_differentiable: OK")


# ============================================================
# 4. straight-through デコーダ
# ============================================================
def test_straight_through() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(8, sm.IN_CH, sm.NUM_SLOTS, requires_grad=True)
    y = sm.straight_through(x0)
    assert y.shape == x0.shape

    # 前向きは厳密に one-hot。★括弧を落とすと 6.0e-08 ずれてここが落ちる
    oh = torch.nn.functional.one_hot(
        torch.softmax(x0.detach(), dim=1).argmax(dim=1), sm.NUM_ACT).permute(0, 2, 1).float()
    assert torch.equal(y.detach(), oh), "前向きが厳密な one-hot になっていない"
    assert float((y.detach().sum(dim=1) - 1.0).abs().max()) == 0.0
    print("  (1) 前向きが厳密に one-hot、活動チャネル方向の和が厳密に 1: OK")

    # 後ろ向きは softmax の微分（argmax なら勾配は恒等的に 0 になるはず）
    y.sum().backward()
    assert x0.grad is not None and float(x0.grad.abs().max()) > 0, \
        "勾配が流れていない（straight-through になっていない）"
    print("  (2) 後ろ向きに勾配が流れる: OK")

    # 群平均が pool_to_rates と一致する ＝ 学習で下げる量と評価で測る量が同じ
    pool = x0.detach().argmax(dim=1).numpy()[None, :, :]           # (1, M, 96)
    rates = sm.pool_to_rates(pool).reshape(1, sm.NUM_ACT, sm.NUM_SLOTS)
    # ★pool_to_rates は float64 で平均を取るので、比較も float64 で行う。
    #   float32 のまま平均すると 1/M の丸めで 2.4e-08 ずれる（straight_through の
    #   欠陥ではなく、平均の精度の違い）
    diff64 = float(np.abs(y.detach().double().mean(0).numpy() - rates[0]).max())
    diff32 = float(np.abs(y.detach().mean(0).numpy() - rates[0]).max())
    assert diff64 == 0.0, f"float64 平均が pool_to_rates と一致しない: {diff64}"
    assert diff32 < 1e-6, f"float32 平均のずれが丸め誤差を超えている: {diff32}"
    print(f"  (3) (a) 群平均が pool_to_rates と一致 (float64 差 {diff64}, "
          f"float32 差 {diff32:.1e}): OK")

    # tau は softmax を鋭くするだけで、前向きの one-hot は変わらない
    y_tau = sm.straight_through(x0.detach(), tau=0.5)
    assert torch.equal(y_tau, oh), "tau を変えると前向きの one-hot が変わる"
    print("  (4) tau を変えても前向きは同じ one-hot: OK")
    print("test_straight_through: OK")


# ============================================================
# 5. 教師 A* と採点
# ============================================================
def test_targets_match_reference() -> None:
    """(h) 移植した load_stula_targets が移植元 (CVAE_Aggregate) と厳密一致すること。

    ★移植元は `from model import AggCVAE` という素の import を持つ。テストから素直に
      読むと sys.path[0] がこのファイルのディレクトリなので DDPM_Aggregate_Simple/model.py
      を掴んで ImportError になる。CVAE のディレクトリを先頭に差し込んでから読み、
      終わったら戻す。設計書 §10.1 が Stage 2 で断とうとしている経路そのものなので、
      本番コードではなくテストの中だけでこの細工をする。
    """
    mine = st.load_stula_targets()
    cvae_dir = REPO_ROOT / "src" / "models" / "CVAE_Aggregate"
    sys.path.insert(0, str(cvae_dir))
    try:
        ref = _load("cvae_japan_match_experiment", cvae_dir / "japan_match_experiment.py")
    finally:
        sys.path.remove(str(cvae_dir))
    reference = ref.load_stula_targets("timeband_weekday")

    for key in ("group_rates_tbl", "pop"):
        a, b = mine[key], reference[key]
        assert a.shape == b.shape, f"{key} の形が違う: {a.shape} vs {b.shape}"
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{key} の NaN 位置が違う"
        d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        assert d == 0.0, f"{key} が移植元と一致しない: max|d|={d}"
    print("  (1) (h) group_rates_tbl と pop が移植元と厳密一致: OK")

    # d_index は model.py と同じ規則でなければならない（群の並びがずれると全部が壊れる）
    for g in range(st.N_G):
        for a in range(st.N_A):
            for e in range(st.N_E):
                assert st.d_index(g, a, e) == sm.d_index(g, a, e)
    assert (st.D_GROUPS, st.NUM_SLOTS) == (sm.D_GROUPS, sm.NUM_SLOTS)
    assert st.NUM_COMMON == sm.NUM_ACT
    print("  (2) (d_idx) d_index と定数が model.py と全28組で一致: OK")
    print("test_targets_match_reference: OK")


def test_target_properties() -> None:
    """(k)(g) 教師 A* の実測性質が設計書 §3.6 から動いていないこと。

    ★食い違ったらテスト側を合わせない。設計書の実測値（勾配配分・必要な n・
      zero-shot 基準線）が全部無効になったという報告すべき事実である。
    """
    tgt = st.load_stula_targets()
    A, pop = tgt["group_rates_tbl"], tgt["pop"]

    assert A.shape == (28, 12, 96)
    n_nan = int(np.isnan(A).sum())
    assert n_nan == 0, f"全国・平日は全セル公表のはずだが NaN が {n_nan} セルある"
    assert int((~np.isnan(A)).sum()) == 32256
    print("  (1) NaN 0 セル / 有効教師セル 32,256: OK")

    # (g) 各 (群,時刻) で12活動の和が 1（公表値の丸めぶんだけ振れる）
    ch = A.sum(axis=1)
    assert 0.9995 <= ch.min() and ch.max() <= 1.0005, f"チャネル和 {ch.min()}..{ch.max()}"
    assert abs(float(ch.mean()) - 1.0) < 1e-4
    print(f"  (2) (g) 12チャネル和 min {ch.min():.4f} / max {ch.max():.4f}: OK")

    # exact 0 が 10.7%（原表の '-' = 行動者ゼロ が 99.3%、単位未満の 0 が 0.7%）
    n_zero = int((A == 0).sum())
    assert n_zero == 3458, f"exact 0 のセル数が 3458 から変わっている: {n_zero}"
    frac_le = float((A <= 0.01).mean())
    assert abs(frac_le - 0.481) < 2e-3, f"<=0.01 の割合が 48.1% から変わっている: {frac_le}"
    assert abs(float(np.median(A)) - 0.0112) < 2e-4
    print(f"  (3) exact 0 が {100*n_zero/A.size:.1f}% / <=0.01 が {100*frac_le:.1f}%: OK")

    # 群人口シェアの開きが 18.7 倍（§7.2 で群を等重みにした根拠）
    share = pop.reshape(28) / pop.sum()
    ratio = float(share.max() / share.min())
    assert abs(float(share.min()) - 0.0045) < 5e-4
    assert abs(float(share.max()) - 0.0839) < 5e-4
    assert abs(ratio - 18.7) < 0.5, f"群人口シェアの比が 18.7 倍から変わっている: {ratio}"
    # 15歳以上人口の 99.6% を覆う（就業状態が不詳の 411千人は公表表に区分が無く対象外）
    assert abs(float(pop.sum()) - 106709.0) < 1.0
    print(f"  (4) 群人口シェア {share.min():.4f}..{share.max():.4f} = {ratio:.1f} 倍: OK")
    print("test_target_properties: OK")


def test_eval_against() -> None:
    tgt = st.load_stula_targets()
    A = tgt["group_rates_tbl"]

    # 教師そのものを渡せば誤差は厳密に 0
    perfect = A.reshape(st.D_GROUPS, st.NUM_COMMON * st.NUM_SLOTS)
    r12 = st.eval_against(perfect, tgt, st.mask_12act())
    assert r12["rate_mae"] == 0.0 and r12["dev_mae"] == 0.0 and r12["max_abs_err"] == 0.0
    assert r12["mask"] == "12act" and r12["n_cells"] == 32256
    print("  (1) 教師そのものを渡すと誤差 0: OK")

    rows = st.eval_both(perfect, tgt)
    assert [r["mask"] for r in rows] == ["12act", "11act"]
    assert rows[1]["n_cells"] == 28 * 11 * 96, "11act のセル数が合わない"
    assert "mae_OTHER_X" in rows[0] and "mae_OTHER_X" not in rows[1]
    print("  (2) eval_both が 12act / 11act の2行を mask 列付きで返す: OK")

    # 相対誤差 = mae / 教師平均率。定数ずらしで手計算と突き合わせる
    shifted = (A + 0.01).reshape(st.D_GROUPS, st.NUM_COMMON * st.NUM_SLOTS)
    r = st.eval_against(shifted, tgt, st.mask_12act())
    assert abs(r["rate_mae"] - 0.01) < 1e-12
    assert abs(r["dev_mae"]) < 1e-12, "一様なずらしは群偏差を動かさないはず"
    for c in st.Common:
        q_bar = float(A[:, int(c), :].mean())
        assert abs(r[f"rel_{c.name}"] - 0.01 / q_bar) < 1e-9, f"rel_{c.name} の定義がずれている"
    print(f"  (3) rel_* = mae / 教師平均率 (例 rel_TRAVEL={r['rel_TRAVEL']:.3f}): OK")
    print("test_eval_against: OK")


def main() -> None:
    test_checkpoint_roundtrip()
    test_reverse_step_matches_inline()
    test_sample_differentiable()
    test_straight_through()
    test_targets_match_reference()
    test_target_properties()
    test_eval_against()
    print("\ntest_stage2: OK")


if __name__ == "__main__":
    main()
