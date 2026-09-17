"""
Stage 2 基盤の単体テスト（Stage2_design.md §10.3）。

対象は §10.2 の実装 1〜11:
    1. 定期チェックポイントと再開            stage2_checkpoint.py
    2. 逆過程1ステップの切り出し             model.Diffusion._reverse_step
    3. 打ち切り逆伝播つきサンプラ             model.Diffusion.sample_differentiable
    4. straight-through デコーダ             model.straight_through
    5. 教師 A* と28群表への採点              stage2_targets.py
    6. 集計損失（split-batch 不偏推定）        stage2_loss.py
    7. 学習ループ                            stage2_finetune.py
    8. teacher_mask と --holdout-groups       stage2_finetune.py
    9. 事後チェックポイント選択                stage2_select.py
   11. 2パス勾配蓄積（gradient caching）      stage2_finetune.py
   12. 学習ループの監視と λ の決め方          stage2_finetune.py / stage2_select.py

検証する内容:
    (ckpt) save_ckpt -> load_ckpt の往復で model/optimizer/step/RNG が戻る。
           一時ファイルが残らない。latest_ckpt が step を数値順で選ぶ
    (np)   ★numpy の RNG（群サブサンプリング）も往復する。torch 側だけ戻しても
           再開後の d_pick は step 1 からの並びを繰り返す（例外は出ない）
    (shape) K=0 / D_sub=0 / 奇数 n を、生成を1回でも回す前に落とす
    (gn)   θ の勾配ノルムが層別 LR の群ごとに取れる
    (x0)   ★clamp の飽和と straight-through の鋭さが観測できる。L_agg も
           g_diagnostics も clamp より上流なので、これ無しでは空回りが見えない
    (lam)  λ=auto が1更新の外れ値で決まらない
    (val)  ★val の ε-MSE が決定的で、学習の乱数もモードも汚さない
    (mem-g) ★暗記チェックが参照集合のサイズ交絡を持ち込まない
    (jsd-mem) jsd は chunk を使えないので K×B で予算判定する
    (crn)  事後選択が全 ckpt を同じ乱数列（common random numbers）で生成する
    (f2)   _reverse_step が旧インライン式と厳密一致し、smoke_test が通る
    (f)    同一シードで sample_differentiable(K=0) が sample と一致
           ＝ 切り出しリファクタで生成の数値が1ビットも変わっていない
    (c)    K=0 では勾配が流れない
    (c2)   ★K>=1 では勾配が流れる。@torch.no_grad() の付け間違いは (c) だけでは
           検出できない（勾配が一切流れなくなっても (c) は通ってしまう）
    (a)    straight-through の前向きが厳密に one-hot で、群平均が pool_to_rates と一致
    (h)    stage2_targets.load_stula_targets が移植元 (CVAE_Aggregate) と厳密一致
    (k)    教師 A* の実測性質が設計書 §3.6 から動いていない
    (g)    A* と Ã が各 (群,時刻) で12チャネルの和 1
    (b)    split-batch 推定が合成データで不偏（多様性への罰が消えている）
    (i)    ε=inf で ω が全要素1、χ² の損失値が素の MSE と一致
    (j)    ★ω の平均が 1。正規化を忘れると実効学習率が24倍ずれる
    (lr)   層別 LR の分割が条件経路と conv/attention を取り違えていない
    (mem)  K×D_sub×n の予算超過を学習前に落とす
    (lgo)  teacher_mask が損失群だけを外し、人口層化が大小を混ぜる
    (sel)  事後選択が §9.8 の出所列を必ず持ち、in-teacher と held-out を分ける
    (d)    ピークメモリが K に線形かつ n に依存しない
    (e)    ★2パス蓄積の勾配が一括計算と一致する（近似ではない）
    (e2)   2パス目に1パス目と同じ zs を渡すと x_0 が一致する
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
import inspect
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    """sys.modules に一意名で載せる。既に同じファイルが同じ名前で入っていれば使い回す。

    ★使い回しが要点。同名で読み直すと sys.modules のエントリは置き換わるが、
      先に読んだ側が掴んでいるモジュールオブジェクトは別のまま残る。すると
      「model.T_STEPS を差し替えたのに、こちらから呼ぶ生成は 1000 ステップのまま」
      のような、例外を出さずに黙って重くなる／数値が変わる食い違いが起きる。
    """
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
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
sl: Any = _load("simple_stage2_loss", HERE / "stage2_loss.py")
ft: Any = _load("simple_stage2_finetune", HERE / "stage2_finetune.py")
se: Any = _load("simple_stage2_select", HERE / "stage2_select.py")
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
        # ★numpy 側も渡す。群サブサンプリング d_pick はこの Generator から引く
        np_rng = np.random.default_rng(42)
        np_rng.integers(0, 100, size=7)       # 保存前に何回か消費しておく
        next_after_save = np_rng.integers(0, 10**9, size=3).tolist()
        np_rng.bit_generator.state = np.random.default_rng(42).bit_generator.state
        np_rng.integers(0, 100, size=7)       # 保存時点の状態へ戻す
        ck.save_ckpt(path, model, opt, 250, {"K": 1, "n": 256}, np_rng=np_rng)
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

        # ★map_location を指定しても RNG が復元できること。指定すると RNG 状態の
        #   テンソルまでそのデバイスへ移るので、CPU へ戻さないと set_rng_state が落ちる
        step_ml, _ = ck.load_ckpt(path, model2, map_location=sm.DEVICE)
        assert step_ml == 250
        print(f"  (2b) map_location={sm.DEVICE} でも RNG を復元できる: OK")

        # ★(np) numpy の RNG（群サブサンプリング）も往復すること。
        #   torch 側だけ戻しても d_pick は step 1 からの並びを繰り返す（例外は出ない）
        fresh = np.random.default_rng(0)             # 保存時とは無関係な状態から始める
        ck.load_ckpt(path, model2, np_rng=fresh)
        assert fresh.integers(0, 10**9, size=3).tolist() == next_after_save, \
            "numpy の RNG が復元されていない（再開後の d_pick が別系列になる）"
        # np_rng を渡さなければ numpy 側は触らない（評価用途で副作用を出さない）
        untouched = np.random.default_rng(0)
        before = untouched.bit_generator.state
        ck.load_ckpt(path, model2)
        assert untouched.bit_generator.state == before
        print("  (2c) (np) numpy RNG も往復し、np_rng 未指定なら触らない: OK")

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


# ============================================================
# 6. 集計損失
# ============================================================
def _fake_onehot(probs: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """各スロットで probs (D,12,96) に従う one-hot を群あたり n 本引く。

    生成器の代わりに使う合成データ。真の平均が probs だと分かっているので、
    推定量の期待値を解析値と突き合わせられる。
    """
    d, n_act, n_slot = probs.shape
    flat = probs.permute(0, 2, 1).reshape(-1, n_act)                  # (D*96, 12)
    idx = torch.multinomial(flat, n, replacement=True, generator=gen)  # (D*96, n)
    oh = torch.nn.functional.one_hot(idx, n_act).float()               # (D*96, n, 12)
    return oh.view(d, n_slot, n, n_act).permute(0, 2, 3, 1).reshape(d * n, n_act, n_slot)


def test_chi2_weights() -> None:
    """(i)(j) 重み ω の性質。"""
    torch.manual_seed(0)
    q = torch.rand(4, sm.NUM_ACT, sm.NUM_SLOTS)
    q = q / q.sum(dim=1, keepdim=True)          # 各スロットで和1（教師と同じ性質）

    # (i) ε=inf は素の MSE。ω が厳密に全要素1
    w_inf = sl.chi2_weights(q, float("inf"))
    assert torch.equal(w_inf, torch.ones_like(q)), "ε=inf で ω が全要素1になっていない"
    print("  (1) (i) ε=inf で ω が厳密に全要素1: OK")

    # (j) ★平均1への正規化。忘れると実効学習率が24倍ずれる
    for eps in (0.01, 0.05, 1.0 / 256):
        w = sl.chi2_weights(q, eps)
        assert abs(float(w.mean()) - 1.0) < 1e-5, f"ω の平均が1でない (eps={eps}): {w.mean()}"
        # 率の低いセルほど重い（逆分散重みの向き）
        assert float(w[q < 0.01].mean()) > float(w[q > 0.2].mean())
    print("  (2) (j) ω の平均が 1、かつ低率セルほど重い: OK")

    # 十分大きい ε は素の MSE に収束する（ε=inf 分岐と連続であること）
    w_big = sl.chi2_weights(q, 1e6)
    assert float((w_big - 1.0).abs().max()) < 1e-3
    print("  (3) 大きい ε は ω=1 に収束（inf 分岐と連続）: OK")

    for bad in (0.0, -1.0):
        try:
            sl.chi2_weights(q, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"eps={bad} が弾かれていない")
    print("  (4) eps<=0 を弾く: OK")
    print("test_chi2_weights: OK")


def test_agg_loss() -> None:
    """(i)(g) 損失値そのものの性質。"""
    torch.manual_seed(0)
    d_sub, n = 4, 8
    q = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS)
    q = q / q.sum(dim=1, keepdim=True)
    gen = torch.Generator().manual_seed(1)
    y = _fake_onehot(q, n, gen)

    # (g) Ã 側も各 (群,時刻) で12チャネルの和が1（one-hot の平均なので厳密）
    a_A, a_B = sl.group_rates_split(y, n)
    for a in (a_A, a_B):
        assert a.shape == (d_sub, sm.NUM_ACT, sm.NUM_SLOTS)
        assert float((a.sum(dim=1) - 1.0).abs().max()) == 0.0
    print("  (1) (g) Ã の12チャネル和が厳密に 1: OK")

    # A と B は重ならない（同じ本が両方の半分に入らない）
    grouped = y.view(d_sub, n, sm.NUM_ACT, sm.NUM_SLOTS)
    assert torch.equal(a_A, grouped[:, :n // 2].mean(dim=1))
    assert torch.equal(a_B, grouped[:, n // 2:].mean(dim=1))
    print("  (2) A/B が群ごとに前半・後半へ重なりなく分かれる: OK")

    # (i) ε=inf の χ² が素の MSE と一致する ＝ 1本のコードで両方走る
    w_inf = sl.chi2_weights(q, float("inf"))
    plain = ((a_A - q) * (a_B - q)).mean()
    assert torch.allclose(sl.agg_loss(y, q, w_inf, n), plain, atol=0, rtol=0)
    print("  (3) (i) ε=inf の損失値が素の MSE と厳密一致: OK")

    # 教師そのものを生成したことにすれば損失は 0
    perfect = q.unsqueeze(1).expand(d_sub, n, sm.NUM_ACT, sm.NUM_SLOTS).reshape(
        d_sub * n, sm.NUM_ACT, sm.NUM_SLOTS)
    assert abs(float(sl.agg_loss(perfect, q, w_inf, n))) < 1e-12
    print("  (4) 教師と同じ率を生成すると損失 0: OK")

    # 奇数の n は split-batch にできない
    try:
        sl.group_rates_split(y, 7)
    except ValueError:
        print("  (5) 奇数の n を弾く: OK")
    else:
        raise AssertionError("奇数の n が弾かれていない")
    print("test_agg_loss: OK")


def test_split_batch_unbiased() -> None:
    """(b) ★split-batch 推定が不偏で、素朴な二乗和は多様性への罰を持つこと。

    合成データの真の平均 p と教師 q を別に置くと、解析値が分かる:
        E[split-batch] = mean_c ω (p−q)²                        （bias のみ）
        E[素朴]        = mean_c ω (p−q)² + (1/n) mean_c ω Var(y_c)  （罰つき）
    Var(y_c) = p_c(1−p_c)（one-hot なのでベルヌーイ）。
    """
    torch.manual_seed(0)
    d_sub, n, reps = 3, 16, 400
    p = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS)
    p = p / p.sum(dim=1, keepdim=True)
    q = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS)
    q = q / q.sum(dim=1, keepdim=True)
    omega = sl.chi2_weights(q, 0.05)

    bias = ((p - q) ** 2 * omega).mean()
    penalty = (omega * p * (1.0 - p)).mean() / n

    gen = torch.Generator().manual_seed(2)
    split_vals, naive_vals = [], []
    for _ in range(reps):
        y = _fake_onehot(p, n, gen)
        a_A, a_B = sl.group_rates_split(y, n)
        split_vals.append(float(sl.agg_loss_from_rates(a_A, a_B, q, omega)))
        a_full = y.view(d_sub, n, sm.NUM_ACT, sm.NUM_SLOTS).mean(dim=1)
        naive_vals.append(float(((a_full - q) ** 2 * omega).mean()))

    split_mean = float(np.mean(split_vals))
    naive_mean = float(np.mean(naive_vals))
    se = float(np.std(split_vals) / np.sqrt(reps))

    assert abs(split_mean - float(bias)) < 4 * se, \
        f"split-batch が不偏でない: {split_mean:.6f} vs 解析値 {float(bias):.6f} (se={se:.6f})"
    print(f"  (1) (b) split-batch の平均 {split_mean:.5f} ≒ bias² {float(bias):.5f} "
          f"(±4se={4*se:.5f}): OK")

    expected_naive = float(bias) + float(penalty)
    assert abs(naive_mean - expected_naive) < 4 * se, \
        f"素朴推定が解析値と合わない: {naive_mean:.6f} vs {expected_naive:.6f}"
    assert naive_mean > split_mean, "素朴推定が split-batch を上回っていない"
    print(f"  (2) 素朴推定の平均 {naive_mean:.5f} ≒ bias²+罰 {expected_naive:.5f} "
          f"（罰 {float(penalty):.5f} が実在する）: OK")
    print("test_split_batch_unbiased: OK")


def test_loss_grad_and_diagnostics() -> None:
    """loss_grad の解析形が autograd と一致し、g_diagnostics が値を返すこと。"""
    torch.manual_seed(0)
    d_sub, n = 4, 8
    q = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS)
    q = q / q.sum(dim=1, keepdim=True)
    omega = sl.chi2_weights(q, 0.01)
    gen = torch.Generator().manual_seed(3)
    y = _fake_onehot(q, n, gen)
    a_A, a_B = sl.group_rates_split(y, n)

    lhs_A = a_A.clone().requires_grad_(True)
    lhs_B = a_B.clone().requires_grad_(True)
    sl.agg_loss_from_rates(lhs_A, lhs_B, q, omega).backward()
    g_A, g_B = sl.loss_grad(a_A, a_B, q, omega)
    assert lhs_A.grad is not None and lhs_B.grad is not None
    assert float((g_A - lhs_A.grad).abs().max()) < 1e-9, "g_A が autograd と一致しない"
    assert float((g_B - lhs_B.grad).abs().max()) < 1e-9, "g_B が autograd と一致しない"
    print("  (1) loss_grad の解析形が autograd と一致: OK")

    # ★A半分に流す勾配は B半分の誤差で決まる（split-batch の帰結）
    scale = d_sub * sm.NUM_ACT * sm.NUM_SLOTS      # 群・活動・時刻の3軸とも平均
    assert torch.allclose(g_A, omega * (a_B - q) / scale)
    print("  (2) g_A が B半分の誤差で決まる: OK")

    diag = sl.g_diagnostics(g_A, q, n, act_names=sm.ACT_NAMES)
    for key in ("g_abs_mean", "g_abs_max", "g_frac_negligible",
                "g_share_unidentifiable", "g_sign_pos_frac"):
        assert key in diag and math.isfinite(diag[key]), key
    shares = [diag[f"g_share_{a}"] for a in sm.ACT_NAMES]
    assert abs(sum(shares) - 1.0) < 1e-5, "活動別シェアの和が1でない"
    print(f"  (3) g_diagnostics: 活動シェアの和 {sum(shares):.6f}、"
          f"識別不能セルへのシェア {diag['g_share_unidentifiable']:.3f}: OK")
    print("test_loss_grad_and_diagnostics: OK")


def test_jsd_loss() -> None:
    """アブレーション用 JSD が定義どおりで、床の値に依存すること。"""
    torch.manual_seed(0)
    d_sub, n = 3, 8
    q = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS)
    q = q / q.sum(dim=1, keepdim=True)
    perfect = q.unsqueeze(1).expand(d_sub, n, sm.NUM_ACT, sm.NUM_SLOTS).reshape(
        d_sub * n, sm.NUM_ACT, sm.NUM_SLOTS)
    assert abs(float(sl.jsd_loss(perfect, q, n))) < 1e-6, "同一分布で JSD が 0 でない"

    gen = torch.Generator().manual_seed(4)
    y = _fake_onehot(q, n, gen)
    v = float(sl.jsd_loss(y, q, n))
    assert 0.0 < v <= math.log(2.0) + 1e-6, f"JSD が [0, ln2] に入っていない: {v}"
    print(f"  (1) 同一分布で 0、一般には (0, ln2] に入る (実測 {v:.4f}): OK")

    # ★床の値で数値が動く。報告時に床を明記する必要があることの担保
    v_low, v_high = float(sl.jsd_loss(y, q, n, 1e-12)), float(sl.jsd_loss(y, q, n, 1.0 / 256))
    assert v_low != v_high, "床を変えても値が動かない（床が効いていない）"
    print(f"  (2) 床 1e-12 で {v_low:.4f} / 床 1/256 で {v_high:.4f} と動く: OK")
    print("test_jsd_loss: OK")


# ============================================================
# 7-8. 学習ループと LGO
# ============================================================
def test_layered_lr() -> None:
    """(lr) 層別 LR の分割（§8.3）。条件経路と conv/attention を取り違えていないこと。"""
    model = _model()
    opt = ft.build_optimizer(model)
    assert ft.PARAM_GROUP_NAMES == ("cond", "emb", "conv")
    assert len(opt.param_groups) == 3
    cond_g, emb_g, conv_g = opt.param_groups
    assert cond_g["lr"] == ft.LR_COND and emb_g["lr"] == ft.LR_EMB \
        and conv_g["lr"] == ft.LR_CONV
    assert ft.LR_COND > ft.LR_EMB > ft.LR_CONV, \
        "cond > emb > conv の順でなければならない（emb は時刻と条件の共有路）"

    groups = ft.split_param_groups(model)
    names = {k: [n for n, p in model.named_parameters()
                 if any(p is q for q in v)] for k, v in groups.items()}
    # cond には「群ごとに違う値を持つ」ものだけが入る
    assert any("cond_embeds" in n for n in names["cond"])
    assert any("cond_proj" in n for n in names["cond"])
    assert "null_emb" in names["cond"]
    # ★emb_proj は cond ではなく emb へ。時刻埋め込みとの和を受けるため
    assert not any("emb_proj" in n for n in names["cond"]), \
        "emb_proj が cond 群へ混入している（時刻応答まで 1e-4 で動いてしまう）"
    assert all("emb_proj" in n for n in names["emb"]) and names["emb"]
    assert not any(".conv1." in n or ".conv2." in n for n in names["cond"]), \
        "畳み込みが条件経路へ混入している"
    assert any("out_conv" in n for n in names["conv"])
    assert any("attn" in n for n in names["conv"])

    n = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    assert sum(n.values()) == sum(p.numel() for p in model.parameters()) == 1_759_124
    # §8.1: cond_embeds 72 + cond_proj 4,352 + null_emb 256 = 4,680、emb_proj×11 = 312,512
    assert n["cond"] == 4_680, f"群専用の条件パラメータ数が §8.1 と違う: {n['cond']}"
    assert n["emb"] == 312_512, f"emb_proj のパラメータ数が §8.1 と違う: {n['emb']}"
    print(f"  (1) (lr) cond {n['cond']:,} (0.27%) / emb {n['emb']:,} / "
          f"conv {n['conv']:,}（§8.1 の内訳と一致）: OK")
    print(f"  (2) emb_proj は cond ではなく emb 群（LR {ft.LR_EMB:g}）: OK")
    print("test_layered_lr: OK")


def test_memory_budget() -> None:
    """(mem) 1パス版のピーク K×D_sub×n を学習前に検査すること（§4.5）。"""
    ft.check_memory_budget(1, 24, 256)          # 6,144 <= 6,600 ： 通る
    ft.check_memory_budget(1, 7, 256)           # 1,792         ： 通る
    for bad in ((1, 28, 256), (4, 7, 256), (1, 7, 1024)):    # 7,168 / 7,168 / 7,168
        try:
            ft.check_memory_budget(*bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"予算超過が弾かれていない: {bad}")
    print("  (1) (mem) D_sub=24 は通り、28 と K=4 と n=1024 は落ちる: OK")
    print("test_memory_budget: OK")


def test_check_shapes() -> None:
    """(shape) K / D_sub / n の前提を、生成を1回でも回す前に落とすこと。

    ★どれも以前は「生成（本番で約40秒）の後に」壊れていた。K=0 は resolve_chunk の
      budget//K が ZeroDivisionError、奇数 n は group_rates_split の ValueError。
    """
    ft.check_shapes(1, 7, 256)
    ft.check_shapes(3, 1, 2)
    for bad, why in (((0, 7, 256), "K=0"), ((-1, 7, 256), "K<0"),
                     ((1, 0, 256), "D_sub=0"),
                     ((1, 7, 255), "n が奇数"), ((1, 7, 0), "n=0")):
        try:
            ft.check_shapes(*bad)
        except SystemExit:
            continue
        raise AssertionError(f"{why} が弾かれていない: {bad}")
    print("  (1) (shape) K=0 / D_sub=0 / 奇数 n を SystemExit で弾く: OK")

    # ★K=0 は resolve_chunk / check_memory_budget 単体でも ZeroDivisionError ではなく
    #   SystemExit にする（テストが K=0 を別用途で使うので CLI から届いてしまう）
    for fn, args in ((ft.resolve_chunk, (0, 7, 256)),
                     (ft.check_memory_budget, (0, 7, 256))):
        try:
            fn(*args)
        except SystemExit:
            continue
        except ZeroDivisionError as e:
            raise AssertionError(f"{fn.__name__} が ZeroDivisionError のまま: {e}") from e
        raise AssertionError(f"{fn.__name__} が K=0 を弾いていない")
    print("  (2) K=0 が resolve_chunk / check_memory_budget でも SystemExit: OK")
    print("test_check_shapes: OK")


def test_teacher_mask_and_holdout() -> None:
    """(lgo) teacher_mask と人口層化ホールドアウト（§8.4）。"""
    m = ft.build_teacher_mask([])
    assert m.shape == (28,) and m.all(), "既定は28群すべてが教師"
    m = ft.build_teacher_mask([0, 5, 27])
    assert int(m.sum()) == 25 and not m[0] and not m[5] and not m[27]
    print("  (1) (lgo) 既定は全28群、指定した群だけが外れる: OK")

    for bad in ([28], [-1], list(range(28))):
        try:
            ft.build_teacher_mask(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"不正な指定が弾かれていない: {bad}")
    print("  (2) 範囲外と全群除外を弾く: OK")

    # ★人口シェアで層化する。群人口シェアは18.7倍の開きがあるので、
    #   大小を混ぜないと「小さい群ばかり外す」ことになりうる
    pop = st.load_stula_targets()["pop"]
    picked = ft.stratified_holdout(pop, 4, seed=0)
    assert len(picked) == 4 and len(set(picked)) == 4
    share = (pop.reshape(28) / pop.sum())[picked]
    assert share.max() / share.min() > 3.0, \
        f"層化しても大小が混ざっていない: {share.min():.4f}..{share.max():.4f}"
    print(f"  (3) 人口層化の4群 {picked} のシェア "
          f"{share.min():.4f}..{share.max():.4f}（{share.max()/share.min():.1f}倍）: OK")
    print("test_teacher_mask_and_holdout: OK")


# ============================================================
# 9. 事後チェックポイント選択
# ============================================================
def test_eval_against_subset() -> None:
    """eval_against が群数非依存であること（LGO で教師群と held-out 群を分けて測る前提）。"""
    tgt = st.load_stula_targets()
    A = tgt["group_rates_tbl"]
    sel = np.zeros(28, dtype=bool)
    sel[[0, 3, 27]] = True
    sub = {"group_rates_tbl": A[sel], "pop": tgt["pop"].reshape(28)[sel]}
    r = st.eval_against(A[sel].reshape(3, st.NUM_COMMON * st.NUM_SLOTS), sub, st.mask_12act())
    assert r["n_cells"] == 3 * 12 * 96 and r["rate_mae"] == 0.0
    # 全28群でも同じ関数が動く（既存の呼び出しが壊れていない）
    full = st.eval_against(A.reshape(28, -1), tgt, st.mask_12act())
    assert full["n_cells"] == 32256 and full["rate_mae"] == 0.0
    print("  (1) eval_against が3群でも28群でも同じ定義で動く: OK")
    print("test_eval_against_subset: OK")


def _fake_pool(probs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """各スロットで probs に従う活動 index を群あたり n 本引く, (D,12,96) -> (D,n,96)

    生成器の代わりに使う合成データ。真の群平均が probs だと分かっているので、
    推定量の期待値を解析値と突き合わせられる。numpy 版なのは評価側
    (stage2_select / pool_to_rates) が numpy の int プールを扱うためで、
    torch 版の _fake_onehot とは用途が別である。

    Args:
        probs: 各群・各スロットの活動分布, dtype=float64, (D, 12, 96)
            活動方向の和が 1 であること
        n: 群あたりの本数
        rng: 乱数生成器

    Returns:
        活動 index のプール, dtype=int64, (D, n, 96)
    """
    d, _, n_slot = probs.shape
    cdf = probs.transpose(0, 2, 1).cumsum(axis=2)                  # (D,96,12)
    u = rng.random((d, n_slot, n, 1))
    idx = (u > cdf[:, :, None, :]).sum(axis=3)                     # (D,96,n) 逆関数法
    return idx.transpose(0, 2, 1).astype(np.int64)                 # (D,n,96)


def test_rate_mse_split() -> None:
    """rate_mse_split が生成側のモンテカルロ雑音を落とすことを確認する（設計書 §9.4）。

    素の rate_mse は E[(ā − A*)²] = (E[ā] − A*)² + Var(ā) で、第2項がモデルの
    良し悪しと無関係な床になる。ここでは「モデルが完璧」= 教師と同じ分布から
    引くので bias² = 0 であり、素の mse は床そのもの、split は 0 になるはずである。
    """
    rng = np.random.default_rng(0)
    d, n, reps = 3, 40, 30

    # 真の率。各スロットで活動方向の和が 1。NaN は置かない（マスクの効果を混ぜないため）
    raw = rng.random((d, st.NUM_COMMON, st.NUM_SLOTS)) + 0.05
    probs = raw / raw.sum(axis=1, keepdims=True)
    tgt = {"group_rates_tbl": probs, "pop": np.full(d, 1.0)}
    shape3 = (d, st.NUM_COMMON, st.NUM_SLOTS)

    def _measure(pool: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        half = n // 2
        ra = sm.pool_to_rates(pool[:, :half])
        rb = sm.pool_to_rates(pool[:, half:])
        return st.eval_against(sm.pool_to_rates(pool), tgt, st.mask_12act(), (ra, rb)), ra, rb

    # (1) 学習側 agg_loss_from_rates を ω=1 で呼んだ値と厳密に一致すること。
    #     教師に NaN が無く 12act 全通しなので、eval_against のマスクは全 True になり
    #     両者は同じセル集合を平均する
    r, ra, rb = _measure(_fake_pool(probs, n, rng))
    ref = sl.agg_loss_from_rates(torch.from_numpy(ra.reshape(shape3)),
                                 torch.from_numpy(rb.reshape(shape3)),
                                 torch.from_numpy(probs),
                                 torch.ones(shape3, dtype=torch.float64))
    assert abs(r["rate_mse_split"] - float(ref)) < 1e-12, \
        "学習側 agg_loss_from_rates(ω=1) と定義がずれている"
    print(f"  (1) agg_loss_from_rates(ω=1) と一致 ({r['rate_mse_split']:+.3e}): OK")

    # (2) mu_hat_split を渡さなければ返さない（後方互換）
    assert "rate_mse_split" not in st.eval_against(
        sm.pool_to_rates(_fake_pool(probs, n, rng)), tgt, st.mask_12act())
    print("  (2) mu_hat_split 未指定なら rate_mse_split を返さない: OK")

    # (3) ★本題。素の mse は解析的な床 Var(ā)=mean p(1-p)/n に一致し、split は 0 に寄る
    mse_s, split_s = [], []
    for _ in range(reps):
        r, _, _ = _measure(_fake_pool(probs, n, rng))
        mse_s.append(r["rate_mse"])
        split_s.append(r["rate_mse_split"])
    floor = float((probs * (1.0 - probs)).mean() / n)
    mse_bar, split_bar = float(np.mean(mse_s)), float(np.mean(split_s))
    assert abs(mse_bar - floor) < 0.1 * floor, \
        f"素の rate_mse が解析的な床と合わない: {mse_bar:.3e} vs {floor:.3e}"
    assert abs(split_bar) < 0.1 * floor, \
        f"split が 0 に寄っていない: {split_bar:+.3e} (床 {floor:.3e})"
    print(f"  (3) bias=0 のとき 素 {mse_bar:.3e} ≒ 床 {floor:.3e} / "
          f"split {split_bar:+.3e} ≒ 0: OK")
    print("test_rate_mse_split: OK")


def test_checkpoint_selection() -> None:
    """(sel) 事後選択が §9.8 の出所列を持ち、LGO で in-teacher と held-out を分けること。

    ★出力は 1 指標 1 行の縦持ちである（実装項目 14）。横持ちだと循環した値
      （教師適合）と非循環な値（ガードレール）が同じ 1 行に混ざり、後から
      どちらの性質の数字か区別できなくなる。
    """
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tgt = st.load_stula_targets()
    cond_idx, sched_real, w_real, _ = sm.load_data()
    d_real = sm.cond_to_d(cond_idx)

    with tempfile.TemporaryDirectory() as tmp, _short_T():
        d = Path(tmp)
        # 教師28群（LGO なし）と、4群を外した LGO の2世代
        ck.save_ckpt(ck.ckpt_path(d, 1), model, opt, 1, {"K": 1, "n": 4, "holdout": []})
        ck.save_ckpt(ck.ckpt_path(d, 2), model, opt, 2, {"K": 1, "n": 4, "holdout": [0, 7, 14, 21]})

        rows: list[dict] = []
        for step in (1, 2):
            rows.extend(se.evaluate_ckpt(ck.ckpt_path(d, step), tgt, sched_real, d_real,
                                         w_real, n=2, device=DEVICE))

    # ★§9.8 の出所列。このリポジトリは数値の出所取り違えを2回起こしている
    for r in rows:
        for col in ("teacher_groups", "eval_kind", "reference", "statistic",
                    "mask", "weight_basis", "metric", "value", "vs_zeroshot"):
            assert col in r, f"出所の列が欠けている: {col}"
        assert r["mask"] in ("11act", "12act")
        assert r["eval_kind"] in ("in-teacher", "held-out", "all")
        assert r["weight_basis"] in ("stula_pop", "atus_comp", "none")
    print("  (1) (sel) 全行が §9.8 の出所列（statistic / weight_basis を含む）を持つ: OK")

    # 軸1（循環）と軸2（非循環）が statistic で機械的に分かれること
    ax1 = [r for r in rows if r["reference"] == "teacher"]
    ax2 = [r for r in rows if r["reference"] == "atus"]
    assert ax1 and ax2
    assert all(r["statistic"] == "slot_rate" and r["weight_basis"] == "stula_pop"
               for r in ax1), "軸1 は教師と同じ統計量なので slot_rate / stula_pop"
    assert all(r["eval_kind"] == "all" for r in ax2), \
        "軸2 は群で分けずプール全体で測るので eval_kind=all"
    assert any(r["statistic"] != "slot_rate" for r in ax2)
    print("  (2) 軸1 は slot_rate（循環）、軸2 は eval_kind=all で分かれる: OK")

    # LGO なしは in-teacher だけ、LGO ありは held-out も出る
    s1 = [r for r in ax1 if r["step"] == 1]
    s2 = [r for r in ax1 if r["step"] == 2]
    assert {r["eval_kind"] for r in s1} == {"in-teacher"}
    assert all(r["teacher_groups"] == 28 for r in s1)
    assert {r["eval_kind"] for r in s2} == {"in-teacher", "held-out"}
    assert all(r["teacher_groups"] == 24 for r in s2)
    held_cells = [r["value"] for r in s2 if r["eval_kind"] == "held-out"
                  and r["mask"] == "12act" and r["metric"] == "n_cells"]
    assert held_cells == [float(4 * 12 * 96)], "held-out のセル数が4群ぶんでない"
    print("  (3) LGO 無しは in-teacher のみ、有りは held-out 4群が分かれて出る: OK")

    # 軸2 の指標集合が GUARDRAIL_META と厳密に一致すること。
    # ★片側だけ増やすと statistic / weight_basis の分類漏れが起きる。meta_of が
    #   KeyError で落とす側と、この検査で META の書き過ぎを止める側の両方が要る
    got = {r["metric"] for r in ax2}
    assert got == set(se.GUARDRAIL_META), \
        f"GUARDRAIL_META とずれている: 不足={set(se.GUARDRAIL_META) - got} 余分={got - set(se.GUARDRAIL_META)}"
    for key in se.GUARDRAIL_KEYS:
        assert key in got, f"ガードレール欠落: {key}"
    print(f"  (4) 軸2 の {len(got)} 指標が GUARDRAIL_META と一致: OK")

    # 実装項目 13（妥当性12）と多様性・暗記が配線されていること
    by_stat: dict[str, set[str]] = {}
    for r in ax2:
        by_stat.setdefault(r["statistic"], set()).add(r["metric"])
    assert len(by_stat["plausibility"]) == 15, "妥当性12 + feasibility3 が揃っていない"
    assert len(by_stat["diversity"]) == 6, "多様性6 指標が揃っていない"
    assert len(by_stat["memorization"]) == 5
    assert "sleep_holder_rate" in by_stat["plausibility"]
    assert "pairwise_hamming_std" in by_stat["diversity"]
    print(f"  (5) statistic 別: " + " / ".join(
        f"{k}={len(v)}" for k, v in sorted(by_stat.items())) + ": OK")

    # zero-shot 基準がある指標だけ vs_zeroshot が有限になる
    for r in ax2:
        has_base = r["metric"] in se.ZERO_SHOT_GUARDRAILS
        assert math.isfinite(r["vs_zeroshot"]) == has_base, \
            f"{r['metric']}: vs_zeroshot の有無が基準の有無と合っていない"
    print("  (6) vs_zeroshot は zero-shot 実測がある指標にだけ入る: OK")

    # 実装項目 15 の rate_mse_split が軸1 に載る
    assert any(r["metric"] == "rate_mse_split" for r in ax1), "rate_mse_split が軸1 に無い"
    print("  (7) rate_mse_split が軸1 に載る（λ パレート曲線の横軸）: OK")

    # 端末表示の pivot が落ちないこと。
    # ★CSV は pivot より前に書き終わっているので落ちても結果は失われないが、
    #   12 世代 × 6 本を評価した最後で落ちると読む手が止まる
    ax1_tbl, ax2_tbl = se.summarize(pd.DataFrame(rows))
    assert list(ax1_tbl.columns) == list(se.SUMMARY_AXIS1)
    assert list(ax2_tbl.columns) == list(se.SUMMARY_AXIS2)
    # 軸1 は step1 の in-teacher と step2 の in-teacher / held-out で3行
    assert len(ax1_tbl) == 3, f"軸1 の行数が合わない: {len(ax1_tbl)}"
    # 軸2 は ckpt ごとに1行（群で分けないので eval_kind では増えない）
    assert len(ax2_tbl) == 2, f"軸2 の行数が合わない: {len(ax2_tbl)}"
    print("  (8) summarize が軸1（step × eval_kind）と軸2（step）を分けて返す: OK")
    print("test_checkpoint_selection: OK")


# ============================================================
# 11. 2パス勾配蓄積
# ============================================================
def _twopass_fixture(d_sub: int = 3, n: int = 8, K: int = 2):
    """2パス蓄積の検証に使うモデル・条件・教師をまとめて作る。"""
    model = _model()
    diff = sm.Diffusion(device=DEVICE)
    cond = torch.as_tensor(sm.cond_grid()[:d_sub], device=DEVICE).repeat_interleave(n, dim=0)
    torch.manual_seed(11)
    q = torch.rand(d_sub, sm.NUM_ACT, sm.NUM_SLOTS, device=DEVICE)
    q = q / q.sum(dim=1, keepdim=True)
    omega = sl.chi2_weights(q, 0.01)
    return model, diff, cond, q, omega


def _grads(model) -> dict:
    return {k: (v.grad.clone() if v.grad is not None else None)
            for k, v in model.named_parameters()}


def _max_grad_diff(a: dict, b: dict) -> tuple[float, float]:
    keys = [k for k in a if a[k] is not None and b[k] is not None]
    assert len(keys) == len([k for k in a if a[k] is not None]), "勾配が付いていないパラメータがある"
    absd = max(float((a[k] - b[k]).abs().max()) for k in keys)
    scale = max(float(a[k].abs().max()) for k in keys)
    return absd, absd / scale


def test_naive_accumulation_is_wrong() -> None:
    """普通の勾配蓄積が使えない理由を、設計書 §2.9 の数値例で固定する。

    損失がバッチ全体の非線形関数なので、部分から全体を復元できない。
    """
    # ★float64 で計算する。float32 だと 0.0025 が 0.00250000110 になり、
    #   ここで見たい「65倍」という桁の話がまるめ誤差の話に見えてしまう
    y = torch.tensor([1.0, 0.9, 0.1, 0.2], dtype=torch.float64)
    target = 0.5
    correct = float((y.mean() - target) ** 2)
    partial = float(torch.stack([(y[:2].mean() - target) ** 2,
                                 (y[2:].mean() - target) ** 2]).mean())
    assert abs(correct - 0.0025) < 1e-12, correct
    assert abs(partial - 0.1625) < 1e-12, partial
    assert abs(partial / correct - 65.0) < 1e-9
    print(f"  (1) 正しい損失 {correct:.4f} に対し部分和は {partial:.4f} で"
          f"{partial/correct:.0f}倍ずれる: OK")
    print("test_naive_accumulation_is_wrong: OK")


def test_two_pass_matches_full_batch() -> None:
    """(e)(e2) ★2パス蓄積の勾配が一括計算と一致すること。近似ではない。"""
    with _short_T():
        d_sub, n, K = 3, 8, 2
        total = d_sub * n
        model, diff, cond, q, omega = _twopass_fixture(d_sub, n, K)

        # --- 参照: 一括 autograd。zs と x_K の引き順を2パス版と揃える ---
        torch.manual_seed(7)
        zs = {ti: torch.randn(total, sm.IN_CH, sm.NUM_SLOTS, device=DEVICE)
              for ti in range(1, K)}
        x_K = diff._sample_head(model, cond, K)
        model.zero_grad(set_to_none=True)
        x0 = diff._sample_tail(model, x_K, K, cond, zs=zs)
        y = sm.straight_through(x0)
        a_A, a_B = sl.group_rates_split(y, n)
        ref_loss = sl.agg_loss_from_rates(a_A, a_B, q, omega)
        ref_loss.backward()
        ref = _grads(model)
        ref_val = float(ref_loss.detach())

        # (e2) 同じ zs をチャンクへ切って渡すと x_0 が再現される
        with torch.no_grad():
            parts = [diff._sample_tail(model, x_K[s:s + 6], K, cond[s:s + 6],
                                       zs={ti: z[s:s + 6] for ti, z in zs.items()})
                     for s in range(0, total, 6)]
        cat = torch.cat(parts, dim=0)
        dmax = float((x0.detach() - cat).abs().max())
        assert dmax < 1e-5, f"チャンクで x_0 が再現できない: {dmax}"
        # ★実際に集計へ入るのは argmax 後の one-hot なので、そちらは厳密一致でなければならない
        assert torch.equal(x0.detach().argmax(dim=1), cat.argmax(dim=1)), \
            "チャンクで argmax がずれる（one-hot が変わる＝別のサンプルを見ている）"
        print(f"  (1) (e2) 同じ zs でチャンク再現: max|Δx_0|={dmax:.1e}、argmax は厳密一致: OK")

        # --- (e) 2パス蓄積を chunk を変えて回し、参照と突き合わせる ---
        for chunk in (total, 12, 6, 1):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(7)          # zs と x_T を参照と同じに引き直す
            loss, got_A, got_B, diag = ft._aggregate_step_two_pass(
                diff, model, cond, K, n, q, omega, chunk, 1.0)
            assert abs(loss - ref_val) < 1e-6, f"損失が一致しない (chunk={chunk})"
            assert torch.equal(got_A, a_A.detach()) and torch.equal(got_B, a_B.detach())
            absd, rel = _max_grad_diff(_grads(model), ref)
            # float32 の丸め誤差レベル。設計書 §2.9 の実測は 5.96e-08（絶対）
            assert rel < 1e-5, f"勾配が一致しない (chunk={chunk}): 相対 {rel:.2e}"
            # x_0 診断は chunk に依らない（同じ x_0 を見ているため）
            assert set(diag) == {"x0_floor_frac", "x0_max", "st_p_max_mean"}, diag
            print(f"  (2) (e) chunk={chunk:3d}: 勾配 max|Δ|={absd:.2e}（相対 {rel:.1e}）: OK")

        # aggregate_step が chunk>=B で1パス、chunk<B で2パスへ分かれること
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        one, _, _, _ = ft.aggregate_step(diff, model, cond, K, n, q, omega, total, "sq")
        one_g = _grads(model)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        two, _, _, _ = ft.aggregate_step(diff, model, cond, K, n, q, omega, 6, "sq")
        assert abs(one - two) < 1e-6, "1パスと2パスで損失が違う"
        _, rel = _max_grad_diff(_grads(model), one_g)
        assert rel < 1e-5, f"1パスと2パスで勾配が違う: 相対 {rel:.2e}"
        print(f"  (3) aggregate_step の1パス／2パスが一致（相対 {rel:.1e}）: OK")
    print("test_two_pass_matches_full_batch: OK")


def _saved_bytes(fn) -> int:
    """fn の実行中に autograd が保存したテンソルの総バイト数。

    ★重複排除しない（設計書 §4.2 の実測と同じ流儀）。ここで見たいのは
      K と n に対する増え方の比なので、共有ストレージの二重計上は両辺で相殺される。
    """
    total = 0

    def pack(t: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += t.nbytes
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total


def test_two_pass_memory_scaling() -> None:
    """(d) ★勾配を保持するメモリが K に線形で、n（＝B）に依存しないこと。

    これが2パス蓄積を入れる理由そのもの。1パス版のピークは K×D_sub×n なので n に
    比例して増えるが、2パスなら K×chunk だけで決まる。
    """
    def peak(K: int, d_sub: int, n: int, chunk: int) -> float:
        model, diff, cond, q, omega = _twopass_fixture(d_sub, n, K)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        n_chunk = -(-(d_sub * n) // chunk)          # 切り上げ
        total = _saved_bytes(lambda: ft._aggregate_step_two_pass(
            diff, model, cond, K, n, q, omega, chunk, 1.0))
        return total / n_chunk                       # 1チャンクあたり＝ピーク

    with _short_T():
        # n を倍にしても1チャンクあたりの保持量は変わらない
        p_n8 = peak(K=1, d_sub=2, n=8, chunk=4)
        p_n16 = peak(K=1, d_sub=2, n=16, chunk=4)
        assert abs(p_n16 / p_n8 - 1.0) < 0.02, \
            f"n に依存している: n=8 で {p_n8:,.0f}B / n=16 で {p_n16:,.0f}B"
        print(f"  (1) (d) n=8 と n=16 でピークが同じ "
              f"({p_n8/1e6:.2f}MB vs {p_n16/1e6:.2f}MB): OK")

        # K を倍にすると保持量も倍になる
        p_k1 = peak(K=1, d_sub=2, n=8, chunk=8)
        p_k2 = peak(K=2, d_sub=2, n=8, chunk=8)
        assert 1.9 < p_k2 / p_k1 < 2.1, f"K に線形でない: 比 {p_k2/p_k1:.3f}"
        print(f"  (2) (d) K=1 -> K=2 でピークが {p_k2/p_k1:.2f} 倍（線形）: OK")

        # chunk を半分にすると保持量も半分に近づく（固定費ぶんだけ完全な半分にはならない）
        p_c8 = peak(K=1, d_sub=2, n=8, chunk=8)
        p_c4 = peak(K=1, d_sub=2, n=8, chunk=4)
        assert p_c4 < p_c8, "chunk を下げてもピークが下がらない"
        print(f"  (3) chunk 8 -> 4 でピークが {p_c8/1e6:.2f}MB -> {p_c4/1e6:.2f}MB: OK")
    print("test_two_pass_memory_scaling: OK")


def test_resolve_chunk() -> None:
    """chunk の自動決定と、予算チェックが chunk を見ること。"""
    # 予算に収まるなら B のまま（＝1パス）
    assert ft.resolve_chunk(1, 7, 256) == 1792
    assert ft.resolve_chunk(1, 24, 256) == 6144
    # 収まらないなら予算いっぱいまで下げる（＝2パス）
    assert ft.resolve_chunk(1, 28, 256) == ft.MEMORY_BUDGET
    assert ft.resolve_chunk(4, 7, 256) == ft.MEMORY_BUDGET // 4
    # 明示指定は B を超えない範囲で尊重する
    assert ft.resolve_chunk(1, 7, 256, chunk=512) == 512
    assert ft.resolve_chunk(1, 7, 256, chunk=99999) == 1792
    print("  (1) chunk の自動決定が予算 K×chunk <= "
          f"{ft.MEMORY_BUDGET:,} に収まる: OK")

    # ★予算チェックは chunk を見る。2パスなら D_sub=28 も通る
    ft.check_memory_budget(1, 28, 256, chunk=ft.resolve_chunk(1, 28, 256))
    ft.check_memory_budget(4, 7, 256, chunk=ft.resolve_chunk(4, 7, 256))
    try:
        ft.check_memory_budget(1, 28, 256, chunk=7168)     # 明示指定で超過
    except SystemExit:
        pass
    else:
        raise AssertionError("chunk 明示指定の予算超過が弾かれていない")
    print("  (2) 2パスなら D_sub=28 も通り、chunk 明示指定の超過は落ちる: OK")
    print("test_resolve_chunk: OK")


# ============================================================
# 12. 学習ループの監視（θ の勾配・x_0 診断・λ の決め方）
# ============================================================
def test_grad_norms() -> None:
    """(gn) 層別 LR の群ごとに θ の勾配ノルムが取れること。

    ★これが「Stage 2 が実際に θ を動かしているか」を見る唯一の量である。
      L_agg も rate_mae も g_diagnostics も straight-through と clamp の上流なので、
      代理勾配が潰れて θ が全く動かなくても正常値を出す。
    """
    model = _model()
    opt = ft.build_optimizer(model)

    # 勾配が無い状態では全て 0
    zero = ft.grad_norms(opt, "agg")
    assert zero == {f"agg_gnorm_{k}": 0.0 for k in ft.PARAM_GROUP_NAMES}, zero

    model(torch.randn(2, sm.IN_CH, sm.NUM_SLOTS), torch.zeros(2, dtype=torch.long),
          _cond(2)).square().mean().backward()
    got = ft.grad_norms(opt, "total")
    assert set(got) == {f"total_gnorm_{k}" for k in ft.PARAM_GROUP_NAMES}, got
    assert all(v > 0 for v in got.values()), got

    # param_groups の並びが split_param_groups と一致していること。
    # ★ここがずれると「conv のノルムを cond として報告する」壊れ方をする
    groups = ft.split_param_groups(model)
    for name in ft.PARAM_GROUP_NAMES:
        sq = sum(float(p.grad.pow(2).sum()) for p in groups[name] if p.grad is not None)
        assert abs(got[f"total_gnorm_{name}"] - math.sqrt(sq)) < 1e-5, name
    print("  (1) (gn) " + " / ".join(
        f"{k} {got[f'total_gnorm_{k}']:.3e}" for k in ft.PARAM_GROUP_NAMES)
        + "、群の割り当ても一致: OK")
    print("test_grad_norms: OK")


def test_x0_diagnostics() -> None:
    """(x0) clamp の飽和と straight-through の鋭さが観測できること。

    ★逆過程の各段は x0_hat を clamp(0,1) する。clamp は飽和した要素の勾配を
      厳密に 0 にするので、飽和が増えるほど代理勾配が痩せる。L_agg からは見えない。
    """
    # 下側 clamp は厳密に 0.0 になる（post_coef_xt[0] が厳密 0 なので混ざらない）
    x0 = torch.zeros(2, sm.NUM_ACT, sm.NUM_SLOTS)
    x0[:, 3] = 0.9998
    d = ft._x0_diagnostics(x0)
    # 12チャネル中 11 が 0 に張り付いている
    assert abs(d["x0_floor_frac"] - 11 / 12) < 1e-6, d
    assert abs(d["x0_max"] - 0.9998) < 1e-6, d
    # softmax(τ=1) の最大値。設計書 §2.8 の最悪ケース 0.198 と一致する
    assert abs(d["st_p_max_mean"] - 0.198) < 5e-3, d
    print(f"  (1) (x0) floor {d['x0_floor_frac']:.4f} / "
          f"p_max {d['st_p_max_mean']:.4f}（§2.8 の 0.198）: OK")

    # 飽和が無ければ floor は 0
    d2 = ft._x0_diagnostics(torch.full((2, sm.NUM_ACT, sm.NUM_SLOTS), 0.5))
    assert d2["x0_floor_frac"] == 0.0 and abs(d2["st_p_max_mean"] - 1 / 12) < 1e-6, d2
    print("  (2) 飽和が無ければ floor=0、p_max は一様 1/12: OK")

    # 実際の生成（短い T）でも同じキーが出て、値域が壊れていないこと
    with _short_T() as diff:
        model = _model()
        cond = _cond(2).repeat_interleave(2, dim=0)
        _, _, _, diag = ft.aggregate_step(
            diff, model, cond, 1, 2,
            torch.rand(2, sm.NUM_ACT, sm.NUM_SLOTS),
            torch.ones(2, sm.NUM_ACT, sm.NUM_SLOTS), cond.size(0), "sq")
    assert 0.0 <= diag["x0_floor_frac"] <= 1.0 and diag["st_p_max_mean"] >= 1 / 12
    print(f"  (3) aggregate_step からも取れる（floor={diag['x0_floor_frac']:.3f}）: OK")
    print("test_x0_diagnostics: OK")


def test_val_epsilon_mse() -> None:
    """(val) val の ε-MSE が決定的で、学習の乱数もモードも汚さないこと。

    ★Diffusion.loss は t と ε を大域 RNG から引く。素朴に呼ぶと学習側の乱数列が
      ずれて --resume の再現性が壊れる。また Stage 2 は train と eval を 1 更新の
      中で行き来するので、モードを戻さないと集計側に dropout が乗る。
    """
    with _short_T() as diff:
        model = _model()
        cond_idx = torch.as_tensor(sm.cond_grid()[:8], device=DEVICE)
        sched = torch.randint(0, sm.NUM_ACT, (8, sm.NUM_SLOTS), device=DEVICE)
        loader = [(cond_idx[:4], sched[:4]), (cond_idx[4:], sched[4:])]

        # 同じ seed なら何度呼んでも同じ値（毎回 t を引き直さない）
        a = ft.val_epsilon_mse(diff, model, loader, DEVICE)
        b = ft.val_epsilon_mse(diff, model, loader, DEVICE)
        assert a == b, f"val が決定的でない: {a} vs {b}"
        assert ft.val_epsilon_mse(diff, model, loader, DEVICE, seed=1) != a, \
            "seed を変えても値が変わらない（t/ε を引いていない）"
        print(f"  (1) (val) 同じ seed で決定的（{a:.6f}）、seed を変えると動く: OK")

        # 大域 RNG を汚さない
        torch.manual_seed(123)
        before = torch.randn(3)
        torch.manual_seed(123)
        ft.val_epsilon_mse(diff, model, loader, DEVICE)
        assert torch.equal(torch.randn(3), before), \
            "val の評価が学習側の乱数列をずらしている（--resume が壊れる）"
        print("  (2) 大域 RNG を退避・復元している: OK")

        # モードを戻す
        for want in (True, False):
            model.train(want)
            ft.val_epsilon_mse(diff, model, loader, DEVICE)
            assert model.training is want, "呼び出し前のモードへ戻っていない"
        print("  (3) model のモードも復元する: OK")

    # run() が実際に呼び、ログへ載せること
    src = inspect.getsource(ft.run)
    assert "val_epsilon_mse" in src and "L_atus_val" in src, \
        "run() が val を測っていない"
    print("  (4) run() が --val-every ごとに測って L_atus_val へ載せる: OK")
    print("test_val_epsilon_mse: OK")


def test_memorization_guardrail() -> None:
    """(mem-g) 暗記チェックが参照集合のサイズ交絡を持ち込まないこと。

    ★DCR_gap は「holdout への最近傍距離 − train への最近傍距離」だが、最近傍距離は
      参照集合が大きいほど小さくなる。ATUS 平日は train 3,363 / val 373 で 9 倍違うので、
      間引かないと暗記が無くても gap が正に出る。
    """
    rng = np.random.default_rng(0)
    real = rng.integers(0, sm.NUM_ACT, size=(400, sm.NUM_SLOTS))
    gen = rng.integers(0, sm.NUM_ACT, size=(60, sm.NUM_SLOTS))

    out = se.memorization_guardrail(gen, real, seed=0)
    assert set(out) >= {"dcr_train", "dcr_holdout", "dcr_gap",
                        "exact_copy_rate", "n_ref_per_side"}, out
    # 参照集合は train / holdout とも同数（小さい方＝val に合わせる）
    train_idx, val_idx = sm.split_indices(len(real))
    assert out["n_ref_per_side"] == min(len(train_idx), len(val_idx))
    assert abs(out["dcr_gap"] - (out["dcr_holdout"] - out["dcr_train"])) < 1e-9
    print(f"  (1) (mem-g) 参照集合を両側 {int(out['n_ref_per_side'])} 本へ揃える"
          f"（train {len(train_idx)} / val {len(val_idx)}）: OK")

    # 乱数データなので train 側だけ近いはずがない ＝ gap は 0 の近く
    assert abs(out["dcr_gap"]) < 3.0, f"交絡が残っている: dcr_gap={out['dcr_gap']}"
    print(f"  (2) 無相関データで dcr_gap={out['dcr_gap']:+.3f}（交絡なし）: OK")

    # 事後選択の行に載ること
    src = inspect.getsource(se.evaluate_ckpt)
    assert "memorization_guardrail" in src
    print("  (3) evaluate_ckpt が呼んで CSV の列にする: OK")
    print("test_memorization_guardrail: OK")


def test_lam_warmup_is_robust() -> None:
    """(lam) λ=auto が1更新の外れ値で決まらないこと。

    ★L_agg は split-batch 推定量なので、期待値が bias² でも実現値は大きく振れ、
      負にもなる。初回がたまたま 0 近傍だと λ≈0 が全ステップ固定され、
      リハーサル項が実質無効のまま予算を回し切る。
    """
    assert ft.LAM_WARMUP_STEPS >= 3, "中央値を採るのに更新数が足りない"

    # 1更新目だけ 0 近傍という最悪ケース。中央値なら引きずられない
    l_aggs = [1e-9, 0.020, 0.024, 0.019, 0.022]
    l_atus = [0.50] * len(l_aggs)
    naive = abs(l_aggs[0]) / l_atus[0]
    robust = float(np.median(l_aggs[:ft.LAM_WARMUP_STEPS])
                   / np.median(l_atus[:ft.LAM_WARMUP_STEPS]))
    assert naive < 1e-8, naive
    assert 0.03 < robust < 0.05, robust
    print(f"  (1) (lam) 初回のみ採ると λ={naive:.1e}（リハーサル無効）、"
          f"{ft.LAM_WARMUP_STEPS} 更新の中央値なら λ={robust:.4f}: OK")

    # run() が warmup 中も暫定 λ で回すこと（0 で回す期間を作らない）
    src = inspect.getsource(ft.run)
    assert "lam_samples.append" in src and "np.median" in src, \
        "run() が warmup 中央値で λ を決めていない"
    assert "prev_config.get(\"lam\")" in src, "run() が --resume で λ を引き継いでいない"
    print("  (2) run() が warmup 中央値で決め、--resume では ckpt の λ を継ぐ: OK")
    print("test_lam_warmup_is_robust: OK")


def test_jsd_memory_gate() -> None:
    """(jsd-mem) jsd は chunk を使えないので K×B で予算判定すること。

    ★jsd は群平均の非線形関数で split-batch が使えず、2パスに要る g（∂L/∂ā の
      解析形）を持たない。aggregate_step は chunk を無視して1パスで回るので、
      K×chunk だけを見ると「両方のゲートを通ってから OOM」になる。
    """
    # sq なら chunk が下がって通る設定
    ft.check_memory_budget(1, 28, 256, chunk=ft.resolve_chunk(1, 28, 256))
    # jsd の実ピークは K×B = 7,168 > 6,600 なので落ちなければならない
    try:
        ft.check_memory_budget(1, 28, 256, chunk=28 * 256)
    except SystemExit:
        pass
    else:
        raise AssertionError("jsd 相当の K×B 判定が弾かれていない")

    src = inspect.getsource(ft.run)
    assert 'loss_kind == "jsd"' in src and "chunk=d_sub * n" in src, \
        "run() が jsd を K×B で判定していない"
    print("  (1) (jsd-mem) run() が jsd のとき K×B で予算判定する: OK")
    print("test_jsd_memory_gate: OK")


def test_select_common_random_numbers() -> None:
    """(crn) 事後選択が全 ckpt を同じ乱数列で生成すること。

    ★ck.load_ckpt は学習時の torch RNG を復元する副作用を持つ。seed を置き直さないと
      ckpt ごとに別の乱数列でプールを作ることになり、rate_mae の差がモデル差か
      乱数差か区別できなくなる（n=2000 のセル当たり MC 標準偏差は最大 0.0112、
      rate_mae の水準 0.0288 と同じ桁）。
    """
    src = inspect.getsource(se.evaluate_ckpt)
    load_at = src.index("ck.load_ckpt")
    seed_at = src.index("torch.manual_seed(pool_seed)")
    pool_at = src.index("sm.group_pool")
    assert load_at < seed_at < pool_at, \
        "torch.manual_seed が load_ckpt の後・group_pool の前に無い"
    print("  (1) (crn) load_ckpt -> manual_seed(pool_seed) -> group_pool の順: OK")

    # 出所の列として CSV に残ること
    assert '"pool_seed": pool_seed' in src
    print("  (2) pool_seed が出力行に載る（§9.8 の出所列）: OK")
    print("test_select_common_random_numbers: OK")


def main() -> None:
    test_checkpoint_roundtrip()
    test_reverse_step_matches_inline()
    test_sample_differentiable()
    test_straight_through()
    test_targets_match_reference()
    test_target_properties()
    test_eval_against()
    test_chi2_weights()
    test_agg_loss()
    test_split_batch_unbiased()
    test_loss_grad_and_diagnostics()
    test_jsd_loss()
    test_layered_lr()
    test_memory_budget()
    test_check_shapes()
    test_teacher_mask_and_holdout()
    test_eval_against_subset()
    test_rate_mse_split()
    test_checkpoint_selection()
    test_naive_accumulation_is_wrong()
    test_two_pass_matches_full_batch()
    test_two_pass_memory_scaling()
    test_resolve_chunk()
    test_grad_norms()
    test_x0_diagnostics()
    test_val_epsilon_mse()
    test_memorization_guardrail()
    test_lam_warmup_is_robust()
    test_jsd_memory_gate()
    test_select_common_random_numbers()
    print("\ntest_stage2: OK")


if __name__ == "__main__":
    main()
