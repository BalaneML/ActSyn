"""
stage2_loss.py
==============
Stage 2 の集計損失（Stage2_design.md §6, §7）

日本の公表集計表 A* に生成分布を合わせるための損失。設計上の要点は4つ。

1. 12チャネルをそのまま教師にする（§6 修正3）
   OTHER_X を損失から外すと「損失に現れないまま確率質量を奪い合う無拘束の自由度」に
   なる。11活動へ再正規化して打ち消す案は実測で悪化した（rate_mae 0.02876 -> 0.02902）
   ため撤回。再正規化という非線形段が消えるので split-batch が厳密に不偏になる副次効果もある。

2. 集計は損失の内側で取る（§6 修正2）
   1本ずつ集計目標に合わせると平均へ引き寄せる力になり、群内多様性が潰れる。

3. split-batch 不偏推定（§7.5）
   集計してからでも E‖(1/n)Σy − A*‖² = ‖bias‖² + (1/n)tr Var となり、第2項が
   多様性そのものへの罰として残る。n本をA/Bに二分して内積を取ると ȳ_A ⊥ ȳ_B より
   この項が厳密に消える。追加コストはゼロ。

4. 群をまたぐ集約は等重み（§7.2）
   人口加重にしない。群人口シェアは18.7倍の開きがあるので、人口加重にすると
   小さい群がほぼ学習されない。評価側 (eval_against) もセル単位の単純平均なので揃う。

損失の形（§7.4 の決定）:

    dist(p,q) = (1/12) Σ_c ω_c (p_c − q_c)²,   ω_c = (1/(q_c+ε)) / mean(1/(q+ε))

    ★活動数 12 で割る。こうすると ε=inf で損失が「セル単位の平均二乗誤差」そのものになり、
      報告指標（rate_mae / rate_rmse はいずれも (d,c,s) 全セルの平均）と同じ尺度になる
      （√L_agg ≈ rate_rmse）。割らないと MSE の 12 倍の量を MSE と呼ぶことになる。
      学習への影響はない：--lam auto は λ = |L_agg| / L_atus と置くので定数 1/12 が
      そのまま λ に吸収され、損失全体が一様に 1/12 倍されるだけで AdamW は不変。

    ε=inf   主A  素の MSE。勾配配分が実誤差分布と一致し、報告指標 rate_mae と整合
    ε=0.01  主B  χ²。逆分散重みに近い側で、稀活動と相対誤差を取りに行く
    ε=0.05  掃引 AとBの中間
    JSD          アブレーション1点のみ。f-ダイバージェンスを採らない理由を実験で示す
    TV/Hellinger²/KL  却下。実装しない

★重みの平均1への正規化は必須。正規化しないと χ² は勾配が24倍大きくなり、
  損失の違いと実効学習率の違いが交絡して比較にならない。

★λ でリハーサル項と足し合わせるのはこのモジュールではなく学習ループ側の仕事。
  L_agg.backward() と (λ·L_atus).backward() を別々に呼ぶとピークメモリが
  max(両者) で済み和にならない（§4.5）ので、合成したスカラーを返さない。

使い方:
    q     = teacher_tensor(tgt, device)              # (28,12,96)
    omega = chi2_weights(q, eps=float("inf"))        # 主A なら全要素1
    ...
    y = straight_through(diffusion.sample_differentiable(model, cond, K), tau)
    a_A, a_B = group_rates_split(y, n)
    L_agg = agg_loss_from_rates(a_A, a_B, q[d_sub], omega[d_sub])
"""
import math

import numpy as np
import torch

# JSD の対数に入れる床の既定値。★この値で勾配配分が大きく動く（0<q<=.01 帯へ配る
# 割合が床 1e-12 で 87%、床 1/256 で 39%）ので、報告時は必ず床の値を明記する。
# 二次形式（MSE / χ²）は床を必要とせず再現可能で、§7.4 の決定はそちらに依存している。
JSD_FLOOR = 1e-12


def teacher_tensor(tgt: dict, device: str | torch.device,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """stage2_targets.load_stula_targets の教師を (28,12,96) の torch テンソルにする。

    ★全国・平日は非公表セルが0個なので NaN は現れない。地域軸へ拡張したときに
      NaN が入りうるので、ここで検出して落とす（黙って NaN を損失へ流さない）。
    """
    a = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    n_nan = int(np.isnan(a).sum())
    if n_nan:
        raise ValueError(f"教師に NaN が {n_nan} セルある。teacher_mask で除くか前処理を見直すこと")
    return torch.as_tensor(a, dtype=dtype, device=device)


def chi2_weights(q: torch.Tensor, eps: float) -> torch.Tensor:
    """教師 q (D,12,96) から二次形式の重み ω を作る。★全体平均が厳密に 1 になる。

    args:
        eps: 逆分散重み 1/q の発散を止める床。inf で全要素1（＝素の MSE）。
             ε→∞ が素の MSE と厳密に一致するので、主A と主B は同じ1本のコードで走る。

    ★平均1への正規化を全セル (d,c,s) にわたって行う。群ごとや活動ごとに正規化すると
      「どの群・どの活動を重く見るか」という別の設計判断が混入する。ここで決めたいのは
      率の水準に応じた重み付けだけなので、正規化は損失全体のスケールを揃える目的に留める。
    """
    if math.isinf(eps):
        return torch.ones_like(q)
    if eps <= 0.0:
        raise ValueError(f"eps は正か inf でなければならない: {eps}")
    w = 1.0 / (q + eps)
    return w / w.mean()


def group_rates_split(y: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(D_sub*n, 12, 96) の one-hot を群ごとに半分ずつ平均して (D_sub,12,96) を2つ返す。

    ★入力は群優先の並び（cond_grid()[d_sub].repeat_interleave(n, dim=0) の順）。
      先頭 n/2 本が A、残り n/2 本が B。同じ本が両方の半分に入ってはいけない。
    ★A/B の分け方が群をまたがないこと（群ごとに半分ずつ）が要点。群をまたいで
      分けると ȳ_A と ȳ_B が別の群の平均になり、内積が bias² を推定しなくなる。
    """
    if n % 2 != 0:
        raise ValueError(f"split-batch には n が偶数である必要がある: {n}")
    d_sub = y.size(0) // n
    if d_sub * n != y.size(0):
        raise ValueError(f"本数 {y.size(0)} が n={n} で割り切れない")
    grouped = y.view(d_sub, n, y.size(1), y.size(2))
    half = n // 2
    return grouped[:, :half].mean(dim=1), grouped[:, half:].mean(dim=1)


def agg_loss_from_rates(a_A: torch.Tensor, a_B: torch.Tensor,
                        q: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """split-batch 不偏推定の集計損失。すべて (D_sub,12,96) で群は揃えて渡す。

        L = mean_{d,c,s} ω[d,c,s] · (ā_A − q)[d,c,s] · (ā_B − q)[d,c,s]

    ★群 d・活動 c・時刻 s の 3 軸とも平均（§7.1(5) の定義）。ω は全セル平均 1 なので
      ε=inf ではこれがそのままセル単位の MSE になり、rate_rmse と同じ尺度で読める。
    ★ā_A ⊥ ā_B なので E[L] = Σ ω (E[ā]−q)² となり、多様性への罰 (1/n)tr Var が
      厳密に消える。素朴な二乗和はこの項を含むので、集計を合わせるほど群内の
      個票が互いに似ていく圧力がかかる。
    ★値が負になりうる。ā_A と ā_B の誤差の符号が逆なら内積が負になる。損失として
      不自然に見えるが、期待値が bias² で下限0なのは変わらない（推定量の分散の話）。
    """
    return ((a_A - q) * (a_B - q) * omega).mean()


def agg_loss(y: torch.Tensor, q: torch.Tensor, omega: torch.Tensor,
             n: int) -> torch.Tensor:
    """生成 one-hot から集計損失まで一息に。group_rates_split + agg_loss_from_rates。

    2パス勾配蓄積では Ã の推定と勾配計算を分けたいので、その場合は
    group_rates_split と agg_loss_from_rates を個別に呼ぶ。
    """
    a_A, a_B = group_rates_split(y, n)
    return agg_loss_from_rates(a_A, a_B, q, omega)


def jsd_loss(y: torch.Tensor, q: torch.Tensor, n: int,
             floor: float = JSD_FLOOR) -> torch.Tensor:
    """アブレーション用の JSD。★split-batch は使えない（二次形式でないため）。

    採らない理由を実験で示すために1点だけ回す。実測では勾配の91.9%が MC ノイズを
    追いかけており（n=256）、しかもそれは分散ではなくバイアス（Jensen ギャップ）なので
    ステップ平均で消えない。n=1024 でも71.4%、n=4096 でも38.7%で現実的な n では改善しない。

    ★split-batch を使わないので、この損失には多様性への罰 (1/n)tr Var が乗ったままである。
      MSE/χ² と比べるときは「損失の形」と「不偏化の有無」の2つが同時に変わっている
      ことを踏まえて読むこと。
    """
    a = y.view(y.size(0) // n, n, y.size(1), y.size(2)).mean(dim=1)
    m = 0.5 * (a + q)

    def _kl(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # p=0 のセルは p*(...) が 0 になるので床は log の中だけに入れる
        return (p * (torch.log(p.clamp_min(floor)) - torch.log(r.clamp_min(floor)))).sum(dim=1)

    return (0.5 * _kl(a, m) + 0.5 * _kl(q, m)).mean()


# ============================================================
# g = ∂L/∂Ã の解析形と診断（§2.3, §11.3）
# ============================================================
def loss_grad(a_A: torch.Tensor, a_B: torch.Tensor, q: torch.Tensor,
              omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """g_A = ∂L/∂ā_A と g_B = ∂L/∂ā_B を解析的に返す。autograd を使わない。

    ★split-batch なので、A半分に流す勾配は B半分の誤差で決まる（その逆も同じ）:
        g_A = ω·(ā_B − q) / (|D_sub|·12·96),   g_B = ω·(ā_A − q) / (|D_sub|·12·96)
      素朴版の g = 2(Ã − A*) とは形が違う。2パス蓄積で g を注入するときに
      取り違えないこと。

    用途は2つ:
      - 学習過程の診断（§11.3）。追加コストなしに「教師のどの部分が効いているか」が測れる
      - 2パス勾配蓄積（§10.2-11）で上流勾配として注入する値そのもの
    """
    scale = a_A.size(0) * a_A.size(1) * a_A.size(2)    # |D_sub| × 12 × 96（3軸とも平均）
    return omega * (a_B - q) / scale, omega * (a_A - q) / scale


def per_sample_grad(g_A: torch.Tensor, g_B: torch.Tensor, n: int) -> torch.Tensor:
    """群平均への勾配 g_A/g_B を個票ごとの上流勾配 (D_sub*n, 12, 96) へ展開する。

    2パス勾配蓄積（Stage2_design.md §2.9, §11.2）で `y_c.backward(gradient=...)` に
    渡す値そのもの。群 d・半分 h に属する個票 i について

        ∂L/∂y_i = ∂L/∂ā_h[d] · ∂ā_h[d]/∂y_i = g_h[d] / (n/2)

    ★割る数は n ではなく n/2。ā_A は n/2 本の平均なので ∂ā_A/∂y_i = 1/(n/2) である。
      §2.9 の説明は split-batch を使わない素朴版（g/n）で書いてあるので取り違えないこと。
    ★並びは group_rates_split と同じ「群優先・群の中は前半がA・後半がB」。
      ここがずれると別の群の勾配を別の群の個票へ流すことになり、しかも例外は出ない。
    """
    if n % 2 != 0:
        raise ValueError(f"split-batch には n が偶数である必要がある: {n}")
    d_sub, n_act, n_slot = g_A.shape
    half = n // 2
    per = torch.cat([g_A.unsqueeze(1).expand(d_sub, half, n_act, n_slot),
                     g_B.unsqueeze(1).expand(d_sub, half, n_act, n_slot)], dim=1)
    return (per / half).reshape(d_sub * n, n_act, n_slot)


def g_diagnostics(g: torch.Tensor, q: torch.Tensor, n: int,
                  act_names: list[str] | None = None) -> dict[str, float]:
    """g の統計を dict で返す。生成済みの量だけで計算できるので毎更新で呼べる。

    設計書 §11.3「g の追跡で学習過程を評価する」。g は (12,96) の「集計をどちらへ
    どれだけ動かしたいか」を書いた表なので、全ステップ記録すれば「教師のどの部分が
    実際に効いているか」を勾配という直接の量で測れる。

    返す量:
        g_abs_mean / g_abs_max   押している量の水準
        g_frac_negligible        全体平均の1%未満しか押されていないセルの割合
                                 ＝ 効いていない部分の大きさ
        g_share_unidentifiable   q <= 2σ のセル（生成 n 本では教師の信号と
                                 サンプリング揺らぎを区別できないセル）へ配られた
                                 |g| のシェア。σ = √(q(1−q)/n)
        g_sign_pos_frac          正（その活動を減らしたい）のセルの割合。
                                 学習ステップをまたいで見ると符号の一貫性が読める
        g_share_<活動>           活動別の |g| シェア（act_names を渡したときのみ）

    args:
        n: 1群あたりの生成本数。σ の計算にだけ使う
    """
    with torch.no_grad():
        g_abs = g.abs()
        mean_abs = float(g_abs.mean())
        # 教師セルが持つ MC ノイズ。率 q のセルで n 本生成したときの標準偏差
        sigma = torch.sqrt(torch.clamp(q * (1.0 - q), min=0.0) / n)
        out: dict[str, float] = {
            "g_abs_mean": mean_abs,
            "g_abs_max": float(g_abs.max()),
            # 効いていない部分。全体平均の1%未満しか押されていないセル
            "g_frac_negligible": float((g_abs < 0.01 * mean_abs).float().mean()),
            # 教師が識別できないセル（q <= 2σ）が損失に占める勾配シェア
            "g_share_unidentifiable": float(
                g_abs[q <= 2.0 * sigma].sum() / g_abs.sum()) if float(g_abs.sum()) > 0 else 0.0,
            "g_sign_pos_frac": float((g > 0).float().mean()),
        }
        if act_names is not None:
            share = g_abs.sum(dim=(0, 2)) / g_abs.sum()
            for i, name in enumerate(act_names):
                out[f"g_share_{name}"] = float(share[i])
        return out
