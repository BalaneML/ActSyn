"""
stage2_loss.py
==============
Stage 2 の集計損失

日本の公表集計表 A* に生成分布を合わせるための損失

損失:
    dist(p,q) = (1/12) Σ_c ω_c (p_c − q_c)²,   ω_c = (1/(q_c+ε)) / mean(1/(q+ε))

    L_agg   = (1 / (|D_sub|·96)) Σ_{d∈D_sub} Σ_{s=0}^{95} dist( Ã[d,·,s], A*[d,·,s] )
            = (1 / (|D_sub|·12·96)) Σ_{d∈D_sub} Σ_{c=0}^{11} Σ_{s=0}^{95} ω[d,c,s] · ( Ã[d,c,s] − A*[d,c,s] )²

    ω[d,c,s] = ( 1 / (A*[d,c,s] + ε) ) / ( (1/(28·12·96)) Σ_{d',c',s'} 1/(A*[d',c',s'] + ε) )
        ε=inf   候補A  素の MSE。勾配配分が実誤差分布と一致し、報告指標 rate_mae と整合
        ε=0.01  候補B  χ²。逆分散重みに近い側で、稀活動と相対誤差を取りに行く
        ε=0.05  掃引 AとBの中間
        JSD     アブレーション1点のみ。f-ダイバージェンスを採らない理由を実験で示す

使い方:
    a_star = teacher_tensor(tgt, device)             # (28,12,96)
    omega  = chi2_weights(a_star, eps=float("inf"))  # A なら全要素1
    ...
    y = straight_through(diffusion.sample_differentiable(model, cond, K), tau)
    a_A, a_B = group_rates_split(y, n)
    L_agg = agg_loss_from_rates(a_A, a_B, a_star[d_sub], omega[d_sub])
"""
import math

import numpy as np
import torch

# JSD用
JSD_FLOOR = 1e-12


def teacher_tensor(tgt: dict,
                    device: str | torch.device,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """stage2_targets.load_stula_targets の教師A*を (28,12,96) の torch テンソルにする

    Args:
        tgt:
            target tensor, dict{行動者率, 推定人口}
            tgt["group_rates_tbl"][d]=郡dの行動者率 (12, 96)
            tgt["group_rates_tbl"][d, c, s]=郡d, 活動c, 時刻区分sの行動者率, [0, 1]
            tgt["pop"][g, a, e]=郡d (gender, age, employment) の推定人口
        device: 出力テンソルの置き場
        dtype: 出力テンソルのdtype, float64->float32

    Returns:
        教師 A* (28, 12, 96)
    """
    a = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)
    n_nan = int(np.isnan(a).sum())
    if n_nan:
        raise ValueError(f"教師に NaN が {n_nan} セルある。teacher_mask で除くか前処理を見直すこと")
    return torch.as_tensor(a, dtype=dtype, device=device)


def chi2_weights(a_star: torch.Tensor, eps: float) -> torch.Tensor:
    """教師 A* (28,12,96) から損失の重み ω を作る

    ω[d,c,s] = ( 1 / (A*[d,c,s] + ε) ) / mean_{d',c',s'}( 1 / (A*[d',c',s'] + ε) )

    誤差を重く見るセルを求める
    ε=inf: 全セル一様, 素の MSE
    ε が小さいほど率の低いセル（稀な活動）の相対誤差を重く見る χ² に寄る

    Args:
        a_star: 教師 A* (28, 12, 96), 値域[0, 1]
        eps:
            ε=inf: 素の MSE
            ε=0.01: χ²誤差

    Returns:
        各セルの重みω, (28, 12, 96)
    """
    if math.isinf(eps):
        return torch.ones_like(a_star)
    if eps <= 0.0:
        raise ValueError(f"eps は正か inf でなければならない: {eps}")
    w = 1.0 / (a_star + eps)
    return w / w.mean()


def group_rates_split(y: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(D_sub*n, 12, 96) の one-hot を群ごとに半分ずつ割り, それぞれの群平均して ā_A, ā_B (D_sub,12,96) を2つ返す
    
    split-batch不偏推定のための分割
    重複しない2つの半分から独立の郡平均を2つ作り, 内積をとることで多様性への罰 (1/n)tr Var を消す

    Note:
        ★A/B の分け方が群をまたがないこと（群ごとに半分ずつ）が要点。群をまたいで
        分けると ā_A と ā_B が別の群の平均になり、内積が bias² を推定しなくなる。
        前半・後半という決め打ちでよいのは、群内の n 本が同じ cond から独立なノイズで
        生成された i.i.d. だからで、必要なのは2つの半分が独立であることだけである。
        ★並びが群優先であることは検査できない。y の中身から群の境界は読めないので、
        呼び出し側が並びを崩しても例外は出ず、別の群の本が混ざった平均が返る。
        ★d_sub は y.size(0) // n で逆算する。n を取り違えても割り切れれば通ってしまう
        （B=512 の n=256 へ n=128 を渡すと d_sub=4 として通る）。群の境界が半分ずれた
        まま学習が進み、損失値は出るのに合わせる先が違う、という壊れ方をする。

    Args:
        y: 微分可能なone-hot, dtype=float32, (D_sub*n, IN_CH, NUM_SLOTS) = (D_sub*n, 12, 96)
        n: 郡あたりの生成本数, 偶数であることが必須

    Returns:
        (ā_A, ā_B), dtype=float32, (D_sub, 12, 96)
        ā_A[d] = 群dの前半 n/2 本の平均, ā_B[d] = 群dの後半 n/2 本の平均
        y の勾配グラフを保つので、そのまま損失へ繋いで逆伝播できる
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
                        a_star: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """split-batch 不偏推定による集計損失 L_agg を1つのスカラーで返す

    L = mean_{d,c,s} ω[d,c,s] · (ā_A[d,c,s] − A*[d,c,s]) · (ā_B[d,c,s] − A*[d,c,s])
    二乗ではなく, 重複しない2つの半分から作った誤差の積をとる
    群 d・活動 c・時刻 s の3軸すべてで平均する

    Note:
        ★ā_A ⊥ ā_B なので E[L] = Σ ω (E[ā] − A*)² となり, 多様性への罰 (1/n)tr Var が
        厳密に消える。素朴な二乗誤差はこの項を含むので, 集計を合わせるほど群内の個票が互いに似ていく圧力がかかる。
        ★値が負になりうる。ā_A と ā_B の誤差の符号が逆のセルでは積が負になる。
        損失値として不自然に見えるが, 期待値が bias² で下限0なのは変わらない
        (推定量の分散が見えているだけ)。学習ログの l_agg < 0 は異常ではない。
        ★4引数の群の並びが一致していることが前提。a_A, a_B は cond を作った順,
        a_star, omega は d_pick_t で切り出した順であり, 呼び出し側が揃える。
        ずれても例外は出ず, 別の群の教師に合わせにいく。
        ★ε=inf では ω が全要素1なのでセル単位の素の MSE と厳密に一致し, rate_rmse と
        同じ尺度で読める。有限の ε の ω は28群全体で平均1に正規化されているため,
        D_sub 群だけを切り出したこの計算では平均はちょうど1にならない
        (ステップ間で L_agg を比較できるようにするための意図した挙動)。

    Args:
        a_A: 群dの前半 n/2 本の平均 ā_A, dtype=float32, (D_sub, 12, 96), 勾配グラフを保つ
        a_B: 群dの後半 n/2 本の平均 ā_B, dtype=float32, (D_sub, 12, 96), 勾配グラフを保つ
        a_star: 教師 A* のうち当該ステップの D_sub 群, (D_sub, 12, 96), 値域[0, 1]
        omega: 損失の重み ω のうち当該ステップの D_sub 群, (D_sub, 12, 96)

    Returns:
        集計損失 L_agg, 0次元テンソル (スカラー), dtype=float32
        a_A, a_B 経由の勾配グラフを保つので, そのまま backward() できる
    """
    return ((a_A - a_star) * (a_B - a_star) * omega).mean()


def agg_loss(y: torch.Tensor, a_star: torch.Tensor,
                omega: torch.Tensor,n: int) -> torch.Tensor:
    """生成 one-hot -> 分割 -> 集計損失 L_agg

    group_rates_split   -> 生成スケジュール y を ā_A, ā_B に分ける
    agg_loss_from_rates -> 損失にする

    Args:
        y: 微分可能なone-hot, dtype=float32, (D_sub*n, 12, 96)
        a_star: 教師 A* のうち当該ステップの D_sub 群, (D_sub, 12, 96), 値域[0, 1]
        omega: 損失の重み ω のうち当該ステップの D_sub 群, (D_sub, 12, 96)
        n: 群あたりの生成本数, 偶数であることが必須

    Returns:
        集計損失 L_agg, 0次元テンソル (スカラー), dtype=float32
        y 経由の勾配グラフを保つので, そのまま backward() できる
    """
    a_A, a_B = group_rates_split(y, n)
    return agg_loss_from_rates(a_A, a_B, a_star, omega)


def jsd_loss(y: torch.Tensor, a_star: torch.Tensor, n: int,
             floor: float = JSD_FLOOR) -> torch.Tensor:
    """JSD"""
    a = y.view(y.size(0) // n, n, y.size(1), y.size(2)).mean(dim=1)
    m = 0.5 * (a + a_star)

    def _kl(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return (p * (torch.log(p.clamp_min(floor)) - torch.log(r.clamp_min(floor)))).sum(dim=1)

    return (0.5 * _kl(a, m) + 0.5 * _kl(a_star, m)).mean()


# ============================================================
# g = ∂L/∂Ã の解析形と診断
# ============================================================
def loss_grad(a_A: torch.Tensor, a_B: torch.Tensor, a_star: torch.Tensor,
                omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """郡平均への勾配 g_A = ∂L/∂ā_A, g_B = ∂L/∂ā_B を解析的に返す (autograd を使わない)

    g_A = ω·(ā_B − A*) / (|D_sub|·12·96)
    g_B = ω·(ā_A − A*) / (|D_sub|·12·96)

    Args:
        a_A: 群dの前半 n/2 本の平均 ā_A, dtype=float32, (D_sub, 12, 96)
        a_B: 群dの後半 n/2 本の平均 ā_B, dtype=float32, (D_sub, 12, 96)
        a_star: 教師 A* のうち当該ステップの D_sub 群, (D_sub, 12, 96), 値域[0, 1]
        omega: 損失の重み ω のうち当該ステップの D_sub 群, (D_sub, 12, 96)

    Returns:
        (g_A, g_B), dtype=float32, ともに (D_sub, 12, 96)
        g_A[d, c, s] = 群d, 活動c, 時刻区分s の ā_A を動かすべき向きと大きさ
    """
    scale = a_A.size(0) * a_A.size(1) * a_A.size(2)    # |D_sub| × 12 × 96（3軸とも平均）
    return omega * (a_B - a_star) / scale, omega * (a_A - a_star) / scale


def per_sample_grad(g_A: torch.Tensor, g_B: torch.Tensor, n: int) -> torch.Tensor:
    """群平均への勾配 g_A, g_B を個票ごとの上流勾配 ∂L/∂y_i へ展開する

    2パス勾配蓄積 (§2.9, §11.2) で y_c.backward(gradient=...) に渡す値そのもの
    群 d・半分 h に属する個票 i について

        ∂L/∂y_i = (∂L/∂ā_h[d]) · (∂ā_h[d]/∂y_i) = g_h[d] / (n/2)

    同じ半分に属する個票は全員が同じ勾配を受け取る (群 d と半分 h だけで決まる)
    group_rates_split の逆操作であり, 並びの規約を共有する

    Note:
        ★割る数は n ではなく n/2。ā_A は n/2 本の平均なので ∂ā_A/∂y_i = 1/(n/2) である。
        §2.9 の説明は split-batch を使わない素朴版 (g/n) なので取り違えないこと。
        ★並びは group_rates_split と同じ「群優先・群の中は前半がA・後半がB」。
        cat の順序がこの規約に対応しているため, 片方だけ変えると壊れる。
        ★並びは検査できない。y の並びがずれていても例外は出ず, 別の群の勾配を
        別の群の個票へ流したまま学習が進む。
        ★検算: 群dのA半分の勾配を全て足すと half·g_A[d]/half = g_A[d] となり,
        群平均側の勾配に一致する。

    Args:
        g_A: 群平均 ā_A への勾配 ∂L/∂ā_A, dtype=float32, (D_sub, 12, 96)
        g_B: 群平均 ā_B への勾配 ∂L/∂ā_B, dtype=float32, (D_sub, 12, 96)
        n: 群あたりの生成本数, 偶数であることが必須

    Raises:
        ValueError: n が奇数のとき
        RuntimeError: g_B の形が g_A と一致しないとき (expand が送出する)

    Returns:
        個票ごとの上流勾配 ∂L/∂y_i, dtype=float32, (D_sub*n, 12, 96)
        並びは y と同一なので, チャンク分割してそのまま backward(gradient=...) に渡せる
    """
    if n % 2 != 0:
        raise ValueError(f"split-batch には n が偶数である必要がある: {n}")
    d_sub, n_act, n_slot = g_A.shape
    half = n // 2
    per = torch.cat([g_A.unsqueeze(1).expand(d_sub, half, n_act, n_slot),
                    g_B.unsqueeze(1).expand(d_sub, half, n_act, n_slot)], dim=1)
    return (per / half).reshape(d_sub * n, n_act, n_slot)


def g_diagnostics(g: torch.Tensor, a_star: torch.Tensor, n: int,
                    act_names: list[str] | None = None) -> dict[str, float]:
    """勾配 g の要約統計を dict で返す, g の追跡で学習過程を評価する

    g は「集計をどちらへどれだけ動かしたいか」を書いた (D_sub, 12, 96) の表である
    全ステップ記録すれば, 教師のどの部分が実際に効いているかを勾配という直接の量で測れる

    Note:
        ★本番では g_A だけを渡す。g_A = ω(ā_B − A*)/S なので, 読んでいるのは
        「B半分の誤差が A半分へ流す勾配」である。
        ★g_abs_mean と g_abs_max は 1/(|D_sub|·12·96) を含むため, D_sub や ε を
        変えると比較できない。同一設定内の時間変化を見る量である。
        残る3つ (g_frac_negligible, g_share_unidentifiable, g_sign_pos_frac) と
        g_share_<活動> は g の定数倍で変わらないので設定をまたいで比較できる。
        ★σ の分母は n だが, g_A を駆動する ā_B は n/2 本の平均であり揺らぎは √2 倍。
        識別可能性の閾値を「全 n 本で報告する群平均」基準で定めた結果であり,
        g_A に対する判定としては緩い側に立つ。
        ★jsd では呼ばない。loss_grad が二次形式の勾配なので, 別の損失の勾配を
        報告することになる (呼び出し側で分岐している)。

    Args:
        g: 群平均への勾配 ∂L/∂ā, dtype=float32, (D_sub, 12, 96)
        a_star: 教師 A* のうち当該ステップの D_sub 群, (D_sub, 12, 96), 値域[0, 1]
        n: 群あたりの生成本数。σ = √(A*(1−A*)/n) の計算にだけ使う
        act_names: 活動12種の名前。渡したときだけ g_share_<活動> を追加する

    Returns:
        統計量の dict[str, float], wandb へ log.update() で流し込める
            g_abs_mean: |g| の平均。押している量の水準
            g_abs_max: |g| の最大。突出して押されているセルの大きさ
            g_frac_negligible: |g| < 0.01·mean|g| のセルの割合 = 効いていない部分の大きさ
            g_share_unidentifiable: A* <= 2σ のセルへ配られた |g| のシェア
                σ は生成 n 本の標本平均が持つ標準偏差を率 A* で評価した値であり,
                A* <= 2σ は教師の信号が生成側の揺らぎに埋もれて合わせようがないセルを指す
            g_sign_pos_frac: g > 0 のセルの割合。正は「その活動を減らしたい」
                (勾配降下は −g 方向へ動かすため)。ステップをまたぐと符号の一貫性が読める
            g_share_<活動>: 群軸と時刻軸で潰した活動別の |g| シェア。12個の合計が1
    """
    with torch.no_grad():
        g_abs = g.abs()
        mean_abs = float(g_abs.mean())
        # 教師セルが持つ MC ノイズ。率 A* のセルで n 本生成したときの標準偏差
        sigma = torch.sqrt(torch.clamp(a_star * (1.0 - a_star), min=0.0) / n)
        out: dict[str, float] = {
            "g_abs_mean": mean_abs,
            "g_abs_max": float(g_abs.max()),
            # 効いていない部分。全体平均の1%未満しか押されていないセル
            "g_frac_negligible": float((g_abs < 0.01 * mean_abs).float().mean()),
            # 教師が識別できないセル（A* <= 2σ）が損失に占める勾配シェア
            "g_share_unidentifiable": float(
                g_abs[a_star <= 2.0 * sigma].sum() / g_abs.sum()) if float(g_abs.sum()) > 0 else 0.0,
            "g_sign_pos_frac": float((g > 0).float().mean()),
        }
        if act_names is not None:
            share = g_abs.sum(dim=(0, 2)) / g_abs.sum()
            for i, name in enumerate(act_names):
                out[f"g_share_{name}"] = float(share[i])
        return out
