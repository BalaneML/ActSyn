"""
stage2_checkpoint.py
====================
Stage 2 の定期チェックポイントと再開（Stage2_design.md §4.8）

なぜ最初に要るか:
    Stage 1 の学習は 5 分で終わるので「最後に 1 回保存する」で足りていた
    （model.py:673-676）。Stage 2 は 1 パラメータ更新が生成 1 回ぶん（実測 155 秒、
    B=7,168）かかり、300 更新で 3〜13 時間になる。SQUID の elapstim_req を超過すると
    保存前に強制終了され、途中再開の口が無いと学習が丸ごと失われる。

atomic な差し替えについて:
    強制終了は保存の最中にも来る。torch.save で既存ファイルを直接上書きすると、
    書き込み途中で落ちたときに「古いチェックポイントごと壊れる」。
    一時ファイルへ書いてから Path.replace で差し替えれば、結果は
    「古いものがそのまま残る」か「新しいものに切り替わる」かのどちらかにしかならない。
    Path.replace は同一ファイルシステム上で atomic であり、rename と違って
    宛先が存在していても失敗しない。

RNG 状態を保存する理由:
    Stage 2 は毎更新で x_T ~ N(0,I) を引き、群サブサンプリングも乱数で行う。
    RNG を復元しないと、再開した学習は「同じ設定の別の実験」になる。

    ★乱数源は2つある。どちらか片方だけ戻しても再開は元の学習の続きにならない。
        torch  : x_T ~ N(0,I) と末尾 K 区間の雑音 zs（save/load が常に扱う）
        numpy  : 群サブサンプリング d_pick（np_rng 引数を渡したときだけ扱う）
      numpy 側を渡し忘れると、再開後の d_pick が step 1 からの並びを繰り返す。
      例外は出ず、学習ログも正常に見える壊れ方をする。

使い方:
    from stage2_checkpoint import save_ckpt, load_ckpt, ckpt_path, latest_ckpt

    # 学習ループの中（np_rng は群サブサンプリングに使っている Generator そのもの）
    if step % save_every == 0:
        save_ckpt(ckpt_path(ckpt_dir, step), model, optimizer, step, config, np_rng=rng)

    # 再開（rng は in-place で保存時の状態へ戻る）
    path = latest_ckpt(ckpt_dir)
    start_step = 0 if path is None else load_ckpt(path, model, optimizer, np_rng=rng)[0]
"""
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ckpt_path が作り latest_ckpt が読む命名規則。両者の唯一の出所として置く
CKPT_STEM = "stage2_step"
_CKPT_RE = re.compile(rf"^{CKPT_STEM}(\d+)\.pt$")


def ckpt_path(ckpt_dir: Path, step: int) -> Path:
    """step 世代のチェックポイントのパス。

    ★step を名前に含めて複数世代を残す。Stage 2 は早期終了を使わず固定ステップ予算で
      回し切り、学習後に (教師適合, ガードレール) の2軸で選ぶ（§8.4 事後チェックポイント選択）。
      1 ファイルを上書きし続けると、この事後選択ができない。
    """
    return ckpt_dir / f"{CKPT_STEM}{step}.pt"


def latest_ckpt(ckpt_dir: Path) -> Path | None:
    """step が最大のチェックポイント。1 つも無ければ None（＝新規に学習を始める）。

    ★step は数値として比較する。文字列順だと stage2_step9 > stage2_step10 になる。
    """
    if not ckpt_dir.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in ckpt_dir.iterdir()
             if (m := _CKPT_RE.match(p.name)) is not None]
    return max(found)[1] if found else None


def save_ckpt(path: Path,
              model: nn.Module,
              optimizer: torch.optim.Optimizer,
              step: int,
              config: dict[str, Any] | None = None,
              np_rng: np.random.Generator | None = None) -> None:
    """チェックポイントを atomic に保存する。

    Args:
        path: 保存先。ckpt_path で作る
        model: 学習中のモデル（凍結 Stage 1 モデルではない）
        optimizer: AdamW。1次・2次モーメントを含めないと再開時に挙動が変わる
        step: 完了済みの更新回数。再開はこの次から始まる
        config: K, n, chunk, lambda など。どの設定で作られた重みかを重みと一緒に残す
        np_rng: 群サブサンプリングに使っている numpy の Generator。
            渡すと bit_generator.state を保存し、load_ckpt が同じ Generator を
            その状態へ戻せるようになる。None なら numpy 側は保存しない
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # ★numpy 側は dict（BitGenerator の状態）。torch の ByteTensor とは別物なので
        #   キーを分けてある。np_rng を渡さなかった場合は None を書いて区別できるようにする
        "numpy_rng": np_rng.bit_generator.state if np_rng is not None else None,
        "config":    config if config is not None else {},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)   # ★atomic。途中で落ちても既存の path は壊れない


def load_ckpt(path: Path,
              model: nn.Module,
              optimizer: torch.optim.Optimizer | None = None,
              map_location: str | torch.device | None = None,
              np_rng: np.random.Generator | None = None) -> tuple[int, dict[str, Any]]:
    """チェックポイントを読み、model / optimizer / RNG を復元して (step, config) を返す。

    Args:
        model: 重みを書き込む先。state_dict の形が保存時と一致している必要がある
        optimizer: None なら optimizer 状態を復元しない（評価だけしたい場合）
        map_location: 保存時と違うデバイスで読むときに指定する
        np_rng: 群サブサンプリングに使っている numpy の Generator。
            渡すと保存時の状態へ **in-place で** 戻す（新しい Generator は返さない）。
            チェックポイントに numpy_rng が無い（古い世代、または np_rng 無しで
            保存した）場合は何もしない

    Returns:
        (step, config) の2要素タプル
            step: 完了済みの更新回数。学習の再開はこの次から始める
            config: save_ckpt に渡した設定 dict のコピー

    Note:
        ★weights_only=False で読む。RNG 状態（torch.ByteTensor）と config を含むため。
          読むのは自分で書いたファイルだけなので信頼できる。
        ★torch の RNG はこの関数を呼ぶだけで必ず書き換わる。評価目的で複数の
          チェックポイントを読むときは、読んだ後に torch.manual_seed で
          共通乱数へ揃え直すこと（stage2_select.evaluate_ckpt がそうしている）。
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    # RNG は CPU 側を必ず戻す。CUDA 側は保存時と同じ基数のときだけ戻す
    # （GPU 数が変わると set_rng_state_all が落ちるので、学習は続けられる側に倒す）
    # ★.cpu() が必須。map_location を指定すると RNG 状態のテンソルまでそのデバイスへ
    #   移り、set_rng_state が "RNG state must be a torch.ByteTensor" で落ちる
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    cuda_rng = ckpt.get("cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available() \
            and len(cuda_rng) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng])
    # numpy 側（群サブサンプリング）。torch と違い、受け皿の Generator を
    # 呼び出し側から渡してもらわないと戻せない
    numpy_rng = ckpt.get("numpy_rng")
    if np_rng is not None and numpy_rng is not None:
        np_rng.bit_generator.state = numpy_rng
    return int(ckpt["step"]), dict(ckpt.get("config", {}))
