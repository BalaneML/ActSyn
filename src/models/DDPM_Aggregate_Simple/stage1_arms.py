"""
stage1_arms.py
==============
Stage 1 アブレーションの arm の表（唯一の出所）

`model.py --arm NAME` と評価（`src/eval/diagnostics/stage1_clock_replicates.py`）の両方が
この表を読む。保存先の名前を shell と Python で二重に組み立てない
（過去に生成プールの取り違えを起こしている）。

    arm 名            構造 (ArchSpec の引数)                          損失
    noclock           時刻符号なし                                    ε のみ
    clock             倍音 K=4                                        ε のみ
    clock_rate3       倍音 K=4                                        ε + 3·L_rate (batch)
    clock_h12         倍音 K=12                                       ε のみ
    clock_h48         倍音 K=48（96 スロットの全関数）                  ε のみ
    clock_tf96        Transformer 型 96 次元                          ε のみ
    clock_rope        倍音 K=4 + attention に RoPE                     ε のみ
    clock_rope_attn96 倍音 K=4 + RoPE + 96 解像度の attention          ε のみ
    clock_condclock   倍音 K=4 + 条件×時刻のバイアス (R=4)             ε のみ

段 2・3 の arm は、前の段の判定で土台が決まってから下の ARMS に足す。

保存先の接尾辞は model.py の既存の規則と同じ:
    {arm の suffix}{_s{seed}（seed != 42 のとき）}
ただし noclock の種 42 だけは本編の 20260819 版を指す（legacy_seed42_tag）。
"""
from dataclasses import dataclass, field
from typing import Any

# 学習の種の既定値（model.SEED と同じ）。この種の実行には _s{seed} を付けない
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ArmSpec:
    """1 つの arm の学習設定と保存先の接尾辞

    Attributes:
        arch: model.ArchSpec の引数。model.py を import しないため dict で持つ
        rate_lam: L_rate の重み λ_rate。0 なら ε のみ
        rate_mode: L_rate の偏りを平均する単位（model.RATE_MODES）
        suffix: 保存先の接尾辞（先頭の "_" を含む）
        legacy_seed42_tag: 種 42 だけ別名で保存されている既存 arm の名前。
            None でなければ種 42 の接尾辞は "_{legacy_seed42_tag}" になり、学習は禁止する
            （本編の成果物を上書きしないため）
        note: 報告書に出す短い説明
    """
    arch: dict[str, Any] = field(default_factory=dict)
    rate_lam: float = 0.0
    rate_mode: str = "batch"
    suffix: str = ""
    legacy_seed42_tag: str | None = None
    note: str = ""


def _h(k: int) -> dict[str, Any]:
    """倍音 K の時刻符号の ArchSpec 引数"""
    return {"clock_kind": "harmonic", "clock_harmonics": k}


ARMS: dict[str, ArmSpec] = {
    # --- 既存（比較の基準） ---
    "noclock": ArmSpec(suffix="", legacy_seed42_tag="20260819", note="時刻符号なし"),
    "clock": ArmSpec(arch=_h(4), suffix="_clock", note="倍音 K=4"),
    "clock_rate3": ArmSpec(arch=_h(4), rate_lam=3.0, suffix="_clock_rate3",
                           note="倍音 K=4 + L_rate(batch) λ=3"),
    # --- 段 1: 時刻符号の種類と解像度（損失は ε のみ） ---
    "clock_h12": ArmSpec(arch=_h(12), suffix="_clock_h12", note="倍音 K=12"),
    "clock_h48": ArmSpec(arch=_h(48), suffix="_clock_h48", note="倍音 K=48（全基底）"),
    "clock_tf96": ArmSpec(arch={"clock_kind": "transformer"}, suffix="_clock_tf96",
                          note="Transformer 型 96 次元"),
    # --- 段 2: 計算ブロック（土台は段 1 の勝者 clock = 倍音 K=4、損失は ε のみ） ---
    # 段 1 の判定（2026-09-27, stage1_replicates_stage1_judge.csv）: K=12 / K=48 / Transformer 型は
    # 行動者率を大きく改善したが、3 つとも switch_emd が基準の 3 本全てより悪く失格。勝者は clock
    "clock_rope": ArmSpec(arch={**_h(4), "attn_rope": True}, suffix="_clock_rope",
                          note="倍音 K=4 + attention に RoPE"),
    "clock_rope_attn96": ArmSpec(arch={**_h(4), "attn_rope": True, "attn96": True},
                                 suffix="_clock_rope_attn96",
                                 note="倍音 K=4 + RoPE + 96 解像度の attention"),
    "clock_condclock": ArmSpec(arch={**_h(4), "cond_clock_rank": 4}, suffix="_clock_condclock",
                               note="倍音 K=4 + 条件×時刻のバイアス (R=4)"),
}


def arm_suffix(name: str, seed: int) -> str:
    """arm と種から保存先の接尾辞を返す

    Args:
        name: ARMS のキー
        seed: 学習の種

    Returns:
        接尾辞（先頭の "_" を含む。noclock の種 42 は "_20260819"）

    Raises:
        KeyError: 未知の arm 名
    """
    if name not in ARMS:
        raise KeyError(f"未知の arm: {name}（既知: {sorted(ARMS)}）")
    arm = ARMS[name]
    if seed == DEFAULT_SEED and arm.legacy_seed42_tag is not None:
        return f"_{arm.legacy_seed42_tag}"
    return arm.suffix + ("" if seed == DEFAULT_SEED else f"_s{seed}")
