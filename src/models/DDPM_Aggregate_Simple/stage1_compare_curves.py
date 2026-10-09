"""
stage1_compare_curves.py
========================
Stage 1 の 3 つのモデルの時刻別行動者率を、ATUS 実データと 1 枚に重ねて描く。

    系列                                    重み（学習の種はすべて 42）
    ε-MSE損失のみ（時刻符号なし）           ddpm_simple_pretrain_common12_weekday_20260819.pt
    ε-MSE損失のみ（時刻符号あり）           ddpm_simple_pretrain_common12_weekday_clock.pt
    ε-MSE+行動者率損失（時刻符号あり, λ）   ddpm_simple_pretrain_common12_weekday_clock_{arm}.pt

★28 群は ATUS の調査ウェイト TUFINLWGT の群シェアでまとめる（stage1_rate_curves と同じ）。
  Stage 1 の目標は ATUS なので、生成も同じ構成でまとめ、差を「群の中の違い」だけにする。
★生成プールは mtime を重みと照合してから読む（stage2_teacher_fit.load_pool_csv）。
★描画と配色は stage1_rate_curves.render_page を使う。生成 3 本の色（青・橙・緑）は全組の色覚検証を通る。
★--with-stula で社会生活基本調査（日本, 教師A*）を灰色の破線で足す。これも ATUS の群シェアで
  まとめるので、他の線との差は群の中の日米差だけになる（atus_vs_stula_curves の π_d 版とは値が違う）。

データフロー:

```mermaid
flowchart LR
    NC["ddpm_simple_pretrain_samples_20260819.csv"] --> LP["tf.load_pool_csv(path, label, ckpt)<br/>mtime を重みと照合 → rates (28,12,96)"]
    CK["ddpm_simple_pretrain_samples_clock.csv"] --> LP
    RT["ddpm_simple_pretrain_samples_clock_{arm}.csv"] --> LP
    ATUS["tf.load_atus_rates()<br/>atus_rates (28,12,96), atus_share (28,)"] --> OV
    STULA["--with-stula のときだけ<br/>st.load_stula_targets()['group_rates_tbl'] = a_star (28,12,96)"] --> OV
    LP --> OV["overall: np.einsum('d,dcs->cs', atus_share, rates)<br/>(12,96)"]
    OV --> FIG["rc.render_page(title, curves, legend_ncol)"]
    FIG --> PNG["outputs/figures/stage1_rate/stage1_compare_{arm}[_stula].png"]
```

使い方:
    .venv/bin/python src/models/DDPM_Aggregate_Simple/stage1_compare_curves.py
    .venv/bin/python src/models/DDPM_Aggregate_Simple/stage1_compare_curves.py --arm rate10
    .venv/bin/python src/models/DDPM_Aggregate_Simple/stage1_compare_curves.py --arm rate1 --with-stula
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent

FloatArr = npt.NDArray[np.float64]


def _load(name: str, path: Path) -> Any:
    """sys.modules に一意名で載せる（stage2_teacher_fit.py と同じ規則）。"""
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# パス規則・描画・ATUS 読み込みは stage1_rate_curves（とその先の stage2_teacher_fit）と共有する
rc: Any = _load("compare_stage1_rate_curves", HERE / "stage1_rate_curves.py")
tf: Any = rc.tf

# 時刻符号なし・種 42。stage1_clock_replicates.BASELINE_NOCLOCK_CSV と同じプール
NOCLOCK_CSV = rc.GEN_DIR / "ddpm_simple_pretrain_samples_20260819.csv"
NOCLOCK_CKPT = rc.CKPT_DIR / "ddpm_simple_pretrain_common12_weekday_20260819.pt"

# 行動者率損失の条件（stage1_rate_curves.ARMS のキー）→ 凡例に出す λ・γ
ARM_PARAMS: dict[str, str] = {
    "rate1": "λ=1",
    "rate3": "λ=3",
    "rate10": "λ=10",
    "rate0.1g0.01": "λ=0.1, γ=0.01",
}
LABEL_NOCLOCK = "ε-MSE損失のみ（時刻符号なし）"
LABEL_CLOCK = "ε-MSE損失のみ（時刻符号あり）"
LABEL_STULA = "社会生活基本調査（日本, 教師A*）"   # atus_vs_stula_curves と同じ語


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="rate3", choices=list(ARM_PARAMS),
                    help="行動者率損失の条件, default=rate3（種 42 の掃引で 12:00 の昼食の山が最も高い）")
    ap.add_argument("--with-stula", action="store_true",
                    help="社会生活基本調査（日本, 教師A*）の線を灰色の破線で足す")
    ap.add_argument("--out", type=Path, default=None,
                    help="出力 PNG, default=outputs/figures/stage1_rate/stage1_compare_{arm}[_stula].png")
    args = ap.parse_args()
    out: Path = args.out or rc.OUT_DIR / f"stage1_compare_{args.arm}{'_stula' if args.with_stula else ''}.png"

    label_rate = f"ε-MSE+行動者率損失（時刻符号あり, {ARM_PARAMS[args.arm]}）"
    clock_csv, clock_ckpt = rc.pool_paths("")
    rate_csv, rate_ckpt = rc.pool_paths(args.arm)

    atus_rates, atus_share = tf.load_atus_rates()          # (28,12,96), (28,)
    noclock = tf.load_pool_csv(NOCLOCK_CSV, LABEL_NOCLOCK, NOCLOCK_CKPT)
    clock = tf.load_pool_csv(clock_csv, LABEL_CLOCK, clock_ckpt)
    rate = tf.load_pool_csv(rate_csv, label_rate, rate_ckpt)

    def overall(rates: FloatArr) -> FloatArr:
        """28 群を ATUS の群シェアでまとめる, (28,12,96) -> (12,96)"""
        return np.einsum("d,dcs->cs", atus_share, rates)

    # (役割, 凡例のラベル, 曲線)。役割は rc.ROLE_STYLE のキーで、色と線幅を決める。
    # 並びは描く順と凡例の順。データの線を先に描き、生成 3 本を上に重ねる
    curves: list[tuple[str, str, FloatArr]] = [("target", rc.LABEL_ATUS, overall(atus_rates))]
    if args.with_stula:
        a_star = np.asarray(tf.cv.st.load_stula_targets()["group_rates_tbl"], dtype=np.float64)
        curves.append(("stula", LABEL_STULA, overall(a_star)))
    curves += [
        ("noclock", LABEL_NOCLOCK, overall(noclock.rates)),
        ("base", LABEL_CLOCK, overall(clock.rates)),
        ("arm", label_rate, overall(rate.rates)),
    ]

    tf._setup_fonts()
    import matplotlib.pyplot as plt

    # ★5 本のときは 3 列にする。凡例は列から埋まるので、列が「データ / ε-MSE のみ / +行動者率損失」に揃う
    # ★題は名前だけにする。28 群を ATUS の調査ウェイトでまとめたことはモジュール docstring に書く
    fig = rc.render_page("Stage 1 の時刻別行動者率",
                         curves, legend_ncol=3 if args.with_stula else 4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for pr in (noclock, clock, rate):
        print(f"[stage1_compare] {pr.label:<40} <- {pr.ckpt}（群あたり {pr.n} 本）")
    print(f"[stage1_compare] 図を書いた: {out}")


if __name__ == "__main__":
    main()
