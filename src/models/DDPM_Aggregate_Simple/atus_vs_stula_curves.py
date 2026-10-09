"""
atus_vs_stula_curves.py
=======================
平日の時刻別行動者率を、ATUS 実データ（米国）と社会生活基本調査（日本、教師A*）で重ねて描く。

    系列                              出所
    社会生活基本調査（日本, 教師A*）  stage2_targets.load_stula_targets()['group_rates_tbl']
    ATUS実データ（米国）              stage2_teacher_fit.load_atus_rates()（TUFINLWGT で群の中を加重）

★28 群は両方とも日本の群人口 π_d でまとめる（cv.weighted_slot_rates）。
  重みを揃えないと、曲線の差に「群構成の日米差」が混ざる（plot_atus_vs_astar の既定と同じ）。
  この全体曲線は stage2_curves.atus_real_curve と同じ量である。
★描画と配色は stage1_rate_curves.render_page を使う（教師A* は黒の太線、ATUS は #199e70）。

データフロー:

```mermaid
flowchart LR
    TGT["st.load_stula_targets()<br/>tgt['group_rates_tbl'] = a_star (28,12,96)<br/>tgt['pop'] → π_d (28,)"] --> W
    ATUS["tf.load_atus_rates()<br/>atus_rates (28,12,96)"] --> W
    W["cv.weighted_slot_rates(rates, tgt)<br/>π_d で 28 群をまとめる → (12,96)"] --> FIG["rc.render_page(title, curves)"]
    FIG --> PNG["outputs/figures/atus_vs_stula.png"]
```

使い方:
    .venv/bin/python src/models/DDPM_Aggregate_Simple/atus_vs_stula_curves.py
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


# 描画は stage1_rate_curves、教師と ATUS の読み込みは stage2_teacher_fit（とその先の stage2_curves）と共有する
rc: Any = _load("atusstula_stage1_rate_curves", HERE / "stage1_rate_curves.py")
tf: Any = rc.tf
cv: Any = tf.cv

LABEL_STULA = "社会生活基本調査（日本, 教師A*）"
LABEL_ATUS = "ATUS実データ（米国）"
OUT_PNG = rc.REPO_ROOT / "outputs" / "figures" / "atus_vs_stula.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_PNG)
    args = ap.parse_args()

    tgt = cv.st.load_stula_targets()
    a_star = np.asarray(tgt["group_rates_tbl"], dtype=np.float64)     # (28,12,96)
    atus_rates, _ = tf.load_atus_rates()                               # (28,12,96)

    # (役割, 凡例のラベル, 曲線)。役割は rc.ROLE_STYLE のキーで、色と線幅を決める
    curves: list[tuple[str, str, FloatArr]] = [
        ("target", LABEL_STULA, cv.weighted_slot_rates(a_star, tgt)),
        ("atus", LABEL_ATUS, cv.weighted_slot_rates(atus_rates, tgt)),
    ]

    tf._setup_fonts()
    import matplotlib.pyplot as plt

    # ★題は名前だけにする。平日であること・28 群を π_d でまとめたことはモジュール docstring に書く
    fig = rc.render_page("時刻別行動者率", curves, legend_ncol=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[atus_vs_stula] 図を書いた: {args.out}")


if __name__ == "__main__":
    main()
