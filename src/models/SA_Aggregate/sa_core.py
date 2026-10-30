"""
sa_core.py
==========
Matsumura & Murata (IEEE SMC 2025) の焼きなまし法を 1 群ぶん解く本体（SA_setup.md §3〜§5）

    状態     X (N, 96)        人 i のスロット s の活動（20 分類, 0 始まり）
    目的関数  f1 ルール違反 > f2 切り替え回数 > f3 日次行動者数のずれ（辞書式）
    近傍     1 スロット t の中で n 人の活動を並べ替える（時刻別行動者率を構成上保つ）

1 回の反復で変わるのは、選んだ n 人のスロット t だけである。そのため f1〜f3 は差分で更新する。

    人 i ごとに持つ量   viol[i]（違反数）・groups[i, r]（ルール r のグループ数）・sw[i]（切り替え回数）・
                        cnt[i, k]（活動 k のスロット数）
    群で持つ量          NA[k] = #{i : cnt[i, k] > 0}

```mermaid
flowchart TD
    T["スロット t を一様に選ぶ"] --> N["n ~ 切断正規 [2, N]<br/>平均 (temp − t_conv)/(t_init − t_conv)·N, 標準偏差 N/2"]
    N --> SEL["n 人を重み viol[i] + 1 で選ぶ（重複なし）"]
    SEL --> POOL["pool_cnt[k] = 選んだ人のスロット t の活動の数"]
    POOL --> ASSIGN["1 人ずつ、前後のスロットと同じ活動を 4.0 / 2.0 / 1.0 の重みで引く"]
    ASSIGN --> DELTA["Δf1・Δf2・Δf3 を差分で計算"]
    DELTA --> ACC{"辞書式に受理?"}
    ACC -- yes --> APPLY["X・viol・sw・cnt・NA を更新<br/>最良なら best_X に写す"]
    ACC -- no --> T
    APPLY --> T
```

★numba の njit 関数だけを置く。データの読み込みと保存は sa_run.py が担う。
"""
import numpy as np
import numpy.typing as npt
from numba import njit

NUM_ACT20 = 20
NUM_SLOTS = 96

# ---- ルール R1〜R4（SA_setup.md §3.1）。活動の番号は 20 分類の コード − 1 ----
SLEEP, MEAL, COMMUTE, WORK, SCHOOL = 0, 2, 3, 4, 5
SHOPPING, TRAVEL, MEDIA, REST, SOCIAL, MEDICAL = 9, 10, 11, 12, 17, 18

RULE_MAIN = np.array([SLEEP, COMMUTE, WORK, SCHOOL], dtype=np.int64)
RULE_CAP = np.array([1, 2, 1, 1], dtype=np.int64)
_INSERTS: list[list[int]] = [
    [],                                         # R1 睡眠
    [MEAL, SHOPPING, TRAVEL, SOCIAL, MEDICAL],  # R2 通勤・通学
    [MEAL, SHOPPING, TRAVEL, MEDIA, REST],      # R3 仕事
    [MEAL, SHOPPING, TRAVEL, MEDIA, REST],      # R4 学業（R3 と同じ）
]
# RULE_MEMBER[r, k]: 活動 k がルール r のグループに入れるか（主の活動か、途中に入ってよい活動）
RULE_MEMBER = np.zeros((len(RULE_MAIN), NUM_ACT20), dtype=np.bool_)
for _r, _main in enumerate(RULE_MAIN):
    RULE_MEMBER[_r, _main] = True
    RULE_MEMBER[_r, _INSERTS[_r]] = True
# どれかのルールに関わる活動。a・b の両方が関わらなければ、その人の違反数は変わらない
RULE_RELEVANT: npt.NDArray[np.bool_] = np.asarray(RULE_MEMBER.any(axis=0), dtype=np.bool_)

# f3 の重み w_k：1 次・2 次活動（01〜10）は 10、3 次活動（11〜20）は 1
F3_WEIGHT = np.array([10] * 10 + [1] * 10, dtype=np.int64)

# 履歴の列（温度ごとに 1 行）
HIST_COLS = ["temp", "f1", "f2", "f3", "best_f1", "best_f2", "best_f3", "accept_rate", "mean_n"]


@njit(cache=True)
def count_groups(row: npt.NDArray[np.int8], member: npt.NDArray[np.bool_], main: int) -> int:
    """1 人の 1 日で、ルールの活動のグループ数を円環として数える

    グループは member に入る活動だけが続く最長の区間のうち、main を 1 スロット以上含むもの。
    スロット 95 と 0 をつなぐので、04:00 をまたぐ夜の睡眠は 1 グループになる。

    Args:
        row: (96,) 1 人の活動
        member: (20,) グループに入れる活動
        main: ルールの主の活動

    Returns:
        グループ数
    """
    n_slots = row.shape[0]
    start = -1
    for s in range(n_slots):
        if not member[row[s]]:
            start = s
            break
    if start == -1:                     # 1 日中 member の活動だけ
        for s in range(n_slots):
            if row[s] == main:
                return 1
        return 0
    groups = 0
    in_run = False
    has_main = False
    # start は member でないので、start の次から 1 周して start で必ず区間が閉じる
    for j in range(1, n_slots + 1):
        a = row[(start + j) % n_slots]
        if member[a]:
            if not in_run:
                in_run = True
                has_main = False
            if a == main:
                has_main = True
        elif in_run:
            if has_main:
                groups += 1
            in_run = False
    return groups


@njit(cache=True)
def group_delta(row: npt.NDArray[np.int8], t: int, b: int, member: npt.NDArray[np.bool_], main: int) -> int:
    """row[t] を b に変えたときのグループ数の変化（row は書き換えない）

    スロット t の左に続く member の区間 L と、右に続く区間 R だけを見る。
    t が member なら L・t・R は 1 つの区間、そうでなければ L と R は別の区間である。
    count_groups(変更後) − count_groups(変更前) と一致する（test_sa.py の a2 で確かめる）。

    Args:
        row: (96,) 1 人の活動
        t: 変えるスロット
        b: 新しい活動
        member: (20,) グループに入れる活動
        main: ルールの主の活動

    Returns:
        グループ数の増減
    """
    n_slots = row.shape[0]
    a = row[t]
    has_l = False
    len_l = 0
    s = (t - 1) % n_slots
    while len_l < n_slots - 1 and member[row[s]]:
        if row[s] == main:
            has_l = True
        len_l += 1
        s = (s - 1) % n_slots
    if len_l == n_slots - 1:
        # t 以外がすべて member：区間は t を含めても含めなくても 1 つ
        before = 1 if (has_l or a == main) else 0
        after = 1 if (has_l or b == main) else 0
        return after - before
    has_r = False
    s = (t + 1) % n_slots
    # 左の区間の外側に member でないスロットがあるので、右へ歩いても必ずそこで止まる
    while member[row[s]]:
        if row[s] == main:
            has_r = True
        s = (s + 1) % n_slots
    if member[a]:
        before = 1 if (has_l or has_r or a == main) else 0
    else:
        before = (1 if has_l else 0) + (1 if has_r else 0)
    if member[b]:
        after = 1 if (has_l or has_r or b == main) else 0
    else:
        after = (1 if has_l else 0) + (1 if has_r else 0)
    return after - before


@njit(cache=True)
def person_violation(row: npt.NDArray[np.int8], rule_member: npt.NDArray[np.bool_],
                     rule_main: npt.NDArray[np.int64], rule_cap: npt.NDArray[np.int64]) -> int:
    """1 人の違反数 Σ_r max(0, グループ数 − 上限)"""
    v = 0
    for r in range(rule_main.shape[0]):
        over = count_groups(row, rule_member[r], rule_main[r]) - rule_cap[r]
        if over > 0:
            v += over
    return v


@njit(cache=True)
def person_switches(row: npt.NDArray[np.int8]) -> int:
    """1 人の切り替え回数。スロット 0〜95 の中だけを数え、95 → 0 は数えない"""
    c = 0
    for s in range(row.shape[0] - 1):
        if row[s] != row[s + 1]:
            c += 1
    return c


@njit(cache=True)
def objectives(X: npt.NDArray[np.int8], sna: npt.NDArray[np.int64], f3_weight: npt.NDArray[np.int64],
               rule_member: npt.NDArray[np.bool_], rule_main: npt.NDArray[np.int64],
               rule_cap: npt.NDArray[np.int64], use_rules: bool) -> tuple[int, int, int]:
    """f1・f2・f3 を差分を使わずに最初から計算する（テストと初期値に使う）"""
    n_people = X.shape[0]
    f1 = 0
    f2 = 0
    na = np.zeros(NUM_ACT20, dtype=np.int64)
    seen = np.zeros(NUM_ACT20, dtype=np.bool_)
    for i in range(n_people):
        if use_rules:
            f1 += person_violation(X[i], rule_member, rule_main, rule_cap)
        f2 += person_switches(X[i])
        seen[:] = False
        for s in range(X.shape[1]):
            seen[X[i, s]] = True
        for k in range(NUM_ACT20):
            if seen[k]:
                na[k] += 1
    f3 = 0
    for k in range(NUM_ACT20):
        f3 += f3_weight[k] * abs(sna[k] - na[k])
    return f1, f2, f3


@njit(cache=True)
def n_temperatures(t_init: float, t_conv: float, cooling_rate: float) -> int:
    """温度を下げる回数。anneal の外側のループと同じ浮動小数の計算で数える"""
    temp = t_init
    n = 0
    while temp > t_conv:
        n += 1
        temp *= cooling_rate
    return n


@njit(cache=True)
def _accept(delta: int, temp: float) -> bool:
    """メトロポリス受理。改善は必ず、悪化は確率 exp(−Δ/T) で受理する"""
    if delta < 0:
        return True
    return np.random.random() < np.exp(-delta / temp)


@njit(cache=True)
def _lexi_less(a1: int, a2: int, a3: int, b1: int, b2: int, b3: int) -> bool:
    """(a1, a2, a3) が (b1, b2, b3) より辞書式に小さいか"""
    if a1 != b1:
        return a1 < b1
    if a2 != b2:
        return a2 < b2
    return a3 < b3


@njit(cache=True)
def anneal(X: npt.NDArray[np.int8], sna: npt.NDArray[np.int64], f3_weight: npt.NDArray[np.int64],
           rule_member: npt.NDArray[np.bool_], rule_main: npt.NDArray[np.int64],
           rule_cap: npt.NDArray[np.int64], rule_relevant: npt.NDArray[np.bool_], use_rules: bool,
           cooling_interval: int, t_init: float, t_conv: float, cooling_rate: float,
           seed: int) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """1 群ぶんの SA を回し、最良解と温度ごとの履歴を返す

    Args:
        X: (N, 96) 初期解。★その場で書き換える（呼び出し側で写しを渡す）
        sna: (20,) 日次行動者数の目標 round(r_k × N)
        f3_weight: (20,) f3 の重み
        rule_member, rule_main, rule_cap: ルール R1〜R4
        rule_relevant: (20,) どれかのルールに関わる活動
        use_rules: False なら f1 を使わない版（norules）。人を選ぶ重みはすべて 1
        cooling_interval: 冷却の間隔 K（同じ温度で行う反復の回数）
        t_init, t_conv, cooling_rate: 初期温度・収束温度・冷却率
        seed: numba の乱数の種

    Returns:
        best_X (N, 96) と、履歴 (温度を下げる回数, len(HIST_COLS))
    """
    np.random.seed(seed)
    n_people, n_slots = X.shape

    # ---- 人ごと・群ごとの量を初期化する ----
    n_rules = rule_main.shape[0]
    viol = np.zeros(n_people, dtype=np.int64)
    groups = np.zeros((n_people, n_rules), dtype=np.int64)   # 人 i のルール r のグループ数
    sw = np.zeros(n_people, dtype=np.int64)
    cnt = np.zeros((n_people, NUM_ACT20), dtype=np.int64)
    na = np.zeros(NUM_ACT20, dtype=np.int64)
    for i in range(n_people):
        if use_rules:
            for r in range(n_rules):
                groups[i, r] = count_groups(X[i], rule_member[r], rule_main[r])
                if groups[i, r] > rule_cap[r]:
                    viol[i] += groups[i, r] - rule_cap[r]
        sw[i] = person_switches(X[i])
        for s in range(n_slots):
            cnt[i, X[i, s]] += 1
        for k in range(NUM_ACT20):
            if cnt[i, k] > 0:
                na[k] += 1
    f1 = 0
    f2 = 0
    for i in range(n_people):
        f1 += viol[i]
        f2 += sw[i]
    f3 = 0
    for k in range(NUM_ACT20):
        f3 += f3_weight[k] * abs(sna[k] - na[k])
    best_X = X.copy()
    best1, best2, best3 = f1, f2, f3

    # ---- 1 回の反復で使う作業用の配列 ----
    keys = np.empty(n_people)
    sel = np.empty(n_people, dtype=np.int64)
    new_act = np.empty(n_people, dtype=np.int64)
    new_viol = np.empty(n_people, dtype=np.int64)
    new_groups = np.empty((n_people, n_rules), dtype=np.int64)
    new_sw = np.empty(n_people, dtype=np.int64)
    pool_cnt = np.zeros(NUM_ACT20, dtype=np.int64)
    act_w = np.zeros(NUM_ACT20)
    d_na = np.zeros(NUM_ACT20, dtype=np.int64)

    hist = np.zeros((n_temperatures(t_init, t_conv, cooling_rate), 9))
    temp = t_init
    h = 0
    while temp > t_conv:
        n_accept = 0
        n_sum = 0
        for _ in range(cooling_interval):
            t = np.random.randint(0, n_slots)

            # 手順 2：入れ替える人数 n（[2, N] で切った正規分布を棄却法で引く）
            mu = (temp - t_conv) / (t_init - t_conv) * n_people
            while True:
                v = mu + 0.5 * n_people * np.random.standard_normal()
                if 2.0 <= v <= n_people:
                    break
            n = min(max(int(np.round(v)), 2), n_people)
            n_sum += n

            # 手順 3：重み viol + 1 で n 人を重複なしに選ぶ。
            # Efraimidis–Spirakis 法：鍵 log(u)/w の大きい順に n 人を取ると、重み付きの非復元抽出になる
            if use_rules:
                for i in range(n_people):
                    keys[i] = np.log(np.random.random()) / (viol[i] + 1.0)
                # 全員を並べ替えず、上位 n 人の境目の値だけを求める（選ばれる集合は並べ替えと同じ）
                thr = np.partition(keys, n_people - n)[n_people - n]
                m = 0
                for i in range(n_people):
                    if keys[i] > thr:
                        sel[m] = i
                        m += 1
                for i in range(n_people):          # 境目と同じ値の人で残りを埋める
                    if m >= n:
                        break
                    if keys[i] == thr:
                        sel[m] = i
                        m += 1
            else:
                # 重みがすべて 1 なら、部分的な Fisher–Yates で一様に選ぶ
                for i in range(n_people):
                    sel[i] = i
                for j in range(n):
                    r = np.random.randint(j, n_people)
                    tmp = sel[j]
                    sel[j] = sel[r]
                    sel[r] = tmp
                # sel[:n] は既に一様な順なので、手順 5 の並べ替えも済んでいる
            if use_rules:
                # 手順 5 で 1 人ずつ取る順をランダムにする
                for j in range(n - 1, 0, -1):
                    r = np.random.randint(0, j + 1)
                    tmp = sel[j]
                    sel[j] = sel[r]
                    sel[r] = tmp

            # 手順 4：選んだ人のスロット t の活動 ID を集める（活動ごとの数で持つ）
            pool_cnt[:] = 0
            for j in range(n):
                pool_cnt[X[sel[j], t]] += 1

            # 手順 5：前後のスロットと同じ活動を引きやすくする（4.0 / 2.0 / 1.0）。
            # 配列 A から要素を 1 つ引くことは、活動 k を pool_cnt[k] × 重み に比例して引くことと同じ
            for j in range(n):
                i = sel[j]
                prev = X[i, (t - 1) % n_slots]
                nxt = X[i, (t + 1) % n_slots]
                tot = 0.0
                for k in range(NUM_ACT20):
                    if pool_cnt[k] == 0:
                        act_w[k] = 0.0
                    else:
                        same = (1 if prev == k else 0) + (1 if nxt == k else 0)
                        act_w[k] = pool_cnt[k] * (4.0 if same == 2 else (2.0 if same == 1 else 1.0))
                    tot += act_w[k]
                u = np.random.random() * tot
                k_new = NUM_ACT20 - 1
                acc_w = 0.0
                for k in range(NUM_ACT20):
                    acc_w += act_w[k]
                    if u < acc_w and act_w[k] > 0.0:
                        k_new = k
                        break
                while pool_cnt[k_new] == 0:       # 丸めで末尾に落ちた場合の保険
                    k_new -= 1
                new_act[j] = k_new
                pool_cnt[k_new] -= 1

            # ---- Δf1・Δf2・Δf3 を差分で計算する ----
            d1 = 0
            d2 = 0
            d_na[:] = 0
            for j in range(n):
                i = sel[j]
                a = X[i, t]
                b = new_act[j]
                if a == b:
                    continue
                ds = 0
                if t > 0:
                    ds += (1 if X[i, t - 1] != b else 0) - (1 if X[i, t - 1] != a else 0)
                if t < n_slots - 1:
                    ds += (1 if X[i, t + 1] != b else 0) - (1 if X[i, t + 1] != a else 0)
                new_sw[j] = sw[i] + ds
                d2 += ds
                # 同じ人が 1 回の反復で 2 度変わることはないので、人ごとの寄与を足せばよい
                if cnt[i, a] == 1:
                    d_na[a] -= 1
                if cnt[i, b] == 0:
                    d_na[b] += 1
                new_viol[j] = viol[i]
                if use_rules and (rule_relevant[a] or rule_relevant[b]):
                    # ルール r のグループ数は「r のグループに入れるか」と「r の主の活動か」の並びだけで決まる。
                    # スロット t でこの 2 つがどちらも変わらないルールは、数え直さなくてよい
                    for r in range(n_rules):
                        if (rule_member[r, a] == rule_member[r, b]
                                and (a == rule_main[r]) == (b == rule_main[r])):
                            new_groups[j, r] = groups[i, r]
                            continue
                        g_new = groups[i, r] + group_delta(X[i], t, b, rule_member[r], rule_main[r])
                        new_groups[j, r] = g_new
                        new_viol[j] += max(g_new - rule_cap[r], 0) - max(groups[i, r] - rule_cap[r], 0)
                    d1 += new_viol[j] - viol[i]
                elif use_rules:
                    for r in range(n_rules):
                        new_groups[j, r] = groups[i, r]
            f3_new = 0
            for k in range(NUM_ACT20):
                f3_new += f3_weight[k] * abs(sna[k] - (na[k] + d_na[k]))
            d3 = f3_new - f3

            # ---- 辞書式の受理（論文 Fig. 3） ----
            if d1 != 0:
                ok = _accept(d1, temp)
            elif d2 != 0:
                ok = _accept(d2, temp)
            elif d3 != 0:
                ok = _accept(d3, temp)
            else:
                ok = True
            if not ok:
                continue

            n_accept += 1
            for j in range(n):
                i = sel[j]
                a = X[i, t]
                b = new_act[j]
                if a == b:
                    continue
                X[i, t] = b
                sw[i] = new_sw[j]
                viol[i] = new_viol[j]
                if use_rules:
                    for r in range(n_rules):
                        groups[i, r] = new_groups[j, r]
                cnt[i, a] -= 1
                cnt[i, b] += 1
            for k in range(NUM_ACT20):
                na[k] += d_na[k]
            f1 += d1
            f2 += d2
            f3 = f3_new
            if _lexi_less(f1, f2, f3, best1, best2, best3):
                best_X[:, :] = X
                best1, best2, best3 = f1, f2, f3

        hist[h, 0] = temp
        hist[h, 1] = f1
        hist[h, 2] = f2
        hist[h, 3] = f3
        hist[h, 4] = best1
        hist[h, 5] = best2
        hist[h, 6] = best3
        hist[h, 7] = n_accept / cooling_interval
        hist[h, 8] = n_sum / cooling_interval
        h += 1
        temp *= cooling_rate
    return best_X, hist
