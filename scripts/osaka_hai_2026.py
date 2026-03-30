"""2026 大阪杯 期待値分析

オッズから市場確率を算出し、Harvilleモデルで全馬券種の期待値を計算する。
実データのモデルがないため、オッズベースの確率に独自の調整を加えて分析。
"""

import numpy as np
import pandas as pd

# =============================================================================
# 出走馬データ
# =============================================================================
entries = [
    {"no": 1, "name": "エコロヴァルツ",     "sex_age": "牡5", "jockey": "浜中",   "trainer": "牧浦(栗東)",  "odds": 21.6, "pop": 6},
    {"no": 2, "name": "エコロディノス",     "sex_age": "牡4", "jockey": "池添",   "trainer": "大久保(栗東)", "odds": 22.5, "pop": 8},
    {"no": 3, "name": "オニャンコポン",     "sex_age": "セ7", "jockey": "未定",   "trainer": "小島(美浦)",  "odds": 272.4, "pop": 15},
    {"no": 4, "name": "クロワデュノール",   "sex_age": "牡4", "jockey": "北村友", "trainer": "斉藤崇(栗東)", "odds": 4.4, "pop": 2},
    {"no": 5, "name": "サンストックトン",   "sex_age": "牡7", "jockey": "未定",   "trainer": "堀内(美浦)",  "odds": 518.6, "pop": 16},
    {"no": 6, "name": "ショウヘイ",         "sex_age": "牡4", "jockey": "川田",   "trainer": "友道(栗東)",  "odds": 5.7, "pop": 4},
    {"no": 7, "name": "セイウンハーデス",   "sex_age": "セ7", "jockey": "幸",     "trainer": "橋口(栗東)",  "odds": 33.4, "pop": 11},
    {"no": 8, "name": "タガノデュード",     "sex_age": "牡5", "jockey": "古川吉", "trainer": "宮(栗東)",    "odds": 22.2, "pop": 7},
    {"no": 9, "name": "ダノンデサイル",     "sex_age": "牡5", "jockey": "坂井",   "trainer": "安田(栗東)",  "odds": 4.4, "pop": 1},
    {"no": 10, "name": "デビットバローズ",  "sex_age": "セ7", "jockey": "岩田望", "trainer": "上村(栗東)",  "odds": 25.7, "pop": 9},
    {"no": 11, "name": "ファウストラーゼン", "sex_age": "牡4", "jockey": "岩田康", "trainer": "須貝(栗東)",  "odds": 87.1, "pop": 14},
    {"no": 12, "name": "ボルドグフーシュ",  "sex_age": "牡7", "jockey": "松山",   "trainer": "宮本(栗東)",  "odds": 66.2, "pop": 13},
    {"no": 13, "name": "マテンロウレオ",    "sex_age": "牡7", "jockey": "横山典", "trainer": "昆(栗東)",    "odds": 54.3, "pop": 12},
    {"no": 14, "name": "メイショウタバル",  "sex_age": "牡5", "jockey": "武豊",   "trainer": "石橋(栗東)",  "odds": 4.5, "pop": 3},
    {"no": 15, "name": "ヨーホーレイク",    "sex_age": "牡8", "jockey": "西村淳", "trainer": "友道(栗東)",  "odds": 30.4, "pop": 10},
    {"no": 16, "name": "レーベンスティール", "sex_age": "牡6", "jockey": "ルメール", "trainer": "田中博(美浦)", "odds": 9.4, "pop": 5},
]

df = pd.DataFrame(entries)

# =============================================================================
# Step 1: 市場確率の算出 (オッズ → 確率)
# =============================================================================
# JRA控除率: 単勝 20%, 複勝 20%, 馬連 22.5%, ワイド 22.5%,
#             馬単 25%, 三連複 25%, 三連単 27.5%
TAKEOUT = {
    "単勝": 0.20, "複勝": 0.20, "馬連": 0.225, "ワイド": 0.225,
    "馬単": 0.25, "三連複": 0.25, "三連単": 0.275,
}

# Raw implied probability (includes overround)
df["raw_prob"] = 1.0 / df["odds"]
overround = df["raw_prob"].sum()  # Should be > 1
print(f"オーバーラウンド: {overround:.3f} ({(overround-1)*100:.1f}% 超過)")

# Normalized market probability (remove overround)
df["market_prob"] = df["raw_prob"] / overround

# =============================================================================
# Step 2: モデル確率の推定
# =============================================================================
# 実際のモデルがないため、市場確率をベースに「エッジ要因」で調整
# 以下は定性的な調整例（実モデルではここが機械学習に置き換わる）
#
# 調整要因:
# - 4歳馬は春G1で上昇中 → 若干プラス
# - 7歳以上は衰えリスク → 若干マイナス
# - トップジョッキー(ルメール,川田,坂井) → 若干プラス
# - 騎手未定 → マイナス

adjustments = {}
for _, row in df.iterrows():
    adj = 1.0  # multiplier

    # Age factor
    age_str = row["sex_age"]
    age = int(age_str[-1]) if age_str[-1].isdigit() else 5
    if age == 4:
        adj *= 1.08  # 4歳馬の春G1アドバンテージ
    elif age >= 7:
        adj *= 0.90  # 高齢馬ディスカウント
    elif age == 8:
        adj *= 0.80

    # Jockey factor
    top_jockeys = {"ルメール": 1.10, "川田": 1.08, "坂井": 1.06, "北村友": 1.03, "武豊": 1.04}
    jockey_adj = top_jockeys.get(row["jockey"], 1.0)
    if row["jockey"] == "未定":
        jockey_adj = 0.85
    adj *= jockey_adj

    adjustments[row["no"]] = adj

df["adj_factor"] = df["no"].map(adjustments)
df["model_prob_raw"] = df["market_prob"] * df["adj_factor"]
df["model_prob"] = df["model_prob_raw"] / df["model_prob_raw"].sum()  # Re-normalize

# =============================================================================
# Step 3: Harville モデルで着順確率を計算
# =============================================================================
win_probs = df["model_prob"].values
n = len(win_probs)


def harville_exacta(probs, i, j):
    """P(i着→j着)"""
    if probs[i] >= 1 or probs[i] <= 0:
        return 0
    return probs[i] * probs[j] / (1 - probs[i])


def harville_trifecta(probs, i, j, k):
    """P(i着→j着→k着)"""
    if probs[i] >= 1 or (1 - probs[i]) <= 0:
        return 0
    p_j = probs[j] / (1 - probs[i])
    remaining = 1 - probs[i] - probs[j]
    if remaining <= 0:
        return 0
    p_k = probs[k] / remaining
    return probs[i] * p_j * p_k


# =============================================================================
# Step 4: 各馬券種の期待値計算
# =============================================================================

print("\n" + "=" * 70)
print("  2026 大阪杯 (G1) 期待値分析")
print("  阪神 芝2000m | 16頭 | 定量58kg")
print("=" * 70)

# --- 単勝 期待値 ---
print("\n" + "-" * 70)
print("【単勝】期待値ランキング")
print("-" * 70)
print(f"{'No':>3} {'馬名':　<10} {'オッズ':>7} {'市場確率':>8} {'モデル確率':>8} {'期待値':>7} {'判定':>6}")
print("-" * 70)

win_results = []
for _, row in df.iterrows():
    ev = row["model_prob"] * row["odds"]
    verdict = "★買い" if ev > 1.1 else "○妙味" if ev > 1.0 else "  ー"
    win_results.append({**row.to_dict(), "ev": ev, "verdict": verdict})
    print(
        f"{row['no']:>3}  {row['name']:　<9} "
        f"{row['odds']:>7.1f} "
        f"{row['market_prob']:>7.1%} "
        f"{row['model_prob']:>8.1%} "
        f"{ev:>7.2f} "
        f"{verdict}"
    )

# --- 複勝 期待値 ---
print("\n" + "-" * 70)
print("【複勝】期待値ランキング (3着以内確率)")
print("-" * 70)

# Compute place probabilities via Monte Carlo (faster than exact enumeration)
np.random.seed(42)
N_SIM = 100_000
place_counts = np.zeros(n)

for _ in range(N_SIM):
    # Simulate race using probabilities
    remaining = list(range(n))
    probs_left = win_probs.copy()
    for pos in range(3):
        probs_norm = probs_left[remaining] / probs_left[remaining].sum()
        chosen_idx = np.random.choice(len(remaining), p=probs_norm)
        chosen = remaining[chosen_idx]
        place_counts[chosen] += 1
        remaining.pop(chosen_idx)

df["place_prob"] = place_counts / N_SIM

# Estimate place odds (roughly odds * 0.25~0.35 for favorites)
print(f"{'No':>3} {'馬名':　<10} {'3着内確率':>8} {'推定複勝':>7} {'期待値':>7} {'判定':>6}")
print("-" * 70)

for _, row in df.iterrows():
    pp = row["place_prob"]
    # Rough place odds estimate: (1 - takeout) / place_prob
    est_place_odds = (1 - TAKEOUT["複勝"]) / pp if pp > 0 else 999
    est_place_odds = max(1.1, round(est_place_odds, 1))
    ev = pp * est_place_odds
    verdict = "★買い" if ev > 1.05 else "○妙味" if ev > 1.0 else "  ー"
    print(
        f"{row['no']:>3}  {row['name']:　<9} "
        f"{pp:>7.1%} "
        f"{est_place_odds:>7.1f} "
        f"{ev:>7.2f} "
        f"{verdict}"
    )

# --- 馬連・馬単 期待値 TOP20 ---
print("\n" + "-" * 70)
print("【馬連】期待値 TOP15")
print("-" * 70)

quinella_evs = []
for i in range(n):
    for j in range(i + 1, n):
        p_ij = harville_exacta(win_probs, i, j) + harville_exacta(win_probs, j, i)
        # Estimate quinella odds
        est_odds = (1 - TAKEOUT["馬連"]) / p_ij if p_ij > 0 else 9999
        ev = p_ij * est_odds
        quinella_evs.append({
            "combo": f"{df.iloc[i]['no']:>2}-{df.iloc[j]['no']:<2}",
            "names": f"{df.iloc[i]['name']}-{df.iloc[j]['name']}",
            "prob": p_ij,
            "est_odds": round(est_odds, 1),
            "ev": ev,
        })

quinella_evs.sort(key=lambda x: x["prob"], reverse=True)
print(f"{'組合せ':>8} {'馬名':　<22} {'確率':>7} {'推定配当':>8} {'期待値':>6}")
print("-" * 70)
for q in quinella_evs[:15]:
    verdict = " ★" if q["ev"] > 1.05 else ""
    print(f"{q['combo']:>8}  {q['names']:　<20} {q['prob']:>6.2%} {q['est_odds']:>8.1f} {q['ev']:>6.2f}{verdict}")

# --- 三連複 期待値 TOP15 ---
print("\n" + "-" * 70)
print("【三連複】期待値 TOP15")
print("-" * 70)

# Top 8 horses only (to keep computation manageable)
top_indices = np.argsort(win_probs)[::-1][:8]
trio_evs = []

for idx_i, i in enumerate(top_indices):
    for idx_j, j in enumerate(top_indices):
        if idx_j <= idx_i:
            continue
        for idx_k, k in enumerate(top_indices):
            if idx_k <= idx_j:
                continue
            # Sum all 6 permutations
            p = 0
            for perm in [(i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)]:
                p += harville_trifecta(win_probs, *perm)

            est_odds = (1 - TAKEOUT["三連複"]) / p if p > 0 else 99999
            ev = p * est_odds

            nums = sorted([df.iloc[i]['no'], df.iloc[j]['no'], df.iloc[k]['no']])
            names = [df.iloc[x]['name'] for x in [i, j, k]]
            trio_evs.append({
                "combo": f"{nums[0]}-{nums[1]}-{nums[2]}",
                "names": "-".join(n[:4] for n in names),
                "prob": p,
                "est_odds": round(est_odds),
                "ev": ev,
            })

trio_evs.sort(key=lambda x: x["prob"], reverse=True)
print(f"{'組合せ':>10} {'確率':>7} {'推定配当':>9} {'期待値':>6}")
print("-" * 70)
for t in trio_evs[:15]:
    verdict = " ★" if t["ev"] > 1.05 else ""
    print(f"{t['combo']:>10}  {t['prob']:>6.2%}  {t['est_odds']:>8,}  {t['ev']:>6.2f}{verdict}")

# --- 三連単 期待値 TOP15 ---
print("\n" + "-" * 70)
print("【三連単】期待値 TOP15")
print("-" * 70)

trifecta_evs = []
top6 = np.argsort(win_probs)[::-1][:6]

for i in top6:
    for j in top6:
        if j == i:
            continue
        for k in top6:
            if k == i or k == j:
                continue
            p = harville_trifecta(win_probs, i, j, k)
            est_odds = (1 - TAKEOUT["三連単"]) / p if p > 0 else 999999
            ev = p * est_odds

            combo = f"{df.iloc[i]['no']}-{df.iloc[j]['no']}-{df.iloc[k]['no']}"
            trifecta_evs.append({
                "combo": combo,
                "names": f"{df.iloc[i]['name'][:4]}→{df.iloc[j]['name'][:4]}→{df.iloc[k]['name'][:4]}",
                "prob": p,
                "est_odds": round(est_odds),
                "ev": ev,
            })

trifecta_evs.sort(key=lambda x: x["prob"], reverse=True)
print(f"{'組合せ':>10} {'確率':>7} {'推定配当':>10} {'期待値':>6}")
print("-" * 70)
for t in trifecta_evs[:15]:
    verdict = " ★" if t["ev"] > 1.05 else ""
    print(f"{t['combo']:>10}  {t['prob']:>6.3%}  {t['est_odds']:>9,}  {t['ev']:>6.2f}{verdict}")

# =============================================================================
# Summary: 推奨ベット
# =============================================================================
print("\n" + "=" * 70)
print("  推奨まとめ (期待値 > 1.0 の馬券)")
print("=" * 70)

print("\n注意: この分析はオッズベースの確率にヒューリスティック調整を加えたものです。")
print("実際のモデル(過去データ学習済み)とは精度が大きく異なります。")
print("投資判断は自己責任でお願いします。")

# Value bet detection
print("\n【バリューベット候補 (モデル確率が市場確率を上回る馬)】")
print("-" * 70)
for _, row in df.iterrows():
    edge = row["model_prob"] - row["market_prob"]
    if edge > 0.005:  # 0.5%以上のエッジ
        ev = row["model_prob"] * row["odds"]
        print(
            f"  {row['name']:　<10} "
            f"市場{row['market_prob']:.1%} → モデル{row['model_prob']:.1%} "
            f"(+{edge:.1%}) "
            f"単勝EV={ev:.2f}"
        )

print("\n【ノーバリュー (市場に対してモデルが低評価)】")
print("-" * 70)
for _, row in df.iterrows():
    edge = row["model_prob"] - row["market_prob"]
    if edge < -0.005:
        print(
            f"  {row['name']:　<10} "
            f"市場{row['market_prob']:.1%} → モデル{row['model_prob']:.1%} "
            f"({edge:.1%})"
        )
