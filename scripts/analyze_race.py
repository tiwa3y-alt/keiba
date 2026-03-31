"""出馬表コピペ → 即期待値分析ツール

netkeiba.comの出馬表ページからテキストをコピペするだけで
全馬券種の期待値を自動計算します。

使い方:
    python scripts/analyze_race.py

    実行すると入力を求められるので、netkeibaの出馬表ページの
    テキストをコピペして、空行でEnterを押してください。
"""

import re
import sys

import numpy as np


def parse_entry_text(text: str) -> dict:
    """netkeibaの出馬表コピペテキストから出走馬データを抽出。

    以下のような形式を想定:
    - 馬名, 性齢, 斤量, 騎手名, 厩舎, オッズ, 人気
    - 各馬の情報が行に含まれている
    """
    lines = text.strip().split("\n")
    entries = []
    race_info = {"name": "", "distance": 0, "surface": "", "venue": "", "grade": ""}

    # レース情報の抽出
    for line in lines[:10]:
        # レース名
        if re.search(r"(G[1-3Ⅰ-Ⅲ]|GI|GII|GIII)", line):
            race_info["name"] = line.strip()
            for g, gn in [("G1", "G1"), ("GⅠ", "G1"), ("GI", "G1"),
                           ("G2", "G2"), ("GⅡ", "G2"), ("GII", "G2"),
                           ("G3", "G3"), ("GⅢ", "G3"), ("GIII", "G3")]:
                if g in line:
                    race_info["grade"] = gn
                    break
        # 距離・馬場
        m = re.search(r"(芝|ダート|ダ)\s*(\d{3,4})\s*m", line)
        if m:
            race_info["surface"] = "芝" if m.group(1) == "芝" else "ダート"
            race_info["distance"] = int(m.group(2))
        # 競馬場
        for venue in ["東京", "中山", "阪神", "京都", "中京", "小倉", "札幌", "函館", "福島", "新潟"]:
            if venue in line:
                race_info["venue"] = venue
                break

    # 出走馬の抽出
    # オッズ（数字.数字）と人気（数字）のパターンで馬を検出
    horse_pattern = re.compile(
        r"([ァ-ヶー]+(?:[ァ-ヶー\s]*[ァ-ヶー]+)*)"  # 馬名（カタカナ）
    )

    # 一行に馬名、性齢、斤量、騎手、オッズ、人気が含まれるパターン
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # オッズを探す（xx.x形式の数字）
        odds_matches = re.findall(r"(\d+\.\d+)", line)
        if not odds_matches:
            continue

        # 馬名を探す（カタカナ3文字以上の連続）
        name_match = re.search(r"([ァ-ヶー]{3,}(?:[ァ-ヶー]*)*)", line)
        if not name_match:
            continue

        horse_name = name_match.group(1)

        # 既知のUIテキストを除外
        skip_words = ["プレミアム", "サービス", "ログイン", "ログアウト", "トップ",
                       "ニュース", "データベース", "コラム", "ショップ", "イベント",
                       "ブックマーク", "キャンペーン", "マスター", "コース"]
        if any(w in horse_name for w in skip_words):
            continue

        # 性齢を探す
        sex_age = ""
        sa_match = re.search(r"([牡牝セ][2-9])", line)
        if sa_match:
            sex_age = sa_match.group(1)

        # 斤量を探す
        weight_match = re.search(r"(\d{2}\.\d)", line)
        weight_carried = float(weight_match.group(1)) if weight_match else 58.0

        # 騎手名を探す（漢字2-4文字）
        jockey = ""
        jockey_match = re.findall(r"([一-龥]{2,4})", line)
        if jockey_match:
            # 厩舎名等を除外
            for j in jockey_match:
                if j not in [horse_name, "栗東", "美浦"] and len(j) <= 4:
                    jockey = j
                    break

        # オッズ: 最も妥当な値を選ぶ
        odds = float(odds_matches[0])
        # 斤量っぽい値(54-58)を除外
        for o in odds_matches:
            val = float(o)
            if val > 1.0 and not (53.0 <= val <= 59.0):
                odds = val
                break

        # 人気を探す
        pop_match = re.findall(r"(\d{1,2})\s*$", line)
        pop = int(pop_match[-1]) if pop_match else 0

        if odds > 1.0:
            entries.append({
                "name": horse_name,
                "sex_age": sex_age,
                "weight_carried": weight_carried,
                "jockey": jockey,
                "odds": odds,
                "popularity": pop,
            })

    # 人気順でソート（未設定なら推定）
    if entries:
        entries.sort(key=lambda x: x["odds"])
        for i, e in enumerate(entries):
            if e["popularity"] == 0:
                e["popularity"] = i + 1

    return {"race_info": race_info, "entries": entries}


def compute_expected_values(entries: list[dict], race_info: dict) -> None:
    """全馬券種の期待値を計算して表示。"""

    n = len(entries)
    if n < 2:
        print("出走馬が2頭未満のため分析できません。")
        return

    # JRA控除率
    TAKEOUT = {
        "単勝": 0.20, "複勝": 0.20, "馬連": 0.225, "ワイド": 0.225,
        "馬単": 0.25, "三連複": 0.25, "三連単": 0.275,
    }

    # 市場確率（オッズから逆算、正規化）
    odds = np.array([e["odds"] for e in entries])
    raw_prob = 1.0 / odds
    overround = raw_prob.sum()
    market_prob = raw_prob / overround

    # モデル確率（ヒューリスティック調整）
    adj = np.ones(n)
    for i, e in enumerate(entries):
        # 年齢調整
        age = int(e["sex_age"][-1]) if e["sex_age"] and e["sex_age"][-1].isdigit() else 5
        if age == 4:
            adj[i] *= 1.05
        elif age >= 7:
            adj[i] *= 0.92

        # トップジョッキー調整
        top_jockeys = {"ルメール": 1.08, "川田": 1.06, "坂井": 1.05,
                        "武豊": 1.04, "戸崎": 1.04, "横山武": 1.04}
        for name, bonus in top_jockeys.items():
            if name in e.get("jockey", ""):
                adj[i] *= bonus
                break

    model_prob_raw = market_prob * adj
    model_prob = model_prob_raw / model_prob_raw.sum()

    # ============================================================
    # 表示
    # ============================================================
    print("\n" + "=" * 70)
    title = race_info.get("name", "レース分析")
    surface = race_info.get("surface", "")
    distance = race_info.get("distance", "")
    venue = race_info.get("venue", "")
    grade = race_info.get("grade", "")
    print(f"  {title}")
    if venue or surface or distance:
        print(f"  {venue} {surface}{distance}m | {n}頭 | {grade}")
    print("=" * 70)

    # --- 単勝 ---
    print(f"\n{'No':>3} {'馬名':<12} {'オッズ':>7} {'市場確率':>7} {'モデル':>7} {'単勝EV':>7} {'判定':>6}")
    print("-" * 60)

    for i, e in enumerate(entries):
        ev = model_prob[i] * odds[i]
        if ev > 1.1:
            verdict = "★買い"
        elif ev > 1.0:
            verdict = "○妙味"
        else:
            verdict = ""
        print(
            f"{e['popularity']:>3}  {e['name']:<11} "
            f"{odds[i]:>7.1f} "
            f"{market_prob[i]:>6.1%} "
            f"{model_prob[i]:>6.1%} "
            f"{ev:>7.2f} "
            f"{verdict}"
        )

    # --- 複勝 (モンテカルロ) ---
    print(f"\n--- 複勝 (3着以内確率) ---")
    np.random.seed(42)
    N_SIM = 50_000
    place_counts = np.zeros(n)
    for _ in range(N_SIM):
        remaining = list(range(n))
        probs_left = model_prob.copy()
        for pos in range(min(3, n)):
            probs_norm = probs_left[remaining] / probs_left[remaining].sum()
            chosen_idx = np.random.choice(len(remaining), p=probs_norm)
            chosen = remaining[chosen_idx]
            place_counts[chosen] += 1
            remaining.pop(chosen_idx)
    place_prob = place_counts / N_SIM

    print(f"{'No':>3} {'馬名':<12} {'3着内確率':>8} {'推定複勝':>7} {'EV':>6}")
    print("-" * 45)
    for i, e in enumerate(entries):
        est_place_odds = max(1.1, round((1 - TAKEOUT["複勝"]) / place_prob[i], 1)) if place_prob[i] > 0 else 999
        ev = place_prob[i] * est_place_odds
        print(f"{e['popularity']:>3}  {e['name']:<11} {place_prob[i]:>7.1%} {est_place_odds:>7.1f} {ev:>6.2f}")

    # --- Harville モデル ---
    def harville_exacta(p, i, j):
        if p[i] >= 1 or p[i] <= 0:
            return 0
        return p[i] * p[j] / (1 - p[i])

    def harville_trifecta(p, i, j, k):
        if p[i] >= 1 or (1 - p[i]) <= 0:
            return 0
        pj = p[j] / (1 - p[i])
        rem = 1 - p[i] - p[j]
        if rem <= 0:
            return 0
        pk = p[k] / rem
        return p[i] * pj * pk

    # --- 馬連 TOP10 ---
    print(f"\n--- 馬連 TOP10 ---")
    quinella = []
    for i in range(n):
        for j in range(i + 1, n):
            p = harville_exacta(model_prob, i, j) + harville_exacta(model_prob, j, i)
            est_odds = max(1.0, (1 - TAKEOUT["馬連"]) / p) if p > 0 else 9999
            quinella.append({"combo": f"{entries[i]['name'][:5]}-{entries[j]['name'][:5]}",
                            "prob": p, "odds": round(est_odds, 1), "ev": p * est_odds})
    quinella.sort(key=lambda x: x["prob"], reverse=True)
    print(f"{'組合せ':<15} {'確率':>7} {'推定配当':>8} {'EV':>6}")
    print("-" * 40)
    for q in quinella[:10]:
        mark = " ★" if q["ev"] > 1.0 else ""
        print(f"{q['combo']:<14} {q['prob']:>6.2%} {q['odds']:>8.1f} {q['ev']:>6.2f}{mark}")

    # --- 三連複 TOP10 ---
    print(f"\n--- 三連複 TOP10 ---")
    top_idx = np.argsort(model_prob)[::-1][:8]
    trio = []
    for ii, i in enumerate(top_idx):
        for jj, j in enumerate(top_idx):
            if jj <= ii:
                continue
            for kk, k in enumerate(top_idx):
                if kk <= jj:
                    continue
                p = sum(harville_trifecta(model_prob, *perm)
                        for perm in [(i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)])
                est_odds = max(1.0, (1 - TAKEOUT["三連複"]) / p) if p > 0 else 99999
                names = f"{entries[i]['name'][:4]}-{entries[j]['name'][:4]}-{entries[k]['name'][:4]}"
                trio.append({"combo": names, "prob": p, "odds": round(est_odds), "ev": p * est_odds})
    trio.sort(key=lambda x: x["prob"], reverse=True)
    print(f"{'組合せ':<18} {'確率':>7} {'推定配当':>9} {'EV':>6}")
    print("-" * 45)
    for t in trio[:10]:
        mark = " ★" if t["ev"] > 1.0 else ""
        print(f"{t['combo']:<17} {t['prob']:>6.2%} {t['odds']:>8,} {t['ev']:>6.2f}{mark}")

    # --- バリューベット候補 ---
    print(f"\n{'=' * 70}")
    print("  バリューベット候補 (モデル確率 > 市場確率)")
    print("=" * 70)
    found = False
    for i, e in enumerate(entries):
        edge = model_prob[i] - market_prob[i]
        if edge > 0.003:
            ev = model_prob[i] * odds[i]
            print(f"  {e['name']:<11} 市場{market_prob[i]:.1%} → モデル{model_prob[i]:.1%} "
                  f"(+{edge:.1%}) 単勝EV={ev:.2f}")
            found = True
    if not found:
        print("  なし（市場確率とモデル確率がほぼ一致）")

    print(f"\n※ オッズベース + ヒューリスティック調整の簡易分析です")
    print(f"※ 実データ学習モデルとは精度が異なります")


def main():
    print("=" * 70)
    print("  出馬表コピペ分析ツール")
    print("=" * 70)
    print()
    print("netkeibaの出馬表ページからテキストをコピペしてください。")
    print("入力が終わったら空行でEnterを押してください。")
    print("(Ctrl+D でも終了できます)")
    print()

    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except EOFError:
        pass

    text = "\n".join(lines)
    if not text.strip():
        print("入力がありません。")
        return

    result = parse_entry_text(text)
    entries = result["entries"]
    race_info = result["race_info"]

    if not entries:
        print("\n出走馬を検出できませんでした。")
        print("以下の形式でもう一度試してください:")
        print()
        print("馬名    性齢  斤量  騎手    オッズ  人気")
        print("ダノンデサイル  牡5  58.0  坂井    4.4    1")
        print("クロワデュノール 牡4  58.0  北村友   4.4    2")
        return

    print(f"\n{len(entries)}頭の出走馬を検出しました:")
    for e in entries:
        print(f"  {e['popularity']:>2}番人気 {e['name']:<11} {e['sex_age']} {e['odds']:>6.1f}倍 {e['jockey']}")

    compute_expected_values(entries, race_info)


if __name__ == "__main__":
    main()
