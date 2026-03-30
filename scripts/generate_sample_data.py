"""Generate realistic synthetic grade race data for pipeline testing.

Simulates 10 years (2016-2025) of G1/G2/G3 races with:
- Realistic horse/jockey/trainer pools
- Correlated performance (good horses win more often)
- Realistic odds derived from true ability
- Proper payout calculations for all bet types
"""

import random
from datetime import date, timedelta

import numpy as np

from keiba.db.connection import get_session, init_db
from keiba.db.schema import Horse, Payout, Race, RaceResult

random.seed(42)
np.random.seed(42)

# --- Constants ---
VENUES = ["東京", "中山", "阪神", "京都", "中京", "小倉", "札幌", "函館", "福島", "新潟"]
VENUE_CODES = {"東京": "05", "中山": "06", "阪神": "09", "京都": "08", "中京": "07",
               "小倉": "10", "札幌": "01", "函館": "02", "福島": "03", "新潟": "04"}
SURFACES = ["芝", "ダート"]
CONDITIONS = ["良", "稍重", "重", "不良"]
WEATHER = ["晴", "曇", "雨"]
GRADES = ["G1", "G2", "G3"]
GRADE_COUNTS = {"G1": 24, "G2": 35, "G3": 65}  # per year, roughly JRA
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000, 3200]
SIRE_NAMES = [
    "ディープインパクト", "キングカメハメハ", "ロードカナロア", "ハーツクライ",
    "エピファネイア", "ドゥラメンテ", "モーリス", "オルフェーヴル",
    "ゴールドシップ", "キタサンブラック", "サトノダイヤモンド", "コントレイル",
    "イクイノックス", "リバティアイランド", "ジャスティンミラノ",
]
DAM_SIRE_NAMES = [
    "サンデーサイレンス", "ストームキャット", "キングカメハメハ", "フレンチデピュティ",
    "クロフネ", "シンボリクリスエス", "ダイワメジャー", "スペシャルウィーク",
]


def generate_horse_pool(n=500):
    """Generate a pool of horses with stable ability ratings."""
    horses = []
    for i in range(n):
        horse_id = f"H{i:04d}"
        sex = random.choice(["牡", "牝", "セン"]) if random.random() > 0.85 else random.choice(["牡", "牝"])
        birth_year = random.randint(2012, 2023)
        sire_idx = random.randint(0, len(SIRE_NAMES) - 1)
        bms_idx = random.randint(0, len(DAM_SIRE_NAMES) - 1)
        # Ability rating: higher = better horse
        ability = np.random.normal(50, 15)
        # Surface preference: positive = turf, negative = dirt
        surface_pref = np.random.normal(0, 1)
        # Distance preference: center distance in meters
        dist_pref = random.choice([1400, 1600, 1800, 2000, 2200, 2400])

        horses.append({
            "horse_id": horse_id,
            "horse_name": f"テスト馬{i:03d}",
            "birth_date": date(birth_year, random.randint(1, 5), random.randint(1, 28)),
            "sex": sex,
            "sire_id": f"S{sire_idx:03d}",
            "sire_name": SIRE_NAMES[sire_idx],
            "dam_id": f"D{i:04d}",
            "dam_name": f"テスト母馬{i:03d}",
            "broodmare_sire_id": f"BS{bms_idx:03d}",
            "broodmare_sire_name": DAM_SIRE_NAMES[bms_idx],
            "owner": f"オーナー{i % 50}",
            "breeder": f"牧場{i % 30}",
            "ability": ability,
            "surface_pref": surface_pref,
            "dist_pref": dist_pref,
        })
    return horses


def generate_jockey_pool(n=80):
    """Generate a pool of jockeys with skill ratings."""
    jockeys = []
    for i in range(n):
        skill = np.random.normal(0, 1)
        jockeys.append({
            "jockey_id": f"J{i:03d}",
            "jockey_name": f"テスト騎手{i:02d}",
            "skill": skill,
        })
    return jockeys


def generate_trainer_pool(n=60):
    trainers = []
    for i in range(n):
        skill = np.random.normal(0, 0.5)
        trainers.append({
            "trainer_id": f"T{i:03d}",
            "trainer_name": f"テスト調教師{i:02d}",
            "skill": skill,
        })
    return trainers


def simulate_race(horses_in_race, jockeys_in_race, trainers_in_race,
                  surface, distance, condition):
    """Simulate a race and return finish order based on ability + noise."""
    scores = []
    for h, j, t in zip(horses_in_race, jockeys_in_race, trainers_in_race):
        # Base ability
        score = h["ability"]
        # Jockey skill
        score += j["skill"] * 5
        # Trainer skill
        score += t["skill"] * 3
        # Surface aptitude
        if surface == "芝":
            score += h["surface_pref"] * 3
        else:
            score -= h["surface_pref"] * 3
        # Distance aptitude (penalty for being far from preferred distance)
        dist_diff = abs(distance - h["dist_pref"]) / 200
        score -= dist_diff * 2
        # Track condition effect (heavy favors some horses)
        if condition in ("重", "不良"):
            score += np.random.normal(0, 5)
        # Age factor (peak at 4-5)
        age = 2025 - h["birth_date"].year  # approximate
        age_factor = -0.5 * (age - 4.5) ** 2
        score += age_factor
        # Random noise (race day form)
        score += np.random.normal(0, 8)
        scores.append(score)

    # Rank by score (highest = 1st)
    order = np.argsort(scores)[::-1]
    finish_positions = np.empty(len(scores), dtype=int)
    for rank, idx in enumerate(order):
        finish_positions[idx] = rank + 1

    return finish_positions, scores


def generate_odds(scores, finish_positions):
    """Generate realistic odds from true ability scores."""
    # Convert scores to probabilities via softmax
    shifted = np.array(scores) - np.max(scores)
    probs = np.exp(shifted / 10) / np.exp(shifted / 10).sum()
    # Add market noise
    probs = probs * np.random.uniform(0.8, 1.2, len(probs))
    probs = probs / probs.sum()
    # Convert to odds (with ~20% takeout)
    odds = []
    for p in probs:
        if p > 0.01:
            raw_odds = 0.80 / p  # 20% takeout
            # Round to realistic odds
            raw_odds = max(1.1, round(raw_odds, 1))
        else:
            raw_odds = 999.9
        odds.append(raw_odds)
    return odds, probs


def generate_payouts(finish_positions, horse_numbers, odds):
    """Generate payout data for all bet types."""
    payouts = []
    n = len(finish_positions)

    # Map finish position to horse number
    pos_to_num = {}
    for i, fp in enumerate(finish_positions):
        pos_to_num[fp] = horse_numbers[i]

    first = pos_to_num.get(1)
    second = pos_to_num.get(2)
    third = pos_to_num.get(3)

    if not all([first, second, third]):
        return payouts

    # 単勝
    winner_idx = list(horse_numbers).index(first)
    win_payout = int(odds[winner_idx] * 100)
    payouts.append({"bet_type": "単勝", "combination": str(first), "payout": win_payout, "popularity": 1})

    # 複勝
    for pos in [1, 2, 3]:
        num = pos_to_num[pos]
        idx = list(horse_numbers).index(num)
        place_payout = int(odds[idx] * 100 * random.uniform(0.2, 0.45))
        place_payout = max(110, place_payout)
        payouts.append({"bet_type": "複勝", "combination": str(num), "payout": place_payout, "popularity": pos})

    # 馬連
    combo = f"{min(first, second)}-{max(first, second)}"
    umaren_payout = int(win_payout * random.uniform(1.5, 5.0))
    payouts.append({"bet_type": "馬連", "combination": combo, "payout": umaren_payout, "popularity": 1})

    # ワイド (3 combinations)
    for a, b in [(first, second), (first, third), (second, third)]:
        combo = f"{min(a, b)}-{max(a, b)}"
        wide_payout = int(random.uniform(150, umaren_payout * 0.6))
        wide_payout = max(110, wide_payout)
        payouts.append({"bet_type": "ワイド", "combination": combo, "payout": wide_payout, "popularity": 1})

    # 馬単
    combo = f"{first}-{second}"
    umatan_payout = int(umaren_payout * random.uniform(1.5, 3.0))
    payouts.append({"bet_type": "馬単", "combination": combo, "payout": umatan_payout, "popularity": 1})

    # 三連複
    nums = sorted([first, second, third])
    combo = f"{nums[0]}-{nums[1]}-{nums[2]}"
    trio_payout = int(umaren_payout * random.uniform(2.0, 10.0))
    payouts.append({"bet_type": "三連複", "combination": combo, "payout": trio_payout, "popularity": 1})

    # 三連単
    combo = f"{first}-{second}-{third}"
    trifecta_payout = int(trio_payout * random.uniform(3.0, 8.0))
    payouts.append({"bet_type": "三連単", "combination": combo, "payout": trifecta_payout, "popularity": 1})

    return payouts


def generate_finish_time(distance, finish_pos, condition):
    """Generate realistic finish time in seconds."""
    # Base speed: ~60 seconds per 1000m for turf
    base_time = distance / 1000 * 60
    # Add variation for condition
    if condition == "稍重":
        base_time *= 1.01
    elif condition == "重":
        base_time *= 1.02
    elif condition == "不良":
        base_time *= 1.03
    # Slower for worse positions
    base_time += (finish_pos - 1) * 0.3
    # Random variation
    base_time += np.random.normal(0, 0.5)
    return round(base_time, 1)


def generate_last_3f(finish_pos):
    """Generate 上がり3F based on finish position."""
    base = 34.0
    # Better finishers tend to have faster last 3F (差し/追込)
    if finish_pos <= 3:
        base -= np.random.uniform(0.3, 1.5)
    elif finish_pos <= 6:
        base += np.random.uniform(-0.3, 0.5)
    else:
        base += np.random.uniform(0.3, 1.5)
    return round(base + np.random.normal(0, 0.3), 1)


def generate_passing_order(finish_pos, field_size):
    """Generate realistic passing order string."""
    # Running style based on finish position
    if random.random() < 0.15:  # 逃げ
        positions = [random.randint(1, 3) for _ in range(3)] + [finish_pos]
    elif random.random() < 0.4:  # 先行
        positions = [random.randint(2, max(3, field_size // 3)) for _ in range(3)] + [finish_pos]
    else:  # 差し/追込
        start = random.randint(field_size // 3, field_size)
        positions = [start, start - random.randint(0, 2),
                     max(1, start - random.randint(1, 4)), finish_pos]
    positions = [max(1, min(p, field_size)) for p in positions]
    return "-".join(str(p) for p in positions)


def main():
    print("Initializing database...")
    init_db()
    session = get_session()

    print("Generating horse/jockey/trainer pools...")
    horses = generate_horse_pool(500)
    jockeys = generate_jockey_pool(80)
    trainers = generate_trainer_pool(60)

    # Save horse profiles
    for h in horses:
        session.add(Horse(
            horse_id=h["horse_id"], horse_name=h["horse_name"],
            birth_date=h["birth_date"], sex=h["sex"],
            sire_id=h["sire_id"], sire_name=h["sire_name"],
            dam_id=h["dam_id"], dam_name=h["dam_name"],
            broodmare_sire_id=h["broodmare_sire_id"],
            broodmare_sire_name=h["broodmare_sire_name"],
            owner=h["owner"], breeder=h["breeder"],
        ))
    session.commit()
    print(f"  Saved {len(horses)} horses")

    race_count = 0
    result_count = 0

    for year in range(2016, 2026):
        race_r = 11  # Grade races are usually R11
        year_races = 0
        for grade, count in GRADE_COUNTS.items():
            for race_num in range(count):
                # Generate race metadata
                venue = random.choice(VENUES)
                venue_code = VENUE_CODES[venue]
                surface = random.choice(SURFACES)
                distance = random.choice(DISTANCES)
                condition = random.choices(CONDITIONS, weights=[60, 20, 15, 5])[0]
                weather = random.choices(WEATHER, weights=[50, 30, 20])[0]

                month = random.randint(1, 12)
                day = random.randint(1, 28)
                race_date = date(year, month, day)

                # Use global counter for unique race_id
                race_count += 1
                race_id = f"{year}{venue_code}{race_count:06d}"

                field_size = random.randint(8, 18)

                # Select horses for this race (age-appropriate)
                eligible = [h for h in horses
                            if h["birth_date"].year <= year - 2
                            and h["birth_date"].year >= year - 9]
                if len(eligible) < field_size:
                    eligible = horses
                race_horses = random.sample(eligible, min(field_size, len(eligible)))
                race_jockeys = random.choices(jockeys, k=len(race_horses))
                race_trainers = random.choices(trainers, k=len(race_horses))

                # Simulate race
                finish_positions, scores = simulate_race(
                    race_horses, race_jockeys, race_trainers,
                    surface, distance, condition
                )
                horse_numbers = list(range(1, len(race_horses) + 1))
                odds_list, _ = generate_odds(scores, finish_positions)

                # Prize money
                prize_map = {"G1": 20000, "G2": 10000, "G3": 6000}
                prize = prize_map[grade]

                # Race name
                race_name = f"テスト{grade}レース{year_races + 1}"

                # Save race
                session.add(Race(
                    race_id=race_id, race_name=race_name, race_date=race_date,
                    venue=venue, course_type=surface, distance=distance,
                    track_condition=condition, weather=weather, grade=grade,
                    race_class=grade, head_count=len(race_horses), prize_1st=prize,
                ))

                # Save results
                for i, (h, j, t) in enumerate(zip(race_horses, race_jockeys, race_trainers)):
                    fp = int(finish_positions[i])
                    age = year - h["birth_date"].year
                    sex_age = f"{h['sex']}{age}"
                    weight = random.randint(430, 530)

                    session.add(RaceResult(
                        race_id=race_id,
                        horse_id=h["horse_id"],
                        finish_order=fp,
                        frame_number=((horse_numbers[i] - 1) // 2) + 1,
                        horse_number=horse_numbers[i],
                        horse_name=h["horse_name"],
                        sex_age=sex_age,
                        weight_carried=random.choice([54.0, 55.0, 56.0, 57.0, 58.0]),
                        jockey_id=j["jockey_id"],
                        jockey_name=j["jockey_name"],
                        trainer_id=t["trainer_id"],
                        trainer_name=t["trainer_name"],
                        finish_time=generate_finish_time(distance, fp, condition),
                        margin="",
                        passing_order=generate_passing_order(fp, len(race_horses)),
                        last_3f=generate_last_3f(fp),
                        horse_weight=weight,
                        weight_change=random.randint(-10, 10),
                        odds=odds_list[i],
                        popularity=int(np.argsort(odds_list)[list(range(len(odds_list))).index(i)] + 1)
                        if i < len(odds_list) else i + 1,
                    ))
                    result_count += 1

                # Save payouts
                payout_list = generate_payouts(finish_positions, horse_numbers, odds_list)
                for p in payout_list:
                    session.add(Payout(
                        race_id=race_id, bet_type=p["bet_type"],
                        combination=p["combination"], payout=p["payout"],
                        popularity=p["popularity"],
                    ))

                race_count += 1
                year_races += 1

        session.commit()
        print(f"  Year {year}: {year_races} races generated")

    session.close()
    print(f"\nTotal: {race_count} races, {result_count} results")
    print("Done!")


if __name__ == "__main__":
    main()
