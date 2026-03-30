"""Scraper for netkeiba.com - race results, horse profiles, odds, and payouts."""

import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from keiba.config import NETKEIBA_BASE_URL, NETKEIBA_DB_URL, REQUEST_INTERVAL_SEC
from keiba.db.connection import get_session, init_db
from keiba.db.schema import Horse, OddsHistory, Payout, Race, RaceResult

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
)


def _fetch(url: str) -> BeautifulSoup:
    """Fetch a URL with rate limiting and return parsed HTML."""
    time.sleep(REQUEST_INTERVAL_SEC)
    resp = _SESSION.get(url, timeout=30)
    resp.encoding = "EUC-JP"
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def _parse_time_to_seconds(time_str: str) -> float | None:
    """Convert "1:34.5" or "1.34.5" format to seconds."""
    if not time_str or time_str.strip() in ("", "-"):
        return None
    time_str = time_str.strip().replace(".", ":", 1)
    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None
    return None


def _safe_int(val: str) -> int | None:
    """Safely parse integer from string."""
    try:
        return int(re.sub(r"[^\d\-]", "", val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> float | None:
    """Safely parse float from string."""
    try:
        return float(re.sub(r"[^\d.\-]", "", val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Race list scraping
# ---------------------------------------------------------------------------


def scrape_grade_race_list(year: int) -> list[str]:
    """Get list of grade race IDs for a given year.

    Scrapes the netkeiba grade race calendar page.

    Returns:
        List of race_id strings like "202405050811".
    """
    url = f"{NETKEIBA_BASE_URL}/top/race_list.html?kaisai_date={year}"
    soup = _fetch(url)
    race_ids = []
    for a_tag in soup.select("a[href*='/race/']"):
        href = a_tag.get("href", "")
        match = re.search(r"/race/(\d{12})", href)
        if match:
            race_id = match.group(1)
            if race_id not in race_ids:
                race_ids.append(race_id)
    return race_ids


def scrape_grade_race_list_by_search(year: int) -> list[str]:
    """Get grade race IDs via the netkeiba search interface.

    Uses the race search to find G1/G2/G3 races for the given year.
    This is more reliable than the calendar-based approach.

    Returns:
        List of race_id strings.
    """
    race_ids = []
    for grade_code in [1, 2, 3]:  # G1=1, G2=2, G3=3
        page = 1
        while True:
            url = (
                f"{NETKEIBA_DB_URL}/?pid=race_list"
                f"&start_year={year}&end_year={year}"
                f"&grade%5B%5D={grade_code}"
                f"&sort=date&list=100&page={page}"
            )
            soup = _fetch(url)
            links = soup.select("a[href*='/race/']")
            if not links:
                break
            for a_tag in links:
                href = a_tag.get("href", "")
                match = re.search(r"/race/(\d{12})", href)
                if match:
                    rid = match.group(1)
                    if rid not in race_ids:
                        race_ids.append(rid)
            # Check for next page
            next_link = soup.select_one("a.nk_pager_next")
            if not next_link:
                break
            page += 1
    return race_ids


# ---------------------------------------------------------------------------
# Race result scraping
# ---------------------------------------------------------------------------


def scrape_race_result(race_id: str) -> dict | None:
    """Scrape full race result page.

    Returns:
        Dict with keys: race_info (dict), results (list[dict]), payouts (list[dict])
        or None if parsing fails.
    """
    url = f"{NETKEIBA_BASE_URL}/race/result.html?race_id={race_id}"
    soup = _fetch(url)

    race_info = _parse_race_info(soup, race_id)
    if race_info is None:
        return None

    results = _parse_result_table(soup, race_id)
    payouts = _parse_payout_table(soup, race_id)

    return {"race_info": race_info, "results": results, "payouts": payouts}


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> dict | None:
    """Parse race metadata from result page."""
    # Race name
    race_name_tag = soup.select_one("dl.racedata > dt")
    if not race_name_tag:
        return None
    race_name = race_name_tag.get_text(strip=True)

    # Race details line: "芝右 2000m / 天候 : 晴 / 芝 : 良"
    detail_tag = soup.select_one("dl.racedata > dd > p > diary_snap_cut > span")
    if not detail_tag:
        detail_tag = soup.select_one("dl.racedata > dd span")

    course_type = ""
    distance = 0
    track_condition = ""
    weather = ""

    if detail_tag:
        detail_text = detail_tag.get_text(strip=True)
        # Parse course type and distance
        m = re.search(r"(芝|ダ(?:ート)?|障).*?(\d{3,4})m", detail_text)
        if m:
            ct = m.group(1)
            course_type = "ダート" if ct.startswith("ダ") else ct
            distance = int(m.group(2))
        # Weather
        m = re.search(r"天候\s*[:：]\s*(\S+)", detail_text)
        if m:
            weather = m.group(1)
        # Track condition
        m = re.search(r"(?:芝|ダート)\s*[:：]\s*(\S+)", detail_text)
        if m:
            track_condition = m.group(1)

    # Race date from race_id: YYYYMMDDNNRR -> need to look at page
    date_tag = soup.select_one("p.smalltxt, div.race_otherdata p")
    race_date = None
    if date_tag:
        m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_tag.get_text())
        if m:
            race_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Venue from race_id
    venue_codes = {
        "01": "札幌",
        "02": "函館",
        "03": "福島",
        "04": "新潟",
        "05": "東京",
        "06": "中山",
        "07": "中京",
        "08": "京都",
        "09": "阪神",
        "10": "小倉",
    }
    venue_code = race_id[4:6]
    venue = venue_codes.get(venue_code, "")

    # Grade detection
    grade = ""
    grade_tag = soup.select_one("img[src*='icon_grade']")
    if grade_tag:
        src = grade_tag.get("src", "")
        if "01" in src:
            grade = "G1"
        elif "02" in src:
            grade = "G2"
        elif "03" in src:
            grade = "G3"
    if not grade:
        for text in [race_name]:
            if "（G1）" in text or "(G1)" in text or "GI" in text:
                grade = "G1"
            elif "（G2）" in text or "(G2)" in text or "GII" in text:
                grade = "G2"
            elif "（G3）" in text or "(G3)" in text or "GIII" in text:
                grade = "G3"

    return {
        "race_id": race_id,
        "race_name": race_name,
        "race_date": race_date,
        "venue": venue,
        "course_type": course_type,
        "distance": distance,
        "track_condition": track_condition,
        "weather": weather,
        "grade": grade,
        "race_class": "",
        "head_count": 0,  # Updated after parsing results
        "prize_1st": 0,
    }


def _parse_result_table(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """Parse the result table rows."""
    results = []
    table = soup.select_one("table.race_table_01")
    if not table:
        return results

    rows = table.select("tr")[1:]  # Skip header
    for row in rows:
        cols = row.select("td")
        if len(cols) < 13:
            continue

        # Extract horse_id from link
        horse_link = cols[3].select_one("a[href*='horse']")
        horse_id = ""
        if horse_link:
            m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
            if m:
                horse_id = m.group(1)

        # Extract jockey_id
        jockey_link = cols[6].select_one("a[href*='jockey']")
        jockey_id = ""
        if jockey_link:
            m = re.search(r"/jockey/(?:result/recent/)?(\w+)", jockey_link.get("href", ""))
            if m:
                jockey_id = m.group(1)

        # Extract trainer_id
        trainer_link = cols[12].select_one("a[href*='trainer']")
        trainer_id = ""
        if trainer_link:
            m = re.search(r"/trainer/(?:result/recent/)?(\w+)", trainer_link.get("href", ""))
            if m:
                trainer_id = m.group(1)

        # Parse finish order (handle 取消, 除外, 中止)
        finish_text = cols[0].get_text(strip=True)
        finish_order = _safe_int(finish_text) if finish_text.isdigit() else 0

        # Horse weight and change: "480(+2)" or "480(-4)"
        weight_text = cols[14].get_text(strip=True) if len(cols) > 14 else ""
        horse_weight = None
        weight_change = None
        wm = re.match(r"(\d+)\(([+\-]?\d+)\)", weight_text)
        if wm:
            horse_weight = int(wm.group(1))
            weight_change = int(wm.group(2))

        result = {
            "race_id": race_id,
            "horse_id": horse_id,
            "finish_order": finish_order,
            "frame_number": _safe_int(cols[1].get_text(strip=True)),
            "horse_number": _safe_int(cols[2].get_text(strip=True)),
            "horse_name": cols[3].get_text(strip=True),
            "sex_age": cols[4].get_text(strip=True),
            "weight_carried": _safe_float(cols[5].get_text(strip=True)),
            "jockey_id": jockey_id,
            "jockey_name": cols[6].get_text(strip=True),
            "trainer_id": trainer_id,
            "trainer_name": cols[12].get_text(strip=True) if len(cols) > 12 else "",
            "finish_time": _parse_time_to_seconds(cols[7].get_text(strip=True)),
            "margin": cols[8].get_text(strip=True) if len(cols) > 8 else "",
            "passing_order": cols[10].get_text(strip=True) if len(cols) > 10 else "",
            "last_3f": _safe_float(cols[11].get_text(strip=True)) if len(cols) > 11 else None,
            "horse_weight": horse_weight,
            "weight_change": weight_change,
            "odds": _safe_float(cols[12].get_text(strip=True)) if len(cols) > 12 else None,
            "popularity": _safe_int(cols[13].get_text(strip=True)) if len(cols) > 13 else None,
        }

        # Odds and popularity are usually in the last columns
        if len(cols) >= 18:
            result["odds"] = _safe_float(cols[15].get_text(strip=True))
            result["popularity"] = _safe_int(cols[16].get_text(strip=True))

        results.append(result)

    return results


def _parse_payout_table(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """Parse the payout (払戻) table."""
    payouts = []
    payout_tables = soup.select("table.pay_table_01")

    bet_type_map = {
        "単勝": "単勝",
        "複勝": "複勝",
        "枠連": "枠連",
        "馬連": "馬連",
        "ワイド": "ワイド",
        "馬単": "馬単",
        "三連複": "三連複",
        "三連単": "三連単",
    }

    for table in payout_tables:
        rows = table.select("tr")
        for row in rows:
            header = row.select_one("th")
            if not header:
                continue
            bet_type_text = header.get_text(strip=True)
            bet_type = bet_type_map.get(bet_type_text)
            if not bet_type:
                continue

            cols = row.select("td")
            if len(cols) < 2:
                continue

            # Combination and payout can have multiple values (e.g., 複勝 has 3 rows)
            combo_texts = cols[0].get_text(separator="\n").strip().split("\n")
            payout_texts = cols[1].get_text(separator="\n").strip().split("\n")
            pop_texts = (
                cols[2].get_text(separator="\n").strip().split("\n") if len(cols) > 2 else []
            )

            for i, (combo, pay) in enumerate(zip(combo_texts, payout_texts)):
                combo = combo.strip().replace(" ", "").replace("\u3000", "")
                combo = re.sub(r"[→ー−]+", "-", combo)
                pay_val = _safe_int(pay.replace(",", ""))
                pop_val = _safe_int(pop_texts[i]) if i < len(pop_texts) else None

                if combo and pay_val is not None:
                    payouts.append(
                        {
                            "race_id": race_id,
                            "bet_type": bet_type,
                            "combination": combo,
                            "payout": pay_val,
                            "popularity": pop_val,
                        }
                    )

    return payouts


# ---------------------------------------------------------------------------
# Horse profile scraping
# ---------------------------------------------------------------------------


def scrape_horse_profile(horse_id: str) -> dict | None:
    """Scrape horse profile and pedigree info.

    Returns:
        Dict with horse info or None if not found.
    """
    url = f"{NETKEIBA_DB_URL}/horse/{horse_id}"
    soup = _fetch(url)

    name_tag = soup.select_one("div.horse_title h1")
    if not name_tag:
        return None

    horse_name = name_tag.get_text(strip=True)

    # Profile table
    profile = {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "birth_date": None,
        "sex": "",
        "sire_id": "",
        "sire_name": "",
        "dam_id": "",
        "dam_name": "",
        "broodmare_sire_id": "",
        "broodmare_sire_name": "",
        "owner": "",
        "breeder": "",
    }

    # Parse profile table
    for row in soup.select("table.db_prof_table tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if not th or not td:
            continue
        label = th.get_text(strip=True)
        if label == "生年月日":
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", td.get_text())
            if m:
                profile["birth_date"] = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        elif label == "性別":
            profile["sex"] = td.get_text(strip=True)
        elif label == "馬主":
            profile["owner"] = td.get_text(strip=True)
        elif label == "生産者":
            profile["breeder"] = td.get_text(strip=True)

    # Pedigree table
    pedigree_table = soup.select_one("table.blood_table")
    if pedigree_table:
        links = pedigree_table.select("a[href*='/horse/']")
        # Typically: sire, sire's sire, sire's dam, dam, dam's sire (BMS), dam's dam
        pedigree_links = []
        for link in links:
            href = link.get("href", "")
            m = re.search(r"/horse/(\w+)", href)
            if m:
                pedigree_links.append((m.group(1), link.get_text(strip=True)))

        if len(pedigree_links) >= 1:
            profile["sire_id"] = pedigree_links[0][0]
            profile["sire_name"] = pedigree_links[0][1]
        if len(pedigree_links) >= 4:
            profile["dam_id"] = pedigree_links[3][0]
            profile["dam_name"] = pedigree_links[3][1]
        if len(pedigree_links) >= 5:
            profile["broodmare_sire_id"] = pedigree_links[4][0]
            profile["broodmare_sire_name"] = pedigree_links[4][1]

    return profile


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_race_data(data: dict, session=None) -> None:
    """Save scraped race data to database.

    Args:
        data: Result from scrape_race_result()
        session: Optional SQLAlchemy session (created if not provided)
    """
    own_session = session is None
    if own_session:
        init_db()
        session = get_session()

    try:
        race_info = data["race_info"]
        results = data["results"]
        payouts = data["payouts"]

        race_info["head_count"] = len(results)

        # Upsert race
        existing_race = session.get(Race, race_info["race_id"])
        if existing_race:
            for k, v in race_info.items():
                setattr(existing_race, k, v)
        else:
            session.add(Race(**race_info))

        # Delete existing results and re-insert
        session.query(RaceResult).filter_by(race_id=race_info["race_id"]).delete()
        for r in results:
            session.add(RaceResult(**r))

        # Delete existing payouts and re-insert
        session.query(Payout).filter_by(race_id=race_info["race_id"]).delete()
        for p in payouts:
            session.add(Payout(**p))

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def save_horse(profile: dict, session=None) -> None:
    """Save horse profile to database."""
    own_session = session is None
    if own_session:
        init_db()
        session = get_session()

    try:
        existing = session.get(Horse, profile["horse_id"])
        if existing:
            for k, v in profile.items():
                setattr(existing, k, v)
        else:
            session.add(Horse(**profile))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
