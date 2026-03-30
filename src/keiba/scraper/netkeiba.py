"""Scraper for netkeiba.com - race results, horse profiles, odds, and payouts.

Uses db.netkeiba.com as primary data source (most stable URL format).
Encoding auto-detection: EUC-JP or UTF-8.
"""

import hashlib
import logging
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from keiba.config import REQUEST_INTERVAL_SEC
from keiba.db.connection import get_session, init_db
from keiba.db.schema import Horse, Payout, Race, RaceResult

logger = logging.getLogger(__name__)

# URLs
DB_BASE = "https://db.netkeiba.com"
RACE_BASE = "https://race.netkeiba.com"

# Raw HTML cache directory
_CACHE_DIR = Path("data/raw")

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)

_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5, 10]


def _load_cookie_string() -> str:
    """Load cookie string from file or environment."""
    import os

    cookie_string = os.environ.get("NETKEIBA_COOKIE", "")
    if not cookie_string:
        cookie_file = Path("data/cookie.txt")
        if cookie_file.exists():
            cookie_string = cookie_file.read_text().strip()

    if cookie_string:
        logger.info(f"Loaded cookie ({len(cookie_string)} chars)")
    else:
        logger.warning(
            "No netkeiba cookie found. Set NETKEIBA_COOKIE env var "
            "or create data/cookie.txt. See README for instructions."
        )
    return cookie_string


# Load cookie as raw header (more reliable than parsing into individual cookies)
_COOKIE_STRING = _load_cookie_string()
if _COOKIE_STRING:
    _SESSION.headers["Cookie"] = _COOKIE_STRING


def _cache_path(url: str) -> Path:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return _CACHE_DIR / f"{url_hash}.html"


def _fetch(url: str, use_cache: bool = True) -> BeautifulSoup:
    """Fetch URL with rate limiting, retry, caching, and auto encoding."""
    if use_cache:
        cache = _cache_path(url)
        if cache.exists():
            html = cache.read_text(encoding="utf-8")
            return BeautifulSoup(html, "lxml")

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            time.sleep(REQUEST_INTERVAL_SEC)
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()

            # Auto-detect encoding: netkeiba uses EUC-JP on db.*, UTF-8 on race.*
            if resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            elif "db.netkeiba" in url:
                resp.encoding = "EUC-JP"
            else:
                resp.encoding = "UTF-8"

            html = resp.text

            if use_cache:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                _cache_path(url).write_text(html, encoding="utf-8")

            return BeautifulSoup(html, "lxml")

        except requests.RequestException as e:
            last_error = e
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            logger.warning(f"Retry {attempt + 1}/{_MAX_RETRIES} for {url}: {e}")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {_MAX_RETRIES} retries: {url}: {last_error}")


# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------

def _parse_time_to_seconds(time_str: str) -> float | None:
    if not time_str or time_str.strip() in ("", "-"):
        return None
    time_str = time_str.strip()
    m = re.match(r"^(\d+):(\d+\.?\d*)$", time_str)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"^(\d+)\.(\d{2}\.\d+)$", time_str)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(time_str)
    except ValueError:
        return None


def _safe_int(val: str) -> int | None:
    try:
        return int(re.sub(r"[^\d\-]", "", val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.\-]", "", val))
    except (ValueError, TypeError):
        return None


VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


# ---------------------------------------------------------------------------
# Race list: use db.netkeiba.com/race/list/ (most reliable)
# ---------------------------------------------------------------------------

def scrape_grade_race_list_by_search(year: int) -> list[str]:
    """Get grade race IDs for a given year.

    Strategy:
    1. Try race.netkeiba.com top page date-based listing
    2. Fall back to db.netkeiba.com search (may need login)
    """
    race_ids = []

    # Method 1: db.netkeiba.com search (works if not login-gated)
    for grade_code in [1, 2, 3]:
        page = 1
        while True:
            url = (
                f"{DB_BASE}/?pid=race_list"
                f"&start_year={year}&end_year={year}"
                f"&grade%5B%5D={grade_code}"
                f"&sort=date&list=100&page={page}"
            )
            soup = _fetch(url, use_cache=False)

            links = soup.select("a[href*='/race/']")
            found = 0
            for a_tag in links:
                href = a_tag.get("href", "")
                match = re.search(r"/race/(\d{12})", href)
                if match:
                    rid = match.group(1)
                    if rid not in race_ids:
                        race_ids.append(rid)
                        found += 1

            if found == 0:
                break

            pager = soup.select("a[href*='page=']")
            has_next = any(f"page={page + 1}" in a.get("href", "") for a in pager)
            if not has_next:
                break
            page += 1

    # Method 2: If db.netkeiba returned nothing, try race calendar
    if not race_ids:
        logger.info("db.netkeiba search returned no results, trying race calendar...")
        import calendar

        for month in range(1, 13):
            _, last_day = calendar.monthrange(year, month)
            # Scan each Saturday and Sunday (typical race days)
            for day in range(1, last_day + 1):
                d = date(year, month, day)
                if d.weekday() not in (5, 6):  # Sat=5, Sun=6
                    continue
                if d > date.today():
                    continue

                date_str = d.strftime("%Y%m%d")
                url = f"{RACE_BASE}/top/race_list.html?kaisai_date={date_str}"
                try:
                    soup = _fetch(url)
                    for a_tag in soup.select("a[href*='/race/']"):
                        href = a_tag.get("href", "")
                        match = re.search(r"/race/(\d{12})", href)
                        if match:
                            rid = match.group(1)
                            if rid not in race_ids:
                                race_ids.append(rid)
                except Exception:
                    continue

    logger.info(f"Found {len(race_ids)} races for {year}")
    return race_ids


# ---------------------------------------------------------------------------
# Race result scraping: db.netkeiba.com/race/{race_id}/
# ---------------------------------------------------------------------------

def scrape_race_result(race_id: str) -> dict | None:
    """Scrape race result page.

    Tries race.netkeiba.com first (free, no login required),
    then falls back to db.netkeiba.com (may need login).
    """
    # Primary: race.netkeiba.com (free access)
    url = f"{RACE_BASE}/race/result.html?race_id={race_id}"
    soup = _fetch(url)

    race_info = _parse_race_info_db(soup, race_id)
    results = _parse_result_table_db(soup, race_id) if race_info else []

    if not race_info or not results:
        # Fallback: db.netkeiba.com
        url2 = f"{DB_BASE}/race/{race_id}/"
        soup = _fetch(url2)
        race_info = _parse_race_info_db(soup, race_id)
        results = _parse_result_table_db(soup, race_id) if race_info else []

    if not race_info or not results:
        return None

    payouts = _parse_payout_table_db(soup, race_id)

    return {"race_info": race_info, "results": results, "payouts": payouts}


def _parse_race_info_db(soup: BeautifulSoup, race_id: str) -> dict | None:
    """Parse race metadata. Tries multiple selector patterns."""

    # --- Race name ---
    race_name = ""
    for sel in [
        "dl.racedata > dt",          # Classic db.netkeiba
        "div.RaceName",               # New format
        "h1.RaceName_main",           # Alternate new
        "div.data_intro h1",          # db.netkeiba 2024+
        "div.data_intro dt",          # db.netkeiba variant
        "h1",                         # Last resort
    ]:
        tag = soup.select_one(sel)
        if tag:
            race_name = tag.get_text(strip=True)
            if race_name and len(race_name) > 1:
                break

    # Fallback: try title tag
    if not race_name and soup.title:
        title = soup.title.get_text(strip=True)
        # netkeiba titles are like "レース名 | netkeiba"
        race_name = title.split("|")[0].strip()

    if not race_name:
        logger.warning(f"Could not parse race name for {race_id}")
        return None

    # --- Race details (distance, surface, condition, weather) ---
    detail_text = ""
    for sel in [
        "dl.racedata > dd",           # Classic
        "div.RaceData01",             # New format
        "div.RaceData",               # Alternate
        "p.smalltxt",                 # Fallback
        "diary_snap_cut span",        # Very old format
    ]:
        tag = soup.select_one(sel)
        if tag:
            detail_text = tag.get_text(separator=" ", strip=True)
            if "m" in detail_text or "芝" in detail_text or "ダ" in detail_text:
                break

    course_type = ""
    distance = 0
    track_condition = ""
    weather = ""

    if detail_text:
        m = re.search(r"(芝|ダ(?:ート)?|障).*?(\d{3,4})m", detail_text)
        if m:
            ct = m.group(1)
            course_type = "ダート" if ct.startswith("ダ") else ct
            distance = int(m.group(2))
        m = re.search(r"天候\s*[:：]\s*(\S+)", detail_text)
        if m:
            weather = m.group(1)
        m = re.search(r"(?:芝|ダート|ダ)\s*[:：]\s*(\S+)", detail_text)
        if m:
            track_condition = m.group(1)

    # --- Race date ---
    race_date = None
    # Try meta / text content
    all_text = soup.get_text()
    m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", all_text)
    if m:
        try:
            race_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # --- Venue from race_id ---
    venue_code = race_id[4:6]
    venue = VENUE_CODES.get(venue_code, "")

    # --- Grade detection ---
    grade = ""
    # Method 1: icon image
    grade_tag = soup.select_one("img[src*='icon_grade'], img[src*='ico_grade']")
    if grade_tag:
        src = grade_tag.get("src", "")
        if "01" in src or "g1" in src.lower():
            grade = "G1"
        elif "02" in src or "g2" in src.lower():
            grade = "G2"
        elif "03" in src or "g3" in src.lower():
            grade = "G3"

    # Method 2: class/span with grade info
    if not grade:
        for tag in soup.select("span.Icon_GradeType1, span[class*='Grade']"):
            t = tag.get_text(strip=True)
            if "G1" in t or "GI" == t:
                grade = "G1"
            elif "G2" in t or "GII" == t:
                grade = "G2"
            elif "G3" in t or "GIII" == t:
                grade = "G3"

    # Method 3: text in race name or title
    if not grade:
        combined = race_name + " " + soup.title.get_text() if soup.title else race_name
        for pattern, g in [
            (r"\(G1\)|（G1）|GⅠ|\bGI\b", "G1"),
            (r"\(G2\)|（G2）|GⅡ|\bGII\b", "G2"),
            (r"\(G3\)|（G3）|GⅢ|\bGIII\b", "G3"),
        ]:
            if re.search(pattern, combined):
                grade = g
                break

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
        "head_count": 0,
        "prize_1st": 0,
    }


def _parse_result_table_db(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """Parse race result table with multiple format support."""
    results = []

    # Try multiple table selectors
    table = None
    for sel in [
        "table.race_table_01",          # Classic db.netkeiba
        "table.nk_tb_common",           # db.netkeiba 2024+
        "table.RaceTable01",             # race.netkeiba new
        "table.Shutuba_Table",           # Race card
        "table[class*='race_table']",    # Fuzzy match
        "table[summary*='レース結果']",    # Summary attribute
        "table[summary*='全着順']",       # Full results summary
    ]:
        table = soup.select_one(sel)
        if table:
            break

    if not table:
        # Last resort: find the table with most rows containing links to horses
        best_table = None
        best_count = 0
        for t in soup.select("table"):
            horse_links = t.select("a[href*='/horse/']")
            if len(horse_links) > best_count:
                best_count = len(horse_links)
                best_table = t
        if best_table and best_count >= 3:
            table = best_table
        else:
            return results

    rows = table.select("tr")
    if len(rows) < 2:
        return results

    # Detect header to understand column layout
    header_row = rows[0]
    headers = [th.get_text(strip=True) for th in header_row.select("th, td")]

    # Build column index map
    col_map = {}
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if h_clean in ("着順", "着\u3000順"):
            col_map["finish"] = i
        elif h_clean in ("枠番", "枠"):
            col_map["frame"] = i
        elif h_clean in ("馬番", "馬\u3000番"):
            col_map["number"] = i
        elif h_clean in ("馬名",):
            col_map["name"] = i
        elif h_clean in ("性齢",):
            col_map["sex_age"] = i
        elif h_clean in ("斤量", "負担重量"):
            col_map["weight_carried"] = i
        elif h_clean in ("騎手",):
            col_map["jockey"] = i
        elif h_clean in ("タイム", "走破タイム"):
            col_map["time"] = i
        elif h_clean in ("着差",):
            col_map["margin"] = i
        elif h_clean in ("通過", "通過順位"):
            col_map["passing"] = i
        elif h_clean in ("上り", "上がり", "上がり3F"):
            col_map["last3f"] = i
        elif h_clean in ("単勝", "単勝オッズ", "オッズ"):
            col_map["odds"] = i
        elif h_clean in ("人気", "人\u3000気"):
            col_map["popularity"] = i
        elif h_clean in ("馬体重",):
            col_map["horse_weight"] = i
        elif h_clean in ("調教師",):
            col_map["trainer"] = i

    # Fallback: if no headers detected, use classic db.netkeiba layout
    if not col_map:
        col_map = {
            "finish": 0, "frame": 1, "number": 2, "name": 3,
            "sex_age": 4, "weight_carried": 5, "jockey": 6,
            "time": 7, "margin": 8, "passing": 10, "last3f": 11,
            "trainer": 12, "odds": 12, "popularity": 13, "horse_weight": 14,
        }

    for row in rows[1:]:
        cols = row.select("td")
        if len(cols) < 5:
            continue

        def _get(key, default=""):
            idx = col_map.get(key)
            if idx is not None and idx < len(cols):
                return cols[idx].get_text(strip=True)
            return default

        def _get_link(key, pattern):
            idx = col_map.get(key)
            if idx is not None and idx < len(cols):
                link = cols[idx].select_one(f"a[href*='{pattern}']")
                if link:
                    m = re.search(rf"/{pattern}/(?:result/recent/)?(\w+)", link.get("href", ""))
                    if m:
                        return m.group(1)
            return ""

        # Finish order
        finish_text = _get("finish")
        finish_order = _safe_int(finish_text) if finish_text and finish_text[0].isdigit() else 0

        # Horse weight and change
        weight_text = _get("horse_weight")
        horse_weight = None
        weight_change = None
        wm = re.match(r"(\d+)\(([+\-]?\d+)\)", weight_text)
        if wm:
            horse_weight = int(wm.group(1))
            weight_change = int(wm.group(2))

        result = {
            "race_id": race_id,
            "horse_id": _get_link("name", "horse"),
            "finish_order": finish_order,
            "frame_number": _safe_int(_get("frame")),
            "horse_number": _safe_int(_get("number")),
            "horse_name": _get("name"),
            "sex_age": _get("sex_age"),
            "weight_carried": _safe_float(_get("weight_carried")),
            "jockey_id": _get_link("jockey", "jockey"),
            "jockey_name": _get("jockey"),
            "trainer_id": _get_link("trainer", "trainer"),
            "trainer_name": _get("trainer"),
            "finish_time": _parse_time_to_seconds(_get("time")),
            "margin": _get("margin"),
            "passing_order": _get("passing"),
            "last_3f": _safe_float(_get("last3f")),
            "horse_weight": horse_weight,
            "weight_change": weight_change,
            "odds": _safe_float(_get("odds")),
            "popularity": _safe_int(_get("popularity")),
        }

        results.append(result)

    return results


def _parse_payout_table_db(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """Parse payout table with multiple format support."""
    payouts = []

    # Find payout tables
    payout_tables = soup.select("table.pay_table_01, table.Payout_Table, table[class*='pay']")
    if not payout_tables:
        # Try finding by content
        for table in soup.select("table"):
            text = table.get_text()
            if "単勝" in text and ("払戻" in text or "配当" in text or any(
                c.isdigit() for c in text[:100]
            )):
                payout_tables.append(table)
                break

    bet_type_map = {
        "単勝": "単勝", "複勝": "複勝", "枠連": "枠連",
        "馬連": "馬連", "ワイド": "ワイド", "馬単": "馬単",
        "三連複": "三連複", "三連単": "三連単",
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

            combo_texts = cols[0].get_text(separator="\n").strip().split("\n")
            payout_texts = cols[1].get_text(separator="\n").strip().split("\n")
            pop_texts = (
                cols[2].get_text(separator="\n").strip().split("\n") if len(cols) > 2 else []
            )

            for i, (combo, pay) in enumerate(zip(combo_texts, payout_texts)):
                combo = combo.strip().replace(" ", "").replace("\u3000", "")
                combo = re.sub(r"[→ー−－]+", "-", combo)
                pay_val = _safe_int(pay.replace(",", ""))
                pop_val = _safe_int(pop_texts[i]) if i < len(pop_texts) else None

                if combo and pay_val is not None:
                    payouts.append({
                        "race_id": race_id,
                        "bet_type": bet_type,
                        "combination": combo,
                        "payout": pay_val,
                        "popularity": pop_val,
                    })

    return payouts


# ---------------------------------------------------------------------------
# Horse profile
# ---------------------------------------------------------------------------

def scrape_horse_profile(horse_id: str) -> dict | None:
    """Scrape horse profile from db.netkeiba.com."""
    url = f"{DB_BASE}/horse/{horse_id}"
    soup = _fetch(url)

    # Find horse name - multiple patterns
    horse_name = ""
    for sel in ["div.horse_title h1", "h1.Horse_Name", "h1"]:
        tag = soup.select_one(sel)
        if tag:
            horse_name = tag.get_text(strip=True)
            if horse_name:
                break

    if not horse_name:
        return None

    profile = {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "birth_date": None,
        "sex": "",
        "sire_id": "", "sire_name": "",
        "dam_id": "", "dam_name": "",
        "broodmare_sire_id": "", "broodmare_sire_name": "",
        "owner": "", "breeder": "",
    }

    # Profile table
    for row in soup.select("table.db_prof_table tr, table.Horse_Profile tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if not th or not td:
            continue
        label = th.get_text(strip=True)
        if "生年月日" in label:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", td.get_text())
            if m:
                profile["birth_date"] = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        elif "性別" in label or "性齢" in label:
            profile["sex"] = td.get_text(strip=True)[:1]
        elif "馬主" in label:
            profile["owner"] = td.get_text(strip=True)
        elif "生産者" in label:
            profile["breeder"] = td.get_text(strip=True)

    # Pedigree table
    for sel in ["table.blood_table", "table[class*='blood']", "table[class*='Pedigree']"]:
        ped_table = soup.select_one(sel)
        if ped_table:
            links = ped_table.select("a[href*='/horse/']")
            ped_links = []
            for link in links:
                href = link.get("href", "")
                m = re.search(r"/horse/(\w+)", href)
                if m:
                    ped_links.append((m.group(1), link.get_text(strip=True)))

            if len(ped_links) >= 1:
                profile["sire_id"] = ped_links[0][0]
                profile["sire_name"] = ped_links[0][1]
            if len(ped_links) >= 4:
                profile["dam_id"] = ped_links[3][0]
                profile["dam_name"] = ped_links[3][1]
            if len(ped_links) >= 5:
                profile["broodmare_sire_id"] = ped_links[4][0]
                profile["broodmare_sire_name"] = ped_links[4][1]
            break

    return profile


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_race_data(data: dict, session=None) -> None:
    """Save scraped race data to database."""
    own_session = session is None
    if own_session:
        init_db()
        session = get_session()

    try:
        race_info = data["race_info"]
        results = data["results"]
        payouts = data["payouts"]
        race_info["head_count"] = len(results)

        existing_race = session.get(Race, race_info["race_id"])
        if existing_race:
            for k, v in race_info.items():
                setattr(existing_race, k, v)
        else:
            session.add(Race(**race_info))

        session.query(RaceResult).filter_by(race_id=race_info["race_id"]).delete()
        for r in results:
            session.add(RaceResult(**r))

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
