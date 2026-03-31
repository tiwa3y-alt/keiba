"""Scrape race results from JRA official website (jra.go.jp).

JRA公式サイトは無料・ログイン不要でレース結果を閲覧できます。
netkeiba.comがログイン必須のため、こちらをメインのデータソースとして使用。
"""

import re
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

from keiba.config import REQUEST_INTERVAL_SEC
from keiba.db.connection import get_session, init_db
from keiba.db.schema import Horse, Payout, Race, RaceResult

_CACHE_DIR = Path("data/raw_jra")

VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

# JRA grade race definitions (手動定義 - 最も確実な方法)
# Format: (month, day_approx, race_name, venue_code, distance, surface, grade)
# 主要G1レースのみ抜粋。実際の日程は毎年変わるため、カレンダーから取得する方が望ましい


def _fetch_jra(url: str, use_cache: bool = True) -> str:
    """Fetch URL using curl (JRA is free, no login needed)."""
    cache_path = _CACHE_DIR / (re.sub(r"[^\w]", "_", url)[-80:] + ".html")

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    time.sleep(REQUEST_INTERVAL_SEC)

    result = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "30",
         "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
         url],
        capture_output=True, timeout=35,
    )

    raw = result.stdout
    # JRA uses UTF-8
    html = raw.decode("utf-8", errors="replace")

    if use_cache and html:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")

    return html


def scrape_jra_race_list(year: int) -> list[dict]:
    """Scrape race IDs from JRA's access page.

    JRA URL pattern:
    https://www.jra.go.jp/JRADB/accessD.html?CESSION=YYYYPPDD
    where PP=場所コード, DD=日

    Alternative: Use the calendar to find race days.
    """
    # JRA doesn't have a simple race list API
    # Instead, we use the Yahoo Keiba calendar which is free
    races = []

    for month in range(1, 13):
        url = f"https://keiba.yahoo.co.jp/schedule/list/{year}/?month={month}"
        html = _fetch_jra(url)
        soup = BeautifulSoup(html, "lxml")

        # Find race links
        for a_tag in soup.select("a[href*='/race/result/']"):
            href = a_tag.get("href", "")
            match = re.search(r"/race/result/(\d{12})", href)
            if match:
                race_id = match.group(1)
                races.append({"race_id": race_id})

    return races


def scrape_yahoo_race_result(race_id: str) -> dict | None:
    """Scrape race result from Yahoo Keiba (free, no login).

    URL: https://keiba.yahoo.co.jp/race/result/XXXXXXXXXXXX/
    """
    url = f"https://keiba.yahoo.co.jp/race/result/{race_id}/"
    html = _fetch_jra(url)
    soup = BeautifulSoup(html, "lxml")

    # Parse race info
    race_name = ""
    title_tag = soup.select_one("h1.fntB, div.resultName h1, h1")
    if title_tag:
        race_name = title_tag.get_text(strip=True)

    if not race_name:
        return None

    # Race details
    detail_text = ""
    for sel in ["p.race_detail", "div.raceData", "p.data", "div.resultData"]:
        tag = soup.select_one(sel)
        if tag:
            detail_text = tag.get_text(separator=" ", strip=True)
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

    # Date
    race_date = None
    all_text = soup.get_text()
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", all_text)
    if m:
        try:
            race_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Venue
    venue_code = race_id[4:6]
    venue = VENUE_CODES.get(venue_code, "")

    # Grade
    grade = ""
    for pattern, g in [
        (r"\(G1\)|（G1）|GⅠ|\bGI\b|Ｇ１", "G1"),
        (r"\(G2\)|（G2）|GⅡ|\bGII\b|Ｇ２", "G2"),
        (r"\(G3\)|（G3）|GⅢ|\bGIII\b|Ｇ３", "G3"),
    ]:
        if re.search(pattern, race_name + " " + detail_text):
            grade = g
            break

    race_info = {
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

    # Parse result table
    results = _parse_yahoo_result_table(soup, race_id)
    if not results:
        return None

    race_info["head_count"] = len(results)

    # Parse payouts
    payouts = _parse_yahoo_payout_table(soup, race_id)

    return {"race_info": race_info, "results": results, "payouts": payouts}


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


def _parse_time(time_str: str) -> float | None:
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


def _parse_yahoo_result_table(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """Parse Yahoo Keiba result table."""
    results = []

    # Find result table
    table = None
    for sel in ["table#raceScore", "table.resultTbl", "table.dataLs"]:
        table = soup.select_one(sel)
        if table:
            break

    if not table:
        # Find table with most rows
        for t in soup.select("table"):
            if len(t.select("tr")) > 5:
                table = t
                break

    if not table:
        return results

    rows = table.select("tr")
    if len(rows) < 2:
        return results

    # Detect column layout from header
    headers = [th.get_text(strip=True) for th in rows[0].select("th, td")]
    col_map = {}
    for i, h in enumerate(headers):
        if h in ("着順", "着"):
            col_map["finish"] = i
        elif h in ("枠番", "枠"):
            col_map["frame"] = i
        elif h in ("馬番",):
            col_map["number"] = i
        elif h in ("馬名",):
            col_map["name"] = i
        elif h in ("性齢",):
            col_map["sex_age"] = i
        elif h in ("斤量",):
            col_map["weight_carried"] = i
        elif h in ("騎手",):
            col_map["jockey"] = i
        elif h in ("タイム",):
            col_map["time"] = i
        elif h in ("着差",):
            col_map["margin"] = i
        elif h in ("通過", "通過順"):
            col_map["passing"] = i
        elif h in ("上り", "上がり"):
            col_map["last3f"] = i
        elif h in ("単勝", "オッズ"):
            col_map["odds"] = i
        elif h in ("人気",):
            col_map["popularity"] = i
        elif h in ("馬体重",):
            col_map["horse_weight"] = i
        elif h in ("調教師",):
            col_map["trainer"] = i

    if not col_map:
        col_map = {
            "finish": 0, "frame": 1, "number": 2, "name": 3,
            "sex_age": 4, "weight_carried": 5, "jockey": 6,
            "time": 7, "margin": 8, "passing": 10, "last3f": 11,
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

        def _get_link_id(key, pattern):
            idx = col_map.get(key)
            if idx is not None and idx < len(cols):
                link = cols[idx].select_one(f"a[href*='{pattern}']")
                if link:
                    m = re.search(rf"/{pattern}/(\w+)", link.get("href", ""))
                    if m:
                        return m.group(1)
            return ""

        finish_text = _get("finish")
        finish_order = _safe_int(finish_text) if finish_text and finish_text[0].isdigit() else 0

        weight_text = _get("horse_weight")
        horse_weight = None
        weight_change = None
        wm = re.match(r"(\d+)\(([+\-]?\d+)\)", weight_text)
        if wm:
            horse_weight = int(wm.group(1))
            weight_change = int(wm.group(2))

        result = {
            "race_id": race_id,
            "horse_id": _get_link_id("name", "horse") or _get_link_id("name", "directory/horse"),
            "finish_order": finish_order,
            "frame_number": _safe_int(_get("frame")),
            "horse_number": _safe_int(_get("number")),
            "horse_name": _get("name"),
            "sex_age": _get("sex_age"),
            "weight_carried": _safe_float(_get("weight_carried")),
            "jockey_id": _get_link_id("jockey", "jockey") or _get_link_id("jockey", "directory/jocky"),
            "jockey_name": _get("jockey"),
            "trainer_id": _get_link_id("trainer", "trainer") or _get_link_id("trainer", "directory/trainer"),
            "trainer_name": _get("trainer"),
            "finish_time": _parse_time(_get("time")),
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


def _parse_yahoo_payout_table(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """Parse Yahoo Keiba payout table."""
    payouts = []

    for table in soup.select("table"):
        text = table.get_text()
        if "単勝" not in text:
            continue

        bet_type_map = {
            "単勝": "単勝", "複勝": "複勝", "枠連": "枠連",
            "馬連": "馬連", "ワイド": "ワイド", "馬単": "馬単",
            "三連複": "三連複", "三連単": "三連単",
        }

        for row in table.select("tr"):
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
            pop_texts = cols[2].get_text(separator="\n").strip().split("\n") if len(cols) > 2 else []

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


# --- Persistence (reuse from netkeiba module) ---

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

        existing = session.get(Race, race_info["race_id"])
        if existing:
            for k, v in race_info.items():
                setattr(existing, k, v)
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
