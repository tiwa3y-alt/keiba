"""Scraper for JRA official data - race calendar and race cards."""

import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from keiba.config import REQUEST_INTERVAL_SEC

JRA_BASE_URL = "https://www.jra.go.jp"

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
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def fetch_grade_race_calendar(year: int) -> list[dict]:
    """Fetch grade race calendar from JRA.

    Returns:
        List of dicts with keys: race_name, race_date, venue, grade
    """
    url = f"{JRA_BASE_URL}/keiba/program/graderace/"
    soup = _fetch(url)
    races = []

    for row in soup.select("table.tbl-data-04 tbody tr, table.grade-race tbody tr"):
        cols = row.select("td")
        if len(cols) < 4:
            continue

        date_text = cols[0].get_text(strip=True)
        m = re.search(r"(\d{1,2})/(\d{1,2})", date_text)
        if not m:
            continue

        month, day = int(m.group(1)), int(m.group(2))
        try:
            race_date = date(year, month, day)
        except ValueError:
            continue

        race_name = cols[1].get_text(strip=True)
        venue = cols[2].get_text(strip=True) if len(cols) > 2 else ""

        # Detect grade
        grade = ""
        grade_text = cols[3].get_text(strip=True) if len(cols) > 3 else ""
        for g in ["G1", "GI", "G2", "GII", "G3", "GIII"]:
            if g in grade_text or g in race_name:
                grade = g[:2].replace("GI", "G1").replace("GI", "G2").rstrip("I")
                break

        races.append(
            {
                "race_name": race_name,
                "race_date": race_date,
                "venue": venue,
                "grade": grade,
            }
        )

    return races


def fetch_race_card(race_date: date, venue: str, race_number: int) -> list[dict] | None:
    """Fetch race card (出馬表) for a specific race.

    Returns:
        List of entry dicts with horse_name, horse_number, jockey_name, weight_carried, etc.
        or None if not available.
    """
    # JRA uses specific URL patterns for race cards
    date_str = race_date.strftime("%Y%m%d")
    url = f"{JRA_BASE_URL}/JRADB/accessS.html?CESSION={date_str}{race_number:02d}"
    try:
        soup = _fetch(url)
    except requests.RequestException:
        return None

    entries = []
    table = soup.select_one("table.race-denma, table.tbl-data-04")
    if not table:
        return None

    for row in table.select("tbody tr"):
        cols = row.select("td")
        if len(cols) < 6:
            continue

        entry = {
            "horse_number": int(cols[0].get_text(strip=True) or 0),
            "horse_name": cols[2].get_text(strip=True),
            "sex_age": cols[3].get_text(strip=True),
            "weight_carried": float(cols[4].get_text(strip=True) or 0),
            "jockey_name": cols[5].get_text(strip=True),
        }
        entries.append(entry)

    return entries if entries else None
