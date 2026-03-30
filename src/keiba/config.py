"""Configuration management for keiba project."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'keiba.db'}")

# Scraper settings
NETKEIBA_BASE_URL = "https://race.netkeiba.com"
NETKEIBA_DB_URL = "https://db.netkeiba.com"
REQUEST_INTERVAL_SEC = 1.0

# Grade race filter
GRADES = ("G1", "G2", "G3")

# Feature engineering
RECENT_RACES_WINDOW = 5
STATS_LOOKBACK_DAYS = 365

# Model
RANDOM_SEED = 42
