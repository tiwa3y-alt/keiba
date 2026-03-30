# CLAUDE.md - Development Guide for keiba

## Project Overview
AI-powered Japanese horse racing (keiba) prediction system targeting grade races (G1/G2/G3).

## Quick Start
```bash
pip install -e ".[dev]"      # Install with dev dependencies
keiba status                  # Check database/model status
python -m pytest tests/ -v    # Run tests
```

## Architecture
```
src/keiba/
├── scraper/     # netkeiba.com + JRA data collection
├── features/    # 53 features (horse/jockey/trainer/race/pedigree/market)
├── models/      # LightGBM LambdaRank + probability calibration
├── backtest/    # All 7 bet types, Kelly criterion, value betting
└── cli.py       # CLI commands
```

## CLI Commands
```bash
keiba scrape --year 2024                         # Scrape single year
keiba scrape --start-year 2016 --end-year 2025   # Scrape range (resume-safe)
keiba features --rebuild                          # Build feature matrix
keiba train --start-test-year 2021                # Walk-forward training
keiba backtest --bet-types 単勝,馬連,三連複       # Run backtest
keiba predict 202405050811                        # Predict specific race
keiba status                                      # Show DB/model status
```

## Full Pipeline (Optimized)
```bash
python scripts/run_pipeline.py    # Features → Train → Backtest (vectorized, ~30s)
python scripts/generate_sample_data.py  # Generate synthetic data for testing
```

## Key Design Decisions
- **No data leakage**: All features use strictly prior race data
- **Walk-forward validation**: Train on past, test on future, never peek
- **Harville model**: Derives multi-horse combination probabilities from win probabilities
- **HTML caching**: Scraped pages cached to `data/raw/` for instant re-parsing

## Testing
```bash
python -m pytest tests/ -v       # All tests
python -m pytest tests/test_backtest.py -v   # Backtest tests only
```

## Data Flow
1. `scrape` → SQLite DB (`data/keiba.db`): races, race_results, horses, payouts
2. `features` → Parquet (`data/features.parquet`): 53 features per entry
3. `train` → Models (`models/*.txt`): LightGBM per test year
4. `backtest` → Reports (`reports/`): ROI, drawdown, monthly charts

## Important Files
- `src/keiba/features/pipeline.py:FEATURE_COLUMNS` — master feature list
- `src/keiba/models/trainer.py` — walk-forward training logic
- `src/keiba/backtest/engine.py` — all 7 bet type simulation
- `src/keiba/models/predictor.py:compute_harville_probs` — combination probability math
