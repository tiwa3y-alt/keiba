# keiba - AI競馬予想システム

重賞レース(G1/G2/G3)に特化した、期待値ベースのAI競馬予想システムです。

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/tiwa3y-alt/keiba.git
cd keiba
```

### 2. Python環境の構築

Python 3.11以上が必要です。

```bash
# venvの作成（推奨）
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

# パッケージのインストール
pip install -e ".[dev]"
```

### 3. 環境変数の設定

```bash
cp .env.example .env
# .env を編集（現時点では特に設定不要）
```

## 使い方

### Step 1: データ収集（初回のみ、約2-3時間）

netkeiba.comから過去10年分の重賞レースデータを収集します。

```bash
# 10年分を一括取得（推奨）
keiba scrape --year 2016 --start-year 2016 --end-year 2025

# 1年ずつ取得する場合
keiba scrape --year 2016
keiba scrape --year 2017
# ... 以下同様

# 2026年のデータも取得
keiba scrape --year 2026
```

**ポイント:**
- リクエスト間隔1秒 + リトライ機能付き（サーバーに優しい設計）
- HTMLキャッシュ機能あり（`data/raw/`に保存）。中断しても再実行すればキャッシュから再開
- 既にDBに保存済みのレースは自動スキップ

### Step 2: 現在の状態を確認

```bash
keiba status
```

出力例:
```
=== Database Status ===
  Races:        1,240
  Results:      15,931
  Horses:       500
  Date range:   2016-01-03 ~ 2025-12-28
    G1: 240 races
    G2: 350 races
    G3: 650 races
```

### Step 3: パイプライン実行（特徴量→学習→バックテスト）

```bash
python scripts/run_pipeline.py
```

これ1つで以下を自動実行します（約30秒）:
1. **特徴量構築** — 53特徴量をベクトル化計算
2. **Walk-forward学習** — 年ごとにLightGBMを再学習
3. **バックテスト** — 4戦略 × 複数馬券種で期待値検証
4. **レポート生成** — `reports/` にチャート出力

### Step 4: 特定レースの予測

```bash
# レースIDはnetkeibaのURL末尾の数字12桁
# 例: https://race.netkeiba.com/race/result.html?race_id=202409030811
keiba predict 202409030811
```

### Step 5: 大阪杯の期待値分析

```bash
# 出走馬データを直接スクリプトに記載して分析
python scripts/osaka_hai_2026.py
```

## バックテスト戦略

```bash
# 単勝バリューベット（EV > 1.2 で賭ける）
keiba backtest --bet-types 単勝 --strategy value --ev-threshold 1.2

# Kelly基準（保守的1/4 Kelly）
keiba backtest --bet-types 単勝 --strategy fractional_kelly

# 全馬券種（単勝+馬連+三連複）
keiba backtest --bet-types 単勝,馬連,三連複 --strategy value --ev-threshold 1.3

# 初期資金を変えてテスト
keiba backtest --bankroll 500000 --bet-types 単勝
```

## プロジェクト構成

```
keiba/
├── src/keiba/
│   ├── scraper/       # netkeiba.com + JRA スクレイパー
│   ├── features/      # 53特徴量（馬/騎手/調教師/血統/市場/レース）
│   ├── models/        # LightGBM LambdaRank + 確率キャリブレーション
│   ├── backtest/      # 全7馬券種対応バックテストエンジン
│   └── cli.py         # CLIコマンド
├── scripts/
│   ├── run_pipeline.py          # 全工程一括実行
│   ├── generate_sample_data.py  # テスト用合成データ生成
│   └── osaka_hai_2026.py        # 大阪杯期待値分析
├── tests/             # テスト（41テスト）
├── data/              # SQLiteデータベース（gitignore）
├── models/            # 学習済みモデル（gitignore）
└── reports/           # レポート・チャート（gitignore）
```

## テスト

```bash
python -m pytest tests/ -v
```

## ライセンス

個人利用を想定しています。netkeiba.comのデータ利用は同サイトの利用規約に従ってください。
