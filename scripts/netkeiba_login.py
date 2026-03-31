"""One-time login to netkeiba.com using Playwright.

Opens a browser window where you log in manually.
Saves the browser state (cookies, session) to data/browser_state/
for future headless scraping.

Usage:
    python scripts/netkeiba_login.py
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_DIR = Path("data/browser_state")


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / "state.json"

    print("=" * 50)
    print("netkeiba.com ログインツール")
    print("=" * 50)
    print()
    print("ブラウザが開きます。")
    print("1. netkeiba.com にログインしてください")
    print("2. ログイン完了後、ターミナルで Enter を押してください")
    print()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(locale="ja-JP")
        page = context.new_page()

        # Navigate to netkeiba login page
        page.goto("https://www.netkeiba.com/")
        print("ブラウザが開きました。")
        print("右上の「ログイン」ボタンからログインしてください...")
        print()

        input("ログイン完了後、ここで Enter を押してください >>> ")

        # Verify login
        page.goto("https://db.netkeiba.com/race/202409050811/")
        page.wait_for_timeout(3000)
        html = page.content()
        tables = html.count("<table")
        print(f"\n確認中... テーブル数: {tables}")

        if tables >= 5:
            print("✓ ログイン成功！データが読み取れました。")
        else:
            print("△ テーブルが少ないですが、状態を保存します。")

        # Save browser state
        context.storage_state(path=str(state_file))
        print(f"\nブラウザ状態を保存しました: {state_file}")
        print("これ以降、keiba scrape はこの状態を使って自動実行されます。")

        browser.close()

    print("\n完了！")


if __name__ == "__main__":
    main()
