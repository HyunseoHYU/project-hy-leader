"""
비트코인 관련 뉴스 수집기 (3개 소스 → CSV 1개)
============================================

소스
    CryptoPanic API   암호화폐 뉴스 모음      (CRYPTOPANIC_API_KEY 필요)
    NewsAPI           일반 뉴스 검색          (NEWSAPI_KEY 필요)
    RSS               CoinDesk·Cointelegraph·Decrypt  (키 불필요)

    API 키가 없는 소스는 경고만 출력하고 건너뛴다 → 일부 키만 있어도 실행 가능.

출력
    data/crypto_news.csv   (이미 저장된 id는 다시 저장하지 않음 → 여러 번 실행해도 중복 없음)

실행
    python collectors/collect_crypto_news.py     (주기적으로 실행해 누적 수집)
"""

import os
import time
from datetime import datetime, timezone

import feedparser
import requests

from common import DATA_DIR, append_rows_csv, load_seen_ids

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
OUTPUT_PATH = os.path.join(DATA_DIR, "crypto_news.csv")
COLUMNS = ["source", "id", "published_at", "title", "body", "url"]

RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
}
BITCOIN_KEYWORDS = ("bitcoin", "btc")  # RSS는 전체 암호화폐 기사라 이 단어가 있는 것만 남김


def clean_text(text) -> str:
    """줄바꿈을 공백으로 바꾸고 앞뒤 공백 제거 (CSV 한 칸에 한 줄로 들어가게)."""
    return (text or "").replace("\n", " ").strip()


def is_bitcoin_related(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in BITCOIN_KEYWORDS)


# ---------------------------------------------------------------------------
# 소스 1: CryptoPanic (여러 페이지)
# ---------------------------------------------------------------------------
def fetch_cryptopanic(api_key):
    if not api_key:
        print("  [cryptopanic] skipped: CRYPTOPANIC_API_KEY not set")
        return []

    rows = []
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": api_key, "currencies": "BTC", "kind": "news", "public": "true"}

    # 응답의 "next" 주소를 따라 마지막 페이지까지
    while url:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        for post in data.get("results", []):
            rows.append([
                "cryptopanic",
                str(post["id"]),
                post.get("published_at"),
                clean_text(post.get("title", "")),
                "",  # CryptoPanic은 본문을 주지 않음
                post.get("url", ""),
            ])

        url = data.get("next")
        params = None  # "next" 주소에 검색 조건이 이미 들어 있음
        if url:
            time.sleep(1)  # 페이지 사이 1초 쉬기 (요청 한도 보호)

    print(f"  [cryptopanic] {len(rows)} articles")
    return rows


# ---------------------------------------------------------------------------
# 소스 2: NewsAPI (최신 100건)
# ---------------------------------------------------------------------------
def fetch_newsapi(api_key):
    if not api_key:
        print("  [newsapi] skipped: NEWSAPI_KEY not set")
        return []

    params = {
        "q": "bitcoin OR BTC",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
    response.raise_for_status()

    rows = []
    for article in response.json().get("articles", []):
        rows.append([
            "newsapi",
            article.get("url", ""),  # NewsAPI에는 고유 번호가 없어서 URL을 id로 사용
            article.get("publishedAt"),
            clean_text(article.get("title")),
            clean_text(article.get("description")),
            article.get("url", ""),
        ])

    print(f"  [newsapi] {len(rows)} articles")
    return rows


# ---------------------------------------------------------------------------
# 소스 3: RSS 피드
# ---------------------------------------------------------------------------
def fetch_rss():
    rows = []
    for feed_name, feed_url in RSS_FEEDS.items():
        feed = feedparser.parse(feed_url)

        bitcoin_entry_count = 0
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not is_bitcoin_related(f"{title} {summary}"):
                continue

            published_at = entry.get("published") or entry.get("updated") or ""
            rows.append([
                f"rss:{feed_name}",
                entry.get("id", entry.get("link", "")),
                published_at,
                clean_text(title),
                clean_text(summary),
                entry.get("link", ""),
            ])
            bitcoin_entry_count += 1

        print(f"  [rss:{feed_name}] {bitcoin_entry_count} bitcoin-related entries")
    return rows


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    already_saved_ids = load_seen_ids(OUTPUT_PATH, "id")

    # --- 1. 세 소스에서 모두 받기 ---
    all_rows = []
    all_rows += fetch_cryptopanic(os.environ.get("CRYPTOPANIC_API_KEY"))
    all_rows += fetch_newsapi(os.environ.get("NEWSAPI_KEY"))
    all_rows += fetch_rss()

    # --- 2. 이미 저장된 기사 제외 후 덧붙이기 (row[1] = id) ---
    new_rows = [row for row in all_rows if row[1] not in already_saved_ids]
    if new_rows:
        append_rows_csv(OUTPUT_PATH, COLUMNS, new_rows)

    print(f"Collected {len(new_rows)} new articles -> {OUTPUT_PATH} at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
