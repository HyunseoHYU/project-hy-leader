"""
X(트위터) 비트코인 게시물 수집기 (API v2 recent search)
=====================================================

주의
    recent search는 '최근 7일' 게시물만 검색된다 (표준 요금제).
    더 긴 기간을 모으려면 매일 실행해 7일 창이 계속 겹치도록 해야 한다.

필요한 환경변수 (.env)
    X_BEARER_TOKEN

출력
    data/x_bitcoin_posts.csv   (이미 저장된 id는 건너뜀)

실행
    python collectors/collect_x_posts.py
"""

import os
import sys
import time
from datetime import datetime, timezone

import requests

from common import DATA_DIR, append_rows_csv, load_seen_ids, require_env

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
# 비트코인 관련 영어 게시물, 리트윗 제외 (같은 내용 중복 방지)
QUERY = "(bitcoin OR btc OR #bitcoin OR #btc) lang:en -is:retweet"
MAX_RESULTS_PER_PAGE = 100
MAX_RETRIES = 5
MIN_RATE_LIMIT_WAIT_SEC = 5
OUTPUT_PATH = os.path.join(DATA_DIR, "x_bitcoin_posts.csv")

COLUMNS = [
    "id",
    "created_at",
    "author_id",
    "text",
    "like_count",
    "retweet_count",
    "reply_count",
    "quote_count",
    "lang",
]


def search_page(bearer_token, next_token=None):
    """
    검색 결과 한 페이지(최대 100건)를 가져온다.
    429(요청 한도 초과)면 응답 헤더의 '한도 초기화 시각'까지 기다렸다가 다시 요청.
    """
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": QUERY,
        "max_results": MAX_RESULTS_PER_PAGE,
        "tweet.fields": "created_at,author_id,public_metrics,lang",
    }
    if next_token:
        params["next_token"] = next_token  # 다음 페이지 표시

    for _ in range(MAX_RETRIES):
        response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=15)

        if response.status_code == 429:
            reset_unix_time = int(response.headers.get("x-rate-limit-reset", time.time() + 60))
            wait_seconds = max(reset_unix_time - int(time.time()), MIN_RATE_LIMIT_WAIT_SEC)
            print(f"Rate limited, waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("Failed to fetch tweets after retries")


def tweet_to_row(tweet: dict) -> list:
    metrics = tweet.get("public_metrics", {})
    return [
        tweet["id"],
        tweet.get("created_at"),
        tweet.get("author_id"),
        tweet.get("text", "").replace("\n", " ").strip(),
        metrics.get("like_count", 0),
        metrics.get("retweet_count", 0),
        metrics.get("reply_count", 0),
        metrics.get("quote_count", 0),
        tweet.get("lang"),
    ]


def main():
    bearer_token = require_env("X_BEARER_TOKEN")
    already_saved_ids = load_seen_ids(OUTPUT_PATH, "id")

    new_rows = []
    next_token = None
    page_number = 0

    # 다음 페이지 표시(next_token)가 없을 때까지 계속
    while True:
        page = search_page(bearer_token, next_token)
        page_number += 1
        tweets_on_page = page.get("data", [])

        for tweet in tweets_on_page:
            if tweet["id"] in already_saved_ids:
                continue
            new_rows.append(tweet_to_row(tweet))
            already_saved_ids.add(tweet["id"])

        print(f"  page {page_number}: {len(tweets_on_page)} tweets ({len(new_rows)} new so far)")

        next_token = page.get("meta", {}).get("next_token")
        if not next_token:
            break
        time.sleep(1)  # 페이지 사이 1초 쉬기

    if new_rows:
        append_rows_csv(OUTPUT_PATH, COLUMNS, new_rows)
    print(f"Collected {len(new_rows)} new tweets -> {OUTPUT_PATH} at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
