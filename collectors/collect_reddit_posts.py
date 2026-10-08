"""
Reddit 비트코인 게시물 수집기 (PRAW)
=================================

대상
    r/Bitcoin          비트코인 전용 → 최신 글 전부
    r/CryptoCurrency   여러 코인 섞임 → "bitcoin OR btc" 검색 결과만
    r/BitcoinMarkets   시세 토론 → 검색 결과만

필요한 환경변수 (.env, .env.example 참고)
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

출력
    data/reddit_bitcoin_posts.csv   (이미 저장된 게시물 id는 건너뜀)

실행
    python collectors/collect_reddit_posts.py
"""

import os
import sys
from datetime import datetime, timezone

import praw

from common import DATA_DIR, append_rows_csv, load_seen_ids, require_env

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
SUBREDDITS = ["Bitcoin", "CryptoCurrency", "BitcoinMarkets"]
BITCOIN_ONLY_SUBREDDIT = "Bitcoin"  # 이 서브레딧은 검색 없이 최신 글을 그대로 수집
SEARCH_QUERY = "bitcoin OR btc"
POSTS_PER_SUBREDDIT = 200  # Reddit 목록 1회 조회 한도에 맞춤
OUTPUT_PATH = os.path.join(DATA_DIR, "reddit_bitcoin_posts.csv")

COLUMNS = [
    "id",
    "created_utc",  # 작성 시각 (UTC)
    "subreddit",
    "author",
    "title",
    "selftext",  # 본문
    "score",  # 추천 − 비추천
    "num_comments",
    "url",
]


def clean_text(text) -> str:
    return (text or "").replace("\n", " ").strip()


def build_client():
    """읽기 전용 Reddit 클라이언트."""
    reddit = praw.Reddit(
        client_id=require_env("REDDIT_CLIENT_ID"),
        client_secret=require_env("REDDIT_CLIENT_SECRET"),
        user_agent=require_env("REDDIT_USER_AGENT"),
    )
    reddit.read_only = True
    return reddit


def main():
    reddit = build_client()
    already_saved_ids = load_seen_ids(OUTPUT_PATH, "id")
    new_rows = []

    for subreddit_name in SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)

        # --- 1. 게시물 목록: 비트코인 전용 서브레딧은 최신순 전부, 나머지는 키워드 검색 ---
        if subreddit_name == BITCOIN_ONLY_SUBREDDIT:
            posts = subreddit.new(limit=POSTS_PER_SUBREDDIT)
        else:
            posts = subreddit.search(SEARCH_QUERY, sort="new", limit=POSTS_PER_SUBREDDIT)

        # --- 2. 새 게시물만 행으로 변환 ---
        new_post_count = 0
        for post in posts:
            if post.id in already_saved_ids:
                continue

            new_rows.append([
                post.id,
                datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                subreddit_name,
                str(post.author) if post.author else "[deleted]",
                clean_text(post.title),
                clean_text(post.selftext),
                post.score,
                post.num_comments,
                post.url,
            ])
            already_saved_ids.add(post.id)  # 여러 서브레딧에 교차 게시된 글 중복 방지
            new_post_count += 1

        print(f"  r/{subreddit_name}: {new_post_count} new posts")

    # --- 3. 저장 ---
    if new_rows:
        append_rows_csv(OUTPUT_PATH, COLUMNS, new_rows)
    print(f"Collected {len(new_rows)} new posts -> {OUTPUT_PATH} at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
