"""
BTC/USD 1분봉 10년치 수집기 (Bitstamp 공개 REST API)
=================================================

목적
    비트코인 장기 통계(분포·변동성·계절성)를 보기 위한 10년치 1분봉을 모은다.

데이터 소스
    Bitstamp  GET /api/v2/ohlc/btcusd/?step=60&limit=1000&start=<unix초>
    - Binance BTCUSDT는 2017-08 상장이라 10년을 채울 수 없어 Bitstamp를 사용
    - Bitstamp는 거래가 없던 분도 volume=0 캔들로 채워서 돌려준다
    - API 키 불필요 (공개 시장 데이터)

출력
    data/raw/bitstamp_btcusd_1m_{YYYY}.parquet   연도별 1개 파일, zstd 압축
    컬럼: time(UTC, 캔들 시작 시각), open, high, low, close, volume(BTC)

이어받기(resume)
    연도 파일이 이미 있으면 그 해는 건너뛴다 → 중간에 끊겨도 다시 실행하면 이어서 수집

실행
    python collectors/collect_bitstamp_1m.py [--start 2016-10-01] [--end 2026-10-01] [--workers 4]
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
CANDLE_SECONDS = 60  # 1분봉
CANDLES_PER_REQUEST = 1000  # Bitstamp 1회 요청 최대 캔들 수

# Bitstamp 요청 한도는 10분에 8,000회(≈초당 13회). 여유 있게 초당 8회로 제한한다.
MAX_REQUESTS_PER_SECOND = 8.0
MAX_RETRIES = 8  # 한 페이지당 최대 재시도 횟수
MAX_RETRY_WAIT_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# 요청 속도 제한: 여러 스레드가 동시에 요청해도 전체 속도가 한도를 넘지 않게 한다
# ---------------------------------------------------------------------------
_rate_limit_lock = threading.Lock()
_last_request_time = 0.0


def wait_for_rate_limit() -> None:
    """직전 요청 후 최소 간격(1/초당 요청 수)이 지날 때까지 기다린다."""
    global _last_request_time
    minimum_interval = 1.0 / MAX_REQUESTS_PER_SECOND

    with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        _last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# 수집
# ---------------------------------------------------------------------------
def fetch_one_page(session: requests.Session, start_unix: int) -> list[dict]:
    """
    start_unix(초)부터 최대 1,000개의 1분봉을 가져온다.
    실패하면 1, 2, 4, 8 … 초(최대 60초) 기다렸다가 다시 시도한다 (지수 백오프).
    """
    params = {"step": CANDLE_SECONDS, "limit": CANDLES_PER_REQUEST, "start": start_unix}

    for attempt in range(MAX_RETRIES):
        wait_for_rate_limit()
        try:
            response = session.get(BITSTAMP_OHLC_URL, params=params, timeout=20)

            # 429/418 = 요청이 너무 많음, 5xx = 서버 오류 → 재시도 대상
            is_retryable_status = response.status_code in (429, 418) or response.status_code >= 500
            if is_retryable_status:
                raise requests.HTTPError(f"HTTP {response.status_code}")

            response.raise_for_status()
            return response.json()["data"]["ohlc"]

        except (requests.RequestException, ValueError, KeyError) as error:
            wait_seconds = min(2**attempt, MAX_RETRY_WAIT_SECONDS)
            print(f"  [retry] start={start_unix} {error} → {wait_seconds}s 대기", flush=True)
            time.sleep(wait_seconds)

    raise RuntimeError(f"start={start_unix} 페이지 수집 실패")


def collect_range(start_unix: int, end_unix: int, worker_count: int) -> pd.DataFrame:
    """
    [start_unix, end_unix) 구간을 1,000분 단위 페이지로 나눠 여러 스레드로 동시에 수집한다.
    결과는 시간순으로 정렬된 DataFrame (time, open, high, low, close, volume).
    """
    # --- 1. 페이지 시작 시각 목록: start, start+1000분, start+2000분, ... ---
    seconds_per_page = CANDLE_SECONDS * CANDLES_PER_REQUEST
    page_start_times = list(range(start_unix, end_unix, seconds_per_page))

    # --- 2. 병렬 수집 (각 페이지는 캔들 dict의 리스트) ---
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pages = list(executor.map(lambda page_start: fetch_one_page(session, page_start), page_start_times))

    all_candles = []
    for page in pages:
        all_candles.extend(page)
    candles = pd.DataFrame(all_candles)

    # --- 3. 정리: 구간 밖·중복 제거, 숫자형 변환, 시각을 UTC datetime으로 ---
    candles["timestamp"] = candles["timestamp"].astype("int64")
    is_inside_range = (candles["timestamp"] >= start_unix) & (candles["timestamp"] < end_unix)
    candles = candles[is_inside_range]
    candles = candles.drop_duplicates("timestamp").sort_values("timestamp")

    for column in ["open", "high", "low", "close", "volume"]:
        candles[column] = candles[column].astype("float64")

    candle_start_time = pd.to_datetime(candles.pop("timestamp"), unit="s", utc=True)
    candles.insert(0, "time", candle_start_time)
    return candles.reset_index(drop=True)


def save_parquet_atomically(candles: pd.DataFrame, output_path: Path) -> None:
    """
    임시 파일에 먼저 쓴 뒤 이름을 바꾼다.
    쓰는 도중 중단돼도 불완전한 파일이 최종 이름으로 남지 않는다 (이어받기 판단이 안전해짐).
    """
    temporary_path = output_path.with_suffix(".parquet.tmp")
    candles.to_parquet(temporary_path, compression="zstd", index=False)
    temporary_path.rename(output_path)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-10-01", help="수집 시작일 (UTC, 포함)")
    parser.add_argument("--end", default="2026-10-01", help="수집 종료일 (UTC, 미포함)")
    parser.add_argument("--workers", type=int, default=4, help="동시 요청 스레드 수")
    args = parser.parse_args()

    collection_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    collection_end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 연도 단위로 나눠 수집·저장 (연도별 파일 1개)
    for year in range(collection_start.year, collection_end.year + 1):
        # --- 1. 이 해의 수집 구간 = [전체 구간] ∩ [그해 1/1 ~ 다음해 1/1) ---
        year_start = max(collection_start, datetime(year, 1, 1, tzinfo=timezone.utc))
        year_end = min(collection_end, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
        if year_start >= year_end:
            continue

        # --- 2. 이미 수집한 해는 건너뛰기 ---
        output_path = RAW_DATA_DIR / f"bitstamp_btcusd_1m_{year}.parquet"
        if output_path.exists():
            print(f"{year}: 이미 존재 → skip ({output_path.name})")
            continue

        # --- 3. 수집 & 저장 ---
        started_at = time.time()
        candles = collect_range(int(year_start.timestamp()), int(year_end.timestamp()), args.workers)
        save_parquet_atomically(candles, output_path)

        # --- 4. 누락 점검: 기대 캔들 수 = 구간 길이(초) / 60 ---
        expected_count = int((year_end - year_start).total_seconds() // CANDLE_SECONDS)
        missing_count = expected_count - len(candles)
        elapsed_seconds = time.time() - started_at
        print(
            f"{year}: {len(candles):,} rows / 기대 {expected_count:,} "
            f"(누락 {missing_count:,}) — {elapsed_seconds:.0f}s → {output_path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
