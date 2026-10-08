"""
Binance BTCUSDT 1분봉 전체 기간 수집기 (REST API)
==============================================

목적
    Binance 상장(2017-08-17) 이후 전체 1분봉을 모은다.
    Bitstamp와 달리 '시장가 매수 거래량(taker buy volume)'과 체결 건수가 있어
    주문흐름 불균형(OFI)·Kyle's λ 같은 미시구조 지표를 근사할 수 있다.

데이터 소스
    Binance  GET /api/v3/klines   (API 키 불필요, 1회 최대 1,000개)

출력
    data/binance/btcusdt_1m_max.csv   (856MB, git 제외 → src/convert_binance_1m_parquet.py 로 연도별 Parquet 변환)

이어받기
    출력 CSV가 이미 있으면 마지막 행 다음 분부터 이어서 받는다.

실행
    python collectors/collect_binance_1m.py
"""

import csv
import os
import time
from datetime import datetime, timezone

import requests

from common import DATA_DIR

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
CANDLES_PER_REQUEST = 1000  # Binance 1회 요청 최대 캔들 수
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
START_DT = datetime(2016, 1, 1, tzinfo=timezone.utc)  # 하한. 실제 데이터는 상장일(2017-08-17)부터 옴
OUTPUT_PATH = os.path.join(DATA_DIR, "binance", "btcusdt_1m_max.csv")

MAX_RETRIES = 5
PAUSE_BETWEEN_REQUESTS_SEC = 0.05  # Binance 요청 한도는 넉넉하지만 예의상 짧게 쉰다
PROGRESS_EVERY_N_REQUESTS = 100

# CSV 컬럼 = Binance가 돌려주는 캔들 배열의 필드 순서 (마지막 '무시' 필드 제외)
COLUMNS = [
    "open_time",  # 캔들 시작 시각
    "open",
    "high",
    "low",
    "close",
    "volume",  # 거래량 (BTC)
    "close_time",  # 캔들 종료 시각 (시작 + 59.999초)
    "quote_asset_volume",  # 거래대금 (USDT)
    "num_trades",  # 체결 건수
    "taker_buy_base_volume",  # 시장가 매수 거래량 (BTC) → 매도 거래량 = volume − 이것
    "taker_buy_quote_volume",  # 시장가 매수 거래대금 (USDT)
]


def ms_to_iso(milliseconds: int) -> str:
    """밀리초 타임스탬프 → ISO 문자열 (UTC)."""
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def kline_to_csv_row(kline: list) -> list:
    """
    Binance 캔들 배열 하나를 CSV 행으로 바꾼다 (시각만 읽기 쉬운 ISO 문자열로 변환).
    배열 순서: [시작시각, 시, 고, 저, 종, 거래량, 종료시각, 거래대금, 체결수, 매수거래량, 매수거래대금, 무시]
    """
    return [
        ms_to_iso(kline[0]),
        kline[1], kline[2], kline[3], kline[4], kline[5],
        ms_to_iso(kline[6]),
        kline[7], kline[8], kline[9], kline[10],
    ]


def fetch_klines(symbol, interval, start_ms, end_ms, limit=CANDLES_PER_REQUEST):
    """
    [start_ms, end_ms] 구간의 캔들을 최대 limit개 가져온다.
    - 429/418(요청 과다): 서버가 알려준 Retry-After 초만큼 기다렸다가 재시도
    - 네트워크 오류: 1, 2, 4, 8, 16초 기다리며 재시도
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)

            is_rate_limited = response.status_code in (429, 418)
            if is_rate_limited:
                wait_seconds = int(response.headers.get("Retry-After", 5))
                print(f"Rate limited, waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            wait_seconds = 2**attempt
            print(f"Request error: {error}, retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError("Failed to fetch klines after retries")


def resume_start_ms(path):
    """
    기존 CSV의 마지막 행을 읽어 '그 다음 캔들의 시작 시각(ms)'을 돌려준다. 이어받을 게 없으면 None.
    856MB 파일 전체를 읽지 않도록 끝에서 4KB만 읽는다.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    with open(path, "rb") as csv_file:
        csv_file.seek(0, os.SEEK_END)
        csv_file.seek(max(0, csv_file.tell() - 4096))
        last_lines = csv_file.read().decode().strip().splitlines()

    only_header_exists = len(last_lines) < 2 or last_lines[-1].startswith("open_time")
    if only_header_exists:
        return None

    last_close_time_iso = last_lines[-1].split(",")[6]  # 7번째 컬럼 = close_time
    last_close_time_ms = int(datetime.fromisoformat(last_close_time_iso).timestamp() * 1000)
    return last_close_time_ms + 1


def main():
    # --- 1. 수집 구간: START_DT(또는 이어받기 지점) ~ 지금 ---
    end_dt = datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(START_DT.timestamp() * 1000)

    resume_ms = resume_start_ms(OUTPUT_PATH)
    if resume_ms:
        start_ms = max(start_ms, resume_ms)
        print(f"Resuming from {ms_to_iso(start_ms)}")

    print(f"Fetching {SYMBOL} {INTERVAL} klines from {ms_to_iso(start_ms)} to {end_dt}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    is_new_file = not resume_ms
    next_start_ms = start_ms
    request_count = 0
    total_rows = 0

    # --- 2. 1,000개씩 페이지를 넘기며 받아서 바로바로 파일에 덧붙인다 (메모리 절약) ---
    with open(OUTPUT_PATH, "w" if is_new_file else "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if is_new_file:
            writer.writerow(COLUMNS)

        while next_start_ms < end_ms:
            klines = fetch_klines(SYMBOL, INTERVAL, next_start_ms, end_ms)
            request_count += 1
            if not klines:
                break

            # 아직 끝나지 않은(진행 중인) 캔들은 버린다 → 완성된 캔들만 저장
            closed_klines = [kline for kline in klines if kline[6] < end_ms]
            if not closed_klines:
                break

            rows = [kline_to_csv_row(kline) for kline in closed_klines]
            writer.writerows(rows)
            total_rows += len(rows)

            # 다음 페이지 = 마지막 캔들 종료 시각 + 1ms
            next_start_ms = closed_klines[-1][6] + 1

            if request_count % PROGRESS_EVERY_N_REQUESTS == 0:
                csv_file.flush()  # 중간에 끊겨도 여기까지는 디스크에 남도록
                print(f"  {total_rows} rows so far... (up to {rows[-1][0]})")

            time.sleep(PAUSE_BETWEEN_REQUESTS_SEC)

    print(f"Total rows fetched: {total_rows} in {request_count} requests")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
