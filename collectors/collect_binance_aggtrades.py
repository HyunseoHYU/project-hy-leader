"""
Binance 현물 BTCUSDT aggTrades(체결 기록) 과거 데이터 수집기
=========================================================

목적
    호가창 원본은 과거분을 구할 수 없지만, 체결(aggTrades)은 Binance 공개 저장소에
    일 단위로 올라와 있다. 이를 받아 주문흐름 불균형(OFI)·체결 기반 Kyle's λ 등
    체결 기반 유동성 지표의 장기 표본으로 쓴다.

데이터 소스
    https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{YYYY-MM-DD}.zip
    - API 키 불필요, 하루 1개 zip(CSV, 머리글 없음)
    - 타임스탬프 단위가 바뀌었다: 2025-01-01 이후 파일은 마이크로초(µs), 이전은 밀리초(ms)
      → 값의 자릿수로 판별해 모두 UTC datetime으로 통일한다

출력
    data/aggtrades/binance_btcusdt_aggtrades_{YYYY-MM-DD}.parquet   하루 1개, zstd 압축 (git 제외)
    컬럼: agg_trade_id, price(USDT), quantity(BTC), first_trade_id, last_trade_id,
          trade_time(UTC), is_buyer_maker, is_best_match
    - is_buyer_maker=True  → 매수자가 호가에 걸어둔 쪽(메이커), 즉 시장가 **매도**가 체결을 일으킴
    - is_buyer_maker=False → 시장가 **매수**가 체결을 일으킴

이어받기(resume)
    날짜 파일이 이미 있으면 건너뛴다 → 끊겨도 다시 실행하면 이어서 받는다

실행
    python collectors/collect_binance_aggtrades.py [--start 2023-10-11] [--end 2026-10-10] [--workers 4]
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "aggtrades"

DEFAULT_START = "2023-10-11"   # 최근 3년의 시작일
DEFAULT_END = "2026-10-10"     # 공개 저장소의 최신 완료일 (오늘 파일은 다음날 올라온다)
DEFAULT_WORKERS = 4            # 동시 다운로드 수 (공개 저장소에 부담 주지 않도록 적게)
MAX_RETRIES = 4                # 일시적 네트워크 오류 재시도 횟수
REQUEST_TIMEOUT_SEC = 120
MICROSECOND_THRESHOLD = 10**14  # 이보다 큰 타임스탬프는 µs 단위 (ms 값은 2026년에도 ~1.8e12)

CSV_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
]


def output_path_for(day: date) -> Path:
    return OUTPUT_DIR / f"binance_btcusdt_aggtrades_{day.isoformat()}.parquet"


def download_zip(day: date) -> bytes | None:
    """하루치 zip을 내려받는다. 파일이 없으면(404) None, 그 외 실패는 재시도 후 예외."""
    url = f"{BASE_URL}/BTCUSDT-aggTrades-{day.isoformat()}.zip"
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * attempt)  # 점점 길게 쉬었다가 재시도
    raise RuntimeError(f"{day} 다운로드 실패: {last_error}")


def parse_trades(zip_bytes: bytes) -> pd.DataFrame:
    """zip 안의 CSV를 읽어 컬럼 이름·시간 단위를 통일한 DataFrame으로 만든다."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as csv_file:
            trades = pd.read_csv(csv_file, header=None, names=CSV_COLUMNS)

    # --- 타임스탬프: 자릿수로 ms / µs 구분 후 UTC로 변환 ---
    is_microsecond = trades["timestamp"] > MICROSECOND_THRESHOLD
    timestamp_unit = "us" if is_microsecond.any() else "ms"
    trades["trade_time"] = pd.to_datetime(trades["timestamp"], unit=timestamp_unit, utc=True)
    trades = trades.drop(columns=["timestamp"])

    # --- 용량을 줄이기 위해 가격·수량은 float64, 불리언은 bool로 ---
    trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)
    trades["is_best_match"] = trades["is_best_match"].astype(bool)
    return trades


def fetch_one_day(day: date) -> str:
    """하루치를 받아 Parquet으로 저장한다. 결과 상태 문자열을 돌려준다."""
    output_path = output_path_for(day)
    if output_path.exists():
        return "skip"

    zip_bytes = download_zip(day)
    if zip_bytes is None:
        return "missing"

    trades = parse_trades(zip_bytes)
    temp_path = output_path.with_suffix(".parquet.tmp")  # 쓰다 끊겨도 완성본으로 오인되지 않게
    trades.to_parquet(temp_path, compression="zstd", index=False)
    temp_path.rename(output_path)
    return f"ok:{len(trades)}"


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Binance BTCUSDT aggTrades 과거 데이터 수집")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    start_day = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_day = datetime.strptime(args.end, "%Y-%m-%d").date()
    days = list(date_range(start_day, end_day))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{start_day} ~ {end_day}: {len(days)}일, workers={args.workers}", flush=True)

    status_counts = {"ok": 0, "skip": 0, "missing": 0, "failed": 0}
    failed_days = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one_day, day): day for day in days}
        for done_count, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                status = future.result().split(":")[0]
            except Exception as error:  # 한 날짜 실패가 전체를 멈추지 않게
                status = "failed"
                failed_days.append(day)
                print(f"  ! {day}: {error}", flush=True)
            status_counts[status] += 1
            if done_count % 50 == 0 or done_count == len(days):
                print(f"  {done_count}/{len(days)} {status_counts}", flush=True)

    print(f"완료: {status_counts}", flush=True)
    if failed_days:
        print("실패한 날짜(다시 실행하면 이어받음):", sorted(failed_days), flush=True)


if __name__ == "__main__":
    main()
