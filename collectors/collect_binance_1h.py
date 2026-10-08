"""
Binance BTCUSDT 1시간봉 수집기 (2026-01-01 ~ 현재)
================================================

collect_binance_1m.py 의 요청·재시도·행 변환 함수를 그대로 쓰고, 주기와 기간만 바꾼다.
완성된(마감된) 시간봉만 저장한다.

출력
    data/binance/btcusdt_1h_2026jan_sep.csv

실행
    python collectors/collect_binance_1h.py
"""

import csv
import os
import time
from datetime import datetime, timezone

from collect_binance_1m import COLUMNS, PAUSE_BETWEEN_REQUESTS_SEC, SYMBOL, fetch_klines, kline_to_csv_row
from common import DATA_DIR

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
INTERVAL = "1h"
START_DT = datetime(2026, 1, 1, tzinfo=timezone.utc)
OUTPUT_PATH = os.path.join(DATA_DIR, "binance", "btcusdt_1h_2026jan_sep.csv")


def main():
    # --- 1. 수집 구간: 1월 1일 ~ 마지막으로 마감된 정시 직전 ---
    current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_ms = int(START_DT.timestamp() * 1000)
    end_ms = int(current_hour.timestamp() * 1000) - 1  # 진행 중인 시간봉 제외

    print(f"Fetching {SYMBOL} {INTERVAL} klines from {START_DT} to {current_hour}")

    # --- 2. 페이지 단위로 받기 (9개월 ≈ 6,500개라 메모리에 모아도 충분) ---
    all_rows = []
    next_start_ms = start_ms
    while next_start_ms < end_ms:
        klines = fetch_klines(SYMBOL, INTERVAL, next_start_ms, end_ms)
        if not klines:
            break

        all_rows.extend(kline_to_csv_row(kline) for kline in klines)
        next_start_ms = klines[-1][6] + 1  # 다음 페이지 = 마지막 캔들 종료 시각 + 1ms
        time.sleep(PAUSE_BETWEEN_REQUESTS_SEC)

    print(f"Total rows fetched: {len(all_rows)} (up to {all_rows[-1][0]})")

    # --- 3. 저장 (매번 전체를 새로 씀) ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(COLUMNS)
        writer.writerows(all_rows)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
