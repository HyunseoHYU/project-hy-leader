"""
호가창 수집 커버리지 점검 도구
============================

목적
    저장된 호가창 Parquet 파일을 훑어서 '빠진 시간'이 있는지 보고한다.
    - 스모크테스트 직후 확인용
    - 장기 수집 중 주 1회 건강검진용 (docs/deployment.md)

보고 내용 (스트림 × 호스트별)
    - 시간(hour)별 행 수와 중앙값
    - 하드 갭(hard gap): 처음~마지막 시간 사이에 폴더 자체가 없는 시간 → 완전히 끊긴 구간
    - 소프트 갭(soft gap): 행 수가 중앙값의 20% 미만인 시간 → 재연결 등으로 일부만 수집된 구간
    - depth20: 0.1초마다 오므로 시간당 약 36,000행이 정상. 절반 미만이면 따로 표시

실행
    python collectors/verify_orderbook_coverage.py [--stream bookticker] [--host local]
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import pyarrow.parquet as pq

from orderbook_common import ORDERBOOK_DIR

# 파일 경로에서 스트림·호스트·날짜·시간을 뽑아내는 정규식
#   예: .../orderbook/depth20/host=vps/date=2026-10-08/hour=13/xxx.parquet
PARTITION_PATTERN = re.compile(
    r"orderbook/(?P<stream>[^/]+)/host=(?P<host>[^/]+)/date=(?P<date>[^/]+)/hour=(?P<hour>\d{2})/"
)

# depth20@100ms = 초당 10회 × 3,600초 → 심볼 하나당 시간당 약 36,000행 (참고용 기준)
DEPTH20_EXPECTED_ROWS_PER_HOUR_PER_SYMBOL = 10 * 3600

SOFT_GAP_RATIO = 0.2  # 중앙값의 20% 미만이면 소프트 갭
DEPTH20_LOW_RATIO = 0.5  # depth20 기대치의 50% 미만이면 경고


def scan_partitions(stream_filter: str | None, host_filter: str | None) -> dict:
    """
    모든 Parquet 파일의 행 수를 (스트림, 호스트, 날짜, 시간) 단위로 합산한다.
    반환: {(stream, host, date, hour): {"rows": 행 수, "files": 파일 수}}
    파일 내용 전체를 읽지 않고 메타데이터(행 수)만 읽어서 빠르다.
    """
    file_pattern = os.path.join(ORDERBOOK_DIR, "*", "host=*", "date=*", "hour=*", "*.parquet")
    counts: dict = defaultdict(lambda: {"rows": 0, "files": 0})

    for path in glob.glob(file_pattern):
        match = PARTITION_PATTERN.search(path.replace(os.sep, "/"))
        if not match:
            continue

        stream, host, date, hour = match["stream"], match["host"], match["date"], match["hour"]
        if stream_filter and stream != stream_filter:
            continue
        if host_filter and host != host_filter:
            continue

        try:
            row_count = pq.ParquetFile(path).metadata.num_rows
        except Exception as error:
            print(f"  WARN: could not read {path}: {error}")
            continue

        partition_key = (stream, host, date, hour)
        counts[partition_key]["rows"] += row_count
        counts[partition_key]["files"] += 1

    return counts


def every_hour_between(first_hour: datetime, last_hour: datetime) -> list[tuple[str, str]]:
    """first_hour ~ last_hour 사이의 모든 (날짜, 시) 쌍. 빠진 시간을 찾는 기준 목록."""
    all_hours = []
    current_hour = first_hour
    while current_hour <= last_hour:
        all_hours.append((current_hour.strftime("%Y-%m-%d"), current_hour.strftime("%H")))
        current_hour += timedelta(hours=1)
    return all_hours


def report_one_stream(stream: str, host: str, hourly_rows: list[tuple[str, str, int, int]]) -> None:
    """스트림 하나 × 호스트 하나의 커버리지 보고. hourly_rows = [(날짜, 시, 행 수, 파일 수), ...]"""
    hourly_rows.sort()

    # --- 1. 하드 갭: 범위 안의 모든 시간 중 데이터가 없는 시간 ---
    present_hours = {(date, hour) for date, hour, _, _ in hourly_rows}
    first_hour = datetime.strptime(f"{hourly_rows[0][0]} {hourly_rows[0][1]}", "%Y-%m-%d %H")
    last_hour = datetime.strptime(f"{hourly_rows[-1][0]} {hourly_rows[-1][1]}", "%Y-%m-%d %H")
    expected_hours = every_hour_between(first_hour, last_hour)
    missing_hours = [date_hour for date_hour in expected_hours if date_hour not in present_hours]

    # --- 2. 소프트 갭: 행 수가 중앙값보다 훨씬 적은 시간 ---
    row_counts = sorted(row_count for _, _, row_count, _ in hourly_rows)
    median_rows = row_counts[len(row_counts) // 2]
    soft_gap_threshold = max(1, median_rows * SOFT_GAP_RATIO)
    soft_gaps = [(date, hour, rows) for date, hour, rows, _ in hourly_rows if rows < soft_gap_threshold]

    # --- 3. 출력 ---
    print(f"\n=== {stream} / host={host} ===")
    print(f"  hours present: {len(hourly_rows)}  span: {expected_hours[0]} .. {expected_hours[-1]}")
    print(f"  median rows/hour: {median_rows}")

    if missing_hours:
        print(f"  HARD GAPS (missing hours, {len(missing_hours)}):")
        for date, hour in missing_hours:
            print(f"    {date} {hour}:00")
    else:
        print("  no hard gaps (no fully-missing hours in range)")

    if soft_gaps:
        print(f"  SOFT GAPS (rows far below median, {len(soft_gaps)}):")
        for date, hour, rows in soft_gaps:
            print(f"    {date} {hour}:00 -> {rows} rows (median {median_rows})")

    # --- 4. depth20 전용: 0.1초 주기 기대치 대비 ---
    if stream == "depth20":
        expected_rows = DEPTH20_EXPECTED_ROWS_PER_HOUR_PER_SYMBOL  # 심볼이 여러 개면 직접 나눠서 볼 것
        low_hours = [(date, hour, rows) for date, hour, rows, _ in hourly_rows if rows < expected_rows * DEPTH20_LOW_RATIO]
        if low_hours:
            print(f"  depth20 rows well below the ~{expected_rows}/hr/symbol expected cadence, {len(low_hours)} hour(s):")
            for date, hour, rows in low_hours:
                print(f"    {date} {hour}:00 -> {rows} rows")


def report(counts: dict) -> None:
    # (스트림, 호스트)별로 묶기
    rows_by_stream_host = defaultdict(list)
    for (stream, host, date, hour), totals in counts.items():
        rows_by_stream_host[(stream, host)].append((date, hour, totals["rows"], totals["files"]))

    if not rows_by_stream_host:
        print("No order-book Parquet files found under", ORDERBOOK_DIR)
        return

    for (stream, host), hourly_rows in sorted(rows_by_stream_host.items()):
        report_one_stream(stream, host, hourly_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", choices=["bookticker", "depth20", "trades"], default=None)
    parser.add_argument("--host", default=None)
    args = parser.parse_args()

    counts = scan_partitions(args.stream, args.host)
    report(counts)


if __name__ == "__main__":
    main()
