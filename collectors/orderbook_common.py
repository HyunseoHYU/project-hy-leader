"""
실시간 호가창 수집기 공통 부품
============================

collect_orderbook_ws.py(수집기), watchdog_orderbook.py(감시), verify_orderbook_coverage.py(점검)가
같이 쓰는 저장 경로 규칙, Parquet 버퍼 저장, 재연결 대기시간 계산, 생존신호(heartbeat) 파일 처리.

왜 common.py의 CSV 방식과 따로 두는가
    뉴스·Reddit 수집기는 몇 분에 한 번 소량을 CSV에 덧붙이면 충분하다.
    호가창은 초당 수십 건이 들어오므로, 메모리에 모았다가 일정 주기마다
    작은 Parquet 파일로 한꺼번에 쓰는 편이 빠르고 안전하다.

저장 경로 규칙
    data/orderbook/<스트림>/host=<수집 호스트>/date=YYYY-MM-DD/hour=HH/<스트림>_<시각>_<난수>.parquet
    - host 폴더로 VPS·로컬 두 수집기의 결과가 섞이지 않는다
    - 날짜·시간 폴더로 원하는 구간만 골라 읽을 수 있다
"""

import json
import os
import random
import time
from datetime import datetime, timezone

import pandas as pd

from common import DATA_DIR

ORDERBOOK_DIR = os.path.join(DATA_DIR, "orderbook")

# depth20 스트림은 20단계를 주지만, 저장 용량을 고려해 상위 10단계만 저장한다
# (스모크테스트 기준 시간당 약 13MB → 20단계면 약 2배)
DEPTH_LEVELS = 10


def partition_path(stream: str, source_host: str, event_dt: datetime) -> str:
    """스트림·호스트·날짜·시간에 해당하는 저장 폴더 경로를 만든다."""
    return os.path.join(
        ORDERBOOK_DIR,
        stream,
        f"host={source_host}",
        f"date={event_dt.strftime('%Y-%m-%d')}",
        f"hour={event_dt.strftime('%H')}",
    )


def flatten_depth_levels(bids, asks, levels: int = DEPTH_LEVELS) -> dict:
    """
    Binance 호가 배열을 고정된 컬럼들로 펼친다.

    입력:  bids = [["가격", "수량"], ...]  (최우선 호가부터 순서대로, 문자열)
    출력:  {"bid_px_1": 가격, "bid_qty_1": 수량, ..., "ask_px_10": ..., "ask_qty_10": ...}

    호가가 얇아 단계 수가 모자라면 남는 칸은 None 으로 채운다 (오류 대신).
    """
    row = {}
    for side_name, side_levels in (("bid", bids), ("ask", asks)):
        for level_index in range(levels):
            level_number = level_index + 1  # 컬럼 이름은 1부터

            has_this_level = level_index < len(side_levels)
            if has_this_level:
                price_text, quantity_text = side_levels[level_index][0], side_levels[level_index][1]
                row[f"{side_name}_px_{level_number}"] = float(price_text)
                row[f"{side_name}_qty_{level_number}"] = float(quantity_text)
            else:
                row[f"{side_name}_px_{level_number}"] = None
                row[f"{side_name}_qty_{level_number}"] = None
    return row


class BufferedParquetWriter:
    """
    스트림 하나의 행들을 메모리에 모아 두었다가, flush() 때 새 Parquet 파일 하나로 쓴다.

    매번 '새 파일'로 쓰는 이유
        - 기존 파일을 고치지 않으므로, 쓰는 도중 프로세스가 죽어도 이미 저장된 데이터는 안전
        - 재시작 후 파일 잠금 같은 것을 신경 쓸 필요가 없음
    """

    def __init__(self, stream: str, source_host: str, columns: list[str]):
        self.stream = stream
        self.source_host = source_host
        self.columns = columns
        self._buffered_rows: list[dict] = []

    def add(self, row: dict) -> None:
        self._buffered_rows.append(row)

    def __len__(self) -> int:
        return len(self._buffered_rows)

    def flush(self) -> str | None:
        """모아 둔 행을 Parquet 파일 하나로 쓰고 버퍼를 비운다. 쓴 파일 경로를 반환 (비어 있으면 None)."""
        if not self._buffered_rows:
            return None

        # --- 1. 버퍼를 꺼내고 즉시 새 빈 버퍼로 교체 ---
        rows_to_write = self._buffered_rows
        self._buffered_rows = []
        table = pd.DataFrame(rows_to_write, columns=self.columns)

        # --- 2. 저장 폴더: flush 시점의 날짜·시간 기준 ---
        # (flush 주기가 20초라 한 파일의 행들은 시간 경계를 거의 넘지 않는다)
        now = datetime.now(timezone.utc)
        output_dir = partition_path(self.stream, self.source_host, now)
        os.makedirs(output_dir, exist_ok=True)

        # --- 3. 파일 이름: 스트림_시각(마이크로초)_4자리난수 → 같은 초에 두 번 써도 충돌 없음 ---
        timestamp_text = now.strftime("%Y%m%dT%H%M%S%f")
        random_suffix = f"{random.randint(0, 9999):04d}"
        output_path = os.path.join(output_dir, f"{self.stream}_{timestamp_text}_{random_suffix}.parquet")

        # --- 4. 임시 파일에 쓴 뒤 이름 바꾸기 (같은 디스크 안에서는 원자적 → 반쯤 쓴 파일이 안 보임) ---
        temporary_path = output_path + ".tmp"
        table.to_parquet(temporary_path, engine="pyarrow", index=False)
        os.replace(temporary_path, output_path)
        return output_path


class ReconnectBackoff:
    """
    재연결 대기시간 계산기 (지수 백오프).

        대기시간 = min(상한, 기본값 × 2^연속실패횟수)  →  1, 2, 4, 8, ... 최대 60초

    연결이 healthy_after 초 이상 안정적으로 유지됐다면, 한 번 끊긴 것은 일시적 문제로 보고
    실패 횟수를 0으로 되돌린다 (오래 잘 돌던 연결이 한 번 끊겼다고 오래 기다리지 않게).
    """

    def __init__(self, base: float = 1.0, cap: float = 60.0, healthy_after: float = 120.0):
        self.base = base  # 첫 대기시간 (초)
        self.cap = cap  # 최대 대기시간 (초)
        self.healthy_after = healthy_after  # 이만큼 유지되면 '안정적 연결'로 인정 (초)
        self._consecutive_failures = 0
        self._connected_at: float | None = None

    def on_connect(self) -> None:
        self._connected_at = time.monotonic()

    def on_disconnect(self) -> float:
        """연결이 끊겼을 때 호출. 재연결 전에 기다릴 시간(초)을 돌려준다."""
        was_connected = self._connected_at is not None
        if was_connected:
            connection_lifetime = time.monotonic() - self._connected_at
            if connection_lifetime >= self.healthy_after:
                self._consecutive_failures = 0

        self._connected_at = None
        delay = min(self.cap, self.base * (2**self._consecutive_failures))
        self._consecutive_failures += 1
        return delay


def write_heartbeat(path: str, state: dict) -> None:
    """
    생존신호(heartbeat) 파일을 쓴다. 감시 프로세스(watchdog)가 이 파일의 갱신 시각과
    메시지 수를 보고 수집기가 살아 있는지 판단한다.
    임시 파일 → 이름 바꾸기 방식이라 watchdog이 반쯤 쓴 파일을 읽는 일이 없다.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {**state, "written_at": datetime.now(timezone.utc).isoformat()}

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as heartbeat_file:
        json.dump(payload, heartbeat_file)
    os.replace(temporary_path, path)


def read_heartbeat(path: str) -> dict | None:
    """생존신호 파일을 읽는다. 없거나 깨져 있으면 None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as heartbeat_file:
            return json.load(heartbeat_file)
    except (json.JSONDecodeError, OSError):
        return None
