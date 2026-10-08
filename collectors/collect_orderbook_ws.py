"""
실시간 BTC 호가창 수집기 (Binance WebSocket)
==========================================

목적
    감성 이벤트 전후의 유동성 변화를 측정하기 위한 원자료(호가·체결)를 24시간 쉬지 않고 모은다.

받는 스트림 (심볼마다 3개, 하나의 WebSocket 연결로 묶어서 수신)
    <심볼>@bookTicker      최우선 매수·매도 호가와 잔량. 최우선 호가가 바뀔 때마다 즉시 전송
    <심볼>@depth20@100ms   상위 20단계 호가 스냅샷. 0.1초마다 전송 (저장은 상위 10단계)
    <심볼>@aggTrade        체결 내역. 같은 가격·같은 주문의 체결은 하나로 묶여 옴
                            is_buyer_maker=True → 매도자가 시장가로 친 '매도 주도 체결'

    API 키 불필요 (공개 시장 데이터)

출력
    data/orderbook/<스트림>/host=<SOURCE_HOST>/date=.../hour=.../*.parquet
    data/orderbook/HEARTBEAT_<SOURCE_HOST>.json     생존신호 (watchdog이 읽음)

무중단 설계 (3중 안전장치)
    1. 이 스크립트 자체: 연결이 끊기면 대기 후 자동 재연결 (지수 백오프)
    2. systemd Restart=always: 프로세스가 죽으면 5초 후 재시작
    3. watchdog_orderbook.py: 프로세스는 살아 있는데 데이터가 안 들어오는 '조용한 정지'를 감지해 재시작
    + VPS·로컬 두 곳에서 독립적으로 실행 (docs/deployment.md)

설정 (.env)
    SOURCE_HOST         이 수집기의 이름 (vps / local). 저장 폴더 host=... 에 쓰임
    ORDERBOOK_SYMBOLS   수집할 심볼, 쉼표로 구분 (기본 btcusdt)
    FLUSH_INTERVAL_SEC  메모리 → 파일 저장 주기 (기본 20초)

실행
    cd collectors && python collect_orderbook_ws.py
"""

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import websocket

from common import DATA_DIR
from orderbook_common import (
    DEPTH_LEVELS,
    BufferedParquetWriter,
    ReconnectBackoff,
    flatten_depth_levels,
    write_heartbeat,
)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
SOURCE_HOST = os.environ.get("SOURCE_HOST", "local")
SYMBOLS = [symbol.strip().lower() for symbol in os.environ.get("ORDERBOOK_SYMBOLS", "btcusdt").split(",") if symbol.strip()]
FLUSH_INTERVAL_SEC = float(os.environ.get("FLUSH_INTERVAL_SEC", "20"))
HEARTBEAT_PATH = os.path.join(DATA_DIR, "orderbook", f"HEARTBEAT_{SOURCE_HOST}.json")

# 여러 스트림을 하나의 연결로 받는 'combined stream' 주소
STREAM_NAMES = []
for symbol in SYMBOLS:
    STREAM_NAMES += [f"{symbol}@bookTicker", f"{symbol}@depth20@100ms", f"{symbol}@aggTrade"]
COMBINED_STREAM_URL = "wss://stream.binance.com:9443/stream?streams=" + "/".join(STREAM_NAMES)

# WebSocket ping: 20초마다 ping을 보내고 10초 안에 응답이 없으면 끊긴 것으로 판단
# (네트워크가 조용히 죽어 close 신호조차 오지 않는 경우를 잡기 위함)
PING_INTERVAL_SEC = 20
PING_TIMEOUT_SEC = 10

# ---------------------------------------------------------------------------
# 저장 컬럼
# ---------------------------------------------------------------------------
BOOKTICKER_COLUMNS = [
    "event_time",  # 수신 시각 (bookTicker에는 거래소 시각이 없음)
    "symbol",
    "best_bid_price",  # 최우선 매수호가
    "best_bid_qty",  # 최우선 매수호가 잔량 (BTC)
    "best_ask_price",  # 최우선 매도호가
    "best_ask_qty",  # 최우선 매도호가 잔량 (BTC)
    "update_id",  # 호가창 갱신 번호 (순서 확인용)
    "source_host",
]

DEPTH20_COLUMNS = ["event_time", "symbol"]
for side in ("bid", "ask"):
    for field in ("px", "qty"):
        for level in range(1, DEPTH_LEVELS + 1):
            DEPTH20_COLUMNS.append(f"{side}_{field}_{level}")  # 예: bid_px_1, ask_qty_10
DEPTH20_COLUMNS.append("source_host")

TRADES_COLUMNS = [
    "event_time",  # 거래소가 이벤트를 보낸 시각
    "trade_time",  # 실제 체결 시각
    "symbol",
    "price",
    "quantity",  # 체결 수량 (BTC)
    "is_buyer_maker",  # True = 매수자가 지정가(maker) → 매도 주도 체결
    "agg_trade_id",  # 묶음 체결 번호 (연속성 확인용)
    "recv_time",  # 우리 쪽 수신 시각 (지연 측정용)
    "source_host",
]


def to_iso_utc(milliseconds: float | None = None) -> str:
    """밀리초 타임스탬프 → ISO 문자열(UTC). 인자가 없으면 현재 시각."""
    seconds = milliseconds / 1000.0 if milliseconds is not None else time.time()
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def float_or_none(data: dict, key: str) -> float | None:
    """data[key]를 float으로. 키가 없으면 None (메시지 형식이 바뀌어도 수집이 멈추지 않게)."""
    return float(data[key]) if key in data else None


class OrderbookCollector:
    """WebSocket 메시지를 받아 스트림별 버퍼에 쌓고, 주기적으로 파일 저장 + 생존신호 갱신."""

    def __init__(self):
        self.writers = {
            "bookticker": BufferedParquetWriter("bookticker", SOURCE_HOST, BOOKTICKER_COLUMNS),
            "depth20": BufferedParquetWriter("depth20", SOURCE_HOST, DEPTH20_COLUMNS),
            "trades": BufferedParquetWriter("trades", SOURCE_HOST, TRADES_COLUMNS),
        }
        # 메시지 수신 스레드와 저장 스레드가 같은 버퍼를 만지므로 잠금(lock)으로 보호
        self._lock = threading.Lock()
        self._message_counts = {stream: 0 for stream in self.writers}  # 이번 저장 주기 동안 받은 메시지 수
        self._last_receive_time = {stream: None for stream in self.writers}
        self._stop_event = threading.Event()  # 저장 스레드 정지 신호
        self._shutdown_requested = threading.Event()  # 프로그램 종료 요청 (SIGTERM/Ctrl+C)
        self._current_ws: websocket.WebSocketApp | None = None  # 종료 시 닫을 현재 연결
        self.backoff = ReconnectBackoff()

    # -----------------------------------------------------------------------
    # 메시지 처리
    # -----------------------------------------------------------------------
    def on_message(self, ws, raw_message: str) -> None:
        """
        combined stream 메시지 형식:  {"stream": "btcusdt@aggTrade", "data": {...}}
        stream 이름으로 종류를 구분해 각 처리 함수로 넘긴다.
        """
        receive_time = to_iso_utc()
        try:
            message = json.loads(raw_message)
            stream_name = message.get("stream", "")
            data = message.get("data", {})
        except (json.JSONDecodeError, AttributeError):
            return  # 깨진 메시지는 버리고 계속

        symbol = stream_name.split("@", 1)[0].upper() if stream_name else None

        with self._lock:
            if stream_name.endswith("@bookTicker"):
                self._handle_bookticker(data, receive_time)
            elif "@depth20" in stream_name:
                self._handle_depth20(data, symbol, receive_time)
            elif stream_name.endswith("@aggTrade"):
                self._handle_trade(data, receive_time)

    def _handle_bookticker(self, data: dict, receive_time: str) -> None:
        # Binance 필드: s=심볼, b/B=매수호가/잔량, a/A=매도호가/잔량, u=갱신번호
        row = {
            "event_time": receive_time,
            "symbol": data.get("s"),
            "best_bid_price": float_or_none(data, "b"),
            "best_bid_qty": float_or_none(data, "B"),
            "best_ask_price": float_or_none(data, "a"),
            "best_ask_qty": float_or_none(data, "A"),
            "update_id": data.get("u"),
            "source_host": SOURCE_HOST,
        }
        self.writers["bookticker"].add(row)
        self._record_receipt("bookticker", receive_time)

    def _handle_depth20(self, data: dict, symbol: str, receive_time: str) -> None:
        # depth20 메시지에는 거래소 시각도 심볼도 없어서, 수신 시각과 스트림 이름에서 가져온다
        row = {"event_time": receive_time, "symbol": symbol}
        row.update(flatten_depth_levels(data.get("bids", []), data.get("asks", [])))
        row["source_host"] = SOURCE_HOST

        self.writers["depth20"].add(row)
        self._record_receipt("depth20", receive_time)

    def _handle_trade(self, data: dict, receive_time: str) -> None:
        # Binance 필드: E=이벤트시각, T=체결시각(ms), p=가격, q=수량, m=매수자가 maker인지, a=묶음체결번호
        row = {
            "event_time": to_iso_utc(data.get("E")),
            "trade_time": to_iso_utc(data.get("T")),
            "symbol": data.get("s"),
            "price": float_or_none(data, "p"),
            "quantity": float_or_none(data, "q"),
            "is_buyer_maker": bool(data.get("m")),
            "agg_trade_id": data.get("a"),
            "recv_time": receive_time,
            "source_host": SOURCE_HOST,
        }
        self.writers["trades"].add(row)
        self._record_receipt("trades", receive_time)

    def _record_receipt(self, stream: str, receive_time: str) -> None:
        """생존신호에 쓸 통계(메시지 수, 마지막 수신 시각) 갱신."""
        self._message_counts[stream] += 1
        self._last_receive_time[stream] = receive_time

    # -----------------------------------------------------------------------
    # 주기적 저장 & 생존신호
    # -----------------------------------------------------------------------
    def flush_and_heartbeat(self) -> None:
        """버퍼를 파일로 저장하고, 이번 주기 메시지 수를 생존신호 파일에 기록한다."""
        # 잠금 안에서는 '버퍼 꺼내기 + 카운터 초기화'만 빠르게 하고
        with self._lock:
            message_counts = dict(self._message_counts)
            last_receive_time = dict(self._last_receive_time)
            self._message_counts = {stream: 0 for stream in self.writers}
            written_files = {stream: writer.flush() for stream, writer in self.writers.items()}

        # 생존신호 쓰기는 잠금 밖에서
        heartbeat_state = {
            "pid": os.getpid(),
            "source_host": SOURCE_HOST,
            "symbols": SYMBOLS,
            "msg_counts_last_interval": message_counts,  # watchdog: 전부 0이면 '조용한 정지'
            "last_recv_time": last_receive_time,
            "files_written_last_flush": {stream: path for stream, path in written_files.items() if path},
        }
        write_heartbeat(HEARTBEAT_PATH, heartbeat_state)

    def _flush_loop(self) -> None:
        """별도 스레드: FLUSH_INTERVAL_SEC 마다 저장. 메시지가 안 와도 생존신호는 계속 갱신된다."""
        while not self._stop_event.wait(FLUSH_INTERVAL_SEC):
            self.flush_and_heartbeat()

    # -----------------------------------------------------------------------
    # 연결 상태 콜백
    # -----------------------------------------------------------------------
    def on_open(self, ws) -> None:
        self.backoff.on_connect()
        print(f"[{to_iso_utc()}] connected: {COMBINED_STREAM_URL}", flush=True)

    def on_error(self, ws, error) -> None:
        print(f"[{to_iso_utc()}] websocket error: {error}", flush=True)

    def on_close(self, ws, close_status_code, close_message) -> None:
        print(f"[{to_iso_utc()}] closed: {close_status_code} {close_message}", flush=True)

    # -----------------------------------------------------------------------
    # 종료 요청 (SIGTERM / Ctrl+C)
    # -----------------------------------------------------------------------
    def request_shutdown(self, signal_number=None, frame=None) -> None:
        """
        종료 신호를 받으면 '종료 요청' 표시를 하고 현재 연결을 닫는다.

        주의: 예외(KeyboardInterrupt)를 던지는 방식은 쓰면 안 된다.
        websocket-client 라이브러리가 그 예외를 '연결 오류'로 삼켜 버려서,
        아래 재연결 루프가 그냥 다시 연결해 버린다 (실제로 확인된 버그).
        """
        print(f"[{to_iso_utc()}] shutdown requested (signal {signal_number})", flush=True)
        self._shutdown_requested.set()
        if self._current_ws is not None:
            # 참고: 라이브러리 내부 읽기 루프가 최대 PING_TIMEOUT_SEC(10초) 기다리므로 실제 종료까지 ~10초 걸릴 수 있다.
            #       systemd 정지 대기시간(TimeoutStopSec=30) 안쪽이고, 종료 직전 버퍼 저장은 그대로 실행된다.
            self._current_ws.close(timeout=1)

    # -----------------------------------------------------------------------
    # 메인 루프: 연결 → (끊기면) 대기 → 재연결 … 을 종료 요청이 올 때까지 반복
    # -----------------------------------------------------------------------
    def run_forever(self) -> None:
        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()

        try:
            while not self._shutdown_requested.is_set():
                self._current_ws = websocket.WebSocketApp(
                    COMBINED_STREAM_URL,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                # 연결이 끊기거나 종료 요청으로 닫힐 때까지 여기서 멈춰 있음
                self._current_ws.run_forever(ping_interval=PING_INTERVAL_SEC, ping_timeout=PING_TIMEOUT_SEC)

                if self._shutdown_requested.is_set():
                    break

                # 종료 요청이 아닌데 끊겼다면 → 대기 후 재연결
                # (대기 중에도 종료 요청이 오면 바로 깨어나도록 sleep 대신 Event.wait 사용)
                delay_seconds = self.backoff.on_disconnect()
                print(f"[{to_iso_utc()}] reconnecting in {delay_seconds:.1f}s...", flush=True)
                self._shutdown_requested.wait(delay_seconds)

        finally:
            # 종료 직전 마지막 저장: 버퍼에 남은 최대 20초치 데이터를 잃지 않기 위함
            print(f"[{to_iso_utc()}] shutting down, flushing buffers...", flush=True)
            self._stop_event.set()
            flush_thread.join(timeout=5)
            self.flush_and_heartbeat()


def main() -> None:
    print(f"source_host={SOURCE_HOST} symbols={SYMBOLS} flush_interval={FLUSH_INTERVAL_SEC}s", flush=True)
    collector = OrderbookCollector()

    # systemd 정지(SIGTERM)와 Ctrl+C(SIGINT)를 같은 방식으로 처리 → 버퍼 저장 후 깔끔하게 종료
    signal.signal(signal.SIGTERM, collector.request_shutdown)
    signal.signal(signal.SIGINT, collector.request_shutdown)

    collector.run_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Fatal error: {error}", file=sys.stderr)
        sys.exit(1)  # 비정상 종료 → systemd가 재시작
