"""
호가창 수집기 감시 프로세스 (watchdog)
====================================

목적
    수집기가 스스로는 알아차리지 못하는 고장을 바깥에서 감지해 재시작한다.
    그래서 일부러 수집기와 '별도 프로세스'로 실행한다.

감지하는 고장 2가지 (수집기가 남기는 생존신호 파일 HEARTBEAT_<호스트>.json 을 읽어서 판단)
    1. 생존신호 파일이 오래됨     → 수집기가 멈췄거나(hang) 죽었음
    2. 파일은 제때 갱신되는데,
       직전 주기 메시지 수가 전부 0 → '조용한 정지': 연결은 열려 보이지만 데이터가 안 들어옴
                                    (BTC 최우선 호가는 20초 동안 한 번도 안 바뀌는 일이 사실상 없음)

실행 방법
    python watchdog_orderbook.py           자체 루프로 계속 감시
    python watchdog_orderbook.py --once    한 번 점검하고 종료 (cron 2분마다 등록용)
    → 연결 방법은 docs/deployment.md 참고

설정 (.env)
    ORDERBOOK_RESTART_CMD        재시작 명령 (기본: systemctl restart hy-leader-orderbook.service)
    WATCHDOG_STALE_SEC           생존신호가 이보다 오래되면 고장으로 판단 (기본: 저장 주기 × 4, 최소 90초)
    WATCHDOG_CHECK_INTERVAL_SEC  루프 모드의 점검 간격 (기본: 저장 주기 × 3, 최소 60초)
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

from common import DATA_DIR
from orderbook_common import read_heartbeat

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
SOURCE_HOST = os.environ.get("SOURCE_HOST", "local")
HEARTBEAT_PATH = os.path.join(DATA_DIR, "orderbook", f"HEARTBEAT_{SOURCE_HOST}.json")
FLUSH_INTERVAL_SEC = float(os.environ.get("FLUSH_INTERVAL_SEC", "20"))

# 판단 기준을 저장 주기에 비례시킨다 → 저장 주기를 늘려도 멀쩡한 수집기를 재시작하는 일이 없음
DEFAULT_STALE_SEC = max(90.0, FLUSH_INTERVAL_SEC * 4)
DEFAULT_CHECK_INTERVAL_SEC = max(60.0, FLUSH_INTERVAL_SEC * 3)
STALE_SEC = float(os.environ.get("WATCHDOG_STALE_SEC", DEFAULT_STALE_SEC))
CHECK_INTERVAL_SEC = float(os.environ.get("WATCHDOG_CHECK_INTERVAL_SEC", DEFAULT_CHECK_INTERVAL_SEC))

RESTART_CMD = os.environ.get("ORDERBOOK_RESTART_CMD", "systemctl restart hy-leader-orderbook.service")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def restart_collector() -> None:
    """설정된 재시작 명령을 실행한다. 실패해도 watchdog 자체는 죽지 않는다."""
    log(f"restarting: {RESTART_CMD}")
    try:
        subprocess.run(shlex.split(RESTART_CMD), check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        log(f"restart command failed to run: {error} (fix ORDERBOOK_RESTART_CMD for this host)")


def check_once() -> bool:
    """한 번 점검한다. 정상이면 True, 재시작을 실행했으면 False."""
    heartbeat = read_heartbeat(HEARTBEAT_PATH)

    # --- 점검 0: 생존신호 파일이 아예 없음 ---
    if heartbeat is None:
        log(f"no heartbeat found at {HEARTBEAT_PATH} - collector never started or crashed before first flush")
        restart_collector()
        return False

    # --- 점검 1: 생존신호가 너무 오래됨 → 멈춤/사망 ---
    written_at = datetime.fromisoformat(heartbeat["written_at"])
    heartbeat_age_sec = (datetime.now(timezone.utc) - written_at).total_seconds()
    if heartbeat_age_sec > STALE_SEC:
        log(f"heartbeat stale ({heartbeat_age_sec:.0f}s > {STALE_SEC:.0f}s) - process likely hung or dead")
        restart_collector()
        return False

    # --- 점검 2: 생존신호는 새것인데 메시지가 하나도 없음 → 조용한 정지 ---
    message_counts = heartbeat.get("msg_counts_last_interval", {})
    received_nothing = bool(message_counts) and all(count == 0 for count in message_counts.values())
    if received_nothing:
        log(f"heartbeat fresh ({heartbeat_age_sec:.0f}s old) but zero messages last interval - silent stall: {message_counts}")
        restart_collector()
        return False

    log(f"healthy - heartbeat {heartbeat_age_sec:.0f}s old, msg_counts={message_counts}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="check once and exit (for cron-style invocation)")
    args = parser.parse_args()

    # cron 모드: 한 번 점검하고 종료 (종료코드 0 = 정상, 1 = 재시작함)
    if args.once:
        is_healthy = check_once()
        sys.exit(0 if is_healthy else 1)

    # 루프 모드: CHECK_INTERVAL_SEC 마다 계속 점검
    log(f"watchdog loop starting: source_host={SOURCE_HOST} stale_sec={STALE_SEC} check_interval={CHECK_INTERVAL_SEC}s")
    while True:
        check_once()
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
