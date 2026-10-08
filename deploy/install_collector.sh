#!/usr/bin/env bash
# =============================================================================
# 호가창 수집기 설치 스크립트 (VPS / 로컬 WSL 겸용)
# =============================================================================
#
# 사용법
#   bash deploy/install_collector.sh vps      VPS: 시스템 서비스로 설치 (sudo 필요)
#   bash deploy/install_collector.sh local    로컬 WSL: 사용자 서비스로 설치 (sudo 불필요)
#
# 하는 일
#   1. .venv 가상환경 생성 + requirements.txt 설치          (이미 있으면 패키지만 맞춤)
#   2. .env 생성 (없을 때만) + SOURCE_HOST 를 vps/local 로 설정
#   3. systemd 서비스 등록: 수집기 (죽으면 5초 뒤 자동 재시작)
#   4. systemd 타이머 등록: 2분마다 watchdog 점검 (조용한 정지 감지 → 재시작)
#   5. 바로 시작하고 상태 출력
#
# 여러 번 실행해도 안전함 (같은 설정으로 덮어씀). 코드 업데이트 후에는 다시 실행하면 재시작까지 됨.
# =============================================================================
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "vps" && "$MODE" != "local" ]]; then
    echo "사용법: bash deploy/install_collector.sh [vps|local]" >&2
    exit 1
fi

# --- 경로: 이 스크립트 위치(deploy/)의 한 단계 위 = 프로젝트 루트 ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
SERVICE_NAME="hy-leader-orderbook"
WATCHDOG_NAME="hy-leader-watchdog"

echo "[1/5] 가상환경 & 패키지 ($PROJECT_DIR/.venv)"
if [[ ! -x "$PYTHON" ]]; then
    python3 -m venv "$PROJECT_DIR/.venv"
fi
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r "$PROJECT_DIR/requirements.txt"

echo "[2/5] .env 설정 (SOURCE_HOST=$MODE)"
ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
fi
# SOURCE_HOST 줄을 이 모드 값으로 바꾼다 (없으면 추가)
if grep -q '^SOURCE_HOST=' "$ENV_FILE"; then
    sed -i "s/^SOURCE_HOST=.*/SOURCE_HOST=$MODE/" "$ENV_FILE"
else
    echo "SOURCE_HOST=$MODE" >> "$ENV_FILE"
fi

# --- 모드별 차이: 서비스 파일 위치, systemctl 명령, 재시작 명령 ---
if [[ "$MODE" == "vps" ]]; then
    UNIT_DIR="/etc/systemd/system"
    SYSTEMCTL="sudo systemctl"
    WRITE="sudo tee"
    RESTART_CMD="systemctl restart $SERVICE_NAME.service"       # watchdog 타이머가 root로 실행되므로 sudo 불필요
    RUN_AS_LINE="User=$(id -un)"                                 # 수집기 자체는 일반 사용자 권한으로
    WANTED_BY="multi-user.target"
else
    UNIT_DIR="$HOME/.config/systemd/user"
    SYSTEMCTL="systemctl --user"
    WRITE="tee"
    RESTART_CMD="systemctl --user restart $SERVICE_NAME.service"
    RUN_AS_LINE=""                                               # 사용자 서비스는 이미 내 권한
    WANTED_BY="default.target"
    mkdir -p "$UNIT_DIR"
fi

echo "[3/5] 수집기 서비스 등록 ($UNIT_DIR/$SERVICE_NAME.service)"
$WRITE "$UNIT_DIR/$SERVICE_NAME.service" > /dev/null <<EOF
[Unit]
Description=HY-Leader real-time order-book collector ($MODE)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
$RUN_AS_LINE
WorkingDirectory=$PROJECT_DIR/collectors
EnvironmentFile=-$ENV_FILE
ExecStart=$PYTHON collect_orderbook_ws.py
# 프로세스가 어떤 이유로든 끝나면 5초 뒤 다시 시작
Restart=always
RestartSec=5
# 정지 시 SIGTERM → 수집기가 버퍼를 저장하고 스스로 종료 (최대 30초 대기)
TimeoutStopSec=30

[Install]
WantedBy=$WANTED_BY
EOF

echo "[4/5] watchdog 타이머 등록 (2분마다)"
$WRITE "$UNIT_DIR/$WATCHDOG_NAME.service" > /dev/null <<EOF
[Unit]
Description=HY-Leader order-book watchdog check ($MODE)

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_DIR/collectors
EnvironmentFile=-$ENV_FILE
Environment=ORDERBOOK_RESTART_CMD=$RESTART_CMD
ExecStart=$PYTHON watchdog_orderbook.py --once
# 재시작을 실행한 경우 종료코드 1 → '실패'가 아니라 정상 동작으로 취급
SuccessExitStatus=1
EOF

$WRITE "$UNIT_DIR/$WATCHDOG_NAME.timer" > /dev/null <<EOF
[Unit]
Description=Run HY-Leader watchdog every 2 minutes

[Timer]
# 수집기가 첫 생존신호를 쓸 시간을 주기 위해 부팅 3분 뒤부터
OnBootSec=3min
OnUnitActiveSec=2min

[Install]
WantedBy=timers.target
EOF

echo "[5/5] 시작"
$SYSTEMCTL daemon-reload
$SYSTEMCTL enable "$SERVICE_NAME.service" "$WATCHDOG_NAME.timer"
$SYSTEMCTL restart "$SERVICE_NAME.service"
$SYSTEMCTL restart "$WATCHDOG_NAME.timer"

if [[ "$MODE" == "local" ]]; then
    # 로그아웃해도 사용자 서비스가 계속 돌도록 (권한이 없으면 안내만)
    loginctl enable-linger "$(id -un)" 2>/dev/null \
        || echo "  (참고) 'sudo loginctl enable-linger $(id -un)' 을 한 번 실행하면 로그아웃 후에도 계속 수집됩니다."
fi

sleep 3
$SYSTEMCTL --no-pager status "$SERVICE_NAME.service" | head -n 12
echo
echo "설치 완료. 확인 명령:"
echo "  로그:       journalctl $( [[ $MODE == local ]] && echo --user ) -u $SERVICE_NAME -f"
echo "  생존신호:   cat $PROJECT_DIR/data/orderbook/HEARTBEAT_$MODE.json"
echo "  커버리지:   $PYTHON $PROJECT_DIR/collectors/verify_orderbook_coverage.py --host $MODE"
echo "  watchdog:   journalctl $( [[ $MODE == local ]] && echo --user ) -u $WATCHDOG_NAME -n 20"
