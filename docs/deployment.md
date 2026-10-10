# 실시간 호가창 수집기 배포 (이중 네트워크망 · 이중 보관)

`collectors/collect_orderbook_ws.py` 를 **서로 독립된 두 곳**(VPS + 로컬 WSL)에서 각각 실행한다.
한 프로세스가 두 곳에 쓰는 '미러링'이 아니라 완전히 별개의 프로세스 두 개다 — 한쪽 회선·서버가
죽어도 다른 쪽이 계속 수집한다. 두 결과는 `host=vps` / `host=local` 폴더로 나뉘어 저장되고
파일명이 겹치지 않으므로 나중에 그대로 합치면 된다.

설치는 스크립트 하나로 끝난다: `deploy/install_collector.sh [vps|local]`

---

## 1. 로컬 WSL (현재 가동 중, 2026-10-08~)

```bash
bash deploy/install_collector.sh local
```

- 사용자 systemd 서비스로 설치 (sudo 불필요). 로그아웃해도 계속 돈다 (linger 설정됨)
- **한계**: Windows가 꺼지거나 절전·WSL 종료 시 멈춘다 → 그래서 VPS가 주 수집기, 로컬은 보조

## 2. VPS

### 2.1 서버 고르기
| 항목 | 권장 | 이유 |
|---|---|---|
| 리전 | **도쿄 또는 싱가포르** | Binance.com은 미국 IP를 차단. 도쿄는 Binance 매칭엔진(AWS 도쿄)과 가까워 지연도 작음 |
| 디스크 | **100GB 이상** | 스모크테스트 실측 약 19MB/시간 ≈ 0.5GB/일 → 10월~2월 약 75GB (변동성 큰 날은 더 많음) |
| 사양 | 1 vCPU, 1~2GB RAM | 수집기 메모리 약 50MB |
| OS | Ubuntu 24.04 | systemd 기본 |

### 2.2 설치
```bash
ssh <user>@<vps-ip>
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/<계정>/<repo>.git ~/HY_LEADER
cd ~/HY_LEADER
bash deploy/install_collector.sh vps
```

스크립트가 하는 일:
1. `.venv` 생성 + `requirements.txt` 설치
2. `.env` 생성 + `SOURCE_HOST=vps`
3. 수집기 서비스 등록 (`Restart=always`: 죽으면 5초 뒤 재시작)
4. watchdog 타이머 등록 (2분마다, root로 실행 → 별도 sudo 설정 없이 재시작 가능)
5. 시작 + 상태 출력

코드를 업데이트한 뒤에는 `git pull && bash deploy/install_collector.sh vps` 로 다시 실행하면 재시작까지 된다.

## 3. 무중단 구조 (3중 안전장치)

| 단계 | 담당 | 잡아내는 고장 |
|---|---|---|
| 1 | 수집기 자체 재연결 (지수 백오프 1→60초) | 네트워크 끊김, 거래소 연결 종료 |
| 2 | systemd `Restart=always` | 프로세스 사망 (오류, 메모리 등) |
| 3 | watchdog 타이머 (2분마다) | 생존신호가 멈춤(hang) 또는 메시지 0건(조용한 정지) |

정지 시 (`systemctl stop`) 수집기는 SIGTERM을 받아 버퍼를 파일로 저장한 뒤 종료한다 (최대 약 10초).

## 4. 확인 명령

```bash
# 로컬은 systemctl/journalctl 에 --user, VPS는 sudo
systemctl --user status hy-leader-orderbook          # 가동 상태
journalctl --user -u hy-leader-orderbook -f          # 실시간 로그 (연결·재연결 기록)
journalctl --user -u hy-leader-watchdog -n 20        # watchdog 점검 기록 ("healthy - ...")
cat data/orderbook/HEARTBEAT_local.json              # 마지막 생존신호 (20초마다 갱신)
.venv/bin/python collectors/verify_orderbook_coverage.py --host local   # 빠진 시간 점검
```

주 1회 `verify_orderbook_coverage.py` 로 하드 갭(통째로 빠진 시간)이 없는지 확인한다.

## 5. VPS 데이터를 로컬로 백업 (이중 보관)

```bash
rsync -avz <user>@<vps-ip>:~/HY_LEADER/data/orderbook/ ~/HY_LEADER/data/orderbook/
```

새 파일만 복사하므로 하루 1번이면 충분. WSL에서는 cron이 안 돌 수 있어 수동 또는 Windows 작업 스케줄러 권장.

## 6. 배포 후 한 번은 확인할 것

1. 15~20분 뒤 생존신호가 계속 갱신되고 Parquet 파일이 쌓이는지
2. `python impact/analyze_liquidity_orderbook.py --host vps` 가 돌아가는지 (스키마 점검 겸 10월 지표 계산)
3. 강제 단절 (방화벽으로 Binance 아웃바운드 잠깐 차단) → 재연결 로그가 뜨는지
4. 조용한 정지 흉내 (`kill -STOP <pid>`) → 2분 안에 watchdog이 재시작하는지
5. VPS·로컬이 겹치는 구간에서 `verify_orderbook_coverage.py` 로 두 호스트 모두 공백 없는지
