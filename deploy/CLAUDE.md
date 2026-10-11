# deploy/ — 실시간 수집기 배포

> 상위: [../CLAUDE.md](../CLAUDE.md) · 절차 문서: [../docs/deployment.md](../docs/deployment.md) · 대상: [../collectors/CLAUDE.md](../collectors/CLAUDE.md)

## 역할
호가창 수집기를 systemd 서비스로 설치한다. VPS와 로컬 WSL 두 곳에 **독립적으로** 설치한다 (미러링 아님).

## 파일
| 파일 | 내용 |
|---|---|
| `install_collector.sh [vps\|local]` | venv + 패키지 → `.env`(`SOURCE_HOST`) → 수집기 서비스 → 2분 주기 watchdog 타이머 → 시작 |

- `vps`: 시스템 서비스 (sudo 필요) / `local`: 사용자 서비스 (sudo 불필요, linger)
- 서비스명: `hy-leader-orderbook`, `hy-leader-watchdog` (타이머)
- 여러 번 실행해도 안전(idempotent). 코드 업데이트 후 재실행하면 재시작까지 된다

## 규칙
- VPS는 **도쿄·싱가포르 리전** (Binance는 미국 IP 차단)
- 스크립트를 고치면 idempotent 성질을 유지하고, 절차 변경은 `docs/deployment.md`에도 반영한다
- 설치·재시작은 가동 중인 수집기를 끊으므로, 실행 전 사용자에게 확인한다 (데이터 누적 기간이 곧 표본 크기)
- 현재 상태: 로컬 가동 중(2026-10-08~), VPS 배포 대기 — 진행은 `docs/progress.md`
