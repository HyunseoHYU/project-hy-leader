# collectors/ — 데이터 수집

> 상위: [../CLAUDE.md](../CLAUDE.md) · 이 폴더 파일을 만지기 전에 읽는다.
> 소비처: [../impact/CLAUDE.md](../impact/CLAUDE.md), [../prediction/CLAUDE.md](../prediction/CLAUDE.md) · 배포: [../deploy/CLAUDE.md](../deploy/CLAUDE.md) · 저장 위치: [../data/CLAUDE.md](../data/CLAUDE.md)

## 역할
두 트랙(impact / prediction)이 공유하는 **원자료를 모으는 곳**. 분석 코드는 여기에 두지 않는다.

## 파일
| 파일 | 내용 | 출력 |
|---|---|---|
| `collect_orderbook_ws.py` | ★ Binance WebSocket 실시간 호가·체결 (bookTicker, depth20@100ms, aggTrade), 자동 재연결 | `data/orderbook/<스트림>/host=<H>/date=/hour=/*.parquet` |
| `watchdog_orderbook.py` | 생존신호(heartbeat) 감시, 조용한 정지 시 재시작 — 수집기와 **별도 프로세스** | — |
| `verify_orderbook_coverage.py` | 시간별 행 수로 수집 공백(hard/soft gap) 보고 | 콘솔 |
| `orderbook_common.py` | 호가창 수집기 공통 부품 (경로 규칙, Parquet 버퍼, 백오프, heartbeat) | — |
| `common.py` | CSV 수집기 공통 (`.env` 로드, `DATA_DIR`) | — |
| `collect_bitstamp_1m.py` | Bitstamp BTC/USD 1분봉 10년 (이어받기 지원) | `data/raw/bitstamp_btcusd_1m_{YYYY}.parquet` |
| `collect_binance_1m.py`, `collect_binance_1h.py` | Binance BTCUSDT 1분봉·1시간봉 REST | `data/binance/*.csv` (git 제외) |
| `collect_crypto_news.py`, `collect_reddit_posts.py`, `collect_x_posts.py` | 감성 텍스트 (뉴스·Reddit·X) | `data/*.csv` |

## 규칙
- 모든 시각은 UTC. 캔들 `time`은 시작 시각 (루트 CLAUDE.md §2.2)
- 호가창 수집기는 **무중단**이 최우선: 연결 유실·재연결은 코드에서 처리하고, VPS·로컬 두 인스턴스는 독립 프로세스 (`host=` 파티션)
- 호가창은 `orderbook_common.py`, CSV 수집기는 `common.py`를 쓴다 — 저장 방식을 섞지 않는다
- 감성 텍스트는 수집 시각(`detected_at`)과 발행 시각을 **모두** 기록한다 (이벤트 시각 강건성 검정용)
- API 키는 `.env`에서만 읽고, 키가 없는 소스는 경고 후 건너뛴다
- 실행 위치: 프로젝트 루트 또는 `collectors/`. 예: `python collectors/collect_bitstamp_1m.py`
- 새 수집기는 파일 맨 위 docstring(목적·소스·출력·실행법)을 루트 CLAUDE.md §5 형식으로 작성

## 수집 상태 확인
```bash
python collectors/verify_orderbook_coverage.py --host local   # 공백 리포트
systemctl --user status hy-leader-orderbook                   # 로컬 서비스 (deploy/ 참고)
```
