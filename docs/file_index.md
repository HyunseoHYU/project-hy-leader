# 폴더 · 파일 명단

저장소의 모든 폴더와 파일이 무엇인지 한눈에 보기 위한 색인. (git에 올라가지 않는 파일은 맨 아래 별도 표시)

```
HY_LEADER/
├── README.md, CLAUDE.md, requirements.txt, .env.example, .gitignore, .gitattributes
├── .github/workflows/ci.yml
├── collectors/      데이터 수집 (11)
├── src/             분석 (16)
├── deploy/          실시간 수집기 설치 (1)
├── docs/            문서 (4)
├── data/            데이터
└── results/         분석 결과 (주제별 7개 폴더)
```

---

## 루트

| 파일 | 내용 |
|---|---|
| `README.md` | 프로젝트 소개, 구조, 재현 방법, 결과 링크 |
| `CLAUDE.md` | 연구 설계서 — 연구 질문, 설계 원칙, 가설 H1~H4, 리스크·개선안, 코딩 스타일 |
| `requirements.txt` | 파이썬 패키지 (버전 고정) |
| `.env.example` | API 키·수집기 설정 양식 (실제 `.env`는 git 제외) |
| `.gitignore`, `.gitattributes` | git 제외 목록, 줄바꿈·바이너리 규칙 |
| `.github/workflows/ci.yml` | push 때 자동 검사: 문법 컴파일 + 1분봉 데이터 무결성 |

## `collectors/` — 데이터 수집

| 파일 | 내용 | 출력 |
|---|---|---|
| `collect_orderbook_ws.py` | ★ 실시간 호가·체결 수집 (Binance WebSocket: bookTicker, depth20@100ms, aggTrade), 자동 재연결 | `data/orderbook/` |
| `watchdog_orderbook.py` | 수집기 감시 — 생존신호 정지·메시지 0건(조용한 정지) 감지 시 재시작 | 로그 |
| `verify_orderbook_coverage.py` | 호가창 수집 공백 점검 (하드·소프트 갭 보고) | 화면 출력 |
| `orderbook_common.py` | 호가창 수집 공통 부품: Parquet 버퍼 저장, 재연결 대기, 생존신호 | — |
| `common.py` | 수집기 공통: `.env` 로드, 경로, CSV 증분 저장 | — |
| `collect_bitstamp_1m.py` | Bitstamp BTC/USD 1분봉 10년 (2016-10~2026-09) | `data/raw/bitstamp_*.parquet` |
| `collect_binance_1m.py` | Binance BTCUSDT 1분봉 전체 기간 (이어받기 가능) | `data/binance/btcusdt_1m_max.csv` |
| `collect_binance_1h.py` | Binance BTCUSDT 1시간봉 (2026-01~) | `data/binance/btcusdt_1h_*.csv` |
| `collect_crypto_news.py` | 비트코인 뉴스: CryptoPanic, NewsAPI, RSS | `data/crypto_news.csv` |
| `collect_reddit_posts.py` | Reddit r/Bitcoin 등 게시물 | `data/reddit_bitcoin_posts.csv` |
| `collect_x_posts.py` | X(트위터) 최근 7일 게시물 | `data/x_bitcoin_posts.csv` |

## `src/` — 분석

**공통 모듈 (공식·부품)**

| 파일 | 내용 |
|---|---|
| `liquidity_metrics.py` | ★ 미시구조 지표 공식: 유효 스프레드 분해, 슬리피지, OFI, Kyle's λ, Amihud, Roll, Abdi–Ranaldo, VPIN, PIN, 강건 회귀 |
| `baseline_factors.py` | ★ 베이스라인 팩터 모형: N분 봉 패널, HAR 팩터, 모형(M0~M3), 워크포워드, 평가 검정(DM·CW·PT), Newey–West 회귀 |
| `technical_indicators.py` | 기술지표 13개: RSI, MACD, 볼린저, 스토캐스틱, 이동평균 괴리, ROC, ATR, OBV, MFI 등 |
| `kline_eda.py` | Binance 캔들 공통 EDA 루틴 |
| `plot_style.py` | 공통 차트 색상·서식·저장 함수 |

**실행 스크립트** (모두 `python src/<파일>.py`)

| 파일 | 하는 일 | 결과 폴더 | 소요 |
|---|---|---|---|
| `descriptive_stats.py` | Bitstamp 10년 1분봉 기초통계 | `results/descriptive/` | 10초 |
| `eda_binance_1m_30d.py` | Binance 1분봉 30일 + 호가창 스모크테스트 EDA | `results/eda_binance/1m_30d/` | 10초 |
| `eda_binance_1h.py` | Binance 1시간봉 2026년 EDA | `results/eda_binance/1h/` | 5초 |
| `analyze_liquidity_klines.py` | 1분봉 9년 → 일별 유동성 근사 지표 | `results/liquidity/` | 30초 |
| `analyze_liquidity_orderbook.py` | 실시간 호가창 → 정밀 유동성 지표 (`--host`) | `results/orderbook/` | 30초 |
| `analyze_baseline_factors.py` | 1시간 단위 베이스라인 팩터 모형 | `results/baseline_factors/` | 1.5분 |
| `analyze_horizons.py` | 5분~1일 시간 단위별 평균회귀성·예측 성능·기술지표 비교 | `results/horizons/` | 20~40분 |
| `convert_binance_1m_parquet.py` | Binance 1분봉 CSV → 연도별 Parquet | `data/raw/binance_*.parquet` | 1분 |

## `deploy/`

| 파일 | 내용 |
|---|---|
| `install_collector.sh` | 실시간 수집기 설치 `[vps \| local]` — 가상환경, `.env`, systemd 서비스, 2분 watchdog 타이머 |

## `docs/`

| 파일 | 내용 |
|---|---|
| `research_summary.md` | ★ 지금까지의 연구 결과 종합 요약 (보고서·발표용) |
| `progress.md` | 월별 진행 기록 (장학금 보고서 근거) |
| `deployment.md` | 실시간 수집기 배포 절차 (VPS·로컬) |
| `file_index.md` | 이 문서 |

## `data/`

| 경로 | 내용 | git |
|---|---|---|
| `raw/bitstamp_btcusd_1m_{2016..2026}.parquet` | Bitstamp BTC/USD 1분봉 (11개 파일, 5,258,880행) | ✅ |
| `raw/binance_btcusdt_1m_{2017..2026}.parquet` | Binance BTCUSDT 1분봉 + 시장가 매수량 (10개 파일, 4,796,862행) | ✅ |
| `binance/btcusdt_1m_30d.csv`, `btcusdt_1h_2026jan_sep.csv` | EDA용 소형 CSV | ✅ |
| `crypto_news.csv` | 뉴스 수집 샘플 | ✅ |
| `binance/btcusdt_1m_max.csv` | Binance 1분봉 원본 (856MB) | ❌ 크기 초과 |
| `orderbook/<stream>/host=<호스트>/date=/hour=/` | 실시간 호가·체결 Parquet | ❌ 계속 증가 (rsync로 동기화) |

## `results/` — 주제별 결과 (각 폴더: `summary.md` + `tables/*.csv` + `figures/*.png`)

| 폴더 | 내용 | 핵심 결론 |
|---|---|---|
| `descriptive/` | 10년 1분봉 기초통계 | 두꺼운 꼬리, 1분 미시구조 잡음, 변동성 군집, 14시 UTC 피크 |
| `eda_binance/` | Binance 1차 EDA (`1m_30d/`, `1h/`) | 9월 수집 데이터 검증 |
| `liquidity/` | 9년 일별 유동성 지표 | Kyle's λ·Amihud·VPIN 유효, 1분봉 스프레드 추정치는 2022년 이후 무의미 |
| `orderbook/` | 호가창 정밀 지표 (스모크테스트) | 유효 스프레드 중앙값 1틱, OFI 회귀 R² 0.74 |
| `baseline_factors/` | 1시간 베이스라인 팩터 모형 | 유동성 예측 가능, 가격 예측 불가 |
| `horizons/` | 시간 단위별 분석 + 기술지표 비교 | `docs/research_summary.md` 참고 |

## git에 올라가지 않는 것

| 경로 | 이유 |
|---|---|
| `.venv/` | 가상환경 (`requirements.txt`로 재생성) |
| `.env` | API 키 등 비밀 정보 |
| `data/orderbook/` | 실시간 데이터, 크기가 계속 커짐 |
| `data/binance/btcusdt_1m_max.csv` | 856MB (같은 내용이 `data/raw/binance_*.parquet`에 있음) |
| `playground/` | 개인 실습 노트북 (프로젝트와 무관) |
| `*.log`, `__pycache__/` | 로그, 파이썬 캐시 |
