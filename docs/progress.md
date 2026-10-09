# 진행 상황 (Progress Log)

한양리더 장학 프로젝트 월별 실행 경과 트래킹. 새 항목은 위에 추가.

## 9월 (미시구조 거래 환경 조성)

- [x] 실시간 Websocket 데이터 수집기 구현 (`collectors/collect_orderbook_ws.py`) — Binance
  `bookTicker`/`depth20@100ms`/`aggTrade` 3개 스트림, 재연결 backoff, heartbeat, Parquet
  버퍼링 저장(`collectors/orderbook_common.py`). 별도 감시 프로세스(`collectors/watchdog_orderbook.py`)와
  커버리지 점검 도구(`collectors/verify_orderbook_coverage.py`) 포함. 로컬 스모크테스트 완료.
- [ ] VPS·로컬 두 인스턴스 실제 배포 및 24시간 무중단 검증 — 로컬은 10/8 가동, VPS 대기 (10월 항목 참고)
- [ ] 이중 네트워크망·이중 보관 구성 (설계·구현 완료, 실제 두 호스트에서의 동시 가동은 배포 후 확인)
- [ ] 미시구조 이론 학습
- [x] BTCUSDT 1분봉 수집 스크립트 작성 (2016-01-01 이후 최대 기간, 상장 2017-08부터 반환, 재개 가능) (`collectors/collect_binance_1m.py`)
- [x] 1차 EDA: Binance 1분봉 30일 + 호가창 스모크테스트 1시간분 (`src/eda_binance_1m_30d.py`), Binance 1시간봉 2026.1~9 (`src/eda_binance_1h.py`) → `results/eda_binance/`

## 10월 (미시구조 이론·알고리즘 테스트)

- [x] (10/5) BTC/USD 1분봉 10년 수집 — Bitstamp, 2016-10-01 ~ 2026-09-30 UTC, 5,258,880행, 누락 0분
  (`collectors/collect_bitstamp_1m.py` → `data/raw/bitstamp_btcusd_1m_{YYYY}.parquet`)
- [x] (10/5) 10년 1분봉 기초통계량 (`src/descriptive_stats.py` → `results/descriptive/summary.md`, 표 11개·그림 8개)
  - 전 시간 단위 정규성 기각, 초과첨도 1m 87 / 1h 35 / 1d 13 (fat tail)
  - 1분 수익률 lag-1 자기상관 −0.058, 분산비 VR(1d)=0.74 → bid-ask bounce 등 미시구조 잡음
  - |r| 자기상관 lag-60분 0.24 → 강한 변동성 군집
  - 14시 UTC(미국장 개장) 변동성·거래량 최대, 주말 거래량 약 40% 감소
  - 주의: 2016 Q4는 거래 없는 분 37% → 초기 구간 1분 지표 해석 주의
- [x] (10/6) Binance BTCUSDT 1분봉 전체 기간 연도별 Parquet 변환 — 2017-08-17 ~ 2026-10-06, 4,796,862행,
  taker buy volume·체결 건수 포함 (`src/convert_binance_1m_parquet.py` → `data/raw/binance_btcusdt_1m_{YYYY}.parquet`)
- [x] (10/8) 저장소 정리: Git 커밋, `.gitignore`(856MB 원본 CSV·호가창 데이터·로그 제외), requirements 버전 고정, CI(문법·데이터 무결성)
- [x] (10/8) 코드 전체 리팩터링 — 읽기 쉬운 이름·단계별 주석·공식 주석 (CLAUDE.md §5). 결과물 동일성 검증 완료
  - 수집기 버그 수정: SIGTERM(systemd 정지)을 WebSocket 라이브러리가 삼켜 종료되지 않고 재연결하던 문제 → 버퍼 저장 후 정상 종료
- [x] (10/8) 미시구조 지표 구현 (`src/liquidity_metrics.py`)
  - [x] 유효 스프레드 = 실현 스프레드 + 가격 충격(역선택 비용) 분해 — 체결 메시지가 호가보다 ~50ms 늦게 도착하는 것을 실측해 보정
  - [x] 슬리피지 곡선 (상위 10단계) — 10단계는 ±10틱($0.10)뿐이라 5 BTC 이상은 대부분 계산 불가 → 더 깊은 호가 수집 필요 (개선 과제)
  - [x] Kyle's λ — 1분봉 일별 추정(9년) + 체결 기반(10초) / OFI 회귀(Cont et al. 2014): 스모크테스트 R² 0.74
  - [x] PIN (분기별 최대우도) + VPIN — PIN은 81% 분기에서 경계해(Poisson 과대분산) → 진단용, VPIN 병행
  - [x] Amihud, Roll, Abdi–Ranaldo — 1분봉 스프레드 추정치는 2022년 이후 무의미(실제 스프레드 1틱 ≈ 0.001bp)함을 확인
  - 결과: `results/liquidity/summary.md` (9년 1분봉), `results/orderbook/summary.md` (호가창 스모크테스트)
- [x] (10/8) 저장소 구조 정리 — 수집 코드는 `collectors/`, 분석은 `src/`, 결과는 `results/<주제>/`로 통일. 감성 파이프라인 스켈레톤(harness)은 12월에 이벤트 스터디 설계에 맞춰 새로 작성하기로 하고 삭제
- [x] (10/9) 베이스라인 팩터 모형 (감성 제외) — `src/baseline_factors.py`, `results/baseline_factors/summary.md`
  - 1시간 단위, 팩터 22개(HAR·반전/모멘텀·주문 불균형·비정상 거래량·시간 패턴), 워크포워드 2021~2026
  - 유동성: 표본 외 R²(직전 값 대비) Amihud 28%, Kyle's λ 41%, 변동성 22% → **M2(선형)를 1월 '기대 유동성' 모형으로 채택** (10/10 설계 개선 후 수치는 아래 항목)
  - 가격: R² 음수, Clark–West 비유의, 수수료 후 샤프 크게 음수 → 예측 불가 (단기 반전 약 2bp만 존재 → 이벤트 분석 시 통제)
  - 검증 중 바로잡은 함정 2가지: 트리 모형의 범위 밖 외삽 실패(편차 학습으로 해결), 쏠린 예측의 방향 적중률 과대평가(Pesaran–Timmermann으로 교체)
- [x] (10/10) 시간 단위별 분석 + 기술지표 팩터 — `src/analyze_horizons.py`, `src/technical_indicators.py`, `results/horizons/summary.md`
  - 5분 · 15분 · 1시간 · 4시간 · 1일, 미시구조 vs 기술지표(13개) vs 결합, 가격·유동성 대상 구분
  - 평균회귀성: 유동성 충격 반감기 5분 봉 3~5분 ~ 1일 봉 30~40시간, 수익률은 전 시간 단위에서 1차 자기상관 음수
  - 유동성: 미시구조 팩터 우세, 기술지표 추가 시 +0.2~2.7%p / 가격: 5·15분 기술지표 부스팅만 R² +0.2%(안정적)이나 수수료 후 손실
  - 베이스라인 설계 개선(유동성 편차 학습·팩터 절단·가격용 정상 팩터) → 1시간 베이스라인 R²: Amihud 28→22.8%, λ 41.5→40.9%, 변동성 21.8→20.0% (더 보수적·강건한 설계로 교체, 수치 하락 투명 공개)
- [x] (10/10) 결과 종합 요약 `docs/research_summary.md`, 파일 명단 `docs/file_index.md` 작성
- [x] (10/8) 로컬 WSL 실시간 수집 가동 (`deploy/install_collector.sh local`, systemd 사용자 서비스 + 2분 watchdog 타이머)
- [ ] VPS 배포 (`deploy/install_collector.sh vps`) — GitHub 연결 완료(10/8), 서버 대여 대기
  - ⚠️ 로컬 수집 공백 확인 (10/9): 10/8 14:00 ~ 10/9 05:00 UTC 16시간 통째로 비어 있음 (노트북 절전 추정), 복구 후 자동 재개.
    → 로컬 단독으로는 24시간 수집 불가 — VPS 배포가 최우선
- [ ] 9월 수집 데이터로 EDA → 로컬/VPS 실시간 데이터가 1~2주 쌓이면 `analyze_liquidity_orderbook.py` 재실행

## 11월 (EDA 정리 및 중간보고)

- [ ] EDA 결과 보고서 작성
- [ ] 경제금융대학 학술제 참여

## 12월 (감성 데이터 수집 파이프라인 구축)

- [x] 뉴스/Reddit/X 텍스트 수집기 초안 작성 (`collectors/collect_x_posts.py`, `collect_reddit_posts.py`, `collect_crypto_news.py`) — 원래 12월 예정이나 API 키 확보 시점에 맞춰 9월에 선작업
  - X: v2 recent search (최근 7일 한도, 표준 티어)
  - Reddit: PRAW로 r/Bitcoin, r/CryptoCurrency, r/BitcoinMarkets 수집
  - 뉴스: CryptoPanic API + NewsAPI + RSS(CoinDesk/Cointelegraph/Decrypt) 3중 소스, 키 없는 소스는 자동 skip
  - 크레덴셜은 `.env` (gitignore 처리, `.env.example` 참고)로 주입
- [ ] 감성지수 NLP 모델 파인튜닝
- [ ] LLM 교차검증
- [ ] 기존 파이프라인과 병합

## 1월 (감성지수 호가창 영향 분석 — 핵심 과제)

- [ ] 이벤트 전후 지표 변화 추적
- [ ] Selective Inference 방법론으로 통계 검정

## 2월 (최종 결론 및 보고)

- [ ] 결과 정리
- [ ] 코드/프레임워크 오픈소스 공유
- [ ] 후속 연구 방향 탐색
