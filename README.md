# HY-Leader: Sentiment Shocks & Crypto Order-Book Liquidity

```
한양리더 장학 프로젝트 (2026.09 – 2027.02).

연구 과제 : 감성 신호 충격이 암호화폐 호가창 유동성에 미치는 영향을 시장 미시구조 관점에서 실증하고 이를 기반으로 한 예측 모델을 생성한다.

현재 단계: 미시구조 지표를 활용한 베이스라인 구축 및 연구의 이론적 배경 확립
```

## 두 트랙

| 트랙 | 질문 | 위치 | 상태 |
|---|---|---|---|
| **impact** (영향력 검증) | 감성 충격이 스프레드·깊이·Kyle's λ·PIN 등 유동성에 통계적으로 유의한 영향을 주는가? | [`impact/`](impact/CLAUDE.md) | 지표·베이스라인 완료, 감성 이벤트 분석은 1월 |
| **prediction** (예측 모델) | 가격·감성 피처로 시장 국면(상승/횡보/하락)을 판별하고 수익률을 예측할 수 있는가? | [`prediction/`](prediction/CLAUDE.md) | MoE 국면 라우팅 파이프라인 병합 (합성 데이터 기준 우위 미입증) |

두 트랙은 [`collectors/`](collectors/CLAUDE.md)가 모으는 같은 데이터(호가창·분봉·감성 텍스트)를 공유한다.

## 구조

각 폴더에는 그 폴더의 역할·파일·규칙을 설명하는 `CLAUDE.md`가 있다. 폴더 작업 전에 먼저 읽고, 상단 링크로 인접 폴더로 이동한다 (전체 지도: [`CLAUDE.md`](CLAUDE.md) §6.1).

```
.
├── CLAUDE.md                          # 프로젝트 전체 맥락·규칙 + 폴더별 가이드 지도
├── collectors/  → CLAUDE.md           # 데이터 수집
│   ├── collect_orderbook_ws.py        # ★ 실시간 호가·체결 (Binance WebSocket, 24시간)
│   ├── watchdog_orderbook.py, verify_orderbook_coverage.py   # 감시·공백 점검
│   ├── orderbook_common.py, common.py #   수집기 공통 부품
│   ├── collect_bitstamp_1m.py         # Bitstamp BTC/USD 1분봉 10년
│   ├── collect_binance_1m.py, collect_binance_1h.py   # Binance BTCUSDT 1분봉·1시간봉
│   └── collect_crypto_news.py, collect_reddit_posts.py, collect_x_posts.py   # 감성 텍스트
├── impact/      → CLAUDE.md           # 트랙 1: 감성 → 유동성·시장구조 영향력
│   ├── liquidity_metrics.py           # ★ 미시구조 지표 공식 (유효 스프레드 분해, 슬리피지, OFI, Kyle's λ, VPIN, PIN …)
│   ├── baseline_factors.py            # ★ 감성 제외 베이스라인 → 기대 유동성
│   ├── analyze_liquidity_orderbook.py, analyze_liquidity_klines.py, analyze_baseline_factors.py
│   └── plot_style.py, table_format.py #   공통 차트·표 서식
├── prediction/  → CLAUDE.md           # 트랙 2: MoE 국면 라우팅 예측
│   ├── moe_regime/                    #   라우터(게이트) + 국면별 ridge expert
│   └── run_pipeline.py, configs/default.yaml, tests/
├── deploy/      → CLAUDE.md           # install_collector.sh [vps|local]
├── docs/        → CLAUDE.md           # research_summary, file_index, progress, deployment
├── data/        → CLAUDE.md           # raw/ (연도별 parquet), orderbook/ (git 제외), crypto_news.csv
├── results/     → CLAUDE.md           # liquidity/, baseline_factors/, orderbook/  (summary.md + tables + figures)
└── .github/     → CLAUDE.md           # workflows/ci.yml — push 시 문법 + 테스트 + 데이터 무결성 검사
```

(탐색 단계의 EDA·기초통계·시간 단위별 분석은 `archive/pre-restructure` 브랜치에 보존)

## 재현

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# impact 트랙
python collectors/collect_bitstamp_1m.py                  # Bitstamp 10년 수집 (약 12분, 있으면 skip)
python impact/analyze_liquidity_klines.py                 # → results/liquidity/   (약 30초)
python impact/analyze_baseline_factors.py                 # → results/baseline_factors/ (약 1.5분)
python impact/analyze_liquidity_orderbook.py --host <smoketest|local|vps>   # → results/orderbook/

# prediction 트랙
python prediction/run_pipeline.py [--config PATH] [--csv date,close[,volume,sentiment].csv]
python -m pytest prediction                               # 인과성·라벨 검증
```

실시간 수집기 설치: `bash deploy/install_collector.sh [vps|local]` — 자세한 내용은 [`docs/deployment.md`](docs/deployment.md)

## 데이터

| 데이터 | 소스 | 기간 | 용도 |
|---|---|---|---|
| BTC/USD 1분봉 | Bitstamp REST | 2016-10 ~ 2026-09 (5.26M행, 누락 0) | 장기 통계적 성질 |
| BTCUSDT 1분봉 | Binance REST | 2017-08 ~ 2026-10 (4.80M행) | 시장가 매수량 포함 → 유동성 근사 지표 |
| 호가·체결 | Binance WebSocket | 2026-10-08 ~ (가동 중) | 정밀 미시구조 지표, 감성 이벤트 분석 |

## 결과

- [`results/liquidity/summary.md`](results/liquidity/summary.md) — 9년 유동성 지표 (Kyle's λ, Amihud, VPIN, PIN …)
- [`results/baseline_factors/summary.md`](results/baseline_factors/summary.md) — 유동성은 예측 가능(표본 외 R² 20~41%), 가격은 예측 불가(수수료 후 손실)
- [`results/orderbook/summary.md`](results/orderbook/summary.md) — 호가창 정밀 지표 (스모크테스트로 파이프라인 검증)
- [`docs/research_summary.md`](docs/research_summary.md) — 탐색 단계 결과 종합 (기초통계·시간 단위별 분석 포함, 해당 산출물은 archive 브랜치)

## prediction 트랙 메모

- 라벨은 라우터 지도학습에만 쓰이고, 학습/검증은 시간순 분할 + `horizon` 행 purge. 라우터 입력은 인과적(테스트로 보장)
- CSV 형식: `date, close[, volume, sentiment]`. CSV 없이 실행하면 합성 국면전환 시계열 사용
- 합성 데이터에서 라우터 정확도 ≈ 0.5 (우연 0.33), 단일 ridge 대비 우위 미입증 — 실데이터 검증 전 결론 금지
- 출처: `Impact_of_Sentiment_on_CryptoMarket` 저장소 (브랜치 `claude/compassionate-franklin-bv69cv`) 병합
