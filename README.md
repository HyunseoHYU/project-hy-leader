# HY-Leader: Sentiment Shocks & Crypto Order-Book Liquidity

한양리더 장학 프로젝트 (2026.09 – 2027.02). 암호화폐 감성지수가 호가창 유동성·시장구조에 미치는 영향을 검증하고(**impact 트랙**),
이를 바탕으로 시장 상태를 예측하는 모델을 만든다(**prediction 트랙**).

현재 단계: **10월 — 미시구조 지표·베이스라인 팩터 완료, 실시간 호가창 수집 가동 (로컬), VPS 배포 대기**
(탐색 단계의 EDA·기초통계·시간 단위별 분석은 `archive/pre-restructure` 브랜치에 보존)

## 두 트랙

| 트랙 | 질문 | 위치 | 상태 |
|---|---|---|---|
| **impact** (영향력 검증) | 감성 충격이 스프레드·깊이·Kyle's λ·PIN 등 유동성에 통계적으로 유의한 영향을 주는가? | `impact/` | 지표·베이스라인 완료, 감성 이벤트 분석은 1월 |
| **prediction** (예측 모델) | 가격·감성 피처로 시장 국면(상승/횡보/하락)을 판별하고 수익률을 예측할 수 있는가? | `prediction/` | MoE 국면 라우팅 파이프라인 병합 (합성 데이터 기준 우위 미입증) |

두 트랙은 `collectors/`가 모으는 같은 데이터(호가창·분봉·감성 텍스트)를 공유한다.
impact의 결과(유의한 유동성 반응)가 확인되면 그 지표를 prediction의 피처·타깃으로 연결한다.

## 구조

```
.
├── collectors/                  # 공통 데이터 수집
│   ├── collect_orderbook_ws.py  # ★ 실시간 호가·체결 (Binance WebSocket, 24시간)
│   ├── watchdog_orderbook.py, verify_orderbook_coverage.py   # 감시·공백 점검
│   ├── collect_bitstamp_1m.py, collect_binance_1m.py, collect_binance_1h.py   # 과거 분봉
│   └── collect_crypto_news.py, collect_reddit_posts.py, collect_x_posts.py    # 감성 텍스트
├── impact/                      # 트랙 1: 감성 → 유동성·시장구조 영향력
│   ├── liquidity_metrics.py     # ★ 미시구조 지표 공식 (유효 스프레드 분해, 슬리피지, OFI, Kyle's λ, VPIN, PIN …)
│   ├── baseline_factors.py      # ★ 감성 제외 베이스라인 → 기대 유동성 (이벤트 효과 = 실제 − 기대)
│   ├── analyze_liquidity_orderbook.py   # 실시간 호가창 → 정밀 지표
│   ├── analyze_liquidity_klines.py      # 1분봉 → 유동성 근사 지표
│   ├── analyze_baseline_factors.py      # 워크포워드 검증
│   └── plot_style.py, table_format.py   # 공통 차트·표 서식
├── prediction/                  # 트랙 2: 예측 모델
│   ├── moe_regime/              # MoE: 라우터(게이트) + 국면별 ridge expert
│   ├── run_pipeline.py, configs/default.yaml, tests/
├── deploy/install_collector.sh  # 실시간 수집기 설치 (vps | local)
├── docs/                        # research_summary.md, file_index.md, progress.md, deployment.md
├── data/                        # raw/ (연도별 parquet), orderbook/ (git 제외), crypto_news.csv
├── results/                     # liquidity/, baseline_factors/, orderbook/  (summary.md + tables + figures)
└── .github/workflows/ci.yml     # push 시 문법 + 테스트 + 데이터 무결성 검사
```

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
