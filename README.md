# HY-Leader: Sentiment Shocks & Crypto Order-Book Liquidity

한양리더 장학 프로젝트 (2026.09 – 2027.02). 감성 신호 충격이 암호화폐 호가창 유동성에 미치는 영향을
시장 미시구조 관점에서 실증 검증한다.

현재 단계: **10월 — 미시구조 지표·베이스라인 팩터 모형 완료, 실시간 호가창 수집 가동 (로컬), VPS 배포 대기**

## 구조

```
.
├── collectors/                        # 데이터 수집
│   ├── collect_orderbook_ws.py        # ★ 실시간 호가·체결 (Binance WebSocket, 24시간)
│   ├── watchdog_orderbook.py          #   수집기 감시 (조용한 정지 감지 → 재시작)
│   ├── verify_orderbook_coverage.py   #   수집 공백 점검
│   ├── collect_bitstamp_1m.py         # Bitstamp BTC/USD 1분봉 10년
│   ├── collect_binance_1m.py, collect_binance_1h.py   # Binance BTCUSDT 1분봉·1시간봉
│   └── collect_crypto_news.py, collect_reddit_posts.py, collect_x_posts.py   # 감성 텍스트
├── src/                               # 분석
│   ├── liquidity_metrics.py           # ★ 미시구조 지표 공식 (유효 스프레드 분해, 슬리피지, OFI, Kyle's λ, VPIN, PIN …)
│   ├── analyze_liquidity_orderbook.py # 실시간 호가창 → 정밀 지표
│   ├── analyze_liquidity_klines.py    # Binance 1분봉 9년 → 유동성 근사 지표
│   ├── baseline_factors.py            # ★ 베이스라인 팩터 모형 (감성 제외 유동성·가격 예측)
│   ├── analyze_baseline_factors.py    #   워크포워드 검증 → results/baseline_factors/
│   ├── descriptive_stats.py           # Bitstamp 10년 기초통계
│   ├── eda_binance_1m_30d.py, eda_binance_1h.py, kline_eda.py   # Binance 1차 EDA
│   ├── convert_binance_1m_parquet.py  # Binance 1분봉 CSV → 연도별 parquet
│   └── plot_style.py                  # 공통 차트 스타일
├── deploy/install_collector.sh        # 실시간 수집기 설치 (vps | local)
├── docs/                              # progress.md (월별 경과), deployment.md (배포 절차)
├── data/
│   ├── raw/                           # bitstamp_btcusd_1m_{YYYY}, binance_btcusdt_1m_{YYYY}.parquet
│   ├── binance/                       # Binance 1분봉 30일·1시간봉 CSV (EDA용)
│   ├── orderbook/                     # 실시간 호가창 parquet (git 제외)
│   └── crypto_news.csv
├── results/                           # 주제별: summary.md + tables/ + figures/
│   ├── descriptive/                   # 10년 기초통계
│   ├── eda_binance/                   # Binance 1차 EDA
│   ├── liquidity/                     # 1분봉 유동성 지표
│   ├── baseline_factors/              # 베이스라인 팩터 모형 (기대 유동성)
│   └── orderbook/                     # 호가창 정밀 지표
└── .github/workflows/ci.yml           # push 시 문법 + 데이터 무결성 검사
```

## 재현

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python collectors/collect_bitstamp_1m.py     # Bitstamp 10년 수집 (약 12분, 있으면 skip)
python src/descriptive_stats.py              # → results/descriptive/
python src/analyze_liquidity_klines.py       # → results/liquidity/   (약 30초)
python src/analyze_baseline_factors.py       # → results/baseline_factors/ (약 1.5분)
python src/analyze_liquidity_orderbook.py --host <smoketest|local|vps>   # → results/orderbook/
```

실시간 수집기 설치: `bash deploy/install_collector.sh [vps|local]` — 자세한 내용은 [`docs/deployment.md`](docs/deployment.md)

## 데이터

| 데이터 | 소스 | 기간 | 용도 |
|---|---|---|---|
| BTC/USD 1분봉 | Bitstamp REST | 2016-10 ~ 2026-09 (5.26M행, 누락 0) | 장기 통계적 성질 |
| BTCUSDT 1분봉 | Binance REST | 2017-08 ~ 2026-10 (4.80M행) | 시장가 매수량 포함 → 유동성 근사 지표 |
| 호가·체결 | Binance WebSocket | 2026-10-08 ~ (가동 중) | 정밀 미시구조 지표, 감성 이벤트 분석 |

## 결과

- [`results/descriptive/summary.md`](results/descriptive/summary.md) — 10년 기초통계 (두꺼운 꼬리, 미시구조 잡음, 변동성 군집, 시간대 효과)
- [`results/liquidity/summary.md`](results/liquidity/summary.md) — 9년 유동성 지표 (Kyle's λ, Amihud, VPIN, PIN …)
- [`results/baseline_factors/summary.md`](results/baseline_factors/summary.md) — 베이스라인 팩터 모형: 유동성은 예측 가능(표본 외 R² 22~41%), 가격은 예측 불가(수수료 후 손실)
- [`results/orderbook/summary.md`](results/orderbook/summary.md) — 호가창 정밀 지표 (스모크테스트로 파이프라인 검증)
