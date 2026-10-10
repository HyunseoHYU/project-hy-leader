# 폴더 · 파일 명단

두 트랙(impact / prediction) 구조 기준. 탐색 단계 파일(EDA·기초통계·horizons)은 `archive/pre-restructure` 브랜치 참고.
git에 올라가지 않는 것: `.env`, `data/orderbook/`, `data/binance/btcusdt_1m_max.csv`, `prediction/outputs/`, 로그.

```
HY_LEADER/
├── README.md, CLAUDE.md, requirements.txt, .env.example, .gitignore, .gitattributes
├── .github/workflows/ci.yml   문법 컴파일 + 예측 트랙 테스트 + 1분봉 무결성
├── collectors/   공통 데이터 수집 (호가창 WebSocket, 과거 분봉, 감성 텍스트)
├── impact/       트랙 1: 감성 → 유동성 영향력 (지표 공식, 베이스라인, 분석 스크립트)
├── prediction/   트랙 2: MoE 국면 라우팅 예측 모델
├── deploy/       실시간 수집기 설치 (install_collector.sh)
├── docs/         research_summary, file_index, progress, deployment
├── data/         raw/ (연도별 parquet), orderbook/, crypto_news.csv
└── results/      liquidity/, baseline_factors/, orderbook/
```

## `collectors/`
| 파일 | 내용 |
|---|---|
| `collect_orderbook_ws.py` | ★ 실시간 호가·체결 수집 (bookTicker, depth20@100ms, aggTrade), 자동 재연결 → `data/orderbook/` |
| `watchdog_orderbook.py` | 조용한 정지 감지 시 재시작 |
| `verify_orderbook_coverage.py` | 수집 공백 점검 |
| `orderbook_common.py`, `common.py` | 수집기 공통 부품 |
| `collect_bitstamp_1m.py`, `collect_binance_1m.py`, `collect_binance_1h.py` | 과거 분봉 |
| `collect_crypto_news.py`, `collect_reddit_posts.py`, `collect_x_posts.py` | 감성 텍스트 |

## `impact/`
| 파일 | 내용 | 출력 |
|---|---|---|
| `liquidity_metrics.py` | ★ 미시구조 지표 공식 | — |
| `baseline_factors.py` | ★ 감성 제외 베이스라인 (기대 유동성) | — |
| `analyze_liquidity_orderbook.py` | 실시간 호가창 정밀 지표 | `results/orderbook/` |
| `analyze_liquidity_klines.py` | 1분봉 유동성 근사 지표 | `results/liquidity/` |
| `analyze_baseline_factors.py` | 워크포워드 검증 | `results/baseline_factors/` |
| `plot_style.py`, `table_format.py` | 차트·표 서식 | — |

## `prediction/`
| 파일 | 내용 |
|---|---|
| `moe_regime/` | data, features(인과적), labels, router(게이트), experts(ridge), moe, metrics, pipeline |
| `run_pipeline.py`, `configs/default.yaml` | 실행 진입점·설정 |
| `tests/test_pipeline.py` | 피처 인과성·라벨·라우터 테스트 |
