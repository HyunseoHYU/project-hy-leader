# results/ — 분석 결과

> 상위: [../CLAUDE.md](../CLAUDE.md) · 생성: [../impact/CLAUDE.md](../impact/CLAUDE.md) · 입력: [../data/CLAUDE.md](../data/CLAUDE.md)

## 구성 — 주제별 `summary.md` + `tables/*.csv` + `figures/*.png`
| 주제 | 내용 | 만드는 스크립트 | 요약 |
|---|---|---|---|
| `liquidity/` | 9년 1분봉 유동성 근사 지표 (Kyle's λ, Amihud, Roll, VPIN, PIN) | `impact/analyze_liquidity_klines.py` | [summary](liquidity/summary.md) |
| `baseline_factors/` | 감성 제외 베이스라인 — 유동성은 예측 가능, 가격은 불가 | `impact/analyze_baseline_factors.py` | [summary](baseline_factors/summary.md) |
| `orderbook/` | 실시간 호가창 정밀 지표 (현재 스모크테스트로 파이프라인 검증) | `impact/analyze_liquidity_orderbook.py` | [summary](orderbook/summary.md) |

(옛 `descriptive`, `eda_binance`, `horizons` 결과는 `archive/pre-restructure` 브랜치)

## 규칙
- **직접 편집 금지** — 이 폴더는 스크립트 출력물이다. 고치려면 스크립트를 고치고 다시 실행한다
- 새 주제 폴더는 `summary.md` 상단에 데이터·검증 방식·**재현 명령 한 줄**을 적는다. 명령의 경로는 `impact/…` (옛 `src/…` 아님)
- 유의/비유의 모두 정식 결과로 쓴다. 비유의 결과를 지우거나 숨기지 않는다
- 요약 문서(`docs/research_summary.md`)에 인용되는 수치가 바뀌면 그쪽도 갱신한다
