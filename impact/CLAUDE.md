# impact/ — 트랙 1: 감성 → 유동성·시장구조 영향력

> 상위: [../CLAUDE.md](../CLAUDE.md) · 이 폴더 파일을 만지기 전에 읽는다.
> 입력: [../data/CLAUDE.md](../data/CLAUDE.md) (수집: [../collectors/CLAUDE.md](../collectors/CLAUDE.md)) · 출력: [../results/CLAUDE.md](../results/CLAUDE.md) · 연결 트랙: [../prediction/CLAUDE.md](../prediction/CLAUDE.md)

## 역할
연구 질문 **"감성 충격이 호가창 유동성에 유의한 영향을 미치는가"** 를 검정하는 트랙.
지금(10월)은 *감성 없는* 지표 구현과 베이스라인(기대 유동성)까지 완료. 감성 이벤트 분석은 1월.

## 파일
| 파일 | 역할 | 출력 |
|---|---|---|
| `liquidity_metrics.py` | ★ 미시구조 지표 공식 모음 (유효 스프레드 분해, 슬리피지, OFI, Kyle's λ, Amihud, Roll, Abdi–Ranaldo, VPIN, PIN). **새 지표는 여기에 추가** | — |
| `baseline_factors.py` | ★ 감성 제외 베이스라인. 비정상 유동성 = 실제 − 기대 유동성 | — |
| `trade_flow_backtest.py` | 체결 기반 주문흐름 백테스트 계산 (봉 집계, 워크포워드 릿지, CW·PT 검정, 전략 손익, 플라시보) | — |
| `analyze_aggtrades_backtest.py` | aggTrades 3년 백테스트 실행 | `results/aggtrades_backtest/` |
| `analyze_liquidity_orderbook.py` | 실시간 호가창 → 정밀 지표 (`--host <smoketest\|local\|vps>`) | `results/orderbook/` |
| `analyze_liquidity_klines.py` | 1분봉 → 유동성 근사 지표(장기 기준선) | `results/liquidity/` |
| `analyze_baseline_factors.py` | 워크포워드 검증 | `results/baseline_factors/` |
| `plot_style.py`, `table_format.py` | 공통 차트·표 서식 | — |

## 규칙
- **계산(`liquidity_metrics.py`, `baseline_factors.py`)과 실행(`analyze_*.py`)을 분리**한다. 공식은 계산 모듈에, 로드·저장은 실행 스크립트에
- 지표마다 수식을 주석으로 (예: `# 유효 스프레드 = 2 × |체결가 − 중간가| / 중간가`), 단위(bp, 틱, BTC)를 변수명에 드러낸다
- 파라미터(창 크기, 임계값, 기간)는 파일 상단 상수로. 매직 넘버 금지
- 모든 결과는 스크립트 한 번 실행으로 재생성 가능해야 한다 → `results/<주제>/summary.md` 상단에 재현 명령을 적는다
- 워크포워드만 사용 — 학습에 미래 정보를 섞지 않는다
- 1월 분석 전 **분석 계획을 먼저 고정** (H1–H4, 창 [−30분, +60분]). 결과를 보고 가설을 바꾸지 않는다 (루트 CLAUDE.md §2.2, §3)
- 1분 단위 지표는 2018년 이후만, 실현변동성은 5분 이상 샘플링
- 차트는 `dataviz` 스킬 팔레트, 이중 y축 금지
- 실행: `python impact/<파일>.py` (프로젝트 루트에서)

## 알려진 정리 필요 사항
일부 docstring/summary에 옛 경로 `src/…`가 남아 있다 — 현재 경로는 `impact/…`. 파일을 수정할 때 함께 고친다.
