# data/ — 데이터

> 상위: [../CLAUDE.md](../CLAUDE.md) · 생성: [../collectors/CLAUDE.md](../collectors/CLAUDE.md) · 사용: [../impact/CLAUDE.md](../impact/CLAUDE.md), [../prediction/CLAUDE.md](../prediction/CLAUDE.md)

## 구성
| 경로 | 내용 | git |
|---|---|---|
| `raw/bitstamp_btcusd_1m_{YYYY}.parquet` | BTC/USD 1분봉 2016-10~. 컬럼 `time, open, high, low, close, volume` | 추적 |
| `raw/binance_btcusdt_1m_{YYYY}.parquet` | BTCUSDT 1분봉 2017-08~. 위 컬럼 + `close_time, quote_asset_volume, num_trades, taker_buy_base_volume, taker_buy_quote_volume` | 추적 |
| `orderbook/{bookticker,depth20,trades}/host=<H>/date=/hour=/*.parquet` | 실시간 호가창. `event_time`, 최우선 호가·잔량 / 상위 10단계 / 체결(`is_buyer_maker`) | **제외** |
| `orderbook/HEARTBEAT_<host>.json` | 수집기 생존신호 | 제외 |
| `binance/btcusdt_1m_max.csv` | Binance 1분봉 원본 CSV (856MB) | 제외 |
| `crypto_news.csv` (+ `reddit_*`, `x_*`) | 감성 텍스트 | 추적 |

## 규칙
- 모든 시각은 UTC, 캔들 `time`은 **시작 시각**
- 저장 형식은 Parquet(zstd), 연도 또는 날짜·시간 단위 분할. GitHub 파일당 100MB 제한 → 큰 파일은 git에 올리지 않는다
- `data/raw`의 두 소스는 기간이 겹치므로 **따로** 읽고 검사한다 (CI도 동일)
- `host=` 파티션: `smoketest`(검증용 1회분), `local`, `vps`. 분석 시 호스트를 명시하고, 병합은 `event_time` 기준 중복 제거 후 수행
- 호가창 데이터는 수집 중인 파일이 있으니 **읽기 전용**으로만 다루고 삭제·이동하지 않는다. VPS ↔ 로컬 동기화는 rsync ([../docs/deployment.md](../docs/deployment.md))
- 2016 Q4는 무거래 분 37% → 분 단위 지표는 2018년 이후만
