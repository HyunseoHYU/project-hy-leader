"""
실시간 호가창 기반 유동성 지표 (정밀)
==================================

목적
    collectors/collect_orderbook_ws.py 가 모은 호가·체결 원자료로 10월 핵심 지표를 계산한다.
    지금은 9/22 스모크테스트 1시간분으로 '계산 파이프라인이 동작하는지' 검증하는 단계이고,
    VPS 수집이 쌓이면 같은 스크립트를 그대로 다시 돌린다.

계산 지표 (정의는 src/liquidity_metrics.py 주석 참고)
    1. 호가 스프레드          최우선 매도호가 − 최우선 매수호가 (틱, bp)
    2. 유효 스프레드 분해     유효 스프레드 = 실현 스프레드 + 가격 충격(역선택 비용), Δ = 1·5·30·60초
    3. 슬리피지 곡선          0.1 ~ 10 BTC 시장가 주문의 체결 비용
    4. 가격 충격 회귀         Δ중간가 = λ · 주문흐름  (10초 구간)
                              ① OFI(호가창 주문흐름 불균형, Cont et al. 2014)
                              ② 순매수 체결량 (Kyle's λ의 체결 기반 버전)

입력
    data/orderbook/{bookticker,trades,depth20}/host=<HOST>/**/*.parquet

출력
    results/orderbook/tables/*.csv, results/orderbook/figures/*.png, results/orderbook/summary.md

실행
    python src/analyze_liquidity_orderbook.py [--host smoketest]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import liquidity_metrics as lm
from plot_style import BLUE, MUTED, ORANGE, save_figure

# ---------------------------------------------------------------------------
# 경로 & 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORDERBOOK_DIR = PROJECT_ROOT / "data" / "orderbook"
OUTPUT_DIR = PROJECT_ROOT / "results" / "orderbook"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

TICK_SIZE_USDT = 0.01  # BTCUSDT 최소 호가 단위
PRICE_IMPACT_HORIZONS_SECONDS = [1, 5, 30, 60]  # 유효 스프레드 분해에서 '체결 후 Δ초'
SLIPPAGE_ORDER_SIZES_BTC = [0.1, 0.5, 1, 2, 5, 10]
REGRESSION_INTERVAL = "10s"  # 가격 충격 회귀의 집계 구간 길이
DEPTH_LEVELS = 10


# ===========================================================================
# 1. 데이터 로드
# ===========================================================================
def load_stream(stream: str, host: str) -> pd.DataFrame:
    """한 스트림의 parquet 파일을 모두 읽는다. 시각 문자열은 UTC datetime으로 변환."""
    files = sorted((ORDERBOOK_DIR / stream / f"host={host}").glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"{stream} 데이터 없음: {ORDERBOOK_DIR / stream / f'host={host}'}")

    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    for time_column in ["event_time", "trade_time", "recv_time"]:
        if time_column in frame.columns:
            # 마이크로초가 있는 값과 없는 값이 섞여 있어 ISO8601 형식으로 유연하게 읽는다
            frame[time_column] = pd.to_datetime(frame[time_column], format="ISO8601", utc=True)
    return frame


def load_quotes(host: str) -> pd.DataFrame:
    """최우선 호가(bookTicker) + 중간가·스프레드."""
    quotes = load_stream("bookticker", host).sort_values("event_time").reset_index(drop=True)
    quotes["mid"] = (quotes["best_bid_price"] + quotes["best_ask_price"]) / 2
    quotes["spread_usdt"] = quotes["best_ask_price"] - quotes["best_bid_price"]
    quotes["spread_ticks"] = (quotes["spread_usdt"] / TICK_SIZE_USDT).round()
    quotes["spread_bps"] = quotes["spread_usdt"] / quotes["mid"] * lm.BPS
    return quotes


# ===========================================================================
# 2. 지표 계산
# ===========================================================================
def quoted_spread_summary(quotes: pd.DataFrame) -> pd.DataFrame:
    """
    호가 스프레드 분포. 호가가 바뀔 때마다 1행이 생기므로 '시간 가중'으로 본다
    (다음 갱신까지 유지된 시간만큼 가중치) — 짧게 스쳐 간 넓은 스프레드가 과대 반영되지 않게.
    """
    seconds_until_next_update = quotes["event_time"].diff().shift(-1).dt.total_seconds().fillna(0)
    spread_ticks = quotes["spread_ticks"]

    total_seconds = seconds_until_next_update.sum()
    one_tick_time_share = seconds_until_next_update[spread_ticks == 1].sum() / total_seconds

    return pd.DataFrame(
        {
            "value": {
                "quote_updates": len(quotes),
                "time_weighted_mean_spread_ticks": np.average(spread_ticks, weights=seconds_until_next_update + 1e-9),
                "time_weighted_mean_spread_bps": np.average(quotes["spread_bps"], weights=seconds_until_next_update + 1e-9),
                "share_of_time_at_1_tick": one_tick_time_share,
                "max_spread_ticks": spread_ticks.max(),
                "mean_best_bid_qty_btc": quotes["best_bid_qty"].mean(),
                "mean_best_ask_qty_btc": quotes["best_ask_qty"].mean(),
            }
        }
    )


def price_impact_regressions(quotes: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    10초 구간마다 [중간가 변화(bp)] 를 [주문흐름] 에 회귀한다.
        ① OFI(BTC)          : 호가창에서 측정한 매수 압력
        ② 순매수 체결량(BTC) : 체결에서 측정한 매수 압력 = Kyle's λ
    기울기 = 주문흐름 1 BTC당 가격이 몇 bp 움직이는가 (100 BTC 단위로도 표시).
    """
    # --- 구간별 OFI 합계, 구간 시작·끝 중간가 ---
    quote_frame = quotes.set_index("event_time")
    quote_frame["ofi"] = lm.order_flow_imbalance(quotes).to_numpy()

    interval_mid_first = quote_frame["mid"].resample(REGRESSION_INTERVAL).first()
    interval_mid_last = quote_frame["mid"].resample(REGRESSION_INTERVAL).last()
    interval_ofi = quote_frame["ofi"].resample(REGRESSION_INTERVAL).sum()

    # --- 구간별 순매수 체결량 (매수 주도 +, 매도 주도 −) ---
    trade_direction = np.where(trades["is_buyer_maker"], -1, 1)
    signed_quantity = pd.Series(trade_direction * trades["quantity"].to_numpy(), index=trades["recv_time"])
    interval_signed_volume = signed_quantity.resample(REGRESSION_INTERVAL).sum()

    intervals = pd.DataFrame(
        {
            "mid_change_bps": (interval_mid_last / interval_mid_first - 1) * lm.BPS,
            "ofi_btc": interval_ofi,
            "signed_trade_volume_btc": interval_signed_volume,
        }
    ).dropna()

    # --- 회귀 두 개 ---
    rows = []
    for flow_column, label in [("ofi_btc", "OFI (Cont et al. 2014)"), ("signed_trade_volume_btc", "signed trade volume (Kyle)")]:
        regression = lm.ols_with_robust_se(intervals[flow_column], intervals["mid_change_bps"])
        rows.append(
            {
                "order_flow_measure": label,
                "slope_bps_per_btc": regression["slope"],
                "slope_bps_per_100btc": regression["slope"] * 100,
                "t_stat_robust": regression["t_stat"],
                "r_squared": regression["r_squared"],
                "n_intervals": regression["n"],
            }
        )
    return pd.DataFrame(rows).set_index("order_flow_measure"), intervals


# ===========================================================================
# 3. 차트
# ===========================================================================
def plot_spread_decomposition(decomposition: pd.DataFrame) -> None:
    """Δ초가 길어질수록 '가격 충격(역선택)'이 커지고 '실현 스프레드(공급자 몫)'는 줄어드는 모습."""
    horizons = decomposition.index.astype(str) + "s"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(horizons, decomposition["price_impact_bps"], color=ORANGE, marker="o", ms=8, label="price impact (adverse selection)")
    ax.plot(horizons, decomposition["realized_spread_bps"], color=BLUE, marker="o", ms=8, label="realized spread (to liquidity provider)")
    ax.axhline(decomposition["effective_spread_bps"].iloc[0], color=MUTED, ls="--", lw=1, label="effective spread")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("horizon Δ after trade")
    ax.set_ylabel("bp (volume-weighted)")
    ax.set_title("Effective spread decomposition — smoke test", loc="left")
    ax.legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "spread_decomposition.png")


def plot_slippage_curve(slippage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for side, color in [("buy", BLUE), ("sell", ORANGE)]:
        rows = slippage[slippage["side"] == side]
        ax.plot(rows["order_size_btc"], rows["median_bps"], color=color, marker="o", ms=8, label=f"market {side}")
    ax.set_xscale("log")
    ax.set_xlabel("order size (BTC, log)")
    ax.set_ylabel("median slippage vs mid (bp)")
    ax.set_title(f"Slippage curve from top-{DEPTH_LEVELS} depth — smoke test", loc="left")
    ax.legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "slippage_curve.png")


def plot_price_impact_scatter(intervals: pd.DataFrame, regressions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [("ofi_btc", "OFI (Cont et al. 2014)"), ("signed_trade_volume_btc", "signed trade volume (Kyle)")]
    for ax, (column, label) in zip(axes, panels):
        slope = regressions.loc[label, "slope_bps_per_btc"]
        intercept_free_line = np.linspace(intervals[column].min(), intervals[column].max(), 50)

        ax.scatter(intervals[column], intervals["mid_change_bps"], s=10, color=BLUE, alpha=0.5)
        ax.plot(intercept_free_line, slope * intercept_free_line, color=ORANGE, lw=1.5)
        ax.set_xlabel(f"{label} per {REGRESSION_INTERVAL} (BTC)")
        ax.set_ylabel(f"mid change per {REGRESSION_INTERVAL} (bp)")
        r_squared = regressions.loc[label, "r_squared"]
        ax.set_title(f"{label}\nslope {slope * 100:.2f} bp / 100 BTC, R² {r_squared:.2f}", loc="left")
    save_figure(fig, FIGURES_DIR / "price_impact_regressions.png")


# ===========================================================================
# 4. 요약
# ===========================================================================
def write_summary(host: str, quotes: pd.DataFrame, trades: pd.DataFrame, spread_summary: pd.DataFrame,
                  decomposition: pd.DataFrame, slippage: pd.DataFrame, regressions: pd.DataFrame) -> None:
    from descriptive_stats import to_markdown_table

    start_time = quotes["event_time"].min()
    end_time = quotes["event_time"].max()
    duration_minutes = (end_time - start_time).total_seconds() / 60
    thin_book_share_10btc = slippage.loc[slippage["order_size_btc"] == 10, "book_too_thin_share"].max()

    # 수집 공백: 호가 갱신 사이 간격이 가장 길었던 곳 (BTC는 보통 0.01초마다 갱신되므로 수십 초 이상이면 수집 끊김)
    largest_gap_seconds = quotes["event_time"].diff().dt.total_seconds().max()

    markdown = f"""# 실시간 호가창 유동성 지표 (host={host})

- 구간: {start_time:%Y-%m-%d %H:%M:%S} ~ {end_time:%H:%M:%S} UTC ({duration_minutes:.0f}분)
- 호가 갱신 {len(quotes):,}건, 체결 {len(trades):,}건, 최대 수집 공백 {largest_gap_seconds:.0f}초
- 재현: `python src/analyze_liquidity_orderbook.py --host {host}` / 지표 정의: `src/liquidity_metrics.py`
- ⚠️ 표본이 짧아 수치 자체보다 **계산 파이프라인 검증** 목적. VPS 수집분이 쌓이면 다시 실행.

## 1. 호가 스프레드 (시간 가중)

{to_markdown_table(spread_summary, ".4g")}

## 2. 유효 스프레드 분해 (거래량 가중, bp)

유효 스프레드 = 실현 스프레드(유동성 공급자 몫) + 가격 충격(역선택 비용)

{to_markdown_table(decomposition)}

## 3. 슬리피지 곡선 (상위 {DEPTH_LEVELS}단계 호가 기준)

{to_markdown_table(slippage.set_index(["side", "order_size_btc"]))}

> 상위 {DEPTH_LEVELS}단계는 가격으로 고작 {DEPTH_LEVELS} 틱(≈ ${DEPTH_LEVELS * TICK_SIZE_USDT:.2f}) 범위다.
> 10 BTC 주문은 스냅샷의 {thin_book_share_10btc:.0%}에서 보이는 잔량이 모자라 계산 불가 → 큰 주문의 슬리피지를 보려면
> 더 깊은 호가가 필요 (개선안: REST 호가 스냅샷 주기 수집 또는 diff-depth 스트림으로 전체 호가창 유지).

## 4. 가격 충격 회귀 ({REGRESSION_INTERVAL} 구간, 이분산 강건 t값)

{to_markdown_table(regressions)}

## 그림

| | |
|---|---|
| ![](figures/spread_decomposition.png) | ![](figures/slippage_curve.png) |
| ![](figures/price_impact_regressions.png) | |
"""
    (OUTPUT_DIR / "summary.md").write_text(markdown, encoding="utf-8")


# ===========================================================================
# 실행
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="smoketest", help="분석할 수집기 이름 (data/orderbook/*/host=<이름>)")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 데이터 로드")
    quotes = load_quotes(args.host)
    trades = load_stream("trades", args.host).sort_values("recv_time").reset_index(drop=True)
    depth = load_stream("depth20", args.host).sort_values("event_time").reset_index(drop=True)

    print("2) 지표 계산")
    spread_summary = quoted_spread_summary(quotes)
    _per_trade, decomposition = lm.effective_spread_decomposition(trades, quotes, PRICE_IMPACT_HORIZONS_SECONDS)
    slippage = lm.slippage_curve(depth, SLIPPAGE_ORDER_SIZES_BTC, levels=DEPTH_LEVELS)
    regressions, intervals = price_impact_regressions(quotes, trades)

    print("3) 저장")
    spread_summary.to_csv(TABLES_DIR / "quoted_spread_summary.csv")
    decomposition.to_csv(TABLES_DIR / "effective_spread_decomposition.csv")
    slippage.to_csv(TABLES_DIR / "slippage_curve.csv", index=False)
    regressions.to_csv(TABLES_DIR / "price_impact_regressions.csv")

    plot_spread_decomposition(decomposition)
    plot_slippage_curve(slippage)
    plot_price_impact_scatter(intervals, regressions)
    write_summary(args.host, quotes, trades, spread_summary, decomposition, slippage, regressions)

    print(f"완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
