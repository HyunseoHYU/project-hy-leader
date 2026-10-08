"""
Binance BTCUSDT 1분봉 30일 + 호가창 스모크테스트 EDA
=================================================

목적
    9월 수집 데이터가 분석에 쓸 만한지 1차로 확인한다.

입력
    1. data/binance/btcusdt_1m_30d.csv            BTCUSDT 1분봉 30일 (OHLCV)
    2. data/orderbook/depth20/host=smoketest/**    호가창 상위 20단계 스냅샷 (9/22 스모크테스트 107분)

출력
    results/eda_binance/1m_30d/*.png              통계표 이미지 + 분포·시계열 차트

실행
    python src/eda_binance_1m_30d.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kline_eda import KlineSettings, load_klines_csv, run_kline_eda
from plot_style import AQUA, BLUE, INK_SECONDARY, MUTED, ORANGE, save_figure, save_table_as_image

# ---------------------------------------------------------------------------
# 경로 & 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KLINES_CSV_PATH = PROJECT_ROOT / "data" / "binance" / "btcusdt_1m_30d.csv"
# 9/22 스모크테스트분만 분석 (host=smoketest). 이후 실시간 수집분은 analyze_liquidity_orderbook.py 에서 다룸
DEPTH20_DIR = PROJECT_ROOT / "data" / "orderbook" / "depth20" / "host=smoketest"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda_binance" / "1m_30d"

TICK_SIZE_USDT = 0.01  # Binance BTCUSDT 최소 호가 단위 (1틱 = 0.01달러)
DEPTH_LEVELS = 10  # 수집기가 저장하는 호가 단계 수 (상위 10단계)

KLINE_SETTINGS = KlineSettings(
    label="1-min Klines (30d)",
    return_label="1-min",
    periods_per_year=365 * 24 * 60,  # 1년 안의 1분 개수
    rolling_window=60,  # 60분 이동 변동성
    rolling_label="60-min",
    volume_bar_width_days=0.0006,  # 1분 ≈ 0.0007일보다 살짝 좁게
    output_dir=OUTPUT_DIR,
)


# ===========================================================================
# 1. 호가창 스냅샷 로드 & 지표 계산
# ===========================================================================
def load_depth_snapshots() -> pd.DataFrame:
    """depth20 parquet 파일을 모두 읽어 시간순으로 정렬한다. 파일이 없으면 빈 DataFrame."""
    parquet_files = sorted(DEPTH20_DIR.glob("**/*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    snapshots = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    snapshots["event_time"] = pd.to_datetime(snapshots["event_time"], format="ISO8601")
    snapshots = snapshots.sort_values("event_time").reset_index(drop=True)
    return snapshots


def add_orderbook_metrics(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    스냅샷마다 기본 미시구조 지표를 계산해 컬럼으로 추가한다.

        중간가(mid)        = (최우선 매수호가 + 최우선 매도호가) / 2
        스프레드           = 최우선 매도호가 − 최우선 매수호가
        스프레드(bp)       = 스프레드 / 중간가 × 10,000
        호가 깊이          = 상위 10단계 잔량 합계 (BTC)
        깊이 불균형        = (매수 깊이 − 매도 깊이) / (매수 깊이 + 매도 깊이),  −1 ~ +1
                             +면 매수 대기 물량이 더 많음
    """
    best_bid = snapshots["bid_px_1"]
    best_ask = snapshots["ask_px_1"]

    snapshots["mid_price"] = (best_bid + best_ask) / 2
    snapshots["spread"] = best_ask - best_bid
    snapshots["spread_bps"] = snapshots["spread"] / snapshots["mid_price"] * 10_000
    snapshots["spread_ticks"] = (snapshots["spread"] / TICK_SIZE_USDT).round().astype(int)

    bid_quantity_columns = [f"bid_qty_{level}" for level in range(1, DEPTH_LEVELS + 1)]
    ask_quantity_columns = [f"ask_qty_{level}" for level in range(1, DEPTH_LEVELS + 1)]
    snapshots["bid_depth_top10"] = snapshots[bid_quantity_columns].sum(axis=1)
    snapshots["ask_depth_top10"] = snapshots[ask_quantity_columns].sum(axis=1)

    total_depth = snapshots["bid_depth_top10"] + snapshots["ask_depth_top10"]
    depth_difference = snapshots["bid_depth_top10"] - snapshots["ask_depth_top10"]
    snapshots["order_flow_imbalance"] = depth_difference / total_depth
    return snapshots


# ===========================================================================
# 2. 호가창 차트
# ===========================================================================
def analyze_orderbook() -> None:
    snapshots = load_depth_snapshots()
    if snapshots.empty:
        print("  호가창 데이터 없음 — 건너뜀")
        return

    snapshots = add_orderbook_metrics(snapshots)

    # --- (1) 지표 기술통계표 ---
    metric_columns = ["mid_price", "spread", "spread_bps", "bid_depth_top10", "ask_depth_top10", "order_flow_imbalance"]
    metric_stats = snapshots[metric_columns].describe().T
    metric_stats["skew"] = snapshots[metric_columns].skew()
    save_table_as_image(
        metric_stats,
        f"Orderbook Snapshots — Descriptive Statistics (n={len(snapshots):,}, smoke test)",
        OUTPUT_DIR / "orderbook_summary_stats.png",
    )

    # --- (2) 중간가 시계열 ---
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(snapshots["event_time"], snapshots["mid_price"], color=BLUE, linewidth=0.8)
    ax.set_ylabel("Mid Price (USDT)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("BTCUSDT Orderbook Mid Price — Smoke Test Window", loc="left", fontweight="bold")
    save_figure(fig, OUTPUT_DIR / "orderbook_mid_price_timeseries.png")

    # --- (3) 스프레드 분포 (틱 단위) ---
    # 스냅샷 대부분이 정확히 1틱이라, y축을 로그로 해야 드물게 넓어진 스프레드도 보인다
    spread_tick_counts = snapshots["spread_ticks"].value_counts().sort_index()
    is_one_tick = snapshots["spread_ticks"] == 1
    one_tick_percent = is_one_tick.mean() * 100
    one_tick_count = int(is_one_tick.sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(spread_tick_counts.index, spread_tick_counts.values, color=ORANGE, width=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Best bid-ask spread (ticks, 1 tick = $0.01)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("BTCUSDT Best Bid-Ask Spread Distribution", loc="left", fontweight="bold")
    ax.annotate(
        f"{one_tick_percent:.2f}% of snapshots\nat 1 tick (n={one_tick_count:,})",
        xy=(1, spread_tick_counts.loc[1]),
        xytext=(0.35, 0.55),
        textcoords="axes fraction",
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1),
    )
    save_figure(fig, OUTPUT_DIR / "orderbook_spread_distribution.png")

    # --- (4) 호가 단계별 평균 잔량 (매수 vs 매도) ---
    levels = list(range(1, DEPTH_LEVELS + 1))
    mean_bid_quantity = [snapshots[f"bid_qty_{level}"].mean() for level in levels]
    mean_ask_quantity = [snapshots[f"ask_qty_{level}"].mean() for level in levels]
    bar_positions = np.arange(len(levels))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bar_positions - bar_width / 2, mean_bid_quantity, bar_width, label="Bid", color=BLUE)
    ax.bar(bar_positions + bar_width / 2, mean_ask_quantity, bar_width, label="Ask", color=ORANGE)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([f"L{level}" for level in levels])
    ax.set_ylabel("Average quantity (BTC)")
    ax.set_title("Average Depth by Level (Top 10)", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    save_figure(fig, OUTPUT_DIR / "orderbook_depth_by_level.png")

    # --- (5) 깊이 불균형 시계열 ---
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(snapshots["event_time"], snapshots["order_flow_imbalance"], color=AQUA, linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylabel("Order Flow Imbalance")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Orderbook Order Flow Imbalance (Top10 bid-ask qty)", loc="left", fontweight="bold")
    save_figure(fig, OUTPUT_DIR / "orderbook_flow_imbalance.png")


# ===========================================================================
# 실행
# ===========================================================================
def main() -> None:
    print("1) BTCUSDT 1분봉 30일 EDA")
    klines = load_klines_csv(KLINES_CSV_PATH)
    run_kline_eda(klines, KLINE_SETTINGS)

    print("2) 호가창 depth20 스냅샷 EDA")
    analyze_orderbook()

    print(f"완료 — {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
