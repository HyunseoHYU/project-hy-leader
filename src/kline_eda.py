"""
Binance 캔들(kline) 공통 EDA 루틴
================================

eda_binance_1m_30d.py (1분봉 30일)와 eda_binance_1h.py (1시간봉 9개월)가 똑같은 분석을
캔들 주기만 바꿔 수행하므로, 그 공통 부분을 이 모듈 하나로 모았다.

만드는 결과물 (output_dir 아래 PNG):
    btc_klines_summary_stats.png      전체 컬럼 기술통계표
    btc_returns_summary_stats.png     로그수익률 기술통계표 (+ 연율화 변동성)
    btc_price_volume_timeseries.png   종가 & 거래량 시계열
    btc_returns_distribution.png      로그수익률 히스토그램
    btc_rolling_volatility.png        이동 실현변동성
    btc_correlation_heatmap.png       컬럼 간 상관계수
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SURFACE, save_figure, save_table_as_image

# Binance kline CSV의 숫자 컬럼 (collectors/collect_binance_1m.py 출력 형식)
KLINE_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",  # 거래량 (BTC)
    "quote_asset_volume",  # 거래대금 (USDT)
    "num_trades",  # 체결 건수
    "taker_buy_base_volume",  # 시장가 매수 거래량 (BTC) — 매수 압력 지표
    "taker_buy_quote_volume",  # 시장가 매수 거래대금 (USDT)
]


@dataclass
class KlineSettings:
    """캔들 주기마다 달라지는 설정값."""

    label: str  # 차트 제목에 들어갈 이름, 예: "1-min Klines (30d)"
    return_label: str  # 수익률 이름, 예: "1-min"
    periods_per_year: int  # 연율화 계수, 1분봉 = 365×24×60, 1시간봉 = 365×24
    rolling_window: int  # 이동 변동성 창 길이 (캔들 개수)
    rolling_label: str  # 예: "60-min", "24-hour"
    volume_bar_width_days: float  # 거래량 막대 폭 (일 단위; 1분 ≈ 0.0007일)
    output_dir: Path


def load_klines_csv(csv_path: Path) -> pd.DataFrame:
    """kline CSV를 읽어 시간순 정렬하고 로그수익률 컬럼을 추가한다."""
    klines = pd.read_csv(csv_path)
    klines["open_time"] = pd.to_datetime(klines["open_time"])
    klines = klines.sort_values("open_time").reset_index(drop=True)

    # 로그수익률 r_t = ln(P_t / P_{t-1})
    klines["log_return"] = np.log(klines["close"]).diff()
    return klines


def run_kline_eda(klines: pd.DataFrame, settings: KlineSettings) -> None:
    """공통 EDA 6종을 차례로 실행한다."""
    save_summary_tables(klines, settings)
    plot_price_and_volume(klines, settings)
    plot_return_histogram(klines, settings)
    plot_rolling_volatility(klines, settings)
    plot_correlation_heatmap(klines, settings)


# ---------------------------------------------------------------------------
# 1. 기술통계표
# ---------------------------------------------------------------------------
def save_summary_tables(klines: pd.DataFrame, settings: KlineSettings) -> None:
    # --- 전체 컬럼: describe() + 왜도·첨도 ---
    column_stats = klines[KLINE_NUMERIC_COLUMNS].describe().T
    column_stats["skew"] = klines[KLINE_NUMERIC_COLUMNS].skew()
    column_stats["kurtosis"] = klines[KLINE_NUMERIC_COLUMNS].kurtosis()
    save_table_as_image(
        column_stats,
        f"BTCUSDT {settings.label} — Descriptive Statistics",
        settings.output_dir / "btc_klines_summary_stats.png",
    )

    # --- 로그수익률: describe() + 왜도·첨도 + 연율화 변동성(= 표준편차 × √연간 캔들 수) ---
    log_return = klines["log_return"]
    return_stats = klines[["log_return"]].describe().T
    return_stats["skew"] = log_return.skew()
    return_stats["kurtosis"] = log_return.kurtosis()
    return_stats["annualized_vol"] = log_return.std() * np.sqrt(settings.periods_per_year)
    save_table_as_image(
        return_stats,
        f"BTCUSDT {settings.return_label} Log Returns — Descriptive Statistics",
        settings.output_dir / "btc_returns_summary_stats.png",
    )


# ---------------------------------------------------------------------------
# 2. 종가 & 거래량 (단위가 달라 이중 축 대신 위아래 두 패널)
# ---------------------------------------------------------------------------
def plot_price_and_volume(klines: pd.DataFrame, settings: KlineSettings) -> None:
    fig, (price_ax, volume_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])

    price_ax.plot(klines["open_time"], klines["close"], color=BLUE, linewidth=1)
    price_ax.set_ylabel("Close Price (USDT)")
    price_ax.set_title(f"BTCUSDT Close Price & Volume — {settings.label}", loc="left", fontweight="bold")

    volume_ax.bar(klines["open_time"], klines["volume"], color=ORANGE,
                  width=settings.volume_bar_width_days, linewidth=0)
    volume_ax.set_ylabel("Volume (BTC)")
    volume_ax.set_xlabel("Time (UTC)")

    save_figure(fig, settings.output_dir / "btc_price_volume_timeseries.png")


# ---------------------------------------------------------------------------
# 3. 로그수익률 분포 (평균 ±6σ 범위만 표시)
# ---------------------------------------------------------------------------
def plot_return_histogram(klines: pd.DataFrame, settings: KlineSettings) -> None:
    log_return = klines["log_return"].dropna()
    mean_return = log_return.mean()
    std_return = log_return.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log_return, bins=200, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.2)

    # 평균(주황 점선)과 ±1 표준편차(회색 점선) 표시
    ax.axvline(mean_return, color=ORANGE, linewidth=1.5, linestyle="--", label=f"mean={mean_return:.2e}")
    ax.axvline(mean_return + std_return, color=MUTED, linewidth=1, linestyle=":", label=f"±1 std={std_return:.2e}")
    ax.axvline(mean_return - std_return, color=MUTED, linewidth=1, linestyle=":")

    ax.set_xlim(mean_return - 6 * std_return, mean_return + 6 * std_return)
    ax.set_xlabel(f"{settings.return_label} log return")
    ax.set_ylabel("Frequency")
    ax.set_title(f"BTCUSDT {settings.return_label} Log Return Distribution", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    save_figure(fig, settings.output_dir / "btc_returns_distribution.png")


# ---------------------------------------------------------------------------
# 4. 이동 실현변동성 = 최근 N개 캔들 수익률의 표준편차 × √N  (창 길이 단위의 변동성)
# ---------------------------------------------------------------------------
def plot_rolling_volatility(klines: pd.DataFrame, settings: KlineSettings) -> None:
    window = settings.rolling_window
    rolling_volatility = klines["log_return"].rolling(window).std() * np.sqrt(window)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(klines["open_time"], rolling_volatility, color=AQUA, linewidth=1)
    ax.set_ylabel(f"{settings.rolling_label} rolling realized volatility")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(f"BTCUSDT {settings.rolling_label} Rolling Realized Volatility", loc="left", fontweight="bold")

    save_figure(fig, settings.output_dir / "btc_rolling_volatility.png")


# ---------------------------------------------------------------------------
# 5. 컬럼 간 상관계수 히트맵
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(klines: pd.DataFrame, settings: KlineSettings) -> None:
    correlation = klines[KLINE_NUMERIC_COLUMNS].corr()
    column_count = len(KLINE_NUMERIC_COLUMNS)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlation, cmap="Blues", vmin=-1, vmax=1)

    ax.set_xticks(range(column_count))
    ax.set_yticks(range(column_count))
    ax.set_xticklabels(KLINE_NUMERIC_COLUMNS, rotation=45, ha="right")
    ax.set_yticklabels(KLINE_NUMERIC_COLUMNS)

    # 칸마다 상관계수 숫자 표시 (진한 칸은 흰 글자로)
    for row in range(column_count):
        for col in range(column_count):
            value = correlation.iloc[row, col]
            text_color = "white" if abs(value) > 0.6 else INK
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title(f"BTCUSDT {settings.label.split(' (')[0]} — Feature Correlation", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8)

    save_figure(fig, settings.output_dir / "btc_correlation_heatmap.png")
