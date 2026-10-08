"""
Binance BTCUSDT 1시간봉 (2026년 1월~9월) EDA
==========================================

목적
    최근 9개월의 시장 국면(레짐) 변화를 1시간 해상도로 확인한다.
    분석 구성은 1분봉 스크립트(eda_binance_1m_30d.py)와 같고, 월별 요약표 1장을 추가한다.

입력
    data/binance/btcusdt_1h_2026jan_sep.csv

출력
    results/eda_binance/1h/*.png

실행
    python src/eda_binance_1h.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kline_eda import KlineSettings, load_klines_csv, run_kline_eda
from plot_style import save_table_as_image

# ---------------------------------------------------------------------------
# 경로 & 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KLINES_CSV_PATH = PROJECT_ROOT / "data" / "binance" / "btcusdt_1h_2026jan_sep.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda_binance" / "1h"

HOURS_PER_YEAR = 365 * 24  # 연율화 계수


def build_monthly_summary(klines: pd.DataFrame) -> pd.DataFrame:
    """
    월별 요약표.
        month_return_%   월간 수익률 = (월말 종가 / 월초 시가 − 1) × 100
        ann_vol_%        1시간 수익률 표준편차를 연율화한 변동성
        taker_buy_ratio  시장가 매수 거래량 / 전체 거래량  (0.5보다 크면 매수 우위)
    """
    month_of_candle = klines["open_time"].dt.strftime("%Y-%m")
    candles_by_month = klines.groupby(month_of_candle)

    month_open = candles_by_month["open"].first()
    month_close = candles_by_month["close"].last()

    return pd.DataFrame(
        {
            "open": month_open,
            "close": month_close,
            "high": candles_by_month["high"].max(),
            "low": candles_by_month["low"].min(),
            "month_return_%": (month_close / month_open - 1) * 100,
            "ann_vol_%": candles_by_month["log_return"].std() * np.sqrt(HOURS_PER_YEAR) * 100,
            "avg_volume_btc": candles_by_month["volume"].mean(),
            "avg_trades": candles_by_month["num_trades"].mean(),
            "taker_buy_ratio": candles_by_month["taker_buy_base_volume"].sum() / candles_by_month["volume"].sum(),
        }
    )


def main() -> None:
    # --- 1. 데이터 로드 & 기간 문자열 (차트 제목용) ---
    klines = load_klines_csv(KLINES_CSV_PATH)
    first_day = klines["open_time"].min()
    last_day = klines["open_time"].max()
    period_text = f"{first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d}"

    # --- 2. 공통 EDA (1분봉 스크립트와 같은 6종) ---
    settings = KlineSettings(
        label=f"1-hour Klines ({period_text})",
        return_label="1-hour",
        periods_per_year=HOURS_PER_YEAR,
        rolling_window=24,  # 24시간 이동 변동성
        rolling_label="24-hour",
        volume_bar_width_days=1 / 24,  # 1시간 = 1/24일
        output_dir=OUTPUT_DIR,
    )
    run_kline_eda(klines, settings)

    # --- 3. 추가: 월별 요약표 ---
    monthly_summary = build_monthly_summary(klines)
    save_table_as_image(
        monthly_summary,
        "BTCUSDT Monthly Summary (from 1-hour Klines)",
        OUTPUT_DIR / "btc_monthly_summary.png",
    )

    print(f"완료 — {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
