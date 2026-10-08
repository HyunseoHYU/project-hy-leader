"""
기술지표(Technical Indicators) 팩터
=================================

트레이더가 차트에서 흔히 쓰는 지표들을 '팩터'로 만들어, 미시구조 기반 베이스라인 팩터와
예측력을 비교하기 위한 모듈. (실행: src/analyze_horizons.py)

원칙
    - 모든 지표는 봉 t가 끝난 시점까지의 가격·거래량만 사용 (미래 정보 없음)
    - 가격 수준(달러)에 따라 값이 커지지 않도록 비율·bp·0~100 지수로 정규화
      → 2018년(6천 달러)과 2026년(8만 달러)을 같은 기준으로 비교 가능
    - 기간(14, 20, 12/26/9 …)은 업계 표준값을 그대로 쓴다. 데이터를 보고 최적 기간을 고르면
      과최적화(데이터 스누핑)가 되므로 일부러 조정하지 않는다.

지표 목록 (컬럼명: 정의)
    rsi_14            RSI = 100 − 100/(1 + 평균상승폭/평균하락폭), 14봉 (Wilder 평활)     0~100, 70↑ 과매수 · 30↓ 과매도
    macd_hist_bps     MACD 히스토그램 = (EMA12 − EMA26) − 시그널(9)  을 종가 대비 bp로
    bollinger_pctb    볼린저 %B = (종가 − 하단) / (상단 − 하단),  밴드 = SMA20 ± 2σ          1↑ 상단 돌파 · 0↓ 하단 이탈
    bollinger_width   볼린저 밴드폭 = (상단 − 하단) / SMA20                                 변동성 수축·확장
    stoch_k           스토캐스틱 %K = (종가 − 14봉 최저) / (14봉 최고 − 14봉 최저) × 100
    stoch_d           스토캐스틱 %D = %K 의 3봉 이동평균
    ma_gap_10_50      이동평균 괴리 = SMA10 / SMA50 − 1                                     골든·데드크로스의 연속형
    price_vs_sma200   장기 추세 = 종가 / SMA200 − 1
    roc_10_bps        변화율(ROC) = ln(종가_t / 종가_{t−10}) × 10⁴
    atr_14_bps        ATR(평균 실제 범위, 14봉)를 종가 대비 bp로                             변동성
    obv_slope_20      OBV 기울기 = 최근 20봉 OBV 변화 / 최근 20봉 거래량 합                 −1~+1, 상승봉 거래량 우위
    volume_ratio_20   거래량 비율 = log(거래량 / 20봉 평균 거래량)
    mfi_14            MFI(자금흐름지수) = 거래량 가중 RSI, 14봉                              0~100
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e4

# ---------------------------------------------------------------------------
# 표준 기간 (업계 관행값 — 데이터에 맞춰 조정하지 않음)
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLLINGER_PERIOD, BOLLINGER_WIDTH_SIGMA = 20, 2.0
STOCHASTIC_PERIOD, STOCHASTIC_SMOOTHING = 14, 3
SHORT_MA, MEDIUM_MA, LONG_MA = 10, 50, 200
ROC_PERIOD = 10
ATR_PERIOD = 14
OBV_PERIOD = 20
VOLUME_RATIO_PERIOD = 20
MFI_PERIOD = 14

TECHNICAL_COLUMNS = [
    "rsi_14",
    "macd_hist_bps",
    "bollinger_pctb",
    "bollinger_width",
    "stoch_k",
    "stoch_d",
    "ma_gap_10_50",
    "price_vs_sma200",
    "roc_10_bps",
    "atr_14_bps",
    "obv_slope_20",
    "volume_ratio_20",
    "mfi_14",
]


def wilder_average(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder 평활 이동평균 (RSI·ATR 원정의).  평활계수 α = 1/period 인 지수이동평균과 같다.
        평균_t = 평균_{t−1} + (값_t − 평균_{t−1}) / period
    """
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def relative_strength_index(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI = 100 − 100 / (1 + 평균 상승폭 / 평균 하락폭)."""
    price_change = close.diff()
    gain = price_change.clip(lower=0)
    loss = (-price_change).clip(lower=0)

    average_gain = wilder_average(gain, period)
    average_loss = wilder_average(loss, period)
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def macd_histogram_bps(close: pd.Series) -> pd.Series:
    """
    MACD = EMA(12) − EMA(26),  시그널 = MACD의 EMA(9),  히스토그램 = MACD − 시그널.
    가격 수준에 비례하므로 종가로 나눠 bp로 표시.
    """
    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    return (macd_line - signal_line) / close * BPS


def bollinger_bands(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """볼린저 %B 와 밴드폭.  밴드 = SMA(20) ± 2 × 표준편차(20)."""
    middle_band = close.rolling(BOLLINGER_PERIOD).mean()
    band_std = close.rolling(BOLLINGER_PERIOD).std()
    upper_band = middle_band + BOLLINGER_WIDTH_SIGMA * band_std
    lower_band = middle_band - BOLLINGER_WIDTH_SIGMA * band_std

    percent_b = (close - lower_band) / (upper_band - lower_band)
    band_width = (upper_band - lower_band) / middle_band
    return percent_b, band_width


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """%K = (종가 − 최근 14봉 최저가) / (최근 14봉 최고가 − 최저가) × 100,  %D = %K의 3봉 평균."""
    lowest_low = low.rolling(STOCHASTIC_PERIOD).min()
    highest_high = high.rolling(STOCHASTIC_PERIOD).max()
    percent_k = (close - lowest_low) / (highest_high - lowest_low) * 100
    percent_d = percent_k.rolling(STOCHASTIC_SMOOTHING).mean()
    return percent_k, percent_d


def average_true_range_bps(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    실제 범위(True Range) = max(고가 − 저가, |고가 − 전 종가|, |저가 − 전 종가|)  ← 갭까지 포함한 봉의 변동폭
    ATR = TR의 Wilder 평균(14봉), 종가 대비 bp.
    """
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return wilder_average(true_range, ATR_PERIOD) / close * BPS


def on_balance_volume_slope(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV(누적 거래량 지표): 종가가 오른 봉의 거래량은 +, 내린 봉은 − 로 누적.
    여기서는 누적값 대신 '최근 20봉 OBV 변화 / 최근 20봉 총 거래량' (−1 ~ +1)으로 정규화.
        +1에 가까울수록 상승봉에 거래량이 몰림
    """
    signed_volume = np.sign(close.diff()) * volume
    obv_change = signed_volume.rolling(OBV_PERIOD).sum()
    total_volume = volume.rolling(OBV_PERIOD).sum()
    return obv_change / total_volume


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    MFI = 거래량을 반영한 RSI.
        대표가격 = (고가 + 저가 + 종가) / 3,   자금흐름 = 대표가격 × 거래량
        대표가격이 오른 봉의 자금흐름 합 / 내린 봉의 자금흐름 합 (14봉) → 비율 R
        MFI = 100 − 100 / (1 + R)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    price_went_up = typical_price.diff() > 0
    price_went_down = typical_price.diff() < 0

    positive_flow = money_flow.where(price_went_up, 0.0).rolling(MFI_PERIOD).sum()
    negative_flow = money_flow.where(price_went_down, 0.0).rolling(MFI_PERIOD).sum()
    return 100 - 100 / (1 + positive_flow / negative_flow)


def build_technical_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    봉 패널(open, high, low, close, btc_volume)에서 기술지표 13개를 계산한다.

    결측 봉(거래소 점검 등) 처리 — 실제로 겪은 문제
        결측 봉을 그대로 두면 200봉 이동평균 같은 긴 지표가 결측 하나 때문에 이후 200봉 동안 전부 비었다.
        1일 봉에서는 점검일 14일 때문에 2021년 전체가 분석에서 빠졌다.
        → 차트 프로그램처럼 결측 봉을 '건너뛰고' 남은 봉끼리 이어서 계산한 뒤 원래 시간축에 다시 맞춘다.
          (빈 구간을 '가격 변화 없음'으로 채우지는 않는다)
    """
    has_price = panel[["close", "high", "low", "btc_volume"]].notna().all(axis=1)
    available_bars = panel[has_price]

    close = available_bars["close"]
    high = available_bars["high"]
    low = available_bars["low"]
    volume = available_bars["btc_volume"]

    factors = pd.DataFrame(index=available_bars.index)
    factors["rsi_14"] = relative_strength_index(close)
    factors["macd_hist_bps"] = macd_histogram_bps(close)
    factors["bollinger_pctb"], factors["bollinger_width"] = bollinger_bands(close)
    factors["stoch_k"], factors["stoch_d"] = stochastic_oscillator(high, low, close)

    short_ma = close.rolling(SHORT_MA).mean()
    medium_ma = close.rolling(MEDIUM_MA).mean()
    long_ma = close.rolling(LONG_MA).mean()
    factors["ma_gap_10_50"] = short_ma / medium_ma - 1
    factors["price_vs_sma200"] = close / long_ma - 1

    factors["roc_10_bps"] = np.log(close / close.shift(ROC_PERIOD)) * BPS
    factors["atr_14_bps"] = average_true_range_bps(high, low, close)
    factors["obv_slope_20"] = on_balance_volume_slope(close, volume)

    average_volume = volume.rolling(VOLUME_RATIO_PERIOD).mean()
    with np.errstate(divide="ignore"):  # 거래량 0인 봉의 log(0) 경고는 아래에서 결측 처리하므로 숨김
        factors["volume_ratio_20"] = np.log(volume / average_volume)
    factors["mfi_14"] = money_flow_index(high, low, close, volume)

    # 0으로 나누기·log(0)(거래량 0인 봉) 등에서 생긴 무한대는 결측으로
    factors = factors[TECHNICAL_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # 원래 시간축(결측 봉 포함)으로 되돌림 — 결측 봉 자리는 지표도 결측
    return factors.reindex(panel.index)
