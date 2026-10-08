"""
BTC/USD 1분봉 10년치 기초통계량 분석
==================================

목적
    10년치 1분봉으로 비트코인 가격의 장기 통계적 성질(분포, 변동성, 자기상관, 시간대 패턴)을
    파악한다. 이후 감성 이벤트 분석에서 "평소 수준"을 정하는 기준선(baseline)으로 쓴다.

입력
    data/raw/bitstamp_btcusd_1m_*.parquet  (collectors/collect_bitstamp_1m.py 결과)

출력
    results/descriptive/tables/*.csv     통계표
    results/descriptive/figures/*.png    차트
    results/descriptive/summary.md       핵심 결과 요약 (자동 생성)

주요 정의
    - 로그수익률  r_t = ln(P_t / P_{t-1}),  P = 1분봉 종가
    - 연율화      변동성 × √(1년 안의 구간 수). 암호화폐는 휴장이 없으므로 1년 = 365일 × 24시간
    - 누락된 분은 직전 종가로 채우고 거래량은 0으로 둔다 → 모든 시간 단위를 같은 1분 격자에서 계산

실행
    python src/descriptive_stats.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_style import BLUE, GRID, MUTED, ORANGE, save_figure

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "descriptive"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# 분석 설정값
# ---------------------------------------------------------------------------
MINUTES_PER_YEAR = 365 * 24 * 60  # 연율화 계수 계산용 (휴장 없음)
DAYS_PER_YEAR = 365

# 수익률을 계산할 시간 단위: 이름 → 분 단위 길이
HORIZONS_IN_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

ACF_MAX_LAG_MINUTES = 60  # 1분 수익률 자기상관은 1~60분 시차까지
ACF_MAX_LAG_DAYS = 30  # 일별 수익률 자기상관은 1~30일 시차까지
RV_SMOOTHING_DAYS = 30  # 실현변동성 차트의 이동평균 기간
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ===========================================================================
# 1. 데이터 로드 & 품질 점검
# ===========================================================================
def load_minute_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    연도별 parquet을 모두 읽어 하나의 1분 격자 DataFrame으로 만든다.

    반환: (bars, raw_bars)
        raw_bars = 원본 그대로 (품질 점검용)
        bars     = 1분 격자로 채운 데이터, 컬럼은 아래와 같음

    bars 컬럼:
        open, high, low, close, volume  원본 OHLCV (거래량 단위 BTC)
        is_missing                      원본에 없던 분이면 True
        usd_volume                      거래대금 근사치 = 거래량 × 종가
        log_return_1m                   1분 로그수익률
    """
    # --- 1) 연도별 파일 읽어서 이어 붙이기 ---
    parquet_files = sorted(RAW_DATA_DIR.glob("bitstamp_btcusd_1m_*.parquet"))
    if not parquet_files:
        raise SystemExit("data/raw 에 parquet 없음 — 먼저 collectors/collect_bitstamp_1m.py 실행")

    yearly_frames = [pd.read_parquet(path) for path in parquet_files]
    raw_bars = pd.concat(yearly_frames).set_index("time").sort_index()
    raw_bars = raw_bars[~raw_bars.index.duplicated()]  # 혹시 모를 중복 시각 제거

    # --- 2) 빈틈 없는 1분 격자로 맞추기 ---
    full_minute_grid = pd.date_range(raw_bars.index[0], raw_bars.index[-1], freq="1min", tz="UTC")
    bars = raw_bars.reindex(full_minute_grid)

    # --- 3) 누락된 분 채우기: 가격은 직전 종가 유지, 거래량은 0 ---
    bars["is_missing"] = bars["close"].isna()
    bars["close"] = bars["close"].ffill()
    for price_column in ["open", "high", "low"]:
        bars[price_column] = bars[price_column].fillna(bars["close"])
    bars["volume"] = bars["volume"].fillna(0.0)

    # --- 4) 파생 컬럼 ---
    bars["usd_volume"] = bars["volume"] * bars["close"]
    bars["log_return_1m"] = np.log(bars["close"]).diff()

    return bars, raw_bars


def build_coverage_table(bars: pd.DataFrame, raw_bars: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 데이터 품질 점검표.
        - 누락된 분 수, 거래 없는 분(거래량 0) 수
        - OHLC 정합성 위반 수: 고가 < max(시가, 종가) 이거나 저가 > min(시가, 종가) 이거나 가격 ≤ 0
    """
    # --- OHLC 정합성 위반 여부 (원본 데이터 기준) ---
    high_too_low = raw_bars["high"] < raw_bars[["open", "close"]].max(axis=1)
    low_too_high = raw_bars["low"] > raw_bars[["open", "close"]].min(axis=1)
    non_positive_price = (raw_bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
    is_ohlc_violation = high_too_low | low_too_high | non_positive_price

    # --- 연도별 집계 ---
    year_of_bar = bars.index.year
    year_of_raw_bar = raw_bars.index.year

    coverage = pd.DataFrame(
        {
            "minutes_expected": bars.groupby(year_of_bar).size(),
            "minutes_missing": bars["is_missing"].groupby(year_of_bar).sum(),
            "zero_volume_minutes": (bars["volume"] == 0).groupby(year_of_bar).sum(),
            "ohlc_violations": is_ohlc_violation.groupby(year_of_raw_bar).sum(),
        }
    )
    coverage["missing_pct"] = 100 * coverage["minutes_missing"] / coverage["minutes_expected"]
    coverage["zero_volume_pct"] = 100 * coverage["zero_volume_minutes"] / coverage["minutes_expected"]
    coverage.index.name = "year"
    return coverage


# ===========================================================================
# 2. 통계 계산 도구
# ===========================================================================
def make_returns_by_horizon(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """
    시간 단위별 로그수익률을 만든다.
    겹치지 않는 구간을 쓰기 위해 h분마다 한 번씩 종가를 뽑아 차분한다
    (예: 1시간 수익률 = 60분 간격으로 뽑은 로그종가의 차이).
    """
    log_close = np.log(bars["close"])

    returns_by_horizon = {}
    for horizon_name, minutes in HORIZONS_IN_MINUTES.items():
        sampled_log_close = log_close.iloc[::minutes]
        returns_by_horizon[horizon_name] = sampled_log_close.diff().dropna()
    return returns_by_horizon


def describe_returns(returns: pd.Series, periods_per_year: float) -> dict:
    """
    수익률 분포 요약.
        ann_vol          연율화 변동성 = 표준편차 × √(연간 구간 수)
        excess_kurtosis  초과첨도 (정규분포 = 0). 클수록 극단값이 잦은 '두꺼운 꼬리'
        jarque_bera      정규성 검정 통계량. p값이 작으면 정규분포가 아니다
        zero_share       수익률이 정확히 0인 비율 (거래 없는 분이 많을수록 큼)
    """
    returns = returns.dropna()
    quantiles = returns.quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    jarque_bera_result = stats.jarque_bera(returns)

    return {
        "n": len(returns),
        "mean": returns.mean(),
        "std": returns.std(),
        "ann_mean": returns.mean() * periods_per_year,
        "ann_vol": returns.std() * np.sqrt(periods_per_year),
        "skew": stats.skew(returns),
        "excess_kurtosis": stats.kurtosis(returns),
        "min": returns.min(),
        "q0.1%": quantiles[0.001],
        "q1%": quantiles[0.01],
        "q5%": quantiles[0.05],
        "median": quantiles[0.5],
        "q95%": quantiles[0.95],
        "q99%": quantiles[0.99],
        "q99.9%": quantiles[0.999],
        "max": returns.max(),
        "zero_share": (returns == 0).mean(),
        "jarque_bera": jarque_bera_result.statistic,
        "jb_pvalue": jarque_bera_result.pvalue,
    }


def autocorrelation(series: pd.Series, max_lag: int) -> np.ndarray:
    """
    표본 자기상관계수 ρ(k), k = 1..max_lag.
        ρ(k) = Σ (x_t − x̄)(x_{t+k} − x̄) / Σ (x_t − x̄)²
    """
    values = series.dropna().to_numpy()
    deviations = values - values.mean()
    total_variance = np.dot(deviations, deviations)

    acf_values = []
    for lag in range(1, max_lag + 1):
        covariance_at_lag = np.dot(deviations[:-lag], deviations[lag:])
        acf_values.append(covariance_at_lag / total_variance)
    return np.array(acf_values)


def max_drawdown(prices: pd.Series) -> float:
    """최대낙폭 = 이전 고점 대비 가장 크게 떨어진 비율 (음수, 예: −0.82 = 82% 하락)."""
    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1
    return float(drawdown.min())


# ===========================================================================
# 3. 통계표 만들기
# ===========================================================================
def build_returns_by_horizon_table(returns_by_horizon: dict[str, pd.Series]) -> pd.DataFrame:
    """시간 단위(1m~1d)별 수익률 분포 요약표."""
    rows = {}
    for horizon_name, returns in returns_by_horizon.items():
        periods_per_year = MINUTES_PER_YEAR / HORIZONS_IN_MINUTES[horizon_name]
        rows[horizon_name] = describe_returns(returns, periods_per_year)
    return pd.DataFrame(rows).T


def build_yearly_summary_table(bars: pd.DataFrame, daily_log_return: pd.Series) -> pd.DataFrame:
    """연도별 가격·수익률·변동성·거래량 요약."""
    rows = {}
    for year, bars_in_year in bars.groupby(bars.index.year):
        daily_returns_in_year = daily_log_return[daily_log_return.index.year == year].dropna()
        has_enough_days = len(daily_returns_in_year) > 3

        rows[year] = {
            "first_close": bars_in_year["close"].iloc[0],
            "last_close": bars_in_year["close"].iloc[-1],
            "min_low": bars_in_year["low"].min(),
            "max_high": bars_in_year["high"].max(),
            "mean_close": bars_in_year["close"].mean(),
            "log_return": np.log(bars_in_year["close"].iloc[-1] / bars_in_year["close"].iloc[0]),
            "max_drawdown": max_drawdown(bars_in_year["close"]),
            # 같은 해의 변동성을 1분 수익률과 일별 수익률로 각각 연율화해 비교
            # (1분 기반이 더 크면 미시구조 잡음이 섞여 있다는 뜻)
            "ann_vol_1m": bars_in_year["log_return_1m"].std() * np.sqrt(MINUTES_PER_YEAR),
            "ann_vol_1d": daily_returns_in_year.std() * np.sqrt(DAYS_PER_YEAR),
            "kurtosis_1m": stats.kurtosis(bars_in_year["log_return_1m"].dropna()),
            "kurtosis_1d": stats.kurtosis(daily_returns_in_year) if has_enough_days else np.nan,
            "btc_volume": bars_in_year["volume"].sum(),
            "usd_volume": bars_in_year["usd_volume"].sum(),
            "mean_btc_per_min": bars_in_year["volume"].mean(),
        }
    return pd.DataFrame(rows).T.rename_axis("year")


def build_autocorrelation_tables(returns_by_horizon: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    수익률(r)과 절대수익률(|r|)의 자기상관표.
        - r 의 자기상관이 0 근처  → 방향은 예측하기 어렵다 (효율적 시장)
        - |r| 의 자기상관이 양수 → 큰 움직임 뒤에 큰 움직임이 온다 (변동성 군집)
    """
    returns_1m = returns_by_horizon["1m"]
    returns_1d = returns_by_horizon["1d"]

    acf_1m_table = pd.DataFrame(
        {
            "lag_min": range(1, ACF_MAX_LAG_MINUTES + 1),
            "acf_r": autocorrelation(returns_1m, ACF_MAX_LAG_MINUTES),
            "acf_abs_r": autocorrelation(returns_1m.abs(), ACF_MAX_LAG_MINUTES),
        }
    ).set_index("lag_min")

    acf_1d_table = pd.DataFrame(
        {
            "lag_day": range(1, ACF_MAX_LAG_DAYS + 1),
            "acf_r": autocorrelation(returns_1d, ACF_MAX_LAG_DAYS),
            "acf_abs_r": autocorrelation(returns_1d.abs(), ACF_MAX_LAG_DAYS),
        }
    ).set_index("lag_day")

    return acf_1m_table, acf_1d_table


def build_seasonality_tables(bars: pd.DataFrame, daily_log_return: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    시간대(UTC 시) · 요일별 평균 변동 크기와 거래량.
    감성 이벤트 효과를 볼 때 "그 시간대라서 원래 큰 것"을 걸러내는 기준으로 쓴다.
    """
    absolute_return_bp = bars["log_return_1m"].abs() * 1e4  # 1bp = 0.01%
    is_zero_volume = bars["volume"] == 0
    hour_of_day = bars.index.hour
    day_of_week = bars.index.dayofweek

    by_hour = pd.DataFrame(
        {
            "mean_abs_r_bp": absolute_return_bp.groupby(hour_of_day).mean(),
            "mean_btc_volume": bars["volume"].groupby(hour_of_day).mean(),
            "zero_volume_share": is_zero_volume.groupby(hour_of_day).mean(),
        }
    ).rename_axis("hour_utc")

    by_weekday = pd.DataFrame(
        {
            "mean_abs_r_bp": absolute_return_bp.groupby(day_of_week).mean(),
            "mean_btc_volume": bars["volume"].groupby(day_of_week).mean(),
            "mean_daily_r": daily_log_return.groupby(daily_log_return.index.dayofweek).mean(),
        }
    )
    by_weekday = by_weekday.set_axis(WEEKDAY_NAMES).rename_axis("weekday")

    return by_hour, by_weekday


def build_variance_ratio_table(returns_by_horizon: dict[str, pd.Series]) -> pd.DataFrame:
    """
    분산비(Variance Ratio) — 가격이 무작위 보행(random walk)을 따르는지 확인.
        VR(h) = Var(h분 수익률) / (h × Var(1분 수익률))
        VR = 1 : 무작위 보행
        VR < 1 : 단기 움직임이 되돌려짐 (bid-ask bounce 등 미시구조 잡음)
        VR > 1 : 추세가 이어짐 (모멘텀)
    """
    std_1m = returns_by_horizon["1m"].std()

    table = pd.DataFrame(
        {
            "horizon_min": list(HORIZONS_IN_MINUTES.values()),
            "std": [returns_by_horizon[name].std() for name in HORIZONS_IN_MINUTES],
            # 무작위 보행이라면 기대되는 표준편차 = 1분 표준편차 × √h
            "sqrt_h_scaled_std": [std_1m * np.sqrt(minutes) for minutes in HORIZONS_IN_MINUTES.values()],
        },
        index=list(HORIZONS_IN_MINUTES),
    )
    table["variance_ratio"] = (table["std"] / table["sqrt_h_scaled_std"]) ** 2
    return table


def build_extreme_moves_table(bars: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """1분 수익률 하위·상위 top_n개 (급락·급등 순간)."""
    returns_1m = bars["log_return_1m"].dropna()
    extreme_returns = pd.concat([returns_1m.nsmallest(top_n), returns_1m.nlargest(top_n)])

    table = extreme_returns.to_frame("log_return_1m")
    table["pct"] = 100 * (np.exp(table["log_return_1m"]) - 1)  # 로그수익률 → 단순 수익률(%)
    table["close"] = bars.loc[table.index, "close"]
    table["btc_volume"] = bars.loc[table.index, "volume"]
    return table.rename_axis("time")


def build_daily_realized_volatility_table(bars: pd.DataFrame, daily_close: pd.Series,
                                          daily_log_return: pd.Series) -> pd.DataFrame:
    """
    일별 실현변동성(Realized Volatility).
        RV_day = Σ (그날 1분 수익률)²        ← 하루 동안의 분산 추정치
        연율화 RV = √(RV_day × 365)
    """
    daily_realized_variance = (bars["log_return_1m"] ** 2).resample("1D").sum()

    return pd.DataFrame(
        {
            "close": daily_close,
            "log_return": daily_log_return,
            "realized_vol_ann": np.sqrt(daily_realized_variance * DAYS_PER_YEAR),
            "btc_volume": bars["volume"].resample("1D").sum(),
        }
    ).rename_axis("date")


def build_all_tables(bars: pd.DataFrame, returns_by_horizon: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """모든 통계표를 만들어 {파일이름: 표} 사전으로 돌려준다."""
    # 일별 종가·수익률은 여러 표에서 같이 쓰므로 한 번만 계산
    daily_close = bars["close"].resample("1D").last()
    daily_log_return = np.log(daily_close).diff()

    acf_1m_table, acf_1d_table = build_autocorrelation_tables(returns_by_horizon)
    by_hour_table, by_weekday_table = build_seasonality_tables(bars, daily_log_return)

    level_stats_table = bars[["close", "volume", "usd_volume"]].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T

    return {
        "returns_by_horizon": build_returns_by_horizon_table(returns_by_horizon),
        "yearly_summary": build_yearly_summary_table(bars, daily_log_return),
        "level_stats_1m": level_stats_table,
        "acf_1m": acf_1m_table,
        "acf_1d": acf_1d_table,
        "intraday_by_hour_utc": by_hour_table,
        "by_weekday": by_weekday_table,
        "variance_ratio": build_variance_ratio_table(returns_by_horizon),
        "extreme_1m_moves": build_extreme_moves_table(bars),
        "daily_realized_vol": build_daily_realized_volatility_table(bars, daily_close, daily_log_return),
    }


# ===========================================================================
# 4. 차트
# ===========================================================================
def plot_price_and_volume(bars: pd.DataFrame, daily_table: pd.DataFrame) -> None:
    """일별 종가(로그 축)와 주간 거래량. 단위가 달라 이중 축 대신 위아래 두 패널로 나눈다."""
    weekly_volume = bars["volume"].resample("1W").sum()

    fig, (price_ax, volume_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])

    price_ax.plot(daily_table.index, daily_table["close"], color=BLUE, lw=1.2)
    price_ax.set_yscale("log")  # 600달러 → 10만 달러: 로그 축이어야 초기 변화도 보임
    price_ax.set_ylabel("Close (USD, log)")
    price_ax.set_title("BTC/USD daily close & volume — Bitstamp, 2016-10 ~ 2026-09", loc="left")

    volume_ax.bar(weekly_volume.index, weekly_volume.values, width=5, color=BLUE, alpha=0.8)
    volume_ax.set_ylabel("Weekly volume (BTC)")

    save_figure(fig, FIGURES_DIR / "price_volume.png")


def plot_return_distribution(returns_by_horizon: dict[str, pd.Series]) -> None:
    """
    표준화 수익률의 분포를 정규분포와 비교 (y축 로그).
    로그 축에서 정규분포는 포물선, 실제 분포는 꼬리가 훨씬 높게 남는다 → 두꺼운 꼬리.
    """
    histogram_bins = np.linspace(-12, 12, 241)  # −12σ ~ +12σ, 0.1σ 간격

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, horizon_name in zip(axes, ["1m", "1h", "1d"]):
        returns = returns_by_horizon[horizon_name]

        # --- 표준화: (r − 평균) / 표준편차 → 단위가 σ ---
        standardized = (returns - returns.mean()) / returns.std()
        standardized = standardized[standardized.abs() < 12]

        # --- 히스토그램을 밀도로 계산 ---
        density, bin_edges = np.histogram(standardized, bins=histogram_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = np.where(density > 0, density, np.nan)  # 빈 구간은 선을 끊어 표시

        ax.plot(bin_centers, density, color=BLUE, marker="o", ms=2, lw=1, label="empirical")
        ax.plot(bin_centers, stats.norm.pdf(bin_centers), color=ORANGE, ls="--", label="N(0,1)")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2)
        ax.set_xlabel("σ")
        excess_kurtosis = stats.kurtosis(returns)
        ax.set_title(f"{horizon_name} log return (standardized)\nexcess kurtosis = {excess_kurtosis:.1f}", loc="left")

    axes[0].set_ylabel("density (log)")
    axes[0].legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "return_distribution.png")


def plot_qq(returns_by_horizon: dict[str, pd.Series]) -> None:
    """
    정규 QQ 플롯: 점이 대각선에서 벗어날수록 정규분포가 아니다.
    1분 수익률은 0이 많아(거래 없는 분) 0을 제외하고 그린다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, horizon_name in zip(axes, ["1m", "1d"]):
        returns = returns_by_horizon[horizon_name]
        if horizon_name == "1m":
            returns = returns[returns != 0]

        standardized = (returns - returns.mean()) / returns.std()
        (theoretical_quantiles, sample_quantiles), _ = stats.probplot(standardized, dist="norm")

        # 수백만 점을 다 그리면 느리므로 균등 간격으로 4,000개만 표시
        plot_index = np.unique(np.linspace(0, len(theoretical_quantiles) - 1, 4000).astype(int))
        ax.scatter(theoretical_quantiles[plot_index], sample_quantiles[plot_index], s=8, color=BLUE)

        axis_limit = max(abs(theoretical_quantiles).max(), 1)
        ax.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], color=ORANGE, ls="--", lw=1.2)

        suffix = " (non-zero)" if horizon_name == "1m" else ""
        ax.set_title(f"Normal QQ — {horizon_name} returns{suffix}", loc="left")
        ax.set_xlabel("theoretical quantile")
        ax.set_ylabel("sample quantile (σ)")
    save_figure(fig, FIGURES_DIR / "qq_plot.png")


def plot_autocorrelation(tables: dict[str, pd.DataFrame], returns_by_horizon: dict[str, pd.Series]) -> None:
    """r 과 |r| 의 자기상관. 회색 띠는 95% 신뢰구간(±1.96/√n): 띠 안이면 0과 구분되지 않음."""
    panels = [
        ("acf_1m", "1m", "minutes"),
        ("acf_1d", "1d", "days"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (table_name, horizon_name, lag_unit) in zip(axes, panels):
        acf_table = tables[table_name]
        sample_size = len(returns_by_horizon[horizon_name])
        confidence_band = 1.96 / np.sqrt(sample_size)

        ax.plot(acf_table.index, acf_table["acf_abs_r"], color=ORANGE, marker="o", ms=3, label="|r|")
        ax.plot(acf_table.index, acf_table["acf_r"], color=BLUE, marker="o", ms=3, label="r")
        ax.axhspan(-confidence_band, confidence_band, color=GRID, alpha=0.8, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(f"lag ({lag_unit})")
        ax.set_title(f"Autocorrelation — {horizon_name} returns", loc="left")
        ax.legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "autocorrelation.png")


def plot_intraday_seasonality(by_hour_table: pd.DataFrame) -> None:
    """UTC 시간대별 평균 |1분 수익률| 과 평균 거래량."""
    fig, (volatility_ax, volume_ax) = plt.subplots(1, 2, figsize=(12, 4))

    volatility_ax.bar(by_hour_table.index, by_hour_table["mean_abs_r_bp"], color=BLUE, width=0.8)
    volatility_ax.set_title("Mean |1m return| by hour (UTC, bp)", loc="left")
    volatility_ax.set_xlabel("hour (UTC)")

    volume_ax.bar(by_hour_table.index, by_hour_table["mean_btc_volume"], color=BLUE, width=0.8)
    volume_ax.set_title("Mean 1m volume by hour (UTC, BTC)", loc="left")
    volume_ax.set_xlabel("hour (UTC)")

    save_figure(fig, FIGURES_DIR / "intraday_seasonality.png")


def plot_realized_volatility(daily_table: pd.DataFrame) -> None:
    """일별 실현변동성(연율화)과 30일 이동평균."""
    realized_vol = daily_table["realized_vol_ann"]
    smoothed_realized_vol = realized_vol.rolling(RV_SMOOTHING_DAYS).mean()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(realized_vol.index, realized_vol, color=BLUE, alpha=0.35, lw=0.8, label="daily RV")
    ax.plot(realized_vol.index, smoothed_realized_vol, color=BLUE, lw=1.8, label=f"{RV_SMOOTHING_DAYS}d mean")
    ax.set_yscale("log")
    ax.set_ylabel("annualized RV (log)")
    ax.set_title("Daily realized volatility from 1m returns", loc="left")
    ax.legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "realized_volatility.png")


def plot_yearly_volatility(yearly_table: pd.DataFrame) -> None:
    """연도별 연율화 변동성: 1분 수익률 기반 vs 일별 수익률 기반."""
    bar_positions = np.arange(len(yearly_table))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(bar_positions - 0.2, yearly_table["ann_vol_1m"], width=bar_width, color=BLUE, label="from 1m returns")
    ax.bar(bar_positions + 0.2, yearly_table["ann_vol_1d"], width=bar_width, color=ORANGE, label="from 1d returns")
    ax.set_xticks(bar_positions, [str(year) for year in yearly_table.index])
    ax.set_ylabel("annualized volatility")
    ax.set_title("Annualized volatility by year (2016 = Q4 only, 2026 = Jan–Sep)", loc="left")
    ax.legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "yearly_volatility.png")


def plot_variance_ratio(variance_ratio_table: pd.DataFrame) -> None:
    """시간 단위별 분산비. 1보다 작을수록 단기 되돌림(미시구조 잡음)이 크다."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(variance_ratio_table["horizon_min"], variance_ratio_table["variance_ratio"], color=BLUE, marker="o", ms=6)
    ax.axhline(1, color=MUTED, ls="--", lw=1)  # 무작위 보행 기준선
    ax.set_xscale("log")
    ax.set_xticks(variance_ratio_table["horizon_min"], variance_ratio_table.index)
    ax.set_xlabel("horizon")
    ax.set_ylabel("VR = σ²(h) / (h·σ²(1m))")
    ax.set_title("Variance ratio vs horizon (1 = random walk)", loc="left")
    save_figure(fig, FIGURES_DIR / "variance_ratio.png")


def plot_all(bars: pd.DataFrame, returns_by_horizon: dict[str, pd.Series], tables: dict[str, pd.DataFrame]) -> None:
    plot_price_and_volume(bars, tables["daily_realized_vol"])
    plot_return_distribution(returns_by_horizon)
    plot_qq(returns_by_horizon)
    plot_autocorrelation(tables, returns_by_horizon)
    plot_intraday_seasonality(tables["intraday_by_hour_utc"])
    plot_realized_volatility(tables["daily_realized_vol"])
    plot_yearly_volatility(tables["yearly_summary"])
    plot_variance_ratio(tables["variance_ratio"])


# ===========================================================================
# 5. 요약 리포트 (results/descriptive/summary.md)
# ===========================================================================
def format_number(value, float_format: str = ".4g"):
    """
    표 안의 숫자를 읽기 쉽게 문자열로 바꾼다.
        정수 값         → 천 단위 쉼표   (5258880 → "5,258,880")
        1000 이상 실수  → 쉼표 + 정수    (13880.2 → "13,880")
        그 외           → 유효숫자 4자리 (0.0010937 → "0.001094")
    """
    is_number = isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
    if not is_number:
        return value
    if pd.isna(value):
        return ""
    if float(value).is_integer() and abs(value) >= 1:
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return format(value, float_format)


def to_markdown_table(table: pd.DataFrame, float_format: str = ".4g") -> str:
    formatted = table.map(lambda value: format_number(value, float_format))
    # disable_numparse: 이미 만든 문자열("13,880")을 tabulate가 다시 숫자로 바꾸지 않게
    return formatted.to_markdown(stralign="right", disable_numparse=True)


def write_summary(bars: pd.DataFrame, coverage: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    # --- 요약에 넣을 표·값 준비 ---
    returns_summary = tables["returns_by_horizon"][
        ["n", "mean", "std", "ann_vol", "skew", "excess_kurtosis", "min", "max", "zero_share", "jb_pvalue"]
    ]
    yearly_summary = tables["yearly_summary"][
        ["first_close", "last_close", "log_return", "max_drawdown", "ann_vol_1m", "ann_vol_1d", "kurtosis_1d", "usd_volume"]
    ]
    acf_1m = tables["acf_1m"]
    acf_1d = tables["acf_1d"]
    by_hour = tables["intraday_by_hour_utc"]

    total_missing_minutes = int(coverage["minutes_missing"].sum())
    total_zero_volume_minutes = int(coverage["zero_volume_minutes"].sum())

    # --- 마크다운 본문 ---
    markdown = f"""# BTC/USD 1분봉 10년 기초통계량

- 데이터: Bitstamp BTC/USD 1분봉, {bars.index[0]:%Y-%m-%d %H:%M} ~ {bars.index[-1]:%Y-%m-%d %H:%M} UTC
- 관측치: {len(bars):,}분 (누락 {total_missing_minutes:,}분, 거래량 0인 분 {total_zero_volume_minutes:,}분)
- 재현: `python collectors/collect_bitstamp_1m.py && python src/descriptive_stats.py`

## 1. 데이터 커버리지 (연도별)

{to_markdown_table(coverage, ".3g")}

> 2016~2017 초기에는 거래 없는 분(volume=0, 직전 종가 유지)의 비중이 높아 1분 수익률에 0이 많이 섞인다.
> 미시구조 분석 시 초기 구간의 1분 해상도 지표는 해석에 주의.

## 2. 시간 단위별 로그수익률 분포

{to_markdown_table(returns_summary)}

- 모든 시간 단위에서 Jarque–Bera 검정의 정규성 귀무가설 기각 (p≈0), 초과첨도는 고빈도일수록 크다 (fat tail).

## 3. 연도별 요약

{to_markdown_table(yearly_summary)}

## 4. 자기상관 (변동성 군집)

| | lag 1 | lag 5 | lag 30/60 |
|---|---|---|---|
| 1m r | {acf_1m['acf_r'].iloc[0]:.4f} | {acf_1m['acf_r'].iloc[4]:.4f} | {acf_1m['acf_r'].iloc[59]:.4f} (60m) |
| 1m \\|r\\| | {acf_1m['acf_abs_r'].iloc[0]:.4f} | {acf_1m['acf_abs_r'].iloc[4]:.4f} | {acf_1m['acf_abs_r'].iloc[59]:.4f} (60m) |
| 1d r | {acf_1d['acf_r'].iloc[0]:.4f} | {acf_1d['acf_r'].iloc[4]:.4f} | {acf_1d['acf_r'].iloc[29]:.4f} (30d) |
| 1d \\|r\\| | {acf_1d['acf_abs_r'].iloc[0]:.4f} | {acf_1d['acf_abs_r'].iloc[4]:.4f} | {acf_1d['acf_abs_r'].iloc[29]:.4f} (30d) |

- 1분 수익률의 lag-1 음의 자기상관은 bid-ask bounce 등 미시구조 잡음과 일치 (Roll 1984).
- 절대수익률의 자기상관은 장기간 양수로 유지 → 변동성 군집.

## 5. 분산비 (Variance Ratio)

{to_markdown_table(tables['variance_ratio'])}

## 6. 장중 계절성 (UTC)

- 평균 |1분 수익률| 최대: {by_hour['mean_abs_r_bp'].idxmax()}시 UTC ({by_hour['mean_abs_r_bp'].max():.2f}bp), 최소: {by_hour['mean_abs_r_bp'].idxmin()}시 UTC ({by_hour['mean_abs_r_bp'].min():.2f}bp)
- 평균 1분 거래량 최대: {by_hour['mean_btc_volume'].idxmax()}시 UTC, 최소: {by_hour['mean_btc_volume'].idxmin()}시 UTC

{to_markdown_table(tables['by_weekday'])}

## 7. 극단 1분 수익률 (하위·상위 10)

{to_markdown_table(tables['extreme_1m_moves'])}

## 그림

| | |
|---|---|
| ![](figures/price_volume.png) | ![](figures/realized_volatility.png) |
| ![](figures/return_distribution.png) | ![](figures/qq_plot.png) |
| ![](figures/autocorrelation.png) | ![](figures/intraday_seasonality.png) |
| ![](figures/yearly_volatility.png) | ![](figures/variance_ratio.png) |
"""
    (RESULTS_DIR / "summary.md").write_text(markdown, encoding="utf-8")


# ===========================================================================
# 실행
# ===========================================================================
def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 데이터 로드")
    bars, raw_bars = load_minute_bars()
    coverage = build_coverage_table(bars, raw_bars)

    print("2) 통계표 계산")
    returns_by_horizon = make_returns_by_horizon(bars)
    tables = build_all_tables(bars, returns_by_horizon)
    tables["coverage"] = coverage
    for table_name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{table_name}.csv", float_format="%.10g")

    print("3) 차트 저장")
    plot_all(bars, returns_by_horizon, tables)

    print("4) 요약 리포트 작성")
    write_summary(bars, coverage, tables)

    print(f"완료: 표 {len(tables)}개 → {TABLES_DIR}, 그림 → {FIGURES_DIR}, 요약 → {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
