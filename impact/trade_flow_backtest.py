"""
체결(aggTrades) 기반 주문흐름 백테스트 — 계산 모듈
=================================================

질문
    감성 없이, 체결 기록만으로 만든 '주문흐름 불균형'이 다음 구간 수익률을 예측하는가?
    예측한다면 거래비용을 낸 뒤에도 돈이 남는가?
    → 1월 감성 분석의 기준선("감성 없는 체결 기반 신호의 성과")을 만든다.

사전 고정한 분석 계획 (결과를 보기 전에 정함 — CLAUDE.md §2.2-5)
    - 봉 크기: 1분 · 5분 · 15분 · 60분 (겹치지 않는 봉)
    - 팩터는 모두 '봉 t 끝까지 알려진 정보'만 사용, 예측 대상은 봉 t+1 수익률
    - 검증: 워크포워드 — 분기마다 '그 이전 전체'로 재학습, 첫 12개월은 학습 전용
    - 모형: M0 수익률 0 / M1 과거 수익률만(릿지) / M2 과거 수익률 + 주문흐름(릿지)
    - 규칙 신호(학습 없음): 직전 봉 불균형 방향을 따른다(추종) / 반대로 간다(역추세)
    - 포지션은 -1(매도)·0·+1(매수), 비용은 포지션이 바뀐 양 × 편도 수수료
    - 비용 시나리오: 0bp(이론) · 1bp(대형 마켓메이커 수준) · 10bp(Binance 현물 기본 테이커)
    - 유의성: Clark–West(크기), Pesaran–Timmermann(방향), 무작위 신호 플라시보(샤프)

단위 약속
    수익률: bp (1bp = 0.01%) / 수량: BTC / 시각: UTC, 봉의 '시작 시각'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
BPS = 1e4
MINUTES_PER_DAY = 1440
DAYS_PER_YEAR = 365

RIDGE_ALPHA = 10.0                 # 릿지 규제 세기 (표준화된 팩터 기준, 과적합 방지)
CLIP_LOWER_QUANTILE = 0.005        # 학습 구간 0.5~99.5% 밖의 팩터 값은 잘라냄 (극단값 방어)
CLIP_UPPER_QUANTILE = 0.995
RETURN_LAGS = (0, 1, 2)            # ret_t, ret_{t-1}, ret_{t-2}
IMBALANCE_LAGS = (0, 1, 2)         # imb_t, imb_{t-1}, imb_{t-2}
IMBALANCE_SMOOTH_BARS = 12         # 최근 12개 봉 평균 불균형 (더 느린 주문흐름)
ACTIVITY_WINDOW_DAYS = 1           # 체결 건수 '평소 대비' 기준 창 (하루)
MIN_TRAIN_BARS = 1000              # 학습 표본이 이보다 적으면 그 분기는 건너뜀
TRAIN_ONLY_MONTHS = 12             # 맨 앞 12개월은 학습에만 사용
RETRAIN_FREQ = "QS"                # 분기 시작 시점마다 재학습
PLACEBO_DRAWS = 2000               # 무작위 신호 반복 횟수
PLACEBO_SEED = 20261011            # 재현성을 위한 난수 고정


# ===========================================================================
# 1. 체결 → 1분 / N분 봉
# ===========================================================================
def aggregate_trades_to_minute_bars(trades: pd.DataFrame) -> pd.DataFrame:
    """
    체결 기록(하루치)을 1분 봉으로 요약한다.

    입력 컬럼: price, quantity, is_buyer_maker, trade_time(UTC)
    출력 컬럼 (index = 분 시작 시각):
        close      그 분의 마지막 체결가
        buy_qty    시장가 매수가 일으킨 체결량 (is_buyer_maker=False)
        sell_qty   시장가 매도가 일으킨 체결량 (is_buyer_maker=True)
        n_trades   체결 건수 (aggTrades 행 수)
        usd_volume 거래대금 = Σ 가격 × 수량
    """
    is_sell_initiated = trades["is_buyer_maker"].to_numpy()
    quantity = trades["quantity"].to_numpy()
    price = trades["price"].to_numpy()

    work = pd.DataFrame(
        {
            "minute": trades["trade_time"].dt.floor("1min"),
            "price": price,
            "buy_qty": np.where(is_sell_initiated, 0.0, quantity),
            "sell_qty": np.where(is_sell_initiated, quantity, 0.0),
            "usd_volume": price * quantity,
        }
    )
    grouped = work.groupby("minute")
    minute_bars = pd.DataFrame(
        {
            "close": grouped["price"].last(),
            "buy_qty": grouped["buy_qty"].sum(),
            "sell_qty": grouped["sell_qty"].sum(),
            "n_trades": grouped["price"].size(),
            "usd_volume": grouped["usd_volume"].sum(),
        }
    )
    return minute_bars


def build_bar_panel(minute_bars: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """
    1분 봉을 bar_minutes 분 봉으로 합치고 팩터·예측 대상을 만든다.

    - ret_bp          = 100 × (봉 종가 / 직전 봉 종가 - 1) × 100  [bp, 로그 수익률]
    - imbalance       = (매수량 - 매도량) / (매수량 + 매도량)  ∈ [-1, 1]
                         → 이 봉에서 시장가 매수가 우세했으면 +, 매도 우세면 -
    - log_activity_dev= log(체결 건수) - 최근 1일 평균 log(체결 건수)  → 평소보다 활발한가
    - next_ret_bp     = 다음 봉 수익률 (예측 대상). 마지막 봉은 NaN
    """
    full_index = pd.date_range(minute_bars.index.min(), minute_bars.index.max(), freq="1min")
    minute_bars = minute_bars.reindex(full_index)

    # 거래가 없던 분(점검 등)은 합계 0, 종가는 비워 둔다
    for sum_column in ("buy_qty", "sell_qty", "n_trades", "usd_volume"):
        minute_bars[sum_column] = minute_bars[sum_column].fillna(0.0)

    resampler = minute_bars.resample(f"{bar_minutes}min", label="left", closed="left")
    bars = pd.DataFrame(
        {
            "close": resampler["close"].last(),
            "buy_qty": resampler["buy_qty"].sum(),
            "sell_qty": resampler["sell_qty"].sum(),
            "n_trades": resampler["n_trades"].sum(),
            "usd_volume": resampler["usd_volume"].sum(),
        }
    )
    # 봉 안에 체결이 하나도 없으면(거래소 점검) 그 봉은 결측으로 본다
    bars = bars[bars["n_trades"] > 0].copy()

    # --- 수익률 (연속한 봉일 때만 유효: 사이에 빠진 봉이 있으면 NaN) ---
    bar_delta = pd.Timedelta(minutes=bar_minutes)
    previous_close = bars["close"].shift(1)
    is_consecutive = bars.index.to_series().diff() == bar_delta
    bars["ret_bp"] = np.log(bars["close"] / previous_close) * BPS
    bars.loc[~is_consecutive, "ret_bp"] = np.nan

    # --- 주문 불균형: (매수-매도)/(매수+매도) ---
    total_qty = bars["buy_qty"] + bars["sell_qty"]
    bars["imbalance"] = (bars["buy_qty"] - bars["sell_qty"]) / total_qty

    # --- 체결 활발도: 평소(하루 평균) 대비 ---
    bars_per_day = max(int(MINUTES_PER_DAY * ACTIVITY_WINDOW_DAYS / bar_minutes), 2)
    log_trades = np.log(bars["n_trades"])
    rolling_mean = log_trades.rolling(bars_per_day, min_periods=bars_per_day // 2).mean()
    bars["log_activity_dev"] = log_trades - rolling_mean.shift(1)  # 현재 봉은 평균에서 제외(누출 방지)

    # --- 예측 대상: 다음 봉 수익률 (연속한 봉일 때만) ---
    bars["next_ret_bp"] = bars["ret_bp"].shift(-1)
    return bars


def build_factor_table(bars: pd.DataFrame) -> pd.DataFrame:
    """봉 t까지 알려진 정보로 만든 팩터 표. 컬럼 이름이 곧 의미."""
    factors = pd.DataFrame(index=bars.index)
    for lag in RETURN_LAGS:
        factors[f"ret_lag{lag}"] = bars["ret_bp"].shift(lag)
    for lag in IMBALANCE_LAGS:
        factors[f"imb_lag{lag}"] = bars["imbalance"].shift(lag)
    factors["imb_smooth"] = bars["imbalance"].rolling(IMBALANCE_SMOOTH_BARS).mean()
    factors["log_activity_dev"] = bars["log_activity_dev"]
    return factors


RETURN_FACTOR_NAMES = [f"ret_lag{lag}" for lag in RETURN_LAGS]
FLOW_FACTOR_NAMES = [f"imb_lag{lag}" for lag in IMBALANCE_LAGS] + ["imb_smooth", "log_activity_dev"]


# ===========================================================================
# 2. 워크포워드 예측
# ===========================================================================
def fit_predict_ridge(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> np.ndarray:
    """학습 구간 기준으로 팩터를 절단·표준화한 뒤 릿지 회귀로 예측한다."""
    lower = train_x.quantile(CLIP_LOWER_QUANTILE)
    upper = train_x.quantile(CLIP_UPPER_QUANTILE)
    train_clipped = train_x.clip(lower, upper, axis=1)
    test_clipped = test_x.clip(lower, upper, axis=1)

    mean = train_clipped.mean()
    std = train_clipped.std().replace(0, 1.0)
    train_scaled = (train_clipped - mean) / std
    test_scaled = (test_clipped - mean) / std

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(train_scaled, train_y)
    return model.predict(test_scaled)


def walk_forward_forecast(bars: pd.DataFrame, factor_names: list[str]) -> pd.Series:
    """
    분기마다 '그 이전 전체 데이터'로만 학습해 그 분기를 예측한다 (미래 정보 누출 없음).
    맨 앞 TRAIN_ONLY_MONTHS 개월은 학습 전용이라 예측하지 않는다.
    """
    factors = build_factor_table(bars)[factor_names]
    target = bars["next_ret_bp"]
    usable = factors.join(target).dropna()

    first_test_time = usable.index.min() + pd.DateOffset(months=TRAIN_ONLY_MONTHS)
    quarter_starts = pd.date_range(
        first_test_time.normalize(), usable.index.max(), freq=RETRAIN_FREQ, tz=usable.index.tz
    )
    boundaries = [first_test_time] + [t for t in quarter_starts if t > first_test_time]
    boundaries.append(usable.index.max() + pd.Timedelta(seconds=1))

    forecasts = []
    for quarter_start, quarter_end in zip(boundaries[:-1], boundaries[1:]):
        train = usable[usable.index < quarter_start]
        test = usable[(usable.index >= quarter_start) & (usable.index < quarter_end)]
        if len(train) < MIN_TRAIN_BARS or test.empty:
            continue
        predicted = fit_predict_ridge(train[factor_names], train["next_ret_bp"], test[factor_names])
        forecasts.append(pd.Series(predicted, index=test.index))
    return pd.concat(forecasts)


# ===========================================================================
# 3. 평가: 예측력 검정
# ===========================================================================
def out_of_sample_r2(actual: pd.Series, forecast: pd.Series) -> float:
    """표본 외 R² = 1 - Σ(실제-예측)² / Σ(실제-0)²  (기준 예측 = 수익률 0)."""
    squared_error_model = ((actual - forecast) ** 2).sum()
    squared_error_zero = (actual ** 2).sum()
    return 1.0 - squared_error_model / squared_error_zero


def clark_west_p_value(actual: pd.Series, forecast: pd.Series) -> tuple[float, float]:
    """
    Clark–West (2007) 검정: 모형이 '수익률 0'보다 유의하게 정확한가 (단측).
    조정 오차 f = e0² - e1² + (예측)²,  H0: E[f] = 0 vs H1: E[f] > 0
    """
    error_zero = actual
    error_model = actual - forecast
    adjusted_loss_gap = error_zero ** 2 - error_model ** 2 + forecast ** 2
    mean_gap = adjusted_loss_gap.mean()
    standard_error = adjusted_loss_gap.std(ddof=1) / np.sqrt(len(adjusted_loss_gap))
    t_value = mean_gap / standard_error
    p_value = 1.0 - stats.norm.cdf(t_value)
    return float(t_value), float(p_value)


def direction_test(actual: pd.Series, forecast: pd.Series) -> dict:
    """
    방향 적중률과 Pesaran–Timmermann (1992) 검정.
    예측과 실제가 독립일 때의 기대 적중률 P* 대비 실제 적중률 P가 유의하게 높은가.
    """
    actual_up = actual > 0
    forecast_up = forecast > 0
    hit_rate = float((actual_up == forecast_up).mean())

    p_actual_up = actual_up.mean()
    p_forecast_up = forecast_up.mean()
    expected_hit = p_actual_up * p_forecast_up + (1 - p_actual_up) * (1 - p_forecast_up)

    n = len(actual)
    var_hit = expected_hit * (1 - expected_hit) / n
    var_expected = (
        (2 * p_actual_up - 1) ** 2 * p_forecast_up * (1 - p_forecast_up) / n
        + (2 * p_forecast_up - 1) ** 2 * p_actual_up * (1 - p_actual_up) / n
        + 4 * p_actual_up * p_forecast_up * (1 - p_actual_up) * (1 - p_forecast_up) / n ** 2
    )
    denominator = np.sqrt(max(var_hit - var_expected, 1e-18))
    statistic = (hit_rate - expected_hit) / denominator
    p_value = 1.0 - stats.norm.cdf(statistic)
    return {"hit_rate": hit_rate, "hit_rate_if_independent": float(expected_hit), "pt_p": float(p_value)}


# ===========================================================================
# 4. 전략 성과
# ===========================================================================
def positions_from_forecast(forecast: pd.Series) -> pd.Series:
    """예측 부호대로 매수(+1)·매도(-1). 예측이 정확히 0이면 관망(0)."""
    return np.sign(forecast).astype(float)


def positions_from_imbalance(imbalance: pd.Series, follow: bool) -> pd.Series:
    """
    학습 없는 규칙 신호.
    follow=True : 직전 봉에서 매수 우세(imbalance>0)면 매수 — 주문흐름 추종(모멘텀)
    follow=False: 반대로 — 주문흐름 역추세(반전)
    """
    direction = np.sign(imbalance).astype(float)
    return direction if follow else -direction


def strategy_pnl_bp(position: pd.Series, next_ret_bp: pd.Series, one_way_cost_bp: float) -> pd.Series:
    """
    봉별 손익(bp) = 포지션 × 다음 봉 수익률 - 편도수수료 × |포지션 변화량|.
    포지션을 새로 잡거나(0→±1) 뒤집으면(+1→-1은 변화량 2) 그만큼 수수료를 낸다.
    """
    aligned = pd.concat([position.rename("pos"), next_ret_bp.rename("ret")], axis=1).dropna()
    traded_amount = aligned["pos"].diff().abs().fillna(aligned["pos"].abs())
    return aligned["pos"] * aligned["ret"] - one_way_cost_bp * traded_amount


def annualized_sharpe(pnl_bp: pd.Series, bars_per_year: float) -> float:
    """연율화 샤프 = 평균 / 표준편차 × √(연간 봉 수). 무위험수익률 0 가정."""
    std = pnl_bp.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(pnl_bp.mean() / std * np.sqrt(bars_per_year))


def max_drawdown_pct(pnl_bp: pd.Series) -> float:
    """누적 손익(단리, 자본 1 기준)의 고점 대비 최대 낙폭 [%]."""
    cumulative = pnl_bp.cumsum() / BPS
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    return float(drawdown.min() * 100)


def summarize_strategy(position: pd.Series, next_ret_bp: pd.Series, bar_minutes: int,
                       one_way_cost_bp: float) -> dict:
    """한 전략·한 비용 시나리오의 성과 요약."""
    bars_per_year = MINUTES_PER_DAY * DAYS_PER_YEAR / bar_minutes
    pnl = strategy_pnl_bp(position, next_ret_bp, one_way_cost_bp)
    traded = position.reindex(pnl.index).diff().abs().fillna(position.reindex(pnl.index).abs())
    days = len(pnl) * bar_minutes / MINUTES_PER_DAY
    return {
        "mean_bp_per_bar": float(pnl.mean()),
        "sharpe": annualized_sharpe(pnl, bars_per_year),
        "total_return_pct": float(pnl.sum() / BPS * 100),
        "max_drawdown_pct": max_drawdown_pct(pnl),
        "trades_per_day": float(traded.sum() / days),
        "n_bars": int(len(pnl)),
    }


def placebo_sharpe_p_value(position: pd.Series, next_ret_bp: pd.Series, bar_minutes: int,
                           one_way_cost_bp: float) -> tuple[float, float]:
    """
    플라시보: 실제 신호의 '포지션 순서'를 무작위로 섞어 같은 분석을 반복한다.
    → 섞은 신호의 샤프 분포에서 실제 샤프가 얼마나 극단적인지 (단측 p값).
    포지션 값의 구성비(롱/숏/관망 비율)는 그대로 두고 시점만 무작위로 만든다.
    """
    bars_per_year = MINUTES_PER_DAY * DAYS_PER_YEAR / bar_minutes
    aligned = pd.concat([position.rename("pos"), next_ret_bp.rename("ret")], axis=1).dropna()
    pos_values = aligned["pos"].to_numpy()
    ret_values = aligned["ret"].to_numpy()

    def sharpe_of(positions: np.ndarray) -> float:
        traded = np.abs(np.diff(positions, prepend=0.0))
        pnl = positions * ret_values - one_way_cost_bp * traded
        std = pnl.std(ddof=1)
        return pnl.mean() / std * np.sqrt(bars_per_year) if std > 0 else np.nan

    actual_sharpe = sharpe_of(pos_values)
    random_generator = np.random.default_rng(PLACEBO_SEED)
    placebo_sharpes = np.array(
        [sharpe_of(random_generator.permutation(pos_values)) for _ in range(PLACEBO_DRAWS)]
    )
    p_value = float((np.sum(placebo_sharpes >= actual_sharpe) + 1) / (PLACEBO_DRAWS + 1))
    return float(actual_sharpe), p_value
