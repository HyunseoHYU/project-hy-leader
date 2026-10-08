"""
베이스라인 팩터 모델 — 감성지수 없이 유동성·가격을 얼마나 예측할 수 있는가
====================================================================

왜 필요한가
    1월 핵심 분석(감성 충격 → 호가창 유동성)에서 "감성 때문에 변했다"고 말하려면,
    감성이 없어도 원래 예상되던 수준(기대 유동성)을 먼저 알아야 한다.
        비정상 유동성(abnormal liquidity) = 실제 유동성 − 베이스라인 모형의 예측값
    가격도 마찬가지로, 감성 없이도 잡히는 예측력(모멘텀·주문흐름 등)을 먼저 걷어내야
    감성의 '추가' 예측력을 정직하게 잴 수 있다.

이 모듈의 구성 (실행은 src/analyze_baseline_factors.py)
    1. build_hourly_panel      1분봉 → 1시간 단위 변수 (수익률, 실현변동성, Amihud, Kyle's λ, 주문 불균형 …)
    2. build_factor_table      t시점까지의 정보만으로 만든 팩터 + t+1시점 예측 대상
    3. 모형 4종                M0 단순 기준 / M1 자기 과거값(HAR) / M2 전체 팩터 선형 / M3 전체 팩터 부스팅
    4. walk_forward_forecast   연도별로 '그 이전 데이터로만' 학습 → 그 해 예측 (미래 정보 누출 방지)
    5. 평가                    표본 외 R², Diebold–Mariano, Clark–West, 방향 적중률, 수수료 반영 전략 성과
    6. ols_newey_west          전체 표본 회귀 + 자기상관·이분산 강건(HAC) 표준오차 → 팩터 해석용

단위 약속
    수익률·변동성: bp (1bp = 0.01%) / 거래대금: 백만 달러 / 시각: UTC, 시간봉의 '시작 시각'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

import liquidity_metrics as lm

BPS = 1e4
HOURS_PER_YEAR = 365 * 24
MIN_MINUTES_PER_HOUR = 50  # 60분 중 이보다 적게 남은 시간(거래소 점검 등)은 결측 처리


# ===========================================================================
# 1. 1분봉 → 1시간 패널
# ===========================================================================
def build_hourly_panel(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """
    1분봉(liquidity_metrics.prepare_minute_bars 결과)을 1시간 단위로 모은다.

    만드는 변수 (모두 '그 1시간 동안'의 값)
        return_bps        시간 수익률 = ln(종가_t / 종가_{t−1}) × 10⁴
        rv_bps            실현변동성 = √Σ(1분 수익률²)                    (bp)
        range_bps         고가·저가 범위 = ln(고가/저가) × 10⁴             (Parkinson 변동성 재료)
        usd_volume_musd   거래대금 (백만 달러)
        order_imbalance   주문 불균형 = (매수 주도량 − 매도 주도량) / 거래량   (−1 ~ +1)
        avg_trade_btc     평균 체결 크기 = 거래량 / 체결 수                  (BTC)
        amihud            Amihud 비유동성 = Σ|1분 수익률| / 거래대금         (bp per $1M)
                          ※ |시간 수익률| 대신 1분 수익률 절댓값의 합(가격 경로 길이)을 써서 잡음을 줄임
        kyle_lambda       Kyle's λ = 1분 수익률(bp)을 순매수량(BTC)에 회귀한 기울기 × 100  (bp per 100 BTC)
    """
    bars = minute_bars.copy()
    bars["hour"] = bars.index.floor("h")
    bars["abs_return_bps"] = bars["return_bps"].abs()
    bars["squared_return_bps"] = bars["return_bps"] ** 2

    by_hour = bars.groupby("hour")
    hourly = pd.DataFrame(
        {
            "close": by_hour["close"].last(),
            "high": by_hour["high"].max(),
            "low": by_hour["low"].min(),
            "btc_volume": by_hour["volume"].sum(),
            "buy_volume": by_hour["buy_volume"].sum(),
            "usd_volume_musd": by_hour["usd_volume"].sum() / 1e6,
            "num_trades": by_hour["num_trades"].sum(),
            "path_length_bps": by_hour["abs_return_bps"].sum(),
            "realized_variance_bps2": by_hour["squared_return_bps"].sum(),
            "minutes_available": by_hour.size(),
        }
    )

    # --- 빈 시간도 행을 만들어 시간 간격을 일정하게 (shift/rolling 이 '1시간 전'을 정확히 가리키도록) ---
    full_hour_grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_hour_grid)
    hourly.index.name = "hour"

    # --- 파생 변수 ---
    hourly["return_bps"] = np.log(hourly["close"]).diff() * BPS
    hourly["rv_bps"] = np.sqrt(hourly["realized_variance_bps2"])
    hourly["range_bps"] = np.log(hourly["high"] / hourly["low"]) * BPS
    sell_volume = hourly["btc_volume"] - hourly["buy_volume"]
    hourly["order_imbalance"] = (hourly["buy_volume"] - sell_volume) / hourly["btc_volume"]
    hourly["avg_trade_btc"] = hourly["btc_volume"] / hourly["num_trades"]
    hourly["amihud"] = hourly["path_length_bps"] / hourly["usd_volume_musd"]

    kyle = lm.regression_slope_by_group(minute_bars["signed_volume"], minute_bars["return_bps"],
                                        minute_bars.index.floor("h").to_series(index=minute_bars.index))
    hourly["kyle_lambda"] = kyle["slope"] * 100

    # --- 데이터가 많이 빈 시간은 전부 결측으로 ---
    is_incomplete_hour = hourly["minutes_available"].fillna(0) < MIN_MINUTES_PER_HOUR
    hourly.loc[is_incomplete_hour, hourly.columns.difference(["minutes_available"])] = np.nan
    return hourly


# ===========================================================================
# 2. 팩터 & 예측 대상
# ===========================================================================
LIQUIDITY_TARGETS = {
    # 이름: (원천 컬럼, 변환)   — 로그 변환: 분포가 한쪽으로 매우 치우친 변수라 비율 변화를 보는 게 자연스러움
    "log_amihud": ("amihud", "log"),
    "log_rv": ("rv_bps", "log"),
    "kyle_lambda": ("kyle_lambda", "winsorize"),  # 음수도 나올 수 있어 로그 대신 극단값만 1%/99%로 자름
}
PRICE_TARGET = "return_bps"


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """상·하위 극단값을 분위수 경계로 자른다 (몇 개의 이상치가 회귀를 좌우하지 않게)."""
    return series.clip(series.quantile(lower), series.quantile(upper))


def har_components(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    HAR(Heterogeneous AutoRegressive, Corsi 2009) 구성요소: 서로 다른 시간 규모의 과거 평균.
        _1h   직전 1시간 값          (초단기 참여자)
        _24h  최근 24시간 평균       (일 단위 참여자)
        _168h 최근 1주(168시간) 평균 (주 단위 참여자)
    변동성·유동성은 이 세 가지로 대부분 설명된다는 것이 정형화된 사실.
    """
    return pd.DataFrame(
        {
            f"{prefix}_1h": series,
            f"{prefix}_24h": series.rolling(24, min_periods=18).mean(),
            f"{prefix}_168h": series.rolling(168, min_periods=120).mean(),
        }
    )


def build_factor_table(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    한 행 = 시각 t.  팩터는 t시간이 끝난 시점까지 알 수 있는 정보만, 대상(target_*)은 t+1시간의 값.

    반환:
        table          팩터 + 대상 컬럼
        factor_groups  {"own_<대상>": 그 대상의 자기 과거값 팩터, "seasonal": 시간 패턴, "all": 전체 팩터}
    """
    table = pd.DataFrame(index=hourly.index)
    factor_groups: dict[str, list[str]] = {}

    # --- (1) 유동성·변동성 변수의 HAR 구성요소 ---
    transformed = {
        "log_amihud": np.log(hourly["amihud"].where(hourly["amihud"] > 0)),
        "log_rv": np.log(hourly["rv_bps"].where(hourly["rv_bps"] > 0)),
        "kyle_lambda": winsorize(hourly["kyle_lambda"]),
    }
    for name, series in transformed.items():
        har = har_components(series, name)
        table = table.join(har)
        factor_groups[f"own_{name}"] = list(har.columns)

    # --- (2) 가격 팩터: 반전(reversal)·모멘텀 ---
    #     과거 수익률 합계. 1시간은 '단기 반전', 1주는 '모멘텀'을 잡는 용도
    returns = hourly["return_bps"]
    table["ret_1h"] = returns
    table["ret_24h"] = returns.rolling(24, min_periods=18).sum()
    table["ret_168h"] = returns.rolling(168, min_periods=120).sum()
    factor_groups["own_return"] = ["ret_1h", "ret_24h", "ret_168h"]

    # --- (3) 주문흐름 팩터 ---
    table["oib_1h"] = hourly["order_imbalance"]
    table["oib_24h"] = hourly["order_imbalance"].rolling(24, min_periods=18).mean()

    # --- (4) 거래 활동 팩터 ---
    log_volume = np.log(hourly["usd_volume_musd"].where(hourly["usd_volume_musd"] > 0))
    # 비정상 거래량 = 이번 시간 로그 거래대금 − 최근 1주 평균  (평소보다 몇 배 많은가)
    table["abnormal_volume"] = log_volume - log_volume.rolling(168, min_periods=120).mean()
    table["log_avg_trade_btc"] = np.log(hourly["avg_trade_btc"].where(hourly["avg_trade_btc"] > 0))
    table["log_range"] = np.log(hourly["range_bps"].where(hourly["range_bps"] > 0))

    # --- (5) 시간 패턴: '예측하려는 시간(t+1)'의 시각·요일 — 미리 알 수 있는 정보라 사용 가능 ---
    next_hour = table.index + pd.Timedelta(hours=1)
    hour_angle = 2 * np.pi * next_hour.hour / 24
    table["hour_sin"] = np.sin(hour_angle)  # 하루 주기를 원 위의 좌표로 표현 (23시와 0시가 가깝게)
    table["hour_cos"] = np.cos(hour_angle)
    table["hour_sin2"] = np.sin(2 * hour_angle)  # 반나절 주기 (아시아·미국 세션 두 봉우리)
    table["hour_cos2"] = np.cos(2 * hour_angle)
    table["is_weekend"] = (next_hour.dayofweek >= 5).astype(float)
    table["hour_of_day"] = next_hour.hour.astype(float)  # 부스팅 모형 전용 (선형 모형에는 sin/cos 사용)
    factor_groups["seasonal"] = ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2", "is_weekend"]

    # --- 전체 팩터 목록 (선형 모형용; hour_of_day 는 부스팅에만 추가) ---
    all_factors = []
    for group_name, columns in factor_groups.items():
        all_factors += columns
    all_factors += ["oib_1h", "oib_24h", "abnormal_volume", "log_avg_trade_btc", "log_range"]
    factor_groups["all"] = list(dict.fromkeys(all_factors))  # 중복 제거, 순서 유지

    # --- (6) 예측 대상: 한 시간 뒤의 값 (shift(-1)) ---
    for name in LIQUIDITY_TARGETS:
        table[f"target_{name}"] = transformed[name].shift(-1)
    table["target_return_bps"] = returns.shift(-1)

    return table, factor_groups


# ===========================================================================
# 3. 모형
# ===========================================================================
class PersistenceModel:
    """M0 (유동성): '다음 시간 = 이번 시간' — 가장 단순한 기준. 이것도 못 이기면 모형이 쓸모없다."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, features: pd.DataFrame, target: pd.Series):
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features[self.column].to_numpy()


class ZeroModel:
    """M0 (가격): '다음 시간 수익률 = 0' — 효율적 시장(랜덤워크) 가설에 해당하는 기준."""

    def fit(self, features: pd.DataFrame, target: pd.Series):
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features))


class StandardizedOLS:
    """
    선형회귀 (M1, M2).
    학습 구간의 평균·표준편차로 팩터를 표준화한 뒤 최소제곱 추정 → 계수 크기를 서로 비교할 수 있다.
    예측할 때도 '학습 구간'의 평균·표준편차를 써야 미래 정보가 섞이지 않는다.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, features: pd.DataFrame, target: pd.Series):
        x = features[self.columns]
        self.mean_ = x.mean()
        self.std_ = x.std().replace(0, 1)  # 상수 컬럼(예: 학습 구간에 주말이 없는 경우) 방어
        design = np.column_stack([np.ones(len(x)), ((x - self.mean_) / self.std_).to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(design, target.to_numpy(), rcond=None)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        x = (features[self.columns] - self.mean_) / self.std_
        design = np.column_stack([np.ones(len(x)), x.to_numpy()])
        return design @ self.coef_


class BoostingModel:
    """
    M3: 그래디언트 부스팅 (비선형·상호작용을 자동으로 잡는 트리 앙상블).
    선형 모형보다 나은지 확인하는 용도. 과적합을 막기 위해 보수적으로 설정:
        얕은 트리(잎 15개), 작은 학습률, 큰 최소 표본(잎당 200시간), 학습 구간 끝 10%로 조기 종료

    anchor_column (유동성 대상에서 사용)
        트리 모형은 학습 때 본 값의 범위 밖을 예측하지 못한다. 그런데 Amihud 비유동성은 해마다 낮아져서
        2021년 테스트 구간의 67%가 학습 구간 하위 1%보다도 낮았다 → 부스팅이 M0보다도 못한 결과(R² −17%).
        그래서 수준 대신 '최근 1주 평균 대비 편차'(대상 − anchor)를 학습하고, 예측할 때 anchor를 다시 더한다.
        편차는 연도가 바뀌어도 분포가 안정적이라 트리가 다룰 수 있다.
    """

    def __init__(self, columns: list[str], anchor_column: str | None = None, random_state: int = 42):
        self.columns = columns
        self.anchor_column = anchor_column
        self.random_state = random_state

    def _anchor(self, features: pd.DataFrame) -> np.ndarray:
        if self.anchor_column is None:
            return np.zeros(len(features))
        return features[self.anchor_column].to_numpy()

    def fit(self, features: pd.DataFrame, target: pd.Series):
        self.model_ = HistGradientBoostingRegressor(
            max_leaf_nodes=15,
            learning_rate=0.05,
            min_samples_leaf=200,
            l2_regularization=1.0,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state,  # 재현성
        )
        deviation_from_anchor = target.to_numpy() - self._anchor(features)
        self.model_.fit(features[self.columns], deviation_from_anchor)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(features[self.columns]) + self._anchor(features)


def make_models(target_name: str, factor_groups: dict[str, list[str]]) -> dict:
    """대상별 모형 4종 (M0~M3)을 만든다."""
    boosting_columns = factor_groups["all"] + ["hour_of_day"]

    if target_name == "return_bps":
        own_factors = factor_groups["own_return"]
        benchmark = ZeroModel()
        boosting_anchor = None  # 수익률은 이미 0 근처에서 안정적
    else:
        own_factors = factor_groups[f"own_{target_name}"]
        benchmark = PersistenceModel(f"{target_name}_1h")
        boosting_anchor = f"{target_name}_168h"  # 최근 1주 평균 대비 편차를 학습

    return {
        "M0_benchmark": benchmark,
        "M1_own_HAR": StandardizedOLS(own_factors + factor_groups["seasonal"]),
        "M2_all_linear": StandardizedOLS(factor_groups["all"]),
        "M3_all_boosting": BoostingModel(boosting_columns, anchor_column=boosting_anchor),
    }


# ===========================================================================
# 4. 워크포워드(walk-forward) 예측
# ===========================================================================
def walk_forward_forecast(table: pd.DataFrame, target_name: str, factor_groups: dict[str, list[str]],
                          train_start: str, test_years: list[int]) -> pd.DataFrame:
    """
    시간 순서를 지키는 표본 외 예측.
        테스트 연도 Y 마다:  학습 = [train_start, Y년 1월 1일)   →   예측 = Y년 전체
    즉 어떤 예측도 그 시점 이후의 데이터로 학습되지 않는다 (KOSPI 프로젝트의 시점 고정 원칙과 동일).

    반환: 행 = 시각, 컬럼 = actual + 모형별 예측값
    """
    target_column = f"target_{target_name}"
    needed_columns = list(dict.fromkeys(factor_groups["all"] + ["hour_of_day", target_column]))
    usable = table[needed_columns].dropna()
    usable = usable[usable.index >= train_start]

    yearly_forecasts = []
    for year in test_years:
        year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        year_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        train = usable[usable.index < year_start]
        test = usable[(usable.index >= year_start) & (usable.index < year_end)]
        if test.empty:
            continue

        forecasts = pd.DataFrame({"actual": test[target_column]}, index=test.index)
        for model_name, model in make_models(target_name, factor_groups).items():
            model.fit(train, train[target_column])
            forecasts[model_name] = model.predict(test)
        yearly_forecasts.append(forecasts)

    return pd.concat(yearly_forecasts)


# ===========================================================================
# 5. 평가
# ===========================================================================
def newey_west_variance_of_mean(series: np.ndarray, lags: int) -> float:
    """
    시계열 평균의 분산을 Newey–West(1987) 방식으로 추정한다.
    시간봉 오차는 서로 자기상관이 있어서(오늘 크게 틀리면 다음 시간도 틀리기 쉬움) 일반 분산 공식은 과소추정한다.
        Var(평균) = [γ₀ + 2 Σ_{k=1..L} (1 − k/(L+1)) γ_k] / n
    """
    values = series - series.mean()
    sample_size = len(values)
    long_run_variance = np.dot(values, values) / sample_size
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        autocovariance = np.dot(values[:-lag], values[lag:]) / sample_size
        long_run_variance += 2 * weight * autocovariance
    return long_run_variance / sample_size


def out_of_sample_r2(actual: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray) -> float:
    """
    표본 외 R² (Campbell & Thompson 2008) = 1 − Σ(실제 − 모형)² / Σ(실제 − 기준)²
        > 0 : 기준 모형보다 오차가 작다,  < 0 : 기준보다 못하다
    """
    return 1 - np.sum((actual - forecast) ** 2) / np.sum((actual - benchmark) ** 2)


def diebold_mariano(actual: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray, lags: int = 24) -> tuple[float, float]:
    """
    Diebold–Mariano(1995) 검정: 두 예측의 제곱오차 차이 평균이 0인가?
        d_t = (실제 − 기준)² − (실제 − 모형)²      → 양수면 모형이 더 정확
        통계량 = 평균(d) / √Var_NW(평균(d))         → 단측 p값 (모형이 더 낫다는 대립가설)
    """
    loss_difference = (actual - benchmark) ** 2 - (actual - forecast) ** 2
    statistic = loss_difference.mean() / np.sqrt(newey_west_variance_of_mean(loss_difference, lags))
    return statistic, 1 - stats.norm.cdf(statistic)


def clark_west(actual: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray, lags: int = 24) -> tuple[float, float]:
    """
    Clark–West(2007) 검정: 기준 모형(예: 수익률 0)이 큰 모형에 '포함된' 경우(nested)의 예측력 비교.
    큰 모형은 쓸모없는 모수 추정 잡음 때문에 오차가 커지는 불이익을 받는데, 이를 보정한다.
        f_t = (실제 − 기준)² − [(실제 − 모형)² − (기준 − 모형)²]
        통계량 = 평균(f) / √Var_NW(평균(f))   → 단측 p값
    """
    adjusted = (actual - benchmark) ** 2 - ((actual - forecast) ** 2 - (benchmark - forecast) ** 2)
    statistic = adjusted.mean() / np.sqrt(newey_west_variance_of_mean(adjusted, lags))
    return statistic, 1 - stats.norm.cdf(statistic)


def direction_hit_rate(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """
    방향 적중률과 Pesaran–Timmermann(1992) 검정.

    왜 '50%와 비교'하면 안 되는가 (실제로 겪은 함정)
        2021~2026년 상승 시간 비율은 50.5%이고, 모형은 '상승'을 57~72% 예측했다.
        이렇게 한쪽으로 쏠린 예측은 방향을 전혀 몰라도 50%보다 높게 맞을 수 있어서,
        50% 기준 이항검정은 p = 10⁻²⁴ 같은 엉터리 유의성을 준다.

    Pesaran–Timmermann: '예측과 실제가 서로 독립일 때 기대되는 적중률'과 비교한다.
        p_a = 실제 상승 비율,  p_f = 예측 상승 비율
        독립일 때 기대 적중률  P* = p_a·p_f + (1−p_a)(1−p_f)
        통계량 = (적중률 − P*) / √[Var(적중률) − Var(P*)]   ~ N(0,1), 단측
    실제 수익률이 정확히 0인 시간은 제외.
    """
    is_nonzero = actual != 0
    actual_up = actual[is_nonzero] > 0
    forecast_up = forecast[is_nonzero] > 0
    sample_size = actual_up.size

    hit_rate = np.mean(actual_up == forecast_up)
    share_actual_up = actual_up.mean()
    share_forecast_up = forecast_up.mean()
    expected_hit_rate_if_independent = (share_actual_up * share_forecast_up
                                        + (1 - share_actual_up) * (1 - share_forecast_up))

    variance_hit = expected_hit_rate_if_independent * (1 - expected_hit_rate_if_independent) / sample_size
    variance_expected = (
        (2 * share_actual_up - 1) ** 2 * share_forecast_up * (1 - share_forecast_up) / sample_size
        + (2 * share_forecast_up - 1) ** 2 * share_actual_up * (1 - share_actual_up) / sample_size
        + 4 * share_actual_up * share_forecast_up * (1 - share_actual_up) * (1 - share_forecast_up) / sample_size**2
    )
    statistic = (hit_rate - expected_hit_rate_if_independent) / np.sqrt(variance_hit - variance_expected)

    return {
        "hit_rate": hit_rate,
        "hit_rate_if_independent": expected_hit_rate_if_independent,
        "forecast_up_share": share_forecast_up,
        "pesaran_timmermann_stat": statistic,
        "pesaran_timmermann_p": 1 - stats.norm.cdf(statistic),
    }


def sign_strategy_performance(actual_bps: np.ndarray, forecast_bps: np.ndarray, cost_bps_per_trade: float) -> dict:
    """
    '예측 부호대로 1단위 롱/숏' 전략의 성과 — 통계적 예측력이 경제적으로 의미 있는지 확인.
        포지션_t = sign(예측_t)  (+1 매수, −1 매도)
        수수료  = |포지션 변화| × 편도 수수료     (롱→숏 전환은 2단위 거래)
    연율화: 1년 = 8,760시간.
    """
    position = np.sign(forecast_bps)
    gross_return = position * actual_bps
    turnover = np.abs(np.diff(position, prepend=0))
    net_return = gross_return - turnover * cost_bps_per_trade

    def annualized_sharpe(returns):
        return returns.mean() / returns.std() * np.sqrt(HOURS_PER_YEAR) if returns.std() > 0 else np.nan

    return {
        "gross_mean_bps_per_hour": gross_return.mean(),
        "net_mean_bps_per_hour": net_return.mean(),
        "gross_sharpe": annualized_sharpe(gross_return),
        "net_sharpe": annualized_sharpe(net_return),
        "trades_per_day": turnover.sum() / len(turnover) * 24,
    }


def evaluate_forecasts(forecasts: pd.DataFrame, target_name: str, cost_bps_per_trade: float) -> pd.DataFrame:
    """모형별 표본 외 성과표. 기준(M0) 대비로 비교한다."""
    actual = forecasts["actual"].to_numpy()
    benchmark = forecasts["M0_benchmark"].to_numpy()
    is_price_target = target_name == "return_bps"

    rows = []
    for model_name in [column for column in forecasts.columns if column.startswith("M")]:
        forecast = forecasts[model_name].to_numpy()
        row = {
            "model": model_name,
            "n_hours": len(actual),
            "rmse": np.sqrt(np.mean((actual - forecast) ** 2)),
            "oos_r2_vs_M0": out_of_sample_r2(actual, forecast, benchmark),
        }
        if model_name != "M0_benchmark":
            if is_price_target:
                row["clark_west_stat"], row["clark_west_p"] = clark_west(actual, forecast, benchmark)
            else:
                row["dm_stat"], row["dm_p"] = diebold_mariano(actual, forecast, benchmark)

        if is_price_target and model_name != "M0_benchmark":
            row.update(direction_hit_rate(actual, forecast))
            row.update(sign_strategy_performance(actual, forecast, cost_bps_per_trade))
        elif not is_price_target:
            row["corr_actual_forecast"] = np.corrcoef(actual, forecast)[0, 1]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ===========================================================================
# 6. 전체 표본 회귀 (해석용) — Newey–West HAC 표준오차
# ===========================================================================
def ols_newey_west(features: pd.DataFrame, target: pd.Series, lags: int = 24) -> pd.DataFrame:
    """
    표준화한 팩터로 OLS를 추정하고, 계수의 표준오차를 Newey–West HAC로 계산한다.
        β = (X'X)⁻¹X'y
        Var(β) = (X'X)⁻¹ · S · (X'X)⁻¹,   S = Σ_k w_k Σ_t (x_t e_t)(x_{t−k} e_{t−k})',  w_k = 1 − k/(L+1)
    계수 = 팩터 1 표준편차 변화당 대상의 변화 (대상 단위 그대로).
    """
    data = features.join(target.rename("target")).dropna()
    x = data[features.columns]
    standardized = (x - x.mean()) / x.std().replace(0, 1)
    design = np.column_stack([np.ones(len(data)), standardized.to_numpy()])
    y = data["target"].to_numpy()

    xtx_inverse = np.linalg.inv(design.T @ design)
    beta = xtx_inverse @ design.T @ y
    residuals = y - design @ beta

    # --- HAC 가운데 행렬 S ---
    scores = design * residuals[:, None]  # 각 행: x_t · e_t
    middle = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        cross = scores[lag:].T @ scores[:-lag]
        middle += weight * (cross + cross.T)

    covariance = xtx_inverse @ middle @ xtx_inverse
    standard_errors = np.sqrt(np.diag(covariance))

    names = ["intercept"] + list(features.columns)
    table = pd.DataFrame({"coef": beta, "nw_se": standard_errors}, index=names)
    table["t_stat"] = table["coef"] / table["nw_se"]
    table["p_value"] = 2 * (1 - stats.norm.cdf(table["t_stat"].abs()))
    table.attrs["r_squared"] = 1 - residuals.var() / y.var()
    table.attrs["n"] = len(y)
    return table
