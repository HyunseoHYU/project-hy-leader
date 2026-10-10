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
# 1. 1분봉 → N분 봉 패널 (5분 · 15분 · 1시간 · 4시간 · 1일 …)
# ===========================================================================
MINUTES_PER_DAY = 1440
MINUTES_PER_WEEK = 7 * MINUTES_PER_DAY

# 봉 하나에 필요한 최소 1분봉 비율 (60분 중 50분 = 83%). 이보다 적으면 거래소 점검 등으로 보고 결측 처리
MIN_COVERAGE_RATIO = MIN_MINUTES_PER_HOUR / 60


def build_bar_panel(minute_bars: pd.DataFrame, bar_minutes: int = 60) -> pd.DataFrame:
    """
    1분봉(liquidity_metrics.prepare_minute_bars 결과)을 bar_minutes 분 단위 봉으로 모은다.

    만드는 변수 (모두 '그 봉 동안'의 값)
        return_bps        봉 수익률 = ln(종가_t / 종가_{t−1}) × 10⁴
        rv_bps            실현변동성 = √Σ(1분 수익률²)                    (bp)
        range_bps         고가·저가 범위 = ln(고가/저가) × 10⁴             (Parkinson 변동성 재료)
        usd_volume_musd   거래대금 (백만 달러)
        order_imbalance   주문 불균형 = (매수 주도량 − 매도 주도량) / 거래량   (−1 ~ +1)
        avg_trade_btc     평균 체결 크기 = 거래량 / 체결 수                  (BTC)
        amihud            Amihud 비유동성 = Σ|1분 수익률| / 거래대금         (bp per $1M)
                          ※ |봉 수익률| 대신 1분 수익률 절댓값의 합(가격 경로 길이)을 써서 잡음을 줄임
        kyle_lambda       Kyle's λ = 1분 수익률(bp)을 순매수량(BTC)에 회귀한 기울기 × 100  (bp per 100 BTC)
                          ※ 봉 안의 1분봉 개수(5분봉이면 5개)로 추정하므로 짧은 봉일수록 잡음이 크다
        open, volume      기술지표 계산용 (시가, BTC 거래량)
    """
    bar_frequency = f"{bar_minutes}min"
    bars = minute_bars.copy()
    bars["bar_start"] = bars.index.floor(bar_frequency)
    bars["abs_return_bps"] = bars["return_bps"].abs()
    bars["squared_return_bps"] = bars["return_bps"] ** 2

    by_bar = bars.groupby("bar_start")
    panel = pd.DataFrame(
        {
            "open": by_bar["open"].first(),
            "close": by_bar["close"].last(),
            "high": by_bar["high"].max(),
            "low": by_bar["low"].min(),
            "btc_volume": by_bar["volume"].sum(),
            "buy_volume": by_bar["buy_volume"].sum(),
            "usd_volume_musd": by_bar["usd_volume"].sum() / 1e6,
            "num_trades": by_bar["num_trades"].sum(),
            "path_length_bps": by_bar["abs_return_bps"].sum(),
            "realized_variance_bps2": by_bar["squared_return_bps"].sum(),
            "minutes_available": by_bar.size(),
        }
    )

    # --- 빈 봉도 행을 만들어 간격을 일정하게 (shift/rolling 이 '한 봉 전'을 정확히 가리키도록) ---
    full_grid = pd.date_range(panel.index.min(), panel.index.max(), freq=bar_frequency)
    panel = panel.reindex(full_grid)
    panel.index.name = "bar_start"

    # --- 파생 변수 ---
    panel["return_bps"] = np.log(panel["close"]).diff() * BPS
    panel["rv_bps"] = np.sqrt(panel["realized_variance_bps2"])
    panel["range_bps"] = np.log(panel["high"] / panel["low"]) * BPS
    sell_volume = panel["btc_volume"] - panel["buy_volume"]
    panel["order_imbalance"] = (panel["buy_volume"] - sell_volume) / panel["btc_volume"]
    panel["avg_trade_btc"] = panel["btc_volume"] / panel["num_trades"]
    panel["amihud"] = panel["path_length_bps"] / panel["usd_volume_musd"]

    bar_of_each_minute = minute_bars.index.floor(bar_frequency).to_series(index=minute_bars.index)
    kyle = lm.regression_slope_by_group(minute_bars["signed_volume"], minute_bars["return_bps"], bar_of_each_minute)
    panel["kyle_lambda"] = kyle["slope"] * 100

    # --- 데이터가 많이 빈 봉은 전부 결측으로 ---
    minimum_minutes = round(bar_minutes * MIN_COVERAGE_RATIO)
    is_incomplete_bar = panel["minutes_available"].fillna(0) < minimum_minutes
    panel.loc[is_incomplete_bar, panel.columns.difference(["minutes_available"])] = np.nan
    return panel


def build_hourly_panel(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """1시간 봉 패널 (기존 베이스라인 분석용 바로가기)."""
    return build_bar_panel(minute_bars, bar_minutes=60)


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


def har_windows(bar_minutes: int) -> dict[str, tuple[int, int]]:
    """
    HAR 구성요소의 창 길이 (봉 개수, 최소 유효 봉 개수).
        day  = 하루에 해당하는 봉 수,   단 최소 5봉   (1일 봉이면 5일)
        week = 1주에 해당하는 봉 수,    단 최소 30봉  (1일 봉이면 30일)
    최소 유효 봉 개수는 1시간 봉 기준값(24봉 중 18, 168봉 중 120)과 같은 비율.
    """
    day_bars = max(MINUTES_PER_DAY // bar_minutes, 5)
    week_bars = max(MINUTES_PER_WEEK // bar_minutes, 30)
    return {
        "day": (day_bars, round(day_bars * 18 / 24)),
        "week": (week_bars, round(week_bars * 120 / 168)),
    }


def har_components(series: pd.Series, prefix: str, bar_minutes: int = 60) -> pd.DataFrame:
    """
    HAR(Heterogeneous AutoRegressive, Corsi 2009) 구성요소: 서로 다른 시간 규모의 과거 평균.
        _last  직전 봉 값           (초단기 참여자)
        _day   최근 하루 평균       (일 단위 참여자)
        _week  최근 1주 평균        (주 단위 참여자)
    변동성·유동성은 이 세 가지로 대부분 설명된다는 것이 정형화된 사실.
    """
    windows = har_windows(bar_minutes)
    day_bars, day_min = windows["day"]
    week_bars, week_min = windows["week"]
    return pd.DataFrame(
        {
            f"{prefix}_last": series,
            f"{prefix}_day": series.rolling(day_bars, min_periods=day_min).mean(),
            f"{prefix}_week": series.rolling(week_bars, min_periods=week_min).mean(),
        }
    )


def build_factor_table(panel: pd.DataFrame, bar_minutes: int = 60) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    한 행 = 봉 t.  팩터는 봉 t가 끝난 시점까지 알 수 있는 정보만, 대상(target_*)은 다음 봉(t+1)의 값.

    반환:
        table          팩터 + 대상 컬럼
        factor_groups  {"own_<대상>": 그 대상의 자기 과거값 팩터, "seasonal": 시간 패턴,
                        "all": 전체 베이스라인 팩터, "boosting_extra": 부스팅 전용 추가 컬럼}
    """
    table = pd.DataFrame(index=panel.index)
    factor_groups: dict[str, list[str]] = {}
    windows = har_windows(bar_minutes)
    day_bars, day_min = windows["day"]
    week_bars, week_min = windows["week"]

    # --- (1) 유동성·변동성 변수의 HAR 구성요소 ---
    transformed = {
        "log_amihud": np.log(panel["amihud"].where(panel["amihud"] > 0)),
        "log_rv": np.log(panel["rv_bps"].where(panel["rv_bps"] > 0)),
        "kyle_lambda": winsorize(panel["kyle_lambda"]),
    }
    for name, series in transformed.items():
        har = har_components(series, name, bar_minutes)
        table = table.join(har)
        factor_groups[f"own_{name}"] = list(har.columns)

    # --- (2) 가격 팩터: 반전(reversal)·모멘텀 ---
    #     과거 수익률 합계. 직전 봉은 '단기 반전', 1주는 '모멘텀'을 잡는 용도
    returns = panel["return_bps"]
    table["ret_last"] = returns
    table["ret_day"] = returns.rolling(day_bars, min_periods=day_min).sum()
    table["ret_week"] = returns.rolling(week_bars, min_periods=week_min).sum()
    factor_groups["own_return"] = ["ret_last", "ret_day", "ret_week"]

    # --- (3) 주문흐름 팩터 ---
    table["oib_last"] = panel["order_imbalance"]
    table["oib_day"] = panel["order_imbalance"].rolling(day_bars, min_periods=day_min).mean()

    # --- (4) 거래 활동 팩터 ---
    #     수준 대신 '최근 1주 평균 대비 편차'로 만든다. 거래대금·평균 체결 크기는 해마다 크게 변해서
    #     (예: log 평균 체결 크기 2018년 −1.9 → 2026년 −5.3) 수준을 그대로 쓰면 학습 범위 밖 값이 들어온다.
    def deviation_from_week_mean(series):
        return series - series.rolling(week_bars, min_periods=week_min).mean()

    log_volume = np.log(panel["usd_volume_musd"].where(panel["usd_volume_musd"] > 0))
    log_trade_size = np.log(panel["avg_trade_btc"].where(panel["avg_trade_btc"] > 0))
    log_range = np.log(panel["range_bps"].where(panel["range_bps"] > 0))
    table["abnormal_volume"] = deviation_from_week_mean(log_volume)  # 평소보다 거래대금이 몇 배인가 (로그)
    table["abnormal_trade_size"] = deviation_from_week_mean(log_trade_size)  # 평소보다 큰 주문이 들어오는가
    table["log_range"] = log_range
    table["abnormal_range"] = deviation_from_week_mean(log_range)

    # 유동성·변동성의 '편차' 버전 (가격 예측용, 아래 설명)
    for name in transformed:
        table[f"{name}_deviation"] = table[f"{name}_last"] - table[f"{name}_week"]

    # --- (5) 시간 패턴: '예측하려는 봉(t+1)'의 시각·요일 — 미리 알 수 있는 정보라 사용 가능 ---
    next_bar_start = table.index + pd.Timedelta(minutes=bar_minutes)
    table["is_weekend"] = (next_bar_start.dayofweek >= 5).astype(float)
    if bar_minutes < MINUTES_PER_DAY:
        hour_angle = 2 * np.pi * next_bar_start.hour / 24
        table["hour_sin"] = np.sin(hour_angle)  # 하루 주기를 원 위의 좌표로 표현 (23시와 0시가 가깝게)
        table["hour_cos"] = np.cos(hour_angle)
        table["hour_sin2"] = np.sin(2 * hour_angle)  # 반나절 주기 (아시아·미국 세션 두 봉우리)
        table["hour_cos2"] = np.cos(2 * hour_angle)
        table["hour_of_day"] = next_bar_start.hour.astype(float)  # 부스팅 모형 전용 (선형 모형에는 sin/cos)
        factor_groups["seasonal"] = ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2", "is_weekend"]
        factor_groups["boosting_extra"] = ["hour_of_day"]
    else:
        # 1일 봉은 '시각'이 항상 0시라 의미 없음 → 요일(주말) 효과만
        factor_groups["seasonal"] = ["is_weekend"]
        factor_groups["boosting_extra"] = []

    # --- 전체 팩터 목록 (선형 모형용) ---
    all_factors = []
    for group_name in ["own_log_amihud", "own_log_rv", "own_kyle_lambda", "own_return", "seasonal"]:
        all_factors += factor_groups[group_name]
    all_factors += ["oib_last", "oib_day", "abnormal_volume", "abnormal_trade_size", "log_range"]
    factor_groups["all"] = list(dict.fromkeys(all_factors))  # 중복 제거, 순서 유지

    # --- 가격 예측용 팩터: 추세 없는(정상, stationary) 변수만 ---
    #     유동성 수준(예: log Amihud)은 해마다 낮아지는 추세가 있어, 수익률 회귀에 넣으면 학습 기간의 우연한
    #     관계가 테스트 기간의 '범위 밖' 값과 만나 폭발한다 (1일 봉에서 2022년 일 수익률 +1,700bp 예측이 실제 발생).
    #     → 수익률 예측에는 수준 대신 '1주 평균 대비 편차'만 쓴다.
    price_factors = factor_groups["own_return"] + factor_groups["seasonal"]
    price_factors += [f"{name}_deviation" for name in transformed]
    price_factors += ["oib_last", "oib_day", "abnormal_volume", "abnormal_trade_size", "abnormal_range"]
    factor_groups["price"] = price_factors

    # --- (6) 예측 대상: 다음 봉의 값 (shift(-1)) ---
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


# 학습 구간 분위수 밖의 팩터 값을 잘라내는 경계 (0.5% / 99.5%)
FEATURE_CLIP_QUANTILES = (0.005, 0.995)


class StandardizedOLS:
    """
    선형회귀 (M1, M2 …).
    학습 구간의 평균·표준편차로 팩터를 표준화한 뒤 최소제곱 추정 → 계수 크기를 서로 비교할 수 있다.
    예측할 때도 '학습 구간'의 통계만 써야 미래 정보가 섞이지 않는다.

    극단값 방어 (실제로 겪은 문제)
        1일 봉처럼 학습 표본이 작을 때, 테스트 기간에 학습 범위를 크게 벗어난 팩터 값이 들어오면
        선형 예측이 폭발했다 (R² −300% 이상). → 팩터를 학습 구간의 0.5%·99.5% 분위수로 잘라서 사용.

    anchor_column (선택)
        주어지면 '대상 − anchor'를 학습하고 예측 때 anchor를 더한다 (BoostingModel과 같은 방식).
    """

    def __init__(self, columns: list[str], anchor_column: str | None = None):
        self.columns = columns
        self.anchor_column = anchor_column

    def _anchor(self, features: pd.DataFrame) -> np.ndarray:
        if self.anchor_column is None:
            return np.zeros(len(features))
        return features[self.anchor_column].to_numpy()

    def _standardize(self, features: pd.DataFrame) -> np.ndarray:
        clipped = features[self.columns].clip(self.lower_bound_, self.upper_bound_, axis=1)
        return ((clipped - self.mean_) / self.std_).to_numpy()

    def fit(self, features: pd.DataFrame, target: pd.Series):
        x = features[self.columns]
        lower_quantile, upper_quantile = FEATURE_CLIP_QUANTILES
        self.lower_bound_ = x.quantile(lower_quantile)
        self.upper_bound_ = x.quantile(upper_quantile)

        clipped = x.clip(self.lower_bound_, self.upper_bound_, axis=1)
        self.mean_ = clipped.mean()
        self.std_ = clipped.std().replace(0, 1)  # 상수 컬럼(예: 학습 구간에 주말이 없는 경우) 방어

        design = np.column_stack([np.ones(len(x)), self._standardize(features)])
        target_after_anchor = target.to_numpy() - self._anchor(features)
        self.coef_, *_ = np.linalg.lstsq(design, target_after_anchor, rcond=None)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(features)), self._standardize(features)])
        return design @ self.coef_ + self._anchor(features)


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


def benchmark_model(target_name: str):
    """M0: 유동성은 '다음 봉 = 이번 봉', 수익률은 '0'."""
    if target_name == "return_bps":
        return ZeroModel()
    return PersistenceModel(f"{target_name}_last")


def boosting_anchor_column(target_name: str) -> str | None:
    """모형이 '최근 1주 평균 대비 편차'를 학습하도록 기준이 될 컬럼 (수익률은 이미 0 근처라 불필요)."""
    if target_name == "return_bps":
        return None
    return f"{target_name}_week"


def own_factor_columns(target_name: str, factor_groups: dict[str, list[str]]) -> list[str]:
    """대상 자신의 과거값 팩터 (HAR 구성요소 또는 과거 수익률)."""
    if target_name == "return_bps":
        return factor_groups["own_return"]
    return factor_groups[f"own_{target_name}"]


def baseline_factor_columns(target_name: str, factor_groups: dict[str, list[str]]) -> list[str]:
    """베이스라인 전체 팩터: 수익률 대상은 추세 없는 '가격용' 팩터, 유동성 대상은 HAR 수준 포함 전체."""
    if target_name == "return_bps":
        return factor_groups["price"]
    return factor_groups["all"]


def make_models(target_name: str, factor_groups: dict[str, list[str]]) -> dict:
    """베이스라인 모형 4종 (M0~M3)."""
    own_factors = own_factor_columns(target_name, factor_groups)
    baseline_columns = baseline_factor_columns(target_name, factor_groups)
    boosting_columns = baseline_columns + factor_groups["boosting_extra"]

    # 유동성 대상은 선형·부스팅 모두 '최근 1주 평균 대비 편차'를 학습 (anchor).
    # 수준을 직접 맞히게 하면, 팩터를 학습 범위로 자르는(clip) 극단값 방어와 충돌한다:
    # Amihud 수준이 학습 기간 최저보다 더 낮아지면 예측이 그 바닥에 붙어 버림 (실제로 R² +28% → −17%).
    anchor = boosting_anchor_column(target_name)
    return {
        "M0_benchmark": benchmark_model(target_name),
        "M1_own_HAR": StandardizedOLS(own_factors + factor_groups["seasonal"], anchor_column=anchor),
        "M2_all_linear": StandardizedOLS(baseline_columns, anchor_column=anchor),
        "M3_all_boosting": BoostingModel(boosting_columns, anchor_column=anchor),
    }


def columns_used_by(models: dict) -> list[str]:
    """모형들이 쓰는 컬럼 전체 (워크포워드에서 결측 제거 기준 — 모든 모형이 같은 표본으로 비교되도록)."""
    columns = []
    for model in models.values():
        columns += getattr(model, "columns", [])
        for attribute in ("column", "anchor_column"):
            value = getattr(model, attribute, None)
            if value:
                columns.append(value)
    return list(dict.fromkeys(columns))


# ===========================================================================
# 4. 워크포워드(walk-forward) 예측
# ===========================================================================
def walk_forward_forecast(table: pd.DataFrame, target_name: str, factor_groups: dict[str, list[str]],
                          train_start: str, test_years: list[int], model_factory=None) -> pd.DataFrame:
    """
    시간 순서를 지키는 표본 외 예측.
        테스트 연도 Y 마다:  학습 = [train_start, Y년 1월 1일)   →   예측 = Y년 전체
    즉 어떤 예측도 그 시점 이후의 데이터로 학습되지 않는다 (KOSPI 프로젝트의 시점 고정 원칙과 동일).

    model_factory: (target_name, factor_groups) → {모형이름: 모형}.  기본값은 베이스라인 make_models.
    반환: 행 = 시각, 컬럼 = actual + 모형별 예측값
    """
    if model_factory is None:
        model_factory = make_models

    target_column = f"target_{target_name}"
    needed_columns = columns_used_by(model_factory(target_name, factor_groups)) + [target_column]
    usable = table[list(dict.fromkeys(needed_columns))].dropna()
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
        for model_name, model in model_factory(target_name, factor_groups).items():
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


def sign_strategy_performance(actual_bps: np.ndarray, forecast_bps: np.ndarray, cost_bps_per_trade: float,
                              bars_per_year: float = HOURS_PER_YEAR) -> dict:
    """
    '예측 부호대로 1단위 롱/숏' 전략의 성과 — 통계적 예측력이 경제적으로 의미 있는지 확인.
        포지션_t = sign(예측_t)  (+1 매수, −1 매도)
        수수료  = |포지션 변화| × 편도 수수료     (롱→숏 전환은 2단위 거래)
    연율화: 샤프 × √(1년 봉 개수)   (1시간 봉 = 8,760)
    """
    position = np.sign(forecast_bps)
    gross_return = position * actual_bps
    turnover = np.abs(np.diff(position, prepend=0))
    net_return = gross_return - turnover * cost_bps_per_trade

    def annualized_sharpe(returns):
        return returns.mean() / returns.std() * np.sqrt(bars_per_year) if returns.std() > 0 else np.nan

    bars_per_day = bars_per_year / 365

    return {
        "gross_mean_bps_per_hour": gross_return.mean(),
        "net_mean_bps_per_hour": net_return.mean(),
        "gross_sharpe": annualized_sharpe(gross_return),
        "net_sharpe": annualized_sharpe(net_return),
        "trades_per_day": turnover.sum() / len(turnover) * bars_per_day,
    }


def newey_west_lags(bars_per_year: float) -> int:
    """
    Newey–West 시차 수 = 하루치 봉 개수 (오차의 자기상관이 대략 하루 안에 사라진다고 보고),
    단 5 ~ 48 사이로 제한 (1일 봉은 5일, 5분 봉도 계산량을 위해 48개까지).
    """
    bars_per_day = round(bars_per_year / 365)
    return int(min(max(bars_per_day, 5), 48))


def evaluate_forecasts(forecasts: pd.DataFrame, target_name: str, cost_bps_per_trade: float,
                       bars_per_year: float = HOURS_PER_YEAR) -> pd.DataFrame:
    """모형별 표본 외 성과표. 기준(M0) 대비로 비교한다."""
    lags = newey_west_lags(bars_per_year)
    actual = forecasts["actual"].to_numpy()
    benchmark = forecasts["M0_benchmark"].to_numpy()
    is_price_target = target_name == "return_bps"

    rows = []
    model_names = [column for column in forecasts.columns if column != "actual"]
    for model_name in model_names:
        forecast = forecasts[model_name].to_numpy()
        row = {
            "model": model_name,
            "n_hours": len(actual),
            "rmse": np.sqrt(np.mean((actual - forecast) ** 2)),
            "oos_r2_vs_M0": out_of_sample_r2(actual, forecast, benchmark),
        }
        if model_name != "M0_benchmark":
            if is_price_target:
                row["clark_west_stat"], row["clark_west_p"] = clark_west(actual, forecast, benchmark, lags)
            else:
                row["dm_stat"], row["dm_p"] = diebold_mariano(actual, forecast, benchmark, lags)

        if is_price_target and model_name != "M0_benchmark":
            row.update(direction_hit_rate(actual, forecast))
            row.update(sign_strategy_performance(actual, forecast, cost_bps_per_trade, bars_per_year))
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
