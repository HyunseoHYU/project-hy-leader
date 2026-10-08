"""
예측 시간 단위별 분석 — 평균회귀성 · 예측 성능 · 기술지표 vs 미시구조 팩터
=====================================================================

질문
    1. 평균회귀성: 유동성·변동성 충격은 얼마나 오래 지속되고(반감기), 수익률은 얼마나 되돌려지는가?
                  그리고 그 성질이 시간 단위(5분 ~ 1일)에 따라 어떻게 달라지는가?
    2. 시간 단위별 예측 성능: 다음 봉의 유동성·수익률 예측력이 시간 단위에 따라 어떻게 변하는가?
    3. 기술지표 관점: RSI·MACD·볼린저 등 차트 지표로 만든 팩터가 예측력을 주는가?
                    → '가격' 대상과 '유동성' 대상으로 나눠 미시구조 베이스라인 팩터와 비교

시간 단위   5분 · 15분 · 1시간 · 4시간 · 1일 (각 봉의 '다음 봉'을 예측)

모형 (대상마다, 모든 모형이 같은 표본에서 비교되도록 결측을 공통으로 제거)
    M0  기준          유동성 = 직전 값, 수익률 = 0
    M2  베이스라인 선형    미시구조 팩터 (HAR·주문 불균형·비정상 거래량·시간 패턴 …)  ← src/baseline_factors.py
    M3  베이스라인 부스팅
    T1  기술지표 선형      기술지표 13개만                                       ← src/technical_indicators.py
    T2  기술지표 부스팅
    C1  결합 선형          베이스라인 + 기술지표  → 기술지표의 '추가' 예측력 확인

검증    워크포워드 2021~2026 (각 해를 그 이전 데이터로만 학습), 수수료 편도 10bp

출력    results/horizons/summary.md, tables/*.csv, figures/*.png
실행    python src/analyze_horizons.py        (약 20~40분, 5분 봉의 부스팅이 대부분)
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import baseline_factors as bf
import liquidity_metrics as lm
import technical_indicators as ti
from plot_style import AQUA, BLUE, GRID, MUTED, ORANGE, save_figure

# ---------------------------------------------------------------------------
# 경로 & 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "horizons"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

HORIZONS_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
TRAIN_START = "2018-01-01"
TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
COST_BPS_PER_TRADE = 10.0
LIQUIDITY_TARGETS = ["log_amihud", "kyle_lambda", "log_rv"]
PRICE_TARGET = "return_bps"
ALL_TARGETS = LIQUIDITY_TARGETS + [PRICE_TARGET]

TARGET_LABELS = {
    "log_amihud": "log Amihud illiquidity",
    "kyle_lambda": "Kyle's λ",
    "log_rv": "log realized volatility",
    "return_bps": "next-bar return",
}
MODEL_LABELS = {
    "M2_base_linear": "baseline (linear)",
    "M3_base_boost": "baseline (boosting)",
    "T1_tech_linear": "technical (linear)",
    "T2_tech_boost": "technical (boosting)",
    "C1_combined_linear": "baseline + technical (linear)",
}

# 기술지표 '회귀성' 점검 구간 (업계 관행 기준선)
RSI_OVERSOLD, RSI_OVERBOUGHT = 30, 70


# ===========================================================================
# 1. 시간 단위별 팩터 테이블
# ===========================================================================
def build_table_for_horizon(minute_bars: pd.DataFrame, bar_minutes: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """한 시간 단위의 베이스라인 팩터 + 기술지표 + 예측 대상."""
    panel = bf.build_bar_panel(minute_bars, bar_minutes)
    table, factor_groups = bf.build_factor_table(panel, bar_minutes)
    technical = ti.build_technical_factors(panel)
    table = table.join(technical)
    factor_groups["technical"] = ti.TECHNICAL_COLUMNS
    return table, factor_groups


def make_comparison_models(target_name: str, factor_groups: dict[str, list[str]]) -> dict:
    """
    베이스라인 vs 기술지표 vs 결합 모형 (모듈 상단 설명 참고).

    유동성 대상의 공정 비교: 모든 모형이 '최근 1주 평균 대비 편차'를 예측한다 (anchor = 대상의 1주 평균).
        기술지표에는 Amihud 같은 유동성의 '수준' 정보가 없어서, 수준을 그대로 맞히게 하면
        해마다 낮아지는 추세를 못 따라가 R²가 −2,000%까지 떨어진다 (당연한 실패라 비교가 무의미).
        수준은 1주 평균으로 고정하고 '평소 대비 충격'만 경쟁시키면 공정하고,
        이는 1월 이벤트 분석에서 필요한 '비정상 유동성'과도 같은 개념이다.
    수익률 대상은 anchor 없음 (이미 0 근처에서 안정적).
    """
    baseline_columns = bf.baseline_factor_columns(target_name, factor_groups)  # 수익률이면 추세 없는 가격용 팩터
    technical_columns = factor_groups["technical"]
    boosting_extra = factor_groups["boosting_extra"]
    anchor = bf.boosting_anchor_column(target_name)

    return {
        "M0_benchmark": bf.benchmark_model(target_name),
        "M2_base_linear": bf.StandardizedOLS(baseline_columns, anchor_column=anchor),
        "M3_base_boost": bf.BoostingModel(baseline_columns + boosting_extra, anchor_column=anchor),
        # 기술지표 모형은 차트 지표만 쓴다 (대상의 과거값·시간 패턴 없음, 유동성이면 1주 평균 anchor만)
        "T1_tech_linear": bf.StandardizedOLS(technical_columns, anchor_column=anchor),
        "T2_tech_boost": bf.BoostingModel(technical_columns, anchor_column=anchor),
        "C1_combined_linear": bf.StandardizedOLS(baseline_columns + technical_columns, anchor_column=anchor),
    }


# ===========================================================================
# 2. 평균회귀성 분석
# ===========================================================================
def ar1_coefficient(series: pd.Series) -> float:
    """
    AR(1) 계수 φ:  x_t − 평균 = φ · (x_{t−1} − 평균) + e_t
        φ ≈ 1 : 충격이 거의 사라지지 않음 (강한 지속성)
        φ ≈ 0 : 다음 봉이면 충격이 사라짐
        φ < 0 : 다음 봉에 반대로 튐 (되돌림)
    """
    values = series.dropna()
    current = values.iloc[1:].to_numpy()
    previous = values.shift(1).iloc[1:].to_numpy()
    is_valid = np.isfinite(previous)
    return np.corrcoef(current[is_valid], previous[is_valid])[0, 1]


def half_life_in_hours(phi: float, bar_minutes: int) -> float:
    """
    반감기 = 충격의 절반이 사라지는 데 걸리는 시간.
        φ^k = 0.5  →  k = ln(0.5) / ln(φ)  봉,   시간으로 환산하려면 × (봉 길이 / 60)
    φ ≤ 0 이면 한 봉 안에 사라진 것으로 보고 반감기 = 봉 길이의 절반으로 표시하지 않고 NaN.
    """
    if not 0 < phi < 1:
        return np.nan
    half_life_bars = np.log(0.5) / np.log(phi)
    return half_life_bars * bar_minutes / 60


def mean_reversion_table(table: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """
    유동성·변동성: 두 가지 지속성
        level      값 자체의 AR(1) — 장기 추세(예: Amihud의 해마다 하락)까지 포함
        deviation  '최근 1주 평균 대비 편차'의 AR(1) — 추세를 뺀 순수한 단기 충격의 지속성 ← 이벤트 분석에 중요
    수익률: 1차 자기상관 (음수 = 되돌림)과 그 t값, |수익률|의 1차 자기상관 (변동성 군집)
    """
    sample = table.loc[TRAIN_START:]
    rows = []

    for name in LIQUIDITY_TARGETS:
        level = sample[f"{name}_last"]
        deviation = level - sample[f"{name}_week"]
        phi_level = ar1_coefficient(level)
        phi_deviation = ar1_coefficient(deviation)
        rows.append(
            {
                "variable": name,
                "ar1_level": phi_level,
                "ar1_deviation": phi_deviation,
                "half_life_level_hours": half_life_in_hours(phi_level, bar_minutes),
                "half_life_deviation_hours": half_life_in_hours(phi_deviation, bar_minutes),
            }
        )

    returns = sample["ret_last"].dropna()
    return_autocorrelation = ar1_coefficient(returns)
    rows.append(
        {
            "variable": "return_bps",
            "ar1_level": return_autocorrelation,
            "ar1_t_stat": return_autocorrelation * np.sqrt(len(returns)),  # 근사: ρ / (1/√n)
            "ar1_abs_return": ar1_coefficient(returns.abs()),
        }
    )
    return pd.DataFrame(rows).set_index("variable")


def technical_signal_reversion(table: pd.DataFrame) -> pd.DataFrame:
    """
    기술지표의 '회귀성' 직접 점검: 과매수·과매도 구간 다음 봉 수익률이 실제로 반대로 움직이는가?
        RSI < 30 (과매도) → 반등 기대,  RSI > 70 (과매수) → 하락 기대
        %B < 0 (하단 이탈) → 반등 기대,  %B > 1 (상단 돌파) → 하락 기대
    구간별 다음 봉 평균 수익률(bp)과 t값.  주의: 겹치지 않는 독립 표본이 아니라 t값은 다소 과대.
    """
    sample = table.loc[TRAIN_START:, ["rsi_14", "bollinger_pctb", "target_return_bps"]].dropna()
    next_return = sample["target_return_bps"]

    zones = {
        "RSI<30 (oversold)": sample["rsi_14"] < RSI_OVERSOLD,
        "RSI 30-70": (sample["rsi_14"] >= RSI_OVERSOLD) & (sample["rsi_14"] <= RSI_OVERBOUGHT),
        "RSI>70 (overbought)": sample["rsi_14"] > RSI_OVERBOUGHT,
        "%B<0 (below band)": sample["bollinger_pctb"] < 0,
        "%B 0-1": (sample["bollinger_pctb"] >= 0) & (sample["bollinger_pctb"] <= 1),
        "%B>1 (above band)": sample["bollinger_pctb"] > 1,
    }
    rows = []
    for zone_name, in_zone in zones.items():
        zone_returns = next_return[in_zone]
        rows.append(
            {
                "zone": zone_name,
                "share_of_bars": in_zone.mean(),
                "mean_next_return_bps": zone_returns.mean(),
                "t_stat": zone_returns.mean() / (zone_returns.std() / np.sqrt(len(zone_returns))),
            }
        )
    return pd.DataFrame(rows).set_index("zone")


# ===========================================================================
# 3. 실행 루프
# ===========================================================================
def run_one_horizon(minute_bars: pd.DataFrame, horizon_name: str, bar_minutes: int) -> dict:
    started = time.time()
    table, factor_groups = build_table_for_horizon(minute_bars, bar_minutes)
    bars_per_year = 365 * bf.MINUTES_PER_DAY / bar_minutes

    reversion = mean_reversion_table(table, bar_minutes)
    signal_reversion = technical_signal_reversion(table)

    evaluations = []
    yearly_price_rows = []
    for target in ALL_TARGETS:
        forecasts = bf.walk_forward_forecast(table, target, factor_groups, TRAIN_START, TEST_YEARS,
                                             model_factory=make_comparison_models)
        evaluation = bf.evaluate_forecasts(forecasts, target, COST_BPS_PER_TRADE, bars_per_year)
        evaluation.insert(0, "target", target)
        evaluations.append(evaluation)

        if target == PRICE_TARGET:
            for year, rows in forecasts.groupby(forecasts.index.year):
                yearly_row = {"year": year}
                for model in MODEL_LABELS:
                    yearly_row[model] = 100 * bf.out_of_sample_r2(
                        rows["actual"].to_numpy(), rows[model].to_numpy(), rows["M0_benchmark"].to_numpy()
                    )
                yearly_price_rows.append(yearly_row)

    print(f"   {horizon_name}: {len(table):,}봉, {time.time() - started:.0f}초", flush=True)
    return {
        "evaluation": pd.concat(evaluations),
        "reversion": reversion,
        "signal_reversion": signal_reversion,
        "yearly_price": pd.DataFrame(yearly_price_rows).set_index("year"),
    }


def stack_by_horizon(results: dict, key: str) -> pd.DataFrame:
    """시간 단위별 표를 'horizon' 컬럼을 붙여 하나로 합친다."""
    frames = []
    for horizon_name, result in results.items():
        frame = result[key].copy()
        frame.insert(0, "horizon", horizon_name)
        frames.append(frame)
    return pd.concat(frames)


# ===========================================================================
# 4. 차트
# ===========================================================================
def plot_mean_reversion(reversion: pd.DataFrame) -> None:
    """좌: 유동성 충격(1주 평균 대비 편차)의 반감기,  우: 수익률 1차 자기상관 (0 아래 = 되돌림)."""
    horizons = list(HORIZONS_MINUTES)
    fig, (half_life_ax, acf_ax) = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, color in zip(LIQUIDITY_TARGETS, [BLUE, ORANGE, AQUA]):
        rows = reversion[reversion.index == name].set_index("horizon").reindex(horizons)
        half_life_ax.plot(horizons, rows["half_life_deviation_hours"], color=color, marker="o", ms=7, label=TARGET_LABELS[name])
    half_life_ax.set_yscale("log")
    half_life_ax.set_ylabel("half-life of shock (hours, log)")
    half_life_ax.set_xlabel("bar size")
    half_life_ax.set_title("Persistence of liquidity shocks\n(deviation from 1-week mean)", loc="left")
    half_life_ax.legend(frameon=False)

    return_rows = reversion[reversion.index == "return_bps"].set_index("horizon").reindex(horizons)
    acf_ax.bar(horizons, return_rows["ar1_level"], color=BLUE, width=0.6)
    for position, (value, t_stat) in enumerate(zip(return_rows["ar1_level"], return_rows["ar1_t_stat"])):
        acf_ax.text(position, value, f"{value:.3f}\n(t={t_stat:.1f})", ha="center",
                    va="top" if value < 0 else "bottom", fontsize=8)
    acf_ax.axhline(0, color=MUTED, lw=0.8)
    acf_ax.set_xlabel("bar size")
    acf_ax.set_ylabel("lag-1 autocorrelation of returns")
    acf_ax.set_title("Return reversal by bar size (below 0 = reversal)", loc="left")
    save_figure(fig, FIGURES_DIR / "mean_reversion_by_horizon.png")


def plot_r2_by_horizon(evaluation: pd.DataFrame) -> None:
    """대상별 표본 외 R² vs 시간 단위 — 베이스라인 / 기술지표 / 결합 (선형 기준)."""
    horizons = list(HORIZONS_MINUTES)
    series_to_plot = [("M2_base_linear", BLUE), ("T1_tech_linear", ORANGE), ("C1_combined_linear", AQUA)]

    fig, axes = plt.subplots(1, len(ALL_TARGETS), figsize=(17, 4.2))
    for ax, target in zip(axes, ALL_TARGETS):
        for model, color in series_to_plot:
            rows = evaluation[(evaluation["target"] == target) & (evaluation.index == model)]
            values = rows.set_index("horizon")["oos_r2_vs_M0"].reindex(horizons) * 100
            ax.plot(horizons, values, color=color, marker="o", ms=7, label=MODEL_LABELS[model])
        ax.axhline(0, color=MUTED, lw=0.8)
        if target == PRICE_TARGET:
            ax.set_ylim(-3, 1)  # 가격은 0 근처의 작은 차이가 핵심 — 과도하게 음수인 값은 아래로 잘림
        ax.set_title(TARGET_LABELS[target], loc="left")
        ax.set_xlabel("bar size")
    axes[0].set_ylabel("out-of-sample R² vs M0 (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Forecast skill by horizon: microstructure baseline vs technical indicators (walk-forward 2021–2026)",
                 x=0.01, ha="left", fontsize=11)
    save_figure(fig, FIGURES_DIR / "oos_r2_by_horizon.png")


def plot_price_economics(evaluation: pd.DataFrame) -> None:
    """가격 예측의 경제적 가치: 수수료 전·후 샤프 (선형 베이스라인 vs 기술지표)."""
    horizons = list(HORIZONS_MINUTES)
    price = evaluation[evaluation["target"] == PRICE_TARGET]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for ax, column, title in [(axes[0], "gross_sharpe", "before costs"), (axes[1], "net_sharpe", f"after {COST_BPS_PER_TRADE:.0f}bp costs")]:
        for model, color in [("M2_base_linear", BLUE), ("T1_tech_linear", ORANGE), ("C1_combined_linear", AQUA)]:
            values = price[price.index == model].set_index("horizon")[column].reindex(horizons)
            ax.plot(horizons, values, color=color, marker="o", ms=7, label=MODEL_LABELS[model])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Sign-strategy Sharpe, {title}", loc="left")
        ax.set_xlabel("bar size")
    axes[1].set_yscale("symlog")
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, FIGURES_DIR / "price_strategy_sharpe_by_horizon.png")


def plot_signal_reversion(signal_reversion: pd.DataFrame) -> None:
    """과매수·과매도 구간의 다음 봉 평균 수익률 (bp) — 시간 단위별."""
    zones = ["RSI<30 (oversold)", "RSI>70 (overbought)", "%B<0 (below band)", "%B>1 (above band)"]
    horizons = list(HORIZONS_MINUTES)
    fig, axes = plt.subplots(1, len(zones), figsize=(16, 3.8), sharey=False)
    for ax, zone in zip(axes, zones):
        rows = signal_reversion[signal_reversion.index == zone].set_index("horizon").reindex(horizons)
        colors = [BLUE if value >= 0 else ORANGE for value in rows["mean_next_return_bps"]]
        ax.bar(horizons, rows["mean_next_return_bps"], color=colors, width=0.6)
        for position, (value, t_stat) in enumerate(zip(rows["mean_next_return_bps"], rows["t_stat"])):
            ax.text(position, value, f"t={t_stat:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(zone, loc="left")
        ax.set_xlabel("bar size")
    axes[0].set_ylabel("mean next-bar return (bp)")
    fig.suptitle("Do overbought/oversold signals revert? (2018–2026)", x=0.01, ha="left", fontsize=11)
    save_figure(fig, FIGURES_DIR / "technical_signal_reversion.png")


# ===========================================================================
# 5. 요약
# ===========================================================================
def pivot_metric(evaluation: pd.DataFrame, target: str, metric: str, models: list[str]) -> pd.DataFrame:
    """행 = 시간 단위, 열 = 모형 인 표."""
    rows = evaluation[(evaluation["target"] == target) & (evaluation.index.isin(models))]
    pivot = rows.reset_index().pivot(index="horizon", columns="model", values=metric)
    return pivot.reindex(index=list(HORIZONS_MINUTES), columns=models)


def write_summary(evaluation: pd.DataFrame, reversion: pd.DataFrame, signal_reversion: pd.DataFrame,
                  yearly_price: pd.DataFrame) -> None:
    from descriptive_stats import to_markdown_table

    models = list(MODEL_LABELS)
    r2_percent = {target: pivot_metric(evaluation, target, "oos_r2_vs_M0", models) * 100 for target in ALL_TARGETS}

    def best_model_line(target):
        table = r2_percent[target]
        lines = []
        for horizon in HORIZONS_MINUTES:
            best_model = table.loc[horizon].idxmax()
            lines.append(f"{horizon} {MODEL_LABELS[best_model]} ({table.loc[horizon, best_model]:.1f}%)")
        return " · ".join(lines)

    def technical_gain(target, horizon):
        """결합 − 베이스라인 선형: 기술지표를 더했을 때 늘어난 R² (%p)."""
        table = r2_percent[target]
        return table.loc[horizon, "C1_combined_linear"] - table.loc[horizon, "M2_base_linear"]

    liquidity_reversion = reversion[reversion.index.isin(LIQUIDITY_TARGETS)].reset_index()
    liquidity_reversion = liquidity_reversion.pivot(index="horizon", columns="variable", values="half_life_deviation_hours")
    liquidity_reversion = liquidity_reversion.reindex(list(HORIZONS_MINUTES))[LIQUIDITY_TARGETS]
    return_reversion = reversion[reversion.index == "return_bps"].set_index("horizon")[["ar1_level", "ar1_t_stat", "ar1_abs_return"]]
    return_reversion = return_reversion.reindex(list(HORIZONS_MINUTES))

    price_detail_columns = ["oos_r2_vs_M0", "clark_west_p", "hit_rate", "hit_rate_if_independent",
                            "pesaran_timmermann_p", "gross_sharpe", "net_sharpe", "trades_per_day"]
    price_rows = evaluation[(evaluation["target"] == PRICE_TARGET) & (evaluation.index != "M0_benchmark")]
    price_detail = price_rows.reset_index().set_index(["horizon", "model"])[price_detail_columns]

    signal_table = signal_reversion.reset_index().pivot(index="zone", columns="horizon", values="mean_next_return_bps")
    signal_table = signal_table[list(HORIZONS_MINUTES)]
    signal_t = signal_reversion.reset_index().pivot(index="zone", columns="horizon", values="t_stat")[list(HORIZONS_MINUTES)]

    gains_rows = {}
    for target in ALL_TARGETS:
        gains_rows[TARGET_LABELS[target]] = {horizon: technical_gain(target, horizon) for horizon in HORIZONS_MINUTES}
    technical_gain_table = pd.DataFrame(gains_rows).T

    markdown = f"""# 시간 단위별 예측 분석 — 평균회귀성 · 예측 성능 · 기술지표 vs 미시구조

- 데이터: Binance BTCUSDT 1분봉 → 5분 · 15분 · 1시간 · 4시간 · 1일 봉, 학습 {TRAIN_START}~, 워크포워드 검증 {TEST_YEARS[0]}~{TEST_YEARS[-1]}
- 모형: M0 기준 / **M2·M3 미시구조 베이스라인**(선형·부스팅) / **T1·T2 기술지표만**(선형·부스팅) / **C1 결합**(선형)
- 기술지표 13개: RSI, MACD 히스토그램, 볼린저 %B·밴드폭, 스토캐스틱 %K·%D, 이동평균 괴리(10/50), 200봉 괴리, ROC, ATR, OBV 기울기, 거래량 비율, MFI — 모두 업계 표준 기간 (데이터에 맞춰 조정하지 않음)
- 재현: `python src/analyze_horizons.py` / 정의: `src/technical_indicators.py`, `src/baseline_factors.py`
- 스프레드: 1분봉 기반 스프레드 추정치는 2022년 이후 무의미(1틱 ≈ 0.001bp, `results/liquidity/summary.md`)하므로 유동성 대상은 **Amihud · Kyle's λ · 변동성**으로 비교

## 1. 평균회귀성 (2018~2026 전체)

### 유동성 충격의 반감기 (시간) — '1주 평균 대비 편차'가 절반으로 줄어드는 데 걸리는 시간
{to_markdown_table(liquidity_reversion, ".3g")}

### 수익률 되돌림 — 1차 자기상관 (음수 = 되돌림), |수익률| 자기상관 (양수 = 변동성 군집)
{to_markdown_table(return_reversion, ".3g")}

### 기술지표 과매수·과매도 신호의 되돌림 — 다음 봉 평균 수익률 (bp)
{to_markdown_table(signal_table, ".3g")}

t값:
{to_markdown_table(signal_t, ".2f")}

## 2. 시간 단위별 예측 성능 — 표본 외 R² (%, M0 대비)

### 유동성: log Amihud
{to_markdown_table(r2_percent["log_amihud"], ".3g")}

### 유동성: Kyle's λ
{to_markdown_table(r2_percent["kyle_lambda"], ".3g")}

### 변동성: log 실현변동성
{to_markdown_table(r2_percent["log_rv"], ".3g")}

### 가격: 다음 봉 수익률
{to_markdown_table(r2_percent["return_bps"], ".3g")}

시간 단위별 최고 모형
- Amihud: {best_model_line("log_amihud")}
- Kyle's λ: {best_model_line("kyle_lambda")}
- 변동성: {best_model_line("log_rv")}
- 수익률: {best_model_line("return_bps")}

## 3. 기술지표의 추가 예측력 — C1(결합) − M2(베이스라인) R² 차이 (%p)

{to_markdown_table(technical_gain_table, ".3g")}

## 4. 가격 예측 상세 (방향 · 경제적 가치)

수수료 편도 {COST_BPS_PER_TRADE:.0f}bp. 방향 검정은 Pesaran–Timmermann (예측·실제가 독립일 때 기대 적중률 대비).

{to_markdown_table(price_detail, ".3g")}

### 연도별 표본 외 R² (%) — 시간 단위 × 모형
{to_markdown_table(yearly_price, ".3g")}

## 그림

| | |
|---|---|
| ![](figures/mean_reversion_by_horizon.png) | ![](figures/oos_r2_by_horizon.png) |
| ![](figures/technical_signal_reversion.png) | ![](figures/price_strategy_sharpe_by_horizon.png) |
"""
    (OUTPUT_DIR / "summary.md").write_text(markdown, encoding="utf-8")


# ===========================================================================
# 실행
# ===========================================================================
def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 1분봉 로드", flush=True)
    parquet_files = sorted(RAW_DATA_DIR.glob("binance_btcusdt_1m_*.parquet"))
    klines = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    minute_bars = lm.prepare_minute_bars(klines)

    print("2) 시간 단위별 분석", flush=True)
    results = {}
    for horizon_name, bar_minutes in HORIZONS_MINUTES.items():
        results[horizon_name] = run_one_horizon(minute_bars, horizon_name, bar_minutes)

    evaluation = stack_by_horizon(results, "evaluation")
    reversion = stack_by_horizon(results, "reversion")
    signal_reversion = stack_by_horizon(results, "signal_reversion")
    yearly_price = stack_by_horizon(results, "yearly_price").reset_index().set_index(["horizon", "year"])

    print("3) 저장", flush=True)
    evaluation.to_csv(TABLES_DIR / "oos_evaluation_by_horizon.csv", float_format="%.6g")
    reversion.to_csv(TABLES_DIR / "mean_reversion_by_horizon.csv", float_format="%.6g")
    signal_reversion.to_csv(TABLES_DIR / "technical_signal_reversion.csv", float_format="%.6g")
    yearly_price.to_csv(TABLES_DIR / "price_oos_r2_by_year_horizon.csv", float_format="%.6g")

    plot_mean_reversion(reversion)
    plot_r2_by_horizon(evaluation)
    plot_price_economics(evaluation)
    plot_signal_reversion(signal_reversion)
    write_summary(evaluation, reversion, signal_reversion, yearly_price)
    print(f"완료 → {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
