"""
베이스라인 팩터 모형 실행 — 감성지수 제외, 1시간 단위 유동성·가격 예측
==================================================================

질문
    감성 정보 없이, 과거 가격·거래·주문흐름만으로 '다음 1시간'의
    (1) 유동성(Amihud, Kyle's λ)과 변동성  (2) 수익률  을 얼마나 예측할 수 있는가?

이 결과의 용도
    - 유동성: 예측이 잘 될수록 좋다 → 1월 이벤트 분석의 '기대 유동성'으로 그대로 쓴다
    - 가격:   예측이 안 되는 것이 정상(효율적 시장). 감성이 이 기준을 넘어서는지가 이후 검정 대상

설계 (자세한 정의는 src/baseline_factors.py)
    데이터    Binance BTCUSDT 1분봉 → 1시간, 2018-01 ~ 2026-10
    모형      M0 단순 기준 / M1 자기 과거값(HAR)+시간 패턴 / M2 전체 팩터 선형 / M3 전체 팩터 부스팅
    검증      워크포워드: 2021~2026년 각 해를 '그 이전 데이터로만' 학습한 모형으로 예측
    수수료    전략 성과에 편도 10bp (Binance 현물 기본 테이커 수수료 수준) 반영

출력
    results/baseline_factors/summary.md, tables/*.csv, figures/*.png

실행
    python src/analyze_baseline_factors.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baseline_factors as bf
import liquidity_metrics as lm
from plot_style import BLUE, MUTED, ORANGE, save_figure

# ---------------------------------------------------------------------------
# 경로 & 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_factors"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

TRAIN_START = "2018-01-01"  # 2017년 상장 초기는 시장 구조가 너무 달라 제외 (Kyle's λ가 10배 이상)
TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
COST_BPS_PER_TRADE = 10.0  # 편도 수수료 (bp)
TARGETS = ["log_amihud", "kyle_lambda", "log_rv", "return_bps"]

TARGET_LABELS = {
    "log_amihud": "log Amihud illiquidity",
    "kyle_lambda": "Kyle's λ (bp/100 BTC)",
    "log_rv": "log realized volatility",
    "return_bps": "1h return (bp)",
}


# ===========================================================================
# 1. 데이터 준비
# ===========================================================================
def load_hourly_factor_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    parquet_files = sorted(RAW_DATA_DIR.glob("binance_btcusdt_1m_*.parquet"))
    klines = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)

    minute_bars = lm.prepare_minute_bars(klines)
    hourly = bf.build_hourly_panel(minute_bars)
    return bf.build_factor_table(hourly)


# ===========================================================================
# 2. 차트
# ===========================================================================
def plot_oos_r2(evaluation: pd.DataFrame) -> None:
    """대상별 · 모형별 표본 외 R² (M0 대비). 0보다 크면 기준 모형보다 낫다."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 4))
    for ax, target in zip(axes, TARGETS):
        rows = evaluation[(evaluation["target"] == target) & (evaluation.index != "M0_benchmark")]
        values = rows["oos_r2_vs_M0"] * 100
        colors = [BLUE if value >= 0 else ORANGE for value in values]
        ax.bar(range(len(rows)), values, color=colors, width=0.6)
        ax.set_xticks(range(len(rows)), [name.split("_", 1)[1] for name in rows.index], rotation=20)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(TARGET_LABELS[target], loc="left")
        for position, value in enumerate(values):
            ax.text(position, value, f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    axes[0].set_ylabel("out-of-sample R² vs M0 (%)")
    fig.suptitle("Baseline models, walk-forward 2021–2026 (blue = beats benchmark, orange = worse)",
                 x=0.01, ha="left", fontsize=11)
    save_figure(fig, FIGURES_DIR / "oos_r2_by_model.png")


def plot_factor_t_stats(coefficients: dict[str, pd.DataFrame]) -> None:
    """전체 표본 회귀(M2)의 팩터별 Newey–West t값. |t| > 1.96 이 대략 5% 유의 수준."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(17, 7), sharey=True)
    for ax, target in zip(axes, TARGETS):
        table = coefficients[target].drop(index="intercept")
        colors = [BLUE if value >= 0 else ORANGE for value in table["t_stat"]]
        ax.barh(table.index, table["t_stat"].clip(-40, 40), color=colors)
        ax.axvline(1.96, color=MUTED, ls="--", lw=0.8)
        ax.axvline(-1.96, color=MUTED, ls="--", lw=0.8)
        ax.set_title(f"{TARGET_LABELS[target]}\nR² {table.attrs.get('r_squared', np.nan):.2f}", loc="left", fontsize=10)
        ax.set_xlabel("Newey–West t (clipped ±40)")
    axes[0].invert_yaxis()
    fig.suptitle("Full-sample factor loadings (standardized, 2018–2026)", x=0.01, ha="left", fontsize=11)
    save_figure(fig, FIGURES_DIR / "factor_t_stats.png")


def plot_liquidity_forecast_example(forecasts: dict[str, pd.DataFrame]) -> None:
    """실제 vs 예측 (최근 2주) — 베이스라인이 '기대 유동성'을 얼마나 따라가는지 눈으로 확인."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, target in zip(axes, ["log_amihud", "log_rv"]):
        recent = forecasts[target].iloc[-24 * 14:]
        ax.plot(recent.index, recent["actual"], color=MUTED, lw=1, label="actual")
        ax.plot(recent.index, recent["M2_all_linear"], color=BLUE, lw=1.5, label="M2 forecast")
        ax.set_title(TARGET_LABELS[target], loc="left")
    axes[0].legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "liquidity_forecast_recent.png")


def plot_strategy_equity(forecasts: pd.DataFrame) -> None:
    """가격 예측 부호 전략의 누적 수익 (수수료 전·후). 수수료 후에도 오르는지가 '쓸 수 있는 신호'의 기준."""
    actual = forecasts["actual"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    for model_name, color in [("M2_all_linear", BLUE), ("M3_all_boosting", ORANGE)]:
        position = np.sign(forecasts[model_name].to_numpy())
        gross = position * actual
        net = gross - np.abs(np.diff(position, prepend=0)) * COST_BPS_PER_TRADE
        ax.plot(forecasts.index, np.cumsum(gross) / 100, color=color, lw=1.2, label=f"{model_name} gross")
        ax.plot(forecasts.index, np.cumsum(net) / 100, color=color, lw=1.2, ls="--", label=f"{model_name} net of {COST_BPS_PER_TRADE:.0f}bp")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("cumulative log return (%, summed, not compounded)")
    ax.set_title("Sign strategy on 1h return forecasts, out-of-sample 2021–2026", loc="left")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, FIGURES_DIR / "price_strategy_equity.png")


# ===========================================================================
# 3. 요약
# ===========================================================================
def write_summary(evaluation: pd.DataFrame, yearly_price: pd.DataFrame, coefficients: dict[str, pd.DataFrame],
                  table: pd.DataFrame, forecasts_price: pd.DataFrame) -> None:
    from descriptive_stats import to_markdown_table

    def view(target, columns):
        rows = evaluation[evaluation["target"] == target]
        return rows[[column for column in columns if column in rows.columns]]

    liquidity_columns = ["n_hours", "rmse", "oos_r2_vs_M0", "dm_stat", "dm_p", "corr_actual_forecast"]
    price_columns = ["oos_r2_vs_M0", "clark_west_stat", "clark_west_p", "hit_rate", "hit_rate_if_independent",
                     "pesaran_timmermann_p", "gross_sharpe", "net_sharpe", "trades_per_day"]

    # 참고 기준: 같은 기간 '그냥 사서 보유'의 샤프 — 롱 쪽으로 쏠린 예측의 성과가 단순 상승장 덕분인지 구분용
    price_actual = forecasts_price["actual"]
    buy_and_hold_sharpe = price_actual.mean() / price_actual.std() * np.sqrt(bf.HOURS_PER_YEAR)

    def top_factors(target, count=6):
        coef = coefficients[target].drop(index="intercept")
        top = coef.reindex(coef["t_stat"].abs().sort_values(ascending=False).index).head(count)
        return to_markdown_table(top[["coef", "t_stat", "p_value"]])

    def corr_of(target, model):
        return evaluation.loc[(evaluation["target"] == target) & (evaluation.index == model), "corr_actual_forecast"].iloc[0]

    def r2_of(target, model):
        return evaluation.loc[(evaluation["target"] == target) & (evaluation.index == model), "oos_r2_vs_M0"].iloc[0]

    usable_hours = table.dropna().loc[TRAIN_START:].shape[0]

    m1_row = evaluation[(evaluation["target"] == "return_bps") & (evaluation.index == "M1_own_HAR")].iloc[0]
    m1_hit = m1_row["hit_rate"]
    m1_expected = m1_row["hit_rate_if_independent"]
    m1_gross = m1_row["gross_sharpe"]

    markdown = f"""# 베이스라인 팩터 모형 (감성지수 제외)

- 데이터: Binance BTCUSDT 1분봉 → 1시간, 학습 시작 {TRAIN_START}, 사용 가능 {usable_hours:,}시간
- 검증: 워크포워드 — {TEST_YEARS[0]}~{TEST_YEARS[-1]}년 각 해를 그 이전 데이터로만 학습한 모형으로 예측
- 모형: M0 단순 기준(유동성 = 직전 값, 수익률 = 0) / M1 자기 과거값(HAR)+시간 패턴 / M2 전체 팩터 선형 / M3 전체 팩터 부스팅
- 재현: `python src/analyze_baseline_factors.py` / 정의: `src/baseline_factors.py`
- 설계 메모: 유동성 모형은 모두 '최근 1주 평균 대비 편차'를 학습(수준 추세 대응), 선형 모형의 팩터는 학습 구간 0.5~99.5% 분위수로 자름(극단값 방어),
  가격 모형에는 추세 없는(정상) 팩터만 사용 — 각각의 이유는 `src/baseline_factors.py` 주석 참고

## 1. 유동성·변동성 예측 — 잘 된다 (기대 유동성으로 사용 가능)

표본 외 R²는 M0(직전 값을 그대로 쓰는 예측) 대비 오차 감소율. DM = Diebold–Mariano 검정 (단측, 모형이 M0보다 정확).

### log Amihud 비유동성
{to_markdown_table(view("log_amihud", liquidity_columns))}

### Kyle's λ (bp / 100 BTC)
{to_markdown_table(view("kyle_lambda", liquidity_columns))}

### log 실현변동성
{to_markdown_table(view("log_rv", liquidity_columns))}

**해석**
- 세 지표 모두 M1(자기 과거값 + 시간 패턴)만으로 M0 대비 오차가 크게 줄어든다 → 유동성·변동성은 **강한 지속성 + 하루 주기**를 가진다.
- M2(다른 팩터 추가)의 개선폭: Amihud {100 * (r2_of('log_amihud', 'M2_all_linear') - r2_of('log_amihud', 'M1_own_HAR')):+.1f}%p,
  Kyle's λ {100 * (r2_of('kyle_lambda', 'M2_all_linear') - r2_of('kyle_lambda', 'M1_own_HAR')):+.1f}%p,
  변동성 {100 * (r2_of('log_rv', 'M2_all_linear') - r2_of('log_rv', 'M1_own_HAR')):+.1f}%p.
- M3(부스팅) − M2(선형) 차이: Amihud {100 * (r2_of('log_amihud', 'M3_all_boosting') - r2_of('log_amihud', 'M2_all_linear')):+.1f}%p,
  Kyle's λ {100 * (r2_of('kyle_lambda', 'M3_all_boosting') - r2_of('kyle_lambda', 'M2_all_linear')):+.1f}%p,
  변동성 {100 * (r2_of('log_rv', 'M3_all_boosting') - r2_of('log_rv', 'M2_all_linear')):+.1f}%p
  → 비선형 모형의 이득이 없거나 미미하다. 선형 HAR 구조로 충분하므로 해석이 쉬운 **M2를 1월 이벤트 분석의 기대 유동성 모형으로 채택**.
- 참고: 부스팅은 처음에 Amihud에서 M0보다도 못했다(R² −17%). Amihud가 해마다 낮아져 테스트 구간이 학습 범위 밖으로 나갔기 때문
  (트리는 본 적 없는 범위를 예측 못 함) → '최근 1주 평균 대비 편차'를 학습하도록 고쳐 해결.
- Kyle's λ는 M0 대비 개선폭(R²)은 가장 크지만, 1시간(60개 1분봉) 회귀로 추정한 값이라 측정 잡음 자체가 크다
  (직전 값과의 상관 {corr_of('kyle_lambda', 'M0_benchmark'):.2f}, 예측값과 실제의 상관 {corr_of('kyle_lambda', 'M2_all_linear'):.2f} — Amihud·변동성은 0.86 이상)
  → 이벤트 분석에서는 Amihud·변동성을 주 지표로, λ는 보조로.

## 2. 가격(1시간 수익률) 예측 — 사실상 안 된다 (효율적 시장과 일치)

M0 = '수익률 0' 예측. CW = Clark–West 검정 (단측). 방향 검정 = Pesaran–Timmermann (예측·실제가 독립일 때의 적중률 `hit_rate_if_independent` 대비).
전략 = 예측 부호대로 롱/숏, 수수료 편도 {COST_BPS_PER_TRADE:.0f}bp. 참고: 같은 기간 단순 보유(buy & hold) 샤프 = {buy_and_hold_sharpe:.2f}

{to_markdown_table(view("return_bps", price_columns))}

### 연도별 표본 외 R² (%) — 일관성 확인
{to_markdown_table(yearly_price, ".3f")}

**해석**
- **크기 예측은 실패**: 세 모형 모두 표본 외 R²가 음수(M0 '수익률 0'보다 못함)이고 Clark–West 검정도 유의하지 않다.
  연도별 R²도 대부분 음수로, 안정적인 예측력이 없다.
- **방향은 아주 약하게 맞춘다**: M1의 적중률 {m1_hit:.1%}는 우연 기대치 {m1_expected:.1%}보다 높다 (Pesaran–Timmermann p ≈ 0).
  원천은 직전 1시간 수익률의 음의 계수, 즉 **단기 반전(reversal)** 이다 (아래 3절, 1σ당 약 −2bp).
  단, PT 검정은 적중 여부가 시간마다 독립이라고 가정하므로 유의성은 다소 과대평가됐을 수 있다.
- **경제적으로는 무의미**: 반전 효과(약 2bp)가 수수료(편도 {COST_BPS_PER_TRADE:.0f}bp)보다 훨씬 작고 하루 10회 이상 거래해야 해서
  수수료 후 샤프가 크게 음수다. 수수료 전 샤프({m1_gross:.2f})도 단순 보유({buy_and_hold_sharpe:.2f})와 비슷한 수준.
- **결론**: 감성 없는 베이스라인으로는 1시간 가격을 실질적으로 예측할 수 없다 (효율적 시장과 일치).
  이후 감성 신호는 이 기준을 같은 검정(CW, Pesaran–Timmermann, 연도별 일관성, 수수료 후 성과)으로 **유의하게** 넘어야 한다.
  또한 단기 반전이 존재하므로, 감성 이벤트 직후 수익률을 볼 때 **직전 1시간 수익률을 통제**해야 한다.

## 3. 팩터 해석 (전체 표본 OLS, Newey–West t값, 팩터는 표준화)

계수 = 팩터 1 표준편차 변화당 대상 변화. 표본이 7만 시간이 넘어 작은 효과도 t값이 커지므로 **크기(coef)와 표본 외 성과를 함께** 볼 것.

### log Amihud — 상위 팩터
{top_factors("log_amihud")}

### Kyle's λ — 상위 팩터
{top_factors("kyle_lambda")}

### log 실현변동성 — 상위 팩터
{top_factors("log_rv")}

### 1시간 수익률 — 상위 팩터
{top_factors("return_bps")}

## 4. 1월 분석과의 연결

- 감성 이벤트 시각 t에 대해: **비정상 유동성 = 실제 유동성(t+1) − M2 예측값(t+1)** (예측은 t까지의 정보만 사용)
- 이 잔차는 시간대 효과·변동성 군집·직전 거래 활동이 이미 제거된 값 → CLAUDE.md §4의 "변동성 군집·계절성 혼재" 리스크 대응
- 가격은 베이스라인 예측력이 0이므로, 감성 신호의 예측력 검정 기준선 = 수익률 0 (M0)
- 한계: 1분봉 근사 지표 기반. 실시간 호가창이 쌓이면 스프레드·깊이 대상으로 같은 프레임 재적용

## 그림

| | |
|---|---|
| ![](figures/oos_r2_by_model.png) | ![](figures/factor_t_stats.png) |
| ![](figures/liquidity_forecast_recent.png) | ![](figures/price_strategy_equity.png) |
"""
    (OUTPUT_DIR / "summary.md").write_text(markdown, encoding="utf-8")


# ===========================================================================
# 실행
# ===========================================================================
def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 1분봉 → 1시간 팩터 테이블")
    table, factor_groups = load_hourly_factor_table()
    print(f"   {len(table):,}시간, 팩터 {len(factor_groups['all'])}개")

    print("2) 워크포워드 예측 & 평가")
    forecasts = {}
    evaluations = []
    for target in TARGETS:
        print(f"   - {target}")
        forecasts[target] = bf.walk_forward_forecast(table, target, factor_groups, TRAIN_START, TEST_YEARS)
        evaluation = bf.evaluate_forecasts(forecasts[target], target, COST_BPS_PER_TRADE)
        evaluation.insert(0, "target", target)
        evaluations.append(evaluation)
    evaluation = pd.concat(evaluations)

    # 가격 예측의 연도별 표본 외 R² (%)
    price_forecasts = forecasts["return_bps"]
    yearly_rows = {}
    for year, rows in price_forecasts.groupby(price_forecasts.index.year):
        yearly_rows[year] = {
            model: 100 * bf.out_of_sample_r2(rows["actual"].to_numpy(), rows[model].to_numpy(), rows["M0_benchmark"].to_numpy())
            for model in ["M1_own_HAR", "M2_all_linear", "M3_all_boosting"]
        }
    yearly_price = pd.DataFrame(yearly_rows).T.rename_axis("year")

    print("3) 전체 표본 팩터 회귀 (Newey–West)")
    full_sample = table.loc[TRAIN_START:]
    coefficients = {}
    for target in TARGETS:
        factor_columns = bf.baseline_factor_columns(target, factor_groups)  # 수익률은 추세 없는 가격용 팩터
        coefficients[target] = bf.ols_newey_west(full_sample[factor_columns], full_sample[f"target_{target}"])

    print("4) 저장")
    evaluation.to_csv(TABLES_DIR / "oos_evaluation.csv", float_format="%.6g")
    yearly_price.to_csv(TABLES_DIR / "price_oos_r2_by_year.csv", float_format="%.6g")
    for target, coefficient_table in coefficients.items():
        coefficient_table.to_csv(TABLES_DIR / f"factor_loadings_{target}.csv", float_format="%.6g")

    # 1월 이벤트 분석용 '기대 유동성' (M2 표본 외 예측) — 이벤트 시각에 맞춰 잔차를 계산할 때 사용
    expected_liquidity = pd.DataFrame(
        {f"expected_{target}": forecasts[target]["M2_all_linear"] for target in ["log_amihud", "kyle_lambda", "log_rv"]}
    )
    expected_liquidity.to_csv(TABLES_DIR / "expected_liquidity_M2_oos.csv", float_format="%.6g")

    plot_oos_r2(evaluation)
    plot_factor_t_stats(coefficients)
    plot_liquidity_forecast_example(forecasts)
    plot_strategy_equity(price_forecasts)
    write_summary(evaluation, yearly_price, coefficients, table, price_forecasts)
    print(f"완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
