"""
Binance 1분봉 기반 유동성 지표 (2017-08 ~ 2026-10)
================================================

목적
    실시간 호가창 데이터가 쌓이기 전에, 9년치 1분봉으로 유동성 지표의 '장기 기준선'을 만든다.
    1월 이벤트 분석에서 "이 시기·이 시간대의 평소 유동성"을 정하는 데 쓴다.

계산 지표 (정의는 src/liquidity_metrics.py 주석 참고)
    일별  Kyle's λ, Amihud 비유동성, Roll 스프레드, Abdi–Ranaldo 스프레드, VPIN
    분기별 PIN
    시간대별 Kyle's λ (UTC 시 단위)

입력
    data/raw/binance_btcusdt_1m_*.parquet

출력
    results/liquidity/tables/*.csv
    results/liquidity/figures/*.png
    results/liquidity/summary.md

실행
    python src/analyze_liquidity_klines.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import liquidity_metrics as lm
from plot_style import AQUA, BLUE, MUTED, ORANGE, save_figure

# ---------------------------------------------------------------------------
# 경로 & 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "liquidity"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

MIN_MINUTES_PER_DAY = 1200  # 1,440분 중 이보다 적게 남은 날(거래소 점검 등)은 일별 지표에서 제외
SMOOTHING_DAYS = 30  # 차트용 이동 중앙값 기간
FIRST_FULL_DAY = "2017-08-18"  # 상장일(8/17)은 하루가 다 차지 않으므로 다음 날부터


# ===========================================================================
# 1. 데이터 로드
# ===========================================================================
def load_binance_minute_bars() -> pd.DataFrame:
    parquet_files = sorted(RAW_DATA_DIR.glob("binance_btcusdt_1m_*.parquet"))
    if not parquet_files:
        raise SystemExit("data/raw/binance_btcusdt_1m_*.parquet 없음 — src/convert_binance_1m_parquet.py 먼저 실행")

    klines = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    bars = lm.prepare_minute_bars(klines)
    return bars[bars.index >= FIRST_FULL_DAY]


# ===========================================================================
# 2. 지표 계산
# ===========================================================================
def build_daily_liquidity_table(bars: pd.DataFrame) -> pd.DataFrame:
    """날짜별 유동성 지표를 한 표로 모은다."""
    print("  - Kyle's λ")
    kyle = lm.daily_kyle_lambda(bars)
    print("  - Amihud / Roll / Abdi-Ranaldo")
    amihud = lm.daily_amihud_illiquidity(bars)
    roll = lm.daily_roll_spread(bars)
    abdi_ranaldo = lm.daily_abdi_ranaldo_spread(bars)
    print("  - VPIN")
    vpin = lm.daily_vpin(bars)

    # 비교·통제용 일별 기본값
    bars_by_day = bars.groupby("date")

    # 실현변동성(연율화) = √(그날 1분 수익률² 합계 × 365)
    squared_return = (bars["return_bps"] / lm.BPS) ** 2
    daily_realized_variance = squared_return.groupby(bars["date"]).sum()

    daily_basics = pd.DataFrame(
        {
            "close": bars_by_day["close"].last(),
            "btc_volume": bars_by_day["volume"].sum(),
            "usd_volume_musd": bars_by_day["usd_volume"].sum() / 1e6,
            "num_trades": bars_by_day["num_trades"].sum(),
            "buy_volume_share": bars_by_day["buy_volume"].sum() / bars_by_day["volume"].sum(),
            "realized_vol_ann": np.sqrt(daily_realized_variance * 365),
            "minutes_available": bars_by_day.size(),
        }
    )

    daily = daily_basics.join([kyle, amihud, roll, abdi_ranaldo]).join(vpin)
    daily.index.name = "date"

    # 데이터가 많이 빈 날은 지표를 신뢰하기 어려우므로 지표 칸을 비운다
    is_incomplete_day = daily["minutes_available"] < MIN_MINUTES_PER_DAY
    metric_columns = [
        "kyle_lambda_bps_per_btc", "kyle_lambda_bps_per_100btc", "kyle_r_squared",
        "amihud_bps_per_musd", "roll_spread_bps", "abdi_ranaldo_spread_bps", "vpin",
    ]
    daily.loc[is_incomplete_day, metric_columns] = np.nan
    return daily


def build_yearly_table(daily: pd.DataFrame) -> pd.DataFrame:
    """연도별 중앙값 (평균은 극단값에 끌려가므로 중앙값 사용)."""
    columns = [
        "kyle_lambda_bps_per_100btc", "kyle_r_squared", "amihud_bps_per_musd",
        "roll_spread_bps", "abdi_ranaldo_spread_bps", "vpin", "realized_vol_ann", "usd_volume_musd",
    ]
    yearly = daily[columns].groupby(daily.index.year).median()
    yearly.index.name = "year"
    return yearly


def build_quarterly_pin_table(bars: pd.DataFrame) -> pd.DataFrame:
    """분기마다 PIN 모형을 따로 추정한다 (분기 ≈ 90일 → 추정에 필요한 '날' 표본)."""
    daily_counts = lm.daily_trade_counts(bars)
    quarter_of_day = daily_counts.index.tz_localize(None).to_period("Q")  # UTC 기준 분기

    rows = {}
    for quarter, counts_in_quarter in daily_counts.groupby(quarter_of_day):
        if len(counts_in_quarter) < 60:  # 마지막 분기처럼 날짜가 적으면 건너뜀
            continue
        rows[str(quarter)] = lm.estimate_pin(counts_in_quarter["buys"], counts_in_quarter["sells"])
    table = pd.DataFrame(rows).T
    table.index.name = "quarter"
    return table


def build_hourly_kyle_lambda_table(bars: pd.DataFrame) -> pd.DataFrame:
    """
    UTC 시간대별 Kyle's λ — 전체 기간의 같은 시간대 1분봉을 모아 하나의 회귀로 추정.
    감성 이벤트 효과를 볼 때 '그 시간대 평소 λ'를 빼 주기 위한 기준값.
    최근 시장 구조를 반영하도록 2024년 이후만 사용.
    """
    recent_bars = bars[bars.index >= "2024-01-01"]
    rows = []
    for hour, bars_in_hour in recent_bars.groupby(recent_bars.index.hour):
        regression = lm.ols_with_robust_se(bars_in_hour["signed_volume"], bars_in_hour["return_bps"])
        rows.append(
            {
                "hour_utc": hour,
                "kyle_lambda_bps_per_100btc": regression["slope"] * 100,
                "t_stat": regression["t_stat"],
                "r_squared": regression["r_squared"],
                "n_minutes": regression["n"],
            }
        )
    return pd.DataFrame(rows).set_index("hour_utc")


# ===========================================================================
# 3. 차트
# ===========================================================================
def plot_metric_panels(daily: pd.DataFrame) -> None:
    """주요 지표 4개를 위아래로 (각 패널은 일별 값 옅게 + 30일 이동 중앙값 진하게)."""
    panels = [
        ("kyle_lambda_bps_per_100btc", "Kyle's λ (bp per 100 BTC net buy)", True),
        ("amihud_bps_per_musd", "Amihud illiquidity (bp per $1M)", True),
        ("abdi_ranaldo_spread_bps", "Abdi–Ranaldo spread (bp)", True),
        ("vpin", "VPIN", False),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 11), sharex=True)
    for ax, (column, title, use_log_scale) in zip(axes, panels):
        values = daily[column]
        if use_log_scale:
            values = values.where(values > 0)  # 로그 축에 0·음수는 못 그림
        smoothed = values.rolling(SMOOTHING_DAYS, min_periods=10).median()

        ax.plot(values.index, values, color=BLUE, alpha=0.25, lw=0.6)
        ax.plot(smoothed.index, smoothed, color=BLUE, lw=1.6)
        if use_log_scale:
            ax.set_yscale("log")
        ax.set_title(title, loc="left")

    axes[0].text(0.99, 0.95, f"light: daily · dark: {SMOOTHING_DAYS}d median", transform=axes[0].transAxes,
                 ha="right", va="top", color=MUTED, fontsize=9)
    fig.suptitle("Binance BTCUSDT liquidity from 1-minute bars, 2017-08 ~ 2026-10", x=0.01, ha="left", fontsize=12)
    save_figure(fig, FIGURES_DIR / "daily_liquidity_metrics.png")


def plot_spread_estimators(daily: pd.DataFrame) -> None:
    """두 스프레드 추정치 비교 (30일 이동 중앙값)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for column, color, label in [
        ("roll_spread_bps", ORANGE, "Roll (1984)"),
        ("abdi_ranaldo_spread_bps", BLUE, "Abdi–Ranaldo (2017)"),
    ]:
        smoothed = daily[column].rolling(SMOOTHING_DAYS, min_periods=10).median()
        ax.plot(smoothed.index, smoothed, color=color, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("spread estimate (bp, log)")
    ax.set_title(f"Spread estimators from 1-minute bars ({SMOOTHING_DAYS}d median)", loc="left")
    ax.legend(frameon=False)
    save_figure(fig, FIGURES_DIR / "spread_estimators.png")


def plot_quarterly_pin(pin_table: pd.DataFrame) -> None:
    """분기별 PIN. 경계해(α나 δ가 0·1에 붙은 추정)는 회색으로 구분."""
    fig, ax = plt.subplots(figsize=(11, 4))
    positions = np.arange(len(pin_table))
    colors = [MUTED if is_boundary else BLUE for is_boundary in pin_table["boundary_flag"]]
    ax.bar(positions, pin_table["pin"].astype(float), color=colors, width=0.8)

    tick_step = 4  # 1년에 한 번만 라벨
    ax.set_xticks(positions[::tick_step], pin_table.index[::tick_step])
    ax.set_ylabel("PIN")
    ax.set_title("Quarterly PIN (EKOP 1996) — grey = boundary solution, interpret with care", loc="left")
    save_figure(fig, FIGURES_DIR / "quarterly_pin.png")


def plot_hourly_kyle_lambda(hourly_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(hourly_table.index, hourly_table["kyle_lambda_bps_per_100btc"], color=BLUE, width=0.8)
    ax.set_xlabel("hour (UTC)")
    ax.set_ylabel("bp per 100 BTC")
    ax.set_title("Kyle's λ by hour of day (2024-01 ~ ), pooled 1-minute regression", loc="left")
    save_figure(fig, FIGURES_DIR / "kyle_lambda_by_hour.png")


def plot_lambda_vs_volatility(daily: pd.DataFrame) -> None:
    """유동성이 나쁜 날(λ 큼)은 변동성도 큰가? 로그-로그 산점도."""
    usable = daily[(daily["kyle_lambda_bps_per_100btc"] > 0) & (daily["realized_vol_ann"] > 0)]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(usable["realized_vol_ann"], usable["kyle_lambda_bps_per_100btc"], s=8, color=AQUA, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("daily realized volatility (annualized, log)")
    ax.set_ylabel("Kyle's λ (bp / 100 BTC, log)")
    ax.set_title("Illiquidity vs volatility, daily", loc="left")
    save_figure(fig, FIGURES_DIR / "lambda_vs_volatility.png")


# ===========================================================================
# 4. 요약
# ===========================================================================
def write_summary(daily: pd.DataFrame, yearly: pd.DataFrame, pin_table: pd.DataFrame,
                  hourly: pd.DataFrame, correlation: pd.DataFrame) -> None:
    from descriptive_stats import to_markdown_table  # 같은 숫자 서식 재사용

    pin_view = pin_table[["pin", "alpha", "delta", "boundary_flag"]].copy()
    pin_view["pin"] = pin_view["pin"].astype(float)
    pin_view["alpha"] = pin_view["alpha"].astype(float)
    pin_view["delta"] = pin_view["delta"].astype(float)
    boundary_share = pin_table["boundary_flag"].astype(bool).mean()

    # 스프레드 추정치 한계 설명용 숫자
    recent_days = daily[daily.index >= "2022-01-01"]
    zero_spread_share_recent = (recent_days["abdi_ranaldo_spread_bps"] == 0).mean()
    one_tick_bps = 0.01 / daily["close"].iloc[-1] * 1e4

    peak_hour = hourly["kyle_lambda_bps_per_100btc"].idxmax()
    calm_hour = hourly["kyle_lambda_bps_per_100btc"].idxmin()

    markdown = f"""# Binance BTCUSDT 유동성 지표 (1분봉 근사)

- 데이터: Binance BTCUSDT 1분봉 {daily.index[0]:%Y-%m-%d} ~ {daily.index[-1]:%Y-%m-%d} UTC, {len(daily):,}일
- 하루 {MIN_MINUTES_PER_DAY}분 미만 남은 날(거래소 점검 등) {int((daily['minutes_available'] < MIN_MINUTES_PER_DAY).sum())}일은 지표 제외
- 재현: `python src/analyze_liquidity_klines.py` / 지표 정의: `src/liquidity_metrics.py`
- 호가 데이터 없이 1분봉으로 만든 **근사치** — 실시간 호가창 기반 정밀 지표는 `results/orderbook/summary.md`

## 1. 연도별 중앙값

{to_markdown_table(yearly)}

- `kyle_lambda_bps_per_100btc`: 순매수 100 BTC가 1분 수익률을 몇 bp 움직이는가 (클수록 비유동)
- `amihud_bps_per_musd`: 거래대금 100만 달러당 일 수익률 bp
- `roll_spread_bps` / `abdi_ranaldo_spread_bps`: 1분봉 가격 움직임으로 역산한 스프레드 추정치
- `vpin`: 거래량 버킷 주문 불균형 (0~1)

> **스프레드 추정치의 한계**: Abdi–Ranaldo 추정치가 0인 날의 비율이 {zero_spread_share_recent:.0%}(2022년 이후)다.
> BTCUSDT 실제 스프레드는 거의 항상 1틱(0.01달러 ≈ 현재 {one_tick_bps:.4f}bp)으로, 1분봉이 구분할 수 있는 크기보다
> 훨씬 작기 때문이다. Roll 추정치(약 2bp)도 스프레드가 아니라 1분 단위 가격 되돌림 잡음을 잡는 것으로 봐야 한다.
> → **최근 구간의 스프레드는 실시간 호가창 데이터로만 측정 가능.** 1분봉 지표 중에서는 Kyle's λ, Amihud, VPIN이 유효.

## 2. 지표 간 상관 (일별, 스피어만 순위상관)

{to_markdown_table(correlation, ".2f")}

## 3. 시간대별 Kyle's λ (2024-01 이후, UTC)

- 가장 비유동적: {peak_hour}시 ({hourly.loc[peak_hour, 'kyle_lambda_bps_per_100btc']:.2f} bp/100 BTC), 가장 유동적: {calm_hour}시 ({hourly.loc[calm_hour, 'kyle_lambda_bps_per_100btc']:.2f} bp/100 BTC)

{to_markdown_table(hourly)}

## 4. 분기별 PIN

- 경계해(α 또는 δ가 0·1에 붙음) 비율: {boundary_share:.0%} — 하루 체결 수십만 건에서 Poisson 가정이 어긋나는 알려진 문제.
  PIN은 진단용으로만 쓰고 고빈도 비교에는 VPIN을 함께 본다.

{to_markdown_table(pin_view)}

## 그림

| | |
|---|---|
| ![](figures/daily_liquidity_metrics.png) | ![](figures/spread_estimators.png) |
| ![](figures/kyle_lambda_by_hour.png) | ![](figures/lambda_vs_volatility.png) |
| ![](figures/quarterly_pin.png) | |
"""
    (OUTPUT_DIR / "summary.md").write_text(markdown, encoding="utf-8")


# ===========================================================================
# 실행
# ===========================================================================
def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 데이터 로드")
    bars = load_binance_minute_bars()

    print("2) 일별 지표 계산")
    daily = build_daily_liquidity_table(bars)
    yearly = build_yearly_table(daily)

    print("3) 시간대별 Kyle's λ")
    hourly = build_hourly_kyle_lambda_table(bars)

    print("4) 분기별 PIN 추정 (시간이 조금 걸림)")
    pin_table = build_quarterly_pin_table(bars)

    correlation_columns = [
        "kyle_lambda_bps_per_100btc", "amihud_bps_per_musd", "roll_spread_bps",
        "abdi_ranaldo_spread_bps", "vpin", "realized_vol_ann", "usd_volume_musd",
    ]
    correlation = daily[correlation_columns].corr(method="spearman")

    print("5) 저장")
    tables = {
        "daily_liquidity": daily,
        "yearly_liquidity": yearly,
        "kyle_lambda_by_hour": hourly,
        "quarterly_pin": pin_table,
        "liquidity_correlation": correlation,
    }
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", float_format="%.10g")

    plot_metric_panels(daily)
    plot_spread_estimators(daily)
    plot_quarterly_pin(pin_table)
    plot_hourly_kyle_lambda(hourly)
    plot_lambda_vs_volatility(daily)
    write_summary(daily, yearly, pin_table, hourly, correlation)

    print(f"완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
