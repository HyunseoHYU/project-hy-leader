"""
체결(aggTrades) 기반 주문흐름 전체 기간 백테스트 — 실행 스크립트
==============================================================

목적
    감성 없이 체결 기록만으로 만든 주문흐름 불균형이 다음 봉 수익률을 예측하는지,
    예측한다면 거래비용 후에도 이익이 남는지 전체 기간(최근 3년)에 대해 검증한다.
    분석 계획과 공식은 impact/trade_flow_backtest.py 상단 docstring 참고 (결과 보기 전에 고정).

입력
    data/aggtrades/binance_btcusdt_aggtrades_{YYYY-MM-DD}.parquet
    (collectors/collect_binance_aggtrades.py 로 수집)

출력
    results/aggtrades_backtest/summary.md
    results/aggtrades_backtest/tables/*.csv
    results/aggtrades_backtest/figures/*.png
    data/aggtrades/_minute_bars.parquet   (일별 체결을 1분 봉으로 요약한 캐시, git 제외)

실행 (프로젝트 루트에서)
    python impact/analyze_aggtrades_backtest.py [--rebuild-cache]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trade_flow_backtest as tfb  # noqa: E402
from plot_style import AQUA, BLUE, MUTED, ORANGE, save_figure  # noqa: E402
from table_format import to_markdown_table  # noqa: E402

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRADES_DIR = PROJECT_ROOT / "data" / "aggtrades"
MINUTE_BAR_CACHE = TRADES_DIR / "_minute_bars.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "aggtrades_backtest"

BAR_MINUTES_LIST = [1, 5, 15, 60]          # 검증할 봉 크기
COST_SCENARIOS_BP = [0.0, 1.0, 10.0]       # 편도 수수료 시나리오
TAKER_COST_BP = 10.0                       # Binance 현물 기본 테이커 수수료 (프로젝트 공통 가정)
AGGREGATION_WORKERS = 3                    # 일별 요약 병렬 수 (메모리 7GB 환경 고려)
SIGNIFICANCE_LEVEL = 0.05

# 그림 범례용 영어 라벨 (기본 폰트에 한글 글리프가 없어 그림은 영어로 표기)
STRATEGY_LABELS_EN = {
    "M1_returns_only": "M1 past returns (ridge)",
    "M2_returns_plus_flow": "M2 returns + order flow (ridge)",
    "flow_follow": "Rule: follow imbalance",
    "flow_fade": "Rule: fade imbalance",
    "buy_and_hold": "Buy and hold",
}

STRATEGY_LABELS = {
    "M1_returns_only": "M1 과거수익률 릿지",
    "M2_returns_plus_flow": "M2 수익률+주문흐름 릿지",
    "flow_follow": "규칙: 불균형 추종",
    "flow_fade": "규칙: 불균형 역추세",
    "buy_and_hold": "단순 보유",
}


# ===========================================================================
# 1. 데이터 로드: 일별 체결 → 1분 봉 (캐시)
# ===========================================================================
def aggregate_one_file(path: Path) -> pd.DataFrame:
    """하루치 Parquet을 읽어 1분 봉으로 요약한다 (프로세스 병렬 실행용)."""
    trades = pd.read_parquet(path, columns=["price", "quantity", "is_buyer_maker", "trade_time"])
    return tfb.aggregate_trades_to_minute_bars(trades)


def load_minute_bars(rebuild: bool) -> pd.DataFrame:
    if MINUTE_BAR_CACHE.exists() and not rebuild:
        return pd.read_parquet(MINUTE_BAR_CACHE)

    paths = sorted(TRADES_DIR.glob("binance_btcusdt_aggtrades_*.parquet"))
    print(f"일별 체결 {len(paths)}개를 1분 봉으로 요약 중…", flush=True)
    with ProcessPoolExecutor(max_workers=AGGREGATION_WORKERS) as pool:
        daily_bars = list(pool.map(aggregate_one_file, paths, chunksize=8))
    minute_bars = pd.concat(daily_bars).sort_index()
    minute_bars = minute_bars[~minute_bars.index.duplicated(keep="last")]
    minute_bars.to_parquet(MINUTE_BAR_CACHE, compression="zstd")
    return minute_bars


def describe_data(minute_bars: pd.DataFrame, n_files: int) -> dict:
    """데이터 범위·품질 요약 (summary 머리말용)."""
    full_index = pd.date_range(minute_bars.index.min(), minute_bars.index.max(), freq="1min")
    empty_minutes = len(full_index) - len(minute_bars)
    return {
        "start": minute_bars.index.min(),
        "end": minute_bars.index.max(),
        "n_files": n_files,
        "n_trades": int(minute_bars["n_trades"].sum()),
        "n_minutes": len(minute_bars),
        "empty_minutes": empty_minutes,
        "empty_share_pct": empty_minutes / len(full_index) * 100,
        "first_close": float(minute_bars["close"].iloc[0]),
        "last_close": float(minute_bars["close"].iloc[-1]),
    }


# ===========================================================================
# 2. 한 봉 크기에 대한 분석
# ===========================================================================
def analyze_bar_size(minute_bars: pd.DataFrame, bar_minutes: int):
    """예측력 표, 전략 성과 표, 플라시보 표, 연도별 표, 누적손익을 만든다."""
    bars = tfb.build_bar_panel(minute_bars, bar_minutes)

    # --- 워크포워드 예측 (M1, M2) ---
    forecasts = {
        "M1_returns_only": tfb.walk_forward_forecast(bars, tfb.RETURN_FACTOR_NAMES),
        "M2_returns_plus_flow": tfb.walk_forward_forecast(
            bars, tfb.RETURN_FACTOR_NAMES + tfb.FLOW_FACTOR_NAMES
        ),
    }
    # 모든 모형이 공통으로 예측한 시점만 비교 (같은 표본에서 공정 비교)
    test_index = forecasts["M1_returns_only"].index.intersection(forecasts["M2_returns_plus_flow"].index)
    actual = bars.loc[test_index, "next_ret_bp"]

    # --- 포지션 만들기 ---
    positions = {
        name: tfb.positions_from_forecast(forecast.loc[test_index]) for name, forecast in forecasts.items()
    }
    imbalance = bars.loc[test_index, "imbalance"]
    positions["flow_follow"] = tfb.positions_from_imbalance(imbalance, follow=True)
    positions["flow_fade"] = tfb.positions_from_imbalance(imbalance, follow=False)
    positions["buy_and_hold"] = pd.Series(1.0, index=test_index)

    # --- 표 A. 예측력 ---
    prediction_rows = []
    for model_name, forecast in forecasts.items():
        forecast = forecast.loc[test_index]
        cw_t, cw_p = tfb.clark_west_p_value(actual, forecast)
        direction = tfb.direction_test(actual, forecast)
        prediction_rows.append(
            {
                "bar_min": bar_minutes,
                "model": model_name,
                "n_bars": len(test_index),
                "oos_r2_pct": tfb.out_of_sample_r2(actual, forecast) * 100,
                "clark_west_t": cw_t,
                "clark_west_p": cw_p,
                "hit_rate": direction["hit_rate"],
                "hit_rate_if_independent": direction["hit_rate_if_independent"],
                "pt_p": direction["pt_p"],
            }
        )

    # --- 표 B. 전략 성과 (비용 시나리오별) + 손익분기 비용 ---
    strategy_rows = []
    for strategy_name, position in positions.items():
        gross_summary = tfb.summarize_strategy(position, actual, bar_minutes, 0.0)
        pnl_gross = tfb.strategy_pnl_bp(position, actual, 0.0)
        traded_amount = position.diff().abs().fillna(position.abs())
        # 손익분기 비용 = 비용 0일 때 총손익 ÷ 총 거래량 (편도 bp)
        total_traded = traded_amount.sum()
        breakeven_cost_bp = pnl_gross.sum() / total_traded if total_traded > 0 else np.nan
        if strategy_name == "buy_and_hold":
            breakeven_cost_bp = np.nan  # 거래가 거의 없어 의미 없는 값
        for cost_bp in COST_SCENARIOS_BP:
            summary = tfb.summarize_strategy(position, actual, bar_minutes, cost_bp)
            strategy_rows.append(
                {
                    "bar_min": bar_minutes,
                    "strategy": strategy_name,
                    "cost_bp": cost_bp,
                    "breakeven_cost_bp": breakeven_cost_bp,
                    **summary,
                }
            )

    # --- 표 C. 플라시보 (비용 0, 학습 없는 규칙 + M2) ---
    placebo_rows = []
    for strategy_name in ("M2_returns_plus_flow", "flow_follow", "flow_fade"):
        sharpe, p_value = tfb.placebo_sharpe_p_value(positions[strategy_name], actual, bar_minutes, 0.0)
        placebo_rows.append(
            {"bar_min": bar_minutes, "strategy": strategy_name, "gross_sharpe": sharpe, "placebo_p": p_value}
        )

    # --- 표 D. 연도별 일관성: 비용 전 봉당 평균 손익(bp) ---
    yearly_rows = []
    for strategy_name in ("M2_returns_plus_flow", "flow_follow", "flow_fade"):
        pnl_gross = tfb.strategy_pnl_bp(positions[strategy_name], actual, 0.0)
        by_year = pnl_gross.groupby(pnl_gross.index.year).mean()
        for year, mean_bp in by_year.items():
            yearly_rows.append(
                {"bar_min": bar_minutes, "strategy": strategy_name, "year": int(year), "gross_mean_bp_per_bar": mean_bp}
            )

    # --- 누적 손익 곡선 (그림용, 비용 0 / 10bp) ---
    equity_curves = {}
    for strategy_name in ("M2_returns_plus_flow", "flow_follow", "flow_fade", "buy_and_hold"):
        for cost_bp in (0.0, TAKER_COST_BP):
            pnl = tfb.strategy_pnl_bp(positions[strategy_name], actual, cost_bp)
            equity_curves[(strategy_name, cost_bp)] = pnl.cumsum() / tfb.BPS * 100  # 누적 수익률 [%]

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(strategy_rows),
        pd.DataFrame(placebo_rows),
        pd.DataFrame(yearly_rows),
        equity_curves,
        (test_index.min(), test_index.max()),
    )


# ===========================================================================
# 3. 그림
# ===========================================================================
def plot_equity_curves(all_equity: dict, output_path: Path) -> None:
    """봉 크기별 누적 수익률(비용 전). 한 패널에 한 축만 사용 (이중 y축 없음)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    colors = {"M2_returns_plus_flow": BLUE, "flow_follow": ORANGE, "flow_fade": AQUA, "buy_and_hold": MUTED}
    for axis, bar_minutes in zip(axes.ravel(), BAR_MINUTES_LIST):
        for strategy_name, color in colors.items():
            curve = all_equity[bar_minutes][(strategy_name, 0.0)]
            step = max(len(curve) // 2000, 1)  # 그림이 무거워지지 않게 솎아서 그림
            axis.plot(curve.index[::step], curve.values[::step], color=color, label=STRATEGY_LABELS_EN[strategy_name])
        axis.set_title(f"{bar_minutes}-min bars: cumulative return, before fees (%)", loc="left", fontsize=10)
        axis.axhline(0, color=MUTED, linewidth=0.8)
    axes[0, 0].legend(fontsize=8, loc="best")
    save_figure(fig, output_path)


def plot_breakeven_costs(strategy_table: pd.DataFrame, output_path: Path) -> None:
    """전략이 버틸 수 있는 편도 수수료(손익분기) vs 실제 테이커 수수료 10bp."""
    fig, axis = plt.subplots(figsize=(8, 4.5))
    gross = strategy_table[strategy_table["cost_bp"] == 0.0]
    names = ["M2_returns_plus_flow", "flow_follow", "flow_fade"]
    colors = [BLUE, ORANGE, AQUA]
    bar_width = 0.25
    positions_x = np.arange(len(BAR_MINUTES_LIST))
    for offset, (name, color) in enumerate(zip(names, colors)):
        values = [
            gross[(gross["strategy"] == name) & (gross["bar_min"] == bar_minutes)]["breakeven_cost_bp"].iloc[0]
            for bar_minutes in BAR_MINUTES_LIST
        ]
        axis.bar(positions_x + (offset - 1) * bar_width, values, bar_width, color=color, label=STRATEGY_LABELS_EN[name])
    axis.axhline(TAKER_COST_BP, color=INK_LINE, linestyle="--", linewidth=1)
    axis.text(len(BAR_MINUTES_LIST) - 0.5, TAKER_COST_BP, "taker fee 10bp", va="bottom", ha="right", fontsize=8)
    axis.set_xticks(positions_x)
    axis.set_xticklabels([f"{m}-min" for m in BAR_MINUTES_LIST])
    axis.set_ylabel("break-even one-way fee (bp)")
    axis.set_title("Fee a strategy can bear vs actual taker fee", loc="left")
    axis.legend(fontsize=8)
    save_figure(fig, output_path)


INK_LINE = "#0b0b0b"


# ===========================================================================
# 4. summary.md 작성
# ===========================================================================
def build_key_findings(prediction: pd.DataFrame, strategy: pd.DataFrame, placebo: pd.DataFrame) -> list[str]:
    """표에서 기계적으로 뽑은 핵심 사실 (해석은 숫자에서 자동 생성 — 손으로 고치지 않는다)."""
    findings = []

    significant_cw = prediction[prediction["clark_west_p"] < SIGNIFICANCE_LEVEL]
    if significant_cw.empty:
        findings.append("**크기 예측**: 어떤 봉·모형도 Clark–West 검정에서 '수익률 0'을 유의하게 이기지 못했다.")
    else:
        names = ", ".join(f"{r.bar_min}분 {STRATEGY_LABELS[r.model]}(R² {r.oos_r2_pct:.3f}%)" for r in significant_cw.itertuples())
        findings.append(f"**크기 예측**: Clark–West 유의(p<{SIGNIFICANCE_LEVEL}) — {names}. (R²가 0 근처이거나 음수인데 유의한 것은 Clark–West의 성질이며, 실질적 크기는 R²로 판단)")

    gross = strategy[(strategy["cost_bp"] == 0.0) & (strategy["strategy"] != "buy_and_hold")]
    best_gross = gross.sort_values("sharpe", ascending=False).iloc[0]
    findings.append(
        f"**비용 전 최고 샤프**: {int(best_gross['bar_min'])}분 {STRATEGY_LABELS[best_gross['strategy']]} "
        f"(샤프 {best_gross['sharpe']:.2f}, 손익분기 수수료 {best_gross['breakeven_cost_bp']:.2f}bp)."
    )

    for cost_bp in COST_SCENARIOS_BP[1:]:
        net = strategy[(strategy["cost_bp"] == cost_bp) & (strategy["strategy"] != "buy_and_hold")]
        winners = net[net["sharpe"] > 0]
        if winners.empty:
            findings.append(f"**편도 {cost_bp:g}bp 수수료 후**: 샤프가 양수인 전략이 하나도 없다.")
        else:
            labels = ", ".join(
                f"{r.bar_min}분 {STRATEGY_LABELS[r.strategy]}(샤프 {r.sharpe:.2f})" for r in winners.itertuples()
            )
            findings.append(f"**편도 {cost_bp:g}bp 수수료 후** 샤프 양수: {labels}.")

    significant_placebo = placebo[placebo["placebo_p"] < SIGNIFICANCE_LEVEL]
    if significant_placebo.empty:
        findings.append(f"**플라시보**: 비용 전 샤프가 무작위 신호보다 유의하게 높은 경우가 없다(p<{SIGNIFICANCE_LEVEL} 기준).")
    else:
        labels = ", ".join(
            f"{r.bar_min}분 {STRATEGY_LABELS[r.strategy]}(p={r.placebo_p:.3f})" for r in significant_placebo.itertuples()
        )
        findings.append(f"**플라시보 유의(비용 전)**: {labels}.")
    return findings


def write_summary(data_info: dict, test_period: tuple, prediction: pd.DataFrame, strategy: pd.DataFrame,
                  placebo: pd.DataFrame, yearly: pd.DataFrame) -> None:
    findings = build_key_findings(prediction, strategy, placebo)

    strategy_display = strategy.copy()
    strategy_display["strategy"] = strategy_display["strategy"].map(STRATEGY_LABELS)
    strategy_columns = ["bar_min", "strategy", "cost_bp", "mean_bp_per_bar", "sharpe", "total_return_pct",
                        "max_drawdown_pct", "trades_per_day", "breakeven_cost_bp"]

    placebo_display = placebo.copy()
    placebo_display["strategy"] = placebo_display["strategy"].map(STRATEGY_LABELS)
    yearly_pivot = yearly.pivot_table(
        index=["bar_min", "strategy"], columns="year", values="gross_mean_bp_per_bar"
    ).reset_index()
    yearly_pivot["strategy"] = yearly_pivot["strategy"].map(STRATEGY_LABELS)
    yearly_pivot.columns = [str(c) for c in yearly_pivot.columns]

    prediction_display = prediction.copy()
    prediction_display["model"] = prediction_display["model"].map(STRATEGY_LABELS)

    lines = [
        "# 체결(aggTrades) 기반 주문흐름 백테스트 — 전체 기간",
        "",
        f"- 데이터: Binance 현물 BTCUSDT aggTrades {data_info['n_files']}일, "
        f"{data_info['start']:%Y-%m-%d} ~ {data_info['end']:%Y-%m-%d} UTC, 체결 {data_info['n_trades']:,}건",
        f"- 빈 분(체결 없음): {data_info['empty_minutes']:,}분 ({data_info['empty_share_pct']:.3f}%) — 점검 등, 해당 봉은 결측 처리",
        f"- 가격: {data_info['first_close']:,.0f} → {data_info['last_close']:,.0f} USDT",
        f"- 검증 구간(표본 외): {test_period[0]:%Y-%m-%d} ~ {test_period[1]:%Y-%m-%d} "
        f"(첫 {tfb.TRAIN_ONLY_MONTHS}개월은 학습 전용, 이후 분기마다 재학습하는 워크포워드)",
        f"- 비용: 편도 {', '.join(f'{c:g}' for c in COST_SCENARIOS_BP)}bp 시나리오 (10bp = Binance 현물 기본 테이커)",
        "- 재현: `python impact/analyze_aggtrades_backtest.py` / 공식·분석 계획: `impact/trade_flow_backtest.py`",
        "- 분석 계획은 결과를 보기 전에 고정했다 (봉 크기·모형·비용·검정 모두 코드 상단 상수). 결과를 보고 바꾸지 않았다.",
        "",
        "## 핵심 결과 (표에서 자동 생성)",
        "",
        *[f"- {finding}" for finding in findings],
        "",
        "## 1. 예측력 (표본 외)",
        "",
        "표본 외 R²는 '수익률 0' 예측 대비 오차 감소율. Clark–West = 크기 예측력, PT = Pesaran–Timmermann 방향 예측력(적중률이 독립일 때 기대치보다 높은가).",
        "표본 수가 수십만~백만 개라 아주 작은 효과도 유의해질 수 있으므로 **크기(R²·적중률)와 아래 비용 후 성과를 함께** 본다.",
        "",
        to_markdown_table(prediction_display),
        "",
        "## 2. 전략 성과 (비용 시나리오별)",
        "",
        "`mean_bp_per_bar` 봉당 평균 손익(bp), `sharpe` 연율화 샤프, `total_return_pct` 단리 누적 수익률, "
        "`trades_per_day` 하루 거래량(포지션 변화량), `breakeven_cost_bp` 이 전략이 비용 0에서 번 돈을 편도 수수료로 나눈 값 — **실제 수수료가 이보다 낮아야 이익**.",
        "",
        to_markdown_table(strategy_display[strategy_columns]),
        "",
        "## 3. 플라시보 (비용 전)",
        "",
        f"포지션 순서를 무작위로 섞어 {tfb.PLACEBO_DRAWS:,}번 반복한 샤프 분포 대비 실제 샤프의 단측 p값.",
        "",
        to_markdown_table(placebo_display),
        "",
        "## 4. 연도별 일관성 (비용 전, 봉당 평균 손익 bp)",
        "",
        to_markdown_table(yearly_pivot),
        "",
        "## 그림",
        "",
        "| | |",
        "|---|---|",
        "| ![](figures/equity_curves.png) | ![](figures/breakeven_costs.png) |",
        "",
        "## 한계",
        "",
        "- 체결 기록(aggTrades)만 사용: 호가(스프레드·깊이)가 없어 **시장가 체결 비용을 수수료로만 근사**했다. 실제로는 스프레드·슬리피지가 더해진다(BTCUSDT는 1틱이라 작지만 0은 아님).",
        "- 다음 봉 종가 기준 체결을 가정(신호 시점 즉시 체결 불가·지연 미반영).",
        "- 단일 거래소·단일 자산·3년 표본. 국면(상승/하락)별 성과는 별도 분석 필요.",
        "- 이 결과는 **감성이 없는 기준선**이다. 1월 감성 분석은 같은 검정으로 이 기준선을 넘어야 의미가 있다.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# 5. 실행
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="aggTrades 주문흐름 전체 기간 백테스트")
    parser.add_argument("--rebuild-cache", action="store_true", help="1분 봉 캐시를 다시 만든다")
    args = parser.parse_args()

    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    minute_bars = load_minute_bars(args.rebuild_cache)
    n_files = len(list(TRADES_DIR.glob("binance_btcusdt_aggtrades_*.parquet")))
    data_info = describe_data(minute_bars, n_files)
    print(f"1분 봉 {data_info['n_minutes']:,}개, 체결 {data_info['n_trades']:,}건", flush=True)

    prediction_tables, strategy_tables, placebo_tables, yearly_tables = [], [], [], []
    all_equity = {}
    test_period = None
    for bar_minutes in BAR_MINUTES_LIST:
        print(f"  분석: {bar_minutes}분 봉", flush=True)
        prediction, strategy, placebo, yearly, equity, period = analyze_bar_size(minute_bars, bar_minutes)
        prediction_tables.append(prediction)
        strategy_tables.append(strategy)
        placebo_tables.append(placebo)
        yearly_tables.append(yearly)
        all_equity[bar_minutes] = equity
        test_period = test_period or period

    prediction_all = pd.concat(prediction_tables, ignore_index=True)
    strategy_all = pd.concat(strategy_tables, ignore_index=True)
    placebo_all = pd.concat(placebo_tables, ignore_index=True)
    yearly_all = pd.concat(yearly_tables, ignore_index=True)

    prediction_all.to_csv(OUTPUT_DIR / "tables" / "prediction_oos.csv", index=False)
    strategy_all.to_csv(OUTPUT_DIR / "tables" / "strategy_performance.csv", index=False)
    placebo_all.to_csv(OUTPUT_DIR / "tables" / "placebo.csv", index=False)
    yearly_all.to_csv(OUTPUT_DIR / "tables" / "yearly_consistency.csv", index=False)

    plot_equity_curves(all_equity, OUTPUT_DIR / "figures" / "equity_curves.png")
    plot_breakeven_costs(strategy_all, OUTPUT_DIR / "figures" / "breakeven_costs.png")
    write_summary(data_info, test_period, prediction_all, strategy_all, placebo_all, yearly_all)
    print(f"완료: {OUTPUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
