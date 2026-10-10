"""
시장 미시구조 · 유동성 지표 계산 모듈
==================================

이 프로젝트의 핵심 측정 도구. 계산 함수만 모아 두었고, 실행은 아래 두 스크립트가 한다.
    src/analyze_liquidity_klines.py      Binance 1분봉 9년치로 '근사' 지표 계산
    src/analyze_liquidity_orderbook.py   실시간 호가창·체결 데이터로 '정밀' 지표 계산

지표 목록
    [A] 호가창·체결 데이터로 계산 (정밀)
        effective_spread_decomposition   유효 스프레드 = 실현 스프레드 + 가격 충격(역선택 비용)
        slippage_curve                   주문 크기별 체결 비용
        order_flow_imbalance             호가창 주문흐름 불균형 (Cont–Kukanov–Stoikov 2014)
    [B] 1분봉으로 계산 (근사, 장기 기준선용)
        daily_kyle_lambda                Kyle's λ: 순매수 1 BTC당 가격 변화
        daily_amihud_illiquidity         Amihud 비유동성: 거래대금 대비 가격 변동
        daily_roll_spread                Roll(1984) 스프레드 추정치
        daily_abdi_ranaldo_spread        Abdi–Ranaldo(2017) 고가·저가·종가 스프레드 추정치
        daily_vpin                       VPIN: 거래량 기준 주문 독성(정보거래 위험)
        estimate_pin                     PIN (Easley et al. 1996): 정보거래 확률
    [공통]
        ols_with_robust_se               단순 회귀 + 이분산 강건 표준오차

단위 약속
    bp(basis point) = 0.01% = 1e-4.  스프레드·가격 변화는 모두 bp로 맞춘다.
    거래 방향 D: +1 = 매수 주도(시장가 매수), −1 = 매도 주도(시장가 매도)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

BPS = 1e4  # 비율 → bp 변환 계수 (0.0001 → 1bp)


# ===========================================================================
# [공통] 회귀 도구
# ===========================================================================
def ols_with_robust_se(x: np.ndarray, y: np.ndarray) -> dict:
    """
    단순 회귀 y = a + b·x + e 를 최소제곱으로 추정한다.

    표준오차는 White(1980) 이분산 강건(HC0) 방식:
        Var(b) = Σ (x_i − x̄)² e_i²  /  [Σ (x_i − x̄)²]²
    금융 데이터는 변동성이 시기마다 달라 일반 표준오차가 신뢰구간을 과소평가하기 쉽다.

    반환: slope(b), intercept(a), t_stat(b의 t값), r_squared, n
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    is_valid = np.isfinite(x) & np.isfinite(y)
    x = x[is_valid]
    y = y[is_valid]

    sample_size = len(x)
    if sample_size < 3:
        return {"slope": np.nan, "intercept": np.nan, "t_stat": np.nan, "r_squared": np.nan, "n": sample_size}

    x_deviation = x - x.mean()
    y_deviation = y - y.mean()
    sum_squared_x = np.sum(x_deviation**2)
    if sum_squared_x == 0:
        return {"slope": np.nan, "intercept": np.nan, "t_stat": np.nan, "r_squared": np.nan, "n": sample_size}

    # --- 계수 ---
    slope = np.sum(x_deviation * y_deviation) / sum_squared_x
    intercept = y.mean() - slope * x.mean()

    # --- 잔차와 R² ---
    residuals = y - (intercept + slope * x)
    total_sum_squares = np.sum(y_deviation**2)
    r_squared = 1 - np.sum(residuals**2) / total_sum_squares if total_sum_squares > 0 else np.nan

    # --- 이분산 강건 표준오차 ---
    robust_variance = np.sum(x_deviation**2 * residuals**2) / sum_squared_x**2
    t_stat = slope / np.sqrt(robust_variance) if robust_variance > 0 else np.nan

    return {"slope": slope, "intercept": intercept, "t_stat": t_stat, "r_squared": r_squared, "n": sample_size}


# ===========================================================================
# [A-1] 유효 스프레드 분해 (호가 + 체결)
# ===========================================================================
QUOTE_LAG = pd.Timedelta(milliseconds=50)


def attach_prevailing_midquote(trades: pd.DataFrame, quotes: pd.DataFrame, horizon: pd.Timedelta | None = None,
                               column_name: str = "mid", quote_lag: pd.Timedelta = QUOTE_LAG) -> pd.DataFrame:
    """
    각 체결 시점(또는 체결 + horizon 시점)에 '그 직전까지 유효했던' 중간가를 붙인다 (as-of 조인).

    trades: recv_time 컬럼 필요 (우리 쪽 수신 시각)
    quotes: event_time, mid 컬럼 필요 (bookTicker 수신 시각 기준 중간가)

    시각 맞추기
        두 데이터 모두 '같은 컴퓨터의 수신 시각'으로 맞춘다 (거래소 시각과 섞으면 시계 차이만큼 어긋남).
        단, 체결 메시지(aggTrade)는 호가 메시지(bookTicker)보다 약 50ms 늦게 도착한다.
        그래서 체결 수신 시각 그대로 조회하면, 그 체결 '이후'의 호가를 체결 전 호가로 착각한다.
        스모크테스트 실측 (체결 비용이 양수로 나와야 정상인 체결 비율):
            지연 보정 0ms → 72.8%,   50ms → 99.2%,   100ms → 98.9%
        → 체결 수신 시각에서 quote_lag(50ms)를 뺀 시점의 호가를 '체결 직전 호가'로 쓴다.
    """
    trade_time_on_quote_clock = trades["recv_time"] - quote_lag
    lookup_time = trade_time_on_quote_clock if horizon is None else trade_time_on_quote_clock + horizon

    left = pd.DataFrame({"lookup_time": lookup_time, "row_order": np.arange(len(trades))}).sort_values("lookup_time")
    right = quotes[["event_time", "mid"]].rename(columns={"event_time": "lookup_time", "mid": column_name})
    right = right.sort_values("lookup_time")

    merged = pd.merge_asof(left, right, on="lookup_time", direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("row_order")

    result = trades.copy()
    result[column_name] = merged[column_name].to_numpy()
    return result


def effective_spread_decomposition(trades: pd.DataFrame, quotes: pd.DataFrame,
                                   horizons_seconds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    체결 하나하나의 거래 비용을 계산하고, 시간이 지난 뒤 가격이 어떻게 움직였는지로 분해한다.

        D   = 거래 방향 (+1 매수 주도, −1 매도 주도)
        P   = 체결가,  M_t = 체결 직전 중간가,  M_{t+Δ} = Δ초 뒤 중간가

        유효 스프레드  ES = 2·D·(P − M_t) / M_t                  ← 시장가 주문자가 실제로 낸 비용
        가격 충격      PI = 2·D·(M_{t+Δ} − M_t) / M_t            ← 체결 뒤 가격이 그 방향으로 이동한 정도
                                                                    = 정보를 가진 상대에게 진 '역선택 비용'
        실현 스프레드  RS = 2·D·(P − M_{t+Δ}) / M_t   = ES − PI  ← 유동성 공급자(지정가)가 실제로 번 몫

    해석: 감성 충격 직후 PI가 커지면 "정보거래가 늘어 유동성 공급자가 손해를 본다"는 뜻 → 스프레드 확대로 이어질 가설(H1, H3)

    반환:
        per_trade  체결별 ES, PI_Δ, RS_Δ (bp)
        summary    Δ별 거래량 가중 평균 (bp)
    """
    # --- 1. 거래 방향: is_buyer_maker=True 면 매수자가 지정가 → 매도자가 시장가로 친 '매도 주도' ---
    per_trade = attach_prevailing_midquote(trades, quotes, horizon=None, column_name="mid_before")
    per_trade["direction"] = np.where(per_trade["is_buyer_maker"], -1, 1)

    # --- 2. 유효 스프레드 ---
    price_minus_mid = per_trade["price"] - per_trade["mid_before"]
    per_trade["effective_spread_bps"] = 2 * per_trade["direction"] * price_minus_mid / per_trade["mid_before"] * BPS

    # --- 3. Δ초 뒤 중간가로 가격 충격 / 실현 스프레드 ---
    for seconds in horizons_seconds:
        mid_after_column = f"mid_after_{seconds}s"
        per_trade = attach_prevailing_midquote(per_trade, quotes, pd.Timedelta(seconds=seconds), mid_after_column)

        mid_change = per_trade[mid_after_column] - per_trade["mid_before"]
        per_trade[f"price_impact_{seconds}s_bps"] = 2 * per_trade["direction"] * mid_change / per_trade["mid_before"] * BPS
        per_trade[f"realized_spread_{seconds}s_bps"] = (
            per_trade["effective_spread_bps"] - per_trade[f"price_impact_{seconds}s_bps"]
        )

    # --- 4. 요약: 거래량 가중 평균 (큰 체결일수록 비용 비중이 큼) + 중앙값 (급변 구간 영향 확인용) ---
    summary_rows = []
    for seconds in horizons_seconds:
        usable = per_trade.dropna(subset=["mid_before", f"mid_after_{seconds}s"])
        weights = usable["quantity"]
        summary_rows.append(
            {
                "horizon_s": seconds,
                "n_trades": len(usable),
                "effective_spread_bps": np.average(usable["effective_spread_bps"], weights=weights),
                "price_impact_bps": np.average(usable[f"price_impact_{seconds}s_bps"], weights=weights),
                "realized_spread_bps": np.average(usable[f"realized_spread_{seconds}s_bps"], weights=weights),
                "median_effective_spread_bps": usable["effective_spread_bps"].median(),
                "median_price_impact_bps": usable[f"price_impact_{seconds}s_bps"].median(),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("horizon_s")
    return per_trade, summary


# ===========================================================================
# [A-2] 슬리피지 곡선 (호가창 깊이)
# ===========================================================================
def slippage_curve(depth_snapshots: pd.DataFrame, order_sizes_btc: list[float], levels: int = 10) -> pd.DataFrame:
    """
    "지금 시장가로 Q BTC를 사면/팔면 중간가보다 얼마나 불리하게 체결되는가?"

    호가창을 위에서부터 차례로 먹어 들어가며 평균 체결가(VWAP)를 계산한다.
        Q BTC 매수 시:  i단계에서 체결되는 양 = min(남은 양, i단계 잔량)
                        VWAP = Σ(체결량_i × 가격_i) / Q
        슬리피지(bp)    = (VWAP − 중간가) / 중간가 × 10,000   (매도는 부호 반대로, 항상 양수가 '비용')

    상위 levels 단계 잔량을 다 합쳐도 Q보다 적으면 계산 불가(NaN) → 그 비율도 함께 보고한다.

    반환: 주문 크기별 슬리피지의 평균·중앙값·95% 분위수, 계산 불가 비율 (매수·매도 각각)
    """
    mid_price = (depth_snapshots["bid_px_1"] + depth_snapshots["ask_px_1"]) / 2

    rows = []
    for side in ("buy", "sell"):
        # 매수 주문은 매도호가(ask)를, 매도 주문은 매수호가(bid)를 먹는다
        book_side = "ask" if side == "buy" else "bid"
        prices = depth_snapshots[[f"{book_side}_px_{level}" for level in range(1, levels + 1)]].to_numpy()
        quantities = depth_snapshots[[f"{book_side}_qty_{level}" for level in range(1, levels + 1)]].to_numpy()
        quantities = np.nan_to_num(quantities)  # 비어 있는 단계는 잔량 0

        # 각 단계 '앞'까지 쌓인 잔량 (1단계 앞 = 0)
        quantity_before_level = np.cumsum(quantities, axis=1) - quantities
        total_visible_quantity = quantities.sum(axis=1)

        for order_size in order_sizes_btc:
            # 단계별 체결량 = 0 ~ 잔량 사이로 자른 (주문량 − 앞 단계까지 누적)
            filled_at_level = np.clip(order_size - quantity_before_level, 0, quantities)
            average_fill_price = np.nansum(filled_at_level * prices, axis=1) / order_size

            if side == "buy":
                slippage_bps = (average_fill_price - mid_price) / mid_price * BPS
            else:
                slippage_bps = (mid_price - average_fill_price) / mid_price * BPS

            is_book_too_thin = total_visible_quantity < order_size
            slippage_bps = slippage_bps.where(~is_book_too_thin)

            rows.append(
                {
                    "side": side,
                    "order_size_btc": order_size,
                    "mean_bps": slippage_bps.mean(),
                    "median_bps": slippage_bps.median(),
                    "p95_bps": slippage_bps.quantile(0.95),
                    "book_too_thin_share": is_book_too_thin.mean(),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# [A-3] 주문흐름 불균형 OFI (Cont, Kukanov & Stoikov 2014)
# ===========================================================================
def order_flow_imbalance(quotes: pd.DataFrame) -> pd.Series:
    """
    최우선 호가가 바뀔 때마다 '매수 쪽 압력이 얼마나 늘었는가'를 수량(BTC)으로 잰다.

    n번째 갱신의 OFI 기여분 e_n:
        매수 쪽:  매수호가가 올랐거나 같으면 +새 잔량,  내렸거나 같으면 −이전 잔량
        매도 쪽:  매도호가가 내렸거나 같으면 −새 잔량,  올랐거나 같으면 +이전 잔량

        e_n = 1{b_n ≥ b_{n−1}}·qb_n − 1{b_n ≤ b_{n−1}}·qb_{n−1}
            − 1{a_n ≤ a_{n−1}}·qa_n + 1{a_n ≥ a_{n−1}}·qa_{n−1}

    구간 OFI = 구간 안 e_n 의 합.  양수면 매수 압력, 음수면 매도 압력.
    Cont et al. 은 짧은 구간의 중간가 변화가 OFI와 거의 선형 관계임을 보였다 → 그 기울기가 가격 충격 계수.

    quotes 컬럼: best_bid_price, best_bid_qty, best_ask_price, best_ask_qty (시간순)
    반환: 갱신별 e_n (첫 행은 비교 대상이 없어 NaN)
    """
    bid_price = quotes["best_bid_price"]
    bid_quantity = quotes["best_bid_qty"]
    ask_price = quotes["best_ask_price"]
    ask_quantity = quotes["best_ask_qty"]

    previous_bid_price = bid_price.shift(1)
    previous_bid_quantity = bid_quantity.shift(1)
    previous_ask_price = ask_price.shift(1)
    previous_ask_quantity = ask_quantity.shift(1)

    # 조건(True/False)을 반드시 1.0/0.0 숫자로 바꿔서 쓴다.
    # 주의: pandas에서 -(불리언 Series)는 '음수'가 아니라 '논리 부정(NOT)'이 된다 → 부호가 뒤집히는 버그의 원인
    bid_rose_or_same = (bid_price >= previous_bid_price).astype(float)
    bid_fell_or_same = (bid_price <= previous_bid_price).astype(float)
    ask_fell_or_same = (ask_price <= previous_ask_price).astype(float)
    ask_rose_or_same = (ask_price >= previous_ask_price).astype(float)

    bid_contribution = bid_rose_or_same * bid_quantity - bid_fell_or_same * previous_bid_quantity
    ask_contribution = ask_rose_or_same * previous_ask_quantity - ask_fell_or_same * ask_quantity

    ofi = bid_contribution + ask_contribution
    ofi.iloc[0] = np.nan
    return ofi


# ===========================================================================
# [B] 1분봉 기반 근사 지표 — 공통 준비
# ===========================================================================
def prepare_minute_bars(klines: pd.DataFrame) -> pd.DataFrame:
    """
    Binance 1분봉에 미시구조 계산용 컬럼을 추가한다.

        매수 주도 거래량 = taker_buy_base_volume       (시장가 매수로 체결된 양)
        매도 주도 거래량 = volume − 매수 주도 거래량
        순매수 거래량    = 매수 주도 − 매도 주도         (Kyle 모형의 '주문흐름' x)
        1분 수익률(bp)   = ln(종가_t / 종가_{t−1}) × 10,000
    """
    bars = klines.sort_values("time").set_index("time").copy()

    bars["buy_volume"] = bars["taker_buy_base_volume"]
    bars["sell_volume"] = bars["volume"] - bars["buy_volume"]
    bars["signed_volume"] = bars["buy_volume"] - bars["sell_volume"]

    bars["return_bps"] = np.log(bars["close"]).diff() * BPS

    # 거래소 점검 등으로 1분 이상 비는 경우, 그 사이 수익률은 '1분 수익률'이 아니므로 제외
    minutes_since_previous_bar = bars.index.to_series().diff().dt.total_seconds() / 60
    bars.loc[minutes_since_previous_bar != 1, "return_bps"] = np.nan

    bars["usd_volume"] = bars["quote_asset_volume"]
    bars["date"] = bars.index.floor("D")
    return bars


def regression_slope_by_group(x: pd.Series, y: pd.Series, group: pd.Series) -> pd.DataFrame:
    """
    그룹(예: 날짜)마다 y = a + b·x 회귀의 기울기 b와 R²를 한꺼번에 계산한다 (반복문 없이 합계로).
        b  = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²
        R² = [Σ(x−x̄)(y−ȳ)]² / [Σ(x−x̄)² · Σ(y−ȳ)²]
    """
    frame = pd.DataFrame({"x": x, "y": y, "group": group}).dropna()
    grouped = frame.groupby("group")

    frame["x_dev"] = frame["x"] - grouped["x"].transform("mean")
    frame["y_dev"] = frame["y"] - grouped["y"].transform("mean")
    frame["xy"] = frame["x_dev"] * frame["y_dev"]
    frame["xx"] = frame["x_dev"] ** 2
    frame["yy"] = frame["y_dev"] ** 2

    sums = frame.groupby("group")[["xy", "xx", "yy"]].sum()
    result = pd.DataFrame(index=sums.index)
    result["slope"] = sums["xy"] / sums["xx"]
    result["r_squared"] = sums["xy"] ** 2 / (sums["xx"] * sums["yy"])
    result["n"] = frame.groupby("group").size()
    return result


# ===========================================================================
# [B-1] Kyle's λ (일별)
# ===========================================================================
def daily_kyle_lambda(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Kyle(1985) 모형: 가격 변화 = λ × 순주문흐름 + 잡음
        1분 수익률(bp) = a + λ · 순매수 거래량(BTC) + e      ← 하루 1,440개 관측치로 매일 추정

    λ(bp/BTC) = 순매수 1 BTC가 가격을 몇 bp 움직이는가.  클수록 시장이 얕다(비유동적).
    보기 쉽게 '100 BTC당 bp'로도 표시한다.
    """
    result = regression_slope_by_group(bars["signed_volume"], bars["return_bps"], bars["date"])
    result = result.rename(columns={"slope": "kyle_lambda_bps_per_btc", "r_squared": "kyle_r_squared"})
    result["kyle_lambda_bps_per_100btc"] = result["kyle_lambda_bps_per_btc"] * 100
    return result


# ===========================================================================
# [B-2] Amihud 비유동성 (일별)
# ===========================================================================
def daily_amihud_illiquidity(bars: pd.DataFrame) -> pd.Series:
    """
    Amihud(2002):  ILLIQ = |일 수익률| / 일 거래대금
        여기서는 |일 수익률|(bp) / 일 거래대금(백만 달러) → "100만 달러 거래당 가격이 몇 bp 움직였나"
    거래대금에 비해 가격이 크게 움직일수록 비유동적. 호가 데이터 없이 쓸 수 있는 대표 지표.
    """
    daily_close = bars["close"].groupby(bars["date"]).last()
    daily_absolute_return_bps = np.log(daily_close).diff().abs() * BPS
    daily_usd_volume_millions = bars["usd_volume"].groupby(bars["date"]).sum() / 1e6
    return (daily_absolute_return_bps / daily_usd_volume_millions).rename("amihud_bps_per_musd")


# ===========================================================================
# [B-3] Roll(1984) 스프레드 (일별)
# ===========================================================================
def daily_roll_spread(bars: pd.DataFrame) -> pd.Series:
    """
    체결가가 매수호가·매도호가를 번갈아 오가면(bid-ask bounce) 연속 가격 변화가 음의 상관을 갖는다.
    Roll 은 그 공분산으로 스프레드를 역산했다:
        스프레드 = 2 · √( −Cov(ΔP_t, ΔP_{t−1}) )      (공분산이 양수면 정의되지 않음 → NaN)

    여기서는 1분 로그 수익률을 쓰므로 결과는 비율 → bp로 변환.
    """
    return_fraction = bars["return_bps"] / BPS
    previous_return_fraction = return_fraction.groupby(bars["date"]).shift(1)

    frame = pd.DataFrame({"r": return_fraction, "r_prev": previous_return_fraction, "date": bars["date"]}).dropna()
    daily_autocovariance = frame.groupby("date").apply(lambda day: np.cov(day["r"], day["r_prev"])[0, 1])

    roll_spread = 2 * np.sqrt(-daily_autocovariance.where(daily_autocovariance < 0)) * BPS
    return roll_spread.rename("roll_spread_bps")


# ===========================================================================
# [B-4] Abdi–Ranaldo(2017) 스프레드 (일별)
# ===========================================================================
def daily_abdi_ranaldo_spread(bars: pd.DataFrame) -> pd.Series:
    """
    고가·저가의 중간(η)을 '효율 가격'의 대용치로 보고, 종가가 거기서 얼마나 튀는지로 스프레드를 추정한다.
        η_t = (ln 고가_t + ln 저가_t) / 2,     c_t = ln 종가_t
        s²  = 4 · 평균[ (c_t − η_t) · (c_t − η_{t+1}) ]
        s   = √max(s², 0)
    Roll 보다 잡음에 강하고 음수 공분산 문제로 값이 비는 날이 적다.
    """
    log_close = np.log(bars["close"])
    log_mid_range = (np.log(bars["high"]) + np.log(bars["low"])) / 2
    next_log_mid_range = log_mid_range.groupby(bars["date"]).shift(-1)

    product = (log_close - log_mid_range) * (log_close - next_log_mid_range)
    mean_product = product.groupby(bars["date"]).mean()

    squared_spread = 4 * mean_product
    spread_bps = np.sqrt(squared_spread.clip(lower=0)) * BPS
    return spread_bps.rename("abdi_ranaldo_spread_bps")


# ===========================================================================
# [B-5] VPIN (Easley, López de Prado & O'Hara 2012)
# ===========================================================================
def daily_vpin(bars: pd.DataFrame, buckets_per_day: int = 50, window_buckets: int = 50) -> pd.Series:
    """
    거래량 시계(volume clock) 기반 주문 독성 지표.

        1) 거래량을 같은 크기 V의 '버킷'으로 자른다.  V = (그 달 하루 평균 거래량) / buckets_per_day
           → 거래가 활발할 때는 버킷이 빨리 차서, 시간 대신 '거래량'으로 시간을 잰다
        2) 버킷마다 주문 불균형 = |매수 주도량 − 매도 주도량| / 버킷 거래량
        3) VPIN = 최근 window_buckets 개 버킷 불균형의 평균   (0 ~ 1)

    VPIN이 높다 = 한쪽 방향 주문이 몰린다 = 정보거래 위험이 커 유동성 공급자가 물러날 가능성.
    원 논문은 체결을 '대량 분류(BVC)'로 나누지만, Binance는 시장가 매수량을 직접 주므로 그대로 쓴다.
    1분봉 단위로 버킷에 넣기 때문에(분 안에서는 쪼개지 않음) 근사치다.

    반환: 하루 마지막 시점의 VPIN 대신, 그날 끝난 버킷들의 VPIN 평균
    """
    frame = bars[["volume", "buy_volume", "sell_volume"]].copy()
    frame["month"] = frame.index.tz_localize(None).to_period("M")  # 모든 시각이 UTC라 시간대 정보는 떼도 됨

    # --- 1. 월별 버킷 크기 V (거래량 규모가 해마다 크게 달라 월 단위로 조정) ---
    daily_volume = frame["volume"].groupby(frame.index.floor("D")).sum()
    mean_daily_volume_by_month = daily_volume.groupby(daily_volume.index.tz_localize(None).to_period("M")).mean()
    bucket_size_by_month = mean_daily_volume_by_month / buckets_per_day
    frame["bucket_size"] = frame["month"].map(bucket_size_by_month)

    # --- 2. 1분봉을 버킷에 배정: 누적 거래량 / V 의 정수 부분 = 버킷 번호 ---
    #     (달마다 V가 달라지므로 '버킷 단위로 환산한 거래량'을 누적한다)
    volume_in_bucket_units = frame["volume"] / frame["bucket_size"]
    frame["bucket_id"] = np.floor(volume_in_bucket_units.cumsum()).astype(int)

    # --- 3. 버킷별 불균형 ---
    frame["time"] = frame.index
    buckets = frame.groupby("bucket_id").agg(
        end_time=("time", "last"),  # 버킷이 다 찬 시각
        volume=("volume", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
    )
    buckets = buckets[buckets["volume"] > 0]
    buckets["imbalance"] = (buckets["buy_volume"] - buckets["sell_volume"]).abs() / buckets["volume"]

    # --- 4. 이동평균 → VPIN, 그리고 하루 단위로 평균 ---
    buckets["vpin"] = buckets["imbalance"].rolling(window_buckets).mean()
    end_date = pd.to_datetime(buckets["end_time"]).dt.floor("D")
    return buckets["vpin"].groupby(end_date.to_numpy()).mean().rename("vpin")


# ===========================================================================
# [B-6] PIN (Easley, Kiefer, O'Hara & Paperman 1996)
# ===========================================================================
def pin_log_likelihood(params: np.ndarray, buys: np.ndarray, sells: np.ndarray) -> float:
    """
    PIN 모형의 로그우도 (부호 반대 → 최소화용).

    하루 동안 일어나는 일:
        확률 1−α : 정보 없음        → 매수 ~ Poisson(εb),       매도 ~ Poisson(εs)
        확률 α·δ : 나쁜 소식        → 매수 ~ Poisson(εb),       매도 ~ Poisson(εs + μ)   (정보거래자가 매도)
        확률 α(1−δ): 좋은 소식      → 매수 ~ Poisson(εb + μ),   매도 ~ Poisson(εs)       (정보거래자가 매수)

    수치 안정성: 하루 체결 수가 수십만 건이라 확률을 그대로 곱하면 0으로 사라진다.
    → 각 경우의 '로그 확률'을 구한 뒤 log-sum-exp 로 더한다. 모수와 무관한 ln(B!)·ln(S!)은 생략.
    """
    alpha, delta, mu, epsilon_buy, epsilon_sell = params

    def log_poisson_kernel(count, rate):
        return count * np.log(rate) - rate  # ln P(count | rate) 에서 −ln(count!) 생략

    def safe_log(probability):
        return np.log(max(probability, 1e-300))  # 확률이 정확히 0이 되어 log(0) 경고가 나는 것 방지

    log_no_news = safe_log(1 - alpha) + log_poisson_kernel(buys, epsilon_buy) + log_poisson_kernel(sells, epsilon_sell)
    log_bad_news = safe_log(alpha * delta) + log_poisson_kernel(buys, epsilon_buy) + log_poisson_kernel(sells, epsilon_sell + mu)
    log_good_news = safe_log(alpha * (1 - delta)) + log_poisson_kernel(buys, epsilon_buy + mu) + log_poisson_kernel(sells, epsilon_sell)

    daily_log_likelihood = special.logsumexp(np.vstack([log_no_news, log_bad_news, log_good_news]), axis=0)
    return -np.sum(daily_log_likelihood)


def estimate_pin(daily_buys: np.ndarray, daily_sells: np.ndarray) -> dict:
    """
    일별 매수·매도 체결 건수로 PIN 모형을 최대우도 추정한다.

        PIN = α·μ / (α·μ + εb + εs)
            = 전체 주문 중 '정보를 가진 거래자'의 주문이 차지하는 비율 추정치

    최적화가 국소해에 빠지기 쉬워 여러 출발점(α, δ 격자)에서 시작해 가장 좋은 해를 고른다.

    수치 최적화 요령 (실제로 겪은 문제)
        하루 체결이 수백만 건이라 μ·ε를 그대로 최적화하면 로그우도 곡면이 너무 가팔라
        최적화기가 출발점에서 움직이지 못했다(α=0.5, PIN=0.25 그대로 반환).
        → 모수를 '제약 없는 값'으로 바꿔서 최적화한다:
            α, δ  = 로지스틱(z)              (항상 0~1)
            μ, ε  = exp(z) × 하루 평균 체결 수  (항상 양수, 크기 1 근처로 정규화)

    주의 (해석 시 필수)
        암호화폐처럼 하루 체결이 수십만~수백만 건이면 Poisson 가정이 크게 어긋나(과대분산)
        α 또는 δ가 0·1에 붙는 '경계해'가 흔하다. 반환값의 boundary_flag 로 표시한다.
        그래서 이 프로젝트는 PIN을 '진단용'으로만 쓰고, 고빈도 비교에는 VPIN을 함께 본다.
    """
    buys = np.asarray(daily_buys, dtype=float)
    sells = np.asarray(daily_sells, dtype=float)
    rate_scale = (buys.mean() + sells.mean()) / 2  # 하루 평균 체결 수 (정규화 기준)

    def to_model_params(z: np.ndarray) -> np.ndarray:
        """제약 없는 값 z → 모형 모수 (α, δ, μ, εb, εs)."""
        alpha = special.expit(z[0])
        delta = special.expit(z[1])
        mu, epsilon_buy, epsilon_sell = np.exp(z[2:]) * rate_scale
        return np.array([alpha, delta, mu, epsilon_buy, epsilon_sell])

    def negative_log_likelihood(z: np.ndarray) -> float:
        return pin_log_likelihood(to_model_params(z), buys, sells)

    best_result = None
    for alpha_start in (0.1, 0.3, 0.5, 0.7, 0.9):
        for delta_start in (0.1, 0.5, 0.9):
            # 출발점: 평균 체결의 75%는 정보 없는 거래(ε), 정보거래 도착률 μ는 평균의 절반
            start = np.array(
                [
                    special.logit(alpha_start),
                    special.logit(delta_start),
                    np.log(0.5),
                    np.log(0.75 * buys.mean() / rate_scale),
                    np.log(0.75 * sells.mean() / rate_scale),
                ]
            )
            result = optimize.minimize(
                negative_log_likelihood, start, method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-6},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    alpha, delta, mu, epsilon_buy, epsilon_sell = to_model_params(best_result.x)
    pin = alpha * mu / (alpha * mu + epsilon_buy + epsilon_sell)
    is_boundary = alpha < 0.01 or alpha > 0.99 or delta < 0.01 or delta > 0.99

    return {
        "pin": pin,
        "alpha": alpha,  # 정보 사건이 일어나는 날의 비율
        "delta": delta,  # 정보 사건 중 나쁜 소식의 비율
        "mu": mu,  # 정보거래자의 하루 주문 도착률
        "epsilon_buy": epsilon_buy,  # 정보 없는 매수 도착률
        "epsilon_sell": epsilon_sell,  # 정보 없는 매도 도착률
        "n_days": len(buys),
        "converged": bool(best_result.success),
        "boundary_flag": is_boundary,
    }


def daily_trade_counts(bars: pd.DataFrame) -> pd.DataFrame:
    """
    PIN 입력용 일별 매수·매도 체결 '건수' 추정치.
    Binance 1분봉은 체결 건수(num_trades)만 주고 매수·매도별 건수는 주지 않으므로,
    각 분의 건수를 그 분의 매수 거래량 비중으로 나눈다:
        매수 건수 ≈ num_trades × (매수 주도 거래량 / 전체 거래량)
    """
    buy_share = (bars["buy_volume"] / bars["volume"]).fillna(0.5)  # 거래량 0인 분은 반반
    estimated_buy_trades = bars["num_trades"] * buy_share
    estimated_sell_trades = bars["num_trades"] - estimated_buy_trades

    counts = pd.DataFrame({"buys": estimated_buy_trades, "sells": estimated_sell_trades})
    return counts.groupby(bars["date"]).sum().round()
