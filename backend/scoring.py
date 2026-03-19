"""
Scoring engine: score each ticker -100 to +100, map to BUY/HOLD/SELL.
Market Regime computation.
"""

import pandas as pd
import numpy as np
from config import TICKER_GROUPS


def score_ticker(df: pd.DataFrame) -> dict:
    """
    Score a single ticker based on the last bar of indicator data.
    Returns dict with all signal fields.
    """
    if len(df) == 0:
        return _empty_signal("???")

    last = df.iloc[-1]
    symbol = str(last.get('symbol', '???'))
    close = float(last.get('close', 0))

    # ─── Daily Score ───
    score = 0

    # DeMark signals (highest weight)
    td_setup_buy = int(last.get('td_setup_buy', 0))
    td_setup_sell = int(last.get('td_setup_sell', 0))
    td_cd_buy = int(last.get('td_countdown_buy', 0))
    td_cd_sell = int(last.get('td_countdown_sell', 0))

    td9_daily = None
    td13_daily = None

    if td_setup_buy == 9:
        score += 30
        td9_daily = "TD9 BUY"
    elif td_setup_sell == 9:
        score -= 30
        td9_daily = "TD9 SELL"

    if td_cd_buy == 13:
        score += 40
        td13_daily = "TD13 BUY"
    elif td_cd_sell == 13:
        score -= 40
        td13_daily = "TD13 SELL"

    # Active countdown in progress
    if td_cd_buy > 0 and td_cd_buy < 13:
        score += 20
    elif td_cd_sell > 0 and td_cd_sell < 13:
        score -= 20

    # RSI
    rsi = float(last.get('rsi14', 50))
    if rsi < 30:
        score += int(15 * (30 - rsi) / 30)
    elif rsi > 70:
        score -= int(15 * (rsi - 70) / 30)

    # Slow Stochastic
    stoch_k = float(last.get('stoch_k', 50))
    stoch_d = float(last.get('stoch_d', 50))
    prev_stoch_k = float(df.iloc[-2].get('stoch_k', 50)) if len(df) > 1 else 50
    prev_stoch_d = float(df.iloc[-2].get('stoch_d', 50)) if len(df) > 1 else 50

    # Crossover detection
    if stoch_k < 20 and stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d:
        score += 10
    elif stoch_k > 80 and stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d:
        score -= 10

    # MACD
    macd_line = float(last.get('macd_line', 0))
    macd_signal = float(last.get('macd_signal', 0))
    macd_hist = float(last.get('macd_hist', 0))
    prev_macd_line = float(df.iloc[-2].get('macd_line', 0)) if len(df) > 1 else 0
    prev_macd_signal = float(df.iloc[-2].get('macd_signal', 0)) if len(df) > 1 else 0

    # MACD crossover
    if macd_line > macd_signal and prev_macd_line <= prev_macd_signal:
        score += 10
    elif macd_line < macd_signal and prev_macd_line >= prev_macd_signal:
        score -= 10

    # MACD histogram direction
    prev_hist = float(df.iloc[-2].get('macd_hist', 0)) if len(df) > 1 else 0
    if macd_hist > prev_hist:
        score += 5
    elif macd_hist < prev_hist:
        score -= 5

    # Bollinger Bands
    bb_lower = float(last.get('bb_lower', close * 0.98))
    bb_upper = float(last.get('bb_upper', close * 1.02))
    if close < bb_lower:
        score += 5
    elif close > bb_upper:
        score -= 5

    # Relative to SPY
    rel_spy = float(last.get('rel_to_spy', 0))
    if rel_spy > 0:
        score += 5
    else:
        score -= 5

    # Clamp
    daily_score = max(-100, min(100, score))
    daily_signal = _score_to_signal(daily_score)

    # ─── BB %B ───
    bb_mid = float(last.get('bb_mid', close))
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    # ─── % Changes ───
    pct_chg_1d = 0.0
    pct_chg_5d = 0.0
    if len(df) > 1:
        prev_close = float(df.iloc[-2]['close'])
        if prev_close > 0:
            pct_chg_1d = ((close - prev_close) / prev_close) * 100
    if len(df) > 5:
        close_5d = float(df.iloc[-6]['close'])
        if close_5d > 0:
            pct_chg_5d = ((close - close_5d) / close_5d) * 100

    # ─── Weekly score (simplified: use daily * scaling factor) ───
    weekly_score = max(-100, min(100, int(daily_score * 0.85)))
    weekly_signal = _score_to_signal(weekly_score)

    # ─── TD Weekly (from last few bars) ───
    td9_weekly = None
    td13_weekly = None
    setup_count = int(last.get('td_setup_count', 0))
    cd_count = int(last.get('td_countdown_count', 0))

    return {
        "symbol": symbol,
        "lastClose": round(close, 2),
        "pctChg1d": round(pct_chg_1d, 2),
        "pctChg5d": round(pct_chg_5d, 2),
        "dailyScore": daily_score,
        "dailySignal": daily_signal,
        "weeklyScore": weekly_score,
        "weeklySignal": weekly_signal,
        "td9Daily": td9_daily,
        "td13Daily": td13_daily,
        "td9Weekly": td9_weekly,
        "td13Weekly": td13_weekly,
        "tdSetupCount": abs(setup_count),
        "tdCountdownCount": abs(cd_count),
        "rsi14": round(rsi, 1),
        "stochK": round(stoch_k, 1),
        "stochD": round(stoch_d, 1),
        "macdHist": round(macd_hist, 4),
        "macdLine": round(macd_line, 4),
        "macdSignal": round(macd_signal, 4),
        "bbPct": round(bb_pct, 2),
        "bbUpper": round(bb_upper, 2),
        "bbMid": round(bb_mid, 2),
        "bbLower": round(bb_lower, 2),
        "relSpy20d": round(rel_spy, 2),
    }


def compute_market_regime(signals: dict) -> dict:
    """Compute market regime from index signals."""
    index_scores = []
    for ticker in TICKER_GROUPS["INDICES"]:
        if ticker in signals:
            index_scores.append(signals[ticker]["dailyScore"])

    if not index_scores:
        return {
            "regime": "NEUTRAL",
            "avgScore": 0,
            "volatilityElevated": False,
            "justification": "No data available",
        }

    avg_score = sum(index_scores) / len(index_scores)
    buy_count = sum(1 for s in index_scores if s >= 40)
    sell_count = sum(1 for s in index_scores if s <= -40)
    total = len(index_scores)

    if avg_score > 25:
        regime = "RISK ON"
        justification = f"{buy_count} of {total} indices showing bullish DeMark signals"
    elif avg_score < -25:
        regime = "RISK OFF"
        justification = f"{sell_count} of {total} indices showing bearish signals with elevated selling pressure"
    else:
        regime = "NEUTRAL"
        neutral_count = total - buy_count - sell_count
        justification = f"Mixed signals across indices — {buy_count} bullish, {sell_count} bearish, {neutral_count} neutral"

    # VIX proxy: compute SPY 20-day realized vol
    vol_elevated = False  # Will be computed with actual data

    return {
        "regime": regime,
        "avgScore": round(avg_score, 1),
        "volatilityElevated": vol_elevated,
        "justification": justification,
    }


def _score_to_signal(score: int) -> str:
    if score >= 40:
        return "BUY"
    elif score <= -40:
        return "SELL"
    return "HOLD"


def _empty_signal(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "lastClose": 0, "pctChg1d": 0, "pctChg5d": 0,
        "dailyScore": 0, "dailySignal": "HOLD",
        "weeklyScore": 0, "weeklySignal": "HOLD",
        "td9Daily": None, "td13Daily": None,
        "td9Weekly": None, "td13Weekly": None,
        "tdSetupCount": 0, "tdCountdownCount": 0,
        "rsi14": 50, "stochK": 50, "stochD": 50,
        "macdHist": 0, "macdLine": 0, "macdSignal": 0,
        "bbPct": 0.5, "bbUpper": 0, "bbMid": 0, "bbLower": 0,
        "relSpy20d": 0,
    }
