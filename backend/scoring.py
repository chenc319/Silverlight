"""
Scoring engine: score each ticker -100 to +100, map to BUY/HOLD/SELL.
Market Regime computation.
Now includes TD Combo setup count and both Sequential + Combo countdown numbers.
"""

import pandas as pd
import numpy as np
from config import TICKER_GROUPS


def _find_active_demark_state(df: pd.DataFrame) -> dict:
    """
    Walk backward from the last bar to find the currently active DeMark state:
    - Active setup count and side (from the most recent incomplete or just-completed setup)
    - Active Sequential countdown number (from most recent in-progress seq countdown)
    - Active Combo countdown number (from most recent in-progress combo countdown)
    """
    n = len(df)
    if n == 0:
        return {
            'setup_count': 0, 'setup_side': None,
            'seq_cd_num': 0, 'seq_cd_side': None,
            'combo_cd_num': 0, 'combo_cd_side': None,
        }

    # Find the last non-zero setup count (active or completed)
    setup_count = 0
    setup_side = None
    for i in range(n - 1, max(n - 15, -1), -1):
        sb = int(df.iloc[i].get('td_setup_buy', 0))
        ss = int(df.iloc[i].get('td_setup_sell', 0))
        if sb > 0:
            setup_count = sb
            setup_side = 'buy'
            break
        elif ss > 0:
            setup_count = ss
            setup_side = 'sell'
            break

    # Find the highest recent sequential countdown number + bars ago
    seq_cd_num = 0
    seq_cd_side = None
    seq_cd_bars_ago = 0
    for i in range(n - 1, max(n - 60, -1), -1):
        scb = int(df.iloc[i].get('seq_cd_buy', 0))
        scs = int(df.iloc[i].get('seq_cd_sell', 0))
        if scb > 0:
            seq_cd_num = scb
            seq_cd_side = 'buy'
            seq_cd_bars_ago = (n - 1) - i
            break
        elif scs > 0:
            seq_cd_num = scs
            seq_cd_side = 'sell'
            seq_cd_bars_ago = (n - 1) - i
            break

    # Find the highest recent combo countdown number + bars ago
    combo_cd_num = 0
    combo_cd_side = None
    combo_cd_bars_ago = 0
    for i in range(n - 1, max(n - 60, -1), -1):
        ccb = int(df.iloc[i].get('combo_cd_buy', 0))
        ccs = int(df.iloc[i].get('combo_cd_sell', 0))
        if ccb > 0:
            combo_cd_num = ccb
            combo_cd_side = 'buy'
            combo_cd_bars_ago = (n - 1) - i
            break
        elif ccs > 0:
            combo_cd_num = ccs
            combo_cd_side = 'sell'
            combo_cd_bars_ago = (n - 1) - i
            break

    return {
        'setup_count': setup_count,
        'setup_side': setup_side,
        'seq_cd_num': seq_cd_num,
        'seq_cd_side': seq_cd_side,
        'seq_cd_bars_ago': seq_cd_bars_ago,
        'combo_cd_num': combo_cd_num,
        'combo_cd_side': combo_cd_side,
        'combo_cd_bars_ago': combo_cd_bars_ago,
    }


def score_ticker(df: pd.DataFrame) -> dict:
    """
    Score a single ticker based on the last bar of indicator data.
    Returns dict with all signal fields including TD Combo.
    """
    if len(df) == 0:
        return _empty_signal("???")

    last = df.iloc[-1]
    symbol = str(last.get('symbol', '???'))
    close = float(last.get('close', 0))

    # ─── Active DeMark state ───
    dm = _find_active_demark_state(df)

    # ─── Daily Score ───
    score = 0

    # Setup signals
    td_setup_buy = int(last.get('td_setup_buy', 0))
    td_setup_sell = int(last.get('td_setup_sell', 0))

    td9_daily = None
    if td_setup_buy == 9:
        score += 30
        perfected = int(last.get('td_perfected', 0))
        td9_daily = "TD9 BUY" + (" ●" if perfected else "")
    elif td_setup_sell == 9:
        score -= 30
        perfected = int(last.get('td_perfected', 0))
        td9_daily = "TD9 SELL" + (" ●" if perfected else "")

    # Sequential countdown signals
    seq_cd_buy = int(last.get('seq_cd_buy', 0))
    seq_cd_sell = int(last.get('seq_cd_sell', 0))
    td13_seq = None
    if seq_cd_buy == 13:
        score += 40
        td13_seq = "SEQ 13 BUY"
    elif seq_cd_sell == 13:
        score -= 40
        td13_seq = "SEQ 13 SELL"
    elif dm['seq_cd_num'] > 0 and dm['seq_cd_num'] < 13:
        if dm['seq_cd_side'] == 'buy':
            score += 15
        else:
            score -= 15

    # Combo countdown signals
    combo_cd_buy = int(last.get('combo_cd_buy', 0))
    combo_cd_sell = int(last.get('combo_cd_sell', 0))
    td13_combo = None
    if combo_cd_buy == 13:
        score += 40
        td13_combo = "COMBO 13 BUY"
    elif combo_cd_sell == 13:
        score -= 40
        td13_combo = "COMBO 13 SELL"
    elif dm['combo_cd_num'] > 0 and dm['combo_cd_num'] < 13:
        if dm['combo_cd_side'] == 'buy':
            score += 10
        else:
            score -= 10

    # RSI
    rsi = float(last.get('rsi14', 50))
    if rsi < 30:
        score += int(15 * (30 - rsi) / 30)
    elif rsi > 70:
        score -= int(15 * (rsi - 70) / 30)

    # Slow Stochastic (14,3,3)
    # Bullish if %K > %D; bearish if %K < %D
    # Bullish crossover under 20 in last ~2 bars = strong buy signal
    # Bearish crossover above 80 in last ~2 bars = strong sell signal
    stoch_k = float(last.get('stoch_k', 50))
    stoch_d = float(last.get('stoch_d', 50))
    prev_stoch_k = float(df.iloc[-2].get('stoch_k', 50)) if len(df) > 1 else 50
    prev_stoch_d = float(df.iloc[-2].get('stoch_d', 50)) if len(df) > 1 else 50
    prev2_stoch_k = float(df.iloc[-3].get('stoch_k', 50)) if len(df) > 2 else 50
    prev2_stoch_d = float(df.iloc[-3].get('stoch_d', 50)) if len(df) > 2 else 50

    # Base: bullish if %K > %D, bearish if %K < %D
    if stoch_k > stoch_d:
        score += 5
    elif stoch_k < stoch_d:
        score -= 5

    # Strong signal: %K crossed above %D under 20 within last 2 bars
    cross_up_today = (stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d)
    cross_up_yesterday = (prev_stoch_k > prev_stoch_d and prev2_stoch_k <= prev2_stoch_d)
    if (cross_up_today or cross_up_yesterday) and stoch_k < 20:
        score += 15  # Strong oversold bullish crossover
    elif (cross_up_today or cross_up_yesterday) and stoch_k < 30:
        score += 10  # Moderate oversold bullish crossover

    # Strong signal: %K crossed below %D above 80 within last 2 bars
    cross_dn_today = (stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d)
    cross_dn_yesterday = (prev_stoch_k < prev_stoch_d and prev2_stoch_k >= prev2_stoch_d)
    if (cross_dn_today or cross_dn_yesterday) and stoch_k > 80:
        score -= 15  # Strong overbought bearish crossover
    elif (cross_dn_today or cross_dn_yesterday) and stoch_k > 70:
        score -= 10  # Moderate overbought bearish crossover

    # MACD
    macd_line = float(last.get('macd_line', 0))
    macd_signal = float(last.get('macd_signal', 0))
    macd_hist = float(last.get('macd_hist', 0))
    prev_macd_line = float(df.iloc[-2].get('macd_line', 0)) if len(df) > 1 else 0
    prev_macd_signal = float(df.iloc[-2].get('macd_signal', 0)) if len(df) > 1 else 0

    if macd_line > macd_signal and prev_macd_line <= prev_macd_signal:
        score += 10
    elif macd_line < macd_signal and prev_macd_line >= prev_macd_signal:
        score -= 10

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

    # BB %B
    bb_mid = float(last.get('bb_mid', close))
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    # % Changes
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

    # Weekly score (simplified)
    weekly_score = max(-100, min(100, int(daily_score * 0.85)))
    weekly_signal = _score_to_signal(weekly_score)

    # ─── Richer DeMark labels ───
    # Setup label: "Bearish N" (buy setup = bearish), "Bullish N" (sell setup = bullish)
    setup_label = None
    setup_label_color = None
    if dm['setup_count'] > 0 and dm['setup_count'] <= 9:
        if dm['setup_side'] == 'buy':
            setup_label = f"Bearish {dm['setup_count']}"
            setup_label_color = "red"
        elif dm['setup_side'] == 'sell':
            setup_label = f"Bullish {dm['setup_count']}"
            setup_label_color = "green"

    # Countdown labels: buy CD = bullish (green), sell CD = bearish (red)
    # RULE: If the most recent countdown bar is more than 12 bars ago, show dash (null)
    seq_cd_label = None
    seq_cd_label_color = None
    if dm['seq_cd_num'] > 0 and dm['seq_cd_num'] <= 13 and dm['seq_cd_bars_ago'] <= 12:
        if dm['seq_cd_side'] == 'buy':
            seq_cd_label = f"Bullish {dm['seq_cd_num']}"
            seq_cd_label_color = "green"
        elif dm['seq_cd_side'] == 'sell':
            seq_cd_label = f"Bearish {dm['seq_cd_num']}"
            seq_cd_label_color = "red"

    combo_cd_label = None
    combo_cd_label_color = None
    if dm['combo_cd_num'] > 0 and dm['combo_cd_num'] <= 13 and dm['combo_cd_bars_ago'] <= 12:
        if dm['combo_cd_side'] == 'buy':
            combo_cd_label = f"Bullish {dm['combo_cd_num']}"
            combo_cd_label_color = "green"
        elif dm['combo_cd_side'] == 'sell':
            combo_cd_label = f"Bearish {dm['combo_cd_num']}"
            combo_cd_label_color = "red"

    # ─── DeMark action signal (demarkSignal) ───
    # Priority: 13 completion > perfected 9 + TDST
    demark_signal = _compute_demark_signal(df, dm, close)

    return {
        "symbol": symbol,
        "lastClose": round(close, 2),
        "pctChg1d": round(pct_chg_1d, 2),
        "pctChg5d": round(pct_chg_5d, 2),
        "dailyScore": daily_score,
        "dailySignal": daily_signal,
        "weeklyScore": weekly_score,
        "weeklySignal": weekly_signal,
        # DeMark completed signals
        "td9Daily": td9_daily,
        "td13Seq": td13_seq,
        "td13Combo": td13_combo,
        # Active DeMark state (for dashboard display)
        "setupLabel": setup_label,
        "setupLabelColor": setup_label_color,
        "seqCdLabel": seq_cd_label,
        "seqCdLabelColor": seq_cd_label_color,
        "comboCdLabel": combo_cd_label,
        "comboCdLabelColor": combo_cd_label_color,
        "demarkSignal": demark_signal,
        "tdSetupCount": dm['setup_count'],
        "tdSetupSide": dm['setup_side'],
        "seqCdNum": dm['seq_cd_num'],
        "seqCdSide": dm['seq_cd_side'],
        "comboCdNum": dm['combo_cd_num'],
        "comboCdSide": dm['combo_cd_side'],
        # Traditional indicators
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


def _compute_demark_signal(df: pd.DataFrame, dm: dict, close: float) -> str:
    """
    Compute the top-level DeMark action signal: BUY, SELL, 13+, or None.

    Rules (checked in priority order):
    1. Countdown 13 completion:
       - Buy 13 printed today or <=12 bars ago AND still active → "BUY"
       - Sell 13 printed today or <=12 bars ago AND still active → "SELL"
       - If no active countdown but a 13 printed within 12 bars → "13+"
    2. Perfected 9 + TDST:
       - Perfected buy 9 ABOVE active TDST sell support → "BUY" (active 4 bars)
       - Perfected sell 9 BELOW active TDST buy resistance → "SELL" (active 4 bars)
    """
    n = len(df)
    if n == 0:
        return None

    lookback_13 = min(12, n - 1)  # 12 bars window
    lookback_9 = min(4, n - 1)    # 4 bars window

    # --- Check for Countdown 13 completions within last 12 bars ---
    found_buy_13 = False
    found_sell_13 = False
    for i in range(n - 1, max(n - 1 - lookback_13, -1), -1):
        scb = int(df.iloc[i].get('seq_cd_buy', 0))
        ccb = int(df.iloc[i].get('combo_cd_buy', 0))
        scs = int(df.iloc[i].get('seq_cd_sell', 0))
        ccs = int(df.iloc[i].get('combo_cd_sell', 0))
        if scb == 13 or ccb == 13:
            found_buy_13 = True
        if scs == 13 or ccs == 13:
            found_sell_13 = True

    # Is there an active countdown still running?
    active_buy_cd = (dm['seq_cd_side'] == 'buy' and 0 < dm['seq_cd_num'] < 13) or \
                    (dm['combo_cd_side'] == 'buy' and 0 < dm['combo_cd_num'] < 13)
    active_sell_cd = (dm['seq_cd_side'] == 'sell' and 0 < dm['seq_cd_num'] < 13) or \
                     (dm['combo_cd_side'] == 'sell' and 0 < dm['combo_cd_num'] < 13)

    if found_buy_13:
        # Still active if buy countdown was completed (cd_num might be 13 or reset)
        return "BUY"
    if found_sell_13:
        return "SELL"

    # --- Check for Perfected 9 + TDST within last 4 bars ---
    for i in range(n - 1, max(n - 1 - lookback_9, -1), -1):
        sb = int(df.iloc[i].get('td_setup_buy', 0))
        ss = int(df.iloc[i].get('td_setup_sell', 0))
        perf = int(df.iloc[i].get('td_perfected', 0))
        if not perf:
            continue

        tdst_sell_level = float(df.iloc[i].get('tdst_sell_level', 0))
        tdst_buy_level = float(df.iloc[i].get('tdst_buy_level', 0))
        bar_close = float(df.iloc[i].get('close', 0))

        if sb == 9:
            # Perfected buy 9: if ABOVE active TDST sell support → BUY
            if tdst_sell_level > 0 and bar_close > tdst_sell_level:
                return "BUY"

        if ss == 9:
            # Perfected sell 9: if BELOW active TDST buy resistance → SELL
            if tdst_buy_level > 0 and bar_close < tdst_buy_level:
                return "SELL"

    return None


def compute_market_regime(signals: dict) -> dict:
    index_scores = []
    for ticker in TICKER_GROUPS["INDICES"]:
        if ticker in signals and "error" not in signals[ticker]:
            index_scores.append(signals[ticker]["dailyScore"])

    if not index_scores:
        return {"regime": "NEUTRAL", "avgScore": 0, "volatilityElevated": False, "justification": "No data available"}

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

    return {
        "regime": regime,
        "avgScore": round(avg_score, 1),
        "volatilityElevated": False,
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
        "td9Daily": None, "td13Seq": None, "td13Combo": None,
        "setupLabel": None, "setupLabelColor": None,
        "seqCdLabel": None, "seqCdLabelColor": None,
        "comboCdLabel": None, "comboCdLabelColor": None,
        "demarkSignal": None,
        "tdSetupCount": 0, "tdSetupSide": None,
        "seqCdNum": 0, "seqCdSide": None,
        "comboCdNum": 0, "comboCdSide": None,
        "rsi14": 50, "stochK": 50, "stochD": 50,
        "macdHist": 0, "macdLine": 0, "macdSignal": 0,
        "bbPct": 0.5, "bbUpper": 0, "bbMid": 0, "bbLower": 0,
        "relSpy20d": 0,
    }
