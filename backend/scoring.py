"""
Scoring engine v2: DeMark-first directional scoring framework.

Score range: -10 to +10 (signed)
  DeMark core:       ±6 max (Countdown ±3, Setup ±2, Perfection ±1, TDST ±1)
  Confirming:        ±4 max (Stochastics ±1, RSI ±1, MACD ±1, MA alignment ±1)

Signal thresholds:
  +7 to +10  → STRONG BUY
  +4 to +6   → BUY
  +1 to +3   → HOLD (lean bullish)
  -1 to +1   → NEUTRAL
  -3 to -1   → HOLD (lean bearish)
  -6 to -4   → SELL
  -10 to -7  → STRONG SELL

Signal decay:
  Setup 9:     full 4 bars → half 5-8 bars → zero after 8
  Countdown 13: full 12 bars → half 13-24 bars → zero after 24

Multi-timeframe alignment (5-tier):
  Weekly Strong Buy/Buy (+4..+10): daily agree 1.0x, conflict 0.0x
  Weekly Weak Buy (+1..+3):        daily agree 0.75x, conflict 0.25x
  Weekly Neutral (-1..+1):         always 0.5x
  Weekly Weak Sell (-3..-1):       daily agree 0.75x, conflict 0.25x
  Weekly Strong Sell/Sell (-10..-4): daily agree 1.0x, conflict 0.0x
"""

import pandas as pd
import numpy as np
from config import TICKER_GROUPS


# ═══════════════════════════════════════
# Signal decay helpers
# ═══════════════════════════════════════

def _decay_setup(bars_ago: int) -> float:
    """Setup 9: full for 4 bars, half for 5-8, zero after 8."""
    if bars_ago <= 4:
        return 1.0
    elif bars_ago <= 8:
        return 0.5
    return 0.0


def _decay_countdown(bars_ago: int) -> float:
    """Countdown 13: full for 12 bars, half for 13-24, zero after 24."""
    if bars_ago <= 12:
        return 1.0
    elif bars_ago <= 24:
        return 0.5
    return 0.0


# ═══════════════════════════════════════
# DeMark state extraction
# ═══════════════════════════════════════

def _find_demark_state(df: pd.DataFrame) -> dict:
    """
    Walk backward from the last bar to find:
    - Most recent completed Setup 9 (buy or sell), bars_ago, and perfection status
    - Most recent Sequential Countdown 13 (buy or sell), bars_ago
    - Most recent Combo Countdown 13 (buy or sell), bars_ago
    - Active in-progress countdown number/side
    - Active TDST levels and whether they're holding
    - Active setup count (for dashboard display)
    """
    n = len(df)
    if n == 0:
        return _empty_dm_state()

    last = df.iloc[-1]
    close = float(last.get('close', 0))

    # --- Find most recent Setup 9 ---
    setup9_side = None
    setup9_bars_ago = 0
    setup9_perfected = False
    for i in range(n - 1, max(n - 30, -1), -1):
        sb = int(df.iloc[i].get('td_setup_buy', 0))
        ss = int(df.iloc[i].get('td_setup_sell', 0))
        if sb == 9:
            setup9_side = 'buy'
            setup9_bars_ago = (n - 1) - i
            setup9_perfected = bool(int(df.iloc[i].get('td_perfected', 0)))
            break
        elif ss == 9:
            setup9_side = 'sell'
            setup9_bars_ago = (n - 1) - i
            setup9_perfected = bool(int(df.iloc[i].get('td_perfected', 0)))
            break

    # --- Find most recent Sequential Countdown 13 ---
    seq13_side = None
    seq13_bars_ago = 0
    for i in range(n - 1, max(n - 60, -1), -1):
        scb = int(df.iloc[i].get('seq_cd_buy', 0))
        scs = int(df.iloc[i].get('seq_cd_sell', 0))
        if scb == 13:
            seq13_side = 'buy'
            seq13_bars_ago = (n - 1) - i
            break
        elif scs == 13:
            seq13_side = 'sell'
            seq13_bars_ago = (n - 1) - i
            break

    # --- Find most recent Combo Countdown 13 ---
    combo13_side = None
    combo13_bars_ago = 0
    for i in range(n - 1, max(n - 60, -1), -1):
        ccb = int(df.iloc[i].get('combo_cd_buy', 0))
        ccs = int(df.iloc[i].get('combo_cd_sell', 0))
        if ccb == 13:
            combo13_side = 'buy'
            combo13_bars_ago = (n - 1) - i
            break
        elif ccs == 13:
            combo13_side = 'sell'
            combo13_bars_ago = (n - 1) - i
            break

    # --- Active in-progress countdown (for dashboard labels, not scoring) ---
    active_seq_num = 0
    active_seq_side = None
    active_seq_bars_ago = 0
    for i in range(n - 1, max(n - 60, -1), -1):
        scb = int(df.iloc[i].get('seq_cd_buy', 0))
        scs = int(df.iloc[i].get('seq_cd_sell', 0))
        if scb > 0:
            active_seq_num = scb
            active_seq_side = 'buy'
            active_seq_bars_ago = (n - 1) - i
            break
        elif scs > 0:
            active_seq_num = scs
            active_seq_side = 'sell'
            active_seq_bars_ago = (n - 1) - i
            break

    active_combo_num = 0
    active_combo_side = None
    active_combo_bars_ago = 0
    for i in range(n - 1, max(n - 60, -1), -1):
        ccb = int(df.iloc[i].get('combo_cd_buy', 0))
        ccs = int(df.iloc[i].get('combo_cd_sell', 0))
        if ccb > 0:
            active_combo_num = ccb
            active_combo_side = 'buy'
            active_combo_bars_ago = (n - 1) - i
            break
        elif ccs > 0:
            active_combo_num = ccs
            active_combo_side = 'sell'
            active_combo_bars_ago = (n - 1) - i
            break

    # --- Active setup count (for dashboard) ---
    active_setup_count = 0
    active_setup_side = None
    for i in range(n - 1, max(n - 15, -1), -1):
        sb = int(df.iloc[i].get('td_setup_buy', 0))
        ss = int(df.iloc[i].get('td_setup_sell', 0))
        if sb > 0:
            active_setup_count = sb
            active_setup_side = 'buy'
            break
        elif ss > 0:
            active_setup_count = ss
            active_setup_side = 'sell'
            break

    # --- TDST status ---
    tdst_buy_level = float(last.get('tdst_buy_level', 0))   # Buy TDST = resistance
    tdst_sell_level = float(last.get('tdst_sell_level', 0))  # Sell TDST = support

    return {
        'setup9_side': setup9_side,
        'setup9_bars_ago': setup9_bars_ago,
        'setup9_perfected': setup9_perfected,
        'seq13_side': seq13_side,
        'seq13_bars_ago': seq13_bars_ago,
        'combo13_side': combo13_side,
        'combo13_bars_ago': combo13_bars_ago,
        'active_seq_num': active_seq_num,
        'active_seq_side': active_seq_side,
        'active_seq_bars_ago': active_seq_bars_ago,
        'active_combo_num': active_combo_num,
        'active_combo_side': active_combo_side,
        'active_combo_bars_ago': active_combo_bars_ago,
        'active_setup_count': active_setup_count,
        'active_setup_side': active_setup_side,
        'tdst_buy_level': tdst_buy_level,
        'tdst_sell_level': tdst_sell_level,
        'close': close,
    }


def _empty_dm_state():
    return {
        'setup9_side': None, 'setup9_bars_ago': 0, 'setup9_perfected': False,
        'seq13_side': None, 'seq13_bars_ago': 0,
        'combo13_side': None, 'combo13_bars_ago': 0,
        'active_seq_num': 0, 'active_seq_side': None, 'active_seq_bars_ago': 0,
        'active_combo_num': 0, 'active_combo_side': None, 'active_combo_bars_ago': 0,
        'active_setup_count': 0, 'active_setup_side': None,
        'tdst_buy_level': 0, 'tdst_sell_level': 0,
        'close': 0,
    }


# ═══════════════════════════════════════
# Core scoring function (single timeframe)
# ═══════════════════════════════════════

def _score_demark_core(dm: dict) -> tuple:
    """
    Compute DeMark core score (±6 max).
    Returns (score, breakdown_dict) for transparency.
    """
    score = 0.0
    breakdown = {}
    close = dm['close']

    # --- 1. Countdown 13 (±3 max) ---
    # Use the best (most recent, least decayed) of Seq and Combo
    cd13_score = 0.0
    cd13_label = None

    # Sequential 13
    seq13_raw = 0.0
    if dm['seq13_side'] == 'buy':
        seq13_raw = 3.0 * _decay_countdown(dm['seq13_bars_ago'])
    elif dm['seq13_side'] == 'sell':
        seq13_raw = -3.0 * _decay_countdown(dm['seq13_bars_ago'])

    # Combo 13
    combo13_raw = 0.0
    if dm['combo13_side'] == 'buy':
        combo13_raw = 3.0 * _decay_countdown(dm['combo13_bars_ago'])
    elif dm['combo13_side'] == 'sell':
        combo13_raw = -3.0 * _decay_countdown(dm['combo13_bars_ago'])

    # Best of the two (highest absolute value = least decayed / most relevant)
    if abs(seq13_raw) >= abs(combo13_raw) and seq13_raw != 0:
        cd13_score = seq13_raw
        cd13_label = f"Seq13 {'buy' if seq13_raw > 0 else 'sell'} ({dm['seq13_bars_ago']}bars)"
    elif combo13_raw != 0:
        cd13_score = combo13_raw
        cd13_label = f"Combo13 {'buy' if combo13_raw > 0 else 'sell'} ({dm['combo13_bars_ago']}bars)"

    score += cd13_score
    breakdown['countdown_13'] = round(cd13_score, 2)

    # --- 2. Setup 9 (±2 max) ---
    # Only if no active Countdown 13 is dominating
    setup_score = 0.0
    if dm['setup9_side']:
        decay = _decay_setup(dm['setup9_bars_ago'])
        if dm['setup9_side'] == 'buy':
            # Buy Setup 9 = bullish exhaustion of selling → bullish
            setup_score = 2.0 * decay
        elif dm['setup9_side'] == 'sell':
            # Sell Setup 9 = bullish exhaustion of buying → bearish
            setup_score = -2.0 * decay

    score += setup_score
    breakdown['setup_9'] = round(setup_score, 2)

    # --- 3. Perfection (±1 max) ---
    perf_score = 0.0
    if dm['setup9_perfected'] and dm['setup9_side']:
        # Perfection decays with its parent (Setup 9)
        decay = _decay_setup(dm['setup9_bars_ago'])
        if dm['setup9_side'] == 'buy':
            perf_score = 1.0 * decay
        else:
            perf_score = -1.0 * decay

    score += perf_score
    breakdown['perfection'] = round(perf_score, 2)

    # --- 4. TDST Status (±1 max) ---
    tdst_score = 0.0
    tdst_buy_level = dm['tdst_buy_level']   # Resistance (from buy setup)
    tdst_sell_level = dm['tdst_sell_level']  # Support (from sell setup)

    # TDST sell support holding = bearish context remains valid (sell countdown has room)
    # → -1 if price is above sell TDST support (support holding, bearish call intact)
    # TDST buy resistance holding = bullish context remains valid
    # → +1 if price is below buy TDST resistance (resistance holding, bullish call intact)
    # Sign flips on confirmed break

    if tdst_sell_level > 0 and tdst_buy_level > 0:
        # Both levels active — check which one is more relevant
        if close > tdst_buy_level:
            # Broke above resistance → bullish extension
            tdst_score = 1.0
        elif close < tdst_sell_level:
            # Broke below support → bearish extension
            tdst_score = -1.0
        else:
            # Between levels — use the direction of the most recent DeMark signal
            if cd13_score != 0:
                tdst_score = 1.0 if cd13_score > 0 else -1.0
            elif setup_score != 0:
                tdst_score = 1.0 if setup_score > 0 else -1.0
    elif tdst_buy_level > 0:
        # Only resistance active
        if close < tdst_buy_level:
            # Below resistance = resistance holding = bearish context
            tdst_score = -1.0
        else:
            # Broke above = bullish
            tdst_score = 1.0
    elif tdst_sell_level > 0:
        # Only support active
        if close > tdst_sell_level:
            # Above support = support holding = bullish (sell setup support intact)
            tdst_score = -1.0  # Sell TDST support holding means bearish call is valid
        else:
            # Broke below = bearish extension confirmed
            tdst_score = -1.0

    score += tdst_score
    breakdown['tdst'] = round(tdst_score, 2)

    return round(score, 2), breakdown


def _score_confirming(df: pd.DataFrame) -> tuple:
    """
    Compute confirming indicator score (±4 max).
    Returns (score, breakdown_dict).
    """
    if len(df) == 0:
        return 0.0, {}

    last = df.iloc[-1]
    score = 0.0
    breakdown = {}

    # --- 1. Slow Stochastic (±1) ---
    stoch_k = float(last.get('stoch_k', 50))
    stoch_d = float(last.get('stoch_d', 50))
    stoch_score = 0.0

    if stoch_k > stoch_d:
        # K > D = bullish
        if stoch_k < 20:
            stoch_score = 1.0   # Oversold bullish crossover — maximum bullish
        elif stoch_k < 30:
            stoch_score = 1.0   # Oversold zone
        elif stoch_k > 80:
            stoch_score = 0.0   # Overbought but still bullish cross — conflicting, neutral
        else:
            stoch_score = 1.0   # Mid-range bullish
    elif stoch_k < stoch_d:
        # K < D = bearish
        if stoch_k > 80:
            stoch_score = -1.0  # Overbought bearish crossover — maximum bearish
        elif stoch_k > 70:
            stoch_score = -1.0  # Overbought zone
        elif stoch_k < 20:
            stoch_score = 0.0   # Oversold but still bearish cross — conflicting
        else:
            stoch_score = -1.0  # Mid-range bearish

    score += stoch_score
    breakdown['stochastic'] = stoch_score

    # --- 2. RSI (±1) ---
    rsi = float(last.get('rsi14', 50))
    rsi_score = 0.0
    if rsi < 30:
        rsi_score = 1.0    # Oversold → bullish
    elif rsi > 70:
        rsi_score = -1.0   # Overbought → bearish
    # 30-70 = neutral, 0

    score += rsi_score
    breakdown['rsi'] = rsi_score

    # --- 3. MACD (±1) ---
    macd_line = float(last.get('macd_line', 0))
    macd_signal = float(last.get('macd_signal', 0))
    macd_hist = float(last.get('macd_hist', 0))
    prev_hist = float(df.iloc[-2].get('macd_hist', 0)) if len(df) > 1 else 0
    macd_score = 0.0

    if macd_line > macd_signal:
        macd_score = 1.0   # Bullish — MACD above signal
    elif macd_line < macd_signal:
        macd_score = -1.0  # Bearish — MACD below signal
    # If equal or very close, check histogram momentum
    elif macd_hist > prev_hist:
        macd_score = 0.5
    elif macd_hist < prev_hist:
        macd_score = -0.5

    score += macd_score
    breakdown['macd'] = macd_score

    # --- 4. MA Alignment (±1) ---
    # 50-day vs 200-day moving average
    # We need to compute these from the dataframe
    ma_score = 0.0
    if len(df) >= 200:
        close_series = df['close'].astype(float)
        ma50 = close_series.rolling(50).mean().iloc[-1]
        ma200 = close_series.rolling(200).mean().iloc[-1]
        if pd.notna(ma50) and pd.notna(ma200):
            if ma50 > ma200:
                ma_score = 1.0   # Golden cross structure
            else:
                ma_score = -1.0  # Death cross structure
    elif len(df) >= 50:
        # Not enough for 200 DMA, just check price vs 50 DMA
        close_series = df['close'].astype(float)
        ma50 = close_series.rolling(50).mean().iloc[-1]
        last_close = float(df.iloc[-1].get('close', 0))
        if pd.notna(ma50):
            if last_close > ma50:
                ma_score = 0.5
            else:
                ma_score = -0.5

    score += ma_score
    breakdown['ma_alignment'] = ma_score

    return round(score, 2), breakdown


# ═══════════════════════════════════════
# Signal thresholds
# ═══════════════════════════════════════

def _score_to_signal(score: float) -> str:
    """Map score to signal label."""
    if score >= 7:
        return "STRONG BUY"
    elif score >= 4:
        return "BUY"
    elif score > 1:
        return "HOLD"
    elif score >= -1:
        return "NEUTRAL"
    elif score > -4:
        return "HOLD"
    elif score > -7:
        return "SELL"
    else:
        return "STRONG SELL"


# ═══════════════════════════════════════
# Multi-timeframe alignment
# ═══════════════════════════════════════

def _weekly_category(weekly_score: float) -> str:
    """Classify weekly score into 5 tiers."""
    if weekly_score >= 4:
        return "strong_bull"
    elif weekly_score >= 1:
        return "weak_bull"
    elif weekly_score >= -1:
        return "neutral"
    elif weekly_score >= -4:
        return "weak_bear"
    else:
        return "strong_bear"


def _apply_mtf_alignment(daily_score: float, weekly_score: float) -> float:
    """
    Apply multi-timeframe alignment multiplier.
    Returns the adjusted daily score.
    """
    weekly_cat = _weekly_category(weekly_score)

    # Determine if daily agrees or conflicts with weekly
    daily_direction = 1 if daily_score > 0 else (-1 if daily_score < 0 else 0)
    weekly_direction = 1 if weekly_score > 0 else (-1 if weekly_score < 0 else 0)
    agrees = (daily_direction == weekly_direction) or daily_direction == 0

    multipliers = {
        "strong_bull": (1.0, 0.0),   # (agree, conflict)
        "weak_bull":   (0.75, 0.25),
        "neutral":     (0.5, 0.5),
        "weak_bear":   (0.75, 0.25),
        "strong_bear": (1.0, 0.0),
    }

    agree_mult, conflict_mult = multipliers[weekly_cat]
    mult = agree_mult if agrees else conflict_mult

    return round(daily_score * mult, 2)


# ═══════════════════════════════════════
# Weekly resampling
# ═══════════════════════════════════════

def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLC data to weekly bars.
    Preserves all indicator columns by taking the last value of each week.
    """
    if len(df) == 0:
        return df

    wdf = df.copy()
    wdf['date_dt'] = pd.to_datetime(wdf['date'])
    wdf = wdf.set_index('date_dt')

    # OHLC resampling
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'date': 'last',
    }

    # For indicator columns, take the last value of the week
    indicator_cols = [c for c in wdf.columns if c not in
                      ['open', 'high', 'low', 'close', 'volume', 'date', 'symbol']
                      and c in wdf.columns]
    for col in indicator_cols:
        agg_dict[col] = 'last'

    if 'symbol' in wdf.columns:
        agg_dict['symbol'] = 'last'

    weekly = wdf.resample('W-FRI').agg(agg_dict)
    weekly = weekly.dropna(subset=['close'])
    weekly = weekly.reset_index(drop=True)

    return weekly


# ═══════════════════════════════════════
# Score a single timeframe
# ═══════════════════════════════════════

def _score_single_timeframe(df: pd.DataFrame) -> tuple:
    """
    Score a single timeframe (daily or weekly).
    Returns (total_score, dm_state, dm_breakdown, confirming_breakdown).
    """
    dm = _find_demark_state(df)
    dm_score, dm_breakdown = _score_demark_core(dm)
    conf_score, conf_breakdown = _score_confirming(df)

    total = round(dm_score + conf_score, 2)
    # Clamp to ±10
    total = max(-10.0, min(10.0, total))

    return total, dm, dm_breakdown, conf_breakdown


# ═══════════════════════════════════════
# DeMark action signal
# ═══════════════════════════════════════

def _compute_demark_signal(df: pd.DataFrame, dm: dict) -> str:
    """
    Compute the top-level DeMark action signal: BUY, SELL, 13+, or None.

    Rules (checked in priority order):
    1. Countdown 13 completion:
       - Buy 13 printed today or <=12 bars ago → "BUY"
       - Sell 13 printed today or <=12 bars ago → "SELL"
       - If 13 printed >12 bars ago but <=24 → "13+" (aging)
    2. Perfected 9 + TDST (within 4 bars):
       - Perfected buy 9 ABOVE active TDST sell support → "BUY"
       - Perfected sell 9 BELOW active TDST buy resistance → "SELL"
    """
    n = len(df)
    if n == 0:
        return None

    close = dm['close']

    # --- Check for Countdown 13 completions ---
    # Sequential or Combo buy 13
    if dm['seq13_side'] == 'buy' and dm['seq13_bars_ago'] <= 12:
        return "BUY"
    if dm['combo13_side'] == 'buy' and dm['combo13_bars_ago'] <= 12:
        return "BUY"
    if dm['seq13_side'] == 'sell' and dm['seq13_bars_ago'] <= 12:
        return "SELL"
    if dm['combo13_side'] == 'sell' and dm['combo13_bars_ago'] <= 12:
        return "SELL"

    # 13+ (aging but not dead)
    if dm['seq13_side'] == 'buy' and dm['seq13_bars_ago'] <= 24:
        return "13+"
    if dm['combo13_side'] == 'buy' and dm['combo13_bars_ago'] <= 24:
        return "13+"
    if dm['seq13_side'] == 'sell' and dm['seq13_bars_ago'] <= 24:
        return "13+"
    if dm['combo13_side'] == 'sell' and dm['combo13_bars_ago'] <= 24:
        return "13+"

    # --- Check for Perfected 9 + TDST (within 4 bars) ---
    if dm['setup9_side'] and dm['setup9_perfected'] and dm['setup9_bars_ago'] <= 4:
        tdst_sell_level = dm['tdst_sell_level']
        tdst_buy_level = dm['tdst_buy_level']

        if dm['setup9_side'] == 'buy':
            # Perfected buy 9: BUY if above TDST sell support
            if tdst_sell_level > 0 and close > tdst_sell_level:
                return "BUY"
            elif tdst_sell_level == 0:
                return "BUY"  # No opposing TDST, take the signal

        elif dm['setup9_side'] == 'sell':
            # Perfected sell 9: SELL if below TDST buy resistance
            if tdst_buy_level > 0 and close < tdst_buy_level:
                return "SELL"
            elif tdst_buy_level == 0:
                return "SELL"

    return None


# ═══════════════════════════════════════
# Main entry point: score a ticker
# ═══════════════════════════════════════

def score_ticker(df: pd.DataFrame) -> dict:
    """
    Score a single ticker with the DeMark-first framework.
    Computes daily score, weekly score (from resampled data),
    and applies multi-timeframe alignment.
    """
    if len(df) == 0:
        return _empty_signal("???")

    last = df.iloc[-1]
    symbol = str(last.get('symbol', '???'))
    close = float(last.get('close', 0))

    # ─── Daily scoring ───
    daily_raw, dm_daily, dm_bd, conf_bd = _score_single_timeframe(df)

    # ─── Weekly scoring (resample daily to weekly, score independently) ───
    weekly_df = _resample_to_weekly(df)
    weekly_raw = 0.0
    dm_weekly = _empty_dm_state()
    if len(weekly_df) >= 10:
        weekly_raw, dm_weekly, _, _ = _score_single_timeframe(weekly_df)
    else:
        # Not enough weekly data — use daily as proxy (dampened)
        weekly_raw = round(daily_raw * 0.5, 2)

    # ─── Multi-timeframe alignment ───
    aligned_daily = _apply_mtf_alignment(daily_raw, weekly_raw)

    # Final scores
    daily_score = round(aligned_daily, 1)
    weekly_score = round(weekly_raw, 1)

    daily_signal = _score_to_signal(daily_score)
    weekly_signal = _score_to_signal(weekly_score)

    # ─── DeMark labels (for dashboard display) ───
    dm = dm_daily  # Use daily DeMark state for labels

    # Setup label
    setup_label = None
    setup_label_color = None
    if dm['active_setup_count'] > 0 and dm['active_setup_count'] <= 9:
        if dm['active_setup_side'] == 'buy':
            setup_label = f"Bearish {dm['active_setup_count']}"
            setup_label_color = "red"
        elif dm['active_setup_side'] == 'sell':
            setup_label = f"Bullish {dm['active_setup_count']}"
            setup_label_color = "green"

    # Countdown labels (stale after 12 bars = dash)
    seq_cd_label = None
    seq_cd_label_color = None
    if dm['active_seq_num'] > 0 and dm['active_seq_num'] <= 13 and dm['active_seq_bars_ago'] <= 12:
        if dm['active_seq_side'] == 'buy':
            seq_cd_label = f"Bullish {dm['active_seq_num']}"
            seq_cd_label_color = "green"
        elif dm['active_seq_side'] == 'sell':
            seq_cd_label = f"Bearish {dm['active_seq_num']}"
            seq_cd_label_color = "red"

    combo_cd_label = None
    combo_cd_label_color = None
    if dm['active_combo_num'] > 0 and dm['active_combo_num'] <= 13 and dm['active_combo_bars_ago'] <= 12:
        if dm['active_combo_side'] == 'buy':
            combo_cd_label = f"Bullish {dm['active_combo_num']}"
            combo_cd_label_color = "green"
        elif dm['active_combo_side'] == 'sell':
            combo_cd_label = f"Bearish {dm['active_combo_num']}"
            combo_cd_label_color = "red"

    # ─── DeMark action signal ───
    demark_signal = _compute_demark_signal(df, dm_daily)

    # ─── Traditional indicator values (for display) ───
    rsi = float(last.get('rsi14', 50))
    stoch_k = float(last.get('stoch_k', 50))
    stoch_d = float(last.get('stoch_d', 50))
    macd_line = float(last.get('macd_line', 0))
    macd_signal_val = float(last.get('macd_signal', 0))
    macd_hist = float(last.get('macd_hist', 0))
    bb_lower = float(last.get('bb_lower', close * 0.98))
    bb_upper = float(last.get('bb_upper', close * 1.02))
    bb_mid = float(last.get('bb_mid', close))
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range if bb_range > 0 else 0.5
    rel_spy = float(last.get('rel_to_spy', 0))

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

    return {
        "symbol": symbol,
        "lastClose": round(close, 2),
        "pctChg1d": round(pct_chg_1d, 2),
        "pctChg5d": round(pct_chg_5d, 2),
        # ─── New scoring ───
        "dailyScore": daily_score,
        "dailySignal": daily_signal,
        "weeklyScore": weekly_score,
        "weeklySignal": weekly_signal,
        "dailyRaw": round(daily_raw, 2),       # Pre-alignment daily score
        "weeklyRaw": round(weekly_raw, 2),
        # ─── Score breakdown (for transparency) ───
        "dmBreakdown": dm_bd,
        "confBreakdown": conf_bd,
        # ─── DeMark labels ───
        "td9Daily": None,   # Deprecated — use setupLabel
        "td13Seq": None,    # Deprecated — use seqCdLabel
        "td13Combo": None,  # Deprecated — use comboCdLabel
        "setupLabel": setup_label,
        "setupLabelColor": setup_label_color,
        "seqCdLabel": seq_cd_label,
        "seqCdLabelColor": seq_cd_label_color,
        "comboCdLabel": combo_cd_label,
        "comboCdLabelColor": combo_cd_label_color,
        "demarkSignal": demark_signal,
        "tdSetupCount": dm['active_setup_count'],
        "tdSetupSide": dm['active_setup_side'],
        "seqCdNum": dm['active_seq_num'],
        "seqCdSide": dm['active_seq_side'],
        "comboCdNum": dm['active_combo_num'],
        "comboCdSide": dm['active_combo_side'],
        # ─── Traditional indicators ───
        "rsi14": round(rsi, 1),
        "stochK": round(stoch_k, 1),
        "stochD": round(stoch_d, 1),
        "macdHist": round(macd_hist, 4),
        "macdLine": round(macd_line, 4),
        "macdSignal": round(macd_signal_val, 4),
        "bbPct": round(bb_pct, 2),
        "bbUpper": round(bb_upper, 2),
        "bbMid": round(bb_mid, 2),
        "bbLower": round(bb_lower, 2),
        "relSpy20d": round(rel_spy, 2),
    }


# ═══════════════════════════════════════
# Market Regime
# ═══════════════════════════════════════

def compute_market_regime(signals: dict) -> dict:
    """Compute market regime from index scores."""
    index_scores = []
    for ticker in TICKER_GROUPS["INDICES"]:
        if ticker in signals and "error" not in signals[ticker]:
            index_scores.append(signals[ticker]["dailyScore"])

    if not index_scores:
        return {
            "regime": "NEUTRAL", "avgScore": 0,
            "volatilityElevated": False,
            "justification": "No data available",
        }

    avg_score = sum(index_scores) / len(index_scores)
    buy_count = sum(1 for s in index_scores if s >= 4)
    sell_count = sum(1 for s in index_scores if s <= -4)
    total = len(index_scores)

    if avg_score >= 3:
        regime = "RISK ON"
        justification = f"{buy_count} of {total} indices showing bullish DeMark signals"
    elif avg_score <= -3:
        regime = "RISK OFF"
        justification = f"{sell_count} of {total} indices showing bearish signals"
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


# ═══════════════════════════════════════
# Empty signal template
# ═══════════════════════════════════════

def _empty_signal(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "lastClose": 0, "pctChg1d": 0, "pctChg5d": 0,
        "dailyScore": 0, "dailySignal": "NEUTRAL",
        "weeklyScore": 0, "weeklySignal": "NEUTRAL",
        "dailyRaw": 0, "weeklyRaw": 0,
        "dmBreakdown": {}, "confBreakdown": {},
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
