"""
Indicator computations: TD Sequential, TD Combo, RSI, MACD, Slow Stochastic, Bollinger Bands.
All pure Python/pandas — no TA-Lib dependency.

TD Sequential: Setup (1-9) → Countdown (1-13) using close vs low/high 2 bars back (single condition)
TD Combo: Setup (shared) → Countdown (1-13) using 4 simultaneous conditions (Version II, relaxed 11-13)
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════
# TD Setup (shared by Sequential and Combo)
# ═══════════════════════════════════════

def compute_td_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TD Setup phase with Price Flip tracking.
    
    Buy setup: 9 consecutive closes < close 4 bars earlier (after bearish price flip)
    Sell setup: 9 consecutive closes > close 4 bars earlier (after bullish price flip)
    
    Adds columns: td_setup_buy (0-9), td_setup_sell (0-9), td_setup_ext_len (for recycling)
    """
    n = len(df)
    if n < 5:
        df['td_setup_buy'] = 0
        df['td_setup_sell'] = 0
        df['td_setup_count'] = 0
        return df

    close = df['close'].values

    # h/l classification: compare close to close 4 bars back
    hl = np.zeros(n, dtype=int)  # 1=h, -1=l, 0=NA
    for i in range(4, n):
        if close[i] > close[i - 4]:
            hl[i] = 1   # 'h'
        elif close[i] < close[i - 4]:
            hl[i] = -1  # 'l'

    # Setup enumeration with linear scan + price flip tracking
    setup_buy = np.zeros(n, dtype=int)
    setup_sell = np.zeros(n, dtype=int)

    # Track completed setups with their bar positions and extension lengths
    completed_setups = []  # list of dicts with side, bar1_pos, bar9_pos, ext_len, bars[]

    current_side = 0   # +1 = sell counting, -1 = buy counting
    current_count = 0
    current_start = 0
    current_bars = []

    for i in range(4, n):
        if hl[i] == 0:
            # No signal — don't break streak, just skip
            continue

        # Detect price flip
        if current_side == -1 and hl[i] == 1:
            # Flip from buy to sell
            # If current buy streak was < 9, cancel it (remove partial setup marks)
            if current_count < 9:
                for b in current_bars:
                    setup_buy[b] = 0
            current_side = 1
            current_count = 1
            current_start = i
            current_bars = [i]
            setup_sell[i] = 1
            continue

        elif current_side == 1 and hl[i] == -1:
            # Flip from sell to buy
            if current_count < 9:
                for b in current_bars:
                    setup_sell[b] = 0
            current_side = -1
            current_count = 1
            current_start = i
            current_bars = [i]
            setup_buy[i] = 1
            continue

        # Continue same direction
        if hl[i] == -1:
            if current_side == -1:
                current_count += 1
                current_bars.append(i)
                if current_count <= 9:
                    setup_buy[i] = current_count
                if current_count == 9:
                    completed_setups.append({
                        'side': 'buy',
                        'bar1_pos': current_start,
                        'bar9_pos': i,
                        'bars': list(current_bars[:9]),
                        'ext_len': 9,
                    })
                elif current_count > 9 and completed_setups and completed_setups[-1]['bar1_pos'] == current_start:
                    completed_setups[-1]['ext_len'] = current_count
            else:
                # First buy bar
                current_side = -1
                current_count = 1
                current_start = i
                current_bars = [i]
                setup_buy[i] = 1

        elif hl[i] == 1:
            if current_side == 1:
                current_count += 1
                current_bars.append(i)
                if current_count <= 9:
                    setup_sell[i] = current_count
                if current_count == 9:
                    completed_setups.append({
                        'side': 'sell',
                        'bar1_pos': current_start,
                        'bar9_pos': i,
                        'bars': list(current_bars[:9]),
                        'ext_len': 9,
                    })
                elif current_count > 9 and completed_setups and completed_setups[-1]['bar1_pos'] == current_start:
                    completed_setups[-1]['ext_len'] = current_count
            else:
                current_side = 1
                current_count = 1
                current_start = i
                current_bars = [i]
                setup_sell[i] = 1

    df['td_setup_buy'] = setup_buy
    df['td_setup_sell'] = setup_sell

    # Active setup count for the last bar
    td_setup_count = np.zeros(n, dtype=int)
    for i in range(n):
        if setup_buy[i] > 0:
            td_setup_count[i] = -setup_buy[i]  # Negative = buy side
        elif setup_sell[i] > 0:
            td_setup_count[i] = setup_sell[i]   # Positive = sell side
    df['td_setup_count'] = td_setup_count

    # Store completed setups for countdown use
    df.attrs['completed_setups'] = completed_setups
    return df


# ═══════════════════════════════════════
# TD Sequential Countdown
# ═══════════════════════════════════════

def compute_td_sequential_countdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    TD Sequential Countdown after completed setups.
    Buy countdown: close <= low 2 bars earlier (single condition)
    Sell countdown: close >= high 2 bars earlier (single condition)
    Bars need NOT be consecutive. Requires 13 qualifying bars.
    """
    n = len(df)
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values

    seq_cd_buy = np.zeros(n, dtype=int)
    seq_cd_sell = np.zeros(n, dtype=int)

    buy_cd_active = False
    sell_cd_active = False
    buy_cd_count = 0
    sell_cd_count = 0

    setup_buy = df['td_setup_buy'].values
    setup_sell = df['td_setup_sell'].values

    for i in range(4, n):
        # Completed buy setup → start buy countdown
        if setup_buy[i] == 9:
            buy_cd_active = True
            buy_cd_count = 0

        # Completed sell setup → start sell countdown
        if setup_sell[i] == 9:
            sell_cd_active = True
            sell_cd_count = 0

        # Buy countdown: close <= low 2 bars earlier
        if buy_cd_active and i >= 2 and buy_cd_count < 13:
            if close[i] <= low[i - 2]:
                buy_cd_count += 1
                seq_cd_buy[i] = buy_cd_count
                if buy_cd_count == 13:
                    buy_cd_active = False

        # Sell countdown: close >= high 2 bars earlier
        if sell_cd_active and i >= 2 and sell_cd_count < 13:
            if close[i] >= high[i - 2]:
                sell_cd_count += 1
                seq_cd_sell[i] = sell_cd_count
                if sell_cd_count == 13:
                    sell_cd_active = False

        # Cancellation: opposite setup completes
        if setup_sell[i] == 9 and buy_cd_active:
            buy_cd_active = False
            buy_cd_count = 0
        if setup_buy[i] == 9 and sell_cd_active:
            sell_cd_active = False
            sell_cd_count = 0

    df['seq_cd_buy'] = seq_cd_buy
    df['seq_cd_sell'] = seq_cd_sell

    # Current sequential countdown count (signed: negative=buy, positive=sell)
    seq_cd_count = np.zeros(n, dtype=int)
    for i in range(n):
        if seq_cd_buy[i] > 0:
            seq_cd_count[i] = -seq_cd_buy[i]
        elif seq_cd_sell[i] > 0:
            seq_cd_count[i] = seq_cd_sell[i]
    df['seq_countdown_count'] = seq_cd_count

    return df


# ═══════════════════════════════════════
# TD Combo Countdown (Version II)
# ═══════════════════════════════════════

def compute_td_combo_countdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    TD Combo Countdown Version II (retrospective from bar 1 of setup).
    
    Bars 1-10 (Buy): ALL 4 conditions must be met:
      1. close <= low 2 bars earlier
      2. low <= low of prior bar
      3. close < previous Combo countdown close
      4. close < close of prior bar
    
    Bars 11-13: Only requires successively lower closes (condition #3 only)
    Bar 13 (Termination): Close OR Open < previous Combo countdown close
    
    Sell is the mirror.
    """
    n = len(df)
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    opn = df['open'].values

    combo_cd_buy = np.zeros(n, dtype=int)
    combo_cd_sell = np.zeros(n, dtype=int)

    setup_buy = df['td_setup_buy'].values
    setup_sell = df['td_setup_sell'].values

    completed_setups = df.attrs.get('completed_setups', [])

    for setup in completed_setups:
        side = setup['side']
        bar1_pos = setup['bar1_pos']

        cdn_num = 1
        last_cdn_pos = None
        last_cdn_close = None

        # Combo countdown starts retrospectively from bar 1 of setup
        for i in range(bar1_pos, n):
            if cdn_num > 13:
                break
            if i < 2:
                continue

            c = close[i]
            l = low[i]
            h = high[i]
            o = opn[i]
            c_prev = close[i - 1]
            l_prev = low[i - 1]
            h_prev = high[i - 1]
            l_2back = low[i - 2]
            h_2back = high[i - 2]

            if side == 'buy':
                if cdn_num <= 10:
                    # All 4 conditions
                    cond1 = c <= l_2back
                    cond2 = l <= l_prev
                    cond4 = c < c_prev

                    if not (cond1 and cond2 and cond4):
                        continue

                    # Condition 3: close < previous combo countdown close
                    if last_cdn_close is not None and cdn_num > 1:
                        if c >= last_cdn_close:
                            continue

                    combo_cd_buy[i] = cdn_num
                    last_cdn_pos = i
                    last_cdn_close = c
                    cdn_num += 1

                elif cdn_num < 13:
                    # Bars 11-12: successively lower closes only
                    if last_cdn_close is not None and c < last_cdn_close:
                        combo_cd_buy[i] = cdn_num
                        last_cdn_pos = i
                        last_cdn_close = c
                        cdn_num += 1

                elif cdn_num == 13:
                    # Bar 13 (termination): close OR open < previous combo countdown close
                    if last_cdn_close is not None and (c < last_cdn_close or o < last_cdn_close):
                        combo_cd_buy[i] = cdn_num
                        last_cdn_pos = i
                        last_cdn_close = c
                        cdn_num += 1

            else:  # sell
                if cdn_num <= 10:
                    cond1 = c >= h_2back
                    cond2 = h >= h_prev
                    cond4 = c > c_prev

                    if not (cond1 and cond2 and cond4):
                        continue

                    if last_cdn_close is not None and cdn_num > 1:
                        if c <= last_cdn_close:
                            continue

                    combo_cd_sell[i] = cdn_num
                    last_cdn_pos = i
                    last_cdn_close = c
                    cdn_num += 1

                elif cdn_num < 13:
                    if last_cdn_close is not None and c > last_cdn_close:
                        combo_cd_sell[i] = cdn_num
                        last_cdn_pos = i
                        last_cdn_close = c
                        cdn_num += 1

                elif cdn_num == 13:
                    if last_cdn_close is not None and (c > last_cdn_close or o > last_cdn_close):
                        combo_cd_sell[i] = cdn_num
                        last_cdn_pos = i
                        last_cdn_close = c
                        cdn_num += 1

            # Cancellation: opposite setup 9 kills active countdown
            if side == 'buy' and setup_sell[i] == 9:
                break
            if side == 'sell' and setup_buy[i] == 9:
                break

    df['combo_cd_buy'] = combo_cd_buy
    df['combo_cd_sell'] = combo_cd_sell

    # Current combo countdown count (signed)
    combo_cd_count = np.zeros(n, dtype=int)
    for i in range(n):
        if combo_cd_buy[i] > 0:
            combo_cd_count[i] = -combo_cd_buy[i]
        elif combo_cd_sell[i] > 0:
            combo_cd_count[i] = combo_cd_sell[i]
    df['combo_countdown_count'] = combo_cd_count

    return df


# ═══════════════════════════════════════
# Perfection check
# ═══════════════════════════════════════

def compute_perfection(df: pd.DataFrame) -> pd.DataFrame:
    """Check setup perfection on completed 9s."""
    n = len(df)
    df['td_perfected'] = 0
    low = df['low'].values
    high = df['high'].values
    setup_buy = df['td_setup_buy'].values
    setup_sell = df['td_setup_sell'].values

    for i in range(8, n):
        if setup_buy[i] == 9:
            # Buy perfected: low of bar 8 OR 9 <= low of BOTH bars 6 AND 7
            low6 = low[i - 3]
            low7 = low[i - 2]
            low8 = low[i - 1]
            low9 = low[i]
            if (low8 <= low6 and low8 <= low7) or (low9 <= low6 and low9 <= low7):
                df.iloc[i, df.columns.get_loc('td_perfected')] = 1

        elif setup_sell[i] == 9:
            high6 = high[i - 3]
            high7 = high[i - 2]
            high8 = high[i - 1]
            high9 = high[i]
            if (high8 >= high6 and high8 >= high7) or (high9 >= high6 and high9 >= high7):
                df.iloc[i, df.columns.get_loc('td_perfected')] = 1

    return df


# ═══════════════════════════════════════
# TDST and Risk Levels
# ═══════════════════════════════════════

def compute_tdst_and_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TDST support/resistance levels and Risk Levels from completed
    setups and countdowns.  Must run AFTER setups AND countdowns.

    TDST lines:
      - Buy Setup TDST Resistance = highest True High across bars 1-9
      - Sell Setup TDST Support  = lowest True Low across bars 1-9
      - Starts retroactively at the date of the extreme True High/Low
      - Persists until replaced by next same-direction setup or countdown-13

    Risk Levels (only for Countdown 13):
      - Buy CD 13:  find bar with lowest True Low in countdown window →
                    Risk Level = True Low − True Range of that bar
      - Sell CD 13: find bar with highest True High in countdown window →
                    Risk Level = True High + True Range of that bar
    """
    n = len(df)
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values

    # ── True High / True Low / True Range ──
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    true_high = np.maximum(high, prev_close)
    true_low  = np.minimum(low,  prev_close)
    true_range = true_high - true_low

    # ── Per-bar level arrays (0 = not active) ──
    tdst_buy_level  = np.zeros(n)
    tdst_sell_level = np.zeros(n)
    risk_seq_buy_level   = np.zeros(n)
    risk_seq_sell_level  = np.zeros(n)
    risk_combo_buy_level  = np.zeros(n)
    risk_combo_sell_level = np.zeros(n)

    completed_setups = df.attrs.get('completed_setups', [])

    # ── TDST levels ──
    # For each completed setup, find the extreme true high/low across bars 1-9
    # and record the start position (date of that extreme) and level.
    # A new same-direction setup replaces the prior TDST.
    buy_tdst_segments  = []  # (start_pos, level)
    sell_tdst_segments = []  # (start_pos, level)

    for setup in completed_setups:
        bars = setup['bars']  # positions of bars 1-9
        if len(bars) < 9:
            continue

        if setup['side'] == 'buy':
            # TDST Resistance = highest True High across bars 1-9
            best_th = -np.inf
            best_pos = bars[0]
            for b in bars:
                if b < n and true_high[b] > best_th:
                    best_th = true_high[b]
                    best_pos = b
            buy_tdst_segments.append((best_pos, best_th))

        else:  # sell
            # TDST Support = lowest True Low across bars 1-9
            best_tl = np.inf
            best_pos = bars[0]
            for b in bars:
                if b < n and true_low[b] < best_tl:
                    best_tl = true_low[b]
                    best_pos = b
            sell_tdst_segments.append((best_pos, best_tl))

    # Forward-fill TDST: each segment starts at start_pos and persists
    # until the next same-direction segment starts (staircase).
    def _forward_fill_segments(segments, arr, n):
        if not segments:
            return
        # Sort by start position
        segments.sort(key=lambda s: s[0])
        for idx, (start, level) in enumerate(segments):
            end = segments[idx + 1][0] if idx + 1 < len(segments) else n
            for j in range(start, end):
                arr[j] = level

    _forward_fill_segments(buy_tdst_segments, tdst_buy_level, n)
    _forward_fill_segments(sell_tdst_segments, tdst_sell_level, n)

    # ── Risk Levels (Countdown 13 only) ──
    # We need to scan the countdown arrays to find completed 13s and their
    # windows (from the triggering setup bar1 to bar13 position).

    seq_cd_buy  = df['seq_cd_buy'].values.astype(int)
    seq_cd_sell = df['seq_cd_sell'].values.astype(int)
    combo_cd_buy  = df['combo_cd_buy'].values.astype(int)
    combo_cd_sell = df['combo_cd_sell'].values.astype(int)

    def _find_countdown_windows(cd_arr, side_label):
        """Return list of (window_start, bar13_pos) for completed countdowns."""
        windows = []
        # Walk through and find 13-completions, then backtrack to find
        # the most recent setup-9 (or the 1 of the countdown) as window start.
        # The countdown window runs from the first qualifying bar (cd==1) to cd==13.
        i = 0
        while i < n:
            if cd_arr[i] == 13:
                bar13 = i
                # Walk backward to find the bar with cd == 1
                bar1 = i
                for j in range(i - 1, -1, -1):
                    if cd_arr[j] == 1:
                        bar1 = j
                        break
                windows.append((bar1, bar13))
            i += 1
        return windows

    def _compute_risk_segments(cd_arr, side, arr_out):
        windows = _find_countdown_windows(cd_arr, side)
        segments = []
        for (w_start, w_end) in windows:
            if side == 'buy':
                # Find bar with lowest True Low in the window
                best_pos = w_start
                best_tl = true_low[w_start]
                for j in range(w_start, min(w_end + 1, n)):
                    if true_low[j] < best_tl:
                        best_tl = true_low[j]
                        best_pos = j
                risk = best_tl - true_range[best_pos]
                segments.append((w_end, risk))  # starts at the 13 bar
            else:  # sell
                best_pos = w_start
                best_th = true_high[w_start]
                for j in range(w_start, min(w_end + 1, n)):
                    if true_high[j] > best_th:
                        best_th = true_high[j]
                        best_pos = j
                risk = best_th + true_range[best_pos]
                segments.append((w_end, risk))
        _forward_fill_segments(segments, arr_out, n)

    _compute_risk_segments(seq_cd_buy,   'buy',  risk_seq_buy_level)
    _compute_risk_segments(seq_cd_sell,  'sell', risk_seq_sell_level)
    _compute_risk_segments(combo_cd_buy,  'buy',  risk_combo_buy_level)
    _compute_risk_segments(combo_cd_sell, 'sell', risk_combo_sell_level)

    # ── Store columns ──
    df['tdst_buy_level']  = tdst_buy_level
    df['tdst_sell_level'] = tdst_sell_level
    df['risk_seq_buy_level']   = risk_seq_buy_level
    df['risk_seq_sell_level']  = risk_seq_sell_level
    df['risk_combo_buy_level']  = risk_combo_buy_level
    df['risk_combo_sell_level'] = risk_combo_sell_level

    # ── Store last active levels in attrs for scoring.py ──
    last_tdst_buy  = float(tdst_buy_level[-1])  if n > 0 and tdst_buy_level[-1]  != 0 else None
    last_tdst_sell = float(tdst_sell_level[-1]) if n > 0 and tdst_sell_level[-1] != 0 else None
    df.attrs['last_tdst_buy_level']  = last_tdst_buy
    df.attrs['last_tdst_sell_level'] = last_tdst_sell

    return df


# ═══════════════════════════════════════
# RSI (14-period)
# ═══════════════════════════════════════

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))
    df['rsi14'] = df['rsi14'].fillna(50)
    return df


# ═══════════════════════════════════════
# Slow Stochastic (%K=14, %D=3, slowing=3)
# ═══════════════════════════════════════

def compute_slow_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    fast_k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    df['stoch_k'] = fast_k.rolling(window=slowing).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    df['stoch_k'] = df['stoch_k'].fillna(50)
    df['stoch_d'] = df['stoch_d'].fillna(50)
    return df


# ═══════════════════════════════════════
# MACD (fast=12, slow=26, signal=9)
# ═══════════════════════════════════════

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    return df


# ═══════════════════════════════════════
# Bollinger Bands (20-period SMA ± 2 std)
# ═══════════════════════════════════════

def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df['bb_mid'] = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_mid'] + num_std * std
    df['bb_lower'] = df['bb_mid'] - num_std * std
    df['bb_mid'] = df['bb_mid'].fillna(df['close'])
    df['bb_upper'] = df['bb_upper'].fillna(df['close'] * 1.02)
    df['bb_lower'] = df['bb_lower'].fillna(df['close'] * 0.98)
    return df


# ═══════════════════════════════════════
# Relative performance vs SPY
# ═══════════════════════════════════════

def compute_relative_to_spy(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    if len(spy_close) == 0 or len(df) == 0:
        df['rel_to_spy'] = 0.0
        return df
    try:
        ticker_ret_20 = df['close'].pct_change(20)
        spy_aligned = spy_close.reindex(df.index)
        spy_ret_20 = spy_aligned.pct_change(20)
        df['rel_to_spy'] = (ticker_ret_20 - spy_ret_20) * 100
    except Exception:
        df['rel_to_spy'] = 0.0
    df['rel_to_spy'] = df['rel_to_spy'].fillna(0)
    return df


# ═══════════════════════════════════════
# Master computation
# ═══════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame, spy_close: pd.Series = None) -> pd.DataFrame:
    """Run all indicator computations on a DataFrame with OHLC columns."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # DeMark: Setup → Sequential Countdown → Combo Countdown → Perfection → TDST/Risk
    df = compute_td_setups(df)
    df = compute_td_sequential_countdown(df)
    df = compute_td_combo_countdown(df)
    df = compute_perfection(df)
    df = compute_tdst_and_risk_levels(df)

    # Traditional indicators
    df = compute_rsi(df)
    df = compute_slow_stochastic(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)

    if spy_close is not None:
        df = compute_relative_to_spy(df, spy_close)
    else:
        df['rel_to_spy'] = 0.0

    return df
