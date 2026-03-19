"""
Indicator computations: TD Sequential, TD Combo, RSI, MACD, Slow Stochastic, Bollinger Bands.
All pure Python/pandas — no TA-Lib dependency.

TD Sequential: Setup (1-9) → Countdown (1-13) using close vs low/high 2 bars back (single condition)
TD Combo: Setup (shared) → Countdown (1-13) using 4 simultaneous conditions (Version II, relaxed 11-13)

DeMark rules implemented per the DeMark Technician skill:
  - Setup cancellation: Price Flip cancels in-progress setup; cancelled counts removed
  - Countdown cancellation: Opposite Setup 9 completion OR TDST violation
  - TDST lifecycle: dashed → qualified (3-bar) → confirmed → solid entire line → stop printing
  - Risk levels: Only for Countdown 13, same breakout lifecycle, stay in lane
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
    
    Setup cancellation: Price Flip cancels in-progress partial setups (counts removed).
    Setup extends beyond 9 until Price Flip (important for recycling).
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
# TDST computation (needed before countdowns for cancellation)
# ═══════════════════════════════════════

def _compute_true_values(df: pd.DataFrame):
    """Compute true high, true low, true range arrays."""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    true_high = np.maximum(high, prev_close)
    true_low = np.minimum(low, prev_close)
    true_range = true_high - true_low

    return true_high, true_low, true_range


def _compute_raw_tdst(df: pd.DataFrame):
    """
    Compute raw TDST levels from completed setups (before lifecycle processing).
    Returns buy_tdst_segments and sell_tdst_segments.
    Each segment: (start_pos, level, setup_bar9_pos)
    """
    n = len(df)
    true_high, true_low, _ = _compute_true_values(df)
    completed_setups = df.attrs.get('completed_setups', [])

    buy_tdst_segments = []
    sell_tdst_segments = []

    for setup in completed_setups:
        bars = setup['bars']
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
            buy_tdst_segments.append((best_pos, best_th, setup['bar9_pos']))
        else:  # sell
            # TDST Support = lowest True Low across bars 1-9
            best_tl = np.inf
            best_pos = bars[0]
            for b in bars:
                if b < n and true_low[b] < best_tl:
                    best_tl = true_low[b]
                    best_pos = b
            sell_tdst_segments.append((best_pos, best_tl, setup['bar9_pos']))

    return buy_tdst_segments, sell_tdst_segments


def _get_tdst_level_at_bar(segments, bar_idx):
    """
    Get the active TDST level at a given bar index.
    segments: list of (start_pos, level, setup_bar9_pos) sorted by start_pos
    Returns the level value, or 0 if no active TDST.
    """
    active_level = 0.0
    for (start, level, _bar9) in segments:
        if start <= bar_idx:
            active_level = level
        else:
            break
    return active_level


# ═══════════════════════════════════════
# TD Sequential Countdown (with proper cancellation)
# ═══════════════════════════════════════

def compute_td_sequential_countdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    TD Sequential Countdown after completed setups.
    Buy countdown: close <= low 2 bars earlier (single condition)
    Sell countdown: close >= high 2 bars earlier (single condition)
    Bars need NOT be consecutive. Requires 13 qualifying bars.

    Cancellation rules (per DeMark skill):
      1. Opposite Setup 9 completion cancels in-progress countdown
      2. TDST violation: Buy CD cancelled if true_low > TDST resistance (buy TDST)
                         Sell CD cancelled if true_high < TDST support (sell TDST)
    """
    n = len(df)
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values

    seq_cd_buy = np.zeros(n, dtype=int)
    seq_cd_sell = np.zeros(n, dtype=int)

    setup_buy = df['td_setup_buy'].values
    setup_sell = df['td_setup_sell'].values

    # Get TDST levels for cancellation checks
    true_high, true_low, _ = _compute_true_values(df)
    buy_tdst_segs, sell_tdst_segs = _compute_raw_tdst(df)
    buy_tdst_segs.sort(key=lambda s: s[0])
    sell_tdst_segs.sort(key=lambda s: s[0])

    buy_cd_active = False
    sell_cd_active = False
    buy_cd_count = 0
    sell_cd_count = 0
    # Track the TDST level that was active when the countdown's setup completed
    buy_cd_tdst_ref = 0.0   # Buy TDST resistance for buy countdown cancellation
    sell_cd_tdst_ref = 0.0  # Sell TDST support for sell countdown cancellation

    for i in range(4, n):
        # Completed buy setup → start buy countdown
        if setup_buy[i] == 9:
            buy_cd_active = True
            buy_cd_count = 0
            # Reference TDST for cancellation: the buy TDST resistance active at this point
            buy_cd_tdst_ref = _get_tdst_level_at_bar(buy_tdst_segs, i)

        # Completed sell setup → start sell countdown
        if setup_sell[i] == 9:
            sell_cd_active = True
            sell_cd_count = 0
            # Reference TDST: the sell TDST support active at this point
            sell_cd_tdst_ref = _get_tdst_level_at_bar(sell_tdst_segs, i)

        # --- Cancellation checks ---
        # Cancel buy countdown if opposite (sell) setup 9 completes
        if buy_cd_active and setup_sell[i] == 9:
            buy_cd_active = False
            buy_cd_count = 0

        # Cancel sell countdown if opposite (buy) setup 9 completes
        if sell_cd_active and setup_buy[i] == 9:
            sell_cd_active = False
            sell_cd_count = 0

        # TDST violation cancellation:
        # Buy CD: if true_low > TDST resistance → prior downtrend concluded → cancel
        if buy_cd_active and buy_cd_tdst_ref > 0:
            if true_low[i] > buy_cd_tdst_ref:
                buy_cd_active = False
                buy_cd_count = 0

        # Sell CD: if true_high < TDST support → prior uptrend concluded → cancel
        if sell_cd_active and sell_cd_tdst_ref > 0:
            if true_high[i] < sell_cd_tdst_ref:
                sell_cd_active = False
                sell_cd_count = 0

        # --- Countdown qualification ---
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
# TD Combo Countdown (Version II, with proper cancellation)
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

    Cancellation:
      1. Opposite Setup 9 completion
      2. TDST violation (same as Sequential)
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

    # Get TDST levels for cancellation checks
    true_high, true_low, _ = _compute_true_values(df)
    buy_tdst_segs, sell_tdst_segs = _compute_raw_tdst(df)
    buy_tdst_segs.sort(key=lambda s: s[0])
    sell_tdst_segs.sort(key=lambda s: s[0])

    for setup in completed_setups:
        side = setup['side']
        bar1_pos = setup['bar1_pos']
        bar9_pos = setup['bar9_pos']

        cdn_num = 1
        last_cdn_pos = None
        last_cdn_close = None

        # Get reference TDST level at the time of setup completion
        if side == 'buy':
            tdst_ref = _get_tdst_level_at_bar(buy_tdst_segs, bar9_pos)
        else:
            tdst_ref = _get_tdst_level_at_bar(sell_tdst_segs, bar9_pos)

        cancelled = False

        # Combo countdown starts retrospectively from bar 1 of setup
        for i in range(bar1_pos, n):
            if cdn_num > 13:
                break
            if i < 2:
                continue
            if cancelled:
                break

            # --- Cancellation checks ---
            # Opposite setup 9 kills this countdown
            if side == 'buy' and setup_sell[i] == 9:
                cancelled = True
                break
            if side == 'sell' and setup_buy[i] == 9:
                cancelled = True
                break

            # TDST violation cancellation
            if side == 'buy' and tdst_ref > 0:
                if true_low[i] > tdst_ref:
                    cancelled = True
                    break
            if side == 'sell' and tdst_ref > 0:
                if true_high[i] < tdst_ref:
                    cancelled = True
                    break

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
    """
    Check setup perfection on completed 9s.
    Buy perfected: low of bar 8 OR 9 <= low of BOTH bars 6 AND 7
    Sell perfected: high of bar 8 OR 9 >= high of BOTH bars 6 AND 7

    Uses actual bar positions from the completed setup records.
    """
    n = len(df)
    df['td_perfected'] = 0
    low = df['low'].values
    high = df['high'].values
    completed_setups = df.attrs.get('completed_setups', [])

    for setup in completed_setups:
        bars = setup['bars']
        if len(bars) < 9:
            continue

        bar9_pos = setup['bar9_pos']
        # bars[5]=bar6, bars[6]=bar7, bars[7]=bar8, bars[8]=bar9
        b6 = bars[5]
        b7 = bars[6]
        b8 = bars[7]
        b9 = bars[8]

        if setup['side'] == 'buy':
            low6 = low[b6]
            low7 = low[b7]
            low8 = low[b8]
            low9 = low[b9]
            if (low8 <= low6 and low8 <= low7) or (low9 <= low6 and low9 <= low7):
                df.iloc[bar9_pos, df.columns.get_loc('td_perfected')] = 1
        else:
            high6 = high[b6]
            high7 = high[b7]
            high8 = high[b8]
            high9 = high[b9]
            if (high8 >= high6 and high8 >= high7) or (high9 >= high6 and high9 >= high7):
                df.iloc[bar9_pos, df.columns.get_loc('td_perfected')] = 1

    return df


# ═══════════════════════════════════════
# Breakout qualification/confirmation (3-bar sequence)
# Used by TDST lines and Risk Levels
# ═══════════════════════════════════════

def _check_breakout_sequence(close, high, low, opn, true_high, true_low, level, direction, start_pos, end_pos):
    """
    Check for a qualified AND confirmed 3-bar breakout sequence.

    For UPSIDE breakout (direction='up'):
      Bar 1: close < prior close (down close)
      Bar 2: close > prior close AND close > level
      Bar 3: open > bar3's open (gap up or above level), higher high than prior 3 bars,
              close > prior close

    For DOWNSIDE breakout (direction='down'):
      Bar 1: close > prior close (up close)
      Bar 2: close < prior close AND close < level
      Bar 3: open < bar3's open (gap down), lower low than prior 3 bars,
              close < prior close

    Returns: (confirmed, confirmation_bar_idx) or (False, None)
    """
    n = len(close)

    for i in range(max(start_pos, 1), min(end_pos, n - 2)):
        if direction == 'up':
            # Bar 1: down close
            if close[i] >= close[i - 1]:
                continue

            # Bar 2 (i+1): close above prior close AND above the level
            j = i + 1
            if j >= n:
                break
            if not (close[j] > close[j - 1] and close[j] > level):
                continue

            # Bar 3 (i+2): confirmation
            k = i + 2
            if k >= n:
                break
            # Open higher than own open — this is from the DeMark spec literally:
            # "Open higher than bar 3's own open" — interpreted as open > prior close
            # (the gap-up condition)
            cond_open = opn[k] > close[k - 1]
            # Higher high than prior 3 bars
            max_high_prior = max(high[k - 1], high[k - 2], high[k - 3]) if k >= 3 else high[k - 1]
            cond_high = high[k] > max_high_prior
            # Close > prior close
            cond_close = close[k] > close[k - 1]

            if cond_open and cond_high and cond_close:
                return True, k  # Confirmed!

        else:  # direction == 'down'
            # Bar 1: up close
            if close[i] <= close[i - 1]:
                continue

            # Bar 2 (i+1): close below prior close AND below level
            j = i + 1
            if j >= n:
                break
            if not (close[j] < close[j - 1] and close[j] < level):
                continue

            # Bar 3 (i+2): confirmation
            k = i + 2
            if k >= n:
                break
            cond_open = opn[k] < close[k - 1]
            min_low_prior = min(low[k - 1], low[k - 2], low[k - 3]) if k >= 3 else low[k - 1]
            cond_low = low[k] < min_low_prior
            cond_close = close[k] < close[k - 1]

            if cond_open and cond_low and cond_close:
                return True, k

    return False, None


# ═══════════════════════════════════════
# TDST and Risk Levels with full lifecycle
# ═══════════════════════════════════════

def compute_tdst_and_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TDST support/resistance levels and Risk Levels from completed
    setups and countdowns.  Must run AFTER setups AND countdowns.

    TDST lines:
      - Buy Setup TDST Resistance = highest True High across bars 1-9
      - Sell Setup TDST Support  = lowest True Low across bars 1-9
      - Starts retroactively at the date of the extreme True High/Low
      - Lifecycle: dashed by default → qualified & confirmed breakout → solid entire line → stop

    Risk Levels (only for Countdown 13):
      - Buy CD 13:  find bar with lowest True Low in countdown window →
                    Risk Level = True Low − True Range of that bar
      - Sell CD 13: find bar with highest True High in countdown window →
                    Risk Level = True High + True Range of that bar
      - Same breakout lifecycle as TDST
      - Stay in their lane: Combo risk only from Combo signals, Sequential from Sequential
    """
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    opn = df['open'].values

    # ── True High / True Low / True Range ──
    true_high, true_low, true_range = _compute_true_values(df)

    # ── Per-bar level arrays (0 = not active) ──
    tdst_buy_level = np.zeros(n)
    tdst_sell_level = np.zeros(n)
    tdst_buy_status = np.full(n, '', dtype=object)   # 'dashed', 'solid', or ''
    tdst_sell_status = np.full(n, '', dtype=object)
    risk_seq_buy_level = np.zeros(n)
    risk_seq_sell_level = np.zeros(n)
    risk_combo_buy_level = np.zeros(n)
    risk_combo_sell_level = np.zeros(n)
    risk_seq_buy_status = np.full(n, '', dtype=object)
    risk_seq_sell_status = np.full(n, '', dtype=object)
    risk_combo_buy_status = np.full(n, '', dtype=object)
    risk_combo_sell_status = np.full(n, '', dtype=object)

    completed_setups = df.attrs.get('completed_setups', [])

    # ── TDST levels with lifecycle ──
    buy_tdst_raw = []   # (start_pos, level, setup_bar9_pos)
    sell_tdst_raw = []

    for setup in completed_setups:
        bars = setup['bars']
        if len(bars) < 9:
            continue

        if setup['side'] == 'buy':
            best_th = -np.inf
            best_pos = bars[0]
            for b in bars:
                if b < n and true_high[b] > best_th:
                    best_th = true_high[b]
                    best_pos = b
            buy_tdst_raw.append((best_pos, best_th, setup['bar9_pos']))
        else:
            best_tl = np.inf
            best_pos = bars[0]
            for b in bars:
                if b < n and true_low[b] < best_tl:
                    best_tl = true_low[b]
                    best_pos = b
            sell_tdst_raw.append((best_pos, best_tl, setup['bar9_pos']))

    buy_tdst_raw.sort(key=lambda s: s[0])
    sell_tdst_raw.sort(key=lambda s: s[0])

    def _process_tdst_segments(segments, level_arr, status_arr, direction):
        """
        Process TDST segments with lifecycle:
        - Each segment starts at start_pos, persists until next same-direction segment or confirmed breakout.
        - Check for qualified/confirmed breakout (3-bar sequence).
        - If confirmed: entire line turns solid, then stops printing.
        - If not confirmed: line stays dashed.
        """
        for idx, (start, level, _bar9) in enumerate(segments):
            # Determine natural end: when next same-direction TDST starts
            natural_end = segments[idx + 1][0] if idx + 1 < len(segments) else n

            # Check for breakout: for buy TDST (resistance), check upside breakout
            # For sell TDST (support), check downside breakout
            breakout_dir = 'up' if direction == 'buy' else 'down'
            confirmed, confirm_bar = _check_breakout_sequence(
                close, high, low, opn, true_high, true_low,
                level, breakout_dir, start, natural_end
            )

            if confirmed and confirm_bar is not None:
                # Entire line from start to confirmation date turns solid, then stops
                for j in range(start, min(confirm_bar + 1, n)):
                    level_arr[j] = level
                    status_arr[j] = 'solid'
                # Stop printing after confirmation
            else:
                # Line stays dashed until natural end
                for j in range(start, min(natural_end, n)):
                    level_arr[j] = level
                    status_arr[j] = 'dashed'

    _process_tdst_segments(buy_tdst_raw, tdst_buy_level, tdst_buy_status, 'buy')
    _process_tdst_segments(sell_tdst_raw, tdst_sell_level, tdst_sell_status, 'sell')

    # ── Risk Levels (Countdown 13 only, with lifecycle) ──
    seq_cd_buy = df['seq_cd_buy'].values.astype(int)
    seq_cd_sell = df['seq_cd_sell'].values.astype(int)
    combo_cd_buy = df['combo_cd_buy'].values.astype(int)
    combo_cd_sell = df['combo_cd_sell'].values.astype(int)

    def _find_countdown_windows(cd_arr):
        """Return list of (bar1_pos, bar13_pos) for completed countdowns."""
        windows = []
        i = 0
        while i < n:
            if cd_arr[i] == 13:
                bar13 = i
                bar1 = i
                for j in range(i - 1, -1, -1):
                    if cd_arr[j] == 1:
                        bar1 = j
                        break
                windows.append((bar1, bar13))
            i += 1
        return windows

    def _compute_risk_segments(cd_arr, side, level_arr, status_arr, indicator_type):
        """
        Compute risk level segments with lifecycle.
        Risk levels stay in their lane:
          - Combo risk only replaced by next Combo risk event
          - Sequential risk only replaced by next Sequential risk event
        """
        windows = _find_countdown_windows(cd_arr)
        segments = []

        for (w_start, w_end) in windows:
            if side == 'buy':
                best_pos = w_start
                best_tl = true_low[w_start]
                for j in range(w_start, min(w_end + 1, n)):
                    if true_low[j] < best_tl:
                        best_tl = true_low[j]
                        best_pos = j
                risk = best_tl - true_range[best_pos]
                segments.append((w_end, risk))
            else:
                best_pos = w_start
                best_th = true_high[w_start]
                for j in range(w_start, min(w_end + 1, n)):
                    if true_high[j] > best_th:
                        best_th = true_high[j]
                        best_pos = j
                risk = best_th + true_range[best_pos]
                segments.append((w_end, risk))

        segments.sort(key=lambda s: s[0])

        # Process each segment with lifecycle
        for idx, (start, level) in enumerate(segments):
            # Natural end: next same-type risk segment or end of data
            natural_end = segments[idx + 1][0] if idx + 1 < len(segments) else n

            # Check for breakout: buy risk = support → check downside breakout
            # sell risk = resistance → check upside breakout
            breakout_dir = 'down' if side == 'buy' else 'up'
            confirmed, confirm_bar = _check_breakout_sequence(
                close, high, low, opn, true_high, true_low,
                level, breakout_dir, start, natural_end
            )

            if confirmed and confirm_bar is not None:
                for j in range(start, min(confirm_bar + 1, n)):
                    level_arr[j] = level
                    status_arr[j] = 'solid'
            else:
                for j in range(start, min(natural_end, n)):
                    level_arr[j] = level
                    status_arr[j] = 'dashed'

    _compute_risk_segments(seq_cd_buy, 'buy', risk_seq_buy_level, risk_seq_buy_status, 'sequential')
    _compute_risk_segments(seq_cd_sell, 'sell', risk_seq_sell_level, risk_seq_sell_status, 'sequential')
    _compute_risk_segments(combo_cd_buy, 'buy', risk_combo_buy_level, risk_combo_buy_status, 'combo')
    _compute_risk_segments(combo_cd_sell, 'sell', risk_combo_sell_level, risk_combo_sell_status, 'combo')

    # ── Store columns ──
    df['tdst_buy_level'] = tdst_buy_level
    df['tdst_sell_level'] = tdst_sell_level
    df['tdst_buy_status'] = tdst_buy_status
    df['tdst_sell_status'] = tdst_sell_status
    df['risk_seq_buy_level'] = risk_seq_buy_level
    df['risk_seq_sell_level'] = risk_seq_sell_level
    df['risk_combo_buy_level'] = risk_combo_buy_level
    df['risk_combo_sell_level'] = risk_combo_sell_level
    df['risk_seq_buy_status'] = risk_seq_buy_status
    df['risk_seq_sell_status'] = risk_seq_sell_status
    df['risk_combo_buy_status'] = risk_combo_buy_status
    df['risk_combo_sell_status'] = risk_combo_sell_status

    # ── Store last active levels in attrs for scoring.py ──
    last_tdst_buy = float(tdst_buy_level[-1]) if n > 0 and tdst_buy_level[-1] != 0 else None
    last_tdst_sell = float(tdst_sell_level[-1]) if n > 0 and tdst_sell_level[-1] != 0 else None
    df.attrs['last_tdst_buy_level'] = last_tdst_buy
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
