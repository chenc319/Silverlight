"""
Indicator computations: TD Sequential, RSI, MACD, Slow Stochastic, Bollinger Bands.
All pure Python/pandas — no TA-Lib dependency.
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════
# TD Sequential (pure Python/pandas)
# ═══════════════════════════════════════

def compute_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TD Sequential Setup (1-9) and Countdown (1-13) on OHLC data.
    
    Setup: Count consecutive closes higher (sell) or lower (buy) than close 4 bars prior.
    Countdown: After completed 9-bar setup, count bars meeting countdown conditions up to 13.
    
    Returns DataFrame with added columns:
      td_setup_buy, td_setup_sell, td_countdown_buy, td_countdown_sell,
      td_setup_count, td_countdown_count
    """
    n = len(df)
    if n < 5:
        for col in ['td_setup_buy', 'td_setup_sell', 'td_countdown_buy', 'td_countdown_sell',
                     'td_setup_count', 'td_countdown_count']:
            df[col] = 0
        return df

    close = df['close'].values
    low = df['low'].values
    high = df['high'].values

    # Setup arrays
    setup_buy = np.zeros(n, dtype=int)    # 1-9 for buy setup counts
    setup_sell = np.zeros(n, dtype=int)   # 1-9 for sell setup counts
    countdown_buy = np.zeros(n, dtype=int)
    countdown_sell = np.zeros(n, dtype=int)

    # Track active setup
    buy_count = 0
    sell_count = 0

    for i in range(4, n):
        # Compare close to close 4 bars back
        if close[i] < close[i - 4]:
            # Potential buy setup bar
            if buy_count > 0 or sell_count == 0:
                buy_count += 1
                sell_count = 0
            else:
                # Price flip from sell to buy
                sell_count = 0
                buy_count = 1

            if buy_count <= 9:
                setup_buy[i] = buy_count
            if buy_count > 9:
                buy_count = buy_count  # Continue tracking for recycling

        elif close[i] > close[i - 4]:
            # Potential sell setup bar
            if sell_count > 0 or buy_count == 0:
                sell_count += 1
                buy_count = 0
            else:
                # Price flip from buy to sell
                buy_count = 0
                sell_count = 1

            if sell_count <= 9:
                setup_sell[i] = sell_count
            if sell_count > 9:
                sell_count = sell_count

        else:
            # Equal — reset both
            buy_count = 0
            sell_count = 0

    # Countdown phase
    # Find completed setups (count == 9) and run countdown from there
    buy_cd_count = 0
    sell_cd_count = 0
    buy_cd_active = False
    sell_cd_active = False

    for i in range(4, n):
        # Check for completed buy setup → start buy countdown
        if setup_buy[i] == 9:
            buy_cd_active = True
            buy_cd_count = 0

        # Check for completed sell setup → start sell countdown
        if setup_sell[i] == 9:
            sell_cd_active = True
            sell_cd_count = 0

        # Buy countdown: close <= low 2 bars earlier
        if buy_cd_active and i >= 2 and buy_cd_count < 13:
            if close[i] <= low[i - 2]:
                buy_cd_count += 1
                countdown_buy[i] = buy_cd_count
                if buy_cd_count == 13:
                    # Check 13 vs 8 deferral: low of bar 13 must be <= close of bar 8
                    buy_cd_active = False

        # Sell countdown: close >= high 2 bars earlier
        if sell_cd_active and i >= 2 and sell_cd_count < 13:
            if close[i] >= high[i - 2]:
                sell_cd_count += 1
                countdown_sell[i] = sell_cd_count
                if sell_cd_count == 13:
                    sell_cd_active = False

        # Cancellation: opposite setup completes
        if setup_sell[i] == 9 and buy_cd_active:
            buy_cd_active = False
            buy_cd_count = 0
        if setup_buy[i] == 9 and sell_cd_active:
            sell_cd_active = False
            sell_cd_count = 0

    df['td_setup_buy'] = setup_buy
    df['td_setup_sell'] = setup_sell
    df['td_countdown_buy'] = countdown_buy
    df['td_countdown_sell'] = countdown_sell

    # Current active count for scoring
    td_setup_count = np.zeros(n, dtype=int)
    td_countdown_count = np.zeros(n, dtype=int)
    for i in range(n):
        if setup_buy[i] > 0:
            td_setup_count[i] = -setup_buy[i]  # Negative = buy side
        elif setup_sell[i] > 0:
            td_setup_count[i] = setup_sell[i]   # Positive = sell side
        if countdown_buy[i] > 0:
            td_countdown_count[i] = -countdown_buy[i]
        elif countdown_sell[i] > 0:
            td_countdown_count[i] = countdown_sell[i]

    df['td_setup_count'] = td_setup_count
    df['td_countdown_count'] = td_countdown_count

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
    df['stoch_k'] = fast_k.rolling(window=slowing).mean()  # Slow %K
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()  # %D
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
    # Fill NaN with reasonable defaults
    df['bb_mid'] = df['bb_mid'].fillna(df['close'])
    df['bb_upper'] = df['bb_upper'].fillna(df['close'] * 1.02)
    df['bb_lower'] = df['bb_lower'].fillna(df['close'] * 0.98)
    return df


# ═══════════════════════════════════════
# Relative performance vs SPY
# ═══════════════════════════════════════

def compute_relative_to_spy(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Compute 20-day relative performance vs SPY."""
    if len(spy_close) == 0 or len(df) == 0:
        df['rel_to_spy'] = 0.0
        return df

    # Align by date
    try:
        ticker_ret_20 = df['close'].pct_change(20)
        # Match SPY data by index
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

    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]

    df = compute_td_sequential(df)
    df = compute_rsi(df)
    df = compute_slow_stochastic(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)

    if spy_close is not None:
        df = compute_relative_to_spy(df, spy_close)
    else:
        df['rel_to_spy'] = 0.0

    return df
