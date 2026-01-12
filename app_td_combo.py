### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### MAGS DATA ###
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA']
each_mags_df = pd.DataFrame()
for mag_ticker in mags_tickers:
    mag_string = mag_ticker + '.csv'
    with open(Path(DATA_DIR) / mag_string, 'rb') as file:
        mag_df = pd.read_csv(file)
        mag_df.index = pd.to_datetime(mag_df['Date'].values)
        close_df = pd.DataFrame(mag_df['Close'])
        close_df.columns = [mag_ticker]
    each_mags_df = merge_dfs([each_mags_df, close_df])
each_mags_df = each_mags_df.dropna()
mags_monthly_pct = each_mags_df.pct_change()

### MAGS WEIGHTS ###
with open(Path(DATA_DIR) / 'mags_weights.xlsx', 'rb') as file:
    mags_weights_df = pd.read_excel(file,sheet_name='Sheet1')
    mags_weights_df.index = mags_weights_df['Date'].values
    mags_weights_df.drop('Date', axis=1, inplace=True)
    mags_weights_df['sum'] = mags_weights_df.sum(axis=1)
normalized_mags_weights = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA'])
normalized_mags_pct = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA'])
for col in normalized_mags_weights.columns:
    normalized_mags_weights[col] = (mags_weights_df[col] / mags_weights_df['sum']).resample('ME').last()
    col_df = merge_dfs([mags_monthly_pct[col],normalized_mags_weights[col]]).ffill().dropna()
    col_df.columns = ['pct','weights']
    normalized_mags_pct[col] = col_df['pct'] * col_df['weights']

mock_mags_monthly_pct = pd.DataFrame(normalized_mags_pct.sum(axis=1))
mock_mags_monthly_pct.columns = ['mags']

c = pd.DataFrame(each_mags_df['GOOGL'])





# Use close series, not whole DataFrame, in the logic
def td_setup(c: pd.DataFrame) -> pd.DataFrame:
    close = c.iloc[:, 0]  # 'GOOGL' column as Series

    # price flip
    flip_buy  = (close.shift(1) > close.shift(5)) & (close <= close.shift(4))
    flip_sell = (close.shift(1) < close.shift(5)) & (close >= close.shift(4))

    setup_buy = pd.Series(0, index=c.index, dtype=int)
    setup_sell = pd.Series(0, index=c.index, dtype=int)

    for i in range(len(c)):
        # BUY SETUP 1–9
        if flip_buy.iat[i]:
            setup_buy.iat[i] = 1
        elif (
            i >= 4 and
            setup_buy.iat[i-1] > 0 and
            setup_buy.iat[i-1] < 9 and
            close.iat[i] < close.iat[i-4]
        ):
            setup_buy.iat[i] = setup_buy.iat[i-1] + 1
        else:
            # reset when condition fails
            if setup_buy.iat[i-1] < 9 if i > 0 else True:
                setup_buy.iat[i] = 0

        # SELL SETUP 1–9 (mirror)
        if flip_sell.iat[i]:
            setup_sell.iat[i] = 1
        elif (
            i >= 4 and
            setup_sell.iat[i-1] > 0 and
            setup_sell.iat[i-1] < 9 and
            close.iat[i] > close.iat[i-4]
        ):
            setup_sell.iat[i] = setup_sell.iat[i-1] + 1
        else:
            if setup_sell.iat[i-1] < 9 if i > 0 else True:
                setup_sell.iat[i] = 0

    out = c.copy()
    out['td_setup_buy'] = setup_buy
    out['td_setup_sell'] = setup_sell
    return out

# apply to your GOOGL close series
c = pd.DataFrame(each_mags_df['GOOGL'])
df = td_setup(c)



def td_combo_buy(df: pd.DataFrame) -> pd.Series:
    """
    TD Combo buy countdown (1–13) built on top of td_setup_buy.
    Assumes df has columns: 'GOOGL' (or 'Close'), 'td_setup_buy'.
    """
    close = df.iloc[:, 0]          # your GOOGL close
    # if you later add High/Low, change this to df['Low']; for now approximate with close
    low = close                    # placeholder if you only have closes

    setup = df['td_setup_buy']

    cd = pd.Series(0, index=df.index, dtype=int)
    last_cd_idx = None
    k = 0  # current countdown number (0–13)

    for i in range(len(df)):
        # only count while a buy setup is active (1–9)
        if setup.iat[i] == 0:
            continue

        if k >= 13:
            # countdown complete for this setup
            continue

        qualifies = False

        if k < 10:
            # STRICT RULES for countdown 1–10
            if i >= 2:
                cond1 = close.iat[i] <= low.iat[i-2]       # close <= low 2 bars ago
            else:
                cond1 = False
            if i >= 1:
                cond2 = low.iat[i] < low.iat[i-1]          # low < prior low
                cond3 = close.iat[i] < close.iat[i-1]      # close < prior close
            else:
                cond2 = cond3 = False

            cond4 = True if last_cd_idx is None else close.iat[i] < close.iat[last_cd_idx]
            qualifies = cond1 and cond2 and cond3 and cond4

        else:
            # RELAXED RULES for countdown 11–13:
            # each new countdown bar must close below the previous countdown bar
            cond = True if last_cd_idx is None else close.iat[i] < close.iat[last_cd_idx]
            qualifies = cond

        if qualifies:
            k += 1
            cd.iat[i] = k
            last_cd_idx = i

    return cd

df['td_combo_buy'] = td_combo_buy(df)



# text labels
df['td_setup_buy_label']  = df['td_setup_buy'].astype(str).where(df['td_setup_buy']  > 0, '')
df['td_combo_buy_label']  = df['td_combo_buy'].astype(str).where(df['td_combo_buy']  > 0, '')

# quick sanity check
print(df[['GOOGL','td_setup_buy','td_combo_buy']].tail(25))


import pandas as pd

df['td_setup_buy_label'] = df['td_setup_buy'].astype(str).where(df['td_setup_buy'] > 0, '')
df['td_combo_buy_label'] = df['td_combo_buy'].astype(str).where(df['td_combo_buy'] > 0, '')




df = df['2025-01-01':]
import plotly.graph_objects as go

fig = go.Figure()

# Candles (using close only; if you later add Open/High/Low, plug them in here)
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['GOOGL'],   # if you only have close, you can duplicate it for mock candles
        high=df['GOOGL'],
        low=df['GOOGL'],
        close=df['GOOGL'],
        name='GOOGL'
    )
)

# Y positions for labels: small offsets above/below price
y_setup = df['GOOGL'] * 0.995
y_combo = df['GOOGL'] * 1.005

# Green setup numbers below bars
mask_setup = df['td_setup_buy'] > 0
fig.add_trace(
    go.Scatter(
        x=df.index[mask_setup],
        y=y_setup[mask_setup],
        mode='text',
        text=df.loc[mask_setup, 'td_setup_buy_label'],
        textposition='bottom center',
        textfont=dict(color='green', size=10),
        showlegend=False
    )
)

# Red combo numbers above bars
mask_combo = df['td_combo_buy'] > 0
fig.add_trace(
    go.Scatter(
        x=df.index[mask_combo],
        y=y_combo[mask_combo],
        mode='text',
        text=df.loc[mask_combo, 'td_combo_buy_label'],
        textposition='top center',
        textfont=dict(color='red', size=10),
        showlegend=False
    )
)

fig.update_layout(
    title='GOOGL with TD Setup (green) and TD Combo (red)',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_white',
    height=700,
    width=1200
)

fig.show()



