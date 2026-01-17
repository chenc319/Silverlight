### ------------------------------------------------------------------------------------ ###
### ------------------------------------- TD COMBO ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ------------------------------------------------------------------------------------ ###
### ------------------------------------- TD COMBO ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### OHLC MAGS DATA ###
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA']
mags_ohlc_dict = dict()
for mag_ticker in mags_tickers:
    mag_string = mag_ticker + '.csv'
    with open(Path(DATA_DIR) / mag_string, 'rb') as file:
        mag_df = pd.read_csv(file)
        mag_df.index = pd.to_datetime(mag_df['Date'].values)
        mag_df.drop('Date', axis=1, inplace=True)
    mags_ohlc_dict[mag_ticker] = mag_df

### ------------------------------------------------------------------------------------ ###
### -------------------------------------- SETUP --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

sample_df = pd.DataFrame(mags_ohlc_dict['GOOGL'])
sample_df['h/l'] = close_hl_setup(sample_df,'Close')
sample_df['sell_setup'] = td_combo_setup(sample_df,'h')
sample_df['buy_setup'] = td_combo_setup(sample_df,'l')

sample_df = build_td_combo_v2_setups(sample_df)
sample_df['perfected'] = get_perfected_9(sample_df)

### ------------------------------------------------------------------------------------ ###
### ------------------------------- BUY SETUP COUNTDOWN -------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### COUNTDOWN BUY SETUP ###
df = sample_df['2025-01-01':]
df,setup_dict_key_index = compute_setup_countdown_pairs(df)



















### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

def plot_googl_initial_results():
    df = setup_dict_key_index[list(setup_dict_key_index.keys())[0]]['dataframe']
    df['Date'] = df.index
    df['DateStr'] = df.index.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)

    # Create series that span the whole visible window
    df['TDST'] = setup_dict_key_index[list(setup_dict_key_index.keys())[0]]['tdst_val']
    df['TD_Risk'] = setup_dict_key_index[list(setup_dict_key_index.keys())[0]]['risk_lvl']



    # --- 3. Base OHLC candlestick chart over ALL rows ---
    fig = go.Figure()

    # 1) Candles over ALL bars, x = DateStr (categorical, no gaps)
    fig.add_trace(
        go.Ohlc(
            x=df['DateStr'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='white',
            decreasing_line_color='white',
            name='Price'
        )
    )

    # 2) Setup labels
    setup_mask = df['setup'] > 0
    fig.add_trace(
        go.Scatter(
            x=df.loc[setup_mask, 'DateStr'],
            y=df.loc[setup_mask, 'High'] * 1.01,
            mode='text',
            text=df.loc[setup_mask, 'setup'].astype(int).astype(str),
            textfont=dict(color='lime', size=12),
            textposition='top center',
            name='TD Setup'
        )
    )

    # 3) Countdown labels
    cd_mask = df['buy_countdown'] > 0
    fig.add_trace(
        go.Scatter(
            x=df.loc[cd_mask, 'DateStr'],
            y=df.loc[cd_mask, 'Low'] * 0.99,
            mode='text',
            text=df.loc[cd_mask, 'buy_countdown'].astype(int).astype(str),
            textfont=dict(color='magenta', size=12),
            textposition='bottom center',
            name='TD Buy Countdown'
        )
    )

    # 4) Single TDST + TD Risk lines
    fig.add_trace(
        go.Scatter(
            x=df['DateStr'],
            y=df['TDST'],
            mode='lines',
            line=dict(color='limegreen', width=1.5),
            name='TD TDST Level'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['DateStr'],
            y=df['TD_Risk'],
            mode='lines',
            line=dict(color='magenta', width=1.5, dash='dot'),
            name='TD Risk Level'
        )
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            type='category',      # ensure categorical axis (no date gaps)
            rangeslider_visible=False
        ),
        yaxis=dict(showgrid=False),
        height=700,
        width=1200
    )

    st.plotly_chart(fig, use_container_width=True)

