### ------------------------------------------------------------------------------------ ###
### ------------------------------------- TD COMBO ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### FUNCTIONS ###
def close_hl_setup(df,close_col_name):
    df['h/l'] = np.nan
    for idx in range(4,len(df)):
        row_name = df.index[idx]
        row_4 = df.index[idx-4]
        if df.loc[row_name,close_col_name] > df.loc[row_4,close_col_name]:
            df.loc[row_name, 'h/l'] = 'h'
        else:
            df.loc[row_name, 'h/l'] = 'l'
    return df['h/l']

def td_combo_setup(df,setup_type):
    ### CONSECUTIVE DAYS ###
    df['setup'] = 0
    for idx in range(1,len(df.index)):
        row_name = df.index[idx]
        prev_row = df.index[idx-1]
        if df.loc[prev_row,'setup'] != 9:
            if df.loc[row_name, 'h/l'] == setup_type:
                df.loc[row_name, 'setup'] = df.loc[prev_row, 'setup'] + 1

    ### FILTER OUT INCOMPLETE 9S ###
    for idx in range(8,len(df.index)):
        row_name = df.index[idx]
        prev_row = df.index[idx - 1]
        if ((df.loc[row_name, 'setup'] < df.loc[prev_row, 'setup'])
                and (df.loc[prev_row, 'setup'] != 9)):
            total_rows_to_delete = df.loc[prev_row, 'setup']
            for num_row_to_delete in range(0,total_rows_to_delete+1):
                df.loc[df.index[idx-num_row_to_delete],'setup'] = 0
    return df['setup']

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

### ------------------------------------------------------------------------------------ ###
### ------------------------------- BUY SETUP COUNTDOWN -------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### COUNTDOWN BUY SETUP ###
df = sample_df['2025-01-01':]
buy_setup_rows_to_iterate = df[df['buy_setup'] == 1].index
buy_setup_countdown_dict = dict()

buy_setup_dict_key_index = 1
first_bar_of_setup_date = buy_setup_rows_to_iterate[0]

for setup_start in buy_setup_rows_to_iterate:

    ### TDST ###
    prev_bar = df.index[df.index.get_loc(setup_start) - 1]
    bar9 = df.index[df.index.get_loc(setup_start) + 8]

    tdst_win = df.loc[prev_bar:bar9].copy()
    prev_close = tdst_win['Close'].shift(1)

    tdst_win['true_high'] = tdst_win['High'].combine(prev_close, max)
    tdst_win = tdst_win.iloc[1:]  # drop pre-setup bar

    tdst_val  = tdst_win['true_high'].max()
    tdst_date = tdst_win['true_high'].idxmax()
    tdst_setup_num = tdst_win.loc[tdst_date, 'buy_setup']

    first_tdst_break = (df.loc[setup_start:, 'Close'] > tdst_val).idxmax()

    ### COUNTDOWN ###
    df['buy_countdown'] = 0
    setup_idx = df.index.get_loc(setup_start)
    cdn_num   = 1
    last_cdn_bar = None
    for i in range(setup_idx, len(df.index)):
        if cdn_num > 13:
            break
        row = df.index[i]
        row_t1 = df.index[i - 1]
        row_t2 = df.index[i - 2]
        c = df.loc[row, 'Close']
        l = df.loc[row, 'Low']

        ### BARS 1-10 ###
        if cdn_num <= 10:
            base_ok = (
                c <= df.loc[row_t2, 'Low'] and
                l <= df.loc[row_t1, 'Low'] and
                c <= df.loc[row_t1, 'Close']
            )
            if not base_ok:
                continue

            if last_cdn_bar is None or cdn_num == 1:
                df.loc[row, 'buy_countdown'] = cdn_num
                last_cdn_bar = row
                cdn_num += 1
                continue

            if c < df.loc[last_cdn_bar, 'Close']:
                df.loc[row, 'buy_countdown'] = cdn_num
                last_cdn_bar = row
                cdn_num += 1

        ### BARS 11-13 ###
        else:
            if (last_cdn_bar is not None
                    and (c < df.loc[last_cdn_bar, 'Close'] or
                         c < df.loc[last_cdn_bar, 'Open'])):
                df.loc[row, 'buy_countdown'] = cdn_num
                last_cdn_bar = row
                cdn_num += 1

    ### STORE DATA IN DICTIONARIES ###
    window = df.loc[setup_start:first_tdst_break]
    name   = f"buy_setup_countdown #{buy_setup_dict_key_index}"
    buy_setup_dict_key_index += 1

    result = {
        'tdst_val': tdst_val,
        'tdst_date': tdst_date,
        'tdst_setup_start_bar': tdst_setup_num,
    }

    if cdn_num == 14 and last_cdn_bar is not None:
        risk_win = window.loc[setup_start:last_cdn_bar].copy()
        prev_close = risk_win['Close'].shift(1)

        risk_win['true_high'] = risk_win['High'].combine(prev_close, max)
        risk_win['true_low'] = risk_win['Low'].combine(prev_close, min)
        risk_win['true_range'] = risk_win['true_high'] - risk_win['true_low']
        risk_win = risk_win.iloc[1:]

        risk_date  = risk_win['true_low'].idxmin()
        risk_low   = risk_win.loc[risk_date, 'true_low']
        risk_range = risk_win.loc[risk_date, 'true_range']
        risk_val   = risk_low - risk_range

        result.update({
            'valid setup-cdn_pair': 'yes',
            'dataframe': window.loc[setup_start:last_cdn_bar],
            'risk_lvl':  risk_val,
            'risk_date': risk_date,
        })
    else:
        result.update({
            'valid setup-cdn_pair': 'no',
            'dataframe': window,
            'risk_lvl':  None,
            'risk_date': None,
        })
    buy_setup_countdown_dict[name] = result

### ------------------------------------------------------------------------------------ ###
### ------------------------------- SELL SETUP COUNTDOWN ------------------------------- ###
### ------------------------------------------------------------------------------------ ###





















### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

def plot_googl_initial_results():
    df = buy_setup_countdown_dict[list(buy_setup_countdown_dict.keys())[0]]['dataframe']
    df['Date'] = df.index
    df['DateStr'] = df.index.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)

    # Create series that span the whole visible window
    df['TDST'] = buy_setup_countdown_dict[list(buy_setup_countdown_dict.keys())[0]]['tdst_val']
    df['TD_Risk'] = buy_setup_countdown_dict[list(buy_setup_countdown_dict.keys())[0]]['risk_lvl']



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

