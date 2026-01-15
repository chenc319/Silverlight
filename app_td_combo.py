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

sample_df = pd.DataFrame(mags_ohlc_dict['GOOGL'])

### ------------------------------------------------------------------------------------ ###
### -------------------------------------- SETUP --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### SETUP ###
def close_hl_setup(df,close_col_name):
    df['h/l'] = np.nan
    for idx in range(4,len(df)):
        row = df.index[idx]
        row_4 = df.index[idx-4]
        if df.loc[row,close_col_name] > df.loc[row_4,close_col_name]:
            df.loc[row,'h/l'] = 'h'
        else:
            df.loc[row, 'h/l'] = 'l'
    return df['h/l']

def td_combo_setup(df,setup_type):
    ### CONSECUTIVE DAYS ###
    df['setup'] = 0
    for idx in range(1,len(df.index)):
        row = df.index[idx]
        prev_row = df.index[idx-1]
        if df.loc[prev_row,'setup'] != 9:
            if df.loc[row,'h/l'] == setup_type:
                df.loc[row,'setup'] = df.loc[prev_row,'setup'] + 1

    ### FILTER OUT INCOMPLETE 9S ###
    for idx in range(8,len(df.index)):
        row = df.index[idx]
        prev_row = df.index[idx - 1]
        if ((df.loc[row, 'setup'] < df.loc[prev_row, 'setup'])
                and (df.loc[prev_row, 'setup'] != 9)):
            total_rows_to_delete = df.loc[prev_row, 'setup']
            for num_row_to_delete in range(0,total_rows_to_delete+1):
                df.loc[df.index[idx-num_row_to_delete],'setup'] = 0
    return df['setup']

sample_df['h/l'] = close_hl_setup(sample_df,'Close')
sample_df['sell_setup'] = td_combo_setup(sample_df,'h')
sample_df['buy_setup'] = td_combo_setup(sample_df,'l')

### ------------------------------------------------------------------------------------ ###
### ------------------------------------ COUNTDOWN ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### COUNTDOWN BUY SETUP ###
df = sample_df['2025-01-01':]
buy_setup_rows_to_iterate = df[df['buy_setup'] == 1].index
buy_setup_countdown_dict = dict()

### ------------------------------------ BUY SETUP ------------------------------------- ###

buy_setup_dict_key_index = 1
for first_bar_of_setup_date in buy_setup_rows_to_iterate:
    ### TDST FILTER ###
    tdst_possible_candidates = df[date_before_first_bar_setup:date_t_9_after_first_bar_setup].copy()
    prev_close = tdst_possible_candidates['Close'].shift(1)
    # true high for each of the 9 setup bars
    tdst_possible_candidates['true_high'] = tdst_possible_candidates['High'].combine(prev_close, max)
    tdst_possible_candidates = tdst_possible_candidates.iloc[1:]
    # 1) TDST resistance value
    tdst_resistance = tdst_possible_candidates['true_high'].max()
    # 2) index where that max occurs
    tdst_idx = tdst_possible_candidates['true_high'].idxmax()
    # 3) buy_setup value on that row
    tdst_buy_setup_value = tdst_possible_candidates.loc[tdst_idx, 'buy_setup']

    ### TD RISK LEVEL ###
    s = tdst_possible_candidates.copy()

    prev_close = s['Close'].shift(1)
    s['true_high'] = s['High'].combine(prev_close, max)
    s['true_low'] = s['Low'].combine(prev_close, min)
    s['true_range'] = s['true_high'] - s['true_low']

    # bar with lowest true low
    risk_idx = s['true_low'].idxmin()

    risk_true_low = s.loc[risk_idx, 'true_low']
    risk_true_range = s.loc[risk_idx, 'true_range']

    td_risk_level_buy = risk_true_low - risk_true_range


    ### COUNTDOWN PARAMETERS ###
    df['buy_countdown'] = 0
    setup_row_idx = df.index.get_loc(first_bar_of_setup_date)
    td_buy_setup_int = 1

    ### SET COUNTDOWN BAR ROWS AND IDX ###
    prev_countdown_bar_idx = None
    prev_countdown_bar_row = None

    for idx in range(setup_row_idx, len(df.index)):
        if td_buy_setup_int > 13:
            break

        row = df.index[idx]
        t_2_prev_row = df.index[idx - 2]
        t_1_prev_row = df.index[idx - 1]

        ### BARS 1-10 ###
        if td_buy_setup_int <= 10:
            if (df.loc[row, 'Close'] <= df.loc[t_2_prev_row, 'Low'] and
                    df.loc[row, 'Low'] <= df.loc[t_1_prev_row, 'Low'] and
                    df.loc[row, 'Close'] <= df.loc[t_1_prev_row, 'Close']):

                # Condition 4 only if we have a previous countdown bar
                if prev_countdown_bar_row is not None:
                    if df.loc[row, 'Close'] < df.loc[prev_countdown_bar_row, 'Close']:
                        df.loc[row, 'buy_countdown'] = td_buy_setup_int
                        prev_countdown_bar_idx = idx
                        prev_countdown_bar_row = row
                        td_buy_setup_int += 1
                else:
                    # first countdown bar: skip condition 4
                    df.loc[row, 'buy_countdown'] = td_buy_setup_int
                    prev_countdown_bar_idx = idx
                    prev_countdown_bar_row = row
                    td_buy_setup_int += 1

        ### BARS 11–13 ###
        elif 11 <= td_buy_setup_int <= 13:
            if prev_countdown_bar_row is not None:
                # Version 2: close must be less than previous COUNTDOWN bar close
                if df.loc[row, 'Close'] < df.loc[prev_countdown_bar_row, 'Close']:
                    df.loc[row, 'buy_countdown'] = td_buy_setup_int
                    prev_countdown_bar_idx = idx
                    prev_countdown_bar_row = row
                    td_buy_setup_int += 1

    ### CHECK TO SEE IF THERE ARE ANY 13s ###
    filtered_df = df[first_bar_of_setup_date:first_tdst_idx]
    final_dict = dict()
    dict_name = 'buy_setup_countdown #' + str(buy_setup_dict_key_index)
    buy_setup_dict_key_index += 1

    if td_buy_setup_int == 14:
        final_dict['valid setup-countown_pair'] = 'yes'
        final_dict['dataframe'] = filtered_df[first_bar_of_setup_date:
                                              prev_countdown_bar_row]
    else:
        final_dict['valid setup-countown_pair'] = 'no'
        final_dict['dataframe'] = filtered_df

    ### TDST ###
    final_dict['tdst_value'] = tdst_resistance
    final_dict['tdst_date'] = tdst_idx
    final_dict['tdst_setup_bar_start'] = tdst_buy_setup_value

    ### TD RISK LEVEL ###
    final_dict['td_risk_level_val'] =
    buy_setup_countdown_dict[dict_name] = final_dict

### ------------------------------------ SELL SETUP ------------------------------------- ###



### ------------------------------------------------------------------------------------ ###
### --------------------------- TDST RESISTANCE/SUPPORT LINES -------------------------- ###
### ------------------------------------------------------------------------------------ ###

### TDST RESISTANCE - BUY SETUP ###


list(buy_setup_countdown_dict.keys())




















### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###


import matplotlib.pyplot as plt

# assumes df has: 'close', 'sell_setup', 'buy_setup'
fig, ax = plt.subplots(figsize=(14, 6))

# 1) Price line
ax.plot(df.index, df['Close'], color='white', linewidth=1.2, label='Close')

# 2) SELL setups – light green numbers only (above price)
sell_mask = df['sell_setup'] > 0
for t, n in zip(df.index[sell_mask], df['sell_setup'][sell_mask]):
    y = df.loc[t, 'Close'] * 1.01
    ax.text(
        t, y, str(int(n)),
        color='#7CFC00',        # light green
        ha='center', va='bottom',
        fontsize=7
    )

# 3) BUY setups – dark green numbers only (below price)
buy_mask = df['buy_setup'] > 0
for t, n in zip(df.index[buy_mask], df['buy_setup'][buy_mask]):
    y = df.loc[t, 'Close'] * 0.99
    ax.text(
        t, y, str(int(n)),
        color='#006400',        # dark green
        ha='center', va='top',
        fontsize=7
    )

# 4) Styling – dark chart look
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

ax.set_xlabel('')
ax.set_ylabel('GOOGL Close', color='white')
ax.set_title('GOOGL – TD Buy/Sell Setups', color='white')
ax.grid(False)

plt.tight_layout()
plt.show()




