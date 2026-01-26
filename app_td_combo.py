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
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA','TSM']
mags_ohlc_dict = dict()
for mag_ticker in mags_tickers:
    mag_string = mag_ticker + '.csv'
    with open(Path(DATA_DIR) / mag_string, 'rb') as file:
        mag_df = pd.read_csv(file)
        mag_df.index = pd.to_datetime(mag_df['Date'].values)
        mag_df.drop('Date', axis=1, inplace=True)
    mags_ohlc_dict[mag_ticker] = mag_df

### ------------------------------------------------------------------------------------ ###
### ------------------------------- SETUP AND COUNTDOWN -------------------------------- ###
### ------------------------------------------------------------------------------------ ###

mags_td_combo_complete_stitches = dict()
mags_td_combo_individual_stitches = dict()
for mag_ticker in mags_tickers:

    ### GET DATAFRAME ###
    mags_df = pd.DataFrame(mags_ohlc_dict[mag_ticker])
    mags_df['h/l'] = close_hl_setup(mags_df,'Close')

    ### SETUP AND PERFECTED ###
    mags_df =  build_td_combo_v2_setups(mags_df)
    mags_df['perfected'] = get_perfected_9(mags_df)
    mini_mag_df = mags_df['2025-01-01':]

    ### COUNTDOWN DICTIONARY ###
    mag_dict = compute_setup_countdown_pairs(
        df = mini_mag_df['2025-01-01':]
    )
    test_stitch = build_stitched_td_df(mini_mag_df,mag_dict)
    mags_td_combo_individual_stitches[mag_ticker] = mag_dict
    mags_td_combo_complete_stitches[mag_ticker] = test_stitch


### ------------------------------------------------------------------------------------ ###
### ------------------------------------- BACKTEST ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

each_mags_backtest_dict = mags_td_combo_individual_stitches['GOOGL']
each_stitch_df = each_mags_backtest_dict['setup_countdown #1']['dataframe']
each_mags_backtest_dict.keys()




each_mags_return_stream_dict = dict()
sum_cumulative_ret = 0
for each_stitch in each_mags_backtest_dict:
    each_stitch_df = each_mags_backtest_dict[each_stitch]['dataframe']
    if each_stitch_df['countdown'][-1] == 13:\

        ### FIND THE FIRST 9 BAR SETUP OCCURRENCE ###
        mask = abs(each_stitch_df['setup']) == 9
        first_idx = each_stitch_df[mask].index[0]

        ### ISOLATE DF SO IT STARTS FROM 9 TO 13 ###
        setup9_to_cd13_df = each_stitch_df[first_idx:]

        ### BUY SETUP AND COUNTDOWN ###
        if each_stitch_df['setup'][first_idx] == -9:
            setup9_to_cd13_df['pct'] = setup9_to_cd13_df['Close'].pct_change()
            setup9_to_cd13_df['cumulative_ret'] = (1 + setup9_to_cd13_df['pct']).cumprod() - 1
            daily_returns_cumulative_returns_df = (setup9_to_cd13_df[['pct','cumulative_ret']] * -1).dropna()

        ### SELL SETUP AND COUNTDOWN ###
        elif each_stitch_df['setup'][first_idx] == 9:
            setup9_to_cd13_df['pct'] = setup9_to_cd13_df['Close'].pct_change()
            setup9_to_cd13_df['cumulative_ret'] = (1 + setup9_to_cd13_df['pct']).cumprod() - 1
            daily_returns_cumulative_returns_df = (setup9_to_cd13_df[['pct','cumulative_ret']] * 1).dropna()

        each_mags_return_stream_dict[each_stitch] = daily_returns_cumulative_returns_df

        sum_cumulative_ret += daily_returns_cumulative_returns_df['cumulative_ret'][-1]


# 0) Build px and strat as you already do
px = pd.DataFrame(mags_ohlc_dict['GOOGL']['2025-01-01':]['Close'])
strat = pd.DataFrame(pd.concat(each_mags_return_stream_dict.values()))

# 1) Ensure datetime index and sorted
px.index = pd.to_datetime(px.index)
strat.index = pd.to_datetime(strat.index)

px = px.sort_index()
strat = strat.sort_index()

# 2) Ensure indices are UNIQUE (drop any duplicate dates)
px = px[~px.index.duplicated(keep='first')]
strat = strat[~strat.index.duplicated(keep='first')]
# (use keep='last' instead if that makes more sense for you)[web:176][web:173]

# 3) Keep only the 'pct' column from strat (if it has others)
#    adjust if your column is named differently
strat = strat[['pct']]

# 4) Reindex strategy to full trading-day index of px
strat_full = strat.reindex(px.index)

# 5) Fill missing pct with 0 on days the strategy is not live
strat_full['pct'] = strat_full['pct'].fillna(0.0)

# 6) Recompute cumulative_ret from the filled pct
strat_full['cumulative_ret'] = (1 + strat_full['pct']).cumprod() - 1

plt.plot(strat_full['cumulative_ret'])
plt.show()


(strat_full['pct'].mean() * 252) / (strat_full['pct'].std() * 252**0.5)


mags_ohlc_dict['GOOGL']


### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

# plot_td_combo_case_study_stitched(test_stitch,'n')
# plot_td_combo_case_study_test_run(googl_dict,1,'n')


def plot_stitched_googl():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['GOOGL'])

def plot_stitched_amzn():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['AMZN'])

def plot_stitched_aapl():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['AAPL'])

def plot_stitched_meta():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['META'])

def plot_stitched_msft():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['MSFT'])

def plot_stitched_nvda():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['NVDA'])

def plot_stitched_tsla():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['TSLA'])

