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
### -------------------------------------- SETUP --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

mags_td_combo_stitches = dict()
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
    mags_td_combo_stitches[mag_ticker] = test_stitch

### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

# plot_td_combo_case_study_stitched(test_stitch,'n')
# plot_td_combo_case_study_test_run(googl_dict,1,'n')


def plot_stitched_googl():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['GOOGL'])

def plot_stitched_amzn():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['AMZN'])

def plot_stitched_aapl():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['AAPL'])

def plot_stitched_meta():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['META'])

def plot_stitched_msft():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['MSFT'])

def plot_stitched_nvda():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['NVDA'])

def plot_stitched_tsla():
    plot_td_combo_case_study_stitched(mags_td_combo_stitches['TSLA'])

