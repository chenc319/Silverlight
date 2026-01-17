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

### GET DATAFRAME ###
googl_df = pd.DataFrame(mags_ohlc_dict['GOOGL'])
googl_df['h/l'] = close_hl_setup(googl_df,'Close')

### SETUP AND PERFECTED ###
googl_df =  build_td_combo_v2_setups(googl_df)
googl_df['perfected'] = get_perfected_9(googl_df)
mini_googl_df = googl_df['2025-01-01':]

### COUNTDOWN DICTIONARY ###
googl_dict = compute_setup_countdown_pairs(
    df = mini_googl_df['2025-01-01':]
)

### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

def plot_googl_case_study_1():
    plot_td_combo_case_study(googl_dict,0)