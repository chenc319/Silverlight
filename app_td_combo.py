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
normalized_mags_weights = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL',
                                                  'META','MSFT','NVDA','TSLA'])
normalized_mags_pct = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL',
                                              'META','MSFT','NVDA','TSLA'])
for col in normalized_mags_weights.columns:
    normalized_mags_weights[col] = (mags_weights_df[col] /
                                    mags_weights_df['sum']).resample('ME').last()
    col_df = merge_dfs([mags_monthly_pct[col],
                        normalized_mags_weights[col]]).ffill().dropna()
    col_df.columns = ['pct','weights']
    normalized_mags_pct[col] = col_df['pct'] * col_df['weights']

mock_mags_monthly_pct = pd.DataFrame(normalized_mags_pct.sum(axis=1))
mock_mags_monthly_pct.columns = ['mags']


### ------------------------------------------------------------------------------------ ###
### ------------------------------------- TD COMBO ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

sample_df = pd.DataFrame(each_mags_df['GOOGL'])
sample_df.columns = ['close']

def close_higher_lower_t_4(df):
    df['h/l'] = np.nan
    input_col = df.columns[0]
    for idx in range(4,len(df)):
        row = df.index[idx]
        row_4 = df.index[idx-4]
        if df.loc[row,input_col] > df.loc[row_4,input_col]:
            df.loc[row,'h/l'] = 'h'
        else:
            df.loc[row, 'h/l'] = 'l'
    return df.dropna()

def setup_countdown(df, setup_type) -> pd.Series:




countdown_df = close_higher_lower_t_4(sample_df)
countdown_df['sell_setup'] = setup_countdown(sample_df,'h')
countdown_df['buy_setup'] = setup_countdown(sample_df,'l')

df = countdown_df['2025-01-01':]















import matplotlib.pyplot as plt

# assumes df has: 'close', 'sell_setup', 'buy_setup'
fig, ax = plt.subplots(figsize=(14, 6))

# 1) Price line
ax.plot(df.index, df['close'], color='white', linewidth=1.2, label='Close')

# 2) SELL setups – light green
sell_mask = df['sell_setup'] > 0
ax.scatter(
    df.index[sell_mask],
    df['close'][sell_mask] * 1.01,
    c='#7CFC00',          # light green
    s=25,
    zorder=3
)
for t, y, n in zip(df.index[sell_mask],
                   df['close'][sell_mask] * 1.01,
                   df['sell_setup'][sell_mask]):
    ax.text(t, y, str(int(n)),
            color='#7CFC00', ha='center', va='bottom', fontsize=7)

# 3) BUY setups – dark green
buy_mask = df['buy_setup'] > 0
ax.scatter(
    df.index[buy_mask],
    df['close'][buy_mask] * 0.99,
    c='#006400',          # dark green
    s=25,
    zorder=3
)
for t, y, n in zip(df.index[buy_mask],
                   df['close'][buy_mask] * 0.99,
                   df['buy_setup'][buy_mask]):
    ax.text(t, y, str(int(n)),
            color='#006400', ha='center', va='top', fontsize=7)

# 4) Styling (same dark background look)
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




