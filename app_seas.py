### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- SEASONALITY MODEL ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
from matplotlib import pyplot as plt
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- SEASONALITY MODEL ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']
spx_monthly_pct = spx_monthly.pct_change().dropna()

### MAGS ###
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
mags_monthly_pct = each_mags_df.resample('ME').last().pct_change().dropna()

### BONDS ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

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

def get_seasonality(df,target_col):
    df['month_num'] = df.index.month
    df['month_name'] = df.index.month_name()
    seasonality = (
        df.groupby('month_num')[target_col]
        .mean()
        .to_frame(name='avg_spread')
    )
    seasonality['month'] = seasonality.index.map(lambda m: pd.Timestamp(2000, m, 1).strftime('%b'))
    seasonality = seasonality[['month', 'avg_spread']]
    return(seasonality)
    print(seasonality)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- SEASONALITY MODEL ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### SPX AND MAGS RV ###
spx_mags_merge = merge_dfs([spx_monthly_pct,mock_mags_monthly_pct]).dropna()
spx_mags_merge['rv_diff'] = spx_mags_merge['mags'] - spx_mags_merge['spx']
spx_mags_merge['rv_mean'] = spx_mags_merge['rv_diff'].rolling(12).mean()
spx_mags_merge['rv_std'] = spx_mags_merge['rv_diff'].rolling(12).std()
spx_mags_merge['rv_z'] = (spx_mags_merge['rv_diff'] - spx_mags_merge['rv_mean']) / spx_mags_merge['rv_std']
plt.plot(spx_mags_merge['rv_z'])
plt.show()

get_seasonality(spx_mags_merge,'rv_z')


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- SEASONALITY MODEL ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### MERGE DFS AND PREPARE ###
spx_bonds_merge = merge_dfs([spx_monthly,bonds_monthly]).pct_change().dropna()
spx_bonds_merge_mean = spx_bonds_merge.rolling(12).mean()
spx_bonds_merge_std = spx_bonds_merge.rolling(12).std()
spx_bonds_merge_z = ((spx_bonds_merge - spx_bonds_merge_mean) / spx_bonds_merge_std).dropna()
spx_bonds_merge_z['spread'] = spx_bonds_merge_z['spx'] - spx_bonds_merge_z['bonds']

spx_bonds_merge_z['month_num'] = spx_bonds_merge_z.index.month
spx_bonds_merge_z['month_name'] = spx_bonds_merge_z.index.month_name()

# group by month number, compute average spread
seasonality = (
    spx_bonds_merge_z.groupby('month_num')['spread']
      .mean()
      .to_frame(name='avg_spread')
)

# optional: attach month names for readability
seasonality['month'] = seasonality.index.map(lambda m: pd.Timestamp(2000, m, 1).strftime('%b'))

# final seasonality DataFrame with Jan..Dec in order
seasonality = seasonality[['month', 'avg_spread']]
print(seasonality)




