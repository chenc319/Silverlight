### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_weekly = pd.DataFrame(sp500['Close']).resample('W-FRI').last()
spx_weekly.columns = ['spx']
spx_daily = pd.DataFrame(sp500['Close'])
spx_daily.columns = ['spx']
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_weekly = pd.DataFrame(agg['Close']).resample('W-FRI').last()
bonds_weekly.columns = ['bonds']
bonds_daily = pd.DataFrame(agg['Close'])
bonds_daily.columns = ['bonds']
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

with open(Path(DATA_DIR) / '^BCOM.csv', 'rb') as file:
    bcom = pd.read_csv(file)
bcom.index = pd.to_datetime(bcom['Date']).values
bcom.drop('Date', axis=1, inplace=True)
bcom_weekly = pd.DataFrame(bcom['Close']).resample('W-FRI').last()
bcom_weekly.columns = ['bcom']
bcom_daily = pd.DataFrame(bcom['Close'])
bcom_daily.columns = ['bcom']
bcom_monthly = pd.DataFrame(bcom['Close']).resample('ME').last()
bcom_monthly.columns = ['bcom']

### MERGE DFS ###
cross_asset_daily_merge = merge_dfs([spx_daily,bonds_daily,bcom_daily]).dropna()
cross_asset_weekly_merge = merge_dfs([spx_weekly,bonds_weekly,bcom_weekly]).dropna()
cross_asset_monthly_merge = merge_dfs([spx_monthly,bonds_monthly,bcom_monthly]).dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

regime_archetypes = {
    "Goldilocks":  np.array([ 1,   0,  -1]),  # Equities best, Commodities worst, Bonds neutral
    "Reflation":   np.array([ 0,  -1,   1]),  # Commodities best, Bonds worst, Equities neutral
    "Stagflation": np.array([-1,  0,   1]),   # Commodities best, Equities worst, Bonds neutral
    "Deflation":   np.array([-1,  1,   0]),   # Bonds best, Equities worst, Commodities neutral
}
asset_cols = ['spx', 'bonds', 'bcom']

### Z SCORE CALCULATION ###
window = 12
df = cross_asset_monthly_merge.pct_change().dropna()
df_rolling_mean = df.rolling(window).mean()
df_rolling_std = df.rolling(window).std()
df_rolling_z = ((df - df_rolling_mean) / df_rolling_std).dropna()

### FUNCTION ###
lambda_ = 1.0
def assign_regime_probs(z_scores_row, regime_archetypes, lambda_=1.0):
    distances = {k: np.linalg.norm(z_scores_row - v) for k, v in regime_archetypes.items()}
    exp_dists = {k: np.exp(-lambda_ * d) for k, d in distances.items()}
    regime_probs = {k: exp_dists[k] / sum(exp_dists.values()) for k in regime_archetypes}
    nearest_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
    return nearest_regime, regime_probs

### BACKTEST ###
results = []
for idx, row in df_rolling_z[asset_cols].iterrows():
    regime, probs = assign_regime_probs(row.values, regime_archetypes, lambda_)
    results.append({"date": idx, "closest_regime": regime, **probs})
regime_df = pd.DataFrame(results)
regime_df.index = regime_df['date'].values
regime_df.drop('date', axis=1, inplace=True)

regime_backtest = merge_dfs([regime_df,df.shift(-1).dropna()])
regime_backtest['equity_bt'] = np.nan
regime_backtest['bonds_bt'] = np.nan
regime_backtest['bcom_bt'] = np.nan
for row in regime_backtest.index:
    if regime_backtest.loc[row, 'closest_regime'] == 'Goldilocks':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx'] * 1.5
        regime_backtest.loc[row,'bonds_bt'] = regime_backtest.loc[row,'bonds'] * 0.8
        regime_backtest.loc[row,'bcom_bt'] = regime_backtest.loc[row,'bcom'] * 0.6
    elif regime_backtest.loc[row, 'closest_regime'] == 'Reflation':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx'] * 1
        regime_backtest.loc[row,'bonds_bt'] = regime_backtest.loc[row,'bonds'] * 0.6
        regime_backtest.loc[row,'bcom_bt'] = regime_backtest.loc[row,'bcom'] * 1
    elif regime_backtest.loc[row, 'closest_regime'] == 'Stagflation':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx'] * 0.8
        regime_backtest.loc[row,'bonds_bt'] = regime_backtest.loc[row,'bonds'] * 0.8
        regime_backtest.loc[row,'bcom_bt'] = regime_backtest.loc[row,'bcom'] * 1
    elif regime_backtest.loc[row, 'closest_regime'] == 'Deflation':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx'] * 0.6
        regime_backtest.loc[row,'bonds_bt'] = regime_backtest.loc[row,'bonds'] * 1
        regime_backtest.loc[row,'bcom_bt'] = regime_backtest.loc[row,'bcom'] * 0.6
regime_backtest = regime_backtest.dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

equities_prometheus_return_metrics = return_metrics(
    regime_backtest[['equity_bt','spx']],
    regime_backtest[['spx']],
    12
)
equities_prometheus_return_metrics['Return/Risk']

bonds_prometheus_return_metrics = return_metrics(
    regime_backtest[['bonds_bt','bonds']],
    regime_backtest[['bonds']],
    12
)
bonds_prometheus_return_metrics['Return/Risk']

bcom_prometheus_return_metrics = return_metrics(
    regime_backtest[['bcom_bt','bcom']],
    regime_backtest[['bcom']],
    12
)
bcom_prometheus_return_metrics['Return/Risk']

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def equity_prometheus_results():
    streamlit_plot(df=(1 + regime_backtest[['equity_bt', 'spx']]).cumprod() - 1,
                   columns_array=['equity_bt', 'spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(equities_prometheus_return_metrics)

def bonds_prometheus_results():
    streamlit_plot(df=(1 + regime_backtest[['bonds_bt', 'bonds']]).cumprod() - 1,
                   columns_array=['bonds_bt', 'bonds'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Bonds Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(bonds_prometheus_return_metrics)

def bcom_prometheus_results():
    streamlit_plot(df=(1 + regime_backtest[['bcom_bt', 'bcom']]).cumprod() - 1,
                   columns_array=['bcom_bt', 'bcom'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='BCOM Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(bcom_prometheus_return_metrics)




