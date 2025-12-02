### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- CROSS-ASSET REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###
import pandas as pd

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- CROSS-ASSET REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

with open(Path(DATA_DIR) / '^BCOM.csv', 'rb') as file:
    bcom = pd.read_csv(file)
bcom.index = pd.to_datetime(bcom['Date']).values
bcom.drop('Date', axis=1, inplace=True)
bcom_monthly = pd.DataFrame(bcom['Close']).resample('ME').last()
bcom_monthly.columns = ['bcom']

with open(Path(DATA_DIR) / 'DXY.csv', 'rb') as file:
    dxy = pd.read_csv(file)
dxy.index = pd.to_datetime(dxy['Date']).values
dxy.drop('Date', axis=1, inplace=True)
dxy_monthly = pd.DataFrame(dxy['Close']).resample('ME').last()
dxy_monthly.columns = ['dxy']

### MERGE DFS ###
cross_asset_monthly_merge = merge_dfs([
    spx_monthly,
    bonds_monthly,
    bcom_monthly,
    dxy_monthly]
).dropna()

# 1) Base returns
df = cross_asset_monthly_merge.pct_change().dropna()
df.columns = ['spx_ret', 'bonds_ret', 'bcom_ret','dxy_ret']

# 2) 12m realized vol for each asset
vol_window = 12
vol = df.rolling(vol_window).std()
vol.columns = ['spx_vol', 'bonds_vol', 'bcom_vol','dxy_vol']

# 4) Merge into full feature matrix
features = merge_dfs([df, vol]).dropna()

feature_cols = [
    'spx_ret', 'bonds_ret', 'bcom_ret','dxy_ret',
    'spx_vol', 'bonds_vol', 'bcom_vol','dxy_vol'
]

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- Z SCORE CALCULATION --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

window = 36
feat_rolling_mean = features[feature_cols].rolling(window).mean()
feat_rolling_std  = features[feature_cols].rolling(window).std()
feat_rolling_z = ((features[feature_cols] - feat_rolling_mean) / feat_rolling_std).dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- REGIME ARCHETYPES ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

regime_archetypes = {
    "Goldilocks": np.array([
        1,   # spx_ret    (equities up)
        0,   # bonds_ret  (bonds flat/moderate)
        -1,  # bcom_ret   (commodities soft)
        -1,  # dxy_ret    (dollar weak / risk-on)
        -1,  # spx_vol    (low equity vol)
        0,   # bonds_vol  (neutral bond vol)
        0,   # bcom_vol   (neutral commodity vol)
        -1   # dxy_vol    (low FX vol / calm dollar)
    ]),
    "Reflation": np.array([
        1,   # spx_ret    (equities up)
        -1,  # bonds_ret  (bonds down, yields up)
        1,   # bcom_ret   (commodities up)
        -1,  # dxy_ret    (dollar weaker on global growth)
        1,   # spx_vol    (higher equity vol)
        0,   # bonds_vol  (neutral to modestly higher)
        1,   # bcom_vol   (higher commodity vol)
        0    # dxy_vol    (neutral FX vol; not a crisis)
    ]),
    "Stagflation": np.array([
        -1,  # spx_ret    (equities down)
        -1,  # bonds_ret  (bonds pressured by inflation)
        1,   # bcom_ret   (commodities up on inflation)
        1,   # dxy_ret    (dollar up on stress/inflation)
        1,   # spx_vol    (high equity vol)
        1,   # bonds_vol  (high rates/bond vol)
        1,   # bcom_vol   (high commodity vol)
        1    # dxy_vol    (high FX vol / stress)
    ]),
    "Deflation": np.array([
        -1,  # spx_ret    (equities down)
        1,   # bonds_ret  (bonds up, yields fall)
        -1,  # bcom_ret   (commodities weak)
        1,   # dxy_ret    (dollar up as safe haven)
        0,   # spx_vol    (elevated but not as extreme as stagflation)
        0,   # bonds_vol  (moderate; yields grinding lower)
        -1,  # bcom_vol   (low commodity vol / demand collapse)
        1    # dxy_vol    (higher FX vol on deleveraging)
    ])
}


lambda_ = 1.0

def assign_regime_probs(z_scores_row, regime_archetypes, lambda_=1.0):
    distances = {k: np.linalg.norm(z_scores_row - v) for k, v in regime_archetypes.items()}
    exp_dists = {k: np.exp(-lambda_ * d) for k, d in distances.items()}
    regime_probs = {k: exp_dists[k] / sum(exp_dists.values()) for k in regime_archetypes}
    nearest_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
    return nearest_regime, regime_probs

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- BACKTEST -------------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

results = []
for idx, row in feat_rolling_z.iterrows():
    regime, probs = assign_regime_probs(row.values, regime_archetypes, lambda_)
    results.append({"date": idx, "closest_regime": regime, **probs})

regime_df = pd.DataFrame(results)
regime_df.index = regime_df['date'].values
regime_df.drop('date', axis=1, inplace=True)

# Use next-month *returns* from the original 3-asset return df
regime_backtest = merge_dfs([regime_df, df.shift(-1).dropna()])
regime_backtest['equity_bt'] = np.nan
regime_backtest['bonds_bt'] = np.nan
regime_backtest['bcom_bt'] = np.nan

for row in regime_backtest.index:
    if regime_backtest.loc[row, 'closest_regime'] == 'Goldilocks':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx_ret']   * 1.0
        regime_backtest.loc[row,'bonds_bt']  = regime_backtest.loc[row,'bonds_ret'] * 0.8
        regime_backtest.loc[row,'bcom_bt']   = regime_backtest.loc[row,'bcom_ret']  * 0.6
    elif regime_backtest.loc[row, 'closest_regime'] == 'Reflation':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx_ret']   * 0.9
        regime_backtest.loc[row,'bonds_bt']  = regime_backtest.loc[row,'bonds_ret'] * 0.6
        regime_backtest.loc[row,'bcom_bt']   = regime_backtest.loc[row,'bcom_ret']  * 1.0
    elif regime_backtest.loc[row, 'closest_regime'] == 'Stagflation':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx_ret']   * 0.7
        regime_backtest.loc[row,'bonds_bt']  = regime_backtest.loc[row,'bonds_ret'] * 0.8
        regime_backtest.loc[row,'bcom_bt']   = regime_backtest.loc[row,'bcom_ret']  * 1.0
    elif regime_backtest.loc[row, 'closest_regime'] == 'Deflation':
        regime_backtest.loc[row,'equity_bt'] = regime_backtest.loc[row,'spx_ret']   * 0.6
        regime_backtest.loc[row,'bonds_bt']  = regime_backtest.loc[row,'bonds_ret'] * 1.0
        regime_backtest.loc[row,'bcom_bt']   = regime_backtest.loc[row,'bcom_ret']  * 0.6

regime_backtest = regime_backtest.dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- CROSS-ASSET REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

equities_prometheus_return_metrics = return_metrics(
    regime_backtest[['equity_bt','spx_ret']],
    regime_backtest[['spx_ret']],
    12
)
equities_prometheus_return_metrics['Return/Risk']

bonds_prometheus_return_metrics = return_metrics(
    regime_backtest[['bonds_bt','bonds_ret']],
    regime_backtest[['bonds_ret']],
    12
)
bonds_prometheus_return_metrics['Return/Risk']

bcom_prometheus_return_metrics = return_metrics(
    regime_backtest[['bcom_bt','bcom_ret']],
    regime_backtest[['bcom_ret']],
    12
)
bcom_prometheus_return_metrics['Return/Risk']

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- CROSS-ASSET REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_colorcoded_regime():
    regime_merge = merge_dfs([
        pd.DataFrame(regime_backtest['closest_regime']),
        spx_monthly,
        pd.DataFrame(regime_backtest['equity_bt'])
    ]).dropna()
    color_coded_regime_plot(regime_merge,
                            y_col='spx',
                            regime_col='closest_regime',
                            title="S&P 500 Level by Macro Regime")
    stats_df = calculate_regime_statistics(regime_merge,
                                           regime_col_name='closest_regime',
                                           return_cols=['equity_bt'])
    plot_streamlit_regime_statistics(stats_df)

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



