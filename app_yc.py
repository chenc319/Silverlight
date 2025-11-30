### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
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

### BONDS ###
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

### YIELDS ###
with open(Path(DATA_DIR) / 'treasury_1m.pkl', 'rb') as file:
    treasury_1m = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_2y.pkl', 'rb') as file:
    treasury_2y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_5y.pkl', 'rb') as file:
    treasury_5y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_10y.pkl', 'rb') as file:
    treasury_10y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_30y.pkl', 'rb') as file:
    treasury_30y = pd.read_pickle(file)

treasury_merge = merge_dfs([treasury_1m,treasury_2y, treasury_5y, treasury_10y,treasury_30y]).dropna()
treasury_merge.index = pd.to_datetime(treasury_merge.index).values
treasury_merge.columns = ['1m','2y','5y','10y','30y']
treasury_monthly_df = treasury_merge.resample('ME').last()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### CALCULATE SLOPES AND DIFFERENCES ###
treasury_diff = treasury_monthly_df.diff().dropna()
treasury_diff['front_end'] = treasury_diff['2y'] - treasury_diff['1m']
treasury_diff['belly'] = treasury_diff['10y'] - treasury_diff['2y']
treasury_diff['back_end'] = treasury_diff['30y'] - treasury_diff['10y']

### CALCULATE CLASSIFICATIONS ###
treasury_diff['total_yc_direction'] = treasury_diff[['1m','2y','10y','30y']].mean(axis=1)
treasury_diff['level_class'] = ['Bull' if x <= 0 else 'Bear' for x in treasury_diff['total_yc_direction']]
treasury_diff['front_class'] = ['Flattening' if x <= 0 else 'Steepening' for x in treasury_diff['front_end']]
treasury_diff['belly_class'] = ['Flattening' if x <= 0 else 'Steepening' for x in treasury_diff['belly']]
treasury_diff['back_class'] = ['Flattening' if x <= 0 else 'Steepening' for x in treasury_diff['back_end']]

### REGIMES ###
treasury_diff['front_regime'] = treasury_diff['level_class'] + ' ' + treasury_diff['front_class']
treasury_diff['belly_regime'] = treasury_diff['level_class'] + ' ' + treasury_diff['belly_class']
treasury_diff['back_regime'] = treasury_diff['level_class'] + ' ' + treasury_diff['back_class']

treasury_diff['spx_monthly_pct'] = spx_monthly.pct_change().shift(-1)

front_regime_score_map = {
    'Bear Steepening':  +1,
    'Bear Flattening':  +0.9,
    'Bull Flattening':  +0.8,
    'Bull Steepening':  +0.7
}
belly_regime_score_map = {
    'Bear Steepening':  +1,
    'Bull Flattening':  +0.9,
    'Bear Flattening':  +0.8,
    'Bull Steepening':  +0.7
}
back_regime_score_map = {
    'Bear Steepening':  +1,
    'Bull Flattening':  +0.9,
    'Bear Flattening':  +0.8,
    'Bull Steepening':  +0.7
}

bear_steepening = treasury_diff[treasury_diff['front_regime'] == 'Bear Steepening']['spx_monthly_pct'].mean()
bear_flattening = treasury_diff[treasury_diff['front_regime'] == 'Bear Flattening']['spx_monthly_pct'].mean()
bull_flattening = treasury_diff[treasury_diff['front_regime'] == 'Bull Flattening']['spx_monthly_pct'].mean()
bull_steepening = treasury_diff[treasury_diff['front_regime'] == 'Bull Steepening']['spx_monthly_pct'].mean()

bear_steepening = treasury_diff[treasury_diff['belly_regime'] == 'Bear Steepening']['spx_monthly_pct'].mean()
bull_flattening = treasury_diff[treasury_diff['belly_regime'] == 'Bull Flattening']['spx_monthly_pct'].mean()
bear_flattening = treasury_diff[treasury_diff['belly_regime'] == 'Bear Flattening']['spx_monthly_pct'].mean()
bull_steepening = treasury_diff[treasury_diff['belly_regime'] == 'Bull Steepening']['spx_monthly_pct'].mean()

bear_steepening = treasury_diff[treasury_diff['back_regime'] == 'Bear Steepening']['spx_monthly_pct'].mean()
bull_flattening = treasury_diff[treasury_diff['back_regime'] == 'Bull Flattening']['spx_monthly_pct'].mean()
bear_flattening = treasury_diff[treasury_diff['back_regime'] == 'Bear Flattening']['spx_monthly_pct'].mean()
bull_steepening = treasury_diff[treasury_diff['back_regime'] == 'Bull Steepening']['spx_monthly_pct'].mean()

treasury_diff['front_score'] = treasury_diff['front_regime'].map(front_regime_score_map)
treasury_diff['belly_score'] = treasury_diff['belly_regime'].map(belly_regime_score_map)
treasury_diff['back_score']  = treasury_diff['back_regime'].map(back_regime_score_map)

treasury_diff['total_score'] = (
    treasury_diff['belly_score']
)

treasury_diff['bt_returns'] = treasury_diff['spx_monthly_pct'] * treasury_diff['total_score']
treasury_diff = treasury_diff['2000-01-01':]

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

equities_yc_return_metrics = return_metrics(
    treasury_diff[['bt_returns','spx_monthly_pct']],
    treasury_diff[['spx_monthly_pct']],
    12
)
equities_yc_return_metrics['Return/Risk']

def plot_colorcoded_regime():
    regime_merge = merge_dfs([
        pd.DataFrame(treasury_diff['belly_regime']),
        spx_monthly,
        pd.DataFrame(treasury_diff['bt_returns'])
    ]).dropna()
    color_coded_regime_plot(regime_merge,
                            y_col='spx',
                            regime_col='belly_regime',
                            title="Level by Macro Regime")
    stats_df = calculate_regime_statistics(regime_merge,
                                           regime_col_name='belly_regime',
                                           return_cols = ['bt_returns'])
    plot_streamlit_regime_statistics(stats_df)

def equity_yc_results():
    streamlit_plot(df=(1 + treasury_diff[['bt_returns', 'spx_monthly_pct']]).cumprod() - 1,
                   columns_array=['bt_returns', 'spx_monthly_pct'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(equities_yc_return_metrics)




