### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
DATA_DIR = os.getenv('DATA_DIR', 'data')

def transform_series(series, tcode):
    if tcode == 1:  # No transformation
        return series.diff(12).diff(3)
    elif tcode == 2:  # First difference
        return series.diff(12).diff(3)
    elif tcode == 3:  # Second difference
        return series.diff(12).diff(3)
    elif tcode == 4:  # Log
        return series.diff(12).diff(3)
    elif tcode == 5:  # Log first difference
        return series.diff(12).diff(3)
    elif tcode == 6:  # Log second difference
        return series.diff(12).diff(3)
    elif tcode == 7:  # Demeaned
        return series.diff(12).diff(3)
    else:
        raise ValueError(f"Unknown tcode {tcode}")

def transform_fredmd(df, tcodes):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        transformed = transform_series(df[col], tcodes[col])
        out[col] = transformed
    return out.dropna()

# growth_vars_lag1 = pdr.DataReader([
#     'USALOLITOAASTSAM','ICSA','RPI', 'W875RX1', 'DPCERA3M086SBEA',
#     'INDPRO','IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD','IPNCONGD',
#     'IPBUSEQ', 'IPMAT', 'IPDMAT','IPNMAT','IPMANSICS', 'IPB51222S',
#     'IPFUELS','CUMFNS',  'HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS',
#     'HOUSTW','PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS','PERMITW',
#     'BUSLOANS', 'REALLN'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# growth_vars_lag1.index = growth_vars_lag1.index + pd.DateOffset(months=1)
# growth_vars_lag1 = growth_vars_lag1.resample('ME').last()
#
# growth_vars_lag2 = pdr.DataReader([
#     'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN','UEMPLT5',
#     'UEMP5TO14', 'UEMP15OV', 'UEMP15T26','UEMP27OV',
#     'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS','MANEMP',
#     'DMANEMP', 'NDMANEMP','SRVPRD', 'USTPU', 'USWTRADE',
#     'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN',
#     'AWHMAN'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# growth_vars_lag2.index = growth_vars_lag2.index + pd.DateOffset(months=2)
# growth_vars_lag2 = growth_vars_lag2.resample('ME').last()
#
# growth_lagged_features = merge_dfs([growth_vars_lag1, growth_vars_lag2]).dropna()
# with open(Path(DATA_DIR) / 'growth_lagged_features.pkl', 'wb') as file:
#     pickle.dump(growth_lagged_features, file)
#
# inflation_vars_lag1 = pdr.DataReader([
#     'M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES',
#     'FEDFUNDS','TB3MS', 'TB6MS','GS1', 'GS5',
#     'GS10', 'AAA', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM','T5YFFM', 'T10YFFM',
#     'AAAFFM', 'BAAFFM','WPSFD49207', 'WPSFD49502','WPSID61','WPSID62',
#     'PPICMM', 'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL','CPIMEDSL', 'CUSR0000SAC',
#     'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL','CUSR0000SA0L2', 'CUSR0000SA0L5',
#     'PCEPI','DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# inflation_vars_lag1.index = inflation_vars_lag1.index + pd.DateOffset(months=1)
# inflation_vars_lag1 = inflation_vars_lag1.resample('ME').last()
#
# inflation_vars_lag2 = pdr.DataReader([
#     'CES0600000008',
#     'CES2000000008',
#     'CES3000000008'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# inflation_vars_lag2.index = inflation_vars_lag2.index + pd.DateOffset(months=2)
# inflation_vars_lag2 = inflation_vars_lag2.resample('ME').last()
#
# inflation_lagged_features = merge_dfs([inflation_vars_lag1, inflation_vars_lag2]).dropna()
# with open(Path(DATA_DIR) / 'inflation_lagged_features.pkl', 'wb') as file:
#     pickle.dump(inflation_lagged_features, file)

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###


### GROWTH TARGET ###
growth = pdr.DataReader('PCEC96',
                        'fred',
                        '1949-12-31',
                        '2025-12-31')
growth.index = pd.to_datetime(growth.index) + pd.DateOffset(months=1)
growth = growth.resample('ME').last()

### INFLATION TARGET ###
inflation = pdr.DataReader(
    'CPIAUCSL',
    'fred',
    '1949-12-31',
    '2025-12-31')
inflation.index = pd.to_datetime(inflation.index) + pd.DateOffset(months=1)
inflation = inflation.resample('ME').last()

with open(Path(DATA_DIR) / 'growth_lagged_features.pkl', 'rb') as file:
    growth_lagged_features = pd.read_pickle(file)

with open(Path(DATA_DIR) / 'inflation_lagged_features.pkl', 'rb') as file:
    inflation_lagged_features = pd.read_pickle(file)

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

with open(Path(DATA_DIR) / 'sector_industry_returns.xlsx', 'rb') as file:
    sector_excel = pd.read_excel(file, sheet_name='sector')
    sector_excel.index = pd.to_datetime(sector_excel['Dates'].values)
    sector_excel.drop('Dates', axis=1, inplace=True)
    sector_excel = sector_excel[::-1].dropna()

    industry_excel = pd.read_excel(file, sheet_name='industry')
    industry_excel.index = pd.to_datetime(industry_excel['Dates'].values)
    industry_excel.drop('Dates', axis=1, inplace=True)
    industry_excel = industry_excel[::-1].dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### GROWTH DATA ###
growth_roc_2 = growth.pct_change(12).diff(3)
growth_roc_2.columns = ['growth_roc_2']
lagged_growth_roc_2 = growth_roc_2.shift(-1)
lagged_growth_roc_2.columns = ['growth_roc_2_lag']

### INFLATION DATA ###
inflation_roc_2 = inflation.pct_change(12).diff(3)
inflation_roc_2.columns = ['inflation_roc_2']
lagged_inflation_roc_2 = inflation_roc_2.shift(-1)
lagged_inflation_roc_2.columns = ['inflation_roc_2_lag']

### MERGE TARGETS AND FEATURES ###
target_feature_merge = merge_dfs([
    growth_lagged_features.diff(12).diff(3),
    inflation_lagged_features.diff(12).diff(3),
    lagged_growth_roc_2,
    lagged_inflation_roc_2,
]).dropna()


### CREATE FEATURES AND TARGET ###
growth_vars = growth_lagged_features.columns
inflation_vars = inflation_lagged_features.columns

growth_features_targets = target_feature_merge[growth_vars]
growth_features_targets['growth_roc_2_lag'] = target_feature_merge['growth_roc_2_lag']
inflation_features_targets = target_feature_merge[inflation_vars]
inflation_features_targets['inflation_roc_2_lag'] = target_feature_merge['inflation_roc_2_lag']

inflation_features_targets = inflation_features_targets.dropna()
growth_features_targets = growth_features_targets.dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

lookback_window = 12
rolling_pred = []
rolling_dates = []
rolling_actual = []

for row in range(lookback_window + 1, len(growth_features_targets)+1):
    # Extract rolling window
    growth_subset = growth_features_targets[row - lookback_window - 1:row]
    target_growth_subset = growth_subset[:lookback_window]
    current_growth_subset = growth_subset[lookback_window:]

    # Split into features and target
    X_window = target_growth_subset.iloc[:, :-1]  # Features in window
    y_window = target_growth_subset.iloc[:, -1]  # Targets in window
    X_current = current_growth_subset.iloc[:, :-1]  # Current features
    y_current = current_growth_subset.iloc[:, -1]  # Current features

    # Standardize features in the window
    scaler = StandardScaler()
    X_window_scaled = scaler.fit_transform(X_window)
    X_current_scaled = scaler.transform(X_current)

    # PCA on standardized features
    pca = PCA(n_components=10)
    factors_window = pca.fit_transform(X_window_scaled)
    factors_current = pca.transform(X_current_scaled)

    # Regression of targets on window factors
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    model_gb = LinearRegression()
    model_gb.fit(factors_window, y_window)
    current_pred = model_gb.predict(factors_current)

    rolling_pred.append(current_pred[0])
    rolling_actual.append(y_current[0])
    rolling_dates.append(target_feature_merge.index[row-1])

df = pd.DataFrame({
    'prediction': rolling_pred,
    'actual': target_feature_merge['growth_roc_2_lag'][lookback_window:210],
}, index=rolling_dates)
df_valid = df.dropna(subset=['prediction', 'actual'])


wins = (np.sign(df_valid['prediction']) == np.sign(df_valid['actual'])).astype(int)
wins.sum() / len(wins)

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- PROMETHEUS REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

lookback_window = 12
rolling_pred = []
rolling_dates = []
rolling_actual = []

for row in range(lookback_window + 1, len(inflation_features_targets)+1):
    # Extract rolling window
    inflation_subset = inflation_features_targets[row - lookback_window - 1:row]
    target_inflation_subset = inflation_subset[:lookback_window]
    current_inflation_subset = inflation_subset[lookback_window:]

    # Split into features and target
    X_window = target_inflation_subset.iloc[:, :-1]  # Features in window
    y_window = target_inflation_subset.iloc[:, -1]  # Targets in window
    X_current = current_inflation_subset.iloc[:, :-1]  # Current features
    y_current = current_inflation_subset.iloc[:, -1]  # Current features

    # Standardize features in the window
    scaler = StandardScaler()
    X_window_scaled = scaler.fit_transform(X_window)
    X_current_scaled = scaler.transform(X_current)

    # PCA on standardized features
    pca = PCA(n_components=10)
    factors_window = pca.fit_transform(X_window_scaled)
    factors_current = pca.transform(X_current_scaled)

    # Regression of targets on window factors
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    model_gb = LinearRegression()
    model_gb.fit(factors_window, y_window)
    current_pred = model_gb.predict(factors_current)

    rolling_pred.append(current_pred[0])
    rolling_actual.append(y_current[0])
    rolling_dates.append(target_feature_merge.index[row-1])

df = pd.DataFrame({
    'prediction': rolling_pred,
    'actual': target_feature_merge['inflation_roc_2_lag'][lookback_window:210],
}, index=rolling_dates)
df_valid = df.dropna(subset=['prediction', 'actual'])


wins = (np.sign(df_valid['prediction']) == np.sign(df_valid['actual'])).astype(int)
wins.sum() / len(wins)

