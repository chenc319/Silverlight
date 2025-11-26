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
#     'USALOLITOAASTSAM','ICSA',
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
# growth_vars_lag3 = pdr.DataReader([
#     'RPI', 'W875RX1', 'DPCERA3M086SBEA',
#     'INDPRO','IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD','IPNCONGD',
#     'IPBUSEQ', 'IPMAT', 'IPDMAT','IPNMAT','IPMANSICS', 'IPB51222S',
#     'IPFUELS','CUMFNS',  'HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS',
#     'HOUSTW','PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS','PERMITW'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# growth_vars_lag3.index = growth_vars_lag3.index + pd.DateOffset(months=3)
# growth_vars_lag3 = growth_vars_lag3.resample('ME').last()
#
# growth_lagged_features = merge_dfs([
#     growth_vars_lag1,
#     growth_vars_lag2,
#     growth_vars_lag3,
# ]).dropna()
# with open(Path(DATA_DIR) / 'growth_lagged_features.pkl', 'wb') as file:
#     pickle.dump(growth_lagged_features, file)

# inflation_vars_lag1 = pdr.DataReader([
#     'M1SL', 'M2SL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES',
#     'FEDFUNDS','TB3MS', 'TB6MS','GS1', 'GS5',
#     'GS10', 'AAA', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM','T5YFFM', 'T10YFFM',
#     'AAAFFM', 'BAAFFM'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# inflation_vars_lag1.index = inflation_vars_lag1.index + pd.DateOffset(months=1)
# inflation_vars_lag1 = inflation_vars_lag1.resample('ME').last()
#
# inflation_vars_lag2 = pdr.DataReader([
#     'M2REAL',
#     'CES0600000008',
#     'CES2000000008',
#     'CES3000000008','WPSFD49207', 'WPSFD49502','WPSID61','WPSID62',
#     'PPICMM', 'CPIAPPSL', 'CPITRNSL','CPIMEDSL', 'CUSR0000SAC',
#     'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL','CUSR0000SA0L2', 'CUSR0000SA0L5'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# inflation_vars_lag2.index = inflation_vars_lag2.index + pd.DateOffset(months=2)
# inflation_vars_lag2 = inflation_vars_lag2.resample('ME').last()
#
# inflation_vars_lag3 = pdr.DataReader([
#     'PCEPI','DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA'
# ],
#     'fred',
#     '1949-12-31',
#     '2025-12-31'
# )
# inflation_vars_lag3.index = inflation_vars_lag3.index + pd.DateOffset(months=3)
# inflation_vars_lag3 = inflation_vars_lag3.resample('ME').last()
#
# inflation_lagged_features = merge_dfs([
#     inflation_vars_lag1,
#     inflation_vars_lag2,
#     inflation_vars_lag3,
# ]).dropna()
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
growth.index = pd.to_datetime(growth.index) + pd.DateOffset(months=3)
growth = growth.resample('ME').last()

### INFLATION TARGET ###
inflation = pdr.DataReader(
    'CPIAUCSL',
    'fred',
    '1949-12-31',
    '2025-12-31')
inflation.index = pd.to_datetime(inflation.index) + pd.DateOffset(months=2)
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

def linear_regression_pca_predictor(
        feature_target_df,
        feature_col_name_array,
        target_col_name,
        lookback_window,
        pca_num_components):

    rolling_pred = []
    rolling_dates = []
    rolling_actual = []

    for row in range(lookback_window + 1, len(feature_target_df) + 1):
        ### EXTRACT SUBSETS ###
        inflation_subset = feature_target_df[row - lookback_window - 1:row]
        target_inflation_subset = inflation_subset[:lookback_window]
        current_inflation_subset = inflation_subset[lookback_window:]

        ### SPLITS INTO FEATURES AND TARGETS ###
        X_window = target_inflation_subset.loc[:, feature_col_name_array]  # Features in window
        y_window = target_inflation_subset.loc[:, target_col_name]  # Targets in window
        X_current = current_inflation_subset.loc[:, feature_col_name_array]  # Current features
        y_current = current_inflation_subset.loc[:, target_col_name]  # Current features

        ### STANDARDIZE AND PCA ###
        scaler = StandardScaler()
        X_window_scaled = scaler.fit_transform(X_window)
        X_current_scaled = scaler.transform(X_current)
        pca = PCA(n_components=pca_num_components)
        factors_window = pca.fit_transform(X_window_scaled)
        factors_current = pca.transform(X_current_scaled)

        ### LINEAR REGRESSION ###
        model = LinearRegression()
        model.fit(factors_window, y_window)
        current_pred = model.predict(factors_current)

        ### RESULTS ###
        rolling_pred.append(current_pred[0])
        rolling_actual.append(y_current[0])
        rolling_dates.append(target_feature_merge.index[row - 1])

    df = pd.DataFrame({
        'prediction': rolling_pred,
        'actual': target_feature_merge[target_col_name][lookback_window:len(feature_target_df)+1],
    }, index=rolling_dates)

    wins = (np.sign(df['prediction']) == np.sign(df['actual'])).astype(int)
    print(wins.sum() / len(wins))

    return(df)

def logistic_regression_pca_sign_predictor(
        feature_target_df,
        feature_col_name_array,
        target_col_name,
        lookback_window,
        pca_num_components):

    rolling_pred_sign = []
    rolling_proba_pos = []
    rolling_actual_sign = []
    rolling_low_conf_flag = []
    rolling_dates = []

    for row in range(lookback_window + 1, len(feature_target_df) + 1):
        ### EXTRACT SUBSETS ###
        window_subset = feature_target_df[row - lookback_window - 1:row]
        target_window_subset = window_subset[:lookback_window]
        current_subset = window_subset[lookback_window:]

        ### SPLIT INTO FEATURES AND TARGET ###
        X_window = target_window_subset.loc[:, feature_col_name_array]
        y_window = target_window_subset.loc[:, target_col_name]
        X_current = current_subset.loc[:, feature_col_name_array]
        y_current = current_subset.loc[:, target_col_name]

        # Numeric current value and sign
        y_current_val = y_current.iloc[0]
        actual_sign = 1 if y_current_val > 0 else -1

        ### BUILD CLASS LABELS: 1 = y > 0, 0 = y <= 0 ###
        y_window_class = (y_window > 0).astype(int)

        ### STANDARDIZE AND PCA ###
        scaler = StandardScaler()
        X_window_scaled = scaler.fit_transform(X_window)
        X_current_scaled = scaler.transform(X_current)

        pca = PCA(n_components=pca_num_components)
        factors_window = pca.fit_transform(X_window_scaled)
        factors_current = pca.transform(X_current_scaled)

        ### LOGISTIC REGRESSION OR FALLBACK ###
        if y_window_class.nunique() >= 2:
            clf = LogisticRegression()
            clf.fit(factors_window, y_window_class)
            proba = clf.predict_proba(factors_current)[0]   # [P(class=0), P(class=1)]
            proba_pos = proba[1]
            pred_class = clf.predict(factors_current)[0]
        else:
            only_class = int(y_window_class.iloc[0])
            pred_class = only_class
            proba_pos = 1.0 if only_class == 1 else 0.0

        pred_sign = 1 if pred_class == 1 else -1

        ### LOW PROBABILITY FLAG ###
        low_conf_flag = (proba_pos >= 0.45) and (proba_pos <= 0.55)

        ### RESULTS ###
        rolling_pred_sign.append(pred_sign)
        rolling_proba_pos.append(proba_pos)
        rolling_actual_sign.append(actual_sign)
        rolling_low_conf_flag.append(low_conf_flag)
        rolling_dates.append(feature_target_df.index[row - 1])

    df = pd.DataFrame({
        'pred_sign': rolling_pred_sign,
        'proba_pos': rolling_proba_pos,
        'actual_sign': rolling_actual_sign,
        'proba_low_confidence': rolling_low_conf_flag,
    }, index=rolling_dates)

    wins = (df['pred_sign'] == df['actual_sign']).astype(int)
    print(wins.sum() / len(wins))

    return df

def combine_signals_regression_logistic(
        df_reg,
        df_cls,
        low_conf_lower=0.45,
        low_conf_upper=0.55):
    """
    Combines regression and logistic sign signals into an ensemble sign.

    Parameters
    ----------
    df_reg : DataFrame
        Must have columns: 'prediction', 'actual'.
        Index should be datetime (or compatible with df_cls).
    df_cls : DataFrame
        Must have columns: 'pred_sign', 'proba_pos', 'actual_sign'.
        Index should be datetime, overlapping df_reg.
    low_conf_lower : float
        Lower bound of low-confidence band for proba_pos.
    low_conf_upper : float
        Upper bound of low-confidence band for proba_pos.

    Returns
    -------
    df_ens : DataFrame
        Columns:
            'reg_pred'        : regression prediction (magnitude)
            'reg_sign'        : sign of regression prediction
            'cls_sign'        : sign from logistic classifier
            'proba_pos'       : P(y > 0) from classifier
            'combined_sign'   : ensemble sign (+1, -1, or 0 for neutral)
            'actual_sign'     : realized sign from df_reg['actual']
        Index: common dates.
    """

    # Align on common index
    common_index = df_reg.index.intersection(df_cls.index)
    df_reg_aligned = df_reg.loc[common_index].copy()
    df_cls_aligned = df_cls.loc[common_index].copy()

    # Regression sign
    reg_pred = df_reg_aligned['prediction']
    reg_sign = np.sign(reg_pred).replace(0, 0)  # 0 if exactly 0, otherwise ±1
    reg_sign = reg_sign.where(reg_sign != 0, 1)  # if 0, arbitrarily treat as +1

    # Classifier pieces
    cls_sign = df_cls_aligned['pred_sign']
    proba_pos = df_cls_aligned['proba_pos']

    # Actual sign (from regression actual to keep it consistent)
    actual = df_reg_aligned['actual']
    actual_sign = np.sign(actual).replace(0, 0)
    actual_sign = actual_sign.where(actual_sign != 0, 1)

    # Low-confidence flag from probabilities
    proba_low_conf = (proba_pos >= low_conf_lower) & (proba_pos <= low_conf_upper)

    combined_sign_list = []

    for dt in common_index:
        r_sign = reg_sign.loc[dt]
        c_sign = cls_sign.loc[dt]
        p_pos = proba_pos.loc[dt]
        low_conf = proba_low_conf.loc[dt]

        # Ensemble rules:
        # 1) If both agree on sign, use that sign.
        if r_sign == c_sign:
            combined = r_sign

        else:
            # 2) If they disagree:
            #    - if classifier is high-confidence, follow regression
            #    - else neutral
            if not low_conf:
                # High conviction classifier (outside [low_conf_lower, low_conf_upper])
                combined = 0
            else:
                combined = 0  # neutral / no-trade

        combined_sign_list.append(combined)

    df_ens = pd.DataFrame({
        'reg_pred': reg_pred.loc[common_index],
        'reg_sign': reg_sign.loc[common_index],
        'cls_sign': cls_sign.loc[common_index],
        'proba_pos': proba_pos.loc[common_index],
        'combined_sign': combined_sign_list,
        'actual_sign': actual_sign.loc[common_index],
    }, index=common_index)

    # Win ratios
    reg_wins = (df_ens['reg_sign'] == df_ens['actual_sign']).astype(int)
    cls_wins = (df_ens['cls_sign'] == df_ens['actual_sign']).astype(int)

    mask_trades = df_ens['combined_sign'] != 0
    combined_wins = (df_ens.loc[mask_trades, 'combined_sign'] ==
                     df_ens.loc[mask_trades, 'actual_sign']).astype(int)

    reg_hit_rate = reg_wins.mean()
    cls_hit_rate = cls_wins.mean()
    combined_hit_rate = combined_wins.mean() if mask_trades.any() else np.nan
    trade_frac = mask_trades.mean()

    print("Regression sign hit rate:", reg_hit_rate)
    print("Classifier sign hit rate:", cls_hit_rate)
    print("Combined sign hit rate (conditional on taking a position):", combined_hit_rate)
    print("Fraction of periods traded by combined signal:", trade_frac)

    return df_ens



lin_pca_growth_results = linear_regression_pca_predictor(
    feature_target_df = growth_features_targets,
    feature_col_name_array = growth_features_targets.columns[:-1],
    target_col_name = growth_features_targets.columns[-1],
    lookback_window = 12,
    pca_num_components = 10)

log_pca_growth_results = logistic_regression_pca_sign_predictor(
    feature_target_df = growth_features_targets,
    feature_col_name_array = growth_features_targets.columns[:-1],
    target_col_name = growth_features_targets.columns[-1],
    lookback_window = 60,
    pca_num_components = 30)

lin_pca_inflation_results = linear_regression_pca_predictor(
    feature_target_df = inflation_features_targets,
    feature_col_name_array = inflation_features_targets.columns[:-1],
    target_col_name = inflation_features_targets.columns[-1],
    lookback_window = 12,
    pca_num_components = 10)

log_pca_inflation_results = logistic_regression_pca_sign_predictor(
    feature_target_df = inflation_features_targets,
    feature_col_name_array = inflation_features_targets.columns[:-1],
    target_col_name = inflation_features_targets.columns[-1],
    lookback_window = 60,
    pca_num_components = 20)


combine_signals_regression_logistic(
    df_reg=lin_pca_growth_results,
    df_cls=log_pca_growth_results,
    low_conf_lower=0.05,
    low_conf_upper=0.95,
)

combine_signals_regression_logistic(
    df_reg=lin_pca_inflation_results,
    df_cls=log_pca_inflation_results,
    low_conf_lower=0.05,
    low_conf_upper=0.95,
)

