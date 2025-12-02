### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

### BONDS ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

### BCOM ###
with open(Path(DATA_DIR) / '^BCOM.csv', 'rb') as file:
    bcom = pd.read_csv(file)
bcom.index = pd.to_datetime(bcom['Date']).values
bcom.drop('Date', axis=1, inplace=True)
bcom_monthly = pd.DataFrame(bcom['Close']).resample('ME').last()
bcom_monthly.columns = ['bcom']

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

### LIQUIDITY ###
with open(Path(DATA_DIR) / 'treasury.pkl', 'rb') as file:
    treasury = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'reserves.pkl', 'rb') as file:
    reserves = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'tga.pkl', 'rb') as file:
    tga = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'rrp_volume.pkl', 'rb') as file:
    rrp_volume = pd.read_pickle(file)

### POSITIONING DATA ###
with open(Path(DATA_DIR) / 'spx_positioning_df.pkl', 'rb') as file:
    spx_positioning_df = pd.read_pickle(file)[['dealer_spread', 'asset_mgr_spread', 'lev_funds_spread']]
    spx_positioning_df.index = pd.to_datetime(spx_positioning_df.index)
with open(Path(DATA_DIR) / 'emini_spx_positioning_df.pkl', 'rb') as file:
    emini_spx_positioning_df = pd.read_pickle(file)[['dealer_spread', 'asset_mgr_spread', 'lev_funds_spread']]
    emini_spx_positioning_df.index = pd.to_datetime(emini_spx_positioning_df.index)
with open(Path(DATA_DIR) / 'vix_positioning_df.pkl', 'rb') as file:
    vix_positioning_df = pd.read_pickle(file)[['dealer_spread', 'asset_mgr_spread', 'lev_funds_spread']]
    vix_positioning_df.index = pd.to_datetime(vix_positioning_df.index)

treasury_merge = merge_dfs([treasury_1m,treasury_2y, treasury_5y, treasury_10y,treasury_30y]).dropna()
treasury_merge.index = pd.to_datetime(treasury_merge.index).values
treasury_merge.columns = ['1m','2y','5y','10y','30y']
treasury_monthly_df = treasury_merge.resample('ME').last()


# Color mapping for regimes (customize as desired)
regime_colors = {
    "Goldilocks": "#28a745",  # Green
    "Reflation": "#90ee90",  # Super light green
    "Stagflation": "#ffc107",  # Yellow
    "Deflation": "#dc3545"  # Red
}

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_grid_nowcast():
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
            'actual': target_feature_merge[target_col_name][lookback_window:len(feature_target_df) + 1],
        }, index=rolling_dates)

        wins = (np.sign(df['prediction']) == np.sign(df['actual'])).astype(int)
        print(wins.sum() / len(wins))

        return (df)


    lin_pca_growth_results = linear_regression_pca_predictor(
        feature_target_df=growth_features_targets,
        feature_col_name_array=growth_features_targets.columns[:-1],
        target_col_name=growth_features_targets.columns[-1],
        lookback_window=12,
        pca_num_components=10)

    lin_pca_inflation_results = linear_regression_pca_predictor(
        feature_target_df=inflation_features_targets,
        feature_col_name_array=inflation_features_targets.columns[:-1],
        target_col_name=inflation_features_targets.columns[-1],
        lookback_window=12,
        pca_num_components=10)


    growth_prediction = lin_pca_growth_results['prediction'][-1]
    inflation_prediction = lin_pca_inflation_results['prediction'][-1]

    if growth_prediction > 0 and inflation_prediction < 0:
        upcoming_grid_regime = 'Goldilocks'
    elif growth_prediction > 0 and inflation_prediction > 0:
        upcoming_grid_regime = 'Reflation'
    elif growth_prediction < 0 and inflation_prediction > 0:
        upcoming_grid_regime = 'Stagflation'
    elif growth_prediction < 0 and inflation_prediction < 0:
        upcoming_grid_regime = 'Deflation'

    regime_color = regime_colors.get(upcoming_grid_regime, "gray")


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Growth Prediction**")
        st.markdown(f"<span style='font-size:1.5em;font-weight:bold;'>{growth_prediction:+.2%}</span>",
                    unsafe_allow_html=True)
        st.caption("3-Month Change of YoY Real PCE")
    with col2:
        st.markdown("**Inflation**")
        st.markdown(f"<span style='font-size:1.5em;font-weight:bold;'>{inflation_prediction:+.2%}</span>",
                    unsafe_allow_html=True)

        st.caption("3-Month Change of YoY CPI")
    with col3:
        st.markdown("**Quad Regime**")
        st.markdown(
            f"<span style='background-color:{regime_color};color:white;padding:0.25em 0.75em;border-radius:0.3em;font-weight:bold;font-size:1.2em'>{upcoming_grid_regime}</span>",
            unsafe_allow_html=True
        )
        if upcoming_grid_regime == 'Goldilocks':
            st.caption("Growth + Inflation -")
        elif upcoming_grid_regime == 'Reflation':
            st.caption("Growth + Inflation +")
        elif upcoming_grid_regime == 'Deflation':
            st.caption("Growth - Inflation -")
        elif upcoming_grid_regime == 'Stagflation':
            st.caption("Growth - Inflation +")


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_flowcluster_nowcast():


    ### ---------------------------------------------------------------------------------------------------------- ###
    ### -------------------------------------- FLOWCLUSTER LIQUIDITY REGIME -------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    ### AGGREGATE DATA AND RESAMPLE ###
    liquidity_df = merge_dfs([treasury, reserves, tga, rrp_volume]).dropna()
    liquidity_df.index = liquidity_df.index.values
    liquidity_df.columns = ['treasury', 'reserves', 'tga', 'onrrp']
    liquidity_df = liquidity_df.resample('ME').last()

    spx_positioning_diff = spx_positioning_df.resample('ME').mean().dropna()
    spx_positioning_diff.columns = ['spx_dealer', 'spx_asset_mgr', 'spx_lev_funds']

    flow_cluster_df = merge_dfs([
        liquidity_df['treasury'].diff(),
        liquidity_df['treasury'].diff(12).diff(3),
        spx_positioning_diff['spx_lev_funds'].diff(),
        spx_positioning_diff['spx_lev_funds'].diff(12).diff(3),
        spx_monthly.pct_change().shift(-1),
        bonds_monthly.pct_change().shift(-1)
    ]).dropna()

    flow_cluster_df.columns = [
        'liquidity_1_roc',
        'liquidity_2_roc',
        'positioning_1_roc',
        'positioning_2_roc',
        'spx',
        'bonds']

    def regime_label(row):
        if row['liquidity_2_roc'] > 0 and row['positioning_2_roc'] > 0:
            return 'Goldilocks'
        elif row['liquidity_2_roc'] > 0 and row['positioning_2_roc'] < 0:
            return 'Reflation'
        elif row['liquidity_2_roc'] < 0 and row['positioning_2_roc'] < 0:
            return 'Stagflation'
        elif row['liquidity_2_roc'] < 0 and row['positioning_2_roc'] > 0:
            return 'Deflation'
        return np.nan

    flow_cluster_df['regime_label'] = flow_cluster_df.apply(regime_label, axis=1)

    liquidity_indicator = flow_cluster_df['liquidity_2_roc'][-1]
    positioning_indicator = flow_cluster_df['positioning_2_roc'][-1]

    if liquidity_indicator > 0 and positioning_indicator > 0:
        upcoming_grid_regime = 'Goldilocks'
    elif liquidity_indicator > 0 and positioning_indicator < 0:
        upcoming_grid_regime = 'Reflation'
    elif liquidity_indicator < 0 and positioning_indicator < 0:
        upcoming_grid_regime = 'Stagflation'
    elif liquidity_indicator < 0 and positioning_indicator > 0:
        upcoming_grid_regime = 'Deflation'

    regime_color = regime_colors.get(upcoming_grid_regime, "gray")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Liquidity Indicator**")
        st.markdown(f"<span style='font-size:1.5em;font-weight:bold;'>{liquidity_indicator:+.2%}</span>",
                    unsafe_allow_html=True)
        st.caption("3-Month Change of YoY QE Treasuries")
    with col2:
        st.markdown("**Positioning Indicator**")
        st.markdown(f"<span style='font-size:1.5em;font-weight:bold;'>{positioning_indicator:+.2%}</span>",
                    unsafe_allow_html=True)

        st.caption("3-Month Change of YoY SPX Positioning")
    with col3:
        st.markdown("**Quad Regime**")
        st.markdown(
            f"<span style='background-color:{regime_color};color:white;padding:0.25em 0.75em;border-radius:0.3em;font-weight:bold;font-size:1.2em'>{upcoming_grid_regime}</span>",
            unsafe_allow_html=True
        )
        if upcoming_grid_regime == 'Goldilocks':
            st.caption("Liquidity + Positioning +")
        elif upcoming_grid_regime == 'Reflation':
            st.caption("Liquidity + Positioning -")
        elif upcoming_grid_regime == 'Deflation':
            st.caption("Liquidity - Positioning -")
        elif upcoming_grid_regime == 'Stagflation':
            st.caption("Liquidity - Positioning +")



### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_crossasset_nowcast():
    print('hi')


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_yc_nowcast():
    print('hi')




