### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')
end = pd.to_datetime('today')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
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
                            end)
    growth.index = pd.to_datetime(growth.index) + pd.DateOffset(months=1)
    growth = growth.resample('ME').last()

    ### INFLATION TARGET ###
    inflation = pdr.DataReader(
        'CPIAUCSL',
        'fred',
        '1949-12-31',
        end)
    inflation.index = pd.to_datetime(inflation.index) + pd.DateOffset(months=1)
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
        st.markdown(
            f"<span style='font-size:1.5em;font-weight:bold;'>"
            f"{liquidity_indicator / 1e9:+.2f} Bn"
            f"</span>",
            unsafe_allow_html=True
        )
        st.caption("3-Month Change in YoY QE Treasuries (USD billions)")

    with col2:
        st.markdown("**Positioning Indicator**")
        st.markdown(
            f"<span style='font-size:1.5em;font-weight:bold;'>"
            f"{positioning_indicator:+,.0f}"
            f"</span>",
            unsafe_allow_html=True
        )
        st.caption("3-Month Change in YoY SPX Positioning (number of contracts)")

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
    ### MERGE DFS ###
    cross_asset_monthly_merge = merge_dfs([
        spx_monthly,
        bonds_monthly,
        bcom_monthly,
        dxy_monthly
    ]).dropna()

    # Monthly returns
    df = cross_asset_monthly_merge.pct_change().dropna()
    df.columns = ['spx_ret', 'bonds_ret', 'bcom_ret', 'dxy_ret']

    feature_cols = ['spx_ret', 'bonds_ret', 'bcom_ret', 'dxy_ret']

    window = 36
    feat_rolling_mean = df[feature_cols].rolling(window).mean()
    feat_rolling_std = df[feature_cols].rolling(window).std()
    feat_rolling_z = ((df[feature_cols] - feat_rolling_mean) / feat_rolling_std).dropna()

    regime_archetypes = {
        "Goldilocks": np.array([
            1.0,  # spx_ret: clear winner
            0.5,  # bonds_ret: modestly positive
            -0.5,  # bcom_ret: mildly soft
            -0.5  # dxy_ret: dollar gently weaker
        ]),
        "Reflation": np.array([
            0.5,  # spx_ret: positive but choppier than Goldilocks
            -1.0,  # bonds_ret: clear underperformer (yields up)
            1.0,  # bcom_ret: big winner
            0.0  # dxy_ret: roughly mixed/flat
        ]),
        "Stagflation": np.array([
            -1.0,  # spx_ret: worst for equities
            -0.5,  # bonds_ret: mildly negative in nominal terms
            1.0,  # bcom_ret: strong relative winner
            0.5  # dxy_ret: often firm
        ]),
        "Deflation": np.array([
            -0.5,  # spx_ret: weak but not as bad as stagflation
            1.0,  # bonds_ret: strongest (duration bull)
            -1.0,  # bcom_ret: clear loser
            1.0  # dxy_ret: strong safe haven
        ])
    }

    lambda_ = 1.0

    def assign_regime_probs(z_scores_row, regime_archetypes, lambda_=1.0):
        distances = {k: np.linalg.norm(z_scores_row - v) for k, v in regime_archetypes.items()}
        exp_dists = {k: np.exp(-lambda_ * d) for k, d in distances.items()}
        regime_probs = {k: exp_dists[k] / sum(exp_dists.values()) for k in regime_archetypes}
        nearest_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
        return nearest_regime, regime_probs

    results = []
    for idx, row in feat_rolling_z.iterrows():
        regime, probs = assign_regime_probs(row.values, regime_archetypes, lambda_)
        results.append({"date": idx, "closest_regime": regime, **probs})

    regime_df = pd.DataFrame(results)
    regime_df.index = regime_df['date'].values
    regime_df.drop('date', axis=1, inplace=True)

    # Get last row
    last = regime_df.iloc[-1]
    last_regime = last['closest_regime']
    gold_prob = float(last['Goldilocks'])
    refl_prob = float(last['Reflation'])
    stag_prob = float(last['Stagflation'])
    defl_prob = float(last['Deflation'])

    # Use your existing regime_colors dict
    regime_color = regime_colors.get(last_regime, "#808080")

    # Caption based on archetype matrix signs:
    # Goldilocks:   Equities +  Bonds 0   Commodities -  Dollar 0
    # Reflation:    Equities +  Bonds -   Commodities +  Dollar 0
    # Stagflation:  Equities -  Bonds -   Commodities +  Dollar +
    # Deflation:    Equities -  Bonds +   Commodities -  Dollar +
    regime_caption_map = {
        "Goldilocks": "Equities +   Bonds 0   Commodities -   Dollar 0",
        "Reflation": "Equities +   Bonds -   Commodities +   Dollar 0",
        "Stagflation": "Equities -   Bonds -   Commodities +   Dollar +",
        "Deflation": "Equities -   Bonds +   Commodities -   Dollar +",
    }
    regime_caption = regime_caption_map.get(last_regime, "")

    # Order: col1 = Regime Probabilities, col2 = Most Likely, col3 = Quad Regime
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Regime Probabilities**")
        st.markdown(
            f"<span style='font-size:1.0em;'>"
            f"Goldilocks: <b>{gold_prob:5.1%}</b><br>"
            f"Reflation:&nbsp;&nbsp; <b>{refl_prob:5.1%}</b><br>"
            f"Stagflation: <b>{stag_prob:5.1%}</b><br>"
            f"Deflation:&nbsp;&nbsp; <b>{defl_prob:5.1%}</b>"
            f"</span>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("**Most Likely Regime Prob**")
        max_regime = max(
            [
                ("Goldilocks", gold_prob),
                ("Reflation", refl_prob),
                ("Stagflation", stag_prob),
                ("Deflation", defl_prob),
            ],
            key=lambda x: x[1],
        )
        st.markdown(
            f"<span style='font-size:1.5em;font-weight:bold;'>"
            f"{max_regime[0]}: {max_regime[1]:.1%}"
            f"</span>",
            unsafe_allow_html=True
        )
        st.caption("Last observation regime probability snapshot")

    with col3:
        st.markdown("**Quad Regime**")
        st.markdown(
            f"<span style='background-color:{regime_color};color:white;"
            f"padding:0.25em 0.75em;border-radius:0.3em;font-weight:bold;"
            f"font-size:1.2em'>{last_regime}</span>",
            unsafe_allow_html=True
        )
        st.caption(regime_caption)


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_yc_nowcast():
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

    treasury_merge = merge_dfs([treasury_1m, treasury_2y, treasury_5y, treasury_10y, treasury_30y]).dropna()
    treasury_merge.index = pd.to_datetime(treasury_merge.index).values
    treasury_merge.columns = ['1m', '2y', '5y', '10y', '30y']
    treasury_monthly_df = treasury_merge.resample('ME').last()


    ### CALCULATE SLOPES AND DIFFERENCES ###
    treasury_diff = treasury_monthly_df.diff(3).dropna()
    treasury_diff['belly'] = treasury_diff['10y'] - treasury_diff['2y']
    treasury_diff['total_yc_direction'] = treasury_diff[['2y', '5y', '10y']].mean(axis=1)
    treasury_diff['level_class'] = ['Bull' if x <= 0 else 'Bear' for x in treasury_diff['2y']]
    treasury_diff['belly_class'] = ['Flattening' if x <= 0 else 'Steepening' for x in treasury_diff['belly']]

    ### REGIMES ###
    treasury_diff['belly_regime'] = treasury_diff['level_class'] + ' ' + treasury_diff['belly_class']
    treasury_diff['spx_monthly_pct'] = spx_monthly.pct_change(1).shift(-1)
    treasury_diff = treasury_diff.dropna()

    # Get last row from treasury_diff
    last_yc = treasury_diff.iloc[-1]
    last_belly_regime = last_yc['belly_regime']  # e.g. "Bull Steepening"
    last_spx_next = float(last_yc['spx_monthly_pct'])  # next-month SPX pct

    # Map belly_regime -> macro quad regime
    belly_quad_regime_map = {
        'Bear Steepening': 'Reflation',
        'Bull Flattening': 'Goldilocks',
        'Bear Flattening': 'Deflation',
        'Bull Steepening': 'Stagflation'
    }
    mapped_quad = belly_quad_regime_map.get(last_belly_regime, "Unknown")

    # Caption for curve regime (intuitive description)
    belly_caption_map = {
        'Bear Steepening': "Yields ↑, long-end ↑↑ (growth + inflation scare)",
        'Bull Flattening': "Yields ↓, long-end ↓ less (late-cycle / disinflation)",
        'Bear Flattening': "Yields ↑, front-end ↑↑ (late-cycle tightening)",
        'Bull Steepening': "Yields ↓, front-end ↓↓ (cuts into slowdown/recession)"
    }
    belly_caption = belly_caption_map.get(last_belly_regime, "")

    # Caption for mapped macro quad (consistent with your archetypes)
    quad_caption_map = {
        "Goldilocks": "Bull Flattening",
        "Reflation": "Bear Steepening",
        "Stagflation": "Bear Flattening",
        "Deflation": "Bull Steepening",
    }
    mapped_quad_caption = quad_caption_map.get(mapped_quad, "")

    # Optional color for belly_regime (define your own)
    belly_regime_colors = {
        'Bear Steepening': "#1f77b4",
        'Bull Flattening': "#2ca02c",
        'Bear Flattening': "#d62728",
        'Bull Steepening': "#ff7f0e",
    }
    belly_color = belly_regime_colors.get(last_belly_regime, "#808080")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Yield Curve Regime**")
        st.markdown(
            f"<span style='background-color:{belly_color};color:white;"
            f"padding:0.25em 0.75em;border-radius:0.3em;font-weight:bold;"
            f"font-size:1.1em'>{last_belly_regime}</span>",
            unsafe_allow_html=True
        )
        st.caption(belly_caption)


    with col2:
        st.markdown("**Next-Month SPX (Curve-based)**")
        st.markdown(
            f"<span style='font-size:1.5em;font-weight:bold;'>"
            f"{last_spx_next:+.2%}"
            f"</span>",
            unsafe_allow_html=True
        )
        st.caption("S&P 500 1M return following this yield-curve regime")

    with col3:
        st.markdown("**Quad Regime**")
        mapped_color = regime_colors.get(mapped_quad, "#808080")
        st.markdown(
            f"<span style='background-color:{mapped_color};color:white;"
            f"padding:0.25em 0.75em;border-radius:0.3em;font-weight:bold;"
            f"font-size:1.1em'>{mapped_quad}</span>",
            unsafe_allow_html=True
        )
        st.caption(mapped_quad_caption)






