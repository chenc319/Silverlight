### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- GRID -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

spx_sectors = {
    "XLC": "Comm Services",
    "XLY": "Cons Disc",
    "XLP": "Cons Stap",
    "XLE": "Energy",
    "XLF": "Financial",
    "XLV": "Healthcare",
    "XLI": "Industrial",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Tech",
    "XLU": "utilities"
}
growth_dict = {
    'USALOLITOAASTSAM': 'cli_amplitude_adjusted',
    'INDPRO': 'industrial_production',
    'BOPGSTB': 'trade_balance_goods_and_services',
    'RSXFS': 'advanced_retail_sales_retail_trade',
    'TLMFGCONS': 'manufacturing_spending',
    'PAYEMS': 'all_employees_total_nonfarm',
    'USGOOD': 'goods_producing_employment',
    'MANEMP': 'all_employees_manufacturing',
    'CES0500000011': 'avg_earnings_all_private_employees',
    'PCEC96': 'real_personal_consumption_expenditures',
    'RRSFS': 'real_retail_food_services_sales',
    'TOTALSA': 'total_vehicle_sales'
}
inflation_dict = {
    'CPIAUCSL': 'cpi_all_items',
    'CPILFESL': 'cpi_less_food_energy',
    'CPIUFDSL': 'cpi_food',
    'CPIENGSL': 'cpi_energy',
    'CUSR0000SAH3': 'cpi_household_furnishings',
    'CPIAPPSL': 'cpi_apparel',
    'CPIMEDSL': 'cpi_medical_care',
    'CPITRNSL': 'cpi_transportation',
    'CUSR0000SAF116': 'cpi_alcohol',
    'CUSR0000SETB': 'cpi_motor_fuel',
    'CUSR0000SASLE': 'cpi_services_less_energy'
}
spx_sectors = {
    "XLC": "Comm Services",
    "XLY": "Cons Disc",
    "XLP": "Cons Stap",
    "XLE": "Energy",
    "XLF": "Financial",
    "XLV": "Healthcare",
    "XLI": "Industrial",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Tech",
    "XLU": "utilities"
}

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- GRID -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### SPX DATA ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']
spx_monthly_pct = spx_monthly.pct_change().dropna()

### BONDS DATA ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg = pd.DataFrame(agg['Close']).resample('ME').last()

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
mags_monthly_pct = each_mags_df.resample('ME').last().pct_change().dropna()

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

### SECTOR DATA ###
spx_sectors_merge = pd.DataFrame()
for each_factor in list(spx_sectors.keys()):
    with open(Path(DATA_DIR) / (each_factor + '.csv'), 'rb') as file:
        df = pd.read_csv(file)
    df.index = pd.to_datetime(df['Date']).values
    df = pd.DataFrame(df['Close'])
    df.columns = [spx_sectors[each_factor]]
    spx_sectors_merge = merge_dfs([spx_sectors_merge, df])

### GROWTH VARIABLE ###
with open(Path(DATA_DIR) / 'cli.pkl', 'rb') as file:
    cli = pd.read_pickle(file)

### INFLATION VARIABLE ###
factor_features = [
    'CPILFESL',
    'CPIUFDSL',
    'CPIENGSL',
    'CUSR0000SAH3',
    'CPIAPPSL',
    'CPIMEDSL',
    'CPITRNSL',
    'CUSR0000SAF116',
    'CUSR0000SETB',
    'CUSR0000SASLE'
]
with open(Path(DATA_DIR) / 'inflation_variables_merge.pkl', 'rb') as file:
    inflation_variables_merge = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'di_reserves.pkl', 'rb') as file:
    di_reserves = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'm2_money_supply.pkl', 'rb') as file:
    m2_money_supply = pd.read_pickle(file)
inflation_variables_merge = merge_dfs([inflation_variables_merge, di_reserves, m2_money_supply])
target_feature_df = inflation_variables_merge.pct_change(12)
target_feature_df['TOTRESNS'] = target_feature_df['TOTRESNS'] * -1
target_feature_df['M2SL'] = target_feature_df['M2SL'] * -1
target_feature_df['CPIAUCSL'] = target_feature_df['CPIAUCSL'].shift(-1)
target_feature_df = target_feature_df.dropna()

result_factor = []
window = 36
for i in range(window, len(target_feature_df)):
    train = target_feature_df.iloc[i - window:i]
    test = target_feature_df.iloc[i:i + 1]

    # Simple factor: average of features
    factor_train = train[factor_features].mean(axis=1)
    factor_test = test[factor_features].mean(axis=1)

    model = LinearRegression()
    model.fit(factor_train.values.reshape(-1, 1), train['CPIAUCSL'].values)
    pred = model.predict(factor_test.values.reshape(-1, 1))[0]
    true = test['CPIAUCSL'].values[0]
    result_factor.append({
        'prediction': pred,
        'actual': true
    })

inflation_prediction = pd.DataFrame(result_factor, index=target_feature_df.index[window:])
inflation_prediction.index = inflation_prediction.index + pd.DateOffset(months=1)
inflation_prediction['inflation_signal'] = inflation_prediction['prediction'].diff()

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- PREPARE BACKTEST DATAFRAMES --------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def regime_label(row):
    if row['inflation'] > 0 and row['growth'] > 0:
        return 0  # Reflation
    elif row['inflation'] > 0 and row['growth'] < 0:
        return 1  # Stagflation
    elif row['inflation'] < 0 and row['growth'] > 0:
        return 2  # Goldilocks
    elif row['inflation'] < 0 and row['growth'] < 0:
        return 3  # Deflation
    else:
        return np.nan
regime_labels = {
    0: 'Reflation',
    1: 'Stagflation',
    2: 'Goldilocks',
    3: 'Deflation'
}
regime_colors = {
    0: '#90ee90',  # Reflation (red)
    1: '#ffc107',  # Stagflation (yellow)
    2: '#28a745',  # Goldilocks (green)
    3: '#dc3545'  # Deflation (blue)
    }

grid_growth_inflation_spx = merge_dfs([
    cli.pct_change(),
    inflation_prediction['inflation_signal'].resample('ME').last(),
    spx_monthly.pct_change().shift(-1)
]).dropna()
grid_growth_inflation_spx.columns = ['growth', 'inflation', 'spx']
grid_growth_inflation_spx = grid_growth_inflation_spx['2005-01-01':]

grid_growth_inflation_agg = merge_dfs([
    cli.pct_change(),
    inflation_prediction['inflation_signal'].resample('ME').last(),
    agg.pct_change().shift(-1)
]).dropna()
grid_growth_inflation_agg.columns = ['growth', 'inflation', 'bonds']
grid_growth_inflation_agg = grid_growth_inflation_agg['2005-01-01':]

grid_growth_inflation_mags = merge_dfs([
    cli.pct_change(),
    inflation_prediction['inflation_signal'].resample('ME').last(),
    mock_mags_monthly_pct.shift(-1)
]).dropna()
grid_growth_inflation_mags.columns = ['growth', 'inflation', 'mags']
grid_growth_inflation_mags = grid_growth_inflation_mags['2005-01-01':]

grid_growth_inflation_spx['regime_code'] = grid_growth_inflation_spx.apply(regime_label, axis=1)
grid_growth_inflation_spx['regime_label'] = grid_growth_inflation_spx['regime_code'].map(regime_labels)
grid_growth_inflation_spx['regime_color'] = grid_growth_inflation_spx['regime_code'].map(regime_colors)

grid_growth_inflation_agg['regime_code'] = grid_growth_inflation_agg.apply(regime_label, axis=1)
grid_growth_inflation_agg['regime_label'] = grid_growth_inflation_agg['regime_code'].map(regime_labels)
grid_growth_inflation_agg['regime_color'] = grid_growth_inflation_agg['regime_code'].map(regime_colors)

grid_growth_inflation_mags['regime_code'] = grid_growth_inflation_mags.apply(regime_label, axis=1)
grid_growth_inflation_mags['regime_label'] = grid_growth_inflation_mags['regime_code'].map(regime_labels)
grid_growth_inflation_mags['regime_color'] = grid_growth_inflation_mags['regime_code'].map(regime_colors)

def grid_equity_weights(row):
    regime_weights = {
        'Goldilocks': 1,
        'Reflation': 0.75,
        'Deflation': 0.5,
        'Stagflation': 0.25
    }
    return regime_weights.get(row['regime_label'], np.nan)

def grid_bond_weights(row):
    regime_weights = {
        'Goldilocks': 1,
        'Reflation': 0.75,
        'Deflation': 0.5,
        'Stagflation': 0.25
    }
    return regime_weights.get(row['regime_label'], np.nan)

occurrences_table = pd.DataFrame(columns = ['Goldilocks','Reflation','Deflation','Stagflation'],
                                     index = ['% of Occurrences'])
occurrences_table.loc['% of Occurrences','Goldilocks'] = (
        len(grid_growth_inflation_spx[grid_growth_inflation_spx['regime_label'] == 'Goldilocks'])
        / len(grid_growth_inflation_spx))
occurrences_table.loc['% of Occurrences', 'Reflation'] = (
        len(grid_growth_inflation_spx[grid_growth_inflation_spx['regime_label'] == 'Reflation'])
        / len(grid_growth_inflation_spx))
occurrences_table.loc['% of Occurrences', 'Deflation'] = (
        len(grid_growth_inflation_spx[grid_growth_inflation_spx['regime_label'] == 'Deflation'])
        / len(grid_growth_inflation_spx))
occurrences_table.loc['% of Occurrences', 'Stagflation'] = (
        len(grid_growth_inflation_spx[grid_growth_inflation_spx['regime_label'] == 'Stagflation'])
        / len(grid_growth_inflation_spx))

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- GRID -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def grid_regime_nowcast():
    with open(Path(DATA_DIR) / 'inflation_variables_merge.pkl', 'rb') as file:
        inflation_variables_merge = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'di_reserves.pkl', 'rb') as file:
        di_reserves = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'm2_money_supply.pkl', 'rb') as file:
        m2_money_supply = pd.read_pickle(file)
    inflation_variables_merge = merge_dfs([inflation_variables_merge, di_reserves, m2_money_supply])
    target_feature_df = inflation_variables_merge.pct_change(12)
    target_feature_df['TOTRESNS'] = target_feature_df['TOTRESNS'] * -1
    target_feature_df['M2SL'] = target_feature_df['M2SL'] * -1
    target_feature_df['CPIAUCSL'] = target_feature_df['CPIAUCSL'].shift(-1)
    target_feature_df = target_feature_df.dropna()

    train = target_feature_df.iloc[len(target_feature_df) - 37:len(target_feature_df) - 1]
    test = target_feature_df.iloc[len(target_feature_df) - 1:len(target_feature_df)]
    factor_features = [
        'CPILFESL',
        'CPIUFDSL',
        'CPIENGSL',
        'CUSR0000SAH3',
        'CPIAPPSL',
        'CPIMEDSL',
        'CPITRNSL',
        'CUSR0000SAF116',
        'CUSR0000SETB',
        'CUSR0000SASLE'
    ]
    factor_train = train[factor_features].mean(axis=1)
    factor_test = test[factor_features].mean(axis=1)
    model = LinearRegression()
    model.fit(factor_train.values.reshape(-1, 1), train['CPIAUCSL'].values)
    inflation_yoy_pred = model.predict(factor_test.values.reshape(-1, 1))[0]
    inflation_2nd_order_diff = (inflation_yoy_pred - target_feature_df['CPIAUCSL'][-1]) / target_feature_df['CPIAUCSL'][-1]
    cli_1st_order_change = cli.pct_change().iloc[-1][0]

    if inflation_2nd_order_diff > 0 and cli_1st_order_change > 0:
        upcoming_grid_regime = 'Reflation'
    elif inflation_2nd_order_diff > 0 and cli_1st_order_change < 0:
        upcoming_grid_regime = 'Stagflation'
    elif inflation_2nd_order_diff < 0 and cli_1st_order_change > 0:
        upcoming_grid_regime = 'Goldilocks'
    elif inflation_2nd_order_diff < 0 and cli_1st_order_change < 0:
        upcoming_grid_regime = 'Deflation'

    # Color mapping for regimes (customize as desired)
    regime_colors = {
        "Goldilocks": "#28a745",  # Green
        "Reflation": "#90ee90",  # Super light green
        "Stagflation": "#ffc107",  # Yellow
        "Deflation": "#dc3545"  # Red
    }
    regime_color = regime_colors.get(upcoming_grid_regime, "gray")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Growth**")
        st.markdown(f"<span style='font-size:1.5em;font-weight:bold;'>{cli_1st_order_change:+.2%}</span>",
                    unsafe_allow_html=True)
        st.caption("OECD CLI 1st Order % Change")
    with col2:
        st.markdown("**Inflation**")
        st.markdown(f"<span style='font-size:1.5em;font-weight:bold;'>{inflation_2nd_order_diff:+.2%}</span>",
                    unsafe_allow_html=True)

        st.caption("SAM CPI 2nd Order % Change")
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

    st.dataframe(occurrences_table)

def grid_equity_backtest():
    grid_growth_inflation_spx['weights'] = grid_growth_inflation_spx.apply(grid_equity_weights, axis=1)
    grid_growth_inflation_spx['bt_returns'] = (grid_growth_inflation_spx['weights'] * grid_growth_inflation_spx['spx']).shift(1)
    grid_growth_inflation_spx['cumsum_spx'] = (1 + grid_growth_inflation_spx['spx']).cumprod() -1
    grid_growth_inflation_spx['cumsum_bt'] = (1 + grid_growth_inflation_spx['bt_returns']).cumprod() -1

    # Drawdown calculations using your helper
    grid_growth_inflation_spx['drawdown_bt'] = compute_drawdown(grid_growth_inflation_spx['cumsum_bt'])
    grid_growth_inflation_spx['drawdown_spx'] = compute_drawdown(grid_growth_inflation_spx['cumsum_spx'])

    # Performance metrics table with your helper function
    grid_metrics = return_metrics(
        backtest_returns_data=grid_growth_inflation_spx[['bt_returns','spx']],
        benchmark_data=grid_growth_inflation_spx[['spx']],
        ann_factor=12
    )

    # Display main summary table
    streamlit_return_metrics_table(grid_metrics)

    regime_stats_df = return_metrics_by_regime(base_df = grid_growth_inflation_spx,
                                               return_col = 'bt_returns',
                                               benchmark_col = 'spx',
                                               regime_col='regime_label', ann_factor=12)
    desired_order = ['Goldilocks', 'Reflation', 'Deflation', 'Stagflation']
    regime_stats_df = regime_stats_df.reindex(desired_order)

    # Styled regime table
    streamlit_return_metrics_table(regime_stats_df)

    # Cum return plot
    streamlit_plot(
        df=grid_growth_inflation_spx,
        columns_array=['cumsum_bt', 'cumsum_spx'],
        colors_array=['#5FB3FF', '#2DCDB2'],
        graph_title="GRID Backtest",
        y_axis_label="Cumulative Return"
    )

    # Drawdown plot
    streamlit_drawdown_plot(
        df=grid_growth_inflation_spx,
        graph_labels=['GRID', 'SPX'],
        df_columns_to_plot=['drawdown_bt', 'drawdown_spx'],
        line_colors=['rgba(95,179,255,1)', 'rgba(45,205,178,1)'],
        fill_colors=['rgba(95,179,255,0.3)', 'rgba(45,205,178,0.3)']
    )

    # Regime return distribution subplots
    plot_regime_return_histograms(
        grid_growth_inflation_spx,
        regime_col='regime_label',
        return_col='spx',
        regimes=['Goldilocks', 'Reflation', 'Deflation', 'Stagflation']
    )


def grid_bonds_backtest():
    grid_growth_inflation_agg['weights'] = grid_growth_inflation_agg.apply(grid_bond_weights, axis=1)
    grid_growth_inflation_agg['bt_returns'] = (grid_growth_inflation_agg['weights'] * grid_growth_inflation_agg['bonds']).shift(1)
    grid_growth_inflation_agg['cumsum_bonds'] = (1 + grid_growth_inflation_agg['bonds']).cumprod() -1
    grid_growth_inflation_agg['cumsum_bt'] = (1 + grid_growth_inflation_agg['bt_returns']).cumprod() -1

    # Drawdown calculations using your helper
    grid_growth_inflation_agg['drawdown_bt'] = compute_drawdown(grid_growth_inflation_agg['cumsum_bt'])
    grid_growth_inflation_agg['drawdown_bonds'] = compute_drawdown(grid_growth_inflation_agg['cumsum_bonds'])

    # Performance metrics table with your helper function
    grid_metrics = return_metrics(
        backtest_returns_data=grid_growth_inflation_agg[['bt_returns', 'bonds']],
        benchmark_data=grid_growth_inflation_agg[['bonds']],
        ann_factor=12
    )

    # Display main summary table
    streamlit_return_metrics_table(grid_metrics)

    regime_stats_df = return_metrics_by_regime(base_df=grid_growth_inflation_agg,
                                               return_col='bt_returns',
                                               benchmark_col='bonds',
                                               regime_col='regime_label', ann_factor=12)
    desired_order = ['Goldilocks', 'Reflation', 'Deflation', 'Stagflation']
    regime_stats_df = regime_stats_df.reindex(desired_order)

    # Styled regime table
    streamlit_return_metrics_table(regime_stats_df)

    # Cum return plot
    streamlit_plot(
        df=grid_growth_inflation_agg,
        columns_array=['cumsum_bt', 'cumsum_bonds'],
        colors_array=['#5FB3FF', '#2DCDB2'],
        graph_title="GRID Backtest",
        y_axis_label="Cumulative Return"
    )

    # Drawdown plot
    streamlit_drawdown_plot(
        df=grid_growth_inflation_agg,
        graph_labels=['GRID', 'Bonds'],
        df_columns_to_plot=['drawdown_bt', 'drawdown_bonds'],
        line_colors=['rgba(95,179,255,1)', 'rgba(45,205,178,1)'],
        fill_colors=['rgba(95,179,255,0.3)', 'rgba(45,205,178,0.3)']
    )

    # Regime return distribution subplots
    plot_regime_return_histograms(
        grid_growth_inflation_agg,
        regime_col='regime_label',
        return_col='bonds',
        regimes=['Goldilocks', 'Reflation', 'Deflation', 'Stagflation']
    )

def grid_mags_backtest():
    grid_growth_inflation_mags['weights'] = grid_growth_inflation_mags.apply(grid_equity_weights, axis=1)
    grid_growth_inflation_mags['bt_returns'] = (
                grid_growth_inflation_mags['weights'] * grid_growth_inflation_mags['mags']).shift(1)
    grid_growth_inflation_mags['cumsum_mags'] = (1 + grid_growth_inflation_mags['mags']).cumprod() -1
    grid_growth_inflation_mags['cumsum_bt'] = (1 + grid_growth_inflation_mags['bt_returns']).cumprod() -1

    # Drawdown calculations using your helper
    grid_growth_inflation_mags['drawdown_bt'] = compute_drawdown(grid_growth_inflation_mags['cumsum_bt'])
    grid_growth_inflation_mags['drawdown_mags'] = compute_drawdown(grid_growth_inflation_mags['cumsum_mags'])

    # Performance metrics table with your helper function
    grid_metrics = return_metrics(
        backtest_returns_data=grid_growth_inflation_mags[['bt_returns', 'mags']],
        benchmark_data=grid_growth_inflation_mags[['mags']],
        ann_factor=12
    )

    # Display main summary table
    streamlit_return_metrics_table(grid_metrics)

    regime_stats_df = return_metrics_by_regime(base_df=grid_growth_inflation_mags,
                                               return_col='bt_returns',
                                               benchmark_col='mags',
                                               regime_col='regime_label', ann_factor=12)
    desired_order = ['Goldilocks', 'Reflation', 'Deflation', 'Stagflation']
    regime_stats_df = regime_stats_df.reindex(desired_order)

    # Styled regime table
    streamlit_return_metrics_table(regime_stats_df)

    # Cum return plot
    streamlit_plot(
        df=grid_growth_inflation_mags,
        columns_array=['cumsum_bt', 'cumsum_mags'],
        colors_array=['#5FB3FF', '#2DCDB2'],
        graph_title="GRID Backtest",
        y_axis_label="Cumulative Return"
    )

    # Drawdown plot
    streamlit_drawdown_plot(
        df=grid_growth_inflation_mags,
        graph_labels=['GRID', 'MAGS'],
        df_columns_to_plot=['drawdown_bt', 'drawdown_mags'],
        line_colors=['rgba(95,179,255,1)', 'rgba(45,205,178,1)'],
        fill_colors=['rgba(95,179,255,0.3)', 'rgba(45,205,178,0.3)']
    )

    # Regime return distribution subplots
    plot_regime_return_histograms(
        grid_growth_inflation_mags,
        regime_col='regime_label',
        return_col='mags',
        regimes=['Goldilocks', 'Reflation', 'Deflation', 'Stagflation']
    )

