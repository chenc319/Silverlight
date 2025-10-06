import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

DATA_DIR = os.getenv('DATA_DIR', 'data')

def growth_inflation_model():
    st.title("Growth & Inflation PCA OLS Backtest")

    # DATA PULL
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
    spx_monthly.columns = ['spx']

    with open(Path(DATA_DIR) / 'growth.pkl', 'rb') as file:
        growth = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'grid_growth_variables.pkl', 'rb') as file:
        grid_growth_variables = pd.read_pickle(file)

    chosen_growth_dict = {
        'INDPRO': 'industrial_production',
        'RSXFS': 'advanced_retail_sales_retail_trade',
        'TLMFGCONS': 'manufacturing_spending',
        'PAYEMS': 'all_employees_total_nonfarm',
        'MANEMP': 'all_employees_manufacturing',
        'PCEC96': 'real_personal_consumption_expenditures',
        'TOTALSA': 'total_vehicle_sales'
    }
    def merge_dfs(array_of_dfs):
        return ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), array_of_dfs)

    chose_growth_variables = merge_dfs([grid_growth_variables[chosen_growth_dict.keys()], spx_monthly])
    feature_cols = chose_growth_variables.columns
    df = chose_growth_variables.copy()
    for col in feature_cols:
        df[f"{col}_diff"] = df[col].diff()
        df[f"{col}_pct"] = df[col].pct_change()
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_lag6"] = df[col].shift(6)
        df[f"{col}_ma3"] = df[col].rolling(3).mean()
        df[f"{col}_ma6"] = df[col].rolling(6).mean()
        df[f"{col}_std3"] = df[col].rolling(3).std()
        df[f"{col}_std6"] = df[col].rolling(6).std()
        df[f"{col}_min6"] = df[col].rolling(6).min()
        df[f"{col}_max6"] = df[col].rolling(6).max()
        df[f"{col}_zscore12"] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(12).std()

    merge_df = merge_dfs([growth.diff().shift(-1), df]).dropna()

    X = merge_df.iloc[:, 1:]
    y = merge_df.iloc[:, 0]
    window = 12
    n_components = 2

    base, up, down = [], [], []

    for i in range(window, len(y)):
        X_train = X.iloc[i-window:i]
        y_train = y.iloc[i-window:i]
        X_test = X.iloc[[i]]

        X_tr_mean = X_train.mean()
        X_tr_std = X_train.std().replace(0, 1)
        X_train_std = (X_train - X_tr_mean) / X_tr_std
        X_test_std = (X_test - X_tr_mean) / X_tr_std

        pca = PCA(n_components=n_components)
        X_pca_train = pca.fit_transform(X_train_std)
        X_pca_test = pca.transform(X_test_std)

        model = LinearRegression()
        model.fit(X_pca_train, y_train)
        pred = model.predict(X_pca_test)[0]
        resids = y_train - model.predict(X_pca_train)
        sigma = resids.std()

        base.append(pred)
        up.append(pred + sigma)
        down.append(pred - sigma)

    result_index = merge_df.index[window:]
    results = pd.DataFrame({
        'Actual': y.iloc[window:].values,
        'Base': base,
        'Upside': up,
        'Downside': down
    }, index=result_index)

    # PLOTS
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', line=dict(color="#222", width=2)))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Base'], name='Base', line=dict(color="#007AFF", width=2)))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Upside'], name='Upside', line=dict(color="#63dd63", width=1, dash='dash')))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Downside'], name='Downside', line=dict(color="#ff6961", width=1, dash='dash')))
    fig1.update_layout(title="PCA OLS Backtest: Actual vs Model", yaxis_title="Growth: 1M % Change", hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

    # Metrics/Stats
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Base']))
    mae = mean_absolute_error(results['Actual'], results['Base'])
    st.write(f"**Overall RMSE:** {rmse:.4f} &nbsp;&nbsp; **MAE:** {mae:.4f} &nbsp;&nbsp; **Baseline StdDev:** {y.std():.4f}")

    if not isinstance(results.index, pd.DatetimeIndex):
        results.index = pd.to_datetime(results.index)

    # Pre/Post COVID stats
    pre_covid = results.loc[:'2019-12-31']
    post_covid = results.loc['2022-01-01':]
    rmse_pre = np.sqrt(mean_squared_error(pre_covid['Actual'], pre_covid['Base']))
    mae_pre = mean_absolute_error(pre_covid['Actual'], pre_covid['Base'])
    std_pre = pre_covid['Actual'].std()
    rmse_post = np.sqrt(mean_squared_error(post_covid['Actual'], post_covid['Base']))
    mae_post = mean_absolute_error(post_covid['Actual'], post_covid['Base'])
    std_post = post_covid['Actual'].std()

    st.markdown(f"""#### Regime Stats
    - **Pre-COVID (to Dec 2019):** RMSE={rmse_pre:.4f}, MAE={mae_pre:.4f}, StdDev={std_pre:.4f}
    - **Post-COVID (Jan 2022+):** RMSE={rmse_post:.4f}, MAE={mae_post:.4f}, StdDev={std_post:.4f}
    """)

    st.dataframe(results.tail(12))

# In your Streamlit app run: growth_inflation_model()
