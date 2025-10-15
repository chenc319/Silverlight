### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- INFLATION ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import functools as ft
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- INFLATION ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_inflation_predictor():
    # --- Load Data ---
    factor_features = [
        'CPILFESL',
        'PPIACO',
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
    inflation_variables_merge = merge_dfs([inflation_variables_merge,di_reserves,m2_money_supply])
    target_feature_df = inflation_variables_merge.copy()
    target_feature_df['CPIAUCSL'] = target_feature_df['CPIAUCSL'].pct_change(12)
    target_feature_df[factor_features] = target_feature_df[factor_features].pct_change(12)
    target_feature_df.index = target_feature_df.index + pd.DateOffset(months=1)
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

    df_factor = pd.DataFrame(result_factor, index=target_feature_df.index[window:])
    errors = df_factor['prediction'] - df_factor['actual']

    # --- Dynamic Conditional Upside/Downside Case (Rolling Quantiles) ---
    rolling_err_window = 24  # history length for scenarios
    upside = []
    downside = []
    for i in range(len(df_factor)):
        if i == 0:
            upside.append(df_factor['prediction'].iloc[i])
            downside.append(df_factor['prediction'].iloc[i])
        else:
            hist_e = (df_factor['prediction'].iloc[max(0, i - rolling_err_window):i]
                      - df_factor['actual'].iloc[max(0, i - rolling_err_window):i])
            q_up = np.quantile(hist_e, 0.90) if len(hist_e) > 0 else 0
            q_dn = np.quantile(hist_e, 0.10) if len(hist_e) > 0 else 0
            upside.append(df_factor['prediction'].iloc[i] + q_up)
            downside.append(df_factor['prediction'].iloc[i] + q_dn)
    df_factor['upside'] = upside
    df_factor['downside'] = downside

    # --- Metrics ---
    tracking_error = np.mean(np.abs(errors)) * 1e4  # bp
    correct_direction = np.mean(
        np.sign(df_factor['prediction']) == np.sign(df_factor['actual'])
    )
    rmse = np.sqrt(np.mean(errors ** 2))
    target_std = df_factor['actual'].std()
    rmse_improvement = (1 - rmse / target_std) if target_std > 0 else np.nan

    st.title("CPI Inflation: Factor Model Backtest (Dynamic Scenarios)")

    # --- Main Chart: Actual vs Predicted and Dynamic Scenarios ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_factor.index,
        y=df_factor['actual'],
        name='Actual',
        mode='lines',
        line=dict(color='#2056AE', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_factor.index,
        y=df_factor['prediction'],
        name='Predicted',
        mode='lines',
        line=dict(color='#F2552C', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_factor.index,
        y=df_factor['upside'],
        name='Upside (90th percentile error)',
        mode='lines',
        line=dict(color='#6AC47E', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df_factor.index,
        y=df_factor['downside'],
        name='Downside (10th percentile error)',
        mode='lines',
        line=dict(color='#E74C3C', dash='dot')
    ))
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title="CPI Inflation: Actual vs Predicted and Conditional Scenarios"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Tracking Error Chart ---
    tracking_errors_history = np.abs(df_factor['prediction'] - df_factor['actual']) * 1e4  # in bps
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(
        x=df_factor.index,
        y=tracking_errors_history,
        name='Tracking Error (bp)',
        mode='lines',
        line=dict(color='#F1C40F', width=2)
    ))
    fig_error.update_layout(
        height=300,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title='Historical Tracking Error (basis points)'
    )
    st.plotly_chart(fig_error, use_container_width=True)

    # --- Metrics Table ---
    st.title("Prediction Performance Metrics")
    metrics = pd.DataFrame({
        'Metric': [
            'Avg Tracking Error (bp)',
            'Sign Prediction Accuracy (%)',
            'RMSE',
            'STD of Target',
            'RMSE Improvement (%)'
        ],
        'Value': [
            f"{tracking_error:.2f}",
            f"{100*correct_direction:.2f}",
            f"{rmse:.6f}",
            f"{target_std:.6f}",
            f"{100*rmse_improvement:.2f}"
        ]
    })
    st.table(metrics)

    # --- RMSE Validity Alert ---
    if rmse_improvement >= 0.10:
        st.success(f"RMSE is at least 10% lower than the standard deviation of the target (Improvement: {100*rmse_improvement:.2f}%)")
    else:
        st.warning(f"RMSE improvement is only {100*rmse_improvement:.2f}%. Recommend model tuning.")

def plot_cpi_nowcast():
    # --- Load Data ---
    with open(Path(DATA_DIR) / 'inflation_variables_merge.pkl', 'rb') as file:
        inflation_variables_merge = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'di_reserves.pkl', 'rb') as file:
        di_reserves = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'm2_money_supply.pkl', 'rb') as file:
        m2_money_supply = pd.read_pickle(file)
    inflation_variables_merge = merge_dfs([inflation_variables_merge, di_reserves, m2_money_supply])
    target_feature_df = inflation_variables_merge.pct_change(12)
    target_feature_df.index = target_feature_df.index + pd.DateOffset(months=1)
    target_feature_df['TOTRESNS'] *= -1
    target_feature_df['M2SL'] *= -1
    target_feature_df['CPIAUCSL'] = target_feature_df['CPIAUCSL'].shift(-1)
    target_feature_df = target_feature_df.dropna()

    train = target_feature_df.iloc[len(target_feature_df) - 37:len(target_feature_df)-1]
    test = target_feature_df.iloc[len(target_feature_df)-1:len(target_feature_df)]
    factor_features = [
        'CPILFESL',
        'PPIACO',
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
    pred = model.predict(factor_test.values.reshape(-1, 1))[0]
    train_pred = model.predict(factor_train.values.reshape(-1, 1))

    window = 24  # number of months to look back
    hist_errors = train['CPIAUCSL'].values - train_pred
    recent_errors = hist_errors[-window:] if len(hist_errors) >= window else hist_errors
    upside_shift = np.quantile(recent_errors, 0.80)  # upper 80th percentile error
    downside_shift = np.quantile(recent_errors, 0.20)  # lower 20th percentile error
    upside_pred = pred + upside_shift
    downside_pred = pred + downside_shift

    # --- Prepare Data for Chart ---
    cpi_actual = target_feature_df['CPIAUCSL'].iloc[-12:]
    history_dates = cpi_actual.index
    forecast_date = history_dates[-1] + pd.DateOffset(months=1)

    fig = go.Figure()

    # Actual CPI history
    fig.add_trace(go.Scatter(
        x=history_dates,
        y=cpi_actual,
        mode='lines+markers',
        name='Actual CPI YoY',
        line=dict(color='black', width=2)
    ))

    base_pct = f"{100 * pred:.2f}%"
    upside_pct = f"{100 * upside_pred:.2f}%"
    downside_pct = f"{100 * downside_pred:.2f}%"

    # Model predictions as three points with labels
    fig.add_trace(go.Scatter(
        x=[forecast_date],
        y=[pred],
        mode='markers+text',
        name='Base Case',
        marker=dict(color='blue', size=12),
        text=[f"{base_pct}"],
        textposition='middle right'
    ))
    fig.add_trace(go.Scatter(
        x=[forecast_date],
        y=[upside_pred],
        mode='markers+text',
        name='Upside',
        marker=dict(color='green', size=12),
        text=[f"{upside_pct}"],
        textposition='top right'
    ))
    fig.add_trace(go.Scatter(
        x=[forecast_date],
        y=[downside_pred],
        mode='markers+text',
        name='Downside',
        marker=dict(color='red', size=12),
        text=[f"{downside_pct}"],
        textposition='bottom right'
    ))

    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        xaxis_title="Month",
        yaxis_title="CPI YoY (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

