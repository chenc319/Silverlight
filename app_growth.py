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
def add_monthly_pct_change_features(df):
    """
    Adds 1, 3, 6 month percent change features for every column, including UNRATE.
    Assumes df has monthly frequency.
    Returns expanded DataFrame.
    """
    lags = [1, 3, 6, 12]
    out = df.copy()
    for col in df.columns:
        for lag in lags:
            out[f"{col}_pct{lag}"] = df[col].pct_change(lag)
    return out

def plot_growth_predictor():
    # --- Load Data ---
    with open(Path(DATA_DIR) / 'growth_variables_merge.pkl', 'rb') as file:
        growth_variables_merge = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'di_reserves.pkl', 'rb') as file:
        di_reserves = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'm2_money_supply.pkl', 'rb') as file:
        m2_money_supply = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'initial_claims.pkl', 'rb') as file:
        initial_claims = pd.read_pickle(file)
    growth_variables_merge = merge_dfs([growth_variables_merge,di_reserves,m2_money_supply,initial_claims])
    target_feature_df = growth_variables_merge.pct_change()
    # target_feature_df['UNRATE'] = target_feature_df['UNRATE'] * -1
    # target_feature_df['TOTRESNS'] = target_feature_df['TOTRESNS'] * -1
    # target_feature_df['M2SL'] = target_feature_df['M2SL'] * -1
    # target_feature_df['ICSA'] = target_feature_df['ICSA'] * -1
    target_feature_df['PCEC96'] = target_feature_df['PCEC96'].shift(-1)
    target_feature_df = target_feature_df.dropna()

    target_feature_df.columns
    # --- Model Setup ---
    result_factor = []
    window = 36  # Rolling window
    factor_features = ['USALOLITOAASTSAM', 'RETAILSMSA', 'RSXFS', 'INDPRO', 'IPMAN', 'IPCONGD',
       'PAYEMS', 'UNRATE', 'pce_goods', 'PCEDG', 'TOTRESNS',
       'ICSA']


    for i in range(window, len(target_feature_df)):
        train = target_feature_df.iloc[i - window:i].copy()
        test = target_feature_df.iloc[i:i + 1].copy()

        # Get correlations with the target for all features except 'PCEC96' itself
        corr_weights = train.corr()['PCEC96'].drop('PCEC96')

        # Only apply weights to the feature columns (not the target column itself)
        train_features = train[corr_weights.index] * corr_weights
        test_features = test[corr_weights.index] * corr_weights

        # Reconstruct train/test DataFrames with weighted features and the target
        train_weighted = train_features.copy()
        train_weighted['PCEC96'] = train['PCEC96']

        test_weighted = test_features.copy()
        test_weighted['PCEC96'] = test['PCEC96']

        # Now train_weighted and test_weighted have no NaNs in Y

        # Simple factor: average of features
        factor_train = train[factor_features].mean(axis=1)
        factor_test = test[factor_features].mean(axis=1)

        model = LinearRegression()
        model.fit(factor_train.values.reshape(-1, 1), train['PCEC96'].values)
        pred = model.predict(factor_test.values.reshape(-1, 1))[0]
        true = test['PCEC96'].values[0]
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

    st.title("Real PCE Growth: Factor Model Backtest (Dynamic Scenarios)")

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
        title="PCE Growth: Actual vs Predicted and Conditional Scenarios"
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

