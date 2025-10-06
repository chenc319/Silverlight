### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------ GROWTH PCE PREDICTOR ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from plotly.subplots import make_subplots

DATA_DIR = os.getenv('DATA_DIR', 'data')

def plot_growth_predictor():
    # --- Load Data ---
    with open(Path(DATA_DIR) / 'growth_variables_merge.pkl', 'rb') as file:
        growth_variables_merge = pd.read_pickle(file)
    target_feature_df = growth_variables_merge.pct_change()
    target_feature_df['PCEC96'] = target_feature_df['PCEC96'].shift(-1)
    target_feature_df = target_feature_df.dropna()

    # --- Model Setup ---
    result_factor = []
    window = 36  # Rolling window
    factor_features = ['RETAILSMSA', 'USALOLITOAASTSAM', 'INDPRO', 'CES0600000007']

    for i in range(window, len(target_feature_df)):
        train = target_feature_df.iloc[i - window:i]
        test = target_feature_df.iloc[i:i + 1]

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

    # --- Upside/Downside Case ---
    std_err = errors.std()
    df_factor['upside'] = df_factor['prediction'] + std_err
    df_factor['downside'] = df_factor['prediction'] - std_err

    # --- Metrics ---
    tracking_error = np.mean(np.abs(errors)) * 1e4  # bp
    correct_direction = np.mean(
        np.sign(df_factor['prediction']) == np.sign(df_factor['actual'])
    )
    rmse = np.sqrt(np.mean(errors ** 2))
    target_std = df_factor['actual'].std()
    rmse_improvement = (1 - rmse / target_std) if target_std > 0 else np.nan

    ### PLOTS ###
    st.title("Real PCE Growth: Factor Model Backtest")

    # Prediction visual: Actual vs Predicted (+/– error bands)
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
        name='Upside (1 stdev error)',
        mode='lines',
        line=dict(color='#6AC47E', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df_factor.index,
        y=df_factor['downside'],
        name='Downside (1 stdev error)',
        mode='lines',
        line=dict(color='#E74C3C', dash='dot')
    ))
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title="PCE Growth: Actual vs Predicted and Scenarios"
    )
    st.plotly_chart(fig, use_container_width=True)

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
    if rmse_improvement >= 0.20:
        st.success(f"RMSE is at least 20% lower than the standard deviation of the target (Improvement: {100*rmse_improvement:.2f}%)")
    else:
        st.warning(f"RMSE improvement is only {100*rmse_improvement:.2f}%. Recommend model tuning.")

# In app.py use:
# plot_growth_predictor()
