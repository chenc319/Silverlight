### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import plotly.subplots as sp
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import functools as ft
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
DATA_DIR = os.getenv('DATA_DIR', 'data')

barra_factors = {
    "SPHB": "beta",                # High Beta ETF
    "VLUE": "book_to_price",       # Value/Book-to-Price ETF
    "VYM": "dividend_yield",       # High Dividend Yield ETF
    "IWD": "earnings_yield",       # Value ETF (earnings yield as core metric)
    "IWF": "growth",               # Large Cap Growth ETF
    "SPLV": "leverage",            # Low volatility, proxies low leverage
    "SPY": "liquidity",            # S&P 500 ETF (liquidity proxy)
    "IWR": "mid_cap",              # Mid Cap ETF
    "MTUM": "momentum",            # Momentum Style ETF
    "XMMO": "profitability",       # S&P SmallCap Momentum ETF (profitability proxy)
    "USMV": "residual_volatility", # Minimum Volatility ETF
    "IWM": "size"                  # Small Cap ETF
}

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

barra_factors_df = pd.DataFrame()
for each_factor in list(barra_factors.keys()):
    with open(Path(DATA_DIR) / (each_factor + '.csv'), 'rb') as file:
        factor_df = pd.read_csv(file)
    factor_df.index = pd.to_datetime(factor_df['Date']).values
    factor_df = pd.DataFrame(factor_df['Close'])
    factor_df.columns = [barra_factors[each_factor]]
    barra_factors_df = merge_dfs([barra_factors_df, factor_df])

with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    spx_daily = pd.DataFrame(sp500['Close'])
    spx_daily.columns = ['spx']


def plot_barra_predictor():
    target_feature_df = merge_dfs([spx_daily.pct_change().shift(-1), barra_factors_df.pct_change()]).dropna()

    target_feature_df.columns
    result_factor = []
    window = 63
    factor_features = ['beta', 'book_to_price', 'dividend_yield', 'earnings_yield',
                       'growth', 'leverage', 'liquidity', 'mid_cap', 'momentum',
                       'profitability', 'residual_volatility', 'size']

    for i in range(window, len(target_feature_df)):
        train = target_feature_df.iloc[i - window:i]
        test = target_feature_df.iloc[i:i + 1]

        # Simple factor: average of features
        factor_train = train[factor_features].mean(axis=1)
        factor_test = test[factor_features].mean(axis=1)

        model = LinearRegression()
        model.fit(factor_train.values.reshape(-1, 1), train['spx'].values)
        pred = model.predict(factor_test.values.reshape(-1, 1))[0]
        true = test['spx'].values[0]
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



### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_barra_factors(start, end, **kwargs):
    df = barra_factors_df.copy().resample('ME').last()
    # Define a gentle neutral palette (no neons)
    palette = [
        "#35c9c3",  # Teal
        "#f9c6bb",  # Peach
        "#98e3f9",  # Light Blue
        "#59b758",  # Leaf Green
        "#e54d42",  # Soft Red
        "#fff8a9",  # Pale Yellow
        "#c4b7f4",  # Lavender
        "#bbf6c2",  # Mint Green
        "#ecbe9d",  # Apricot
        "#6bb7f4",  # Sky Blue
    ]

    ### PLOT ###
    columns_to_plot = barra_factors_df.columns
    fig = sp.make_subplots(rows=4, cols=3, subplot_titles=columns_to_plot)
    for i, col in enumerate(columns_to_plot):
        row = i // 3 + 1
        col_pos = i % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=palette[i % len(palette)], width=2)
            ),
            row=row,
            col=col_pos
        )
    for row in range(1, 6):
        for col in range(1, 4):
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Value", row=row, col=col)
    fig.update_layout(
        showlegend=False,
        height=1800,
        width=1200
    )
    st.plotly_chart(fig, use_container_width=True)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_barra_predictor():
    spx_factor_merge = merge_dfs([
        spx_daily,
        barra_factors_df.pct_change(),
    ])
    target_feature_df = spx_factor_merge.pct_change()
    target_feature_df['spx'] = target_feature_df['spx'].shift(-1)

    target_feature_df.corr()
    target_feature_df['beta'] = target_feature_df['beta'] * -1
    target_feature_df['dividend_yield'] = target_feature_df['dividend_yield'] * -1
    target_feature_df['liquidity'] = target_feature_df['liquidity'] * -1
    target_feature_df['profitability'] = target_feature_df['profitability'] * -1
    target_feature_df['residual_volatility'] = target_feature_df['residual_volatility'] * -1
    target_feature_df['size'] = target_feature_df['size'] * -1
    target_feature_df = target_feature_df.dropna()
    target_feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    target_feature_df.dropna(inplace=True)

    result_factor = []
    window = 63
    factor_features = target_feature_df.columns[1:]

    for i in range(window, len(target_feature_df)):
        train = target_feature_df.iloc[i - window:i]
        test = target_feature_df.iloc[i:i + 1]

        # Simple factor: average of features
        factor_train = train[factor_features].mean(axis=1)
        factor_test = test[factor_features].mean(axis=1)

        model = LinearRegression()
        model.fit(factor_train.values.reshape(-1, 1), train['spx'].values)
        pred = model.predict(factor_test.values.reshape(-1, 1))[0]
        true = test['spx'].values[0]
        result_factor.append({
            'prediction': pred,
            'actual': true
        })

    df_factor = pd.DataFrame(result_factor, index=target_feature_df.index[window:])
    errors = df_factor['prediction'] - df_factor['actual']

    # --- Dynamic Conditional Upside/Downside Case (Rolling Quantiles) ---
    rolling_err_window = 21  # history length for scenarios
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








