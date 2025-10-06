import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import plotly.graph_objs as go

DATA_DIR = os.getenv('DATA_DIR', 'data')


def plot_growth_predictor():
    # --- Load Data ---
    with open(Path(DATA_DIR) / 'growth_variables_merge.pkl', 'rb') as file:
        growth_variables_merge = pd.read_pickle(file)
    target_feature_df = growth_variables_merge.pct_change()
    target_feature_df['PCEC96'] = target_feature_df['PCEC96'].shift(-1)
    target_feature_df = target_feature_df.dropna()
    idx = target_feature_df.index

    # Optionally filter COVID-19 outlier window (which can distort metrics)
    covid_mask = ~target_feature_df.index.to_series().between('2020-04-01', '2020-08-31')
    filtered = target_feature_df[covid_mask]

    features_all = [col for col in filtered.columns if col != 'PCEC96']
    window = 36
    max_features = 4

    # --- Feature Grid Search ---
    best_rmse = np.inf
    best_result = None
    best_features = None
    best_model_type = None

    for n_feats in range(2, max_features + 1):
        for c in combinations(features_all, n_feats):
            preds = []
            actuals = []
            for i in range(window, len(filtered)):
                train = filtered.iloc[i - window:i]
                test = filtered.iloc[i:i + 1]

                X_train = train[list(c)].values
                y_train = train['PCEC96'].values
                X_test = test[list(c)].values
                y_test = test['PCEC96'].values

                scaler_X = StandardScaler()
                X_train_sc = scaler_X.fit_transform(X_train)
                X_test_sc = scaler_X.transform(X_test)
                scaler_y = StandardScaler()
                y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

                # Try both OLS and Ridge, pick better in-sample RMSE
                reg1 = LinearRegression().fit(X_train_sc, y_train_sc)
                reg2 = Ridge(alpha=2.0).fit(X_train_sc, y_train_sc)
                preds_ols = scaler_y.inverse_transform(reg1.predict(X_test_sc).reshape(-1, 1)).ravel()
                preds_ridge = scaler_y.inverse_transform(reg2.predict(X_test_sc).reshape(-1, 1)).ravel()

                # Pick lower in-sample error at each step (can change to out-of-sample metric if needed)
                preds.append(preds_ridge[0] if mean_squared_error(y_train, scaler_y.inverse_transform(
                    reg2.predict(X_train_sc).reshape(-1, 1)).ravel()) <
                                               mean_squared_error(y_train, scaler_y.inverse_transform(
                                                   reg1.predict(X_train_sc).reshape(-1, 1)).ravel()) else preds_ols[0])
                actuals.append(y_test[0])

            # Evaluate RMSE metric
            rmse = np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_features = c
                best_result = (np.array(preds), np.array(actuals))

    pred, true = best_result
    errors = pred - true

    # Upside/downside
    std_err = errors.std()
    upside = pred + std_err
    downside = pred - std_err
    tracking_error = np.mean(np.abs(errors)) * 1e4
    correct_direction = np.mean(np.sign(pred) == np.sign(true))
    target_std = np.std(true)
    rmse = np.sqrt(np.mean(errors ** 2))
    rmse_improvement = (1 - rmse / target_std) if target_std > 0 else np.nan

    # --- Streamlit Output ---
    st.title("Real PCE Growth: Factor Model Backtest (Tuned)")
    st.write(f"**Best feature set:** {best_features}  |  **Rolling window:** {window}m  |  **Model:** Ridge/OLS select")

    # PLOT
    fig = go.Figure()
    st_dt = filtered.index[window:]
    fig.add_trace(go.Scatter(x=st_dt, y=true, name='Actual', mode='lines', line=dict(color='#2056AE', width=2)))
    fig.add_trace(go.Scatter(x=st_dt, y=pred, name='Predicted', mode='lines', line=dict(color='#F2552C', width=2)))
    fig.add_trace(go.Scatter(x=st_dt, y=upside, name='Upside (1 stdev error)', mode='lines',
                             line=dict(color='#6AC47E', dash='dot')))
    fig.add_trace(go.Scatter(x=st_dt, y=downside, name='Downside (1 stdev error)', mode='lines',
                             line=dict(color='#E74C3C', dash='dot')))
    fig.update_layout(height=450, hovermode='x unified', legend=dict(title='Legend', orientation='h', y=-0.25),
                      margin=dict(t=30, b=30), title="PCE Growth: Actual vs Predicted and Scenarios")
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
            f"{100 * correct_direction:.2f}",
            f"{rmse:.6f}",
            f"{target_std:.6f}",
            f"{100 * rmse_improvement:.2f}"
        ]
    })
    st.table(metrics)

    if rmse_improvement >= 0.20:
        st.success(
            f"RMSE is at least 20% lower than the standard deviation of the target (Improvement: {100 * rmse_improvement:.2f}%)")
    else:
        st.warning(f"RMSE improvement is only {100 * rmse_improvement:.2f}%. Recommend more feature/model tuning.")

# In app.py:
# plot_growth_predictor()
