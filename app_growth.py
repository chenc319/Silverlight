# growth_inflation_model_nowcast_bayes.py

import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.getenv('DATA_DIR', 'data')

# -----------------------------
# Utilities
# -----------------------------
def merge_dfs(array_of_dfs):
    return ft.reduce(lambda l, r: pd.merge(l, r, left_index=True, right_index=True, how='outer'), array_of_dfs)

def monthly_eom(df):
    return df.resample('ME').last()

def rates_of_change_panel(df, win_short=3, win_long=6, z_win_mean=24, z_win_std=12):
    out = df.copy()
    for col in df.columns:
        s = df[col]
        out[f"{col}_m1"] = s.pct_change()                # 1M % change
        out[f"{col}_m3"] = s.pct_change(3) / 3.0         # 3M avg monthly % change
        out[f"{col}_m6"] = s.pct_change(6) / 6.0         # 6M avg monthly % change
        out[f"{col}_diff"] = s.diff()
        out[f"{col}_ma3"] = s.rolling(win_short).mean()
        out[f"{col}_ma6"] = s.rolling(win_long).mean()
        out[f"{col}_std3"] = s.rolling(win_short).std()
        out[f"{col}_std6"] = s.rolling(win_long).std()
        out[f"{col}_min6"] = s.rolling(win_long).min()
        out[f"{col}_max6"] = s.rolling(win_long).max()
        out[f"{col}_z"] = (s - s.rolling(z_win_mean).mean()) / s.rolling(z_win_std).std()
    return out

def make_tracker_features(panel, target_colname):
    # drop target leakage columns identical to target
    X = panel.drop(columns=[c for c in panel.columns if c == target_colname])
    # drop columns that are entirely NaN or constant
    nunique = X.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = X[keep_cols]
    # forward-fill then drop early NaNs
    X = X.ffill().bfill()
    return X

def time_series_kde_regression(X_train, y_train, X_test, cv_splits=5):
    # Nonlinear interpolation via kernel weighting: estimate conditional mean E[y|x] ≈ kernel-smoothed local average
    # We fit KernelDensity on [X, y] jointly and then compute Nadaraya–Watson style weights over X_train.
    # To keep it efficient, we standardize and use Gaussian kernel with bandwidth selected by CV.
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train.values)
    Xt = scaler.transform(X_test.values)

    # Select bandwidth by time-series CV on a proxy kernel smoother (grid)
    grid = {'bandwidth': np.geomspace(0.1, 3.0, 12)}
    # Use 1D proxy on principal score: project to first standardized PC-like axis via mean-weighting
    proxy = Xs @ np.ones(Xs.shape[1]) / np.sqrt(Xs.shape[1])
    proxy = proxy.reshape(-1, 1)
    tscv = TimeSeriesSplit(n_splits=min(cv_splits, len(X_train)//24 if len(X_train)>48 else 3))
    kde_cv = GridSearchCV(KernelDensity(kernel='gaussian'), grid, cv=tscv)
    kde_cv.fit(proxy)
    bw = float(kde_cv.best_params_['bandwidth'])

    # Compute weights with Gaussian kernel on standardized Euclidean distance in feature space
    # w_i = exp(-||x - x_i||^2 / (2*bw^2))
    def kernel_weights(X_base, xq, bandwidth):
        d2 = np.sum((X_base - xq)**2, axis=1)
        w = np.exp(-0.5 * d2 / (bandwidth**2))
        return w

    preds = []
    for xq in Xt:
        w = kernel_weights(Xs, xq, bw)
        if np.all(w == 0) or np.isnan(w).any():
            preds.append(float(np.nan))
        else:
            wnorm = w / np.sum(w)
            preds.append(float(np.sum(wnorm * y_train.values)))
    return np.array(preds), scaler, bw

def residual_sigma(y_true, y_hat):
    r = y_true - y_hat
    return float(np.nanstd(r, ddof=1))

# -----------------------------
# Comparative Base Effects + Bayesian update
# -----------------------------
def compute_base_effects(y_level, months=12):
    """
    Comparative base effects for monthly series:
    - year-ago comp: y_{t-12}
    - 2Y average comp per methodology language: mean growth over prior 24 months
    Returns a DataFrame with:
      comp_ya: prior-year level
      roc_12m: 12M % change
      roc_24m_avg: rolling 24M average monthly % change
    """
    df = pd.DataFrame(index=y_level.index)
    df['level'] = y_level
    df['comp_ya'] = y_level.shift(months)
    df['roc_12m'] = y_level.pct_change(months)
    df['roc_24m_avg'] = (y_level.pct_change(24) / 24.0)
    return df

def bayesian_base_update(baseline_growth, base_effects_df, t, kappa=1.0, prior_var=0.0025, obs_var=0.0050):
    """
    Bayesian adjustment that moves baseline inversely and proportionally to the marginal change in base effects.
    - baseline_growth: prior forecast for growth at time t
    - base_effects_df: DataFrame from compute_base_effects
    - t: timestamp
    - kappa: proportionality to marginal change in base effects (higher -> stronger adjustment)
    - prior_var, obs_var: variances for Bayesian shrinkage
    We measure the marginal change in base effects as delta of 12M ROC and 24M avg ROC.
    """
    if t not in base_effects_df.index:
        return baseline_growth

    row = base_effects_df.loc[t]
    # Marginal change signals
    d12 = row['roc_12m'] - base_effects_df['roc_12m'].shift(1).loc[t]
    d24 = row['roc_24m_avg'] - base_effects_df['roc_24m_avg'].shift(1).loc[t]
    # Inverse relationship per description: adjust opposite the marginal change
    delta = -kappa * np.nanmean([d for d in [d12, d24] if pd.notna(d)])
    if np.isnan(delta):
        delta = 0.0

    # Bayesian blend: posterior mean with prior = baseline_growth; "observation" = baseline_growth + delta
    prior_mean = baseline_growth
    obs_mean = baseline_growth + delta
    post_var = 1.0 / (1.0/prior_var + 1.0/obs_var)
    post_mean = post_var * (prior_mean/prior_var + obs_mean/obs_var)
    return float(post_mean)

# -----------------------------
# Main app
# -----------------------------
def growth_inflation_model():
    st.title("Growth & Inflation Nowcast + Bayesian Base Effects (Hedgeye-style)")

    # DATA PULL
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
    spx_monthly.columns = ['spx']

    with open(Path(DATA_DIR) / 'real_pce.pkl', 'rb') as file:
        real_pce = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'grid_growth_variables.pkl', 'rb') as file:
        grid_growth_variables = pd.read_pickle(file)

    # Align monthly end-of-month
    real_pce = monthly_eom(real_pce)
    grid_growth_variables = monthly_eom(grid_growth_variables)
    spx_monthly = monthly_eom(spx_monthly)

    # Choose variables (user-provided mapping)
    chosen_growth_dict = {
        'USALOLITOAASTSAM': 'cli',
        'INDPRO': 'industrial_production',
        'RSXFS': 'advanced_retail_sales_retail_trade',
        'TLMFGCONS': 'manufacturing_spending',
        'PAYEMS': 'all_employees_total_nonfarm',
        'MANEMP': 'all_employees_manufacturing',
        'PCEC96': 'real_personal_consumption_expenditures',
        'TOTALSA': 'total_vehicle_sales'
    }

    # Build working panel
    base_df = grid_growth_variables[list(chosen_growth_dict.keys())].copy()
    base_df.columns = [chosen_growth_dict[c] for c in base_df.columns]
    panel_raw = merge_dfs([base_df, spx_monthly])
    panel_raw = panel_raw.sort_index()

    # Target: real PCE level, and growth definitions
    target = real_pce.copy()
    target.columns = ['real_pce']
    # One-month % change is the primary growth metric to predict
    real_pce_pct = target['real_pce'].pct_change()

    # Feature engineering per dynamic-factor-like “rate of change” language
    panel = rates_of_change_panel(panel_raw, win_short=3, win_long=6, z_win_mean=24, z_win_std=12)

    # Merge and align target with features
    merged = merge_dfs([real_pce_pct.to_frame('real_pce_m1'), panel]).dropna()
    # Shift target by -1 to nowcast next month change, mirroring original code’s intention
    merged['target_next_m'] = merged['real_pce_m1'].shift(-1)
    merged = merged.dropna().copy()

    # Split features/target
    X = make_tracker_features(merged.drop(columns=['target_next_m']), target_colname='target_next_m')
    y = merged['target_next_m']

    # Rolling nowcast using kernel regression (nonlinear interpolation from factors to base rate)
    window = 36
    base_preds = []
    sigmas = []
    dates = []

    for i in range(window, len(y)):
        X_tr = X.iloc[i-window:i].copy()
        y_tr = y.iloc[i-window:i].copy()
        X_te = X.iloc[[i]].copy()

        # Nonlinear tracker
        yhat_arr, scaler, bw = time_series_kde_regression(X_tr, y_tr, X_te, cv_splits=5)
        yhat = float(yhat_arr[0])

        # Estimate residual sigma from in-sample errors for bands
        # Refit in-sample predictions for sigma
        yhat_in, _, _ = time_series_kde_regression(X_tr, y_tr, X_tr, cv_splits=3)
        sigma = residual_sigma(y_tr.values, yhat_in)

        base_preds.append(yhat)
        sigmas.append(sigma)
        dates.append(X_te.index[0])

    results = pd.DataFrame({
        'Base_Tracker': base_preds,
        'Sigma': sigmas
    }, index=pd.DatetimeIndex(dates))

    # Compute comparative base effects on levels and construct Bayesian out-month adjustment
    base_effects_df = compute_base_effects(target['real_pce'], months=12)

    # Bayesian update: apply to the tracker to form final base path
    updated = []
    for t, base in results['Base_Tracker'].items():
        # create a simple baseline as the tracker estimate
        baseline_growth = base
        adj = bayesian_base_update(baseline_growth, base_effects_df, t, kappa=1.0, prior_var=0.0025, obs_var=0.0050)
        updated.append(adj)

    results['Base'] = updated
    results['Upside'] = results['Base'] + results['Sigma']
    results['Downside'] = results['Base'] - results['Sigma']

    # Actual next-month growth to evaluate
    actual = y.loc[results.index]
    results['Actual'] = actual

    # Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', line=dict(color="#222", width=2)))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Base'], name='Base', line=dict(color="#007AFF", width=2)))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Upside'], name='Upside', line=dict(color="#63dd63", width=1, dash='dash')))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Downside'], name='Downside', line=dict(color="#ff6961", width=1, dash='dash')))
    fig1.update_layout(title="Nowcast + Bayesian Base Effects: Real PCE 1M % Change", yaxis_title="Growth: 1M % Change", hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

    # Metrics/Stats
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Base']))
    mae = mean_absolute_error(results['Actual'], results['Base'])
    st.write(f"**Overall RMSE:** {rmse:.4f}   **MAE:** {mae:.4f}")

    # Regime stats
    if not isinstance(results.index, pd.DatetimeIndex):
        results.index = pd.to_datetime(results.index)

    pre_covid = results.loc[:'2019-12-31']
    post_covid = results.loc['2022-01-01':]
    if len(pre_covid) > 5:
        rmse_pre = np.sqrt(mean_squared_error(pre_covid['Actual'], pre_covid['Base']))
        mae_pre = mean_absolute_error(pre_covid['Actual'], pre_covid['Base'])
        std_pre = pre_covid['Actual'].std()
    else:
        rmse_pre = mae_pre = std_pre = np.nan

    if len(post_covid) > 5:
        rmse_post = np.sqrt(mean_squared_error(post_covid['Actual'], post_covid['Base']))
        mae_post = mean_absolute_error(post_covid['Actual'], post_covid['Base'])
        std_post = post_covid['Actual'].std()
    else:
        rmse_post = mae_post = std_post = np.nan

    st.markdown(f"""#### Regime Stats
- Pre-COVID (to Dec 2019): RMSE={rmse_pre:.4f}, MAE={mae_pre:.4f}, StdDev={std_pre:.4f}
- Post-COVID (Jan 2022+): RMSE={rmse_post:.4f}, MAE={mae_post:.4f}, StdDev={std_post:.4f}
""")

    st.dataframe(results.tail(12))

