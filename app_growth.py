# growth_pce_nowcast_bayes_robust.py

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
    # Ensure DateTimeIndex and resample to month end
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        if 'Date' in df.columns:
            df.index = pd.to_datetime(df['Date'])
            df = df.drop(columns=['Date'])
        else:
            df.index = pd.to_datetime(df.index)
    return df.resample('ME').last()

def sanitize_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop all-NaN columns
    df = df.dropna(axis=1, how='all')
    # Drop constant columns
    nunique = df.nunique(dropna=True)
    df = df.loc[:, nunique[nunique > 1].index]
    # Replace inf
    df = df.replace([np.inf, -np.inf], np.nan)
    # Limit small local gaps only (avoid creating lookahead)
    df = df.ffill(limit=2).bfill(limit=2)
    return df

def safe_inner_join_on_index(dfs):
    idx = None
    for d in dfs:
        idx = d.index if idx is None else idx.intersection(d.index)
    aligned = [d.loc[idx] for d in dfs]
    return merge_dfs(aligned)

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

def time_series_kde_regression(X_train, y_train, X_query, cv_splits=5):
    """
    Nonlinear interpolation via kernel-weighted local averaging (Nadaraya–Watson style).
    Training and query matrices must be finite; no NaNs allowed.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train.values)
    Xq = scaler.transform(X_query.values)

    # Bandwidth selection using TS CV on a 1D proxy to avoid dimensionality curse
    grid = {'bandwidth': np.geomspace(0.1, 3.0, 12)}
    proxy = Xs @ np.ones(Xs.shape[1]) / np.sqrt(Xs.shape[1])
    proxy = proxy.reshape(-1, 1)
    n_splits = max(3, min(cv_splits, len(X_train) // 24 if len(X_train) > 48 else 3))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    kde_cv = GridSearchCV(KernelDensity(kernel='gaussian'), grid, cv=tscv)
    kde_cv.fit(proxy)
    bw = float(kde_cv.best_params_['bandwidth'])

    def kernel_weights(X_base, xq, bandwidth):
        d2 = np.sum((X_base - xq)**2, axis=1)
        w = np.exp(-0.5 * d2 / (bandwidth**2))
        return w

    preds = []
    for xq in Xq:
        w = kernel_weights(Xs, xq, bw)
        if np.all(w == 0) or np.isnan(w).any():
            preds.append(np.nan)
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
    df = pd.DataFrame(index=y_level.index)
    df['level'] = y_level
    df['comp_ya'] = y_level.shift(months)
    df['roc_12m'] = y_level.pct_change(months)
    df['roc_24m_avg'] = (y_level.pct_change(24) / 24.0)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def bayesian_base_update(baseline_growth, base_effects_df, t, kappa=1.0, prior_var=0.0025, obs_var=0.0050):
    if t not in base_effects_df.index:
        return baseline_growth
    row = base_effects_df.loc[t]
    d12_prev = base_effects_df['roc_12m'].shift(1).loc[t]
    d24_prev = base_effects_df['roc_24m_avg'].shift(1).loc[t]
    if pd.isna(row['roc_12m']) and pd.isna(row['roc_24m_avg']):
        return baseline_growth
    d12 = (row['roc_12m'] - d12_prev) if pd.notna(row['roc_12m']) and pd.notna(d12_prev) else np.nan
    d24 = (row['roc_24m_avg'] - d24_prev) if pd.notna(row['roc_24m_avg']) and pd.notna(d24_prev) else np.nan
    candidates = [d for d in [d12, d24] if pd.notna(d)]
    delta = -kappa * (np.mean(candidates) if candidates else 0.0)

    # Bayesian blend: posterior mean of prior (baseline) and adjusted observation
    prior_mean = baseline_growth
    obs_mean = baseline_growth + delta
    post_var = 1.0 / (1.0/prior_var + 1.0/obs_var)
    post_mean = post_var * (prior_mean/prior_var + obs_mean/obs_var)
    return float(post_mean)

# -----------------------------
# Main app: true rolling backtest without lookahead
# -----------------------------
def growth_inflation_model():
    st.title("Real PCE Nowcast + Bayesian Base Effects (True Rolling Backtest)")

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

    # Enforce EOM alignment everywhere
    real_pce = monthly_eom(real_pce)
    grid_growth_variables = monthly_eom(grid_growth_variables)
    spx_monthly = monthly_eom(spx_monthly)

    # Variable mapping
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

    base_df = grid_growth_variables[list(chosen_growth_dict.keys())].copy()
    base_df.columns = [chosen_growth_dict[c] for c in base_df.columns]

    # Merge features and sanitize
    panel_raw = merge_dfs([base_df, spx_monthly]).sort_index()
    panel_raw = sanitize_panel(panel_raw)

    # Target level and growth (monthly)
    target = real_pce.copy()
    target.columns = ['real_pce']
    y_m1 = target['real_pce'].pct_change()

    # Build factor rates-of-change panel
    panel = rates_of_change_panel(panel_raw, win_short=3, win_long=6, z_win_mean=24, z_win_std=12)
    panel = sanitize_panel(panel)

    # Strict inner alignment (no future merging)
    merged = safe_inner_join_on_index([
        y_m1.to_frame('real_pce_m1'),
        panel
    ])
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(how='any').copy()

    # Define backtest target: predict y at time t using info up to t-1 (no lookahead)
    # So we align X at time t-1 with y at time t via shift(+1) on X when constructing rolling windows.
    # Create X_base (features) and y_true (target at same index as merged)
    X_base = merged.drop(columns=['real_pce_m1'])
    y_true = merged['real_pce_m1']

    # Rolling parameters
    window = 36

    # Precompute base effects on level series for Bayesian adjustment
    base_effects_df = compute_base_effects(target['real_pce'], months=12)

    preds_base = []
    preds_up = []
    preds_dn = []
    actuals = []
    eval_dates = []

    # Iterate t from window+1 to end so that training uses up to t-1 and predicts y_t
    for i in range(window + 1, len(merged)):
        # Training sample indices cover [i-window-1, i-1] in X aligned to those dates,
        # because X at date s represents information available at end of s (no future shift)
        train_idx = merged.index[i - window - 1 : i - 1]
        # Query index is t-1 in X to predict y at t
        query_idx = merged.index[i - 1]
        target_idx = merged.index[i]

        X_tr = X_base.loc[train_idx].copy()
        y_tr = y_true.loc[train_idx].copy()
        X_q = X_base.loc[[query_idx]].copy()

        # Defensive: drop any columns with NaNs in train or query
        cols_good = X_tr.columns[~X_tr.isna().any()].intersection(X_q.columns[~X_q.isna().any()])
        X_tr = X_tr[cols_good]
        X_q = X_q[cols_good]

        # Skip if insufficient features or NaNs remain
        if X_tr.empty or X_q.isna().any().any() or y_tr.isna().any():
            continue

        # Fit nonlinear tracker on train, predict for query
        yhat_arr, scaler, bw = time_series_kde_regression(X_tr, y_tr, X_q, cv_splits=5)
        yhat_tracker = float(yhat_arr[0])

        # In-sample sigma from re-predicting train
        yhat_in, _, _ = time_series_kde_regression(X_tr, y_tr, X_tr, cv_splits=3)
        sigma = residual_sigma(y_tr.values, yhat_in)

        # Bayesian adjustment uses information available up to target date t only:
        # base_effects_df at t is constructed from levels through t (which would not be known until t is realized).
        # To avoid lookahead when forming a forecast at end of t-1, use base effects as of t-1 (the most recent known).
        t_for_update = query_idx  # last known date when forecasting t
        yhat_bayes = bayesian_base_update(yhat_tracker, base_effects_df, t_for_update,
                                          kappa=1.0, prior_var=0.0025, obs_var=0.0050)

        preds_base.append(yhat_bayes)
        preds_up.append(yhat_bayes + sigma)
        preds_dn.append(yhat_bayes - sigma)
        actuals.append(y_true.loc[target_idx])
        eval_dates.append(target_idx)

    # Assemble results aligned to the realized target dates y_t
    results = pd.DataFrame({
        'Actual': actuals,
        'Base': preds_base,
        'Upside': preds_up,
        'Downside': preds_dn
    }, index=pd.DatetimeIndex(eval_dates)).dropna(how='any')

    # Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', line=dict(color="#222", width=2)))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Base'], name='Base', line=dict(color="#007AFF", width=2)))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Upside'], name='Upside', line=dict(color="#63dd63", width=1, dash='dash')))
    fig1.add_trace(go.Scatter(x=results.index, y=results['Downside'], name='Downside', line=dict(color="#ff6961", width=1, dash='dash')))
    fig1.update_layout(title="Real PCE: Nowcast + Bayesian Base Effects (True Rolling, No Lookahead)",
                       yaxis_title="1M % Change", hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

    # Metrics/Stats
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Base']))
    mae = mean_absolute_error(results['Actual'], results['Base'])
    st.write(f"**Overall RMSE:** {rmse:.4f}   **MAE:** {mae:.4f}")

    # Regime stats
    pre_covid = results.loc[:'2019-12-31']
    post_covid = results.loc['2022-01-01':]
    def safe_stats(df):
        if len(df) >= 6:
            return (np.sqrt(mean_squared_error(df['Actual'], df['Base'])),
                    mean_absolute_error(df['Actual'], df['Base']),
                    df['Actual'].std())
        return (np.nan, np.nan, np.nan)
    rmse_pre, mae_pre, std_pre = safe_stats(pre_covid)
    rmse_post, mae_post, std_post = safe_stats(post_covid)

    st.markdown(f"""#### Regime Stats
- Pre-COVID (to Dec 2019): RMSE={rmse_pre:.4f}, MAE={mae_pre:.4f}, StdDev={std_pre:.4f}
- Post-COVID (Jan 2022+): RMSE={rmse_post:.4f}, MAE={mae_post:.4f}, StdDev={std_post:.4f}
""")

    st.dataframe(results.tail(12))

