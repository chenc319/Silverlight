### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- REAL PCE BRIDGE NOWCAST ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###
### PACKAGES ###
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error



def render_real_pce_bridge_nowcast(DATA_DIR='data', window=60, alpha=1):


    ### ------------------------------------ HELPERS ------------------------------------ ###
    def monthly_eom(df):
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index(pd.to_datetime(df['Date'])).drop(columns=['Date'])
            else:
                df.index = pd.to_datetime(df.index)
        return df.resample('ME').last()

    def sanitize(df):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill(limit=2).bfill(limit=2)
        nunique = df.nunique(dropna=True)
        return df.loc[:, nunique[nunique > 1].index].dropna(how='all')

    def growth_features(panel, cols, m_changes=(1,3)):
        out = pd.DataFrame(index=panel.index)
        for c in cols:
            s = panel[c].astype(float)
            for m in m_changes:
                out[f'{c}_d{m}'] = s.pct_change(m) / m
        return out

    def build_dataset(real_pce, indicators, sel_cols):
        y = real_pce['real_pce'].pct_change()
        X_base = growth_features(indicators, sel_cols, m_changes=(1,3))
        X = X_base.copy()
        X['y_lag1'] = y.shift(1)
        X['y_lag2'] = y.shift(2)
        df = pd.concat([y.rename('y'), X], axis=1)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
        return df

    def rolling_nowcast(df, window=36, alpha=0.5):
        dates = df.index
        y_true, y_pred, eval_dates = [], [], []
        scaler = StandardScaler()
        for i in range(window + 1, len(df)):
            train = df.iloc[i - window - 1: i - 1]   # info up to t-1
            query = df.iloc[[i - 1]]                  # features at t-1
            target = df.iloc[i]['y']                  # realized y at t

            X_tr = train.drop(columns=['y']).values
            y_tr = train['y'].values
            X_q = query.drop(columns=['y']).values

            Xs = scaler.fit_transform(X_tr)
            Xq = scaler.transform(X_q)
            mdl = Ridge(alpha=alpha, fit_intercept=True)
            mdl.fit(Xs, y_tr)
            yhat = float(mdl.predict(Xq)[0])

            y_true.append(target)
            y_pred.append(yhat)
            eval_dates.append(dates[i])

        res = pd.DataFrame({'Actual': y_true, 'Pred': y_pred}, index=pd.DatetimeIndex(eval_dates))
        res['DirOK'] = np.sign(res['Actual']) == np.sign(res['Pred'])
        rmse = float(np.sqrt(mean_squared_error(res['Actual'], res['Pred'])))
        mda = float(res['DirOK'].mean())
        return res, rmse, mda

    ### ------------------------------------ DATA PULL ------------------------------------ ###
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as f:
        spx = pd.read_csv(f)
    spx = spx.set_index(pd.to_datetime(spx['Date']))[['Close']].rename(columns={'Close':'spx'})
    spx = monthly_eom(spx)

    with open(Path(DATA_DIR) / 'real_pce.pkl', 'rb') as f:
        real_pce = pd.read_pickle(f)
    if real_pce.shape[1] == 1:
        real_pce.columns = ['real_pce']
    elif 'PCEC96' in real_pce.columns:
        real_pce = real_pce[['PCEC96']].rename(columns={'PCEC96':'real_pce'})
    real_pce = monthly_eom(real_pce)

    with open(Path(DATA_DIR) / 'grid_growth_variables.pkl', 'rb') as f:
        grid = pd.read_pickle(f)
    grid = monthly_eom(grid)

    # Small, strong indicator set
    sel = [
        'INDPRO',               # industrial production
        'PAYEMS',               # payrolls
        'RSXFS',                # retail sales (adv retail trade)
        'USALOLITOAASTSAM',     # OECD CLI
        'TOTALSA'               # light vehicle sales
    ]
    panel = grid[sel].copy()
    panel.columns = ['indpro','payems','retail','cli','autos']
    panel = pd.concat([panel, spx.rename(columns={'spx':'spx'})], axis=1)

    # Clean and align
    real_pce = sanitize(real_pce)
    panel = sanitize(panel)
    idx = real_pce.index.intersection(panel.index)
    real_pce = real_pce.loc[idx]
    panel = panel.loc[idx]

    ### ------------------------------------ MODEL + DISPLAY ------------------------------------ ###
    df = build_dataset(real_pce, panel, ['indpro','payems','retail','cli','autos','spx'])
    if len(df) < window + 24:
        st.warning("Not enough data after cleaning to run the rolling backtest. Consider reducing the window.")
        return pd.DataFrame()

    results, rmse, mda = rolling_nowcast(df, window=window, alpha=alpha)

    # Plot: Actual vs Pred
    st.title("Real PCE Nowcast: Simple Bridge (Ridge, True Rolling)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', line=dict(color="#222", width=2)))
    fig.add_trace(go.Scatter(x=results.index, y=results['Pred'],   name='Pred',   line=dict(color="#007AFF", width=2)))
    fig.update_layout(yaxis_title="1M % Change", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.write(f"**RMSE:** {rmse:.4f}   **MDA (Directional Accuracy):** {mda:.3f}")

    # Regime stats
    pre_covid = results.loc[:'2019-12-31']
    post_covid = results.loc['2022-01-01':]
    def stats_block(df_):
        if len(df_) >= 6:
            return np.sqrt(mean_squared_error(df_['Actual'], df_['Pred'])), df_['DirOK'].mean(), df_['Actual'].std()
        return (np.nan, np.nan, np.nan)
    rmse_pre, mda_pre, std_pre = stats_block(pre_covid)
    rmse_post, mda_post, std_post = stats_block(post_covid)
    st.markdown(f"""#### Regime Stats
- Pre-COVID (to Dec 2019): RMSE={rmse_pre:.4f}, MDA={mda_pre:.3f}, StdDev={std_pre:.4f}
- Post-COVID (Jan 2022+): RMSE={rmse_post:.4f}, MDA={mda_post:.3f}, StdDev={std_post:.4f}
""")

    st.dataframe(results.tail(12))
    return results
