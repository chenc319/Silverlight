### ----------------------------------------------------------------------------------------------- ###
### ------------------------------------------ FUNCTIONS ------------------------------------------ ###
### ----------------------------------------------------------------------------------------------- ###

### FUNCTIONS ###
import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
import pandas as pd
import functools as ft
import pickle
import pandas_datareader as pdr
import numpy as np
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression


def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

def static_beta(return_ts, benchmark_ts,):
    returns = merge_dfs([return_ts,benchmark_ts])
    rolling_cov = returns.iloc[:,0].cov(returns.iloc[:,1])
    rolling_var = returns.iloc[:,1].var()
    individual_beta = rolling_cov / rolling_var
    return individual_beta

def rolling_beta(return_ts, benchmark_ts,window):
    returns = merge_dfs([return_ts,benchmark_ts])
    rolling_cov = returns.iloc[:,0].rolling(window).cov(returns.iloc[:,1])
    rolling_var = returns.iloc[:,1].rolling(window).var()
    individual_beta = rolling_cov / rolling_var
    return individual_beta

def rolling_beta_sign(y, x, window, thresh=-0.2):
    """
    Return 1 if beta >= thresh, else -1.
    """
    beta = rolling_beta(y, x, window)
    sign = np.where(beta >= thresh, 1, -1)
    return pd.Series(sign, index=beta.index)

def return_metrics(backtest_returns_data, benchmark_data, ann_factor):
    backtest_returns_data = pd.DataFrame(backtest_returns_data)
    benchmark_data = pd.DataFrame(benchmark_data)
    return_metrics_df = pd.DataFrame(
        columns=['Total Return', 'Avg Return', 'Avg Upside Return', 'Avg Downside Return',
                 'Win Ratio', 'Ann. Return', 'Ann. Volatility', 'Return/Risk',
                 'Max Return', 'Min Return',
                 'Upside Capture', 'Downside Capture', 'Capture Ratio','Beta']
    )
    benchmark_returns = benchmark_data.iloc[:,0].ffill().dropna()

    for x in range(0, len(backtest_returns_data.columns)):
        col = backtest_returns_data.columns[x]
        data = pd.DataFrame(backtest_returns_data[col]).ffill().dropna()
        data.columns = ['returns']
        total_return = ((1 + data['returns']).cumprod()-1)[-1]
        mean_return = data['returns'].mean()
        avg_win_return = data[data['returns'] > 0]['returns'].mean()
        avg_lose_return = data[data['returns'] < 0]['returns'].mean()
        win_ratio = len(data[data['returns'] > 0]) / len(data)
        ann_return = ((1+ mean_return) ** ann_factor)-1
        ann_vol = data['returns'].std() * (ann_factor ** 0.5)
        return_risk = ann_return / ann_vol if ann_vol != 0 else None
        max_return = data['returns'].max()
        min_return = data['returns'].min()

        # Upside/Downside Capture
        upside_mask = benchmark_returns > 0
        downside_mask = benchmark_returns < 0
        upside_capture = (data['returns'][upside_mask].mean() / benchmark_returns[upside_mask].mean()) if upside_mask.any() else None
        downside_capture = (data['returns'][downside_mask].mean() / benchmark_returns[downside_mask].mean()) if downside_mask.any() else None

        # Capture Ratio: Upside / |Downside|
        capture_ratio = (upside_capture / abs(downside_capture)) if (upside_capture is not None and downside_capture not in (None, 0)) else None

        beta = static_beta(data['returns'],benchmark_data)

        return_metrics_df.loc[col] = [
            total_return, mean_return, avg_win_return, avg_lose_return, win_ratio,
            ann_return, ann_vol, return_risk, max_return, min_return,
            upside_capture, downside_capture, capture_ratio,beta
        ]
    return return_metrics_df

def return_metrics_by_regime(base_df, return_col, benchmark_col, regime_col='regime_label', ann_factor=12):
    """
    Computes return metrics for each unique regime label in the base DataFrame.

    Args:
        base_df: pd.DataFrame. Must include regime_col, return_col, benchmark_col.
        return_col: str. Name of the column to use as the individual strategy stream.
        benchmark_col: str. Name of the column to use for the benchmark stream.
        regime_col: str. The column containing the regime labels (default 'regime_label').
        ann_factor: int. Annualization factor for metrics (default 12 for monthly).

    Returns:
        pd.DataFrame. Regime-by-regime metrics, indexed by regime label.
    """

    unique_regimes = base_df[regime_col].dropna().unique()
    metrics_list = []

    for regime in unique_regimes:
        mask = base_df[regime_col] == regime
        regime_returns = base_df.loc[mask, return_col]
        benchmark_returns = base_df.loc[mask, benchmark_col]

        metric_df = return_metrics(
            pd.DataFrame({str(regime): regime_returns}),
            pd.DataFrame({benchmark_col: benchmark_returns}),
            ann_factor
        )
        # metric_df will have index=str(regime), so we pull out the values as a dict and add regime
        metrics_dict = metric_df.loc[str(regime)].to_dict()
        metrics_dict['Regime'] = regime
        metrics_list.append(metrics_dict)

    # Recombine into final result DataFrame
    regime_metrics_df = pd.DataFrame(metrics_list)
    regime_metrics_df.set_index('Regime', inplace=True)
    return regime_metrics_df



def posneg_only_red_green(val, min_pos, max_pos, min_neg, max_neg):
    if pd.isnull(val):
        return 'background-color: rgb(240,255,240); color: black'
    if val > 0:
        # Strictly green gradient: lightest = min_pos, strongest = max_pos
        ratio = 0 if max_pos == min_pos else (val - min_pos) / (max_pos - min_pos)
        # From light green to green: (230,255,230) -> (0,180,0)
        r = int(230 - 230 * ratio)
        g = int(255 - 75 * ratio)
        b = int(230 - 230 * ratio)
        return f'background-color: rgb({r},{g},{b}); color: black'
    elif val < 0:
        # Strictly red gradient: lightest = max_neg, strongest = min_neg
        ratio = 0 if min_neg == max_neg else (val - max_neg) / (min_neg - max_neg)
        # From light red to red: (255,230,230) -> (255,0,0)
        r = 255
        g = int(230 - 230 * ratio)
        b = int(230 - 230 * ratio)
        return f'background-color: rgb({r},{g},{b}); color: black'
    else:
        # Zero assigned very pale green
        return 'background-color: rgb(240,255,240); color: black'


def streamlit_return_metrics_table(df):
    fmt_dict = {
        'Total Return': '{:,.2%}',
        'Avg Return': '{:,.2%}',
        'Avg Upside Return': '{:.2%}',
        'Avg Downside Return': '{:.2%}',
        'Win Ratio': '{:.2%}',
        'Ann. Return': '{:.2%}',
        'Ann. Volatility': '{:.2%}',
        'Return/Risk': '{:.2f}',
        'Max Return': '{:.2%}',
        'Min Return': '{:.2%}',
        'Upside Capture': '{:.2%}',
        'Downside Capture': '{:.2%}',
        'Capture Ratio': '{:.2f}',
        'Beta': '{:.2f}'
    }
    styler = df.style.format(fmt_dict)

    for col in df.columns:
        vals = df[col].dropna()
        pos = vals[vals > 0]
        neg = vals[vals < 0]
        min_pos = pos.min() if len(pos) else 0.001
        max_pos = pos.max() if len(pos) else 1
        min_neg = neg.min() if len(neg) else -1
        max_neg = neg.max() if len(neg) else -0.001

        def style_cell(val, mp=min_pos, xp=max_pos, mn=min_neg, xn=max_neg):
            return posneg_only_red_green(val, mp, xp, mn, xn)

        styler = styler.applymap(style_cell, subset=[col])

    return st.dataframe(styler)

def compute_drawdown(daily_returns):
    # Assume daily_returns is a pandas Series or numpy array of percent returns, e.g., 0.01 for 1%
    cumret = (1 + pd.Series(daily_returns)).cumprod()
    roll_max = cumret.cummax()
    drawdown = cumret / roll_max - 1
    return drawdown

def streamlit_drawdown_plot(df,
                            graph_labels,
                            df_columns_to_plot,
                            line_colors,
                            fill_colors):
    fig = go.Figure()
    for col, line, fill, label in zip(df_columns_to_plot, line_colors, fill_colors, graph_labels):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=label,
            line=dict(color=line, width=2),
            fill='tozeroy',
            fillcolor=fill,
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Drawdown: %{{y:.2%}}<extra></extra>",
            showlegend=True
        ))
    fig.update_layout(
        title="Drawdown Analysis",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat='.0%',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor='bottom', y=1.02, xanchor='center', x=0.5, title=None),
        margin=dict(l=40, r=40, t=70, b=40),
        plot_bgcolor='#f9f9f9'
    )
    st.plotly_chart(fig, use_container_width=True)

def streamlit_plot(df,columns_array,colors_array,graph_title,y_axis_label):
    fig = go.Figure()
    for name, color in zip(columns_array, colors_array):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[name],
            name=name,
            mode='lines',
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title=graph_title,
        yaxis_title=y_axis_label
    )
    st.plotly_chart(fig, use_container_width=True)

def streamlit_spread_plot(df, columns_array, graph_title, y_axis_label):
    fig = go.Figure()
    for name in columns_array:
        y = df[name]
        x = df.index

        # Main black line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            name=name,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=True
        ))

        # Positive shading (light green)
        fig.add_trace(go.Scatter(
            x=x,
            y=y.where(y > 0, 0),
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            fill='tozeroy',
            fillcolor='rgba(144,238,144,0.5)',  # Light green
            showlegend=False,
            hoverinfo='skip'
        ))

        # Negative shading (light red)
        fig.add_trace(go.Scatter(
            x=x,
            y=y.where(y < 0, 0),
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255,182,193,0.5)',  # Light red
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title=graph_title,
        yaxis_title=y_axis_label,
        template='plotly_white',
        plot_bgcolor='#f9f9f9'
    )
    st.plotly_chart(fig, use_container_width=True)


def streamlit_subplot(df, columns_array, colors_array, row_nums, col_nums):
    fig = sp.make_subplots(rows=row_nums, cols=col_nums, subplot_titles=columns_array)
    for i, col in enumerate(columns_array):
        row = i // col_nums + 1
        col_pos = i % col_nums + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=colors_array[i % len(colors_array)], width=2)
            ),
            row=row,
            col=col_pos
        )
    fig.update_layout(
        showlegend=False,
        height=800,
        width=1000
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_regime_return_histograms(df, regime_col, return_col, regimes):
    """
    Plots subplot histograms for a set of regimes' return distributions, with KDE overlay for each regime.

    Args:
        df: pd.DataFrame containing your data.
        regime_col: str, name of regime label column.
        return_col: str, name of return column.
        regimes: list of regime names, defines subplot order.
    """
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde, kurtosis
    import numpy as np

    subplot_titles = []
    for regime in regimes:
        subdata = df[df[regime_col] == regime][return_col].dropna()
        k_value = kurtosis(subdata, fisher=True, nan_policy='omit')
        subplot_titles.append(f"{regime} (kurt={k_value:.2f})")

    default_colors = ['#28a745', '#90ee90', '#dc3545', '#ffc107']
    regime_colors = {regimes[i]: default_colors[i % len(default_colors)] for i in range(len(regimes))}

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.20,
    )
    min_bound = df[return_col].min()
    max_bound = df[return_col].max()
    x_grid = np.linspace(min_bound, max_bound, 300)

    for i, regime in enumerate(regimes):
        row = i // 2 + 1
        col = i % 2 + 1
        subdata = df[df[regime_col] == regime][return_col].dropna()
        # Plot histogram
        fig.add_trace(
            go.Histogram(
                x=subdata,
                name=f"{regime} hist",
                marker=dict(
                    color=regime_colors[regime],
                    line=dict(
                        width=1,
                        color='#444'   # Subtle separation
                    )
                ),
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=row,
            col=col
        )
        # KDE overlay
        if len(subdata) > 1:
            kde = gaussian_kde(subdata)
            density = kde(x_grid) * len(subdata) * (x_grid[1] - x_grid[0])  # Scale to histogram
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=density,
                    mode='lines',
                    line=dict(color='navy', width=2),
                    name=f"{regime} KDE",
                    showlegend=False
                ),
                row=row,
                col=col
            )
        fig.update_xaxes(
            title_text="Equity % Return",
            row=row,
            col=col,
            range=[min_bound, max_bound],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(190, 190, 190, 0.2)',
            ticks="outside",
            tickfont=dict(size=12, family='Arial'),
        )
        fig.update_yaxes(
            title_text="Count",
            row=row,
            col=col,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(190, 190, 190, 0.2)',
            ticks="outside",
            tickfont=dict(size=12, family='Arial'),
        )
    fig.update_layout(
        showlegend=False,
        height=750,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14, color="#222"),
        margin=dict(l=60, r=60, t=80, b=60),
        title=dict(
            text="Regime Return Distributions",
            font=dict(size=18, family="Arial"),
            x=0.5,
            y=0.98,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

def color_coded_regime_plot(df, y_col, regime_col,
                                  title="Asset Price by Regime"):
    df[regime_col] = df[regime_col].astype('category')
    regime_colors = {
        "Goldilocks": "#28a745",  # Green
        "Reflation": "#90ee90",  # Light green
        "Stagflation": "#ffc107",  # Yellow
        "Deflation": "#dc3545"  # Red
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[y_col],
        mode='lines',
        line=dict(color='black', width=2),
        name=y_col,
        showlegend=False,
        hoverinfo='skip'
    ))
    for regime, color in regime_colors.items():
        mask = df[regime_col] == regime
        fig.add_trace(go.Scatter(
            x=df.index[mask],
            y=df[y_col][mask],
            mode='markers',
            marker=dict(color=color, size=8),
            name=regime,
            showlegend=True,
            hovertemplate=f"Regime: {regime}<br>{y_col}: %{{y}}<br>Date: %{{x}}<extra></extra>"
        ))
    fig.update_layout(
        title=title,
        hovermode='closest',
        legend=dict(title='Regime', orientation='h', y=-0.15),
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_regime_statistics(df, regime_col_name, return_cols=['sp500_pct']):
    regimes = ['Goldilocks', 'Reflation', 'Stagflation', 'Deflation']
    quad_labels = ['Quad 1', 'Quad 2', 'Quad 3', 'Quad 4']
    regime_short = {
        'Goldilocks': 'I-G+',
        'Reflation': 'I+G+',
        'Stagflation': 'I+G-',
        'Deflation': 'I-G-'
    }
    results = []

    for regime, quad in zip(regimes, quad_labels):
        regime_data = df[df[regime_col_name] == regime]
        row = {
            'Quad': quad,
            'Regime': regime,
            'Regime Code': regime_short[regime],
            '% of Occurrences': (len(regime_data) / len(df.dropna(subset=[regime_col_name]))) * 100
        }

        # Each asset is a separate column: key = asset, value = mean return
        for asset in return_cols:
            asset_mean = regime_data[asset].mean() * 100 if not regime_data.empty else float('nan')
            row[asset] = asset_mean

        results.append(row)

    return pd.DataFrame(results)

def plot_streamlit_regime_statistics(stats_df):
    asset_cols = [col for col in stats_df.columns if col not in ('Quad', 'Regime', 'Regime Code', '% of Occurrences')]
    cmap = LinearSegmentedColormap.from_list(
        'red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256
    )
    styled = stats_df.style \
        .format({col: "{:.2f}%" for col in asset_cols + ['% of Occurrences']}) \
        .background_gradient(cmap=cmap, subset=asset_cols)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)


def z_score_bucket(df, z_col, target_col):
    l2 = df[df[z_col] < -2][target_col].mean()
    l1g2 = df[(df[z_col] < -1) & (df[z_col] > -2)][target_col].mean()
    l0g1 = df[(df[z_col] < 0) & (df[z_col] > -1)][target_col].mean()
    g0l1 = df[(df[z_col] < 1) & (df[z_col] > 0)][target_col].mean()
    g1l2 = df[(df[z_col] < 2) & (df[z_col] > 1)][target_col].mean()
    g2 = df[df[z_col] > 2][target_col].mean()

    l5 = df[df[z_col] < -0.5][target_col].mean()
    g5 = df[df[z_col] > 0.5][target_col].mean()

    z_score_df = pd.DataFrame()
    z_score_df['buckets'] = ['<-2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '>2', '<-0.5', '>0.5']
    z_score_df['mean_returns'] = [l2, l1g2, l0g1, g0l1, g1l2, g2, l5, g5]
    print(z_score_df)
    return (z_score_df)

### ----------------------------------------------------------------------------------------------- ###
### -------------------------------------- DEMARK FUNCTIONS --------------------------------------- ###
### ----------------------------------------------------------------------------------------------- ###

### FUNCTIONS ###
def close_hl_setup(df,close_col_name):
    df['h/l'] = np.nan
    for idx in range(4,len(df)):
        row_name = df.index[idx]
        row_4 = df.index[idx-4]
        if df.loc[row_name,close_col_name] > df.loc[row_4,close_col_name]:
            df.loc[row_name, 'h/l'] = 'h'
        else:
            df.loc[row_name, 'h/l'] = 'l'
    return df['h/l']

def td_combo_setup(df,setup_type):
    ### CONSECUTIVE DAYS ###
    df['setup'] = 0
    for idx in range(1,len(df.index)):
        row_name = df.index[idx]
        prev_row = df.index[idx-1]
        if df.loc[prev_row,'setup'] != 9:
            if df.loc[row_name, 'h/l'] == setup_type:
                df.loc[row_name, 'setup'] = df.loc[prev_row, 'setup'] + 1

    ### FILTER OUT INCOMPLETE 9S ###
    for idx in range(8,len(df.index)):
        row_name = df.index[idx]
        prev_row = df.index[idx - 1]
        if ((df.loc[row_name, 'setup'] < df.loc[prev_row, 'setup'])
                and (df.loc[prev_row, 'setup'] != 9)):
            total_rows_to_delete = df.loc[prev_row, 'setup']
            for num_row_to_delete in range(0,total_rows_to_delete+1):
                df.loc[df.index[idx-num_row_to_delete],'setup'] = 0
    return df['setup']

def build_td_combo_v2_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single 'setup' column:
      +1..+9 = completed Sell setup
      -1..-9 = completed Buy setup
       0     = no completed setup

    Uses existing df['h/l'] with 'h' / 'l' values.
    Enforces:
      - Bullish flip (l -> h) required before a Sell setup can begin.
      - Bearish flip (h -> l) required before a Buy setup can begin.
      - Flip must occur AFTER the previous completed setup (of any side).
      - Only fully completed 1–9 / -1–-9 sequences are printed; partial runs are discarded.
    """
    df = df.copy()

    side = df['h/l'].map({'h': 1, 'l': -1}).fillna(0)

    bull_flip = (side.shift(1) == -1) & (side == 1)   # l -> h
    bear_flip = (side.shift(1) == 1) & (side == -1)   # h -> l

    # Track “eligibility windows” for starting a new setup of each type
    # after an appropriate flip has occurred.
    last_bull_flip_idx = None
    last_bear_flip_idx = None
    last_completed_setup_idx = None   # index of bar 9 of the last completed setup

    # This will hold only completed sequences.
    df['setup'] = 0

    idx_list = df.index.to_list()
    n = len(idx_list)

    i = 0
    while i < n:
        idx = idx_list[i]

        # Update flip trackers
        if bull_flip.iloc[i]:
            last_bull_flip_idx = idx
        if bear_flip.iloc[i]:
            last_bear_flip_idx = idx

        # Decide if we *can* start a new setup here and in which direction.
        # Conditions:
        #   1) Side at this bar must be 'h' (sell) or 'l' (buy).
        #   2) There must be a flip of the correct type AFTER the last completed setup.
        #   3) That flip must occur at or before this bar.
        direction = 0

        if side.iloc[i] == 1:
            # Candidate Sell bar (needs bullish flip since last completed setup)
            if last_bull_flip_idx is not None:
                if (last_completed_setup_idx is None) or (last_bull_flip_idx > last_completed_setup_idx):
                    # The bullish flip that enables this sell setup window is after the last completed setup.
                    # Setup can start on any later 'h' bar.
                    direction = +1

        elif side.iloc[i] == -1:
            # Candidate Buy bar (needs bearish flip since last completed setup)
            if last_bear_flip_idx is not None:
                if (last_completed_setup_idx is None) or (last_bear_flip_idx > last_completed_setup_idx):
                    direction = -1

        # If we cannot start a setup here, just advance.
        if direction == 0:
            i += 1
            continue

        # Attempt to build a full 9‑bar run starting at i.
        ok = True
        if i + 9 > n:
            ok = False
        else:
            for k in range(9):
                j = i + k
                if direction == +1:
                    # Sell setup requires 'h' (close > close[-4])
                    if side.iloc[j] != 1:
                        ok = False
                        break
                else:
                    # Buy setup requires 'l' (close < close[-4])
                    if side.iloc[j] != -1:
                        ok = False
                        break

        if ok:
            # Completed setup: write +1..+9 or -1..-9 into df['setup'].
            for k in range(1, 10):
                j = i + (k - 1)
                df.at[idx_list[j], 'setup'] = direction * k

            # Mark completion index; new setups of either side must wait for a new flip after this.
            last_completed_setup_idx = idx_list[i + 8]

            # Jump past the completed sequence.
            i = i + 9
        else:
            # Not a valid 9‑bar run; do nothing, leave setup as 0 and step forward.
            i += 1

    return df

def get_perfected_9(df):
    df['perfected'] = None

    # All 9s of either side
    idx9 = df.index[(df['setup'] == -9) | (df['setup'] == 9)]

    for i in idx9:
        pos = df.index.get_loc(i)
        if pos < 8:
            continue

        idx6 = df.index[pos - 3]  # bar 6
        idx7 = df.index[pos - 2]  # bar 7
        idx8 = df.index[pos - 1]  # bar 8
        idx9_bar = df.index[pos]  # bar 9

        if df.at[idx9_bar, 'setup'] == -9:
            # Buy perfected rule: low(8 or 9) <= lows of 6 and 7
            low6 = df.at[idx6, 'Low']
            low7 = df.at[idx7, 'Low']
            low8 = df.at[idx8, 'Low']
            low9 = df.at[idx9_bar, 'Low']

            cond = ((low8 <= low6 and low8 <= low7) or
                    (low9 <= low6 and low9 <= low7))
        else:
            # Sell perfected rule: high(8 or 9) >= highs of 6 and 7
            high6 = df.at[idx6, 'High']
            high7 = df.at[idx7, 'High']
            high8 = df.at[idx8, 'High']
            high9 = df.at[idx9_bar, 'High']

            cond = ((high8 >= high6 and high8 >= high7) or
                    (high9 >= high6 and high9 >= high7))

        if cond:
            df.at[idx9_bar, 'perfected'] = "Perfected"
    return df['perfected']

def compute_setup_countdown_pairs(df):
    """
    Uses unified 'setup' column:
      -1..-9 = Buy setup
      +1..+9 = Sell setup

    For each setup start (setup == -1 or +1), runs the
    corresponding TD Combo-style countdown into 'countdown'.

    Returns dict of setup-countdown pairs (for both sides).
    """
    df = df.copy()

    setup_col     = "setup"
    countdown_col = "countdown"
    df[countdown_col] = 0

    # Setup starts: -1 (buy) or +1 (sell)
    setup_rows_to_iterate = df[(df[setup_col] == -1) | (df[setup_col] == 1)].index

    setup_countdown_dict = dict()
    setup_dict_key_index = 1

    for setup_start in setup_rows_to_iterate:

        n = len(df.index)
        setup_pos = df.index.get_loc(setup_start)

        # need one bar before and 8 bars after: indices [setup_pos-1 .. setup_pos+8]
        if setup_pos - 1 < 0 or setup_pos + 8 >= n:
            continue

        # ----- DETECT SIDE FROM SIGN -----
        start_val = df.at[setup_start, setup_col]
        side = "buy" if start_val < 0 else "sell"

        # ---------- TDST ----------
        prev_bar = df.index[setup_pos - 1]
        bar9     = df.index[setup_pos + 8]

        tdst_win   = df.loc[prev_bar:bar9].copy()
        prev_close = tdst_win['Close'].shift(1)

        if side == "buy":
            tdst_win['true_high'] = tdst_win['High'].combine(prev_close, max)
            tdst_win = tdst_win.iloc[1:]  # drop first row with NaN prev_close

            tdst_val  = tdst_win['true_high'].max()
            tdst_date = tdst_win['true_high'].idxmax()
        else:
            tdst_win['true_low'] = tdst_win['Low'].combine(prev_close, min)
            tdst_win = tdst_win.iloc[1:]

            tdst_val  = tdst_win['true_low'].min()
            tdst_date = tdst_win['true_low'].idxmin()

        tdst_setup_num = tdst_win.loc[tdst_date, setup_col]

        # First TDST break
        if side == "buy":
            first_tdst_break = (df.loc[setup_start:, 'Close'] > tdst_val).idxmax()
        else:
            first_tdst_break = (df.loc[setup_start:, 'Close'] < tdst_val).idxmax()

        # ---------- COUNTDOWN ----------
        # (Note: this reinitializes countdown for each setup window, just like your original.)
        df[countdown_col] = 0
        setup_idx    = setup_pos
        cdn_num      = 1
        last_cdn_bar = None

        for i in range(setup_idx, len(df.index)):
            if cdn_num > 13:
                break

            row    = df.index[i]
            row_t1 = df.index[i - 1]
            row_t2 = df.index[i - 2]

            c = df.loc[row, 'Close']
            l = df.loc[row, 'Low']
            h = df.loc[row, 'High']
            o = df.loc[row, 'Open']

            if side == "buy":
                # ---------- BUY COUNTDOWN ----------
                if cdn_num <= 10:
                    base_ok = (
                        c <= df.loc[row_t2, 'Low'] and
                        l <= df.loc[row_t1, 'Low'] and
                        c <= df.loc[row_t1, 'Close']
                    )
                    if not base_ok:
                        continue

                    if last_cdn_bar is None or cdn_num == 1:
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1
                        continue

                    if c < df.loc[last_cdn_bar, 'Close']:
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1

                else:
                    if (10 < cdn_num < 13 and last_cdn_bar is not None
                            and (c < df.loc[last_cdn_bar, 'Close'])):
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1
                    elif (cdn_num == 13 and last_cdn_bar is not None
                          and (c < df.loc[last_cdn_bar, 'Close']
                               or o < df.loc[last_cdn_bar, 'Close'])):
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1

            else:
                # ---------- SELL COUNTDOWN (mirrored) ----------
                if cdn_num <= 10:
                    base_ok = (
                        c >= df.loc[row_t2, 'High'] and
                        h >= df.loc[row_t1, 'High'] and
                        c >= df.loc[row_t1, 'Close']
                    )
                    if not base_ok:
                        continue

                    if last_cdn_bar is None or cdn_num == 1:
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1
                        continue

                    if c > df.loc[last_cdn_bar, 'Close']:
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1

                else:
                    if (10 < cdn_num < 13 and last_cdn_bar is not None
                            and (c > df.loc[last_cdn_bar, 'Close'])):
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1
                    elif (cdn_num == 13 and last_cdn_bar is not None
                          and (c > df.loc[last_cdn_bar, 'Close']
                               or o > df.loc[last_cdn_bar, 'Close'])):
                        df.loc[row, countdown_col] = cdn_num
                        last_cdn_bar = row
                        cdn_num += 1

        # ---------- STORE DATA ----------
        window = df.loc[setup_start:first_tdst_break]
        name   = f"setup_countdown #{setup_dict_key_index}"
        setup_dict_key_index += 1

        result = {
            'side': side,
            'tdst_val': tdst_val,
            'tdst_date': tdst_date,
            'tdst_setup_start_bar': tdst_setup_num,
        }

        if cdn_num == 14 and last_cdn_bar is not None:
            risk_win = window.loc[setup_start:last_cdn_bar].copy()

            if len(risk_win) <= 1:
                result.update({
                    'valid setup-cdn_pair': 'no',
                    'dataframe': window,
                    'risk_lvl':  None,
                    'risk_date': None,
                })
            else:
                prev_close = risk_win['Close'].shift(1)
                risk_win['true_high']  = risk_win['High'].combine(prev_close, max)
                risk_win['true_low']   = risk_win['Low'].combine(prev_close, min)
                risk_win['true_range'] = risk_win['true_high'] - risk_win['true_low']
                risk_win = risk_win.iloc[1:]

                if side == "buy":
                    risk_date  = risk_win['true_low'].idxmin()
                    risk_low   = risk_win.loc[risk_date, 'true_low']
                    risk_range = risk_win.loc[risk_date, 'true_range']
                    risk_val   = risk_low - risk_range
                else:
                    risk_date  = risk_win['true_high'].idxmax()
                    risk_high  = risk_win.loc[risk_date, 'true_high']
                    risk_range = risk_win.loc[risk_date, 'true_range']
                    risk_val   = risk_high + risk_range

                result.update({
                    'valid setup-cdn_pair': 'yes',
                    'dataframe': window.loc[setup_start:last_cdn_bar],
                    'risk_lvl':  risk_val,
                    'risk_date': risk_date,
                })
        else:
            result.update({
                'valid setup-cdn_pair': 'no',
                'dataframe': window,
                'risk_lvl':  None,
                'risk_date': None,
            })

        setup_countdown_dict[name] = result

    return setup_countdown_dict


def plot_td_combo_case_study(setup_countdown_dict,index_val):
    df = setup_countdown_dict[list(setup_countdown_dict.keys())[index_val]]['dataframe']
    df['Date'] = df.index
    df['DateStr'] = df.index.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)

    # Create series that span the whole visible window
    df['TDST'] = setup_countdown_dict[list(setup_countdown_dict.keys())[index_val]]['tdst_val']
    df['TD_Risk'] = setup_countdown_dict[list(setup_countdown_dict.keys())[index_val]]['risk_lvl']

    # --- 3. Base OHLC candlestick chart over ALL rows ---
    fig = go.Figure()

    # 1) Candles over ALL bars, x = DateStr (categorical, no gaps)
    fig.add_trace(
        go.Ohlc(
            x=df['DateStr'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='white',
            decreasing_line_color='white',
            name='Price'
        )
    )

    # 2) Setup labels
    setup_mask = df['setup'] > 0
    fig.add_trace(
        go.Scatter(
            x=df.loc[setup_mask, 'DateStr'],
            y=df.loc[setup_mask, 'High'] * 1.01,
            mode='text',
            text=df.loc[setup_mask, 'setup'].astype(int).astype(str),
            textfont=dict(color='lime', size=12),
            textposition='top center',
            name='TD Setup'
        )
    )

    # 3) Countdown labels
    cd_mask = df['countdown'] > 0
    fig.add_trace(
        go.Scatter(
            x=df.loc[cd_mask, 'DateStr'],
            y=df.loc[cd_mask, 'Low'] * 0.99,
            mode='text',
            text=df.loc[cd_mask, 'countdown'].astype(int).astype(str),
            textfont=dict(color='magenta', size=12),
            textposition='bottom center',
            name='TD Countdown'
        )
    )

    # 4) Single TDST + TD Risk lines
    fig.add_trace(
        go.Scatter(
            x=df['DateStr'],
            y=df['TDST'],
            mode='lines',
            line=dict(color='limegreen', width=1.5),
            name='TD TDST Level'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['DateStr'],
            y=df['TD_Risk'],
            mode='lines',
            line=dict(color='magenta', width=1.5, dash='dot'),
            name='TD Risk Level'
        )
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            type='category',      # ensure categorical axis (no date gaps)
            rangeslider_visible=False
        ),
        yaxis=dict(showgrid=False),
        height=700,
        width=1200
    )

    st.plotly_chart(fig, use_container_width=True)

