### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- MAG7 SAM CORE EQUITY ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

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
from sklearn.linear_model import LinearRegression


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

def calculate_regime_statistics(df, return_cols=['sp500_pct', 'bonds_pct']):
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
        regime_data = df[df['regime_label'] == regime]
        regime_returns = (regime_data[return_cols].mean() * 100).values

        results.append({
            'Quad': quad,
            'Regime': regime,
            'Regime Code': regime_short[regime],
            'Equities': regime_returns[0],
            'Bonds': regime_returns[1],
            '% of Occurrences': (len(regime_data) / len(df.dropna(subset=['regime_label']))) * 100
        })

    return pd.DataFrame(results)

def plot_streamlit_regime_statistics(stats_df):
    cmap = LinearSegmentedColormap.from_list(
        'red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256
    )
    styled = stats_df.style \
        .format({'Equities': "{:.2f}%", 'Bonds': "{:.2f}%", '% of Occurrences': "{:.2f}%"}) \
        .background_gradient(cmap=cmap, subset=['Equities', 'Bonds'])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)