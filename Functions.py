### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- MAG7 SAM CORE EQUITY ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### FUNCTIONS ###
import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
import pandas as pd
import functools as ft

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
        columns = ['Total Return',
                   'Avg Return',
                   'Avg Upside Return',
                   'Avg Downside Return',
                   'Win Ratio',
                   'Ann. Return',
                   'Ann. Volatility',
                   'Return/Risk',
                   'Max Return',
                   'Min Return',
                   'Upside Capture',
                   'Downside Capture',
                   'Beta']
    )
    benchmark_returns = benchmark_data.iloc[:,0].ffill().dropna()

    for x in range(0, len(backtest_returns_data.columns)):
        col = backtest_returns_data.columns[x]
        data = pd.DataFrame(backtest_returns_data[col]).ffill().dropna()
        data.columns = ['returns']
        total_return = data['returns'].sum()
        mean_return = data['returns'].mean()
        avg_win_return = data[data['returns'] > 0]['returns'].mean()
        avg_lose_return = data[data['returns'] < 0]['returns'].mean()
        win_ratio = len(data[data['returns'] > 0]) / len(data)
        ann_return = mean_return * ann_factor
        ann_vol = data['returns'].std() * (ann_factor ** 0.5)
        return_risk = ann_return / ann_vol if ann_vol != 0 else None
        max_return = data['returns'].max()
        min_return = data['returns'].min()

        upside_mask = benchmark_returns > 0
        downside_mask = benchmark_returns < 0
        upside_capture = (data['returns'][upside_mask].mean() / benchmark_returns[upside_mask].mean()) if upside_mask.any() else None
        downside_capture = (data['returns'][downside_mask].mean() / benchmark_returns[downside_mask].mean()) if downside_mask.any() else None

        beta = static_beta(benchmark_data, data['returns'])

        return_metrics_df.loc[col] = [
            total_return, mean_return, avg_win_return, avg_lose_return, win_ratio,
            ann_return, ann_vol, return_risk, max_return, min_return,
            upside_capture, downside_capture, beta
        ]
    return return_metrics_df


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

def compute_drawdown(cumret):
    roll_max = cumret.cummax()
    drawdown = (cumret - roll_max) / roll_max
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
