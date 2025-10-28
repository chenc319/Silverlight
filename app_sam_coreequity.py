### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- MAG7 SAM CORE EQUITY ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### FUNCTIONS ###
import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import os
import functools as ft
from scipy import stats
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### SAM ###
data = '''
2025  4.63  -0.78  -3.15  0.83  4.56  2.87  ---   ---   ---   ---   ---   ---
2024  0.98  3.55  3.37  -3.74  4.04  0.02  2.43  2.21  1.53  0.30  5.52  -5.72
2023  5.51  -2.06  -1.20  -2.46  -2.93  4.99  2.58  -0.03  -2.44  -1.55  7.53  4.06
2022  -3.34  0.28  3.34  -5.60  1.41  -7.83  6.72  -1.06  -6.80  9.74  6.91  -3.64
2021  -0.92  4.57  3.46  3.48  3.20  0.73  1.67  -0.14  -2.54  3.01  -1.95  2.21
2020  -2.36  -7.47  -12.19  7.33  6.33  5.07  5.63  3.89  4.08  5.92  5.68  16.89
2019  7.35  1.92  2.36  3.57  -5.27  5.90  1.33  0.69  1.42  1.09  5.25  2.57
2018  5.94  -4.00  -0.97  -0.09  2.75  1.71  2.17  1.94  0.44  -5.65  4.32  -7.63
2017  1.69  3.76  0.07  0.24  1.13  0.93  -0.04  -0.36  2.13  1.89  3.88  0.77
2016  -1.77  2.76  6.68  -2.15  1.48  -0.02  -1.98  -1.50  -2.00  -1.98  5.23  2.22
2015  -1.98  5.94  -0.31  1.83  1.16  -1.72  -3.57  -5.68  -2.57  5.87  -1.69  -1.24
2014  -2.60  4.32  1.57  0.34  3.95  1.12  -1.75  3.19  -1.43  4.10  4.07  -0.36
2013  4.23  0.23  3.88  3.35  1.97  -0.18  4.91  -3.39  3.55  5.47  5.32  3.37
'''

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
all_dates = []
all_vals = []

for row in data.strip().split('\n'):
    parts = row.split()
    year = parts[0]
    vals = parts[1:13]
    for i, val in enumerate(vals):
        date_str = f'{year}-{months[i]}-01'  # Use the first day of each month
        if val in ['---', '--', '—']:
            decimal = float('nan')
        else:
            decimal = float(val) / 100.0
        all_dates.append(date_str)
        all_vals.append(decimal)

series = pd.Series(all_vals, index=pd.to_datetime(all_dates), name='Return')
df = series.to_frame()
df.sort_index(inplace=True)
df = df.dropna()
sam_core_equity = df.resample('ME').last()
sam_core_equity.columns = ['SAM']

### MAGS ###
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA']
each_mags_df = pd.DataFrame()
for mag_ticker in mags_tickers:
    mag_string = mag_ticker + '.csv'
    with open(Path(DATA_DIR) / mag_string, 'rb') as file:
        mag_df = pd.read_csv(file)
        mag_df.index = pd.to_datetime(mag_df['Date'].values)
        close_df = pd.DataFrame(mag_df['Close'])
        close_df.columns = [mag_ticker]
    each_mags_df = merge_dfs([each_mags_df, close_df])
each_mags_df = each_mags_df.dropna()
mags_monthly_pct = each_mags_df.resample('ME').last().pct_change().dropna()

### SPX ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    spx_df = pd.read_csv(file)
    spx_df.index = pd.to_datetime(spx_df['Date'].values)
    spx_df = pd.DataFrame(spx_df['Close'])
spx_monthly_pct = spx_df.resample('ME').last().pct_change().dropna()
spx_monthly_pct.columns = ['SPX']

### MERGE ###
sam_mags_merge = merge_dfs([mags_monthly_pct, spx_monthly_pct,sam_core_equity]).dropna()



### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- MAG7 SAM CORE EQUITY ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def core_equity_mags_spx():
    fig = go.Figure()
    colors = ['#006400', '#228B22', '#2E8B57', '#3CB371', '#36C88B', '#77DD77', '#ADFF2F', '#2056AE', '#E74C3C']
    for name, color in zip(sam_mags_merge.columns, colors):
        fig.add_trace(go.Scatter(
            x=sam_mags_merge.index,
            y=sam_mags_merge[name],
            name=name,
            mode='lines',
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title="Monthly Returns"
    )
    st.plotly_chart(fig, use_container_width=True)

    cumulative_returns = sam_mags_merge.cumsum()
    fig = go.Figure()
    colors = ['#006400', '#228B22', '#2E8B57', '#3CB371', '#36C88B', '#77DD77', '#ADFF2F', '#2056AE', '#E74C3C']

    for name, color in zip(cumulative_returns.columns, colors):
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[name],
            name=name,
            mode='lines',
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title="Summative Returns"
    )
    st.plotly_chart(fig, use_container_width=True)


def core_equity_correlation():
    sam_corr_matrix = sam_mags_merge.corr()['SAM']
    spx_corr_matrix = sam_mags_merge.corr()['SPX']
    def highlight_red_green(val):
        if val < 0:
            color = 'background-color: #ffcccc'
        elif val > 0:
            color = 'background-color: #ccffcc'
        else:
            color = ''
        return color

    styled_df = spx_corr_matrix.style.format({"Correlation": "{:.3f}"}) \
        .applymap(highlight_red_green, subset=['Correlation'])
    st.write("Correlation with S&P 500")
    st.write(styled_df, unsafe_allow_html=True)

    def highlight_red_green(val):
        if val < 0:
            color = 'background-color: #ffcccc'
        elif val > 0:
            color = 'background-color: #ccffcc'
        else:
            color = ''
        return color

    styled_df = sam_corr_matrix.style.format({"Correlation": "{:.3f}"}) \
        .applymap(highlight_red_green, subset=['Correlation'])
    st.write("Correlation with S&P 500")
    st.write(styled_df, unsafe_allow_html=True)

def sam_core_equity_rolling_alpha():

    rolling_alpha_to_spx = pd.DataFrame(columns = ['SAM','GOOGL','AMZN','AAPL','MSFT','NVDA','TSLA'],
                                        index = sam_mags_merge.index)
    for row in range(11,len(sam_mags_merge)):
        subset = sam_mags_merge.iloc[row-11:row+1]
        sam_beta, sam_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['SAM'])
        rolling_alpha_to_spx.loc[subset.index[11],'SAM'] = sam_alpha
        goog_beta, goog_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['GOOGL'])
        rolling_alpha_to_spx.loc[subset.index[11],'GOOGL'] = goog_alpha
        amzn_beta, amzn_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['AMZN'])
        rolling_alpha_to_spx.loc[subset.index[11],'AMZN'] = amzn_alpha
        aapl_beta, aapl_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['AAPL'])
        rolling_alpha_to_spx.loc[subset.index[11],'AAPL'] = aapl_alpha
        meta_beta, meta_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['META'])
        rolling_alpha_to_spx.loc[subset.index[11],'META'] = meta_alpha
        msft_beta, msft_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['MSFT'])
        rolling_alpha_to_spx.loc[subset.index[11],'MSFT'] = msft_alpha
        nvda_beta, nvda_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['NVDA'])
        rolling_alpha_to_spx.loc[subset.index[11],'NVDA'] = nvda_alpha
        tsla_beta, tsla_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['TSLA'])
        rolling_alpha_to_spx.loc[subset.index[11],'TSLA'] = tsla_alpha

    rolling_alpha_to_spx = rolling_alpha_to_spx.dropna()
    rolling_alpha_to_spx['MAGS'] = rolling_alpha_to_spx[mags_tickers].mean(axis=1)

    fig = go.Figure()
    colors = ['#2056AE', '#E74C3C']
    for name, color in zip(['MAGS','SAM'], colors):
        fig.add_trace(go.Scatter(
            x=rolling_alpha_to_spx.index,
            y=rolling_alpha_to_spx[name],
            name=name,
            mode='lines',
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(title='Legend', orientation='h', y=-0.25),
        margin=dict(t=30, b=30),
        title="Rolling 12m Alpha"
    )
    st.plotly_chart(fig, use_container_width=True)

    alpha_correlation = rolling_alpha_to_spx.corr()['SAM'][1:]
    def highlight_red_green(val):
        if val < 0:
            color = 'background-color: #ffcccc'  # light red
        elif val > 0:
            color = 'background-color: #ccffcc'  # light green
        else:
            color = ''  # no highlight for zero
        return color

    def style_percent(df):
        col = df.columns[0]
        return df.style.format({col: "{:.2f}%"}) \
            .applymap(highlight_red_green, subset=[col])

    ### PLOT ###
    st.title("Alpha Correlation")
    cols = st.columns(1)
    with cols[0]:
        st.write(style_percent(alpha_correlation), unsafe_allow_html=True)

