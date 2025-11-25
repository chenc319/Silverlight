### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH INFLATION MODEL ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH INFLATION MODEL ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### FUNCTIONS ###
def load_price_data(filename, resample_freq='ME'):
    """Load and resample price data from CSV."""
    with open(Path(DATA_DIR) / filename, 'rb') as file:
        df = pd.read_csv(file)
    df.index = pd.to_datetime(df['Date'])
    df = pd.DataFrame(df['Close']).resample(resample_freq).last()
    return df

def load_multiple_tickers(ticker_dict, resample_freq='ME'):
    """Load multiple ticker CSVs and merge into single DataFrame."""
    merged_df = pd.DataFrame()
    for ticker, label in ticker_dict.items():
        df = load_price_data(f'{ticker}.csv', resample_freq)
        df.columns = [label]
        merged_df = merge_dfs([merged_df, df])
    return merged_df

def regime_label(row):
    if row['inflation_roc_2'] > 0 and row['growth_roc_2'] > 0:
        return 'Reflation'
    elif row['inflation_roc_2'] > 0 and row['growth_roc_2'] < 0:
        return 'Stagflation'
    elif row['inflation_roc_2'] < 0 and row['growth_roc_2'] > 0:
        return 'Goldilocks'
    elif row['inflation_roc_2'] < 0 and row['growth_roc_2'] < 0:
        return 'Deflation'
    return np.nan

def calculate_regime_statistics(df, return_cols=['sp500_pct', 'bonds_pct']):
    regimes = ['Goldilocks', 'Reflation', 'Stagflation', 'Deflation']
    quad_labels = ['Quad 1', 'Quad 2', 'Quad 3', 'Quad 4']
    results = []

    for regime, quad in zip(regimes, quad_labels):
        regime_data = df[df['regime_label'] == regime]
        regime_returns = (regime_data[return_cols].mean() * 100).values

        results.append({
            'Regime': f"{quad}: {regime} ({'I-G+' if regime == 'Goldilocks' else 'I+G+' if regime == 'Reflation' else 'I+G-' if regime == 'Stagflation' else 'I-G-'})",
            'Equities': regime_returns[0],
            'Bonds': regime_returns[1],
            '% of Occurrences': (len(regime_data) / len(df.dropna(subset=['regime_label']))) * 100
        })

    return pd.DataFrame(results)

def calculate_regime_performance(base_df, asset_returns_df, regime_col='regime_label'):
    """Calculate average returns by regime for multiple assets."""
    # UPDATED ORDER
    regimes = ['Goldilocks', 'Reflation', 'Stagflation', 'Deflation']
    combined_df = merge_dfs([base_df, asset_returns_df])

    results = {}
    for regime in regimes:
        regime_data = combined_df[combined_df[regime_col] == regime]
        avg_returns = (regime_data[asset_returns_df.columns].mean() * 100).sort_values(ascending=False)
        results[regime] = pd.DataFrame({regime: avg_returns.round(2)})

    return results

### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------ DATA LOADING -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### DICTIONARIES ###
spx_sectors = {
    "XLC": "comm_serv", "XLY": "cons_disc", "XLP": "cons_stap",
    "XLE": "energy", "XLF": "financials", "XLV": "healthcare",
    "XLI": "industrial", "XLB": "materials", "XLRE": "real_estate",
    "XLK": "tech", "XLU": "utilities"
}
quad_regime_factors = {
    "SPHB": "high_beta", "SPLV": "low_beta", "IWM": "small_caps",
    "IWR": "mid_caps", "MGK": "mega_cap_growth", "IYT": "cyclicals",
    "DEF": "defensives", "OEF": "size", "QUAL": "quality",
    "SPHD": "dividends", "MTUM": "momentum", "IWD": "value",
    "IWF": "equity_growth", "IWB": "large_caps"
}
regime_colors = {
    "Goldilocks": "#28a745",  # Green
    "Reflation": "#90ee90",  # Light green
    "Stagflation": "#ffc107",  # Yellow
    "Deflation": "#dc3545"  # Red
}
regime_code_map = {
    0: 'Reflation',
    1: 'Stagflation',
    2: 'Goldilocks',
    3: 'Deflation'
}

# ### GROWTH ###
# growth = pdr.DataReader('PCE',
#                         'fred',
#                         '1900-01-01',
#                         pd.to_datetime('today'))
# growth.index = growth.index + pd.DateOffset(months=1)
# growth = growth.resample('ME').last()
# with open(Path(DATA_DIR) / 'growth.pkl', 'wb') as file:
#     pickle.dump(growth, file)
#
# ### GROWTH ###
# inflation = pdr.DataReader('CPIAUCSL',
#                            'fred',
#                            '1900-01-01',
#                            pd.to_datetime('today'))
# inflation.index = inflation.index + pd.DateOffset(months=1)
# inflation = inflation.resample('ME').last()
# with open(Path(DATA_DIR) / 'inflation.pkl', 'wb') as file:
#     pickle.dump(inflation, file)

### LOAD DATA ###
sp500 = load_price_data('SPX.csv')
sp500.columns = ['spx']
agg = load_price_data('AGG.csv')
agg.columns = ['bonds']

with open(Path(DATA_DIR) / 'growth.pkl', 'rb') as file:
    growth = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'inflation.pkl', 'rb') as file:
    inflation = pd.read_pickle(file)

# Load sector and factor data
spx_sectors_df = load_multiple_tickers(spx_sectors)
quad_factors_df = load_multiple_tickers(quad_regime_factors)


### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- REGIME CLASSIFICATION ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### CREATE GROWTH AND INFLATION DF ###
growth_inflation_df = merge_dfs([
    growth.shift(-2),
    inflation.shift(-2),
    sp500,
    agg
]).dropna()

growth_inflation_df.columns = ['growth', 'inflation', 'sp500', 'bonds']

growth_inflation_df['growth_roc'] = growth_inflation_df['growth'].diff(12)
growth_inflation_df['growth_roc_2'] = growth_inflation_df['growth_roc'].diff(3)
growth_inflation_df['inflation_roc'] = growth_inflation_df['inflation'].diff(12)
growth_inflation_df['inflation_roc_2'] = growth_inflation_df['inflation_roc'].diff(3)

growth_inflation_df['sp500_pct'] = growth_inflation_df['sp500'].pct_change()
growth_inflation_df['bonds_pct'] = growth_inflation_df['bonds'].pct_change()

growth_inflation_df['regime_label'] = growth_inflation_df.apply(regime_label, axis=1)
growth_inflation_df = growth_inflation_df.dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###


def plot_growth_inflation(start=None, end=None, **kwargs):
    """Main plotting function for growth/inflation regime analysis."""

    df = growth_inflation_df.copy()

    # FIX: Convert start/end to pd.Timestamp to avoid datetime comparison error
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    # ===== CLI/CPI Dual Y-Axis Plot =====
    st.title("Growth and Inflation Inputs")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['CLI (Growth)', 'CPI (Inflation)'],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )

    colors = {
        'CLI': '#2056AE',
        'CLI ROC': '#6AC47E',
        'CPI': '#F2552C',
        'CPI ROC': '#F7BC38'
    }

    # CLI subplot
    fig.add_trace(
        go.Scatter(x=df.index, y=df['growth'], name='CLI Outright',
                   mode='lines', line=dict(color=colors['CLI'], width=2)),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['growth_roc'], name='CLI ROC',
                   mode='lines', line=dict(color=colors['CLI ROC'], dash='dot', width=2)),
        row=1, col=1, secondary_y=True
    )

    # CPI subplot
    fig.add_trace(
        go.Scatter(x=df.index, y=df['inflation'], name='CPI Outright',
                   mode='lines', line=dict(color=colors['CPI'], width=2)),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['inflation_roc'], name='CPI ROC',
                   mode='lines', line=dict(color=colors['CPI ROC'], dash='dot', width=2)),
        row=1, col=2, secondary_y=True
    )

    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.2)
    )
    fig.update_yaxes(title_text="Outright", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ROC", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Outright", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="ROC", row=1, col=2, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # ===== Regime-Colored Asset Charts =====
    st.title("Equity and Fixed Income by Regime")
    color_coded_regime_plot(df, 'sp500','regime_label', title="SP500 by Regime")
    color_coded_regime_plot(df, 'bonds','regime_label', title="Bonds by Regime")

    # ===== Statistics Table =====
    st.title("Growth and Inflation Historical Performance")
    stats_df = calculate_regime_statistics(df)

    cmap = LinearSegmentedColormap.from_list('red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256)
    styled = stats_df.style \
        .format({'Equities': "{:.2f}%", 'Bonds': "{:.2f}%", '% of Occurrences': "{:.2f}%"}) \
        .background_gradient(cmap=cmap, subset=['Equities', 'Bonds'])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)

    # ===== Return Distributions =====
    # UPDATED ORDER
    regimes = ['Goldilocks', 'Reflation', 'Stagflation', 'Deflation']

    st.title("Equity Return Distributions")
    plot_regime_return_histograms(df, 'regime_label', 'sp500_pct', regimes)

    st.title("Bonds Return Distributions")
    plot_regime_return_histograms(df, 'regime_label', 'bonds_pct', regimes)

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- SECTOR/FACTOR PLOTTING ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_spx_sector_regimes(start=None, end=None, **kwargs):
    """Plot sector and factor performance by regime."""

    # Calculate returns
    sector_returns = spx_sectors_df.resample('ME').last().pct_change()
    factor_returns = quad_factors_df.resample('ME').last().pct_change()

    # Filter by date range if provided
    if start:
        sector_returns = sector_returns[sector_returns.index >= pd.Timestamp(start)]
        factor_returns = factor_returns[factor_returns.index >= pd.Timestamp(start)]
    if end:
        sector_returns = sector_returns[sector_returns.index <= pd.Timestamp(end)]
        factor_returns = factor_returns[factor_returns.index <= pd.Timestamp(end)]

    # Calculate regime performance
    sector_performance = calculate_regime_performance(growth_inflation_df, sector_returns)
    factor_performance = calculate_regime_performance(growth_inflation_df, factor_returns)

    # Styling function
    def style_percent_table(df):
        def highlight_red_green(val):
            if val < 0:
                return 'background-color: #ffcccc'
            elif val > 0:
                return 'background-color: #ccffcc'
            return ''

        col = df.columns[0]
        return df.style.format({col: "{:.2f}%"}).applymap(highlight_red_green, subset=[col])

    # ===== Sector Performance =====
    # UPDATED ORDER with Quad labels
    st.title("Top/Bottom SPX Sector Performance")
    cols = st.columns(4)
    regimes = ['Goldilocks', 'Reflation', 'Stagflation', 'Deflation']
    quad_labels = ['Quad 1', 'Quad 2', 'Quad 3', 'Quad 4']

    for i, (regime, quad) in enumerate(zip(regimes, quad_labels)):
        with cols[i]:
            st.subheader(f"{quad}: {regime}")
            st.write(style_percent_table(sector_performance[regime]), unsafe_allow_html=True)

    # ===== Factor Performance =====
    st.title("Top/Bottom SPX Factor Performance")
    cols = st.columns(4)
    for i, (regime, quad) in enumerate(zip(regimes, quad_labels)):
        with cols[i]:
            st.subheader(f"{quad}: {regime}")
            st.write(style_percent_table(factor_performance[regime]), unsafe_allow_html=True)
