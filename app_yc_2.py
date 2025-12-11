### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_weekly = pd.DataFrame(sp500['Close']).resample('W-FRI').last()
spx_weekly.columns = ['spx']
spx_daily = pd.DataFrame(sp500['Close'])
spx_daily.columns = ['spx']
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

### BONDS ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_weekly = pd.DataFrame(agg['Close']).resample('W-FRI').last()
bonds_weekly.columns = ['bonds']
bonds_daily = pd.DataFrame(agg['Close'])
bonds_daily.columns = ['bonds']
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

### YIELDS ###
with open(Path(DATA_DIR) / 'treasury_1m.pkl', 'rb') as file:
    treasury_1m = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_2y.pkl', 'rb') as file:
    treasury_2y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_5y.pkl', 'rb') as file:
    treasury_5y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_10y.pkl', 'rb') as file:
    treasury_10y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_30y.pkl', 'rb') as file:
    treasury_30y = pd.read_pickle(file)

treasury_merge = merge_dfs([treasury_1m,
                            treasury_2y,
                            treasury_5y,
                            treasury_10y,
                            treasury_30y]).dropna()
treasury_merge.index = pd.to_datetime(treasury_merge.index).values
treasury_merge.columns = ['1m','2y','5y','10y','30y']
treasury_monthly_df = treasury_merge.resample('ME').last()


level = treasury_monthly_df.mean(axis=1)
slope = treasury_monthly_df["10y"] - treasury_monthly_df["2y"]
curv = (2 * treasury_monthly_df["5y"] -
        treasury_monthly_df["2y"] - treasury_monthly_df["10y"])

factors = pd.DataFrame({
    "level": level,
    "slope": slope,
    "curv":  curv,
})

df = factors.copy()
for col in ["level", "slope", "curv"]:
    df[f"d_{col}"] = df[col].diff(3)

# 3m rolling mean on changes to smooth noise
for col in ["d_level", "d_slope", "d_curv"]:
    df[f"{col}_ma"] = df[col].rolling(3).mean()

df = df.dropna()

alpha = 0.5   # weight on front-end move in growth proxy
beta  = 0.5   # weight on curvature in inflation proxy

# Front-end level move: use 3m
d_front = treasury_monthly_df["1m"].diff(3).reindex(df.index)

G = df["d_slope_ma"] - alpha * d_front
Pi = df["d_level_ma"] + beta * df["d_curv_ma"]

sig = pd.DataFrame({
    "G":  G,
    "Pi": Pi,
}).dropna()

# Align with factors
df = df.join(sig, how="inner")

def rolling_z(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window).mean()
    s = x.rolling(window).std()
    return (x - m) / s

window = 36  # 10 years of monthly data
df['G_z']  = rolling_z(df['G'],  window)
df['Pi_z'] = rolling_z(df['Pi'], window)

def quad_scores(g_z: float, pi_z: float) -> dict:
    """
    Latent scores for each named quad given G_z and Pi_z.
    Continuous, sign-consistent with the usual GIP framework.
    """
    s = {}

    s['Goldilocks']  = (+1.0 * g_z) + (-1.0 * pi_z)   # G↑, Pi↓
    s['Reflation']   = (+1.0 * g_z) + (+1.0 * pi_z)   # G↑, Pi↑
    s['Stagflation'] = (-1.0 * g_z) + (+1.0 * pi_z)   # G↓, Pi↑
    s['Deflation']   = (-1.0 * g_z) + (-1.0 * pi_z)   # G↓, Pi↓

    return s

def scores_to_probs(score_dict: dict, temp: float = 1.0) -> dict:
    vals = np.array(list(score_dict.values()), dtype=float)
    exps = np.exp(vals / temp - np.max(vals / temp))
    probs = exps / exps.sum()
    return {k: p for k, p in zip(score_dict.keys(), probs)}

quad_names = ['Goldilocks', 'Reflation', 'Stagflation', 'Deflation']
prob_cols = [f'prob_{q}' for q in quad_names]

for col in prob_cols:
    df[col] = np.nan

best_quad = []

for dt, row in df.iterrows():
    g_z  = row['G_z']
    pi_z = row['Pi_z']

    if np.isnan(g_z) or np.isnan(pi_z):
        best_quad.append(np.nan)
        continue

    scores = quad_scores(g_z, pi_z)
    probs  = scores_to_probs(scores, temp=1.0)

    for q in quad_names:
        df.loc[dt, f'prob_{q}'] = probs[q]

    # No "Neutral": always pick argmax
    best_quad.append(max(probs, key=probs.get))

df['quad'] = best_quad




### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### CALCULATE SLOPES AND DIFFERENCES ###
treasury_diff = treasury_monthly_df.copy()

treasury_diff['level'] = treasury_monthly_df.mean(axis=1)
treasury_diff['slope'] = treasury_monthly_df["10y"] - treasury_monthly_df["2y"]
treasury_diff['curve'] = 2 * treasury_monthly_df["5y"] - treasury_monthly_df["2y"] - treasury_monthly_df["10y"]


treasury_diff = treasury_monthly_df.diff(3).dropna()
treasury_diff['belly'] = treasury_diff['10y'] - treasury_diff['2y']
treasury_diff['total_yc_direction'] = treasury_diff[['2y','5y','10y']].mean(axis=1)
treasury_diff['level_class'] = ['Bull' if x <= 0 else 'Bear' for x in treasury_diff['2y']]
treasury_diff['belly_class'] = ['Flattening' if x <= 0 else 'Steepening' for x in treasury_diff['belly']]

### REGIMES ###
treasury_diff['belly_regime'] = treasury_diff['level_class'] + ' ' + treasury_diff['belly_class']
treasury_diff['spx_monthly_pct'] = spx_monthly.pct_change(1).shift(-1)

belly_regime_score_map = {
    'Bear Steepening':  +0.9,
    'Bull Flattening':  +1,
    'Bear Flattening':  +0.6,
    'Bull Steepening':  +0.7
}
belly_quad_regime_map = {
    'Bear Steepening':  'Reflation',
    'Bull Flattening':  'Goldilocks',
    'Bear Flattening':  'Deflation',
    'Bull Steepening':  'Stagflation'
}

treasury_diff['total_score'] = treasury_diff['belly_regime'].map(belly_regime_score_map)
treasury_diff['bt_returns'] = treasury_diff['spx_monthly_pct'] * treasury_diff['total_score']
treasury_diff['regime_label'] = treasury_diff['belly_regime'].map(belly_quad_regime_map)

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- YIELD CURVE REGIME MODEL ---------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

equities_yc_return_metrics = return_metrics(
    treasury_diff[['bt_returns','spx_monthly_pct']],
    treasury_diff[['spx_monthly_pct']],
    12
)
equities_yc_return_metrics['Return/Risk']

def plot_colorcoded_regime():
    regime_merge = merge_dfs([
        pd.DataFrame(treasury_diff['regime_label']),
        spx_monthly,
        pd.DataFrame(treasury_diff['bt_returns'])
    ]).dropna()
    color_coded_regime_plot(regime_merge,
                            y_col='spx',
                            regime_col='regime_label',
                            title="Level by Macro Regime")
    stats_df = calculate_regime_statistics(regime_merge,
                                           regime_col_name='regime_label',
                                           return_cols = ['bt_returns'])
    plot_streamlit_regime_statistics(stats_df)

def equity_yc_results():
    streamlit_plot(df=(1 + treasury_diff[['bt_returns', 'spx_monthly_pct']]).cumprod() - 1,
                   columns_array=['bt_returns', 'spx_monthly_pct'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(equities_yc_return_metrics)

