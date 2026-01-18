### ------------------------------------------------------------------------------------ ###
### ------------------------------------- TD COMBO ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ------------------------------------------------------------------------------------ ###
### ------------------------------------- TD COMBO ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### OHLC MAGS DATA ###
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA']
mags_ohlc_dict = dict()
for mag_ticker in mags_tickers:
    mag_string = mag_ticker + '.csv'
    with open(Path(DATA_DIR) / mag_string, 'rb') as file:
        mag_df = pd.read_csv(file)
        mag_df.index = pd.to_datetime(mag_df['Date'].values)
        mag_df.drop('Date', axis=1, inplace=True)
    mags_ohlc_dict[mag_ticker] = mag_df

### ------------------------------------------------------------------------------------ ###
### -------------------------------------- SETUP --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### GET DATAFRAME ###
googl_df = pd.DataFrame(mags_ohlc_dict['GOOGL'])
googl_df['h/l'] = close_hl_setup(googl_df,'Close')

### SETUP AND PERFECTED ###
googl_df =  build_td_combo_v2_setups(googl_df)
googl_df['perfected'] = get_perfected_9(googl_df)
mini_googl_df = googl_df['2025-01-01':]

### COUNTDOWN DICTIONARY ###
googl_dict = compute_setup_countdown_pairs(
    df = mini_googl_df['2025-01-01':]
)

def resolve_countdown(series):
    """Pick the countdown that should win for a given date."""
    non_na = series.dropna()
    if non_na.empty:
        return np.nan
    # choose the highest countdown value (earlier in the sequence)
    return non_na.max()

def stitch_td_dataframes(dfs):
    # concat all, aligning on index (dates)
    big = pd.concat(dfs, axis=0)

    # if your index is not DatetimeIndex yet:
    # big.index = pd.to_datetime(big.index)

    # group by date index
    grouped = big.groupby(big.index)

    # columns that should be 'any non-null' (these should not conflict across dfs)
    any_cols = ["Open", "High", "Low", "Close", "setup", "perfected"]
    # countdown has special rule
    cd_col = "countdown"

    out_parts = []

    for date, g in grouped:
        row = {}
        # handle the straightforward columns
        for col in any_cols:
            vals = g[col].dropna()
            row[col] = vals.iloc[0] if not vals.empty else np.nan

        # handle countdown with custom resolver
        row[cd_col] = resolve_countdown(g[cd_col])

        out_parts.append(pd.Series(row, name=date))

    stitched = pd.DataFrame(out_parts).sort_index()

    return stitched


def build_stitched_td_df(setup_countdown_dict):
    """
    Convert your setup_countdown_dict into one time‑series with:
      Open, High, Low, Close, setup, perfected, countdown,
      tdst_level, risk_level

    TDST: horizontal from each leg's tdst_date to just before the next leg's tdst_date.
    Risk: horizontal from each leg's risk_date (if not None) to just before the next leg's risk_date.
    """

    # 1) collect all per‑leg data
    leg_frames = []
    tdst_events = []   # (date, level)
    risk_events = []   # (date, level)

    for name, rec in setup_countdown_dict.items():
        df_leg = rec["dataframe"].copy()
        df_leg.index = pd.to_datetime(df_leg.index)

        # keep original structure; we will fill tdst_level / risk_level AFTER stitching
        leg_frames.append(df_leg)

        # TDST event
        tdst_val = rec.get("tdst_val", None)
        tdst_date = rec.get("tdst_date", None)
        if (tdst_val is not None) and (tdst_date is not None):
            tdst_events.append((pd.to_datetime(tdst_date), float(tdst_val)))

        # Risk event
        risk_val = rec.get("risk_lvl", None)
        risk_date = rec.get("risk_date", None)
        if (risk_val is not None) and (risk_date is not None):
            risk_events.append((pd.to_datetime(risk_date), float(risk_val)))

    # 2) stitch price / setup / countdown etc.
    big = pd.concat(leg_frames, axis=0)
    big.index.name = "Date"

    grouped = big.groupby(big.index)

    def first_non_nan(series):
        s = series.dropna()
        return s.iloc[0] if not s.empty else np.nan

    def max_countdown(series):
        s = series.dropna()
        return s.max() if not s.empty else np.nan

    rows = []
    for dt, g in grouped:
        row = {}
        for col in ["Open", "High", "Low", "Close", "setup", "perfected"]:
            row[col] = first_non_nan(g[col])
        row["countdown"] = max_countdown(g["countdown"])
        rows.append(pd.Series(row, name=dt))

    df = pd.DataFrame(rows).sort_index()

    # 3) build TDST staircase from tdst_events
    df["tdst_level"] = np.nan
    if tdst_events:
        tdst_events_sorted = sorted(tdst_events, key=lambda x: x[0])
        for i, (start_date, level) in enumerate(tdst_events_sorted):
            start_date = pd.to_datetime(start_date)
            end_date = (
                tdst_events_sorted[i + 1][0] - pd.Timedelta(days=1)
                if i + 1 < len(tdst_events_sorted)
                else df.index.max()
            )
            mask = (df.index >= start_date) & (df.index <= end_date)
            df.loc[mask, "tdst_level"] = level

    # 4) build Risk staircase from risk_events
    df["risk_level"] = np.nan
    if risk_events:
        risk_events_sorted = sorted(risk_events, key=lambda x: x[0])
        for i, (start_date, level) in enumerate(risk_events_sorted):
            start_date = pd.to_datetime(start_date)
            end_date = (
                risk_events_sorted[i + 1][0] - pd.Timedelta(days=1)
                if i + 1 < len(risk_events_sorted)
                else df.index.max()
            )
            mask = (df.index >= start_date) & (df.index <= end_date)
            df.loc[mask, "risk_level"] = level

    return df



test_stitch = build_stitched_td_df(googl_dict)
def plot_td_combo_case_study_stitched(stitched_df):
    df = stitched_df.copy()
    df.index = pd.to_datetime(df.index)
    df["DateStr"] = df.index.strftime("%Y-%m-%d")

    # masks
    setup_mask = df["setup"] != 0
    buy_mask = df["setup"] < 0
    sell_mask = df["setup"] > 0
    cd_mask = df["countdown"] > 0
    perfected_mask = df["perfected"].fillna("").astype(str).str.lower().eq("perfected")

    # magnitude reference
    price_span = df["High"].max() - df["Low"].min()
    if price_span == 0:
        price_span = max(df["Close"].abs().max(), 1.0)
    small_off = 0.01 * price_span
    large_off = 0.02 * price_span

    # label Y positions
    setup_y_buy = df["Low"] - (large_off + 0.5 * small_off)
    cd_y_buy = df["Low"] - (large_off + 2.5 * small_off)

    setup_y_sell = df["High"] + (large_off + 0.5 * small_off)
    cd_y_sell = df["High"] + (large_off + 2.5 * small_off)

    setup_y = pd.Series(index=df.index, dtype=float)
    setup_y[buy_mask] = setup_y_buy[buy_mask]
    setup_y[sell_mask] = setup_y_sell[sell_mask]

    cd_y = pd.Series(index=df.index, dtype=float)
    cd_y[cd_mask & buy_mask] = cd_y_buy[cd_mask & buy_mask]
    cd_y[cd_mask & sell_mask] = cd_y_sell[cd_mask & sell_mask]

    setup_text = pd.Series(index=df.index, dtype=object)
    setup_text[buy_mask] = df.loc[buy_mask, "setup"].abs().astype(int).astype(str)
    setup_text[sell_mask] = df.loc[sell_mask, "setup"].astype(int).astype(str)

    fig = go.Figure()

    # candles
    fig.add_trace(
        go.Ohlc(
            x=df["DateStr"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="white",
            decreasing_line_color="white",
            name="Price",
        )
    )

    # setup labels
    fig.add_trace(
        go.Scatter(
            x=df.loc[setup_mask, "DateStr"],
            y=setup_y.loc[setup_mask],
            mode="text",
            text=setup_text.loc[setup_mask],
            textfont=dict(color="lime", size=12),
            textposition="middle center",
            name="TD Setup",
        )
    )

    # countdown labels
    fig.add_trace(
        go.Scatter(
            x=df.loc[cd_mask, "DateStr"],
            y=cd_y.loc[cd_mask],
            mode="text",
            text=df.loc[cd_mask, "countdown"].astype(int).astype(str),
            textfont=dict(color="magenta", size=12),
            textposition="middle center",
            name="TD Countdown",
        )
    )

    # TDST staircase (already piecewise horizontal in column)
    tdst_mask = df["tdst_level"].notna()
    if tdst_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[tdst_mask, "DateStr"],
                y=df.loc[tdst_mask, "tdst_level"],
                mode="lines",
                line=dict(color="limegreen", width=1.5),
                name="TDST Level",
            )
        )

    # Risk staircase
    risk_mask = df["risk_level"].notna()
    if risk_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[risk_mask, "DateStr"],
                y=df.loc[risk_mask, "risk_level"],
                mode="lines",
                line=dict(color="magenta", width=1.5, dash="dot"),
                name="TD Risk Level",
            )
        )

    # perfected arrows
    for dt in df.index[perfected_mask]:
        date_str = dt.strftime("%Y-%m-%d")
        y_arrow = df.loc[dt, "High"] + large_off
        fig.add_annotation(
            x=date_str,
            y=y_arrow,
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=1.5,
            arrowcolor="red",
            ax=0,
            ay=-15,
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        xaxis=dict(
            showgrid=False,
            type="category",
            rangeslider_visible=False,
        ),
        yaxis=dict(showgrid=False),
        height=700,
        width=1200,
    )

    st.plotly_chart(fig, use_container_width=True)



### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

def plot_googl_case_study_1():
    plot_td_combo_case_study(googl_dict,0)


def plot_googl_case_study_2():
    plot_td_combo_case_study(googl_dict,1)


def plot_googl_case_study_3():
    plot_td_combo_case_study(googl_dict,2)


def plot_googl_case_study_4():
    plot_td_combo_case_study(googl_dict,3)


def plot_googl_case_study_5():
    plot_td_combo_case_study(googl_dict,4)


def plot_googl_case_study_6():
    plot_td_combo_case_study(googl_dict,5)

def plot_stitched_googl():
    plot_td_combo_case_study_stitched(test_stitch)
