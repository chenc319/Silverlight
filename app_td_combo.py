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
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA','TSM']
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

### GET DATAFRAME ###
tsm_df = pd.DataFrame(mags_ohlc_dict['TSM'])
tsm_df['h/l'] = close_hl_setup(tsm_df,'Close')

### SETUP AND PERFECTED ###
tsm_df =  build_td_combo_v2_setups(tsm_df)
tsm_df['perfected'] = get_perfected_9(tsm_df)
mini_tsm_df = tsm_df['2025-01-01':]

### COUNTDOWN DICTIONARY ###
tsm_dict = compute_setup_countdown_pairs(
    df = mini_tsm_df['2025-01-01':]
)

def build_stitched_td_df(full_df: pd.DataFrame, setup_countdown_dict: dict) -> pd.DataFrame:
    """
    full_df: the master time series with columns like:
       Open, High, Low, Close, h/l, setup, perfected (and possibly others)

    setup_countdown_dict: output of compute_setup_countdown_pairs, where each entry has:
       'dataframe' with columns including: Open, High, Low, Close, setup, perfected, countdown

    Returns stitched_df with index == full_df.index and columns:
       Open, High, Low, Close, h/l, setup, perfected,
       countdown, tdst_level, risk_level
    """

    # Ensure datetime index on full_df
    base = full_df.copy()
    base.index = pd.to_datetime(base.index)

    # Initialize countdown / levels on the full index
    base["countdown"] = 0.0
    base["tdst_level"] = np.nan
    base["risk_level"] = np.nan

    # Collect TDST and risk events (date, level)
    tdst_events = []
    risk_events = []

    for _, rec in setup_countdown_dict.items():
        leg_df = rec["dataframe"].copy()
        leg_df.index = pd.to_datetime(leg_df.index)

        # Align leg countdown to the full index
        # Reindex with 0 for missing dates so we don't contaminate other periods
        aligned_cdn = (
            leg_df["countdown"]
            .reindex(base.index)
            .fillna(0.0)
        )

        # Keep the max countdown per bar across legs
        base["countdown"] = np.maximum(base["countdown"], aligned_cdn)

        # Collect TDST / risk events for the staircase
        tdst_val = rec.get("tdst_val")
        tdst_date = rec.get("tdst_date")
        if tdst_val is not None and tdst_date is not None:
            tdst_events.append((pd.to_datetime(tdst_date), float(tdst_val)))

        risk_val = rec.get("risk_lvl")
        risk_date = rec.get("risk_date")
        if risk_val is not None and risk_date is not None:
            risk_events.append((pd.to_datetime(risk_date), float(risk_val)))

    # Build TDST staircase on the full index
    if tdst_events:
        tdst_events_sorted = sorted(tdst_events, key=lambda x: x[0])
        for i, (start_date, level) in enumerate(tdst_events_sorted):
            start_date = pd.to_datetime(start_date)
            end_date = (
                tdst_events_sorted[i + 1][0] - pd.Timedelta(days=1)
                if i + 1 < len(tdst_events_sorted)
                else base.index.max()
            )
            mask = (base.index >= start_date) & (base.index <= end_date)
            base.loc[mask, "tdst_level"] = level

    # Build Risk staircase on the full index
    if risk_events:
        risk_events_sorted = sorted(risk_events, key=lambda x: x[0])
        for i, (start_date, level) in enumerate(risk_events_sorted):
            start_date = pd.to_datetime(start_date)
            end_date = (
                risk_events_sorted[i + 1][0] - pd.Timedelta(days=1)
                if i + 1 < len(risk_events_sorted)
                else base.index.max()
            )
            mask = (base.index >= start_date) & (base.index <= end_date)
            base.loc[mask, "risk_level"] = level

    # Ensure countdown is integer-like
    base["countdown"] = base["countdown"].astype(int)

    # You now have one cohesive dataframe on the original time series
    return base

test_stitch = build_stitched_td_df(mini_googl_df,googl_dict)
tsm_stitch = build_stitched_td_df(mini_tsm_df,tsm_dict)



### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###


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
    cd_y_buy    = df["Low"] - (large_off + 2.5 * small_off)

    setup_y_sell = df["High"] + (large_off + 0.5 * small_off)
    cd_y_sell    = df["High"] + (large_off + 2.5 * small_off)

    setup_y = pd.Series(index=df.index, dtype=float)
    setup_y[buy_mask]  = setup_y_buy[buy_mask]
    setup_y[sell_mask] = setup_y_sell[sell_mask]

    # countdown y: independent of setup, but side‑aware
    cd_y = pd.Series(index=df.index, dtype=float)
    cd_y[cd_mask & buy_mask]  = cd_y_buy[cd_mask & buy_mask]
    cd_y[cd_mask & sell_mask] = cd_y_sell[cd_mask & sell_mask]

    # if there is a countdown but no setup (e.g. continuation), decide side
    # by last non‑zero setup sign
    no_setup_cd = cd_mask & ~buy_mask & ~sell_mask
    if no_setup_cd.any():
        last_setup = df["setup"].replace(0, np.nan).ffill()
        extra_buy  = no_setup_cd & (last_setup < 0)
        extra_sell = no_setup_cd & (last_setup > 0)
        cd_y[extra_buy]  = cd_y_buy[extra_buy]
        cd_y[extra_sell] = cd_y_sell[extra_sell]

    # setup text: abs for buys, raw for sells
    setup_text = pd.Series(index=df.index, dtype=object)
    setup_text[buy_mask]  = df.loc[buy_mask,  "setup"].abs().astype(int).astype(str)
    setup_text[sell_mask] = df.loc[sell_mask, "setup"].astype(int).astype(str)

    fig = go.Figure()

    # 1) Candles
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

    # 2) Setup labels
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

    # 3) Countdown labels (magenta numbers should appear wherever countdown > 0)
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

    # 4) TDST pure horizontal segments
    tdst_mask = df["tdst_level"].notna()
    if tdst_mask.any():
        # find contiguous ranges of same tdst_level
        level = df["tdst_level"]
        change = (level != level.shift()).fillna(False)
        start_idx = df.index[change & tdst_mask]

        for start in start_idx:
            current_level = level.loc[start]
            # extend until level changes or becomes NaN
            seg_mask = (df.index >= start) & (tdst_mask) & (level == current_level)
            seg_dates = df.index[seg_mask]
            if len(seg_dates) < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df.loc[seg_dates, "DateStr"],
                    y=[current_level] * len(seg_dates),
                    mode="lines",
                    line=dict(color="limegreen", width=1.5),
                    name="TDST Level",
                    showlegend=False,
                )
            )

    # 5) Risk pure horizontal segments
    risk_mask = df["risk_level"].notna()
    if risk_mask.any():
        level = df["risk_level"]
        change = (level != level.shift()).fillna(False)
        start_idx = df.index[change & risk_mask]

        for start in start_idx:
            current_level = level.loc[start]
            seg_mask = (df.index >= start) & (risk_mask) & (level == current_level)
            seg_dates = df.index[seg_mask]
            if len(seg_dates) < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df.loc[seg_dates, "DateStr"],
                    y=[current_level] * len(seg_dates),
                    mode="lines",
                    line=dict(color="magenta", width=1.5, dash="dot"),
                    name="TD Risk Level",
                    showlegend=False,
                )
            )

    # 6) Perfected arrows (higher for sell setups)
    # use last non‑zero setup sign to decide side per perfected bar
    last_setup = df["setup"].replace(0, np.nan).ffill()
    for dt in df.index[perfected_mask]:
        date_str = dt.strftime("%Y-%m-%d")
        side_val = last_setup.loc[dt]
        if side_val > 0:  # sell: push arrow further up
            y_arrow = df.loc[dt, "High"] + (large_off + 3 * small_off)
        else:             # buy or unknown: just above bar
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

def plot_stitched_tsm():
    plot_td_combo_case_study_stitched(tsm_stitch)