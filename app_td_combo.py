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
### ------------------------------- SETUP AND COUNTDOWN -------------------------------- ###
### ------------------------------------------------------------------------------------ ###

mags_td_combo_complete_stitches = dict()
mags_td_combo_individual_stitches = dict()
for mag_ticker in mags_tickers:

    ### GET DATAFRAME ###
    mags_df = pd.DataFrame(mags_ohlc_dict[mag_ticker])
    mags_df['h/l'] = close_hl_setup(mags_df,'Close')

    ### SETUP AND PERFECTED ###
    mags_df =  build_td_combo_v2_setups(mags_df)
    mags_df['perfected'] = get_perfected_9(mags_df)
    mini_mag_df = mags_df['2025-01-01':]

    ### COUNTDOWN DICTIONARY ###
    mag_dict = compute_setup_countdown_pairs(
        df = mini_mag_df['2025-01-01':]
    )
    test_stitch = build_stitched_td_df(mini_mag_df,mag_dict)
    mags_td_combo_individual_stitches[mag_ticker] = mag_dict
    mags_td_combo_complete_stitches[mag_ticker] = test_stitch




def add_td_recycle_buy_and_sell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'recycle' column marking 'R' on bars where a TD Buy/Sell Countdown bar 13
    would be recycled by an 18-bar extended Setup in the same direction.

    Assumes:
      - index is chronological (Date-like)
      - columns: 'Close', 'setup', 'countdown'
      - buy setups:  -1 .. -9
      - sell setups:  1 ..  9
      - countdown > 0 for both buy & sell Countdowns (direction inferred from setup sign nearby)
    """
    df = df.copy()
    df['recycle'] = None

    close = df['Close']

    # --- 1) Build Sequential-style extension conditions ---

    # Buy-type extension: close < close[ t-4 ]
    cond_buy_ext = close < close.shift(4)

    # Sell-type extension (mirror): close > close[ t-4 ]
    cond_sell_ext = close > close.shift(4)

    def build_run_lengths(cond_series: pd.Series) -> pd.Series:
        """Return a Series of run lengths of consecutive True values in cond_series."""
        idx = cond_series.index
        run_id = 0
        run_ids = np.zeros(len(idx), dtype=int)
        run_lengths = np.zeros(len(idx), dtype=int)

        prev_true = False
        for i, (_, is_true) in enumerate(cond_series.items()):
            if is_true:
                if not prev_true:
                    run_id += 1
                run_ids[i] = run_id
                prev_true = True
            else:
                prev_true = False

        run_id_series = pd.Series(run_ids, index=idx)
        for rid in range(1, run_id + 1):
            mask = run_id_series == rid
            length = mask.sum()
            run_lengths[mask.values] = length

        return pd.Series(run_lengths, index=idx)

    buy_run_len  = build_run_lengths(cond_buy_ext)
    sell_run_len = build_run_lengths(cond_sell_ext)

    # --- 2) Helpers to test for 18-bar extended Setup near a given bar ---

    def has_extended_buy_setup_near(idx_label, window=5, min_len=22):
        if idx_label not in df.index:
            return False
        pos = df.index.get_loc(idx_label)
        start = max(0, pos - window)
        end = min(len(df.index) - 1, pos + window)
        local_idx = df.index[start:end+1]
        return (buy_run_len.loc[local_idx] >= min_len).any()

    def has_extended_sell_setup_near(idx_label, window=5, min_len=22):
        if idx_label not in df.index:
            return False
        pos = df.index.get_loc(idx_label)
        start = max(0, pos - window)
        end = min(len(df.index) - 1, pos + window)
        local_idx = df.index[start:end+1]
        return (sell_run_len.loc[local_idx] >= min_len).any()

    # --- 3) Mark recycles for buy and sell Countdowns at bar 13 ---

    for t in df.index:
        cdn = df.at[t, 'countdown']
        if cdn != 13:
            continue

        # Infer direction of Countdown: look back a bit at setup sign around this bar
        # If the dominant setup sign nearby is negative => buy, positive => sell.
        pos = df.index.get_loc(t)
        start = max(0, pos - 15)
        end = pos
        local_setups = df['setup'].iloc[start:end+1]

        neg_count = (local_setups < 0).sum()
        pos_count = (local_setups > 0).sum()

        if neg_count > pos_count:
            # Treat as buy Countdown
            if has_extended_buy_setup_near(t):
                df.at[t, 'recycle'] = 'R'
        elif pos_count > neg_count:
            # Treat as sell Countdown
            if has_extended_sell_setup_near(t):
                df.at[t, 'recycle'] = 'R'
        else:
            # If ambiguous, you can skip or choose one side; here we skip.
            continue

    return df


def rebuild_tdst_and_risk_compact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuilds tdst_level and risk_level so that:
      - Each row has either a valid level for that bar or NaN.
      - Lines start at the correct event bars and stop when terminated.
      - Multiple legs can overlap, but are collapsed into one value per bar.
    Assumes:
      - 'Date' column or datetime index
      - 'Close','High','Low','setup','tdst_level','risk_level'
      - buy setups: -1..-9, sell setups: 1..9
    """
    df = df.copy()

    # Ensure datetime index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    df = df.sort_index()

    # store originals
    orig_tdst = df['tdst_level'].copy()
    orig_risk = df['risk_level'].copy()

    # reset working columns
    df['tdst_level'] = np.nan
    df['risk_level'] = np.nan

    # -------- utilities --------
    def infer_side(at_time):
        local = df.loc[:at_time, 'setup'].tail(15)
        neg = (local < 0).sum()
        pos = (local > 0).sum()
        return 'buy' if neg >= pos else 'sell'

    # Each leg: (start, end, level, side)
    tdst_legs = []
    risk_legs = []

    # -------- build TDST legs --------
    # find all event starts from original data
    tdst_events = []
    for t in df.index:
        v = orig_tdst.loc[t]
        if pd.notna(v):
            tdst_events.append((t, float(v), infer_side(t)))

    for i, (start_t, level, side) in enumerate(tdst_events):
        # walk forward to find termination
        end_t = df.index[-1]
        for t in df.index[df.index >= start_t]:
            # TDST break
            if side == 'buy':
                broken = df.at[t, 'Close'] > level
            else:
                broken = df.at[t, 'Close'] < level

            # optional: terminate if a *different* TDST event (any side) starts here
            is_new_event = (t != start_t) and pd.notna(orig_tdst.loc[t])

            if broken or is_new_event:
                # stop at previous bar if not at start
                pos = df.index.get_loc(t)
                end_t = df.index[pos - 1] if pos > 0 else start_t
                break
        tdst_legs.append((start_t, end_t, level, side))

    # -------- build Risk legs --------
    risk_events = []
    for t in df.index:
        v = orig_risk.loc[t]
        if pd.notna(v):
            risk_events.append((t, float(v), infer_side(t)))

    for i, (start_t, level, side) in enumerate(risk_events):
        end_t = df.index[-1]
        for t in df.index[df.index >= start_t]:
            # risk violation
            if side == 'buy':
                broken = df.at[t, 'Low'] < level
            else:
                broken = df.at[t, 'High'] > level

            is_new_event = (t != start_t) and pd.notna(orig_risk.loc[t])

            if broken or is_new_event:
                pos = df.index.get_loc(t)
                end_t = df.index[pos - 1] if pos > 0 else start_t
                break
        risk_legs.append((start_t, end_t, level, side))

    # -------- per-bar combine logic --------
    # For each bar, collect all active levels and pick one.
    for t in df.index:
        # TDST: all legs active on this bar
        active_tdst = [level for (s, e, level, side)
                       in tdst_legs if (t >= s and t <= e)]
        # Risk:
        active_risk = [level for (s, e, level, side)
                       in risk_legs if (t >= s and t <= e)]

        # Define your combine rule:
        # Example: if multiple TDST lines, keep the *latest* one (by start date).
        if active_tdst:
            # take the leg whose start is latest <= t
            candidates = [(s, level) for (s, e, level, side) in tdst_legs
                          if (t >= s and t <= e)]
            latest_start, chosen_level = max(candidates, key=lambda x: x[0])
            df.at[t, 'tdst_level'] = chosen_level

        if active_risk:
            candidates = [(s, level) for (s, e, level, side) in risk_legs
                          if (t >= s and t <= e)]
            latest_start, chosen_level = max(candidates, key=lambda x: x[0])
            df.at[t, 'risk_level'] = chosen_level

    return df

export_csv = mags_td_combo_complete_stitches['GOOGL']
recycled_export_df = add_td_recycle_buy_and_sell(export_csv)

test_export_df = rebuild_tdst_and_risk_compact(export_csv)

### ------------------------------------------------------------------------------------ ###
### ------------------------------------- BACKTEST ------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

each_mags_backtest_dict = mags_td_combo_individual_stitches['GOOGL']
each_stitch_df = each_mags_backtest_dict['setup_countdown #1']['dataframe']
each_mags_backtest_dict.keys()




each_mags_return_stream_dict = dict()
sum_cumulative_ret = 0
for each_stitch in each_mags_backtest_dict:
    each_stitch_df = each_mags_backtest_dict[each_stitch]['dataframe']
    if each_stitch_df['countdown'][-1] == 13:\

        ### FIND THE FIRST 9 BAR SETUP OCCURRENCE ###
        mask = abs(each_stitch_df['setup']) == 9
        first_idx = each_stitch_df[mask].index[0]

        ### ISOLATE DF SO IT STARTS FROM 9 TO 13 ###
        setup9_to_cd13_df = each_stitch_df[first_idx:]

        ### BUY SETUP AND COUNTDOWN ###
        if each_stitch_df['setup'][first_idx] == -9:
            setup9_to_cd13_df['pct'] = setup9_to_cd13_df['Close'].pct_change()
            setup9_to_cd13_df['cumulative_ret'] = (1 + setup9_to_cd13_df['pct']).cumprod() - 1
            daily_returns_cumulative_returns_df = (setup9_to_cd13_df[['pct','cumulative_ret']] * -1).dropna()

        ### SELL SETUP AND COUNTDOWN ###
        elif each_stitch_df['setup'][first_idx] == 9:
            setup9_to_cd13_df['pct'] = setup9_to_cd13_df['Close'].pct_change()
            setup9_to_cd13_df['cumulative_ret'] = (1 + setup9_to_cd13_df['pct']).cumprod() - 1
            daily_returns_cumulative_returns_df = (setup9_to_cd13_df[['pct','cumulative_ret']] * 1).dropna()

        each_mags_return_stream_dict[each_stitch] = daily_returns_cumulative_returns_df

        sum_cumulative_ret += daily_returns_cumulative_returns_df['cumulative_ret'][-1]


# 0) Build px and strat as you already do
px = pd.DataFrame(mags_ohlc_dict['GOOGL']['2025-01-01':]['Close'])
strat = pd.DataFrame(pd.concat(each_mags_return_stream_dict.values()))

# 1) Ensure datetime index and sorted
px.index = pd.to_datetime(px.index)
strat.index = pd.to_datetime(strat.index)

px = px.sort_index()
strat = strat.sort_index()

# 2) Ensure indices are UNIQUE (drop any duplicate dates)
px = px[~px.index.duplicated(keep='first')]
strat = strat[~strat.index.duplicated(keep='first')]
# (use keep='last' instead if that makes more sense for you)[web:176][web:173]

# 3) Keep only the 'pct' column from strat (if it has others)
#    adjust if your column is named differently
strat = strat[['pct']]

# 4) Reindex strategy to full trading-day index of px
strat_full = strat.reindex(px.index)

# 5) Fill missing pct with 0 on days the strategy is not live
strat_full['pct'] = strat_full['pct'].fillna(0.0)

# 6) Recompute cumulative_ret from the filled pct
strat_full['cumulative_ret'] = (1 + strat_full['pct']).cumprod() - 1

plt.plot(strat_full['cumulative_ret'])
plt.show()


(strat_full['pct'].mean() * 252) / (strat_full['pct'].std() * 252**0.5)


mags_ohlc_dict['GOOGL']


### ------------------------------------------------------------------------------------ ###
### -------------------------------------- CHARTS -------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

# plot_td_combo_case_study_stitched(test_stitch,'n')
# plot_td_combo_case_study_test_run(googl_dict,1,'n')


def plot_stitched_googl():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['GOOGL'])

def plot_stitched_amzn():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['AMZN'])

def plot_stitched_aapl():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['AAPL'])

def plot_stitched_meta():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['META'])

def plot_stitched_msft():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['MSFT'])

def plot_stitched_nvda():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['NVDA'])

def plot_stitched_tsla():
    plot_td_combo_case_study_stitched(mags_td_combo_complete_stitches['TSLA'])

