import pandas as pd
import numpy as np

from app_td_combo import mini_mag_df


def build_setup_registry(df: pd.DataFrame):
    """
    Build a registry of completed TD setups (buy & sell) with correct TDST levels.

    Assumes:
      - df is chronological, index is date-like
      - columns: 'Open','High','Low','Close','setup'
      - buy setups:  -1 .. -9 (and possibly extended beyond -9)
      - sell setups:  1 ..  9 (and possibly extended beyond 9)

    Returns:
      - list of dicts, each:
        {
          'id': int,
          'side': 'buy'|'sell',
          'start_idx': int,   # index position in df
          'end_idx':   int,   # index position in df (last setup bar, incl. extension)
          'start_date': Timestamp,
          'end_date':   Timestamp,
          'true_high': float,     # over setup window (for range comps)
          'true_low':  float,
          'true_range': float,
          'tdst_val': float,      # TDST resistance/support
          'tdst_date': Timestamp, # anchor bar of TDST
        }
    """

    df = df.copy()
    setup_col = "setup"
    close = df["Close"]

    registry = []
    setup_id = 1

    n = len(df.index)
    i = 0
    while i < n:
        s = df.iat[i, df.columns.get_loc(setup_col)]

        # look for setup bar 1 (buy or sell)
        if s == -1 or s == 1:
            side = "buy" if s == -1 else "sell"
            start_pos = i
            start_date = df.index[start_pos]

            # build setup sequence 1..9, allow extension beyond 9
            end_pos = start_pos
            expected = -1 if side == "buy" else 1
            k = 0

            while end_pos < n:
                setup_val = df.iat[end_pos, df.columns.get_loc(setup_col)]

                if side == "buy":
                    # expect -1,-2,...,-9,... for extension
                    if setup_val != expected:
                        break
                    expected -= 1
                else:
                    # expect 1,2,...,9,...
                    if setup_val != expected:
                        break
                    expected += 1

                k += 1
                end_pos += 1

                if k >= 9:
                    # continued extension 10..18 if present
                    while end_pos < n:
                        next_val = df.iat[end_pos, df.columns.get_loc(setup_col)]
                        if side == "buy" and next_val == expected:
                            expected -= 1
                            k += 1
                            end_pos += 1
                            continue
                        if side == "sell" and next_val == expected:
                            expected += 1
                            k += 1
                            end_pos += 1
                            continue
                        break
                    break

            # require at least 9 bars to treat as completed setup
            if k >= 9 and start_pos > 0:
                end_pos -= 1  # last setup bar
                end_date = df.index[end_pos]

                # ---------- true-range over setup window (bars 1..k) ----------
                setup_win = df.iloc[start_pos:end_pos+1].copy()
                prev_close = setup_win["Close"].shift(1)
                setup_win["true_high"] = setup_win["High"].combine(prev_close, np.maximum)
                setup_win["true_low"]  = setup_win["Low"].combine(prev_close, np.minimum)
                setup_win = setup_win.iloc[1:]  # drop first where prev_close is NaN

                true_high  = float(setup_win["true_high"].max())
                true_low   = float(setup_win["true_low"].min())
                true_range = true_high - true_low

                # ---------- TDST calculation (DeMark window) ----------
                # Window: bar prior to setup bar 1 through setup bar 9
                prev_bar = df.index[start_pos - 1]
                bar9     = df.index[start_pos + 8]

                tdst_win   = df.loc[prev_bar:bar9].copy()
                prev_close_tdst = tdst_win["Close"].shift(1)

                # drop first row (prev_bar) so prev_close is aligned
                tdst_win = tdst_win.iloc[1:]
                prev_close_tdst = prev_close_tdst.iloc[1:]

                if side == "buy":
                    # TD Buy setup -> TDST resistance = max true_high in this window
                    true_high_arr = np.maximum(tdst_win["High"].values,
                                               prev_close_tdst.values)
                    tdst_win = tdst_win.assign(true_high=true_high_arr)
                    tdst_val = float(tdst_win["true_high"].max())
                    tdst_date = tdst_win["true_high"].idxmax()
                else:
                    # TD Sell setup -> TDST support = min true_low in this window
                    true_low_arr = np.minimum(tdst_win["Low"].values,
                                              prev_close_tdst.values)
                    tdst_win = tdst_win.assign(true_low=true_low_arr)
                    tdst_val = float(tdst_win["true_low"].min())
                    tdst_date = tdst_win["true_low"].idxmin()

                registry.append(
                    {
                        "id": setup_id,
                        "side": side,
                        "start_idx": start_pos,
                        "end_idx": end_pos,
                        "start_date": start_date,
                        "end_date": end_date,
                        "true_high": true_high,
                        "true_low": true_low,
                        "true_range": true_range,
                        "tdst_val": tdst_val,
                        "tdst_date": tdst_date,
                    }
                )
                setup_id += 1

                # skip ahead to avoid re-detecting this setup's bar 1
                i = end_pos + 1
                continue

        i += 1

    return registry

def apply_same_side_cancellation_qualifiers(df: pd.DataFrame, registry):
    """
    Apply DeMark TD Countdown Cancellation Qualifier I & II *between same-side setups*.

    For each pair of consecutive setups with same 'side':

      - Compute true-range ratio:
          ratio = new_true_range / prior_true_range

      - Qualifier I (range-based):
          If 1.0 <= ratio < 1.618:
             new setup is large enough to cancel prior countdown,
             and becomes the active setup.

      - Qualifier II (setup within setup):
          If the new setup's *closing range* and its extreme close
          lie entirely within the prior setup's true range,
          then the PRIOR setup remains active and the new one is
          treated as "inside"; prior's countdown stays attached.

    Visual TDST handling:
      - If the NEW setup takes over (Qualifier I), we clip the PRIOR
        setup's TDST at the new setup's bar-9 date.

      - If the PRIOR remains active (Qualifier II), we leave its TDST
        end date unchanged here (other logic like TDST break may still
        modify it later).

    This function adds:
      - 'active_after' : bool (whether this setup remains the active one
                         after considering the next same-side setup)
      - 'tdst_end_date' : updated for prior setups that get cancelled
                          by a new same-side setup.
    """

    # First, sort registry chronologically by start_date
    registry = sorted(registry, key=lambda r: r["start_date"])

    # Precompute for each setup the bar-9 date (end_date of the initial 1..9 part)
    # We assume 'end_idx' may include extension beyond 9; bar-9 is start_idx+8.
    for rec in registry:
        start_pos = rec["start_idx"]
        bar9_pos  = start_pos + 8
        if bar9_pos < len(df.index):
            rec["bar9_date"] = df.index[bar9_pos]
        else:
            rec["bar9_date"] = rec["end_date"]

        # initialize TDST end as None; will be set later
        rec.setdefault("tdst_end_date", None)
        rec["active_after"] = True

    # Group setups by side (buy/sell)
    by_side = {"buy": [], "sell": []}
    for rec in registry:
        by_side[rec["side"]].append(rec)

    for side, setups in by_side.items():
        if len(setups) < 2:
            continue

        # process consecutive same-side pairs
        for i in range(len(setups) - 1):
            prior = setups[i]
            new   = setups[i + 1]

            # Only consider if new starts AFTER prior (chronologically)
            if new["start_date"] <= prior["start_date"]:
                continue

            prior_range = prior["true_range"]
            new_range   = new["true_range"]

            if prior_range <= 0:
                continue

            ratio = new_range / prior_range

            # ---------- Qualifier II: "setup within a setup" ----------
            # New setup's *closing range* and extreme close inside prior true range.
            prior_hi = prior["true_high"]
            prior_lo = prior["true_low"]

            # closing range of new setup: min/max closes in its window
            win_new = df.iloc[new["start_idx"]: new["end_idx"] + 1]
            close_min = win_new["Close"].min()
            close_max = win_new["Close"].max()
            extreme_close = win_new["Close"].iloc[-1]  # last close of setup

            within_true_range = (
                close_min >= prior_lo and close_max <= prior_hi
                and prior_lo <= extreme_close <= prior_hi
            )

            # ---------- Qualifier I: range comparison ----------
            q1_cancels_prior = (ratio >= 1.0 and ratio < 1.618)

            # ---------- Decision logic ----------
            if within_true_range:
                # Qualifier II: prior setup remains active; TDST stays for now
                new["active_after"]   = False   # new is subordinate to prior
                # We do NOT clip prior's TDST here.
                continue

            if q1_cancels_prior:
                # Qualifier I: new setup cancels prior countdown, becomes active
                prior["active_after"] = False

                # VISUAL: clip prior TDST line at new's bar-9 date
                prior["tdst_end_date"] = new["bar9_date"]

                # new remains active_after = True
            else:
                # Neither qualifier triggered: by default allow new to become
                # the active reference going forward, and prior becomes historical.
                prior["active_after"] = False
                prior["tdst_end_date"] = new["bar9_date"]

    # For any setup without a TDST end yet, default to end of data
    last_date = df.index[-1]
    for rec in registry:
        if rec["tdst_end_date"] is None:
            rec["tdst_end_date"] = last_date
        rec["tdst_duration_bars"] = (
            df.index.get_loc(rec["tdst_end_date"]) - df.index.get_loc(rec["tdst_date"])
        )

    return registry

def apply_same_side_tdst_rules(df: pd.DataFrame, registry):
    """
    Apply DeMark same-side TD Countdown Cancellation Qualifier I & II
    to determine *where* each setup's TDST line should end.

    Rules implemented:

      - TDST cannot end before the next same-side setup 9.
      - For a pair of consecutive same-side setups (prior, new):

          * Qualifier II (setup within setup):
                If the NEW setup's closing range and extreme close
                are entirely within the PRIOR setup's true range,
                THEN the PRIOR setup remains active and its TDST
                is NOT forced to end at the new bar-9.
                -> TDST can extend beyond that bar-9.

          * Qualifier I (range comparison):
                Compare true_range_new / true_range_prior.
                If 1.0 <= ratio < 1.618, the NEW setup cancels
                the PRIOR countdown and becomes the active setup.
                -> PRIOR TDST ends at the new setup's bar-9.

      - If neither Qualifier I nor II is satisfied, we still let the
        NEW setup take over visually (to avoid clutter), so the PRIOR
        TDST ends at the new bar-9. You can relax this if your chart
        shows different behaviour.

    registry is modified in-place to include:
      - 'bar9_date'      : date of that setup's bar 9
      - 'tdst_end_date'  : final TDST end (at least next same-side 9)
      - 'tdst_duration_bars'
      - 'active_after'   : whether this setup remains the active one
                           after processing the next same-side setup.
    """

    # sort chronologically
    registry = sorted(registry, key=lambda r: r["start_date"])

    # precompute bar-9 date for each setup
    for rec in registry:
        start_pos = rec["start_idx"]
        bar9_pos  = start_pos + 8
        if bar9_pos < len(df.index):
            rec["bar9_date"] = df.index[bar9_pos]
        else:
            rec["bar9_date"] = rec["end_date"]

        rec["tdst_end_date"] = None
        rec["active_after"]  = True

    # group by side
    by_side = {"buy": [], "sell": []}
    for rec in registry:
        by_side[rec["side"]].append(rec)

    for side, setups in by_side.items():
        if len(setups) < 2:
            continue

        for i in range(len(setups) - 1):
            prior = setups[i]
            new   = setups[i + 1]

            # only forward in time
            if new["start_date"] <= prior["start_date"]:
                continue

            prior_range = prior["true_range"]
            new_range   = new["true_range"]
            if prior_range <= 0:
                continue

            ratio = new_range / prior_range

            prior_hi = prior["true_high"]
            prior_lo = prior["true_low"]

            # window of new setup
            win_new = df.iloc[new["start_idx"]: new["end_idx"] + 1]
            close_min = win_new["Close"].min()
            close_max = win_new["Close"].max()
            extreme_close = win_new["Close"].iloc[-1]

            # Qualifier II: setup within prior's true range
            within_true_range = (
                close_min >= prior_lo and close_max <= prior_hi
                and prior_lo <= extreme_close <= prior_hi
            )

            # Qualifier I: range ratio
            q1_cancels_prior = (ratio >= 1.0 and ratio < 1.618)

            # --- decision ---
            if within_true_range:
                # Qualifier II: prior remains active, TDST can extend beyond new bar-9.
                new["active_after"]   = False
                # Do not set prior['tdst_end_date'] here.
                continue

            if q1_cancels_prior:
                # Qualifier I: new takes over, prior TDST ends at new bar-9.
                prior["active_after"]  = False
                prior["tdst_end_date"] = new["bar9_date"]
                # new remains active_after=True
            else:
                # Neither I nor II: by default, let new take over visually and
                # clip prior at new bar-9 (you can relax this if needed).
                prior["active_after"]  = False
                prior["tdst_end_date"] = new["bar9_date"]

    # any setup that still has no tdst_end_date gets end-of-data
    last_date = df.index[-1]
    for rec in registry:
        if rec["tdst_end_date"] is None:
            rec["tdst_end_date"] = last_date
        rec["tdst_duration_bars"] = (
            df.index.get_loc(rec["tdst_end_date"]) -
            df.index.get_loc(rec["tdst_date"])
        )

    return registry

def apply_tdst_qualified_breaks_after_bar9(df: pd.DataFrame, registry):
    """
    Symmetric TDST break logic for both buy and sell setups.

    For each setup in registry (after same-side TDST rules):

      - Start scanning from bar-9+1 (cannot end before own bar-9).
      - Do not scan beyond any prelim tdst_end_date already set
        by same-side qualifiers (e.g., next same-side bar-9).
      - For BUY setups (TDST resistance above):
            look for QUALIFIED + CONFIRMED *upside* breakout
            through tdst_val.
        For SELL setups (TDST support below):
            look for QUALIFIED + CONFIRMED *downside* breakout
            through tdst_val.

      - On first qualified+confirmed break, set tdst_end_date to
        the CONFIRM bar; otherwise keep prelim end.
    """

    def is_qualified_break(i, level, side):
        """Qualified bar i vs TDST level."""
        if i < 2:
            return False

        c  = df.iloc[i]["Close"]
        c1 = df.iloc[i-1]["Close"]
        c2 = df.iloc[i-2]["Close"]

        if side == "buy":
            # upside break of resistance:
            #  - close > level
            #  - up close
            #  - prior bar down close
            return (c > level) and (c > c1) and (c1 < c2)

        else:  # side == "sell"
            # downside break of support:
            #  - close < level
            #  - down close
            #  - prior bar up close
            return (c < level) and (c < c1) and (c1 > c2)

    def is_confirmed_break(i, side):
        """Confirm on bar i+1."""
        if i + 1 >= len(df.index):
            return False

        # qualified bar
        o_q = df.iloc[i]["Open"]
        h_q = df.iloc[i]["High"]
        l_q = df.iloc[i]["Low"]
        c_q = df.iloc[i]["Close"]

        # confirm bar
        o_c = df.iloc[i+1]["Open"]
        h_c = df.iloc[i+1]["High"]
        l_c = df.iloc[i+1]["Low"]
        c_c = df.iloc[i+1]["Close"]

        if side == "buy":
            # upside confirmation:
            #  - open > prior open
            #  - high > max(high of prior 3 bars)
            #  - close > prior close
            hi_prev3 = df["High"].iloc[max(0, i-2): i+1].max()
            cond_open  = o_c > o_q
            cond_high  = h_c > hi_prev3
            cond_close = c_c > c_q
            return cond_open and cond_high and cond_close

        else:  # sell
            # downside confirmation:
            #  - open < prior open
            #  - low < min(low of prior 3 bars)
            #  - close < prior close
            lo_prev3 = df["Low"].iloc[max(0, i-2): i+1].min()
            cond_open  = o_c < o_q
            cond_low   = l_c < lo_prev3
            cond_close = c_c < c_q
            return cond_open and cond_low and cond_close

    for rec in registry:
        side      = rec["side"]          # 'buy' or 'sell'
        level     = rec["tdst_val"]
        tdst_date = rec["tdst_date"]
        bar9_date = rec["bar9_date"]

        tdst_pos  = df.index.get_loc(tdst_date)
        bar9_pos  = df.index.get_loc(bar9_date)

        # Never end before own bar-9
        start_pos = bar9_pos + 1

        prelim_end   = rec.get("tdst_end_date", df.index[-1])
        prelim_end_pos = df.index.get_loc(prelim_end)

        if start_pos > prelim_end_pos:
            rec["tdst_end_date"] = prelim_end
            rec["tdst_duration_bars"] = prelim_end_pos - tdst_pos
            continue

        break_idx = None

        for i in range(start_pos, prelim_end_pos + 1):
            if is_qualified_break(i, level, side) and is_confirmed_break(i, side):
                break_idx = i + 1  # confirm bar
                break

        if break_idx is not None:
            rec["tdst_end_date"] = df.index[break_idx]
        else:
            rec["tdst_end_date"] = prelim_end

        rec["tdst_duration_bars"] = (
            df.index.get_loc(rec["tdst_end_date"]) - tdst_pos
        )

    return registry




setups = build_setup_registry(mini_mag_df)
setups_end_dates = apply_same_side_tdst_rules(mini_mag_df, setups)
filtred_setups_end_dates = apply_tdst_qualified_breaks_after_bar9(mini_mag_df, setups_end_dates)


import pandas as pd
import numpy as np

def _find_setup_rec_for_start(df, registry, setup_start_ts, side):
    for rec in registry:
        if rec["side"] == side and rec["start_date"] == setup_start_ts:
            return rec
    return None


def compute_setup_countdown_pairs(df: pd.DataFrame, registry):
    df = df.copy()

    setup_col = "setup"
    countdown_col = "countdown"
    df[countdown_col] = 0

    setup_rows_to_iterate = df[(df[setup_col] == -1) | (df[setup_col] == 1)].index

    setup_countdown_dict = dict()
    setup_dict_key_index = 1

    for setup_start in setup_rows_to_iterate:

        n = len(df.index)
        setup_pos = df.index.get_loc(setup_start)

        if setup_pos - 1 < 0 or setup_pos + 8 >= n:
            continue

        start_val = df.at[setup_start, setup_col]
        side = "buy" if start_val < 0 else "sell"

        # ---------- TDST FROM REGISTRY ----------
        setup_start_ts = setup_start
        rec = _find_setup_rec_for_start(df, registry, setup_start_ts, side)

        if rec is None:
            tdst_val = None
            tdst_date = None
            tdst_end_date = None
            tdst_duration_bars = None
            tdst_setup_num = start_val
        else:
            tdst_val = rec["tdst_val"]
            tdst_date = rec["tdst_date"]
            tdst_end_date = rec["tdst_end_date"]
            tdst_duration_bars = rec["tdst_duration_bars"]
            tdst_setup_num = df.loc[tdst_date, setup_col]

        # ---------- FIRST (RAW) TDST BREAK WINDOW ----------
        if tdst_val is None:
            # if we have no TDST, just use last bar as window end
            first_tdst_break = df.index[-1]
        else:
            if side == "buy":
                mask = df.loc[setup_start:, 'Close'] > tdst_val
            else:
                mask = df.loc[setup_start:, 'Close'] < tdst_val

            first_tdst_break = mask.idxmax()
            if first_tdst_break == setup_start:
                first_tdst_break = df.index[-1]

        # ---------- COUNTDOWN ----------
        df[countdown_col] = 0
        setup_idx    = setup_pos
        cdn_num      = 1
        last_cdn_bar = None

        for i in range(setup_idx, len(df.index)):
            if cdn_num > 13:
                break
            if i < 2:
                continue

            row    = df.index[i]
            row_t1 = df.index[i - 1]
            row_t2 = df.index[i - 2]

            c = df.loc[row, 'Close']
            l = df.loc[row, 'Low']
            h = df.loc[row, 'High']
            o = df.loc[row, 'Open']

            if side == "buy":
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
            else:  # sell
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

        # TDST + static info (now includes tdst_end_date)
        result = {
            'side': side,
            'tdst_val': tdst_val,
            'tdst_date': tdst_date,
            'tdst_end_date': tdst_end_date,
            'tdst_duration_bars': tdst_duration_bars,
            'tdst_setup_start_bar': tdst_setup_num,
        }

        completed_cdn = cdn_num - 1

        if completed_cdn >= 13 and last_cdn_bar is not None:
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
            live_end = last_cdn_bar if last_cdn_bar is not None else window.index[-1]
            result.update({
                'valid setup-cdn_pair': 'live',
                'dataframe': window.loc[setup_start:live_end],
                'risk_lvl':  None,
                'risk_date': None,
            })

        setup_countdown_dict[name] = result

    return setup_countdown_dict


registry = build_setup_registry(mini_mag_df)
registry = apply_same_side_tdst_rules(mini_mag_df, registry)  # or apply_same_side_cancellation_qualifiers
registry = apply_tdst_qualified_breaks_after_bar9(mini_mag_df, registry)

setup_countdown_dict = compute_setup_countdown_pairs(mini_mag_df, registry)





