"""
Chart image renderer using matplotlib.
Generates OHLC bar charts with DeMark annotations per the DeMark Technician spec:
- OHLC bars in light grey (#f2f2f2)
- Setup counts in green (#00ff00)
- Sequential Countdown in red (#ff0000)
- Combo Countdown in magenta (#ff00ff)
- TDST lines: green (buy resistance), magenta (sell support)
- Risk Level lines: red (sequential), magenta (combo)
- Slow Stochastic subplot (optional)
- Bollinger Bands (optional, off by default)
"""

import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# DeMark color scheme
COLOR_BG = '#0a0a0f'
COLOR_SURFACE = '#12121a'
COLOR_BAR = '#c8c8c8'       # Light grey OHLC bars
COLOR_SETUP = '#00ff00'      # Green for all setup counts
COLOR_SEQ_CD = '#ff0000'     # Red for Sequential Countdown
COLOR_COMBO_CD = '#ff00ff'   # Magenta for Combo Countdown
COLOR_BB = '#94a3b844'       # Muted BB lines
COLOR_BB_MID = '#94a3b822'
COLOR_GRID = '#1e1e2e'
COLOR_TEXT = '#94a3b8'
COLOR_GREEN = '#00d4a0'
COLOR_RED = '#ff4d6d'

# TDST / Risk Level colors
COLOR_TDST_BUY = '#00ff00'   # Green dashed — buy TDST resistance
COLOR_TDST_SELL = '#ff00ff'  # Magenta dashed — sell TDST support
COLOR_RISK_SEQ = '#ff0000'   # Red dotted — sequential risk
COLOR_RISK_COMBO = '#ff00ff' # Magenta dotted — combo risk

# Stochastic colors
COLOR_STOCH_K = '#5599ff'
COLOR_STOCH_D = '#ff9933'


def render_chart_image(
    ohlc_data: list,
    ticker: str,
    timeframe: str = 'daily',
    width: int = 1200,
    height: int = 750,
    show_bb: bool = False,
    show_demark: bool = True,
    show_stoch: bool = True,
) -> bytes:
    """
    Render an OHLC bar chart with DeMark annotations as a PNG image.
    Returns PNG bytes.
    """
    if not ohlc_data:
        return _empty_chart_image(ticker, width, height)

    df = pd.DataFrame(ohlc_data)

    # Trim to last N bars for readability
    max_bars = 120 if timeframe == 'daily' else 80
    if len(df) > max_bars:
        df = df.iloc[-max_bars:].reset_index(drop=True)

    n = len(df)
    dates = df['date'].values
    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)

    # Figure setup — with or without stochastic subplot
    dpi = 150
    fig_w = width / dpi
    fig_h = height / dpi

    if show_stoch:
        fig, (ax, ax_stoch) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h), dpi=dpi,
            gridspec_kw={'height_ratios': [4, 1]},
            sharex=True,
        )
        ax_stoch.set_facecolor(COLOR_BG)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
        ax_stoch = None

    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Price range for offset calculations
    price_min = lows.min()
    price_max = highs.max()
    price_span = price_max - price_min
    if price_span == 0:
        price_span = max(abs(closes.mean()), 1.0)

    offset_small = 0.012 * price_span
    offset_large = 0.02 * price_span

    # Draw OHLC bars
    for i in range(n):
        color = COLOR_GREEN if closes[i] >= opens[i] else COLOR_RED
        # Vertical line (high-low)
        ax.plot([i, i], [lows[i], highs[i]], color=COLOR_BAR, linewidth=0.8, solid_capstyle='round')
        # Open tick (left)
        ax.plot([i - 0.3, i], [opens[i], opens[i]], color=COLOR_BAR, linewidth=0.8, solid_capstyle='round')
        # Close tick (right)
        ax.plot([i, i + 0.3], [closes[i], closes[i]], color=COLOR_BAR, linewidth=0.8, solid_capstyle='round')

    # Bollinger Bands (off by default)
    if show_bb:
        bb_upper = df.get('bb_upper', pd.Series(dtype=float)).values.astype(float)
        bb_mid = df.get('bb_mid', pd.Series(dtype=float)).values.astype(float)
        bb_lower = df.get('bb_lower', pd.Series(dtype=float)).values.astype(float)

        valid_bb = ~np.isnan(bb_upper) & (bb_upper > 0)
        x_bb = np.arange(n)[valid_bb]
        if len(x_bb) > 1:
            ax.plot(x_bb, bb_upper[valid_bb], color=COLOR_BB, linewidth=0.7)
            ax.plot(x_bb, bb_mid[valid_bb], color=COLOR_BB_MID, linewidth=0.5, linestyle='--')
            ax.plot(x_bb, bb_lower[valid_bb], color=COLOR_BB, linewidth=0.7)
            ax.fill_between(x_bb, bb_upper[valid_bb], bb_lower[valid_bb], color='#94a3b808')

    # ── TDST lines ──
    if show_demark:
        _draw_level_line(ax, df, n, 'tdst_buy_level', COLOR_TDST_BUY, linestyle='--', linewidth=0.9)
        _draw_level_line(ax, df, n, 'tdst_sell_level', COLOR_TDST_SELL, linestyle='--', linewidth=0.9)

        # ── Risk Level lines ──
        _draw_level_line(ax, df, n, 'risk_seq_buy_level', COLOR_RISK_SEQ, linestyle=':', linewidth=0.8)
        _draw_level_line(ax, df, n, 'risk_seq_sell_level', COLOR_RISK_SEQ, linestyle=':', linewidth=0.8)
        _draw_level_line(ax, df, n, 'risk_combo_buy_level', COLOR_RISK_COMBO, linestyle=':', linewidth=0.8)
        _draw_level_line(ax, df, n, 'risk_combo_sell_level', COLOR_RISK_COMBO, linestyle=':', linewidth=0.8)

    # DeMark annotations
    if show_demark:
        setup_buy = df.get('td_setup_buy', pd.Series(np.zeros(n), dtype=int)).values.astype(int)
        setup_sell = df.get('td_setup_sell', pd.Series(np.zeros(n), dtype=int)).values.astype(int)
        seq_buy = df.get('seq_cd_buy', pd.Series(np.zeros(n), dtype=int)).values.astype(int)
        seq_sell = df.get('seq_cd_sell', pd.Series(np.zeros(n), dtype=int)).values.astype(int)
        combo_buy = df.get('combo_cd_buy', pd.Series(np.zeros(n), dtype=int)).values.astype(int)
        combo_sell = df.get('combo_cd_sell', pd.Series(np.zeros(n), dtype=int)).values.astype(int)
        perfected = df.get('td_perfected', pd.Series(np.zeros(n), dtype=int)).values.astype(int)

        for i in range(n):
            # Setup labels (green, closest to bars)
            if setup_buy[i] > 0:
                count = setup_buy[i]
                y_pos = lows[i] - offset_large
                fontsize = 7 if count < 9 else 9
                fontweight = 'bold' if count == 9 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_SETUP,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='top')
                # Perfection dot — bigger, further below
                if count == 9 and perfected[i]:
                    ax.text(i, y_pos - 2 * offset_small, '●', color=COLOR_SETUP,
                            fontsize=8, ha='center', va='top')

            elif setup_sell[i] > 0:
                count = setup_sell[i]
                y_pos = highs[i] + offset_large
                fontsize = 7 if count < 9 else 9
                fontweight = 'bold' if count == 9 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_SETUP,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='bottom')
                if count == 9 and perfected[i]:
                    ax.text(i, y_pos + 2 * offset_small, '●', color=COLOR_SETUP,
                            fontsize=8, ha='center', va='bottom')

            # Sequential Countdown labels (red, further from bars)
            if seq_buy[i] > 0:
                count = seq_buy[i]
                y_pos = lows[i] - offset_large - 2.5 * offset_small
                fontsize = 6 if count < 13 else 8
                fontweight = 'bold' if count == 13 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_SEQ_CD,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='top')

            elif seq_sell[i] > 0:
                count = seq_sell[i]
                y_pos = highs[i] + offset_large + 2.5 * offset_small
                fontsize = 6 if count < 13 else 8
                fontweight = 'bold' if count == 13 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_SEQ_CD,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='bottom')

            # Combo Countdown labels (magenta, furthest from bars)
            if combo_buy[i] > 0:
                count = combo_buy[i]
                y_pos = lows[i] - offset_large - 5.0 * offset_small
                fontsize = 6 if count < 13 else 8
                fontweight = 'bold' if count == 13 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_COMBO_CD,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='top')

            elif combo_sell[i] > 0:
                count = combo_sell[i]
                y_pos = highs[i] + offset_large + 5.0 * offset_small
                fontsize = 6 if count < 13 else 8
                fontweight = 'bold' if count == 13 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_COMBO_CD,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='bottom')

    # ── Slow Stochastic subplot ──
    if show_stoch and ax_stoch is not None:
        stoch_k = df.get('stoch_k', pd.Series(np.full(n, 50.0))).values.astype(float)
        stoch_d = df.get('stoch_d', pd.Series(np.full(n, 50.0))).values.astype(float)
        x_arr = np.arange(n)

        ax_stoch.plot(x_arr, stoch_k, color=COLOR_STOCH_K, linewidth=0.9, label='%K')
        ax_stoch.plot(x_arr, stoch_d, color=COLOR_STOCH_D, linewidth=0.7, linestyle='--', label='%D')

        # Overbought / oversold reference lines
        ax_stoch.axhline(y=80, color='#ff4d6d44', linewidth=0.5, linestyle='-')
        ax_stoch.axhline(y=20, color='#00d4a044', linewidth=0.5, linestyle='-')

        ax_stoch.set_ylim(0, 100)
        ax_stoch.set_ylabel('%K/%D', color=COLOR_TEXT, fontsize=6, labelpad=2)
        ax_stoch.yaxis.set_label_position('right')
        ax_stoch.yaxis.tick_right()
        ax_stoch.set_yticks([20, 50, 80])
        ax_stoch.tick_params(colors=COLOR_TEXT, labelsize=6)
        ax_stoch.grid(True, axis='y', color=COLOR_GRID, linewidth=0.3, alpha=0.5)
        ax_stoch.spines['top'].set_visible(False)
        ax_stoch.spines['right'].set_color(COLOR_GRID)
        ax_stoch.spines['bottom'].set_color(COLOR_GRID)
        ax_stoch.spines['left'].set_color(COLOR_GRID)

        # Compact legend
        ax_stoch.legend(loc='upper left', fontsize=5,
                        facecolor=COLOR_SURFACE, edgecolor=COLOR_GRID,
                        labelcolor=COLOR_TEXT, framealpha=0.8)

    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT, labelsize=7)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.grid(True, axis='y', color=COLOR_GRID, linewidth=0.3, alpha=0.5)

    # X-axis: show every Nth date label (on bottom axis only)
    target_ax = ax_stoch if ax_stoch is not None else ax
    step = max(1, n // 12)
    tick_positions = list(range(0, n, step))
    tick_labels = [dates[i][:10] if i < n else '' for i in tick_positions]
    target_ax.set_xticks(tick_positions)
    target_ax.set_xticklabels(tick_labels, rotation=0, fontsize=6, color=COLOR_TEXT)

    if ax_stoch is not None:
        ax.tick_params(axis='x', labelbottom=False)

    # Last price annotation
    last_close = closes[-1]
    last_color = COLOR_GREEN if closes[-1] >= opens[-1] else COLOR_RED
    ax.axhline(y=last_close, color=last_color, linewidth=0.4, linestyle='--', alpha=0.5)
    ax.text(n + 0.5, last_close, f'{last_close:.2f}', color=last_color,
            fontsize=7, fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLOR_SURFACE, edgecolor=last_color, alpha=0.9))

    # Title
    ax.set_title(f'{ticker}  {timeframe.capitalize()}', color='#e2e8f0',
                 fontsize=10, fontweight='bold', loc='left', pad=8)

    # Legend for DeMark colors
    if show_demark:
        legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_SETUP, markersize=5, label='Setup'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_SEQ_CD, markersize=5, label='Seq CD'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_COMBO_CD, markersize=5, label='Combo CD'),
            Line2D([0], [0], color=COLOR_TDST_BUY, linewidth=0.8, linestyle='--', label='TDST'),
            Line2D([0], [0], color=COLOR_RISK_SEQ, linewidth=0.8, linestyle=':', label='Risk Lvl'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=6,
                  facecolor=COLOR_SURFACE, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT,
                  framealpha=0.8)

    # Padding
    ax.set_xlim(-1, n + 4)
    y_pad = price_span * 0.15
    ax.set_ylim(price_min - y_pad, price_max + y_pad)

    plt.tight_layout(pad=0.5)

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _draw_level_line(ax, df, n, col_name, color, linestyle='--', linewidth=0.8):
    """
    Draw a horizontal level line for TDST or Risk Level data.
    Only draws segments where the value is non-zero.
    Handles staircase-style data (level changes produce separate segments).
    """
    vals = df.get(col_name, pd.Series(np.zeros(n))).values.astype(float)

    # Find contiguous segments of the same non-zero level
    i = 0
    while i < n:
        if vals[i] == 0 or np.isnan(vals[i]):
            i += 1
            continue
        # Start of a segment
        level = vals[i]
        seg_start = i
        while i < n and vals[i] == level:
            i += 1
        seg_end = i - 1
        # Draw the horizontal segment
        ax.plot([seg_start, seg_end], [level, level],
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7)


def _empty_chart_image(ticker: str, width: int, height: int) -> bytes:
    dpi = 150
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.text(0.5, 0.5, f'No data for {ticker}', color=COLOR_TEXT,
            fontsize=12, ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
