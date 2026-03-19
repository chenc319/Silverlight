"""
Chart image renderer using matplotlib.
Generates OHLC bar charts with DeMark annotations per the DeMark Technician spec:
- OHLC bars in light grey (#f2f2f2)
- Setup counts in green (#00ff00)
- Sequential Countdown in red (#ff0000)
- Combo Countdown in magenta (#ff00ff)
- Bollinger Bands in muted grey
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


def render_chart_image(
    ohlc_data: list,
    ticker: str,
    timeframe: str = 'daily',
    width: int = 1200,
    height: int = 600,
    show_bb: bool = True,
    show_demark: bool = True,
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

    # Figure setup
    dpi = 150
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
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

    # Bollinger Bands
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
                # Perfection dot
                if count == 9 and perfected[i]:
                    ax.text(i, y_pos - offset_small, '●', color=COLOR_SETUP,
                            fontsize=5, ha='center', va='top')

            elif setup_sell[i] > 0:
                count = setup_sell[i]
                y_pos = highs[i] + offset_large
                fontsize = 7 if count < 9 else 9
                fontweight = 'bold' if count == 9 else 'normal'
                ax.text(i, y_pos, str(count), color=COLOR_SETUP,
                        fontsize=fontsize, fontweight=fontweight,
                        ha='center', va='bottom')
                if count == 9 and perfected[i]:
                    ax.text(i, y_pos + offset_small, '●', color=COLOR_SETUP,
                            fontsize=5, ha='center', va='bottom')

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

    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT, labelsize=7)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.grid(True, axis='y', color=COLOR_GRID, linewidth=0.3, alpha=0.5)

    # X-axis: show every Nth date label
    step = max(1, n // 12)
    tick_positions = list(range(0, n, step))
    tick_labels = [dates[i][:10] if i < n else '' for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=6, color=COLOR_TEXT)

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
