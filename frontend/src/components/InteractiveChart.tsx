import { useState, useEffect, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import { fetchOHLC } from '@/lib/api';

// ── DeMark color scheme ──
const COLOR_BG = '#0a0a0f';
const COLOR_SURFACE = '#12121a';
const COLOR_BAR_UP = '#c8c8c8';
const COLOR_BAR_DOWN = '#c8c8c8';
const COLOR_SETUP = '#00ff00';
const COLOR_SEQ_CD = '#ff0000';
const COLOR_COMBO_CD = '#ff00ff';
const COLOR_TDST_BUY = '#00ff00';
const COLOR_TDST_SELL = '#ff00ff';
const COLOR_RISK_SEQ = '#ff0000';
const COLOR_RISK_COMBO = '#ff00ff';
const COLOR_BB = 'rgba(148,163,184,0.25)';
const COLOR_BB_MID = 'rgba(148,163,184,0.13)';
const COLOR_GREEN = '#00d4a0';
const COLOR_RED = '#ff4d6d';
const COLOR_STOCH_K = '#5599ff';
const COLOR_STOCH_D = '#ff9933';
const COLOR_GRID = '#1e1e2e';
const COLOR_TEXT = '#94a3b8';

interface OHLCBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  td_setup_buy: number;
  td_setup_sell: number;
  seq_cd_buy: number;
  seq_cd_sell: number;
  combo_cd_buy: number;
  combo_cd_sell: number;
  td_perfected: number;
  rsi14: number;
  stoch_k: number;
  stoch_d: number;
  macd_line: number;
  macd_signal: number;
  macd_hist: number;
  bb_upper: number;
  bb_mid: number;
  bb_lower: number;
  tdst_buy_level: number;
  tdst_sell_level: number;
  risk_seq_buy_level: number;
  risk_seq_sell_level: number;
  risk_combo_buy_level: number;
  risk_combo_sell_level: number;
}

interface InteractiveChartProps {
  symbol: string;
  timeframe?: 'daily' | 'weekly';
  showBB?: boolean;
  height?: number | string;
  fullscreen?: boolean;
}

export default function InteractiveChart({
  symbol,
  timeframe = 'daily',
  showBB = false,
  height = 500,
  fullscreen = false,
}: InteractiveChartProps) {
  const [data, setData] = useState<OHLCBar[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch OHLC data
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    fetchOHLC(symbol, timeframe).then(result => {
      if (cancelled) return;
      if (!result || result.length === 0) {
        setError(true);
      } else {
        // Trim to last N bars for readability
        const maxBars = timeframe === 'daily' ? 120 : 80;
        setData(result.slice(-maxBars) as OHLCBar[]);
      }
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, [symbol, timeframe]);

  // Build Plotly traces and layout
  const { traces, layout, config } = useMemo(() => {
    if (!data || data.length === 0) {
      return { traces: [], layout: {}, config: {} };
    }
    return buildPlotlyChart(data, symbol, timeframe, showBB, fullscreen, height);
  }, [data, symbol, timeframe, showBB, fullscreen, height]);

  if (loading) {
    return (
      <div className="flex items-center justify-center w-full" style={{ height }}>
        <div className="flex items-center gap-2 text-signal-text-muted text-xs">
          <div className="w-4 h-4 border-2 border-signal-green/30 border-t-signal-green rounded-full animate-spin" />
          Loading chart...
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center w-full" style={{ height }}>
        <div className="text-signal-text-muted text-xs">No chart data for {symbol}</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full" style={{ height }} data-testid={`chart-${symbol}`}>
      <Plot
        data={traces as any}
        layout={layout as any}
        config={config as any}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}


/* ────────────────────────────────────────────────────────
 *  Build all Plotly traces + layout from OHLC data
 * ──────────────────────────────────────────────────────── */
function buildPlotlyChart(
  bars: OHLCBar[],
  ticker: string,
  timeframe: string,
  showBB: boolean,
  fullscreen: boolean,
  chartHeight: number | string,
) {
  const dates = bars.map(b => b.date);
  const opens = bars.map(b => b.open);
  const highs = bars.map(b => b.high);
  const lows = bars.map(b => b.low);
  const closes = bars.map(b => b.close);

  const priceMin = Math.min(...lows);
  const priceMax = Math.max(...highs);
  const priceSpan = priceMax - priceMin || 1;
  const offsetSmall = 0.012 * priceSpan;
  const offsetLarge = 0.02 * priceSpan;

  const traces: any[] = [];
  const annotations: any[] = [];
  const shapes: any[] = [];

  // ── OHLC trace ──
  traces.push({
    type: 'ohlc',
    x: dates,
    open: opens,
    high: highs,
    low: lows,
    close: closes,
    increasing: { line: { color: COLOR_BAR_UP, width: 1 } },
    decreasing: { line: { color: COLOR_BAR_DOWN, width: 1 } },
    name: 'OHLC',
    showlegend: false,
    xaxis: 'x',
    yaxis: 'y',
  });

  // ── Bollinger Bands (optional) ──
  if (showBB) {
    const bbUpper = bars.map(b => b.bb_upper || null);
    const bbMid = bars.map(b => b.bb_mid || null);
    const bbLower = bars.map(b => b.bb_lower || null);

    traces.push({
      type: 'scatter', mode: 'lines', x: dates, y: bbUpper,
      line: { color: COLOR_BB, width: 1 },
      name: 'BB Upper', showlegend: false, xaxis: 'x', yaxis: 'y',
      hoverinfo: 'skip',
    });
    traces.push({
      type: 'scatter', mode: 'lines', x: dates, y: bbMid,
      line: { color: COLOR_BB_MID, width: 0.8, dash: 'dash' },
      name: 'BB Mid', showlegend: false, xaxis: 'x', yaxis: 'y',
      hoverinfo: 'skip',
    });
    traces.push({
      type: 'scatter', mode: 'lines', x: dates, y: bbLower,
      line: { color: COLOR_BB, width: 1 },
      fill: 'tonexty', fillcolor: 'rgba(148,163,184,0.03)',
      name: 'BB Lower', showlegend: false, xaxis: 'x', yaxis: 'y',
      hoverinfo: 'skip',
    });
  }

  // ── TDST level lines ──
  addLevelShapes(shapes, bars, dates, 'tdst_buy_level', COLOR_TDST_BUY, 'dash');
  addLevelShapes(shapes, bars, dates, 'tdst_sell_level', COLOR_TDST_SELL, 'dash');

  // ── Risk level lines ──
  addLevelShapes(shapes, bars, dates, 'risk_seq_buy_level', COLOR_RISK_SEQ, 'dot');
  addLevelShapes(shapes, bars, dates, 'risk_seq_sell_level', COLOR_RISK_SEQ, 'dot');
  addLevelShapes(shapes, bars, dates, 'risk_combo_buy_level', COLOR_RISK_COMBO, 'dot');
  addLevelShapes(shapes, bars, dates, 'risk_combo_sell_level', COLOR_RISK_COMBO, 'dot');

  // ── DeMark annotations ──
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];

    // Setup labels (green, closest to bars)
    if (bar.td_setup_buy > 0) {
      const count = bar.td_setup_buy;
      const yPos = bar.low - offsetLarge;
      annotations.push(demarkAnnotation(dates[i], yPos, String(count), COLOR_SETUP, count === 9, 'top', 'y'));
      if (count === 9 && bar.td_perfected) {
        annotations.push(demarkAnnotation(dates[i], yPos - 2 * offsetSmall, '●', COLOR_SETUP, false, 'top', 'y', 10));
      }
    } else if (bar.td_setup_sell > 0) {
      const count = bar.td_setup_sell;
      const yPos = bar.high + offsetLarge;
      annotations.push(demarkAnnotation(dates[i], yPos, String(count), COLOR_SETUP, count === 9, 'bottom', 'y'));
      if (count === 9 && bar.td_perfected) {
        annotations.push(demarkAnnotation(dates[i], yPos + 2 * offsetSmall, '●', COLOR_SETUP, false, 'bottom', 'y', 10));
      }
    }

    // Sequential Countdown labels (red, further from bars)
    if (bar.seq_cd_buy > 0) {
      const count = bar.seq_cd_buy;
      const yPos = bar.low - offsetLarge - 2.5 * offsetSmall;
      annotations.push(demarkAnnotation(dates[i], yPos, String(count), COLOR_SEQ_CD, count === 13, 'top', 'y'));
    } else if (bar.seq_cd_sell > 0) {
      const count = bar.seq_cd_sell;
      const yPos = bar.high + offsetLarge + 2.5 * offsetSmall;
      annotations.push(demarkAnnotation(dates[i], yPos, String(count), COLOR_SEQ_CD, count === 13, 'bottom', 'y'));
    }

    // Combo Countdown labels (magenta, furthest from bars)
    if (bar.combo_cd_buy > 0) {
      const count = bar.combo_cd_buy;
      const yPos = bar.low - offsetLarge - 5.0 * offsetSmall;
      annotations.push(demarkAnnotation(dates[i], yPos, String(count), COLOR_COMBO_CD, count === 13, 'top', 'y'));
    } else if (bar.combo_cd_sell > 0) {
      const count = bar.combo_cd_sell;
      const yPos = bar.high + offsetLarge + 5.0 * offsetSmall;
      annotations.push(demarkAnnotation(dates[i], yPos, String(count), COLOR_COMBO_CD, count === 13, 'bottom', 'y'));
    }
  }

  // ── Slow Stochastic subplot ──
  const stochK = bars.map(b => b.stoch_k ?? 50);
  const stochD = bars.map(b => b.stoch_d ?? 50);

  traces.push({
    type: 'scatter', mode: 'lines', x: dates, y: stochK,
    line: { color: COLOR_STOCH_K, width: 1.2 },
    name: '%K', xaxis: 'x', yaxis: 'y2',
    hovertemplate: '%K: %{y:.1f}<extra></extra>',
  });
  traces.push({
    type: 'scatter', mode: 'lines', x: dates, y: stochD,
    line: { color: COLOR_STOCH_D, width: 1, dash: 'dash' },
    name: '%D', xaxis: 'x', yaxis: 'y2',
    hovertemplate: '%D: %{y:.1f}<extra></extra>',
  });

  // Stochastic overbought/oversold lines
  shapes.push(
    {
      type: 'line', x0: dates[0], x1: dates[dates.length - 1], y0: 80, y1: 80,
      line: { color: 'rgba(255,77,109,0.25)', width: 0.8, dash: 'dot' },
      xref: 'x', yref: 'y2',
    },
    {
      type: 'line', x0: dates[0], x1: dates[dates.length - 1], y0: 20, y1: 20,
      line: { color: 'rgba(0,212,160,0.25)', width: 0.8, dash: 'dot' },
      xref: 'x', yref: 'y2',
    }
  );

  // ── Last price line ──
  const lastClose = closes[closes.length - 1];
  const lastColor = closes[closes.length - 1] >= opens[opens.length - 1] ? COLOR_GREEN : COLOR_RED;
  shapes.push({
    type: 'line', x0: dates[0], x1: dates[dates.length - 1], y0: lastClose, y1: lastClose,
    line: { color: lastColor, width: 0.6, dash: 'dash' },
    xref: 'x', yref: 'y',
    opacity: 0.5,
  });

  // Last price annotation
  annotations.push({
    x: dates[dates.length - 1],
    y: lastClose,
    xref: 'x',
    yref: 'y',
    text: `<b>${lastClose.toFixed(2)}</b>`,
    showarrow: false,
    font: { color: lastColor, size: 10 },
    bgcolor: COLOR_SURFACE,
    bordercolor: lastColor,
    borderwidth: 1,
    borderpad: 3,
    xanchor: 'left',
    xshift: 10,
  });

  // ── Layout ──
  const yPad = priceSpan * 0.15;
  const layout: any = {
    title: {
      text: `<b>${ticker}</b>  <span style="font-size:12px;color:${COLOR_TEXT}">${timeframe.charAt(0).toUpperCase() + timeframe.slice(1)}</span>`,
      font: { color: '#e2e8f0', size: 14 },
      x: 0.01,
      xanchor: 'left',
    },
    plot_bgcolor: COLOR_BG,
    paper_bgcolor: COLOR_BG,
    font: { color: COLOR_TEXT, family: 'Inter, system-ui, sans-serif', size: 11 },
    margin: { l: 10, r: 60, t: 40, b: 30 },
    hovermode: 'x unified',
    dragmode: 'zoom',
    showlegend: false,

    // Main OHLC axis
    xaxis: {
      type: 'date',
      rangeslider: { visible: false },
      gridcolor: COLOR_GRID,
      gridwidth: 0.5,
      showgrid: false,
      linecolor: COLOR_GRID,
      tickfont: { size: 9, color: COLOR_TEXT },
      domain: [0, 1],
      showspikes: true,
      spikecolor: '#ffffff22',
      spikethickness: 0.5,
      spikemode: 'across',
      spikedash: 'dot',
    },
    yaxis: {
      side: 'right',
      range: [priceMin - yPad, priceMax + yPad],
      gridcolor: COLOR_GRID,
      gridwidth: 0.5,
      showgrid: true,
      linecolor: COLOR_GRID,
      tickfont: { size: 9, color: COLOR_TEXT },
      domain: [0.22, 1],
      fixedrange: false,
    },
    // Stochastic sub-axis
    yaxis2: {
      side: 'right',
      range: [0, 100],
      gridcolor: COLOR_GRID,
      gridwidth: 0.5,
      showgrid: true,
      linecolor: COLOR_GRID,
      tickfont: { size: 8, color: COLOR_TEXT },
      tickvals: [20, 50, 80],
      domain: [0, 0.18],
      fixedrange: true,
      title: { text: '%K/%D', font: { size: 9, color: COLOR_TEXT }, standoff: 5 },
    },

    annotations,
    shapes,
  };

  // Legend for DeMark (compact custom text in top-left)
  const legendAnnotations = [
    { text: '●', color: COLOR_SETUP, label: 'Setup' },
    { text: '●', color: COLOR_SEQ_CD, label: 'Seq CD' },
    { text: '●', color: COLOR_COMBO_CD, label: 'Combo CD' },
    { text: '---', color: COLOR_TDST_BUY, label: 'TDST' },
    { text: '···', color: COLOR_RISK_SEQ, label: 'Risk Lvl' },
  ];
  const legendText = legendAnnotations
    .map(l => `<span style="color:${l.color}">${l.text}</span> <span style="color:${COLOR_TEXT}">${l.label}</span>`)
    .join('&nbsp;&nbsp;');

  layout.annotations.push({
    x: 0.01, y: 0.98,
    xref: 'paper', yref: 'paper',
    text: legendText,
    showarrow: false,
    font: { size: 9 },
    align: 'left',
    bgcolor: 'rgba(18,18,26,0.85)',
    bordercolor: COLOR_GRID,
    borderwidth: 1,
    borderpad: 4,
  });

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
    modeBarStyle: { bgcolor: 'transparent', color: COLOR_TEXT, activecolor: '#e2e8f0' },
    responsive: true,
    scrollZoom: true,
  };

  return { traces, layout, config };
}


/* ────────────────────────────────────────────────────────
 *  Helper: create a DeMark text annotation
 * ──────────────────────────────────────────────────────── */
function demarkAnnotation(
  x: string, y: number, text: string, color: string,
  isBold: boolean, yanchor: string, yref: string, fontSize?: number,
) {
  const defaultSize = isBold ? 10 : 8;
  return {
    x, y,
    xref: 'x',
    yref,
    text: isBold ? `<b>${text}</b>` : text,
    showarrow: false,
    font: { color, size: fontSize ?? defaultSize },
    xanchor: 'center',
    yanchor,
  };
}


/* ────────────────────────────────────────────────────────
 *  Helper: add horizontal level line shapes (TDST, Risk)
 * ──────────────────────────────────────────────────────── */
function addLevelShapes(
  shapes: any[], bars: OHLCBar[], dates: string[],
  field: keyof OHLCBar, color: string, dash: string,
) {
  let i = 0;
  while (i < bars.length) {
    const val = bars[i][field] as number;
    if (!val || val === 0) { i++; continue; }
    const segStart = i;
    while (i < bars.length && (bars[i][field] as number) === val) { i++; }
    const segEnd = i - 1;
    shapes.push({
      type: 'line',
      x0: dates[segStart], x1: dates[segEnd],
      y0: val, y1: val,
      line: { color, width: 1, dash },
      xref: 'x', yref: 'y',
      opacity: 0.7,
    });
  }
}
