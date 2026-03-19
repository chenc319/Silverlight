import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import { fetchOHLC } from '@/lib/api';
import { generateOHLC } from '@/data/seededSignals';

interface OHLCBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  td_setup_buy?: number;
  td_setup_sell?: number;
  td_countdown_buy?: number;
  td_countdown_sell?: number;
  bb_upper?: number;
  bb_mid?: number;
  bb_lower?: number;
}

interface LightweightChartProps {
  symbol: string;
  height?: number;
  showBB?: boolean;
  showRSI?: boolean;
  showMACD?: boolean;
  timeframe?: 'daily' | 'weekly';
}

export default function LightweightChart({
  symbol,
  height = 400,
  showBB = true,
  timeframe = 'daily',
}: LightweightChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<ReturnType<typeof createChart> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!chartRef.current) return;

    let cancelled = false;

    // Clean up previous chart
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }

    const container = chartRef.current;

    async function loadAndRender() {
      setLoading(true);

      // Try API first, fall back to seeded
      let ohlcData: OHLCBar[];
      const apiData = await fetchOHLC(symbol, timeframe);

      if (apiData && apiData.length > 0) {
        ohlcData = apiData as OHLCBar[];
      } else {
        // Fallback to generated seeded data
        const seeded = generateOHLC(symbol, timeframe === 'weekly' ? 104 : 252);
        ohlcData = seeded.map(d => ({
          date: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
          td_setup_buy: d.tdSetup && d.tdSetup < 0 ? Math.abs(d.tdSetup) : 0,
          td_setup_sell: d.tdSetup && d.tdSetup > 0 ? d.tdSetup : 0,
          td_countdown_buy: d.tdCountdown === 13 ? 13 : 0,
          td_countdown_sell: 0,
        }));
      }

      if (cancelled || !container) return;
      setLoading(false);

      const chart = createChart(container, {
        width: container.clientWidth,
        height: height,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: '#94a3b8',
          fontSize: 11,
          fontFamily: 'Inter, system-ui, sans-serif',
        },
        grid: {
          vertLines: { color: 'rgba(30, 30, 46, 0.5)' },
          horzLines: { color: 'rgba(30, 30, 46, 0.5)' },
        },
        crosshair: {
          vertLine: { color: 'rgba(148, 163, 184, 0.3)', labelBackgroundColor: '#1e1e2e' },
          horzLine: { color: 'rgba(148, 163, 184, 0.3)', labelBackgroundColor: '#1e1e2e' },
        },
        rightPriceScale: {
          borderColor: '#1e1e2e',
        },
        timeScale: {
          borderColor: '#1e1e2e',
          timeVisible: false,
        },
      });

      chartInstanceRef.current = chart;

      // Candlestick series
      const candleSeries = chart.addCandlestickSeries({
        upColor: '#00d4a0',
        downColor: '#ff4d6d',
        borderUpColor: '#00d4a0',
        borderDownColor: '#ff4d6d',
        wickUpColor: '#00d4a088',
        wickDownColor: '#ff4d6d88',
      });

      const candleData = ohlcData.map(d => ({
        time: d.date,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));
      candleSeries.setData(candleData as any);

      // DeMark markers from API data
      const markers: any[] = [];
      ohlcData.forEach(d => {
        const setupBuy = d.td_setup_buy || 0;
        const setupSell = d.td_setup_sell || 0;

        // Buy setup (count 1-9)
        if (setupBuy === 9) {
          markers.push({
            time: d.date,
            position: 'belowBar',
            color: '#00d4a0',
            shape: 'arrowUp',
            text: '9',
          });
        } else if (setupBuy >= 1 && setupBuy <= 8) {
          markers.push({
            time: d.date,
            position: 'belowBar',
            color: 'rgba(0,212,160,0.5)',
            shape: 'circle',
            text: String(setupBuy),
          });
        }

        // Sell setup (count 1-9)
        if (setupSell === 9) {
          markers.push({
            time: d.date,
            position: 'aboveBar',
            color: '#ff4d6d',
            shape: 'arrowDown',
            text: '9',
          });
        } else if (setupSell >= 1 && setupSell <= 8) {
          markers.push({
            time: d.date,
            position: 'aboveBar',
            color: 'rgba(255,77,109,0.5)',
            shape: 'circle',
            text: String(setupSell),
          });
        }

        // Countdown 13 signals
        const countdownBuy = d.td_countdown_buy || 0;
        const countdownSell = d.td_countdown_sell || 0;
        if (countdownBuy === 13) {
          markers.push({
            time: d.date,
            position: 'belowBar',
            color: '#3b82f6',
            shape: 'arrowUp',
            text: '13',
          });
        }
        if (countdownSell === 13) {
          markers.push({
            time: d.date,
            position: 'aboveBar',
            color: '#3b82f6',
            shape: 'arrowDown',
            text: '13',
          });
        }
      });

      if (markers.length > 0) {
        markers.sort((a: any, b: any) => a.time.localeCompare(b.time));
        candleSeries.setMarkers(markers);
      }

      // Bollinger Bands from API data
      if (showBB) {
        const hasBBData = ohlcData.some(d => d.bb_upper && d.bb_upper > 0);

        let bbUpper: any[] = [];
        let bbMid: any[] = [];
        let bbLower: any[] = [];

        if (hasBBData) {
          // Use pre-computed BB from API
          ohlcData.forEach(d => {
            if (d.bb_upper && d.bb_upper > 0) {
              bbUpper.push({ time: d.date, value: d.bb_upper });
              bbMid.push({ time: d.date, value: d.bb_mid });
              bbLower.push({ time: d.date, value: d.bb_lower });
            }
          });
        } else {
          // Compute client-side as fallback
          const period = 20;
          const closes = ohlcData.map(d => d.close);
          for (let i = 0; i < ohlcData.length; i++) {
            if (i < period - 1) continue;
            const slice = closes.slice(i - period + 1, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / period;
            const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period);
            bbUpper.push({ time: ohlcData[i].date, value: mean + 2 * std });
            bbMid.push({ time: ohlcData[i].date, value: mean });
            bbLower.push({ time: ohlcData[i].date, value: mean - 2 * std });
          }
        }

        if (bbUpper.length > 0) {
          const upperSeries = chart.addLineSeries({
            color: 'rgba(148, 163, 184, 0.25)',
            lineWidth: 1,
            crosshairMarkerVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
          });
          upperSeries.setData(bbUpper);

          const midSeries = chart.addLineSeries({
            color: 'rgba(148, 163, 184, 0.15)',
            lineWidth: 1,
            lineStyle: 2,
            crosshairMarkerVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
          });
          midSeries.setData(bbMid);

          const lowerSeries = chart.addLineSeries({
            color: 'rgba(148, 163, 184, 0.25)',
            lineWidth: 1,
            crosshairMarkerVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
          });
          lowerSeries.setData(bbLower);
        }
      }

      chart.timeScale().fitContent();

      // Resize handler
      const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
          if (chartInstanceRef.current) {
            chartInstanceRef.current.applyOptions({ width: entry.contentRect.width });
          }
        }
      });
      resizeObserver.observe(container);

      // Return cleanup for resize only (chart cleanup below)
      return () => resizeObserver.disconnect();
    }

    let resizeCleanup: (() => void) | undefined;
    loadAndRender().then(cleanup => {
      resizeCleanup = cleanup;
    });

    return () => {
      cancelled = true;
      if (resizeCleanup) resizeCleanup();
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
        chartInstanceRef.current = null;
      }
    };
  }, [symbol, height, showBB, timeframe]);

  return (
    <div className="relative w-full" style={{ height }}>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex items-center gap-2 text-signal-text-muted text-xs">
            <div className="w-4 h-4 border-2 border-signal-green/30 border-t-signal-green rounded-full animate-spin" />
            Loading chart...
          </div>
        </div>
      )}
      <div
        ref={chartRef}
        data-testid={`chart-${symbol}`}
        className="w-full h-full"
      />
    </div>
  );
}
