import { useEffect, useCallback } from 'react';
import { X, Maximize2 } from 'lucide-react';
import { cn, formatPrice, formatPct, formatScore } from '@/lib/utils';
import SignalBadge from './SignalBadge';
import DeMarkBadge from './DeMarkBadge';
import { useSignalData } from '@/data/DataProvider';
import LightweightChart from './LightweightChart';

interface ChartDrawerProps {
  symbol: string;
  onClose: () => void;
  onNavigateToChart: (symbol: string) => void;
}

export default function ChartDrawer({ symbol, onClose, onNavigateToChart }: ChartDrawerProps) {
  const { signals } = useSignalData();
  const data = signals[symbol];

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (!data) return null;

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/50 z-40 md:bg-black/30"
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        data-testid="chart-drawer"
        className={cn(
          'fixed z-50 bg-signal-bg border-l border-signal-border',
          'animate-slide-in-right',
          'md:top-12 md:right-0 md:bottom-0 md:w-[560px]',
          'top-0 right-0 bottom-0 left-0 md:left-auto'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-signal-border">
          <div className="flex items-center gap-3">
            <span className="text-base font-bold text-signal-text">{symbol}</span>
            <span className="text-sm font-semibold tabular-nums text-signal-text">
              {formatPrice(data.lastClose)}
            </span>
            <span className={cn(
              'text-xs font-semibold tabular-nums',
              data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
            )}>
              {formatPct(data.pctChg1d)}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onNavigateToChart(symbol)}
              data-testid="expand-chart"
              className="p-1.5 rounded hover:bg-white/5 text-signal-text-muted hover:text-signal-text transition-colors"
              title="Open full chart"
            >
              <Maximize2 size={16} />
            </button>
            <button
              onClick={onClose}
              data-testid="close-drawer"
              className="p-1.5 rounded hover:bg-white/5 text-signal-text-muted hover:text-signal-text transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Chart */}
        <div className="h-[300px] md:h-[350px] px-2 pt-2">
          <LightweightChart symbol={symbol} height={290} />
        </div>

        {/* Signal Summary */}
        <div className="px-4 py-3 border-t border-signal-border space-y-3 overflow-y-auto custom-scrollbar"
          style={{ maxHeight: 'calc(100% - 350px - 56px)' }}
        >
          {/* Scores */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-signal-surface border border-signal-border">
              <div className="text-[10px] uppercase tracking-wider text-signal-text-muted mb-1">Daily Score</div>
              <div className="flex items-center gap-2">
                <span className={cn(
                  'text-xl font-bold tabular-nums',
                  data.dailyScore >= 40 ? 'text-signal-green' :
                  data.dailyScore <= -40 ? 'text-signal-red' : 'text-signal-amber'
                )}>
                  {formatScore(data.dailyScore)}
                </span>
                <SignalBadge signal={data.dailySignal} />
              </div>
            </div>
            <div className="p-3 rounded-lg bg-signal-surface border border-signal-border">
              <div className="text-[10px] uppercase tracking-wider text-signal-text-muted mb-1">Weekly Score</div>
              <div className="flex items-center gap-2">
                <span className={cn(
                  'text-xl font-bold tabular-nums',
                  data.weeklyScore >= 40 ? 'text-signal-green' :
                  data.weeklyScore <= -40 ? 'text-signal-red' : 'text-signal-amber'
                )}>
                  {formatScore(data.weeklyScore)}
                </span>
                <SignalBadge signal={data.weeklySignal} />
              </div>
            </div>
          </div>

          {/* DeMark Signals — Setup + Sequential CD + Combo CD */}
          <div className="p-3 rounded-lg bg-signal-surface border border-signal-border">
            <div className="text-[10px] uppercase tracking-wider text-signal-text-muted mb-2">DeMark</div>
            <div className="space-y-2">
              {/* Completed signals */}
              <div className="flex flex-wrap gap-2">
                {data.td9Daily && <DeMarkBadge label={data.td9Daily} />}
                {data.td13Seq && <DeMarkBadge label={data.td13Seq} />}
                {data.td13Combo && <DeMarkBadge label={data.td13Combo} />}
              </div>
              {/* Active state labels */}
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <div>
                  <span className="text-signal-text-muted block">Setup</span>
                  <span className={cn(
                    'font-bold',
                    data.setupLabel ? (data.tdSetupSide === 'buy' ? 'text-signal-green' : 'text-signal-red') : 'text-signal-text-muted'
                  )}>
                    {data.setupLabel || '—'}
                  </span>
                </div>
                <div>
                  <span className="text-signal-text-muted block">Seq CD</span>
                  <span className={cn(
                    'font-bold',
                    data.seqCdLabel ? (data.seqCdSide === 'buy' ? 'text-signal-green' : 'text-signal-red') : 'text-signal-text-muted'
                  )}>
                    {data.seqCdLabel || '—'}
                  </span>
                </div>
                <div>
                  <span className="text-signal-text-muted block">Combo CD</span>
                  <span className={cn(
                    'font-bold',
                    data.comboCdLabel ? (data.comboCdSide === 'buy' ? 'text-[#ff00ff]' : 'text-[#ff00ff]') : 'text-signal-text-muted'
                  )}>
                    {data.comboCdLabel || '—'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Indicators Grid */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: 'RSI (14)', value: data.rsi14.toFixed(1), color: data.rsi14 < 30 ? 'text-signal-green' : data.rsi14 > 70 ? 'text-signal-red' : 'text-signal-text' },
              { label: 'Stoch %K', value: data.stochK.toFixed(1), color: data.stochK < 20 ? 'text-signal-green' : data.stochK > 80 ? 'text-signal-red' : 'text-signal-text' },
              { label: 'Stoch %D', value: data.stochD.toFixed(1), color: 'text-signal-text-secondary' },
              { label: 'MACD Hist', value: data.macdHist.toFixed(3), color: data.macdHist > 0 ? 'text-signal-green' : 'text-signal-red' },
              { label: 'BB %', value: (data.bbPct * 100).toFixed(0) + '%', color: data.bbPct < 0.05 ? 'text-signal-green' : data.bbPct > 0.95 ? 'text-signal-red' : 'text-signal-text' },
              { label: 'Rel SPY 20D', value: formatPct(data.relSpy20d), color: data.relSpy20d > 0 ? 'text-signal-green' : 'text-signal-red' },
            ].map(ind => (
              <div key={ind.label} className="p-2 rounded bg-signal-surface/50 border border-signal-border/50">
                <div className="text-[9px] uppercase tracking-wider text-signal-text-muted">{ind.label}</div>
                <div className={cn('text-sm font-bold tabular-nums mt-0.5', ind.color)}>{ind.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
