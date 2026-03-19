import { cn, formatPrice, formatPct } from '@/lib/utils';
import SignalBadge from './SignalBadge';
import DeMarkBadge from './DeMarkBadge';
import type { TickerSignal } from '@/data/tickers';

interface Mag7CardProps {
  data: TickerSignal;
  onClick: (symbol: string) => void;
}

export default function Mag7Card({ data, onClick }: Mag7CardProps) {
  const isBuy = data.dailySignal === 'BUY';
  const isSell = data.dailySignal === 'SELL';
  const demarkLabel = data.td9Daily || data.td13Daily || null;

  return (
    <button
      onClick={() => onClick(data.symbol)}
      data-testid={`mag7-card-${data.symbol}`}
      className={cn(
        'flex flex-col gap-2.5 p-4 rounded-lg border transition-all text-left',
        'hover:bg-white/[0.02] cursor-pointer',
        'bg-signal-surface border-signal-border',
        isBuy && 'glow-buy',
        isSell && 'muted-sell'
      )}
    >
      {/* Header row */}
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-bold text-signal-text tracking-tight">{data.symbol}</div>
          <div className="text-lg font-bold tabular-nums text-signal-text mt-0.5">
            {formatPrice(data.lastClose)}
          </div>
        </div>
        <span className={cn(
          'text-sm font-semibold tabular-nums',
          data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
        )}>
          {formatPct(data.pctChg1d)}
        </span>
      </div>

      {/* Signals row */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <SignalBadge signal={data.dailySignal} />
        <SignalBadge signal={data.weeklySignal} />
        {demarkLabel && <DeMarkBadge label={demarkLabel} />}
      </div>

      {/* Indicators row */}
      <div className="flex items-center justify-between text-[11px] tabular-nums">
        <div className="flex items-center gap-0.5">
          <span className="text-signal-text-muted">RSI</span>
          <span className={cn(
            'font-medium',
            data.rsi14 < 30 ? 'text-signal-green' :
            data.rsi14 > 70 ? 'text-signal-red' : 'text-signal-text-secondary'
          )}>
            {data.rsi14.toFixed(1)}
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          <span className="text-signal-text-muted">Rel/SPY</span>
          <span className={cn(
            'font-medium',
            data.relSpy20d > 0 ? 'text-signal-green' : 'text-signal-red'
          )}>
            {data.relSpy20d > 0 ? '+' : ''}{data.relSpy20d.toFixed(1)}%
          </span>
        </div>
      </div>
    </button>
  );
}
