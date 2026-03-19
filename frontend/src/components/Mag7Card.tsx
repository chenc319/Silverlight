import { cn, formatPrice, formatPct, hasCountdown13Today, getCountdown13Label } from '@/lib/utils';
import SignalBadge from './SignalBadge';
import type { TickerSignal } from '@/data/tickers';

interface Mag7CardProps {
  data: TickerSignal;
  onClick: (symbol: string) => void;
}

function DeMarkSignalChip({ signal, cd13 }: { signal: string | null; cd13: false | 'buy' | 'sell' }) {
  if (!signal) return null;
  const isBuy = signal === 'BUY';
  const isSell = signal === 'SELL';
  const is13Plus = signal === '13+';
  return (
    <span className={cn(
      'inline-block px-2 py-0.5 rounded font-bold uppercase',
      cd13 ? 'text-sm' : 'text-xs',
      isBuy && 'bg-signal-green/20 text-signal-green',
      isSell && 'bg-signal-red/20 text-signal-red',
      is13Plus && 'bg-amber-500/20 text-amber-400',
    )}>
      {signal}
    </span>
  );
}

export default function Mag7Card({ data, onClick }: Mag7CardProps) {
  const isBuy = data.dailySignal === 'BUY';
  const isSell = data.dailySignal === 'SELL';
  const cd13 = hasCountdown13Today(data);

  return (
    <button
      onClick={() => onClick(data.symbol)}
      data-testid={`mag7-card-${data.symbol}`}
      className={cn(
        'flex flex-col gap-2.5 p-4 rounded-lg border transition-all text-left',
        'hover:bg-white/[0.02] cursor-pointer',
        'bg-signal-surface border-signal-border',
        isBuy && !cd13 && 'glow-buy',
        isSell && !cd13 && 'muted-sell',
        cd13 && cd13.direction === 'buy' && 'countdown-13-today',
        cd13 && cd13.direction === 'sell' && 'countdown-13-today-sell',
      )}
    >
      {/* Header row */}
      <div className="flex items-start justify-between">
        <div>
          <div className={cn(
            'font-bold text-signal-text tracking-tight',
            cd13 ? 'text-xl' : 'text-base'
          )} data-testid={`mag7-ticker-${data.symbol}`}>{data.symbol}</div>
          <div className="text-xl font-bold tabular-nums text-signal-text mt-0.5">
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

      {/* Countdown 13 hero badge */}
      {cd13 && (
        <div className={cn(
          'flex items-center justify-center py-2 rounded-md',
          'font-extrabold text-2xl tracking-wide',
          cd13.direction === 'buy'
            ? 'bg-signal-green/20 text-signal-green'
            : 'bg-signal-red/20 text-signal-red'
        )}>
          {getCountdown13Label(cd13)}
        </div>
      )}

      {/* Signals row */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <SignalBadge signal={data.dailySignal} />
        <SignalBadge signal={data.weeklySignal} />
        <DeMarkSignalChip signal={data.demarkSignal} cd13={cd13} />
      </div>

      {/* DeMark state + indicators row */}
      <div className="space-y-1.5">
        {/* DeMark active state */}
        <div className="flex items-center justify-between text-xs tabular-nums flex-wrap gap-x-2">
          {data.setupLabel && (
            <span className={cn(
              'font-bold',
              data.setupLabelColor === 'green' ? 'text-signal-green' : 'text-signal-red'
            )}>
              Setup {data.setupLabel}
            </span>
          )}
          {data.seqCdLabel && (
            <span className={cn(
              'font-bold',
              data.seqCdLabelColor === 'green' ? 'text-signal-green' : 'text-signal-red'
            )}>
              Seq {data.seqCdLabel}
            </span>
          )}
          {data.comboCdLabel && (
            <span className={cn(
              'font-bold',
              data.comboCdLabelColor === 'green' ? 'text-signal-green' :
              data.comboCdLabelColor === 'red' ? 'text-signal-red' : 'text-[#ff00ff]'
            )}>
              Combo {data.comboCdLabel}
            </span>
          )}
        </div>

        {/* Traditional indicators */}
        <div className="flex items-center justify-between text-xs tabular-nums">
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
      </div>
    </button>
  );
}
