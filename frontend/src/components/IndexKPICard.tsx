import { cn, formatPrice, formatPct, hasCountdown13Today } from '@/lib/utils';
import SignalBadge from './SignalBadge';
import type { TickerSignal } from '@/data/tickers';

interface IndexKPICardProps {
  data: TickerSignal;
  onClick: (symbol: string) => void;
}

export default function IndexKPICard({ data, onClick }: IndexKPICardProps) {
  const isBuy = data.dailySignal === 'BUY';
  const isSell = data.dailySignal === 'SELL';
  const cd13 = hasCountdown13Today(data);

  // Show the most notable DeMark state in compact form
  const demarkLabel = data.setupLabel || data.seqCdLabel || data.comboCdLabel || null;
  const demarkColor = data.setupLabel ? data.setupLabelColor :
    data.seqCdLabel ? data.seqCdLabelColor :
    data.comboCdLabel ? data.comboCdLabelColor : null;

  return (
    <button
      onClick={() => onClick(data.symbol)}
      data-testid={`kpi-card-${data.symbol}`}
      className={cn(
        'flex flex-col gap-2 p-3 rounded-lg border transition-all text-left min-w-0',
        'hover:bg-white/[0.02]',
        'bg-signal-surface border-signal-border',
        isBuy && !cd13 && 'glow-buy',
        isSell && !cd13 && 'muted-sell',
        cd13 === 'buy' && 'countdown-13-today',
        cd13 === 'sell' && 'countdown-13-today-sell',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className={cn(
          'font-bold text-signal-text tracking-tight',
          cd13 ? 'text-lg' : 'text-base'
        )}>{data.symbol}</span>
        <span className={cn(
          'text-sm font-semibold tabular-nums',
          data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
        )}>
          {formatPct(data.pctChg1d)}
        </span>
      </div>

      <div className="text-xl font-bold tabular-nums text-signal-text">
        {formatPrice(data.lastClose)}
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        <SignalBadge signal={data.dailySignal} />
        {data.demarkSignal && (
          <span className={cn(
            'font-bold px-1.5 py-0.5 rounded',
            cd13 ? 'text-sm' : 'text-[11px]',
            data.demarkSignal === 'BUY' && 'bg-signal-green/15 text-signal-green',
            data.demarkSignal === 'SELL' && 'bg-signal-red/15 text-signal-red',
            data.demarkSignal === '13+' && 'bg-amber-500/15 text-amber-400',
          )}>
            {data.demarkSignal}
          </span>
        )}
        {demarkLabel && (
          <span className={cn(
            'text-[11px] font-bold tabular-nums truncate',
            demarkColor === 'green' ? 'text-signal-green' :
            demarkColor === 'red' ? 'text-signal-red' : 'text-signal-text-muted'
          )}>
            {demarkLabel}
          </span>
        )}
      </div>

      {/* Countdown 13 hero badge */}
      {cd13 && (
        <div className={cn(
          'flex items-center justify-center py-1.5 -mx-1 rounded-md mt-0.5',
          'font-extrabold text-xl tracking-wide',
          cd13 === 'buy'
            ? 'bg-signal-green/20 text-signal-green'
            : 'bg-signal-red/20 text-signal-red'
        )}>
          CD 13
        </div>
      )}
    </button>
  );
}
