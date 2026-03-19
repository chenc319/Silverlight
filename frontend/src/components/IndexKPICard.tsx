import { cn, formatPrice, formatPct } from '@/lib/utils';
import SignalBadge from './SignalBadge';
import type { TickerSignal } from '@/data/tickers';

interface IndexKPICardProps {
  data: TickerSignal;
  onClick: (symbol: string) => void;
}

export default function IndexKPICard({ data, onClick }: IndexKPICardProps) {
  const isBuy = data.dailySignal === 'BUY';
  const isSell = data.dailySignal === 'SELL';

  // Show the most notable DeMark state in compact form
  const demarkCompact = data.td9Daily || data.td13Seq || data.td13Combo ||
    data.setupLabel || data.seqCdLabel || data.comboCdLabel || null;

  return (
    <button
      onClick={() => onClick(data.symbol)}
      data-testid={`kpi-card-${data.symbol}`}
      className={cn(
        'flex flex-col gap-2 p-3 rounded-lg border transition-all text-left min-w-0',
        'hover:bg-white/[0.02]',
        'bg-signal-surface border-signal-border',
        isBuy && 'glow-buy',
        isSell && 'muted-sell'
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-bold text-signal-text tracking-tight">{data.symbol}</span>
        <span className={cn(
          'text-xs font-semibold tabular-nums',
          data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
        )}>
          {formatPct(data.pctChg1d)}
        </span>
      </div>

      <div className="text-lg font-bold tabular-nums text-signal-text">
        {formatPrice(data.lastClose)}
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        <SignalBadge signal={data.dailySignal} />
        {demarkCompact && (
          <span className={cn(
            'text-[9px] font-bold tabular-nums truncate',
            data.td9Daily?.includes('BUY') || data.td13Seq?.includes('BUY') || data.td13Combo?.includes('BUY') ||
            data.tdSetupSide === 'buy' || data.seqCdSide === 'buy' || data.comboCdSide === 'buy'
              ? 'text-signal-green'
              : data.comboCdLabel ? 'text-[#ff00ff]' : 'text-signal-red'
          )}>
            {data.td9Daily || data.td13Seq || data.td13Combo || data.setupLabel || data.seqCdLabel || data.comboCdLabel}
          </span>
        )}
      </div>
    </button>
  );
}
