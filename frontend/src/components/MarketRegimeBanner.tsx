import { cn, formatPrice, formatPct } from '@/lib/utils';
import { useSignalData } from '@/data/DataProvider';
import { RefreshCw, AlertTriangle, Clock, Wifi, WifiOff } from 'lucide-react';

export default function MarketRegimeBanner() {
  const { signals, regime, lastUpdated, isLive, isRefreshing, refreshProgress, doRefresh } = useSignalData();
  const spy = signals['SPY'];
  const { regime: regimeStr, justification, volatilityElevated } = regime;

  const regimeColor = {
    'RISK ON': 'bg-signal-green/15 text-signal-green border-signal-green/30',
    'NEUTRAL': 'bg-signal-amber/15 text-signal-amber border-signal-amber/30',
    'RISK OFF': 'bg-signal-red/15 text-signal-red border-signal-red/30',
  }[regimeStr];

  const regimeDot = {
    'RISK ON': 'bg-signal-green',
    'NEUTRAL': 'bg-signal-amber',
    'RISK OFF': 'bg-signal-red',
  }[regimeStr];

  const updated = new Date(lastUpdated);
  const timeStr = updated.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });

  const completedCount = refreshProgress.length;
  const totalTickers = 30;

  return (
    <div className="bg-signal-surface border-b border-signal-border px-4 py-3">
      <div className="max-w-[1600px] mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        {/* Left */}
        <div className="flex flex-wrap items-center gap-3">
          <div
            data-testid="regime-pill"
            className={cn(
              'inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold uppercase tracking-wider',
              regimeColor
            )}
          >
            <span className={cn('w-2 h-2 rounded-full animate-pulse-subtle', regimeDot)} />
            {regimeStr}
          </div>

          {spy && (
            <div className="flex items-center gap-4 text-xs tabular-nums">
              <div className="flex items-center gap-1.5">
                <span className="text-signal-text-muted">SPY</span>
                <span className="font-semibold text-signal-text">{formatPrice(spy.lastClose)}</span>
                <span className={cn(
                  'font-medium',
                  spy.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
                )}>
                  {formatPct(spy.pctChg1d)}
                </span>
              </div>

              {volatilityElevated && (
                <div className="flex items-center gap-1 text-signal-amber">
                  <AlertTriangle size={12} />
                  <span className="font-medium">VOL ELEVATED</span>
                </div>
              )}
            </div>
          )}

          <span className="hidden lg:inline text-[11px] text-signal-text-muted max-w-[360px] truncate">
            {justification}
          </span>
        </div>

        {/* Right */}
        <div className="flex items-center gap-3">
          {/* Live indicator */}
          <div className={cn(
            'flex items-center gap-1 text-[11px]',
            isLive ? 'text-signal-green' : 'text-signal-text-muted'
          )}>
            {isLive ? <Wifi size={11} /> : <WifiOff size={11} />}
            <span>{isLive ? 'Live' : 'Cached'}</span>
          </div>

          <div className="flex items-center gap-1.5 text-[11px] text-signal-text-muted">
            <Clock size={11} />
            <span>{timeStr}</span>
          </div>

          <button
            onClick={doRefresh}
            data-testid="refresh-button"
            disabled={isRefreshing}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium',
              'bg-signal-green/10 text-signal-green border border-signal-green/25',
              'hover:bg-signal-green/20 transition-all',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            <RefreshCw size={12} className={isRefreshing ? 'animate-spin' : ''} />
            {isRefreshing ? `${completedCount}/${totalTickers}` : 'Refresh'}
          </button>
        </div>
      </div>
    </div>
  );
}
