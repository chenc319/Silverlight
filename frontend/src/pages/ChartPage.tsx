import { useState, useEffect, useMemo } from 'react';
import { Search, Maximize2, X } from 'lucide-react';
import { cn, formatPrice, formatPct, formatScore, getScoreColor } from '@/lib/utils';
import SignalBadge from '@/components/SignalBadge';
import InteractiveChart from '@/components/InteractiveChart';
import { useSignalData } from '@/data/DataProvider';
import { ALL_TICKERS } from '@/data/tickers';

export default function ChartPage() {
  const { signals } = useSignalData();
  const [selectedTicker, setSelectedTicker] = useState('SPY');
  const [timeframe, setTimeframe] = useState<'daily' | 'weekly'>('daily');
  const [searchQuery, setSearchQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [showBB, setShowBB] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Listen for external ticker selection
  useEffect(() => {
    const handler = (e: Event) => {
      const ticker = (e as CustomEvent).detail;
      if (typeof ticker === 'string' && ALL_TICKERS.includes(ticker as any)) {
        setSelectedTicker(ticker);
      }
    };
    window.addEventListener('select-ticker', handler);
    return () => window.removeEventListener('select-ticker', handler);
  }, []);

  // Close fullscreen on Escape
  useEffect(() => {
    if (!isFullscreen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsFullscreen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isFullscreen]);

  const filteredTickers = useMemo(() => {
    if (!searchQuery) return ALL_TICKERS;
    return ALL_TICKERS.filter(t => t.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [searchQuery]);

  const data = signals[selectedTicker];

  return (
    <>
      <div className="max-w-[1600px] mx-auto px-4 py-4 animate-fade-in">
        {/* Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 mb-4">
          {/* Ticker selector */}
          <div className="relative">
            <div className="flex items-center gap-2 bg-signal-surface border border-signal-border rounded-lg px-3 py-2">
              <Search size={14} className="text-signal-text-muted" />
              <input
                type="text"
                value={showDropdown ? searchQuery : selectedTicker}
                onChange={(e) => { setSearchQuery(e.target.value); setShowDropdown(true); }}
                onFocus={() => { setShowDropdown(true); setSearchQuery(''); }}
                onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                data-testid="ticker-search"
                placeholder="Search ticker..."
                className="bg-transparent text-sm font-semibold text-signal-text outline-none w-28"
              />
            </div>

            {showDropdown && (
              <div className="absolute top-full mt-1 left-0 w-48 max-h-64 overflow-y-auto bg-signal-surface border border-signal-border rounded-lg shadow-xl z-30 custom-scrollbar">
                {filteredTickers.map(t => (
                  <button
                    key={t}
                    onMouseDown={() => { setSelectedTicker(t); setShowDropdown(false); setSearchQuery(''); }}
                    className={cn(
                      'w-full text-left px-3 py-2 text-xs font-medium hover:bg-white/5 transition-colors',
                      t === selectedTicker ? 'text-signal-green bg-signal-green/5' : 'text-signal-text'
                    )}
                  >
                    {t}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Timeframe toggle */}
          <div className="flex items-center bg-signal-surface border border-signal-border rounded-lg p-0.5">
            {(['daily', 'weekly'] as const).map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                data-testid={`tf-${tf}`}
                className={cn(
                  'px-3 py-1.5 rounded-md text-xs font-medium transition-all capitalize',
                  timeframe === tf
                    ? 'bg-signal-green/15 text-signal-green'
                    : 'text-signal-text-secondary hover:text-signal-text'
                )}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* BB toggle */}
          <button
            onClick={() => setShowBB(prev => !prev)}
            data-testid="bb-toggle"
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-all border',
              showBB
                ? 'bg-signal-amber/15 text-signal-amber border-signal-amber/30'
                : 'bg-signal-surface text-signal-text-muted border-signal-border hover:text-signal-text'
            )}
          >
            BB {showBB ? 'ON' : 'OFF'}
          </button>

          {/* Ticker info */}
          {data && (
            <div className="flex items-center gap-3 ml-auto">
              <span className="text-lg font-bold tabular-nums">{formatPrice(data.lastClose)}</span>
              <span className={cn(
                'text-sm font-semibold tabular-nums',
                data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
              )}>
                {formatPct(data.pctChg1d)}
              </span>
              <SignalBadge signal={data.dailySignal} size="md" />
            </div>
          )}
        </div>

        <div className="flex flex-col lg:flex-row gap-4">
          {/* Main chart */}
          <div className="flex-1 bg-signal-surface rounded-lg border border-signal-border p-2 relative group">
            {/* Fullscreen button */}
            <button
              onClick={() => setIsFullscreen(true)}
              data-testid="chart-fullscreen-btn"
              className="absolute top-2 right-2 z-20 p-1.5 rounded bg-signal-surface/80 border border-signal-border/50 text-signal-text-muted hover:text-signal-text opacity-0 group-hover:opacity-100 transition-all backdrop-blur-sm"
              title="Expand chart fullscreen"
            >
              <Maximize2 size={14} />
            </button>
            <InteractiveChart
              key={`${selectedTicker}-${timeframe}-${showBB}`}
              symbol={selectedTicker}
              height={520}
              timeframe={timeframe}
              showBB={showBB}
            />
          </div>

          {/* Signal sidebar */}
          {data && (
            <div className="lg:w-72 space-y-3">
              {/* Signal summary */}
              <div className="bg-signal-surface rounded-lg border border-signal-border p-3">
                <div className="text-[10px] uppercase tracking-wider text-signal-text-muted mb-2">Signal Summary</div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-signal-text-muted">Daily</span>
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        'text-sm font-bold tabular-nums',
                        getScoreColor(data.dailyScore)
                      )}>
                        {formatScore(data.dailyScore)}
                      </span>
                      <SignalBadge signal={data.dailySignal} />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-signal-text-muted">Weekly</span>
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        'text-sm font-bold tabular-nums',
                        getScoreColor(data.weeklyScore)
                      )}>
                        {formatScore(data.weeklyScore)}
                      </span>
                      <SignalBadge signal={data.weeklySignal} />
                    </div>
                  </div>
                </div>
              </div>

              {/* DeMark */}
              <div className="bg-signal-surface rounded-lg border border-signal-border p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] uppercase tracking-wider text-signal-text-muted">DeMark</span>
                  {data.demarkSignal && (
                    <span className={cn(
                      'inline-block px-1.5 py-0.5 rounded text-[10px] font-bold uppercase',
                      data.demarkSignal === 'BUY' && 'bg-signal-green/20 text-signal-green',
                      data.demarkSignal === 'SELL' && 'bg-signal-red/20 text-signal-red',
                      data.demarkSignal === '13+' && 'bg-amber-500/20 text-amber-400',
                    )}>
                      {data.demarkSignal}
                    </span>
                  )}
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-signal-text-muted">Setup</span>
                    <span className={cn(
                      'font-bold tabular-nums',
                      data.setupLabelColor === 'green' ? 'text-signal-green' :
                      data.setupLabelColor === 'red' ? 'text-signal-red' : 'text-signal-text-muted'
                    )}>
                      {data.setupLabel || '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-signal-text-muted">Sequential CD</span>
                    <span className={cn(
                      'font-bold tabular-nums',
                      data.seqCdLabelColor === 'green' ? 'text-signal-green' :
                      data.seqCdLabelColor === 'red' ? 'text-signal-red' : 'text-signal-text-muted'
                    )}>
                      {data.seqCdLabel || '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-signal-text-muted">Combo CD</span>
                    <span className={cn(
                      'font-bold tabular-nums',
                      data.comboCdLabelColor === 'green' ? 'text-signal-green' :
                      data.comboCdLabelColor === 'red' ? 'text-signal-red' : 'text-signal-text-muted'
                    )}>
                      {data.comboCdLabel || '—'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Indicators */}
              <div className="bg-signal-surface rounded-lg border border-signal-border p-3">
                <div className="text-[10px] uppercase tracking-wider text-signal-text-muted mb-2">Indicators</div>
                <div className="space-y-1.5">
                  {[
                    { label: 'RSI (14)', value: data.rsi14.toFixed(1), warn: data.rsi14 < 30 || data.rsi14 > 70 },
                    { label: 'Stoch %K', value: data.stochK.toFixed(1), warn: data.stochK < 20 || data.stochK > 80 },
                    { label: 'Stoch %D', value: data.stochD.toFixed(1), warn: false },
                    { label: 'MACD Line', value: data.macdLine.toFixed(3), warn: false },
                    { label: 'MACD Signal', value: data.macdSignal.toFixed(3), warn: false },
                    { label: 'MACD Hist', value: data.macdHist.toFixed(3), warn: Math.abs(data.macdHist) > 0.5 },
                    { label: 'BB %B', value: (data.bbPct * 100).toFixed(0) + '%', warn: data.bbPct < 0.05 || data.bbPct > 0.95 },
                    { label: 'Rel/SPY 20D', value: formatPct(data.relSpy20d), warn: false },
                  ].map(ind => (
                    <div key={ind.label} className="flex items-center justify-between text-xs">
                      <span className="text-signal-text-muted">{ind.label}</span>
                      <span className={cn(
                        'font-medium tabular-nums',
                        ind.warn ? 'text-signal-amber' : 'text-signal-text-secondary'
                      )}>
                        {ind.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Performance */}
              <div className="bg-signal-surface rounded-lg border border-signal-border p-3">
                <div className="text-[10px] uppercase tracking-wider text-signal-text-muted mb-2">Performance</div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-signal-text-muted">1D Change</span>
                    <span className={cn('font-medium tabular-nums',
                      data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
                    )}>{formatPct(data.pctChg1d)}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-signal-text-muted">5D Change</span>
                    <span className={cn('font-medium tabular-nums',
                      data.pctChg5d >= 0 ? 'text-signal-green' : 'text-signal-red'
                    )}>{formatPct(data.pctChg5d)}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Fullscreen modal overlay */}
      {isFullscreen && (
        <div
          className="fixed inset-0 z-50 bg-signal-bg flex flex-col"
          data-testid="fullscreen-overlay"
        >
          {/* Fullscreen header */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-signal-border bg-signal-surface/50 backdrop-blur-sm shrink-0">
            <div className="flex items-center gap-3">
              <span className="text-sm font-bold text-signal-text">{selectedTicker}</span>
              <span className="text-xs text-signal-text-secondary capitalize">{timeframe}</span>
              {showBB && <span className="text-[10px] font-medium text-signal-amber">BB ON</span>}
              {data && (
                <>
                  <span className="text-sm font-semibold tabular-nums text-signal-text">{formatPrice(data.lastClose)}</span>
                  <span className={cn(
                    'text-xs font-semibold tabular-nums',
                    data.pctChg1d >= 0 ? 'text-signal-green' : 'text-signal-red'
                  )}>{formatPct(data.pctChg1d)}</span>
                </>
              )}
            </div>
            <div className="flex items-center gap-2">
              {/* Timeframe toggle in fullscreen */}
              <div className="flex items-center bg-signal-surface border border-signal-border rounded-lg p-0.5">
                {(['daily', 'weekly'] as const).map(tf => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={cn(
                      'px-2 py-1 rounded-md text-[10px] font-medium transition-all capitalize',
                      timeframe === tf
                        ? 'bg-signal-green/15 text-signal-green'
                        : 'text-signal-text-secondary hover:text-signal-text'
                    )}
                  >
                    {tf}
                  </button>
                ))}
              </div>
              {/* BB toggle in fullscreen */}
              <button
                onClick={() => setShowBB(prev => !prev)}
                className={cn(
                  'px-2 py-1 rounded-lg text-[10px] font-medium transition-all border',
                  showBB
                    ? 'bg-signal-amber/15 text-signal-amber border-signal-amber/30'
                    : 'bg-signal-surface text-signal-text-muted border-signal-border hover:text-signal-text'
                )}
              >
                BB
              </button>
              {/* Close */}
              <button
                onClick={() => setIsFullscreen(false)}
                data-testid="fullscreen-close"
                className="p-1.5 rounded hover:bg-white/5 text-signal-text-muted hover:text-signal-text transition-colors"
                title="Exit fullscreen (Esc)"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Fullscreen chart body — Plotly handles its own zoom/pan natively */}
          <div className="flex-1 min-h-0 p-2">
            <InteractiveChart
              key={`fs-${selectedTicker}-${timeframe}-${showBB}`}
              symbol={selectedTicker}
              timeframe={timeframe}
              showBB={showBB}
              height="100%"
              fullscreen={true}
            />
          </div>
        </div>
      )}
    </>
  );
}
