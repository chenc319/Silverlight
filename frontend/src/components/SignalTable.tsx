import { useState, useMemo } from 'react';
import { cn, formatPrice, formatPct, formatScore, getPctColor } from '@/lib/utils';
import SignalBadge from './SignalBadge';
import { ChevronUp, ChevronDown } from 'lucide-react';
import type { TickerSignal } from '@/data/tickers';

interface SignalTableProps {
  data: TickerSignal[];
  onTickerClick: (symbol: string) => void;
  showRank?: boolean;
  showSparkline?: boolean;
}

type SortKey = keyof TickerSignal;
type SortDir = 'asc' | 'desc';

const COLUMNS: Array<{ key: SortKey; label: string; width?: string; align?: 'left' | 'right' }> = [
  { key: 'symbol', label: 'Ticker', align: 'left' },
  { key: 'lastClose', label: 'Last Close', align: 'right' },
  { key: 'pctChg1d', label: '1D%', align: 'right' },
  { key: 'pctChg5d', label: '5D%', align: 'right' },
  { key: 'relSpy20d', label: 'Rel/SPY 20D', align: 'right' },
  { key: 'dailySignal', label: 'Daily Signal', align: 'left' },
  { key: 'dailyScore', label: 'D Score', align: 'right' },
  { key: 'weeklySignal', label: 'Weekly Signal', align: 'left' },
  { key: 'weeklyScore', label: 'W Score', align: 'right' },
  { key: 'rsi14', label: 'RSI', align: 'right' },
  { key: 'stochK', label: '%K', align: 'right' },
  { key: 'macdHist', label: 'MACD Hist', align: 'right' },
];

function DeMarkLabel({ label, color }: { label: string | null; color: string | null }) {
  if (!label) return <span className="text-signal-text-muted">—</span>;
  return (
    <span className={cn(
      'font-bold tabular-nums text-xs',
      color === 'green' ? 'text-signal-green' :
      color === 'red' ? 'text-signal-red' : 'text-signal-text-secondary'
    )}>
      {label}
    </span>
  );
}

function DeMarkSignalBadge({ signal }: { signal: string | null }) {
  if (!signal) return null;
  const isBuy = signal === 'BUY';
  const isSell = signal === 'SELL';
  const is13Plus = signal === '13+';
  return (
    <span className={cn(
      'inline-block px-1.5 py-0.5 rounded text-[10px] font-bold uppercase',
      isBuy && 'bg-signal-green/20 text-signal-green',
      isSell && 'bg-signal-red/20 text-signal-red',
      is13Plus && 'bg-amber-500/20 text-amber-400',
    )}>
      {signal}
    </span>
  );
}

export default function SignalTable({ data, onTickerClick, showRank, showSparkline }: SignalTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('dailyScore');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const sorted = useMemo(() => {
    return [...data].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;

      let cmp: number;
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        cmp = aVal.localeCompare(bVal);
      } else {
        cmp = (aVal as number) - (bVal as number);
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  const columns = useMemo(() => {
    const cols = [...COLUMNS];
    if (showRank) {
      cols.splice(1, 0, { key: 'rank' as SortKey, label: 'Rank', align: 'right' });
    }
    return cols;
  }, [showRank]);

  return (
    <div className="overflow-x-auto custom-scrollbar">
      <table className="w-full min-w-[1100px]">
        <thead>
          <tr className="sticky top-0 z-10 bg-signal-surface">
            {columns.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={cn(
                  'px-3 py-2 text-[11px] font-semibold uppercase tracking-wider cursor-pointer',
                  'text-signal-text-muted border-b border-signal-border hover:text-signal-text transition-colors',
                  col.align === 'right' ? 'text-right' : 'text-left'
                )}
              >
                <span className="inline-flex items-center gap-1">
                  {col.label}
                  {sortKey === col.key && (
                    sortDir === 'desc' ? <ChevronDown size={12} /> : <ChevronUp size={12} />
                  )}
                </span>
              </th>
            ))}
            {/* DeMark columns */}
            <th className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-signal-text-muted border-b border-signal-border text-left">
              Setup
            </th>
            <th className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-signal-text-muted border-b border-signal-border text-left">
              Seq CD
            </th>
            <th className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-signal-text-muted border-b border-signal-border text-left">
              Combo CD
            </th>
            <th className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-signal-text-muted border-b border-signal-border text-center">
              DM Signal
            </th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const isBuy = row.dailySignal === 'BUY';
            const isSell = row.dailySignal === 'SELL';

            return (
              <tr
                key={row.symbol}
                onClick={() => onTickerClick(row.symbol)}
                data-testid={`table-row-${row.symbol}`}
                className={cn(
                  'cursor-pointer border-b border-signal-border/50 hover:bg-white/[0.02] transition-colors',
                  isBuy && 'bg-signal-green/[0.03]',
                  isSell && 'bg-signal-red/[0.03]',
                  i % 2 === 0 && !isBuy && !isSell && 'bg-white/[0.01]'
                )}
              >
                {columns.map(col => {
                  const val = row[col.key];
                  return (
                    <td
                      key={col.key}
                      className={cn(
                        'px-3 py-2.5 text-xs tabular-nums',
                        col.align === 'right' ? 'text-right' : 'text-left'
                      )}
                    >
                      {col.key === 'symbol' ? (
                        <span className="font-bold text-signal-text">{val}</span>
                      ) : col.key === 'lastClose' ? (
                        <span className="font-medium">{formatPrice(val as number)}</span>
                      ) : col.key === 'pctChg1d' || col.key === 'pctChg5d' ? (
                        <span className={cn('font-medium', getPctColor(val as number))}>
                          {formatPct(val as number)}
                        </span>
                      ) : col.key === 'relSpy20d' ? (
                        <span className={cn('font-medium', getPctColor(val as number))}>
                          {formatPct(val as number)}
                        </span>
                      ) : col.key === 'dailySignal' || col.key === 'weeklySignal' ? (
                        <SignalBadge signal={val as 'BUY' | 'HOLD' | 'SELL'} />
                      ) : col.key === 'dailyScore' || col.key === 'weeklyScore' ? (
                        <span className={cn(
                          'font-bold',
                          (val as number) >= 40 ? 'text-signal-green' :
                          (val as number) <= -40 ? 'text-signal-red' : 'text-signal-text-secondary'
                        )}>
                          {formatScore(val as number)}
                        </span>
                      ) : col.key === 'rsi14' ? (
                        <span className={cn(
                          'font-medium',
                          (val as number) < 30 ? 'text-signal-green' :
                          (val as number) > 70 ? 'text-signal-red' : 'text-signal-text-secondary'
                        )}>
                          {(val as number).toFixed(1)}
                        </span>
                      ) : col.key === 'stochK' ? (
                        <span className={cn(
                          'font-medium',
                          (val as number) < 20 ? 'text-signal-green' :
                          (val as number) > 80 ? 'text-signal-red' : 'text-signal-text-secondary'
                        )}>
                          {(val as number).toFixed(1)}
                        </span>
                      ) : col.key === 'macdHist' ? (
                        <span className={cn(
                          'font-medium',
                          (val as number) > 0 ? 'text-signal-green' :
                          (val as number) < 0 ? 'text-signal-red' : 'text-signal-text-secondary'
                        )}>
                          {(val as number).toFixed(3)}
                        </span>
                      ) : col.key === 'rank' ? (
                        <span className="font-bold text-signal-text-secondary">#{val}</span>
                      ) : (
                        <span>{String(val ?? '—')}</span>
                      )}
                    </td>
                  );
                })}

                {/* Setup column */}
                <td className="px-3 py-2.5 text-xs">
                  <DeMarkLabel label={row.setupLabel} color={row.setupLabelColor} />
                </td>

                {/* Sequential Countdown column */}
                <td className="px-3 py-2.5 text-xs">
                  <DeMarkLabel label={row.seqCdLabel} color={row.seqCdLabelColor} />
                </td>

                {/* Combo Countdown column */}
                <td className="px-3 py-2.5 text-xs">
                  <DeMarkLabel label={row.comboCdLabel} color={row.comboCdLabelColor} />
                </td>

                {/* DeMark Signal column */}
                <td className="px-3 py-2.5 text-xs text-center">
                  <DeMarkSignalBadge signal={row.demarkSignal} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
