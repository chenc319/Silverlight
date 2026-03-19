import { cn } from '@/lib/utils';
import { TICKER_GROUPS, SECTOR_NAMES } from '@/data/tickers';
import { useSignalData } from '@/data/DataProvider';

interface SectorHeatmapProps {
  onTickerClick: (symbol: string) => void;
}

export default function SectorHeatmap({ onTickerClick }: SectorHeatmapProps) {
  const { signals } = useSignalData();

  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
      {TICKER_GROUPS.SECTORS.map(ticker => {
        const data = signals[ticker];
        if (!data) return null;

        const score = data.dailyScore;
        const intensity = Math.min(Math.abs(score) / 100, 1);

        let bgColor: string;
        let textColor: string;
        if (score >= 40) {
          bgColor = `rgba(0, 212, 160, ${0.1 + intensity * 0.25})`;
          textColor = '#00d4a0';
        } else if (score <= -40) {
          bgColor = `rgba(255, 77, 109, ${0.1 + intensity * 0.25})`;
          textColor = '#ff4d6d';
        } else {
          bgColor = `rgba(245, 158, 11, ${0.05 + intensity * 0.1})`;
          textColor = '#f59e0b';
        }

        return (
          <button
            key={ticker}
            onClick={() => onTickerClick(ticker)}
            data-testid={`heatmap-${ticker}`}
            className={cn(
              'flex flex-col items-center justify-center p-3 rounded-lg border border-signal-border',
              'hover:border-signal-border-light transition-all cursor-pointer min-h-[80px]'
            )}
            style={{ backgroundColor: bgColor }}
          >
            <span className="text-xs font-bold text-signal-text">{ticker}</span>
            <span className="text-[10px] text-signal-text-muted mt-0.5">
              {SECTOR_NAMES[ticker] || ticker}
            </span>
            <span className="text-sm font-bold mt-1 tabular-nums" style={{ color: textColor }}>
              {score >= 0 ? '+' : ''}{score}
            </span>
            <span className="text-[10px] font-semibold uppercase mt-0.5" style={{ color: textColor }}>
              {data.dailySignal}
            </span>
          </button>
        );
      })}
    </div>
  );
}
