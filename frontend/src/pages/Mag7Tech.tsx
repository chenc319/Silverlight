import SignalTable from '@/components/SignalTable';
import { useSignalData } from '@/data/DataProvider';
import { TICKER_GROUPS } from '@/data/tickers';

interface Mag7TechProps {
  onTickerClick: (symbol: string) => void;
}

export default function Mag7Tech({ onTickerClick }: Mag7TechProps) {
  const { signals } = useSignalData();
  const tickers = [...TICKER_GROUPS.MAG7, ...TICKER_GROUPS.MEGA_TECH];
  const data = tickers.map(t => signals[t]).filter(Boolean);

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-4 animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-bold text-signal-text">Mag 7 + Tech</h1>
        <span className="text-sm text-signal-text-muted">{data.length} instruments</span>
      </div>
      <div className="bg-signal-surface rounded-lg border border-signal-border overflow-hidden">
        <SignalTable data={data} onTickerClick={onTickerClick} showRank />
      </div>
    </div>
  );
}
