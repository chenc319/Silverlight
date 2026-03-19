import SignalTable from '@/components/SignalTable';
import { useSignalData } from '@/data/DataProvider';
import { TICKER_GROUPS } from '@/data/tickers';

interface IndicesMacroProps {
  onTickerClick: (symbol: string) => void;
}

export default function IndicesMacro({ onTickerClick }: IndicesMacroProps) {
  const { signals } = useSignalData();
  const data = TICKER_GROUPS.INDICES.map(t => signals[t]).filter(Boolean);

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-4 animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-base font-bold text-signal-text">Indices & Macro</h1>
        <span className="text-xs text-signal-text-muted">{data.length} instruments</span>
      </div>
      <div className="bg-signal-surface rounded-lg border border-signal-border overflow-hidden">
        <SignalTable data={data} onTickerClick={onTickerClick} />
      </div>
    </div>
  );
}
