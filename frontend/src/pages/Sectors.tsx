import SignalTable from '@/components/SignalTable';
import SectorHeatmap from '@/components/SectorHeatmap';
import { useSignalData } from '@/data/DataProvider';
import { TICKER_GROUPS } from '@/data/tickers';

interface SectorsProps {
  onTickerClick: (symbol: string) => void;
}

export default function Sectors({ onTickerClick }: SectorsProps) {
  const { signals } = useSignalData();
  const data = TICKER_GROUPS.SECTORS.map(t => signals[t]).filter(Boolean);

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-4 animate-fade-in space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-base font-bold text-signal-text">Sectors</h1>
        <span className="text-xs text-signal-text-muted">{data.length} sector ETFs</span>
      </div>
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-signal-text-muted mb-3">
          Sector Rotation Heatmap
        </h2>
        <SectorHeatmap onTickerClick={onTickerClick} />
      </section>
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-signal-text-muted mb-3">
          Detailed Signals
        </h2>
        <div className="bg-signal-surface rounded-lg border border-signal-border overflow-hidden">
          <SignalTable data={data} onTickerClick={onTickerClick} showRank />
        </div>
      </section>
    </div>
  );
}
