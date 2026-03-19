import MarketRegimeBanner from '@/components/MarketRegimeBanner';
import IndexKPICard from '@/components/IndexKPICard';
import Mag7Card from '@/components/Mag7Card';
import { useSignalData } from '@/data/DataProvider';
import { TICKER_GROUPS } from '@/data/tickers';

interface OverviewProps {
  onTickerClick: (symbol: string) => void;
}

export default function Overview({ onTickerClick }: OverviewProps) {
  const { signals } = useSignalData();

  return (
    <div className="animate-fade-in">
      <MarketRegimeBanner />

      <div className="max-w-[1600px] mx-auto px-4 py-4 space-y-6">
        {/* Index KPI Cards */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-signal-text-muted mb-3">
            Market Indices
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
            {TICKER_GROUPS.INDICES.map(ticker => {
              const data = signals[ticker];
              if (!data) return null;
              return (
                <IndexKPICard key={ticker} data={data} onClick={onTickerClick} />
              );
            })}
          </div>
        </section>

        {/* Mag 7 Signal Grid */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-signal-text-muted mb-3">
            Magnificent 7
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {TICKER_GROUPS.MAG7.map(ticker => {
              const data = signals[ticker];
              if (!data) return null;
              return (
                <Mag7Card key={ticker} data={data} onClick={onTickerClick} />
              );
            })}
          </div>
        </section>

        {/* Mega Tech */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-signal-text-muted mb-3">
            Mega Tech
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {TICKER_GROUPS.MEGA_TECH.map(ticker => {
              const data = signals[ticker];
              if (!data) return null;
              return (
                <Mag7Card key={ticker} data={data} onClick={onTickerClick} />
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
