import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { fetchSignals, triggerRefresh, fetchRefreshStatus, type APISignalsResponse, type RefreshStatus } from '@/lib/api';
import { seededSignals, marketRegime as seededRegime, lastUpdated as seededLastUpdated } from './seededSignals';
import type { TickerSignal, Signal, MarketRegimeData } from './tickers';

interface DataContextType {
  signals: Record<string, TickerSignal>;
  regime: MarketRegimeData;
  lastUpdated: string;
  isLive: boolean;
  isRefreshing: boolean;
  refreshProgress: RefreshStatus['progress'];
  doRefresh: () => void;
}

const DataContext = createContext<DataContextType>({
  signals: seededSignals,
  regime: seededRegime,
  lastUpdated: seededLastUpdated,
  isLive: false,
  isRefreshing: false,
  refreshProgress: [],
  doRefresh: () => {},
});

export function useSignalData() {
  return useContext(DataContext);
}

function apiToTickerSignal(s: any): TickerSignal {
  return {
    symbol: s.symbol || '???',
    lastClose: s.lastClose || 0,
    pctChg1d: s.pctChg1d || 0,
    pctChg5d: s.pctChg5d || 0,
    dailyScore: s.dailyScore || 0,
    dailySignal: (s.dailySignal || 'HOLD') as Signal,
    weeklyScore: s.weeklyScore || 0,
    weeklySignal: (s.weeklySignal || 'HOLD') as Signal,
    // DeMark completed signals
    td9Daily: s.td9Daily || null,
    td13Seq: s.td13Seq || null,
    td13Combo: s.td13Combo || null,
    // Active DeMark labels
    setupLabel: s.setupLabel || null,
    seqCdLabel: s.seqCdLabel || null,
    comboCdLabel: s.comboCdLabel || null,
    // Numeric DeMark state
    tdSetupCount: s.tdSetupCount || 0,
    tdSetupSide: s.tdSetupSide || null,
    seqCdNum: s.seqCdNum || 0,
    seqCdSide: s.seqCdSide || null,
    comboCdNum: s.comboCdNum || 0,
    comboCdSide: s.comboCdSide || null,
    // Traditional indicators
    rsi14: s.rsi14 || 50,
    stochK: s.stochK || 50,
    stochD: s.stochD || 50,
    macdHist: s.macdHist || 0,
    macdLine: s.macdLine || 0,
    macdSignal: s.macdSignal || 0,
    bbPct: s.bbPct || 0.5,
    bbUpper: s.bbUpper || 0,
    bbMid: s.bbMid || 0,
    bbLower: s.bbLower || 0,
    relSpy20d: s.relSpy20d || 0,
    rank: s.rank,
  };
}

export function DataProvider({ children }: { children: React.ReactNode }) {
  const [signals, setSignals] = useState<Record<string, TickerSignal>>(seededSignals);
  const [regime, setRegime] = useState<MarketRegimeData>(seededRegime);
  const [lastUpdated, setLastUpdated] = useState(seededLastUpdated);
  const [isLive, setIsLive] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshProgress, setRefreshProgress] = useState<RefreshStatus['progress']>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load from API on mount
  useEffect(() => {
    fetchSignals().then(data => {
      if (data && data.tickerCount > 0) {
        applyAPIData(data);
      }
    });
  }, []);

  const applyAPIData = useCallback((data: APISignalsResponse) => {
    const newSignals: Record<string, TickerSignal> = {};
    for (const [ticker, sig] of Object.entries(data.signals)) {
      if (sig && !sig.error) {
        newSignals[ticker] = apiToTickerSignal(sig);
      }
    }

    if (Object.keys(newSignals).length > 0) {
      setSignals(newSignals);
      setIsLive(true);
    }

    setRegime({
      regime: data.regime.regime,
      avgScore: data.regime.avgScore,
      volatilityElevated: data.regime.volatilityElevated,
      justification: data.regime.justification,
    });
    setLastUpdated(data.lastUpdated);
  }, []);

  const doRefresh = useCallback(() => {
    setIsRefreshing(true);
    setRefreshProgress([]);

    triggerRefresh().then(() => {
      // Poll for status
      pollRef.current = setInterval(async () => {
        const status = await fetchRefreshStatus();
        if (status) {
          setRefreshProgress(status.progress);
          if (!status.isRefreshing) {
            // Done — reload signals
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setIsRefreshing(false);

            const data = await fetchSignals();
            if (data && data.tickerCount > 0) {
              applyAPIData(data);
            }
          }
        }
      }, 1000);
    });
  }, [applyAPIData]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  return (
    <DataContext.Provider value={{
      signals,
      regime,
      lastUpdated,
      isLive,
      isRefreshing,
      refreshProgress,
      doRefresh,
    }}>
      {children}
    </DataContext.Provider>
  );
}
