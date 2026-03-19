// ═══════════════════════════════════════
// TICKER UNIVERSE (hard-coded, editable)
// ═══════════════════════════════════════

export const TICKER_GROUPS = {
  INDICES: ['SPY', 'QQQ', 'DIA', 'IWM', 'SMH', 'UUP', 'USO', 'TLT'],
  MAG7: ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA'],
  MEGA_TECH: ['NFLX', 'AVGO', 'AMD', 'ORCL'],
  SECTORS: ['XLK', 'XLC', 'XLY', 'XLF', 'XLV', 'XLU', 'XLRE', 'XLI', 'XLP', 'XLB', 'XLE'],
} as const;

export const ALL_TICKERS = [
  ...TICKER_GROUPS.INDICES,
  ...TICKER_GROUPS.MAG7,
  ...TICKER_GROUPS.MEGA_TECH,
  ...TICKER_GROUPS.SECTORS,
];

export const SECTOR_NAMES: Record<string, string> = {
  XLK: 'Technology',
  XLC: 'Communication',
  XLY: 'Cons. Disc.',
  XLF: 'Financials',
  XLV: 'Health Care',
  XLU: 'Utilities',
  XLRE: 'Real Estate',
  XLI: 'Industrials',
  XLP: 'Cons. Staples',
  XLB: 'Materials',
  XLE: 'Energy',
};

export const BENCHMARK = 'SPY';

export type Signal = 'BUY' | 'HOLD' | 'SELL';

export interface TickerSignal {
  symbol: string;
  lastClose: number;
  pctChg1d: number;
  pctChg5d: number;
  dailyScore: number;
  dailySignal: Signal;
  weeklyScore: number;
  weeklySignal: Signal;
  td9Daily: string | null;   // e.g. "TD9 BUY", "TD9 SELL", null
  td13Daily: string | null;
  td9Weekly: string | null;
  td13Weekly: string | null;
  tdSetupCount: number;      // current active setup count (1-9), 0 if none
  tdCountdownCount: number;  // current countdown count (1-13), 0 if none
  rsi14: number;
  stochK: number;
  stochD: number;
  macdHist: number;
  macdLine: number;
  macdSignal: number;
  bbPct: number;             // (close - lower) / (upper - lower)
  bbUpper: number;
  bbMid: number;
  bbLower: number;
  relSpy20d: number;         // relative perf vs SPY, 20d
  rank?: number;             // rank within group
}

export type MarketRegime = 'RISK ON' | 'NEUTRAL' | 'RISK OFF';

export interface MarketRegimeData {
  regime: MarketRegime;
  avgScore: number;
  volatilityElevated: boolean;
  justification: string;
}
