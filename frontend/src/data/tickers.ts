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

export type Signal = 'STRONG BUY' | 'BUY' | 'HOLD' | 'NEUTRAL' | 'SELL' | 'STRONG SELL';

export interface TickerSignal {
  symbol: string;
  lastClose: number;
  pctChg1d: number;
  pctChg5d: number;
  dailyScore: number;
  dailySignal: Signal;
  weeklyScore: number;
  weeklySignal: Signal;
  // DeMark completed signals
  td9Daily: string | null;      // e.g. "TD9 BUY", "TD9 SELL ●", null
  td13Seq: string | null;       // e.g. "SEQ 13 BUY", null
  td13Combo: string | null;     // e.g. "COMBO 13 SELL", null
  // Active DeMark state labels for dashboard display
  setupLabel: string | null;       // e.g. "Bearish 7" or "Bullish 5" or null
  setupLabelColor: string | null;  // "red" | "green" | null
  seqCdLabel: string | null;       // e.g. "Bullish 10" or "Bearish 8" or null
  seqCdLabelColor: string | null;
  comboCdLabel: string | null;
  comboCdLabelColor: string | null;
  demarkSignal: string | null;     // "BUY" | "SELL" | "13+" | null
  // Numeric DeMark state
  tdSetupCount: number;         // 0-9
  tdSetupSide: string | null;   // "buy" | "sell" | null
  seqCdNum: number;             // 0-13
  seqCdSide: string | null;
  comboCdNum: number;           // 0-13
  comboCdSide: string | null;
  // Traditional indicators
  rsi14: number;
  stochK: number;
  stochD: number;
  macdHist: number;
  macdLine: number;
  macdSignal: number;
  bbPct: number;
  bbUpper: number;
  bbMid: number;
  bbLower: number;
  relSpy20d: number;
  rank?: number;
}

export type MarketRegime = 'RISK ON' | 'NEUTRAL' | 'RISK OFF';

export interface MarketRegimeData {
  regime: MarketRegime;
  avgScore: number;
  volatilityElevated: boolean;
  justification: string;
}
