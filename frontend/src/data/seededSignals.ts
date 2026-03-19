import { TickerSignal, Signal, MarketRegimeData, TICKER_GROUPS, SECTOR_NAMES } from './tickers';

// ─── Deterministic pseudo-random from seed ───
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

const rand = seededRandom(42);

function randRange(min: number, max: number): number {
  return min + rand() * (max - min);
}

function pickSignal(score: number): Signal {
  if (score >= 40) return 'BUY';
  if (score <= -40) return 'SELL';
  return 'HOLD';
}

// ─── Realistic price data ───
const PRICES: Record<string, number> = {
  SPY: 567.82, QQQ: 487.35, DIA: 425.91, IWM: 204.56,
  SMH: 231.44, UUP: 27.31, USO: 72.18, TLT: 88.42,
  AAPL: 227.63, AMZN: 205.18, GOOGL: 176.92, META: 589.34,
  MSFT: 425.67, NVDA: 131.28, TSLA: 271.45,
  NFLX: 987.12, AVGO: 201.33, AMD: 119.87, ORCL: 172.56,
  XLK: 218.34, XLC: 92.15, XLY: 198.72, XLF: 47.83,
  XLV: 146.29, XLU: 78.51, XLRE: 41.23, XLI: 129.45,
  XLP: 82.67, XLB: 87.92, XLE: 88.14,
};

// ─── Generate signals for each ticker ───
function generateTickerSignal(symbol: string): TickerSignal {
  const price = PRICES[symbol] || 100 + randRange(-50, 200);
  const pctChg1d = randRange(-3.5, 3.5);
  const pctChg5d = randRange(-7, 7);

  // Generate indicator values
  const rsi14 = Math.round(randRange(18, 82) * 10) / 10;
  const stochK = Math.round(randRange(10, 90) * 10) / 10;
  const stochD = Math.round((stochK + randRange(-10, 10)) * 10) / 10;
  const macdLine = randRange(-2, 2);
  const macdSignal = macdLine + randRange(-0.5, 0.5);
  const macdHist = macdLine - macdSignal;
  const bbMid = price;
  const bbRange = price * randRange(0.02, 0.06);
  const bbUpper = bbMid + bbRange;
  const bbLower = bbMid - bbRange;
  const bbPct = Math.round(randRange(0, 1) * 100) / 100;
  const relSpy20d = Math.round(randRange(-8, 8) * 100) / 100;

  // Score calculation (simplified for seeded data)
  let dailyScore = 0;

  // DeMark contribution
  const tdRoll = rand();
  let td9Daily: string | null = null;
  let td13Seq: string | null = null;
  let td13Combo: string | null = null;
  let setupLabel: string | null = null;
  let setupLabelColor: string | null = null;
  let seqCdLabel: string | null = null;
  let seqCdLabelColor: string | null = null;
  let comboCdLabel: string | null = null;
  let comboCdLabelColor: string | null = null;
  let demarkSignal: string | null = null;
  let tdSetupCount = 0;
  let tdSetupSide: string | null = null;
  let seqCdNum = 0;
  let seqCdSide: string | null = null;
  let comboCdNum = 0;
  let comboCdSide: string | null = null;

  if (tdRoll < 0.12) {
    td9Daily = 'TD9 BUY';
    dailyScore += 30;
    tdSetupCount = 9;
    tdSetupSide = 'buy';
    setupLabel = 'Bearish 9';
    setupLabelColor = 'red';
  } else if (tdRoll < 0.18) {
    td13Seq = 'SEQ 13 BUY';
    dailyScore += 40;
    seqCdNum = 13;
    seqCdSide = 'buy';
    seqCdLabel = 'Bullish 13';
    seqCdLabelColor = 'green';
    demarkSignal = 'BUY';
  } else if (tdRoll < 0.30) {
    td9Daily = 'TD9 SELL';
    dailyScore -= 30;
    tdSetupCount = 9;
    tdSetupSide = 'sell';
    setupLabel = 'Bullish 9';
    setupLabelColor = 'green';
  } else if (tdRoll < 0.36) {
    td13Combo = 'COMBO 13 SELL';
    dailyScore -= 40;
    comboCdNum = 13;
    comboCdSide = 'sell';
    comboCdLabel = 'Bearish 13';
    comboCdLabelColor = 'red';
    demarkSignal = 'SELL';
  } else if (tdRoll < 0.55) {
    tdSetupCount = Math.floor(randRange(1, 8));
    tdSetupSide = rand() > 0.5 ? 'buy' : 'sell';
    setupLabel = `${tdSetupSide === 'buy' ? 'Bearish' : 'Bullish'} ${tdSetupCount}`;
    setupLabelColor = tdSetupSide === 'buy' ? 'red' : 'green';
    dailyScore += tdSetupCount > 5 ? (tdSetupSide === 'buy' ? 15 : -15) : 0;
  } else if (tdRoll < 0.65) {
    seqCdNum = Math.floor(randRange(1, 12));
    seqCdSide = rand() > 0.5 ? 'buy' : 'sell';
    seqCdLabel = `${seqCdSide === 'buy' ? 'Bullish' : 'Bearish'} ${seqCdNum}`;
    seqCdLabelColor = seqCdSide === 'buy' ? 'green' : 'red';
    dailyScore += seqCdSide === 'buy' ? 15 : -15;
  } else if (tdRoll < 0.75) {
    comboCdNum = Math.floor(randRange(1, 12));
    comboCdSide = rand() > 0.5 ? 'buy' : 'sell';
    comboCdLabel = `${comboCdSide === 'buy' ? 'Bullish' : 'Bearish'} ${comboCdNum}`;
    comboCdLabelColor = comboCdSide === 'buy' ? 'green' : 'red';
    dailyScore += comboCdSide === 'buy' ? 10 : -10;
  }

  // RSI
  if (rsi14 < 30) dailyScore += Math.round(15 * (30 - rsi14) / 30);
  else if (rsi14 > 70) dailyScore -= Math.round(15 * (rsi14 - 70) / 30);

  // Stochastics
  if (stochK < 20 && stochK > stochD) dailyScore += 10;
  else if (stochK > 80 && stochK < stochD) dailyScore -= 10;

  // MACD
  if (macdHist > 0 && macdLine > macdSignal) dailyScore += 10;
  else if (macdHist < 0 && macdLine < macdSignal) dailyScore -= 10;

  // BB
  if (bbPct < 0.05) dailyScore += 5;
  else if (bbPct > 0.95) dailyScore -= 5;

  // Relative
  if (relSpy20d > 0) dailyScore += 5;
  else dailyScore -= 5;

  dailyScore = Math.max(-100, Math.min(100, dailyScore));

  const weeklyScore = Math.max(-100, Math.min(100,
    Math.round(dailyScore * randRange(0.5, 1.2))
  ));

  return {
    symbol,
    lastClose: Math.round(price * 100) / 100,
    pctChg1d: Math.round(pctChg1d * 100) / 100,
    pctChg5d: Math.round(pctChg5d * 100) / 100,
    dailyScore,
    dailySignal: pickSignal(dailyScore),
    weeklyScore,
    weeklySignal: pickSignal(weeklyScore),
    td9Daily,
    td13Seq,
    td13Combo,
    setupLabel,
    setupLabelColor,
    seqCdLabel,
    seqCdLabelColor,
    comboCdLabel,
    comboCdLabelColor,
    demarkSignal,
    tdSetupCount,
    tdSetupSide,
    seqCdNum,
    seqCdSide,
    comboCdNum,
    comboCdSide,
    rsi14,
    stochK: Math.max(0, Math.min(100, stochK)),
    stochD: Math.max(0, Math.min(100, stochD)),
    macdHist: Math.round(macdHist * 1000) / 1000,
    macdLine: Math.round(macdLine * 1000) / 1000,
    macdSignal: Math.round(macdSignal * 1000) / 1000,
    bbPct,
    bbUpper: Math.round(bbUpper * 100) / 100,
    bbMid: Math.round(bbMid * 100) / 100,
    bbLower: Math.round(bbLower * 100) / 100,
    relSpy20d,
  };
}

// ─── Generate all signals ───
const allSignals: Record<string, TickerSignal> = {};
const allTickers = [
  ...TICKER_GROUPS.INDICES,
  ...TICKER_GROUPS.MAG7,
  ...TICKER_GROUPS.MEGA_TECH,
  ...TICKER_GROUPS.SECTORS,
];

for (const t of allTickers) {
  allSignals[t] = generateTickerSignal(t);
}

// Rank within groups
function rankGroup(tickers: readonly string[]) {
  const sorted = [...tickers].sort(
    (a, b) => (allSignals[b]?.dailyScore || 0) - (allSignals[a]?.dailyScore || 0)
  );
  sorted.forEach((t, i) => {
    if (allSignals[t]) allSignals[t].rank = i + 1;
  });
}

rankGroup(TICKER_GROUPS.MAG7);
rankGroup(TICKER_GROUPS.MEGA_TECH);
rankGroup(TICKER_GROUPS.SECTORS);

// ─── Market Regime ───
function computeRegime(): MarketRegimeData {
  const indexScores = TICKER_GROUPS.INDICES.map(t => allSignals[t]?.dailyScore || 0);
  const avgScore = Math.round(indexScores.reduce((a, b) => a + b, 0) / indexScores.length);

  const buyCount = TICKER_GROUPS.INDICES.filter(
    t => allSignals[t]?.dailySignal === 'BUY'
  ).length;
  const sellCount = TICKER_GROUPS.INDICES.filter(
    t => allSignals[t]?.dailySignal === 'SELL'
  ).length;

  let regime: 'RISK ON' | 'NEUTRAL' | 'RISK OFF';
  let justification: string;

  if (avgScore > 25) {
    regime = 'RISK ON';
    justification = `${buyCount} of ${TICKER_GROUPS.INDICES.length} indices showing bullish DeMark signals`;
  } else if (avgScore < -25) {
    regime = 'RISK OFF';
    justification = `${sellCount} of ${TICKER_GROUPS.INDICES.length} indices showing bearish signals with elevated selling pressure`;
  } else {
    regime = 'NEUTRAL';
    justification = `Mixed signals across indices — ${buyCount} bullish, ${sellCount} bearish, ${TICKER_GROUPS.INDICES.length - buyCount - sellCount} neutral`;
  }

  return {
    regime,
    avgScore,
    volatilityElevated: rand() > 0.6,
    justification,
  };
}

export const seededSignals = allSignals;
export const marketRegime = computeRegime();
export const lastUpdated = new Date().toISOString();

// ─── Seeded OHLC data for charts ───
export function generateOHLC(symbol: string, bars: number = 252): Array<{
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  tdSetup?: number;
  tdCountdown?: number;
}> {
  const r = seededRandom(symbol.split('').reduce((a, c) => a + c.charCodeAt(0), 0));
  const basePrice = PRICES[symbol] || 100;
  const data: Array<{
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    tdSetup?: number;
    tdCountdown?: number;
  }> = [];

  let price = basePrice * (0.7 + r() * 0.4);
  const startDate = new Date('2024-03-19');

  // Simple DeMark setup counter
  let setupCount = 0;
  let setupDir = 0; // 1 = sell setup (up), -1 = buy setup (down)
  let countdownCount = 0;

  for (let i = 0; i < bars; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    // Skip weekends
    const day = date.getDay();
    if (day === 0 || day === 6) continue;

    const dailyVol = price * (0.005 + r() * 0.025);
    const direction = r() > 0.48 ? 1 : -1;
    const open = price;
    const close = price + direction * dailyVol * r();
    const high = Math.max(open, close) + dailyVol * r() * 0.5;
    const low = Math.min(open, close) - dailyVol * r() * 0.5;
    const volume = Math.round(1000000 + r() * 50000000);

    // Simple TD Sequential setup simulation
    let tdSetup: number | undefined;
    let tdCountdown: number | undefined;

    if (data.length >= 4) {
      const close4back = data[data.length - 4].close;
      if (close < close4back) {
        if (setupDir === -1) {
          setupCount++;
        } else {
          setupDir = -1;
          setupCount = 1;
        }
      } else if (close > close4back) {
        if (setupDir === 1) {
          setupCount++;
        } else {
          setupDir = 1;
          setupCount = 1;
        }
      } else {
        setupCount = 0;
        setupDir = 0;
      }

      if (setupCount >= 1 && setupCount <= 9) {
        tdSetup = setupDir * setupCount;
      }
      if (setupCount === 9) {
        countdownCount = 1;
      } else if (countdownCount > 0 && countdownCount < 13) {
        if (data.length >= 2) {
          const cond = setupDir === -1
            ? close <= data[data.length - 2].low
            : close >= data[data.length - 2].high;
          if (cond) {
            countdownCount++;
            tdCountdown = countdownCount;
          }
        }
      }
      if (countdownCount >= 13) countdownCount = 0;
    }

    price = Math.max(price * 0.5, close);

    data.push({
      time: date.toISOString().split('T')[0],
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume,
      tdSetup,
      tdCountdown,
    });
  }

  return data;
}
