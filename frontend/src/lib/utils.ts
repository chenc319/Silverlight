import { clsx, type ClassValue } from 'clsx';
import type { Signal, TickerSignal } from '@/data/tickers';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatPrice(value: number): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function formatPct(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

export function formatScore(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}`;
}

/** Color for the score number itself */
export function getScoreColor(score: number): string {
  if (score >= 4) return 'text-signal-green';
  if (score <= -4) return 'text-signal-red';
  if (score >= 1) return 'text-signal-amber';
  if (score <= -1) return 'text-signal-amber';
  return 'text-signal-text-secondary';
}

export function getSignalColor(signal: Signal): string {
  switch (signal) {
    case 'STRONG BUY':
    case 'BUY': return 'text-signal-green';
    case 'STRONG SELL':
    case 'SELL': return 'text-signal-red';
    case 'HOLD':
    case 'NEUTRAL':
    default: return 'text-signal-amber';
  }
}

export function getSignalBg(signal: Signal): string {
  switch (signal) {
    case 'STRONG BUY':
    case 'BUY': return 'bg-signal-green/15 text-signal-green border-signal-green/30';
    case 'STRONG SELL':
    case 'SELL': return 'bg-signal-red/15 text-signal-red border-signal-red/30';
    case 'HOLD':
    case 'NEUTRAL':
    default: return 'bg-signal-amber/15 text-signal-amber border-signal-amber/30';
  }
}

export function getPctColor(value: number): string {
  if (value > 0) return 'text-signal-green';
  if (value < 0) return 'text-signal-red';
  return 'text-signal-text-secondary';
}

/** 
 * Check if a ticker has a Countdown 13 printed today (or very recently).
 * Returns an object with direction and type(s), or false.
 */
export type Countdown13Info = {
  direction: 'buy' | 'sell';
  types: ('sequential' | 'combo')[];
};

export function hasCountdown13Today(data: TickerSignal): false | Countdown13Info {
  const isSeq13 = data.seqCdNum === 13;
  const isCombo13 = data.comboCdNum === 13;

  if (!isSeq13 && !isCombo13) return false;

  const types: ('sequential' | 'combo')[] = [];
  if (isSeq13) types.push('sequential');
  if (isCombo13) types.push('combo');

  // Determine direction
  let direction: 'buy' | 'sell' = 'buy';
  if (data.demarkSignal === 'BUY') direction = 'buy';
  else if (data.demarkSignal === 'SELL') direction = 'sell';
  else if (isSeq13 && data.seqCdSide) direction = data.seqCdSide as 'buy' | 'sell';
  else if (isCombo13 && data.comboCdSide) direction = data.comboCdSide as 'buy' | 'sell';

  return { direction, types };
}

/** Helper: get the display label for a Countdown 13 badge */
export function getCountdown13Label(info: Countdown13Info): string {
  if (info.types.length === 2) return '13 Seq + Combo';
  if (info.types[0] === 'sequential') return '13 Sequential';
  return '13 Combo';
}
