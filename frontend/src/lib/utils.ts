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

/** Check if a ticker has a Countdown 13 printed today (or very recently) */
export function hasCountdown13Today(data: TickerSignal): false | 'buy' | 'sell' {
  // seqCdNum === 13 or comboCdNum === 13 means a 13 just completed
  if (data.seqCdNum === 13 || data.comboCdNum === 13) {
    // demarkSignal tells us the direction
    if (data.demarkSignal === 'BUY') return 'buy';
    if (data.demarkSignal === 'SELL') return 'sell';
    // If no demarkSignal, infer from the countdown side
    if (data.seqCdSide === 'buy' || data.comboCdSide === 'buy') return 'buy';
    if (data.seqCdSide === 'sell' || data.comboCdSide === 'sell') return 'sell';
    return 'buy'; // fallback
  }
  return false;
}
