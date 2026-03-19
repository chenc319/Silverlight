import { clsx, type ClassValue } from 'clsx';

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
  return `${sign}${value}`;
}

export function getSignalColor(signal: 'BUY' | 'HOLD' | 'SELL'): string {
  switch (signal) {
    case 'BUY': return 'text-signal-green';
    case 'SELL': return 'text-signal-red';
    case 'HOLD': return 'text-signal-amber';
  }
}

export function getSignalBg(signal: 'BUY' | 'HOLD' | 'SELL'): string {
  switch (signal) {
    case 'BUY': return 'bg-signal-green/15 text-signal-green border-signal-green/30';
    case 'SELL': return 'bg-signal-red/15 text-signal-red border-signal-red/30';
    case 'HOLD': return 'bg-signal-amber/15 text-signal-amber border-signal-amber/30';
  }
}

export function getPctColor(value: number): string {
  if (value > 0) return 'text-signal-green';
  if (value < 0) return 'text-signal-red';
  return 'text-signal-text-secondary';
}
