import { cn, getSignalBg } from '@/lib/utils';
import type { Signal } from '@/data/tickers';

interface SignalBadgeProps {
  signal: Signal;
  size?: 'sm' | 'md';
  className?: string;
}

/** Short label for compact display */
function shortLabel(signal: Signal): string {
  switch (signal) {
    case 'STRONG BUY': return 'STR BUY';
    case 'STRONG SELL': return 'STR SELL';
    default: return signal;
  }
}

export default function SignalBadge({ signal, size = 'sm', className }: SignalBadgeProps) {
  const label = size === 'sm' ? shortLabel(signal) : signal;
  return (
    <span
      data-testid={`signal-badge-${signal.toLowerCase().replace(' ', '-')}`}
      className={cn(
        'inline-flex items-center justify-center rounded border font-semibold uppercase tracking-wider whitespace-nowrap',
        getSignalBg(signal),
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm',
        className
      )}
    >
      {label}
    </span>
  );
}
