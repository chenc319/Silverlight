import { cn, getSignalBg } from '@/lib/utils';
import type { Signal } from '@/data/tickers';

interface SignalBadgeProps {
  signal: Signal;
  size?: 'sm' | 'md';
  className?: string;
}

export default function SignalBadge({ signal, size = 'sm', className }: SignalBadgeProps) {
  return (
    <span
      data-testid={`signal-badge-${signal.toLowerCase()}`}
      className={cn(
        'inline-flex items-center justify-center rounded border font-semibold uppercase tracking-wider',
        getSignalBg(signal),
        size === 'sm' ? 'px-2 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs',
        className
      )}
    >
      {signal}
    </span>
  );
}
