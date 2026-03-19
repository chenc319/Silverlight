import { cn } from '@/lib/utils';

interface DeMarkBadgeProps {
  label: string | null;
  className?: string;
}

export default function DeMarkBadge({ label, className }: DeMarkBadgeProps) {
  if (!label) return <span className="text-signal-text-muted text-xs">—</span>;

  const isBuy = label.includes('BUY');
  const isSell = label.includes('SELL');

  return (
    <span
      data-testid={`demark-badge`}
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold tracking-wide border',
        isBuy && 'bg-signal-green/10 text-signal-green border-signal-green/25',
        isSell && 'bg-signal-red/10 text-signal-red border-signal-red/25',
        !isBuy && !isSell && 'bg-signal-amber/10 text-signal-amber border-signal-amber/25',
        className
      )}
    >
      <span className={cn(
        'w-1.5 h-1.5 rounded-full',
        isBuy ? 'bg-signal-green animate-pulse-subtle' : 'bg-signal-red animate-pulse-subtle'
      )} />
      {label}
    </span>
  );
}
