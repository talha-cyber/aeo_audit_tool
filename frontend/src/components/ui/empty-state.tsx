import { ReactNode } from 'react';
import { Button } from './button';

interface EmptyStateProps {
  headline: string;
  helper: string;
  actionLabel: string;
  onAction?: () => void;
  actionSlot?: ReactNode;
}

export function EmptyState({ headline, helper, actionLabel, onAction, actionSlot }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-xl border border-border bg-surface p-8">
      <div>
        <h3 className="text-lg font-semibold text-text">{headline}</h3>
        <p className="mt-2 text-sm text-muted">{helper}</p>
      </div>
      {actionSlot ?? (
        <Button variant="primary" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
}
