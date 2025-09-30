import clsx from 'clsx';
import { ReactNode } from 'react';

interface ChartFrameProps {
  title: string;
  description?: string;
  children: ReactNode;
  height?: 'sm' | 'md' | 'lg';
}

const heightMap: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'h-48',
  md: 'h-72',
  lg: 'h-96'
};

export function ChartFrame({ title, description, children, height = 'md' }: ChartFrameProps) {
  return (
    <section className={clsx('flex flex-col rounded-xl border border-border bg-surface p-6', heightMap[height])}>
      <header>
        <h3 className="text-base font-semibold text-text">{title}</h3>
        {description ? <p className="mt-1 text-sm text-muted">{description}</p> : null}
      </header>
      <div className="mt-4 flex-1 rounded-lg border border-dashed border-border bg-elevated/40" aria-label={`${title} placeholder`}>
        {children}
      </div>
    </section>
  );
}
