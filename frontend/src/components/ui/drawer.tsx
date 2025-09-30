'use client';

import clsx from 'clsx';
import { ReactNode } from 'react';
import { Button } from './button';

interface DrawerProps {
  title: string;
  description?: string;
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  width?: 'sm' | 'md';
}

const widthMap: Record<'sm' | 'md', string> = {
  sm: 'max-w-md',
  md: 'max-w-xl'
};

export function Drawer({ title, description, open, onClose, children, width = 'md' }: DrawerProps) {
  return (
    <div
      className={clsx(
        'fixed inset-0 z-50 transition-opacity duration-200',
        open ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'
      )}
      aria-hidden={!open}
    >
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="absolute inset-y-0 right-0 flex max-w-full">
        <div className={clsx('h-full w-screen border-l border-border bg-surface shadow-sheet', widthMap[width])}>
          <div className="flex h-full flex-col">
            <div className="border-b border-border px-6 py-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-text">{title}</h2>
                  {description ? <p className="mt-1 text-sm text-muted">{description}</p> : null}
                </div>
                <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close drawer">
                  Close
                </Button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-6 py-6 text-sm leading-relaxed text-text">{children}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
