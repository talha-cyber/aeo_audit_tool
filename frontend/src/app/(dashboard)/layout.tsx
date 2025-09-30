'use client';

import clsx from 'clsx';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ReactNode } from 'react';
import { Button } from '@/components/ui';
import { NewAuditRunDrawer } from '@/components/audits';
import { useUIStore } from '@/store/ui';

interface DashboardLayoutProps {
  children: ReactNode;
}

const navigation = [
  { label: 'Overview', href: '/overview', helper: 'Pulse of the program' },
  { label: 'Audits', href: '/audits', helper: 'Configure + monitor runs' },
  { label: 'Reports', href: '/reports', helper: 'Executive-ready exports' },
  { label: 'Comparisons', href: '/comparisons', helper: 'Competitor matrix' },
  { label: 'Insights', href: '/insights', helper: 'Opportunities + risks' },
  { label: 'Personas', href: '/personas', helper: 'Journeys & coverage' },
  { label: 'Embeds', href: '/embeds', helper: 'Client-ready widgets' },
  { label: 'Settings', href: '/settings', helper: 'Branding, members, billing' },
  { label: 'Portal', href: '/portal', helper: 'Client read-only mode' }
];

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const pathname = usePathname();
  const { isNavCollapsed, toggleNav, openPalette, isPaletteOpen, closePalette, openNewAuditDrawer } = useUIStore();

  return (
    <div className="flex min-h-screen bg-background text-text">
      <aside
        className={clsx(
          'flex flex-col border-r border-border bg-surface transition-all duration-200',
          isNavCollapsed ? 'w-[72px]' : 'w-[280px]'
        )}
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-5">
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-[0.22em] text-muted">AEO</span>
            <span className="text-sm font-semibold text-text">Graybox Shell</span>
          </div>
          <Button variant="ghost" size="sm" onClick={toggleNav}>
            {isNavCollapsed ? 'Expand' : 'Collapse'}
          </Button>
        </div>
        <nav className="flex-1 overflow-y-auto px-3 py-6">
          <ul className="space-y-2">
            {navigation.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={clsx(
                      'block rounded-lg px-4 py-3 transition-colors',
                      active
                        ? 'bg-elevated text-text shadow-sm'
                        : 'text-muted hover:bg-elevated/60 hover:text-text'
                    )}
                  >
                    <span className="block text-sm font-semibold">{item.label}</span>
                    {!isNavCollapsed ? (
                      <span className="mt-1 block text-xs text-muted">{item.helper}</span>
                    ) : null}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="border-t border-border px-5 py-4 text-xs text-muted">
          <p>Last synced 12m ago</p>
        </div>
      </aside>
      <main className="relative flex-1">
        <header className="border-b border-border bg-surface/70 backdrop-blur px-8 py-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Command</p>
              <h1 className="text-lg font-semibold text-text">Agency Control Room</h1>
            </div>
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={openPalette}>
                ⌘K
              </Button>
              <Button variant="ghost" size="sm">
                Alerts
              </Button>
              <Button variant="primary" size="sm" onClick={openNewAuditDrawer}>
                New audit
              </Button>
            </div>
          </div>
        </header>
        <div className="px-8 py-8">{children}</div>

        {isPaletteOpen ? (
          <div className="fixed inset-0 z-40 flex items-start justify-center bg-black/40 px-6 py-20" role="dialog" aria-modal="true">
            <div className="w-full max-w-2xl rounded-xl border border-border bg-surface p-6 shadow-sheet">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-text">Command palette placeholder</p>
                <Button variant="ghost" size="sm" onClick={closePalette}>
                  Close
                </Button>
              </div>
              <p className="mt-3 text-sm text-muted">Wire to real commands after Phase 0.</p>
            </div>
          </div>
        ) : null}

        <NewAuditRunDrawer />
      </main>
    </div>
  );
}
