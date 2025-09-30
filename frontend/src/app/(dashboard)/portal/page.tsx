'use client';

import { Button, Card, CardContent, CardHeader, EmptyState } from '@/components/ui';
import { useReportSummaries } from '@/lib/api/queries';

export default function PortalPage() {
  const { data: reports = [] } = useReportSummaries();

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Client portal</h2>
          <p className="text-sm text-muted">Locked read-only view for clients and exec sponsors.</p>
        </div>
        <Button size="sm">Preview as client</Button>
      </section>

      <Card corner>
        <CardHeader title="Featured report" description="Quick share link for the latest audit summary." />
        <CardContent>
          {reports.length ? (
            <div className="rounded-xl border border-border bg-elevated/30 p-5 text-sm text-text">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Latest</p>
              <p className="mt-2 text-sm font-semibold text-text">{reports[0].title}</p>
              <p className="mt-2 text-sm text-muted">Coverage {reports[0].coverage.completed}/{reports[0].coverage.total}</p>
              <Button variant="ghost" size="sm" className="mt-4">
                Open in reader
              </Button>
            </div>
          ) : (
            <EmptyState
              headline="No reports yet"
              helper="The portal stays in gray mode until reports publish."
              actionLabel="Queue report"
            />
          )}
        </CardContent>
      </Card>

      <Card corner>
        <CardHeader title="Portal guardrails" description="Remind agents what is safe to expose." />
        <CardContent className="space-y-3 text-sm text-text">
          <p>• All AI responses scrubbed for PII before surfacing.</p>
          <p>• Insights limited to approved verticals.</p>
          <p>• Download rights disabled by default.</p>
        </CardContent>
      </Card>
    </div>
  );
}
