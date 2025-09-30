'use client';

import { useState } from 'react';
import { Button, Card, CardContent, CardHeader, Drawer } from '@/components/ui';
import { useInsights } from '@/lib/api/queries';
import { formatRelative } from '@/lib/utils/format';

export default function InsightsPage() {
  const { data: insights = [] } = useInsights();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = insights.find((insight) => insight.id === selectedId);

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Insight feed</h2>
          <p className="text-sm text-muted">We watch anomalies, opportunities, and regressions for you.</p>
        </div>
        <Button variant="ghost" size="sm">
          Configure detection rules
        </Button>
      </section>

      <Card corner>
        <CardHeader title="Newest signals" description="Sorted by impact." />
        <CardContent className="space-y-4">
          {insights.map((insight) => (
            <button
              key={insight.id}
              type="button"
              onClick={() => setSelectedId(insight.id)}
              className="block w-full rounded-xl border border-border bg-elevated/40 p-4 text-left transition-colors hover:bg-elevated"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-muted">{insight.kind}</p>
                  <p className="mt-1 text-sm font-semibold text-text">{insight.title}</p>
                </div>
                <span className="text-xs text-muted">{formatRelative(insight.detectedAt)}</span>
              </div>
              <p className="mt-2 text-sm text-muted">{insight.summary}</p>
              <p className="mt-3 text-xs text-muted">Impact: {insight.impact}</p>
            </button>
          ))}
        </CardContent>
      </Card>

      <Drawer
        title={selected?.title ?? 'Insight detail'}
        description={selected ? `Detected ${formatRelative(selected.detectedAt)}` : 'Select an insight to inspect root cause.'}
        open={Boolean(selected)}
        onClose={() => setSelectedId(null)}
      >
        {selected ? (
          <div className="space-y-4">
            <p className="text-sm text-text">{selected.summary}</p>
            <div className="rounded-lg border border-border bg-elevated/30 p-4 text-sm text-text">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Root cause</p>
              <p className="mt-2">Competitor Z refreshed landing page copy last week, which is now cited across engines.</p>
            </div>
            <div className="rounded-lg border border-border bg-elevated/30 p-4 text-sm text-text">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Recommended action</p>
              <p className="mt-2">Ship matching positioning tile to client portal and update QBR narrative.</p>
            </div>
            <Button size="sm">Assign follow-up</Button>
          </div>
        ) : (
          <p className="text-sm text-muted">Pick an insight to review supporting evidence.</p>
        )}
      </Drawer>
    </div>
  );
}
