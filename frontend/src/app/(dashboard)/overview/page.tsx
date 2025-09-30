'use client';

import { useMemo } from 'react';
import { Button, Card, CardContent, CardHeader, CardFooter, ChartFrame, EmptyState, Kpi, Table } from '@/components/ui';
import { useAuditRuns, useAuditSummaries, useInsights } from '@/lib/api/queries';
import { formatRelative, formatScore } from '@/lib/utils/format';

export default function OverviewPage() {
  const { data: summaries = [] } = useAuditSummaries();
  const { data: runs = [] } = useAuditRuns();
  const { data: insights = [] } = useInsights();

  const averageHealth = useMemo(() => {
    if (!summaries.length) {
      return 0;
    }
    const total = summaries.reduce((acc, summary) => acc + summary.healthScore, 0);
    return Math.round(total / summaries.length);
  }, [summaries]);

  return (
    <div className="space-y-10">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Program pulse</h2>
          <p className="text-sm text-muted">Track visibility, sentiment, and alerts across active audits.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm">
            Last 30 days
          </Button>
          <Button variant="ghost" size="sm">
            Compare period
          </Button>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        <Kpi label="Active audits" value={`${summaries.length}`} delta={{ value: '+2', trend: 'up' }} />
        <Kpi label="Avg. health" value={formatScore(averageHealth)} delta={{ value: '+6', trend: 'up' }} />
        <Kpi label="Signals resolved" value="74" delta={{ value: '-4', trend: 'down' }} />
        <Kpi label="Portal coverage" value="82%" delta={{ value: '+3', trend: 'up' }} />
      </section>

      <section className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" corner>
          <CardHeader title="Recent audit runs" description="Live progress across the active schedules." action={<Button size="sm">View all runs</Button>} />
          <CardContent>
            <Table
              data={runs}
              columns={[
                { key: 'name', header: 'Run' },
                {
                  key: 'status',
                  header: 'Status',
                  render: (run) => (
                    <span className="text-sm text-text capitalize">{run.status}</span>
                  )
                },
                {
                  key: 'progress',
                  header: 'Completion',
                  render: (run) => `${run.progress.done}/${run.progress.total}`
                },
                {
                  key: 'startedAt',
                  header: 'Updated',
                  render: (run) => (run.startedAt ? formatRelative(run.startedAt) : '—')
                }
              ]}
              emptyState={<EmptyState headline="No runs yet" helper="Start an audit to monitor AI mention coverage." actionLabel="Create audit" />}
            />
          </CardContent>
          <CardFooter>
            <span>Issues flagged: {runs.reduce((acc, run) => acc + run.issues.length, 0)}</span>
            <Button variant="ghost" size="sm">
              Export CSV
            </Button>
          </CardFooter>
        </Card>

        <Card corner>
          <CardHeader title="Alert feed" description="Newest risks and opportunities." action={<Button size="sm" variant="ghost">Mute filters</Button>} />
          <CardContent className="space-y-3">
            {insights.slice(0, 4).map((insight) => (
              <div key={insight.id} className="rounded-lg border border-border bg-elevated/30 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-muted">{insight.kind}</p>
                <p className="mt-1 text-sm font-semibold text-text">{insight.title}</p>
                <p className="mt-2 text-sm text-muted">{insight.summary}</p>
                <p className="mt-2 text-xs text-muted">Detected {formatRelative(insight.detectedAt)} · Impact {insight.impact}</p>
              </div>
            ))}
          </CardContent>
          <CardFooter>
            <Button size="sm" variant="ghost">
              View insights
            </Button>
            <span>Auto-published to portal</span>
          </CardFooter>
        </Card>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <ChartFrame title="Visibility trend" description="Relative share of voice across tracked platforms." height="md">
          <div className="flex h-full items-center justify-center text-sm text-muted">
            Chart placeholder (Connect analytics during Phase 1)
          </div>
        </ChartFrame>
        <ChartFrame title="Sentiment balance" description="Distribution of positive vs. negative mentions." height="md">
          <div className="flex h-full items-center justify-center text-sm text-muted">
            Chart placeholder (Connect analytics during Phase 1)
          </div>
        </ChartFrame>
      </section>
    </div>
  );
}
