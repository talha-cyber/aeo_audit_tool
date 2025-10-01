'use client';

import { useAuditRunDetail } from '@/lib/api/queries';
import { formatRelative } from '@/lib/utils/format';
import { Button, Card, CardContent, CardHeader, ChartFrame, Table } from '@/components/ui';

interface RunDetailProps {
  runId: string;
}

export function RunDetail({ runId }: RunDetailProps) {
  const { data, isLoading } = useAuditRunDetail(runId);

  if (isLoading || !data) {
    return <p className="text-sm text-muted">Loading run detail…</p>;
  }

  const { run, questions } = data;
  const total = run.progress.total || 1;
  const completion = Math.min(100, Math.round((run.progress.done / total) * 100));

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-muted">Run detail</p>
          <h2 className="text-xl font-semibold text-text">{run.name}</h2>
          <p className="text-sm text-muted">Started {run.startedAt ? formatRelative(run.startedAt) : '—'} • Status {run.status}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm">
            Pause run
          </Button>
          <Button size="sm">Re-run</Button>
        </div>
      </section>

      <Card corner>
        <CardHeader title="Progress lane" description="Question batches grouped by platform." />
        <CardContent className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-border bg-elevated/40 p-4">
            <p className="text-sm font-semibold text-text">Completion</p>
            <p className="mt-1 text-xs uppercase tracking-[0.18em] text-muted">{completion}% done</p>
            <div className="mt-3 h-2 rounded-full bg-elevated">
              <div className="h-2 rounded-full bg-text" style={{ width: `${completion}%` }} />
            </div>
            <p className="mt-2 text-sm text-muted">
              {run.progress.done} of {run.progress.total} questions processed.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-elevated/40 p-4">
            <p className="text-sm font-semibold text-text">Issues</p>
            {run.issues.length ? (
              <ul className="mt-3 space-y-2 text-sm text-text">
                {run.issues.map((issue) => (
                  <li key={issue.id}>
                    <span className="font-medium">[{issue.severity}]</span> {issue.label}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-muted">No issues flagged.</p>
            )}
          </div>
        </CardContent>
      </Card>

      <ChartFrame title="Platform tempo" description="Batch completion split by model." height="sm">
        <div className="flex h-full items-center justify-center text-sm text-muted">
          Timeline placeholder
        </div>
      </ChartFrame>

      <Card corner>
        <CardHeader title="Question responses" description="Each row tracks sentiment and brand mentions." />
        <CardContent>
          <Table
            data={questions}
            columns={[
              { key: 'platform', header: 'Platform' },
              { key: 'prompt', header: 'Prompt' },
              {
                key: 'mentions',
                header: 'Brand coverage',
                render: (question) => (
                  <div className="space-y-1">
                    {question.mentions.map((mention) => (
                      <div key={mention.brand} className="flex items-center justify-between text-xs text-text">
                        <span>{mention.brand}</span>
                        <span>
                          {mention.frequency} · {mention.sentiment}
                        </span>
                      </div>
                    ))}
                  </div>
                )
              },
              {
                key: 'sentiment',
                header: 'Sentiment'
              }
            ]}
          />
        </CardContent>
      </Card>
    </div>
  );
}
