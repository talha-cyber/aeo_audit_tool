'use client';

import { Button, Card, CardContent, CardFooter, CardHeader, EmptyState, Table } from '@/components/ui';
import { useReportSummaries } from '@/lib/api/queries';
import { formatRelative } from '@/lib/utils/format';

export default function ReportsPage() {
  const { data: reports = [] } = useReportSummaries();

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Report library</h2>
          <p className="text-sm text-muted">Executive-ready exports, grouped by cadence and audience.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm">
            Filter: executive
          </Button>
          <Button variant="ghost" size="sm">
            Filter: tactical
          </Button>
          <Button size="sm">
            Generate new report
          </Button>
        </div>
      </section>

      <Card corner>
        <CardHeader title="Generated reports" description="Lock copy during graybox; style later with tokens." />
        <CardContent>
          <Table
            data={reports}
            columns={[
              { key: 'title', header: 'Title' },
              {
                key: 'generatedAt',
                header: 'Generated',
                render: (report) => formatRelative(report.generatedAt)
              },
              {
                key: 'coverage',
                header: 'Coverage',
                render: (report) => `${report.coverage.completed}/${report.coverage.total}`
              },
              {
                key: 'auditId',
                header: 'Audit'
              }
            ]}
            emptyState={
              <EmptyState
                headline="No reports yet"
                helper="Run an audit and we will queue the executive summary for you."
                actionLabel="Queue report"
              />
            }
          />
        </CardContent>
        <CardFooter>
          <Button variant="ghost" size="sm" href="/reports/preview">
            Preview reader
          </Button>
          <span className="text-xs text-muted">Exports land in reports/</span>
        </CardFooter>
      </Card>

      <Card corner>
        <CardHeader title="Evidence gallery" description="Curate callouts and proof points." />
        <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((item) => (
            <div key={item} className="rounded-lg border border-border bg-elevated/30 p-4 text-sm text-text">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Artifact {item}</p>
              <p className="mt-2 text-sm text-text">&quot;Competitor Z leads pricing queries with confident packaging.&quot;</p>
              <p className="mt-3 text-xs text-muted">Tag: Pricing · Source: ChatGPT</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
