'use client';

import { Button, Card, CardContent, CardHeader, ChartFrame, Table, TableColumn } from '@/components/ui';
import { useComparisonMatrix } from '@/lib/api/queries';

export default function ComparisonsPage() {
  const { data } = useComparisonMatrix();
  const tableData: Record<string, string>[] = data
    ? data.competitors.map((competitor, index) => ({
        Competitor: competitor,
        ...Object.fromEntries(
          data.signals.map((signal) => [signal.label, `${Math.round((signal.weights[index] ?? 0) * 100)}%`])
        )
      }))
    : [];
  const columns: TableColumn<Record<string, string>>[] = data
    ? [
        { key: 'Competitor', header: 'Competitor' },
        ...data.signals.map((signal) => ({ key: signal.label, header: signal.label }))
      ]
    : [];

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Comparison matrix</h2>
          <p className="text-sm text-muted">Stack up visibility + sentiment against your competitor list.</p>
        </div>
        <Button size="sm">Export comparison</Button>
      </section>

      <Card corner>
        <CardHeader title="Signal matrix" description="Every column is a signal, every row a competitor." />
        <CardContent>
          {data ? (
            <Table data={tableData} columns={columns} />
          ) : (
            <p className="text-sm text-muted">Loading comparison data…</p>
          )}
        </CardContent>
      </Card>

      <ChartFrame title="Trend view" description="Relative momentum over the last six weeks." height="lg">
        <div className="flex h-full items-center justify-center text-sm text-muted">
          Chart placeholder with three competitor lines
        </div>
      </ChartFrame>
    </div>
  );
}
