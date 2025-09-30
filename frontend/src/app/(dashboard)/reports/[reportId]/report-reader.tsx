'use client';

import { Button, Card, CardContent, CardHeader, ChartFrame } from '@/components/ui';
import { useReportSummaries } from '@/lib/api/queries';

interface ReportReaderProps {
  reportId: string;
}

export function ReportReader({ reportId }: ReportReaderProps) {
  const { data: reports = [] } = useReportSummaries();
  const report = reports.find((candidate) => candidate.id === reportId) ?? reports[0];

  if (!report) {
    return <p className="text-sm text-muted">No report found yet. Generate one from the library.</p>;
  }

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-muted">Report reader</p>
          <h2 className="text-xl font-semibold text-text">{report.title}</h2>
          <p className="text-sm text-muted">Coverage {report.coverage.completed}/{report.coverage.total} • Audit {report.auditId}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm">
            Share link
          </Button>
          <Button size="sm">Download PDF</Button>
        </div>
      </section>

      <Card corner>
        <CardHeader title="Executive summary" description="Lock copy, wire data later." />
        <CardContent className="space-y-3 text-sm text-text">
          <p>Visibility is trending up 6% WoW with Claude surfacing us first for onboarding queries.</p>
          <p>Competitive threat: Competitor Z dominates &quot;AI onboarding&quot; with playbooks that resonate with agencies.</p>
          <p>Recommendation: Update client portal tile with the Q3 case study and push weekly digest.</p>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartFrame title="Trend line" description="Share of first mention by engine." height="md">
          <div className="flex h-full items-center justify-center text-sm text-muted">Chart placeholder</div>
        </ChartFrame>
        <ChartFrame title="Sentiment delta" description="Movement vs prior period." height="md">
          <div className="flex h-full items-center justify-center text-sm text-muted">Chart placeholder</div>
        </ChartFrame>
      </div>

      <Card corner>
        <CardHeader title="Evidence" description="Link out to transcripts, screen captures, and callouts." />
        <CardContent className="space-y-4">
          {[1, 2, 3].map((index) => (
            <div key={index} className="rounded-lg border border-border bg-elevated/30 p-4 text-sm text-text">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Callout {index}</p>
              <p className="mt-2">Snippet from ChatGPT highlighting our comparison grid.</p>
              <Button variant="ghost" size="sm" className="mt-3">
                Open transcript
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
