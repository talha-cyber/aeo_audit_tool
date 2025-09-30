'use client';

import { Button, Card, CardContent, CardHeader } from '@/components/ui';

export default function ReportPreviewPage() {
  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Report preview</h2>
          <p className="text-sm text-muted">Lock structure before wiring live data sources.</p>
        </div>
        <Button size="sm" href="/reports">
          Back to library
        </Button>
      </section>

      <Card corner>
        <CardHeader title="Reader shell" description="Placeholder for PDF-to-web rendering." />
        <CardContent className="space-y-3 text-sm text-text">
          <p>Embed executive sections, insights, and callouts here.</p>
          <p>Replace this copy once reports stream from the backend.</p>
        </CardContent>
      </Card>
    </div>
  );
}
