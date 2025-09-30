'use client';

import { Button, Card, CardContent, CardHeader, EmptyState, Table } from '@/components/ui';
import { useWidgets } from '@/lib/api/queries';

export default function EmbedsPage() {
  const { data: widgets = [] } = useWidgets();

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Embeds & portal widgets</h2>
          <p className="text-sm text-muted">Package insights into shareable components.</p>
        </div>
        <Button size="sm">Create widget</Button>
      </section>

      <Card corner>
        <CardHeader title="Widget builder" description="Graybox the flows, connect data later." />
        <CardContent>
          <Table
            data={widgets}
            columns={[
              { key: 'name', header: 'Widget' },
              { key: 'preview', header: 'Preview' },
              { key: 'status', header: 'Status' }
            ]}
            emptyState={
              <EmptyState
                headline="No widgets yet"
                helper="Design the client portal building blocks here first."
                actionLabel="Create widget"
              />
            }
          />
        </CardContent>
      </Card>

      <Card corner>
        <CardHeader title="Portal configurator" description="Assemble the client-safe view." />
        <CardContent className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-border bg-elevated/30 p-4 text-sm text-text">
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Template</p>
            <p className="mt-2">Agency-standard layout with hero KPI strip and insight feed.</p>
            <Button variant="ghost" size="sm" className="mt-3">
              Adjust layout
            </Button>
          </div>
          <div className="rounded-lg border border-border bg-elevated/30 p-4 text-sm text-text">
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Access</p>
            <p className="mt-2">Invite clients and set expiration rules.</p>
            <Button variant="ghost" size="sm" className="mt-3">
              Manage access
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
