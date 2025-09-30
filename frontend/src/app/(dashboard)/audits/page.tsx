'use client';

import { Button, Card, CardContent, CardFooter, CardHeader, EmptyState, Table } from '@/components/ui';
import { useAuditRuns, useAuditSummaries } from '@/lib/api/queries';
import { formatRelative } from '@/lib/utils/format';

export default function AuditsPage() {
  const { data: audits = [] } = useAuditSummaries();
  const { data: runs = [] } = useAuditRuns();

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Audit programs</h2>
          <p className="text-sm text-muted">Manage scenarios, schedules, and owners for each audit.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm">
            Duplicate existing
          </Button>
          <Button size="sm" href="/audits/wizard">
            Launch wizard
          </Button>
        </div>
      </section>

      <Card corner>
        <CardHeader
          title="Audit library"
          description="Each audit bundles a question bank, platform coverage, and an execution cadence."
        />
        <CardContent>
          <Table
            data={audits}
            columns={[
              { key: 'name', header: 'Audit' },
              { key: 'owner', header: 'Owner' },
              {
                key: 'platforms',
                header: 'Platforms',
                render: (audit) => audit.platforms.join(', ')
              },
              {
                key: 'lastRun',
                header: 'Last run',
                render: (audit) => formatRelative(audit.lastRun)
              },
              {
                key: 'cadence',
                header: 'Cadence'
              }
            ]}
            emptyState={
              <EmptyState
                headline="No audits configured"
                helper="Spin up your first audit to monitor competitive visibility."
                actionLabel="Create audit"
              />
            }
          />
        </CardContent>
        <CardFooter>
          <span>{audits.length} tracked audits</span>
          <Button variant="ghost" size="sm">
            Export blueprint
          </Button>
        </CardFooter>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card corner>
          <CardHeader title="Live runs" description="Progress lanes refresh every 60 seconds." />
          <CardContent className="space-y-4">
            {runs.map((run) => {
              const progress = Math.round((run.progress.done / run.progress.total) * 100);
              return (
                <div key={run.id} className="rounded-lg border border-border bg-elevated/40 p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-text">{run.name}</p>
                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-muted">{run.status}</p>
                    </div>
                    <span className="text-sm text-muted">{run.progress.done}/{run.progress.total}</span>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-elevated">
                    <div className="h-2 rounded-full bg-text" style={{ width: `${progress}%` }} />
                  </div>
                  {run.issues.length ? (
                    <p className="mt-2 text-xs text-muted">Issues: {run.issues.map((issue) => issue.label).join(', ')}</p>
                  ) : null}
                  <div className="mt-3 text-xs">
                    <Button href={`/audits/run/${run.id}`} variant="ghost" size="sm">
                      View run detail
                    </Button>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        <Card corner>
          <CardHeader
            title="Audit wizard snapshot"
            description="Preview the three-step wizard before inviting stakeholders."
            action={<Button size="sm">Share flow</Button>}
          />
          <CardContent className="space-y-4">
            {[
              { step: '01', title: 'Scenario focus', copy: 'Select industry vertical and question packs.' },
              { step: '02', title: 'Platform mix', copy: 'Toggle engines + API credentials.' },
              { step: '03', title: 'Alert routing', copy: 'Assign stakeholders and escalation windows.' }
            ].map((item) => (
              <div key={item.step} className="rounded-lg border border-border bg-elevated/30 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-muted">Step {item.step}</p>
                <p className="mt-1 text-sm font-semibold text-text">{item.title}</p>
                <p className="mt-2 text-sm text-muted">{item.copy}</p>
              </div>
            ))}
          </CardContent>
          <CardFooter>
            <Button variant="ghost" size="sm" href="/audits/wizard">
              Resume wizard
            </Button>
            <span className="text-xs text-muted">Autosaves per team</span>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}
