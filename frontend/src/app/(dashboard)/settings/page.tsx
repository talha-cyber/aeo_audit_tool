'use client';

import { Button, Card, CardContent, CardHeader, Table, Tabs } from '@/components/ui';
import { useSettings } from '@/lib/api/queries';

export default function SettingsPage() {
  const { data } = useSettings();

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Settings</h2>
          <p className="text-sm text-muted">Branding, team, billing, and integrations in one place.</p>
        </div>
        <Button variant="ghost" size="sm">
          View audit log
        </Button>
      </section>

      <Card corner>
        <CardHeader title="Control panel" description="Each tab maps directly to a backend contract." />
        <CardContent>
          {data ? (
            <Tabs
              tabs={[
                {
                  id: 'branding',
                  label: 'Branding',
                  content: (
                    <div className="space-y-3 text-sm text-text">
                      <p>Primary color: {data.branding.primaryColor}</p>
                      <p>Voice & tone: {data.branding.tone}</p>
                      <Button variant="ghost" size="sm">
                        Upload logo
                      </Button>
                    </div>
                  )
                },
                {
                  id: 'members',
                  label: 'Members',
                  content: (
                    <Table
                      data={data.members}
                      columns={[
                        { key: 'name', header: 'Name' },
                        { key: 'role', header: 'Role' },
                        { key: 'email', header: 'Email' }
                      ]}
                    />
                  )
                },
                {
                  id: 'billing',
                  label: 'Billing',
                  content: (
                    <div className="space-y-3 text-sm text-text">
                      <p>Plan: {data.billing.plan}</p>
                      <p>Renews on: {new Date(data.billing.renewsOn).toLocaleDateString()}</p>
                      <Button size="sm">Update payment method</Button>
                    </div>
                  )
                },
                {
                  id: 'integrations',
                  label: 'Integrations',
                  content: (
                    <Table
                      data={data.integrations}
                      columns={[
                        { key: 'name', header: 'Integration' },
                        {
                          key: 'connected',
                          header: 'Status',
                          render: (integration) => (integration.connected ? 'Connected' : 'Pending')
                        }
                      ]}
                    />
                  )
                }
              ]}
            />
          ) : (
            <p className="text-sm text-muted">Loading settings…</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
