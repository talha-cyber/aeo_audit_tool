'use client';

import { useState } from 'react';
import { Button, Card, CardContent, CardHeader, Tabs } from '@/components/ui';

const steps = [
  {
    id: 'scenario',
    label: 'Scenario',
    content: (
      <div className="space-y-4">
        <p className="text-sm text-muted">Choose a vertical and focus area to seed the question packs.</p>
        <ul className="space-y-2 text-sm text-text">
          <li>• Industry preset: <strong>Enterprise SaaS</strong></li>
          <li>• Audience focus: <strong>Marketing decision makers</strong></li>
          <li>• Question packs: <strong>Awareness, Evaluation, Retention</strong></li>
        </ul>
      </div>
    )
  },
  {
    id: 'platforms',
    label: 'Platforms',
    content: (
      <div className="space-y-4">
        <p className="text-sm text-muted">Toggle engines and verify API credentials.</p>
        <ul className="space-y-2 text-sm text-text">
          <li>• ChatGPT (gpt-4.1) — connected</li>
          <li>• Claude 3.5 Sonnet — connected</li>
          <li>• Perplexity Advanced — connected</li>
          <li>• Google AI Studio — pending</li>
        </ul>
      </div>
    )
  },
  {
    id: 'routing',
    label: 'Routing',
    content: (
      <div className="space-y-4">
        <p className="text-sm text-muted">Set alert recipients and guardrails.</p>
        <ul className="space-y-2 text-sm text-text">
          <li>• Owner: Strategy Ops</li>
          <li>• Escalate high-severity mentions to #aeo-war-room</li>
          <li>• Weekly digest to client portal</li>
        </ul>
      </div>
    )
  }
];

export default function AuditWizardPage() {
  const [stepIndex, setStepIndex] = useState(0);

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Audit wizard</h2>
          <p className="text-sm text-muted">Three steps to launch a new recurring audit.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setStepIndex((index) => Math.max(index - 1, 0))}>
            Back
          </Button>
          <Button size="sm" onClick={() => setStepIndex((index) => Math.min(index + 1, steps.length - 1))}>
            Next
          </Button>
        </div>
      </section>

      <Card corner>
        <CardHeader title="Guided setup" description="Each step locks copy and data requirements before styling." />
        <CardContent>
          <Tabs
            tabs={steps}
            activeTab={steps[stepIndex].id}
            onTabChange={(tabId) => {
              const idx = steps.findIndex((step) => step.id === tabId);
              if (idx >= 0) {
                setStepIndex(idx);
              }
            }}
          />
        </CardContent>
      </Card>
    </div>
  );
}
