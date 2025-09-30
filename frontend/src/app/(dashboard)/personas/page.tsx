'use client';

import { useState } from 'react';
import { Button, Card, CardContent, CardHeader, Drawer } from '@/components/ui';
import { usePersonas } from '@/lib/api/queries';
import { formatPercent } from '@/lib/utils/format';

export default function PersonasPage() {
  const { data: personas = [] } = usePersonas();
  const [activePersona, setActivePersona] = useState<string | null>(null);
  const selected = personas.find((persona) => persona.id === activePersona);

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Personas & journeys</h2>
          <p className="text-sm text-muted">Ground every audit insight in a human context.</p>
        </div>
        <Button size="sm">Add persona</Button>
      </section>

      <Card corner>
        <CardHeader title="Persona grid" description="Prioritize the voices that guide roadmaps." />
        <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {personas.map((persona) => (
            <button
              key={persona.id}
              type="button"
              onClick={() => setActivePersona(persona.id)}
              className="rounded-xl border border-border bg-elevated/40 p-5 text-left transition-colors hover:bg-elevated"
            >
              <p className="text-xs uppercase tracking-[0.18em] text-muted">{persona.priority} persona</p>
              <p className="mt-2 text-sm font-semibold text-text">{persona.name}</p>
              <p className="mt-1 text-sm text-muted">{persona.segment}</p>
              <p className="mt-3 text-sm text-text">Need: {persona.keyNeed}</p>
              <div className="mt-4 space-y-2 text-xs text-muted">
                {persona.journeyStage.map((stage) => (
                  <div key={stage.stage} className="rounded-lg border border-border bg-surface px-3 py-2">
                    <p className="text-xs font-semibold text-text">{stage.stage}</p>
                    <p className="mt-1 text-xs text-muted">{stage.question}</p>
                    <p className="mt-1 text-xs text-muted">Coverage {formatPercent(stage.coverage, 0)}</p>
                  </div>
                ))}
              </div>
            </button>
          ))}
        </CardContent>
      </Card>

      <Drawer
        title={selected?.name ?? 'Persona editor'}
        description={selected ? `${selected.segment} • ${selected.priority} priority` : 'Select a persona to edit journey coverage.'}
        open={Boolean(selected)}
        onClose={() => setActivePersona(null)}
      >
        {selected ? (
          <div className="space-y-5 text-sm text-text">
            <div className="rounded-lg border border-border bg-elevated/30 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Key need</p>
              <p className="mt-2">{selected.keyNeed}</p>
            </div>
            <div className="space-y-3">
              {selected.journeyStage.map((stage) => (
                <div key={stage.stage} className="rounded-lg border border-border bg-elevated/20 p-4">
                  <p className="text-xs uppercase tracking-[0.18em] text-muted">{stage.stage}</p>
                  <p className="mt-1 text-sm text-text">{stage.question}</p>
                  <p className="mt-2 text-xs text-muted">Coverage {formatPercent(stage.coverage, 0)}</p>
                  <Button variant="ghost" size="sm" className="mt-3">
                    Annotate insight
                  </Button>
                </div>
              ))}
            </div>
            <Button size="sm">Save edits</Button>
          </div>
        ) : (
          <p className="text-sm text-muted">Select a persona to edit journey stages.</p>
        )}
      </Drawer>
    </div>
  );
}
