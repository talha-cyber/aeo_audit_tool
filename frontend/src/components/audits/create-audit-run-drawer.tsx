'use client';

import clsx from 'clsx';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { Button, Card, CardHeader, Drawer } from '@/components/ui';
import { useCreateAuditRun, usePersonaLibrary } from '@/lib/api/queries';
import { CreateAuditRunPayload } from '@/lib/api/schemas';
import { useUIStore } from '@/store/ui';

const AVAILABLE_PLATFORMS = [
  { id: 'openai', name: 'ChatGPT' },
  { id: 'anthropic', name: 'Claude' },
  { id: 'perplexity', name: 'Perplexity' },
  { id: 'google', name: 'Google AI' }
] as const;

const QUESTION_PRESETS = [24, 48, 96, 150] as const;

export function CreateAuditRunDrawer() {
  const router = useRouter();
  const { isNewAuditDrawerOpen, closeNewAuditDrawer } = useUIStore();

  // Form state
  const [runName, setRunName] = useState('');
  const [selectedPersonaIds, setSelectedPersonaIds] = useState<string[]>([]);
  const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>(['openai']);
  const [questionCount, setQuestionCount] = useState<number>(48);

  // Data fetching
  const { data: personaLibraryData, isLoading: personasLoading } = usePersonaLibrary('b2c');
  const createAuditRun = useCreateAuditRun();

  const personas = personaLibraryData?.personas ?? [];

  const togglePersona = (personaId: string) => {
    setSelectedPersonaIds((prev) =>
      prev.includes(personaId) ? prev.filter((id) => id !== personaId) : [...prev, personaId]
    );
  };

  const togglePlatform = (platform: string) => {
    setSelectedPlatforms((prev) =>
      prev.includes(platform) ? prev.filter((p) => p !== platform) : [...prev, platform]
    );
  };

  const handleCreate = async () => {
    if (!runName.trim() || selectedPersonaIds.length === 0 || selectedPlatforms.length === 0) {
      return;
    }

    const payload: CreateAuditRunPayload = {
      name: runName.trim(),
      personaIds: selectedPersonaIds,
      platforms: selectedPlatforms,
      questionCount
    };

    try {
      const run = await createAuditRun.mutateAsync(payload);
      closeNewAuditDrawer();
      // Reset form
      setRunName('');
      setSelectedPersonaIds([]);
      setSelectedPlatforms(['openai']);
      setQuestionCount(48);
      // Navigate to run detail
      router.push(`/audits/run/${run.id}`);
    } catch (error) {
      // Error is handled by the mutation hook
      if (process.env.NODE_ENV !== 'production') {
        console.error('Failed to create audit run', error);
      }
    }
  };

  const canSubmit =
    runName.trim().length > 0 &&
    selectedPersonaIds.length > 0 &&
    selectedPlatforms.length > 0 &&
    !createAuditRun.isPending;

  return (
    <Drawer
      title="Create audit run"
      description="Select personas and platforms to generate intelligent questions for your audit."
      open={isNewAuditDrawerOpen}
      onClose={closeNewAuditDrawer}
    >
      <div className="space-y-6">
        {/* Run Name */}
        <section className="space-y-3">
          <label htmlFor="run-name" className="block text-sm font-semibold text-text">
            Run name
          </label>
          <input
            id="run-name"
            type="text"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder="e.g., Q1 2025 Audit - Enterprise SaaS"
            className="w-full rounded-lg border border-border bg-elevated px-4 py-2.5 text-sm text-text placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent"
            maxLength={255}
          />
        </section>

        {/* Persona Selection */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-text">
              Select personas ({selectedPersonaIds.length})
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push('/personas')}
              type="button"
            >
              Manage personas
            </Button>
          </div>

          {personasLoading ? (
            <p className="text-sm text-muted">Loading personas...</p>
          ) : personas.length === 0 ? (
            <Card corner>
              <CardHeader
                title="No personas found"
                description="Create personas first to use them in audit runs."
              />
              <div className="mt-4">
                <Button size="sm" onClick={() => router.push('/personas')}>
                  Go to Personas
                </Button>
              </div>
            </Card>
          ) : (
            <div className="max-h-64 space-y-2 overflow-y-auto">
              {personas.map((persona) => (
                <button
                  key={persona.id}
                  type="button"
                  onClick={() => togglePersona(persona.id)}
                  className={clsx(
                    'w-full rounded-lg border border-border bg-elevated/40 p-3 text-left transition-colors hover:bg-elevated',
                    selectedPersonaIds.includes(persona.id) ? 'ring-2 ring-accent' : ''
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-text">{persona.name}</p>
                      <p className="mt-0.5 text-xs text-muted">
                        {persona.role} • {persona.segment}
                      </p>
                    </div>
                    {selectedPersonaIds.includes(persona.id) ? (
                      <svg
                        className="h-5 w-5 text-accent"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    ) : null}
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>

        {/* Platform Selection */}
        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-text">
            Select platforms ({selectedPlatforms.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {AVAILABLE_PLATFORMS.map((platform) => (
              <Button
                key={platform.id}
                variant={selectedPlatforms.includes(platform.id) ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => togglePlatform(platform.id)}
                type="button"
              >
                {platform.name}
              </Button>
            ))}
          </div>
        </section>

        {/* Question Count */}
        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-text">Question volume</h3>
          <div className="flex flex-wrap gap-2">
            {QUESTION_PRESETS.map((preset) => (
              <Button
                key={preset}
                variant={questionCount === preset ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => setQuestionCount(preset)}
                type="button"
              >
                {preset} questions
              </Button>
            ))}
          </div>
        </section>

        {/* Error Display */}
        {createAuditRun.isError ? (
          <Card corner>
            <p className="text-sm text-error">
              Failed to create audit run:{' '}
              {createAuditRun.error instanceof Error
                ? createAuditRun.error.message
                : 'Unknown error'}
            </p>
          </Card>
        ) : null}

        {/* Submit */}
        <div className="flex items-center justify-between border-t border-border pt-4">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Next step</p>
            <p className="text-sm text-text">
              Questions will be generated and the audit will begin automatically.
            </p>
          </div>
          <Button size="sm" onClick={handleCreate} disabled={!canSubmit}>
            {createAuditRun.isPending ? 'Creating...' : 'Create & Start'}
          </Button>
        </div>
      </div>
    </Drawer>
  );
}
