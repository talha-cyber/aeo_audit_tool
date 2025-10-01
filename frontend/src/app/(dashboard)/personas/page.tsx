'use client';

import clsx from 'clsx';
import { useCallback, useMemo, useState } from 'react';
import { Button, Card, CardContent, CardHeader, Drawer } from '@/components/ui';
import {
  useClonePersona,
  useCreatePersona,
  useDeletePersona,
  usePersonaCatalog,
  usePersonaLibrary,
  usePersonas,
  useUpdatePersona
} from '@/lib/api/queries';
import type {
  Persona,
  PersonaCatalogVoice,
  PersonaLibraryEntry,
  PersonaMode
} from '@/lib/api/schemas';
import { formatPercent } from '@/lib/utils/format';

const PERSONA_MODES: PersonaMode[] = ['b2c', 'b2b'];

type PersonaDraft = {
  voice: string | null;
  role: string | null;
  driver: string | null;
  contexts: string[];
  name: string;
  segment: string;
  priority: 'primary' | 'secondary';
  keyNeed: string;
  journeyStage: Persona['journeyStage'];
};

const createEmptyDraft = (segmentHint: string): PersonaDraft => ({
  voice: null,
  role: null,
  driver: null,
  contexts: [],
  name: '',
  segment: segmentHint,
  priority: 'secondary',
  keyNeed: '',
  journeyStage: []
});

export default function PersonasPage() {
  const [mode, setMode] = useState<PersonaMode>('b2c');

  const personasQuery = usePersonas(mode);
  const personaLibraryQuery = usePersonaLibrary(mode);
  const catalogQuery = usePersonaCatalog(mode);

  const createPersona = useCreatePersona();
  const updatePersona = useUpdatePersona();
  const clonePersona = useClonePersona();
  const deletePersona = useDeletePersona();

  const [activePersonaId, setActivePersonaId] = useState<string | null>(null);
  const [isComposeOpen, setComposeOpen] = useState(false);
  const [editingPersonaId, setEditingPersonaId] = useState<string | null>(null);
  const [draft, setDraft] = useState<PersonaDraft>(createEmptyDraft(mode.toUpperCase()));

  const personas = personasQuery.data ?? [];
  const personaLibrary = personaLibraryQuery.data?.personas ?? [];
  const catalog = catalogQuery.data;
  const catalogLoading = catalogQuery.isLoading;
  const catalogError = catalogQuery.error as Error | null;

  const isEditing = Boolean(editingPersonaId);
  const selectedPersona = personas.find((persona) => persona.id === activePersonaId) ?? null;
  const segmentHint = useMemo(() => mode.toUpperCase(), [mode]);

  const roleLookup = useMemo(() => new Map(catalog?.roles.map((role) => [role.key, role]) ?? []), [catalog]);
  const driverLookup = useMemo(
    () => new Map(catalog?.drivers.map((driver) => [driver.key, driver]) ?? []),
    [catalog]
  );
  const contextLookup = useMemo(
    () => new Map(catalog?.contexts.map((context) => [context.key, context]) ?? []),
    [catalog]
  );

  const contextLabel = useCallback(
    (key: string) => contextLookup.get(key)?.label ?? key,
    [contextLookup]
  );

  const buildJourneyStage = useCallback(
    (contextKeys: string[], base?: Persona['journeyStage']) => {
      if (!contextKeys.length) {
        return [];
      }
      const labels = contextKeys.map(contextLabel);
      const coverage = Number((labels.length ? 1 / labels.length : 0).toFixed(3));
      return contextKeys.map((key, index) => {
        const label = labels[index];
        const existing = base?.find((stage) => stage.stage === label);
        return {
          stage: label,
          question: existing?.question ?? '',
          coverage: existing?.coverage ?? coverage
        };
      });
    },
    [contextLabel]
  );

  const resetDraft = useCallback(() => {
    setDraft(createEmptyDraft(segmentHint));
    setEditingPersonaId(null);
  }, [segmentHint]);

  const applyVoicePreset = useCallback(
    (voice: PersonaCatalogVoice) => {
      const contexts = [...voice.contexts];
      const journeyStage = buildJourneyStage(contexts);
      const roleLabel = roleLookup.get(voice.role)?.label ?? voice.role;
      const driverLabel = driverLookup.get(voice.driver)?.label ?? voice.driver;

      setDraft((prev) => ({
        ...prev,
        voice: voice.key,
        role: voice.role,
        driver: voice.driver,
        contexts,
        name: prev.name || roleLabel,
        segment: prev.segment || segmentHint,
        keyNeed: prev.keyNeed || driverLabel,
        journeyStage
      }));
    },
    [buildJourneyStage, driverLookup, roleLookup, segmentHint]
  );

  const prepareDraftFromEntry = useCallback(
    (entry: PersonaLibraryEntry) => {
      setDraft({
        voice: entry.voice ?? null,
        role: entry.role,
        driver: entry.driver,
        contexts: [...entry.contextKeys],
        name: entry.name,
        segment: entry.segment,
        priority: entry.priority,
        keyNeed: entry.keyNeed,
        journeyStage: entry.journeyStage
      });
    },
    []
  );

  const handleModeChange = (nextMode: PersonaMode) => {
    if (nextMode === mode) {
      return;
    }
    setMode(nextMode);
    setActivePersonaId(null);
    setComposeOpen(false);
    setDraft(createEmptyDraft(nextMode.toUpperCase()));
    setEditingPersonaId(null);
  };

  const toggleContext = (contextKey: string) => {
    setDraft((prev) => {
      const exists = prev.contexts.includes(contextKey);
      if (exists && prev.contexts.length === 1) {
        return prev;
      }
      const contexts = exists
        ? prev.contexts.filter((key) => key !== contextKey)
        : [...prev.contexts, contextKey];
      return {
        ...prev,
        contexts,
        journeyStage: buildJourneyStage(contexts, prev.journeyStage)
      };
    });
  };

  const openComposeDrawer = (entry?: PersonaLibraryEntry) => {
    if (entry) {
      setEditingPersonaId(entry.id);
      prepareDraftFromEntry(entry);
    } else {
      resetDraft();
      if (catalog?.voices.length) {
        applyVoicePreset(catalog.voices[0]);
      }
    }
    setComposeOpen(true);
  };

  const handleCloseCompose = () => {
    setComposeOpen(false);
    resetDraft();
  };

  const canCompose = Boolean(draft.role && draft.driver && draft.contexts.length && draft.name.trim());

  const submitCompose = async () => {
    if (!canCompose) {
      return;
    }

    const payload = {
      mode,
      voice: draft.voice ?? undefined,
      role: draft.role ?? undefined,
      driver: draft.driver ?? undefined,
      contexts: draft.contexts,
      name: draft.name,
      segment: draft.segment || segmentHint,
      priority: draft.priority,
      keyNeed: draft.keyNeed,
      journeyStage: draft.journeyStage
    } as const;

    try {
      if (isEditing && editingPersonaId) {
        await updatePersona.mutateAsync({ personaId: editingPersonaId, payload });
      } else {
        await createPersona.mutateAsync(payload);
      }
      handleCloseCompose();
    } catch (error) {
      if (process.env.NODE_ENV !== 'production') {
        // eslint-disable-next-line no-console
        console.error('Failed to save persona', error);
      }
    }
  };

  const handleClonePersona = async (entry: PersonaLibraryEntry) => {
    try {
      await clonePersona.mutateAsync({
        personaId: entry.id,
        payload: { mode, name: `${entry.name} Copy` }
      });
    } catch (error) {
      if (process.env.NODE_ENV !== 'production') {
        // eslint-disable-next-line no-console
        console.error('Failed to clone persona', error);
      }
    }
  };

  const handleDeletePersona = async (entry: PersonaLibraryEntry) => {
    const confirmed = window.confirm(`Delete persona "${entry.name}"?`);
    if (!confirmed) {
      return;
    }
    try {
      await deletePersona.mutateAsync({ personaId: entry.id, mode });
    } catch (error) {
      if (process.env.NODE_ENV !== 'production') {
        // eslint-disable-next-line no-console
        console.error('Failed to delete persona', error);
      }
    }
  };

  const draftRoleLabel = draft.role ? roleLookup.get(draft.role)?.label ?? draft.role : 'Select a role';
  const draftDriverLabel = draft.driver
    ? driverLookup.get(draft.driver)?.label ?? draft.driver
    : 'Select a driver';
  const draftContexts = draft.contexts.map(contextLabel);
  const previewCoverage = draftContexts.length ? 1 / draftContexts.length : 0;

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-text">Personas & journeys</h2>
          <p className="text-sm text-muted">Ground every audit insight in a human context.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex gap-2">
            {PERSONA_MODES.map((item) => (
              <Button
                key={item}
                variant={mode === item ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => handleModeChange(item)}
              >
                {item.toUpperCase()}
              </Button>
            ))}
          </div>
          <Button size="sm" onClick={() => openComposeDrawer()} disabled={catalogLoading}>
            {catalogLoading ? 'Loading presets…' : 'Add persona'}
          </Button>
        </div>
      </section>

      <Card corner>
        <CardHeader
          title="Persona library"
          description="Saved compositions stay here for cloning or quick edits."
        />
        <CardContent className="space-y-3">
          {personaLibrary.length ? (
            personaLibrary.map((entry) => (
              <div
                key={entry.id}
                className="flex flex-col gap-3 rounded-xl border border-border bg-elevated/40 p-4 transition-colors hover:bg-elevated md:flex-row md:items-center md:justify-between"
              >
                <div>
                  <p className="text-sm font-semibold text-text">{entry.name}</p>
                  <p className="text-xs uppercase tracking-[0.18em] text-muted">
                    {entry.priority} • {entry.mode.toUpperCase()} persona
                  </p>
                  <p className="mt-2 text-xs text-muted">Need: {entry.keyNeed}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button size="xs" variant="ghost" onClick={() => openComposeDrawer(entry)}>
                    Edit
                  </Button>
                  <Button size="xs" variant="ghost" onClick={() => handleClonePersona(entry)}>
                    Clone
                  </Button>
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => handleDeletePersona(entry)}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted">Compose a persona to start your reusable library.</p>
          )}
        </CardContent>
      </Card>

      <Card corner>
        <CardHeader title="Persona grid" description="Prioritize the voices that guide roadmaps." />
        <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {personas.map((persona) => (
            <button
              key={persona.id}
              type="button"
              onClick={() => setActivePersonaId(persona.id)}
              className="rounded-xl border border-border bg-elevated/40 p-5 text-left transition-colors hover:bg-elevated"
            >
              <p className="text-xs uppercase tracking-[0.18em] text-muted">{persona.priority} persona</p>
              <p className="mt-2 text-sm font-semibold text-text">{persona.name}</p>
              <p className="mt-1 text-sm text-muted">{persona.segment}</p>
              <p className="mt-3 text-sm text-text">Need: {persona.keyNeed}</p>
              <div className="mt-4 space-y-2 text-xs text-muted">
                {persona.journeyStage.map((stage) => (
                  <div key={`${persona.id}-${stage.stage}`} className="rounded-lg border border-border bg-surface px-3 py-2">
                    <p className="text-xs font-semibold text-text">{stage.stage}</p>
                    <p className="mt-1 text-xs text-muted">{stage.question}</p>
                    <p className="mt-1 text-xs text-muted">Coverage {formatPercent(stage.coverage, 0)}</p>
                  </div>
                ))}
              </div>
            </button>
          ))}
          {!personas.length ? (
            <p className="text-sm text-muted">No personas resolved for this mode yet. Create one to get started.</p>
          ) : null}
        </CardContent>
      </Card>

      <Drawer
        title={selectedPersona?.name ?? 'Persona details'}
        description={
          selectedPersona
            ? `${selectedPersona.segment} • ${selectedPersona.priority} priority`
            : 'Select a persona to inspect context coverage.'
        }
        open={Boolean(selectedPersona)}
        onClose={() => setActivePersonaId(null)}
      >
        {selectedPersona ? (
          <div className="space-y-5 text-sm text-text">
            <div className="rounded-lg border border-border bg-elevated/30 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Key need</p>
              <p className="mt-2">{selectedPersona.keyNeed}</p>
            </div>
            <div className="space-y-3">
              {selectedPersona.journeyStage.map((stage) => (
                <div key={`${selectedPersona.id}-${stage.stage}`} className="rounded-lg border border-border bg-elevated/20 p-4">
                  <p className="text-xs uppercase tracking-[0.18em] text-muted">{stage.stage}</p>
                  <p className="mt-1 text-sm text-text">{stage.question || 'Surface insights to enrich this stage.'}</p>
                  <p className="mt-2 text-xs text-muted">Coverage {formatPercent(stage.coverage, 0)}</p>
                  <Button variant="ghost" size="sm" className="mt-3">
                    Annotate insight
                  </Button>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted">Select a persona to inspect journey stages.</p>
        )}
      </Drawer>

      <Drawer
        title={isEditing ? 'Edit persona' : 'Compose persona'}
        description={
          isEditing
            ? 'Update the role, driver, or contexts and save your library entry.'
            : 'Start from a preset voice and fine-tune the journey contexts.'
        }
        open={isComposeOpen}
        onClose={handleCloseCompose}
      >
        {catalogError ? (
          <p className="text-sm text-[#b34242]">Failed to load persona presets: {catalogError.message}</p>
        ) : null}

        {catalogLoading && !catalog ? (
          <p className="text-sm text-muted">Loading persona presets…</p>
        ) : null}

        {catalog ? (
          <div className="space-y-6 text-sm text-text">
            <section className="space-y-3">
              <h3 className="text-sm font-semibold text-text">Persona metadata</h3>
              <div className="space-y-2">
                <input
                  className="w-full rounded-md border border-border bg-elevated/40 px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
                  placeholder="Persona name"
                  value={draft.name}
                  onChange={(event) => setDraft((prev) => ({ ...prev, name: event.target.value }))}
                />
                <input
                  className="w-full rounded-md border border-border bg-elevated/40 px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
                  placeholder="Key need"
                  value={draft.keyNeed}
                  onChange={(event) => setDraft((prev) => ({ ...prev, keyNeed: event.target.value }))}
                />
              </div>
              <div className="flex gap-2">
                {(['primary', 'secondary'] as Array<'primary' | 'secondary'>).map((priority) => (
                  <Button
                    key={priority}
                    variant={draft.priority === priority ? 'primary' : 'ghost'}
                    size="xs"
                    onClick={() => setDraft((prev) => ({ ...prev, priority }))}
                  >
                    {priority.charAt(0).toUpperCase() + priority.slice(1)}
                  </Button>
                ))}
              </div>
            </section>

            <section className="space-y-3">
              <h3 className="text-sm font-semibold text-text">Persona presets</h3>
              <div className="space-y-2">
                {catalog.voices.map((voice) => {
                  const role = roleLookup.get(voice.role);
                  const driver = driverLookup.get(voice.driver);
                  const isActive = draft.voice === voice.key;
                  return (
                    <button
                      key={voice.key}
                      type="button"
                      onClick={() => applyVoicePreset(voice)}
                      className={clsx(
                        'w-full rounded-lg border border-border bg-elevated/40 p-4 text-left transition-colors hover:bg-elevated',
                        isActive ? 'ring-2 ring-accent' : ''
                      )}
                    >
                      <p className="text-sm font-semibold text-text">{role?.label ?? voice.key}</p>
                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-muted">
                        {driver?.label ?? voice.driver}
                      </p>
                      <p className="mt-2 text-xs text-muted">
                        Contexts: {voice.contexts.map(contextLabel).join(', ')}
                      </p>
                    </button>
                  );
                })}
                {!catalog.voices.length ? (
                  <p className="text-sm text-muted">No presets available for this mode.</p>
                ) : null}
              </div>
            </section>

            <section className="space-y-3">
              <h3 className="text-sm font-semibold text-text">Role</h3>
              <div className="flex flex-wrap gap-2">
                {catalog.roles.map((role) => (
                  <Button
                    key={role.key}
                    variant={draft.role === role.key ? 'primary' : 'ghost'}
                    size="sm"
                    onClick={() =>
                      setDraft((prev) => ({
                        ...prev,
                        role: role.key,
                        name: prev.name || role.label
                      }))
                    }
                  >
                    {role.label}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted">
                {draft.role
                  ? roleLookup.get(draft.role)?.description ?? 'Define the day-in-the-life voice.'
                  : 'Select which customer role this persona should represent.'}
              </p>
            </section>

            <section className="space-y-3">
              <h3 className="text-sm font-semibold text-text">Motivation</h3>
              <div className="flex flex-wrap gap-2">
                {catalog.drivers.map((driver) => (
                  <Button
                    key={driver.key}
                    variant={draft.driver === driver.key ? 'primary' : 'ghost'}
                    size="sm"
                    onClick={() =>
                      setDraft((prev) => ({
                        ...prev,
                        driver: driver.key,
                        keyNeed: prev.keyNeed || driver.label
                      }))
                    }
                  >
                    {driver.label}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted">
                {draft.driver
                  ? driverLookup.get(draft.driver)?.emotional_anchor ?? 'Map the emotional anchor to tailor messaging.'
                  : 'Choose the core driver that shapes their decisions.'}
              </p>
            </section>

            <section className="space-y-3">
              <h3 className="text-sm font-semibold text-text">Journey contexts</h3>
              <div className="flex flex-wrap gap-2">
                {catalog.contexts.map((context) => {
                  const isSelected = draft.contexts.includes(context.key);
                  return (
                    <Button
                      key={context.key}
                      variant={isSelected ? 'primary' : 'ghost'}
                      size="sm"
                      onClick={() => toggleContext(context.key)}
                    >
                      {context.label}
                    </Button>
                  );
                })}
              </div>
              <p className="text-xs text-muted">
                Keep at least one journey stage selected to anchor coverage weighting.
              </p>
            </section>

            <section className="rounded-lg border border-border bg-elevated/20 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Preview</p>
              <p className="mt-2 text-sm font-semibold text-text">{draftRoleLabel}</p>
              <p className="mt-1 text-xs text-muted">Need: {draftDriverLabel}</p>
              <div className="mt-3 space-y-2">
                {draftContexts.length ? (
                  draftContexts.map((context) => (
                    <div key={context} className="rounded-md border border-border bg-surface px-3 py-2 text-xs text-muted">
                      <p className="text-xs font-semibold text-text">{context}</p>
                      <p className="mt-1 text-xs text-muted">Coverage {formatPercent(previewCoverage, 0)}</p>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-muted">Select at least one journey context to preview coverage.</p>
                )}
              </div>
            </section>

            {(createPersona.isError || updatePersona.isError) && (
              <p className="text-xs text-[#b34242]">
                Unable to {isEditing ? 'save' : 'compose'} persona. Check your selections and try again.
              </p>
            )}

            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-muted">Outcome</p>
                <p className="text-sm text-text">
                  {isEditing
                    ? 'Updates the stored persona and refreshes your library.'
                    : 'Generates a dashboard persona using the selected presets.'}
                </p>
              </div>
              <Button size="sm" onClick={submitCompose} disabled={!canCompose || createPersona.isPending || updatePersona.isPending}>
                {createPersona.isPending || updatePersona.isPending
                  ? 'Saving…'
                  : isEditing
                  ? 'Save changes'
                  : 'Create persona'}
              </Button>
            </div>
          </div>
        ) : null}
      </Drawer>
    </div>
  );
}
