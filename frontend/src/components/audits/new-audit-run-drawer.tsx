'use client';

import clsx from 'clsx';
import { useRouter } from 'next/navigation';
import { useMemo, useState } from 'react';
import { Button, Card, CardHeader, Drawer } from '@/components/ui';
import { useLaunchTestRun } from '@/lib/api/queries';
import { LaunchTestRunPayload } from '@/lib/api/schemas';
import { useUIStore } from '@/store/ui';

const SCENARIOS = [
  {
    id: 'saas_weekly',
    title: 'Enterprise SaaS pulse',
    summary: 'Monitors awareness + evaluation questions for North America.',
    platforms: ['ChatGPT', 'Claude', 'Perplexity'],
    defaultQuestionCount: 48
  },
  {
    id: 'ecom_flash',
    title: 'Retail flash audit',
    summary: 'Short-burst audit tuned for seasonal campaigns.',
    platforms: ['ChatGPT', 'Google AI'],
    defaultQuestionCount: 32
  },
  {
    id: 'global_watch',
    title: 'Global sentiment watch',
    summary: 'Tracks multilingual sentiment shifts with weekly digests.',
    platforms: ['ChatGPT', 'Claude', 'Perplexity', 'Google AI'],
    defaultQuestionCount: 64
  }
] as const;

const QUESTION_PRESETS = [24, 48, 96] as const;

export function NewAuditRunDrawer() {
  const router = useRouter();
  const { isNewAuditDrawerOpen, closeNewAuditDrawer } = useUIStore();
  const [selectedScenario, setSelectedScenario] = useState<typeof SCENARIOS[number] | null>(SCENARIOS[0]);
  const [questionCount, setQuestionCount] = useState<number>(SCENARIOS[0].defaultQuestionCount);
  const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>(SCENARIOS[0].platforms);
  const launchTestRun = useLaunchTestRun();

  const isUsingMocks = useMemo(() => process.env.NEXT_PUBLIC_USE_MOCKS !== 'false', []);

  const togglePlatform = (platform: string) => {
    setSelectedPlatforms((prev) =>
      prev.includes(platform) ? prev.filter((item) => item !== platform) : [...prev, platform]
    );
  };

  const handleScenarioChange = (scenarioId: string) => {
    const scenario = SCENARIOS.find((item) => item.id === scenarioId) ?? null;
    setSelectedScenario(scenario);
    if (scenario) {
      setQuestionCount(scenario.defaultQuestionCount);
      setSelectedPlatforms(scenario.platforms);
    }
  };

  const handleLaunch = async () => {
    if (!selectedScenario || !isUsingMocks) {
      return;
    }

    if (!selectedPlatforms.length) {
      return;
    }

    const payload: LaunchTestRunPayload = {
      scenarioId: selectedScenario.id,
      questionCount,
      platforms: selectedPlatforms
    };

    try {
      const run = await launchTestRun.mutateAsync(payload);
      closeNewAuditDrawer();
      router.push(`/audits/run/${run.id}`);
    } catch (error) {
      // The hook surfaces the error state which we render below; log for local debugging.
      if (process.env.NODE_ENV !== 'production') {
        // eslint-disable-next-line no-console
        console.error('Failed to launch test audit run', error);
      }
    }
  };

  return (
    <Drawer
      title="Launch test audit"
      description="Spin up a simulated audit run to exercise the end-to-end flow."
      open={isNewAuditDrawerOpen}
      onClose={closeNewAuditDrawer}
    >
      <div className="space-y-6">
        {!isUsingMocks ? (
          <Card corner>
            <CardHeader
              title="Live API mode"
              description="Test runs require the mock data layer. Flip NEXT_PUBLIC_USE_MOCKS=true for the guided simulation."
            />
          </Card>
        ) : null}

        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-text">Scenario template</h3>
          <div className="space-y-3">
            {SCENARIOS.map((scenario) => (
              <button
                key={scenario.id}
                type="button"
                className={clsx(
                  'w-full rounded-lg border border-border bg-elevated/40 p-4 text-left transition-colors hover:bg-elevated',
                  selectedScenario?.id === scenario.id ? 'ring-2 ring-accent' : ''
                )}
                onClick={() => handleScenarioChange(scenario.id)}
              >
                <p className="text-sm font-semibold text-text">{scenario.title}</p>
                <p className="mt-1 text-sm text-muted">{scenario.summary}</p>
                <p className="mt-2 text-xs uppercase tracking-[0.18em] text-muted">
                  Suggested platforms: {scenario.platforms.join(', ')}
                </p>
              </button>
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-text">Question volume</h3>
          <div className="flex flex-wrap gap-2">
            {QUESTION_PRESETS.map((preset) => (
              <Button
                key={preset}
                variant={questionCount === preset ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => setQuestionCount(preset)}
              >
                {preset} prompts
              </Button>
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-text">Platforms</h3>
          <div className="flex flex-wrap gap-2">
            {['ChatGPT', 'Claude', 'Perplexity', 'Google AI'].map((platform) => (
              <Button
                key={platform}
                variant={selectedPlatforms.includes(platform) ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => togglePlatform(platform)}
              >
                {platform}
              </Button>
            ))}
          </div>
        </section>

        {launchTestRun.isError ? (
          <p className="text-sm text-[#b34242]">
            Failed to start the test run: {launchTestRun.error instanceof Error ? launchTestRun.error.message : 'Unknown error'}
          </p>
        ) : null}

        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Outcome</p>
            <p className="text-sm text-text">Creates a mock audit run and redirects to the progress view.</p>
          </div>
          <Button
            size="sm"
            onClick={handleLaunch}
            disabled={
              launchTestRun.isPending || !selectedPlatforms.length || !isUsingMocks
            }
          >
            {launchTestRun.isPending ? 'Launching…' : 'Launch test run'}
          </Button>
        </div>
      </div>
    </Drawer>
  );
}
