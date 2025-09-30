import {
  AuditRun,
  AuditRunDetail,
  AuditSummary,
  ComparisonMatrix,
  Insight,
  LaunchTestRunPayload,
  Persona,
  ReportSummary,
  Settings,
  Widget
} from './schemas';

const now = new Date();

const formatDate = (date: Date) => date.toISOString();

interface MockRunRecord {
  run: AuditRun;
  startedAtMs: number;
  durationMs: number;
  totalQuestions: number;
}

const auditSummariesData: AuditSummary[] = [
  {
    id: 'audit-1',
    name: 'US Agency Benchmark',
    cadence: 'Monthly',
    owner: 'Strategy Ops',
    platforms: ['ChatGPT', 'Claude', 'Perplexity'],
    lastRun: formatDate(new Date(now.getTime() - 1000 * 60 * 60 * 20)),
    healthScore: 78
  },
  {
    id: 'audit-2',
    name: 'Feature Comparison North America',
    cadence: 'Bi-weekly',
    owner: 'Product Marketing',
    platforms: ['ChatGPT', 'Google AI'],
    lastRun: formatDate(new Date(now.getTime() - 1000 * 60 * 60 * 48)),
    healthScore: 64
  }
];

const runStore = new Map<string, MockRunRecord>();
const questionStore = new Map<string, AuditRunDetail['questions']>();

const seedQuestions = (platforms: string[]): AuditRunDetail['questions'] =>
  Array.from({ length: 8 }).map((_, index) => ({
    id: `question-${index + 1}`,
    prompt: `How does AEO rank for scenario ${index + 1}?`,
    platform: platforms[index % platforms.length] ?? platforms[0] ?? 'ChatGPT',
    sentiment: index % 3 === 0 ? 'negative' : index % 3 === 1 ? 'neutral' : 'positive',
    mentions: [
      { brand: 'Primary Brand', frequency: 2 + (index % 2), sentiment: 'positive' },
      { brand: 'Competitor X', frequency: 1, sentiment: 'negative' }
    ]
  }));

const buildQuestionsForRun = (payload: LaunchTestRunPayload): AuditRunDetail['questions'] => {
  const total = Math.min(payload.questionCount, 12);
  const sentiments: Array<'positive' | 'neutral' | 'negative'> = ['positive', 'neutral', 'negative'];

  return Array.from({ length: total }).map((_, index) => ({
    id: `sim-question-${index + 1}`,
    prompt: `Simulated prompt ${index + 1} for ${payload.scenarioId}`,
    platform: payload.platforms[index % payload.platforms.length],
    sentiment: sentiments[index % sentiments.length],
    mentions: [
      { brand: 'Primary Brand', frequency: 1 + ((index + 1) % 3), sentiment: sentiments[(index + 1) % sentiments.length] },
      { brand: 'Competitor Z', frequency: index % 2, sentiment: sentiments[(index + 2) % sentiments.length] }
    ]
  }));
};

const seedRun = (record: MockRunRecord, questions: AuditRunDetail['questions']) => {
  runStore.set(record.run.id, record);
  questionStore.set(record.run.id, questions);
};

const updateRunProgress = (record: MockRunRecord) => {
  const { run, startedAtMs, durationMs, totalQuestions } = record;

  if (run.status === 'completed') {
    return;
  }

  const elapsed = Date.now() - startedAtMs;
  const progressRatio = Math.min(elapsed / durationMs, 1);
  const computedDone = Math.max(1, Math.floor(totalQuestions * progressRatio));

  if (computedDone > run.progress.done) {
    run.progress.done = Math.min(computedDone, totalQuestions);
  }

  if (progressRatio >= 1) {
    run.status = 'completed';
    run.completedAt = formatDate(new Date(startedAtMs + durationMs));
    run.issues = run.issues.slice(0, 1);
  } else {
    run.status = 'running';
  }
};

const ensureSeedData = () => {
  if (runStore.size > 0) {
    return;
  }

  const firstRun: AuditRun = {
    id: 'run-101',
    name: 'Monthly Voice of Customer',
    status: 'running',
    startedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 32)),
    completedAt: undefined,
    progress: { done: 48, total: 120 },
    issues: [
      { id: 'iss-1', label: 'Perplexity quota nearing limit', severity: 'medium' },
      { id: 'iss-2', label: 'New competitor mention detected', severity: 'high' }
    ]
  };

  const secondRun: AuditRun = {
    id: 'run-100',
    name: 'Weekly Feature Monitoring',
    status: 'completed',
    startedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 120)),
    completedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 38)),
    progress: { done: 96, total: 96 },
    issues: []
  };

  seedRun(
    {
      run: firstRun,
      startedAtMs: new Date(firstRun.startedAt ?? formatDate(new Date())).getTime(),
      durationMs: 1000 * 60 * 60,
      totalQuestions: firstRun.progress.total
    },
    seedQuestions(['ChatGPT', 'Claude', 'Perplexity'])
  );

  seedRun(
    {
      run: secondRun,
      startedAtMs: new Date(secondRun.startedAt ?? formatDate(new Date())).getTime(),
      durationMs: 1000 * 60 * 60,
      totalQuestions: secondRun.progress.total
    },
    seedQuestions(['ChatGPT', 'Google AI'])
  );
};

const deepCloneRun = (run: AuditRun): AuditRun => ({
  ...run,
  progress: { ...run.progress },
  issues: run.issues.map((issue) => ({ ...issue }))
});

const listRuns = (): AuditRun[] => {
  ensureSeedData();
  runStore.forEach((record) => updateRunProgress(record));

  return Array.from(runStore.values())
    .map((record) => deepCloneRun(record.run))
    .sort((a, b) => {
      const aTime = a.startedAt ? new Date(a.startedAt).getTime() : 0;
      const bTime = b.startedAt ? new Date(b.startedAt).getTime() : 0;
      return bTime - aTime;
    });
};

const getRunDetail = (runId: string): AuditRunDetail => {
  ensureSeedData();
  const record = runStore.get(runId);
  if (!record) {
    const fallback = Array.from(runStore.values())[0];
    if (!fallback) {
      throw new Error('No runs available');
    }
    updateRunProgress(fallback);
    return { run: fallback.run, questions: questionStore.get(fallback.run.id) ?? seedQuestions(['ChatGPT']) };
  }

  updateRunProgress(record);
  return {
    run: deepCloneRun(record.run),
    questions: questionStore.get(runId)?.map((question) => ({
      ...question,
      mentions: question.mentions.map((mention) => ({ ...mention }))
    })) ?? seedQuestions(['ChatGPT'])
  };
};

const launchTestRun = async (payload: LaunchTestRunPayload): Promise<AuditRun> => {
  ensureSeedData();

  const timestamp = Date.now();
  const runId = `run-${timestamp}`;
  const startedAt = new Date(timestamp);
  const name = `Test audit • ${payload.scenarioId.replace(/_/g, ' ')}`;

  const newRun: AuditRun = {
    id: runId,
    name,
    status: 'running',
    startedAt: formatDate(startedAt),
    completedAt: undefined,
    progress: { done: 0, total: payload.questionCount },
    issues: []
  };

  const durationMs = Math.max(15000, payload.questionCount * 350);

  seedRun(
    {
      run: newRun,
      startedAtMs: startedAt.getTime(),
      durationMs,
      totalQuestions: payload.questionCount
    },
    buildQuestionsForRun(payload)
  );

  // update summary last run timestamp to mimic activity
  auditSummariesData[0] = {
    ...auditSummariesData[0],
    lastRun: formatDate(startedAt)
  };

  const record = runStore.get(runId)!;
  updateRunProgress(record);
  return deepCloneRun(record.run);
};

export const mockClient = {
  auditSummaries: async (): Promise<AuditSummary[]> =>
    auditSummariesData.map((summary) => ({ ...summary, platforms: [...summary.platforms] })),
  auditRuns: async (): Promise<AuditRun[]> => listRuns(),
  auditRun: async (runId: string): Promise<AuditRunDetail> => getRunDetail(runId),
  launchTestRun,
  reportSummaries: async (): Promise<ReportSummary[]> => [
    {
      id: 'report-81',
      title: 'Executive Summary — July',
      generatedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 60 * 5)),
      auditId: 'audit-1',
      coverage: { completed: 112, total: 120 }
    },
    {
      id: 'report-82',
      title: 'Competitive Deep Dive — Pricing',
      generatedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 60 * 72)),
      auditId: 'audit-2',
      coverage: { completed: 96, total: 98 }
    }
  ],

  insights: async (): Promise<Insight[]> => [
    {
      id: 'insight-1',
      title: 'Claude ranks Competitor Z first for “best onboarding”.',
      kind: 'risk',
      summary: 'Competitor Z is dominating onboarding questions with tactical examples that resonate.',
      detectedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 90)),
      impact: 'high'
    },
    {
      id: 'insight-2',
      title: 'ChatGPT surfaces our case study for enterprise comparison queries.',
      kind: 'opportunity',
      summary: 'Position this asset on the portal and embed as recommended content.',
      detectedAt: formatDate(new Date(now.getTime() - 1000 * 60 * 240)),
      impact: 'medium'
    }
  ],

  personas: async (): Promise<Persona[]> => [
    {
      id: 'persona-1',
      name: 'Growth Marketer',
      segment: 'Performance Agencies',
      priority: 'primary',
      keyNeed: 'Proves ROI of AI presence to clients.',
      journeyStage: [
        { stage: 'Discover', question: 'Who is leading AI visibility?', coverage: 0.82 },
        { stage: 'Evaluate', question: 'What keywords drive conversions?', coverage: 0.67 },
        { stage: 'Decide', question: 'Which assets to amplify?', coverage: 0.54 }
      ]
    },
    {
      id: 'persona-2',
      name: 'VP of Client Services',
      segment: 'Agency Leadership',
      priority: 'secondary',
      keyNeed: 'Keeps clients confident during QBRs.',
      journeyStage: [
        { stage: 'Discover', question: 'Where are we falling behind?', coverage: 0.48 },
        { stage: 'Evaluate', question: 'Which competitors trend up?', coverage: 0.41 }
      ]
    }
  ],

  widgets: async (): Promise<Widget[]> => [
    { id: 'widget-1', name: 'AEO Heatmap', preview: '9×9 matrix', status: 'draft' },
    { id: 'widget-2', name: 'Weekly Signal Digest', preview: 'Email embed', status: 'published' }
  ],

  comparison: async (): Promise<ComparisonMatrix> => ({
    competitors: ['Primary Brand', 'Competitor X', 'Competitor Y'],
    signals: [
      { label: 'Awareness Share', weights: [0.62, 0.21, 0.17] },
      { label: 'Sentiment Index', weights: [0.48, 0.32, 0.2] },
      { label: 'Top-of-page Presence', weights: [0.71, 0.18, 0.11] }
    ]
  }),

  settings: async (): Promise<Settings> => ({
    branding: {
      primaryColor: '#111214',
      logoUrl: undefined,
      tone: 'Measured and confident'
    },
    members: [
      { id: 'member-1', name: 'Jamie', role: 'Admin', email: 'jamie@agency.com' },
      { id: 'member-2', name: 'Morgan', role: 'Editor', email: 'morgan@agency.com' }
    ],
    billing: {
      plan: 'Agency Pro',
      renewsOn: formatDate(new Date(now.getTime() + 1000 * 60 * 60 * 24 * 30))
    },
    integrations: [
      { id: 'int-1', name: 'Slack', connected: true },
      { id: 'int-2', name: 'HubSpot', connected: false },
      { id: 'int-3', name: 'Google Drive', connected: true }
    ]
  })
};
