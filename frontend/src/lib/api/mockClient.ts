import {
  AuditRun,
  AuditRunDetail,
  AuditSummary,
  ComparisonMatrix,
  Insight,
  LaunchTestRunPayload,
  Persona,
  PersonaCatalog,
  PersonaComposePayload,
  PersonaLibraryEntry,
  PersonaLibraryResponse,
  PersonaUpdatePayload,
  PersonaClonePayload,
  PersonaMode,
  ReportSummary,
  Settings,
  Widget
} from './schemas';

const now = new Date();

const formatDate = (date: Date) => date.toISOString();

const personaCatalogStore: Record<PersonaMode, PersonaCatalog> = {
  b2c: {
    mode: 'b2c',
    roles: [
      {
        key: 'deal_hunter',
        label: 'Deal Hunter',
        description: 'Tracks promotions and bundles before purchasing.',
        defaultContextStage: 'consider'
      },
      {
        key: 'loyalist',
        label: 'Loyal Customer',
        description: 'Advocates for brands that deliver consistent service.',
        defaultContextStage: 'retain'
      }
    ],
    drivers: [
      {
        key: 'value',
        label: 'Value & Savings',
        emotionalAnchor: 'Avoid missing limited-time offers.',
        weight: 1
      },
      {
        key: 'experience',
        label: 'Experience & Ease',
        emotionalAnchor: 'Wants a frictionless checkout.',
        weight: 0.8
      }
    ],
    contexts: [
      { key: 'discover', label: 'Discover', priority: 0.7 },
      { key: 'consider', label: 'Consider', priority: 0.9 },
      { key: 'retain', label: 'Retain', priority: 0.6 }
    ],
    voices: [
      { key: 'budget_hawk', role: 'deal_hunter', driver: 'value', contexts: ['discover', 'consider'] },
      { key: 'experience_enthusiast', role: 'loyalist', driver: 'experience', contexts: ['consider', 'retain'] }
    ]
  },
  b2b: {
    mode: 'b2b',
    roles: [
      {
        key: 'cfo',
        label: 'Chief Financial Officer',
        description: 'Economic buyer focused on ROI and runway.',
        defaultContextStage: 'validation'
      },
      {
        key: 'it_director',
        label: 'Director of IT',
        description: 'Owns integration and security diligence.',
        defaultContextStage: 'evaluation'
      }
    ],
    drivers: [
      {
        key: 'savings',
        label: 'Cost Optimization',
        emotionalAnchor: 'Avoid unplanned spend.',
        weight: 1.1
      },
      {
        key: 'resilience',
        label: 'Reliability',
        emotionalAnchor: 'Seeks zero downtime commitments.',
        weight: 1
      }
    ],
    contexts: [
      { key: 'evaluation', label: 'Evaluation', priority: 0.8 },
      { key: 'validation', label: 'Validation', priority: 1.0 },
      { key: 'rollout', label: 'Rollout', priority: 0.7 }
    ],
    voices: [
      { key: 'roi_guardian', role: 'cfo', driver: 'savings', contexts: ['validation', 'rollout'] },
      { key: 'integration_anchor', role: 'it_director', driver: 'resilience', contexts: ['evaluation', 'rollout'] }
    ]
  }
};

const defaultPersonaStore: Record<PersonaMode, Persona[]> = {
  b2c: [
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
  b2b: [
    {
      id: 'persona-b2b-1',
      name: 'Security Leader',
      segment: 'Enterprise Risk',
      priority: 'primary',
      keyNeed: 'Quantifies exposure before sign-off.',
      journeyStage: [
        { stage: 'Evaluation', question: 'Can we meet compliance baselines?', coverage: 0.5 },
        { stage: 'Validation', question: 'Where are residual gaps?', coverage: 0.5 }
      ]
    },
    {
      id: 'persona-b2b-2',
      name: 'Revenue Ops Lead',
      segment: 'B2B SaaS',
      priority: 'secondary',
      keyNeed: 'Keeps the GTM engine informed.',
      journeyStage: [
        { stage: 'Evaluation', question: 'How do we benchmark messaging?', coverage: 0.5 },
        { stage: 'Rollout', question: 'Which assets need enablement?', coverage: 0.5 }
      ]
    }
  ]
};

const customPersonaStore: Record<string, Record<PersonaMode, PersonaLibraryEntry[]>> = {};

const getPersonaCatalog = (mode: PersonaMode): PersonaCatalog =>
  personaCatalogStore[mode] ?? personaCatalogStore.b2c;

const ensurePersonaBucket = (ownerId: string, mode: PersonaMode): PersonaLibraryEntry[] => {
  const ownerStore = (customPersonaStore[ownerId] ??= {
    b2c: [],
    b2b: []
  } as Record<PersonaMode, PersonaLibraryEntry[]>);
  return ownerStore[mode];
};

const normalizeJourneyStage = (
  contextLabels: string[],
  presetCoverage: number,
  override?: Persona['journeyStage']
) => {
  if (override && override.length === contextLabels.length) {
    return override.map((stage, index) => ({
      stage: stage.stage,
      question: stage.question ?? '',
      coverage: stage.coverage ?? presetCoverage ?? 0
    }));
  }

  return contextLabels.map((label) => ({
    stage: label,
    question: '',
    coverage: presetCoverage
  }));
};

const composePersonaFromPayload = (
  payload: PersonaComposePayload | PersonaUpdatePayload,
  existing?: PersonaLibraryEntry
): PersonaLibraryEntry => {
  const mode = payload.mode ?? existing?.mode ?? 'b2c';
  const ownerId = payload.ownerId ?? existing?.ownerId ?? 'global';
  const catalog = getPersonaCatalog(mode);

  const voice = payload.voice ?? existing?.voice ?? null;
  const voicePreset = voice
    ? catalog.voices.find((item) => item.key === voice)
    : undefined;

  const roleKey = payload.role ?? existing?.role ?? voicePreset?.role ?? catalog.roles[0]?.key;
  const driverKey = payload.driver ?? existing?.driver ?? voicePreset?.driver ?? catalog.drivers[0]?.key;
  const contextKeys = payload.contexts ?? existing?.contextKeys ?? voicePreset?.contexts ?? [catalog.contexts[0]?.key].filter(Boolean);

  if (!roleKey || !driverKey || !contextKeys?.length) {
    throw new Error('Unable to compose persona with provided data');
  }

  const roleLabel = catalog.roles.find((role) => role.key === roleKey)?.label ?? roleKey;
  const driverLabel = catalog.drivers.find((driver) => driver.key === driverKey)?.label ?? driverKey;
  const contextLabels = contextKeys.map(
    (key) => catalog.contexts.find((context) => context.key === key)?.label ?? key
  );

  const coverage = contextLabels.length
    ? Number((1 / contextLabels.length).toFixed(3))
    : 1;

  const nowIso = formatDate(new Date());
  const journeyStage = normalizeJourneyStage(
    contextLabels,
    coverage,
    payload.journeyStage ?? existing?.journeyStage
  );

  return {
    id: existing?.id ?? `${mode}-${roleKey}-${driverKey}-${Date.now().toString(36)}`,
    ownerId,
    mode,
    role: roleKey,
    driver: driverKey,
    voice,
    contextKeys,
    createdAt: existing?.createdAt ?? nowIso,
    updatedAt: nowIso,
    name: payload.name ?? existing?.name ?? roleLabel,
    segment: payload.segment ?? existing?.segment ?? mode.toUpperCase(),
    priority: payload.priority ?? existing?.priority ?? 'secondary',
    keyNeed: payload.keyNeed ?? existing?.keyNeed ?? driverLabel,
    journeyStage,
    meta: {
      source: 'custom',
      ownerId,
      role: roleKey,
      driver: driverKey,
      voice,
      contextKeys,
      createdAt: existing?.meta?.createdAt ?? existing?.createdAt ?? nowIso,
      updatedAt: nowIso,
      mode
    }
  };
};

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
    owner: { name: 'Strategy Ops', email: 'ops@agency.com' },
    platforms: ['ChatGPT', 'Claude', 'Perplexity'],
    lastRun: formatDate(new Date(now.getTime() - 1000 * 60 * 60 * 20)),
    healthScore: 78
  },
  {
    id: 'audit-2',
    name: 'Feature Comparison North America',
    cadence: 'Bi-weekly',
    owner: { name: 'Product Marketing', email: 'pm@agency.com' },
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

  run.progress.updatedAt = formatDate(new Date());

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
    progress: { done: 48, total: 120, updatedAt: formatDate(new Date()) },
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
    progress: { done: 96, total: 96, updatedAt: formatDate(new Date()) },
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
    progress: { done: 0, total: payload.questionCount, updatedAt: formatDate(startedAt) },
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
    auditSummariesData.map((summary) => ({
      ...summary,
      owner: { ...summary.owner },
      platforms: [...summary.platforms]
    })),
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

  personas: async (mode: PersonaMode = 'b2c', ownerId?: string): Promise<Persona[]> => {
    const catalogEntries = defaultPersonaStore[mode] ?? defaultPersonaStore.b2c;

    if (!ownerId) {
      return [...catalogEntries];
    }

    const bucket = ensurePersonaBucket(ownerId, mode);
    return [
      ...bucket.map((entry) => ({
        id: entry.id,
        name: entry.name,
        segment: entry.segment,
        priority: entry.priority,
        keyNeed: entry.keyNeed,
        journeyStage: entry.journeyStage,
        meta: entry.meta
      })),
      ...catalogEntries
    ];
  },
  personaCatalog: async (mode: PersonaMode = 'b2c'): Promise<PersonaCatalog> =>
    getPersonaCatalog(mode),
  personaLibrary: async (mode: PersonaMode, ownerId: string): Promise<PersonaLibraryResponse> => {
    const bucket = ensurePersonaBucket(ownerId, mode);
    return {
      personas: bucket.map((entry) => ({ ...entry }))
    };
  },
  createPersona: async (payload: PersonaComposePayload): Promise<PersonaLibraryEntry> => {
    const entry = composePersonaFromPayload(payload);
    const bucket = ensurePersonaBucket(payload.ownerId, payload.mode);
    bucket.push(entry);
    return { ...entry };
  },
  updatePersona: async (
    personaId: string,
    payload: PersonaUpdatePayload
  ): Promise<PersonaLibraryEntry> => {
    const bucket = ensurePersonaBucket(payload.ownerId, payload.mode);
    const index = bucket.findIndex((item) => item.id === personaId);
    if (index === -1) {
      throw new Error('Persona not found');
    }
    const updated = composePersonaFromPayload(payload, bucket[index]);
    bucket[index] = updated;
    return { ...updated };
  },
  clonePersona: async (
    personaId: string,
    payload: PersonaClonePayload
  ): Promise<PersonaLibraryEntry> => {
    const bucket = ensurePersonaBucket(payload.ownerId, payload.mode);
    const existing = bucket.find((item) => item.id === personaId);
    if (!existing) {
      throw new Error('Persona not found');
    }
    const clone: PersonaLibraryEntry = {
      ...existing,
      id: `${payload.mode}-${Date.now().toString(36)}`,
      name: payload.name ?? `${existing.name} Copy`,
      createdAt: formatDate(new Date()),
      updatedAt: formatDate(new Date()),
      meta: {
        ...existing.meta,
        updatedAt: formatDate(new Date())
      }
    };
    bucket.push(clone);
    return { ...clone };
  },
  deletePersona: async (personaId: string, ownerId: string): Promise<void> => {
    const ownerStore = customPersonaStore[ownerId];
    if (!ownerStore) {
      return;
    }
    (['b2c', 'b2b'] as PersonaMode[]).forEach((mode) => {
      ownerStore[mode] = ownerStore[mode].filter((entry) => entry.id !== personaId);
    });
  },

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
      logoUrl: null,
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
