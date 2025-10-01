import { z } from 'zod';

export const ProgressSchema = z.object({
  done: z.number().nonnegative(),
  total: z.number().nonnegative(),
  updatedAt: z.string().optional().nullable()
});

export const AuditRunSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.enum(['pending', 'running', 'completed', 'failed', 'cancelled']),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  progress: ProgressSchema,
  issues: z.array(
    z.object({
      id: z.string(),
      label: z.string(),
      severity: z.enum(['low', 'medium', 'high'])
    })
  )
});

export const AuditSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  cadence: z.string(),
  owner: z.object({
    name: z.string().optional().nullable(),
    email: z.string().email().optional().nullable()
  }),
  platforms: z.array(z.string()),
  lastRun: z.string().optional().nullable(),
  healthScore: z.number().min(0).max(100).optional().nullable()
});

export const ReportSummarySchema = z.object({
  id: z.string(),
  title: z.string(),
  generatedAt: z.string(),
  auditId: z.string(),
  coverage: z.object({ completed: z.number(), total: z.number() })
});

export const InsightSchema = z.object({
  id: z.string(),
  title: z.string(),
  kind: z.enum(['opportunity', 'risk', 'signal']),
  summary: z.string(),
  detectedAt: z.string(),
  impact: z.enum(['low', 'medium', 'high'])
});

export const PersonaModeSchema = z.enum(['b2c', 'b2b']);

export const PersonaSchema = z.object({
  id: z.string(),
  name: z.string(),
  segment: z.string(),
  priority: z.enum(['primary', 'secondary']),
  keyNeed: z.string(),
  journeyStage: z.array(
    z.object({
      stage: z.string(),
      question: z.string(),
      coverage: z.number().min(0).max(1)
    })
  ),
  meta: z
    .object({
      source: z.string().optional(),
      ownerId: z.string().optional(),
      role: z.string().optional(),
      driver: z.string().optional(),
      voice: z.string().optional().nullable(),
      contextKeys: z.array(z.string()).optional(),
      createdAt: z.string().optional(),
      updatedAt: z.string().optional(),
      mode: z.string().optional()
    })
    .optional()
});

export const PersonaLibraryEntrySchema = PersonaSchema.extend({
  ownerId: z.string(),
  mode: PersonaModeSchema,
  role: z.string(),
  driver: z.string(),
  voice: z.string().nullable().optional(),
  contextKeys: z.array(z.string()),
  createdAt: z.string(),
  updatedAt: z.string()
});

export const PersonaLibraryResponseSchema = z.object({
  personas: z.array(PersonaLibraryEntrySchema)
});

export const PersonaCatalogRoleSchema = z.object({
  key: z.string(),
  label: z.string(),
  description: z.string().optional().nullable(),
  defaultContextStage: z.string().optional().nullable()
});

export const PersonaCatalogDriverSchema = z.object({
  key: z.string(),
  label: z.string(),
  emotionalAnchor: z.string().optional().nullable(),
  weight: z.number().optional().nullable()
});

export const PersonaCatalogContextSchema = z.object({
  key: z.string(),
  label: z.string(),
  priority: z.number().optional().nullable()
});

export const PersonaCatalogVoiceSchema = z.object({
  key: z.string(),
  role: z.string(),
  driver: z.string(),
  contexts: z.array(z.string()).min(1)
});

export const PersonaCatalogSchema = z.object({
  mode: PersonaModeSchema,
  roles: z.array(PersonaCatalogRoleSchema),
  drivers: z.array(PersonaCatalogDriverSchema),
  contexts: z.array(PersonaCatalogContextSchema),
  voices: z.array(PersonaCatalogVoiceSchema)
});

export const PersonaComposePayloadSchema = z.object({
  mode: PersonaModeSchema,
  ownerId: z.string().optional(),
  voice: z.string().optional(),
  role: z.string().optional(),
  driver: z.string().optional(),
  contexts: z.array(z.string()).optional(),
  name: z.string().optional(),
  segment: z.string().optional(),
  priority: z.enum(['primary', 'secondary']).optional(),
  keyNeed: z.string().optional(),
  journeyStage: z
    .array(
      z.object({
        stage: z.string(),
        question: z.string().optional().nullable().default(''),
        coverage: z.number().min(0).max(1).optional()
      })
    )
    .optional()
});

export const PersonaUpdatePayloadSchema = PersonaComposePayloadSchema.extend({
  contexts: z.array(z.string()).optional()
});

export const PersonaClonePayloadSchema = z.object({
  ownerId: z.string().optional(),
  mode: PersonaModeSchema,
  name: z.string().optional()
});

export const WidgetSchema = z.object({
  id: z.string(),
  name: z.string(),
  preview: z.string(),
  status: z.enum(['draft', 'published'])
});

export const MemberSchema = z.object({
  id: z.string(),
  name: z.string(),
  role: z.string(),
  email: z.string().email()
});

export const AuditRunDetailSchema = z.object({
  run: AuditRunSchema,
  questions: z.array(
    z.object({
      id: z.string(),
      prompt: z.string(),
      platform: z.string(),
      sentiment: z.enum(['positive', 'neutral', 'negative']),
      mentions: z.array(
        z.object({
          brand: z.string(),
          frequency: z.number().int(),
          sentiment: z.enum(['positive', 'neutral', 'negative'])
        })
      )
    })
  )
});

export const ComparisonMatrixSchema = z.object({
  competitors: z.array(z.string()),
  signals: z.array(
    z.object({
      label: z.string(),
      weights: z.array(z.number())
    })
  )
});

export const SettingsSchema = z.object({
  branding: z.object({
    primaryColor: z.string(),
    logoUrl: z.string().nullable().optional(),
    tone: z.string()
  }),
  members: z.array(MemberSchema),
  billing: z.object({
    plan: z.string(),
    renewsOn: z.string()
  }),
  integrations: z.array(
    z.object({ id: z.string(), name: z.string(), connected: z.boolean() })
  )
});

export const LaunchTestRunPayloadSchema = z.object({
  scenarioId: z.string(),
  questionCount: z.number().min(1),
  platforms: z.array(z.string()).min(1)
});

export const LaunchTestRunResponseSchema = z.object({
  run: AuditRunSchema
});

export type AuditRun = z.infer<typeof AuditRunSchema>;
export type AuditSummary = z.infer<typeof AuditSummarySchema>;
export type ReportSummary = z.infer<typeof ReportSummarySchema>;
export type Insight = z.infer<typeof InsightSchema>;
export type Persona = z.infer<typeof PersonaSchema>;
export type PersonaMode = z.infer<typeof PersonaModeSchema>;
export type PersonaLibraryEntry = z.infer<typeof PersonaLibraryEntrySchema>;
export type PersonaLibraryResponse = z.infer<typeof PersonaLibraryResponseSchema>;
export type PersonaCatalog = z.infer<typeof PersonaCatalogSchema>;
export type PersonaCatalogVoice = z.infer<typeof PersonaCatalogVoiceSchema>;
export type PersonaComposePayload = z.infer<typeof PersonaComposePayloadSchema>;
export type PersonaUpdatePayload = z.infer<typeof PersonaUpdatePayloadSchema>;
export type PersonaClonePayload = z.infer<typeof PersonaClonePayloadSchema>;
export type Widget = z.infer<typeof WidgetSchema>;
export type Member = z.infer<typeof MemberSchema>;
export type AuditRunDetail = z.infer<typeof AuditRunDetailSchema>;
export type ComparisonMatrix = z.infer<typeof ComparisonMatrixSchema>;
export type Settings = z.infer<typeof SettingsSchema>;
export type LaunchTestRunPayload = z.infer<typeof LaunchTestRunPayloadSchema>;
export type LaunchTestRunResponse = z.infer<typeof LaunchTestRunResponseSchema>;
