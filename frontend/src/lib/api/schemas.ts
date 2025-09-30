import { z } from 'zod';

export const ProgressSchema = z.object({
  done: z.number().nonnegative(),
  total: z.number().positive()
});

export const AuditRunSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.enum(['pending', 'running', 'completed', 'failed']),
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
  owner: z.string(),
  platforms: z.array(z.string()),
  lastRun: z.string(),
  healthScore: z.number().min(0).max(100)
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
  )
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
    logoUrl: z.string().optional(),
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

export type AuditRun = z.infer<typeof AuditRunSchema>;
export type AuditSummary = z.infer<typeof AuditSummarySchema>;
export type ReportSummary = z.infer<typeof ReportSummarySchema>;
export type Insight = z.infer<typeof InsightSchema>;
export type Persona = z.infer<typeof PersonaSchema>;
export type Widget = z.infer<typeof WidgetSchema>;
export type Member = z.infer<typeof MemberSchema>;
export type AuditRunDetail = z.infer<typeof AuditRunDetailSchema>;
export type ComparisonMatrix = z.infer<typeof ComparisonMatrixSchema>;
export type Settings = z.infer<typeof SettingsSchema>;
export type LaunchTestRunPayload = z.infer<typeof LaunchTestRunPayloadSchema>;
