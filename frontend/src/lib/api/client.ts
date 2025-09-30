import { z } from 'zod';
import {
  AuditRunDetailSchema,
  AuditRunSchema,
  AuditSummarySchema,
  ComparisonMatrixSchema,
  InsightSchema,
  PersonaSchema,
  ReportSummarySchema,
  SettingsSchema,
  WidgetSchema,
  LaunchTestRunPayload
} from './schemas';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api/v1';

async function request<T>(path: string, schema: z.Schema<T>, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json'
    },
    ...init,
    cache: 'no-store'
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const data = await response.json();
  return schema.parse(data);
}

export const apiClient = {
  auditSummaries: () => request('/audits', z.array(AuditSummarySchema)),
  auditRun: (runId: string) => request(`/audits/run/${runId}`, AuditRunDetailSchema),
  auditRuns: () => request('/audits/runs', z.array(AuditRunSchema)),
  reportSummaries: () => request('/reports', z.array(ReportSummarySchema)),
  insights: () => request('/insights', z.array(InsightSchema)),
  personas: () => request('/personas', z.array(PersonaSchema)),
  widgets: () => request('/embeds/widgets', z.array(WidgetSchema)),
  comparison: () => request('/comparisons/matrix', ComparisonMatrixSchema),
  settings: () => request('/settings', SettingsSchema),
  launchTestRun: async (payload: LaunchTestRunPayload) =>
    request('/audits/test-run', AuditRunSchema, {
      method: 'POST',
      body: JSON.stringify(payload)
    })
};
