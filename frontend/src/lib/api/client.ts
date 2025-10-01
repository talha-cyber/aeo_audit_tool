import { z } from 'zod';
import {
  AuditRunDetailSchema,
  AuditRunSchema,
  AuditSummarySchema,
  ComparisonMatrixSchema,
  InsightSchema,
  PersonaSchema,
  PersonaLibraryResponseSchema,
  PersonaLibraryEntrySchema,
  PersonaCatalogSchema,
  PersonaMode,
  PersonaComposePayload,
  PersonaUpdatePayload,
  PersonaClonePayload,
  ReportSummarySchema,
  SettingsSchema,
  WidgetSchema,
  LaunchTestRunPayload,
  LaunchTestRunResponseSchema
} from './schemas';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api/v1';
const DASHBOARD_PREFIX = '/dashboard';

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

const buildQueryString = (params: Record<string, string | undefined>) => {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      query.set(key, value);
    }
  });
  const serialized = query.toString();
  return serialized ? `?${serialized}` : '';
};

export const apiClient = {
  auditSummaries: () => request(`${DASHBOARD_PREFIX}/audits`, z.array(AuditSummarySchema)),
  auditRun: (runId: string) => request(`${DASHBOARD_PREFIX}/audits/run/${runId}`, AuditRunDetailSchema),
  auditRuns: () => request(`${DASHBOARD_PREFIX}/audits/runs`, z.array(AuditRunSchema)),
  reportSummaries: () => request(`${DASHBOARD_PREFIX}/reports`, z.array(ReportSummarySchema)),
  insights: () => request(`${DASHBOARD_PREFIX}/insights`, z.array(InsightSchema)),
  personas: (mode: PersonaMode = 'b2c') =>
    request(
      `${DASHBOARD_PREFIX}/personas${buildQueryString({ mode })}`,
      z.array(PersonaSchema)
    ),
  personaCatalog: (mode: PersonaMode = 'b2c') =>
    request(`${DASHBOARD_PREFIX}/personas/catalog?mode=${mode}`, PersonaCatalogSchema),
  personaLibrary: (mode: PersonaMode = 'b2c') =>
    request(
      `${DASHBOARD_PREFIX}/personas/library${buildQueryString({ mode })}`,
      PersonaLibraryResponseSchema
    ),
  createPersona: (payload: PersonaComposePayload) =>
    request(`${DASHBOARD_PREFIX}/personas/custom`, PersonaLibraryEntrySchema, {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  updatePersona: (personaId: string, payload: PersonaUpdatePayload) =>
    request(`${DASHBOARD_PREFIX}/personas/${personaId}`, PersonaLibraryEntrySchema, {
      method: 'PATCH',
      body: JSON.stringify(payload)
    }),
  clonePersona: (personaId: string, payload: PersonaClonePayload) =>
    request(`${DASHBOARD_PREFIX}/personas/${personaId}/clone`, PersonaLibraryEntrySchema, {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  deletePersona: async (personaId: string) => {
    const response = await fetch(
      `${API_BASE}${DASHBOARD_PREFIX}/personas/${personaId}`,
      {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        },
        cache: 'no-store'
      }
    );

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
  },
  widgets: () => request(`${DASHBOARD_PREFIX}/embeds/widgets`, z.array(WidgetSchema)),
  comparison: () => request(`${DASHBOARD_PREFIX}/comparisons/matrix`, ComparisonMatrixSchema),
  settings: () => request(`${DASHBOARD_PREFIX}/settings`, SettingsSchema),
  launchTestRun: async (payload: LaunchTestRunPayload) =>
    request(`${DASHBOARD_PREFIX}/audits/test-run`, LaunchTestRunResponseSchema, {
      method: 'POST',
      body: JSON.stringify(payload)
    }).then((response) => response.run)
};
