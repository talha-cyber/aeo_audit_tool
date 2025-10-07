import { z } from 'zod';
import { tokenManager } from '@/lib/auth/tokenManager';
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
  LaunchTestRunResponseSchema,
  CreateAuditRunPayload,
  CreateAuditRunResponseSchema,
  ImpersonationResponseSchema,
  AdminActionResponseSchema
} from './schemas';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api/v1';
const DASHBOARD_PREFIX = '/dashboard';

type ApiError = Error & { status?: number; body?: unknown };

async function request<T>(path: string, schema: z.Schema<T>, init?: RequestInit): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(init?.headers || {})
  };

  // Add authentication token if available
  const token = tokenManager.getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      cache: 'no-store'
    });
  } catch (error) {
    const networkError: ApiError = error instanceof Error ? error : new Error('Network request failed');
    networkError.status = 0;
    throw networkError;
  }

  const raw = await response.text();
  let data: unknown;
  try {
    data = raw ? JSON.parse(raw) : undefined;
  } catch {
    data = undefined;
  }

  if (!response.ok) {
    const detail =
      data && typeof data === 'object' && data !== null && 'detail' in data
        ? (data as { detail?: unknown }).detail
        : undefined;
    const message =
      (typeof detail === 'string' && detail) ||
      `Request failed: ${response.status}`;
    const apiError: ApiError = new Error(message);
    apiError.status = response.status;
    apiError.body = data;
    throw apiError;
  }

  if (data === undefined) {
    const parseError: ApiError = new Error('Empty response payload');
    parseError.status = response.status;
    throw parseError;
  }

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

const getActiveClientId = (): string | undefined => {
  const clientId = tokenManager.getDefaultClientId();
  return clientId ?? undefined;
};

type AdminAuditAction = 'force_retry' | 'force_rerun';

type AdminTenantAction =
  | 'set_qe_version'
  | 'toggle_platform'
  | 'set_feature_flag'
  | 'seed_demo_data'
  | 'wipe_demo_data'
  | 'export_debug_bundle';

export type AdminTenantActionPayload = {
  action: AdminTenantAction;
  value?: string;
  platform?: string;
  enabled?: boolean;
  flag?: string;
};

export type AdminAuditActionPayload = {
  action: AdminAuditAction;
};

export const apiClient = {
  auditSummaries: (options?: { excludeInternal?: boolean }) =>
    request(
      `${DASHBOARD_PREFIX}/audits${buildQueryString({
        excludeInternal: options?.excludeInternal ? 'true' : undefined,
      })}`,
      z.array(AuditSummarySchema)
    ),
  auditRun: (runId: string) => request(`${DASHBOARD_PREFIX}/audits/run/${runId}`, AuditRunDetailSchema),
  auditRuns: (options?: { excludeInternal?: boolean }) =>
    request(
      `${DASHBOARD_PREFIX}/audits/runs${buildQueryString({
        excludeInternal: options?.excludeInternal ? 'true' : undefined,
      })}`,
      z.array(AuditRunSchema)
    ),
  reportSummaries: (options?: { excludeInternal?: boolean }) =>
    request(
      `${DASHBOARD_PREFIX}/reports${buildQueryString({
        excludeInternal: options?.excludeInternal ? 'true' : undefined,
      })}`,
      z.array(ReportSummarySchema)
    ),
  insights: () => request(`${DASHBOARD_PREFIX}/insights`, z.array(InsightSchema)),
  personas: (mode: PersonaMode = 'b2c', _ownerId?: string) => {
    void _ownerId;
    const clientId = getActiveClientId();
    return request(
      `${DASHBOARD_PREFIX}/personas${buildQueryString({ mode, clientId })}`,
      z.array(PersonaSchema)
    );
  },
  personaCatalog: (mode: PersonaMode = 'b2c') =>
    request(`${DASHBOARD_PREFIX}/personas/catalog?mode=${mode}`, PersonaCatalogSchema),
  personaLibrary: (mode: PersonaMode = 'b2c', _ownerId?: string) => {
    void _ownerId;
    const clientId = getActiveClientId();
    return request(
      `${DASHBOARD_PREFIX}/personas/library${buildQueryString({ mode, clientId })}`,
      PersonaLibraryResponseSchema
    );
  },
  createPersona: (payload: PersonaComposePayload) =>
    request(`${DASHBOARD_PREFIX}/personas/custom`, PersonaLibraryEntrySchema, {
      method: 'POST',
      body: JSON.stringify(
        (() => {
          const clientId = getActiveClientId();
          return clientId ? { ...payload, clientId } : payload;
        })()
      ),
    }),
  updatePersona: (personaId: string, payload: PersonaUpdatePayload) =>
    request(`${DASHBOARD_PREFIX}/personas/${personaId}`, PersonaLibraryEntrySchema, {
      method: 'PATCH',
      body: JSON.stringify(
        (() => {
          const clientId = getActiveClientId();
          return clientId ? { ...payload, clientId } : payload;
        })()
      ),
    }),
  clonePersona: (personaId: string, payload: PersonaClonePayload) =>
    request(`${DASHBOARD_PREFIX}/personas/${personaId}/clone`, PersonaLibraryEntrySchema, {
      method: 'POST',
      body: JSON.stringify(
        (() => {
          const clientId = getActiveClientId();
          return clientId ? { ...payload, clientId } : payload;
        })()
      ),
    }),
  deletePersona: async (personaId: string, _ownerId?: string) => {
    void _ownerId;
    const clientId = getActiveClientId();
    const query = buildQueryString({ clientId });
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    const token = tokenManager.getAuthToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch(
      `${API_BASE}${DASHBOARD_PREFIX}/personas/${personaId}${query}`,
      {
        method: 'DELETE',
        headers,
        cache: 'no-store'
      }
    );

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
  },
  widgets: () => request(`${DASHBOARD_PREFIX}/embeds/widgets`, z.array(WidgetSchema)),
  comparison: (options?: { excludeInternal?: boolean }) =>
    request(
      `${DASHBOARD_PREFIX}/comparisons/matrix${buildQueryString({
        excludeInternal: options?.excludeInternal ? 'true' : undefined,
      })}`,
      ComparisonMatrixSchema
    ),
  settings: () => request(`${DASHBOARD_PREFIX}/settings`, SettingsSchema),
  launchTestRun: async (payload: LaunchTestRunPayload) =>
    request(`${DASHBOARD_PREFIX}/audits/test-run`, LaunchTestRunResponseSchema, {
      method: 'POST',
      body: JSON.stringify(payload)
    }).then((response) => response.run),
  createAuditRun: async (payload: CreateAuditRunPayload) =>
    request(`${DASHBOARD_PREFIX}/audits/run`, CreateAuditRunResponseSchema, {
      method: 'POST',
      body: JSON.stringify(
        (() => {
          const clientId = payload.clientId ?? getActiveClientId();
          return clientId ? { ...payload, clientId } : payload;
        })()
      ),
    }).then((response) => response.run),
  security: {
    impersonate: (clientId: string) =>
      request('/security/impersonate', ImpersonationResponseSchema, {
        method: 'POST',
        body: JSON.stringify({ target_client_id: clientId })
      }),
  },
  admin: {
    auditAction: (runId: string, payload: AdminAuditActionPayload) =>
      request(`/admin/audits/${runId}`, AdminActionResponseSchema, {
        method: 'POST',
        body: JSON.stringify(payload)
      }),
    tenantAction: (clientId: string, payload: AdminTenantActionPayload) =>
      request(`/admin/tenants/${clientId}`, AdminActionResponseSchema, {
        method: 'POST',
        body: JSON.stringify(payload)
      }),
  },
};
