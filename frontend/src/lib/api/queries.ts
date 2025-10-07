import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from './client';
import { mockClient } from './mockClient';
import {
  AuditRun,
  AuditRunDetail,
  AuditSummary,
  LaunchTestRunPayload,
  CreateAuditRunPayload,
  ComparisonMatrix,
  Insight,
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

const useMocks = process.env.NEXT_PUBLIC_USE_MOCKS === 'true';
const fallbackOwnerId = process.env.NEXT_PUBLIC_DASHBOARD_USER_ID ?? 'demo-user';

const client = useMocks ? mockClient : apiClient;

const shouldFallbackToMock = (error: unknown) => {
  if (useMocks) {
    return false;
  }
  if (error && typeof error === 'object' && 'status' in error && typeof (error as { status?: number }).status === 'number') {
    const status = (error as { status?: number }).status ?? 0;
    return status === 0 || status >= 500;
  }
  return true;
};

const ensureOwnerId = <T extends { ownerId?: string }>(payload: T) => ({
  ...payload,
  ownerId: payload.ownerId ?? fallbackOwnerId
});

export function useAuditSummaries() {
  return useQuery<AuditSummary[], Error>({
    queryKey: ['auditSummaries'],
    queryFn: () => client.auditSummaries()
  });
}

export function useAuditRuns() {
  return useQuery<AuditRun[], Error>({
    queryKey: ['auditRuns'],
    queryFn: () => client.auditRuns(),
    refetchInterval: (query) => {
      const runs = query.state.data ?? [];
      return Array.isArray(runs) && runs.some((run) => run.status === 'running' || run.status === 'pending')
        ? 5000
        : false;
    }
  });
}

export function useAuditRunDetail(runId: string) {
  return useQuery<AuditRunDetail, Error>({
    enabled: Boolean(runId),
    queryKey: ['auditRun', runId],
    queryFn: () => client.auditRun(runId),
    refetchInterval: (query) => {
      const run = query.state.data?.run;
      return run && run.status !== 'completed' ? 4000 : false;
    }
  });
}

export function useReportSummaries() {
  return useQuery<ReportSummary[], Error>({
    queryKey: ['reportSummaries'],
    queryFn: () => client.reportSummaries()
  });
}

export function useInsights() {
  return useQuery<Insight[], Error>({
    queryKey: ['insights'],
    queryFn: () => client.insights()
  });
}

export function usePersonas(mode: PersonaMode = 'b2c') {
  return useQuery<Persona[], Error>({
    queryKey: ['personas', mode, fallbackOwnerId],
    queryFn: async () => {
      try {
        return await client.personas(mode, fallbackOwnerId);
      } catch (error) {
        if (shouldFallbackToMock(error)) {
          return mockClient.personas(mode, fallbackOwnerId);
        }
        throw error;
      }
    }
  });
}

export function usePersonaCatalog(mode: PersonaMode = 'b2c') {
  return useQuery<PersonaCatalog, Error>({
    queryKey: ['personaCatalog', mode],
    queryFn: () => client.personaCatalog(mode),
    enabled: Boolean(mode)
  });
}

export function useCreatePersona() {
  const queryClient = useQueryClient();

  return useMutation<PersonaLibraryEntry, Error, PersonaComposePayload>({
    mutationKey: ['createPersona'],
    mutationFn: async (payload) => {
      const enriched = ensureOwnerId(payload);
      try {
        return await client.createPersona(enriched);
      } catch (error) {
        if (shouldFallbackToMock(error)) {
          return mockClient.createPersona(enriched);
        }
        throw error;
      }
    },
    onSuccess: (persona, variables) => {
      queryClient.invalidateQueries({ queryKey: ['personas', variables.mode, fallbackOwnerId] });
      queryClient.invalidateQueries({ queryKey: ['personaLibrary', variables.mode, fallbackOwnerId] });
    }
  });
}

export function usePersonaLibrary(mode: PersonaMode = 'b2c') {
  return useQuery<PersonaLibraryResponse, Error>({
    queryKey: ['personaLibrary', mode, fallbackOwnerId],
    queryFn: async () => {
      try {
        return await client.personaLibrary(mode, fallbackOwnerId);
      } catch (error) {
        if (shouldFallbackToMock(error)) {
          return mockClient.personaLibrary(mode, fallbackOwnerId);
        }
        throw error;
      }
    },
    enabled: Boolean(mode)
  });
}

export function useUpdatePersona() {
  const queryClient = useQueryClient();

  return useMutation<PersonaLibraryEntry, Error, { personaId: string; payload: PersonaUpdatePayload }>(
    {
      mutationKey: ['updatePersona'],
      mutationFn: async ({ personaId, payload }) => {
        const enriched = ensureOwnerId(payload);
        try {
          return await client.updatePersona(personaId, enriched);
        } catch (error) {
          if (shouldFallbackToMock(error)) {
            return mockClient.updatePersona(personaId, enriched);
          }
          throw error;
        }
      },
      onSuccess: (persona, variables) => {
        const { payload } = variables;
        queryClient.invalidateQueries({ queryKey: ['personas', payload.mode, fallbackOwnerId] });
        queryClient.invalidateQueries({ queryKey: ['personaLibrary', payload.mode, fallbackOwnerId] });
      }
    }
  );
}

export function useClonePersona() {
  const queryClient = useQueryClient();

  return useMutation<PersonaLibraryEntry, Error, { personaId: string; payload: PersonaClonePayload }>(
    {
      mutationKey: ['clonePersona'],
      mutationFn: async ({ personaId, payload }) => {
        const enriched = ensureOwnerId(payload);
        try {
          return await client.clonePersona(personaId, enriched);
        } catch (error) {
          if (shouldFallbackToMock(error)) {
            return mockClient.clonePersona(personaId, enriched);
          }
          throw error;
        }
      },
      onSuccess: (persona, variables) => {
        const { payload } = variables;
        queryClient.invalidateQueries({ queryKey: ['personas', payload.mode, fallbackOwnerId] });
        queryClient.invalidateQueries({ queryKey: ['personaLibrary', payload.mode, fallbackOwnerId] });
      }
    }
  );
}

export function useDeletePersona() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, { personaId: string; mode: PersonaMode }>(
    {
      mutationKey: ['deletePersona'],
      mutationFn: async ({ personaId }) => {
        try {
          await client.deletePersona(personaId, fallbackOwnerId);
        } catch (error) {
          if (shouldFallbackToMock(error)) {
            await mockClient.deletePersona(personaId, fallbackOwnerId);
            return;
          }
          throw error;
        }
      },
      onSuccess: (_, variables) => {
        queryClient.invalidateQueries({ queryKey: ['personas', variables.mode, fallbackOwnerId] });
        queryClient.invalidateQueries({ queryKey: ['personaLibrary', variables.mode, fallbackOwnerId] });
      }
    }
  );
}

export function useWidgets() {
  return useQuery<Widget[], Error>({
    queryKey: ['widgets'],
    queryFn: () => client.widgets()
  });
}

export function useComparisonMatrix() {
  return useQuery<ComparisonMatrix, Error>({
    queryKey: ['comparisonMatrix'],
    queryFn: () => client.comparison()
  });
}

export function useSettings() {
  return useQuery<Settings, Error>({
    queryKey: ['settings'],
    queryFn: () => client.settings()
  });
}

export function useLaunchTestRun() {
  const queryClient = useQueryClient();

  return useMutation<AuditRun, Error, LaunchTestRunPayload>({
    mutationKey: ['launchTestRun'],
    mutationFn: (payload) => client.launchTestRun(payload),
    onSuccess: (run) => {
      queryClient.invalidateQueries({ queryKey: ['auditRuns'] });
      queryClient.invalidateQueries({ queryKey: ['auditSummaries'] });
      queryClient.invalidateQueries({ queryKey: ['auditRun', run.id] });
    }
  });
}

export function useCreateAuditRun() {
  const queryClient = useQueryClient();

  return useMutation<AuditRun, Error, CreateAuditRunPayload>({
    mutationKey: ['createAuditRun'],
    mutationFn: (payload) => apiClient.createAuditRun(payload),
    onSuccess: (run) => {
      queryClient.invalidateQueries({ queryKey: ['auditRuns'] });
      queryClient.invalidateQueries({ queryKey: ['auditSummaries'] });
      queryClient.invalidateQueries({ queryKey: ['auditRun', run.id] });
    }
  });
}
