import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from './client';
import { mockClient } from './mockClient';
import {
  AuditRun,
  AuditRunDetail,
  AuditSummary,
  LaunchTestRunPayload,
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

const client = useMocks ? mockClient : apiClient;

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
    queryKey: ['personas', mode],
    queryFn: () => client.personas(mode)
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
    mutationFn: (payload) => client.createPersona(payload),
    onSuccess: (persona, variables) => {
      queryClient.invalidateQueries({ queryKey: ['personas', variables.mode] });
      queryClient.invalidateQueries({ queryKey: ['personaLibrary', variables.mode] });
    }
  });
}

export function usePersonaLibrary(mode: PersonaMode = 'b2c') {
  return useQuery<PersonaLibraryResponse, Error>({
    queryKey: ['personaLibrary', mode],
    queryFn: () => client.personaLibrary(mode),
    enabled: Boolean(mode)
  });
}

export function useUpdatePersona() {
  const queryClient = useQueryClient();

  return useMutation<PersonaLibraryEntry, Error, { personaId: string; payload: PersonaUpdatePayload }>(
    {
      mutationKey: ['updatePersona'],
      mutationFn: ({ personaId, payload }) => client.updatePersona(personaId, payload),
      onSuccess: (persona, variables) => {
        const { payload } = variables;
        queryClient.invalidateQueries({ queryKey: ['personas', payload.mode] });
        queryClient.invalidateQueries({ queryKey: ['personaLibrary', payload.mode] });
      }
    }
  );
}

export function useClonePersona() {
  const queryClient = useQueryClient();

  return useMutation<PersonaLibraryEntry, Error, { personaId: string; payload: PersonaClonePayload }>(
    {
      mutationKey: ['clonePersona'],
      mutationFn: ({ personaId, payload }) => client.clonePersona(personaId, payload),
      onSuccess: (persona, variables) => {
        const { payload } = variables;
        queryClient.invalidateQueries({ queryKey: ['personas', payload.mode] });
        queryClient.invalidateQueries({ queryKey: ['personaLibrary', payload.mode] });
      }
    }
  );
}

export function useDeletePersona() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, { personaId: string; mode: PersonaMode }>(
    {
      mutationKey: ['deletePersona'],
      mutationFn: ({ personaId }) => client.deletePersona(personaId),
      onSuccess: (_, variables) => {
        queryClient.invalidateQueries({ queryKey: ['personas', variables.mode] });
        queryClient.invalidateQueries({ queryKey: ['personaLibrary', variables.mode] });
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
