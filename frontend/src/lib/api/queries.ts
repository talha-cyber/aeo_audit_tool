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
  ReportSummary,
  Settings,
  Widget
} from './schemas';

const useMocks = process.env.NEXT_PUBLIC_USE_MOCKS !== 'false';

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

export function usePersonas() {
  return useQuery<Persona[], Error>({
    queryKey: ['personas'],
    queryFn: () => client.personas()
  });
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
