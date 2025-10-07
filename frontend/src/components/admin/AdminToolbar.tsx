import clsx from 'clsx';
import { usePathname } from 'next/navigation';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';

import type { AdminActionResponse } from '@/lib/api/schemas';
import {
  apiClient,
  type AdminAuditActionPayload,
  type AdminTenantActionPayload
} from '@/lib/api/client';
import { useImpersonationStore } from '@/store/impersonation';

import { useAdminOverlay } from './AdminOverlayProvider';

type FeedbackState = { type: 'success' | 'error'; message: string };

const RUN_DETAIL_REGEX = /\/audits\/run\/([^\/]+)/;

const extractRunId = (pathname: string | null | undefined): string | null => {
  if (!pathname) {
    return null;
  }
  const match = pathname.match(RUN_DETAIL_REGEX);
  return match ? match[1] : null;
};

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Request failed';

const describeSuccess = (
  response: AdminActionResponse,
  contextLabel: string
): string => {
  const action = response.action;
  const details = response.details ?? {};
  switch (action) {
    case 'set_qe_version':
      return `Switched ${contextLabel} to Question Engine ${String(details.qeVersion ?? details.value ?? '').toUpperCase() || 'target'}.`;
    case 'toggle_platform': {
      const platform = String(details.platform ?? '').toUpperCase() || 'platform';
      const enabled = details.enabled === false ? 'disabled' : 'enabled';
      return `${platform} ${enabled} for ${contextLabel}.`;
    }
    case 'set_feature_flag':
      return `Feature flag ${String(details.flag ?? '')} ${(details.enabled === false ? 'disabled' : 'enabled')} for ${contextLabel}.`;
    case 'seed_demo_data': {
      const runs = typeof details.runsCreated === 'number' ? details.runsCreated : 'new';
      return `Seeded ${runs} demo audit runs for ${contextLabel}.`;
    }
    case 'wipe_demo_data':
      return `Demo data wiped for ${contextLabel}.`;
    case 'export_debug_bundle':
      return `Debug bundle ready for download for ${contextLabel}.`;
    case 'force_retry':
    case 'force_rerun': {
      const runId = String(details.runId ?? '') || 'selected run';
      return `Audit ${runId} ${action === 'force_retry' ? 'retry queued' : 'rerun queued'}.`;
    }
    default:
      return `Action ${action} completed for ${contextLabel}.`;
  }
};

interface OverlayActionButtonProps {
  label: string;
  description?: string;
  onClick: () => void | Promise<void>;
  tone?: 'default' | 'primary' | 'danger';
  disabled?: boolean;
  loading?: boolean;
}

const OverlayActionButton: React.FC<OverlayActionButtonProps> = ({
  label,
  description,
  onClick,
  tone = 'default',
  disabled = false,
  loading = false
}) => {
  const toneClasses = {
    default: 'border-border bg-elevated hover:bg-elevated/80 text-text',
    primary: 'border-accent text-accent bg-accent/10 hover:bg-accent/20',
    danger: 'border-red-500 text-red-100 bg-red-900/30 hover:bg-red-900/40'
  }[tone];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className={clsx(
        'w-full rounded-lg border px-4 py-3 text-left transition focus:outline-none focus:ring-2 focus:ring-accent',
        toneClasses,
        (disabled || loading) && 'cursor-not-allowed opacity-60'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold">{label}</p>
          {description ? <p className="mt-1 text-xs text-muted">{description}</p> : null}
        </div>
        {loading ? <span className="text-xs font-semibold text-muted">…</span> : null}
      </div>
    </button>
  );
};

const FeedbackBanner: React.FC<{ state: FeedbackState; onDismiss: () => void }> = ({
  state,
  onDismiss
}) => (
  <div
    className={clsx(
      'mt-4 rounded-md border px-3 py-2 text-sm',
      state.type === 'success'
        ? 'border-emerald-600 bg-emerald-900/30 text-emerald-100'
        : 'border-red-500 bg-red-900/30 text-red-100'
    )}
  >
    <div className="flex items-start justify-between gap-3">
      <span>{state.message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="text-xs uppercase tracking-wide underline"
      >
        Dismiss
      </button>
    </div>
  </div>
);

const AdminOverlayPanel: React.FC = () => {
  const pathname = usePathname();
  const runId = useMemo(() => extractRunId(pathname), [pathname]);
  const {
    clientId,
    clientName,
    clearImpersonation,
    presentationMode,
    setPresentationMode
  } = useImpersonationStore((state) => ({
    clientId: state.clientId,
    clientName: state.clientName,
    clearImpersonation: state.clearImpersonation,
    presentationMode: state.presentationMode,
    setPresentationMode: state.setPresentationMode
  }));
  const { close } = useAdminOverlay();
  const [feedback, setFeedback] = useState<FeedbackState | null>(null);

  const [tenantLoading, setTenantLoading] = useState(false);
  const [auditLoading, setAuditLoading] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  const busy = tenantLoading || auditLoading;

  const runTenantAction = useCallback(
    async (payload: AdminTenantActionPayload) => {
      if (!clientId) {
        setFeedback({
          type: 'error',
          message: 'Impersonate a client to enable tenant actions.'
        });
        return;
      }
      setTenantLoading(true);
      try {
        const response = await apiClient.admin.tenantAction(clientId, payload);
        setFeedback({
          type: 'success',
          message: describeSuccess(response, clientName ?? 'selected tenant')
        });
      } catch (error) {
        setFeedback({ type: 'error', message: errorMessage(error) });
      } finally {
        setTenantLoading(false);
      }
    },
    [clientId, clientName]
  );

  const runAuditAction = useCallback(
    async (payload: AdminAuditActionPayload) => {
      if (!runId) {
        setFeedback({
          type: 'error',
          message: 'Visit an audit run detail page to manage run actions.'
        });
        return;
      }
      setAuditLoading(true);
      try {
        const response = await apiClient.admin.auditAction(runId, payload);
        setFeedback({
          type: 'success',
          message: describeSuccess(response, runId)
        });
      } catch (error) {
        setFeedback({ type: 'error', message: errorMessage(error) });
      } finally {
        setAuditLoading(false);
      }
    },
    [runId]
  );

  const handleTenantAction = useCallback(
    async (payload: AdminTenantActionPayload, confirmation?: string) => {
      if (confirmation && typeof window !== 'undefined') {
        const confirmed = window.confirm(confirmation);
        if (!confirmed) {
          return;
        }
      }
      await runTenantAction(payload);
    },
    [runTenantAction]
  );

  const handleAuditAction = useCallback(
    async (payload: AdminAuditActionPayload) => {
      await runAuditAction(payload);
    },
    [runAuditAction]
  );

  const togglePresentationMode = () => {
    const next = !presentationMode;
    setPresentationMode(next);
    setFeedback({
      type: 'success',
      message: next
        ? 'Presentation mode enabled. Admin controls hidden until you disable it.'
        : 'Presentation mode disabled. Admin controls restored.'
    });
    if (next) {
      close();
    }
  };

  const clearFeedback = () => setFeedback(null);

  const overlay = (
    <div
      role="dialog"
      aria-modal="false"
      className="fixed right-8 top-24 z-[1000] mt-2 w-[420px] max-h-[calc(100vh-4rem)] overflow-y-auto rounded-xl border border-border bg-surface p-6 shadow-2xl"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-muted">Admin Overlay</p>
          <h2 className="mt-1 text-base font-semibold text-text">Tenant Preview Controls</h2>
          <p className="mt-2 text-xs text-muted">
            Use these controls carefully. Actions are audited and rate limited.
          </p>
        </div>
        <button
          type="button"
          onClick={close}
          className="rounded-md border border-border px-2 py-1 text-xs text-muted hover:bg-elevated"
          aria-label="Close admin overlay"
        >
          Close
        </button>
      </div>

      <div className="mt-6 space-y-6">
        <section>
          <h3 className="text-sm font-semibold text-text">Tenant Actions</h3>
          <p className="mt-1 text-xs text-muted">
            Active tenant:{' '}
            <span className="font-medium text-text">{clientName ?? 'None selected'}</span>
          </p>
          <div className="mt-3 space-y-2">
            <OverlayActionButton
              label="Switch to Question Engine v2"
              description="Use the latest persona-aware engine for this tenant."
              onClick={() => handleTenantAction({ action: 'set_qe_version', value: 'v2' })}
              disabled={busy}
              loading={tenantLoading}
              tone="primary"
            />
            <OverlayActionButton
              label="Switch to Question Engine v1"
              description="Fallback to legacy engine for troubleshooting."
              onClick={() => handleTenantAction({ action: 'set_qe_version', value: 'v1' })}
              disabled={busy}
              loading={tenantLoading}
            />
            <OverlayActionButton
              label="Enable Claude for next run"
              description="Toggle a single platform without editing configs."
              onClick={() =>
                handleTenantAction({ action: 'toggle_platform', platform: 'claude', enabled: true })
              }
              disabled={busy}
              loading={tenantLoading}
            />
            <OverlayActionButton
              label="Disable Claude for next run"
              onClick={() =>
                handleTenantAction({ action: 'toggle_platform', platform: 'claude', enabled: false })
              }
              disabled={busy}
              loading={tenantLoading}
            />
            <OverlayActionButton
              label="Seed demo data"
              description="Load a fresh trio of runs for demos."
              onClick={() =>
                handleTenantAction(
                  { action: 'seed_demo_data' },
                  'Seed demo data for the selected tenant? This queues synthetic runs.'
                )
              }
              disabled={busy}
              loading={tenantLoading}
            />
            <OverlayActionButton
              label="Wipe demo data"
              tone="danger"
              description="Remove seeded runs and personas for this tenant."
              onClick={() =>
                handleTenantAction(
                  { action: 'wipe_demo_data' },
                  'Remove demo data for the selected tenant? This deletes seeded runs and personas.'
                )
              }
              disabled={busy}
              loading={tenantLoading}
            />
            <OverlayActionButton
              label="Export debug bundle"
              description="Collect logs and settings for support review."
              onClick={() => handleTenantAction({ action: 'export_debug_bundle' })}
              disabled={busy}
              loading={tenantLoading}
            />
          </div>
        </section>

        <section>
          <h3 className="text-sm font-semibold text-text">Current Run Actions</h3>
          <p className="mt-1 text-xs text-muted">
            Active run context: <span className="font-medium text-text">{runId ?? 'Open a run'}</span>
          </p>
          <div className="mt-3 space-y-2">
            <OverlayActionButton
              label="Force retry"
              description="Requeue the current run with the same config."
              onClick={() => handleAuditAction({ action: 'force_retry' })}
              disabled={busy}
              loading={auditLoading}
              tone="primary"
            />
            <OverlayActionButton
              label="Force rerun"
              description="Start a brand new run with identical settings."
              onClick={() => handleAuditAction({ action: 'force_rerun' })}
              disabled={busy}
              loading={auditLoading}
            />
          </div>
        </section>

        <section>
          <h3 className="text-sm font-semibold text-text">Session Controls</h3>
          <div className="mt-3 space-y-2">
            <OverlayActionButton
              label={presentationMode ? 'Disable presentation mode' : 'Enable presentation mode'}
              description="Hide admin affordances when screen sharing."
              onClick={togglePresentationMode}
              tone={presentationMode ? 'primary' : 'default'}
            />
            <OverlayActionButton
              label="Exit impersonation"
              description="Return to your admin context."
              onClick={() => {
                clearImpersonation();
                setFeedback({ type: 'success', message: 'Impersonation session cleared.' });
              }}
              disabled={!clientId || busy}
            />
          </div>
        </section>
      </div>

      {feedback ? <FeedbackBanner state={feedback} onDismiss={clearFeedback} /> : null}
    </div>
  );

  if (!mounted) {
    return null;
  }

  return createPortal(overlay, document.body);
};

export const AdminToolbar: React.FC = () => {
  const { isAdmin, isOpen, toggle } = useAdminOverlay();

  if (!isAdmin) {
    return null;
  }

  return (
    <div className="relative z-[160]">
      <button
        aria-label="Open admin overlay"
        onClick={toggle}
        type="button"
        className="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-text hover:bg-elevated z-[161]"
      >
        <span aria-hidden="true">🛡️</span>
        <span className="hidden sm:inline">Admin</span>
      </button>
      {isOpen ? <AdminOverlayPanel /> : null}
    </div>
  );
};
