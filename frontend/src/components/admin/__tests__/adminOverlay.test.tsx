import { render, screen, waitFor } from '@testing-library/react';
import { act } from 'react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { vi } from 'vitest';

import { apiClient } from '@/lib/api/client';
import { useImpersonationStore } from '@/store/impersonation';

import {
  AdminGate,
  AdminOverlayProvider,
  AdminToolbar,
  ImpersonationBanner,
  useAdminOverlay
} from '../index';

let mockedPathname = '/overview';

vi.mock('next/navigation', () => ({
  usePathname: () => mockedPathname,
}));

const setMockedPathname = (value: string) => {
  mockedPathname = value;
};

const tenantActionMock = vi.spyOn(apiClient.admin, 'tenantAction');
const auditActionMock = vi.spyOn(apiClient.admin, 'auditAction');

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const OverlayProbe: React.FC = () => {
  const { isOpen, open, close } = useAdminOverlay();
  return (
    <div>
      <button onClick={open}>open</button>
      <button onClick={close}>close</button>
      {isOpen && <div role="dialog">overlay active</div>}
    </div>
  );
};

describe('Admin overlay components', () => {
  beforeEach(() => {
    tenantActionMock.mockResolvedValue({
      status: 'ok',
      action: 'set_qe_version',
      details: { qeVersion: 'v2' },
    });
    auditActionMock.mockResolvedValue({
      status: 'queued',
      action: 'force_retry',
      details: { runId: 'run-123' },
    });
    useImpersonationStore.setState((state) => ({
      ...state,
      isAdmin: true,
      presentationMode: false,
      clientId: 'client-123',
      clientName: 'Preview Tenant',
      status: 'idle',
      error: null,
    }));
    setMockedPathname('/audits/run/run-123');
  });

  afterEach(() => {
    tenantActionMock.mockReset();
    auditActionMock.mockReset();
    useImpersonationStore.setState((state) => ({
      ...state,
      presentationMode: false,
      clientId: null,
      clientName: null,
      status: 'idle',
      error: null,
    }));
    setMockedPathname('/overview');
  });

  it('renders content only for admins via AdminGate', () => {
    const { rerender } = render(
      <AdminGate isAdmin={false}>
        <div data-testid="sensitive">secret</div>
      </AdminGate>
    );
    expect(screen.queryByTestId('sensitive')).toBeNull();

    rerender(
      <AdminGate isAdmin>
        <div data-testid="sensitive">secret</div>
      </AdminGate>
    );
    expect(screen.getByTestId('sensitive')).toBeInTheDocument();
  });

  it('toggles the admin overlay with the keyboard shortcut', async () => {
    const user = userEvent.setup();

    render(
      <AdminOverlayProvider isAdmin>
        <AdminToolbar />
        <OverlayProbe />
      </AdminOverlayProvider>
    );

    expect(screen.queryByRole('dialog')).toBeNull();

    await user.keyboard('{Meta>}{.}{/Meta}');
    const dialogs = screen.queryAllByRole('dialog');
    expect(dialogs.some((dialog) => dialog.textContent?.includes('overlay active'))).toBe(true);

    await user.keyboard('{Escape}');
    const remainingDialogs = screen.queryAllByRole('dialog');
    expect(remainingDialogs.some((dialog) => dialog.textContent?.includes('overlay active'))).toBe(false);
  });

  it('invokes the tenant admin action when button clicked', async () => {
    const user = userEvent.setup();
    tenantActionMock.mockResolvedValueOnce({
      status: 'ok',
      action: 'set_qe_version',
      details: { qeVersion: 'v2' },
    });

    render(
      <AdminOverlayProvider isAdmin>
        <AdminToolbar />
      </AdminOverlayProvider>
    );

    await user.click(screen.getByRole('button', { name: /open admin overlay/i }));
    await act(async () => {
      await user.click(
        screen.getByRole('button', { name: /switch to question engine v2/i })
      );
    });

    await waitFor(() => {
      expect(tenantActionMock).toHaveBeenCalledWith(
        'client-123',
        expect.objectContaining({ action: 'set_qe_version', value: 'v2' })
      );
    });

    expect(
      screen.getByText(/Switched .* Question Engine/i)
    ).toBeInTheDocument();
  });

  it('enables presentation mode and hides the overlay', async () => {
    const user = userEvent.setup();

    render(
      <AdminOverlayProvider isAdmin>
        <AdminToolbar />
      </AdminOverlayProvider>
    );

    await user.click(screen.getByRole('button', { name: /open admin overlay/i }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    await act(async () => {
      await user.click(
        screen.getByRole('button', { name: /enable presentation mode/i })
      );
    });

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull();
    });

    expect(useImpersonationStore.getState().presentationMode).toBe(true);
  });

  it('renders impersonation banner with exit control', async () => {
    const user = userEvent.setup();
    const handleExit = vi.fn();

    render(
      <ImpersonationBanner clientName="Preview Tenant" onExit={handleExit} />
    );

    expect(screen.getByText(/Viewing as Preview Tenant/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /exit/i }));
    expect(handleExit).toHaveBeenCalledTimes(1);
  });
});
