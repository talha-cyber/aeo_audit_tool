'use client';

import { create } from 'zustand';
import { apiClient } from '@/lib/api/client';
import { tokenManager } from '@/lib/auth/tokenManager';

interface ImpersonationState {
  isAdmin: boolean;
  presentationMode: boolean;
  clientId: string | null;
  clientName: string | null;
  expiresAt: string | null;
  status: 'idle' | 'loading' | 'error';
  error: string | null;
  hydrate: () => void;
  startImpersonation: (clientId: string, clientName: string) => Promise<void>;
  clearImpersonation: () => void;
  setPresentationMode: (enabled: boolean) => void;
}

const defaultIsAdmin = process.env.NEXT_PUBLIC_IS_ADMIN !== 'false';
const defaultClientId = tokenManager.getDefaultClientId();
const defaultClientName = tokenManager.getDefaultClientName();

export const useImpersonationStore = create<ImpersonationState>((set) => ({
  isAdmin: defaultIsAdmin,
  presentationMode: false,
  clientId: defaultClientId ?? null,
  clientName: defaultClientName ?? null,
  expiresAt: null,
  status: 'idle',
  error: null,
  hydrate: () => {
    const session = tokenManager.getSession();
    const fallbackClientId = tokenManager.getDefaultClientId();
    const fallbackClientName = tokenManager.getDefaultClientName();
    set({
      clientId: session?.clientId ?? fallbackClientId ?? null,
      clientName: session?.clientName ?? fallbackClientName ?? null,
      expiresAt: session?.expiresAt ?? null,
      status: 'idle',
      error: null,
    });
  },
  clearImpersonation: () => {
    tokenManager.clearSession();
    const fallbackClientId = tokenManager.getDefaultClientId();
    const fallbackClientName = tokenManager.getDefaultClientName();
    set({
      clientId: fallbackClientId ?? null,
      clientName: fallbackClientName ?? null,
      expiresAt: null,
      status: 'idle',
      error: null,
    });
  },
  setPresentationMode: (enabled: boolean) => set({ presentationMode: enabled }),
  startImpersonation: async (clientId: string, clientName: string) => {
    set({ status: 'loading', error: null });
    try {
      const response = await apiClient.security.impersonate(clientId);
      tokenManager.setSession({
        token: response.token,
        clientId,
        clientName,
        expiresAt: response.expiresAt,
      });
      set({
        clientId,
        clientName,
        expiresAt: response.expiresAt,
        status: 'idle',
        error: null,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to impersonate client';
      set({ status: 'error', error: message });
      throw error;
    }
  },
}));

// Keep store in sync with storage changes (e.g., other tabs or manual clears)
tokenManager.subscribe(() => {
  const session = tokenManager.getSession();
  const fallbackClientId = tokenManager.getDefaultClientId();
  const fallbackClientName = tokenManager.getDefaultClientName();
  useImpersonationStore.setState({
    clientId: session?.clientId ?? fallbackClientId ?? null,
    clientName: session?.clientName ?? fallbackClientName ?? null,
    expiresAt: session?.expiresAt ?? null,
    status: 'idle',
    error: null,
  });
});
