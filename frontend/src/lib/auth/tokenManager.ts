const STORAGE_KEY = 'aeo_act_as_session';

export type ActAsSession = {
  token: string;
  clientId: string;
  clientName: string;
  expiresAt: string;
};

const BASE_TOKEN = process.env.NEXT_PUBLIC_AUTH_TOKEN ?? '';

type TokenClaims = {
  act_as?: { client_id?: string; client?: { id?: string; name?: string } };
  client_id?: string;
  client?: { id?: string; name?: string };
  tenant_id?: string;
  tenant?: { id?: string; name?: string };
  client_name?: string;
  tenant_name?: string;
  name?: string;
  exp?: number;
} & Record<string, unknown>;

const decodeClaims = (token: string | null | undefined): TokenClaims | null => {
  if (!token) {
    return null;
  }
  const parts = token.split('.');
  if (parts.length < 2) {
    return null;
  }
  try {
    const normalized = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = normalized.padEnd(normalized.length + (4 - (normalized.length % 4)) % 4, '=');
    const decoded = typeof window !== 'undefined'
      ? window.atob(padded)
      : Buffer.from(padded, 'base64').toString('utf-8');
    return JSON.parse(decoded) as TokenClaims;
  } catch (error) {
    console.warn('Failed to decode token claims', error);
    return null;
  }
};

const BASE_CLAIMS = decodeClaims(BASE_TOKEN);

const deriveClientId = (claims: TokenClaims | null): string | null => {
  if (!claims) {
    return null;
  }
  return (
    claims.act_as?.client_id ??
    claims.act_as?.client?.id ??
    claims.client_id ??
    (typeof claims.client === 'object' ? claims.client?.id : undefined) ??
    claims.tenant_id ??
    (typeof claims.tenant === 'object' ? claims.tenant?.id : undefined) ??
    null
  );
};

const deriveClientName = (claims: TokenClaims | null): string | null => {
  if (!claims) {
    return null;
  }
  return (
    claims.act_as?.client?.name ??
    claims.client_name ??
    (typeof claims.client === 'object' ? claims.client?.name : undefined) ??
    claims.tenant_name ??
    (typeof claims.tenant === 'object' ? claims.tenant?.name : undefined) ??
    (typeof claims.name === 'string' ? claims.name : undefined) ??
    null
  );
};

let memorySession: ActAsSession | null = null;
const listeners = new Set<() => void>();

const readSession = (): ActAsSession | null => {
  if (typeof window === 'undefined') {
    return memorySession;
  }

  try {
    const stored = window.sessionStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return null;
    }
    const parsed = JSON.parse(stored) as ActAsSession;
    if (parsed && typeof parsed.token === 'string') {
      return parsed;
    }
  } catch (error) {
    console.warn('Failed to read act-as session', error);
  }
  return null;
};

const writeSession = (session: ActAsSession | null) => {
  if (typeof window !== 'undefined') {
    if (session) {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    } else {
      window.sessionStorage.removeItem(STORAGE_KEY);
    }
  }
  memorySession = session;
  listeners.forEach((listener) => {
    try {
      listener();
    } catch (error) {
      console.warn('Act-as listener failed', error);
    }
  });
};

const getFallbackClientId = (): string | null => {
  return deriveClientId(BASE_CLAIMS);
};

const getFallbackClientName = (): string | null => {
  return deriveClientName(BASE_CLAIMS);
};

export const tokenManager = {
  getAuthToken(): string {
    const session = readSession();
    if (session?.token) {
      return session.token;
    }
    return BASE_TOKEN;
  },
  getSession(): ActAsSession | null {
    return readSession();
  },
  getBaseClaims(): TokenClaims | null {
    return BASE_CLAIMS;
  },
  getDefaultClientId(): string | null {
    const session = readSession();
    if (session?.clientId) {
      return session.clientId;
    }
    return getFallbackClientId();
  },
  getDefaultClientName(): string | null {
    const session = readSession();
    if (session?.clientName) {
      return session.clientName;
    }
    return getFallbackClientName();
  },
  setSession(session: ActAsSession): void {
    writeSession(session);
  },
  clearSession(): void {
    writeSession(null);
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};
