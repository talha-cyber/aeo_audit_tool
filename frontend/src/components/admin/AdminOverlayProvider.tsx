import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

interface AdminOverlayContextValue {
  isAdmin: boolean;
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

const AdminOverlayContext = createContext<AdminOverlayContextValue | null>(null);

interface AdminOverlayProviderProps {
  isAdmin: boolean;
  children: React.ReactNode;
}

export const AdminOverlayProvider: React.FC<AdminOverlayProviderProps> = ({ isAdmin, children }) => {
  const [isOpen, setIsOpen] = useState(false);

  const open = useCallback(() => {
    if (isAdmin) {
      setIsOpen(true);
    }
  }, [isAdmin]);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const toggle = useCallback(() => {
    if (isAdmin) {
      setIsOpen((prev) => !prev);
    }
  }, [isAdmin]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isAdmin) {
        return;
      }
      const isModifier = event.metaKey || event.ctrlKey;
      if (isModifier && event.key === '.') {
        event.preventDefault();
        toggle();
      }
      if (event.key === 'Escape') {
        close();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isAdmin, toggle, close]);

  const value = useMemo<AdminOverlayContextValue>(
    () => ({
      isAdmin,
      isOpen,
      open,
      close,
      toggle
    }),
    [isAdmin, isOpen, open, close, toggle]
  );

  return <AdminOverlayContext.Provider value={value}>{children}</AdminOverlayContext.Provider>;
};

export const useAdminOverlay = (): AdminOverlayContextValue => {
  const ctx = useContext(AdminOverlayContext);
  if (!ctx) {
    throw new Error('useAdminOverlay must be used within an AdminOverlayProvider');
  }
  return ctx;
};
