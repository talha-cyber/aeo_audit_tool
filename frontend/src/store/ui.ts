import { create } from 'zustand';

interface UIState {
  isNavCollapsed: boolean;
  isPaletteOpen: boolean;
  isNewAuditDrawerOpen: boolean;
  toggleNav: () => void;
  openPalette: () => void;
  closePalette: () => void;
  openNewAuditDrawer: () => void;
  closeNewAuditDrawer: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  isNavCollapsed: false,
  isPaletteOpen: false,
  isNewAuditDrawerOpen: false,
  toggleNav: () => set((state) => ({ isNavCollapsed: !state.isNavCollapsed })),
  openPalette: () => set({ isPaletteOpen: true }),
  closePalette: () => set({ isPaletteOpen: false }),
  openNewAuditDrawer: () => set({ isNewAuditDrawerOpen: true }),
  closeNewAuditDrawer: () => set({ isNewAuditDrawerOpen: false })
}));
