'use client';

import { useEffect } from 'react';
import { themeManager } from '@/styles/themes/theme-manager';

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Initialize Paleolithic theme
    themeManager.setTheme('paleolithic');

    // Watch for system preference changes
    const unwatch = themeManager.watchSystemPreference();
    return unwatch;
  }, []);

  return <>{children}</>;
}
