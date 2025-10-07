'use client';

import { useEffect, useState } from 'react';
import { themeManager } from '@/styles/themes/theme-manager';
import type { ThemeName, ColorMode } from '@/styles/themes/theme-manager';
import clsx from 'clsx';

export function ThemeSwitcher() {
  const [theme, setThemeState] = useState<ThemeName>('default');
  const [mode, setModeState] = useState<ColorMode>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setThemeState(themeManager.getTheme());
    setModeState(themeManager.getMode());

    // Subscribe to theme changes
    const unsubscribe = themeManager.onChange((config) => {
      setThemeState(config.name);
      setModeState(config.mode);
    });

    return unsubscribe;
  }, []);

  const handleThemeChange = (newTheme: ThemeName) => {
    themeManager.setTheme(newTheme);
    setThemeState(newTheme);
  };

  const handleModeToggle = () => {
    themeManager.toggleDarkMode();
    setModeState(themeManager.getMode());
  };

  // Prevent hydration mismatch
  if (!mounted) {
    return (
      <div className="flex items-center gap-2 p-2 rounded-lg bg-surface border border-border">
        <div className="w-20 h-8 bg-elevated animate-pulse rounded" />
        <div className="w-8 h-8 bg-elevated animate-pulse rounded" />
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-surface border border-border">
      {/* Theme Selector */}
      <div className="flex gap-1">
        <button
          onClick={() => handleThemeChange('default')}
          className={clsx(
            'px-3 py-1.5 text-sm font-medium rounded transition-colors',
            theme === 'default'
              ? 'bg-accent text-background'
              : 'bg-transparent text-muted hover:text-text'
          )}
          title="Default Theme"
        >
          Default
        </button>
        <button
          onClick={() => handleThemeChange('paleolithic')}
          className={clsx(
            'px-3 py-1.5 text-sm font-medium rounded transition-colors',
            theme === 'paleolithic'
              ? 'bg-accent text-background'
              : 'bg-transparent text-muted hover:text-text'
          )}
          title="Paleolithic Theme"
        >
          Paleolithic
        </button>
      </div>

      {/* Dark Mode Toggle */}
      <button
        onClick={handleModeToggle}
        className="p-2 rounded transition-colors hover:bg-elevated"
        title={mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
        aria-label={mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
      >
        {mode === 'light' ? (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </svg>
        )}
      </button>
    </div>
  );
}
