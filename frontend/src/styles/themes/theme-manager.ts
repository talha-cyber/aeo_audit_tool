/**
 * Theme Manager
 *
 * Centralized theme management system that allows easy switching between
 * design themes (grayscale default, paleolithic, etc.)
 *
 * Usage:
 * ```tsx
 * import { themeManager } from '@/styles/themes/theme-manager';
 *
 * // Set theme
 * themeManager.setTheme('paleolithic');
 *
 * // Get current theme
 * const current = themeManager.getTheme();
 *
 * // Toggle dark mode
 * themeManager.toggleDarkMode();
 * ```
 */

export type ThemeName = 'default' | 'paleolithic';
export type ColorMode = 'light' | 'dark';

interface ThemeConfig {
  name: ThemeName;
  mode: ColorMode;
}

class ThemeManager {
  private static instance: ThemeManager;
  private currentTheme: ThemeName = 'default';
  private currentMode: ColorMode = 'light';
  private storageKey = 'aeo-theme-preference';

  private constructor() {
    if (typeof window !== 'undefined') {
      this.loadThemePreference();
    }
  }

  static getInstance(): ThemeManager {
    if (!ThemeManager.instance) {
      ThemeManager.instance = new ThemeManager();
    }
    return ThemeManager.instance;
  }

  /**
   * Load theme preference from localStorage or system preference
   */
  private loadThemePreference(): void {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const config: ThemeConfig = JSON.parse(stored);
        this.setTheme(config.name, config.mode, false);
      } else {
        // Check system preference for dark mode
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        this.setMode(prefersDark ? 'dark' : 'light', false);
      }
    } catch (error) {
      console.warn('Failed to load theme preference:', error);
    }
  }

  /**
   * Save theme preference to localStorage
   */
  private saveThemePreference(): void {
    try {
      const config: ThemeConfig = {
        name: this.currentTheme,
        mode: this.currentMode
      };
      localStorage.setItem(this.storageKey, JSON.stringify(config));
    } catch (error) {
      console.warn('Failed to save theme preference:', error);
    }
  }

  /**
   * Apply theme to document root
   */
  private applyTheme(): void {
    if (typeof document === 'undefined') return;

    const root = document.documentElement;

    // Set theme data attribute
    root.setAttribute('data-theme', this.currentTheme);

    // Set dark mode class
    if (this.currentMode === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }

  /**
   * Set the active theme
   */
  setTheme(theme: ThemeName, mode?: ColorMode, persist = true): void {
    this.currentTheme = theme;
    if (mode) {
      this.currentMode = mode;
    }
    this.applyTheme();
    if (persist) {
      this.saveThemePreference();
    }
  }

  /**
   * Set color mode (light/dark)
   */
  setMode(mode: ColorMode, persist = true): void {
    this.currentMode = mode;
    this.applyTheme();
    if (persist) {
      this.saveThemePreference();
    }
  }

  /**
   * Toggle between light and dark mode
   */
  toggleDarkMode(): void {
    this.setMode(this.currentMode === 'light' ? 'dark' : 'light');
  }

  /**
   * Get current theme name
   */
  getTheme(): ThemeName {
    return this.currentTheme;
  }

  /**
   * Get current color mode
   */
  getMode(): ColorMode {
    return this.currentMode;
  }

  /**
   * Get complete theme config
   */
  getConfig(): ThemeConfig {
    return {
      name: this.currentTheme,
      mode: this.currentMode
    };
  }

  /**
   * Subscribe to theme changes
   */
  onChange(callback: (config: ThemeConfig) => void): () => void {
    const listener = () => {
      callback(this.getConfig());
    };

    // Listen for storage events (theme changes in other tabs)
    window.addEventListener('storage', listener);

    // Return unsubscribe function
    return () => {
      window.removeEventListener('storage', listener);
    };
  }

  /**
   * Listen for system theme preference changes
   */
  watchSystemPreference(): () => void {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const listener = (e: MediaQueryListEvent) => {
      // Only auto-update if user hasn't manually set a preference
      try {
        const stored = localStorage.getItem(this.storageKey);
        if (!stored) {
          this.setMode(e.matches ? 'dark' : 'light', false);
        }
      } catch (error) {
        console.warn('Failed to watch system preference:', error);
      }
    };

    mediaQuery.addEventListener('change', listener);

    // Return cleanup function
    return () => {
      mediaQuery.removeEventListener('change', listener);
    };
  }
}

// Export singleton instance
export const themeManager = ThemeManager.getInstance();

// Export hook for React integration
export function useTheme() {
  if (typeof window === 'undefined') {
    return {
      theme: 'default' as ThemeName,
      mode: 'light' as ColorMode,
      setTheme: () => {},
      setMode: () => {},
      toggleDarkMode: () => {}
    };
  }

  return {
    theme: themeManager.getTheme(),
    mode: themeManager.getMode(),
    setTheme: (theme: ThemeName, mode?: ColorMode) => themeManager.setTheme(theme, mode),
    setMode: (mode: ColorMode) => themeManager.setMode(mode),
    toggleDarkMode: () => themeManager.toggleDarkMode()
  };
}
