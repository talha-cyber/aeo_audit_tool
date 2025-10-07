/**
 * Theme System Index
 *
 * Central export point for the theme system
 */

export { themeManager, useTheme } from './theme-manager';
export type { ThemeName, ColorMode } from './theme-manager';

// Import all theme CSS files
import './paleolithic-theme.css';
import './typography.css';
