const defaultTheme = require('tailwindcss/defaultTheme');

module.exports = {
  content: [
    './src/app/**/*.{ts,tsx}',
    './src/components/**/*.{ts,tsx}',
    './src/styles/**/*.{ts,tsx,css}',
    './src/lib/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--bg)',
        surface: 'var(--surface)',
        elevated: 'var(--elev1)',
        border: 'var(--border)',
        text: 'var(--text)',
        muted: 'var(--muted)',
        accent: 'var(--accent)'
      },
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans]
      },
      borderRadius: {
        lg: '14px',
        xl: '20px'
      },
      boxShadow: {
        sheet: '0 18px 40px rgba(17, 18, 20, 0.08)'
      }
    }
  },
  plugins: []
};
