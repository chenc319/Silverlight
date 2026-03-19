/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        signal: {
          bg: '#0a0a0f',
          surface: '#12121a',
          'surface-2': '#191924',
          border: '#1e1e2e',
          'border-light': '#2a2a3e',
          green: '#00d4a0',
          'green-dim': '#00d4a033',
          red: '#ff4d6d',
          'red-dim': '#ff4d6d33',
          amber: '#f59e0b',
          'amber-dim': '#f59e0b33',
          blue: '#3b82f6',
          text: '#e2e8f0',
          'text-secondary': '#94a3b8',
          'text-muted': '#64748b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
