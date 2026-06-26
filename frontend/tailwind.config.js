/** @type {import('tailwindcss').Config} */
// Token set copied verbatim from app.html so its exact classes render identically.
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'on-tertiary-fixed-variant': '#782e35', 'inverse-primary': '#83d8a6', 'secondary-fixed-dim': '#f6bd56',
        'on-surface': '#181d19', primary: '#086b43', 'on-background': '#181d19', 'on-secondary-container': '#734f00',
        'on-secondary': '#ffffff', 'on-primary-fixed': '#002111', 'surface-dim': '#d7dbd5', 'outline-variant': '#bec9bf',
        'error-container': '#ffdad6', 'on-secondary-fixed': '#271900', 'on-primary-fixed-variant': '#005231', secondary: '#7e5700',
        'inverse-surface': '#2d322e', 'tertiary-fixed-dim': '#ffb3b6', 'on-tertiary-container': '#ffffff', tertiary: '#95444b',
        'on-surface-variant': '#3f4942', 'primary-fixed': '#9ff5c1', 'primary-container': '#2f855a', 'surface-container': '#ebefe9',
        background: '#f7faf4', 'primary-fixed-dim': '#83d8a6', 'secondary-fixed': '#ffdeab', 'secondary-container': '#fdc25b',
        'on-primary': '#ffffff', 'on-error': '#ffffff', 'on-primary-container': '#ffffff', 'inverse-on-surface': '#eef2ec',
        'tertiary-fixed': '#ffdada', 'surface-container-lowest': '#ffffff', 'on-tertiary-fixed': '#3f030d', error: '#ba1a1a',
        'surface-container-highest': '#e0e4de', 'surface-variant': '#e0e4de', 'surface-container-low': '#f1f5ef', 'surface-tint': '#0a6c44',
        surface: '#f7faf4', 'on-secondary-fixed-variant': '#5f4100', 'surface-bright': '#f7faf4', 'tertiary-container': '#b35c62',
        outline: '#6f7a71', 'on-error-container': '#93000a', 'on-tertiary': '#ffffff', 'surface-container-high': '#e5e9e3', success: '#38A169',
      },
      borderRadius: { DEFAULT: '0.25rem', lg: '0.5rem', xl: '0.75rem', full: '9999px' },
      spacing: { xl: '40px', gutter: '24px', 'container-max': '1280px', base: '4px', sm: '8px', xxl: '64px', xs: '4px', md: '16px', lg: '24px' },
      fontFamily: {
        'label-md': ['Inter'], 'headline-lg-mobile': ['Fraunces'], 'title-lg': ['Inter'], 'body-md': ['Inter'], 'body-lg': ['Inter'],
        'headline-lg': ['Fraunces'], 'headline-md': ['Fraunces'], 'label-sm': ['Inter'], 'display-lg': ['Fraunces'],
      },
      fontSize: {
        'label-md': ['14px', { lineHeight: '20px', letterSpacing: '0.01em', fontWeight: '600' }],
        'headline-lg-mobile': ['28px', { lineHeight: '36px', fontWeight: '600' }],
        'title-lg': ['20px', { lineHeight: '28px', fontWeight: '600' }],
        'body-md': ['16px', { lineHeight: '24px', fontWeight: '400' }],
        'body-lg': ['18px', { lineHeight: '28px', fontWeight: '400' }],
        'headline-lg': ['32px', { lineHeight: '40px', fontWeight: '600' }],
        'headline-md': ['24px', { lineHeight: '32px', fontWeight: '600' }],
        'label-sm': ['12px', { lineHeight: '16px', fontWeight: '500' }],
        'display-lg': ['48px', { lineHeight: '56px', letterSpacing: '-0.02em', fontWeight: '700' }],
      },
      boxShadow: { low: '0px 2px 4px rgba(31, 42, 36, 0.05)', high: '0px 12px 24px rgba(31, 42, 36, 0.1)' },
    },
  },
  plugins: [],
}
