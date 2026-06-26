import { createTheme, alpha } from '@mui/material/styles'

// M3 dark seed color: deep purple
const primary = '#D0BCFF'
const secondary = '#CCC2DC'
const tertiary = '#EFB8C8'
const surface = '#131218'
const surfaceVariant = '#1E1B2E'
const surfaceContainer = '#211F2D'
const onSurface = '#E6E1E5'

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: primary, contrastText: '#381E72' },
    secondary: { main: secondary, contrastText: '#332D41' },
    error: { main: '#F2B8B5' },
    background: { default: surface, paper: surfaceContainer },
    text: { primary: onSurface, secondary: '#CAC4D0' },
    divider: alpha('#CAC4D0', 0.12),
  },
  typography: {
    fontFamily: '"Roboto", "Google Sans", sans-serif',
    h4: { fontWeight: 400, letterSpacing: 0 },
    h5: { fontWeight: 400 },
    h6: { fontWeight: 500 },
    body1: { letterSpacing: 0.15 },
    body2: { letterSpacing: 0.25 },
    caption: { letterSpacing: 0.4 },
    button: { letterSpacing: 0.1, fontWeight: 500 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 100,
          textTransform: 'none',
          fontWeight: 500,
          paddingLeft: 24,
          paddingRight: 24,
        },
        contained: {
          boxShadow: 'none',
          '&:hover': { boxShadow: 'none' },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: surfaceContainer,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 8 },
      },
    },
    MuiTextField: {
      defaultProps: { variant: 'outlined' },
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            backgroundColor: alpha(primary, 0.04),
          },
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: { color: primary },
        thumb: { width: 20, height: 20 },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: 'none' },
      },
    },
    MuiBottomNavigation: {
      styleOverrides: {
        root: { backgroundColor: '#1E1B2E', borderTop: `1px solid ${alpha('#CAC4D0', 0.12)}` },
      },
    },
    MuiBottomNavigationAction: {
      styleOverrides: {
        root: { color: '#CAC4D0', '&.Mui-selected': { color: primary } },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { borderRadius: 100, height: 4, backgroundColor: alpha(primary, 0.12) },
        bar: { borderRadius: 100 },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: { backgroundColor: surfaceVariant, backgroundImage: 'none', '&:before': { display: 'none' }, borderRadius: '12px !important', marginBottom: 8 },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: { borderRadius: 12 },
      },
    },
  },
})
