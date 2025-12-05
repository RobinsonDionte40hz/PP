import { createTheme } from '@mui/material/styles';

// Project Color Palette
// #293B5F - Deep Navy (Primary Dark)
// #47597E - Slate Blue (Primary)
// #1C1C1C - Smoke Black (Accent/Background)
// #B2AB8C - Warm Beige (Secondary)

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#47597E',
      dark: '#293B5F',
      light: '#1C1C1C',
    },
    secondary: {
      main: '#B2AB8C',
      light: '#C9C4AD',
      dark: '#9A9377',
    },
    background: {
      default: '#1C1C1C',
      paper: '#ffffff',
    },
    text: {
      primary: '#293B5F',
      secondary: '#47597E',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#B2AB8C',
      dark: '#47597E',
      light: '#E8EFFE',
    },
    secondary: {
      main: '#B2AB8C',
      light: '#C9C4AD',
      dark: '#9A9377',
    },
    background: {
      default: '#293B5F',
      paper: '#47597E',
    },
    text: {
      primary: '#B2AB8C',
      secondary: '#E8EFFE',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});
