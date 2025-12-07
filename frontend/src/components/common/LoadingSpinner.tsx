import { Box, CircularProgress, Typography, useTheme, alpha } from '@mui/material';

interface LoadingSpinnerProps {
  message?: string;
  size?: number;
  /** Makes the spinner take the full viewport height - use for page-level loading */
  fullPage?: boolean;
  /** Minimum height for the container */
  minHeight?: string | number;
}

export default function LoadingSpinner({ 
  message = 'Loading...', 
  size = 40,
  fullPage = false,
  minHeight = '200px'
}: LoadingSpinnerProps) {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: fullPage ? '100vh' : minHeight,
        width: '100%',
        gap: 2,
        py: fullPage ? 0 : 4,
        background: fullPage 
          ? `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.02)} 0%, ${alpha(theme.palette.background.default, 1)} 100%)`
          : 'transparent',
      }}
    >
      <CircularProgress 
        size={size} 
        thickness={4}
        sx={{
          color: theme.palette.primary.main,
        }}
      />
      {message && (
        <Typography 
          variant="body2" 
          color="text.secondary"
          sx={{
            fontWeight: 500,
            letterSpacing: '0.02em',
          }}
        >
          {message}
        </Typography>
      )}
    </Box>
  );
}
