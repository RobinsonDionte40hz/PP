import { Box, LinearProgress, Typography } from '@mui/material';

interface ProgressBarProps {
  value: number;
  label?: string;
  showPercentage?: boolean;
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
  height?: number;
}

export default function ProgressBar({
  value,
  label,
  showPercentage = true,
  color = 'primary',
  height = 8,
}: ProgressBarProps) {
  const percentage = Math.min(100, Math.max(0, value));

  return (
    <Box sx={{ width: '100%' }}>
      {(label || showPercentage) && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          {label && (
            <Typography variant="body2" color="text.secondary">
              {label}
            </Typography>
          )}
          {showPercentage && (
            <Typography variant="body2" color="text.secondary">
              {percentage.toFixed(0)}%
            </Typography>
          )}
        </Box>
      )}
      
      <LinearProgress
        variant="determinate"
        value={percentage}
        color={color}
        sx={{ height, borderRadius: 1 }}
      />
    </Box>
  );
}
