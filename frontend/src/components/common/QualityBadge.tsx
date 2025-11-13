import { Chip } from '@mui/material';

export type QualityLevel = 'excellent' | 'good' | 'acceptable' | 'poor' | 'unknown';

interface QualityBadgeProps {
  quality: QualityLevel;
  metric?: string;
  size?: 'small' | 'medium';
}

export default function QualityBadge({ quality, metric, size = 'small' }: QualityBadgeProps) {
  const getQualityConfig = () => {
    switch (quality) {
      case 'excellent':
        return { label: 'Excellent', color: 'success' as const };
      case 'good':
        return { label: 'Good', color: 'info' as const };
      case 'acceptable':
        return { label: 'Acceptable', color: 'warning' as const };
      case 'poor':
        return { label: 'Poor', color: 'error' as const };
      case 'unknown':
        return { label: 'Unknown', color: 'default' as const };
      default:
        return { label: 'Unknown', color: 'default' as const };
    }
  };

  const config = getQualityConfig();
  const label = metric ? `${metric}: ${config.label}` : config.label;

  return (
    <Chip
      label={label}
      color={config.color}
      size={size}
      variant="outlined"
      sx={{ fontWeight: 500 }}
    />
  );
}
