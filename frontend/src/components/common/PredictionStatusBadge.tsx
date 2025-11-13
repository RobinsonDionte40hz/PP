import { Chip } from '@mui/material';

export type PredictionStatus = 
  | 'pending' 
  | 'running' 
  | 'paused' 
  | 'completed' 
  | 'failed' 
  | 'cancelled';

interface PredictionStatusBadgeProps {
  status: PredictionStatus;
  size?: 'small' | 'medium';
}

export default function PredictionStatusBadge({ status, size = 'small' }: PredictionStatusBadgeProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'pending':
        return { label: 'Pending', color: 'default' as const };
      case 'running':
        return { label: 'Running', color: 'primary' as const };
      case 'paused':
        return { label: 'Paused', color: 'warning' as const };
      case 'completed':
        return { label: 'Completed', color: 'success' as const };
      case 'failed':
        return { label: 'Failed', color: 'error' as const };
      case 'cancelled':
        return { label: 'Cancelled', color: 'default' as const };
      default:
        return { label: 'Unknown', color: 'default' as const };
    }
  };

  const config = getStatusConfig();

  return (
    <Chip
      label={config.label}
      color={config.color}
      size={size}
      sx={{ fontWeight: 500 }}
    />
  );
}
