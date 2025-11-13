import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
  icon?: React.ReactNode;
}

export default function MetricCard({
  title,
  value,
  unit,
  trend = 'neutral',
  trendValue,
  color = 'primary',
  icon,
}: MetricCardProps) {
  const getTrendIcon = () => {
    if (trend === 'up') return <TrendingUpIcon fontSize="small" />;
    if (trend === 'down') return <TrendingDownIcon fontSize="small" />;
    return null;
  };

  const getTrendColor = () => {
    if (trend === 'up') return 'success';
    if (trend === 'down') return 'error';
    return 'default';
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            {title}
          </Typography>
          {icon && (
            <Box sx={{ color: `${color}.main` }}>
              {icon}
            </Box>
          )}
        </Box>
        
        <Typography variant="h4" component="div" sx={{ my: 1, color: `${color}.main` }}>
          {value}
          {unit && (
            <Typography component="span" variant="h6" color="text.secondary" sx={{ ml: 0.5 }}>
              {unit}
            </Typography>
          )}
        </Typography>
        
        {trendValue && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Chip
              label={trendValue}
              size="small"
              color={getTrendColor()}
              icon={getTrendIcon() || undefined}
              sx={{ height: 20, fontSize: '0.75rem' }}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
