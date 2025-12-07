/**
 * QuotaDisplay component - Shows user's prediction quota status
 */
import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Tooltip,
  Chip,
  IconButton,
  Collapse,
  Paper,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  AccessTime as AccessTimeIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { quotaService } from '../../services/quotaService';
import { TIER_CONFIG } from '../../types/quota';

interface QuotaDisplayProps {
  /** Whether to show compact (header) or full (dashboard) version */
  variant?: 'compact' | 'full';
  /** Whether to show the tier badge */
  showTier?: boolean;
}

const QuotaDisplay: React.FC<QuotaDisplayProps> = ({
  variant = 'full',
  showTier = true,
}) => {
  const [expanded, setExpanded] = React.useState(false);

  const { data: quota, isLoading, error } = useQuery({
    queryKey: ['user-quota'],
    queryFn: quotaService.getQuota,
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000,
  });

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography variant="caption" color="text.secondary">
          Loading quota...
        </Typography>
      </Box>
    );
  }

  if (error || !quota) {
    return null; // Silently fail if quota can't be loaded
  }

  const tierConfig = TIER_CONFIG[quota.account_tier] || TIER_CONFIG.free;
  const dailyPercentage = quotaService.getQuotaPercentage(quota.daily.used, quota.daily.limit);
  const monthlyPercentage = quotaService.getQuotaPercentage(quota.monthly.used, quota.monthly.limit);
  const dailyColor = quotaService.getQuotaColor(dailyPercentage);
  const monthlyColor = quotaService.getQuotaColor(monthlyPercentage);

  // Compact version for header
  if (variant === 'compact') {
    const isUnlimited = quota.daily.limit <= 0;
    
    return (
      <Tooltip
        title={
          <Box sx={{ p: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Prediction Quota
            </Typography>
            <Typography variant="body2">
              Daily: {quotaService.formatQuotaRemaining(quota.daily.remaining, quota.daily.limit)}
            </Typography>
            <Typography variant="body2">
              Monthly: {quotaService.formatQuotaRemaining(quota.monthly.remaining, quota.monthly.limit)}
            </Typography>
            {quota.daily.reset_at && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                Resets {quotaService.formatResetTime(quota.daily.reset_at)}
              </Typography>
            )}
          </Box>
        }
        arrow
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {showTier && (
            <Chip
              label={tierConfig.displayName}
              size="small"
              color={tierConfig.color}
              variant="outlined"
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          )}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <TrendingUpIcon sx={{ fontSize: 16, color: `${dailyColor}.main` }} />
            <Typography variant="caption" color="text.secondary">
              {isUnlimited ? '∞' : `${quota.daily.remaining}/${quota.daily.limit}`}
            </Typography>
          </Box>
        </Box>
      </Tooltip>
    );
  }

  // Full version for dashboard
  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TrendingUpIcon color="primary" />
          <Typography variant="subtitle1" fontWeight="bold">
            Prediction Quota
          </Typography>
        </Box>
        {showTier && (
          <Chip
            label={tierConfig.displayName}
            size="small"
            color={tierConfig.color}
          />
        )}
      </Box>

      {/* Daily Quota */}
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="body2" color="text.secondary">
            Today
          </Typography>
          <Typography variant="body2" fontWeight="medium">
            {quotaService.formatQuotaRemaining(quota.daily.remaining, quota.daily.limit)}
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={dailyPercentage}
          color={dailyColor}
          sx={{ height: 8, borderRadius: 4 }}
        />
        {quota.daily.reset_at && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
            <AccessTimeIcon sx={{ fontSize: 12, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              Resets {quotaService.formatResetTime(quota.daily.reset_at)}
            </Typography>
          </Box>
        )}
      </Box>

      {/* Monthly Quota - Collapsible */}
      <Box>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Typography variant="body2" color="text.secondary">
            This Month
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" fontWeight="medium">
              {quotaService.formatQuotaRemaining(quota.monthly.remaining, quota.monthly.limit)}
            </Typography>
            <IconButton size="small">
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
        </Box>
        <Collapse in={expanded}>
          <Box sx={{ mt: 1 }}>
            <LinearProgress
              variant="determinate"
              value={monthlyPercentage}
              color={monthlyColor}
              sx={{ height: 8, borderRadius: 4 }}
            />
            {quota.monthly.reset_at && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                <AccessTimeIcon sx={{ fontSize: 12, color: 'text.secondary' }} />
                <Typography variant="caption" color="text.secondary">
                  Resets {quotaService.formatResetTime(quota.monthly.reset_at)}
                </Typography>
              </Box>
            )}
          </Box>
        </Collapse>
      </Box>

      {/* Warning when approaching limit */}
      {(dailyPercentage >= 80 || monthlyPercentage >= 80) && (
        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: 1,
            bgcolor: dailyPercentage >= 90 || monthlyPercentage >= 90 ? 'error.light' : 'warning.light',
            color: dailyPercentage >= 90 || monthlyPercentage >= 90 ? 'error.contrastText' : 'warning.contrastText',
          }}
        >
          <Typography variant="caption" fontWeight="medium">
            {dailyPercentage >= 100 || monthlyPercentage >= 100
              ? 'Quota limit reached. Please wait for reset or upgrade your plan.'
              : 'Approaching quota limit. Consider upgrading for more predictions.'}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default QuotaDisplay;
