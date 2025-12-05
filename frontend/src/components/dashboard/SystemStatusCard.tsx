import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  useTheme,
  alpha,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  PlayArrow as RunningIcon,
  TrendingUp as TrendingIcon,
} from '@mui/icons-material';

interface SystemStatusCardProps {
  stats?: {
    total: number;
    running: number;
    completed: number;
    failed: number;
    avgRMSD: number;
    successRate: number;
  };
}

const SystemStatusCard: React.FC<SystemStatusCardProps> = ({ stats }) => {
  const theme = useTheme();

  const systemHealth = stats
    ? stats.successRate >= 80
      ? 'excellent'
      : stats.successRate >= 60
      ? 'good'
      : 'needs-attention'
    : 'good';

  const healthColor =
    systemHealth === 'excellent'
      ? theme.palette.success.main
      : systemHealth === 'good'
      ? theme.palette.info.main
      : theme.palette.warning.main;

  const healthLabel =
    systemHealth === 'excellent'
      ? 'Excellent'
      : systemHealth === 'good'
      ? 'Good'
      : 'Needs Attention';

  return (
    <Card
      elevation={2}
      sx={{
        height: '100%',
        background: `linear-gradient(135deg, ${alpha(
          theme.palette.success.main,
          0.05
        )} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
      }}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight="bold">
            System Status
          </Typography>
          <Chip
            label={healthLabel}
            size="small"
            sx={{
              backgroundColor: alpha(healthColor, 0.2),
              color: healthColor,
              fontWeight: 'bold',
            }}
          />
        </Box>

        {stats && (
          <>
            {/* Current Activity */}
            <Box mb={3}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Current Activity
              </Typography>
              <Box display="flex" gap={1}>
                <Box
                  flex={1}
                  display="flex"
                  alignItems="center"
                  gap={1}
                  p={1.5}
                  sx={{
                    backgroundColor: alpha(theme.palette.success.main, 0.1),
                    borderRadius: 1,
                  }}
                >
                  <RunningIcon sx={{ color: theme.palette.success.main }} />
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {stats.running}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Running
                    </Typography>
                  </Box>
                </Box>
                <Box
                  flex={1}
                  display="flex"
                  alignItems="center"
                  gap={1}
                  p={1.5}
                  sx={{
                    backgroundColor: alpha(theme.palette.info.main, 0.1),
                    borderRadius: 1,
                  }}
                >
                  <CheckIcon sx={{ color: theme.palette.info.main }} />
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {stats.completed}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Completed
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Box>

            {/* Performance Metrics */}
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Performance Metrics
              </Typography>
              
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={0.5}>
                  <Typography variant="caption" color="text.secondary">
                    Success Rate
                  </Typography>
                  <Typography variant="caption" fontWeight="bold">
                    {!isNaN(stats.successRate) ? `${stats.successRate.toFixed(1)}%` : 'N/A'}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={!isNaN(stats.successRate) ? stats.successRate : 0}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                    '& .MuiLinearProgress-bar': {
                      backgroundColor:
                        stats.successRate >= 80
                          ? theme.palette.success.main
                          : stats.successRate >= 60
                          ? theme.palette.info.main
                          : theme.palette.warning.main,
                    },
                  }}
                />
              </Box>

              <Box
                display="flex"
                alignItems="center"
                gap={1}
                p={1.5}
                sx={{
                  backgroundColor: alpha(theme.palette.primary.main, 0.05),
                  borderRadius: 1,
                }}
              >
                <TrendingIcon sx={{ color: theme.palette.primary.main }} />
                <Box>
                  <Typography variant="body2" fontWeight="bold">
                    {stats.avgRMSD > 0 && !isNaN(stats.avgRMSD) ? `${stats.avgRMSD.toFixed(2)} Å` : 'N/A'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Avg. RMSD
                  </Typography>
                </Box>
              </Box>
            </Box>
          </>
        )}

        {!stats && (
          <Box display="flex" justifyContent="center" py={4}>
            <Typography variant="body2" color="text.secondary">
              Loading system statistics...
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemStatusCard;
