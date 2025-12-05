import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
  Science as ScienceIcon,
} from '@mui/icons-material';
import {
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { format, subDays, startOfDay } from 'date-fns';
import type { PredictionResponse } from '../../types/api';

interface StatisticsCardProps {
  stats?: {
    total: number;
    running: number;
    completed: number;
    failed: number;
    avgRMSD: number;
    successRate: number;
  };
  predictions: PredictionResponse[];
}

interface StatItem {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}

const StatisticsCard: React.FC<StatisticsCardProps> = ({ stats, predictions }) => {
  const theme = useTheme();

  // Prepare chart data - predictions over last 7 days
  const chartData = useMemo(() => {
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = startOfDay(subDays(new Date(), 6 - i));
      return {
        date: format(date, 'MMM dd'),
        fullDate: date,
        completed: 0,
        failed: 0,
        total: 0,
      };
    });

    predictions.forEach((pred) => {
      const predDate = startOfDay(new Date(pred.created_at));
      const dayData = last7Days.find(
        (day) => day.fullDate.getTime() === predDate.getTime()
      );
      
      if (dayData) {
        dayData.total += 1;
        if (pred.status === 'completed') {
          dayData.completed += 1;
        } else if (pred.status === 'failed') {
          dayData.failed += 1;
        }
      }
    });

    return last7Days.map(({ date, completed, failed, total }) => ({
      date,
      completed,
      failed,
      total,
    }));
  }, [predictions]);

  // RMSD distribution data - use folding_rmsd as fallback when best_rmsd is not available
  const rmsdDistribution = useMemo(() => {
    const completedPredictions = predictions.filter(
      (p) => p.status === 'completed' && (
        (p.best_rmsd !== null && p.best_rmsd !== undefined && p.best_rmsd !== Infinity) ||
        (p.metrics?.folding_rmsd !== null && p.metrics?.folding_rmsd !== undefined)
      )
    );

    const bins = [
      { range: '0-2Å', min: 0, max: 2, count: 0, label: 'Excellent' },
      { range: '2-4Å', min: 2, max: 4, count: 0, label: 'Good' },
      { range: '4-6Å', min: 4, max: 6, count: 0, label: 'Fair' },
      { range: '6+Å', min: 6, max: Infinity, count: 0, label: 'Poor' },
    ];

    completedPredictions.forEach((pred) => {
      const rmsd = (pred.best_rmsd !== null && pred.best_rmsd !== undefined && pred.best_rmsd !== Infinity)
        ? pred.best_rmsd
        : (pred.metrics?.folding_rmsd ?? 0);
      const bin = bins.find((b) => rmsd >= b.min && rmsd < b.max);
      if (bin) bin.count += 1;
    });

    return bins;
  }, [predictions]);

  const statItems: StatItem[] = [
    {
      label: 'Total Predictions',
      value: stats?.total || 0,
      icon: <TimelineIcon />,
      color: theme.palette.primary.main,
    },
    {
      label: 'Completed',
      value: stats?.completed || 0,
      icon: <CompleteIcon />,
      color: theme.palette.success.main,
    },
    {
      label: 'Failed',
      value: stats?.failed || 0,
      icon: <ErrorIcon />,
      color: theme.palette.error.main,
    },
    {
      label: 'Avg Folding',
      value: stats && stats.avgRMSD > 0 ? `${stats.avgRMSD.toFixed(2)} Å` : 'No data',
      icon: <ScienceIcon />,
      color: theme.palette.info.main,
    },
  ];

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" fontWeight="bold" mb={3}>
          Statistics Overview
        </Typography>

        {/* Stat Items Grid */}
        <Box display="flex" flexWrap="wrap" gap={2} mb={4}>
          {statItems.map((item) => (
            <Box
              key={item.label}
              flex="1 1 calc(25% - 16px)"
              minWidth="150px"
              sx={{
                p: 2,
                borderRadius: 1,
                backgroundColor: alpha(item.color, 0.1),
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
              }}
            >
                <Box sx={{ color: item.color, mb: 1 }}>{item.icon}</Box>
                <Typography variant="h5" fontWeight="bold">
                  {item.value}
                </Typography>
              <Typography variant="caption" color="text.secondary">
                {item.label}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Charts Grid */}
        <Box display="flex" flexDirection={{ xs: 'column', md: 'row' }} gap={3}>
          {/* Activity Chart */}
          <Box flex="2" minWidth="0">
            <Typography variant="subtitle2" fontWeight="bold" mb={2}>
              Activity (Last 7 Days)
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor={theme.palette.primary.main}
                      stopOpacity={0.8}
                    />
                    <stop
                      offset="95%"
                      stopColor={theme.palette.primary.main}
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12 }}
                  stroke={theme.palette.text.secondary}
                />
                <YAxis tick={{ fontSize: 12 }} stroke={theme.palette.text.secondary} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 4,
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="total"
                  stroke={theme.palette.primary.main}
                  fillOpacity={1}
                  fill="url(#colorTotal)"
                  name="Total"
                />
                <Line
                  type="monotone"
                  dataKey="completed"
                  stroke={theme.palette.success.main}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Completed"
                />
                <Line
                  type="monotone"
                  dataKey="failed"
                  stroke={theme.palette.error.main}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Failed"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>

          {/* RMSD Distribution */}
          <Box flex="1" minWidth="0">
            <Typography variant="subtitle2" fontWeight="bold" mb={2}>
              RMSD Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rmsdDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                <XAxis
                  dataKey="range"
                  tick={{ fontSize: 11 }}
                  stroke={theme.palette.text.secondary}
                />
                <YAxis tick={{ fontSize: 12 }} stroke={theme.palette.text.secondary} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 4,
                  }}
                />
                <Bar
                  dataKey="count"
                  fill={theme.palette.primary.main}
                  radius={[4, 4, 0, 0]}
                  name="Predictions"
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default StatisticsCard;
