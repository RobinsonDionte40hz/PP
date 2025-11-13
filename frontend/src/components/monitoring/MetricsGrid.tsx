import React from 'react';
import { Box } from '@mui/material';
import { MetricCard } from '../common';
import type { PredictionResponse, PredictionProgress } from '../../types/api';

interface MetricsGridProps {
  prediction: PredictionResponse;
  latestProgress?: PredictionProgress;
}

const MetricsGrid: React.FC<MetricsGridProps> = ({ prediction, latestProgress }) => {
  const metrics = [
    {
      title: 'Current Energy',
      value: latestProgress?.energy?.toFixed(2) || prediction.best_energy?.toFixed(2) || '-',
      unit: 'kcal/mol',
      color: 'primary' as const,
      trend: (latestProgress && prediction.best_energy
        ? latestProgress.energy < prediction.best_energy ? 'down' : 'up'
        : undefined) as 'up' | 'down' | 'neutral' | undefined,
    },
    {
      title: 'Best Energy',
      value: prediction.best_energy?.toFixed(2) || '-',
      unit: 'kcal/mol',
      color: 'success' as const,
    },
    {
      title: 'RMSD',
      value: latestProgress?.rmsd?.toFixed(2) || prediction.best_rmsd?.toFixed(2) || '-',
      unit: 'Å',
      color: 'info' as const,
      trend: (latestProgress?.rmsd && prediction.best_rmsd
        ? latestProgress.rmsd < prediction.best_rmsd ? 'down' : 'up'
        : undefined) as 'up' | 'down' | 'neutral' | undefined,
    },
    {
      title: 'Best RMSD',
      value: prediction.best_rmsd?.toFixed(2) || '-',
      unit: 'Å',
      color: 'success' as const,
    },
    {
      title: 'Aggressiveness',
      value: latestProgress?.aggressiveness?.toFixed(1) || '-',
      unit: '',
      color: 'warning' as const,
    },
    {
      title: 'Consistency',
      value: latestProgress?.consistency?.toFixed(2) || '-',
      unit: '',
      color: 'secondary' as const,
    },
  ];

  return (
    <Box display="flex" flexWrap="wrap" gap={2}>
      {metrics.map((metric) => (
        <Box key={metric.title} flex="1 1 calc(33.333% - 16px)" minWidth="200px">
          <MetricCard {...metric} />
        </Box>
      ))}
    </Box>
  );
};

export default MetricsGrid;
