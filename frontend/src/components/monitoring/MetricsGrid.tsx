import React from 'react';
import { Box } from '@mui/material';
import { MetricCard } from '../common';
import type { PredictionResponse, PredictionProgress } from '../../types/api';

interface MetricsGridProps {
  prediction: PredictionResponse;
  latestProgress?: PredictionProgress;
}

const MetricsGrid: React.FC<MetricsGridProps> = ({ prediction, latestProgress }) => {
  // Extract metrics from nested object - prefer WebSocket data for real-time updates
  const bestEnergy = latestProgress?.best_energy ?? prediction.metrics?.best_energy ?? null;
  const bestRMSD = latestProgress?.best_rmsd ?? prediction.metrics?.best_rmsd ?? null;
  const currentEnergy = latestProgress?.current_energy ?? prediction.metrics?.current_energy ?? bestEnergy;
  const currentRMSD = latestProgress?.current_rmsd ?? prediction.metrics?.current_rmsd ?? bestRMSD;

  const metrics = [
    {
      title: 'Current Energy',
      value: currentEnergy !== null && currentEnergy !== undefined ? currentEnergy.toFixed(2) : '-',
      unit: 'kcal/mol',
      color: 'primary' as const,
      trend: (currentEnergy !== null && bestEnergy !== null
        ? currentEnergy < bestEnergy ? 'down' : 'up'
        : undefined) as 'up' | 'down' | 'neutral' | undefined,
    },
    {
      title: 'Best Energy',
      value: bestEnergy !== null ? bestEnergy.toFixed(2) : '-',
      unit: 'kcal/mol',
      color: 'success' as const,
    },
    {
      title: 'RMSD',
      value: currentRMSD !== null && currentRMSD !== Infinity ? currentRMSD.toFixed(2) : '-',
      unit: 'Å',
      color: 'info' as const,
      trend: (currentRMSD !== null && bestRMSD !== null && currentRMSD !== Infinity && bestRMSD !== Infinity
        ? currentRMSD < bestRMSD ? 'down' : 'up'
        : undefined) as 'up' | 'down' | 'neutral' | undefined,
    },
    {
      title: 'Best RMSD',
      value: bestRMSD !== null && bestRMSD !== Infinity ? bestRMSD.toFixed(2) : 'N/A',
      unit: bestRMSD !== Infinity ? 'Å' : '',
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
