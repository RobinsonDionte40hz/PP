import React from 'react';
import { Box } from '@mui/material';
import { MetricCard } from '../common';
import type { PredictionResponse, PredictionProgress } from '../../types/api';

interface MetricsGridProps {
  prediction: PredictionResponse;
  latestProgress?: PredictionProgress;
}

const MetricsGrid: React.FC<MetricsGridProps> = React.memo(({ prediction, latestProgress }) => {
  // Extract metrics from nested object - prefer WebSocket data for real-time updates
  const bestEnergy = latestProgress?.best_energy ?? prediction.metrics?.best_energy ?? null;
  // Use native RMSD if available, otherwise fall back to folding RMSD
  const bestRMSD = latestProgress?.best_rmsd ?? prediction.metrics?.best_rmsd ?? null;
  const foldingRMSD = latestProgress?.folding_rmsd ?? prediction.metrics?.folding_rmsd ?? null;
  const displayRMSD = (bestRMSD !== null && bestRMSD !== undefined && bestRMSD !== Infinity) ? bestRMSD : foldingRMSD;
  const currentEnergy = latestProgress?.current_energy ?? prediction.metrics?.current_energy ?? bestEnergy;
  const currentRMSD = latestProgress?.current_rmsd ?? prediction.metrics?.current_rmsd ?? displayRMSD;
  const rmsdLabel = (bestRMSD !== null && bestRMSD !== undefined && bestRMSD !== Infinity) ? 'RMSD (Native)' : 'Folding Distance';

  const metrics = [
    {
      title: 'Current Energy',
      value: currentEnergy !== null && currentEnergy !== undefined ? currentEnergy.toFixed(2) : '-',
      unit: 'kcal/mol',
      color: 'primary' as const,
      trend: (currentEnergy !== null && bestEnergy !== null
        ? currentEnergy < bestEnergy ? 'down' : 'up'
        : undefined) as 'up' | 'down' | 'neutral' | undefined,
      tooltip: 'Current potential energy of the protein structure. Lower values indicate more stable conformations.',
    },
    {
      title: 'Best Energy',
      value: bestEnergy !== null ? bestEnergy.toFixed(2) : '-',
      unit: 'kcal/mol',
      color: 'success' as const,
      tooltip: 'Lowest potential energy achieved during this prediction. The optimization goal is to minimize this value.',
    },
    {
      title: rmsdLabel,
      value: currentRMSD !== null && currentRMSD !== undefined && currentRMSD !== Infinity ? currentRMSD.toFixed(2) : 'N/A',
      unit: currentRMSD !== null && currentRMSD !== undefined && currentRMSD !== Infinity ? 'Å' : '',
      color: 'info' as const,
      trend: (currentRMSD !== null && currentRMSD !== undefined && displayRMSD !== null && displayRMSD !== undefined && currentRMSD !== Infinity && displayRMSD !== Infinity
        ? currentRMSD < displayRMSD ? 'down' : 'up'
        : undefined) as 'up' | 'down' | 'neutral' | undefined,
      tooltip: rmsdLabel === 'Folding Distance' 
        ? 'How much the structure changed from the initial extended state. Higher values indicate more folding.'
        : 'Root Mean Square Deviation from native structure. Lower is better. Values <2Å indicate excellent accuracy.',
    },
    {
      title: rmsdLabel === 'Folding Distance' ? 'Best Folding' : 'Best RMSD',
      value: displayRMSD !== null && displayRMSD !== undefined && displayRMSD !== Infinity ? displayRMSD.toFixed(2) : 'N/A',
      unit: displayRMSD !== null && displayRMSD !== undefined && displayRMSD !== Infinity ? 'Å' : '',
      color: 'success' as const,
      tooltip: rmsdLabel === 'Folding Distance'
        ? 'Maximum structural change from initial state. Indicates how much the protein has folded.'
        : 'Best (lowest) RMSD achieved. <2Å = Excellent, 2-4Å = Good, 4-5Å = Acceptable, >5Å = Research phase.',
    },
    {
      title: 'Aggressiveness',
      value: latestProgress?.aggressiveness?.toFixed(1) || '-',
      unit: '',
      color: 'warning' as const,
      tooltip: 'Exploration tempo (3-15). Higher values mean more aggressive conformational exploration. Adapts based on success.',
    },
    {
      title: 'Consistency',
      value: latestProgress?.consistency?.toFixed(2) || '-',
      unit: '',
      color: 'secondary' as const,
      tooltip: 'Behavioral stability (0.2-1.0). Higher values indicate more consistent exploration patterns. Increases with success.',
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
});

MetricsGrid.displayName = 'MetricsGrid';

export default MetricsGrid;
