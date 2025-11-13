import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Divider,
  IconButton,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { useQueries } from '@tanstack/react-query';
import { predictionService, resultService } from '../../services';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';
import PredictionStatusBadge from '../common/PredictionStatusBadge';

interface ComparisonModalProps {
  open: boolean;
  onClose: () => void;
  predictionIds: string[];
}

const ComparisonModal: React.FC<ComparisonModalProps> = ({ open, onClose, predictionIds }) => {
  // Fetch all prediction details
  const predictionQueries = useQueries({
    queries: predictionIds.map((id) => ({
      queryKey: ['prediction', id],
      queryFn: () => predictionService.getPrediction(id),
      enabled: open && !!id,
    })),
  });

  // Fetch all result details
  const resultQueries = useQueries({
    queries: predictionIds.map((id) => ({
      queryKey: ['result', id],
      queryFn: () => resultService.getResultDetail(id),
      enabled: open && !!id,
    })),
  });

  const isLoading = predictionQueries.some((q) => q.isLoading) || resultQueries.some((q) => q.isLoading);
  const hasError = predictionQueries.some((q) => q.isError) || resultQueries.some((q) => q.isError);

  const predictions = predictionQueries.map((q) => q.data).filter(Boolean);
  const results = resultQueries.map((q) => q.data).filter(Boolean);

  const getMetricValue = (predictionId: string, metric: string) => {
    const prediction = predictions.find((p) => p?.id === predictionId);
    const result = results.find((r) => r?.prediction_id === predictionId);

    switch (metric) {
      case 'status':
        return prediction?.status;
      case 'residues':
        return prediction?.sequence?.length || prediction?.protein_sequence?.length || 0;
      case 'energy':
        return result?.final_energy ?? prediction?.best_energy;
      case 'rmsd':
        return result?.final_rmsd ?? prediction?.best_rmsd;
      case 'iterations':
        return `${prediction?.current_iteration}/${prediction?.total_iterations}`;
      case 'gdt_ts':
        return result?.structure_quality?.gdt_ts;
      case 'tm_score':
        return result?.structure_quality?.tm_score;
      case 'total_moves':
        return result?.agent_statistics?.total_moves;
      case 'accepted_moves':
        return result?.agent_statistics?.accepted_moves;
      case 'acceptance_rate':
        if (result?.agent_statistics) {
          const total = result.agent_statistics.total_moves || 0;
          const accepted = result.agent_statistics.accepted_moves || 0;
          return total > 0 ? `${((accepted / total) * 100).toFixed(1)}%` : 'N/A';
        }
        return 'N/A';
      default:
        return 'N/A';
    }
  };

  const formatValue = (value: any, metric: string) => {
    if (value === undefined || value === null) return 'N/A';

    if (metric === 'status') {
      return <PredictionStatusBadge status={value as any} />;
    }

    if (typeof value === 'number') {
      if (metric === 'energy') return `${value.toFixed(2)} kcal/mol`;
      if (metric === 'rmsd') return `${value.toFixed(2)} Å`;
      if (metric === 'gdt_ts' || metric === 'tm_score') return value.toFixed(3);
      return value.toLocaleString();
    }

    return value.toString();
  };

  const comparisonMetrics = [
    { key: 'status', label: 'Status' },
    { key: 'residues', label: 'Residues' },
    { key: 'energy', label: 'Final Energy' },
    { key: 'rmsd', label: 'RMSD' },
    { key: 'iterations', label: 'Iterations' },
    { key: 'gdt_ts', label: 'GDT-TS' },
    { key: 'tm_score', label: 'TM-Score' },
    { key: 'total_moves', label: 'Total Moves' },
    { key: 'accepted_moves', label: 'Accepted Moves' },
    { key: 'acceptance_rate', label: 'Acceptance Rate' },
  ];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Compare Predictions</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {isLoading && <LoadingSpinner message="Loading comparison data..." />}

        {hasError && (
          <ErrorAlert message="Failed to load some prediction data. Comparison may be incomplete." />
        )}

        {!isLoading && !hasError && predictions.length > 0 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom sx={{ mb: 2 }}>
              Comparing {predictionIds.length} predictions
            </Typography>

            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold', minWidth: 150 }}>Metric</TableCell>
                    {predictionIds.map((id) => (
                      <TableCell key={id} sx={{ fontWeight: 'bold' }}>
                        {id.slice(0, 12)}...
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {comparisonMetrics.map((metric) => (
                    <TableRow key={metric.key} hover>
                      <TableCell sx={{ fontWeight: 'medium' }}>{metric.label}</TableCell>
                      {predictionIds.map((id) => (
                        <TableCell key={id}>
                          {formatValue(getMetricValue(id, metric.key), metric.key)}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <Divider sx={{ my: 3 }} />

            {/* Best Performers */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Best Performers
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {/* Best Energy */}
                <Paper variant="outlined" sx={{ p: 2, flex: 1, minWidth: 200 }}>
                  <Typography variant="caption" color="text.secondary">
                    Lowest Energy
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {(() => {
                      const energies = predictionIds
                        .map((id) => ({
                          id,
                          value: getMetricValue(id, 'energy') as number,
                        }))
                        .filter((e) => typeof e.value === 'number');
                      const best = energies.reduce((min, curr) =>
                        curr.value < min.value ? curr : min
                      );
                      return best ? `${best.id.slice(0, 8)}...` : 'N/A';
                    })()}
                  </Typography>
                </Paper>

                {/* Best RMSD */}
                <Paper variant="outlined" sx={{ p: 2, flex: 1, minWidth: 200 }}>
                  <Typography variant="caption" color="text.secondary">
                    Lowest RMSD
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {(() => {
                      const rmsds = predictionIds
                        .map((id) => ({
                          id,
                          value: getMetricValue(id, 'rmsd') as number,
                        }))
                        .filter((r) => typeof r.value === 'number');
                      const best = rmsds.reduce((min, curr) =>
                        curr.value < min.value ? curr : min
                      );
                      return best ? `${best.id.slice(0, 8)}...` : 'N/A';
                    })()}
                  </Typography>
                </Paper>

                {/* Highest Acceptance Rate */}
                <Paper variant="outlined" sx={{ p: 2, flex: 1, minWidth: 200 }}>
                  <Typography variant="caption" color="text.secondary">
                    Highest Acceptance
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {(() => {
                      const rates = predictionIds
                        .map((id) => {
                          const result = results.find((r) => r?.prediction_id === id);
                          if (!result?.agent_statistics) return null;
                          const total = result.agent_statistics.total_moves || 0;
                          const accepted = result.agent_statistics.accepted_moves || 0;
                          return { id, value: total > 0 ? (accepted / total) * 100 : 0 };
                        })
                        .filter((r): r is { id: string; value: number } => r !== null);
                      const best = rates.reduce((max, curr) =>
                        curr.value > max.value ? curr : max
                      , { id: '', value: 0 });
                      return best.id ? `${best.id.slice(0, 8)}...` : 'N/A';
                    })()}
                  </Typography>
                </Paper>
              </Box>
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default ComparisonModal;
