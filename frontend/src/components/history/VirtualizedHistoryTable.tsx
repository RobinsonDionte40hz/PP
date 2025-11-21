/**
 * VirtualizedHistoryTable Component
 * High-performance table with virtual scrolling for large datasets
 */

import React, { useCallback } from 'react';
import {
  Box,
  Typography,
  Checkbox,
  IconButton,
  Tooltip,
  Stack,
  useTheme,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import * as ReactWindow from 'react-window';

// Type for FixedSizeList props
interface FixedSizeListProps {
  height: number;
  itemCount: number;
  itemSize: number;
  width: string | number;
  overscanCount?: number;
  children: React.ComponentType<{ index: number; style: React.CSSProperties }>;
}

// Extract FixedSizeList from react-window namespace
const FixedSizeList = (ReactWindow as unknown as { FixedSizeList: React.ComponentType<FixedSizeListProps> }).FixedSizeList;
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import type { PredictionResponse } from '../../types/api';
import PredictionStatusBadge from '../common/PredictionStatusBadge';

interface VirtualizedHistoryTableProps {
  predictions: PredictionResponse[];
  selectedPredictions: string[];
  onSelectPrediction: (predictionId: string) => void;
  height?: number;
}

const ROW_HEIGHT = 72;
const HEADER_HEIGHT = 56;

const VirtualizedHistoryTable: React.FC<VirtualizedHistoryTableProps> = ({
  predictions,
  selectedPredictions,
  onSelectPrediction,
  height = 600,
}) => {
  const theme = useTheme();
  const navigate = useNavigate();

  const handleViewResult = useCallback((predictionId: string) => {
    navigate(`/results/${predictionId}`);
  }, [navigate]);

  const handleDownload = useCallback((predictionId: string) => {
    // TODO: Implement download
    console.log('Download:', predictionId);
  }, []);

  const handleDelete = useCallback((predictionId: string) => {
    // TODO: Implement delete with confirmation
    console.log('Delete:', predictionId);
  }, []);

  const handleSelectAll = useCallback(() => {
    if (selectedPredictions.length === predictions.length) {
      // Deselect all
      predictions.forEach((p) => {
        if (selectedPredictions.includes(p.id)) {
          onSelectPrediction(p.id);
        }
      });
    } else {
      // Select all
      predictions.forEach((p) => {
        if (!selectedPredictions.includes(p.id)) {
          onSelectPrediction(p.id);
        }
      });
    }
  }, [predictions, selectedPredictions, onSelectPrediction]);

  const isAllSelected = predictions.length > 0 && selectedPredictions.length === predictions.length;
  const isSomeSelected = selectedPredictions.length > 0 && !isAllSelected;

  // Row renderer
  const Row = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const prediction = predictions[index];
    const isSelected = selectedPredictions.includes(prediction.id);

    return (
      <Box
        style={style}
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 2,
          borderBottom: `1px solid ${theme.palette.divider}`,
          backgroundColor: isSelected
            ? theme.palette.action.selected
            : index % 2 === 0
            ? theme.palette.background.default
            : theme.palette.background.paper,
          '&:hover': {
            backgroundColor: theme.palette.action.hover,
          },
        }}
      >
        {/* Checkbox */}
        <Box sx={{ width: '5%', minWidth: 48 }}>
          <Checkbox
            size="small"
            checked={isSelected}
            onChange={() => onSelectPrediction(prediction.id)}
          />
        </Box>

        {/* ID */}
        <Box sx={{ width: '12%', minWidth: 100 }}>
          <Typography variant="body2" fontWeight="medium" noWrap>
            {prediction.id.slice(0, 12)}...
          </Typography>
        </Box>

        {/* Status */}
        <Box sx={{ width: '12%', minWidth: 100 }}>
          <PredictionStatusBadge status={prediction.status} />
        </Box>

        {/* Residues */}
        <Box sx={{ width: '8%', minWidth: 80 }}>
          <Typography variant="body2">
            {prediction.sequence?.length || prediction.protein_sequence?.length || 0}
          </Typography>
        </Box>

        {/* Energy */}
        <Box sx={{ width: '12%', minWidth: 100 }}>
          {prediction.best_energy !== undefined ? (
            <Typography variant="body2">{prediction.best_energy.toFixed(2)}</Typography>
          ) : (
            <Typography variant="body2" color="text.secondary">
              N/A
            </Typography>
          )}
        </Box>

        {/* RMSD */}
        <Box sx={{ width: '12%', minWidth: 100 }}>
          {prediction.best_rmsd !== undefined ? (
            <Typography variant="body2">{prediction.best_rmsd.toFixed(2)}</Typography>
          ) : (
            <Typography variant="body2" color="text.secondary">
              N/A
            </Typography>
          )}
        </Box>

        {/* Iterations */}
        <Box sx={{ width: '15%', minWidth: 120 }}>
          <Typography variant="body2">
            {prediction.current_iteration}/{prediction.total_iterations}
          </Typography>
        </Box>

        {/* Created */}
        <Box sx={{ width: '14%', minWidth: 120 }}>
          <Typography variant="body2" color="text.secondary" noWrap>
            {prediction.created_at
              ? formatDistanceToNow(new Date(prediction.created_at), {
                  addSuffix: true,
                })
              : 'Unknown'}
          </Typography>
        </Box>

        {/* Actions */}
        <Box sx={{ width: '10%', minWidth: 100, display: 'flex', justifyContent: 'flex-end' }}>
          <Stack direction="row" spacing={0.5}>
            <Tooltip title="View details">
              <IconButton
                size="small"
                onClick={() => handleViewResult(prediction.id)}
                color="primary"
              >
                <ViewIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download">
              <IconButton
                size="small"
                onClick={() => handleDownload(prediction.id)}
                disabled={prediction.status !== 'completed'}
              >
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Delete">
              <IconButton
                size="small"
                onClick={() => handleDelete(prediction.id)}
                color="error"
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </Box>
      </Box>
    );
  }, [predictions, selectedPredictions, onSelectPrediction, handleViewResult, handleDownload, handleDelete, theme]);

  if (predictions.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          No predictions found
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Try adjusting your filters or create a new prediction
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Table Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          height: HEADER_HEIGHT,
          px: 2,
          backgroundColor: theme.palette.background.default,
          borderBottom: `2px solid ${theme.palette.divider}`,
          fontWeight: 'bold',
        }}
      >
        <Box sx={{ width: '5%', minWidth: 48 }}>
          <Checkbox
            size="small"
            checked={isAllSelected}
            indeterminate={isSomeSelected}
            onChange={handleSelectAll}
          />
        </Box>
        <Box sx={{ width: '12%', minWidth: 100 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            ID
          </Typography>
        </Box>
        <Box sx={{ width: '12%', minWidth: 100 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            Status
          </Typography>
        </Box>
        <Box sx={{ width: '8%', minWidth: 80 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            Residues
          </Typography>
        </Box>
        <Box sx={{ width: '12%', minWidth: 100 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            Energy
          </Typography>
        </Box>
        <Box sx={{ width: '12%', minWidth: 100 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            RMSD
          </Typography>
        </Box>
        <Box sx={{ width: '15%', minWidth: 120 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            Iterations
          </Typography>
        </Box>
        <Box sx={{ width: '14%', minWidth: 120 }}>
          <Typography variant="subtitle2" fontWeight="bold">
            Created
          </Typography>
        </Box>
        <Box sx={{ width: '10%', minWidth: 100 }}>
          <Typography variant="subtitle2" fontWeight="bold" align="right">
            Actions
          </Typography>
        </Box>
      </Box>

      {/* Virtualized List */}
      <FixedSizeList
        height={height - HEADER_HEIGHT}
        itemCount={predictions.length}
        itemSize={ROW_HEIGHT}
        width="100%"
        overscanCount={5}
      >
        {Row}
      </FixedSizeList>

      {/* Footer with stats */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          px: 2,
          py: 1,
          borderTop: `1px solid ${theme.palette.divider}`,
          backgroundColor: theme.palette.background.default,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Total: {predictions.length} predictions
        </Typography>
        {selectedPredictions.length > 0 && (
          <Typography variant="body2" color="primary">
            {selectedPredictions.length} selected
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default React.memo(VirtualizedHistoryTable);
