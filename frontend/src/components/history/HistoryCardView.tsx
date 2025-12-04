import React from 'react';
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Checkbox,
  Stack,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import type { PredictionResponse } from '../../types/api';
import PredictionStatusBadge from '../common/PredictionStatusBadge';

interface HistoryCardViewProps {
  predictions: PredictionResponse[];
  selectedPredictions: string[];
  onSelectPrediction: (predictionId: string) => void;
}

const HistoryCardView: React.FC<HistoryCardViewProps> = ({
  predictions,
  selectedPredictions,
  onSelectPrediction,
}) => {
  const navigate = useNavigate();

  const handleViewResult = (predictionId: string) => {
    navigate(`/dashboard/results/${predictionId}`);
  };

  const handleDownload = (predictionId: string) => {
    // TODO: Implement download
    console.log('Download:', predictionId);
  };

  const handleDelete = (predictionId: string) => {
    // TODO: Implement delete with confirmation
    console.log('Delete:', predictionId);
  };

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
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: {
          xs: '1fr',
          sm: 'repeat(2, 1fr)',
          md: 'repeat(3, 1fr)',
          lg: 'repeat(4, 1fr)',
        },
        gap: 3,
      }}
    >
      {predictions.map((prediction) => (
        <Card
          key={prediction.id}
          sx={{
            position: 'relative',
            '&:hover': {
              boxShadow: 3,
            },
          }}
        >
          {/* Selection Checkbox */}
          <Checkbox
            checked={selectedPredictions.includes(prediction.id)}
            onChange={() => onSelectPrediction(prediction.id)}
            sx={{
              position: 'absolute',
              top: 8,
              left: 8,
              zIndex: 1,
            }}
          />

          <CardContent sx={{ pt: 5 }}>
            {/* Status Badge */}
            <Box sx={{ mb: 2 }}>
              <PredictionStatusBadge status={prediction.status} />
            </Box>

            {/* ID */}
            <Typography variant="h6" component="div" gutterBottom noWrap>
              {prediction.id.slice(0, 8)}
            </Typography>

            {/* Sequence Info */}
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {prediction.sequence?.length || prediction.protein_sequence?.length || 0} residues
            </Typography>

            {/* Metrics */}
            <Stack spacing={1} sx={{ mt: 2 }}>
              {prediction.best_energy !== undefined && (
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Energy:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {prediction.best_energy.toFixed(2)} kcal/mol
                  </Typography>
                </Box>
              )}
              {prediction.best_rmsd !== undefined && (
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    RMSD:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {prediction.best_rmsd.toFixed(2)} Å
                  </Typography>
                </Box>
              )}
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Iterations:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {prediction.current_iteration}/{prediction.total_iterations}
                </Typography>
              </Box>
            </Stack>

            {/* Date */}
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
              {prediction.created_at
                ? formatDistanceToNow(new Date(prediction.created_at), { addSuffix: true })
                : 'Unknown'}
            </Typography>
          </CardContent>

          <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
            <Button size="small" onClick={() => handleViewResult(prediction.id)} startIcon={<ViewIcon />}>
              View
            </Button>
            <Box>
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
                <IconButton size="small" onClick={() => handleDelete(prediction.id)} color="error">
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </CardActions>
        </Card>
      ))}
    </Box>
  );
};

export default HistoryCardView;
