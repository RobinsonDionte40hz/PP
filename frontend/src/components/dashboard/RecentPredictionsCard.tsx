import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  IconButton,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Visibility as ViewIcon,
  PlayArrow as MonitorIcon,
} from '@mui/icons-material';
import type { NavigateFunction } from 'react-router-dom';
import { PredictionStatusBadge } from '../common';
import { formatDistanceToNow } from 'date-fns';
import type { PredictionResponse } from '../../types/api';

interface RecentPredictionsCardProps {
  predictions: PredictionResponse[];
  onNavigate: NavigateFunction;
  onRefresh: () => void;
}

const RecentPredictionsCard: React.FC<RecentPredictionsCardProps> = ({
  predictions,
  onNavigate,
  onRefresh,
}) => {
  const theme = useTheme();

  const truncateSequence = (seq: string, maxLength: number = 20) => {
    return seq.length > maxLength ? `${seq.substring(0, maxLength)}...` : seq;
  };

  const getProgress = (prediction: PredictionResponse) => {
    if (prediction.current_iteration && prediction.total_iterations) {
      return Math.round(
        (prediction.current_iteration / prediction.total_iterations) * 100
      );
    }
    return 0;
  };

  return (
    <Card elevation={2}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight="bold">
            Recent Predictions
          </Typography>
          <IconButton onClick={onRefresh} size="small" title="Refresh">
            <RefreshIcon />
          </IconButton>
        </Box>

        {predictions.length === 0 ? (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            py={6}
            sx={{ backgroundColor: alpha(theme.palette.background.default, 0.5), borderRadius: 1 }}
          >
            <Typography variant="body2" color="text.secondary" mb={2}>
              No predictions yet
            </Typography>
            <Button
              variant="contained"
              color="primary"
              onClick={() => onNavigate('/predictions/new')}
            >
              Create Your First Prediction
            </Button>
          </Box>
        ) : (
          <List sx={{ pt: 0 }}>
            {predictions.slice(0, 5).map((prediction, index) => (
              <React.Fragment key={prediction.id}>
                {index > 0 && <Divider />}
                <ListItem
                  disablePadding
                  secondaryAction={
                    <Box display="flex" gap={1}>
                      {prediction.status === 'running' && (
                        <IconButton
                          edge="end"
                          size="small"
                          onClick={() => onNavigate(`/monitor/${prediction.id}`)}
                          title="Monitor"
                        >
                          <MonitorIcon />
                        </IconButton>
                      )}
                      {prediction.status === 'completed' && (
                        <IconButton
                          edge="end"
                          size="small"
                          onClick={() => onNavigate(`/results/${prediction.id}`)}
                          title="View Results"
                        >
                          <ViewIcon />
                        </IconButton>
                      )}
                    </Box>
                  }
                >
                  <ListItemButton
                    onClick={() => {
                      if (prediction.status === 'running') {
                        onNavigate(`/monitor/${prediction.id}`);
                      } else if (prediction.status === 'completed') {
                        onNavigate(`/results/${prediction.id}`);
                      }
                    }}
                    sx={{ pr: 10 }}
                  >
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="body1" fontWeight="medium">
                            {truncateSequence(prediction.sequence)}
                          </Typography>
                          <PredictionStatusBadge status={prediction.status} />
                        </Box>
                      }
                      secondary={
                        <Box component="span" mt={0.5}>
                          <Typography variant="caption" color="text.secondary" display="block">
                            {formatDistanceToNow(new Date(prediction.created_at), {
                              addSuffix: true,
                            })}
                          </Typography>
                          {prediction.status === 'running' && getProgress(prediction) > 0 && (
                            <Typography variant="caption" color="primary" display="block">
                              Progress: {getProgress(prediction)}%
                            </Typography>
                          )}
                          {prediction.status === 'completed' &&
                            prediction.best_rmsd && (
                              <Typography variant="caption" color="success.main" display="block">
                                RMSD: {prediction.best_rmsd.toFixed(2)} Å
                              </Typography>
                            )}
                        </Box>
                      }
                    />
                  </ListItemButton>
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        )}

        {predictions.length > 5 && (
          <Box mt={2} display="flex" justifyContent="center">
            <Button variant="text" onClick={() => onNavigate('/history')}>
              View All Predictions
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RecentPredictionsCard;
