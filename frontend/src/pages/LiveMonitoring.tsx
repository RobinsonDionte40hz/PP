import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  IconButton,
  LinearProgress,
  Alert,
  Stack,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  ArrowBack as BackIcon,
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionService } from '../services';
import { useWebSocket } from '../hooks/useWebSocket';
import MetricsGrid from '../components/monitoring/MetricsGrid';
import LiveCharts from '../components/monitoring/LiveCharts';
import EventLog from '../components/monitoring/EventLog';
import StructurePreviewModal from '../components/monitoring/StructurePreviewModal';
import { ErrorAlert } from '../components/common';
import { MetricsGridSkeleton, LiveChartsSkeleton } from '../components/common/skeletons';
import type { PredictionProgress } from '../types/api';

const LiveMonitoring: React.FC = () => {
  const theme = useTheme();
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  const [progressData, setProgressData] = useState<PredictionProgress[]>([]);
  const [events, setEvents] = useState<Array<{ level: 'info' | 'warning' | 'error' | 'success'; message: string; timestamp: string }>>([]);
  const [showStructureModal, setShowStructureModal] = useState(false);

  // Fetch prediction details
  const { data: prediction, isLoading, error } = useQuery({
    queryKey: ['prediction', id],
    queryFn: () => predictionService.getPrediction(id!),
    enabled: !!id,
    refetchInterval: 5000, // Fallback polling if WebSocket fails
  });

  // WebSocket connection
  const { isConnected, latestProgress: wsLatestProgress, messages } = useWebSocket(id);

  // Process WebSocket messages
  useEffect(() => {
    messages.forEach((message) => {
      if (message.type === 'progress') {
        setProgressData((prev) => [...prev.slice(-99), message.data]);
      } else if (message.type === 'log') {
        setEvents((prev) => [...prev.slice(-99), message.data]);
      } else if (message.type === 'status') {
        queryClient.invalidateQueries({ queryKey: ['prediction', id] });
      }
    });
  }, [messages, queryClient, id]);

  // Use WebSocket latest progress or fall back to progressData
  const latestProgress = wsLatestProgress || progressData[progressData.length - 1];
  
  // Debug logging
  useEffect(() => {
    console.log('🔍 LiveMonitoring debug:', {
      wsLatestProgress,
      progressDataLength: progressData.length,
      latestProgress,
      isConnected
    });
  }, [wsLatestProgress, progressData, latestProgress, isConnected]);

  // Control mutations
  const pauseMutation = useMutation({
    mutationFn: () => predictionService.pausePrediction(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prediction', id] });
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => predictionService.resumePrediction(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prediction', id] });
    },
  });

  const stopMutation = useMutation({
    mutationFn: () => predictionService.stopPrediction(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prediction', id] });
    },
  });

  const downloadCheckpoint = async () => {
    if (!id) return;
    try {
      const response = await predictionService.downloadCheckpoint(id);
      const blob = new Blob([JSON.stringify(response, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `checkpoint_${id}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Failed to download checkpoint:', err);
    }
  };

  useEffect(() => {
    // Navigate to results if completed
    if (prediction?.status === 'completed') {
      setTimeout(() => {
        navigate(`/results/${id}`);
      }, 2000);
    }
  }, [prediction?.status, id, navigate]);

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton onClick={() => navigate('/dashboard')}>
            <BackIcon />
          </IconButton>
          <Box>
            <Typography variant="h5" fontWeight="bold">
              Loading Prediction...
            </Typography>
          </Box>
        </Box>
        <LinearProgress sx={{ mb: 3 }} />
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 2, mb: 3 }}>
          <MetricsGridSkeleton count={6} />
        </Box>
        <LiveChartsSkeleton />
      </Box>
    );
  }

  if (error || !prediction) {
    return (
      <Box p={3}>
        <ErrorAlert 
          error={error}
          title="Unable to Load Prediction"
          message={!prediction ? "The prediction could not be found. It may have been deleted." : undefined}
          onRetry={() => window.location.reload()}
        />
        <Button startIcon={<BackIcon />} onClick={() => navigate('/dashboard')} sx={{ mt: 2 }}>
          Back to Dashboard
        </Button>
      </Box>
    );
  }

  const progress = prediction.total_iterations > 0
    ? (prediction.current_iteration / prediction.total_iterations) * 100
    : 0;

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Box display="flex" alignItems="center" gap={2}>
            <IconButton onClick={() => navigate('/dashboard')}>
              <BackIcon />
            </IconButton>
            <Typography variant="h4" fontWeight="bold">
              Live Monitoring
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" ml={7}>
            Prediction ID: {id}
          </Typography>
        </Box>

        {/* Control Buttons */}
        <Stack direction="row" spacing={1}>
          <IconButton
            onClick={() => setShowStructureModal(true)}
            title="Preview Structure"
            color="primary"
          >
            <ViewIcon />
          </IconButton>
          <IconButton
            onClick={downloadCheckpoint}
            title="Download Checkpoint"
            disabled={prediction.status !== 'running' && prediction.status !== 'paused'}
          >
            <DownloadIcon />
          </IconButton>
          {prediction.status === 'running' && (
            <Button
              variant="outlined"
              startIcon={<PauseIcon />}
              onClick={() => pauseMutation.mutate()}
              disabled={pauseMutation.isPending}
            >
              Pause
            </Button>
          )}
          {prediction.status === 'paused' && (
            <Button
              variant="outlined"
              color="success"
              startIcon={<PlayIcon />}
              onClick={() => resumeMutation.mutate()}
              disabled={resumeMutation.isPending}
            >
              Resume
            </Button>
          )}
          {(prediction.status === 'running' || prediction.status === 'paused') && (
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
            >
              Stop
            </Button>
          )}
        </Stack>
      </Box>

      {/* WebSocket Status */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Real-time connection lost. Using fallback polling...
        </Alert>
      )}

      {/* Completion Alert */}
      {prediction.status === 'completed' && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Prediction completed! Redirecting to results...
        </Alert>
      )}

      {/* Progress Bar */}
      <Paper
        elevation={2}
        sx={{
          p: 3,
          mb: 3,
          background: `linear-gradient(135deg, ${alpha(
            theme.palette.primary.main,
            0.05
          )} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
        }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6" fontWeight="bold">
            Progress
          </Typography>
          <Typography variant="h6" color="primary" fontWeight="bold">
            {progress.toFixed(1)}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 12,
            borderRadius: 6,
            mb: 1,
            backgroundColor: alpha(theme.palette.primary.main, 0.1),
            '& .MuiLinearProgress-bar': {
              borderRadius: 6,
            },
          }}
        />
        <Typography variant="caption" color="text.secondary">
          Iteration {prediction.current_iteration.toLocaleString()} of{' '}
          {prediction.total_iterations.toLocaleString()}
        </Typography>
      </Paper>

      {/* Metrics Grid */}
      <Box mb={3}>
        <MetricsGrid prediction={prediction} latestProgress={latestProgress} />
      </Box>

      {/* Charts and Event Log */}
      <Box display="flex" flexDirection={{ xs: 'column', lg: 'row' }} gap={3}>
        {/* Charts - 70% width on large screens */}
        <Box flex={{ lg: '2' }} minWidth="0">
          <LiveCharts progressData={progressData} />
        </Box>

        {/* Event Log - 30% width on large screens */}
        <Box flex={{ lg: '1' }} minWidth="0">
          <EventLog events={events} />
        </Box>
      </Box>

      {/* Structure Preview Modal */}
      <StructurePreviewModal
        open={showStructureModal}
        onClose={() => setShowStructureModal(false)}
        predictionId={id || ''}
      />
    </Box>
  );
};

export default LiveMonitoring;
