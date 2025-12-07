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
  Card,
  CardContent,
  CardActions,
  Chip,
  Grid,
} from '@mui/material';
import {
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  ArrowBack as BackIcon,
  MonitorHeart as MonitorIcon,
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionService } from '../services';
import { useWebSocket } from '../hooks/useWebSocket';
import { usePredictions } from '../hooks/usePredictions';
import MetricsGrid from '../components/monitoring/MetricsGrid';
import LiveCharts from '../components/monitoring/LiveCharts';
import SecondaryStructurePanel from '../components/monitoring/SecondaryStructurePanel';
import StructurePreviewModal from '../components/monitoring/StructurePreviewModal';
import { ErrorAlert } from '../components/common';
import { MetricsGridSkeleton, LiveChartsSkeleton } from '../components/common/skeletons';
import type { PredictionProgress, PredictionResponse } from '../types/api';

// Component for showing list of active predictions
const ActivePredictionsList: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  
  const { data: predictions, isLoading, error } = usePredictions({ status: 'running' });
  const { data: pendingPredictions } = usePredictions({ status: 'pending' });
  
  const activePredictions = [
    ...(predictions || []),
    ...(pendingPredictions || []),
  ];

  if (isLoading) {
    return (
      <Box sx={{ p: 3, width: '100%' }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Active Predictions
        </Typography>
        <Box sx={{ width: '100%', mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
            Loading active predictions...
          </Typography>
        </Box>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <ErrorAlert error={error} title="Failed to load predictions" />
      </Box>
    );
  }

  if (activePredictions.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Active Predictions
        </Typography>
        <Paper
          sx={{
            p: 6,
            textAlign: 'center',
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
          }}
        >
          <MonitorIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Active Predictions
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Start a new prediction to see it here
          </Typography>
          <Button variant="contained" onClick={() => navigate('/dashboard/predict')}>
            New Prediction
          </Button>
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Active Predictions
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        {activePredictions.length} prediction(s) currently running or pending
      </Typography>
      
      <Grid container spacing={3}>
        {activePredictions.map((prediction: PredictionResponse) => {
          const progress = prediction.total_iterations > 0
            ? (prediction.current_iteration / prediction.total_iterations) * 100
            : 0;
          
          return (
            <Grid size={{ xs: 12, md: 6, lg: 4 }} key={prediction.id}>
              <Card
                sx={{
                  height: '100%',
                  cursor: 'pointer',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                  },
                }}
                onClick={() => navigate(`/dashboard/monitor/${prediction.id}`)}
              >
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="subtitle2" color="text.secondary" noWrap sx={{ maxWidth: '60%' }}>
                      {prediction.id.slice(0, 8)}...
                    </Typography>
                    <Chip
                      label={prediction.status}
                      size="small"
                      color={prediction.status === 'running' ? 'success' : 'warning'}
                    />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    Sequence: {prediction.sequence?.slice(0, 20)}...
                  </Typography>
                  
                  <Box mb={1}>
                    <Box display="flex" justifyContent="space-between" mb={0.5}>
                      <Typography variant="caption" color="text.secondary">
                        Progress
                      </Typography>
                      <Typography variant="caption" fontWeight="bold">
                        {progress.toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={progress}
                      sx={{ height: 6, borderRadius: 3 }}
                    />
                  </Box>
                  
                  <Typography variant="caption" color="text.secondary">
                    Iteration {prediction.current_iteration?.toLocaleString() || 0} / {prediction.total_iterations?.toLocaleString() || 0}
                  </Typography>
                </CardContent>
                <CardActions>
                  <Button size="small" color="primary">
                    View Details
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

// Type for secondary structure data received via WebSocket
interface SecondaryStructureData {
  assignments: string;
  helix_count: number;
  sheet_count: number;
  coil_count: number;
  helix_percent: number;
  sheet_percent: number;
  coil_percent: number;
  total_residues: number;
  helix_segments?: [number, number][];
  sheet_segments?: [number, number][];
  coil_segments?: [number, number][];
}

const LiveMonitoring: React.FC = () => {
  const theme = useTheme();
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  // If id is 'active', show the list of active predictions
  if (id === 'active') {
    return <ActivePredictionsList />;
  }
  
  const [progressData, setProgressData] = useState<PredictionProgress[]>([]);
  const [showStructureModal, setShowStructureModal] = useState(false);
  const [secondaryStructure, setSecondaryStructure] = useState<SecondaryStructureData | undefined>(undefined);

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
        // Update secondary structure if available in the progress message
        if (message.data.secondary_structure) {
          setSecondaryStructure(message.data.secondary_structure);
        }
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
      isConnected,
      secondaryStructure: secondaryStructure ? 'present' : 'none'
    });
  }, [wsLatestProgress, progressData, latestProgress, isConnected, secondaryStructure]);

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
        navigate(`/dashboard/results/${id}`);
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

  // For real-time display, prefer WebSocket data over database data
  const currentIteration = latestProgress?.iteration ?? prediction.current_iteration;
  const totalIterations = latestProgress?.total_iterations ?? prediction.total_iterations;
  const displayProgress = totalIterations > 0 
    ? (currentIteration / totalIterations) * 100 
    : progress;

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
            {displayProgress.toFixed(1)}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={displayProgress}
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
          Iteration {currentIteration.toLocaleString()} of{' '}
          {totalIterations.toLocaleString()}
        </Typography>
      </Paper>

      {/* Metrics Grid */}
      <Box mb={3}>
        <MetricsGrid prediction={prediction} latestProgress={latestProgress} />
      </Box>

      {/* Charts and Secondary Structure Panel */}
      <Box display="flex" flexDirection={{ xs: 'column', lg: 'row' }} gap={3}>
        {/* Charts - 70% width on large screens */}
        <Box flex={{ lg: '2' }} minWidth="0">
          <LiveCharts progressData={progressData} />
        </Box>

        {/* Secondary Structure Panel - 30% width on large screens */}
        <Box flex={{ lg: '1' }} minWidth="0">
          <SecondaryStructurePanel 
            sequence={prediction.sequence}
            secondaryStructure={secondaryStructure}
            source={secondaryStructure ? "live" : "sequence_estimate"}
          />
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
