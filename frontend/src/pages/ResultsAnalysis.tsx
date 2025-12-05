import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Button,
  Chip,
  Alert,
  CircularProgress,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
  IconButton,
  Tooltip,
  alpha,
  useTheme,
} from '@mui/material';
import {
  ArrowBack,
  Download,
  Compare,
  Share,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  ViewModule as CardViewIcon,
  ViewList as TableViewIcon,
  Refresh as RefreshIcon,
  CompareArrows as CompareIcon,
  Assessment as ResultsIcon,
} from '@mui/icons-material';
import { predictionService } from '../services/predictionService';
import { usePredictions } from '../hooks/usePredictions';
import {
  SummaryTab,
  DetailedMetricsTab,
  TrajectoryTab,
  GeometricAnalysisTab,
} from '../components/results';
import HistoryFilters from '../components/history/HistoryFilters';
import HistoryCardView from '../components/history/HistoryCardView';
import HistoryTableView from '../components/history/HistoryTableView';
import ComparisonModal from '../components/history/ComparisonModal';
import { ResultsAnalysisSkeleton } from '../components/common/skeletons';
import ErrorAlert from '../components/common/ErrorAlert';
import { TableSkeleton } from '../components/common/skeletons';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`results-tabpanel-${index}`}
      aria-labelledby={`results-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `results-tab-${index}`,
    'aria-controls': `results-tabpanel-${index}`,
  };
}

type ViewMode = 'card' | 'table';

interface FilterState {
  status?: string;
  dateRange?: [Date | null, Date | null];
  qualityLevel?: string;
  sortBy: 'created_at' | 'energy' | 'rmsd' | 'iterations';
  sortOrder: 'asc' | 'desc';
  searchQuery: string;
}

// Component for showing history/past results
const PastResultsList: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState<ViewMode>('card');
  const [filters, setFilters] = useState<FilterState>({
    sortBy: 'created_at',
    sortOrder: 'desc',
    searchQuery: '',
  });
  const [selectedPredictions, setSelectedPredictions] = useState<string[]>([]);
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);

  // Only fetch completed predictions
  const { data: predictions, isLoading, isError, error, refetch } = usePredictions({
    status: 'completed',
  });

  const handleViewModeChange = (_event: React.MouseEvent<HTMLElement>, newMode: ViewMode | null) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };

  const handleFilterChange = (newFilters: Partial<FilterState>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  };

  const handleSelectPrediction = (predictionId: string) => {
    setSelectedPredictions((prev) => {
      if (prev.includes(predictionId)) {
        return prev.filter((id) => id !== predictionId);
      } else {
        return [...prev, predictionId];
      }
    });
  };

  const handleClearSelection = () => {
    setSelectedPredictions([]);
  };

  const handleCompare = () => {
    if (selectedPredictions.length >= 2) {
      setComparisonModalOpen(true);
    }
  };

  const handleRefresh = () => {
    refetch();
  };

  // Apply client-side filtering and sorting
  const filteredAndSortedPredictions = React.useMemo(() => {
    if (!predictions) return [];

    let filtered = [...predictions];

    // Search filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      filtered = filtered.filter(
        (p) =>
          p.id.toLowerCase().includes(query) ||
          p.sequence?.toLowerCase().includes(query)
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      switch (filters.sortBy) {
        case 'created_at':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
        case 'energy':
          comparison = (a.final_energy || 0) - (b.final_energy || 0);
          break;
        case 'rmsd':
          comparison = (a.best_rmsd || 0) - (b.best_rmsd || 0);
          break;
        case 'iterations':
          comparison = (a.total_iterations || 0) - (b.total_iterations || 0);
          break;
      }
      return filters.sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [predictions, filters]);

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Past Results
        </Typography>
        <TableSkeleton rows={5} columns={5} />
      </Box>
    );
  }

  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Past Results
        </Typography>
        <ErrorAlert message={error instanceof Error ? error.message : 'Failed to load results'} />
      </Box>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Past Results
        </Typography>
        <Paper
          sx={{
            p: 6,
            textAlign: 'center',
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
          }}
        >
          <ResultsIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Completed Predictions
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Complete a prediction to see results here
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
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Past Results
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Browse and analyze your completed predictions
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {selectedPredictions.length > 0 && (
            <>
              <Chip
                label={`${selectedPredictions.length} selected`}
                onDelete={handleClearSelection}
                color="primary"
              />
              <Button
                variant="outlined"
                startIcon={<CompareIcon />}
                onClick={handleCompare}
                disabled={selectedPredictions.length < 2}
              >
                Compare
              </Button>
            </>
          )}
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={handleViewModeChange}
            size="small"
          >
            <ToggleButton value="card">
              <CardViewIcon />
            </ToggleButton>
            <ToggleButton value="table">
              <TableViewIcon />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Filters */}
      <HistoryFilters filters={filters} onFilterChange={handleFilterChange} />

      {/* Results */}
      {viewMode === 'card' ? (
        <HistoryCardView
          predictions={filteredAndSortedPredictions}
          selectedPredictions={selectedPredictions}
          onSelectPrediction={handleSelectPrediction}
        />
      ) : (
        <HistoryTableView
          predictions={filteredAndSortedPredictions}
          selectedPredictions={selectedPredictions}
          onSelectPrediction={handleSelectPrediction}
        />
      )}

      {/* Comparison Modal */}
      <ComparisonModal
        open={comparisonModalOpen}
        onClose={() => setComparisonModalOpen(false)}
        predictionIds={selectedPredictions}
      />
    </Box>
  );
};

const ResultsAnalysis: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [currentTab, setCurrentTab] = useState(0);
  
  // If id is 'latest', show the history/past results list
  if (id === 'latest') {
    return <PastResultsList />;
  }

  // Fetch prediction results
  const { data: prediction, isLoading, error } = useQuery({
    queryKey: ['prediction', id],
    queryFn: () => predictionService.getPrediction(id!),
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === 'running' || data?.status === 'pending' ? 5000 : false;
    },
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleDownloadPDB = () => {
    if (prediction?.result_file) {
      window.open(`/api/v1/predictions/${id}/download/pdb`, '_blank');
    }
  };

  const handleDownloadJSON = () => {
    if (prediction) {
      const dataStr = JSON.stringify(prediction, null, 2);
      const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
      const exportFileDefaultName = `prediction_${id}_results.json`;
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    }
  };

  const handleDownloadTrajectory = () => {
    window.open(`/api/v1/predictions/${id}/download/trajectory`, '_blank');
  };

  const handleCompare = () => {
    navigate(`/compare?baseline=${id}`);
  };

  const getStatusChip = (status: string) => {
    const statusConfig = {
      completed: { label: 'Completed', color: 'success' as const, icon: <CheckCircle /> },
      running: { label: 'Running', color: 'primary' as const, icon: <CircularProgress size={16} /> },
      failed: { label: 'Failed', color: 'error' as const, icon: <ErrorIcon /> },
      pending: { label: 'Pending', color: 'default' as const, icon: null },
      paused: { label: 'Paused', color: 'warning' as const, icon: <Warning /> },
    };

    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.pending;

    return (
      <Chip
        label={config.label}
        color={config.color}
        icon={config.icon || undefined}
        size="small"
      />
    );
  };

  const getQualityChip = (rmsd?: number | null) => {
    if (rmsd === undefined || rmsd === null) return null;

    if (rmsd < 2) {
      return <Chip label="Excellent" color="success" size="small" />;
    } else if (rmsd < 4) {
      return <Chip label="Good" color="info" size="small" />;
    } else if (rmsd < 5) {
      return <Chip label="Acceptable" color="warning" size="small" />;
    } else {
      return <Chip label="Poor" color="error" size="small" />;
    }
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <ResultsAnalysisSkeleton />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">
          Failed to load prediction results. {(error as Error).message}
        </Alert>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/dashboard')}
          sx={{ mt: 2 }}
        >
          Back to Dashboard
        </Button>
      </Box>
    );
  }

  if (!prediction) {
    return (
      <Box p={3}>
        <Alert severity="warning">Prediction not found.</Alert>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/dashboard')}
          sx={{ mt: 2 }}
        >
          Back to Dashboard
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Stack spacing={2}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start">
            <Box>
              <Button
                startIcon={<ArrowBack />}
                onClick={() => navigate('/dashboard')}
                sx={{ mb: 1 }}
              >
                Back to Dashboard
              </Button>
              <Typography variant="h4" gutterBottom>
                Results Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Prediction ID: {prediction.id}
              </Typography>
            </Box>

            <Stack direction="row" spacing={1}>
              {getStatusChip(prediction.status)}
              {prediction.best_rmsd !== undefined && prediction.best_rmsd !== null && getQualityChip(prediction.best_rmsd)}
            </Stack>
          </Box>

          <Stack direction="row" spacing={2}>
            <Typography variant="body2">
              <strong>Sequence:</strong> {(prediction.protein_sequence || prediction.sequence || '').substring(0, 50)}
              {(prediction.protein_sequence || prediction.sequence || '').length > 50 ? '...' : ''} ({(prediction.protein_sequence || prediction.sequence || '').length} residues)
            </Typography>
            <Typography variant="body2">
              <strong>Created:</strong> {new Date(prediction.created_at).toLocaleString()}
            </Typography>
            {prediction.completed_at && (
              <Typography variant="body2">
                <strong>Completed:</strong> {new Date(prediction.completed_at).toLocaleString()}
              </Typography>
            )}
          </Stack>

          {/* Action Buttons */}
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleDownloadPDB}
              disabled={!prediction.result_file}
            >
              Download PDB
            </Button>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleDownloadJSON}
            >
              Download JSON
            </Button>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleDownloadTrajectory}
            >
              Download Trajectory
            </Button>
            <Button
              variant="outlined"
              startIcon={<Compare />}
              onClick={handleCompare}
            >
              Compare
            </Button>
            <Button
              variant="outlined"
              startIcon={<Share />}
              onClick={() => {
                navigator.clipboard.writeText(window.location.href);
                // Could add a snackbar notification here
              }}
            >
              Share
            </Button>
          </Stack>
        </Stack>
      </Paper>

      {/* Tabs */}
      <Paper>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="results tabs">
            <Tab label="Summary" {...a11yProps(0)} />
            <Tab label="Detailed Metrics" {...a11yProps(1)} />
            <Tab label="Trajectory" {...a11yProps(2)} />
            <Tab label="Geometric Analysis" {...a11yProps(3)} />
          </Tabs>
        </Box>

        <Box sx={{ p: 3 }}>
          <TabPanel value={currentTab} index={0}>
            <SummaryTab prediction={prediction} />
          </TabPanel>

          <TabPanel value={currentTab} index={1}>
            <DetailedMetricsTab prediction={prediction} />
          </TabPanel>

          <TabPanel value={currentTab} index={2}>
            <TrajectoryTab predictionId={prediction.id} />
          </TabPanel>

          <TabPanel value={currentTab} index={3}>
            <GeometricAnalysisTab predictionId={prediction.id} />
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  );
};

export default ResultsAnalysis;
