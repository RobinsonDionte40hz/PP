import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  ToggleButtonGroup,
  ToggleButton,
  IconButton,
  Tooltip,
  Button,
  Chip,
} from '@mui/material';
import {
  ViewModule as CardViewIcon,
  ViewList as TableViewIcon,
  Refresh as RefreshIcon,
  CompareArrows as CompareIcon,
} from '@mui/icons-material';
import { usePredictions } from '../hooks/usePredictions';
import HistoryFilters from '../components/history/HistoryFilters';
import HistoryCardView from '../components/history/HistoryCardView';
import HistoryTableView from '../components/history/HistoryTableView';
import ComparisonModal from '../components/history/ComparisonModal';
import ErrorAlert from '../components/common/ErrorAlert';
import { TableSkeleton } from '../components/common/skeletons';

// Note: VirtualizedHistoryTable component available in codebase but needs react-window ESM fix for production use

type ViewMode = 'card' | 'table';

interface FilterState {
  status?: string;
  dateRange?: [Date | null, Date | null];
  qualityLevel?: string;
  sortBy: 'created_at' | 'energy' | 'rmsd' | 'iterations';
  sortOrder: 'asc' | 'desc';
  searchQuery: string;
}

const HistoryBrowser: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('card');
  const [filters, setFilters] = useState<FilterState>({
    sortBy: 'created_at',
    sortOrder: 'desc',
    searchQuery: '',
  });
  const [selectedPredictions, setSelectedPredictions] = useState<string[]>([]);
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);

  const { data: predictions, isLoading, isError, error, refetch } = usePredictions({
    status: filters.status,
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

  // Apply client-side filtering and sorting (memoized for performance)
  const filteredAndSortedPredictions = React.useMemo(() => {
    if (!predictions) return [];

    let filtered = [...predictions];

    // Search filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      filtered = filtered.filter(
        (p) =>
          p.id.toLowerCase().includes(query) ||
          p.protein_sequence?.toLowerCase().includes(query) ||
          p.sequence?.toLowerCase().includes(query)
      );
    }

    // Quality filter (placeholder - would need quality calculation logic)
    // TODO: Implement quality filtering based on metrics when quality data is available
    // if (filters.qualityLevel) { ... }

    // Date range filter
    if (filters.dateRange && filters.dateRange[0] && filters.dateRange[1]) {
      const [start, end] = filters.dateRange;
      filtered = filtered.filter((p) => {
        if (!p.created_at) return false;
        const date = new Date(p.created_at);
        return date >= start && date <= end;
      });
    }

    // Sorting
    filtered.sort((a, b) => {
      let aValue: string | number | Date;
      let bValue: string | number | Date;

      switch (filters.sortBy) {
        case 'created_at':
          aValue = a.created_at ? new Date(a.created_at).getTime() : 0;
          bValue = b.created_at ? new Date(b.created_at).getTime() : 0;
          break;
        case 'energy':
          aValue = a.best_energy ?? Infinity;
          bValue = b.best_energy ?? Infinity;
          break;
        case 'rmsd':
          aValue = a.best_rmsd ?? Infinity;
          bValue = b.best_rmsd ?? Infinity;
          break;
        case 'iterations':
          aValue = a.current_iteration ?? 0;
          bValue = b.current_iteration ?? 0;
          break;
        default:
          return 0;
      }

      const comparison = aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      return filters.sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [predictions, filters]);

  if (isLoading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Prediction History
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Loading prediction history...
          </Typography>
        </Box>
        <TableSkeleton rows={10} columns={6} />
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Prediction History
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Browse and compare past structure predictions
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
                variant="contained"
                startIcon={<CompareIcon />}
                onClick={handleCompare}
                disabled={selectedPredictions.length < 2}
              >
                Compare
              </Button>
            </>
          )}
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} color="primary">
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

      {/* Error Alert */}
      {isError && (
        <Box sx={{ mb: 3 }}>
          <ErrorAlert
            message={error instanceof Error ? error.message : 'Failed to load predictions'}
          />
        </Box>
      )}

      {/* Filters */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <HistoryFilters filters={filters} onFilterChange={handleFilterChange} />
      </Paper>

      {/* Results Count */}
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Showing {filteredAndSortedPredictions.length} of {predictions?.length || 0} predictions
        </Typography>
      </Box>

      {/* Content */}
      {viewMode === 'card' ? (
        <HistoryCardView
          predictions={filteredAndSortedPredictions}
          selectedPredictions={selectedPredictions}
          onSelectPrediction={handleSelectPrediction}
        />
      ) : (
        // Use standard table view (virtualization available but needs react-window fix)
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
    </Container>
  );
};

export default HistoryBrowser;
