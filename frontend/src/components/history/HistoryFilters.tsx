import React from 'react';
import {
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';

interface FilterState {
  status?: string;
  dateRange?: [Date | null, Date | null];
  qualityLevel?: string;
  sortBy: 'created_at' | 'energy' | 'rmsd' | 'iterations';
  sortOrder: 'asc' | 'desc';
  searchQuery: string;
}

interface HistoryFiltersProps {
  filters: FilterState;
  onFilterChange: (filters: Partial<FilterState>) => void;
}

const HistoryFilters: React.FC<HistoryFiltersProps> = ({ filters, onFilterChange }) => {
  return (
    <Stack spacing={3}>
      {/* Search Bar */}
      <TextField
        placeholder="Search by ID or sequence..."
        value={filters.searchQuery}
        onChange={(e) => onFilterChange({ searchQuery: e.target.value })}
        fullWidth
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
      />

      {/* Filter Row */}
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
        {/* Status Filter */}
        <FormControl size="small" sx={{ flex: 1, minWidth: 150 }}>
          <InputLabel>Status</InputLabel>
          <Select
            value={filters.status || 'all'}
            label="Status"
            onChange={(e) =>
              onFilterChange({ status: e.target.value === 'all' ? undefined : e.target.value })
            }
          >
            <MenuItem value="all">All Status</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="running">Running</MenuItem>
            <MenuItem value="completed">Completed</MenuItem>
            <MenuItem value="failed">Failed</MenuItem>
            <MenuItem value="cancelled">Cancelled</MenuItem>
          </Select>
        </FormControl>

        {/* Quality Filter */}
        <FormControl size="small" sx={{ flex: 1, minWidth: 150 }}>
          <InputLabel>Quality</InputLabel>
          <Select
            value={filters.qualityLevel || 'all'}
            label="Quality"
            onChange={(e) =>
              onFilterChange({
                qualityLevel: e.target.value === 'all' ? undefined : e.target.value,
              })
            }
          >
            <MenuItem value="all">All Quality</MenuItem>
            <MenuItem value="excellent">Excellent</MenuItem>
            <MenuItem value="good">Good</MenuItem>
            <MenuItem value="acceptable">Acceptable</MenuItem>
            <MenuItem value="poor">Poor</MenuItem>
          </Select>
        </FormControl>

        {/* Sort By */}
        <FormControl size="small" sx={{ flex: 1, minWidth: 150 }}>
          <InputLabel>Sort By</InputLabel>
          <Select
            value={filters.sortBy}
            label="Sort By"
            onChange={(e) =>
              onFilterChange({
                sortBy: e.target.value as 'created_at' | 'energy' | 'rmsd' | 'iterations',
              })
            }
          >
            <MenuItem value="created_at">Date Created</MenuItem>
            <MenuItem value="energy">Energy</MenuItem>
            <MenuItem value="rmsd">RMSD</MenuItem>
            <MenuItem value="iterations">Iterations</MenuItem>
          </Select>
        </FormControl>

        {/* Sort Order */}
        <FormControl size="small" sx={{ flex: 1, minWidth: 120 }}>
          <InputLabel>Order</InputLabel>
          <Select
            value={filters.sortOrder}
            label="Order"
            onChange={(e) => onFilterChange({ sortOrder: e.target.value as 'asc' | 'desc' })}
          >
            <MenuItem value="asc">Ascending</MenuItem>
            <MenuItem value="desc">Descending</MenuItem>
          </Select>
        </FormControl>
      </Stack>
    </Stack>
  );
};

export default HistoryFilters;
