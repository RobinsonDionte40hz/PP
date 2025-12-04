import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Checkbox,
  IconButton,
  Tooltip,
  Typography,
  Stack,
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

interface HistoryTableViewProps {
  predictions: PredictionResponse[];
  selectedPredictions: string[];
  onSelectPrediction: (predictionId: string) => void;
}

const HistoryTableView: React.FC<HistoryTableViewProps> = ({
  predictions,
  selectedPredictions,
  onSelectPrediction,
}) => {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

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

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const allIds = paginatedPredictions.map((p) => p.id);
      allIds.forEach((id) => {
        if (!selectedPredictions.includes(id)) {
          onSelectPrediction(id);
        }
      });
    } else {
      paginatedPredictions.forEach((p) => {
        if (selectedPredictions.includes(p.id)) {
          onSelectPrediction(p.id);
        }
      });
    }
  };

  const paginatedPredictions = predictions.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  const isAllSelected =
    paginatedPredictions.length > 0 &&
    paginatedPredictions.every((p) => selectedPredictions.includes(p.id));

  const isSomeSelected =
    paginatedPredictions.some((p) => selectedPredictions.includes(p.id)) && !isAllSelected;

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
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  checked={isAllSelected}
                  indeterminate={isSomeSelected}
                  onChange={handleSelectAll}
                />
              </TableCell>
              <TableCell>ID</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Residues</TableCell>
              <TableCell>Energy</TableCell>
              <TableCell>RMSD</TableCell>
              <TableCell>Iterations</TableCell>
              <TableCell>Created</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedPredictions.map((prediction) => (
              <TableRow
                key={prediction.id}
                hover
                selected={selectedPredictions.includes(prediction.id)}
              >
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedPredictions.includes(prediction.id)}
                    onChange={() => onSelectPrediction(prediction.id)}
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontWeight="medium">
                    {prediction.id.slice(0, 12)}...
                  </Typography>
                </TableCell>
                <TableCell>
                  <PredictionStatusBadge status={prediction.status} />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {prediction.sequence?.length || prediction.protein_sequence?.length || 0}
                  </Typography>
                </TableCell>
                <TableCell>
                  {prediction.best_energy !== undefined ? (
                    <Typography variant="body2">
                      {prediction.best_energy.toFixed(2)}
                    </Typography>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      N/A
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  {prediction.best_rmsd !== undefined ? (
                    <Typography variant="body2">{prediction.best_rmsd.toFixed(2)}</Typography>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      N/A
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {prediction.current_iteration}/{prediction.total_iterations}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" color="text.secondary">
                    {prediction.created_at
                      ? formatDistanceToNow(new Date(prediction.created_at), {
                          addSuffix: true,
                        })
                      : 'Unknown'}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={0.5} justifyContent="flex-end">
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
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={predictions.length}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[10, 25, 50, 100]}
      />
    </Box>
  );
};

export default HistoryTableView;
