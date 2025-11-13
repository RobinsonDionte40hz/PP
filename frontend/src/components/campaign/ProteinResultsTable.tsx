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
  IconButton,
  Tooltip,
  Typography,
  Chip,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import PredictionStatusBadge from '../common/PredictionStatusBadge';
import QualityBadge from '../common/QualityBadge';

interface ProteinResult {
  id: string;
  protein_name?: string;
  sequence?: string;
  status: string;
  rmsd?: number;
  energy?: number;
  quality?: string;
  phase?: number;
  created_at?: string;
}

interface ProteinResultsTableProps {
  proteins: ProteinResult[];
  campaignId: string;
}

const ProteinResultsTable: React.FC<ProteinResultsTableProps> = ({
  proteins,
}) => {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleViewResult = (proteinId: string) => {
    navigate(`/results/${proteinId}`);
  };

  const handleDownload = (proteinId: string) => {
    // TODO: Implement download single protein result
    console.log('Download protein:', proteinId);
  };

  const paginatedProteins = proteins.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  return (
    <Box>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Protein</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Phase</TableCell>
              <TableCell>RMSD</TableCell>
              <TableCell>Energy</TableCell>
              <TableCell>Quality</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedProteins.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                    No protein results yet
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              paginatedProteins.map((protein) => (
                <TableRow key={protein.id} hover>
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {protein.protein_name || `Protein ${protein.id.slice(0, 8)}`}
                    </Typography>
                    {protein.sequence && (
                      <Typography variant="caption" color="text.secondary">
                        {protein.sequence.length} residues
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <PredictionStatusBadge status={protein.status as 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'} />
                  </TableCell>
                  <TableCell>
                    {protein.phase && (
                      <Chip label={`Phase ${protein.phase}`} size="small" variant="outlined" />
                    )}
                  </TableCell>
                  <TableCell>
                    {protein.rmsd !== undefined ? (
                      <Typography variant="body2">{protein.rmsd.toFixed(2)} Å</Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        N/A
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {protein.energy !== undefined ? (
                      <Typography variant="body2">
                        {protein.energy.toFixed(2)} kcal/mol
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        N/A
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {protein.quality && <QualityBadge quality={protein.quality as 'excellent' | 'good' | 'acceptable' | 'poor'} />}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="View details">
                      <IconButton
                        size="small"
                        onClick={() => handleViewResult(protein.id)}
                        color="primary"
                      >
                        <ViewIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download">
                      <IconButton
                        size="small"
                        onClick={() => handleDownload(protein.id)}
                        disabled={protein.status !== 'completed'}
                      >
                        <DownloadIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={proteins.length}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[5, 10, 25, 50]}
      />
    </Box>
  );
};

export default ProteinResultsTable;
