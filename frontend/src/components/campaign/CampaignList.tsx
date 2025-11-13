import React, { useState } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  IconButton,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  InputAdornment,
  Tooltip,
  Stack,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  PlayArrow as ResumeIcon,
  Pause as PauseIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import type { Campaign } from '../../types/api';
import { formatDistanceToNow } from 'date-fns';
import PredictionStatusBadge from '../common/PredictionStatusBadge';
import ConfirmDialog from '../common/ConfirmDialog';

interface CampaignListProps {
  campaigns: Campaign[];
  onViewCampaign: (campaignId: string) => void;
  statusFilter: string;
  onStatusFilterChange: (status: string) => void;
}

const CampaignList: React.FC<CampaignListProps> = ({
  campaigns,
  onViewCampaign,
  statusFilter,
  onStatusFilterChange,
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedCampaignId, setSelectedCampaignId] = useState<string | null>(null);

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleDeleteClick = (campaignId: string) => {
    setSelectedCampaignId(campaignId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    // TODO: Implement delete campaign API call
    console.log('Delete campaign:', selectedCampaignId);
    setDeleteDialogOpen(false);
    setSelectedCampaignId(null);
  };

  const handleResumeCampaign = (campaignId: string) => {
    // TODO: Implement resume campaign API call
    console.log('Resume campaign:', campaignId);
  };

  const handlePauseCampaign = (campaignId: string) => {
    // TODO: Implement pause campaign API call
    console.log('Pause campaign:', campaignId);
  };

  const handleDownload = (campaignId: string) => {
    // TODO: Implement download campaign results
    console.log('Download campaign:', campaignId);
  };

  // Filter campaigns by search query
  const filteredCampaigns = campaigns.filter((campaign) =>
    campaign.name?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const paginatedCampaigns = filteredCampaigns.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  const getPhaseLabel = (currentPhase: number, totalPhases: number) => {
    return `Phase ${currentPhase}/${totalPhases}`;
  };

  const getSuccessRate = (campaign: Campaign) => {
    if (!campaign.statistics) return 'N/A';
    const total = campaign.statistics.total_proteins || 0;
    const successful = campaign.statistics.successful_predictions || 0;
    if (total === 0) return '0%';
    return `${Math.round((successful / total) * 100)}%`;
  };

  return (
    <Box>
      {/* Filters */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
        <TextField
          placeholder="Search campaigns..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          size="small"
          sx={{ flex: 1, maxWidth: 400 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Status</InputLabel>
          <Select
            value={statusFilter}
            label="Status"
            onChange={(e) => onStatusFilterChange(e.target.value)}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="running">Running</MenuItem>
            <MenuItem value="completed">Completed</MenuItem>
            <MenuItem value="failed">Failed</MenuItem>
            <MenuItem value="paused">Paused</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Campaign Table */}
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Campaign Name</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Phase</TableCell>
              <TableCell>Proteins</TableCell>
              <TableCell>Success Rate</TableCell>
              <TableCell>Created</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedCampaigns.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                    {searchQuery ? 'No campaigns found matching your search' : 'No campaigns yet'}
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              paginatedCampaigns.map((campaign) => (
                <TableRow key={campaign.id} hover>
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {campaign.name || `Campaign ${campaign.id.slice(0, 8)}`}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <PredictionStatusBadge status={campaign.status} />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={getPhaseLabel(
                        campaign.current_phase || 1,
                        campaign.total_phases || 4
                      )}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {campaign.statistics?.total_proteins || 0}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {getSuccessRate(campaign)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {campaign.created_at
                        ? formatDistanceToNow(new Date(campaign.created_at), {
                            addSuffix: true,
                          })
                        : 'N/A'}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                      <Tooltip title="View details">
                        <IconButton
                          size="small"
                          onClick={() => onViewCampaign(campaign.id)}
                          color="primary"
                        >
                          <ViewIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      {campaign.status === 'paused' && (
                        <Tooltip title="Resume campaign">
                          <IconButton
                            size="small"
                            onClick={() => handleResumeCampaign(campaign.id)}
                            color="success"
                          >
                            <ResumeIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      {campaign.status === 'running' && (
                        <Tooltip title="Pause campaign">
                          <IconButton
                            size="small"
                            onClick={() => handlePauseCampaign(campaign.id)}
                            color="warning"
                          >
                            <PauseIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      <Tooltip title="Download results">
                        <IconButton
                          size="small"
                          onClick={() => handleDownload(campaign.id)}
                          disabled={campaign.status !== 'completed'}
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete campaign">
                        <IconButton
                          size="small"
                          onClick={() => handleDeleteClick(campaign.id)}
                          color="error"
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      <TablePagination
        component="div"
        count={filteredCampaigns.length}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[5, 10, 25, 50]}
      />

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        open={deleteDialogOpen}
        onCancel={() => setDeleteDialogOpen(false)}
        onConfirm={handleDeleteConfirm}
        title="Delete Campaign"
        message="Are you sure you want to delete this campaign? This action cannot be undone and all associated data will be lost."
      />
    </Box>
  );
};

export default CampaignList;
