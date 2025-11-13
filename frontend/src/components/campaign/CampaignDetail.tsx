import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Stack,
  Chip,
  Button,
  Divider,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  ArrowBack as BackIcon,
  Download as DownloadIcon,
  PlayArrow as ResumeIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
} from '@mui/icons-material';
import { useCampaign } from '../../hooks/useCampaigns';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';
import PredictionStatusBadge from '../common/PredictionStatusBadge';
import MetricCard from '../common/MetricCard';
import PhaseProgress from './PhaseProgress';
import ProteinResultsTable from './ProteinResultsTable';
import CampaignStatistics from './CampaignStatistics';
import { formatDistanceToNow } from 'date-fns';

interface CampaignDetailProps {
  campaignId: string;
  onBack: () => void;
}

const CampaignDetail: React.FC<CampaignDetailProps> = ({ campaignId, onBack }) => {
  const { data: campaign, isLoading, isError, error } = useCampaign(campaignId);

  const handleResume = () => {
    // TODO: Implement resume campaign API call
    console.log('Resume campaign:', campaignId);
  };

  const handlePause = () => {
    // TODO: Implement pause campaign API call
    console.log('Pause campaign:', campaignId);
  };

  const handleStop = () => {
    // TODO: Implement stop campaign API call
    console.log('Stop campaign:', campaignId);
  };

  const handleExport = () => {
    // TODO: Implement export campaign results
    console.log('Export campaign:', campaignId);
  };

  if (isLoading) {
    return <LoadingSpinner message="Loading campaign details..." />;
  }

  if (isError) {
    return (
      <ErrorAlert
        message={error instanceof Error ? error.message : 'Failed to load campaign details'}
      />
    );
  }

  if (!campaign) {
    return (
      <Alert severity="warning">Campaign not found</Alert>
    );
  }

  const calculateProgress = () => {
    if (!campaign.statistics) return 0;
    const total = campaign.statistics.total_proteins || 0;
    const completed =
      (campaign.statistics.successful_predictions || 0) +
      (campaign.statistics.failed_predictions || 0);
    return total > 0 ? (completed / total) * 100 : 0;
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Button
            startIcon={<BackIcon />}
            onClick={onBack}
            sx={{ mb: 2 }}
          >
            Back to Campaigns
          </Button>
          <Typography variant="h5" component="h2" gutterBottom>
            {campaign.name || `Campaign ${campaign.id.slice(0, 8)}`}
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mt: 1 }}>
            <PredictionStatusBadge status={campaign.status} />
            <Chip
              label={`Phase ${campaign.current_phase || 1}/${campaign.total_phases || 4}`}
              size="small"
              color="primary"
            />
            <Typography variant="body2" color="text.secondary">
              Created {campaign.created_at
                ? formatDistanceToNow(new Date(campaign.created_at), { addSuffix: true })
                : 'N/A'}
            </Typography>
          </Box>
        </Box>

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          {campaign.status === 'paused' && (
            <Button
              variant="contained"
              startIcon={<ResumeIcon />}
              onClick={handleResume}
              color="success"
            >
              Resume
            </Button>
          )}
          {campaign.status === 'running' && (
            <>
              <Button
                variant="outlined"
                startIcon={<PauseIcon />}
                onClick={handlePause}
                color="warning"
              >
                Pause
              </Button>
              <Button
                variant="outlined"
                startIcon={<StopIcon />}
                onClick={handleStop}
                color="error"
              >
                Stop
              </Button>
            </>
          )}
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={campaign.status !== 'completed'}
          >
            Export
          </Button>
        </Box>
      </Box>

      {/* Overall Progress */}
      {campaign.status === 'running' && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Overall Progress
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <LinearProgress
              variant="determinate"
              value={calculateProgress()}
              sx={{ flex: 1, height: 8, borderRadius: 4 }}
            />
            <Typography variant="body2" fontWeight="medium">
              {calculateProgress().toFixed(1)}%
            </Typography>
          </Box>
        </Paper>
      )}

      {/* Key Metrics */}
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={3}
        sx={{ mb: 3 }}
      >
        <Box sx={{ flex: 1 }}>
          <MetricCard
            title="Total Proteins"
            value={campaign.statistics?.total_proteins || 0}
          />
        </Box>
        <Box sx={{ flex: 1 }}>
          <MetricCard
            title="Successful"
            value={campaign.statistics?.successful_predictions || 0}
          />
        </Box>
        <Box sx={{ flex: 1 }}>
          <MetricCard
            title="Failed"
            value={campaign.statistics?.failed_predictions || 0}
          />
        </Box>
        <Box sx={{ flex: 1 }}>
          <MetricCard
            title="Avg RMSD"
            value={
              campaign.statistics?.average_rmsd
                ? `${campaign.statistics.average_rmsd.toFixed(2)} Å`
                : 'N/A'
            }
          />
        </Box>
      </Stack>

      {/* Phase Progress */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Phase Progress
        </Typography>
        <Divider sx={{ mb: 3 }} />
        <PhaseProgress
          currentPhase={campaign.current_phase || 1}
          totalPhases={campaign.total_phases || 4}
          phaseData={campaign.phase_results || []}
        />
      </Paper>

      {/* Statistical Analysis */}
      {campaign.statistics && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Statistical Analysis
          </Typography>
          <Divider sx={{ mb: 3 }} />
          <CampaignStatistics statistics={campaign.statistics} />
        </Paper>
      )}

      {/* Protein Results */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Protein Results
        </Typography>
        <Divider sx={{ mb: 3 }} />
        <ProteinResultsTable
          proteins={campaign.proteins || []}
          campaignId={campaignId}
        />
      </Paper>
    </Box>
  );
};

export default CampaignDetail;
