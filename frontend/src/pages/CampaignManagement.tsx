import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Button,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useCampaigns } from '../hooks/useCampaigns';
import CampaignList from '../components/campaign/CampaignList';
import CampaignDetail from '../components/campaign/CampaignDetail';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

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
      id={`campaign-tabpanel-${index}`}
      aria-labelledby={`campaign-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const CampaignManagement: React.FC = () => {
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [selectedCampaignId, setSelectedCampaignId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const { data: campaigns, isLoading, isError, error, refetch } = useCampaigns({
    status: statusFilter === 'all' ? undefined : statusFilter,
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    if (newValue === 0) {
      setSelectedCampaignId(null);
    }
  };

  const handleCreateCampaign = () => {
    // Navigate to campaign creation form (can be implemented later)
    navigate('/dashboard/predictions/new?mode=campaign');
  };

  const handleViewCampaign = (campaignId: string) => {
    setSelectedCampaignId(campaignId);
    setTabValue(1);
  };

  const handleRefresh = () => {
    refetch();
  };

  if (isLoading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <LoadingSpinner message="Loading campaigns..." />
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Campaign Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage multi-protein structure prediction campaigns
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Tooltip title="Refresh campaigns">
            <IconButton onClick={handleRefresh} color="primary">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleCreateCampaign}
          >
            New Campaign
          </Button>
        </Box>
      </Box>

      {/* Error Alert */}
      {isError && (
        <Box sx={{ mb: 3 }}>
          <ErrorAlert
            message={error instanceof Error ? error.message : 'Failed to load campaigns'}
          />
        </Box>
      )}

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="campaign tabs"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="All Campaigns" id="campaign-tab-0" />
          <Tab label="Campaign Details" id="campaign-tab-1" disabled={!selectedCampaignId} />
        </Tabs>

        {/* Campaign List Tab */}
        <TabPanel value={tabValue} index={0}>
          <CampaignList
            campaigns={campaigns || []}
            onViewCampaign={handleViewCampaign}
            statusFilter={statusFilter}
            onStatusFilterChange={setStatusFilter}
          />
        </TabPanel>

        {/* Campaign Detail Tab */}
        <TabPanel value={tabValue} index={1}>
          {selectedCampaignId ? (
            <CampaignDetail
              campaignId={selectedCampaignId}
              onBack={() => setTabValue(0)}
            />
          ) : (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                Select a campaign to view details
              </Typography>
            </Box>
          )}
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default CampaignManagement;
