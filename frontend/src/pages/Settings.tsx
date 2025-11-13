import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Tabs,
  Tab,
  Button,
  Stack,
} from '@mui/material';
import {
  Save as SaveIcon,
  RestartAlt as ResetIcon,
} from '@mui/icons-material';
import SystemConfigTab from '../components/settings/SystemConfigTab';
import VisualizationTab from '../components/settings/VisualizationTab';
import NotificationsTab from '../components/settings/NotificationsTab';
import AdvancedTab from '../components/settings/AdvancedTab';

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
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface SettingsState {
  system: {
    defaultIterations: number;
    defaultAgents: number;
    defaultConsciousness: number;
    defaultConsistency: number;
    enableQCPP: boolean;
    qcppConfig: string;
    checkpointInterval: number;
    autoSaveCheckpoints: boolean;
  };
  visualization: {
    defaultRepresentation: string;
    defaultColorScheme: string;
    backgroundColor: string;
    enableSmoothAnimation: boolean;
    showHydrogenBonds: boolean;
    showGeometricPatterns: boolean;
    qualityLevel: string;
  };
  notifications: {
    enableDesktopNotifications: boolean;
    notifyOnCompletion: boolean;
    notifyOnError: boolean;
    notifyOnMilestone: boolean;
    emailNotifications: boolean;
    emailAddress: string;
  };
  advanced: {
    enableDebugMode: boolean;
    logLevel: string;
    maxConcurrentPredictions: number;
    cacheEnabled: boolean;
    cacheSize: number;
    apiTimeout: number;
  };
}

const defaultSettings: SettingsState = {
  system: {
    defaultIterations: 1000,
    defaultAgents: 10,
    defaultConsciousness: 8.0,
    defaultConsistency: 0.6,
    enableQCPP: true,
    qcppConfig: 'default',
    checkpointInterval: 100,
    autoSaveCheckpoints: true,
  },
  visualization: {
    defaultRepresentation: 'cartoon',
    defaultColorScheme: 'chainid',
    backgroundColor: '#000000',
    enableSmoothAnimation: true,
    showHydrogenBonds: false,
    showGeometricPatterns: true,
    qualityLevel: 'high',
  },
  notifications: {
    enableDesktopNotifications: true,
    notifyOnCompletion: true,
    notifyOnError: true,
    notifyOnMilestone: false,
    emailNotifications: false,
    emailAddress: '',
  },
  advanced: {
    enableDebugMode: false,
    logLevel: 'info',
    maxConcurrentPredictions: 5,
    cacheEnabled: true,
    cacheSize: 1000,
    apiTimeout: 30000,
  },
};

const Settings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState<SettingsState>(() => {
    // Load settings from localStorage
    const saved = localStorage.getItem('app-settings');
    return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
  });
  const [hasChanges, setHasChanges] = useState(false);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSettingChange = (category: keyof SettingsState, key: string, value: any) => {
    setSettings((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value,
      },
    }));
    setHasChanges(true);
  };

  const handleSave = () => {
    // Save to localStorage
    localStorage.setItem('app-settings', JSON.stringify(settings));
    setHasChanges(false);

    // TODO: Optionally sync with backend API
    console.log('Settings saved:', settings);
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      setSettings(defaultSettings);
      localStorage.removeItem('app-settings');
      setHasChanges(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Settings
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Configure application preferences and behavior
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<ResetIcon />}
            onClick={handleReset}
            disabled={!hasChanges}
          >
            Reset to Defaults
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSave}
            disabled={!hasChanges}
          >
            Save Changes
          </Button>
        </Stack>
      </Box>

      {/* Settings Tabs */}
      <Paper>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="settings tabs"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="System Configuration" id="settings-tab-0" />
          <Tab label="Visualization" id="settings-tab-1" />
          <Tab label="Notifications" id="settings-tab-2" />
          <Tab label="Advanced" id="settings-tab-3" />
        </Tabs>

        {/* System Configuration Tab */}
        <TabPanel value={tabValue} index={0}>
          <SystemConfigTab
            settings={settings.system}
            onChange={(key, value) => handleSettingChange('system', key, value)}
          />
        </TabPanel>

        {/* Visualization Tab */}
        <TabPanel value={tabValue} index={1}>
          <VisualizationTab
            settings={settings.visualization}
            onChange={(key, value) => handleSettingChange('visualization', key, value)}
          />
        </TabPanel>

        {/* Notifications Tab */}
        <TabPanel value={tabValue} index={2}>
          <NotificationsTab
            settings={settings.notifications}
            onChange={(key, value) => handleSettingChange('notifications', key, value)}
          />
        </TabPanel>

        {/* Advanced Tab */}
        <TabPanel value={tabValue} index={3}>
          <AdvancedTab
            settings={settings.advanced}
            onChange={(key, value) => handleSettingChange('advanced', key, value)}
          />
        </TabPanel>
      </Paper>

      {/* Unsaved Changes Warning */}
      {hasChanges && (
        <Paper
          sx={{
            mt: 3,
            p: 2,
            bgcolor: 'warning.light',
            color: 'warning.contrastText',
          }}
        >
          <Typography variant="body2">
            You have unsaved changes. Don't forget to save your settings before leaving this page.
          </Typography>
        </Paper>
      )}
    </Container>
  );
};

export default Settings;
