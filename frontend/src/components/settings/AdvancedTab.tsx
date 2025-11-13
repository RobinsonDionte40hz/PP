import React from 'react';
import {
  Box,
  Typography,
  FormControlLabel,
  Switch,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  Alert,
  Stack,
  Paper,
  Chip,
} from '@mui/material';
import { Warning as WarningIcon } from '@mui/icons-material';

interface AdvancedTabProps {
  settings: {
    enableDebugMode: boolean;
    logLevel: string;
    maxConcurrentPredictions: number;
    cacheEnabled: boolean;
    cacheSize: number;
    apiTimeout: number;
  };
  onChange: (key: string, value: boolean | string | number) => void;
}

const AdvancedTab: React.FC<AdvancedTabProps> = ({ settings, onChange }) => {
  const calculateCacheMemory = () => {
    // Rough estimate: ~1KB per cache entry
    return (settings.cacheSize * 1) / 1024;
  };

  return (
    <Box sx={{ px: 3 }}>
      <Alert severity="warning" icon={<WarningIcon />} sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Warning:</strong> Advanced settings can affect application performance and
          behavior. Only modify these if you understand their implications.
        </Typography>
      </Alert>

      {/* Debug & Logging */}
      <Typography variant="h6" gutterBottom>
        Debug & Logging
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.enableDebugMode}
              onChange={(e) => onChange('enableDebugMode', e.target.checked)}
            />
          }
          label={
            <Box>
              <Typography variant="body2">Enable Debug Mode</Typography>
              <Typography variant="caption" color="text.secondary">
                Shows additional console logs and debugging information
              </Typography>
            </Box>
          }
        />

        <FormControl fullWidth>
          <InputLabel>Log Level</InputLabel>
          <Select
            value={settings.logLevel}
            label="Log Level"
            onChange={(e) => onChange('logLevel', e.target.value)}
          >
            <MenuItem value="error">Error Only</MenuItem>
            <MenuItem value="warn">Warning & Error</MenuItem>
            <MenuItem value="info">Info, Warning & Error</MenuItem>
            <MenuItem value="debug">All (Debug Mode)</MenuItem>
          </Select>
        </FormControl>
      </Stack>

      {/* Performance Settings */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Performance Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        <Box>
          <Typography variant="body2" gutterBottom>
            Max Concurrent Predictions
          </Typography>
          <TextField
            type="number"
            value={settings.maxConcurrentPredictions}
            onChange={(e) => onChange('maxConcurrentPredictions', parseInt(e.target.value))}
            fullWidth
            inputProps={{ min: 1, max: 20 }}
            helperText="Maximum number of predictions that can run simultaneously"
          />
        </Box>

        <Box>
          <Typography variant="body2" gutterBottom>
            API Request Timeout (ms)
          </Typography>
          <TextField
            type="number"
            value={settings.apiTimeout}
            onChange={(e) => onChange('apiTimeout', parseInt(e.target.value))}
            fullWidth
            inputProps={{ min: 5000, max: 120000, step: 1000 }}
            helperText="Timeout for API requests in milliseconds (5000-120000)"
          />
        </Box>
      </Stack>

      {/* Cache Settings */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Cache Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.cacheEnabled}
              onChange={(e) => onChange('cacheEnabled', e.target.checked)}
            />
          }
          label={
            <Box>
              <Typography variant="body2">Enable Response Caching</Typography>
              <Typography variant="caption" color="text.secondary">
                Cache API responses to improve performance
              </Typography>
            </Box>
          }
        />

        <Box>
          <Typography variant="body2" gutterBottom>
            Cache Size (entries)
          </Typography>
          <TextField
            type="number"
            value={settings.cacheSize}
            onChange={(e) => onChange('cacheSize', parseInt(e.target.value))}
            fullWidth
            disabled={!settings.cacheEnabled}
            inputProps={{ min: 100, max: 10000, step: 100 }}
            helperText={`Approximate memory usage: ${calculateCacheMemory().toFixed(1)} MB`}
          />
        </Box>
      </Stack>

      {/* System Information */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        System Information
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">
              Application Version:
            </Typography>
            <Chip label="1.0.0" size="small" />
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">
              API Endpoint:
            </Typography>
            <Typography variant="body2" fontFamily="monospace">
              {import.meta.env.VITE_API_URL || 'http://localhost:8000'}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">
              WebSocket Endpoint:
            </Typography>
            <Typography variant="body2" fontFamily="monospace">
              {import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">
              Browser:
            </Typography>
            <Typography variant="body2">{navigator.userAgent.split(' ').pop()}</Typography>
          </Box>
        </Stack>
      </Paper>

      {/* Data Management */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Data Management
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Alert severity="info">
        <Typography variant="body2">
          To clear all application data (settings, cache, etc.), use your browser's developer tools
          to clear localStorage and indexedDB for this domain.
        </Typography>
      </Alert>
    </Box>
  );
};

export default AdvancedTab;
