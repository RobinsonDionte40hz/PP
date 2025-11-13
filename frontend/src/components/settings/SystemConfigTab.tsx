import React from 'react';
import {
  Box,
  Typography,
  TextField,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Divider,
  Alert,
  Stack,
} from '@mui/material';

interface SystemConfigTabProps {
  settings: {
    defaultIterations: number;
    defaultAgents: number;
    defaultConsciousness: number;
    defaultConsistency: number;
    enableQCPP: boolean;
    qcppConfig: string;
    checkpointInterval: number;
    autoSaveCheckpoints: boolean;
  };
  onChange: (key: string, value: number | boolean | string) => void;
}

const SystemConfigTab: React.FC<SystemConfigTabProps> = ({ settings, onChange }) => {
  return (
    <Box sx={{ px: 3 }}>
      <Alert severity="info" sx={{ mb: 3 }}>
        These settings define the default configuration for new predictions. You can override them
        when creating individual predictions.
      </Alert>

      {/* Prediction Defaults */}
      <Typography variant="h6" gutterBottom>
        Prediction Defaults
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        {/* Default Iterations */}
        <Box>
          <Typography variant="body2" gutterBottom>
            Default Iterations
          </Typography>
          <TextField
            type="number"
            value={settings.defaultIterations}
            onChange={(e) => onChange('defaultIterations', parseInt(e.target.value))}
            fullWidth
            inputProps={{ min: 100, max: 10000, step: 100 }}
            helperText="Number of iterations for structure optimization (100-10000)"
          />
        </Box>

        {/* Default Agents */}
        <Box>
          <Typography variant="body2" gutterBottom>
            Default Number of Agents
          </Typography>
          <TextField
            type="number"
            value={settings.defaultAgents}
            onChange={(e) => onChange('defaultAgents', parseInt(e.target.value))}
            fullWidth
            inputProps={{ min: 1, max: 100 }}
            helperText="Number of agents for parallel exploration (1-100)"
          />
        </Box>

        {/* Consciousness Level (Aggressiveness) */}
        <Box>
          <Typography variant="body2" gutterBottom>
            Consciousness Level (Aggressiveness): {settings.defaultConsciousness.toFixed(1)}
          </Typography>
          <Slider
            value={settings.defaultConsciousness}
            onChange={(_e, value) => onChange('defaultConsciousness', value as number)}
            min={3}
            max={15}
            step={0.5}
            marks
            valueLabelDisplay="auto"
          />
          <Typography variant="caption" color="text.secondary">
            Controls exploration tempo (3.0-15.0)
          </Typography>
        </Box>

        {/* Consistency */}
        <Box>
          <Typography variant="body2" gutterBottom>
            Consistency: {settings.defaultConsistency.toFixed(2)}
          </Typography>
          <Slider
            value={settings.defaultConsistency}
            onChange={(_e, value) => onChange('defaultConsistency', value as number)}
            min={0.2}
            max={1.0}
            step={0.05}
            marks
            valueLabelDisplay="auto"
          />
          <Typography variant="caption" color="text.secondary">
            Controls behavioral stability (0.2-1.0)
          </Typography>
        </Box>
      </Stack>

      {/* QCPP Integration */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        QCPP Integration
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.enableQCPP}
              onChange={(e) => onChange('enableQCPP', e.target.checked)}
            />
          }
          label="Enable QCPP (Quantum Coherence Protein Predictor)"
        />

        <FormControl fullWidth disabled={!settings.enableQCPP}>
          <InputLabel>QCPP Configuration</InputLabel>
          <Select
            value={settings.qcppConfig}
            label="QCPP Configuration"
            onChange={(e) => onChange('qcppConfig', e.target.value)}
          >
            <MenuItem value="default">Default</MenuItem>
            <MenuItem value="high_performance">High Performance</MenuItem>
            <MenuItem value="high_accuracy">High Accuracy</MenuItem>
          </Select>
        </FormControl>
      </Stack>

      {/* Checkpoint Settings */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Checkpoint Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.autoSaveCheckpoints}
              onChange={(e) => onChange('autoSaveCheckpoints', e.target.checked)}
            />
          }
          label="Auto-save Checkpoints"
        />

        <Box>
          <Typography variant="body2" gutterBottom>
            Checkpoint Interval (iterations)
          </Typography>
          <TextField
            type="number"
            value={settings.checkpointInterval}
            onChange={(e) => onChange('checkpointInterval', parseInt(e.target.value))}
            fullWidth
            disabled={!settings.autoSaveCheckpoints}
            inputProps={{ min: 10, max: 500 }}
            helperText="Save checkpoint every N iterations"
          />
        </Box>
      </Stack>
    </Box>
  );
};

export default SystemConfigTab;
