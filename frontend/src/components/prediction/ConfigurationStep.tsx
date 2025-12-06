import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  FormControl,
  FormLabel,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  alpha,
  useTheme,
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material';

interface ConfigurationStepProps {
  formData: {
    iterations?: number;
    consciousness_level?: number;
    consistency?: number;
    enable_qcpp?: boolean;
    qcpp_config?: string;
    checkpoint_interval?: number;
    enable_mediators?: boolean;
    mediator_count?: number;
    enable_refinement?: boolean;
    enable_hierarchical_folding?: boolean;
  };
  onChange: (updates: Partial<ConfigurationStepProps['formData']>) => void;
}

const ConfigurationStep: React.FC<ConfigurationStepProps> = ({ formData, onChange }) => {
  const theme = useTheme();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const presets = [
    {
      value: 'fast',
      label: 'Fast',
      description: 'Quick exploration (500 iterations)',
      config: {
        iterations: 500,
        consciousness_level: 6.0,
        consistency: 0.5,
        enable_qcpp: false,
        checkpoint_interval: 50,
      },
    },
    {
      value: 'balanced',
      label: 'Balanced',
      description: 'Good balance of speed and quality (1000 iterations)',
      config: {
        iterations: 1000,
        consciousness_level: 8.0,
        consistency: 0.6,
        enable_qcpp: true,
        qcpp_config: 'default',
        checkpoint_interval: 100,
      },
    },
    {
      value: 'accurate',
      label: 'Accurate',
      description: 'High-quality prediction (2000 iterations)',
      config: {
        iterations: 2000,
        consciousness_level: 10.0,
        consistency: 0.7,
        enable_qcpp: true,
        qcpp_config: 'high_accuracy',
        checkpoint_interval: 100,
      },
    },
  ];

  const handlePresetChange = (preset: string) => {
    const selected = presets.find((p) => p.value === preset);
    if (selected) {
      onChange(selected.config);
    }
  };

  return (
    <Box>
      <Typography variant="h6" fontWeight="bold" gutterBottom>
        Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        Choose a preset or customize advanced parameters
      </Typography>

      {/* Presets */}
      <Box mb={4}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Quick Presets
        </Typography>
        <Box display="flex" gap={2} flexWrap="wrap">
          {presets.map((preset) => (
            <Box
              key={preset.value}
              onClick={() => handlePresetChange(preset.value)}
              sx={{
                flex: '1 1 calc(33.333% - 16px)',
                minWidth: '200px',
                p: 2,
                border: `2px solid ${
                  formData.iterations === preset.config.iterations
                    ? theme.palette.primary.main
                    : theme.palette.divider
                }`,
                borderRadius: 1,
                cursor: 'pointer',
                backgroundColor:
                  formData.iterations === preset.config.iterations
                    ? alpha(theme.palette.primary.main, 0.05)
                    : 'transparent',
                transition: 'all 0.2s',
                '&:hover': {
                  borderColor: theme.palette.primary.main,
                  backgroundColor: alpha(theme.palette.primary.main, 0.02),
                },
              }}
            >
              <Typography variant="subtitle1" fontWeight="bold">
                {preset.label}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {preset.description}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>

      {/* Basic Configuration */}
      <Box mb={3}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Basic Settings
        </Typography>

        <Stack spacing={3}>
          {/* Iterations */}
          <FormControl fullWidth>
            <FormLabel>
              Max Iterations: {formData.iterations}
            </FormLabel>
            <Slider
              value={formData.iterations || 1000}
              onChange={(_, value) => onChange({ iterations: value as number })}
              min={100}
              max={5000}
              step={100}
              marks={[
                { value: 100, label: '100' },
                { value: 1000, label: '1K' },
                { value: 2000, label: '2K' },
                { value: 5000, label: '5K' },
              ]}
              valueLabelDisplay="auto"
            />
          </FormControl>

          {/* QCPP Integration */}
          <FormControl>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <FormLabel>Quantum Coherence Integration</FormLabel>
                <Typography variant="caption" color="text.secondary" display="block">
                  Enable QCPP for quantum-guided structure prediction
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_qcpp || false}
                    onChange={(e) => onChange({ enable_qcpp: e.target.checked })}
                  />
                }
                label=""
              />
            </Box>
          </FormControl>

          {/* QCPP Config */}
          {formData.enable_qcpp && (
            <FormControl fullWidth>
              <FormLabel>QCPP Configuration</FormLabel>
              <Select
                value={formData.qcpp_config || 'default'}
                onChange={(e) => onChange({ qcpp_config: e.target.value })}
                size="small"
              >
                <MenuItem value="default">Default</MenuItem>
                <MenuItem value="high_performance">High Performance</MenuItem>
                <MenuItem value="high_accuracy">High Accuracy</MenuItem>
              </Select>
            </FormControl>
          )}

          {/* Mediator Agents */}
          <FormControl>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <FormLabel>Mediator Agents</FormLabel>
                <Typography variant="caption" color="text.secondary" display="block">
                  Pattern detection and information relay agents
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_mediators || false}
                    onChange={(e) => onChange({ enable_mediators: e.target.checked })}
                  />
                }
                label=""
              />
            </Box>
          </FormControl>

          {formData.enable_mediators && (
            <FormControl fullWidth>
              <FormLabel>Mediator Count: {formData.mediator_count || 3}</FormLabel>
              <Slider
                value={formData.mediator_count || 3}
                onChange={(_, value) => onChange({ mediator_count: value as number })}
                min={1}
                max={10}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                ]}
                valueLabelDisplay="auto"
              />
            </FormControl>
          )}

          {/* Quantum Refinement */}
          <FormControl>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <FormLabel>Quantum Refinement</FormLabel>
                <Typography variant="caption" color="text.secondary" display="block">
                  Two-stage optimization with automatic geometric targeting (45-58% RMSD improvement)
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_refinement || false}
                    onChange={(e) => onChange({ enable_refinement: e.target.checked })}
                  />
                }
                label=""
              />
            </Box>
          </FormControl>

          {/* Hierarchical Folding */}
          <FormControl>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <FormLabel>Hierarchical Folding</FormLabel>
                <Typography variant="caption" color="text.secondary" display="block">
                  Progressive search confinement - anchors stable secondary structure to reduce search space
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_hierarchical_folding || false}
                    onChange={(e) => onChange({ enable_hierarchical_folding: e.target.checked })}
                  />
                }
                label=""
              />
            </Box>
          </FormControl>
        </Stack>
      </Box>

      {/* Advanced Settings */}
      <Accordion expanded={showAdvanced} onChange={() => setShowAdvanced(!showAdvanced)}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2" fontWeight="bold">
            Advanced Settings
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={3}>
            {/* Consciousness Level (Aggressiveness) */}
            <FormControl fullWidth>
              <FormLabel>
                Exploration Aggressiveness: {formData.consciousness_level?.toFixed(1)}
              </FormLabel>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Controls exploration tempo (3.0 = cautious, 15.0 = aggressive)
              </Typography>
              <Slider
                value={formData.consciousness_level || 8.0}
                onChange={(_, value) => onChange({ consciousness_level: value as number })}
                min={3.0}
                max={15.0}
                step={0.5}
                marks={[
                  { value: 3.0, label: 'Cautious' },
                  { value: 9.0, label: 'Balanced' },
                  { value: 15.0, label: 'Aggressive' },
                ]}
                valueLabelDisplay="auto"
              />
            </FormControl>

            {/* Consistency */}
            <FormControl fullWidth>
              <FormLabel>
                Behavioral Consistency: {formData.consistency?.toFixed(2)}
              </FormLabel>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Controls behavioral stability (0.2 = adaptive, 1.0 = stable)
              </Typography>
              <Slider
                value={formData.consistency || 0.6}
                onChange={(_, value) => onChange({ consistency: value as number })}
                min={0.2}
                max={1.0}
                step={0.05}
                marks={[
                  { value: 0.2, label: 'Adaptive' },
                  { value: 0.6, label: 'Balanced' },
                  { value: 1.0, label: 'Stable' },
                ]}
                valueLabelDisplay="auto"
              />
            </FormControl>

            {/* Checkpoint Interval */}
            <FormControl fullWidth>
              <FormLabel>Checkpoint Interval</FormLabel>
              <TextField
                type="number"
                size="small"
                value={formData.checkpoint_interval || 100}
                onChange={(e) =>
                  onChange({ checkpoint_interval: parseInt(e.target.value) || 100 })
                }
                inputProps={{ min: 10, max: 500, step: 10 }}
                helperText="Save progress every N iterations"
              />
            </FormControl>
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* Estimated Time */}
      <Box
        mt={3}
        p={2}
        sx={{
          backgroundColor: alpha(theme.palette.info.main, 0.05),
          borderRadius: 1,
          border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
        }}
      >
        <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
          Estimated Time
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Approximately{' '}
          {Math.ceil(((formData.iterations || 1000) * 0.5) / 60)} minutes
          {formData.enable_qcpp && ' (with QCPP integration)'}
        </Typography>
      </Box>
    </Box>
  );
};

export default ConfigurationStep;
