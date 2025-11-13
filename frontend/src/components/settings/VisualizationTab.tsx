import React from 'react';
import {
  Box,
  Typography,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  Alert,
  Stack,
  TextField,
} from '@mui/material';

interface VisualizationTabProps {
  settings: {
    defaultRepresentation: string;
    defaultColorScheme: string;
    backgroundColor: string;
    enableSmoothAnimation: boolean;
    showHydrogenBonds: boolean;
    showGeometricPatterns: boolean;
    qualityLevel: string;
  };
  onChange: (key: string, value: boolean | string) => void;
}

const VisualizationTab: React.FC<VisualizationTabProps> = ({ settings, onChange }) => {
  return (
    <Box sx={{ px: 3 }}>
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure default visualization settings for 3D protein structure viewer.
      </Alert>

      {/* Display Settings */}
      <Typography variant="h6" gutterBottom>
        Display Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        {/* Default Representation */}
        <FormControl fullWidth>
          <InputLabel>Default Representation</InputLabel>
          <Select
            value={settings.defaultRepresentation}
            label="Default Representation"
            onChange={(e) => onChange('defaultRepresentation', e.target.value)}
          >
            <MenuItem value="cartoon">Cartoon</MenuItem>
            <MenuItem value="ball+stick">Ball & Stick</MenuItem>
            <MenuItem value="licorice">Licorice</MenuItem>
            <MenuItem value="backbone">Backbone</MenuItem>
            <MenuItem value="ribbon">Ribbon</MenuItem>
            <MenuItem value="surface">Surface</MenuItem>
            <MenuItem value="spacefill">Spacefill</MenuItem>
          </Select>
        </FormControl>

        {/* Default Color Scheme */}
        <FormControl fullWidth>
          <InputLabel>Default Color Scheme</InputLabel>
          <Select
            value={settings.defaultColorScheme}
            label="Default Color Scheme"
            onChange={(e) => onChange('defaultColorScheme', e.target.value)}
          >
            <MenuItem value="chainid">Chain ID</MenuItem>
            <MenuItem value="element">Element</MenuItem>
            <MenuItem value="residueindex">Residue Index</MenuItem>
            <MenuItem value="secondary">Secondary Structure</MenuItem>
            <MenuItem value="bfactor">B-Factor</MenuItem>
            <MenuItem value="hydrophobicity">Hydrophobicity</MenuItem>
            <MenuItem value="uniform">Uniform Color</MenuItem>
          </Select>
        </FormControl>

        {/* Background Color */}
        <Box>
          <Typography variant="body2" gutterBottom>
            Background Color
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center">
            <TextField
              type="color"
              value={settings.backgroundColor}
              onChange={(e) => onChange('backgroundColor', e.target.value)}
              sx={{ width: 80 }}
            />
            <Typography variant="body2" color="text.secondary">
              {settings.backgroundColor}
            </Typography>
          </Stack>
        </Box>

        {/* Quality Level */}
        <FormControl fullWidth>
          <InputLabel>Rendering Quality</InputLabel>
          <Select
            value={settings.qualityLevel}
            label="Rendering Quality"
            onChange={(e) => onChange('qualityLevel', e.target.value)}
          >
            <MenuItem value="low">Low (Better Performance)</MenuItem>
            <MenuItem value="medium">Medium</MenuItem>
            <MenuItem value="high">High (Better Quality)</MenuItem>
            <MenuItem value="auto">Auto</MenuItem>
          </Select>
        </FormControl>
      </Stack>

      {/* Feature Toggles */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Features
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={2}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.enableSmoothAnimation}
              onChange={(e) => onChange('enableSmoothAnimation', e.target.checked)}
            />
          }
          label="Enable Smooth Animations"
        />

        <FormControlLabel
          control={
            <Switch
              checked={settings.showHydrogenBonds}
              onChange={(e) => onChange('showHydrogenBonds', e.target.checked)}
            />
          }
          label="Show Hydrogen Bonds"
        />

        <FormControlLabel
          control={
            <Switch
              checked={settings.showGeometricPatterns}
              onChange={(e) => onChange('showGeometricPatterns', e.target.checked)}
            />
          }
          label="Highlight Geometric Patterns (Golden Ratio)"
        />
      </Stack>

      {/* Info Box */}
      <Alert severity="success" sx={{ mt: 4 }}>
        <Typography variant="body2">
          💡 <strong>Tip:</strong> For large structures (&gt;500 residues), consider using lower quality
          settings or simpler representations for better performance.
        </Typography>
      </Alert>
    </Box>
  );
};

export default VisualizationTab;
