/**
 * ViewerControls Component
 * Control panel for 3D protein viewer
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  ButtonGroup,
  Divider,
  Typography,
  Stack,
  Tooltip
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import {
  CameraAlt,
  FileDownload,
  Refresh,
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  Visibility,
  VisibilityOff
} from '@mui/icons-material';
import * as NGL from 'ngl';
import {
  centerStructure,
  takeScreenshot,
  exportPDB,
  setQuality,
  DEFAULT_REPRESENTATIONS,
  COLOR_SCHEMES
} from '../../utils/nglUtils';

export interface ViewerControlsProps {
  stage: NGL.Stage | null;
  component: NGL.StructureComponent | null;
  onRepresentationChange?: (representation: keyof typeof DEFAULT_REPRESENTATIONS) => void;
  onColorSchemeChange?: (colorScheme: keyof typeof COLOR_SCHEMES) => void;
  onExportPDB?: () => void;
  onScreenshot?: () => void;
}

export const ViewerControls: React.FC<ViewerControlsProps> = ({
  stage,
  component,
  onRepresentationChange,
  onColorSchemeChange,
  onExportPDB,
  onScreenshot
}) => {
  const [representation, setRepresentation] = useState<keyof typeof DEFAULT_REPRESENTATIONS>('cartoon');
  const [colorScheme, setColorScheme] = useState<keyof typeof COLOR_SCHEMES>('secondary structure');
  const [quality, setQualityState] = useState<'low' | 'medium' | 'high' | 'auto'>('high');
  const [visible, setVisible] = useState(true);

  const handleRepresentationChange = (event: SelectChangeEvent<string>) => {
    const newRep = event.target.value as keyof typeof DEFAULT_REPRESENTATIONS;
    setRepresentation(newRep);

    if (component) {
      // Remove all representations
      component.removeAllRepresentations();
      
      // Add new representation
      const rep = DEFAULT_REPRESENTATIONS[newRep];
      component.addRepresentation(rep.type, {
        color: colorScheme,
        ...rep.params
      });
    }

    onRepresentationChange?.(newRep);
  };

  const handleColorSchemeChange = (event: SelectChangeEvent<string>) => {
    const newColor = event.target.value as keyof typeof COLOR_SCHEMES;
    setColorScheme(newColor);

    if (component) {
      // Update color scheme on all representations
      component.eachRepresentation((repr) => {
        repr.setColor(newColor);
      });
    }

    onColorSchemeChange?.(newColor);
  };

  const handleQualityChange = (event: SelectChangeEvent<string>) => {
    const newQuality = event.target.value as 'low' | 'medium' | 'high' | 'auto';
    setQualityState(newQuality);

    if (stage) {
      setQuality(stage, newQuality);
    }
  };

  const handleCenter = () => {
    if (stage) {
      centerStructure(stage, component || undefined);
    }
  };

  const handleZoomIn = () => {
    if (stage) {
      const camera = stage.viewer.camera;
      const newZ = camera.position.z * 0.8;
      camera.position.setZ(newZ);
    }
  };

  const handleZoomOut = () => {
    if (stage) {
      const camera = stage.viewer.camera;
      const newZ = camera.position.z * 1.25;
      camera.position.setZ(newZ);
    }
  };

  const handleReset = () => {
    if (stage && component) {
      centerStructure(stage, component);
    }
  };

  const handleToggleVisibility = () => {
    if (component) {
      if (visible) {
        component.setVisibility(false);
      } else {
        component.setVisibility(true);
      }
      setVisible(!visible);
    }
  };

  const handleScreenshot = async () => {
    if (stage) {
      try {
        const blob = await takeScreenshot(stage, {
          factor: 4,
          antialias: true,
          transparent: false
        });
        
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `protein-structure-${Date.now()}.png`;
        link.click();
        URL.revokeObjectURL(url);

        onScreenshot?.();
      } catch (error) {
        console.error('Failed to take screenshot:', error);
      }
    }
  };

  const handleExport = () => {
    if (component) {
      exportPDB(component, `structure-${Date.now()}.pdb`);
      onExportPDB?.();
    }
  };

  const handleSpin = () => {
    if (stage) {
      stage.setSpin(true);
      stage.setRock(true);
    }
  };

  const handleStopSpin = () => {
    if (stage) {
      stage.setSpin(false);
      stage.setRock(false);
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Stack spacing={2}>
        <Typography variant="h6" gutterBottom>
          Viewer Controls
        </Typography>

        {/* Representation Selection */}
        <FormControl fullWidth size="small">
          <InputLabel>Representation</InputLabel>
          <Select
            value={representation}
            label="Representation"
            onChange={handleRepresentationChange}
            disabled={!component}
          >
            {Object.keys(DEFAULT_REPRESENTATIONS).map(rep => (
              <MenuItem key={rep} value={rep}>
                {rep.charAt(0).toUpperCase() + rep.slice(1).replace(/([A-Z])/g, ' $1')}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Color Scheme Selection */}
        <FormControl fullWidth size="small">
          <InputLabel>Color Scheme</InputLabel>
          <Select
            value={colorScheme}
            label="Color Scheme"
            onChange={handleColorSchemeChange}
            disabled={!component}
          >
            {Object.entries(COLOR_SCHEMES).map(([value, label]) => (
              <MenuItem key={value} value={value}>
                {label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Quality Selection */}
        <FormControl fullWidth size="small">
          <InputLabel>Quality</InputLabel>
          <Select
            value={quality}
            label="Quality"
            onChange={handleQualityChange}
            disabled={!stage}
          >
            <MenuItem value="auto">Auto</MenuItem>
            <MenuItem value="low">Low</MenuItem>
            <MenuItem value="medium">Medium</MenuItem>
            <MenuItem value="high">High</MenuItem>
          </Select>
        </FormControl>

        <Divider />

        {/* Zoom Controls */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            View Controls
          </Typography>
          <ButtonGroup fullWidth size="small" disabled={!stage}>
            <Tooltip title="Zoom In">
              <Button onClick={handleZoomIn} startIcon={<ZoomIn />}>
                Zoom In
              </Button>
            </Tooltip>
            <Tooltip title="Zoom Out">
              <Button onClick={handleZoomOut} startIcon={<ZoomOut />}>
                Zoom Out
              </Button>
            </Tooltip>
            <Tooltip title="Center">
              <Button onClick={handleCenter} startIcon={<CenterFocusStrong />}>
                Center
              </Button>
            </Tooltip>
          </ButtonGroup>
        </Box>

        {/* Rotation Controls */}
        <Box>
          <ButtonGroup fullWidth size="small" disabled={!stage}>
            <Button onClick={handleSpin}>
              Start Spin
            </Button>
            <Button onClick={handleStopSpin}>
              Stop Spin
            </Button>
            <Button onClick={handleReset} startIcon={<Refresh />}>
              Reset
            </Button>
          </ButtonGroup>
        </Box>

        <Divider />

        {/* Export Controls */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Export
          </Typography>
          <Stack spacing={1}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<CameraAlt />}
              onClick={handleScreenshot}
              disabled={!stage}
              size="small"
            >
              Screenshot (PNG)
            </Button>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<FileDownload />}
              onClick={handleExport}
              disabled={!component}
              size="small"
            >
              Export PDB
            </Button>
          </Stack>
        </Box>

        <Divider />

        {/* Additional Controls */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Display
          </Typography>
          <Button
            fullWidth
            variant="outlined"
            startIcon={visible ? <Visibility /> : <VisibilityOff />}
            onClick={handleToggleVisibility}
            disabled={!component}
            size="small"
          >
            {visible ? 'Hide Structure' : 'Show Structure'}
          </Button>
        </Box>
      </Stack>
    </Paper>
  );
};

export default ViewerControls;
