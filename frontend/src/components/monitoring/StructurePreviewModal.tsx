import React, { useState, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Alert,
  CircularProgress,
  IconButton,
  useTheme,
  alpha,
  ToggleButtonGroup,
  ToggleButton,
  Chip,
  Stack,
} from '@mui/material';
import { 
  Close as CloseIcon, 
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  ViewInAr as View3DIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { ProteinViewer } from '../visualization/ProteinViewer';
import api from '../../services/api';

interface StructurePreviewModalProps {
  open: boolean;
  onClose: () => void;
  predictionId: string;
}

type RepresentationType = 'cartoon' | 'ribbon' | 'backbone' | 'surface' | 'ball-stick';

const StructurePreviewModal: React.FC<StructurePreviewModalProps> = ({
  open,
  onClose,
  predictionId,
}) => {
  const theme = useTheme();
  const [representation, setRepresentation] = useState<RepresentationType>('cartoon');
  const [structureStats, setStructureStats] = useState<{
    residueCount?: number;
    atomCount?: number;
  } | null>(null);

  // Fetch current PDB structure from backend
  const { data: structure, isLoading, error, refetch } = useQuery({
    queryKey: ['live-structure', predictionId],
    queryFn: async () => {
      // Use axios with responseType: 'text' to get raw PDB content
      const response = await api.get(`/results/${predictionId}/structure`, {
        responseType: 'text',
        headers: {
          'Accept': 'chemical/x-pdb, text/plain, */*'
        }
      });
      
      const pdbContent = response.data;
      
      if (!pdbContent || typeof pdbContent !== 'string' || pdbContent.length < 10) {
        throw new Error('Invalid PDB data received');
      }
      
      return {
        pdb_content: pdbContent,
        format: 'pdb' as const,
      };
    },
    enabled: open && !!predictionId,
    refetchInterval: false,
    retry: 1,
  });

  const handleRepresentationChange = (
    _event: React.MouseEvent<HTMLElement>,
    newRep: RepresentationType | null
  ) => {
    if (newRep) {
      setRepresentation(newRep);
    }
  };

  const handleStructureLoad = useCallback((stats: { residueCount?: number; atomCount?: number }) => {
    setStructureStats(stats);
  }, []);

  const handleDownload = () => {
    if (!structure?.pdb_content) return;
    
    const blob = new Blob([structure.pdb_content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `structure_${predictionId}.pdb`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6" fontWeight="bold">
            Structure Preview
          </Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        {isLoading && (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            py={8}
          >
            <CircularProgress />
            <Typography variant="body2" color="text.secondary" mt={2}>
              Loading structure...
            </Typography>
          </Box>
        )}

        {error && (
          <Alert 
            severity="warning" 
            action={
              <Button color="inherit" size="small" onClick={() => refetch()}>
                Retry
              </Button>
            }
          >
            {error instanceof Error ? error.message : 'Failed to load structure. The prediction may still be running.'}
          </Alert>
        )}

        {structure && !isLoading && (
          <Box>
            {/* Representation Controls */}
            <Box sx={{ mb: 2 }}>
              <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                <ToggleButtonGroup
                  value={representation}
                  exclusive
                  onChange={handleRepresentationChange}
                  size="small"
                >
                  <ToggleButton value="cartoon">Cartoon</ToggleButton>
                  <ToggleButton value="ribbon">Ribbon</ToggleButton>
                  <ToggleButton value="backbone">Backbone</ToggleButton>
                  <ToggleButton value="surface">Surface</ToggleButton>
                  <ToggleButton value="ball-stick">Ball & Stick</ToggleButton>
                </ToggleButtonGroup>
                
                <Button
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={() => refetch()}
                >
                  Refresh
                </Button>
              </Stack>
            </Box>

            {/* 3D Protein Viewer */}
            <Box sx={{ mb: 2 }}>
              <ProteinViewer
                pdbData={structure.pdb_content}
                height={450}
                representation={representation as 'cartoon' | 'ribbon' | 'backbone' | 'surface'}
                onLoad={handleStructureLoad}
                onError={(err) => console.error('Viewer error:', err)}
              />
            </Box>

            {/* Structure Info */}
            <Box
              sx={{
                p: 2,
                backgroundColor: alpha(theme.palette.info.main, 0.05),
                borderRadius: 1,
                border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
              }}
            >
              <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                    Structure Information
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Format: {structure.format.toUpperCase()}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Size: {(structure.pdb_content.length / 1024).toFixed(2)} KB
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Lines: {structure.pdb_content.split('\n').length}
                  </Typography>
                </Box>
                <Box>
                  {structureStats && (
                    <Stack direction="row" spacing={1}>
                      {structureStats.residueCount && (
                        <Chip 
                          size="small" 
                          label={`${structureStats.residueCount} residues`}
                          icon={<View3DIcon />}
                        />
                      )}
                      {structureStats.atomCount && (
                        <Chip 
                          size="small" 
                          label={`${structureStats.atomCount} atoms`}
                          variant="outlined"
                        />
                      )}
                    </Stack>
                  )}
                </Box>
              </Stack>
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        {structure && (
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleDownload}
          >
            Download PDB
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default StructurePreviewModal;
