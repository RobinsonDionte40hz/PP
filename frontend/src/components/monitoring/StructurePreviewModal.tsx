import React from 'react';
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
} from '@mui/material';
import { Close as CloseIcon, Download as DownloadIcon } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

interface StructurePreviewModalProps {
  open: boolean;
  onClose: () => void;
  predictionId: string;
}

const StructurePreviewModal: React.FC<StructurePreviewModalProps> = ({
  open,
  onClose,
  predictionId,
}) => {
  const theme = useTheme();

  const { data: structure, isLoading, error } = useQuery({
    queryKey: ['structure', predictionId],
    queryFn: async () => {
      // Placeholder - would call actual structure endpoint
      // For now, return mock data
      return {
        pdb_content: 'MOCK PDB FILE CONTENT',
        format: 'pdb' as const,
      };
    },
    enabled: open && !!predictionId,
  });

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
          <Alert severity="error">
            Failed to load structure. The prediction may still be running.
          </Alert>
        )}

        {structure && !isLoading && (
          <Box>
            {/* Placeholder for 3D viewer - would integrate NGL Viewer here */}
            <Box
              sx={{
                height: 400,
                backgroundColor: alpha(theme.palette.background.default, 0.5),
                borderRadius: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                border: `1px solid ${theme.palette.divider}`,
                mb: 2,
              }}
            >
              <Typography variant="h6" color="text.secondary" gutterBottom>
                3D Viewer Placeholder
              </Typography>
              <Typography variant="body2" color="text.secondary" textAlign="center" px={4}>
                In production, this would show an interactive 3D visualization using NGL Viewer.
                <br />
                For now, you can download the PDB file below.
              </Typography>
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
