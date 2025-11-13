/**
 * StructureVisualization Page
 * Integrated 3D protein structure viewer with full controls
 */

import React, { useState, useRef } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Stack,
  Alert,
  TextField,
  Tabs,
  Tab,
  Card,
  CardContent
} from '@mui/material';
import { Upload, CloudDownload } from '@mui/icons-material';
import * as NGL from 'ngl';
import { ProteinViewer, ViewerControls } from '../components/visualization';
import { getStructureStats } from '../utils/nglUtils';

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
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

export const StructureVisualization: React.FC = () => {
  const [pdbData, setPdbData] = useState<string>('');
  const [nativePdbData] = useState<string>('');
  const [pdbFile, setPdbFile] = useState<File | undefined>();
  const [pdbId, setPdbId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<ReturnType<typeof getStructureStats> | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [comparisonMode] = useState(false);
  
  const viewerRef = useRef<HTMLDivElement>(null);
  const [stage] = useState<NGL.Stage | null>(null);
  const [component] = useState<NGL.StructureComponent | null>(null);

  const handleStructureLoad = (structureStats: ReturnType<typeof getStructureStats>) => {
    setStats(structureStats);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setPdbFile(file);
      setPdbData('');
      setPdbId('');
      setError(null);
    }
  };

  const handleFetchPDB = async () => {
    if (!pdbId) {
      setError('Please enter a PDB ID');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`https://files.rcsb.org/download/${pdbId.toUpperCase()}.pdb`);
      if (!response.ok) {
        throw new Error('PDB not found');
      }
      const data = await response.text();
      setPdbData(data);
      setPdbFile(undefined);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch PDB');
      setLoading(false);
    }
  };

  const handleLoadExample = () => {
    setPdbId('1UBQ');
    setTimeout(() => {
      handleFetchPDB();
    }, 100);
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Typography variant="h4" gutterBottom>
        3D Structure Visualization
      </Typography>

      <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
        {/* Left Panel - Input and Controls */}
        <Box sx={{ width: { xs: '100%', md: 350 }, flexShrink: 0 }}>
          <Stack spacing={2}>
            {/* Input Options */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Load Structure
              </Typography>
              
              <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 2 }}>
                <Tab label="PDB ID" />
                <Tab label="File" />
              </Tabs>

              <TabPanel value={tabValue} index={0}>
                <Stack spacing={2}>
                  <TextField
                    label="PDB ID"
                    size="small"
                    value={pdbId}
                    onChange={(e) => setPdbId(e.target.value)}
                    placeholder="e.g., 1UBQ"
                    fullWidth
                  />
                  <Button
                    variant="contained"
                    onClick={handleFetchPDB}
                    disabled={loading || !pdbId}
                    startIcon={<CloudDownload />}
                    fullWidth
                  >
                    Fetch from RCSB
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleLoadExample}
                    size="small"
                    fullWidth
                  >
                    Load Example (1UBQ)
                  </Button>
                </Stack>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<Upload />}
                  fullWidth
                >
                  Upload PDB File
                  <input
                    type="file"
                    hidden
                    accept=".pdb"
                    onChange={handleFileUpload}
                  />
                </Button>
                {pdbFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {pdbFile.name}
                  </Typography>
                )}
              </TabPanel>
            </Paper>

            {/* Structure Info */}
            {stats && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Structure Info
                  </Typography>
                  <Stack spacing={1}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Residues
                      </Typography>
                      <Typography variant="h6">
                        {stats.residueCount}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Atoms
                      </Typography>
                      <Typography variant="h6">
                        {stats.atomCount}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Chains
                      </Typography>
                      <Typography variant="h6">
                        {stats.chainCount}
                      </Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            )}

            {/* Viewer Controls */}
            {(pdbData || pdbFile) && (
              <ViewerControls
                stage={stage}
                component={component}
              />
            )}
          </Stack>
        </Box>

        {/* Right Panel - 3D Viewer */}
        <Box sx={{ flex: 1, minWidth: 0 }} ref={viewerRef}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {!pdbData && !pdbFile && (
            <Paper
              sx={{
                height: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: 2
              }}
            >
              <Typography variant="h6" color="text.secondary">
                No structure loaded
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Load a structure using PDB ID or upload a file
              </Typography>
            </Paper>
          )}

          {(pdbData || pdbFile) && (
            <ProteinViewer
              pdbData={pdbData}
              pdbFile={pdbFile}
              nativePdbData={comparisonMode ? nativePdbData : undefined}
              height={600}
              representation="cartoon"
              onLoad={handleStructureLoad}
              onError={(err) => setError(err.message)}
            />
          )}
        </Box>
      </Box>
    </Container>
  );
};

export default StructureVisualization;
