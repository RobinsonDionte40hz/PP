import React, { useState, useCallback } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  Button,
  Grid,
  Tabs,
  Tab,
  Chip,
  Alert,
  AlertTitle,
  LinearProgress,
  Card,
  CardContent,
  Divider,
  Collapse,
  CircularProgress,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Science as ScienceIcon,
  PlayArrow as PlayIcon,
  Download as DownloadIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Upload as UploadIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { useMutation } from '@tanstack/react-query';
import { screeningService } from '../services/screeningService';
import type {
  ScreeningResult,
  BatchScreeningResponse,
  ScreeningMode,
  AggregationRiskLevel,
} from '../types/screening';
import { RISK_LEVEL_CONFIG, SCREENING_MODE_CONFIG } from '../types/screening';

// Tab panel component
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

// Risk level chip component
const RiskChip: React.FC<{ level: AggregationRiskLevel }> = ({ level }) => {
  const config = RISK_LEVEL_CONFIG[level];
  return (
    <Chip
      label={config.label}
      size="small"
      sx={{
        backgroundColor: config.bgColor,
        color: config.color,
        fontWeight: 600,
      }}
    />
  );
};

// Score bar component
const ScoreBar: React.FC<{ score: number; label: string }> = ({ score, label }) => {
  const getColor = (s: number) => {
    if (s >= 0.7) return '#4caf50';
    if (s >= 0.5) return '#ff9800';
    if (s >= 0.3) return '#f44336';
    return '#b71c1c';
  };

  return (
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="caption" fontWeight={600}>{(score * 100).toFixed(0)}%</Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={score * 100}
        sx={{
          height: 6,
          borderRadius: 3,
          backgroundColor: alpha(getColor(score), 0.2),
          '& .MuiLinearProgress-bar': {
            backgroundColor: getColor(score),
            borderRadius: 3,
          },
        }}
      />
    </Box>
  );
};

// Single result card component
const ResultCard: React.FC<{ result: ScreeningResult; index: number }> = ({ result, index }) => {
  const [expanded, setExpanded] = useState(false);
  const theme = useTheme();

  return (
    <Card 
      variant="outlined" 
      sx={{ 
        mb: 2,
        borderColor: result.passes_screening 
          ? alpha(theme.palette.success.main, 0.3)
          : alpha(theme.palette.error.main, 0.3),
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box sx={{ flex: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="subtitle2" color="text.secondary">
                #{index + 1}
              </Typography>
              <RiskChip level={result.risk_level} />
              {result.passes_screening ? (
                <Chip 
                  icon={<CheckIcon />} 
                  label="Passes" 
                  size="small" 
                  color="success" 
                  variant="outlined" 
                />
              ) : (
                <Chip 
                  icon={<ErrorIcon />} 
                  label="Fails" 
                  size="small" 
                  color="error" 
                  variant="outlined" 
                />
              )}
            </Box>
            <Typography 
              variant="body2" 
              sx={{ 
                fontFamily: 'monospace',
                wordBreak: 'break-all',
                color: 'text.secondary',
              }}
            >
              {result.sequence.length > 50 
                ? `${result.sequence.substring(0, 50)}...` 
                : result.sequence
              }
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'right', minWidth: 120 }}>
            <Typography variant="h5" fontWeight={700} color={
              result.aggregation_score >= 0.7 ? 'success.main' :
              result.aggregation_score >= 0.5 ? 'warning.main' : 'error.main'
            }>
              {(result.aggregation_score * 100).toFixed(0)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Aggregation Score
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
          <Typography variant="caption" color="text.secondary">
            {result.sequence_length} residues • {result.screening_time_ms.toFixed(0)}ms
          </Typography>
          <IconButton size="small" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>

        <Collapse in={expanded}>
          <Divider sx={{ my: 2 }} />
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, md: 6 }}>
              <Typography variant="subtitle2" gutterBottom>Score Breakdown</Typography>
              <ScoreBar score={result.energy_score} label="Energy Stability" />
              <ScoreBar score={result.structure_score} label="Secondary Structure" />
              <ScoreBar score={result.hydrophobic_score} label="Hydrophobic Clustering" />
              <ScoreBar score={result.compactness_score} label="Compactness" />
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <Typography variant="subtitle2" gutterBottom>Raw Values</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Typography variant="caption" color="text.secondary">Energy:</Typography>
                <Typography variant="caption">{result.final_energy.toFixed(1)} kcal/mol</Typography>
                <Typography variant="caption" color="text.secondary">Structure %:</Typography>
                <Typography variant="caption">{result.secondary_structure_pct.toFixed(1)}%</Typography>
                <Typography variant="caption" color="text.secondary">Radius of Gyration:</Typography>
                <Typography variant="caption">{result.radius_of_gyration.toFixed(1)} Å</Typography>
              </Box>
              {result.risk_factors.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="error">Risk Factors</Typography>
                  {result.risk_factors.map((factor: string, i: number) => (
                    <Chip 
                      key={i} 
                      label={factor} 
                      size="small" 
                      color="error" 
                      variant="outlined" 
                      sx={{ mr: 0.5, mb: 0.5 }} 
                    />
                  ))}
                </Box>
              )}
            </Grid>
          </Grid>
        </Collapse>
      </CardContent>
    </Card>
  );
};

// Main Screening Page Component
const AggregationScreening: React.FC = () => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [sequenceInput, setSequenceInput] = useState('');
  const [batchInput, setBatchInput] = useState('');
  const [screeningMode, setScreeningMode] = useState<ScreeningMode>('balanced');
  const [singleResult, setSingleResult] = useState<ScreeningResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchScreeningResponse | null>(null);

  // Single sequence screening mutation
  const singleScreenMutation = useMutation({
    mutationFn: (sequence: string) => screeningService.screenSingle(sequence, screeningMode),
    onSuccess: (result: ScreeningResult) => setSingleResult(result),
  });

  // Batch screening mutation
  const batchScreenMutation = useMutation({
    mutationFn: (sequences: string[]) => 
      screeningService.screenBatch({ sequences, mode: screeningMode }),
    onSuccess: (result: BatchScreeningResponse) => setBatchResult(result),
  });

  // Parse batch input into sequences
  const parseSequences = useCallback((input: string): string[] => {
    return input
      .split(/[\n,;]+/)
      .map(s => s.trim().toUpperCase())
      .filter(s => s.length >= 5 && /^[ACDEFGHIKLMNPQRSTVWY]+$/.test(s));
  }, []);

  const handleSingleScreen = () => {
    const seq = sequenceInput.trim().toUpperCase();
    if (seq.length >= 5) {
      singleScreenMutation.mutate(seq);
    }
  };

  const handleBatchScreen = () => {
    const sequences = parseSequences(batchInput);
    if (sequences.length > 0) {
      batchScreenMutation.mutate(sequences);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setBatchInput(content);
      };
      reader.readAsText(file);
    }
  };

  const handleDownloadCsv = () => {
    if (batchResult?.batch_id) {
      screeningService.downloadCsv(batchResult.batch_id);
    }
  };

  const parsedCount = parseSequences(batchInput).length;

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <ScienceIcon sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1">
              Aggregation Screening
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Fast screening of protein sequences for aggregation risk
            </Typography>
          </Box>
        </Box>

        <Alert severity="info" sx={{ mt: 2 }}>
          <AlertTitle>What is Aggregation Screening?</AlertTitle>
          Unlike full structure prediction, screening quickly answers: 
          <strong> "Will this sequence fold stably, or is it likely to aggregate/clump?"</strong>
          <br />
          Use this to filter large sequence libraries before running expensive predictions.
        </Alert>
      </Box>

      {/* Mode Selection */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>Screening Mode</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {(Object.keys(SCREENING_MODE_CONFIG) as ScreeningMode[]).map((mode) => {
            const config = SCREENING_MODE_CONFIG[mode];
            const isSelected = screeningMode === mode;
            return (
              <Card
                key={mode}
                variant={isSelected ? 'elevation' : 'outlined'}
                onClick={() => setScreeningMode(mode)}
                sx={{
                  flex: 1,
                  cursor: 'pointer',
                  borderColor: isSelected ? 'primary.main' : undefined,
                  borderWidth: isSelected ? 2 : 1,
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: 'primary.main',
                  },
                }}
              >
                <CardContent>
                  <Typography variant="subtitle1" fontWeight={600}>{config.label}</Typography>
                  <Typography variant="body2" color="text.secondary">{config.description}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {config.iterations} iter / {config.agents} agents
                  </Typography>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={(_: React.SyntheticEvent, v: number) => setTabValue(v)}>
          <Tab label="Single Sequence" />
          <Tab label="Batch Screening" />
        </Tabs>

        {/* Single Sequence Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ p: 2 }}>
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Protein Sequence"
              placeholder="Enter amino acid sequence (e.g., MKTAYIAKQRQISFVKSH...)"
              value={sequenceInput}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setSequenceInput(e.target.value)}
              sx={{ mb: 2 }}
              helperText={`${sequenceInput.trim().length} characters`}
            />
            <Button
              variant="contained"
              startIcon={singleScreenMutation.isPending ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
              onClick={handleSingleScreen}
              disabled={sequenceInput.trim().length < 5 || singleScreenMutation.isPending}
            >
              {singleScreenMutation.isPending ? 'Screening...' : 'Screen Sequence'}
            </Button>

            {singleScreenMutation.isError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Screening failed: {(singleScreenMutation.error as Error).message}
              </Alert>
            )}

            {singleResult && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>Result</Typography>
                <ResultCard result={singleResult} index={0} />
              </Box>
            )}
          </Box>
        </TabPanel>

        {/* Batch Screening Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Button
                variant="outlined"
                component="label"
                startIcon={<UploadIcon />}
              >
                Upload File
                <input
                  type="file"
                  hidden
                  accept=".txt,.csv,.fasta"
                  onChange={handleFileUpload}
                />
              </Button>
              <Button
                variant="outlined"
                startIcon={<ClearIcon />}
                onClick={() => { setBatchInput(''); setBatchResult(null); }}
              >
                Clear
              </Button>
            </Box>

            <TextField
              fullWidth
              multiline
              rows={8}
              label="Sequences (one per line, or comma/semicolon separated)"
              placeholder="MKTAYIAKQRQISFVKSH&#10;ACDEFGHIKLMNPQRSTVWY&#10;VVVVVVVVVVVVVVVVVVVV"
              value={batchInput}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setBatchInput(e.target.value)}
              sx={{ mb: 2, fontFamily: 'monospace' }}
              helperText={`${parsedCount} valid sequences detected`}
            />

            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <Button
                variant="contained"
                startIcon={batchScreenMutation.isPending ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
                onClick={handleBatchScreen}
                disabled={parsedCount === 0 || batchScreenMutation.isPending}
              >
                {batchScreenMutation.isPending ? 'Screening...' : `Screen ${parsedCount} Sequences`}
              </Button>
              {batchResult?.status === 'completed' && (
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={handleDownloadCsv}
                >
                  Download CSV
                </Button>
              )}
            </Box>

            {batchScreenMutation.isError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Batch screening failed: {(batchScreenMutation.error as Error).message}
              </Alert>
            )}

            {batchResult && (
              <Box sx={{ mt: 3 }}>
                {/* Summary */}
                <Paper sx={{ p: 2, mb: 2, bgcolor: alpha(theme.palette.primary.main, 0.05) }}>
                  <Typography variant="h6" gutterBottom>Screening Summary</Typography>
                  <Grid container spacing={3}>
                    <Grid size={{ xs: 6, md: 3 }}>
                      <Typography variant="h4">{batchResult.total_sequences}</Typography>
                      <Typography variant="caption" color="text.secondary">Total Screened</Typography>
                    </Grid>
                    <Grid size={{ xs: 6, md: 3 }}>
                      <Typography variant="h4" color="success.main">{batchResult.sequences_passed}</Typography>
                      <Typography variant="caption" color="text.secondary">Passed</Typography>
                    </Grid>
                    <Grid size={{ xs: 6, md: 3 }}>
                      <Typography variant="h4" color="error.main">{batchResult.sequences_failed}</Typography>
                      <Typography variant="caption" color="text.secondary">Failed</Typography>
                    </Grid>
                    <Grid size={{ xs: 6, md: 3 }}>
                      <Typography variant="h4">
                        {((batchResult.sequences_passed / batchResult.total_sequences) * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">Pass Rate</Typography>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" gutterBottom>Risk Distribution</Typography>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    {(Object.entries(batchResult.risk_summary) as [AggregationRiskLevel, number][]).map(([level, count]) => (
                      <Chip
                        key={level}
                        label={`${RISK_LEVEL_CONFIG[level].label}: ${count}`}
                        sx={{
                          backgroundColor: RISK_LEVEL_CONFIG[level].bgColor,
                          color: RISK_LEVEL_CONFIG[level].color,
                        }}
                      />
                    ))}
                  </Box>
                </Paper>

                {/* Results List */}
                <Typography variant="h6" gutterBottom>
                  Results (Ranked by Score)
                </Typography>
                {batchResult.results.map((result: ScreeningResult, index: number) => (
                  <ResultCard key={index} result={result} index={index} />
                ))}
              </Box>
            )}
          </Box>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default AggregationScreening;
