import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Stack,
  alpha,
  useTheme,
  Alert,
  AlertTitle,
  Divider,
} from '@mui/material';
import {
  Check as CheckIcon,
  Science as ScienceIcon,
  Speed as SpeedIcon,
  Tune as TuneIcon,
  Queue as QueueIcon,
  Schedule as ScheduleIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

interface ReviewStepProps {
  formData: {
    sequence: string;
    native_pdb_id?: string;
    iterations?: number;
    agents?: number;
    consciousness_level?: number;
    consistency?: number;
    enable_qcpp?: boolean;
    qcpp_config?: string;
    checkpoint_interval?: number;
    enable_mediators?: boolean;
    mediator_count?: number;
    enable_refinement?: boolean;
    enable_hierarchical_folding?: boolean;
    enable_screening?: boolean;
    screening_mode?: string;
  };
}

const ReviewStep: React.FC<ReviewStepProps> = ({ formData }) => {
  const theme = useTheme();

  const truncateSequence = (seq: string, maxLength: number = 50) => {
    return seq.length > maxLength ? `${seq.substring(0, maxLength)}...` : seq;
  };

  return (
    <Box>
      <Typography variant="h6" fontWeight="bold" gutterBottom>
        Review & Submit
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        Please review your prediction configuration before submitting
      </Typography>

      <Stack spacing={3}>
        {/* Sequence Section */}
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            backgroundColor: alpha(theme.palette.background.default, 0.5),
          }}
        >
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <ScienceIcon color="primary" />
            <Typography variant="subtitle1" fontWeight="bold">
              Sequence Information
            </Typography>
          </Box>

          <Box mb={2}>
            <Typography variant="caption" color="text.secondary" display="block">
              Amino Acid Sequence
            </Typography>
            <Typography
              variant="body2"
              fontFamily="monospace"
              sx={{
                mt: 0.5,
                p: 1,
                backgroundColor: alpha(theme.palette.primary.main, 0.05),
                borderRadius: 0.5,
                wordBreak: 'break-all',
              }}
            >
              {truncateSequence(formData.sequence)}
            </Typography>
            <Typography variant="caption" color="text.secondary" mt={0.5}>
              {formData.sequence.length} amino acids
            </Typography>
          </Box>

          {formData.native_pdb_id && (
            <Box mt={2}>
              <Typography variant="caption" color="text.secondary" display="block">
                Native Structure for RMSD Calculation
              </Typography>
              <Chip
                label={`PDB ID: ${formData.native_pdb_id}`}
                size="small"
                color="success"
                icon={<CheckIcon />}
                sx={{ mt: 0.5 }}
              />
              <Typography variant="caption" color="success.main" display="block" mt={0.5}>
                RMSD will be calculated against native structure
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Configuration Section */}
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            backgroundColor: alpha(theme.palette.background.default, 0.5),
          }}
        >
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <TuneIcon color="primary" />
            <Typography variant="subtitle1" fontWeight="bold">
              Configuration
            </Typography>
          </Box>

          <Stack spacing={2}>
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Max Iterations
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {formData.iterations?.toLocaleString() || 0}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Number of Agents
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {formData.agents || 10}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Exploration Aggressiveness
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {formData.consciousness_level?.toFixed(1) || 0}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Behavioral Consistency
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {formData.consistency?.toFixed(2) || 0}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Checkpoint Interval
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                Every {formData.checkpoint_interval || 100} iterations
              </Typography>
            </Box>

            {formData.enable_mediators && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Mediator Agents
                </Typography>
                <Chip
                  label={`${formData.mediator_count || 3} agents`}
                  size="small"
                  color="info"
                />
              </Box>
            )}

            {formData.enable_refinement && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Quantum Refinement
                </Typography>
                <Chip label="Enabled" size="small" color="success" />
              </Box>
            )}

            {formData.enable_hierarchical_folding && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Hierarchical Folding
                </Typography>
                <Chip label="Enabled" size="small" color="success" />
              </Box>
            )}

            {formData.enable_screening && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Aggregation Screening
                </Typography>
                <Chip
                  label={
                    formData.screening_mode === 'fast' ? 'Fast' :
                    formData.screening_mode === 'thorough' ? 'Thorough' : 'Balanced'
                  }
                  size="small"
                  color="warning"
                />
              </Box>
            )}
          </Stack>
        </Paper>

        {/* QCPP Section */}
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            backgroundColor: formData.enable_qcpp
              ? alpha(theme.palette.success.main, 0.05)
              : alpha(theme.palette.background.default, 0.5),
            borderColor: formData.enable_qcpp
              ? alpha(theme.palette.success.main, 0.3)
              : undefined,
          }}
        >
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <SpeedIcon color={formData.enable_qcpp ? 'success' : 'disabled'} />
            <Typography variant="subtitle1" fontWeight="bold">
              Quantum Coherence Integration
            </Typography>
            <Chip
              label={formData.enable_qcpp ? 'Enabled' : 'Disabled'}
              size="small"
              color={formData.enable_qcpp ? 'success' : 'default'}
              icon={formData.enable_qcpp ? <CheckIcon /> : undefined}
            />
          </Box>

          {formData.enable_qcpp && (
            <Box>
              <Typography variant="body2" color="text.secondary">
                QCPP Configuration:{' '}
                <Typography component="span" variant="body2" fontWeight="medium">
                  {formData.qcpp_config === 'default'
                    ? 'Default'
                    : formData.qcpp_config === 'high_performance'
                    ? 'High Performance'
                    : 'High Accuracy'}
                </Typography>
              </Typography>
            </Box>
          )}

          {!formData.enable_qcpp && (
            <Typography variant="caption" color="text.secondary">
              QCPP integration is disabled. The prediction will use UBF-only optimization.
            </Typography>
          )}
        </Paper>

        {/* Queue System Information */}
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            backgroundColor: alpha(theme.palette.warning.main, 0.05),
            borderColor: alpha(theme.palette.warning.main, 0.3),
          }}
        >
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <QueueIcon color="warning" />
            <Typography variant="subtitle1" fontWeight="bold">
              Queue-Based Processing
            </Typography>
          </Box>

          <Alert severity="info" icon={<InfoIcon />} sx={{ mb: 2 }}>
            <AlertTitle>How Our Queue System Works</AlertTitle>
            <Typography variant="body2" paragraph>
              To ensure fair access and optimal performance, predictions are processed through a first-come, first-served queue system.
            </Typography>
            <Stack spacing={1}>
              <Box display="flex" alignItems="flex-start" gap={1}>
                <ScheduleIcon fontSize="small" color="action" sx={{ mt: 0.3 }} />
                <Typography variant="body2">
                  <strong>Queue Position:</strong> After submission, you'll see your position in the queue and estimated wait time.
                </Typography>
              </Box>
              <Box display="flex" alignItems="flex-start" gap={1}>
                <QueueIcon fontSize="small" color="action" sx={{ mt: 0.3 }} />
                <Typography variant="body2">
                  <strong>Processing Order:</strong> Jobs are processed one at a time in the order they were submitted.
                </Typography>
              </Box>
              <Box display="flex" alignItems="flex-start" gap={1}>
                <SpeedIcon fontSize="small" color="action" sx={{ mt: 0.3 }} />
                <Typography variant="body2">
                  <strong>Real-time Updates:</strong> Monitor your prediction's progress live once it starts processing.
                </Typography>
              </Box>
            </Stack>
          </Alert>

          <Typography variant="caption" color="text.secondary">
            💡 Tip: Smaller sequences (30-150 residues) typically complete faster. During peak hours, 
            queue times may be longer.
          </Typography>
        </Paper>

        {/* Estimated Time */}
        <Paper
          variant="outlined"
          sx={{
            p: 2,
            backgroundColor: alpha(theme.palette.info.main, 0.05),
            borderColor: alpha(theme.palette.info.main, 0.2),
          }}
        >
          <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
            Estimated Completion Time
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Approximately{' '}
            {Math.ceil(((formData.iterations || 1000) * 0.5) / 60)} minutes
            {formData.enable_qcpp && ' (with QCPP integration)'}
          </Typography>
        </Paper>
      </Stack>
    </Box>
  );
};

export default ReviewStep;
