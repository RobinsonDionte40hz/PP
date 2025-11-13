import React from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Typography,
  Chip,
} from '@mui/material';
import { CheckCircle as CheckIcon } from '@mui/icons-material';

interface PhaseProgressProps {
  currentPhase: number;
  totalPhases: number;
  phaseData: Array<{
    phase: number;
    total_proteins?: number;
    successful_predictions?: number;
    failed_predictions?: number;
  }>;
}

const PhaseProgress: React.FC<PhaseProgressProps> = ({
  currentPhase,
  totalPhases,
  phaseData,
}) => {
  const phases = Array.from({ length: totalPhases }, (_, i) => i + 1);

  const getPhaseStatus = (phase: number) => {
    if (phase < currentPhase) return 'completed';
    if (phase === currentPhase) return 'active';
    return 'pending';
  };

  const getPhaseLabel = (phase: number) => {
    return `Phase ${phase}`;
  };

  const getPhaseDescription = (phase: number) => {
    const phaseInfo = phaseData.find((p) => p.phase === phase);
    if (!phaseInfo) {
      return 'Pending...';
    }

    const total = phaseInfo.total_proteins || 0;
    const successful = phaseInfo.successful_predictions || 0;
    const failed = phaseInfo.failed_predictions || 0;
    const completed = successful + failed;

    return `${completed}/${total} proteins completed (${successful} successful, ${failed} failed)`;
  };

  const getPhaseColor = (phase: number) => {
    const status = getPhaseStatus(phase);
    if (status === 'completed') return 'success';
    if (status === 'active') return 'primary';
    return 'default';
  };

  return (
    <Stepper activeStep={currentPhase - 1} orientation="vertical">
      {phases.map((phase) => {
        const status = getPhaseStatus(phase);
        return (
          <Step key={phase} completed={status === 'completed'}>
            <StepLabel
              optional={
                status === 'completed' && (
                  <Chip
                    icon={<CheckIcon />}
                    label="Completed"
                    size="small"
                    color="success"
                    sx={{ mt: 0.5 }}
                  />
                )
              }
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle2">{getPhaseLabel(phase)}</Typography>
                <Chip
                  label={status}
                  size="small"
                  color={getPhaseColor(phase)}
                  variant="outlined"
                />
              </Box>
            </StepLabel>
            <StepContent>
              <Typography variant="body2" color="text.secondary">
                {getPhaseDescription(phase)}
              </Typography>
            </StepContent>
          </Step>
        );
      })}
    </Stepper>
  );
};

export default PhaseProgress;
