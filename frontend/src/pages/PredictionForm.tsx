import React, { useState } from 'react';
import {
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Button,
  Paper,
  alpha,
  useTheme,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { predictionService } from '../services';
import SequenceStep from '../components/prediction/SequenceStep';
import ConfigurationStep from '../components/prediction/ConfigurationStep';
import ReviewStep from '../components/prediction/ReviewStep';
import type { PredictionCreate } from '../types/api';
import { ErrorAlert } from '../components/common';

const steps = ['Sequence Input', 'Configuration', 'Review & Submit'];

interface FormData {
  sequence: string;
  native_pdb_id?: string;
  iterations?: number;
  consciousness_level?: number;
  consistency?: number;
  enable_qcpp?: boolean;
  qcpp_config?: string;
  checkpoint_interval?: number;
  enable_mediators?: boolean;
  mediator_count?: number;
  enable_refinement?: boolean;
}

const PredictionForm: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState<FormData>({
    sequence: '',
    iterations: 1000,
    consciousness_level: 8.0,
    consistency: 0.6,
    enable_qcpp: true,
    qcpp_config: 'default',
    checkpoint_interval: 100,
    enable_mediators: false,
    mediator_count: 3,
    enable_refinement: false,
  });

  const createPredictionMutation = useMutation({
    mutationFn: (data: PredictionCreate) => predictionService.createPrediction(data),
    onSuccess: (response) => {
      // Navigate to monitoring page for the new prediction
      navigate(`/monitor/${response.id}`);
    },
  });

  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      // Submit form - transform data to match backend schema
      const predictionData: PredictionCreate = {
        sequence: formData.sequence,
        configuration: {
          iterations: formData.iterations || 1000,
          agents: 10, // Default value
          diversity: 'balanced', // Default value
          enable_checkpointing: true,
          checkpoint_interval: formData.checkpoint_interval || 100,
          native_pdb: formData.native_pdb_id || undefined,
          qcpp_config: formData.enable_qcpp ? (formData.qcpp_config || 'default') : undefined,
          enable_mediators: formData.enable_mediators || false,
          mediator_count: formData.mediator_count || 3,
          enable_refinement: formData.enable_refinement || false,
        },
      };
      createPredictionMutation.mutate(predictionData);
    } else {
      setActiveStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleFormDataChange = (updates: Partial<FormData>) => {
    setFormData((prev) => ({ ...prev, ...updates }));
  };

  const isStepValid = (): boolean => {
    switch (activeStep) {
      case 0:
        // Sequence step - must have a sequence
        return formData.sequence.length > 0;
      case 1:
        // Configuration step - iterations must be > 0
        return (formData.iterations || 0) > 0;
      case 2:
        // Review step - always valid
        return true;
      default:
        return false;
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <SequenceStep
            formData={formData}
            onChange={handleFormDataChange}
          />
        );
      case 1:
        return (
          <ConfigurationStep
            formData={formData}
            onChange={handleFormDataChange}
          />
        );
      case 2:
        return <ReviewStep formData={formData} />;
      default:
        return null;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          New Prediction
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Create a new protein structure prediction
        </Typography>
      </Box>

      {/* Error Alert */}
      {createPredictionMutation.isError && (
        <Box mb={3}>
          <ErrorAlert
            message={
              createPredictionMutation.error instanceof Error
                ? createPredictionMutation.error.message
                : 'Failed to create prediction'
            }
          />
        </Box>
      )}

      {/* Stepper */}
      <Paper
        elevation={2}
        sx={{
          p: 4,
          background: `linear-gradient(135deg, ${alpha(
            theme.palette.primary.main,
            0.02
          )} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
        }}
      >
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Step Content */}
        <Box sx={{ minHeight: 400 }}>{renderStepContent(activeStep)}</Box>

        {/* Navigation Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            onClick={handleBack}
            disabled={activeStep === 0}
            variant="outlined"
          >
            Back
          </Button>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="text"
              onClick={() => navigate('/dashboard')}
            >
              Cancel
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={!isStepValid() || createPredictionMutation.isPending}
            >
              {activeStep === steps.length - 1
                ? createPredictionMutation.isPending
                  ? 'Submitting...'
                  : 'Submit'
                : 'Next'}
            </Button>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default PredictionForm;
