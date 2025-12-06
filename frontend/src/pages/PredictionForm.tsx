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
  enable_hierarchical_folding?: boolean;
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
    enable_hierarchical_folding: false,
  });

  const createPredictionMutation = useMutation({
    mutationFn: (data: PredictionCreate) => predictionService.createPrediction(data),
    onSuccess: (response) => {
      // Navigate to monitoring page for the new prediction
      navigate(`/dashboard/monitor/${response.id}`);
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
          qcpp_config: formData.enable_qcpp ? ((formData.qcpp_config || 'default') as 'default' | 'high_performance' | 'high_accuracy') : undefined,
          enable_mediators: formData.enable_mediators || false,
          mediator_count: formData.mediator_count || 3,
          enable_refinement: formData.enable_refinement || false,
          enable_hierarchical_folding: formData.enable_hierarchical_folding || false,
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
        // If PDB ID is provided, it must be exactly 4 characters
        const hasValidSequence = formData.sequence.length > 0;
        const pdbIdValid = !formData.native_pdb_id || formData.native_pdb_id.length === 4;
        return hasValidSequence && pdbIdValid;
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
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      {/* Header */}
      <Box mb={{ xs: 2, sm: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom sx={{ fontSize: { xs: '1.5rem', sm: '2.125rem' } }}>
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
          p: { xs: 2, sm: 4 },
          background: `linear-gradient(135deg, ${alpha(
            theme.palette.primary.main,
            0.02
          )} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
        }}
      >
        <Stepper activeStep={activeStep} sx={{ mb: { xs: 2, sm: 4 } }} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel sx={{ '& .MuiStepLabel-label': { fontSize: { xs: '0.75rem', sm: '0.875rem' } } }}>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Step Content */}
        <Box sx={{ minHeight: { xs: 300, sm: 400 } }}>{renderStepContent(activeStep)}</Box>

        {/* Navigation Buttons */}
        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' },
          justifyContent: 'space-between', 
          mt: { xs: 2, sm: 4 },
          gap: { xs: 2, sm: 0 }
        }}>
          <Button
            onClick={handleBack}
            disabled={activeStep === 0}
            variant="outlined"
            fullWidth
            sx={{ display: { xs: activeStep === 0 ? 'none' : 'block', sm: 'block' } }}
          >
            Back
          </Button>
          <Box sx={{ 
            display: 'flex', 
            gap: 2, 
            flexDirection: { xs: 'column-reverse', sm: 'row' },
            width: { xs: '100%', sm: 'auto' }
          }}>
            <Button
              variant="text"
              onClick={() => navigate('/dashboard')}
              fullWidth
              sx={{ width: { xs: '100%', sm: 'auto' } }}
            >
              Cancel
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={!isStepValid() || createPredictionMutation.isPending}
              fullWidth
              sx={{ width: { xs: '100%', sm: 'auto' } }}
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
