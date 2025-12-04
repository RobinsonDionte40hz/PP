import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Checkbox,
  Alert,
  CircularProgress,
  Collapse,
  IconButton,
  Chip,
  Stack,
} from '@mui/material';
import {
  Feedback as FeedbackIcon,
  Send as SendIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  BugReport as BugIcon,
  Lightbulb as FeatureIcon,
  TrendingUp as ImprovementIcon,
  MoreHoriz as OtherIcon,
  CheckCircle as SuccessIcon,
} from '@mui/icons-material';
import { useMutation } from '@tanstack/react-query';
import api from '../../services/api';

interface FeedbackFormData {
  category: 'bug' | 'feature' | 'improvement' | 'other';
  subject: string;
  message: string;
  email: string;
  include_system_info: boolean;
}

interface FeedbackResponse {
  success: boolean;
  message: string;
  feedback_id: string;
}

const FeedbackCard: React.FC = () => {
  const [expanded, setExpanded] = useState(false);
  const [formData, setFormData] = useState<FeedbackFormData>({
    category: 'improvement',
    subject: '',
    message: '',
    email: '',
    include_system_info: false,
  });
  const [submitted, setSubmitted] = useState(false);
  const [feedbackId, setFeedbackId] = useState<string | null>(null);

  const submitFeedback = useMutation({
    mutationFn: async (data: FeedbackFormData): Promise<FeedbackResponse> => {
      const response = await api.post('/api/feedback', {
        category: data.category,
        subject: data.subject,
        message: data.message,
        email: data.email || undefined,
        include_system_info: data.include_system_info,
      });
      return response.data;
    },
    onSuccess: (data) => {
      setSubmitted(true);
      setFeedbackId(data.feedback_id);
      // Reset form after success
      setTimeout(() => {
        setFormData({
          category: 'improvement',
          subject: '',
          message: '',
          email: '',
          include_system_info: false,
        });
      }, 500);
    },
  });

  const handleChange = (field: keyof FeedbackFormData) => (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement> | { target: { value: string } }
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const handleCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({
      ...prev,
      include_system_info: event.target.checked,
    }));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (formData.subject.length >= 5 && formData.message.length >= 10) {
      submitFeedback.mutate(formData);
    }
  };

  const handleNewFeedback = () => {
    setSubmitted(false);
    setFeedbackId(null);
  };

  const isValid = formData.subject.length >= 5 && formData.message.length >= 10;

  return (
    <Card
      sx={{
        background: (theme) =>
          theme.palette.mode === 'dark'
            ? 'linear-gradient(135deg, rgba(41, 59, 95, 0.3) 0%, rgba(71, 89, 126, 0.2) 100%)'
            : 'linear-gradient(135deg, rgba(219, 230, 253, 0.5) 0%, rgba(178, 171, 140, 0.2) 100%)',
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <CardContent>
        {/* Header - Always visible */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <FeedbackIcon color="primary" />
            <Typography variant="h6" fontWeight="bold">
              Send Feedback
            </Typography>
            <Chip
              label="Help us improve"
              size="small"
              variant="outlined"
              sx={{ ml: 1 }}
            />
          </Box>
          <IconButton size="small">
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>

        {/* Expandable Content */}
        <Collapse in={expanded}>
          <Box sx={{ mt: 3 }}>
            {submitted ? (
              /* Success State */
              <Box
                sx={{
                  textAlign: 'center',
                  py: 4,
                }}
              >
                <SuccessIcon
                  sx={{ fontSize: 64, color: 'success.main', mb: 2 }}
                />
                <Typography variant="h6" gutterBottom>
                  Thank you for your feedback!
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Your feedback has been submitted successfully.
                </Typography>
                {feedbackId && (
                  <Chip
                    label={`Reference: ${feedbackId}`}
                    size="small"
                    sx={{ mt: 1 }}
                  />
                )}
                <Box sx={{ mt: 3 }}>
                  <Button
                    variant="outlined"
                    onClick={handleNewFeedback}
                    startIcon={<FeedbackIcon />}
                  >
                    Send More Feedback
                  </Button>
                </Box>
              </Box>
            ) : (
              /* Form State */
              <form onSubmit={handleSubmit}>
                <Stack spacing={2.5}>
                  {/* Category Selection */}
                  <FormControl fullWidth size="small">
                    <InputLabel>Category</InputLabel>
                    <Select
                      value={formData.category}
                      label="Category"
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          category: e.target.value as FeedbackFormData['category'],
                        }))
                      }
                    >
                      <MenuItem value="bug">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <BugIcon fontSize="small" color="error" />
                          Bug Report
                        </Box>
                      </MenuItem>
                      <MenuItem value="feature">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <FeatureIcon fontSize="small" color="primary" />
                          Feature Request
                        </Box>
                      </MenuItem>
                      <MenuItem value="improvement">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <ImprovementIcon fontSize="small" color="success" />
                          Improvement Suggestion
                        </Box>
                      </MenuItem>
                      <MenuItem value="other">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <OtherIcon fontSize="small" />
                          Other
                        </Box>
                      </MenuItem>
                    </Select>
                  </FormControl>

                  {/* Subject */}
                  <TextField
                    fullWidth
                    size="small"
                    label="Subject"
                    placeholder="Brief summary of your feedback"
                    value={formData.subject}
                    onChange={handleChange('subject')}
                    required
                    inputProps={{ minLength: 5, maxLength: 200 }}
                    helperText={`${formData.subject.length}/200 characters (min 5)`}
                    error={formData.subject.length > 0 && formData.subject.length < 5}
                  />

                  {/* Message */}
                  <TextField
                    fullWidth
                    size="small"
                    label="Message"
                    placeholder="Describe your feedback in detail..."
                    value={formData.message}
                    onChange={handleChange('message')}
                    required
                    multiline
                    rows={4}
                    inputProps={{ minLength: 10, maxLength: 5000 }}
                    helperText={`${formData.message.length}/5000 characters (min 10)`}
                    error={formData.message.length > 0 && formData.message.length < 10}
                  />

                  {/* Email (optional) */}
                  <TextField
                    fullWidth
                    size="small"
                    label="Email (optional)"
                    placeholder="your@email.com"
                    type="email"
                    value={formData.email}
                    onChange={handleChange('email')}
                    helperText="Provide if you'd like us to follow up"
                  />

                  {/* Include System Info */}
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={formData.include_system_info}
                        onChange={handleCheckboxChange}
                        size="small"
                      />
                    }
                    label={
                      <Typography variant="body2" color="text.secondary">
                        Include browser and system information
                      </Typography>
                    }
                  />

                  {/* Error Alert */}
                  {submitFeedback.isError && (
                    <Alert severity="error">
                      Failed to submit feedback. Please try again.
                    </Alert>
                  )}

                  {/* Submit Button */}
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                    <Button
                      variant="outlined"
                      onClick={() => setExpanded(false)}
                      disabled={submitFeedback.isPending}
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      variant="contained"
                      disabled={!isValid || submitFeedback.isPending}
                      startIcon={
                        submitFeedback.isPending ? (
                          <CircularProgress size={16} color="inherit" />
                        ) : (
                          <SendIcon />
                        )
                      }
                    >
                      {submitFeedback.isPending ? 'Sending...' : 'Send Feedback'}
                    </Button>
                  </Box>
                </Stack>
              </form>
            )}
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default FeedbackCard;
