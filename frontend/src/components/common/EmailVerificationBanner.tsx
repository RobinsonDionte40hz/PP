/**
 * EmailVerificationBanner - Shows banner for unverified users
 */
import React, { useState } from 'react';
import {
  Alert,
  AlertTitle,
  Button,
  Box,
  CircularProgress,
  Collapse,
  TextField,
  Typography,
} from '@mui/material';
import {
  MarkEmailRead as VerifiedIcon,
  Send as SendIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { oauthService } from '../../services/oauthService';
import { useNotification } from '../../hooks/useNotification';

interface EmailVerificationBannerProps {
  /** Allow user to dismiss the banner temporarily */
  dismissible?: boolean;
}

const EmailVerificationBanner: React.FC<EmailVerificationBannerProps> = ({
  dismissible = true,
}) => {
  const [dismissed, setDismissed] = useState(false);
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [email, setEmail] = useState('');
  const { showSuccess, showError } = useNotification();
  const queryClient = useQueryClient();

  // Fetch verification status
  const { data: status, isLoading, error } = useQuery({
    queryKey: ['verification-status'],
    queryFn: oauthService.getVerificationStatus,
    staleTime: 60000, // 1 minute
    retry: false, // Don't retry on 401/403
  });

  // Send verification email mutation
  const sendMutation = useMutation({
    mutationFn: (emailToVerify?: string) => oauthService.sendVerificationEmail(emailToVerify),
    onSuccess: (response) => {
      showSuccess(`Verification email sent to ${response.email}`, 'Email Sent');
      setShowEmailInput(false);
      setEmail('');
      queryClient.invalidateQueries({ queryKey: ['verification-status'] });
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to send verification email';
      showError(message, 'Error');
    },
  });

  // Don't show if loading, error, or dismissed
  if (isLoading || error || dismissed) return null;

  // Don't show if already verified or verification not required
  if (!status || status.email_verified || !status.verification_required) return null;

  const hasEmail = !!status.email;
  const canResend = status.can_resend;

  const handleSendVerification = () => {
    if (hasEmail) {
      sendMutation.mutate(undefined);
    } else {
      setShowEmailInput(true);
    }
  };

  const handleSubmitEmail = (e: React.FormEvent) => {
    e.preventDefault();
    if (email.trim()) {
      sendMutation.mutate(email.trim());
    }
  };

  return (
    <Collapse in={!dismissed}>
      <Alert
        severity="warning"
        icon={<VerifiedIcon />}
        action={
          dismissible && (
            <Button
              color="inherit"
              size="small"
              onClick={() => setDismissed(true)}
              startIcon={<CloseIcon />}
            >
              Dismiss
            </Button>
          )
        }
        sx={{ mb: 2 }}
      >
        <AlertTitle>Email Verification Required</AlertTitle>
        <Box>
          {hasEmail ? (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Please verify your email address ({status.email}) to create predictions.
              </Typography>
              <Button
                variant="outlined"
                size="small"
                color="warning"
                onClick={handleSendVerification}
                disabled={!canResend || sendMutation.isPending}
                startIcon={sendMutation.isPending ? <CircularProgress size={16} /> : <SendIcon />}
              >
                {sendMutation.isPending
                  ? 'Sending...'
                  : canResend
                  ? 'Resend Verification Email'
                  : 'Verification Email Sent'}
              </Button>
              {!canResend && status.verification_sent_at && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                  You can request another verification email after some time.
                </Typography>
              )}
            </>
          ) : (
            <>
              {!showEmailInput ? (
                <>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Please add and verify an email address to create predictions.
                  </Typography>
                  <Button
                    variant="outlined"
                    size="small"
                    color="warning"
                    onClick={() => setShowEmailInput(true)}
                    startIcon={<SendIcon />}
                  >
                    Add Email Address
                  </Button>
                </>
              ) : (
                <Box component="form" onSubmit={handleSubmitEmail} sx={{ mt: 1 }}>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
                    <TextField
                      size="small"
                      type="email"
                      label="Email Address"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={sendMutation.isPending}
                      sx={{ minWidth: 250 }}
                      autoFocus
                    />
                    <Button
                      type="submit"
                      variant="contained"
                      color="warning"
                      size="medium"
                      disabled={!email.trim() || sendMutation.isPending}
                      sx={{ height: 40 }}
                    >
                      {sendMutation.isPending ? (
                        <CircularProgress size={20} color="inherit" />
                      ) : (
                        'Send'
                      )}
                    </Button>
                    <Button
                      variant="text"
                      color="inherit"
                      size="medium"
                      onClick={() => {
                        setShowEmailInput(false);
                        setEmail('');
                      }}
                      sx={{ height: 40 }}
                    >
                      Cancel
                    </Button>
                  </Box>
                </Box>
              )}
            </>
          )}
        </Box>
      </Alert>
    </Collapse>
  );
};

export default EmailVerificationBanner;
