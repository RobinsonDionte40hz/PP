/**
 * VerifyEmail - Handles email verification token
 */
import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams, useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Button,
  keyframes,
} from '@mui/material';
import {
  CheckCircle,
  Error as ErrorIcon,
  MarkEmailRead as VerifiedIcon,
} from '@mui/icons-material';
import { oauthService } from '../services/oauthService';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';

const pulse = keyframes`
  0%, 100% { opacity: 0.6; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.05); }
`;

const VerifyEmail: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { token: pathToken } = useParams<{ token: string }>();
  const { showSuccess, showError } = useNotification();
  const { user } = useAuth();

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [verifiedEmail, setVerifiedEmail] = useState<string>('');

  useEffect(() => {
    const verifyEmailToken = async () => {
      // Get token from URL path or query param
      const token = pathToken || searchParams.get('token');

      if (!token) {
        setStatus('error');
        setErrorMessage('No verification token provided. Please check your email link.');
        return;
      }

      try {
        const response = await oauthService.verifyEmail(token);
        
        setStatus('success');
        setVerifiedEmail(response.email);
        showSuccess('Your email has been verified successfully!', 'Email Verified');

        // Redirect after a delay
        setTimeout(() => {
          if (user) {
            navigate('/dashboard');
          } else {
            navigate('/login');
          }
        }, 3000);
      } catch (err) {
        console.error('Email verification error:', err);
        const apiError = err as { response?: { data?: { detail?: string } } };
        const message = apiError.response?.data?.detail || 'Verification failed. The token may be invalid or expired.';
        
        setStatus('error');
        setErrorMessage(message);
        showError(message, 'Verification Failed');
      }
    };

    verifyEmailToken();
  }, [pathToken, searchParams, navigate, showSuccess, showError, user]);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: (theme) =>
          theme.palette.mode === 'dark'
            ? 'linear-gradient(-45deg, #293B5F 0%, #47597E 50%, #293B5F 100%)'
            : 'linear-gradient(-45deg, #47597E 0%, #B2AB8C 50%, #47597E 100%)',
      }}
    >
      <Container maxWidth="sm">
        <Paper
          elevation={24}
          sx={{
            p: 4,
            textAlign: 'center',
            borderRadius: 4,
            background: (theme) =>
              theme.palette.mode === 'dark'
                ? 'rgba(30, 30, 30, 0.9)'
                : 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(20px)',
          }}
        >
          {/* Logo */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              mb: 3,
              animation: status === 'loading' ? `${pulse} 2s ease-in-out infinite` : 'none',
            }}
          >
            <img
              src="/emergentfoldslogo.png"
              alt="EmergentFolds"
              style={{ height: 80, width: 'auto' }}
            />
          </Box>

          {/* Loading State */}
          {status === 'loading' && (
            <>
              <CircularProgress size={48} sx={{ mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Verifying Email...
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Please wait while we verify your email address.
              </Typography>
            </>
          )}

          {/* Success State */}
          {status === 'success' && (
            <>
              <VerifiedIcon
                sx={{ fontSize: 64, color: 'success.main', mb: 2 }}
              />
              <Typography variant="h6" gutterBottom>
                Email Verified!
              </Typography>
              <Typography variant="body1" sx={{ mb: 1 }}>
                <strong>{verifiedEmail}</strong>
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Your email has been verified successfully. You can now create predictions.
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Redirecting...
              </Typography>
            </>
          )}

          {/* Error State */}
          {status === 'error' && (
            <>
              <ErrorIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Verification Failed
              </Typography>
              <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
                {errorMessage}
              </Alert>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
                {user ? (
                  <>
                    <Button
                      variant="contained"
                      onClick={() => navigate('/dashboard')}
                    >
                      Go to Dashboard
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => navigate('/dashboard/settings')}
                    >
                      Account Settings
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      variant="contained"
                      onClick={() => navigate('/login')}
                    >
                      Login
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => navigate('/register')}
                    >
                      Create Account
                    </Button>
                  </>
                )}
              </Box>
            </>
          )}
        </Paper>
      </Container>
    </Box>
  );
};

export default VerifyEmail;
