/**
 * OAuthCallback - Handles OAuth provider callbacks
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
import { CheckCircle, Error as ErrorIcon } from '@mui/icons-material';
import { oauthService } from '../services/oauthService';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';

const pulse = keyframes`
  0%, 100% { opacity: 0.6; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.05); }
`;

const OAuthCallback: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { provider } = useParams<{ provider: string }>();
  const { showSuccess, showError } = useNotification();
  const { refreshToken } = useAuth();

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isNewUser, setIsNewUser] = useState(false);

  useEffect(() => {
    const handleCallback = async () => {
      // Get code and state from URL params
      const code = searchParams.get('code');
      const state = searchParams.get('state');
      const error = searchParams.get('error');
      const errorDescription = searchParams.get('error_description');

      // Check for OAuth error
      if (error) {
        setStatus('error');
        setErrorMessage(errorDescription || `OAuth error: ${error}`);
        showError(errorDescription || `OAuth error: ${error}`, 'Authentication Failed');
        return;
      }

      // Validate we have required params
      if (!code || !state) {
        setStatus('error');
        setErrorMessage('Missing authorization code or state parameter');
        showError('Invalid OAuth callback', 'Authentication Failed');
        return;
      }

      // Validate state matches what we stored (CSRF protection)
      const storedState = sessionStorage.getItem('oauth_state');
      const storedProvider = sessionStorage.getItem('oauth_provider');

      if (state !== storedState) {
        setStatus('error');
        setErrorMessage('Invalid state parameter. This may be a security issue.');
        showError('Invalid state parameter', 'Security Error');
        return;
      }

      // Clear stored state
      sessionStorage.removeItem('oauth_state');
      sessionStorage.removeItem('oauth_provider');

      // Determine provider from URL or stored value
      const actualProvider = provider || storedProvider;

      try {
        let response;

        if (actualProvider === 'google') {
          response = await oauthService.googleCallback({ code, state });
        } else if (actualProvider === 'github') {
          response = await oauthService.githubCallback({ code, state });
        } else {
          throw new Error(`Unknown OAuth provider: ${actualProvider}`);
        }

        // Store tokens
        oauthService.storeOAuthTokens(response);

        // Update auth state
        setStatus('success');
        setIsNewUser(response.is_new_user);

        if (response.is_new_user) {
          showSuccess(`Welcome, ${response.user.username}! Your account has been created.`, 'Account Created');
        } else {
          showSuccess(`Welcome back, ${response.user.username}!`, 'Login Successful');
        }

        // Redirect after a short delay
        setTimeout(() => {
          // Force a page reload to update auth state everywhere
          window.location.href = '/dashboard';
        }, 1500);
      } catch (err) {
        console.error('OAuth callback error:', err);
        const apiError = err as { response?: { data?: { detail?: string } } };
        const message = apiError.response?.data?.detail || 'Authentication failed. Please try again.';
        
        setStatus('error');
        setErrorMessage(message);
        showError(message, 'Authentication Failed');
      }
    };

    handleCallback();
  }, [searchParams, provider, navigate, showSuccess, showError, refreshToken]);

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
                Completing Sign In...
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Please wait while we verify your credentials.
              </Typography>
            </>
          )}

          {/* Success State */}
          {status === 'success' && (
            <>
              <CheckCircle
                sx={{ fontSize: 64, color: 'success.main', mb: 2 }}
              />
              <Typography variant="h6" gutterBottom>
                {isNewUser ? 'Account Created!' : 'Welcome Back!'}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {isNewUser
                  ? 'Your account has been created successfully.'
                  : 'You have been signed in successfully.'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Redirecting to dashboard...
              </Typography>
            </>
          )}

          {/* Error State */}
          {status === 'error' && (
            <>
              <ErrorIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Authentication Failed
              </Typography>
              <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
                {errorMessage}
              </Alert>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button
                  variant="contained"
                  onClick={() => navigate('/login')}
                >
                  Back to Login
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => navigate('/register')}
                >
                  Create Account
                </Button>
              </Box>
            </>
          )}
        </Paper>
      </Container>
    </Box>
  );
};

export default OAuthCallback;
