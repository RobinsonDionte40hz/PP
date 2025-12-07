/**
 * OAuthButtons - Social login buttons for Google and GitHub
 */
import React from 'react';
import {
  Box,
  Button,
  Divider,
  Typography,
  CircularProgress,
} from '@mui/material';
import { Google as GoogleIcon, GitHub as GitHubIcon } from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';
import { oauthService } from '../../services/oauthService';
import { useNotification } from '../../hooks/useNotification';

interface OAuthButtonsProps {
  /** Text shown above the buttons */
  dividerText?: string;
  /** Called when OAuth flow starts (to show loading state) */
  onOAuthStart?: () => void;
  /** Size of buttons */
  size?: 'small' | 'medium' | 'large';
  /** Show full width buttons */
  fullWidth?: boolean;
}

const OAuthButtons: React.FC<OAuthButtonsProps> = ({
  dividerText = 'Or continue with',
  onOAuthStart,
  size = 'large',
  fullWidth = true,
}) => {
  const { showError } = useNotification();

  // Fetch OAuth configuration
  const { data: config, isLoading: configLoading } = useQuery({
    queryKey: ['oauth-config'],
    queryFn: oauthService.getOAuthConfig,
    staleTime: 300000, // 5 minutes
  });

  // Google OAuth mutation
  const googleMutation = useMutation({
    mutationFn: () => oauthService.initiateGoogleOAuth(),
    onSuccess: (response) => {
      // Store state for CSRF validation on callback
      sessionStorage.setItem('oauth_state', response.state);
      sessionStorage.setItem('oauth_provider', 'google');
      // Redirect to Google
      window.location.href = response.authorization_url;
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to initiate Google login';
      showError(message, 'OAuth Error');
    },
  });

  // GitHub OAuth mutation
  const githubMutation = useMutation({
    mutationFn: () => oauthService.initiateGitHubOAuth(),
    onSuccess: (response) => {
      // Store state for CSRF validation on callback
      sessionStorage.setItem('oauth_state', response.state);
      sessionStorage.setItem('oauth_provider', 'github');
      // Redirect to GitHub
      window.location.href = response.authorization_url;
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to initiate GitHub login';
      showError(message, 'OAuth Error');
    },
  });

  const handleGoogleClick = () => {
    onOAuthStart?.();
    googleMutation.mutate();
  };

  const handleGitHubClick = () => {
    onOAuthStart?.();
    githubMutation.mutate();
  };

  // Check if any OAuth provider is enabled
  const googleEnabled = config?.google?.enabled;
  const githubEnabled = config?.github?.enabled;
  const anyEnabled = googleEnabled || githubEnabled;

  // Don't render if no OAuth providers are enabled
  if (configLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (!anyEnabled) {
    return null;
  }

  const isLoading = googleMutation.isPending || githubMutation.isPending;

  return (
    <Box sx={{ width: '100%' }}>
      {/* Divider */}
      <Divider sx={{ my: 3 }}>
        <Typography variant="body2" color="text.secondary">
          {dividerText}
        </Typography>
      </Divider>

      {/* OAuth Buttons */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: fullWidth ? 'column' : 'row',
          gap: 2,
        }}
      >
        {/* Google Button */}
        {googleEnabled && (
          <Button
            variant="outlined"
            size={size}
            fullWidth={fullWidth}
            onClick={handleGoogleClick}
            disabled={isLoading}
            startIcon={
              googleMutation.isPending ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <GoogleIcon />
              )
            }
            sx={{
              borderColor: 'divider',
              color: 'text.primary',
              bgcolor: 'background.paper',
              py: size === 'large' ? 1.5 : 1,
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'action.hover',
              },
            }}
          >
            {googleMutation.isPending ? 'Connecting...' : 'Continue with Google'}
          </Button>
        )}

        {/* GitHub Button */}
        {githubEnabled && (
          <Button
            variant="outlined"
            size={size}
            fullWidth={fullWidth}
            onClick={handleGitHubClick}
            disabled={isLoading}
            startIcon={
              githubMutation.isPending ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <GitHubIcon />
              )
            }
            sx={{
              borderColor: 'divider',
              color: 'text.primary',
              bgcolor: 'background.paper',
              py: size === 'large' ? 1.5 : 1,
              '&:hover': {
                borderColor: '#333',
                bgcolor: 'action.hover',
              },
            }}
          >
            {githubMutation.isPending ? 'Connecting...' : 'Continue with GitHub'}
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default OAuthButtons;
