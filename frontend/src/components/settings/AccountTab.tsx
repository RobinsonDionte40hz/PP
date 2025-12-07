/**
 * AccountTab - Account settings including OAuth linking and email verification
 */
import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Divider,
  Alert,
  Chip,
  CircularProgress,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  Google as GoogleIcon,
  GitHub as GitHubIcon,
  Link as LinkIcon,
  LinkOff as LinkOffIcon,
  Email as EmailIcon,
  Lock as LockIcon,
  CheckCircle as VerifiedIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { oauthService } from '../../services/oauthService';
import { useAuth } from '../../hooks/useAuth';
import { useNotification } from '../../hooks/useNotification';
import QuotaDisplay from '../common/QuotaDisplay';

const AccountTab: React.FC = () => {
  const { user } = useAuth();
  const { showSuccess, showError } = useNotification();
  const queryClient = useQueryClient();

  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Fetch linked accounts
  const { data: linkedAccounts, isLoading: linkedLoading } = useQuery({
    queryKey: ['linked-accounts'],
    queryFn: oauthService.getLinkedAccounts,
    retry: false,
  });

  // Fetch verification status
  const { data: verificationStatus, isLoading: verificationLoading } = useQuery({
    queryKey: ['verification-status'],
    queryFn: oauthService.getVerificationStatus,
    retry: false,
  });

  // Fetch OAuth config
  const { data: oauthConfig } = useQuery({
    queryKey: ['oauth-config'],
    queryFn: oauthService.getOAuthConfig,
    staleTime: 300000,
  });

  // Link account mutation
  const linkMutation = useMutation({
    mutationFn: (provider: 'google' | 'github') => oauthService.initiateLink(provider),
    onSuccess: (response) => {
      sessionStorage.setItem('oauth_state', response.state);
      sessionStorage.setItem('oauth_link_mode', 'true');
      window.location.href = response.authorization_url;
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to initiate account linking';
      showError(message, 'Link Error');
    },
  });

  // Unlink account mutation
  const unlinkMutation = useMutation({
    mutationFn: (provider: 'google' | 'github') => oauthService.unlinkAccount(provider),
    onSuccess: () => {
      showSuccess('Account unlinked successfully', 'Success');
      queryClient.invalidateQueries({ queryKey: ['linked-accounts'] });
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to unlink account';
      showError(message, 'Error');
    },
  });

  // Set password mutation
  const setPasswordMutation = useMutation({
    mutationFn: (password: string) => oauthService.setPassword(password),
    onSuccess: () => {
      showSuccess('Password set successfully', 'Success');
      setPasswordDialogOpen(false);
      setNewPassword('');
      setConfirmPassword('');
      queryClient.invalidateQueries({ queryKey: ['linked-accounts'] });
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to set password';
      showError(message, 'Error');
    },
  });

  // Send verification email mutation
  const sendVerificationMutation = useMutation({
    mutationFn: () => oauthService.sendVerificationEmail(),
    onSuccess: (response) => {
      showSuccess(`Verification email sent to ${response.email}`, 'Email Sent');
      queryClient.invalidateQueries({ queryKey: ['verification-status'] });
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const message = err.response?.data?.detail || 'Failed to send verification email';
      showError(message, 'Error');
    },
  });

  const handleLinkAccount = (provider: 'google' | 'github') => {
    linkMutation.mutate(provider);
  };

  const handleUnlinkAccount = (provider: 'google' | 'github') => {
    if (!linkedAccounts?.has_password && getLinkedCount() <= 1) {
      showError(
        'Cannot unlink. Please set a password first or link another account.',
        'Cannot Unlink'
      );
      return;
    }
    unlinkMutation.mutate(provider);
  };

  const handleSetPassword = () => {
    if (newPassword !== confirmPassword) {
      showError('Passwords do not match', 'Validation Error');
      return;
    }
    if (newPassword.length < 8) {
      showError('Password must be at least 8 characters', 'Validation Error');
      return;
    }
    setPasswordMutation.mutate(newPassword);
  };

  const getLinkedCount = () => {
    if (!linkedAccounts) return 0;
    return (linkedAccounts.google ? 1 : 0) + (linkedAccounts.github ? 1 : 0);
  };

  const isLoading = linkedLoading || verificationLoading;

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ px: { xs: 2, sm: 3 } }}>
      {/* Quota Section */}
      <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
        Prediction Quota
      </Typography>
      <QuotaDisplay variant="full" showTier={true} />

      <Divider sx={{ my: 3 }} />

      {/* Email Verification Section */}
      <Typography variant="h6" gutterBottom>
        Email Verification
      </Typography>
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        {user?.email ? (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <EmailIcon color="primary" />
              <Typography>{user.email}</Typography>
              {verificationStatus?.email_verified ? (
                <Chip
                  icon={<VerifiedIcon />}
                  label="Verified"
                  color="success"
                  size="small"
                />
              ) : (
                <Chip
                  icon={<WarningIcon />}
                  label="Not Verified"
                  color="warning"
                  size="small"
                />
              )}
            </Box>
            {!verificationStatus?.email_verified && (
              <Button
                variant="outlined"
                size="small"
                onClick={() => sendVerificationMutation.mutate()}
                disabled={sendVerificationMutation.isPending || !verificationStatus?.can_resend}
                sx={{ mt: 1 }}
              >
                {sendVerificationMutation.isPending ? 'Sending...' : 'Send Verification Email'}
              </Button>
            )}
          </Box>
        ) : (
          <Alert severity="info">
            No email address associated with your account. Add an email to enable account recovery.
          </Alert>
        )}
      </Paper>

      <Divider sx={{ my: 3 }} />

      {/* Linked Accounts Section */}
      <Typography variant="h6" gutterBottom>
        Linked Accounts
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Connect social accounts for easier sign-in. You can link multiple accounts.
      </Typography>

      <Paper variant="outlined">
        <List>
          {/* Google */}
          {oauthConfig?.google?.enabled && (
            <ListItem>
              <ListItemIcon>
                <GoogleIcon />
              </ListItemIcon>
              <ListItemText
                primary="Google"
                secondary={linkedAccounts?.google ? 'Connected' : 'Not connected'}
              />
              <ListItemSecondaryAction>
                {linkedAccounts?.google ? (
                  <Button
                    variant="outlined"
                    color="error"
                    size="small"
                    startIcon={<LinkOffIcon />}
                    onClick={() => handleUnlinkAccount('google')}
                    disabled={unlinkMutation.isPending}
                  >
                    Unlink
                  </Button>
                ) : (
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<LinkIcon />}
                    onClick={() => handleLinkAccount('google')}
                    disabled={linkMutation.isPending}
                  >
                    Link
                  </Button>
                )}
              </ListItemSecondaryAction>
            </ListItem>
          )}

          {/* GitHub */}
          {oauthConfig?.github?.enabled && (
            <>
              <Divider component="li" />
              <ListItem>
                <ListItemIcon>
                  <GitHubIcon />
                </ListItemIcon>
                <ListItemText
                  primary="GitHub"
                  secondary={linkedAccounts?.github ? 'Connected' : 'Not connected'}
                />
                <ListItemSecondaryAction>
                  {linkedAccounts?.github ? (
                    <Button
                      variant="outlined"
                      color="error"
                      size="small"
                      startIcon={<LinkOffIcon />}
                      onClick={() => handleUnlinkAccount('github')}
                      disabled={unlinkMutation.isPending}
                    >
                      Unlink
                    </Button>
                  ) : (
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<LinkIcon />}
                      onClick={() => handleLinkAccount('github')}
                      disabled={linkMutation.isPending}
                    >
                      Link
                    </Button>
                  )}
                </ListItemSecondaryAction>
              </ListItem>
            </>
          )}
        </List>
      </Paper>

      <Divider sx={{ my: 3 }} />

      {/* Password Section */}
      <Typography variant="h6" gutterBottom>
        Password
      </Typography>
      <Paper variant="outlined" sx={{ p: 2 }}>
        {linkedAccounts?.has_password ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <LockIcon color="primary" />
            <Typography>Password is set</Typography>
            <Chip label="Active" color="success" size="small" />
          </Box>
        ) : (
          <Box>
            <Alert severity="warning" sx={{ mb: 2 }}>
              No password set. You can only sign in using linked accounts.
              Set a password to enable email/password login.
            </Alert>
            <Button
              variant="contained"
              startIcon={<LockIcon />}
              onClick={() => setPasswordDialogOpen(true)}
            >
              Set Password
            </Button>
          </Box>
        )}
      </Paper>

      {/* Set Password Dialog */}
      <Dialog open={passwordDialogOpen} onClose={() => setPasswordDialogOpen(false)}>
        <DialogTitle>Set Password</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Set a password to enable email/password login as an alternative to OAuth.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="New Password"
            type="password"
            fullWidth
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            helperText="At least 8 characters with uppercase, lowercase, number, and special character"
          />
          <TextField
            margin="dense"
            label="Confirm Password"
            type="password"
            fullWidth
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPasswordDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSetPassword}
            variant="contained"
            disabled={!newPassword || !confirmPassword || setPasswordMutation.isPending}
          >
            {setPasswordMutation.isPending ? 'Setting...' : 'Set Password'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AccountTab;
