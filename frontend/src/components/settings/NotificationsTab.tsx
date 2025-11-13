import React from 'react';
import {
  Box,
  Typography,
  FormControlLabel,
  Switch,
  TextField,
  Divider,
  Alert,
  Stack,
} from '@mui/material';

interface NotificationsTabProps {
  settings: {
    enableDesktopNotifications: boolean;
    notifyOnCompletion: boolean;
    notifyOnError: boolean;
    notifyOnMilestone: boolean;
    emailNotifications: boolean;
    emailAddress: string;
  };
  onChange: (key: string, value: boolean | string) => void;
}

const NotificationsTab: React.FC<NotificationsTabProps> = ({ settings, onChange }) => {
  return (
    <Box sx={{ px: 3 }}>
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure how and when you receive notifications about prediction status updates.
      </Alert>

      {/* Desktop Notifications */}
      <Typography variant="h6" gutterBottom>
        Desktop Notifications
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={2}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.enableDesktopNotifications}
              onChange={(e) => onChange('enableDesktopNotifications', e.target.checked)}
            />
          }
          label="Enable Desktop Notifications"
        />

        <Box sx={{ ml: 4 }}>
          <Stack spacing={1}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.notifyOnCompletion}
                  onChange={(e) => onChange('notifyOnCompletion', e.target.checked)}
                  disabled={!settings.enableDesktopNotifications}
                />
              }
              label="Notify when prediction completes"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.notifyOnError}
                  onChange={(e) => onChange('notifyOnError', e.target.checked)}
                  disabled={!settings.enableDesktopNotifications}
                />
              }
              label="Notify on errors or failures"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.notifyOnMilestone}
                  onChange={(e) => onChange('notifyOnMilestone', e.target.checked)}
                  disabled={!settings.enableDesktopNotifications}
                />
              }
              label="Notify on milestones (every 25% progress)"
            />
          </Stack>
        </Box>
      </Stack>

      {/* Email Notifications */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Email Notifications
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Stack spacing={3}>
        <FormControlLabel
          control={
            <Switch
              checked={settings.emailNotifications}
              onChange={(e) => onChange('emailNotifications', e.target.checked)}
            />
          }
          label="Enable Email Notifications"
        />

        <TextField
          label="Email Address"
          type="email"
          value={settings.emailAddress}
          onChange={(e) => onChange('emailAddress', e.target.value)}
          fullWidth
          disabled={!settings.emailNotifications}
          helperText="Receive notifications at this email address"
        />
      </Stack>

      {/* Permission Check */}
      {settings.enableDesktopNotifications && (
        <Alert severity="warning" sx={{ mt: 4 }}>
          <Typography variant="body2">
            <strong>Browser Permission Required:</strong> Make sure to allow notifications in your
            browser settings for this website.
          </Typography>
        </Alert>
      )}
    </Box>
  );
};

export default NotificationsTab;
