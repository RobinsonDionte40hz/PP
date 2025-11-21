import { Alert, AlertTitle, Box, Button, Collapse } from '@mui/material';
import { useState } from 'react';
import { toUserFriendlyError } from '../../utils/errorHandling';

interface ErrorAlertProps {
  title?: string;
  message?: string;
  details?: string;
  severity?: 'error' | 'warning' | 'info';
  onRetry?: () => void;
  onDismiss?: () => void;
  error?: unknown; // Raw error object that will be converted to user-friendly format
}

export default function ErrorAlert({
  title,
  message,
  details,
  severity,
  onRetry,
  onDismiss,
  error,
}: ErrorAlertProps) {
  const [showDetails, setShowDetails] = useState(false);

  // If raw error provided, convert to user-friendly format
  const friendlyError = error ? toUserFriendlyError(error) : null;
  
  const finalTitle = title || friendlyError?.title || 'Error';
  const finalMessage = message || friendlyError?.message || 'An unexpected error occurred';
  const finalDetails = details || friendlyError?.details;
  const finalSeverity = severity || friendlyError?.severity || 'error';
  const showRetry = onRetry && (friendlyError?.retryable !== false);

  return (
    <Alert
      severity={finalSeverity}
      onClose={onDismiss}
      action={
        showRetry && (
          <Button color="inherit" size="small" onClick={onRetry}>
            Retry
          </Button>
        )
      }
    >
      <AlertTitle>{finalTitle}</AlertTitle>
      {finalMessage}
      
      {finalDetails && (
        <Box sx={{ mt: 1 }}>
          <Button
            size="small"
            onClick={() => setShowDetails(!showDetails)}
            sx={{ p: 0, minWidth: 'auto', textTransform: 'none' }}
          >
            {showDetails ? 'Hide' : 'Show'} Details
          </Button>
          
          <Collapse in={showDetails}>
            <Box
              sx={{
                mt: 1,
                p: 1,
                backgroundColor: 'background.paper',
                borderRadius: 1,
                fontFamily: 'monospace',
                fontSize: '0.75rem',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {finalDetails}
            </Box>
          </Collapse>
        </Box>
      )}
    </Alert>
  );
}
