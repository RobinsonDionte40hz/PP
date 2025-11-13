import { Alert, AlertTitle, Box, Button, Collapse } from '@mui/material';
import { useState } from 'react';

interface ErrorAlertProps {
  title?: string;
  message: string;
  details?: string;
  severity?: 'error' | 'warning' | 'info';
  onRetry?: () => void;
  onDismiss?: () => void;
}

export default function ErrorAlert({
  title = 'Error',
  message,
  details,
  severity = 'error',
  onRetry,
  onDismiss,
}: ErrorAlertProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <Alert
      severity={severity}
      onClose={onDismiss}
      action={
        onRetry && (
          <Button color="inherit" size="small" onClick={onRetry}>
            Retry
          </Button>
        )
      }
    >
      <AlertTitle>{title}</AlertTitle>
      {message}
      
      {details && (
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
              {details}
            </Box>
          </Collapse>
        </Box>
      )}
    </Alert>
  );
}
