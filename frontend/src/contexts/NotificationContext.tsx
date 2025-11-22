/**
 * Notification Context for managing toast notifications
 */
import React, { createContext, useState, useCallback, useRef } from 'react';
import type { ReactNode } from 'react';
import type { AlertColor } from '@mui/material';
import {
  Snackbar,
  Alert,
  AlertTitle,
  IconButton,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

export interface NotificationOptions {
  title?: string;
  message: string;
  severity?: AlertColor;
  duration?: number;
  action?: ReactNode;
}

interface NotificationContextType {
  showNotification: (options: NotificationOptions) => void;
  showSuccess: (message: string, title?: string) => void;
  showError: (message: string, title?: string) => void;
  showWarning: (message: string, title?: string) => void;
  showInfo: (message: string, title?: string) => void;
  hideNotification: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

interface NotificationProviderProps {
  children: ReactNode;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({ children }) => {
  const [open, setOpen] = useState(false);
  const [notification, setNotification] = useState<NotificationOptions>({
    message: '',
    severity: 'info',
    duration: 6000,
  });
  
  // Queue to handle multiple notifications
  const queueRef = useRef<NotificationOptions[]>([]);
  const isProcessingRef = useRef(false);

  const processQueue = useCallback(() => {
    if (queueRef.current.length > 0 && !isProcessingRef.current) {
      isProcessingRef.current = true;
      const nextNotification = queueRef.current.shift()!;
      setNotification(nextNotification);
      setOpen(true);
    }
  }, []);

  const showNotification = useCallback((options: NotificationOptions) => {
    queueRef.current.push({
      severity: 'info',
      duration: 6000,
      ...options,
    });
    processQueue();
  }, [processQueue]);

  const showSuccess = useCallback((message: string, title?: string) => {
    showNotification({
      message,
      title,
      severity: 'success',
      duration: 4000,
    });
  }, [showNotification]);

  const showError = useCallback((message: string, title?: string) => {
    showNotification({
      message,
      title,
      severity: 'error',
      duration: 8000,
    });
  }, [showNotification]);

  const showWarning = useCallback((message: string, title?: string) => {
    showNotification({
      message,
      title,
      severity: 'warning',
      duration: 6000,
    });
  }, [showNotification]);

  const showInfo = useCallback((message: string, title?: string) => {
    showNotification({
      message,
      title,
      severity: 'info',
      duration: 5000,
    });
  }, [showNotification]);

  const hideNotification = useCallback(() => {
    setOpen(false);
  }, []);

  const handleClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpen(false);
  };

  const handleExited = () => {
    isProcessingRef.current = false;
    processQueue();
  };

  const value: NotificationContextType = {
    showNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    hideNotification,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <Snackbar
        open={open}
        autoHideDuration={notification.duration}
        onClose={handleClose}
        TransitionProps={{ onExited: handleExited }}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert
          onClose={handleClose}
          severity={notification.severity}
          variant="filled"
          sx={{ width: '100%', minWidth: 300 }}
          action={
            notification.action || (
              <IconButton
                size="small"
                aria-label="close"
                color="inherit"
                onClick={handleClose}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            )
          }
        >
          {notification.title && (
            <AlertTitle>{notification.title}</AlertTitle>
          )}
          {notification.message}
        </Alert>
      </Snackbar>
    </NotificationContext.Provider>
  );
};

export default NotificationContext;
