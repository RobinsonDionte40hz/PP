/**
 * Utility functions for handling errors and converting them to user-friendly messages
 */

export interface UserFriendlyError {
  title: string;
  message: string;
  details?: string;
  severity: 'error' | 'warning' | 'info';
  retryable: boolean;
}

/**
 * Convert an error to a user-friendly format
 */
export function toUserFriendlyError(error: unknown): UserFriendlyError {
  // Handle null/undefined
  if (!error) {
    return {
      title: 'Unknown Error',
      message: 'An unexpected error occurred. Please try again.',
      severity: 'error',
      retryable: true,
    };
  }

  // Handle Error objects
  if (error instanceof Error) {
    const errorMessage = error.message.toLowerCase();
    
    // Network errors
    if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
      return {
        title: 'Connection Error',
        message: 'Unable to connect to the server. Please check your internet connection and try again.',
        details: error.message,
        severity: 'error',
        retryable: true,
      };
    }

    // Timeout errors
    if (errorMessage.includes('timeout')) {
      return {
        title: 'Request Timeout',
        message: 'The request took too long to complete. The server might be busy. Please try again.',
        details: error.message,
        severity: 'warning',
        retryable: true,
      };
    }

    // Authentication errors
    if (errorMessage.includes('auth') || errorMessage.includes('unauthorized') || errorMessage.includes('401')) {
      return {
        title: 'Authentication Required',
        message: 'Your session has expired. Please log in again.',
        details: error.message,
        severity: 'warning',
        retryable: false,
      };
    }

    // Permission errors
    if (errorMessage.includes('forbidden') || errorMessage.includes('403')) {
      return {
        title: 'Access Denied',
        message: 'You do not have permission to perform this action.',
        details: error.message,
        severity: 'error',
        retryable: false,
      };
    }

    // Not found errors
    if (errorMessage.includes('not found') || errorMessage.includes('404')) {
      return {
        title: 'Not Found',
        message: 'The requested resource could not be found. It may have been deleted or moved.',
        details: error.message,
        severity: 'warning',
        retryable: false,
      };
    }

    // Validation errors
    if (errorMessage.includes('validation') || errorMessage.includes('invalid')) {
      return {
        title: 'Invalid Input',
        message: 'Please check your input and try again.',
        details: error.message,
        severity: 'warning',
        retryable: false,
      };
    }

    // Server errors
    if (errorMessage.includes('500') || errorMessage.includes('server error')) {
      return {
        title: 'Server Error',
        message: 'The server encountered an error. Our team has been notified. Please try again later.',
        details: error.message,
        severity: 'error',
        retryable: true,
      };
    }

    // Generic error
    return {
      title: 'Error',
      message: 'An error occurred. Please try again.',
      details: error.message,
      severity: 'error',
      retryable: true,
    };
  }

  // Handle API error responses
  if (typeof error === 'object' && error !== null) {
    const err = error as { response?: { status: number; data?: { detail?: string; [key: string]: unknown } } };
    
    if (err.response) {
      const status = err.response.status;
      const data = err.response.data;
      const detail = data?.detail || '';
      
      if (status === 400) {
        return {
          title: 'Invalid Request',
          message: detail || 'The request contains invalid data. Please check your input.',
          details: JSON.stringify(data, null, 2),
          severity: 'warning',
          retryable: false,
        };
      }
      
      if (status === 404) {
        return {
          title: 'Not Found',
          message: detail || 'The requested resource could not be found.',
          details: JSON.stringify(data, null, 2),
          severity: 'warning',
          retryable: false,
        };
      }
      
      if (status === 500) {
        return {
          title: 'Server Error',
          message: 'The server encountered an error. Please try again later.',
          details: JSON.stringify(data, null, 2),
          severity: 'error',
          retryable: true,
        };
      }
      
      return {
        title: `Error ${status}`,
        message: detail || 'An error occurred while processing your request.',
        details: JSON.stringify(data, null, 2),
        severity: 'error',
        retryable: status >= 500,
      };
    }
  }

  // Handle string errors
  if (typeof error === 'string') {
    return {
      title: 'Error',
      message: error,
      severity: 'error',
      retryable: true,
    };
  }

  // Fallback
  return {
    title: 'Unknown Error',
    message: 'An unexpected error occurred. Please try again.',
    details: String(error),
    severity: 'error',
    retryable: true,
  };
}

/**
 * Get a user-friendly message for common operations
 */
export const errorMessages = {
  predictionSubmit: {
    title: 'Submission Failed',
    message: 'Unable to submit prediction. Please check your input and try again.',
  },
  predictionLoad: {
    title: 'Load Failed',
    message: 'Unable to load prediction details. The prediction may have been deleted.',
  },
  predictionControl: {
    title: 'Control Failed',
    message: 'Unable to control prediction. Please try again.',
  },
  resultsLoad: {
    title: 'Results Unavailable',
    message: 'Unable to load results. The prediction may still be running or may have failed.',
  },
  structureLoad: {
    title: 'Structure Load Failed',
    message: 'Unable to load protein structure. The file may be corrupted or unavailable.',
  },
  downloadFailed: {
    title: 'Download Failed',
    message: 'Unable to download file. Please try again.',
  },
  websocketConnection: {
    title: 'Real-time Updates Unavailable',
    message: 'Unable to establish real-time connection. Falling back to periodic updates.',
  },
} as const;
