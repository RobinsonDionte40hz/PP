/**
 * User-friendly error messages for authentication operations
 */

export interface AuthErrorMessage {
  title: string;
  message: string;
  userAction?: string;
  retryable: boolean;
}

/**
 * Map HTTP status codes to user-friendly messages
 */
const statusCodeMessages: Record<number, AuthErrorMessage> = {
  400: {
    title: 'Invalid Request',
    message: 'The information you provided is invalid. Please check your input and try again.',
    retryable: false,
  },
  401: {
    title: 'Authentication Failed',
    message: 'Invalid username or password. Please check your credentials and try again.',
    userAction: 'Double-check your username and password',
    retryable: false,
  },
  403: {
    title: 'Access Denied',
    message: 'You do not have permission to perform this action.',
    retryable: false,
  },
  404: {
    title: 'Not Found',
    message: 'The requested resource could not be found.',
    retryable: false,
  },
  409: {
    title: 'Already Exists',
    message: 'This username or email is already registered. Please use a different one.',
    userAction: 'Try a different username or email',
    retryable: false,
  },
  422: {
    title: 'Validation Error',
    message: 'The information you provided is invalid. Please check the requirements and try again.',
    retryable: false,
  },
  429: {
    title: 'Too Many Requests',
    message: 'You have made too many requests. Please wait a moment and try again.',
    userAction: 'Wait a few minutes before trying again',
    retryable: true,
  },
  500: {
    title: 'Server Error',
    message: 'The server encountered an error. Please try again in a few moments.',
    userAction: 'Try again in a few minutes',
    retryable: true,
  },
  502: {
    title: 'Service Unavailable',
    message: 'The service is temporarily unavailable. Please try again later.',
    userAction: 'Check your connection and try again',
    retryable: true,
  },
  503: {
    title: 'Service Unavailable',
    message: 'The service is temporarily down for maintenance. Please try again later.',
    userAction: 'Try again in a few minutes',
    retryable: true,
  },
};

/**
 * Error messages for specific authentication operations
 */
export const authOperationMessages = {
  login: {
    networkError: {
      title: 'Connection Error',
      message: 'Unable to connect to the server. Please check your internet connection and try again.',
      userAction: 'Check your internet connection',
      retryable: true,
    },
    timeout: {
      title: 'Request Timeout',
      message: 'The login request took too long. The server might be busy. Please try again.',
      userAction: 'Try again in a moment',
      retryable: true,
    },
    invalidCredentials: {
      title: 'Login Failed',
      message: 'Invalid username or password.',
      userAction: 'Double-check your credentials',
      retryable: false,
    },
    accountLocked: {
      title: 'Account Locked',
      message: 'Your account has been locked due to too many failed login attempts.',
      userAction: 'Contact support or wait 15 minutes',
      retryable: false,
    },
    sessionExists: {
      title: 'Session Active',
      message: 'You are already logged in from another device or browser. The previous session has been terminated.',
      userAction: 'Continue with this session',
      retryable: false,
    },
  },
  register: {
    networkError: {
      title: 'Connection Error',
      message: 'Unable to connect to the server. Please check your internet connection and try again.',
      userAction: 'Check your internet connection',
      retryable: true,
    },
    timeout: {
      title: 'Request Timeout',
      message: 'The registration request took too long. Please try again.',
      userAction: 'Try again in a moment',
      retryable: true,
    },
    usernameExists: {
      title: 'Username Taken',
      message: 'This username is already registered. Please choose a different username.',
      userAction: 'Try a different username',
      retryable: false,
    },
    emailExists: {
      title: 'Email Already Registered',
      message: 'This email address is already registered. Please use a different email or try logging in.',
      userAction: 'Use a different email or login',
      retryable: false,
    },
    weakPassword: {
      title: 'Password Too Weak',
      message: 'Your password does not meet the security requirements.',
      userAction: 'Use at least 8 characters with uppercase, lowercase, numbers, and special characters',
      retryable: false,
    },
    invalidEmail: {
      title: 'Invalid Email',
      message: 'Please provide a valid email address.',
      userAction: 'Check your email format',
      retryable: false,
    },
  },
  logout: {
    networkError: {
      title: 'Connection Error',
      message: 'Unable to connect to the server. You have been logged out locally.',
      userAction: 'You can continue - local logout was successful',
      retryable: false,
    },
    timeout: {
      title: 'Request Timeout',
      message: 'The logout request timed out. You have been logged out locally.',
      userAction: 'You can continue - local logout was successful',
      retryable: false,
    },
  },
  refresh: {
    networkError: {
      title: 'Connection Error',
      message: 'Unable to refresh your session. Please log in again.',
      userAction: 'Log in again to continue',
      retryable: false,
    },
    invalidToken: {
      title: 'Session Expired',
      message: 'Your session has expired. Please log in again.',
      userAction: 'Log in to continue',
      retryable: false,
    },
  },
};

/**
 * Parse authentication error and return user-friendly message
 */
export function getAuthErrorMessage(
  error: unknown,
  operation: keyof typeof authOperationMessages = 'login'
): AuthErrorMessage {
  // Handle network errors
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    if (message.includes('network') || message.includes('fetch')) {
      return authOperationMessages[operation].networkError;
    }
    if (message.includes('timeout')) {
      const operationMessages = authOperationMessages[operation];
      if ('timeout' in operationMessages) {
        return operationMessages.timeout;
      }
      return authOperationMessages[operation].networkError;
    }
  }

  // Handle API errors with status codes
  if (typeof error === 'object' && error !== null) {
    const err = error as {
      response?: {
        status?: number;
        data?: {
          detail?: string;
          message?: string;
          [key: string]: unknown;
        };
      };
    };

    if (err.response) {
      const status = err.response.status;
      const detail = err.response.data?.detail || err.response.data?.message;

      // Check for specific error messages
      if (detail) {
        const detailLower = detail.toLowerCase();

        // Login-specific errors
        if (operation === 'login') {
          if (detailLower.includes('invalid') && (detailLower.includes('username') || detailLower.includes('password'))) {
            return authOperationMessages.login.invalidCredentials;
          }
          if (detailLower.includes('locked') || detailLower.includes('blocked')) {
            return authOperationMessages.login.accountLocked;
          }
          if (detailLower.includes('session') && detailLower.includes('active')) {
            return authOperationMessages.login.sessionExists;
          }
        }

        // Registration-specific errors
        if (operation === 'register') {
          if (detailLower.includes('username') && (detailLower.includes('exists') || detailLower.includes('taken'))) {
            return authOperationMessages.register.usernameExists;
          }
          if (detailLower.includes('email') && (detailLower.includes('exists') || detailLower.includes('registered'))) {
            return authOperationMessages.register.emailExists;
          }
          if (detailLower.includes('password') && detailLower.includes('weak')) {
            return authOperationMessages.register.weakPassword;
          }
          if (detailLower.includes('email') && detailLower.includes('invalid')) {
            return authOperationMessages.register.invalidEmail;
          }
        }

        // Refresh-specific errors
        if (operation === 'refresh') {
          if (detailLower.includes('token') && (detailLower.includes('invalid') || detailLower.includes('expired'))) {
            return authOperationMessages.refresh.invalidToken;
          }
        }
      }

      // Use status code mapping
      if (status && statusCodeMessages[status]) {
        return statusCodeMessages[status];
      }

      // Generic error with status
      return {
        title: `Error ${status}`,
        message: detail || 'An error occurred. Please try again.',
        retryable: status ? status >= 500 : false,
      };
    }
  }

  // Generic fallback
  return {
    title: 'Error',
    message: 'An unexpected error occurred. Please try again.',
    retryable: true,
  };
}
