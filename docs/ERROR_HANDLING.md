# Error Handling and User Feedback System

This document describes the comprehensive error handling and notification system implemented for the authentication flow and general application use.

## Overview

The error handling system provides:
- **Error Boundaries** for catching and handling React errors gracefully
- **Toast Notifications** for success/error/warning/info messages
- **Retry Logic** for transient network failures
- **User-Friendly Error Messages** that map technical errors to readable messages
- **Loading States** with spinners for all async operations

## Components

### 1. ErrorBoundary Component

**Location**: `frontend/src/components/common/ErrorBoundary.tsx`

A React Error Boundary that catches JavaScript errors anywhere in the component tree and displays a fallback UI.

**Features**:
- Catches and logs all unhandled errors in child components
- Displays user-friendly error message with details
- Provides "Try Again" and "Go Home" actions
- Supports custom fallback UI via props
- Optional `onError` callback for error reporting services

**Usage**:
```tsx
import ErrorBoundary from './components/common/ErrorBoundary';

// Wrap your app or specific components
<ErrorBoundary onError={(error, errorInfo) => reportToService(error)}>
  <YourApp />
</ErrorBoundary>
```

**Already integrated in**:
- `App.tsx` - Wraps entire application
- `routes/index.tsx` - Wraps individual route sections

### 2. NotificationContext & useNotification Hook

**Location**: 
- Context: `frontend/src/contexts/NotificationContext.tsx`
- Hook: `frontend/src/hooks/useNotification.ts`

A notification system using Material-UI Snackbar for toast notifications.

**Features**:
- Queue-based notification system (handles multiple notifications)
- Auto-dismissal with configurable duration
- Four severity levels: success, error, warning, info
- Custom titles and messages
- Positioned at top-right by default
- Manual dismissal with close button

**API**:
```tsx
const { showSuccess, showError, showWarning, showInfo, showNotification } = useNotification();

// Simple messages
showSuccess('Operation completed!');
showError('Something went wrong!');
showWarning('Please be careful.');
showInfo('Did you know...');

// With custom titles
showSuccess('Changes saved successfully', 'Success');
showError('Failed to connect to server', 'Connection Error');

// Custom options
showNotification({
  title: 'Custom Title',
  message: 'Custom message',
  severity: 'info',
  duration: 5000,
  action: <Button>Custom Action</Button>
});
```

**Usage in Components**:
```tsx
import { useNotification } from '../hooks/useNotification';

function MyComponent() {
  const { showSuccess, showError } = useNotification();
  
  const handleSubmit = async () => {
    try {
      await submitData();
      showSuccess('Data submitted successfully!', 'Success');
    } catch (error) {
      showError('Failed to submit data. Please try again.', 'Error');
    }
  };
  
  return <button onClick={handleSubmit}>Submit</button>;
}
```

**Already integrated in**:
- `App.tsx` - NotificationProvider wraps the router
- `pages/Login.tsx` - Success/error notifications on login
- `pages/Register.tsx` - Success/error notifications on registration

### 3. Retry Logic

**Location**: `frontend/src/utils/retry.ts`

Implements exponential backoff retry logic for network requests.

**Features**:
- Configurable max retries, delays, and backoff multiplier
- Smart retry detection (network errors, 5xx errors)
- Exponential backoff with max delay cap
- Per-attempt callbacks for logging/monitoring
- Pre-configured options for auth and API requests

**API**:
```tsx
import { withRetry, authRetryOptions, apiRetryOptions } from '../utils/retry';

// Wrap an async function with retry logic
const result = await withRetry(
  () => api.post('/endpoint', data),
  authRetryOptions
);

// Or create a retryable function
import { withRetryable } from '../utils/retry';

const retryableLogin = withRetryable(
  (credentials) => api.post('/auth/login', credentials),
  authRetryOptions
);

const response = await retryableLogin(credentials);
```

**Default Configurations**:

**`authRetryOptions`**:
- Max retries: 2
- Initial delay: 1500ms
- Retries on: Network errors, 5xx server errors
- No retry on: 401, 403 (auth errors), 400 (validation errors)

**`apiRetryOptions`**:
- Max retries: 3
- Initial delay: 1000ms
- Max delay: 8000ms
- Retries on: Network errors, 5xx server errors

**Already integrated in**:
- `services/authService.ts` - All auth API calls (login, register, refresh)

### 4. User-Friendly Error Messages

**Location**: `frontend/src/utils/authErrors.ts`

Maps technical error codes and messages to user-friendly explanations with actionable guidance.

**Features**:
- HTTP status code mapping (400, 401, 403, 404, 409, 422, 429, 500, 502, 503)
- Operation-specific messages (login, register, logout, refresh)
- Contextual error messages based on error details
- Retryable flag to indicate if user should retry
- User action suggestions for each error type

**API**:
```tsx
import { getAuthErrorMessage, authOperationMessages } from '../utils/authErrors';

try {
  await login(credentials);
} catch (error) {
  const errorMessage = getAuthErrorMessage(error, 'login');
  
  // errorMessage contains:
  // - title: "Login Failed"
  // - message: "Invalid username or password."
  // - userAction?: "Double-check your credentials"
  // - retryable: false
  
  showError(errorMessage.message, errorMessage.title);
  
  if (errorMessage.userAction) {
    console.log('User should:', errorMessage.userAction);
  }
}
```

**Error Categories**:

**Network Errors**:
- Connection failures → "Unable to connect to the server"
- Timeouts → "The request took too long"
- Retryable: Yes

**Authentication Errors**:
- Invalid credentials (401) → "Invalid username or password"
- Account locked → "Account locked due to too many attempts"
- Session exists → "Already logged in from another device"
- Retryable: No

**Validation Errors**:
- Duplicate username (409) → "Username already registered"
- Weak password → "Password doesn't meet requirements"
- Invalid email → "Please provide a valid email"
- Retryable: No

**Server Errors**:
- 500, 502, 503 → "Server encountered an error"
- Retryable: Yes

**Rate Limiting**:
- 429 → "Too many requests. Wait and try again"
- Retryable: Yes (after waiting)

**Already integrated in**:
- `pages/Login.tsx` - Displays friendly error messages
- `pages/Register.tsx` - Displays friendly error messages

### 5. Loading States

**Location**: Throughout auth components

**Implementation**:
- Material-UI `CircularProgress` spinners
- Disabled form inputs during loading
- Loading text (e.g., "Signing In...", "Creating Account...")
- Button states reflect loading status

**Examples**:

```tsx
// In Login.tsx
<Button
  type="submit"
  disabled={isLoading || isSubmitting || !isFormValid}
>
  {isLoading || isSubmitting ? (
    <>
      <CircularProgress size={24} sx={{ mr: 1 }} color="inherit" />
      Signing In...
    </>
  ) : (
    'Sign In'
  )}
</Button>

// In Register.tsx
<Button
  type="submit"
  disabled={isLoading || isSubmitting || !isFormValid}
>
  {isLoading || isSubmitting ? (
    <>
      <CircularProgress size={24} sx={{ mr: 1 }} color="inherit" />
      Creating Account...
    </>
  ) : (
    'Create Account'
  )}
</Button>
```

**Already integrated in**:
- `pages/Login.tsx` - Loading spinner on submit button
- `pages/Register.tsx` - Loading spinner on submit button
- `contexts/AuthContext.tsx` - Loading state management

## Usage Guide

### Basic Error Handling Pattern

```tsx
import { useNotification } from '../hooks/useNotification';
import { getAuthErrorMessage } from '../utils/authErrors';

function MyAuthComponent() {
  const { showSuccess, showError } = useNotification();
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const handleSubmit = async (data) => {
    setIsSubmitting(true);
    
    try {
      await authOperation(data);
      showSuccess('Operation successful!', 'Success');
    } catch (error) {
      const errorMessage = getAuthErrorMessage(error, 'login');
      showError(errorMessage.message, errorMessage.title);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <button disabled={isSubmitting}>
        {isSubmitting ? (
          <>
            <CircularProgress size={20} />
            Processing...
          </>
        ) : (
          'Submit'
        )}
      </button>
    </form>
  );
}
```

### Adding Retry Logic to New Endpoints

```tsx
import { withRetry, apiRetryOptions } from '../utils/retry';
import api from './api';

export const fetchData = async () => {
  return withRetry(
    () => api.get('/data'),
    apiRetryOptions
  );
};

// Or with custom options
export const criticalOperation = async () => {
  return withRetry(
    () => api.post('/critical'),
    {
      maxRetries: 5,
      initialDelay: 2000,
      onRetry: (error, attempt, delay) => {
        console.log(`Retry attempt ${attempt} in ${delay}ms`);
        showWarning(`Retrying... (attempt ${attempt})`);
      }
    }
  );
};
```

### Creating Custom Error Messages

```tsx
// In utils/authErrors.ts or your own error utility

export const customOperationMessages = {
  dataFetch: {
    networkError: {
      title: 'Connection Error',
      message: 'Unable to fetch data. Check your connection.',
      retryable: true,
    },
    notFound: {
      title: 'Not Found',
      message: 'The requested data could not be found.',
      retryable: false,
    },
  },
};

// Usage
const errorMessage = error.response?.status === 404
  ? customOperationMessages.dataFetch.notFound
  : customOperationMessages.dataFetch.networkError;

showError(errorMessage.message, errorMessage.title);
```

### Wrapping Components with ErrorBoundary

```tsx
import ErrorBoundary from './components/common/ErrorBoundary';

// For a specific section that might throw errors
function RiskySection() {
  return (
    <ErrorBoundary
      fallback={<div>This section failed to load.</div>}
      onError={(error, errorInfo) => {
        logErrorToService(error, errorInfo);
      }}
    >
      <RiskyComponent />
    </ErrorBoundary>
  );
}
```

## Testing

### Testing Error Boundaries

```tsx
import { render, screen } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';

const ThrowError = () => {
  throw new Error('Test error');
};

test('catches errors and displays fallback', () => {
  render(
    <ErrorBoundary>
      <ThrowError />
    </ErrorBoundary>
  );
  
  expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
});
```

### Testing Notifications

```tsx
import { renderHook, act } from '@testing-library/react';
import { NotificationProvider } from './NotificationContext';
import { useNotification } from './useNotification';

test('shows success notification', () => {
  const wrapper = ({ children }) => (
    <NotificationProvider>{children}</NotificationProvider>
  );
  
  const { result } = renderHook(() => useNotification(), { wrapper });
  
  act(() => {
    result.current.showSuccess('Success!', 'Great');
  });
  
  expect(screen.getByText('Success!')).toBeInTheDocument();
});
```

### Testing Retry Logic

```tsx
import { withRetry } from './retry';

test('retries on network error', async () => {
  let attempts = 0;
  const fn = jest.fn(async () => {
    attempts++;
    if (attempts < 3) {
      throw new Error('Network error');
    }
    return 'success';
  });
  
  const result = await withRetry(fn, { maxRetries: 3 });
  
  expect(fn).toHaveBeenCalledTimes(3);
  expect(result).toBe('success');
});
```

## Best Practices

1. **Always provide context**: Use specific error titles and messages
2. **Guide the user**: Include actionable suggestions when possible
3. **Don't over-notify**: Use appropriate severity levels
4. **Log errors**: Use ErrorBoundary's `onError` for reporting
5. **Test error states**: Include error scenarios in tests
6. **Graceful degradation**: Always provide fallback UI
7. **Clear loading states**: Show spinners and disable inputs during operations
8. **Retry intelligently**: Use retry logic for transient failures only
9. **User-friendly messages**: Avoid technical jargon in error messages
10. **Consistent patterns**: Follow the established error handling patterns

## Future Enhancements

- [ ] Add error reporting service integration (e.g., Sentry)
- [ ] Implement offline detection and queue
- [ ] Add error analytics dashboard
- [ ] Create error message localization
- [ ] Add sound notifications for critical errors
- [ ] Implement error recovery strategies
- [ ] Add detailed error logging with stack traces in development
- [ ] Create error message testing utilities

## Summary

The error handling and user feedback system provides a comprehensive solution for managing errors and keeping users informed throughout the application. By combining error boundaries, notifications, retry logic, and user-friendly messages, we ensure a robust and pleasant user experience even when things go wrong.

All authentication flows now include:
✅ Error boundaries to catch unexpected errors
✅ Toast notifications for immediate feedback
✅ Retry logic for network failures
✅ User-friendly error messages with actionable guidance
✅ Loading spinners during async operations
✅ Graceful degradation when services are unavailable
