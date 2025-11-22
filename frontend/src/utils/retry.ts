/**
 * Retry logic for network requests with exponential backoff
 */

export interface RetryOptions {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
  shouldRetry?: (error: unknown, attempt: number) => boolean;
  onRetry?: (error: unknown, attempt: number, delay: number) => void;
}

const DEFAULT_OPTIONS: Required<RetryOptions> = {
  maxRetries: 3,
  initialDelay: 1000, // 1 second
  maxDelay: 10000, // 10 seconds
  backoffMultiplier: 2,
  shouldRetry: (error: unknown) => {
    // Retry on network errors
    if (error instanceof Error) {
      const message = error.message.toLowerCase();
      return (
        message.includes('network') ||
        message.includes('timeout') ||
        message.includes('fetch') ||
        message.includes('connection')
      );
    }

    // Retry on 5xx server errors
    if (typeof error === 'object' && error !== null) {
      const err = error as { response?: { status?: number } };
      if (err.response?.status && err.response.status >= 500) {
        return true;
      }
    }

    return false;
  },
  onRetry: (error, attempt, delay) => {
    console.log(`Retry attempt ${attempt} after ${delay}ms:`, error);
  },
};

/**
 * Sleep utility for delays
 */
const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

/**
 * Calculate delay with exponential backoff
 */
const calculateDelay = (
  attempt: number,
  initialDelay: number,
  maxDelay: number,
  backoffMultiplier: number
): number => {
  const delay = initialDelay * Math.pow(backoffMultiplier, attempt - 1);
  return Math.min(delay, maxDelay);
};

/**
 * Retry a function with exponential backoff
 * @param fn - The async function to retry
 * @param options - Retry configuration options
 * @returns The result of the function
 * @throws The last error if all retries fail
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options?: RetryOptions
): Promise<T> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  let lastError: unknown;

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      // First attempt or retry
      return await fn();
    } catch (error) {
      lastError = error;

      // Check if we should retry
      const shouldRetry = opts.shouldRetry(error, attempt);
      const hasRetriesLeft = attempt < opts.maxRetries;

      if (!shouldRetry || !hasRetriesLeft) {
        throw error;
      }

      // Calculate delay for next retry
      const delay = calculateDelay(
        attempt + 1,
        opts.initialDelay,
        opts.maxDelay,
        opts.backoffMultiplier
      );

      // Call retry callback
      opts.onRetry(error, attempt + 1, delay);

      // Wait before retrying
      await sleep(delay);
    }
  }

  // Should never reach here, but TypeScript needs it
  throw lastError;
}

/**
 * Higher-order function to wrap an async function with retry logic
 * @param fn - The async function to wrap
 * @param options - Retry configuration options
 * @returns A wrapped function with retry logic
 */
export function withRetryable<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => Promise<TReturn>,
  options?: RetryOptions
): (...args: TArgs) => Promise<TReturn> {
  return async (...args: TArgs): Promise<TReturn> => {
    return withRetry(() => fn(...args), options);
  };
}

/**
 * Retry options for authentication requests
 */
export const authRetryOptions: RetryOptions = {
  maxRetries: 2,
  initialDelay: 1500,
  shouldRetry: (error: unknown, attempt: number) => {
    // Don't retry on authentication errors (401, 403)
    if (typeof error === 'object' && error !== null) {
      const err = error as { response?: { status?: number } };
      const status = err.response?.status;
      if (status === 401 || status === 403) {
        return false;
      }
    }

    // Don't retry on validation errors (400)
    if (typeof error === 'object' && error !== null) {
      const err = error as { response?: { status?: number } };
      if (err.response?.status === 400) {
        return false;
      }
    }

    // Retry network errors and 5xx errors
    return DEFAULT_OPTIONS.shouldRetry(error, attempt);
  },
};

/**
 * Retry options for general API requests
 */
export const apiRetryOptions: RetryOptions = {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 8000,
};
