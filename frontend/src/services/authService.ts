/**
 * Authentication service for API calls
 */
import api from './api';
import { withRetry, authRetryOptions } from '../utils/retry';
import type {
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  RegisterResponse,
  RefreshTokenResponse,
  AuthError,
  User,
} from '../types/auth';

/**
 * Login user with credentials (with retry logic for network errors)
 */
export const login = async (credentials: LoginRequest): Promise<LoginResponse> => {
  try {
    const response = await withRetry(
      () => api.post<LoginResponse>('/auth/login', credentials),
      authRetryOptions
    );
    
    // Store tokens in localStorage
    if (response.data.access_token) {
      localStorage.setItem('auth_token', response.data.access_token);
      localStorage.setItem('refresh_token', response.data.refresh_token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error: unknown) {
    const apiError = error as { response?: { data?: { detail?: string }; status?: number } };
    const authError: AuthError = {
      detail: apiError.response?.data?.detail || 'Login failed. Please check your credentials.',
      status: apiError.response?.status,
    };
    throw authError;
  }
};

/**
 * Register new user (with retry logic for network errors)
 */
export const register = async (userData: RegisterRequest): Promise<RegisterResponse> => {
  try {
    const response = await withRetry(
      () => api.post<RegisterResponse>('/auth/register', userData),
      authRetryOptions
    );
    return response.data;
  } catch (error: unknown) {
    const apiError = error as { response?: { data?: { detail?: string }; status?: number } };
    const authError: AuthError = {
      detail: apiError.response?.data?.detail || 'Registration failed. Please try again.',
      status: apiError.response?.status,
    };
    throw authError;
  }
};

/**
 * Logout current user
 */
export const logout = async (): Promise<void> => {
  try {
    await api.post('/auth/logout');
  } catch (error) {
    console.error('Logout API call failed:', error);
    // Continue with local cleanup even if API call fails
  } finally {
    // Clear local storage
    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
  }
};

/**
 * Refresh access token using refresh token (with retry logic)
 */
export const refreshToken = async (): Promise<RefreshTokenResponse> => {
  try {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await withRetry(
      () => api.post<RefreshTokenResponse>('/auth/refresh', {
        refresh_token: refreshToken,
      }),
      {
        ...authRetryOptions,
        maxRetries: 2, // Fewer retries for token refresh
      }
    );

    // Update access token in localStorage
    if (response.data.access_token) {
      localStorage.setItem('auth_token', response.data.access_token);
    }

    return response.data;
  } catch (error: unknown) {
    // If refresh fails, clear all auth data
    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    
    const apiError = error as { response?: { data?: { detail?: string }; status?: number } };
    const authError: AuthError = {
      detail: apiError.response?.data?.detail || 'Session expired. Please login again.',
      status: apiError.response?.status,
    };
    throw authError;
  }
};

/**
 * Get current user from localStorage
 */
export const getCurrentUser = (): User | null => {
  const userStr = localStorage.getItem('user');
  if (!userStr) {
    return null;
  }
  
  try {
    return JSON.parse(userStr) as User;
  } catch (error) {
    console.error('Failed to parse user from localStorage:', error);
    return null;
  }
};

/**
 * Check if user is authenticated
 */
export const isAuthenticated = (): boolean => {
  const token = localStorage.getItem('auth_token');
  const user = getCurrentUser();
  return !!token && !!user;
};

/**
 * Get access token from localStorage
 */
export const getAccessToken = (): string | null => {
  return localStorage.getItem('auth_token');
};

/**
 * Get refresh token from localStorage
 */
export const getRefreshToken = (): string | null => {
  return localStorage.getItem('refresh_token');
};
