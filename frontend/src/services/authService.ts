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
      () => api.post<any>('/auth/login', credentials),
      authRetryOptions
    );
    
    console.log('Login API response:', response.data);
    
    // Handle nested tokens structure (tokens.access_token) or flat structure (access_token)
    const accessToken = response.data.tokens?.access_token || response.data.access_token;
    const refreshToken = response.data.tokens?.refresh_token || response.data.refresh_token;
    const user = response.data.user;
    const expiresIn = response.data.tokens?.expires_in || response.data.expires_in;
    
    // Store tokens in localStorage
    if (accessToken && user) {
      localStorage.setItem('auth_token', accessToken);
      localStorage.setItem('refresh_token', refreshToken);
      localStorage.setItem('user', JSON.stringify(user));
      
      console.log('Auth data stored:', { 
        hasToken: !!accessToken, 
        hasRefresh: !!refreshToken, 
        user: user.username 
      });
    } else {
      console.error('Invalid login response structure:', response.data);
      throw new Error('Invalid response from server');
    }
    
    // Return normalized response
    return {
      user,
      access_token: accessToken,
      refresh_token: refreshToken,
      token_type: 'bearer',
      expires_in: expiresIn || 1800,
    };
  } catch (error: unknown) {
    console.error('Login error:', error);
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
