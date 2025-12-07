/**
 * Authentication types and interfaces for the frontend
 */

/**
 * User profile returned from API
 */
export interface User {
  key_id: string;
  username: string;
  email?: string;
  email_verified?: boolean;
  role?: string;
  account_tier?: 'free' | 'pro' | 'enterprise';
  created_at: string;
  last_login?: string;
  // OAuth linked accounts
  google_id?: string;
  github_id?: string;
}

/**
 * Login request payload
 */
export interface LoginRequest {
  username: string;
  password: string;
}

/**
 * Login response from API
 */
export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

/**
 * Registration request payload
 */
export interface RegisterRequest {
  username: string;
  password: string;
  email?: string;
  captcha_token?: string;
}

/**
 * Registration response from API
 */
export interface RegisterResponse {
  user: User;
  message: string;
}

/**
 * Token refresh request
 */
export interface RefreshTokenRequest {
  refresh_token: string;
}

/**
 * Token refresh response
 */
export interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

/**
 * Auth error response from API
 */
export interface AuthError {
  detail: string;
  status?: number;
}

/**
 * Auth state for context/store
 */
export interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}
