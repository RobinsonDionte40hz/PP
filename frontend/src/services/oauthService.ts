/**
 * OAuth service for social login and account linking
 */
import api from './api';
import type {
  OAuthConfig,
  OAuthInitiateResponse,
  OAuthCallbackRequest,
  OAuthLoginResponse,
  OAuthProvider,
  LinkedAccountsInfo,
  CaptchaConfig,
  VerificationStatus,
  SendVerificationResponse,
  VerifyEmailResponse,
} from '../types/oauth';

// -------------------- OAuth Configuration --------------------

/**
 * Get OAuth configuration (which providers are enabled)
 */
export const getOAuthConfig = async (): Promise<OAuthConfig> => {
  const response = await api.get<OAuthConfig>('/auth/oauth-config');
  return response.data;
};

// -------------------- Google OAuth --------------------

/**
 * Initiate Google OAuth flow
 * @param redirectUri Optional custom redirect URI
 */
export const initiateGoogleOAuth = async (redirectUri?: string): Promise<OAuthInitiateResponse> => {
  const params = redirectUri ? `?redirect_uri=${encodeURIComponent(redirectUri)}` : '';
  const response = await api.get<OAuthInitiateResponse>(`/auth/google${params}`);
  return response.data;
};

/**
 * Complete Google OAuth callback
 */
export const googleCallback = async (data: OAuthCallbackRequest): Promise<OAuthLoginResponse> => {
  const response = await api.post<OAuthLoginResponse>('/auth/google/callback', data);
  return response.data;
};

// -------------------- GitHub OAuth --------------------

/**
 * Initiate GitHub OAuth flow
 * @param redirectUri Optional custom redirect URI
 */
export const initiateGitHubOAuth = async (redirectUri?: string): Promise<OAuthInitiateResponse> => {
  const params = redirectUri ? `?redirect_uri=${encodeURIComponent(redirectUri)}` : '';
  const response = await api.get<OAuthInitiateResponse>(`/auth/github${params}`);
  return response.data;
};

/**
 * Complete GitHub OAuth callback
 */
export const githubCallback = async (data: OAuthCallbackRequest): Promise<OAuthLoginResponse> => {
  const response = await api.post<OAuthLoginResponse>('/auth/github/callback', data);
  return response.data;
};

// -------------------- Account Linking --------------------

/**
 * Get linked OAuth accounts for current user
 */
export const getLinkedAccounts = async (): Promise<LinkedAccountsInfo> => {
  const response = await api.get<LinkedAccountsInfo>('/auth/linked-accounts');
  return response.data;
};

/**
 * Initiate OAuth account linking
 */
export const initiateLink = async (provider: OAuthProvider): Promise<OAuthInitiateResponse> => {
  const response = await api.post<OAuthInitiateResponse>(`/auth/link/${provider}`);
  return response.data;
};

/**
 * Complete OAuth account linking callback
 */
export const completeLinkCallback = async (
  provider: OAuthProvider,
  data: OAuthCallbackRequest
): Promise<{ message: string }> => {
  const response = await api.post<{ message: string }>(`/auth/link/${provider}/callback`, data);
  return response.data;
};

/**
 * Unlink OAuth account
 */
export const unlinkAccount = async (provider: OAuthProvider): Promise<{ message: string }> => {
  const response = await api.delete<{ message: string }>(`/auth/unlink/${provider}`);
  return response.data;
};

/**
 * Set password for OAuth-only user
 */
export const setPassword = async (password: string): Promise<{ message: string }> => {
  const response = await api.post<{ message: string }>('/auth/set-password', { password });
  return response.data;
};

// -------------------- CAPTCHA --------------------

/**
 * Get CAPTCHA configuration
 */
export const getCaptchaConfig = async (): Promise<CaptchaConfig> => {
  const response = await api.get<CaptchaConfig>('/auth/captcha-config');
  return response.data;
};

// -------------------- Email Verification --------------------

/**
 * Get email verification status
 */
export const getVerificationStatus = async (): Promise<VerificationStatus> => {
  const response = await api.get<VerificationStatus>('/auth/verification-status');
  return response.data;
};

/**
 * Send/resend verification email
 */
export const sendVerificationEmail = async (email?: string): Promise<SendVerificationResponse> => {
  const response = await api.post<SendVerificationResponse>('/auth/send-verification', { email });
  return response.data;
};

/**
 * Verify email with token
 */
export const verifyEmail = async (token: string): Promise<VerifyEmailResponse> => {
  const response = await api.post<VerifyEmailResponse>('/auth/verify-email', { token });
  return response.data;
};

// -------------------- Helper Functions --------------------

/**
 * Store OAuth tokens after successful login
 */
export const storeOAuthTokens = (response: OAuthLoginResponse): void => {
  const { user, tokens } = response;
  
  localStorage.setItem('auth_token', tokens.access_token);
  localStorage.setItem('refresh_token', tokens.refresh_token);
  localStorage.setItem('user', JSON.stringify(user));
};

/**
 * Get OAuth provider display name
 */
export const getProviderDisplayName = (provider: OAuthProvider): string => {
  const names: Record<OAuthProvider, string> = {
    google: 'Google',
    github: 'GitHub',
  };
  return names[provider] || provider;
};

/**
 * Get OAuth provider icon name (for MUI icons or similar)
 */
export const getProviderIcon = (provider: OAuthProvider): string => {
  const icons: Record<OAuthProvider, string> = {
    google: 'Google',
    github: 'GitHub',
  };
  return icons[provider] || 'Link';
};

export const oauthService = {
  // Config
  getOAuthConfig,
  getCaptchaConfig,
  
  // Google OAuth
  initiateGoogleOAuth,
  googleCallback,
  
  // GitHub OAuth
  initiateGitHubOAuth,
  githubCallback,
  
  // Account Linking
  getLinkedAccounts,
  initiateLink,
  completeLinkCallback,
  unlinkAccount,
  setPassword,
  
  // Email Verification
  getVerificationStatus,
  sendVerificationEmail,
  verifyEmail,
  
  // Helpers
  storeOAuthTokens,
  getProviderDisplayName,
  getProviderIcon,
};

export default oauthService;
