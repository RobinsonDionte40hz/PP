/**
 * OAuth and verification types for the frontend
 */

/**
 * OAuth provider names
 */
export type OAuthProvider = 'google' | 'github';

/**
 * OAuth configuration from backend
 */
export interface OAuthConfig {
  google: {
    enabled: boolean;
    client_id: string | null;
  };
  github: {
    enabled: boolean;
    client_id: string | null;
  };
}

/**
 * OAuth initiate response
 */
export interface OAuthInitiateResponse {
  authorization_url: string;
  state: string;
}

/**
 * OAuth callback request
 */
export interface OAuthCallbackRequest {
  code: string;
  state: string;
}

/**
 * OAuth login response (same structure as normal login)
 */
export interface OAuthLoginResponse {
  user: {
    key_id: string;
    username: string;
    email?: string;
    created_at: string;
    last_login?: string;
  };
  tokens: {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
  };
  is_new_user: boolean;
}

/**
 * Linked accounts information
 */
export interface LinkedAccountsInfo {
  google: boolean;
  github: boolean;
  has_password: boolean;
}

/**
 * CAPTCHA configuration
 */
export interface CaptchaConfig {
  enabled: boolean;
  provider: 'recaptcha' | 'hcaptcha';
  site_key: string | null;
}

/**
 * Email verification status
 */
export interface VerificationStatus {
  email: string | null;
  email_verified: boolean;
  verification_sent_at: string | null;
  can_resend: boolean;
  verification_required: boolean;
}

/**
 * Send verification response
 */
export interface SendVerificationResponse {
  message: string;
  email: string;
  expires_in_hours: number;
}

/**
 * Verify email response
 */
export interface VerifyEmailResponse {
  message: string;
  email: string;
  verified_at: string;
}
