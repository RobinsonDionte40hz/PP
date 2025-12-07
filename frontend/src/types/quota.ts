/**
 * Quota types and interfaces for the frontend
 */

/**
 * Quota period information (daily or monthly)
 */
export interface QuotaPeriod {
  used: number;
  limit: number;
  remaining: number;
  reset_at: string | null;
}

/**
 * Complete quota information for a user
 */
export interface QuotaInfo {
  account_tier: 'free' | 'pro' | 'enterprise';
  daily: QuotaPeriod;
  monthly: QuotaPeriod;
}

/**
 * Quick quota check response
 */
export interface QuotaCheckResponse {
  can_create: boolean;
  message: string;
  quota: QuotaInfo;
}

/**
 * Tier display information
 */
export interface TierInfo {
  name: string;
  displayName: string;
  color: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  dailyLimit: number;
  monthlyLimit: number;
}

/**
 * Tier configuration map
 */
export const TIER_CONFIG: Record<string, TierInfo> = {
  free: {
    name: 'free',
    displayName: 'Free',
    color: 'default',
    dailyLimit: 20,
    monthlyLimit: 100,
  },
  pro: {
    name: 'pro',
    displayName: 'Pro',
    color: 'primary',
    dailyLimit: 100,
    monthlyLimit: 500,
  },
  enterprise: {
    name: 'enterprise',
    displayName: 'Enterprise',
    color: 'success',
    dailyLimit: -1, // Unlimited
    monthlyLimit: -1, // Unlimited
  },
};
