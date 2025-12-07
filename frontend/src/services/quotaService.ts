/**
 * Quota service for managing user prediction quotas
 */
import api from './api';
import type { QuotaInfo, QuotaCheckResponse } from '../types/quota';

/**
 * Get current user's quota information
 */
export const getQuota = async (): Promise<QuotaInfo> => {
  const response = await api.get<QuotaInfo>('/users/me/quota');
  return response.data;
};

/**
 * Quick check if user can create a prediction
 */
export const checkQuota = async (): Promise<QuotaCheckResponse> => {
  const response = await api.get<QuotaCheckResponse>('/users/me/quota/check');
  return response.data;
};

/**
 * Get quota percentage used for a period
 */
export const getQuotaPercentage = (used: number, limit: number): number => {
  if (limit <= 0) return 0; // Unlimited
  return Math.min(Math.round((used / limit) * 100), 100);
};

/**
 * Format remaining quota for display
 */
export const formatQuotaRemaining = (remaining: number, limit: number): string => {
  if (limit <= 0) return 'Unlimited';
  return `${remaining} / ${limit}`;
};

/**
 * Get quota status color based on percentage used
 */
export const getQuotaColor = (percentage: number): 'success' | 'warning' | 'error' => {
  if (percentage >= 90) return 'error';
  if (percentage >= 70) return 'warning';
  return 'success';
};

/**
 * Format reset time for display
 */
export const formatResetTime = (resetAt: string | null): string => {
  if (!resetAt) return 'Unknown';
  
  const resetDate = new Date(resetAt);
  const now = new Date();
  const diffMs = resetDate.getTime() - now.getTime();
  
  if (diffMs <= 0) return 'Soon';
  
  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
  
  if (hours > 24) {
    const days = Math.floor(hours / 24);
    return `in ${days} day${days > 1 ? 's' : ''}`;
  }
  
  if (hours > 0) {
    return `in ${hours}h ${minutes}m`;
  }
  
  return `in ${minutes}m`;
};

export const quotaService = {
  getQuota,
  checkQuota,
  getQuotaPercentage,
  formatQuotaRemaining,
  getQuotaColor,
  formatResetTime,
};

export default quotaService;
