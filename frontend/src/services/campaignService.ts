import api from './api';
import type {
  CampaignCreate,
  CampaignResponse,
  CampaignStatistics,
  ApiError,
} from '../types/api';

export const campaignService = {
  // Create a new campaign
  async createCampaign(data: CampaignCreate): Promise<CampaignResponse> {
    try {
      const response = await api.post<CampaignResponse>('/campaigns', data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get campaign by ID
  async getCampaign(id: string): Promise<CampaignResponse> {
    try {
      const response = await api.get<CampaignResponse>(`/campaigns/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // List all campaigns
  async listCampaigns(params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<CampaignResponse[]> {
    try {
      const response = await api.get<CampaignResponse[]>('/campaigns', { params });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Resume campaign
  async resumeCampaign(id: string): Promise<void> {
    try {
      await api.post(`/campaigns/${id}/resume`);
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Delete campaign
  async deleteCampaign(id: string): Promise<void> {
    try {
      await api.delete(`/campaigns/${id}`);
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get campaign statistics
  async getCampaignStatistics(id: string): Promise<CampaignStatistics> {
    try {
      const response = await api.get<CampaignStatistics>(`/campaigns/${id}/statistics`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get phase details
  async getPhaseDetails(id: string, phaseNum: number): Promise<unknown> {
    try {
      const response = await api.get(`/campaigns/${id}/phase/${phaseNum}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Error handler
  handleError(error: unknown): ApiError {
    if (typeof error === 'object' && error !== null && 'response' in error) {
      const axiosError = error as { response?: { data?: { detail?: string }; status?: number } };
      return {
        detail: axiosError.response?.data?.detail || 'An unexpected error occurred',
        status: axiosError.response?.status,
      };
    }
    return {
      detail: 'An unexpected error occurred',
    };
  },
};
