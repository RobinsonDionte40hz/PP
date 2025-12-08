import api from './api';
import type {
  PredictionCreate,
  PredictionResponse,
  ApiError,
} from '../types/api';

export const predictionService = {
  // Create a new prediction
  async createPrediction(data: PredictionCreate): Promise<PredictionResponse> {
    try {
      const response = await api.post<PredictionResponse>('/predictions', data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get prediction by ID
  async getPrediction(id: string): Promise<PredictionResponse> {
    try {
      const response = await api.get<PredictionResponse>(`/predictions/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // List all predictions with optional filters
  async listPredictions(params?: {
    status?: string;
    limit?: number;
    page?: number;
    page_size?: number;
  }): Promise<PredictionResponse[]> {
    try {
      // Convert limit to page_size for backend compatibility
      const backendParams = {
        status: params?.status,
        page: params?.page || 1,
        page_size: params?.limit || params?.page_size || 20,
      };
      
      const response = await api.get<{
        predictions: PredictionResponse[];
        total: number;
        page: number;
        page_size: number;
      }>('/predictions', { params: backendParams });
      
      // Extract predictions array from paginated response
      return response.data.predictions || [];
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Pause prediction
  async pausePrediction(id: string): Promise<void> {
    try {
      await api.post(`/predictions/${id}/pause`);
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Resume prediction
  async resumePrediction(id: string): Promise<void> {
    try {
      await api.post(`/predictions/${id}/resume`);
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Stop prediction
  async stopPrediction(id: string): Promise<void> {
    try {
      await api.post(`/predictions/${id}/stop`);
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Delete prediction
  async deletePrediction(id: string): Promise<void> {
    try {
      await api.delete(`/predictions/${id}`);
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Download checkpoint
  async downloadCheckpoint(id: string): Promise<Blob> {
    try {
      const response = await api.get(`/predictions/${id}/checkpoint`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get queue status for a prediction
  async getQueueStatus(id: string): Promise<{
    prediction_id: string;
    status: string;
    queue_position: number;
    total_queued: number;
    estimated_wait_minutes: number;
    message: string;
  }> {
    try {
      const response = await api.get(`/predictions/${id}/queue-status`);
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
