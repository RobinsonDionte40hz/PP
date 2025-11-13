import api from './api';
import type {
  ResultDetail,
  TrajectoryPoint,
  StructureData,
  ApiError,
} from '../types/api';

export const resultService = {
  // Get detailed results
  async getResultDetail(id: string): Promise<ResultDetail> {
    try {
      const response = await api.get<ResultDetail>(`/results/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get structure data
  async getStructure(id: string): Promise<StructureData> {
    try {
      const response = await api.get<StructureData>(`/results/${id}/structure`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get trajectory data
  async getTrajectory(id: string): Promise<TrajectoryPoint[]> {
    try {
      const response = await api.get<TrajectoryPoint[]>(`/results/${id}/trajectory`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Get detailed metrics
  async getMetrics(id: string): Promise<unknown> {
    try {
      const response = await api.get(`/results/${id}/metrics`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Export results
  async exportResults(id: string, format: 'json' | 'pdf' = 'json'): Promise<Blob> {
    try {
      const response = await api.get(`/results/${id}/export`, {
        params: { format },
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  // Compare multiple results
  async compareResults(ids: string[]): Promise<unknown> {
    try {
      const response = await api.post('/results/compare', { ids });
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
