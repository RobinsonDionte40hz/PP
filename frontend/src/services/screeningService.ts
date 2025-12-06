import api from './api';
import type {
  ScreeningResult,
  BatchScreeningRequest,
  BatchScreeningResponse,
  ScreeningCampaignRequest,
  ScreeningCampaignResponse,
} from '../types/screening';

/**
 * Screening Service - Aggregation risk assessment for protein sequences
 * 
 * This service enables fast screening of sequences for aggregation propensity.
 * Unlike full structure prediction, screening answers:
 * "Will this sequence fold stably, or is it likely to aggregate/clump?"
 * 
 * Use Cases:
 * - Screen 100s of sequences before running expensive predictions
 * - Filter aggregation-prone candidates in therapeutic protein development
 * - Pre-filter peptide libraries
 */
class ScreeningService {
  private baseUrl = '/screening';

  /**
   * Screen a single sequence for aggregation risk
   * Returns immediately with results
   */
  async screenSingle(
    sequence: string,
    mode: 'fast' | 'balanced' | 'thorough' = 'fast'
  ): Promise<ScreeningResult> {
    const response = await api.post<ScreeningResult>(
      `${this.baseUrl}/single?mode=${mode}`,
      { sequence }
    );
    return response.data;
  }

  /**
   * Screen multiple sequences in batch
   * Small batches (≤10) return immediately
   * Larger batches are processed in background
   */
  async screenBatch(request: BatchScreeningRequest): Promise<BatchScreeningResponse> {
    const response = await api.post<BatchScreeningResponse>(
      `${this.baseUrl}/batch`,
      request
    );
    return response.data;
  }

  /**
   * Get status/results of a batch screening job
   */
  async getBatchStatus(batchId: string): Promise<BatchScreeningResponse> {
    const response = await api.get<BatchScreeningResponse>(
      `${this.baseUrl}/batch/${batchId}`
    );
    return response.data;
  }

  /**
   * Export batch results as CSV (returns blob)
   */
  async exportBatchCsv(batchId: string): Promise<Blob> {
    const response = await api.get(`${this.baseUrl}/batch/${batchId}/export/csv`, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Create a screening campaign
   * Screens sequences and optionally auto-creates predictions for passing ones
   */
  async createCampaign(request: ScreeningCampaignRequest): Promise<ScreeningCampaignResponse> {
    const response = await api.post<ScreeningCampaignResponse>(
      `${this.baseUrl}/campaign`,
      request
    );
    return response.data;
  }

  /**
   * Get screening campaign status
   */
  async getCampaignStatus(campaignId: string): Promise<ScreeningCampaignResponse> {
    const response = await api.get<ScreeningCampaignResponse>(
      `${this.baseUrl}/campaign/${campaignId}`
    );
    return response.data;
  }

  /**
   * Get detailed campaign results
   */
  async getCampaignResults(
    campaignId: string,
    passedOnly: boolean = false
  ): Promise<{ campaign_id: string; total_results: number; results: ScreeningResult[] }> {
    const response = await api.get(
      `${this.baseUrl}/campaign/${campaignId}/results?passed_only=${passedOnly}`
    );
    return response.data;
  }

  /**
   * Download CSV results for a batch
   */
  downloadCsv(batchId: string): void {
    const url = `${api.defaults.baseURL}${this.baseUrl}/batch/${batchId}/export/csv`;
    const token = localStorage.getItem('auth_token');
    
    // Create a temporary link and click it
    const link = document.createElement('a');
    link.href = `${url}?token=${token}`;
    link.download = `screening_${batchId}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}

export const screeningService = new ScreeningService();
export default screeningService;
