/**
 * Types for Aggregation Screening API
 */

export type ScreeningMode = 'fast' | 'balanced' | 'thorough';

export type AggregationRiskLevel = 'low' | 'moderate' | 'high' | 'critical';

/**
 * Result from screening a single sequence
 */
export interface ScreeningResult {
  sequence: string;
  sequence_length: number;
  
  // Scores (0-1, higher = better/lower risk)
  aggregation_score: number;
  energy_score: number;
  structure_score: number;
  hydrophobic_score: number;
  compactness_score: number;
  
  // Classification
  risk_level: AggregationRiskLevel;
  risk_factors: string[];
  passes_screening: boolean;
  
  // Raw values
  final_energy: number;
  secondary_structure_pct: number;
  radius_of_gyration: number;
  
  // Metadata
  screening_time_ms: number;
}

/**
 * Request for batch screening
 */
export interface BatchScreeningRequest {
  sequences: string[];
  mode?: ScreeningMode;
  name?: string;
}

/**
 * Response from batch screening
 */
export interface BatchScreeningResponse {
  batch_id: string;
  name?: string;
  mode: ScreeningMode;
  status: 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  
  // Summary
  total_sequences: number;
  sequences_passed: number;
  sequences_failed: number;
  
  // By risk level
  risk_summary: {
    low: number;
    moderate: number;
    high: number;
    critical: number;
  };
  
  // Results (sorted by score, best first)
  results: ScreeningResult[];
  
  // Export paths
  csv_path?: string;
  json_path?: string;
}

/**
 * Request to create a screening campaign
 */
export interface ScreeningCampaignRequest {
  name: string;
  sequences: string[];
  mode?: ScreeningMode;
  min_aggregation_score?: number;  // 0-1, default 0.5
  auto_create_predictions?: boolean;
}

/**
 * Response for screening campaign
 */
export interface ScreeningCampaignResponse {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  mode: ScreeningMode;
  created_at: string;
  completed_at?: string;
  
  // Progress
  total_sequences: number;
  screened_sequences: number;
  progress_percentage: number;
  
  // Results summary
  passed_count: number;
  failed_count: number;
  risk_distribution: {
    low: number;
    moderate: number;
    high: number;
    critical: number;
  };
  
  // Linked predictions
  prediction_ids: string[];
  
  // Results availability
  results_available: boolean;
}

/**
 * Risk level display configuration
 */
export const RISK_LEVEL_CONFIG: Record<AggregationRiskLevel, {
  label: string;
  color: string;
  bgColor: string;
  description: string;
}> = {
  low: {
    label: 'Low Risk',
    color: '#2e7d32',
    bgColor: '#e8f5e9',
    description: 'Likely to fold stably',
  },
  moderate: {
    label: 'Moderate Risk',
    color: '#f57c00',
    bgColor: '#fff3e0',
    description: 'May need optimization',
  },
  high: {
    label: 'High Risk',
    color: '#d32f2f',
    bgColor: '#ffebee',
    description: 'Likely to aggregate',
  },
  critical: {
    label: 'Critical Risk',
    color: '#b71c1c',
    bgColor: '#ffcdd2',
    description: 'Almost certainly will aggregate',
  },
};

/**
 * Screening mode display configuration
 */
export const SCREENING_MODE_CONFIG: Record<ScreeningMode, {
  label: string;
  description: string;
  iterations: number;
  agents: number;
}> = {
  fast: {
    label: 'Fast',
    description: 'Quick scan (50 iterations)',
    iterations: 50,
    agents: 2,
  },
  balanced: {
    label: 'Balanced',
    description: 'Default (100 iterations)',
    iterations: 100,
    agents: 3,
  },
  thorough: {
    label: 'Thorough',
    description: 'Detailed (200 iterations)',
    iterations: 200,
    agents: 5,
  },
};
