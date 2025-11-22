// API Response Types
export interface ApiError {
  detail: string;
  status?: number;
}

// Prediction Types
export interface PredictionConfiguration {
  iterations?: number;
  agents?: number;
  diversity?: 'cautious' | 'balanced' | 'aggressive';
  enable_checkpointing?: boolean;
  checkpoint_interval?: number;
  native_pdb?: string;
  qcpp_config?: 'default' | 'high_performance' | 'high_accuracy';
  enable_mediators?: boolean;
  mediator_count?: number;
  enable_refinement?: boolean;
}

export interface PredictionCreate {
  sequence: string;
  configuration?: PredictionConfiguration;
}

export interface PredictionResponse {
  id: string;
  protein_sequence: string;
  sequence: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  current_iteration: number;
  total_iterations: number;
  best_energy?: number;
  best_rmsd?: number;
  final_energy?: number;
  native_pdb_id?: string;
  error_message?: string;
  result_file?: string;
  result_path?: string;
  config?: {
    num_agents?: number;
    qcpp_enabled?: boolean;
    temperature?: number;
  };
  configuration?: {
    agents?: number;
    iterations?: number;
    diversity?: string;
    qcpp_config?: string;
    native_pdb?: string;
    enable_checkpointing?: boolean;
  };
  metrics?: {
    current_iteration?: number;
    max_iterations?: number;
    current_energy?: number;
    best_energy?: number;
    final_energy?: number;
    initial_energy?: number;
    current_rmsd?: number;
    best_rmsd?: number;
    final_rmsd?: number;
    energy_change?: number;
    convergence_rate?: number;
    final_aggressiveness?: number;
    final_consistency?: number;
    conformations_explored?: number;
    unique_structures?: number;
    gdt_ts_score?: number;
    tm_score?: number;
    validation_quality?: string;
    qaap_alignment?: number;
    resonance_40hz?: number;
    water_shielding?: number;
    qcp_score?: number;
    refinement_applied?: boolean;
  };
}

export interface PredictionListResponse {
  predictions: PredictionResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface PredictionProgress {
  prediction_id: string;
  iteration: number;
  total_iterations: number;
  progress_percentage: number;
  current_energy: number;
  current_rmsd?: number;
  conformations_explored?: number;
  best_energy?: number;
  best_rmsd?: number;
  aggressiveness?: number;
  consistency?: number;
  timestamp?: string;
}

// Campaign Types
export interface CampaignCreate {
  name: string;
  protein_sequences: string[];
  iterations_per_protein?: number;
  quality_thresholds?: {
    min_rmsd?: number;
    max_energy?: number;
  };
  enable_qcpp?: boolean;
}

export interface CampaignResponse {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  created_at: string;
  total_proteins: number;
  completed_proteins: number;
  failed_proteins: number;
  current_phase?: number;
  total_phases: number;
  statistics?: {
    total_proteins?: number;
    successful_predictions?: number;
    failed_predictions?: number;
    average_rmsd?: number;
    average_energy?: number;
    average_iterations?: number;
    quality_distribution?: { [key: string]: number };
  };
  phase_results?: Array<{
    phase: number;
    total_proteins?: number;
    successful_predictions?: number;
    failed_predictions?: number;
  }>;
  proteins?: Array<{
    id: string;
    protein_name?: string;
    sequence?: string;
    status: string;
    rmsd?: number;
    energy?: number;
    quality?: string;
    phase?: number;
    created_at?: string;
  }>;
}

export interface CampaignStatistics {
  campaign_id: string;
  avg_energy: number;
  avg_rmsd: number;
  success_rate: number;
  total_iterations: number;
  total_time_seconds: number;
}

// Type alias for cleaner imports
export type Campaign = CampaignResponse;

// Result Types
export interface ResultDetail {
  prediction_id: string;
  final_energy: number;
  final_rmsd?: number;
  structure_quality: {
    rmsd_quality?: 'excellent' | 'good' | 'acceptable' | 'poor';
    energy_quality: 'excellent' | 'good' | 'acceptable' | 'poor';
    gdt_ts?: number;
    tm_score?: number;
  };
  agent_statistics: {
    total_moves: number;
    accepted_moves: number;
    rejected_moves: number;
    memory_usage: number;
  };
  qcpp_metrics?: {
    avg_qcp: number;
    coherence: number;
    stability_score: number;
  };
  geometric_analysis?: {
    icosahedron_score: number;
    dodecahedron_score: number;
    octahedron_score: number;
  };
}

export interface TrajectoryPoint {
  iteration: number;
  energy: number;
  rmsd?: number;
  timestamp: string;
}

export interface StructureData {
  pdb_content: string;
  format: 'pdb';
}

// WebSocket Event Types
export interface WSProgressUpdate {
  type: 'progress';
  data: PredictionProgress;
}

export interface WSMetricsUpdate {
  type: 'metrics';
  data: {
    energy?: number;
    rmsd?: number;
    aggressiveness?: number;
    consistency?: number;
    [key: string]: any;
  };
}

export interface WSAgentUpdate {
  type: 'agent';
  data: {
    agent_id?: string;
    status?: string;
    [key: string]: any;
  };
}

export interface WSStatusUpdate {
  type: 'status';
  data: {
    status: PredictionResponse['status'];
    message?: string;
  };
}

export interface WSLogEvent {
  type: 'log';
  data: {
    level: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: string;
  };
}

export interface WSCompleteEvent {
  type: 'complete';
  data: {
    prediction_id: string;
    final_energy?: number;
    final_rmsd?: number;
    [key: string]: any;
  };
}

export interface WSErrorEvent {
  type: 'error';
  data: {
    message: string;
    details?: string;
  };
}

export type WSMessage = 
  | WSProgressUpdate 
  | WSMetricsUpdate
  | WSAgentUpdate
  | WSStatusUpdate 
  | WSLogEvent
  | WSCompleteEvent
  | WSErrorEvent;
