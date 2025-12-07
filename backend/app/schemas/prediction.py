"""
Prediction schemas for request/response validation
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from app.models.prediction import PredictionStatus
from app.security import SecurityConfig, validate_sequence_security


class PredictionConfigurationSchema(BaseModel):
    """Configuration for prediction"""
    iterations: int = Field(default=1000, ge=SecurityConfig.MIN_ITERATIONS, le=SecurityConfig.MAX_ITERATIONS, description="Number of iterations")
    agents: int = Field(default=10, ge=SecurityConfig.MIN_AGENTS, le=SecurityConfig.MAX_AGENTS, description="Number of agents")
    diversity: str = Field(default="balanced", description="Agent diversity: cautious, balanced, aggressive")
    enable_checkpointing: bool = Field(default=True, description="Enable checkpointing")
    checkpoint_interval: int = Field(default=50, ge=SecurityConfig.MIN_CHECKPOINT_INTERVAL, le=SecurityConfig.MAX_CHECKPOINT_INTERVAL, description="Checkpoint every N iterations")
    native_pdb: Optional[str] = Field(default=None, description="PDB ID for native structure comparison")
    qcpp_config: Optional[str] = Field(default=None, description="QCPP configuration: default, high_performance, high_accuracy")
    enable_mediators: bool = Field(default=False, description="Enable mediator agents for pattern detection")
    mediator_count: int = Field(default=3, ge=1, le=10, description="Number of mediator agents")
    enable_refinement: bool = Field(default=False, description="Enable quantum refinement post-processing")
    enable_hierarchical_folding: bool = Field(default=False, description="Enable hierarchical folding with progressive search confinement")
    enable_screening: bool = Field(default=False, description="Enable aggregation screening analysis")
    screening_mode: Optional[str] = Field(default="balanced", description="Screening mode: fast, balanced, thorough")
    
    @field_validator("diversity")
    @classmethod
    def validate_diversity(cls, v: str) -> str:
        allowed = ["cautious", "balanced", "aggressive"]
        if v not in allowed:
            raise ValueError(f"diversity must be one of {allowed}")
        return v
    
    @field_validator("qcpp_config")
    @classmethod
    def validate_qcpp_config(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = ["default", "high_performance", "high_accuracy"]
        if v not in allowed:
            raise ValueError(f"qcpp_config must be one of {allowed}")
        return v
    
    @field_validator("screening_mode")
    @classmethod
    def validate_screening_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "balanced"
        allowed = ["fast", "balanced", "thorough"]
        if v not in allowed:
            raise ValueError(f"screening_mode must be one of {allowed}")
        return v


class PredictionCreateSchema(BaseModel):
    """Schema for creating a new prediction"""
    sequence: str = Field(..., min_length=3, max_length=1000, description="Protein sequence (amino acids)")
    configuration: Optional[PredictionConfigurationSchema] = Field(default=None, description="Prediction configuration")
    
    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        # Strip whitespace and convert to uppercase
        v = v.strip().upper()
        
        # Length validation (prevent server overload)
        if len(v) < SecurityConfig.MIN_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence too short (minimum {SecurityConfig.MIN_SEQUENCE_LENGTH} amino acids)")
        if len(v) > SecurityConfig.MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence too long (maximum {SecurityConfig.MAX_SEQUENCE_LENGTH} amino acids for performance)")
        
        # Check if valid amino acid sequence
        invalid = set(v) - SecurityConfig.VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(f"Invalid amino acids in sequence: {', '.join(sorted(invalid))}. Only standard 20 amino acids allowed.")
        
        # Additional security validation
        is_valid, error_msg = validate_sequence_security(v)
        if not is_valid:
            raise ValueError(error_msg)
        
        return v


class PredictionMetricsSchema(BaseModel):
    """Current metrics for a prediction"""
    current_iteration: int = 0
    total_iterations: int = 0
    progress_percentage: float = 0.0
    best_energy: Optional[float] = None
    best_rmsd: Optional[float] = None
    current_energy: Optional[float] = None
    current_rmsd: Optional[float] = None
    aggressiveness: Optional[float] = None
    consistency: Optional[float] = None
    memory_count: Optional[int] = None


class PredictionResponseSchema(BaseModel):
    """Schema for prediction response"""
    id: str
    sequence: str
    status: PredictionStatus
    configuration: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    task_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    result_path: Optional[str] = None
    current_iteration: int = 0
    total_iterations: int = 0
    progress_percentage: float = 0.0
    metrics: Dict[str, Any] = {}
    # Top-level convenience fields (extracted from metrics for frontend)
    best_energy: Optional[float] = None
    best_rmsd: Optional[float] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "pred_abc123",
                "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                "status": "running",
                "configuration": {
                    "iterations": 1000,
                    "agents": 10,
                    "diversity": "balanced"
                },
                "created_at": "2025-11-12T10:00:00",
                "updated_at": "2025-11-12T10:05:00",
                "current_iteration": 250,
                "total_iterations": 1000,
                "progress_percentage": 25.0
            }
        }


class PredictionListResponseSchema(BaseModel):
    """Schema for list of predictions"""
    predictions: list[PredictionResponseSchema]
    total: int
    page: int
    page_size: int


class PredictionUpdateSchema(BaseModel):
    """Schema for updating prediction (internal use)"""
    status: Optional[PredictionStatus] = None
    task_id: Optional[str] = None
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None  # Track actual iterations being run
    progress_percentage: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    result_path: Optional[str] = None
