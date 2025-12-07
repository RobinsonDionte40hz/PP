"""
Data schemas for the Prediction Engine API.

These dataclasses define the data structures exchanged between
the prediction engine and external consumers.

All schemas are:
- Immutable (frozen=True) for thread safety
- JSON-serializable for API transport
- Documented for API consumers
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import json


class PredictionPhase(Enum):
    """Phases of the prediction process."""
    INITIALIZING = "initializing"
    EXPLORATION = "exploration"
    REFINEMENT = "refinement"
    VALIDATION = "validation"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QCPPPreset(Enum):
    """QCPP configuration presets."""
    NONE = "none"
    DEFAULT = "default"
    HIGH_PERFORMANCE = "high_performance"
    HIGH_ACCURACY = "high_accuracy"


class AggregationRisk(Enum):
    """Aggregation risk classification."""
    LOW = "low"           # Likely to fold stably
    MODERATE = "moderate" # Some concerns, may need optimization
    HIGH = "high"         # Likely to aggregate
    CRITICAL = "critical" # Almost certainly will aggregate


@dataclass(frozen=True)
class QCPPConfig:
    """
    Configuration for QCPP integration.
    
    Attributes:
        preset: Configuration preset (none, default, high_performance, high_accuracy)
        enable_thz: Enable THz frequency analysis
        enable_field_coherence: Enable field coherence calculations
        coherence_threshold: Minimum coherence score threshold
    """
    preset: QCPPPreset = QCPPPreset.DEFAULT
    enable_thz: bool = True
    enable_field_coherence: bool = True
    coherence_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'preset': self.preset.value,
            'enable_thz': self.enable_thz,
            'enable_field_coherence': self.enable_field_coherence,
            'coherence_threshold': self.coherence_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QCPPConfig':
        """Create from dictionary."""
        preset = data.get('preset', 'default')
        if isinstance(preset, str):
            preset = QCPPPreset(preset)
        return cls(
            preset=preset,
            enable_thz=data.get('enable_thz', True),
            enable_field_coherence=data.get('enable_field_coherence', True),
            coherence_threshold=data.get('coherence_threshold', 0.5),
        )


@dataclass
class PredictionConfig:
    """
    Configuration for protein structure prediction.
    
    This is the primary configuration object for running predictions.
    External code should create this and pass to PredictionRunner.
    
    Attributes:
        sequence: Amino acid sequence (required)
        native_pdb: PDB ID for RMSD validation (optional)
        native_pdb_path: Local PDB file path (optional)
        agents: Number of exploration agents
        iterations: Maximum iterations
        enable_refinement: Enable quantum refinement phase
        enable_mediators: Enable mediator agents
        qcpp_config: QCPP configuration preset or object
        output_dir: Directory for output files
        checkpoint_interval: Save checkpoint every N iterations
        random_seed: Random seed for reproducibility
    """
    
    # Required
    sequence: str
    
    # Native structure (for RMSD validation)
    native_pdb: Optional[str] = None
    native_pdb_path: Optional[str] = None
    
    # Exploration parameters
    agents: int = 10
    iterations: int = 500
    
    # Feature flags
    enable_refinement: bool = True
    enable_mediators: bool = True
    enable_geometric_attractors: bool = True
    
    # QCPP configuration
    qcpp_config: str = "default"  # 'none', 'default', 'high_performance', 'high_accuracy'
    
    # Output configuration
    output_dir: Optional[str] = None
    checkpoint_interval: int = 100
    
    # Reproducibility
    random_seed: Optional[int] = None
    
    # Advanced options (usually left at defaults)
    temperature: float = 1.0
    exploration_weight: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PredictionConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ValidationMetrics:
    """
    Validation metrics for prediction quality assessment.
    
    Attributes:
        rmsd: Root Mean Square Deviation from native (if available)
        tm_score: TM-score (template modeling score)
        gdt_ts: Global Distance Test - Total Score
        clash_score: Steric clash score
        ramachandran_favored: Percentage in favored Ramachandran regions
        energy_total: Total energy score
        qcp_score: Quantum Coherence score (if QCPP enabled)
    """
    rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    gdt_ts: Optional[float] = None
    clash_score: Optional[float] = None
    ramachandran_favored: Optional[float] = None
    energy_total: Optional[float] = None
    qcp_score: Optional[float] = None
    
    # Additional detailed metrics
    bond_violations: int = 0
    angle_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ProgressUpdate:
    """
    Progress update during prediction.
    
    Sent periodically to callbacks during prediction execution.
    
    Attributes:
        iteration: Current iteration number
        total_iterations: Total iterations planned
        phase: Current prediction phase
        percentage: Completion percentage (0-100)
        best_energy: Best energy found so far
        current_rmsd: Current RMSD (if native available)
        message: Human-readable status message
        metrics: Additional metrics dictionary
    """
    iteration: int
    total_iterations: int
    phase: str
    percentage: float
    best_energy: Optional[float] = None
    current_rmsd: Optional[float] = None
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PredictionResults:
    """
    Results from a prediction run.
    
    Contains the predicted structure, validation metrics, and metadata.
    
    Attributes:
        sequence: Input sequence
        pdb_string: Predicted structure in PDB format
        coordinates: CA coordinates as list of [x, y, z]
        metrics: Validation metrics
        trajectory: List of energy values during exploration
        runtime_seconds: Total prediction time
        config: Configuration used
        metadata: Additional metadata (versions, timestamps, etc.)
    """
    sequence: str
    pdb_string: str
    coordinates: List[List[float]]
    metrics: ValidationMetrics
    trajectory: List[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sequence': self.sequence,
            'pdb_string': self.pdb_string,
            'coordinates': self.coordinates,
            'metrics': self.metrics.to_dict() if self.metrics else {},
            'trajectory': self.trajectory,
            'runtime_seconds': self.runtime_seconds,
            'config': self.config,
            'metadata': self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ScreeningConfig:
    """
    Configuration for aggregation screening.
    
    Attributes:
        window_size: Sliding window size for analysis
        threshold: Aggregation propensity threshold
        include_hydrophobic: Include hydrophobic analysis
        include_charge: Include charge pattern analysis
    """
    window_size: int = 7
    threshold: float = 0.5
    include_hydrophobic: bool = True
    include_charge: bool = True
    include_secondary_structure: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregationRegion:
    """
    An identified aggregation-prone region.
    
    Attributes:
        start: Start residue index (0-based)
        end: End residue index (exclusive)
        sequence: Sequence of the region
        score: Aggregation propensity score
        type: Type of aggregation risk
    """
    start: int
    end: int
    sequence: str
    score: float
    type: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningResults:
    """
    Results from aggregation screening.
    
    Full results including all metrics needed by the backend API.
    All scores are 0-1, where higher = better (lower risk).
    
    Attributes:
        sequence: Input sequence
        sequence_length: Length of sequence
        aggregation_score: Overall aggregation score (0-1, higher = less likely to aggregate)
        energy_score: Energy stability score (0-1)
        structure_score: Secondary structure formation score (0-1)
        hydrophobic_score: Hydrophobic core formation score (0-1)
        compactness_score: Compactness/radius of gyration score (0-1)
        risk_level: Classification (LOW, MODERATE, HIGH, CRITICAL)
        risk_factors: List of identified risk factors
        passes_screening: Whether sequence passes basic screening
        final_energy: Raw energy value (kcal/mol)
        secondary_structure_pct: Percentage of secondary structure
        radius_of_gyration: Compactness metric (Angstroms)
        screening_time_ms: Time taken for screening
        regions: List of identified aggregation-prone regions
        per_residue_scores: Score for each residue
        recommendations: Suggested modifications
    """
    sequence: str
    sequence_length: int
    aggregation_score: float
    energy_score: float
    structure_score: float
    hydrophobic_score: float
    compactness_score: float
    risk_level: AggregationRisk
    risk_factors: List[str]
    passes_screening: bool
    final_energy: float
    secondary_structure_pct: float
    radius_of_gyration: float
    screening_time_ms: float = 0.0
    regions: List[AggregationRegion] = field(default_factory=list)
    per_residue_scores: List[float] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sequence': self.sequence,
            'sequence_length': self.sequence_length,
            'aggregation_score': self.aggregation_score,
            'energy_score': self.energy_score,
            'structure_score': self.structure_score,
            'hydrophobic_score': self.hydrophobic_score,
            'compactness_score': self.compactness_score,
            'risk_level': self.risk_level.value,
            'risk_factors': self.risk_factors,
            'passes_screening': self.passes_screening,
            'final_energy': self.final_energy,
            'secondary_structure_pct': self.secondary_structure_pct,
            'radius_of_gyration': self.radius_of_gyration,
            'screening_time_ms': self.screening_time_ms,
            'regions': [r.to_dict() for r in self.regions],
            'per_residue_scores': self.per_residue_scores,
            'recommendations': self.recommendations,
        }
