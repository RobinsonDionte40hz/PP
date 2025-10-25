from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try package-relative imports first
    from .interfaces import MoveType
except ImportError:
    # Fall back to absolute imports from ubf_protein package
    from ubf_protein.interfaces import MoveType

@dataclass
class ConformationalMemory:
    """Memory of a significant conformational state"""
    memory_id: str
    move_type: str  # String representation of MoveType enum
    significance: float  # 0.0-1.0
    energy_change: float
    rmsd_change: float
    success: bool
    timestamp: int
    consciousness_state: ConsciousnessCoordinates
    behavioral_state: BehavioralStateData
    _cached_weight: Optional[float] = None  # Cached influence weight
    _weight_calc_time: Optional[int] = None  # Time when weight was calculated

    def get_influence_weight(self) -> float:
        """Calculate influence weight based on significance and recency (cached for 1 hour)"""
        import time
        current_time = int(time.time() * 1000)
        
        # Cache for 1 hour (3600000 ms) to avoid recalculation
        if (self._cached_weight is not None and 
            self._weight_calc_time is not None and
            (current_time - self._weight_calc_time) < 3600000):
            return self._cached_weight
        
        time_diff_hours = (current_time - self.timestamp) / (1000 * 60 * 60)

        # Exponential decay: more recent = higher weight
        recency_weight = max(0.1, 1.0 / (1.0 + time_diff_hours / 24.0))  # Half-life of 24 hours

        # Success bonus
        success_bonus = 1.2 if self.success else 0.8

        weight = self.significance * recency_weight * success_bonus
        
        # Cache the result
        self._cached_weight = weight
        self._weight_calc_time = current_time
        
        return weight

# ============================================================================
# Consciousness & Behavioral State Data
# ============================================================================

@dataclass
class ConsciousnessCoordinates:
    """Two fundamental coordinates defining agent state"""
    frequency: float  # 3-15 Hz (exploration energy)
    coherence: float  # 0.2-1.0 (structural focus)
    last_update_timestamp: int

    def __post_init__(self):
        assert 3.0 <= self.frequency <= 15.0, "Frequency must be 3-15 Hz"
        assert 0.2 <= self.coherence <= 1.0, "Coherence must be 0.2-1.0"

@dataclass
class BehavioralStateData:
    """Cached behavioral state derived from consciousness coordinates"""
    exploration_energy: float  # Low/Moderate/High
    structural_focus: float  # Scattered/Balanced/Focused
    conformational_bias: float  # Compact vs extended preference
    hydrophobic_drive: float  # 0.0-1.0
    risk_tolerance: float  # 0.0-1.0 (willingness to try radical moves)
    native_state_ambition: float  # 0.0-1.0 (drive toward goal)
    cached_timestamp: int

    @staticmethod
    def from_consciousness(freq: float, coh: float) -> 'BehavioralStateData':
        """Generate behavioral state from consciousness coordinates"""
        return BehavioralStateData(
            exploration_energy=map_frequency_to_energy(freq),
            structural_focus=map_coherence_to_focus(coh),
            conformational_bias=calculate_bias(freq, coh),
            hydrophobic_drive=max(0.0, min(1.0, (freq - 4.0) / 8.0)),  # Clamped 0-1
            risk_tolerance=max(0.0, min(1.0, (freq - 6.0) / 6.0)),     # Clamped 0-1
            native_state_ambition=coh * (freq / 10.0),  # Coherence scaled by normalized frequency
            cached_timestamp=current_time_ms()
        )

# ============================================================================
# Conformational State Data
# ============================================================================

@dataclass
class Conformation:
    """Represents a protein conformational state (mappless - no spatial map)"""
    conformation_id: str
    sequence: str  # Amino acid sequence
    atom_coordinates: List[Tuple[float, float, float]]  # 3D positions
    energy: float  # Current energy (kJ/mol)
    rmsd_to_native: Optional[float]  # If native structure known

    # Structural properties (for capability matching)
    secondary_structure: List[str]  # 'H' (helix), 'E' (sheet), 'C' (coil) per residue
    phi_angles: List[float]  # Backbone dihedral angles
    psi_angles: List[float]

    # Capability metadata (enables mappless matching)
    available_move_types: List[str]  # What moves are feasible from this state
    structural_constraints: Dict[str, Any]  # Constraints limiting moves
    
    # Energy components (for debugging and analysis)
    energy_components: Optional[Dict[str, float]] = None  # Bond, angle, dihedral, VDW, etc.

    def get_capabilities(self) -> Dict[str, bool]:
        """Return capability flags for mappless move matching"""
        return {
            'can_form_helix': self._can_form_helix(),
            'can_form_sheet': self._can_form_sheet(),
            'can_hydrophobic_collapse': self._can_collapse(),
            'can_large_rotation': self._can_large_rotation(),
            'has_flexible_loops': self._has_flexible_loops()
        }

    def _can_form_helix(self) -> bool:
        """Check if helix formation is possible"""
        # Placeholder implementation
        return len([ss for ss in self.secondary_structure if ss == 'C']) >= 4

    def _can_form_sheet(self) -> bool:
        """Check if sheet formation is possible"""
        # Placeholder implementation
        return len([ss for ss in self.secondary_structure if ss == 'C']) >= 4

    def _can_collapse(self) -> bool:
        """Check if hydrophobic collapse is possible"""
        # Placeholder implementation
        return True  # Assume always possible for now

    def _can_large_rotation(self) -> bool:
        """Check if large rotation is possible"""
        # Placeholder implementation
        return len(self.sequence) <= 100  # Smaller proteins can rotate more easily

    def _has_flexible_loops(self) -> bool:
        """Check if flexible loops exist"""
        # Placeholder implementation
        return 'C' in self.secondary_structure

# ============================================================================
# Conformational Moves (Mappless Design)
# ============================================================================

@dataclass
class ConformationalMove:
    """A potential conformational change (mappless - no path, just transition)"""
    move_id: str
    move_type: MoveType
    target_residues: List[int]
    estimated_energy_change: float
    estimated_rmsd_change: float

    # Capability requirements (for mappless matching)
    required_capabilities: Dict[str, bool]
    energy_barrier: float  # Activation energy needed
    structural_feasibility: float  # 0.0-1.0 based on current state

    # Physics-based factors (calculated by physics modules)
    qaap_factor: Optional[float] = None
    resonance_factor: Optional[float] = None
    water_shielding_factor: Optional[float] = None

# ============================================================================
# Outcome & Update Rules
# ============================================================================

@dataclass
class ConformationalOutcome:
    """Result of executing a conformational move"""
    move_executed: ConformationalMove
    new_conformation: Conformation
    energy_change: float
    rmsd_change: float
    success: bool  # Did energy decrease?
    significance: float  # 0.0-1.0 for memory formation

    def get_consciousness_update(self) -> Tuple[float, float]:
        """Get frequency and coherence deltas based on outcome"""
        if self.energy_change < -100:  # Large energy decrease
            return (+0.5, +0.1)
        elif self.energy_change < -50:  # Moderate decrease
            return (+0.3, +0.05)
        elif self.energy_change < -10:  # Small decrease
            return (+0.2, +0.05)
        elif self.energy_change > 50:  # Energy increase (bad move)
            return (-0.3, -0.05)
        elif self.energy_change > 100:  # Large increase (very bad)
            return (-0.5, -0.1)
        else:  # Minimal change
            return (0.0, 0.0)

class OutcomeType(Enum):
    """Outcome categories for consciousness updates"""
    ENERGY_DECREASE_LARGE = "energy_decrease_large"
    ENERGY_DECREASE_SMALL = "energy_decrease_small"
    ENERGY_INCREASE = "energy_increase"
    STRUCTURE_COLLAPSE = "structure_collapse"
    STABLE_MINIMUM_FOUND = "stable_minimum_found"
    HELIX_FORMATION = "helix_formation"
    SHEET_FORMATION = "sheet_formation"
    HYDROPHOBIC_CORE_FORMED = "hydrophobic_core_formed"
    STUCK_IN_LOCAL_MINIMUM = "stuck_in_local_minimum"

# ============================================================================
# Agent Diversity Profiles
# ============================================================================

@dataclass
class AgentProfile:
    """Defines initial consciousness coordinates for agent diversity"""
    profile_name: str
    frequency_range: Tuple[float, float]
    coherence_range: Tuple[float, float]
    description: str

# ============================================================================
# Metrics & Results
# ============================================================================

@dataclass
@dataclass
class ExplorationMetrics:
    """Metrics for tracking agent performance"""
    agent_id: str
    iterations_completed: int
    conformations_explored: int
    memories_created: int
    best_energy_found: float
    best_rmsd_found: float
    learning_improvement: float  # % RMSD improvement over time
    avg_decision_time_ms: float
    stuck_in_minima_count: int
    successful_escapes: int

@dataclass
class ExplorationResults:
    """Results from multi-agent exploration"""
    total_iterations: int
    total_conformations_explored: int
    best_conformation: Optional[Conformation]
    best_energy: float
    best_rmsd: float
    agent_metrics: List[ExplorationMetrics]
    collective_learning_benefit: float  # Multi-agent improvement over single
    total_runtime_seconds: float
    shared_memories_created: int

# ============================================================================
# Visualization & Monitoring
# ============================================================================

@dataclass
class EnergyLandscape:
    """2D projection of explored conformational space"""
    projection_method: str  # 'PCA' or 't-SNE'
    coordinates_2d: List[Tuple[float, float]]  # 2D coordinates for each conformation
    energy_values: List[float]
    rmsd_values: List[float]

# ============================================================================
# Helper Functions (placeholders for now)
# ============================================================================

def map_frequency_to_energy(freq: float) -> float:
    """Map frequency to exploration energy level (0.0-1.0)"""
    # Linear mapping from 3-15 Hz to 0-1
    return (freq - 3.0) / 12.0

def map_coherence_to_focus(coh: float) -> float:
    """Map coherence to structural focus level (0.0-1.0)"""
    # Direct mapping - higher coherence = higher focus
    return coh

def calculate_bias(freq: float, coh: float) -> float:
    """Calculate conformational bias (-1.0 to 1.0, negative=compact, positive=extended)"""
    # Higher frequency + lower coherence = more extended conformations
    # Lower frequency + higher coherence = more compact conformations
    freq_factor = (freq - 9.0) / 6.0  # -1 at 3Hz, +1 at 15Hz
    coh_factor = (coh - 0.6) / 0.4    # -1 at 0.2, +1 at 1.0
    return (freq_factor - coh_factor) / 2.0  # Combined bias

def current_time_ms() -> int:
    """Get current time in milliseconds"""
    import time
    return int(time.time() * 1000)

# ============================================================================
# Visualization & Monitoring
# ============================================================================

@dataclass
class ConformationSnapshot:
    """Snapshot of conformation at a point in time"""
    iteration: int
    timestamp: float
    conformation: Conformation
    agent_id: str
    consciousness_state: ConsciousnessCoordinates
    behavioral_state: BehavioralStateData

# ============================================================================
# Checkpoint & Resume
# ============================================================================

@dataclass
class SystemCheckpoint:
    """Complete system state for checkpoint/resume"""
    timestamp: float
    iteration: int
    protein_sequence: str
    agent_count: int
    configuration: Dict[str, Any]
    agent_states: List[Dict[str, Any]]  # Serialized agent states
    shared_memory_pool: List[ConformationalMemory]
    best_conformation: Optional[Conformation]
    metadata: Dict[str, Any]

# ============================================================================
# Adaptive Configuration
# ============================================================================

class ProteinSizeClass(Enum):
    """Protein size classification"""
    SMALL = "small"    # < 50 residues
    MEDIUM = "medium"  # 50-150 residues
    LARGE = "large"    # > 150 residues

@dataclass
class AdaptiveConfig:
    """Adaptive configuration based on protein size"""
    size_class: ProteinSizeClass
    residue_count: int

    # Consciousness parameters
    initial_frequency_range: Tuple[float, float]
    initial_coherence_range: Tuple[float, float]

    # Local minima detection
    stuck_detection_window: int
    stuck_detection_threshold: float  # kJ/mol, scaled by protein size

    # Memory parameters
    memory_significance_threshold: float
    max_memories_per_agent: int

    # Convergence criteria
    convergence_energy_threshold: float  # kJ/mol
    convergence_rmsd_threshold: float  # Angstroms

    # Performance parameters
    max_iterations: int
    checkpoint_interval: int