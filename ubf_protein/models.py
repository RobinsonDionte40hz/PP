from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING
from enum import Enum

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .qcpp_integration import QCPPMetrics

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


@dataclass
@dataclass
class QCPPValidatedMemory(ConformationalMemory):
    """
    Memory with QCPP validation metrics.
    
    Extends ConformationalMemory to include quantum physics-based
    validation from QCPP system. Allows memory significance to be
    influenced by structural stability and quantum coherence.
    """
    qcpp_metrics: Optional[Any] = None  # QCPPMetrics type, use Any to avoid circular import
    qcpp_significance: float = 0.0  # QCPP contribution to overall significance (0.0-1.0)
    conformation_hash: Optional[str] = None  # Hash for QCPP metrics reuse (coordinate-based)
    
    def __post_init__(self):
        """Validate QCPP-specific fields."""
        if self.qcpp_metrics is not None:
            # Validate qcpp_significance is in range
            if not (0.0 <= self.qcpp_significance <= 1.0):
                raise ValueError(f"qcpp_significance {self.qcpp_significance} must be in [0.0, 1.0]")
            
            # Update total significance to include QCPP contribution
            # This is done after initial significance calculation
            # Total significance remains clamped to [0.0, 1.0]
            pass  # Significance recalculation done in memory system
    
    def is_high_significance(self) -> bool:
        """
        Check if memory meets high-significance criteria.
        
        High significance is defined as:
        - QCPP stability > 1.5 (stable structure)
        - Energy change < -20 kcal/mol (favorable)
        
        Returns:
            True if memory meets high-significance criteria
        """
        if self.qcpp_metrics is None:
            return False
        
        has_high_stability = self.qcpp_metrics.stability_score > 1.5
        has_favorable_energy = self.energy_change < -20.0
        
        return has_high_stability and has_favorable_energy


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
    
    # Validation metrics (Task 5: RMSD validation integration)
    native_structure_ref: Optional[str] = None  # PDB ID or file path for native structure
    gdt_ts_score: Optional[float] = None  # GDT-TS score (0-100)
    tm_score: Optional[float] = None  # TM-score (0-1)

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
    _qcpp_metrics: Optional[Any] = None  # Optional QCPP metrics for this outcome

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
    # Task 5: Add GDT-TS and TM-score tracking
    best_gdt_ts_score: Optional[float] = None  # Best GDT-TS score (0-100)
    best_tm_score: Optional[float] = None  # Best TM-score (0-1)

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
    # Task 5: Add validation quality assessment
    validation_quality: Optional[str] = None  # 'excellent', 'good', 'acceptable', 'poor', or None if no native structure
    best_gdt_ts: Optional[float] = None  # Best GDT-TS score across all agents
    best_tm_score: Optional[float] = None  # Best TM-score across all agents
    # Task 9: QCPP-UBF integration - trajectory and correlation analysis
    qcpp_trajectory_data: Optional[Dict[str, Any]] = None  # Integrated trajectory with QCPP metrics
    qcpp_rmsd_correlations: Optional[Dict[str, float]] = None  # QCPP-RMSD correlation analysis
    qcpp_energy_correlations: Optional[Dict[str, float]] = None  # QCPP-energy correlation analysis
    consciousness_qcpp_correlations: Optional[Dict[str, float]] = None  # Consciousness-QCPP correlations

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

# ============================================================================
# Quantum Refinement Engine (Task 1)
# ============================================================================

@dataclass
class RefinementConfig:
    """
    Configuration for quantum refinement process.
    
    Controls the two-stage optimization pipeline:
    - Stage 1: Global fold optimization (coarse structure)
    - Stage 2: Quantum refinement (fine-grained structure)
    
    Attributes:
        stage1_temperature: Temperature for Stage 1 (global exploration)
        stage1_iterations: Iterations for Stage 1
        stage2_temperature: Temperature for Stage 2 (10x lower for refinement)
        stage2_iterations: Iterations for Stage 2 (10x more for precision)
        restraint_weight: Weight for distance restraints (force constant)
        qcp_weight: Weight for QCP contribution to scoring (0-1)
        qcp_threshold: QCP threshold for quantum core identification
        phi_tolerance: Tolerance for φ-harmonic matching (THz)
        resonance_threshold: Minimum resonance coupling for contacts (0-1)
        water_spacing_nm: Water molecule spacing for shielding effects
        coherence_time_fs: Coherence time in femtoseconds
        max_refinement_time_seconds: Maximum time for refinement
        checkpoint_interval: Iterations between checkpoints
    """
    # Stage 1 (Global) parameters
    stage1_temperature: float = 1.0
    stage1_iterations: int = 1000
    
    # Stage 2 (Refinement) parameters
    stage2_temperature: float = 0.1  # 10x lower
    stage2_iterations: int = 10000  # 10x more
    restraint_weight: float = 10.0
    qcp_weight: float = 0.3  # 30% quantum contribution
    
    # Quantum parameters
    qcp_threshold: float = 7.0  # High coherence threshold
    phi_tolerance: float = 0.1  # THz
    resonance_threshold: float = 0.7
    
    # Water shielding
    water_spacing_nm: float = 0.28
    coherence_time_fs: float = 408.0
    
    # Performance
    max_refinement_time_seconds: float = 300.0  # 5 minutes
    checkpoint_interval: int = 1000
    
    # Geometry validation
    validation_mode: str = "lenient"  # "strict" or "lenient" for input structures


@dataclass
class RefinementResult:
    """
    Complete refinement result with metrics.
    
    Contains all information about the refinement process, including
    initial/final structures, RMSD improvements, quality metrics,
    and detailed diagnostics.
    
    Attributes:
        initial_structure: Input coarse structure (7-14Å RMSD)
        refined_structure: Output refined structure (<5Å RMSD)
        native_structure: Reference native structure (optional)
        initial_rmsd: RMSD before refinement (Ångströms)
        final_rmsd: RMSD after refinement (Ångströms)
        rmsd_improvement: RMSD reduction (Ångströms)
        helix_rmsd: RMSD for helix residues only
        sheet_rmsd: RMSD for sheet residues only
        loop_rmsd: RMSD for loop residues only
        core_rmsd: RMSD for hydrophobic core residues only
        gdt_ts: GDT-TS score (0-100, higher is better)
        tm_score: TM-score (0-1, higher is better)
        energy: Final energy (kcal/mol)
        iterations_used: Total iterations consumed
        refinement_time_seconds: Wall-clock time for refinement
        quantum_cores_identified: Number of quantum cores found
        restraints_applied: Number of distance restraints applied
        contacts_enforced: Number of tertiary contacts enforced
        rmsd_trajectory: RMSD at each iteration
        energy_trajectory: Energy at each iteration
    """
    # Structures
    initial_structure: Conformation
    refined_structure: Conformation
    native_structure: Optional[Any]  # NativeStructure type, avoid circular import
    
    # RMSD metrics
    initial_rmsd: float
    final_rmsd: float
    rmsd_improvement: float  # Angstroms
    
    # Component RMSD breakdown
    helix_rmsd: float
    sheet_rmsd: float
    loop_rmsd: float
    core_rmsd: float
    
    # Quality metrics
    gdt_ts: float
    tm_score: float
    energy: float  # kcal/mol
    
    # Refinement statistics
    iterations_used: int
    refinement_time_seconds: float
    quantum_cores_identified: int
    restraints_applied: int
    contacts_enforced: int
    
    # Convergence tracking
    rmsd_trajectory: List[float]
    energy_trajectory: List[float]
    
    def get_summary(self) -> str:
        """
        Generate human-readable summary of refinement results.
        
        Returns:
            Multi-line summary string with key metrics
        """
        lines = [
            "=" * 60,
            "QUANTUM REFINEMENT RESULTS",
            "=" * 60,
            f"Initial RMSD:    {self.initial_rmsd:6.2f} Å",
            f"Final RMSD:      {self.final_rmsd:6.2f} Å",
            f"Improvement:     {self.rmsd_improvement:6.2f} Å ({self.rmsd_improvement/self.initial_rmsd*100:.1f}%)",
            "",
            "Component RMSD Breakdown:",
            f"  Helix:         {self.helix_rmsd:6.2f} Å",
            f"  Sheet:         {self.sheet_rmsd:6.2f} Å",
            f"  Loop:          {self.loop_rmsd:6.2f} Å",
            f"  Core:          {self.core_rmsd:6.2f} Å",
            "",
            "Quality Metrics:",
            f"  GDT-TS:        {self.gdt_ts:6.2f}",
            f"  TM-score:      {self.tm_score:6.4f}",
            f"  Energy:        {self.energy:6.2f} kcal/mol",
            "",
            "Refinement Statistics:",
            f"  Iterations:    {self.iterations_used:6d}",
            f"  Runtime:       {self.refinement_time_seconds:6.2f} s",
            f"  Quantum cores: {self.quantum_cores_identified:6d}",
            f"  Restraints:    {self.restraints_applied:6d}",
            f"  Contacts:      {self.contacts_enforced:6d}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Quantum Core Analysis (Task 2)
# ============================================================================

@dataclass
class QuantumCore:
    """
    Represents a quantum core region in a protein structure.
    
    A quantum core is a contiguous region of residues with high QCP
    (Quantum Consciousness Potential) values, indicating strong coherence
    and structural stability.
    
    Quantum cores exhibit characteristic THz vibrational modes and can
    couple with other cores through φ-harmonic resonances.
    
    Attributes:
        residue_indices: List of residue indices in the core (contiguous)
        average_qcp: Mean QCP value across all core residues
        coherence: Average coherence metric (0-1)
        center_of_mass: Geometric center (x, y, z) in Ångströms
    
    Example:
        >>> core = QuantumCore(
        ...     residue_indices=[10, 11, 12, 13, 14],
        ...     average_qcp=8.5,
        ...     coherence=0.85,
        ...     center_of_mass=(12.3, 4.5, -8.2)
        ... )
        >>> print(f"Core spans {len(core.residue_indices)} residues")
    """
    residue_indices: List[int]
    average_qcp: float
    coherence: float
    center_of_mass: Tuple[float, float, float]
    
    def __post_init__(self):
        """Validate quantum core data."""
        if len(self.residue_indices) < 3:
            raise ValueError(f"Quantum core must have >= 3 residues, got {len(self.residue_indices)}")
        
        if self.average_qcp < 0:
            raise ValueError(f"average_qcp must be >= 0, got {self.average_qcp}")
        
        if not (0.0 <= self.coherence <= 1.0):
            raise ValueError(f"coherence must be in [0, 1], got {self.coherence}")


@dataclass
class THzMode:
    """
    Represents a THz vibrational mode in a protein structure.
    
    THz modes are collective vibrations in the terahertz frequency range
    (10^12 Hz) that play critical roles in protein conformational dynamics.
    
    Modes can exhibit φ-harmonic resonances at frequencies near golden ratio
    multiples of 1.0 THz (1.618 THz, 2.618 THz, 4.236 THz, etc.).
    
    Attributes:
        frequency: Vibrational frequency in THz
        amplitude: Mode amplitude (relative units)
        participating_residues: List of residue indices involved in this mode
        is_phi_harmonic: True if frequency is near a φ-harmonic
    
    Example:
        >>> mode = THzMode(
        ...     frequency=1.62,  # Near φ × 1.0 THz
        ...     amplitude=0.8,
        ...     participating_residues=[10, 11, 12, 13, 14],
        ...     is_phi_harmonic=True
        ... )
        >>> if mode.is_phi_harmonic:
        ...     print(f"φ-harmonic mode at {mode.frequency:.3f} THz")
    """
    frequency: float  # THz
    amplitude: float  # Relative units
    participating_residues: List[int]
    is_phi_harmonic: bool
    
    def __post_init__(self):
        """Validate THz mode data."""
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be >= 0, got {self.amplitude}")
        
        if len(self.participating_residues) == 0:
            raise ValueError("participating_residues cannot be empty")


# ============================================================================
# Distance Restraint System (Task 3)
# ============================================================================

@dataclass
class DistanceRestraint:
    """
    Represents a φ-harmonic distance restraint for protein refinement.
    
    Distance restraints enforce geometric relationships between high-QCP
    residue pairs, maintaining golden ratio patterns during optimization.
    
    The restraint applies a harmonic potential:
        E = weight × (distance - target_distance)²
    
    when the current distance deviates from the target by more than tolerance.
    
    Attributes:
        residue_i: Index of first residue in restraint
        residue_j: Index of second residue in restraint
        target_distance: Optimal distance in Ångströms (φ-harmonic)
        weight: Force constant (kcal/mol/Å²) - typically 100.0
        tolerance: Allowed deviation in Ångströms - typically 0.5
        is_phi_harmonic: True if distance is a φ-harmonic (d/φ, d, or d×φ)
    
    Example:
        >>> restraint = DistanceRestraint(
        ...     residue_i=10,
        ...     residue_j=25,
        ...     target_distance=6.0,  # Optimal contact distance
        ...     weight=100.0,
        ...     tolerance=0.5,
        ...     is_phi_harmonic=True
        ... )
        >>> energy = restraint.weight * (7.0 - restraint.target_distance) ** 2
        >>> print(f"Restraint energy at 7.0Å: {energy:.2f} kcal/mol")
    """
    residue_i: int
    residue_j: int
    target_distance: float  # Angstroms
    weight: float  # Force constant (kcal/mol/Å²)
    tolerance: float  # Angstroms
    is_phi_harmonic: bool
    
    def __post_init__(self):
        """Validate distance restraint data."""
        if self.residue_i < 0:
            raise ValueError(f"residue_i must be >= 0, got {self.residue_i}")
        
        if self.residue_j < 0:
            raise ValueError(f"residue_j must be >= 0, got {self.residue_j}")
        
        if self.residue_i == self.residue_j:
            raise ValueError(f"residue_i and residue_j must be different, both are {self.residue_i}")
        
        if self.target_distance <= 0:
            raise ValueError(f"target_distance must be > 0, got {self.target_distance}")
        
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0, got {self.weight}")
        
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance}")
    
    def calculate_energy(self, current_distance: float) -> float:
        """
        Calculate harmonic restraint energy.
        
        Energy is zero within tolerance, quadratic outside:
            E = 0                                  if |d - d₀| < tolerance
            E = weight × (d - d₀)²                 otherwise
        
        Args:
            current_distance: Current inter-residue distance in Ångströms
        
        Returns:
            Restraint energy in kcal/mol
        
        Example:
            >>> restraint = DistanceRestraint(0, 1, 6.0, 100.0, 0.5, True)
            >>> restraint.calculate_energy(6.2)  # Within tolerance
            0.0
            >>> restraint.calculate_energy(7.0)  # Outside tolerance
            100.0
        """
        deviation = abs(current_distance - self.target_distance)
        
        if deviation <= self.tolerance:
            return 0.0
        
        # Quadratic penalty outside tolerance
        effective_deviation = deviation - self.tolerance
        return self.weight * effective_deviation ** 2


# ============================================================================
# Secondary Structure Models (Task 4)
# ============================================================================

@dataclass
class HelixRegion:
    """
    Represents an alpha-helix region in a protein structure.
    
    Alpha helices are characterized by:
    - Pitch: ~5.4 Ångströms (distance per turn)
    - Rise: ~1.5 Ångströms (distance per residue)
    - Residues per turn: ~3.6
    - Hydrogen bonds: i to i+4
    
    Quantum-corrected helices (QCP > 7) use modified parameters:
    - Pitch: 5.4Å × (1 + 0.1 × tanh(QCP - 7))
    - Rise: 1.5Å × (1 + 0.05 × tanh(QCP - 7))
    
    Attributes:
        start_residue: First residue index in helix
        end_residue: Last residue index in helix (inclusive)
        average_qcp: Mean QCP value across helix residues
        pitch: Helix pitch in Ångströms
        rise: Rise per residue in Ångströms
        residues_per_turn: Number of residues per helical turn
    
    Example:
        >>> helix = HelixRegion(
        ...     start_residue=10,
        ...     end_residue=25,
        ...     average_qcp=8.2,
        ...     pitch=5.6,
        ...     rise=1.55,
        ...     residues_per_turn=3.61
        ... )
        >>> print(f"Helix spans {helix.end_residue - helix.start_residue + 1} residues")
    """
    start_residue: int
    end_residue: int
    average_qcp: float
    pitch: float  # Ångströms
    rise: float  # Ångströms
    residues_per_turn: float
    
    def __post_init__(self):
        """Validate helix region data."""
        if self.start_residue < 0:
            raise ValueError(f"start_residue must be >= 0, got {self.start_residue}")
        
        if self.end_residue < self.start_residue:
            raise ValueError(f"end_residue {self.end_residue} must be >= start_residue {self.start_residue}")
        
        # Alpha helices need at least 4 residues for one turn
        if (self.end_residue - self.start_residue + 1) < 4:
            raise ValueError(f"Helix must have >= 4 residues, got {self.end_residue - self.start_residue + 1}")
        
        if self.pitch <= 0:
            raise ValueError(f"pitch must be > 0, got {self.pitch}")
        
        if self.rise <= 0:
            raise ValueError(f"rise must be > 0, got {self.rise}")
        
        if self.residues_per_turn <= 0:
            raise ValueError(f"residues_per_turn must be > 0, got {self.residues_per_turn}")
    
    def length(self) -> int:
        """Return number of residues in helix."""
        return self.end_residue - self.start_residue + 1


@dataclass
class SheetRegion:
    """
    Represents a beta-sheet region in a protein structure.
    
    Beta sheets are characterized by:
    - Extended conformation with φ ~ -120°, ψ ~ +120°
    - Hydrogen bonds between strands (parallel or antiparallel)
    - Typical strand length: 5-10 residues
    - Inter-strand distance: ~4.8 Ångströms
    
    Quantum-optimized sheets use 2.618 THz coupling frequency
    (φ² harmonic) for hydrogen bond optimization.
    
    Attributes:
        strand_residues: List of residue ranges for each strand
                         Each range is (start, end) tuple
        average_qcp: Mean QCP value across all sheet residues
        is_parallel: True for parallel, False for antiparallel
        coupling_frequency: THz frequency for H-bond coupling (default 2.618)
    
    Example:
        >>> sheet = SheetRegion(
        ...     strand_residues=[(5, 10), (20, 25), (35, 40)],
        ...     average_qcp=7.5,
        ...     is_parallel=False,
        ...     coupling_frequency=2.618
        ... )
        >>> print(f"Sheet has {len(sheet.strand_residues)} strands")
    """
    strand_residues: List[Tuple[int, int]]  # List of (start, end) tuples
    average_qcp: float
    is_parallel: bool
    coupling_frequency: float = 2.618  # THz (φ² harmonic)
    
    def __post_init__(self):
        """Validate sheet region data."""
        if len(self.strand_residues) < 2:
            raise ValueError(f"Sheet must have >= 2 strands, got {len(self.strand_residues)}")
        
        for i, (start, end) in enumerate(self.strand_residues):
            if start < 0:
                raise ValueError(f"Strand {i} start_residue must be >= 0, got {start}")
            
            if end < start:
                raise ValueError(f"Strand {i} end_residue {end} must be >= start_residue {start}")
            
            # Beta strands need at least 3 residues
            if (end - start + 1) < 3:
                raise ValueError(f"Strand {i} must have >= 3 residues, got {end - start + 1}")
        
        if self.coupling_frequency <= 0:
            raise ValueError(f"coupling_frequency must be > 0, got {self.coupling_frequency}")
    
    def total_residues(self) -> int:
        """Return total number of residues in all strands."""
        return sum(end - start + 1 for start, end in self.strand_residues)


@dataclass
class PackingConstraint:
    """
    Distance constraint for hydrophobic core packing with QCP-weighted forces.
    
    Hydrophobic residues in protein cores pack at optimal distances
    determined by water exclusion zones (0.28 nm water spacing).
    This creates preferred packing distances at 2.8Å intervals.
    
    Force constants are scaled by QCP coupling factor to prioritize
    high-coherence residue pairs:
        force_constant = base_k × (QCP_i + QCP_j) / 2
    
    Attributes:
        residue_i: First residue index (0-based)
        residue_j: Second residue index (0-based)
        target_distance: Optimal packing distance in Ångströms
                         Typically at 2.8Å intervals (water spacing)
        force_constant: Harmonic force constant in kcal/mol/Ř
                        Base value 10.0, scaled by QCP coupling
        qcp_coupling: QCP-based scaling factor (QCP_i + QCP_j) / 2
                      Higher QCP pairs get stronger constraints
    
    Example:
        >>> constraint = PackingConstraint(
        ...     residue_i=10,
        ...     residue_j=25,
        ...     target_distance=5.6,  # 2×2.8Å
        ...     force_constant=75.0,  # 10.0 × 7.5 QCP coupling
        ...     qcp_coupling=7.5
        ... )
        >>> energy = constraint.force_constant * (current_dist - constraint.target_distance)**2
    """
    residue_i: int
    residue_j: int
    target_distance: float  # Angstroms
    force_constant: float  # kcal/mol/Ř
    qcp_coupling: float  # QCP-based scaling factor
    
    def __post_init__(self):
        """Validate packing constraint data."""
        if self.residue_i < 0:
            raise ValueError(f"residue_i must be >= 0, got {self.residue_i}")
        
        if self.residue_j < 0:
            raise ValueError(f"residue_j must be >= 0, got {self.residue_j}")
        
        if self.residue_i == self.residue_j:
            raise ValueError(f"residue_i and residue_j must be different, both are {self.residue_i}")
        
        if self.target_distance <= 0:
            raise ValueError(f"target_distance must be > 0, got {self.target_distance}")
        
        if self.force_constant <= 0:
            raise ValueError(f"force_constant must be > 0, got {self.force_constant}")
        
        if self.qcp_coupling <= 0:
            raise ValueError(f"qcp_coupling must be > 0, got {self.qcp_coupling}")
    
    def calculate_energy(self, current_distance: float) -> float:
        """
        Calculate harmonic restraint energy.
        
        E = k × (r - r₀)²
        
        Args:
            current_distance: Current distance between residues (Å)
        
        Returns:
            Restraint energy in kcal/mol
        """
        deviation = current_distance - self.target_distance
        return self.force_constant * deviation * deviation


@dataclass
class LoopRegion:
    """
    Represents a flexible loop region in a protein structure.
    
    Loops are unstructured regions connecting secondary structure elements
    (helices and sheets). They exhibit high conformational flexibility and
    are often difficult to predict accurately. Loop refinement uses different
    strategies based on quantum coherence (QCP) values:
    
    - Low QCP (<4): Classical loop modeling (sampling/minimization)
    - Medium QCP (4-7): G(φ,t) temporal evolution with quantum decay
    - High QCP (>7): Quantum-corrected geometry constraints
    
    G(φ,t) temporal evolution formula:
        G(φ,t) = exp(-t/τ_c) × φ
        where τ_c = 408 fs (coherence time)
    
    Attributes:
        start_residue: First residue index of loop (0-based)
        end_residue: Last residue index of loop (0-based)
        average_qcp: Mean QCP value across loop residues
        current_conformation: Current 3D coordinates of loop residues
                              List of (x, y, z) tuples, one per residue
        target_conformation: Target 3D coordinates (optional)
                            Used for guided refinement toward known structure
    
    Example:
        >>> loop = LoopRegion(
        ...     start_residue=10,
        ...     end_residue=15,
        ...     average_qcp=5.2,
        ...     current_conformation=[(1.0, 2.0, 3.0), (1.5, 2.5, 3.5), ...],
        ...     target_conformation=None
        ... )
        >>> print(f"Loop has {loop.length()} residues")
        >>> print(f"Strategy: {'quantum' if loop.average_qcp > 4 else 'classical'}")
    """
    start_residue: int
    end_residue: int
    average_qcp: float
    current_conformation: List[Tuple[float, float, float]]
    target_conformation: Optional[List[Tuple[float, float, float]]] = None
    
    def __post_init__(self):
        """Validate loop region data."""
        if self.start_residue < 0:
            raise ValueError(f"start_residue must be >= 0, got {self.start_residue}")
        
        if self.end_residue < self.start_residue:
            raise ValueError(
                f"end_residue {self.end_residue} must be >= start_residue {self.start_residue}"
            )
        
        # Loops should have at least 2 residues
        if self.end_residue - self.start_residue < 1:
            raise ValueError(
                f"Loop must have >= 2 residues, got {self.end_residue - self.start_residue + 1}"
            )
        
        # Validate current_conformation length
        expected_length = self.end_residue - self.start_residue + 1
        if len(self.current_conformation) != expected_length:
            raise ValueError(
                f"current_conformation length {len(self.current_conformation)} "
                f"must match loop length {expected_length}"
            )
        
        # Validate target_conformation length if provided
        if self.target_conformation is not None:
            if len(self.target_conformation) != expected_length:
                raise ValueError(
                    f"target_conformation length {len(self.target_conformation)} "
                    f"must match loop length {expected_length}"
                )
        
        if self.average_qcp < 0:
            raise ValueError(f"average_qcp must be >= 0, got {self.average_qcp}")
    
    def length(self) -> int:
        """Return number of residues in loop."""
        return self.end_residue - self.start_residue + 1
    
    def is_classical_refinement(self) -> bool:
        """Check if loop should use classical refinement (QCP < 4)."""
        return self.average_qcp < 4.0
    
    def is_quantum_refinement(self) -> bool:
        """Check if loop should use quantum G(φ,t) evolution (4 <= QCP < 7)."""
        return 4.0 <= self.average_qcp < 7.0
    
    def is_high_qcp(self) -> bool:
        """Check if loop has high quantum coherence (QCP >= 7)."""
        return self.average_qcp >= 7.0


# ============================================================================
# Tertiary Contact Prediction (Task 7)
# ============================================================================

@dataclass
class TertiaryContact:
    """
    Represents a predicted tertiary contact between distant residues.
    
    Tertiary contacts are long-range interactions between residues that are
    far apart in sequence but close in 3D space. These contacts are critical
    for protein fold stability and are predicted using quantum resonance
    coupling between residue pairs.
    
    Resonance coupling formula:
        R(E₁,E₂,t) = exp[-(E₁(t) - E₂(t) - ℏωγ)²/(2ℏωγ)] × G(φ,t)
    
    where:
        - E₁, E₂: Quantum energies from QCP
        - ωγ: Gamma frequency (40 Hz)
        - G(φ,t): Golden ratio temporal evolution
        - ℏ: Reduced Planck constant
    
    Contacts are classified as probable when:
        - Resonance strength > 0.7
        - Sequence separation >= 5 residues
        - Spatial distance < 8.0 Ångströms (if current structure available)
    
    Attributes:
        residue_i: First residue index (0-based)
        residue_j: Second residue index (0-based)
        resonance_strength: Resonance coupling strength (0-1)
                           Values > 0.7 indicate probable contact
        predicted_distance: Predicted optimal distance in Ångströms
                           Typically 6.0Å for standard contacts
        sequence_separation: |j - i| = distance in sequence
                            Minimum 5 residues for tertiary contacts
    
    Example:
        >>> contact = TertiaryContact(
        ...     residue_i=10,
        ...     residue_j=45,
        ...     resonance_strength=0.85,
        ...     predicted_distance=6.2,
        ...     sequence_separation=35
        ... )
        >>> if contact.is_probable_contact():
        ...     print(f"Strong contact predicted between residues {contact.residue_i} and {contact.residue_j}")
    """
    residue_i: int
    residue_j: int
    resonance_strength: float  # 0-1
    predicted_distance: float  # Angstroms
    sequence_separation: int  # |j - i|
    
    def __post_init__(self):
        """Validate tertiary contact data."""
        if self.residue_i < 0:
            raise ValueError(f"residue_i must be >= 0, got {self.residue_i}")
        
        if self.residue_j < 0:
            raise ValueError(f"residue_j must be >= 0, got {self.residue_j}")
        
        if self.residue_i == self.residue_j:
            raise ValueError(f"residue_i and residue_j must be different, both are {self.residue_i}")
        
        if not (0.0 <= self.resonance_strength <= 1.0):
            raise ValueError(
                f"resonance_strength must be in [0, 1], got {self.resonance_strength}"
            )
        
        if self.predicted_distance <= 0:
            raise ValueError(f"predicted_distance must be > 0, got {self.predicted_distance}")
        
        if self.sequence_separation <= 0:
            raise ValueError(f"sequence_separation must be > 0, got {self.sequence_separation}")
        
        # Verify sequence_separation matches residue indices
        expected_separation = abs(self.residue_j - self.residue_i)
        if self.sequence_separation != expected_separation:
            raise ValueError(
                f"sequence_separation {self.sequence_separation} does not match "
                f"residue indices difference {expected_separation}"
            )
    
    def is_probable_contact(self, threshold: float = 0.7) -> bool:
        """
        Check if this is a probable tertiary contact.
        
        Args:
            threshold: Minimum resonance strength (default 0.7)
        
        Returns:
            True if resonance_strength >= threshold
        """
        return self.resonance_strength >= threshold
    
    def is_long_range(self, min_separation: int = 5) -> bool:
        """
        Check if this is a long-range contact.
        
        Args:
            min_separation: Minimum sequence separation (default 5)
        
        Returns:
            True if sequence_separation >= min_separation
        """
        return self.sequence_separation >= min_separation
    
    def is_valid_contact(self, 
                        resonance_threshold: float = 0.7,
                        min_separation: int = 5) -> bool:
        """
        Check if contact meets all criteria for validity.
        
        Args:
            resonance_threshold: Minimum resonance strength
            min_separation: Minimum sequence separation
        
        Returns:
            True if contact is both probable and long-range
        """
        return (self.is_probable_contact(resonance_threshold) and 
                self.is_long_range(min_separation))


@dataclass(frozen=True)
class RefinementProgress:
    """
    Real-time progress tracking for quantum refinement.
    
    This class tracks refinement progress across iterations, providing
    visibility into RMSD reduction, energy minimization, restraint
    application, and contact formation. It also estimates time remaining
    based on historical iteration times.
    
    Progress tracking is essential for:
    - Monitoring convergence toward native structure
    - Identifying stuck states or divergence
    - Estimating computational resources needed
    - Debugging refinement strategies
    - Providing user feedback
    
    Metrics tracked:
    - RMSD: Distance from native structure (Ångströms)
    - Energy: Molecular mechanics energy (kcal/mol)
    - Active restraints: Number of distance/angle constraints
    - Formed contacts: Number of tertiary contacts established
    - Time: Cumulative wall-clock time (seconds)
    - Iteration rate: Iterations per second
    
    Attributes:
        iteration: Current iteration number (0-based)
        rmsd: Current RMSD from native structure (Å)
        energy: Current molecular mechanics energy (kcal/mol)
        active_restraints: Number of active distance/angle restraints
        formed_contacts: Number of tertiary contacts formed
        elapsed_time: Cumulative wall-clock time (seconds)
        estimated_time_remaining: Estimated seconds until completion (None if unknown)
        convergence_status: 'improving', 'stuck', 'converged', 'diverging'
    
    Example:
        >>> progress = RefinementProgress(
        ...     iteration=100,
        ...     rmsd=8.5,
        ...     energy=-250.0,
        ...     active_restraints=45,
        ...     formed_contacts=12,
        ...     elapsed_time=15.2,
        ...     estimated_time_remaining=45.0,
        ...     convergence_status='improving'
        ... )
        >>> print(f"Progress: {progress.percent_complete():.1f}% complete, "
        ...       f"{progress.iteration_rate():.1f} iter/sec")
    """
    iteration: int
    rmsd: float
    energy: float
    active_restraints: int
    formed_contacts: int
    elapsed_time: float  # seconds
    estimated_time_remaining: Optional[float] = None  # seconds
    convergence_status: str = 'improving'  # 'improving', 'stuck', 'converged', 'diverging'
    
    def __post_init__(self):
        """Validate progress data."""
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")
        
        if self.rmsd < 0:
            raise ValueError(f"rmsd must be >= 0, got {self.rmsd}")
        
        if self.active_restraints < 0:
            raise ValueError(f"active_restraints must be >= 0, got {self.active_restraints}")
        
        if self.formed_contacts < 0:
            raise ValueError(f"formed_contacts must be >= 0, got {self.formed_contacts}")
        
        if self.elapsed_time < 0:
            raise ValueError(f"elapsed_time must be >= 0, got {self.elapsed_time}")
        
        if self.estimated_time_remaining is not None and self.estimated_time_remaining < 0:
            raise ValueError(f"estimated_time_remaining must be >= 0, got {self.estimated_time_remaining}")
        
        valid_statuses = {'improving', 'stuck', 'converged', 'diverging'}
        if self.convergence_status not in valid_statuses:
            raise ValueError(
                f"convergence_status must be one of {valid_statuses}, got '{self.convergence_status}'"
            )
    
    def iteration_rate(self) -> float:
        """
        Calculate iterations per second.
        
        Returns:
            Iterations per second (0 if elapsed_time is 0)
        """
        if self.elapsed_time == 0:
            return 0.0
        return self.iteration / self.elapsed_time
    
    def percent_complete(self, max_iterations: int = 10000) -> float:
        """
        Calculate percentage of iterations complete.
        
        Args:
            max_iterations: Total iterations planned (default 10000)
        
        Returns:
            Percentage complete (0-100)
        """
        if max_iterations <= 0:
            return 100.0
        return min(100.0, (self.iteration / max_iterations) * 100.0)
    
    def is_converged(self, rmsd_threshold: float = 5.0) -> bool:
        """
        Check if refinement has converged.
        
        Args:
            rmsd_threshold: RMSD target in Ångströms (default 5.0)
        
        Returns:
            True if RMSD <= threshold and status is 'converged'
        """
        return self.rmsd <= rmsd_threshold and self.convergence_status == 'converged'
    
    def is_stuck(self, stuck_window: int = 100) -> bool:
        """
        Check if refinement is stuck.
        
        Note: This is a simple check based on convergence_status.
        For more sophisticated stuck detection, use the refinement
        engine's history-based analysis.
        
        Args:
            stuck_window: Number of iterations to look back (not used here)
        
        Returns:
            True if status is 'stuck'
        """
        return self.convergence_status == 'stuck'
    
    def format_status(self) -> str:
        """
        Format progress as human-readable status string.
        
        Returns:
            Formatted status string with key metrics
        """
        rate = self.iteration_rate()
        status_str = (
            f"Iteration {self.iteration}: "
            f"RMSD={self.rmsd:.2f}Å, "
            f"Energy={self.energy:.1f} kcal/mol, "
            f"Restraints={self.active_restraints}, "
            f"Contacts={self.formed_contacts}, "
            f"Rate={rate:.1f} iter/s"
        )
        
        if self.estimated_time_remaining is not None:
            mins = int(self.estimated_time_remaining / 60)
            secs = int(self.estimated_time_remaining % 60)
            status_str += f", ETA={mins}m {secs}s"
        
        status_str += f" [{self.convergence_status}]"
        return status_str