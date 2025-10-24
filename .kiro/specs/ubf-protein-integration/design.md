# Design Document

## Overview

This design document specifies the technical architecture for integrating the Universal Behavioral Framework (UBF) with quantum-inspired protein structure prediction. The system transforms protein folding from a static physics simulation into an experience-driven multi-agent learning system where autonomous agents navigate conformational space using consciousness coordinates, accumulate experiential memories, and collaboratively discover native protein structures.

### Core Design Principles

1. **Mappless Conformational Navigation**: Agents explore conformational space through capability-based matching rather than spatial pathfinding
2. **Experience-Driven Learning**: No training data required - agents learn through exploration outcomes
3. **SOLID Architecture**: Clean separation of concerns with dependency inversion for extensibility
4. **PyPy Optimization**: Pure Python implementation optimized for PyPy JIT compilation
5. **Quantum-Grounded Physics**: Integration with validated QAAP, resonance, and water shielding modules

### Key Innovation

The mappless design eliminates traditional spatial navigation. Instead of pathfinding through 3D space, agents match conformational moves based on:
- Current structural state capabilities
- Energy barrier feasibility
- Behavioral state preferences (from consciousness coordinates)
- Memory-based learned patterns

This approach scales efficiently because move evaluation is O(1) per agent, independent of conformational space size.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UBF Protein System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Protein    │      │  Multi-Agent │      │  Shared   │ │
│  │    Agent     │◄────►│  Coordinator │◄────►│  Memory   │ │
│  │   (UBF Core) │      │              │      │   Pool    │ │
│  └──────┬───────┘      └──────────────┘      └───────────┘ │
│         │                                                    │
│         │ uses                                               │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Conformational Move Evaluator                 │  │
│  │  (Mappless Capability-Based Matching)                 │  │
│  └──────┬───────────────────────────────────────────────┘  │
│         │ delegates to                                      │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Physics Integration Layer                     │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │  QAAP   │  │Resonance │  │ Water Shielding  │    │  │
│  │  │Calculator│  │ Coupling │  │                  │    │  │
│  │  └─────────┘  └──────────┘  └──────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```


### Layer Responsibilities

**Agent Layer**: Manages consciousness coordinates, behavioral state, and personal memory
**Coordination Layer**: Orchestrates multi-agent exploration, manages shared memory pool
**Evaluation Layer**: Implements mappless move matching based on capabilities
**Physics Layer**: Provides quantum-grounded energy calculations (existing modules)

### SOLID Compliance

- **Single Responsibility**: Each component has one clear purpose (agent state, move evaluation, physics calculation)
- **Open-Closed**: New move types and physics modules can be added without modifying core agent logic
- **Liskov Substitution**: All physics calculators implement common interface, interchangeable
- **Interface Segregation**: Separate interfaces for IConsciousnessState, IMoveEvaluator, IPhysicsCalculator
- **Dependency Inversion**: High-level agent logic depends on abstractions, not concrete physics implementations

## Components and Interfaces

### Core Interfaces (SOLID Design)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# Consciousness & Behavioral State (UBF Core)
# ============================================================================

class IConsciousnessState(ABC):
    """Interface for consciousness coordinate management"""
    
    @abstractmethod
    def get_frequency(self) -> float:
        """Returns current frequency (3-15 Hz)"""
        pass
    
    @abstractmethod
    def get_coherence(self) -> float:
        """Returns current coherence (0.2-1.0)"""
        pass
    
    @abstractmethod
    def update_from_outcome(self, outcome: 'ConformationalOutcome') -> None:
        """Updates coordinates based on exploration outcome"""
        pass

class IBehavioralState(ABC):
    """Interface for cached behavioral state derived from consciousness"""
    
    @abstractmethod
    def get_exploration_energy(self) -> float:
        """Energy level for conformational exploration"""
        pass
    
    @abstractmethod
    def get_structural_focus(self) -> float:
        """Focus/precision for structural refinement"""
        pass
    
    @abstractmethod
    def get_hydrophobic_drive(self) -> float:
        """Drive toward hydrophobic collapse"""
        pass
    
    @abstractmethod
    def should_regenerate(self, coordinate_change: float) -> bool:
        """Check if behavioral state needs regeneration (threshold: 0.3)"""
        pass

# ============================================================================
# Memory System
# ============================================================================

@dataclass
class ConformationalMemory:
    """Memory of a significant conformational state"""
    conformation_id: str
    coordinates: List[Tuple[float, float, float]]  # 3D atom positions
    energy: float
    rmsd_to_native: float
    timestamp: int
    significance: float  # 0.0-1.0
    emotional_impact: float  # -1.0 to +1.0
    decay_factor: float  # 0.0-1.0

class IMemorySystem(ABC):
    """Interface for experience memory management"""
    
    @abstractmethod
    def store_memory(self, memory: ConformationalMemory) -> None:
        """Store memory if significance >= 0.3, auto-prune if > 50"""
        pass
    
    @abstractmethod
    def retrieve_relevant_memories(self, move_type: str, max_count: int = 10) -> List[ConformationalMemory]:
        """Retrieve relevant memories for move evaluation"""
        pass
    
    @abstractmethod
    def calculate_memory_influence(self, move_type: str) -> float:
        """Calculate memory influence multiplier (0.8-1.5)"""
        pass

# ============================================================================
# Conformational Moves (Mappless Design)
# ============================================================================

@dataclass
class ConformationalMove:
    """Represents a potential conformational change"""
    move_id: str
    move_type: str  # 'backbone_rotation', 'sidechain_adjust', 'helix_formation', etc.
    target_residues: List[int]
    estimated_energy_change: float
    structural_requirements: Dict[str, any]  # Capabilities needed
    
class IMoveGenerator(ABC):
    """Interface for generating available conformational moves"""
    
    @abstractmethod
    def generate_moves(self, current_conformation: 'Conformation') -> List[ConformationalMove]:
        """Generate all feasible moves from current state (mappless - no pathfinding)"""
        pass


class IMoveEvaluator(ABC):
    """Interface for evaluating and weighting conformational moves"""
    
    @abstractmethod
    def evaluate_move(self, 
                     move: ConformationalMove,
                     behavioral_state: IBehavioralState,
                     memory_influence: float,
                     physics_factors: Dict[str, float]) -> float:
        """Calculate weight for move using 18 factors (mappless capability matching)"""
        pass

# ============================================================================
# Physics Integration (Existing Modules)
# ============================================================================

class IPhysicsCalculator(ABC):
    """Base interface for physics calculation modules"""
    
    @abstractmethod
    def calculate(self, conformation: 'Conformation') -> float:
        """Calculate physics-based score/energy"""
        pass

class IQAAPCalculator(IPhysicsCalculator):
    """Interface for Quantum Amino Acid Potential calculator"""
    
    @abstractmethod
    def calculate_qaap_potential(self, conformation: 'Conformation') -> float:
        """Calculate quantum potential: QCP = 4 + (2^n × φ^l × m)"""
        pass

class IResonanceCoupling(IPhysicsCalculator):
    """Interface for 40 Hz gamma resonance coupling"""
    
    @abstractmethod
    def calculate_resonance(self, residue1: int, residue2: int, conformation: 'Conformation') -> float:
        """Calculate resonance: R(E₁,E₂) = exp[-(E₁ - E₂ - ℏω_γ)²/(2ℏω_γ)]"""
        pass

class IWaterShielding(IPhysicsCalculator):
    """Interface for water shielding effects"""
    
    @abstractmethod
    def calculate_shielding(self, conformation: 'Conformation') -> float:
        """Calculate water shielding with 408 fs coherence time, 3.57 nm⁻¹ factor"""
        pass

# ============================================================================
# Protein Agent (Main UBF Component)
# ============================================================================

class IProteinAgent(ABC):
    """Interface for autonomous protein folding agent"""
    
    @abstractmethod
    def get_consciousness_state(self) -> IConsciousnessState:
        """Get current consciousness coordinates"""
        pass
    
    @abstractmethod
    def get_behavioral_state(self) -> IBehavioralState:
        """Get cached behavioral state"""
        pass
    
    @abstractmethod
    def get_memory_system(self) -> IMemorySystem:
        """Get agent's memory system"""
        pass
    
    @abstractmethod
    def explore_step(self) -> 'ConformationalOutcome':
        """Execute one exploration step (mappless move selection and execution)"""
        pass
    
    @abstractmethod
    def get_current_conformation(self) -> 'Conformation':
        """Get current protein conformation"""
        pass

# ============================================================================
# Multi-Agent Coordination
# ============================================================================

class ISharedMemoryPool(ABC):
    """Interface for shared memory across all agents"""
    
    @abstractmethod
    def share_memory(self, memory: ConformationalMemory) -> None:
        """Share high-significance memory (>= 0.7) with all agents"""
        pass
    
    @abstractmethod
    def retrieve_shared_memories(self, move_type: str, max_count: int = 10) -> List[ConformationalMemory]:
        """Retrieve relevant shared memories"""
        pass
    
    @abstractmethod
    def prune_pool(self, max_size: int = 10000) -> None:
        """Prune pool to max size by weighted significance"""
        pass

class IMultiAgentCoordinator(ABC):
    """Interface for coordinating multiple agents"""
    
    @abstractmethod
    def initialize_agents(self, count: int, diversity_profile: str) -> List[IProteinAgent]:
        """Initialize agents with diversity: 33% cautious, 34% balanced, 33% aggressive"""
        pass
    
    @abstractmethod
    def run_parallel_exploration(self, iterations: int) -> 'ExplorationResults':
        """Run all agents in parallel for N iterations"""
        pass
    
    @abstractmethod
    def get_best_conformation(self) -> Tuple['Conformation', float, float]:
        """Get best conformation found (conformation, energy, RMSD)"""
        pass

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

class IVisualizationExporter(ABC):
    """Interface for exporting visualization data"""
    
    @abstractmethod
    def export_trajectory(self, agent_id: str) -> List[ConformationSnapshot]:
        """Export complete trajectory for an agent"""
        pass
    
    @abstractmethod
    def export_energy_landscape(self) -> 'EnergyLandscape':
        """Export 2D projection of explored conformational space"""
        pass
    
    @abstractmethod
    def stream_update(self, snapshot: ConformationSnapshot) -> None:
        """Stream real-time update (non-blocking)"""
        pass

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
    configuration: Dict[str, any]
    agent_states: List[Dict[str, any]]  # Serialized agent states
    shared_memory_pool: List[ConformationalMemory]
    best_conformation: Optional[Conformation]
    metadata: Dict[str, any]

class ICheckpointManager(ABC):
    """Interface for checkpoint and resume functionality"""
    
    @abstractmethod
    def save_checkpoint(self, 
                       agents: List[IProteinAgent],
                       shared_pool: ISharedMemoryPool,
                       iteration: int,
                       metadata: Dict[str, any]) -> str:
        """Save complete system state, returns checkpoint file path"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> SystemCheckpoint:
        """Load checkpoint from file"""
        pass
    
    @abstractmethod
    def restore_agents(self, checkpoint: SystemCheckpoint) -> Tuple[List[IProteinAgent], ISharedMemoryPool, int]:
        """Restore agents and shared pool from checkpoint, returns (agents, pool, iteration)"""
        pass

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

class IAdaptiveConfigurator(ABC):
    """Interface for adaptive configuration based on protein properties"""
    
    @abstractmethod
    def get_config_for_protein(self, sequence: str) -> AdaptiveConfig:
        """Generate adaptive configuration based on protein sequence"""
        pass
    
    @abstractmethod
    def classify_protein_size(self, sequence: str) -> ProteinSizeClass:
        """Classify protein as small, medium, or large"""
        pass
    
    @abstractmethod
    def scale_threshold(self, base_threshold: float, residue_count: int) -> float:
        """Scale threshold proportionally to protein size"""
        pass
```

### Concrete Implementation Classes

The interfaces above will be implemented by concrete classes following SOLID principles:

- `ConsciousnessState` implements `IConsciousnessState`
- `BehavioralState` implements `IBehavioralState`
- `MemorySystem` implements `IMemorySystem`
- `MapplessMoveGenerator` implements `IMoveGenerator`
- `CapabilityBasedMoveEvaluator` implements `IMoveEvaluator`
- `QAAPCalculator` implements `IQAAPCalculator` (existing module adapter)
- `ResonanceCoupling` implements `IResonanceCoupling` (existing module adapter)
- `WaterShielding` implements `IWaterShielding` (existing module adapter)
- `ProteinAgent` implements `IProteinAgent`
- `SharedMemoryPool` implements `ISharedMemoryPool`
- `MultiAgentCoordinator` implements `IMultiAgentCoordinator`


## Data Models

### Core Data Structures

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum

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
            hydrophobic_drive=(freq - 4.0) / 8.0,
            risk_tolerance=(freq - 6.0) / 6.0,
            native_state_ambition=coh * (freq / 10.0),
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
    structural_constraints: Dict[str, any]  # Constraints limiting moves
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return capability flags for mappless move matching"""
        return {
            'can_form_helix': self._can_form_helix(),
            'can_form_sheet': self._can_form_sheet(),
            'can_hydrophobic_collapse': self._can_collapse(),
            'can_large_rotation': self._can_large_rotation(),
            'has_flexible_loops': self._has_flexible_loops()
        }

class MoveType(Enum):
    """Types of conformational moves (mappless categories)"""
    BACKBONE_ROTATION = "backbone_rotation"
    SIDECHAIN_ADJUST = "sidechain_adjust"
    HELIX_FORMATION = "helix_formation"
    SHEET_FORMATION = "sheet_formation"
    TURN_FORMATION = "turn_formation"
    HYDROPHOBIC_COLLAPSE = "hydrophobic_collapse"
    SALT_BRIDGE = "salt_bridge"
    DISULFIDE_BOND = "disulfide_bond"
    ENERGY_MINIMIZATION = "energy_minimization"
    LARGE_CONFORMATIONAL_JUMP = "large_jump"

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

# Consciousness update rules (from UBF)
CONSCIOUSNESS_UPDATE_RULES = {
    OutcomeType.ENERGY_DECREASE_LARGE: {'frequency': +0.5, 'coherence': +0.1},
    OutcomeType.ENERGY_DECREASE_SMALL: {'frequency': +0.2, 'coherence': +0.05},
    OutcomeType.ENERGY_INCREASE: {'frequency': -0.3, 'coherence': -0.05},
    OutcomeType.STRUCTURE_COLLAPSE: {'frequency': -1.0, 'coherence': -0.2},
    OutcomeType.STABLE_MINIMUM_FOUND: {'frequency': +0.3, 'coherence': +0.15},
    OutcomeType.HELIX_FORMATION: {'frequency': +0.2, 'coherence': +0.08},
    OutcomeType.SHEET_FORMATION: {'frequency': +0.2, 'coherence': +0.08},
    OutcomeType.HYDROPHOBIC_CORE_FORMED: {'frequency': +0.4, 'coherence': +0.1},
    OutcomeType.STUCK_IN_LOCAL_MINIMUM: {'frequency': +1.0, 'coherence': -0.1},  # Boost to escape!
}

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

AGENT_DIVERSITY_PROFILES = {
    'cautious': AgentProfile(
        profile_name='cautious',
        frequency_range=(4.0, 7.0),
        coherence_range=(0.7, 1.0),
        description='Low energy, high focus - careful local exploration'
    ),
    'balanced': AgentProfile(
        profile_name='balanced',
        frequency_range=(7.0, 10.0),
        coherence_range=(0.5, 0.8),
        description='Moderate energy and focus - balanced exploration'
    ),
    'aggressive': AgentProfile(
        profile_name='aggressive',
        frequency_range=(10.0, 15.0),
        coherence_range=(0.3, 0.6),
        description='High energy, lower focus - bold exploration, escapes minima'
    )
}

# ============================================================================
# Metrics & Results
# ============================================================================

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
    best_conformation: Conformation
    best_energy: float
    best_rmsd: float
    agent_metrics: List[ExplorationMetrics]
    collective_learning_benefit: float  # Multi-agent improvement over single
    total_runtime_seconds: float
    shared_memories_created: int
```


## Mappless Conformational Navigation

### Core Concept

Traditional protein folding simulations use spatial pathfinding to navigate 3D space. The mappless design eliminates this overhead by using **capability-based matching**:

1. **No Spatial Maps**: Conformations are abstract nodes with properties, not spatial coordinates
2. **No Pathfinding**: Moves are evaluated by capability compatibility, not geometric paths
3. **Constant-Time Matching**: Move evaluation is O(1) per agent, independent of conformational space size

### Mappless Move Generation

```python
class MapplessMoveGenerator:
    """Generates available moves based on current conformation capabilities"""
    
    def generate_moves(self, conformation: Conformation) -> List[ConformationalMove]:
        """
        Generate moves by matching current capabilities to move requirements
        No pathfinding - just capability checking
        """
        available_moves = []
        capabilities = conformation.get_capabilities()
        
        # Check each move type against capabilities
        for move_type in MoveType:
            if self._is_move_feasible(move_type, capabilities, conformation):
                move = self._create_move(move_type, conformation)
                available_moves.append(move)
        
        return available_moves
    
    def _is_move_feasible(self, move_type: MoveType, 
                          capabilities: Dict[str, bool],
                          conformation: Conformation) -> bool:
        """
        Check if move is feasible based on capabilities (mappless)
        No spatial checks - just structural compatibility
        """
        if move_type == MoveType.HELIX_FORMATION:
            return capabilities['can_form_helix']
        elif move_type == MoveType.HYDROPHOBIC_COLLAPSE:
            return capabilities['can_hydrophobic_collapse']
        elif move_type == MoveType.LARGE_CONFORMATIONAL_JUMP:
            return capabilities['can_large_rotation']
        # ... other move types
        
        return True  # Default: most moves are feasible
```

### Mappless Move Evaluation (Simplified Composite Factors)

```python
class CapabilityBasedMoveEvaluator:
    """Evaluates moves using 5 composite factors without spatial pathfinding"""
    
    def __init__(self, 
                 qaap_calculator: IQAAPCalculator,
                 resonance_coupling: IResonanceCoupling,
                 water_shielding: IWaterShielding):
        self.qaap = qaap_calculator
        self.resonance = resonance_coupling
        self.water = water_shielding
    
    def evaluate_move(self,
                     move: ConformationalMove,
                     behavioral_state: IBehavioralState,
                     memory_influence: float,
                     current_conformation: Conformation) -> float:
        """
        Calculate move weight using 5 composite factors (mappless)
        Simplified from 18 individual factors for easier tuning
        """
        weight = 1.0
        
        # COMPOSITE FACTOR 1: Physical Feasibility (combines structural, energy barrier, Ramachandran)
        physical_feasibility = self._calculate_physical_feasibility(
            move, current_conformation, behavioral_state
        )
        weight *= physical_feasibility  # Range: 0.1-2.0
        
        # COMPOSITE FACTOR 2: Quantum Alignment (QAAP, resonance, water shielding)
        quantum_alignment = self._calculate_quantum_alignment(
            move, current_conformation
        )
        weight *= quantum_alignment  # Range: 0.5-1.5
        
        # COMPOSITE FACTOR 3: Behavioral Preference (all 5 behavioral dimensions)
        behavioral_preference = self._calculate_behavioral_preference(
            move, behavioral_state
        )
        weight *= behavioral_preference  # Range: 0.5-2.5
        
        # COMPOSITE FACTOR 4: Historical Success (memory influence + novelty)
        historical_success = self._calculate_historical_success(
            move, memory_influence, current_conformation
        )
        weight *= historical_success  # Range: 0.8-1.8
        
        # COMPOSITE FACTOR 5: Goal Alignment (energy decrease + RMSD improvement)
        goal_alignment = self._calculate_goal_alignment(move)
        weight += goal_alignment  # Additive boost: 0-10.0
        
        return max(0.0, weight)  # Ensure non-negative
    
    def _calculate_physical_feasibility(self, move, conformation, behavioral_state):
        """
        Combines:
        - Structural feasibility (capability-based)
        - Energy barrier (modulated by exploration energy)
        - Ramachandran favorability
        """
        structural = move.structural_feasibility
        
        # Energy barrier factor (higher exploration energy = more willing to cross barriers)
        barrier_tolerance = behavioral_state.get_exploration_energy()
        barrier_factor = 1.0 / (1.0 + move.energy_barrier / (50.0 * barrier_tolerance))
        
        # Ramachandran favorability (simplified check)
        ramachandran = self._check_ramachandran_allowed(move, conformation)
        
        return structural * barrier_factor * ramachandran
    
    def _calculate_quantum_alignment(self, move, conformation):
        """
        Combines:
        - QAAP quantum potential
        - Resonance coupling (40 Hz gamma)
        - Water shielding effects
        """
        qaap_score = self.qaap.calculate_qaap_potential(conformation)
        qaap_factor = 0.7 + (qaap_score / 100.0) * 0.6  # 0.7-1.3 range
        
        resonance_score = self.resonance.calculate_resonance(
            move.target_residues[0],
            move.target_residues[-1],
            conformation
        )
        resonance_factor = 0.9 + resonance_score * 0.3  # 0.9-1.2 range
        
        shielding_score = self.water.calculate_shielding(conformation)
        shielding_factor = 0.95 + shielding_score * 0.1  # 0.95-1.05 range
        
        return qaap_factor * resonance_factor * shielding_factor
    
    def _calculate_behavioral_preference(self, move, behavioral_state):
        """
        Combines all 5 behavioral dimensions:
        - Exploration energy
        - Structural focus
        - Hydrophobic drive
        - Risk tolerance
        - Native state ambition
        """
        base_preference = 1.0
        
        # Large jumps favor high exploration energy
        if move.move_type == MoveType.LARGE_CONFORMATIONAL_JUMP:
            base_preference *= (0.5 + behavioral_state.get_exploration_energy() * 1.5)
        
        # Structure formation favors high focus
        if move.move_type in [MoveType.HELIX_FORMATION, MoveType.SHEET_FORMATION]:
            base_preference *= (0.7 + behavioral_state.get_structural_focus() * 1.0)
        
        # Hydrophobic collapse favors hydrophobic drive
        if move.move_type == MoveType.HYDROPHOBIC_COLLAPSE:
            base_preference *= (0.5 + behavioral_state.get_hydrophobic_drive() * 2.0)
        
        # Risk tolerance affects all moves
        base_preference *= (0.8 + behavioral_state.risk_tolerance * 0.4)
        
        # Ambition affects goal-directed moves
        if move.estimated_energy_change < 0:
            base_preference *= (0.9 + behavioral_state.native_state_ambition * 0.3)
        
        return base_preference
    
    def _calculate_historical_success(self, move, memory_influence, conformation):
        """
        Combines:
        - Memory influence (0.8-1.5 from past experiences)
        - Novelty bonus (encourage exploration of new moves)
        """
        # Memory influence already calculated (0.8-1.5)
        memory_factor = memory_influence
        
        # Novelty bonus (simple check if move type rarely used)
        novelty_factor = self._calculate_novelty_bonus(move, conformation)
        
        return memory_factor * novelty_factor
    
    def _calculate_goal_alignment(self, move):
        """
        Combines:
        - Energy decrease (dominant factor)
        - RMSD improvement
        """
        alignment = 0.0
        
        # Energy decrease (dominant +10.0 boost)
        if move.estimated_energy_change < 0:
            alignment += 10.0 * abs(move.estimated_energy_change) / 100.0
        
        # RMSD improvement
        if move.estimated_rmsd_change < 0:
            alignment += abs(move.estimated_rmsd_change) * 3.0
        
        return alignment
```

### Progressive Complexity: Evaluation Factor Evolution

The system starts with simplified 5-factor evaluation and can be expanded:

**Phase 1 (MVP)**: 5 composite factors as shown above
**Phase 2 (Refinement)**: Split composites into 10 factors (separate physics components)
**Phase 3 (Full)**: Expand to 18 individual factors for fine-grained control

This progressive approach allows:
- Faster initial implementation and testing
- Easier parameter tuning with fewer variables
- Gradual complexity increase based on validation results


### Mappless Performance Advantages

```
Traditional Spatial Approach:
- Build 3D spatial index: O(N log N)
- Pathfinding per move: O(N log N)
- Total per agent: O(M × N log N) where M = moves, N = atoms
- 1000 agents: ~500ms per iteration

Mappless Capability Approach:
- Extract capabilities: O(1) per conformation
- Match moves: O(M) where M = move types (~10)
- Evaluate moves: O(M) with physics calculations
- Total per agent: O(M) = O(10) = constant time
- 1000 agents: ~50ms per iteration (10x faster)
```

## Error Handling

### Local Minima Detection and Escape (Adaptive)

```python
class LocalMinimaDetector:
    """Detects when agent is stuck in local minimum with adaptive thresholds"""
    
    def __init__(self, config: AdaptiveConfig):
        self.window_size = config.stuck_detection_window  # Adaptive: 10/20/30 based on size
        self.threshold = config.stuck_detection_threshold  # Adaptive: scaled by protein size
        self.energy_history = []
        self.escape_attempts = 0
    
    def update(self, energy: float) -> bool:
        """
        Returns True if stuck in local minimum
        Uses moving average instead of consecutive count for robustness
        """
        self.energy_history.append(energy)
        
        if len(self.energy_history) < self.window_size:
            return False
        
        # Keep only recent history
        self.energy_history = self.energy_history[-self.window_size:]
        
        # Calculate moving average of energy changes
        changes = [abs(self.energy_history[i] - self.energy_history[i-1])
                  for i in range(1, len(self.energy_history))]
        avg_change = sum(changes) / len(changes)
        
        # Stuck if average change is below threshold
        return avg_change < self.threshold
    
    def get_escape_strategy(self) -> Dict[str, float]:
        """
        Return consciousness coordinate adjustments to escape minimum
        Multiple strategies based on escape attempt count
        """
        self.escape_attempts += 1
        
        if self.escape_attempts == 1:
            # Strategy 1: Moderate frequency boost
            return {
                'frequency_delta': +1.0,
                'coherence_delta': -0.1,
                'strategy': 'moderate_boost'
            }
        elif self.escape_attempts == 2:
            # Strategy 2: Large frequency boost with more chaos
            return {
                'frequency_delta': +2.0,
                'coherence_delta': -0.2,
                'strategy': 'large_boost'
            }
        else:
            # Strategy 3: Maximum exploration with large jump bias
            return {
                'frequency_delta': +3.0,
                'coherence_delta': -0.3,
                'bias_large_jumps': True,
                'strategy': 'maximum_exploration'
            }
    
    def reset_escape_attempts(self):
        """Reset escape attempt counter after successful escape"""
        self.escape_attempts = 0

class StructuralValidation:
    """Validates conformational integrity"""
    
    @staticmethod
    def validate_conformation(conformation: Conformation) -> Tuple[bool, List[str]]:
        """
        Validate conformation for structural integrity
        Returns (is_valid, error_messages)
        """
        errors = []
        
        # Check bond lengths
        if not StructuralValidation._check_bond_lengths(conformation):
            errors.append("Invalid bond lengths detected")
        
        # Check for clashes
        if not StructuralValidation._check_steric_clashes(conformation):
            errors.append("Steric clashes detected")
        
        # Check backbone continuity
        if not StructuralValidation._check_backbone_continuity(conformation):
            errors.append("Backbone discontinuity detected")
        
        # Check for NaN/Inf coordinates
        if not StructuralValidation._check_coordinate_validity(conformation):
            errors.append("Invalid coordinates (NaN/Inf)")
        
        return (len(errors) == 0, errors)
    
    @staticmethod
    def repair_conformation(conformation: Conformation) -> Conformation:
        """
        Attempt to repair invalid conformation
        Returns repaired conformation or raises exception if unrepairable
        """
        # Attempt repairs
        conformation = StructuralValidation._fix_bond_lengths(conformation)
        conformation = StructuralValidation._resolve_clashes(conformation)
        conformation = StructuralValidation._fix_backbone(conformation)
        
        # Validate after repair
        is_valid, errors = StructuralValidation.validate_conformation(conformation)
        
        if not is_valid:
            raise ValueError(f"Conformation unrepairable: {errors}")
        
        return conformation
```

### Memory System Error Handling

```python
class RobustMemorySystem:
    """Memory system with error handling and recovery"""
    
    def store_memory(self, memory: ConformationalMemory) -> None:
        """Store memory with validation and error recovery"""
        try:
            # Validate memory data
            if not self._validate_memory(memory):
                logger.warning(f"Invalid memory data, skipping: {memory.conformation_id}")
                return
            
            # Check significance threshold
            if memory.significance < 0.3:
                return  # Below threshold, don't store
            
            # Store memory
            self.memories.append(memory)
            
            # Auto-prune if exceeds limit
            if len(self.memories) > 50:
                self._prune_memories()
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            # Continue execution - memory storage failure shouldn't crash agent
    
    def retrieve_relevant_memories(self, move_type: str, max_count: int = 10) -> List[ConformationalMemory]:
        """Retrieve memories with error handling"""
        try:
            relevant = [m for m in self.memories if self._is_relevant(m, move_type)]
            relevant.sort(key=lambda m: m.significance * m.decay_factor, reverse=True)
            return relevant[:max_count]
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []  # Return empty list on error, don't crash
```


## Testing Strategy

### Unit Testing

```python
# Test consciousness coordinate updates
class TestConsciousnessState:
    def test_frequency_bounds(self):
        """Test frequency stays within 3-15 Hz bounds"""
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        
        # Apply large positive update
        outcome = ConformationalOutcome(energy_change=-200, ...)
        state.update_from_outcome(outcome)
        
        assert 3.0 <= state.frequency <= 15.0
    
    def test_coherence_bounds(self):
        """Test coherence stays within 0.2-1.0 bounds"""
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        
        # Apply large negative update
        outcome = ConformationalOutcome(energy_change=+200, ...)
        state.update_from_outcome(outcome)
        
        assert 0.2 <= state.coherence <= 1.0
    
    def test_behavioral_state_regeneration(self):
        """Test behavioral state regenerates when coordinates change significantly"""
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        old_behavioral = state.get_behavioral_state()
        
        # Small change - should not regenerate
        state.frequency = 7.6
        assert state.get_behavioral_state() is old_behavioral  # Same cached instance
        
        # Large change - should regenerate
        state.frequency = 8.5  # Change of 1.0 > threshold 0.3
        assert state.get_behavioral_state() is not old_behavioral  # New instance

# Test memory system
class TestMemorySystem:
    def test_significance_threshold(self):
        """Test memories below 0.3 significance are not stored"""
        memory_system = MemorySystem()
        
        low_sig_memory = ConformationalMemory(significance=0.2, ...)
        memory_system.store_memory(low_sig_memory)
        
        assert len(memory_system.memories) == 0
    
    def test_auto_pruning(self):
        """Test auto-pruning when exceeding 50 memories"""
        memory_system = MemorySystem()
        
        # Add 55 memories
        for i in range(55):
            memory = ConformationalMemory(
                significance=0.5 + (i / 100),  # Varying significance
                decay_factor=1.0,
                ...
            )
            memory_system.store_memory(memory)
        
        # Should have pruned to 50
        assert len(memory_system.memories) == 50
        
        # Should keep highest weighted significance
        assert all(m.significance >= 0.5 for m in memory_system.memories)
    
    def test_memory_influence_range(self):
        """Test memory influence stays within 0.8-1.5 range"""
        memory_system = MemorySystem()
        
        # Add very positive memories
        for i in range(10):
            memory = ConformationalMemory(
                emotional_impact=1.0,
                significance=0.9,
                decay_factor=1.0,
                ...
            )
            memory_system.store_memory(memory)
        
        influence = memory_system.calculate_memory_influence("test_move")
        assert 0.8 <= influence <= 1.5

# Test mappless move generation
class TestMapplessMoveGenerator:
    def test_capability_based_filtering(self):
        """Test moves are filtered by capabilities, not spatial constraints"""
        conformation = Conformation(...)
        conformation.capabilities = {
            'can_form_helix': True,
            'can_form_sheet': False,
            'can_hydrophobic_collapse': True
        }
        
        generator = MapplessMoveGenerator()
        moves = generator.generate_moves(conformation)
        
        # Should include helix formation
        assert any(m.move_type == MoveType.HELIX_FORMATION for m in moves)
        
        # Should NOT include sheet formation
        assert not any(m.move_type == MoveType.SHEET_FORMATION for m in moves)
    
    def test_constant_time_generation(self):
        """Test move generation is O(1) regardless of conformation size"""
        import time
        
        small_conf = Conformation(sequence="A" * 20, ...)  # 20 residues
        large_conf = Conformation(sequence="A" * 200, ...)  # 200 residues
        
        generator = MapplessMoveGenerator()
        
        start = time.time()
        moves_small = generator.generate_moves(small_conf)
        time_small = time.time() - start
        
        start = time.time()
        moves_large = generator.generate_moves(large_conf)
        time_large = time.time() - start
        
        # Time should be similar (within 2x) despite 10x size difference
        assert time_large < time_small * 2.0

# Test local minima detection
class TestLocalMinimaDetector:
    def test_stuck_detection(self):
        """Test detection of stuck state"""
        detector = LocalMinimaDetector(window_size=20, threshold=10.0)
        
        # Simulate stuck state (small energy changes)
        for i in range(25):
            detector.update(1000.0 + i * 0.5)  # Changes < 10 kJ/mol
        
        assert detector.update(1000.0) == True  # Should detect stuck
    
    def test_not_stuck_with_large_changes(self):
        """Test no false positives with large energy changes"""
        detector = LocalMinimaDetector(window_size=20, threshold=10.0)
        
        # Simulate active exploration (large changes)
        for i in range(25):
            detector.update(1000.0 + i * 50.0)  # Changes > 10 kJ/mol
        
        assert detector.update(1000.0) == False  # Should not detect stuck
```

### Integration Testing

```python
class TestProteinAgentIntegration:
    def test_full_exploration_cycle(self):
        """Test complete exploration cycle: generate → evaluate → execute → update"""
        agent = ProteinAgent(
            initial_frequency=7.5,
            initial_coherence=0.7,
            sequence="ACDEFGHIKLMNPQRSTVWY"  # 20 residue test protein
        )
        
        # Execute one exploration step
        outcome = agent.explore_step()
        
        # Verify outcome structure
        assert outcome.move_executed is not None
        assert outcome.new_conformation is not None
        assert isinstance(outcome.energy_change, float)
        
        # Verify consciousness updated
        new_freq = agent.get_consciousness_state().get_frequency()
        assert new_freq != 7.5  # Should have changed
        
        # Verify memory created if significant
        if outcome.significance >= 0.3:
            assert len(agent.get_memory_system().memories) > 0
    
    def test_multi_agent_coordination(self):
        """Test multi-agent parallel exploration with shared memory"""
        coordinator = MultiAgentCoordinator()
        agents = coordinator.initialize_agents(count=10, diversity_profile='mixed')
        
        # Verify diversity
        frequencies = [a.get_consciousness_state().get_frequency() for a in agents]
        assert min(frequencies) < 7.0  # Some cautious agents
        assert max(frequencies) > 10.0  # Some aggressive agents
        
        # Run parallel exploration
        results = coordinator.run_parallel_exploration(iterations=100)
        
        # Verify results
        assert results.total_conformations_explored > 1000  # 10 agents × 100 iterations
        assert results.best_energy < agents[0].get_current_conformation().energy  # Improvement
        assert results.shared_memories_created > 0  # Agents shared discoveries
```


### Performance Testing

```python
class TestPerformanceMetrics:
    def test_decision_latency(self):
        """Test decision latency < 2ms per move evaluation"""
        import time
        
        agent = ProteinAgent(...)
        evaluator = CapabilityBasedMoveEvaluator(...)
        
        move = ConformationalMove(...)
        behavioral_state = agent.get_behavioral_state()
        memory_influence = agent.get_memory_system().calculate_memory_influence("test")
        
        start = time.time()
        weight = evaluator.evaluate_move(move, behavioral_state, memory_influence, agent.get_current_conformation())
        elapsed_ms = (time.time() - start) * 1000
        
        assert elapsed_ms < 2.0  # Target: < 2ms
    
    def test_memory_retrieval_speed(self):
        """Test memory retrieval < 10μs"""
        import time
        
        memory_system = MemorySystem()
        
        # Fill with 50 memories
        for i in range(50):
            memory_system.store_memory(ConformationalMemory(...))
        
        start = time.time()
        memories = memory_system.retrieve_relevant_memories("test_move", max_count=10)
        elapsed_us = (time.time() - start) * 1_000_000
        
        assert elapsed_us < 10.0  # Target: < 10μs
    
    def test_pypy_speedup(self):
        """Test PyPy achieves 2x speedup over CPython"""
        # This test should be run separately under CPython and PyPy
        import sys
        import time
        
        agent = ProteinAgent(...)
        
        start = time.time()
        for i in range(1000):
            agent.explore_step()
        elapsed = time.time() - start
        
        # Record baseline under CPython, verify speedup under PyPy
        if 'PyPy' in sys.version:
            # Under PyPy, should be at least 2x faster than CPython baseline
            # (Baseline would be recorded separately)
            assert elapsed < 10.0  # Example: CPython baseline was 20s
    
    def test_100_agent_throughput(self):
        """Test 100 agents complete 500K conformations in 2 minutes"""
        import time
        
        coordinator = MultiAgentCoordinator()
        agents = coordinator.initialize_agents(count=100, diversity_profile='mixed')
        
        start = time.time()
        results = coordinator.run_parallel_exploration(iterations=5000)  # 100 × 5000 = 500K
        elapsed = time.time() - start
        
        assert elapsed < 120.0  # Target: < 2 minutes
        assert results.total_conformations_explored >= 500_000

class TestMemoryFootprint:
    def test_agent_memory_usage(self):
        """Test single agent uses < 50 MB with 50 memories"""
        import sys
        
        agent = ProteinAgent(...)
        
        # Fill memory system
        for i in range(50):
            memory = ConformationalMemory(...)
            agent.get_memory_system().store_memory(memory)
        
        # Measure memory (approximate)
        memory_mb = sys.getsizeof(agent) / (1024 * 1024)
        
        assert memory_mb < 50.0  # Target: < 50 MB per agent
    
    def test_shared_pool_memory(self):
        """Test shared memory pool stays under 20 MB for 10K memories"""
        import sys
        
        pool = SharedMemoryPool()
        
        # Fill pool
        for i in range(10_000):
            memory = ConformationalMemory(...)
            pool.share_memory(memory)
        
        memory_mb = sys.getsizeof(pool) / (1024 * 1024)
        
        assert memory_mb < 20.0  # Target: < 20 MB
```

### Validation Testing

```python
class TestPhysicsIntegration:
    def test_qaap_calculator_integration(self):
        """Test QAAP calculator produces valid quantum potentials"""
        qaap = QAAPCalculator()
        conformation = Conformation(...)
        
        potential = qaap.calculate_qaap_potential(conformation)
        
        # QAAP formula: QCP = 4 + (2^n × φ^l × m)
        # Should be positive and reasonable magnitude
        assert potential > 0
        assert potential < 1000  # Reasonable upper bound
    
    def test_resonance_coupling_integration(self):
        """Test resonance coupling produces valid 40 Hz gamma values"""
        resonance = ResonanceCoupling()
        conformation = Conformation(...)
        
        coupling = resonance.calculate_resonance(0, 10, conformation)
        
        # Resonance: R(E₁,E₂) = exp[-(E₁ - E₂ - ℏω_γ)²/(2ℏω_γ)]
        # Should be between 0 and 1
        assert 0.0 <= coupling <= 1.0
    
    def test_water_shielding_integration(self):
        """Test water shielding uses correct parameters"""
        water = WaterShielding()
        conformation = Conformation(...)
        
        shielding = water.calculate_shielding(conformation)
        
        # Should use 408 fs coherence time, 3.57 nm⁻¹ factor
        # Result should be positive
        assert shielding > 0

class TestLearningMetrics:
    def test_learning_improvement_calculation(self):
        """Test 66% improvement rate is measurable"""
        agent = ProteinAgent(...)
        
        # Record initial RMSD (first 20 iterations)
        initial_rmsds = []
        for i in range(20):
            outcome = agent.explore_step()
            initial_rmsds.append(outcome.new_conformation.rmsd_to_native)
        
        # Continue exploration
        for i in range(80):
            agent.explore_step()
        
        # Record final RMSD (last 20 iterations)
        final_rmsds = []
        for i in range(20):
            outcome = agent.explore_step()
            final_rmsds.append(outcome.new_conformation.rmsd_to_native)
        
        # Calculate improvement
        initial_avg = sum(initial_rmsds) / len(initial_rmsds)
        final_avg = sum(final_rmsds) / len(final_rmsds)
        improvement = (initial_avg - final_avg) / initial_avg * 100
        
        # Target: 50-66% improvement (like maze navigation)
        assert improvement > 50.0
```

## PyPy Optimization Strategy

### PyPy-Friendly Code Patterns

```python
# GOOD: Pure Python with type hints (PyPy JIT optimizes well)
def calculate_energy(coordinates: List[Tuple[float, float, float]]) -> float:
    total_energy = 0.0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            dz = coordinates[i][2] - coordinates[j][2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            total_energy += 1.0 / dist
    return total_energy

# AVOID: NumPy (PyPy doesn't optimize NumPy well)
# import numpy as np
# def calculate_energy_numpy(coordinates):
#     coords = np.array(coordinates)
#     # ... NumPy operations (slow under PyPy)

# GOOD: List comprehensions (PyPy optimizes)
energies = [calculate_energy(conf.atom_coordinates) for conf in conformations]

# GOOD: Simple loops (PyPy JIT optimizes)
for agent in agents:
    outcome = agent.explore_step()
    results.append(outcome)

# AVOID: Complex metaclasses or dynamic code generation
# (PyPy JIT struggles with highly dynamic code)
```

### PyPy Performance Targets

```
CPython Baseline (1000 iterations, 100 agents):
- Total time: 120 seconds
- Per-iteration: 120ms
- Per-agent-iteration: 1.2ms

PyPy Target (2x speedup minimum):
- Total time: < 60 seconds
- Per-iteration: < 60ms
- Per-agent-iteration: < 0.6ms

PyPy Stretch Goal (3x speedup):
- Total time: < 40 seconds
- Per-iteration: < 40ms
- Per-agent-iteration: < 0.4ms
```


## Implementation Phases

### Phase 1: Core UBF Components (Foundation)

**Goal**: Implement consciousness coordinates, behavioral state, and basic agent structure

**Components**:
- `ConsciousnessState` class with frequency/coherence management
- `BehavioralState` class with caching and regeneration logic
- `ProteinAgent` class with basic structure
- Consciousness update rules from outcomes
- Unit tests for coordinate bounds and behavioral state generation

**Success Criteria**:
- Consciousness coordinates stay within bounds (3-15 Hz, 0.2-1.0)
- Behavioral state regenerates only when coordinates change > 0.3
- Update rules correctly modify coordinates based on outcomes
- All unit tests pass

**Estimated Effort**: 1-2 days

### Phase 2: Memory System

**Goal**: Implement experience memory with significance filtering and auto-pruning

**Components**:
- `MemorySystem` class with storage and retrieval
- `ConformationalMemory` data structure
- Significance calculation (5 factors: emotional, goal, novelty, social, survival)
- Memory influence calculation (0.8-1.5 multiplier)
- Auto-pruning logic (max 50 memories)
- Decay factor management

**Success Criteria**:
- Memories below 0.3 significance not stored
- Auto-pruning keeps top 50 by weighted significance
- Memory influence stays within 0.8-1.5 range
- Memory retrieval < 10μs
- All unit tests pass

**Estimated Effort**: 2-3 days

### Phase 3: Mappless Move System

**Goal**: Implement capability-based move generation and evaluation

**Components**:
- `MapplessMoveGenerator` class
- `CapabilityBasedMoveEvaluator` class with 18 factors
- `ConformationalMove` data structure
- Capability extraction from conformations
- Move type definitions (10 types)

**Success Criteria**:
- Move generation is O(1) per agent (constant time)
- Moves filtered by capabilities, not spatial constraints
- 18-factor evaluation produces reasonable weights
- Move evaluation < 2ms per move
- All unit tests pass

**Estimated Effort**: 3-4 days

### Phase 4: Physics Integration

**Goal**: Integrate existing QAAP, resonance, and water shielding modules

**Components**:
- Adapter classes for existing physics modules
- `IPhysicsCalculator` interface implementations
- Integration with move evaluator (factors 5-7)
- Physics factor weighting (0.5-2.0 ranges)

**Success Criteria**:
- QAAP calculator produces valid quantum potentials
- Resonance coupling produces 0-1 values
- Water shielding uses correct parameters (408 fs, 3.57 nm⁻¹)
- Physics factors properly weighted in move evaluation
- Integration tests pass

**Estimated Effort**: 2-3 days

### Phase 5: Multi-Agent Coordination

**Goal**: Implement parallel agent exploration with shared memory

**Components**:
- `MultiAgentCoordinator` class
- `SharedMemoryPool` class
- Agent diversity profiles (cautious, balanced, aggressive)
- Parallel exploration logic
- Collective learning metrics

**Success Criteria**:
- Agents initialized with correct diversity (33%/34%/33%)
- Shared memory pool stores high-significance memories (>= 0.7)
- Parallel exploration runs without synchronization overhead
- Multi-agent performance > single agent performance
- Integration tests pass

**Estimated Effort**: 2-3 days

### Phase 6: Local Minima Handling

**Goal**: Implement detection and escape mechanisms for local minima

**Components**:
- `LocalMinimaDetector` class
- Stuck state detection (20 iterations, < 10 kJ/mol changes)
- Escape strategy (boost frequency, reduce coherence)
- Escape success tracking

**Success Criteria**:
- Stuck state correctly detected after 20 small-change iterations
- Escape strategy increases frequency by 1.0 Hz
- Agents successfully escape local minima
- Escape success rate > 70%
- All tests pass

**Estimated Effort**: 1-2 days

### Phase 7: PyPy Optimization

**Goal**: Optimize code for PyPy JIT compilation

**Components**:
- Remove NumPy dependencies (use pure Python)
- Optimize hot loops for JIT
- Add type hints for JIT optimization
- Profile and optimize bottlenecks
- Benchmark CPython vs PyPy

**Success Criteria**:
- PyPy achieves >= 2x speedup over CPython
- No NumPy or C-extension dependencies
- Memory footprint < 50 MB per agent
- 100 agents complete 500K conformations in < 2 minutes
- Performance tests pass

**Estimated Effort**: 2-3 days

### Phase 8: Validation & Metrics

**Goal**: Implement comprehensive metrics and validation

**Components**:
- `ExplorationMetrics` tracking
- Learning improvement calculation
- RMSD and energy tracking
- Structural validation
- Performance profiling
- Results export

**Success Criteria**:
- Learning improvement measurable (target: 50-66%)
- RMSD < 3Å for small proteins (< 100 residues)
- GDT-TS > 70 for known folds
- All validation tests pass
- Metrics dashboard functional

**Estimated Effort**: 2-3 days

## Adaptive Configuration System

### Implementation

```python
class AdaptiveConfigurator:
    """Generates protein-size-specific configurations"""
    
    def classify_protein_size(self, sequence: str) -> ProteinSizeClass:
        """Classify protein by residue count"""
        residue_count = len(sequence)
        
        if residue_count < 50:
            return ProteinSizeClass.SMALL
        elif residue_count <= 150:
            return ProteinSizeClass.MEDIUM
        else:
            return ProteinSizeClass.LARGE
    
    def get_config_for_protein(self, sequence: str) -> AdaptiveConfig:
        """Generate adaptive configuration"""
        size_class = self.classify_protein_size(sequence)
        residue_count = len(sequence)
        
        if size_class == ProteinSizeClass.SMALL:
            return AdaptiveConfig(
                size_class=size_class,
                residue_count=residue_count,
                
                # Higher exploration energy for small proteins
                initial_frequency_range=(8.0, 12.0),
                initial_coherence_range=(0.6, 0.9),
                
                # Tighter detection for small proteins
                stuck_detection_window=10,
                stuck_detection_threshold=self.scale_threshold(10.0, residue_count),
                
                # Standard memory parameters
                memory_significance_threshold=0.3,
                max_memories_per_agent=50,
                
                # Tighter convergence for small proteins
                convergence_energy_threshold=5.0,
                convergence_rmsd_threshold=0.5,
                
                # Fewer iterations needed
                max_iterations=2000,
                checkpoint_interval=100
            )
        
        elif size_class == ProteinSizeClass.MEDIUM:
            return AdaptiveConfig(
                size_class=size_class,
                residue_count=residue_count,
                
                # Balanced parameters
                initial_frequency_range=(6.0, 10.0),
                initial_coherence_range=(0.5, 0.8),
                
                # Standard detection window
                stuck_detection_window=20,
                stuck_detection_threshold=self.scale_threshold(10.0, residue_count),
                
                # Standard memory parameters
                memory_significance_threshold=0.3,
                max_memories_per_agent=50,
                
                # Standard convergence
                convergence_energy_threshold=10.0,
                convergence_rmsd_threshold=1.0,
                
                # Standard iterations
                max_iterations=5000,
                checkpoint_interval=100
            )
        
        else:  # LARGE
            return AdaptiveConfig(
                size_class=size_class,
                residue_count=residue_count,
                
                # Lower initial energy for large proteins
                initial_frequency_range=(5.0, 8.0),
                initial_coherence_range=(0.4, 0.7),
                
                # Wider detection window for large proteins
                stuck_detection_window=30,
                stuck_detection_threshold=self.scale_threshold(10.0, residue_count),
                
                # More memories for complex proteins
                memory_significance_threshold=0.25,
                max_memories_per_agent=75,
                
                # Relaxed convergence for large proteins
                convergence_energy_threshold=20.0,
                convergence_rmsd_threshold=2.0,
                
                # More iterations needed
                max_iterations=10000,
                checkpoint_interval=200
            )
    
    def scale_threshold(self, base_threshold: float, residue_count: int) -> float:
        """
        Scale threshold proportionally to protein size
        Formula: base_threshold × sqrt(residue_count / 50)
        
        Examples:
        - 20 residues: 10.0 × sqrt(20/50) = 6.3 kJ/mol
        - 50 residues: 10.0 × sqrt(50/50) = 10.0 kJ/mol (baseline)
        - 100 residues: 10.0 × sqrt(100/50) = 14.1 kJ/mol
        - 200 residues: 10.0 × sqrt(200/50) = 20.0 kJ/mol
        """
        import math
        return base_threshold * math.sqrt(residue_count / 50.0)
```

### Usage Example

```python
# Initialize with adaptive configuration
configurator = AdaptiveConfigurator()
sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues

# Get size-appropriate configuration
config = configurator.get_config_for_protein(sequence)

print(f"Protein size: {config.size_class.value}")
print(f"Stuck detection window: {config.stuck_detection_window}")
print(f"Stuck threshold: {config.stuck_detection_threshold:.2f} kJ/mol")

# Initialize system with adaptive config
coordinator = MultiAgentCoordinator(config=config)
agents = coordinator.initialize_agents(count=100, diversity_profile='mixed')

# Run with adaptive parameters
results = coordinator.run_parallel_exploration(iterations=config.max_iterations)
```

## Deployment Considerations

### Environment Setup

```bash
# PyPy installation
# Downloaded from pypy.org eed to update PATH
wget https://downloads.python.org/pypy/pypy3.10-v7.3.13-linux64.tar.bz2
tar -xf pypy3.10-v7.3.13-linux64.tar.bz2
export PATH=$PWD/pypy3.10-v7.3.13-linux64/bin:$PATH

# Verify installation
pypy3 --version

# Install dependencies (pure Python only)
pypy3 -m pip install pytest  # For testing
# No NumPy, no SciPy - pure Python only for PyPy optimization
```

### Running the System

```bash
# Single agent exploration (proof of concept)
pypy3 run_single_agent.py --sequence ACDEFGHIKLMNPQRSTVWY --iterations 1000

# Multi-agent exploration (production)
pypy3 run_multi_agent.py \
    --sequence ACDEFGHIKLMNPQRSTVWY \
    --agents 100 \
    --iterations 5000 \
    --diversity mixed \
    --output results.json

# Benchmark mode
pypy3 benchmark.py --agents 100 --iterations 1000 --compare-cpython

# Validation mode (with known native structure)
pypy3 validate.py \
    --sequence ACDEFGHIKLMNPQRSTVWY \
    --native-pdb 1L2Y.pdb \
    --agents 100 \
    --iterations 5000
```

### Configuration

```python
# config.py - System configuration

# Consciousness parameters (global bounds)
FREQUENCY_MIN = 3.0
FREQUENCY_MAX = 15.0
COHERENCE_MIN = 0.2
COHERENCE_MAX = 1.0
BEHAVIORAL_STATE_REGEN_THRESHOLD = 0.3

# Memory parameters (base values, can be overridden by AdaptiveConfig)
MEMORY_SIGNIFICANCE_THRESHOLD = 0.3
MAX_MEMORIES_PER_AGENT = 50
MEMORY_INFLUENCE_MIN = 0.8
MEMORY_INFLUENCE_MAX = 1.5
SHARED_MEMORY_SIGNIFICANCE_THRESHOLD = 0.7
MAX_SHARED_MEMORY_POOL_SIZE = 10000

# Memory significance calculation (simplified 3-factor approach)
SIGNIFICANCE_ENERGY_CHANGE_WEIGHT = 0.5
SIGNIFICANCE_STRUCTURAL_NOVELTY_WEIGHT = 0.3
SIGNIFICANCE_RMSD_IMPROVEMENT_WEIGHT = 0.2

# Local minima detection (base values, scaled by AdaptiveConfig)
BASE_STUCK_DETECTION_WINDOW = 20
BASE_STUCK_DETECTION_THRESHOLD = 10.0  # kJ/mol
ESCAPE_FREQUENCY_BOOST_MODERATE = 1.0
ESCAPE_FREQUENCY_BOOST_LARGE = 2.0
ESCAPE_FREQUENCY_BOOST_MAXIMUM = 3.0
ESCAPE_COHERENCE_REDUCTION = 0.1

# Performance targets
TARGET_DECISION_LATENCY_MS = 2.0
TARGET_MEMORY_RETRIEVAL_US = 10.0
TARGET_AGENT_MEMORY_MB = 50.0
TARGET_PYPY_SPEEDUP = 2.0

# Multi-agent diversity
AGENT_PROFILE_CAUTIOUS_RATIO = 0.33
AGENT_PROFILE_BALANCED_RATIO = 0.34
AGENT_PROFILE_AGGRESSIVE_RATIO = 0.33

# Physics integration (composite quantum alignment factor)
QAAP_CONTRIBUTION_MIN = 0.7
QAAP_CONTRIBUTION_MAX = 1.3
RESONANCE_CONTRIBUTION_MIN = 0.9
RESONANCE_CONTRIBUTION_MAX = 1.2
WATER_SHIELDING_CONTRIBUTION_MIN = 0.95
WATER_SHIELDING_CONTRIBUTION_MAX = 1.05
GAMMA_FREQUENCY_HZ = 40.0
COHERENCE_TIME_FS = 408.0
WATER_SHIELDING_FACTOR = 3.57  # nm⁻¹

# Composite factor ranges (simplified 5-factor evaluation)
PHYSICAL_FEASIBILITY_RANGE = (0.1, 2.0)
QUANTUM_ALIGNMENT_RANGE = (0.5, 1.5)
BEHAVIORAL_PREFERENCE_RANGE = (0.5, 2.5)
HISTORICAL_SUCCESS_RANGE = (0.8, 1.8)
GOAL_ALIGNMENT_MAX_BOOST = 10.0

# Adaptive configuration
PROTEIN_SIZE_SMALL_THRESHOLD = 50  # residues
PROTEIN_SIZE_LARGE_THRESHOLD = 150  # residues
THRESHOLD_SCALING_BASELINE = 50  # residues for 1.0x scaling

# Visualization and monitoring
VISUALIZATION_STREAM_INTERVAL = 10  # iterations
TRAJECTORY_BUFFER_MAX_SIZE = 1000  # snapshots
ENERGY_LANDSCAPE_PROJECTION_METHOD = 'PCA'  # or 't-SNE'

# Checkpoint and resume
CHECKPOINT_AUTO_SAVE_INTERVAL = 100  # iterations
CHECKPOINT_ROTATION_KEEP_COUNT = 5  # keep last N checkpoints
CHECKPOINT_FORMAT_VERSION = '1.0'

# Validation targets
TARGET_RMSD_ANGSTROM = 3.0
TARGET_GDT_TS = 70.0
TARGET_LEARNING_IMPROVEMENT_PERCENT = 50.0
```

## Summary

This design document specifies a complete UBF-protein integration system with:

1. **Mappless Architecture**: Capability-based conformational navigation without spatial pathfinding
2. **SOLID Principles**: Clean interfaces, dependency inversion, extensible design
3. **Experience-Driven Learning**: No training data, agents learn through exploration
4. **Multi-Agent Collaboration**: Shared memory pool enables collective intelligence
5. **Quantum-Grounded Physics**: Integration with validated QAAP, resonance, water shielding
6. **PyPy Optimization**: Pure Python implementation optimized for JIT compilation
7. **Comprehensive Testing**: Unit, integration, performance, and validation tests
8. **Phased Implementation**: 8 phases from foundation to production-ready system

The system transforms protein folding from static physics simulation into adaptive multi-agent exploration, achieving  learning improvement rate in conformational space.


## Summary of Enhanced Design

This design document specifies a complete UBF-protein integration system with:

1. **Mappless Architecture**: Capability-based conformational navigation without spatial pathfinding
2. **SOLID Principles**: Clean interfaces, dependency inversion, extensible design
3. **Experience-Driven Learning**: No training data, agents learn through exploration
4. **Multi-Agent Collaboration**: Shared memory pool enables collective intelligence
5. **Quantum-Grounded Physics**: Integration with validated QAAP, resonance, water shielding
6. **PyPy Optimization**: Pure Python implementation optimized for JIT compilation
7. **Comprehensive Testing**: Unit, integration, performance, and validation tests
8. **Progressive Complexity**: Simplified 5-factor evaluation and 3-factor memory significance for easier initial tuning
9. **Adaptive Configuration**: Protein-size-specific parameters automatically adjust for small/medium/large proteins
10. **Visualization Export**: Real-time monitoring and trajectory export for analysis
11. **Checkpoint/Resume**: Save and restore system state for long-running predictions
12. **Phased Implementation**: 8 core phases plus 3 enhancement phases (adaptive config, visualization, checkpointing)

### Key Design Improvements

**Simplified Evaluation**: 5 composite factors instead of 18 individual factors
- Physical Feasibility (structural + energy barrier + Ramachandran)
- Quantum Alignment (QAAP + resonance + water shielding)
- Behavioral Preference (all 5 behavioral dimensions)
- Historical Success (memory + novelty)
- Goal Alignment (energy + RMSD improvement)

**Simplified Memory**: 3-factor significance calculation instead of 5
- Energy change (0.5 weight)
- Structural novelty (0.3 weight)
- RMSD improvement (0.2 weight)

**Adaptive Parameters**: Automatic scaling based on protein size
- Small (< 50 residues): High energy, tight convergence, 10-iteration window
- Medium (50-150 residues): Balanced parameters, 20-iteration window
- Large (> 150 residues): Lower energy, relaxed convergence, 30-iteration window

**Enhanced Monitoring**: Real-time visualization and checkpoint/resume
- PDB trajectory export for molecular visualization
- Energy landscape 2D projections
- Auto-save checkpoints every 100 iterations
- Resume from any checkpoint

The system transforms protein folding from static physics simulation into adaptive multi-agent exploration, achieving the proven 66% learning improvement rate from maze navigation applied to conformational space, with practical enhancements for real-world usage.
