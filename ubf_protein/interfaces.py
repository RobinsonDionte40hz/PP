from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .models import (
        Conformation, ConformationalMove, ConformationalMemory, ConformationalOutcome,
        ConsciousnessCoordinates, BehavioralStateData, ExplorationResults, EnergyLandscape,
        ConformationSnapshot, SystemCheckpoint, AdaptiveConfig, ProteinSizeClass
    )

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

class IMemorySystem(ABC):
    """Interface for experience memory management"""
    
    @abstractmethod
    def store_memory(self, memory: 'ConformationalMemory') -> None:
        """Store memory if significance >= 0.3, auto-prune if > 50"""
        pass
    
    @abstractmethod
    def retrieve_relevant_memories(self, move_type: str, max_count: int = 10) -> List['ConformationalMemory']:
        """Retrieve relevant memories for move evaluation"""
        pass
    
    @abstractmethod
    def calculate_memory_influence(self, move_type: str) -> float:
        """Calculate memory influence multiplier (0.8-1.5)"""
        pass# ============================================================================
# Conformational Moves (Mappless Design)
# ============================================================================

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

    @abstractmethod
    def get_exploration_metrics(self) -> Dict[str, float]:
        """Get current exploration metrics"""
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

    @abstractmethod
    def get_total_memories(self) -> int:
        """Get total number of memories in the pool"""
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

class IVisualizationExporter(ABC):
    """Interface for exporting visualization data"""

    @abstractmethod
    def export_trajectory(self, agent_id: str) -> List['ConformationSnapshot']:
        """Export complete trajectory for an agent"""
        pass

    @abstractmethod
    def export_energy_landscape(self) -> 'EnergyLandscape':
        """Export 2D projection of explored conformational space"""
        pass

    @abstractmethod
    def stream_update(self, snapshot: 'ConformationSnapshot') -> None:
        """Stream real-time update (non-blocking)"""
        pass

# ============================================================================
# Checkpoint & Resume
# ============================================================================

class ICheckpointManager(ABC):
    """Interface for checkpoint and resume functionality"""

    @abstractmethod
    def save_checkpoint(self,
                       agents: List[IProteinAgent],
                       shared_pool: ISharedMemoryPool,
                       iteration: int,
                       metadata: Dict[str, Any]) -> str:
        """Save complete system state, returns checkpoint file path"""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> 'SystemCheckpoint':
        """Load checkpoint from file"""
        pass

    @abstractmethod
    def restore_agents(self, checkpoint: 'SystemCheckpoint') -> Tuple[List[IProteinAgent], ISharedMemoryPool, int]:
        """Restore agents and shared pool from checkpoint, returns (agents, pool, iteration)"""
        pass

# ============================================================================
# Adaptive Configuration
# ============================================================================

class IAdaptiveConfigurator(ABC):
    """Interface for adaptive configuration based on protein properties"""

    @abstractmethod
    def get_config_for_protein(self, sequence: str) -> 'AdaptiveConfig':
        """Generate adaptive configuration based on protein sequence"""
        pass

    @abstractmethod
    def classify_protein_size(self, sequence: str) -> 'ProteinSizeClass':
        """Classify protein as small, medium, or large"""
        pass

    @abstractmethod
    def scale_threshold(self, base_threshold: float, residue_count: int) -> float:
        """Scale threshold proportionally to protein size"""
        pass