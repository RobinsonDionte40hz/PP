# UBF Protein System - API Documentation

Complete API reference for the UBF Protein System, organized by component.

## Table of Contents

- [Design Principles](#design-principles)
- [Core Interfaces](#core-interfaces)
- [Data Models](#data-models)
- [Consciousness System](#consciousness-system)
- [Behavioral State](#behavioral-state)
- [Memory System](#memory-system)
- [Move Generation](#move-generation)
- [Protein Agent](#protein-agent)
- [Multi-Agent Coordinator](#multi-agent-coordinator)
- [Physics Integration](#physics-integration)
- [Validation & Error Handling](#validation--error-handling)
- [Checkpoint System](#checkpoint-system)
- [Visualization](#visualization)

---

## Design Principles

### SOLID Principles

The UBF Protein System follows SOLID design principles for maintainability and testability:

1. **Single Responsibility**: Each class has one clear purpose
2. **Open/Closed**: Open for extension, closed for modification (via interfaces)
3. **Liskov Substitution**: Interfaces define contracts that implementations must fulfill
4. **Interface Segregation**: Focused interfaces for specific functionality
5. **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations

### Mapless Design

The system uses **mapless navigation** for conformational exploration:

- **No Spatial Maps**: No explicit conformational space maps or transition graphs
- **No Pathfinding**: No A* or Dijkstra algorithms for move planning
- **Capability-Based**: Moves generated based on current conformation capabilities
- **O(1) Performance**: Move generation is constant-time per agent, regardless of space size
- **Dynamic Evaluation**: Each move evaluated independently based on current context

This approach scales to infinite conformational spaces and avoids the "curse of dimensionality" in high-dimensional protein folding.

---

## Core Interfaces

### IProteinAgent

```python
from abc import ABC, abstractmethod
from typing import Optional

class IProteinAgent(ABC):
    """
    Interface for autonomous protein folding agent.
    
    An agent explores conformational space using consciousness-based navigation,
    learning from experience, and adapting to local minima.
    """
    
    @abstractmethod
    def explore_step(self) -> ConformationalOutcome:
        """
        Execute single exploration step in conformational space.
        
        Returns:
            ConformationalOutcome with energy change, RMSD change, and success flag
        """
        pass
    
    @abstractmethod
    def get_consciousness_state(self) -> 'IConsciousnessState':
        """Get current consciousness coordinates."""
        pass
    
    @abstractmethod
    def get_behavioral_state(self) -> 'IBehavioralState':
        """Get current behavioral state derived from consciousness."""
        pass
    
    @abstractmethod
    def get_memory_system(self) -> 'IMemorySystem':
        """Get agent's experience memory system."""
        pass
    
    @abstractmethod
    def get_current_conformation(self) -> Conformation:
        """Get current protein conformation."""
        pass
    
    @abstractmethod
    def get_agent_id(self) -> str:
        """Get unique agent identifier."""
        pass
```

### IConsciousnessState

```python
class IConsciousnessState(ABC):
    """
    Interface for consciousness coordinate system.
    
    Consciousness is represented as a 2D coordinate system:
    - Frequency: 3-15 Hz (exploration tempo)
    - Coherence: 0.2-1.0 (behavioral consistency)
    """
    
    @abstractmethod
    def get_frequency(self) -> float:
        """
        Get consciousness frequency (3-15 Hz).
        
        Higher frequency = faster, more aggressive exploration
        Lower frequency = slower, more cautious exploration
        """
        pass
    
    @abstractmethod
    def get_coherence(self) -> float:
        """
        Get consciousness coherence (0.2-1.0).
        
        Higher coherence = consistent, focused behavior
        Lower coherence = variable, exploratory behavior
        """
        pass
    
    @abstractmethod
    def update_from_outcome(self, outcome: ConformationalOutcome) -> None:
        """
        Update consciousness coordinates based on exploration outcome.
        
        Success increases frequency and coherence
        Failure decreases frequency and coherence
        Stuck state triggers escape strategy
        
        Args:
            outcome: Result of exploration step
        """
        pass
    
    @abstractmethod
    def get_coordinates(self) -> ConsciousnessCoordinates:
        """Get immutable consciousness coordinates."""
        pass
```

### IBehavioralState

```python
class IBehavioralState(ABC):
    """
    Interface for behavioral state derived from consciousness.
    
    Behavioral state consists of 5 dimensions:
    1. Exploration energy (0-1)
    2. Structural focus (0-1)
    3. Hydrophobic drive (0-1)
    4. Risk tolerance (0-1)
    5. Native state ambition (0-1)
    """
    
    @abstractmethod
    def get_exploration_energy(self) -> float:
        """
        Get exploration energy (0-1).
        
        Derived from frequency: higher frequency = higher energy
        Influences: Move magnitude, search radius
        """
        pass
    
    @abstractmethod
    def get_structural_focus(self) -> float:
        """
        Get structural focus (0-1).
        
        Derived from coherence: higher coherence = higher focus
        Influences: Secondary structure formation bias
        """
        pass
    
    @abstractmethod
    def get_hydrophobic_drive(self) -> float:
        """
        Get hydrophobic drive (0-1).
        
        Derived from frequency × coherence
        Influences: Hydrophobic collapse bias
        """
        pass
    
    @abstractmethod
    def get_risk_tolerance(self) -> float:
        """
        Get risk tolerance (0-1).
        
        Derived from frequency, inverse of coherence
        Influences: Large jump acceptance, energy barrier tolerance
        """
        pass
    
    @abstractmethod
    def get_native_state_ambition(self) -> float:
        """
        Get native state ambition (0-1).
        
        Derived from coherence
        Influences: Goal-directed vs exploratory balance
        """
        pass
    
    @abstractmethod
    def get_behavioral_data(self) -> BehavioralStateData:
        """Get immutable behavioral state data."""
        pass
```

### IMemorySystem

```python
class IMemorySystem(ABC):
    """
    Interface for experience memory management.
    
    Stores significant conformational transitions (significance ≥ 0.3)
    and provides memory-based influence for move selection.
    """
    
    @abstractmethod
    def store_memory(self, memory: ConformationalMemory) -> None:
        """
        Store significant conformational memory.
        
        Only memories with significance ≥ 0.3 are stored.
        Auto-prunes when memory count exceeds MAX_MEMORIES_PER_AGENT.
        
        Args:
            memory: Conformational memory to store
        """
        pass
    
    @abstractmethod
    def retrieve_relevant_memories(
        self,
        move_type: str,
        max_count: int = 10
    ) -> List[ConformationalMemory]:
        """
        Retrieve relevant memories for move evaluation.
        
        Returns memories for given move type, sorted by influence weight
        (significance × age_factor).
        
        Args:
            move_type: Type of move being evaluated
            max_count: Maximum memories to return
            
        Returns:
            List of relevant memories, most influential first
        """
        pass
    
    @abstractmethod
    def calculate_memory_influence(
        self,
        memories: List[ConformationalMemory]
    ) -> float:
        """
        Calculate memory influence multiplier for move evaluation.
        
        Returns value in range [0.8, 1.5]:
        - 0.8-1.0: Memories suggest move is risky
        - 1.0: Neutral (no strong memory signal)
        - 1.0-1.5: Memories suggest move is promising
        
        Args:
            memories: Relevant memories for current context
            
        Returns:
            Influence multiplier for move weight
        """
        pass
```

### ISharedMemoryPool

```python
class ISharedMemoryPool(ABC):
    """
    Interface for shared memory pool across agents.
    
    Stores high-significance memories (≥ 0.7) that can be shared
    between agents for collective learning.
    """
    
    @abstractmethod
    def share_memory(self, memory: ConformationalMemory) -> None:
        """
        Share high-significance memory with all agents.
        
        Only memories with significance ≥ 0.7 are shared.
        Auto-prunes when pool exceeds MAX_SHARED_MEMORY_POOL_SIZE.
        
        Args:
            memory: Memory to share
        """
        pass
    
    @abstractmethod
    def retrieve_shared_memories(
        self,
        move_type: str,
        max_count: int = 10
    ) -> List[ConformationalMemory]:
        """
        Retrieve relevant shared memories.
        
        Args:
            move_type: Type of move being evaluated
            max_count: Maximum memories to return
            
        Returns:
            List of relevant shared memories
        """
        pass
```

### IMultiAgentCoordinator

```python
class IMultiAgentCoordinator(ABC):
    """
    Interface for multi-agent coordination system.
    
    Manages population of agents with diverse consciousness profiles,
    coordinates parallel exploration, and aggregates results.
    """
    
    @abstractmethod
    def initialize_agents(
        self,
        count: int,
        diversity_profile: str = "balanced"
    ) -> List[IProteinAgent]:
        """
        Initialize diverse agent population.
        
        Diversity profiles:
        - "cautious": 100% low-frequency, high-coherence agents
        - "balanced": 33% cautious, 34% balanced, 33% aggressive
        - "aggressive": 100% high-frequency, low-coherence agents
        
        Args:
            count: Number of agents to create
            diversity_profile: Distribution of agent types
            
        Returns:
            List of initialized agents
        """
        pass
    
    @abstractmethod
    def run_parallel_exploration(
        self,
        iterations: int
    ) -> ExplorationResults:
        """
        Run all agents in parallel for N iterations.
        
        Agents explore independently, sharing high-significance
        memories (≥ 0.7) to collective pool after each iteration.
        
        Args:
            iterations: Number of exploration steps per agent
            
        Returns:
            ExplorationResults with best conformation and metrics
        """
        pass
    
    @abstractmethod
    def get_best_conformation(self) -> Conformation:
        """
        Get best conformation found across all agents.
        
        Returns conformation with lowest energy or RMSD.
        """
        pass
```

---

## Data Models

### ConsciousnessCoordinates

```python
@dataclass(frozen=True)
class ConsciousnessCoordinates:
    """
    Immutable consciousness coordinate data.
    
    Attributes:
        frequency: Consciousness frequency (3-15 Hz)
        coherence: Consciousness coherence (0.2-1.0)
        last_update_timestamp: Timestamp of last update (milliseconds)
    """
    frequency: float
    coherence: float
    last_update_timestamp: int
    
    def __post_init__(self):
        """Validate coordinate bounds."""
        assert 3.0 <= self.frequency <= 15.0, "Frequency must be 3-15 Hz"
        assert 0.2 <= self.coherence <= 1.0, "Coherence must be 0.2-1.0"
```

### BehavioralStateData

```python
@dataclass(frozen=True)
class BehavioralStateData:
    """
    Immutable behavioral state data.
    
    Attributes:
        exploration_energy: Exploration energy (0-1)
        structural_focus: Structural focus (0-1)
        hydrophobic_drive: Hydrophobic collapse drive (0-1)
        risk_tolerance: Risk tolerance (0-1)
        native_state_ambition: Goal-directed ambition (0-1)
        timestamp: Creation timestamp (milliseconds)
    """
    exploration_energy: float
    structural_focus: float
    hydrophobic_drive: float
    risk_tolerance: float
    native_state_ambition: float
    timestamp: int
```

### ConformationalMemory

```python
@dataclass
class ConformationalMemory:
    """
    Memory of significant conformational transition.
    
    Attributes:
        memory_id: Unique memory identifier
        move_type: Type of move that created this memory
        significance: Memory significance (0-1)
        energy_change: Energy change from move (kcal/mol)
        rmsd_change: RMSD change from move (Angstroms)
        success: Whether move was successful
        timestamp: Creation timestamp (milliseconds)
        consciousness_state: Consciousness coordinates at time of memory
        behavioral_state: Behavioral state at time of memory
    """
    memory_id: str
    move_type: str
    significance: float
    energy_change: float
    rmsd_change: float
    success: bool
    timestamp: int
    consciousness_state: ConsciousnessCoordinates
    behavioral_state: BehavioralStateData
    
    def get_influence_weight(self) -> float:
        """
        Calculate influence weight for move evaluation.
        
        Combines significance with age decay:
        weight = significance × (1 - age_hours / 24)
        
        Returns:
            Influence weight (0-1)
        """
        pass
```

### Conformation

```python
@dataclass
class Conformation:
    """
    Protein conformation representation.
    
    Attributes:
        conformation_id: Unique identifier
        sequence: Amino acid sequence
        atom_coordinates: 3D coordinates for all atoms
        secondary_structure: Secondary structure assignment per residue
        energy: Total potential energy (kcal/mol)
        rmsd: RMSD from native structure (Angstroms)
        phi_angles: Backbone phi angles (degrees)
        psi_angles: Backbone psi angles (degrees)
    """
    conformation_id: str
    sequence: str
    atom_coordinates: List[List[float]]  # [atom_index][x, y, z]
    secondary_structure: List[str]       # ['H', 'E', 'C', ...]
    energy: float
    rmsd: float
    phi_angles: List[float]
    psi_angles: List[float]
```

---

## Consciousness System

### ConsciousnessState

```python
class ConsciousnessState(IConsciousnessState):
    """
    Implementation of consciousness coordinate system.
    
    Manages 2D consciousness coordinates (frequency, coherence)
    and updates based on exploration outcomes.
    """
    
    def __init__(
        self,
        initial_frequency: float = 9.0,
        initial_coherence: float = 0.6
    ):
        """
        Initialize consciousness state.
        
        Args:
            initial_frequency: Starting frequency (3-15 Hz)
            initial_coherence: Starting coherence (0.2-1.0)
        """
        pass
    
    def update_from_outcome(self, outcome: ConformationalOutcome) -> None:
        """
        Update consciousness based on outcome type.
        
        Update rules:
        - Success: +0.5 Hz frequency, +0.05 coherence
        - Minor success: +0.2 Hz frequency, +0.02 coherence
        - Failure: -0.3 Hz frequency, -0.03 coherence
        - Stuck: -0.5 Hz frequency, -0.05 coherence
        
        Bounds enforced: frequency [3, 15], coherence [0.2, 1.0]
        """
        pass
```

### Usage Example

```python
from ubf_protein.consciousness import ConsciousnessState

# Create consciousness state
consciousness = ConsciousnessState(
    initial_frequency=9.0,
    initial_coherence=0.6
)

# Update from success
consciousness.update_from_outcome(
    ConformationalOutcome(
        energy_change=-5.0,
        rmsd_change=-0.5,
        success=True
    )
)

# Get updated coordinates
coords = consciousness.get_coordinates()
print(f"Frequency: {coords.frequency:.2f} Hz")
print(f"Coherence: {coords.coherence:.2f}")
```

---

## Behavioral State

### BehavioralState

```python
class BehavioralState(IBehavioralState):
    """
    Implementation of behavioral state derived from consciousness.
    
    Calculates 5 behavioral dimensions from consciousness coordinates
    using mathematical transformations.
    """
    
    def __init__(self, consciousness_coords: ConsciousnessCoordinates):
        """
        Initialize behavioral state from consciousness.
        
        Args:
            consciousness_coords: Current consciousness coordinates
        """
        pass
    
    @staticmethod
    def from_consciousness(
        consciousness_coords: ConsciousnessCoordinates
    ) -> 'BehavioralState':
        """
        Create behavioral state from consciousness coordinates.
        
        Derivation formulas:
        - exploration_energy = (freq - 3) / 12
        - structural_focus = coherence
        - hydrophobic_drive = sqrt(exploration_energy × structural_focus)
        - risk_tolerance = exploration_energy × (1 - coherence)
        - native_state_ambition = coherence × 0.8 + 0.2
        """
        pass
    
    def should_regenerate(
        self,
        current_coords: ConsciousnessCoordinates
    ) -> bool:
        """
        Check if behavioral state should regenerate.
        
        Regenerates if consciousness has changed significantly
        (Euclidean distance > 0.3 in normalized space).
        
        Args:
            current_coords: Current consciousness coordinates
            
        Returns:
            True if regeneration needed
        """
        pass
```

### Usage Example

```python
from ubf_protein.behavioral_state import BehavioralState
from ubf_protein.models import ConsciousnessCoordinates

# Create from consciousness
coords = ConsciousnessCoordinates(
    frequency=9.0,
    coherence=0.6,
    last_update_timestamp=1234567890
)

behavioral = BehavioralState.from_consciousness(coords)

# Access behavioral dimensions
print(f"Exploration energy: {behavioral.get_exploration_energy():.2f}")
print(f"Structural focus: {behavioral.get_structural_focus():.2f}")
print(f"Hydrophobic drive: {behavioral.get_hydrophobic_drive():.2f}")
print(f"Risk tolerance: {behavioral.get_risk_tolerance():.2f}")
print(f"Native state ambition: {behavioral.get_native_state_ambition():.2f}")
```

---

## Memory System

### MemorySystem

```python
class MemorySystem(IMemorySystem):
    """
    Implementation of experience memory management.
    
    Stores significant conformational transitions and provides
    memory-based influence for move evaluation.
    """
    
    def __init__(self):
        """Initialize empty memory system."""
        pass
    
    def store_memory(self, memory: ConformationalMemory) -> None:
        """
        Store memory if significance ≥ 0.3.
        
        Auto-prunes least influential memories if count exceeds
        MAX_MEMORIES_PER_AGENT (default 50).
        
        Args:
            memory: Memory to store
        """
        pass
    
    def calculate_significance(
        self,
        outcome: ConformationalOutcome
    ) -> float:
        """
        Calculate memory significance from outcome.
        
        3-factor calculation:
        - Energy change impact (weight 0.5)
        - Structural novelty (weight 0.3)
        - RMSD improvement (weight 0.2)
        
        Returns value in range [0, 1].
        
        Args:
            outcome: Conformational exploration outcome
            
        Returns:
            Significance score (0-1)
        """
        pass
```

### SharedMemoryPool

```python
class SharedMemoryPool(ISharedMemoryPool):
    """
    Implementation of shared memory pool.
    
    Stores high-significance memories (≥ 0.7) for collective learning.
    """
    
    def __init__(self):
        """Initialize empty shared pool."""
        pass
    
    def share_memory(self, memory: ConformationalMemory) -> None:
        """
        Share memory if significance ≥ 0.7.
        
        Auto-prunes least influential memories if count exceeds
        MAX_SHARED_MEMORY_POOL_SIZE (default 10000).
        """
        pass
    
    def prune_pool(self, max_size: int = 10000) -> None:
        """
        Prune pool to max size by influence weight.
        
        Removes least influential memories first.
        
        Args:
            max_size: Maximum pool size
        """
        pass
```

### Usage Example

```python
from ubf_protein.memory_system import MemorySystem, SharedMemoryPool
from ubf_protein.models import ConformationalMemory, ConformationalOutcome

# Create memory system
memory = MemorySystem()

# Create and store significant memory
outcome = ConformationalOutcome(
    energy_change=-10.0,
    rmsd_change=-1.0,
    success=True
)

significance = memory.calculate_significance(outcome)
if significance >= 0.3:
    memory.store_memory(
        ConformationalMemory(
            memory_id="mem_001",
            move_type="helix_formation",
            significance=significance,
            energy_change=outcome.energy_change,
            rmsd_change=outcome.rmsd_change,
            success=outcome.success,
            timestamp=time.time_ns() // 1_000_000,
            consciousness_state=consciousness_coords,
            behavioral_state=behavioral_data
        )
    )

# Retrieve relevant memories
relevant = memory.retrieve_relevant_memories("helix_formation", max_count=10)

# Calculate memory influence
influence = memory.calculate_memory_influence(relevant)
print(f"Memory influence multiplier: {influence:.2f}")
```

---

## Move Generation

### Mapless Move Generator

```python
class MapplessMoveGenerator(IMoveGenerator):
    """
    Mapless move generator using capability-based filtering.
    
    Generates moves in O(1) time without spatial maps or pathfinding.
    Moves are filtered based on current conformation capabilities.
    """
    
    def generate_moves(
        self,
        conformation: Conformation,
        behavioral_state: BehavioralStateData
    ) -> List[ConformationalMove]:
        """
        Generate available moves for current conformation.
        
        Move types:
        1. backbone_rotation: Rotate backbone dihedral angles
        2. sidechain_adjust: Adjust sidechain rotamers
        3. helix_formation: Form alpha-helix structure
        4. sheet_formation: Form beta-sheet structure
        5. turn_formation: Form turn/loop structure
        6. hydrophobic_collapse: Collapse hydrophobic core
        7. salt_bridge: Form salt bridge interaction
        8. disulfide_bond: Form disulfide bond (Cys-Cys)
        9. energy_minimization: Local energy minimization
        10. large_jump: Large conformational change
        
        Filtering based on capabilities:
        - can_form_helix: Has sufficient helix-forming residues
        - can_form_sheet: Has sufficient strand-forming residues
        - can_hydrophobic_collapse: Has hydrophobic residues
        - can_large_rotation: Has flexible regions
        - has_flexible_loops: Has loop regions for turns
        
        Args:
            conformation: Current conformation
            behavioral_state: Current behavioral state
            
        Returns:
            List of feasible moves (typically 5-15 moves)
        """
        pass
```

### Capability-Based Move Evaluator

```python
class CapabilityBasedMoveEvaluator(IMoveEvaluator):
    """
    Evaluates moves using 5 composite factors.
    
    Composite factors:
    1. Physical Feasibility (structural + energy + Ramachandran)
    2. Quantum Alignment (QAAP + resonance + water shielding)
    3. Behavioral Preference (all 5 behavioral dimensions)
    4. Historical Success (memory influence + novelty)
    5. Goal Alignment (energy decrease + RMSD improvement)
    
    Final weight = product of all factors × temperature factor
    """
    
    def evaluate_move(
        self,
        move: ConformationalMove,
        conformation: Conformation,
        behavioral_state: BehavioralStateData,
        memory_influence: float,
        temperature: float = 1.0
    ) -> float:
        """
        Evaluate move and return selection weight.
        
        Args:
            move: Move to evaluate
            conformation: Current conformation
            behavioral_state: Current behavioral state
            memory_influence: Memory influence multiplier (0.8-1.5)
            temperature: Selection temperature (higher = more exploration)
            
        Returns:
            Move selection weight (0-1000+)
        """
        pass
    
    def _evaluate_physical_feasibility(
        self,
        move: ConformationalMove,
        conformation: Conformation
    ) -> float:
        """
        Composite Factor 1: Physical Feasibility.
        
        Combines:
        - Structural feasibility: Can move be executed?
        - Energy barrier: Is energy change reasonable?
        - Ramachandran: Are phi/psi angles allowed?
        
        Returns:
            Feasibility score (0.1-1.0)
        """
        pass
    
    def _evaluate_quantum_alignment(
        self,
        move: ConformationalMove,
        conformation: Conformation
    ) -> float:
        """
        Composite Factor 2: Quantum Alignment.
        
        Combines:
        - QAAP: Quantum coherence potential (0.7-1.3)
        - Resonance: 40 Hz gamma coupling (0.9-1.2)
        - Water shielding: 408 fs coherence, 3.57 nm⁻¹ (0.95-1.05)
        
        Returns:
            Quantum alignment score (0.5-1.5)
        """
        pass
```

### Usage Example

```python
from ubf_protein.mapless_moves import MapplessMoveGenerator, CapabilityBasedMoveEvaluator

# Create move generator and evaluator
generator = MapplessMoveGenerator()
evaluator = CapabilityBasedMoveEvaluator()

# Generate moves
moves = generator.generate_moves(conformation, behavioral_state)
print(f"Generated {len(moves)} feasible moves")

# Evaluate each move
move_weights = []
for move in moves:
    weight = evaluator.evaluate_move(
        move=move,
        conformation=conformation,
        behavioral_state=behavioral_state,
        memory_influence=1.2,  # Positive memory signal
        temperature=1.0
    )
    move_weights.append((move, weight))

# Select move (temperature-based selection)
selected_move = select_move_by_weight(move_weights, temperature=1.0)
```

---

## Checkpoint System

### CheckpointManager

```python
class CheckpointManager(ICheckpointManager):
    """
    Checkpoint and resume system for exploration state.
    
    Features:
    - Full agent state serialization (consciousness, memory, metrics)
    - SHA256 integrity checking
    - Checkpoint rotation (keep N most recent)
    - Version-controlled format
    - Graceful error handling
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        pass
    
    def save_checkpoint(
        self,
        agents: List[IProteinAgent],
        shared_pool: ISharedMemoryPool,
        iteration: int,
        metadata: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save complete system state to checkpoint file.
        
        Checkpoint format (JSON):
        {
            "version": "1.0",
            "timestamp": 1234567890,
            "iteration": 100,
            "metadata": {...},
            "agents": [...],
            "shared_pool": {...},
            "integrity_hash": "sha256..."
        }
        
        Args:
            agents: List of agents to save
            shared_pool: Shared memory pool
            iteration: Current iteration number
            metadata: Additional metadata (sequence, config, etc.)
            checkpoint_name: Optional custom name
            
        Returns:
            Path to saved checkpoint file
        """
        pass
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load checkpoint file with validation.
        
        Validates:
        - File exists and is readable
        - JSON format is valid
        - Version is compatible
        - Integrity hash matches
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
            
        Raises:
            ValueError: If checkpoint is invalid or corrupted
        """
        pass
    
    def restore_agents(
        self,
        checkpoint_data: Dict[str, Any],
        agent_class: type
    ) -> Tuple[List[IProteinAgent], ISharedMemoryPool, int]:
        """
        Restore agents from checkpoint data.
        
        Reconstructs:
        - Agent consciousness states
        - Agent behavioral states
        - Agent memory systems
        - Agent conformations
        - Agent metrics
        - Shared memory pool
        
        Args:
            checkpoint_data: Loaded checkpoint data
            agent_class: Class to instantiate (e.g., ProteinAgent)
            
        Returns:
            Tuple of (agents, shared_pool, iteration)
        """
        pass
    
    def set_auto_save_interval(self, interval: int) -> None:
        """
        Set auto-save interval.
        
        Args:
            interval: Save checkpoint every N iterations
        """
        pass
    
    def should_auto_save(self, iteration: int) -> bool:
        """
        Check if auto-save should trigger.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            True if checkpoint should be saved
        """
        pass
```

### Usage Example

```python
from ubf_protein.checkpoint import CheckpointManager
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

# Create coordinator with checkpointing
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints"
)

# Set auto-save every 50 iterations
coordinator._checkpoint_manager.set_auto_save_interval(50)

# Run with auto-save
coordinator.initialize_agents(count=10)
coordinator.run_parallel_exploration(iterations=200)

# Resume from checkpoint
coordinator2 = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints"
)
coordinator2.resume_from_checkpoint()  # Loads latest checkpoint
coordinator2.run_parallel_exploration(iterations=200)  # Continue from iteration 200
```

---

## Visualization

### VisualizationExporter

```python
class VisualizationExporter(IVisualizationExporter):
    """
    Export system for trajectories and energy landscapes.
    
    Supports multiple formats:
    - JSON: Full trajectory with metadata
    - PDB: Molecular visualization format
    - CSV: Energy landscape for plotting
    """
    
    def __init__(self, output_dir: str = "./viz"):
        """
        Initialize visualization exporter.
        
        Args:
            output_dir: Directory for output files
        """
        pass
    
    def export_trajectory_to_json(
        self,
        snapshots: List[ConformationSnapshot],
        output_file: str
    ) -> None:
        """
        Export trajectory to JSON format.
        
        JSON structure:
        {
            "trajectory": [
                {
                    "iteration": 0,
                    "timestamp": 1234567890,
                    "conformation_id": "conf_001",
                    "energy": -123.45,
                    "rmsd": 2.5,
                    "consciousness": {...},
                    "behavioral": {...}
                },
                ...
            ]
        }
        
        Args:
            snapshots: List of conformation snapshots
            output_file: Output file path
        """
        pass
    
    def export_energy_landscape(
        self,
        snapshots: List[ConformationSnapshot],
        output_file: str,
        projection_method: str = "pca"
    ) -> None:
        """
        Export 2D energy landscape projection.
        
        Uses dimensionality reduction (PCA or t-SNE) to project
        high-dimensional conformational space to 2D.
        
        CSV format:
        iteration,x,y,energy,rmsd,frequency,coherence
        0,0.5,0.3,-100.0,3.0,9.0,0.6
        ...
        
        Args:
            snapshots: List of conformation snapshots
            output_file: Output CSV file path
            projection_method: "pca" or "tsne"
        """
        pass
    
    def stream_update(
        self,
        snapshot: ConformationSnapshot,
        stream_file: str
    ) -> None:
        """
        Stream real-time update to file.
        
        Non-blocking append to file for real-time monitoring.
        
        Args:
            snapshot: Current conformation snapshot
            stream_file: Stream file path
        """
        pass
```

### Usage Example

```python
from ubf_protein.visualization import VisualizationExporter
from ubf_protein.protein_agent import ProteinAgent

# Create agent with visualization
agent = ProteinAgent(
    protein_sequence="ACDEFGH",
    enable_visualization=True,
    max_snapshots=1000
)

# Run exploration
for i in range(500):
    agent.explore_step()

# Export trajectory
exporter = VisualizationExporter(output_dir="./viz")
exporter.export_trajectory_to_json(
    snapshots=agent.get_trajectory_snapshots(),
    output_file="trajectory.json"
)

# Export energy landscape
exporter.export_energy_landscape(
    snapshots=agent.get_trajectory_snapshots(),
    output_file="energy_landscape.csv",
    projection_method="pca"
)
```

---

## Performance Optimization

### PyPy Optimization Tips

1. **Type Hints**: Add type hints to all hot-path functions for JIT optimization
2. **List Comprehensions**: Use list comprehensions instead of loops where possible
3. **Avoid NumPy**: Pure Python is faster under PyPy than NumPy
4. **Warm-up**: First 100-200 iterations are slower (JIT compilation)
5. **Batch Processing**: Process agents in batches for optimal JIT performance

### Memory Management

```python
# Limit trajectory snapshots
agent = ProteinAgent(
    protein_sequence=sequence,
    enable_visualization=True,
    max_snapshots=1000  # Prevents memory overflow
)

# Limit memories per agent
config = AdaptiveConfig(
    max_memories=50,  # Auto-prunes oldest memories
    ...
)

# Limit shared memory pool
# Auto-prunes at MAX_SHARED_MEMORY_POOL_SIZE (10000)
```

### Profiling

```python
import cProfile
import pstats

# Profile exploration
profiler = cProfile.Profile()
profiler.enable()

coordinator.run_parallel_exploration(iterations=100)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Error Handling

All public methods use graceful error handling:

- **Non-critical errors** (memory storage, checkpoints) log errors and continue
- **Critical errors** (conformation validation) attempt repair before failing
- **User errors** (invalid parameters) raise ValueError with clear message

### Example Error Handling

```python
try:
    agent.explore_step()
except ValidationError as e:
    # Conformation validation failed
    logger.error(f"Validation failed: {e}")
    # Attempt repair
    repaired = agent._validator.repair_conformation(agent._current_conformation)
    if repaired:
        agent._current_conformation = repaired
    else:
        # Generate new conformation
        agent._current_conformation = agent._generate_initial_conformation()
```

---

## See Also

- [README.md](README.md) - System overview and quick start
- [EXAMPLES.md](EXAMPLES.md) - Detailed usage examples
- [tests/](tests/) - Comprehensive test suite demonstrating API usage
