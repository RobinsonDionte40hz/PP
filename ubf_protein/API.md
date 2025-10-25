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
- [Energy Function](#energy-function)
- [Validation Suite](#validation-suite)
- [Protein Agent](#protein-agent)
- [Multi-Agent Coordinator](#multi-agent-coordinator)
- [Physics Integration](#physics-integration)
- [Validation & Error Handling](#validation--error-handling)
- [Checkpoint System](#checkpoint-system)
- [Visualization](#visualization)
- [QCPP Integration](#qcpp-integration) **← NEW**

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

## Energy Function

### MolecularMechanicsEnergy

```python
class MolecularMechanicsEnergy(IEnergyCalculator):
    """
    AMBER-like force field implementation for protein energy calculation.
    
    Calculates energy using 6 terms:
    1. Bond stretching
    2. Angle bending
    3. Dihedral torsion
    4. Van der Waals (Lennard-Jones)
    5. Electrostatic (Coulomb)
    6. Hydrogen bonding
    
    Energy units: kcal/mol
    """
    
    def calculate_energy(self, conformation: Conformation) -> float:
        """
        Calculate total molecular mechanics energy.
        
        Args:
            conformation: Protein conformation with 3D coordinates
            
        Returns:
            Total energy in kcal/mol
            
        Example:
            >>> calculator = MolecularMechanicsEnergy()
            >>> energy = calculator.calculate_energy(conf)
            >>> print(f"Total energy: {energy:.2f} kcal/mol")
            Total energy: -52.34 kcal/mol
        """
        pass
    
    def calculate_detailed_energy(
        self, 
        conformation: Conformation
    ) -> Dict[str, float]:
        """
        Calculate energy breakdown by component.
        
        Args:
            conformation: Protein conformation
            
        Returns:
            Dictionary with keys:
            - "bond": Bond stretching energy
            - "angle": Angle bending energy
            - "dihedral": Dihedral torsion energy
            - "vdw": Van der Waals energy
            - "electrostatic": Electrostatic energy
            - "hbond": Hydrogen bond energy
            - "total": Sum of all components
            
        Example:
            >>> components = calculator.calculate_detailed_energy(conf)
            >>> for term, energy in components.items():
            ...     print(f"{term}: {energy:.2f} kcal/mol")
            bond: 12.50 kcal/mol
            angle: 18.30 kcal/mol
            dihedral: 8.20 kcal/mol
            vdw: -45.60 kcal/mol
            electrostatic: -28.40 kcal/mol
            hbond: -15.20 kcal/mol
            total: -50.20 kcal/mol
        """
        pass
```

#### Energy Terms

1. **Bond Stretching Energy**
   ```python
   E_bond = Σ k_bond × (r - r₀)²
   ```
   - `k_bond = 500 kcal/(mol·Å²)`: Force constant
   - `r₀ = 1.5 Å`: Ideal bond length (C-C bonds)
   - Penalizes deviation from ideal bond geometry

2. **Angle Bending Energy**
   ```python
   E_angle = Σ k_angle × (θ - θ₀)²
   ```
   - `k_angle = 100 kcal/(mol·rad²)`: Force constant
   - `θ₀ = 109.5° (sp³) or 120° (sp²)`: Ideal angle
   - Maintains local geometry around atoms

3. **Dihedral Torsion Energy**
   ```python
   E_dihedral = Σ V_n/2 × [1 + cos(nφ - γ)]
   ```
   - `V_n = 2.0 kcal/mol`: Barrier height
   - `n = 3`: Periodicity (typical for C-C bonds)
   - Models rotational barriers around bonds

4. **Van der Waals Energy**
   ```python
   E_vdw = Σ 4ε[(σ/r)¹² - (σ/r)⁶]
   ```
   - `ε = 0.5 kcal/mol`: Well depth
   - `σ = 3.5 Å`: Atomic radius
   - Lennard-Jones 12-6 potential for non-bonded interactions

5. **Electrostatic Energy**
   ```python
   E_elec = Σ k_e × q_i × q_j / (ε_r × r_ij)
   ```
   - `k_e = 332 kcal·Å/(mol·e²)`: Coulomb constant
   - `ε_r = 4.0`: Dielectric constant (protein interior)
   - Charged residues: D/E (-1), K/R (+1), H (+0.5)

6. **Hydrogen Bond Energy**
   ```python
   E_hbond = Σ A/r¹² - B/r¹⁰
   ```
   - Distance cutoff: 3.5 Å
   - Angle cutoff: 150° (donor-H-acceptor)
   - Directional H-bond potential

#### Energy Interpretation

- **Folded proteins**: -100 to 0 kcal/mol
- **Native structures**: -50 to -80 kcal/mol (typical)
- **Partially folded**: -30 to 0 kcal/mol
- **Unfolded/clashes**: > 0 kcal/mol (positive energy indicates problems)

#### Configuration

```python
# In config.py
USE_MOLECULAR_MECHANICS_ENERGY = True  # Enable MM energy (default: True)

# Alternative: Simple distance-based energy
USE_MOLECULAR_MECHANICS_ENERGY = False  # Use fast approximate energy
```

---

## Validation Suite

### ValidationSuite

```python
class ValidationSuite:
    """
    Comprehensive validation framework for testing protein predictions.
    
    Provides:
    - Single protein validation against native PDB structures
    - Test suite validation across multiple proteins
    - Baseline comparison (vs random sampling, Monte Carlo)
    - Quality assessment and success criteria
    """
    
    def __init__(
        self,
        test_proteins_file: str = "validation_proteins.json",
        pdb_cache_dir: str = "./pdb_cache"
    ):
        """
        Initialize validation suite.
        
        Args:
            test_proteins_file: Path to test protein configuration JSON
            pdb_cache_dir: Directory to cache downloaded PDB files
        """
        pass
    
    def validate_protein(
        self,
        pdb_id: str,
        num_agents: int = 10,
        iterations: int = 1000,
        use_multi_agent: bool = True
    ) -> ValidationReport:
        """
        Validate prediction against native PDB structure.
        
        Args:
            pdb_id: PDB ID (e.g., "1UBQ" for ubiquitin)
            num_agents: Number of agents for exploration
            iterations: Iterations per agent
            use_multi_agent: Use multi-agent (True) or single agent (False)
            
        Returns:
            ValidationReport with metrics (RMSD, energy, GDT-TS, TM-score)
            
        Example:
            >>> suite = ValidationSuite()
            >>> report = suite.validate_protein("1UBQ", num_agents=10, iterations=1000)
            >>> print(report.get_summary())
            PDB: 1UBQ (76 residues)
            RMSD: 3.45 Å
            Energy: -52.30 kcal/mol
            GDT-TS: 68.2
            TM-score: 0.721
            Quality: GOOD
            Success: ✓
        """
        pass
    
    def run_test_suite(
        self,
        num_agents: int = 10,
        iterations: int = 500
    ) -> TestSuiteResults:
        """
        Run validation on all configured test proteins.
        
        Args:
            num_agents: Number of agents per protein
            iterations: Iterations per agent
            
        Returns:
            TestSuiteResults with aggregated statistics
            
        Example:
            >>> suite = ValidationSuite()
            >>> results = suite.run_test_suite(num_agents=10, iterations=500)
            >>> print(f"Success rate: {results.success_rate:.1f}%")
            Success rate: 75.0%
            >>> print(f"Average RMSD: {results.average_rmsd:.2f} Å")
            Average RMSD: 4.12 Å
        """
        pass
    
    def compare_to_baseline(
        self,
        pdb_id: str,
        num_samples: int = 1000
    ) -> ComparisonReport:
        """
        Compare UBF to baseline methods.
        
        Baselines:
        - Random Sampling: Generate random conformations
        - Monte Carlo: Simple Metropolis algorithm
        
        Args:
            pdb_id: PDB ID for test protein
            num_samples: Number of samples for each baseline
            
        Returns:
            ComparisonReport with improvement percentages
            
        Example:
            >>> comparison = suite.compare_to_baseline("1CRN", num_samples=1000)
            >>> improvements = comparison.get_improvement_summary()
            UBF RMSD: 3.2 Å
            Random Sampling RMSD: 8.7 Å (UBF is 63% better)
            Monte Carlo RMSD: 5.1 Å (UBF is 37% better)
        """
        pass
    
    def save_results(
        self,
        results: Union[ValidationReport, TestSuiteResults],
        output_file: str
    ) -> None:
        """
        Save validation results to JSON file.
        
        Args:
            results: ValidationReport or TestSuiteResults
            output_file: Path to output JSON file
        """
        pass
```

### ValidationReport

```python
@dataclass
class ValidationReport:
    """
    Validation results for single protein prediction.
    
    Attributes:
        pdb_id: PDB identifier (e.g., "1UBQ")
        sequence_length: Number of residues
        best_rmsd: Best RMSD to native structure (Å)
        best_energy: Best total energy (kcal/mol)
        gdt_ts_score: GDT-TS score (0-100)
        tm_score: TM-score (0-1)
        runtime_seconds: Total runtime
        conformations_explored: Number of conformations sampled
        num_agents: Number of agents used
        iterations_per_agent: Iterations per agent
    """
    pdb_id: str
    sequence_length: int
    best_rmsd: float
    best_energy: float
    gdt_ts_score: float
    tm_score: float
    runtime_seconds: float
    conformations_explored: int
    num_agents: int
    iterations_per_agent: int
    
    def assess_quality(self) -> str:
        """
        Assess prediction quality based on RMSD and GDT-TS.
        
        Returns:
            "excellent", "good", "acceptable", or "poor"
            
        Quality Criteria:
            Excellent: RMSD < 2.0 Å AND GDT-TS ≥ 80
            Good: RMSD < 4.0 Å AND GDT-TS ≥ 65
            Acceptable: RMSD < 5.0 Å AND GDT-TS ≥ 50
            Poor: Otherwise
        """
        pass
    
    def is_successful(self) -> bool:
        """
        Check if prediction meets success criteria.
        
        Returns:
            True if all criteria met:
            - RMSD < 5.0 Å
            - Energy < 0 kcal/mol (thermodynamically stable)
            - GDT-TS > 50 (correct fold)
        """
        pass
    
    def get_summary(self) -> str:
        """
        Get formatted summary string.
        
        Returns:
            Multi-line summary with all metrics
        """
        pass
```

### TestSuiteResults

```python
@dataclass
class TestSuiteResults:
    """
    Aggregated results from test suite validation.
    
    Attributes:
        validation_reports: List of individual ValidationReports
        success_rate: Percentage of successful predictions (0-100)
        average_rmsd: Average RMSD across all proteins (Å)
        average_energy: Average energy across all proteins (kcal/mol)
        average_gdt_ts: Average GDT-TS score (0-100)
        average_tm_score: Average TM-score (0-1)
        total_runtime_seconds: Total runtime for all proteins
    """
    validation_reports: List[ValidationReport]
    success_rate: float
    average_rmsd: float
    average_energy: float
    average_gdt_ts: float
    average_tm_score: float
    total_runtime_seconds: float
    
    def get_quality_distribution(self) -> Dict[str, int]:
        """
        Get distribution of quality levels.
        
        Returns:
            Dictionary: {"excellent": count, "good": count, "acceptable": count, "poor": count}
        """
        pass
    
    def get_summary(self) -> str:
        """Get formatted summary of test suite results."""
        pass
```

### ComparisonReport

```python
@dataclass
class ComparisonReport:
    """
    Comparison of UBF to baseline methods.
    
    Attributes:
        pdb_id: PDB identifier
        ubf_rmsd: UBF best RMSD (Å)
        random_rmsd: Random sampling best RMSD (Å)
        monte_carlo_rmsd: Monte Carlo best RMSD (Å)
        ubf_energy: UBF best energy (kcal/mol)
        random_energy: Random sampling best energy (kcal/mol)
        monte_carlo_energy: Monte Carlo best energy (kcal/mol)
    """
    pdb_id: str
    ubf_rmsd: float
    random_rmsd: float
    monte_carlo_rmsd: float
    ubf_energy: float
    random_energy: float
    monte_carlo_energy: float
    
    def get_improvement_summary(self) -> str:
        """
        Get formatted summary of improvements over baselines.
        
        Returns:
            Multi-line summary with improvement percentages
        """
        pass
```

### RMSDCalculator

```python
class RMSDCalculator:
    """
    Calculate RMSD, GDT-TS, and TM-score for structural validation.
    
    Uses Kabsch algorithm for optimal superposition.
    """
    
    def calculate_rmsd(
        self,
        predicted: Conformation,
        native: Conformation
    ) -> float:
        """
        Calculate RMSD between predicted and native structures.
        
        Uses Kabsch algorithm for optimal superposition:
        1. Center both structures at origin
        2. Calculate optimal rotation matrix (SVD)
        3. Rotate predicted structure
        4. Calculate RMSD of aligned structures
        
        Args:
            predicted: Predicted conformation
            native: Native structure from PDB
            
        Returns:
            RMSD in Ångströms
            
        Example:
            >>> calculator = RMSDCalculator()
            >>> rmsd = calculator.calculate_rmsd(pred, native)
            >>> print(f"RMSD: {rmsd:.2f} Å")
            RMSD: 3.45 Å
        """
        pass
    
    def calculate_gdt_ts(
        self,
        predicted: Conformation,
        native: Conformation
    ) -> float:
        """
        Calculate GDT-TS (Global Distance Test - Total Score).
        
        GDT-TS = (P₁ + P₂ + P₄ + P₈) / 4
        
        Where Pₙ is percentage of residues within n Å of native position.
        
        Args:
            predicted: Predicted conformation
            native: Native structure
            
        Returns:
            GDT-TS score (0-100, higher is better)
            
        Example:
            >>> gdt_ts = calculator.calculate_gdt_ts(pred, native)
            >>> print(f"GDT-TS: {gdt_ts:.1f}")
            GDT-TS: 68.2
        """
        pass
    
    def calculate_tm_score(
        self,
        predicted: Conformation,
        native: Conformation
    ) -> float:
        """
        Calculate TM-score (Template Modeling score).
        
        TM-score = (1/L) × Σ 1 / [1 + (d_i/d₀)²]
        
        Where:
        - L: Protein length
        - d_i: Distance of residue i after superposition
        - d₀: Normalization factor (depends on L)
        
        Args:
            predicted: Predicted conformation
            native: Native structure
            
        Returns:
            TM-score (0-1, higher is better)
            - > 0.5: Same fold
            - > 0.6: Similar structure
            - > 0.8: High similarity
            
        Example:
            >>> tm_score = calculator.calculate_tm_score(pred, native)
            >>> print(f"TM-score: {tm_score:.3f}")
            TM-score: 0.721
        """
        pass
```

### NativeStructureLoader

```python
class NativeStructureLoader:
    """
    Load native protein structures from PDB files.
    
    Downloads PDB files from RCSB PDB and extracts CA coordinates.
    """
    
    def __init__(self, pdb_cache_dir: str = "./pdb_cache"):
        """
        Initialize loader with cache directory.
        
        Args:
            pdb_cache_dir: Directory to cache downloaded PDB files
        """
        pass
    
    def load_structure(self, pdb_id: str) -> Conformation:
        """
        Load native structure from PDB.
        
        Downloads PDB file if not cached, extracts CA coordinates,
        and returns as Conformation object.
        
        Args:
            pdb_id: PDB identifier (e.g., "1UBQ")
            
        Returns:
            Conformation with native structure coordinates
            
        Raises:
            ValueError: If PDB ID invalid or structure cannot be loaded
            
        Example:
            >>> loader = NativeStructureLoader()
            >>> native = loader.load_structure("1UBQ")
            >>> print(f"Loaded {len(native.sequence)} residues")
            Loaded 76 residues
        """
        pass
    
    def get_sequence_from_pdb(self, pdb_id: str) -> str:
        """
        Extract amino acid sequence from PDB file.
        
        Args:
            pdb_id: PDB identifier
            
        Returns:
            Single-letter amino acid sequence
        """
        pass
```

---

## QCPP Integration

Integration with Quantum Coherence Protein Predictor (QCPP) for physics-based stability validation during conformational exploration.

### QCPPIntegrationAdapter

```python
class QCPPIntegrationAdapter:
    """
    Adapter for integrating QCPP physics calculations into UBF exploration.
    
    Provides real-time quantum coherence analysis, stability predictions,
    and performance monitoring for conformational exploration.
    
    Features:
    - Caching for repeated conformation analysis
    - Performance timing instrumentation
    - Adaptive frequency adjustment based on performance
    - Optional scipy/numpy dependencies with fallbacks
    """
    
    def __init__(
        self,
        qcpp_predictor,
        config: Optional['QCPPIntegrationConfig'] = None
    ):
        """
        Initialize QCPP integration adapter.
        
        Args:
            qcpp_predictor: QCPP predictor instance (QuantumCoherenceProteinPredictor)
            config: Integration configuration (uses default if None)
            
        Example:
            >>> from protein_predictor import QuantumCoherenceProteinPredictor
            >>> predictor = QuantumCoherenceProteinPredictor()
            >>> adapter = QCPPIntegrationAdapter(predictor)
        """
        pass
    
    def analyze_conformation(
        self,
        sequence: str,
        coordinates: List[Tuple[float, float, float]]
    ) -> 'QCPPMetrics':
        """
        Analyze conformation using QCPP physics calculations.
        
        Calculates:
        - QCP values (Quantum Coherence Protein values)
        - Field coherence
        - Stability prediction
        - THz spectrum
        
        Args:
            sequence: Amino acid sequence (single-letter codes)
            coordinates: CA atom coordinates (x, y, z) in Angstroms
            
        Returns:
            QCPPMetrics with physics calculations and timing information
            
        Example:
            >>> metrics = adapter.analyze_conformation("ACDEFGH", coords)
            >>> print(f"Stability: {metrics.stability_prediction:.3f}")
            >>> print(f"Calculation time: {metrics.calculation_time_ms:.2f}ms")
            Stability: 0.742
            Calculation time: 0.35ms
        """
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache and performance statistics.
        
        Returns:
            Dictionary with:
            - total_requests: Total analysis requests
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Cache hit rate (0-1)
            - avg_calculation_time_ms: Average calculation time
            - total_calculation_time_ms: Total calculation time
            - slow_analyses: List of slow analyses (>5ms threshold)
            
        Example:
            >>> stats = adapter.get_cache_stats()
            >>> print(f"Hit rate: {stats['hit_rate']*100:.1f}%")
            >>> print(f"Avg time: {stats['avg_calculation_time_ms']:.2f}ms")
            Hit rate: 43.5%
            Avg time: 0.30ms
        """
        pass
    
    def should_reduce_analysis_frequency(self) -> bool:
        """
        Check if QCPP analysis frequency should be reduced.
        
        Returns True if:
        - Average calculation time > 5ms
        - Cache hit rate < 20%
        - Adaptive frequency enabled in config
        
        Returns:
            True if frequency should be reduced for performance
            
        Example:
            >>> if adapter.should_reduce_analysis_frequency():
            ...     config.analysis_frequency = max(0.1, config.analysis_frequency / 2)
        """
        pass
    
    def get_performance_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations.
        
        Analyzes current performance metrics and suggests:
        - Cache size adjustments
        - Analysis frequency changes
        - Precomputation strategies
        
        Returns:
            List of recommendation strings
            
        Example:
            >>> for rec in adapter.get_performance_recommendations():
            ...     print(f"⚠️  {rec}")
            ⚠️  Average QCPP calculation time (5.2ms) exceeds 5ms target
            ⚠️  Consider reducing analysis_frequency to 0.5
        ```
        pass
```

### QCPPMetrics

```python
@dataclass(frozen=True)
class QCPPMetrics:
    """
    Quantum Coherence Protein Predictor metrics for a conformation.
    
    Contains physics-based calculations from QCPP system including
    quantum coherence, stability predictions, and THz spectra.
    
    Attributes:
        qcp_values: QCP value per residue (quantum coherence protein)
        field_coherence: Field coherence per residue (0-1)
        stability_prediction: Overall stability score (0-1, higher is better)
        thz_spectrum: THz frequency spectrum amplitudes
        calculation_time_ms: Time taken for QCPP analysis (milliseconds)
    """
    qcp_values: List[float]
    field_coherence: List[float]
    stability_prediction: float
    thz_spectrum: List[float]
    calculation_time_ms: float = 0.0
    
    def average_qcp(self) -> float:
        """Average QCP value across all residues."""
        return sum(self.qcp_values) / len(self.qcp_values) if self.qcp_values else 0.0
    
    def average_coherence(self) -> float:
        """Average field coherence across all residues."""
        return sum(self.field_coherence) / len(self.field_coherence) if self.field_coherence else 0.0
    
    def peak_thz_frequency(self) -> float:
        """Frequency (THz) with maximum amplitude in spectrum."""
        if not self.thz_spectrum:
            return 0.0
        max_idx = self.thz_spectrum.index(max(self.thz_spectrum))
        return max_idx * 0.1  # Assuming 0.1 THz spacing
```

### QCPPIntegrationConfig

```python
@dataclass
class QCPPIntegrationConfig:
    """
    Configuration for QCPP integration behavior.
    
    Controls when and how QCPP analysis is performed during exploration,
    with presets for different performance/accuracy tradeoffs.
    
    Attributes:
        analysis_frequency: Fraction of moves to analyze (0.0-1.0)
        cache_size: Maximum number of cached analyses
        min_energy_change: Minimum energy change to trigger analysis (kcal/mol)
        enable_stability_adjustment: Adjust consciousness based on stability
        stability_weight: Weight for stability in consciousness updates (0-1)
        enable_adaptive_frequency: Auto-adjust frequency based on performance
        min_analysis_frequency: Minimum frequency when adaptive (0.0-1.0)
        max_analysis_frequency: Maximum frequency when adaptive (0.0-1.0)
    """
    analysis_frequency: float = 0.1
    cache_size: int = 1000
    min_energy_change: float = 0.5
    enable_stability_adjustment: bool = True
    stability_weight: float = 0.3
    enable_adaptive_frequency: bool = True
    min_analysis_frequency: float = 0.05
    max_analysis_frequency: float = 1.0
    
    @staticmethod
    def default() -> 'QCPPIntegrationConfig':
        """Balanced configuration (10% analysis, adaptive enabled)."""
        return QCPPIntegrationConfig()
    
    @staticmethod
    def high_performance() -> 'QCPPIntegrationConfig':
        """
        Performance-optimized configuration.
        
        - Lower analysis frequency (5%)
        - Smaller cache (500)
        - Higher energy threshold (1.0 kcal/mol)
        - Adaptive frequency enabled
        
        Use for: Large proteins, high throughput screening
        """
        return QCPPIntegrationConfig(
            analysis_frequency=0.05,
            cache_size=500,
            min_energy_change=1.0,
            min_analysis_frequency=0.01,
            max_analysis_frequency=0.1
        )
    
    @staticmethod
    def high_accuracy() -> 'QCPPIntegrationConfig':
        """
        Accuracy-optimized configuration.
        
        - Higher analysis frequency (50%)
        - Larger cache (5000)
        - Lower energy threshold (0.1 kcal/mol)
        - Adaptive frequency disabled
        
        Use for: Final refinement, critical predictions
        """
        return QCPPIntegrationConfig(
            analysis_frequency=0.5,
            cache_size=5000,
            min_energy_change=0.1,
            enable_adaptive_frequency=False
        )
```

### PhysicsGroundedConsciousness

```python
class PhysicsGroundedConsciousness:
    """
    Consciousness system grounded in QCPP physics calculations.
    
    Updates consciousness coordinates based on:
    - QCPP stability predictions
    - Quantum coherence values
    - THz spectrum features
    
    Replaces heuristic consciousness updates with physics-based adjustments.
    """
    
    def __init__(
        self,
        base_consciousness: 'IConsciousnessState',
        config: QCPPIntegrationConfig
    ):
        """
        Initialize physics-grounded consciousness.
        
        Args:
            base_consciousness: Base consciousness state to modify
            config: Integration configuration
        """
        pass
    
    def update_with_qcpp_metrics(
        self,
        metrics: QCPPMetrics,
        outcome: 'ConformationalOutcome'
    ) -> 'IConsciousnessState':
        """
        Update consciousness based on QCPP metrics.
        
        Adjustments:
        - High stability → Increase coherence
        - Low stability → Decrease coherence, increase frequency
        - High QCP → Increase coherence
        - Strong THz peaks → Fine-tune frequency to match
        
        Args:
            metrics: QCPP analysis results
            outcome: Conformational change outcome
            
        Returns:
            Updated consciousness state
            
        Example:
            >>> new_consciousness = physics_grounded.update_with_qcpp_metrics(
            ...     metrics, outcome
            ... )
            >>> print(f"Coherence: {new_consciousness.coherence:.3f}")
            Coherence: 0.782
        """
        pass
```

### IntegratedTrajectoryRecorder

```python
class IntegratedTrajectoryRecorder:
    """
    Records combined UBF + QCPP trajectory information.
    
    Captures both consciousness-based exploration metrics and
    quantum physics calculations at each step.
    """
    
    def __init__(self):
        """Initialize empty trajectory."""
        self.points: List['IntegratedTrajectoryPoint'] = []
    
    def record_step(
        self,
        iteration: int,
        conformation: 'Conformation',
        consciousness: 'IConsciousnessState',
        behavioral: 'BehavioralState',
        energy: float,
        rmsd: float,
        qcpp_metrics: Optional[QCPPMetrics] = None
    ):
        """
        Record single exploration step.
        
        Args:
            iteration: Step number
            conformation: Current conformation
            consciousness: Consciousness state
            behavioral: Behavioral state
            energy: Total energy (kcal/mol)
            rmsd: RMSD from native (Angstroms)
            qcpp_metrics: Optional QCPP analysis results
        """
        pass
    
    def export_json(self, filepath: str):
        """Export trajectory to JSON file."""
        pass
    
    def export_csv(self, filepath: str):
        """Export trajectory to CSV file."""
        pass
```

### IntegratedTrajectoryPoint

```python
@dataclass(frozen=True)
class IntegratedTrajectoryPoint:
    """
    Single point in combined UBF+QCPP trajectory.
    
    Attributes:
        iteration: Step number
        energy: Total energy (kcal/mol)
        rmsd: RMSD from native (Angstroms)
        consciousness_frequency: Consciousness frequency (Hz)
        consciousness_coherence: Consciousness coherence (0-1)
        exploration_energy: Behavioral exploration energy
        structural_focus: Behavioral structural focus
        qcpp_stability: QCPP stability prediction (0-1, optional)
        qcpp_avg_coherence: Average QCPP field coherence (optional)
        qcpp_avg_qcp: Average QCP value (optional)
        qcpp_calculation_time_ms: QCPP analysis time (milliseconds)
    """
    iteration: int
    energy: float
    rmsd: float
    consciousness_frequency: float
    consciousness_coherence: float
    exploration_energy: float
    structural_focus: float
    qcpp_stability: Optional[float] = None
    qcpp_avg_coherence: Optional[float] = None
    qcpp_avg_qcp: Optional[float] = None
    qcpp_calculation_time_ms: float = 0.0
```

### DynamicParameterAdjuster

```python
class DynamicParameterAdjuster:
    """
    Dynamically adjust UBF parameters based on QCPP stability feedback.
    
    Modifies exploration parameters (temperature, stuck thresholds, etc.)
    based on real-time stability predictions from QCPP.
    
    Strategy:
    - High stability → Reduce temperature (exploit)
    - Low stability → Increase temperature (explore)
    - Stable region → Tighten stuck detection
    - Unstable region → Loosen stuck detection
    """
    
    def __init__(self, base_config: 'SystemConfig'):
        """
        Initialize with base configuration.
        
        Args:
            base_config: Initial system configuration
        """
        pass
    
    def adjust_parameters(
        self,
        metrics: QCPPMetrics,
        current_config: 'SystemConfig'
    ) -> 'SystemConfig':
        """
        Adjust parameters based on QCPP stability.
        
        Args:
            metrics: Current QCPP analysis
            current_config: Current configuration
            
        Returns:
            Adjusted configuration
            
        Example:
            >>> adjuster = DynamicParameterAdjuster(base_config)
            >>> new_config = adjuster.adjust_parameters(metrics, current_config)
            >>> print(f"Temperature: {new_config.temperature:.3f}")
            Temperature: 0.850
        """
        pass
```

---

## See Also

- [README.md](README.md) - System overview and quick start
- [EXAMPLES.md](EXAMPLES.md) - Detailed usage examples
- [tests/](tests/) - Comprehensive test suite demonstrating API usage
- [examples/integrated_exploration.py](examples/integrated_exploration.py) - Complete QCPP integration example
