# UBF Protein System

A consciousness-based protein structure prediction system using the Universal Behavioral Framework (UBF). This system uses autonomous agents with consciousness coordinates to explore conformational space and predict protein structures through mapless, capability-based navigation.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance Metrics](#performance-metrics)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Contributing](#contributing)

## Overview

The UBF Protein System integrates quantum coherence principles with autonomous agent behavior to predict protein structures. Each agent operates in a 2D consciousness coordinate system (frequency 3-15 Hz, coherence 0.2-1.0) that guides conformational exploration through behavioral states derived from consciousness coordinates.

### Key Concepts

- **Consciousness Coordinates**: 2D coordinate system (frequency, coherence) that drives agent behavior
- **Behavioral States**: Five behavioral dimensions derived from consciousness (exploration_energy, structural_focus, hydrophobic_drive, risk_tolerance, native_state_ambition)
- **Mapless Navigation**: O(1) move generation using capability-based filtering (no pathfinding or explicit maps)
- **Memory System**: Experience-based learning with significance-weighted memories and shared memory pool
- **Local Minima Detection**: Adaptive escape strategies using consciousness coordinate adjustments
- **Multi-Agent Coordination**: Parallel exploration with collective learning through shared memory

## Key Features

✅ **Pure Python Implementation** - PyPy-compatible for 2-5x performance boost  
✅ **SOLID Architecture** - Interface-driven design with dependency inversion  
✅ **Molecular Mechanics Energy** - AMBER-like force field with 6 energy terms  
✅ **Structural Validation** - RMSD, GDT-TS, TM-score validation against native structures  
✅ **Adaptive Configuration** - Automatic parameter scaling based on protein size (small/medium/large)  
✅ **Checkpoint & Resume** - Save and restore exploration state with integrity checking  
✅ **Real-Time Visualization** - Export trajectories and energy landscapes in multiple formats  
✅ **Comprehensive Testing** - 100+ unit and integration tests with >90% coverage  
✅ **Performance Optimized** - <2ms move evaluation, <10μs memory retrieval  

## Architecture

### Core Components

```
ubf_protein/
├── interfaces.py              # SOLID interfaces for all components
├── models.py                  # Data models (immutable, type-safe)
├── config.py                  # System configuration parameters
├── consciousness.py           # Consciousness coordinate system
├── behavioral_state.py        # Behavioral state derivation
├── memory_system.py           # Experience memory and shared pool
├── mapless_moves.py          # Capability-based move generation
├── energy_function.py        # Molecular mechanics energy calculator
├── rmsd_calculator.py        # RMSD, GDT-TS, TM-score calculation
├── local_minima_detector.py  # Stuck detection and escape strategies
├── structural_validation.py   # Conformation validation and repair
├── physics_integration.py     # Quantum physics calculators (QAAP, resonance, water shielding)
├── protein_agent.py          # Autonomous protein agent
├── multi_agent_coordinator.py # Multi-agent coordination
├── validation_suite.py       # Validation framework for testing predictions
├── adaptive_config.py        # Size-based configuration
├── checkpoint.py             # Checkpoint and resume system
└── visualization.py          # Trajectory and energy landscape export
```

### Design Principles

1. **SOLID Principles**: All components follow SOLID design for maintainability and testability
2. **Mapless Design**: O(1) conformational navigation without spatial maps or pathfinding
3. **Dependency Inversion**: Interfaces define contracts, implementations are injected
4. **Adaptive Scaling**: Parameters automatically adjust based on protein size
5. **Graceful Degradation**: Non-critical failures (memory, checkpoints) don't crash system

## Installation

### Prerequisites

- Python 3.8+ or PyPy 3.8+
- pip

### Standard Installation

```bash
# Clone the repository
git clone <repository-url>
cd PP/ubf_protein

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### PyPy Installation (Recommended for Performance)

```bash
# Install PyPy (see https://www.pypy.org/download.html)
# On Ubuntu/Debian:
sudo apt-get install pypy3

# Create PyPy virtual environment
pypy3 -m venv pypy_env
source pypy_env/bin/activate  # On Windows: pypy_env\Scripts\activate

# Install dependencies
pypy3 -m pip install -r requirements.txt
```

### Windows-Specific Notes

For Python 3.13+, some dependencies may require C++ Build Tools. We recommend:
- Use Python 3.12 or earlier (has pre-built wheels), OR
- Use PyPy3.9+ (pure Python, no compilation needed), OR
- Install Visual Studio C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Quick Start

### Single Agent Exploration

```python
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import AdaptiveConfig, ProteinSizeClass

# Create agent
agent = ProteinAgent(
    protein_sequence="ACDEFGHIKLMNPQRSTVWY",
    initial_frequency=9.0,
    initial_coherence=0.6
)

# Explore conformational space
for i in range(100):
    outcome = agent.explore_step()
    if i % 10 == 0:
        print(f"Iteration {i}: Energy = {agent._best_energy:.2f}")
```

### Multi-Agent Exploration

```python
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

# Initialize coordinator
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGHIKLMNPQRSTVWY",
    enable_checkpointing=True
)

# Initialize agents with diversity
coordinator.initialize_agents(count=10, diversity_profile="balanced")

# Run parallel exploration
results = coordinator.run_parallel_exploration(iterations=500)

print(f"Best energy: {results.best_energy:.2f}")
print(f"Best RMSD: {results.best_rmsd:.2f}")
print(f"Collective learning benefit: {results.collective_learning_benefit:.2%}")
```

### Command-Line Interface

```bash
# Single agent
python run_single_agent.py --sequence ACDEFGH --iterations 1000 --output results.json

# Multi-agent
python run_multi_agent.py --sequence ACDEFGH --agents 10 --iterations 500 --diversity balanced

# Validation against native structure
python run_multi_agent.py --sequence MQIFVKT --agents 10 --iterations 500 --native 1UBQ

# Benchmark
python benchmark.py --agents 100 --iterations 1000

# Validation suite
python -m ubf_protein.examples.validation_example
```

## Usage Examples

### Example 1: Single Agent with Custom Configuration

```python
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import AdaptiveConfig, ProteinSizeClass

# Custom configuration for small protein
config = AdaptiveConfig(
    size_class=ProteinSizeClass.SMALL,
    residue_count=30,
    initial_frequency_range=(6.0, 12.0),
    initial_coherence_range=(0.4, 0.8),
    stuck_window=20,
    stuck_threshold=5.0,
    max_memories=30,
    convergence_energy_threshold=5.0,
    convergence_rmsd_threshold=1.5,
    checkpoint_interval=100,
    max_iterations=1000
)

agent = ProteinAgent(
    protein_sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
    adaptive_config=config,
    enable_visualization=True
)

# Run exploration
for i in range(config.max_iterations):
    outcome = agent.explore_step()
    
    # Check for convergence
    if outcome.success and outcome.energy_change < -config.convergence_energy_threshold:
        print(f"Converged at iteration {i}")
        break
```

### Example 2: Multi-Agent with Checkpointing

```python
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

# Create coordinator with checkpointing
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGHIKLMNPQRSTVWY",
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints"
)

# Initialize diverse agent population
coordinator.initialize_agents(count=20, diversity_profile="balanced")

# Set auto-save interval (every 50 iterations)
coordinator._checkpoint_manager.set_auto_save_interval(50)

# Run exploration with auto-save
results = coordinator.run_parallel_exploration(iterations=500)

# Resume from checkpoint later
coordinator2 = MultiAgentCoordinator(
    protein_sequence="ACDEFGHIKLMNPQRSTVWY",
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints"
)
coordinator2.resume_from_checkpoint()
coordinator2.run_parallel_exploration(iterations=500)  # Continue from iteration 500
```

### Example 3: Visualization Export

```python
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.visualization import VisualizationExporter

# Create agent with visualization enabled
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

# Export energy landscape (2D PCA projection)
exporter.export_energy_landscape(
    snapshots=agent.get_trajectory_snapshots(),
    output_file="energy_landscape.csv",
    projection_method="pca"
)
```

### Example 4: Memory System Analysis

```python
from ubf_protein.protein_agent import ProteinAgent

agent = ProteinAgent(protein_sequence="ACDEFGH")

# Run exploration
for i in range(200):
    agent.explore_step()

# Analyze memories
memory_system = agent.get_memory_system()
for move_type, memories in memory_system._memories.items():
    print(f"\n{move_type} memories: {len(memories)}")
    for memory in sorted(memories, key=lambda m: m.significance, reverse=True)[:3]:
        print(f"  Significance: {memory.significance:.3f}")
        print(f"  Energy change: {memory.energy_change:.2f}")
        print(f"  Success: {memory.success}")
```

## Energy Function and Validation

### Molecular Mechanics Energy Calculator

The system uses an AMBER-like force field with 6 energy terms for accurate structural evaluation:

#### Energy Components

1. **Bond Stretching Energy**
   ```python
   E_bond = Σ k_bond × (r - r₀)²
   ```
   - Penalizes deviation from ideal bond lengths (1.5 Å for C-C bonds)
   - Force constant: 500 kcal/(mol·Å²)

2. **Angle Bending Energy**
   ```python
   E_angle = Σ k_angle × (θ - θ₀)²
   ```
   - Penalizes deviation from ideal bond angles (109.5° for sp³, 120° for sp²)
   - Force constant: 100 kcal/(mol·rad²)

3. **Dihedral Torsion Energy**
   ```python
   E_dihedral = Σ V_n/2 × [1 + cos(nφ - γ)]
   ```
   - Models rotational barriers around bonds
   - Barrier height: 2.0 kcal/mol

4. **Van der Waals Energy**
   ```python
   E_vdw = Σ 4ε[(σ/r)¹² - (σ/r)⁶]
   ```
   - Lennard-Jones 12-6 potential for non-bonded interactions
   - ε = 0.5 kcal/mol, σ = 3.5 Å

5. **Electrostatic Energy**
   ```python
   E_elec = Σ k_e × q_i × q_j / (ε_r × r_ij)
   ```
   - Coulomb interactions between charged residues
   - Dielectric constant: 4.0 (protein interior)

6. **Hydrogen Bond Energy**
   ```python
   E_hbond = Σ A/r¹² - B/r¹⁰
   ```
   - Directional H-bond potential (donor-acceptor)
   - Distance cutoff: 3.5 Å, angle cutoff: 150°

#### Total Energy

```python
E_total = E_bond + E_angle + E_dihedral + E_vdw + E_elec + E_hbond
```

**Energy ranges for folded proteins**: -100 to 0 kcal/mol  
**Typical native structures**: -50 to -80 kcal/mol  
**Positive energy**: Indicates steric clashes or unfolded state

#### Configuration

Enable molecular mechanics energy in `config.py`:
```python
USE_MOLECULAR_MECHANICS_ENERGY = True  # Default: True
```

### Structural Validation Metrics

The system validates predictions against native PDB structures using three metrics:

#### 1. RMSD (Root Mean Square Deviation)

```python
RMSD = sqrt(Σ(r_pred - r_native)² / N)
```

- Measures average Cα displacement in Ångströms
- **Quality criteria**:
  - Excellent: RMSD < 2.0 Å
  - Good: 2.0 ≤ RMSD < 4.0 Å
  - Acceptable: 4.0 ≤ RMSD < 5.0 Å
  - Poor: RMSD ≥ 5.0 Å

#### 2. GDT-TS (Global Distance Test - Total Score)

```python
GDT-TS = (P₁ + P₂ + P₄ + P₈) / 4
```

- Percentage of residues within 1Å, 2Å, 4Å, 8Å of native positions
- Scale: 0-100 (higher is better)
- **Quality criteria**:
  - Excellent: GDT-TS ≥ 80
  - Good: 65 ≤ GDT-TS < 80
  - Acceptable: 50 ≤ GDT-TS < 65
  - Poor: GDT-TS < 50

#### 3. TM-score (Template Modeling Score)

```python
TM-score = (1/L) × Σ 1 / [1 + (d_i/d₀)²]
```

- Length-independent structure similarity score
- Scale: 0-1 (higher is better)
- **Quality criteria**:
  - Same fold: TM-score > 0.5
  - Similar structure: TM-score > 0.6
  - High similarity: TM-score > 0.8

### Validation Suite

The validation suite provides comprehensive testing against known structures:

```python
from ubf_protein.validation_suite import ValidationSuite

# Initialize suite
suite = ValidationSuite()

# Validate single protein
report = suite.validate_protein(
    pdb_id="1UBQ",          # Ubiquitin
    num_agents=10,
    iterations=1000,
    use_multi_agent=True
)

# Check results
print(f"RMSD: {report.best_rmsd:.2f} Å")
print(f"Energy: {report.best_energy:.2f} kcal/mol")
print(f"GDT-TS: {report.gdt_ts_score:.1f}")
print(f"TM-score: {report.tm_score:.3f}")
print(f"Quality: {report.assess_quality()}")  # "excellent", "good", "acceptable", "poor"
print(f"Success: {report.is_successful()}")   # True if RMSD < 5.0, Energy < 0, GDT-TS > 50

# Run full test suite
results = suite.run_test_suite(num_agents=10, iterations=500)
print(f"Success rate: {results.success_rate:.1f}%")
print(f"Average RMSD: {results.average_rmsd:.2f} Å")
print(f"Average GDT-TS: {results.average_gdt_ts:.1f}")
```

#### Test Proteins

The suite includes 5 validation proteins (configured in `validation_proteins.json`):

| PDB ID | Name | Residues | Difficulty | Expected RMSD |
|--------|------|----------|------------|---------------|
| 1UBQ | Ubiquitin | 76 | Medium | 3-5 Å |
| 1CRN | Crambin | 46 | Easy | 2-4 Å |
| 2MR9 | Villin Headpiece | 35 | Easy | 2-3 Å |
| 1VII | Villin Headpiece (NMR) | 36 | Easy | 2-3 Å |
| 1LYZ | Lysozyme | 129 | Hard | 5-7 Å |

#### Validation Example Script

See `ubf_protein/examples/validation_example.py` for comprehensive usage examples:

```bash
# Run validation examples
python -m ubf_protein.examples.validation_example

# Examples include:
#   1. Single protein validation
#   2. Quality assessment criteria
#   3. Test suite validation
#   4. Baseline comparison
#   5. Programmatic usage
#   6. Command-line validation
```

### Command-Line Validation

Validate predictions directly from the command line:

```bash
# Basic validation with native structure
python run_multi_agent.py MQIFVKT --agents 10 --iterations 500 --native 1UBQ

# Output includes:
#   ✓ Energy components breakdown (6 terms)
#   ✓ RMSD to native structure
#   ✓ GDT-TS and TM-score
#   ✓ Quality assessment (★★★★ to ★)
#   ✓ Energy validation (✓ Negative = folded / ⚠ Positive = unfolded)
```

Example output:
```
🔬 Energy Components:
   Bond Energy:        12.50 kcal/mol
   Angle Energy:       18.30 kcal/mol
   Dihedral Energy:     8.20 kcal/mol
   VDW Energy:        -45.60 kcal/mol
   Electrostatic:     -28.40 kcal/mol
   H-Bond Energy:     -15.20 kcal/mol
   -------------------------------
   Total Energy:      -50.20 kcal/mol  ✓ Negative (folded)

📊 Validation Metrics:
   RMSD to Native:      3.45 Å
   GDT-TS Score:       68.2
   TM-score:            0.721

⭐ Prediction Quality:  ★★★ GOOD
```

## Configuration

### Adaptive Configuration System

The system automatically scales parameters based on protein size:

| Parameter | Small (<50) | Medium (50-150) | Large (>150) |
|-----------|-------------|-----------------|--------------|
| Stuck Window | 20 | 30 | 40 |
| Stuck Threshold | 5.0 | 10.0 | 15.0 |
| Max Memories | 30 | 50 | 75 |
| Convergence Energy | 5.0 | 10.0 | 15.0 |
| Convergence RMSD | 1.5 | 2.0 | 3.0 |
| Max Iterations | 1000 | 2000 | 5000 |

### Agent Diversity Profiles

Multi-agent systems use diversity profiles for initialization:

- **Cautious** (33%): Low frequency (3-6 Hz), high coherence (0.7-1.0) - Conservative exploration
- **Balanced** (34%): Medium frequency (6-12 Hz), medium coherence (0.4-0.7) - Balanced approach
- **Aggressive** (33%): High frequency (12-15 Hz), low coherence (0.2-0.4) - Aggressive exploration

### Memory System Parameters

```python
MEMORY_SIGNIFICANCE_THRESHOLD = 0.3  # Minimum significance to store
MAX_MEMORIES_PER_AGENT = 50          # Max memories before pruning
SHARED_MEMORY_SIGNIFICANCE_THRESHOLD = 0.7  # Threshold for sharing
MAX_SHARED_MEMORY_POOL_SIZE = 10000  # Max shared memories
```

### Consciousness Update Rules

```python
CONSCIOUSNESS_UPDATE_RULES = {
    "success": {"frequency": 0.5, "coherence": 0.05},
    "minor_success": {"frequency": 0.2, "coherence": 0.02},
    "failure": {"frequency": -0.3, "coherence": -0.03},
    "stuck": {"frequency": -0.5, "coherence": -0.05}
}
```

## Performance Metrics

### Expected Performance (PyPy 7.3+)

| Metric | Target | Typical |
|--------|--------|---------|
| Move evaluation latency | <2ms | 0.5-1.5ms |
| Memory retrieval | <10μs | 2-8μs |
| Agent memory footprint | <50MB | 15-30MB |
| 100 agents, 500K conformations | <2min | 60-90s |
| PyPy speedup vs CPython | ≥2x | 2-5x |

### Benchmark Command

```bash
# Run comprehensive benchmark
python benchmark.py --agents 100 --iterations 5000

# Compare PyPy vs CPython
python benchmark.py --agents 10 --iterations 1000 --compare-cpython
```

### Performance Tips

1. **Use PyPy** - 2-5x faster than CPython
2. **Warm up JIT** - First 100-200 iterations slower (JIT compilation)
3. **Batch agents** - Use 10-100 agents for optimal parallelization
4. **Limit snapshots** - Set `max_snapshots=1000` to control memory
5. **Disable visualization** - Set `enable_visualization=False` for production runs

## API Documentation

### Core Interfaces

#### IProteinAgent
```python
def explore_step() -> ConformationalOutcome:
    """Execute single exploration step."""
    
def get_consciousness_state() -> IConsciousnessState:
    """Get current consciousness coordinates."""
    
def get_behavioral_state() -> IBehavioralState:
    """Get current behavioral state."""
    
def get_memory_system() -> IMemorySystem:
    """Get agent's memory system."""
```

#### IConsciousnessState
```python
def get_frequency() -> float:
    """Get consciousness frequency (3-15 Hz)."""
    
def get_coherence() -> float:
    """Get consciousness coherence (0.2-1.0)."""
    
def update_from_outcome(outcome: ConformationalOutcome) -> None:
    """Update consciousness from exploration outcome."""
```

#### IBehavioralState
```python
def get_exploration_energy() -> float:
    """Get exploration energy (0-1)."""
    
def get_structural_focus() -> float:
    """Get structural focus (0-1)."""
    
def get_hydrophobic_drive() -> float:
    """Get hydrophobic drive (0-1)."""
    
def get_risk_tolerance() -> float:
    """Get risk tolerance (0-1)."""
    
def get_native_state_ambition() -> float:
    """Get native state ambition (0-1)."""
```

#### IMemorySystem
```python
def store_memory(memory: ConformationalMemory) -> None:
    """Store significant memory (significance ≥ 0.3)."""
    
def retrieve_relevant_memories(move_type: str, max_count: int = 10) -> List[ConformationalMemory]:
    """Retrieve relevant memories for move evaluation."""
    
def calculate_memory_influence(memories: List[ConformationalMemory]) -> float:
    """Calculate memory influence multiplier (0.8-1.5)."""
```

#### IMultiAgentCoordinator
```python
def initialize_agents(count: int, diversity_profile: str = "balanced") -> List[IProteinAgent]:
    """Initialize diverse agent population."""
    
def run_parallel_exploration(iterations: int) -> ExplorationResults:
    """Run all agents in parallel for N iterations."""
    
def get_best_conformation() -> Conformation:
    """Get best conformation across all agents."""
```

### Data Models

#### ConsciousnessCoordinates
```python
@dataclass(frozen=True)
class ConsciousnessCoordinates:
    frequency: float  # 3-15 Hz
    coherence: float  # 0.2-1.0
    last_update_timestamp: int
```

#### BehavioralStateData
```python
@dataclass(frozen=True)
class BehavioralStateData:
    exploration_energy: float
    structural_focus: float
    hydrophobic_drive: float
    risk_tolerance: float
    native_state_ambition: float
    timestamp: int
```

#### ConformationalMemory
```python
@dataclass
class ConformationalMemory:
    memory_id: str
    move_type: str
    significance: float
    energy_change: float
    rmsd_change: float
    success: bool
    timestamp: int
    consciousness_state: ConsciousnessCoordinates
    behavioral_state: BehavioralStateData
```

## Testing

### Run All Tests

```bash
# Run all tests
pytest ubf_protein/tests/ -v

# Run specific test file
pytest ubf_protein/tests/test_protein_agent.py -v

# Run with coverage
pytest ubf_protein/tests/ --cov=ubf_protein --cov-report=html
```

### Test Structure

```
tests/
├── test_consciousness.py          # Consciousness coordinate tests
├── test_behavioral_state.py       # Behavioral state tests
├── test_memory_system.py          # Memory system tests
├── test_local_minima_detector.py  # Local minima detection tests
├── test_structural_validation.py  # Validation and repair tests
├── test_physics_integration.py    # Physics calculator tests
├── test_protein_agent.py          # Agent integration tests
├── test_multi_agent_coordinator.py # Multi-agent tests
├── test_checkpoint.py             # Checkpoint system tests
└── test_visualization.py          # Visualization export tests
```

### Test Coverage

- Unit tests: 80+ tests
- Integration tests: 30+ tests
- Performance tests: 10+ benchmarks
- Total coverage: >90%

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd PP/ubf_protein

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black mypy

# Install in editable mode
pip install -e .
```

### Code Style

- Follow PEP 8 style guide
- Use type hints for all public methods
- Document all public interfaces with docstrings
- Run `black` formatter before committing
- Run `mypy` for type checking

### Testing Requirements

- All new features must have unit tests
- Integration tests for cross-component features
- Performance benchmarks for critical paths
- Maintain >90% test coverage

## License

[Your License Here]

## Citation

If you use this system in your research, please cite:

```bibtex
@software{ubf_protein,
  title={UBF Protein System: Consciousness-Based Protein Structure Prediction},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## Contact

For questions, issues, or contributions:
- GitHub Issues: [Repository Issues URL]
- Email: [Your Email]

## Acknowledgments

This system integrates concepts from:
- Universal Behavioral Framework (UBF)
- Quantum coherence in biological systems
- Multi-agent reinforcement learning
- Protein structure prediction methodologies
