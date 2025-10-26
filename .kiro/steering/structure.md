---
inclusion: always
---

# Project Organization

## Dual-System Architecture

The project contains two major systems:

### System 1: Quantum Coherence Protein Predictor (QCPP)
**Location**: Root directory  
**Purpose**: Physics-based stability prediction

### System 2: UBF Protein System
**Location**: `ubf_protein/` directory  
**Purpose**: Consciousness-based conformational exploration

---

## QCPP Architecture (Root)

### Prediction Layer
- `protein_predictor.py` - Core prediction algorithms
  - `QuantumCoherenceProteinPredictor` class
  - QCP calculation, coherence metrics, phi-angle analysis
  - THz spectrum prediction

### Pipeline Layer
- `qc_pipeline.py` - Orchestration and workflow
  - `QCProteinPipeline` class
  - Protein download/loading, batch analysis
  - Visualization generation, result comparison

### Validation Layer
- `validation_framework.py` - Experimental validation
  - `QuantumProteinValidator` class
  - THz experiment design and simulation
  - Correlation with experimental stability data

### Module Responsibilities
- **protein_predictor.py**: Main predictor with QCP, coherence, stability
- **quantum_utils.py**: Quantum mechanics functions (resonance, phi matching)
- **stability_predictor.py**: Stability calculation methods
- **simple_quantum_dssp.py**: Secondary structure assignment
- **qc_pipeline.py**: End-to-end workflow orchestration
- **validation_framework.py**: Experimental validation
- **compare_predictions.py**: Experimental data comparison
- **metrics_utils.py**: Evaluation metrics
- **run_analysis.py**: Main entry point

---

## UBF System Architecture (`ubf_protein/`)

### Core Interfaces Layer (`interfaces.py`)
- `IProteinAgent` - Autonomous agent interface
- `IConsciousnessState` - Consciousness coordinate system
- `IBehavioralState` - Behavioral state derivation
- `IMemorySystem` - Experience memory management
- `ISharedMemoryPool` - Collective learning
- `IMultiAgentCoordinator` - Multi-agent orchestration
- All following **SOLID principles** (dependency inversion, interface segregation)

### Data Models Layer (`models.py`)
- `ConsciousnessCoordinates` - Immutable consciousness state
- `BehavioralStateData` - Immutable behavioral dimensions
- `ConformationalMemory` - Significant transition memory
- `Conformation` - Protein structure representation
- `ConformationalMove` - Move action data
- `ConformationalOutcome` - Exploration result
- `AdaptiveConfig` - Size-based configuration
- All models are **immutable** and **type-safe**

### Consciousness System
- `consciousness.py` - ConsciousnessState implementation (frequency, coherence)
- `behavioral_state.py` - BehavioralState derivation (5 behavioral dimensions)
- Updates based on exploration outcomes

### Memory & Learning
- `memory_system.py` - MemorySystem and SharedMemoryPool
- `config.py` - System constants and thresholds
- Significance-based storage (≥0.3 individual, ≥0.7 shared)
- Auto-pruning with influence-weighted retention

### Move Generation (Mapless Design)
- `mapless_moves.py` - MapplessMoveGenerator and CapabilityBasedMoveEvaluator
- **O(1) move generation** without spatial maps
- 10 move types: backbone rotation, sidechain adjust, helix/sheet/turn formation, hydrophobic collapse, salt bridge, disulfide bond, energy minimization, large jump
- 5 composite evaluation factors: physical feasibility, quantum alignment, behavioral preference, historical success, goal alignment

### Physics Integration
- `physics_integration.py` - QAAP, resonance coupling, water shielding calculators
- Quantum alignment factor: QAAP (0.7-1.3) × resonance (0.9-1.2) × water shielding (0.95-1.05)
- 40 Hz gamma resonance, 408 fs coherence time, 3.57 nm⁻¹ decay

### Agent Implementation
- `protein_agent.py` - ProteinAgent class
- `local_minima_detector.py` - Stuck detection and escape strategies
- `structural_validation.py` - Conformation validation and repair
- Full exploration cycle: generate → evaluate → execute → update

### Multi-Agent Coordination
- `multi_agent_coordinator.py` - MultiAgentCoordinator
- Diversity profiles: 33% cautious, 34% balanced, 33% aggressive
- Parallel exploration with shared memory pool
- Collective learning benefit tracking
- **NEW: QCPP integration for quantum-guided exploration**

### QCPP Integration (NEW)
- `qcpp_integration.py` - QCPP adapter and metrics with caching
- `qcpp_config.py` - Integration configuration (default, high-performance, high-accuracy)
- `physics_grounded_consciousness.py` - Physics-based consciousness mapping
- `integrated_trajectory.py` - Combined UBF+QCPP trajectory recording
- `dynamic_adjustment.py` - Stability-based parameter adjustment
- **Performance**: 0.3-2.0ms QCPP analysis, 30-50% cache hit rate
- **Examples**: `examples/integrated_exploration.py` and `README_INTEGRATED.md`

### Energy & Validation
- `energy_function.py` - Molecular mechanics energy calculator (6 energy terms)
  - Bond stretching, angle bending, dihedral torsion
  - Van der Waals, electrostatic, hydrogen bond
  - AMBER-like force field for accurate structural evaluation
- `rmsd_calculator.py` - RMSD, GDT-TS, TM-score calculation against native PDB
- `validation_suite.py` - Validation framework with 5 test proteins
  - Quality assessment: Excellent/Good/Acceptable/Poor
  - Success criteria: RMSD <5Å, Energy <0, GDT-TS >50
- `native_loader.py` - PDB structure loading for validation

### Adaptive System
- `adaptive_config.py` - Size-based configuration
- Auto-scales parameters for small (<50), medium (50-150), large (>150) proteins
- Adaptive thresholds: stuck window, energy thresholds, memory limits

### Persistence & Monitoring
- `checkpoint.py` - CheckpointManager with SHA256 integrity
- `visualization.py` - VisualizationExporter (JSON, PDB, CSV)
- Auto-save, checkpoint rotation, resume capability
- Real-time trajectory and energy landscape export

### CLI & Validation
- `run_single_agent.py` - Single agent exploration script
- `run_multi_agent.py` - Multi-agent exploration script with native validation
- `benchmark.py` - Performance benchmarking
- `validation_suite.py` - Test suite runner with 5 validation proteins
- `examples/validation_example.py` - Comprehensive validation examples
- `examples/integrated_exploration.py` - QCPP-UBF integration examples

### Testing (`tests/`)
- `test_consciousness.py` - Consciousness system tests
- `test_memory_system.py` - Memory and shared pool tests
- `test_protein_agent.py` - Agent integration tests
- `test_multi_agent_coordinator.py` - Multi-agent tests
- `test_checkpoint.py` - Checkpoint system tests (13 tests)
- `test_visualization.py` - Visualization export tests (7 tests)
- `test_validation.py` - Validation suite tests
- `test_rmsd_calculator.py` - RMSD/GDT-TS/TM-score tests
- `test_native_loader.py` - Native structure loading tests
- `test_qcpp_integration.py` - QCPP integration tests (NEW)
- `test_qcpp_memory.py` - QCPP memory system tests (NEW)
- `test_physics_grounded_consciousness.py` - Physics consciousness tests (NEW)
- Plus 5 more test files covering all components
- **Total: 100+ tests, >90% coverage**

### Documentation
- `README.md` (18 KB) - System overview, installation, quick start, QCPP integration
- `API.md` (37 KB) - Complete API reference with examples
- `EXAMPLES.md` (36 KB) - 10 detailed usage examples
- `examples/README_INTEGRATED.md` - QCPP integration documentation

---

## Data Flow

### QCPP Flow
```
PDB File → load_protein() → calculate_qcp() → calculate_field_coherence() 
  → predict_stability() → visualize_results() → save JSON/plots
```

### UBF Flow
```
Initialize Agents → Generate Moves (O(1)) → Evaluate Moves (5 factors + QCPP) 
  → Execute Move → Update Consciousness → Store Memory → Share High-Sig Memories 
  → Detect Minima → Escape Strategy → Checkpoint → Continue
  → Validate against Native (RMSD/GDT-TS/TM-score)
```

### QCPP-UBF Integration Flow (NEW)
```
UBF Agent → Generate Move → QCPP Analysis (cached) → Quantum Alignment Factor
  → Physics-Grounded Consciousness Update → Move Evaluation → Execute
  → Integrated Trajectory Recording (UBF + QCPP metrics)
```

---

## File Organization

### QCPP Results
- `quantum_coherence_proteins/results/{pdb_id}_analysis.json`
- `quantum_coherence_proteins/results/plots/`
- `quantum_coherence_proteins/pdb_files/`

### UBF Results
- `./checkpoints/` - Checkpoint files with iteration metadata
- `./viz/` - Trajectory JSON, energy landscape CSV
- Results embedded in coordinator state

---

## Code Organization Patterns

### QCPP Classes
1. `__init__()` - Initialize constants (phi, base_energy)
2. `load_*()` - Data loading methods
3. `calculate_*()` - Core calculations
4. `predict_*()` - Predictions
5. `analyze_*()` - Analysis
6. `visualize_*()` - Visualizations

### UBF Classes (SOLID)
1. Interface-first design (all components implement interfaces)
2. Dependency injection (physics calculators, memory systems)
3. Immutable data models
4. Graceful error handling (non-critical failures don't crash)
5. Type-safe with comprehensive type hints

---

## Extension Points

### Adding QCPP Metrics
1. Add calculation to `protein_predictor.py`
2. Update `run_full_analysis()`
3. Add visualization
4. Update JSON output

### Adding UBF Move Types
1. Add move generator in `mapless_moves.py`
2. Add evaluation factors in `CapabilityBasedMoveEvaluator`
3. Update move type enum
4. Add tests

### Integrating QCPP with UBF
1. Use `QCPPIntegrationAdapter` to wrap QCPP predictor
2. Pass adapter to `MultiAgentCoordinator` via `qcpp_integration` parameter
3. QCPP metrics automatically used in move evaluation quantum factor
4. Physics-grounded consciousness updates based on QCPP stability
5. Integrated trajectories record both UBF and QCPP metrics
6. Configure with presets: `get_default_config()`, `get_high_performance_config()`, `get_high_accuracy_config()`
