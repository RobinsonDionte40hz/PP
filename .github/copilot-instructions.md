# Dual-System Protein Platform - AI Coding Guidelines

## Project Overview

This project contains **two complementary protein structure prediction systems**:

### 1. QCPP (Quantum Coherence Protein Predictor) - Root Directory
Physics-based stability prediction using quantum coherence and golden ratio patterns.

### 2. UBF Protein System - `ubf_protein/` Directory  
Consciousness-based conformational exploration using autonomous agents (✅ **COMPLETE - ALL 16 TASKS DONE**).

---

## QCPP System Guidelines (Root Directory)

### Architecture Overview
Core components:
- `QuantumCoherenceProteinPredictor`: Calculates QCP values, field coherence, THz spectra
- `QCProteinPipeline`: Orchestrates workflow from PDB loading to validation
- Data flow: PDB → QCP calculation → coherence analysis → stability prediction → THz spectrum

### Key Patterns & Conventions

#### QCP Formula Implementation
Always use exact formula: `qcp = 4.0 + (2**n * (phi**l) * m)`
- `n`: Structural hierarchy (0=coil, 1=helix, 2=sheet, 3=phi-based bends)
- `l`: Neighbor count (scaled 1-3)
- `m`: Hydrophobicity (normalized -1 to 1)

#### Golden Ratio Integration
- Phi angle: `2 * π / φ ≈ 137.5°` (radians: `phi_angle = 2 * π / phi`)
- Use phi harmonics: `[φ⁰, φ¹, φ², φ³, φ⁴]` for frequency calculations
- Distance matching: Check against `3.8 * φ^n` Å

#### Secondary Structure Handling
Prefer `SimpleQuantumDSSP` for SS assignment. Fallback to distance-based approximation if DSSP fails:
```python
# Helical: i,i+3 distances ~5Å
for i in range(len-3):
    if 4.5 < distances[i][i+3] < 5.5:
        residues[i:i+4] = ['H'] * 4
```

#### File Organization
- Results: `quantum_coherence_proteins/results/{pdb_id}_analysis.json`
- Plots: `quantum_coherence_proteins/results/plots/`
- PDB files: `quantum_coherence_proteins/pdb_files/`

### Development Workflow

#### Running Analysis
```bash
python run_analysis.py  # Full pipeline on test proteins
```

#### Adding New Proteins
1. Add PDB ID to `test_proteins` list in `run_analysis.py`
2. Pipeline auto-downloads via `PDBList.retrieve_pdb_file()`
3. Results saved automatically to JSON

#### Validation Integration
Use `QCProteinPipeline.run_validation_pipeline()` with experimental data from `experimental_stability.csv`. Correlate predicted stability with THz spectra and experimental measurements.

#### Visualization Standards
- QCP/coherence plots: Line plots with residue ID x-axis
- THz spectra: Bar plots with phi harmonic markers
- Phi angles: Histograms with 137.5° and 222.5° reference lines

### Common Pitfalls
- Always check `res.has_id('CA')` before accessing CA coordinates
- Handle DSSP failures gracefully with fallback SS calculation
- Normalize coherence values (subtract min) before averaging
- Use absolute paths for file operations in pipeline

---

## UBF System Guidelines (`ubf_protein/` Directory)

### Architecture Overview (SOLID + Mapless Design)

#### Core Principles
1. **SOLID Architecture**: All components implement interfaces, dependency inversion throughout
2. **Mapless Navigation**: O(1) move generation without spatial maps or pathfinding
3. **Pure Python**: PyPy-optimized, no NumPy/C-extensions for 2-5x speedup
4. **Immutable Models**: All data models are frozen dataclasses
5. **Graceful Degradation**: Non-critical failures (memory, checkpoints) log and continue

#### Component Structure
- **Interfaces** (`interfaces.py`): 11 core interfaces defining contracts
- **Models** (`models.py`): 10+ immutable data models with type safety
- **Consciousness** (`consciousness.py`, `behavioral_state.py`): 2D coordinate system (freq 3-15 Hz, coh 0.2-1.0)
- **Memory** (`memory_system.py`): Experience storage with significance thresholds (≥0.3 individual, ≥0.7 shared)
- **Move Generation** (`mapless_moves.py`): O(1) capability-based filtering, 10 move types
- **Move Evaluation**: 5 composite factors (physical, quantum, behavioral, historical, goal)
- **Physics** (`physics_integration.py`): QAAP, 40 Hz resonance, 408 fs water shielding
- **Agent** (`protein_agent.py`): Autonomous exploration with consciousness updates
- **Coordination** (`multi_agent_coordinator.py`): Diverse populations (33% cautious, 34% balanced, 33% aggressive)
- **Adaptive** (`adaptive_config.py`): Auto-scaling for small/medium/large proteins
- **Persistence** (`checkpoint.py`): SHA256 integrity, auto-save, rotation
- **Visualization** (`visualization.py`): JSON/PDB/CSV export with 2D projections

### Key Patterns & Conventions

#### Consciousness System
```python
# Consciousness coordinates
frequency: 3.0-15.0 Hz  # Exploration tempo
coherence: 0.2-1.0      # Behavioral consistency

# Update rules
Success: +0.5 Hz frequency, +0.05 coherence
Failure: -0.3 Hz frequency, -0.03 coherence
Stuck:   -0.5 Hz frequency, -0.05 coherence
```

#### Behavioral Derivation
```python
# 5 dimensions derived from consciousness
exploration_energy = (freq - 3) / 12
structural_focus = coherence
hydrophobic_drive = sqrt(exploration_energy × structural_focus)
risk_tolerance = exploration_energy × (1 - coherence)
native_state_ambition = coherence × 0.8 + 0.2
```

#### Memory Significance
```python
# 3-factor calculation
significance = (energy_impact × 0.5 + 
                structural_novelty × 0.3 + 
                rmsd_improvement × 0.2)

# Storage thresholds
if significance >= 0.3:
    memory_system.store_memory()
if significance >= 0.7:
    shared_pool.share_memory()
```

#### Move Evaluation (5 Composite Factors)
```python
# Factor 1: Physical Feasibility (0.1-1.0)
physical = structural_feasibility × energy_barrier × ramachandran

# Factor 2: Quantum Alignment (0.5-1.5)
quantum = qaap(0.7-1.3) × resonance(0.9-1.2) × water_shielding(0.95-1.05)

# Factor 3: Behavioral Preference (0.5-1.5)
behavioral = combine_all_5_dimensions()

# Factor 4: Historical Success (0.8-1.5)
historical = memory_influence × novelty_bonus

# Factor 5: Goal Alignment (0.5-1.5)
goal = energy_decrease × rmsd_improvement

# Final weight
weight = physical × quantum × behavioral × historical × goal × temperature
```

#### Adaptive Configuration
```python
# Auto-scales by protein size
Small (<50):   stuck_window=20, stuck_threshold=5.0,  max_iter=1000
Medium (50-150): stuck_window=30, stuck_threshold=10.0, max_iter=2000
Large (>150):  stuck_window=40, stuck_threshold=15.0, max_iter=5000
```

### Development Workflow

#### Running Single Agent
```bash
python ubf_protein/run_single_agent.py --sequence ACDEFGH --iterations 1000 --output results.json
```

#### Running Multi-Agent
```bash
python ubf_protein/run_multi_agent.py --sequence ACDEFGH --agents 10 --iterations 500 --diversity balanced
```

#### Running Tests (100+ tests, >90% coverage)
```bash
pytest ubf_protein/tests/ -v                         # All tests
pytest ubf_protein/tests/test_checkpoint.py -v       # 13 checkpoint tests
pytest ubf_protein/tests/test_visualization.py -v    # 7 visualization tests
```

#### Performance Benchmarking
```bash
python ubf_protein/benchmark.py --agents 100 --iterations 5000
```

#### Using Checkpoints
```python
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints"
)
coordinator._checkpoint_manager.set_auto_save_interval(50)
coordinator.run_parallel_exploration(iterations=200)

# Resume later
coordinator2.resume_from_checkpoint()
```

### File Organization
- **Source**: All implementation in `ubf_protein/*.py`
- **Tests**: `ubf_protein/tests/test_*.py` (100+ tests)
- **Docs**: `ubf_protein/{README,API,EXAMPLES}.md` (91.8 KB total)
- **Checkpoints**: `./checkpoints/*.json` with SHA256 integrity
- **Visualizations**: `./viz/{trajectory.json, energy_landscape.csv}`

### Common Pitfalls
- **Don't use NumPy**: Pure Python for PyPy optimization
- **Immutable models**: Use `replace()` or create new instances, never mutate
- **Interface contracts**: Always implement full interface, respect method signatures
- **O(1) moves**: Never use spatial maps, pathfinding, or N² algorithms
- **Type hints**: Add to all public methods for JIT optimization
- **Graceful errors**: Non-critical failures should log and continue, not crash
- **Memory limits**: Respect `max_snapshots`, `max_memories` to prevent overflow
- **Checkpoint naming**: Use iteration numbers or timestamps, avoid conflicts

### Performance Expectations
- Move evaluation: <2ms (target), 0.5-1.5ms typical ✅
- Memory retrieval: <10μs (target), 2-8μs typical ✅
- Agent memory: <50MB with 50 memories, 15-30MB typical ✅
- Multi-agent: 100 agents × 5K conf < 2min, 60-90s typical ✅
- PyPy speedup: ≥2x vs CPython, 2-5x typical ✅

### Documentation Resources
- **README.md** (18 KB): System overview, installation, quick start, configuration
- **API.md** (37 KB): Complete API reference with design principles and examples
- **EXAMPLES.md** (36 KB): 10 detailed usage examples from basic to advanced
- **Test suite**: 100+ tests demonstrating all features

---

## Cross-System Integration (Future)

### Potential Integration Points
1. **Quantum-Guided Moves**: Use QCPP's QCP in UBF's move evaluation quantum factor
2. **Stability Validation**: Validate UBF conformations with QCPP stability predictions
3. **Hybrid Energy**: Combine consciousness-based (UBF) and quantum-based (QCPP) scoring
4. **THz + Trajectory**: Merge QCPP THz spectra with UBF conformational trajectories
5. **Shared Physics**: Both systems use 40 Hz resonance, phi patterns, quantum mechanics

### Current Status
Systems are **independent** but designed for **future integration**. QCPP focuses on stability/validation, UBF focuses on conformational exploration.

---

## Dependencies & Environment

### QCPP Dependencies
- numpy, scipy, pandas, biopython, matplotlib, scikit-learn
- Install: `pip install -e .`
- Python ≥3.8 (≤3.12 recommended for BioPython wheels)

### UBF Dependencies (PyPy-Compatible)
- **Pure Python only**: pytest, dataclasses, typing
- Install: `pip install -r ubf_protein/requirements.txt`
- Python ≥3.8 or PyPy ≥3.8 (PyPy recommended for 2-5x speedup)

### Windows-Specific Setup

#### BioPython (QCPP)
BioPython requires compilation on Windows for Python 3.13+:
1. **Use Python 3.12** (recommended - pre-built wheels)
2. **Install C++ Build Tools** from https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. **Use Conda**: `conda install -c conda-forge biopython`

#### PyPy (UBF)
```bash
# Download from https://www.pypy.org/download.html
# Or use Chocolatey: choco install pypy3
pypy3 -m venv pypy_env
pypy_env\Scripts\activate
pip install -r ubf_protein\requirements.txt
```

---

## Status Summary

### QCPP System
- **Status**: Operational with experimental validation
- **Tests**: Validation through experimental comparison
- **Docs**: Inline documentation

### UBF System
- **Status**: ✅ **COMPLETE** - All 16 tasks finished
- **Tests**: 100+ tests, >90% coverage, all passing ✅
- **Docs**: 91.8 KB comprehensive documentation (README, API, Examples)
- **Performance**: All benchmarks passing ✅
- **Production**: Ready with checkpoint/resume, visualization, error handling