---
inclusion: always
---

# Technical Stack

## Language & Runtime

- **Python 3.8+** (QCPP system and UBF system)
- **PyPy 3.8+** (recommended for UBF - 2-5x speedup)
- Requires Python ≥3.8 for type hints and modern features

---

## QCPP System Dependencies

### Scientific Computing
- `numpy>=1.20.0` - Array operations, mathematical functions
- `scipy>=1.7.0` - Statistical analysis, signal processing, optimization
- `pandas>=1.2.0` - Data structures and analysis

### Bioinformatics
- `biopython>=1.79` - PDB file parsing, protein structure manipulation
- Custom DSSP implementation (`simple_quantum_dssp.py`)

### Visualization
- `matplotlib>=3.4.0` - Plotting and visualization

### Machine Learning (validation only)
- `scikit-learn>=0.24.0` - Correlation analysis, metrics
- `statsmodels` - Statistical modeling

---

## UBF System Dependencies

### Core (PyPy-Compatible)
- **Pure Python only** - No NumPy, SciPy, or C-extensions for optimal JIT
- `pytest>=7.0.0` - Testing framework (100+ tests)
- `dataclasses>=0.6` - Data model support
- `typing>=3.7.4` - Enhanced typing

### Performance Characteristics
- **Decision latency**: <2ms per move evaluation (target), 0.5-1.5ms typical
- **Memory retrieval**: <10μs (target), 2-8μs typical
- **Agent memory**: <50MB with 50 memories, 15-30MB typical
- **Throughput**: 100 agents × 5000 conformations < 2 minutes
- **PyPy speedup**: 2-5x faster than CPython

### Architecture Principles
- **SOLID design** - Interface-driven with dependency inversion
- **Mapless navigation** - O(1) move generation, no spatial maps
- **Pure Python** - Optimized for PyPy JIT compilation
- **Type-safe** - Comprehensive type hints throughout
- **Immutable models** - Thread-safe data structures

---

## Project Structure

```
/
# QCPP System (Root)
├── protein_predictor.py          # QCP calculations, coherence analysis
├── qc_pipeline.py                # Pipeline orchestration
├── validation_framework.py       # Experimental validation
├── quantum_utils.py              # Quantum mechanics utilities
├── stability_predictor.py        # Stability calculations
├── simple_quantum_dssp.py        # Secondary structure
├── metrics_utils.py              # Evaluation metrics
├── compare_predictions.py        # Experimental comparison
├── run_analysis.py               # Main analysis entry
├── setup.py                      # Package config
├── quantum_coherence_proteins/   # QCPP data
│   ├── pdb_files/               # PDB structures
│   └── results/                 # Analysis outputs

# UBF System (ubf_protein/)
└── ubf_protein/                  # UBF system root
    ├── interfaces.py             # SOLID interfaces (11 interfaces)
    ├── models.py                 # Immutable data models (10+ models)
    ├── config.py                 # System configuration
    ├── consciousness.py          # Consciousness coordinate system
    ├── behavioral_state.py       # Behavioral state derivation
    ├── memory_system.py          # Memory and shared pool
    ├── mapless_moves.py          # O(1) move generation & evaluation
    ├── local_minima_detector.py  # Stuck detection & escape
    ├── structural_validation.py  # Validation & repair
    ├── physics_integration.py    # Quantum calculators
    ├── protein_agent.py          # Autonomous agent
    ├── multi_agent_coordinator.py # Multi-agent orchestration
    ├── adaptive_config.py        # Size-based adaptation
    ├── checkpoint.py             # Checkpoint & resume (13 tests)
    ├── visualization.py          # Trajectory & landscape export (7 tests)
    ├── run_single_agent.py       # Single agent CLI
    ├── run_multi_agent.py        # Multi-agent CLI
    ├── benchmark.py              # Performance benchmarks
    ├── validate.py               # Structure validation
    ├── requirements.txt          # PyPy-compatible deps
    ├── README.md                 # System overview (18 KB)
    ├── API.md                    # API reference (37 KB)
    ├── EXAMPLES.md               # 10 usage examples (36 KB)
    └── tests/                    # Test suite (100+ tests)
        ├── test_consciousness.py
        ├── test_behavioral_state.py
        ├── test_memory_system.py
        ├── test_local_minima_detector.py
        ├── test_structural_validation.py
        ├── test_physics_integration.py
        ├── test_protein_agent.py
        ├── test_multi_agent_coordinator.py
        ├── test_checkpoint.py        # 13 tests
        ├── test_visualization.py     # 7 tests
        └── test_helpers.py
```

---

## Key Constants

### QCPP Constants
- **Golden Ratio (φ)**: `1.618033988749895`
- **Planck's Constant (ℏ)**: `1.0545718e-34` J·s
- **Base Energy**: `4.0` - QCP baseline
- **Gamma Frequency**: `40 Hz` - Resonance coupling

### UBF Constants (config.py)
- **Consciousness Frequency Range**: `3.0-15.0 Hz`
- **Consciousness Coherence Range**: `0.2-1.0`
- **Memory Significance Threshold**: `0.3` (individual), `0.7` (shared)
- **Max Memories Per Agent**: `50`
- **Max Shared Memory Pool**: `10000`
- **Consciousness Update Rules**: Success (+0.5 Hz, +0.05 coh), Failure (-0.3 Hz, -0.03 coh)
- **Agent Diversity Profiles**: Cautious (33%), Balanced (34%), Aggressive (33%)

---

## Common Commands

### QCPP System
```bash
# Installation
pip install -e .

# Run analysis
python run_analysis.py

# Compare with experimental
python compare_predictions.py

# Programmatic usage
from qc_pipeline import QCProteinPipeline
pipeline = QCProteinPipeline(data_dir="quantum_coherence_proteins")
results = pipeline.run_complete_analysis(pdb_ids=["1UBQ", "1LYZ"])
```

### UBF System
```bash
# Installation (PyPy recommended)
pypy3 -m venv pypy_env
source pypy_env/bin/activate  # Windows: pypy_env\Scripts\activate
pip install -r ubf_protein/requirements.txt

# Run single agent
python ubf_protein/run_single_agent.py --sequence ACDEFGH --iterations 1000

# Run multi-agent
python ubf_protein/run_multi_agent.py --sequence ACDEFGH --agents 10 --iterations 500

# Run benchmark
python ubf_protein/benchmark.py --agents 100 --iterations 5000

# Run tests (all 100+)
pytest ubf_protein/tests/ -v

# Run specific test suite
pytest ubf_protein/tests/test_checkpoint.py -v        # 13 tests
pytest ubf_protein/tests/test_visualization.py -v     # 7 tests

# Programmatic usage
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    enable_checkpointing=True
)
coordinator.initialize_agents(count=10, diversity_profile="balanced")
results = coordinator.run_parallel_exploration(iterations=500)
```

---

## Build System

### QCPP
- Package: `quantum_protein_predictor` v0.1.0
- Uses standard setuptools
- License: MIT

### UBF
- Package: `ubf_protein` v0.1.0 (editable install)
- Pure Python for PyPy compatibility
- License: MIT
- Development status: Production/Stable

---

## Testing

### QCPP
- No formal test suite
- Validation through experimental comparison
- Visual inspection of plots

### UBF (Comprehensive)
- **100+ tests** across 14 test files
- **>90% code coverage**
- Test types:
  - Unit tests: 80+ tests (consciousness, memory, moves, physics)
  - Integration tests: 30+ tests (agent, coordinator, checkpoint)
  - Performance tests: 10+ benchmarks (latency, memory, throughput)
- All tests passing ✅
- CI/CD ready

---

## Performance Targets & Actuals

### UBF System Benchmarks

| Metric | Target | Typical | Status |
|--------|--------|---------|--------|
| Move evaluation | <2ms | 0.5-1.5ms | ✅ PASS |
| Memory retrieval | <10μs | 2-8μs | ✅ PASS |
| Agent memory | <50MB (50 memories) | 15-30MB | ✅ PASS |
| Multi-agent throughput | 100 agents × 5K conf < 2min | 60-90s | ✅ PASS |
| PyPy speedup | ≥2x vs CPython | 2-5x | ✅ PASS |

### QCPP System
- Runtime: 2-15 minutes per protein (depends on size)
- Memory: ~480MB for 10,000 agents
- Scalability: Linear with agent count

---

## Windows-Specific Setup

### BioPython Installation (QCPP)
BioPython requires compilation on Windows for Python 3.13+. Options:
1. **Use Python 3.12** (recommended - pre-built wheels)
2. **Install C++ Build Tools** from https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. **Use Conda**: `conda install -c conda-forge biopython`

### PyPy Installation (UBF)
```bash
# Download PyPy from https://www.pypy.org/download.html
# Or use Chocolatey:
choco install pypy3

# Create virtual environment
pypy3 -m venv pypy_env
pypy_env\Scripts\activate
pip install -r ubf_protein\requirements.txt
```

---

## Integration Possibilities

### Future: QCPP + UBF Integration
1. **Quantum-Guided Exploration**: Use QCPP's QCP values in UBF's move evaluation
2. **Stability Validation**: Validate UBF conformations with QCPP stability predictions
3. **Hybrid Scoring**: Combine consciousness-based and quantum-based metrics
4. **THz + Trajectory**: Merge THz spectra with conformational trajectories
5. **Shared Physics**: Use QCPP's resonance coupling in UBF's physics integration

Current Status: Systems are independent but designed for future integration
