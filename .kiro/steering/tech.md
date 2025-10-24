---
inclusion: always
---

# Technical Stack

## Language & Runtime

- **Python 3.8+** (primary implementation language)
- Requires Python >=3.8 for type hints and modern features

## Core Dependencies

### Scientific Computing
- `numpy>=1.20.0` - Array operations, mathematical functions
- `scipy>=1.7.0` - Statistical analysis, signal processing, optimization
- `pandas>=1.2.0` - Data structures and analysis

### Bioinformatics
- `biopython>=1.79` - PDB file parsing, protein structure manipulation
- Custom DSSP implementation (`simple_quantum_dssp.py`) for secondary structure

### Visualization
- `matplotlib>=3.4.0` - Plotting and visualization

### Machine Learning (for validation only)
- `scikit-learn>=0.24.0` - Correlation analysis, metrics
- `statsmodels` - Statistical modeling and hypothesis testing

## Project Structure

```
/
├── protein_predictor.py          # Main predictor class (QuantumCoherenceProteinPredictor)
├── qc_pipeline.py                # Pipeline orchestration (QCProteinPipeline)
├── validation_framework.py       # Experimental validation (QuantumProteinValidator)
├── quantum_utils.py              # Quantum mechanics utilities
├── stability_predictor.py        # Stability calculation methods
├── simple_quantum_dssp.py        # Secondary structure assignment
├── metrics_utils.py              # Evaluation metrics
├── compare_predictions.py        # Experimental data comparison
├── run_analysis.py               # Main analysis script
├── setup.py                      # Package configuration
├── quantum_coherence_proteins/   # Data directory
│   ├── pdb_files/               # Downloaded PDB structures
│   └── results/                 # Analysis outputs (JSON, plots)
└── docs/                        # Documentation and papers
```

## Key Constants

- **Golden Ratio (φ)**: `1.618033988749895` - Used throughout for phi-based calculations
- **Planck's Constant (ℏ)**: `1.0545718e-34` J·s - Quantum mechanics
- **Base Energy**: `4.0` - QCP calculation baseline
- **Gamma Frequency**: `40 Hz` - Resonance coupling frequency

## Common Commands

### Installation
```bash
pip install -e .
```

### Run Analysis
```bash
python run_analysis.py
```

### Compare with Experimental Data
```bash
python compare_predictions.py
```

### Run Pipeline Programmatically
```python
from qc_pipeline import QCProteinPipeline

pipeline = QCProteinPipeline(data_dir="quantum_coherence_proteins")
results = pipeline.run_complete_analysis(
    pdb_ids=["1UBQ", "1LYZ"],
    chain_id='A',
    simulate_validation=False
)
```

## Build System

Uses standard Python setuptools:
- Package name: `quantum_protein_predictor`
- Version: `0.1.0`
- Development status: Alpha
- License: MIT

## Testing

No formal test suite currently implemented. Validation is performed through:
- Comparison with experimental stability data (`experimental_stability.csv`)
- Visual inspection of generated plots
- Correlation analysis between predictions and experimental metrics

## Performance Characteristics

- **Memory**: ~48 KB per agent, ~480 MB for 10,000 agents
- **Speed**: ~2.4ms per step, ~41.7 steps/second per agent
- **Scalability**: Linear scaling with number of agents
- **Runtime**: 2-15 minutes for typical proteins (depends on agent count)
