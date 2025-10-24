---
inclusion: always
---

# Project Organization

## Core Architecture

The project follows a **pipeline architecture** with three main layers:

### 1. Prediction Layer
- `protein_predictor.py` - Core prediction algorithms
  - `QuantumCoherenceProteinPredictor` class
  - QCP calculation, coherence metrics, phi-angle analysis
  - THz spectrum prediction

### 2. Pipeline Layer
- `qc_pipeline.py` - Orchestration and workflow
  - `QCProteinPipeline` class
  - Protein download/loading, batch analysis
  - Visualization generation, result comparison

### 3. Validation Layer
- `validation_framework.py` - Experimental validation
  - `QuantumProteinValidator` class
  - THz experiment design and simulation
  - Correlation with experimental stability data

## Module Responsibilities

### Prediction Modules
- **protein_predictor.py**: Main predictor with QCP, coherence, stability calculations
- **quantum_utils.py**: Reusable quantum mechanics functions (resonance coupling, phi matching)
- **stability_predictor.py**: Stability-specific calculation methods
- **simple_quantum_dssp.py**: Secondary structure assignment (helix, sheet, coil)

### Analysis Modules
- **qc_pipeline.py**: End-to-end workflow orchestration
- **validation_framework.py**: Experimental validation and correlation analysis
- **compare_predictions.py**: Compare predictions with experimental data
- **metrics_utils.py**: Evaluation metrics and scoring

### Execution
- **run_analysis.py**: Main entry point for batch analysis
- **setup.py**: Package configuration and dependencies

## Data Flow

```
PDB File → load_protein() → calculate_qcp() → calculate_field_coherence() 
  → predict_stability() → visualize_results() → save JSON/plots
```

## File Naming Conventions

### Analysis Results
- `{protein_id}_analysis.json` - Complete analysis results
- `{protein_id}_qcp.png` - QCP distribution plot
- `{protein_id}_coherence.png` - Coherence distribution plot
- `{protein_id}_thz_spectrum.png` - Predicted THz spectrum
- `{protein_id}_phi_angles.png` - Phi angle distribution
- `protein_comparison_{metric}.png` - Multi-protein comparison
- `thz_experiment_protocol.json` - Experimental protocol design
- `validation_results.json` - Validation analysis results

### Data Files
- `experimental_stability.csv` - Experimental stability measurements
  - Required columns: `protein_id`, `melting_temp_C`, `delta_G_unfolding_kcal_mol`

## Code Organization Patterns

### Class Structure
All main classes follow this pattern:
1. `__init__()` - Initialize constants (phi, base_energy, etc.)
2. `load_*()` - Data loading methods
3. `calculate_*()` - Core calculation methods
4. `predict_*()` - Prediction methods
5. `analyze_*()` - Analysis methods
6. `visualize_*()` - Visualization methods

### Data Storage
- **Internal state**: Stored as class attributes (e.g., `self.qcp_values`, `self.coherence_metric`)
- **Results**: Saved as JSON files with nested dictionaries
- **Visualizations**: Saved as PNG files in `results/plots/`

### Error Handling
- Uses `try/except` blocks with informative error messages
- Warnings for missing data (e.g., DSSP calculation failures)
- Graceful degradation (e.g., fallback to simplified secondary structure)

## Configuration

### Pipeline Configuration
```python
pipeline = QCProteinPipeline(data_dir="quantum_coherence_proteins")
```

### Predictor Configuration
```python
predictor = QuantumCoherenceProteinPredictor()
# Constants are initialized in __init__:
# - phi = 1.618...
# - base_energy = 4.0
# - phi_harmonics = [φ^0, φ^1, φ^2, φ^3, φ^4]
```

### Validator Configuration
```python
validator = QuantumProteinValidator(predictor=predictor)
# Automatically inherits phi and base_frequency from predictor
```

## Extension Points

### Adding New Metrics
1. Add calculation method to `protein_predictor.py`
2. Update `run_full_analysis()` to include new metric
3. Add visualization in `visualize_results()`
4. Update JSON output structure

### Adding New Validation Methods
1. Add method to `validation_framework.py`
2. Update `run_validation_pipeline()` in `qc_pipeline.py`
3. Add correlation analysis in `correlate_with_stability()`

### Supporting New Protein Formats
1. Add parser in `load_protein()` method
2. Ensure CA atom extraction works correctly
3. Update secondary structure calculation if needed
