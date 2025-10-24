# Quantum Protein Predictor - AI Coding Guidelines

## Architecture Overview
This project predicts protein stability using quantum coherence principles and golden ratio patterns. Core components:
- `QuantumCoherenceProteinPredictor`: Calculates QCP values, field coherence, and predicts THz spectra
- `QCProteinPipeline`: Orchestrates analysis workflow from PDB loading to validation
- Data flows: PDB → QCP calculation → coherence analysis → stability prediction → THz spectrum

## Key Patterns & Conventions

### QCP Formula Implementation
Always use the exact formula: `qcp = 4.0 + (2**n * (phi**l) * m)`
- `n`: Structural hierarchy (0=coil, 1=helix, 2=sheet, 3=phi-based bends)
- `l`: Neighbor count (scaled 1-3)
- `m`: Hydrophobicity (normalized -1 to 1)

### Golden Ratio Integration
- Phi angle: `2 * π / φ ≈ 137.5°` (radians: `phi_angle = 2 * π / phi`)
- Use phi harmonics: `[φ⁰, φ¹, φ², φ³, φ⁴]` for frequency calculations
- Distance matching: Check against `3.8 * φ^n` Å

### Secondary Structure Handling
Prefer `SimpleQuantumDSSP` for SS assignment. Fallback to distance-based approximation if DSSP fails:
```python
# Helical: i,i+3 distances ~5Å
for i in range(len-3):
    if 4.5 < distances[i][i+3] < 5.5:
        residues[i:i+4] = ['H'] * 4
```

### File Organization
- Results: `quantum_coherence_proteins/results/{pdb_id}_analysis.json`
- Plots: `quantum_coherence_proteins/results/plots/`
- PDB files: `quantum_coherence_proteins/pdb_files/`

## Development Workflow

### Running Analysis
```bash
python run_analysis.py  # Full pipeline on test proteins
```

### Adding New Proteins
1. Add PDB ID to `test_proteins` list in `run_analysis.py`
2. Pipeline auto-downloads via `PDBList.retrieve_pdb_file()`
3. Results saved automatically to JSON

### Validation Integration
Use `QCProteinPipeline.run_validation_pipeline()` with experimental data from `experimental_stability.csv`. Correlate predicted stability with THz spectra and experimental measurements.

### Visualization Standards
- QCP/coherence plots: Line plots with residue ID x-axis
- THz spectra: Bar plots with phi harmonic markers
- Phi angles: Histograms with 137.5° and 222.5° reference lines

## Dependencies & Environment
- Core: numpy, scipy, biopython, pandas, matplotlib, scikit-learn
- Install: `pip install -e .`
- Python ≥3.8 required

## Common Pitfalls
- Always check `res.has_id('CA')` before accessing CA coordinates
- Handle DSSP failures gracefully with fallback SS calculation
- Normalize coherence values (subtract min) before averaging
- Use absolute paths for file operations in pipeline