# CLI Reference - Protein Structure Prediction

This document provides complete reference for command-line tools in the Protein Prediction Platform.

> **💡 Prefer the Web Interface?** The platform is live at **https://emergentfolds.com** with the same prediction engine. The CLI is ideal for automation, scripting, and batch processing.

## CLI vs Web Interface

| Feature | CLI | Web (emergentfolds.com) |
|---------|-----|-------------------------|
| Single predictions | ✅ | ✅ |
| Batch predictions | ✅ | ✅ (Campaigns) |
| Real-time monitoring | Terminal output | Live charts & WebSocket |
| 3D visualization | Export PDB only | Interactive NGL Viewer |
| Progress tracking | Text-based | Visual progress bars |
| Best for | Automation, scripting | Interactive exploration |

**Both use the same `PredictionRunner` engine**, ensuring identical results.

## Table of Contents

- [test_protein.py](#test_proteinpy---universal-protein-testing)
- [systematic_protein_testing.py](#systematic_protein_testingpy---batch-campaigns)
- [run_analysis.py](#run_analysispy---qcpp-only-analysis)

---

## test_protein.py - Universal Protein Testing

The primary CLI tool for single protein structure prediction. Uses the same `PredictionRunner` as the web interface, ensuring consistent results across both interfaces.

### Basic Usage

```bash
# Test with PDB ID (downloads structure automatically)
python test_protein.py --pdb 1UBQ

# Test with custom amino acid sequence
python test_protein.py --sequence ACDEFGHIKLMNPQRSTVWY

# Quick test on small protein (Villin, 35 residues)
python test_protein.py --quick

# List all available cached PDB files
python test_protein.py --list
```

### Command Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pdb` | string | - | PDB ID to test (e.g., 1UBQ, 1CRN). Auto-downloads if not cached. |
| `--sequence` | string | - | Custom amino acid sequence (1-letter codes) |
| `--agents` | int | auto | Number of exploration agents (auto-configured by protein size) |
| `--iterations` | int | auto | Iterations per agent (auto-configured by protein size) |
| `--target-geometry` | choice | none | Platonic solid geometry for guidance |
| `--enable-mediators` | flag | false | Enable pattern detection mediator agents |
| `--mediator-count` | int | 2 | Number of mediator agents |
| `--enable-refinement` | flag | false | Enable quantum refinement (two-stage optimization) |
| `--enable-hierarchical` | flag | false | Enable hierarchical folding with progressive anchoring |
| `--list` | flag | - | List available test proteins and exit |
| `--quick` | flag | - | Quick test mode (reduced iterations on small protein) |

### Target Geometry Options

The `--target-geometry` option guides exploration toward specific Platonic solid geometries:

| Value | Description |
|-------|-------------|
| `none` | No geometric targeting (default) |
| `tetrahedron` | 4-faced regular polyhedron |
| `cube` | 6-faced regular polyhedron |
| `octahedron` | 8-faced regular polyhedron |
| `dodecahedron` | 12-faced regular polyhedron |
| `icosahedron` | 20-faced regular polyhedron |

**Usage:**
```bash
python test_protein.py --pdb 1UBQ --target-geometry icosahedron
```

**When to use:** Geometric targeting can improve results for proteins with symmetric structures or when exploring golden ratio patterns.

### Quantum Refinement

Quantum refinement applies a second optimization stage using QCPP-derived physics constraints:

```bash
python test_protein.py --pdb 1UBQ --enable-refinement
```

**What it does:**
- Distance restraints from QCPP analysis
- Hydrophobic core packing optimization
- Loop refinement with φ-based dynamics
- Tertiary contact prediction

**Performance:** Typically achieves 45-58% RMSD improvement over base exploration.

**Trade-off:** Adds ~20-40 seconds to prediction time.

### Hierarchical Folding

Hierarchical folding enables progressive search space confinement:

```bash
python test_protein.py --pdb 1UBQ --enable-hierarchical
```

**What it does:**
- Identifies stable secondary structure elements early
- "Anchors" these regions to reduce search space
- Focuses remaining exploration on flexible regions
- Progressively constrains the conformational search

**When to use:** Beneficial for medium-to-large proteins (>50 residues) where the full search space is too large.

### Mediator Agents

Mediator agents perform specialized pattern detection and coordinate information between exploration agents:

```bash
python test_protein.py --pdb 1UBQ --enable-mediators --mediator-count 3
```

**What they do:**
- Detect THz frequency patterns
- Identify geometric relationships
- Relay information between exploration agents
- Detect folding progress patterns

### Auto-Configuration

When `--agents` and `--iterations` are not specified, optimal values are chosen based on sequence length:

| Protein Size | Residues | Agents | Iterations | Category |
|--------------|----------|--------|------------|----------|
| Small | <50 | 15 | 300 | small |
| Medium | 50-99 | 20 | 200 | medium |
| Large | 100-149 | 30 | 250 | large |
| Very Large | 150+ | 50 | 300 | very_large |

### Output

Results are saved to `results/test_results/`:

```
results/test_results/
├── test_1UBQ_results.json      # Comprehensive results
└── ...

results/predicted_structures/
├── 1UBQ_predicted.pdb          # Predicted structure
└── ...
```

### Complete Examples

```bash
# Basic prediction with RMSD validation
python test_protein.py --pdb 1UBQ

# Full-featured prediction
python test_protein.py --pdb 1UBQ \
    --enable-refinement \
    --enable-mediators \
    --enable-hierarchical \
    --target-geometry icosahedron

# Custom configuration
python test_protein.py --pdb 1LYZ \
    --agents 30 \
    --iterations 500 \
    --enable-refinement

# Custom sequence (no native structure for RMSD)
python test_protein.py --sequence MKFLILLFNILCLFPVLAADNHGVGPQGAS

# Quick validation test
python test_protein.py --quick
```

---

## systematic_protein_testing.py - Batch Campaigns

Test multiple proteins systematically with various configurations.

### Basic Usage

```bash
# Test first 10 proteins
python systematic_protein_testing.py --start 0 --count 10

# Test specific protein
python systematic_protein_testing.py --protein 1UBQ

# Resume interrupted campaign
python systematic_protein_testing.py --resume
```

### Command Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start` | int | 0 | Starting index in protein list |
| `--count` | int | 10 | Number of proteins to test |
| `--protein` | string | - | Test a specific protein by PDB ID |
| `--resume` | flag | - | Resume from last checkpoint |

### Test Configurations

Each protein is tested with 6 different configurations:

1. **Base Optimal** + quantum refinement
2. **Mediators** + quantum refinement
3. **Geometric Targeting** + quantum refinement
4. **Full Features** (comprehensive) + quantum refinement
5. **High Agent Count** + quantum refinement
6. **High Iterations** + quantum refinement

### Output

Results are saved to `results/systematic_testing/`:

```
results/systematic_testing/
├── campaign_YYYYMMDD_HHMMSS/
│   ├── summary.json
│   ├── 1UBQ/
│   │   ├── base_optimal.json
│   │   ├── mediators.json
│   │   └── ...
│   └── ...
└── checkpoints/
```

---

## run_analysis.py - QCPP-Only Analysis

⚠️ **Legacy Module**: For QCPP-only analysis without UBF exploration. For production testing, use `test_protein.py` instead.

### Usage

```bash
python run_analysis.py
```

Runs QCPP analysis on predefined test proteins and outputs:
- QCP (Quantum Coherence Potential) values
- THz spectrum analysis
- Golden ratio pattern detection
- Field coherence metrics

### Output

Results saved to `quantum_coherence_proteins/results/`.

---

## Environment Setup

### Prerequisites

```bash
# Install main dependencies
pip install -e .

# For UBF system
pip install -r ubf_protein/requirements.txt
```

### PDB Cache

PDB files are automatically downloaded and cached to `pdb_cache/`:

```
pdb_cache/
├── pdb1ubq.ent
├── pdb1crn.ent
└── ...
```

### PyPy Optimization

For 2-5x speedup, run with PyPy:

```bash
pypy3 test_protein.py --pdb 1UBQ
```

---

## Troubleshooting

### "No PDB files found"

PDB files are auto-downloaded. If download fails:
1. Check internet connection
2. Verify PDB ID is valid at https://www.rcsb.org/
3. Manually download to `pdb_cache/`

### "BioPython not found"

```bash
pip install biopython
```

On Windows with Python 3.13+, you may need Visual C++ Build Tools or use Python 3.12.

### "Module not found: ubf_protein"

Ensure you're running from the project root directory:
```bash
cd /path/to/PP
python test_protein.py --pdb 1UBQ
```

### Slow Performance

- Use PyPy instead of CPython for 2-5x speedup
- Reduce `--agents` and `--iterations` for quick tests
- Use `--quick` flag for rapid validation

---

## See Also

- [User Guide](USER_GUIDE.md) - Web interface documentation
- [API Reference](API.md) - REST API documentation
- [Setup Guide](SETUP.md) - Installation instructions
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
