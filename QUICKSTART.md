# Quick Start Guide - Protein Structure Prediction

**PRIMARY MODULES: Quantum Refinement Engine + Real RMSD Calculations**

---

## 🚀 Installation

```bash
# Install dependencies
pip install -e .

# For UBF system (optional PyPy for 2-5x speedup)
pip install -r ubf_protein/requirements.txt
```

---

## 📊 Basic Usage

### 1. Test Single Protein (PRIMARY MODULE)

```bash
# Test with quantum refinement (recommended)
python test_protein.py --pdb 1UBQ --enable-refinement

# Quick test on small protein
python test_protein.py --quick

# Test custom sequence
python test_protein.py --sequence ACDEFGHIKL

# See all available proteins
python test_protein.py --list
```

**Output**:
- Real RMSD calculations with Kabsch alignment
- Quantum refinement RMSD improvement metrics
- Energy landscape analysis
- Geometric attractor patterns
- JSON results in `results/test_results/`

---

### 2. Systematic Testing Campaign (100+ Proteins)

```bash
# Test first 10 proteins with quantum refinement
python systematic_protein_testing.py --start 0 --count 10

# Test specific protein
python systematic_protein_testing.py --protein 1UBQ

# Quick test mode (fewer configurations)
python systematic_protein_testing.py --quick

# Resume interrupted campaign
python systematic_protein_testing.py --resume
```

**Output**:
- `systematic_test_results/campaign_results.json` - All test data
- `systematic_test_results/campaign_summary.json` - Statistical analysis
- `systematic_test_results/campaign_report.txt` - Human-readable summary

**Test Configurations** (6 per protein):
1. Base optimal + quantum refinement
2. Mediators + quantum refinement
3. Geometric targeting + quantum refinement
4. **Full features + quantum refinement** (most comprehensive)
5. High agent count + quantum refinement
6. High iterations + quantum refinement

---

### 3. Legacy QCPP-only Analysis

```bash
# Run QCPP stability analysis only (without UBF/refinement)
python run_analysis.py
```

⚠️ **Note**: For production testing, use `test_protein.py` or `systematic_protein_testing.py` instead.

---

## 📈 Example Workflows

### Workflow 1: Quick Single Protein Test
```bash
# Test Ubiquitin with all features
python test_protein.py --pdb 1UBQ --enable-refinement

# Expected output:
# - RMSD: ~7-10Å (research phase)
# - Quantum refinement improvement: 45-58%
# - Energy: -100 to -250 kcal/mol (negative = folded)
# - Runtime: ~40-60 seconds
```

### Workflow 2: Systematic Robustness Testing
```bash
# Test 10 proteins systematically
python systematic_protein_testing.py --start 0 --count 10

# Expected output:
# - 60 total test configurations (6 per protein)
# - Real RMSD calculations where native structures available
# - Statistical analysis of all configurations
# - Runtime: ~30-60 minutes for 10 proteins
```

### Workflow 3: Custom Protein Testing
```bash
# Test your own sequence
python test_protein.py --sequence "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTYPYDVPDYAG"

# Note: Without PDB ID, RMSD will be energy-based estimate only
```

---

## 📊 Understanding Results

### RMSD (Root Mean Square Deviation)
- **Real RMSD**: Kabsch alignment with native PDB structure (CA atoms only)
- **Estimated RMSD**: Energy-based fallback when native structure unavailable
- **Quality Thresholds** (literature standards for comparison):
  - Excellent: <2Å (AlphaFold level)
  - Good: 2-4Å
  - Acceptable: 4-5Å
  - Research Phase: 7-10Å (current system)

### Energy
- **Negative values**: Indicate properly folded structures
- **Typical range**: -50 to -300 kcal/mol for small/medium proteins
- **Very positive**: May indicate steric clashes or unfolded states

### Quantum Refinement Improvement
- **Typical**: 45-58% RMSD reduction
- **Mechanism**: Two-stage optimization (global fold → quantum refinement)
- **Features**: Distance restraints, hydrophobic packing, loop refinement

---

## 🎯 Tips for Best Results

### 1. Enable Quantum Refinement
Always use `--enable-refinement` for production testing:
```bash
python test_protein.py --pdb 1UBQ --enable-refinement
```

### 2. Provide Native PDB Structure
Real RMSD requires native structure for comparison:
```bash
python test_protein.py --pdb 1UBQ  # Auto-downloads 1UBQ.pdb
```

### 3. Use Systematic Testing for Robustness
Test multiple configurations to validate system behavior:
```bash
python systematic_protein_testing.py --protein 1UBQ
```

### 4. Check for Errors
Review results JSON for validation metrics:
```json
{
  "exploration_results": {
    "final_rmsd": 8.5,  // Real RMSD (good!)
    "estimated_rmsd": 10.0,  // Fallback estimate
    "best_energy": -150.5  // Negative = folded
  },
  "refinement_result": {
    "rmsd_improvement": 4.2  // Angstroms improved
  }
}
```

---

## 🔧 Troubleshooting

### Issue: "RMSD calculation failed"
**Cause**: Coordinate length mismatch or alignment failure  
**Solution**: This is expected for diverse conformations. System falls back to energy-based estimate.

### Issue: "Could not download PDB file"
**Cause**: Network issue or invalid PDB ID  
**Solution**: Check PDB ID at https://www.rcsb.org/ or provide local PDB file

### Issue: "UnicodeEncodeError in reports"
**Cause**: Windows encoding issue (fixed in latest version)  
**Solution**: Ensure you're using the latest code with UTF-8 encoding

### Issue: Very high positive energy
**Cause**: Steric clashes or unfolded structure  
**Solution**: This is expected during exploration. Quantum refinement addresses this.

---

## 📚 Next Steps

- Read full documentation: `README.md`
- Explore UBF system: `ubf_protein/README.md`
- Review validation results: `systematic_test_results/campaign_report.txt`
- Check test suite: `pytest ubf_protein/tests/`

---

## 🤝 Support

For questions or issues:
1. Check documentation in `docs/`
2. Review test examples in `ubf_protein/EXAMPLES.md`
3. Examine systematic test results for reference

---

**Last Updated**: November 9, 2025  
**Primary Modules**: test_protein.py, systematic_protein_testing.py  
**Key Features**: Quantum Refinement Engine, Real RMSD calculations, UTF-8 reports
