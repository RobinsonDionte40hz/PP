# Easy Protein Testing Guide 🧬

## For Users Unfamiliar with the Tech

### What This Does
Tests any protein to predict its 3D structure and stability using AI + physics.

### Quick Start (3 Commands!)

```bash
# 1. See available proteins
python test_protein.py --list

# 2. Test a protein (example: Ubiquitin)
python test_protein.py --pdb 1UBQ

# 3. Done! Check results in test_1UBQ_results.json
```

### Available Test Proteins

| Protein | Command | Size | Time |
|---------|---------|------|------|
| **Ubiquitin** (small regulatory protein) | `python test_protein.py --pdb 1UBQ` | 76 residues | ~12s |
| **Crambin** (plant protein) | `python test_protein.py --pdb 1CRN` | 46 residues | ~7s |
| **Villin** (QUICK TEST) | `python test_protein.py --quick` | 35 residues | ~4s |
| **Lysozyme** (enzyme) | `python test_protein.py --pdb 1LYZ` | 129 residues | ~35s |
| **BBL** (synthetic protein) | `python test_protein.py --pdb 2MR9` | 47 residues | ~7s |

### Understanding the Results

#### Energy (kcal/mol)
- **-300 to -350:** Excellent (very stable)
- **-250 to -300:** Good (stable)
- **-200 to -250:** Fair (moderately stable)
- **Above -200:** Needs improvement

#### RMSD (Å - Angstroms)
How close to real structure:
- **< 6 Å:** GOOD (accurate prediction)
- **6-8 Å:** FAIR (reasonable prediction)
- **> 8 Å:** Needs improvement

#### RMSE (Temperature & ΔG)
How well physics model predicts stability:
- **GOOD:** < 20% error
- **FAIR:** 20-30% error
- **Needs improvement:** > 30% error

### Advanced Options (Optional)

```bash
# Test any protein from Protein Data Bank
python test_protein.py --pdb 2LZM

# Test custom sequence
python test_protein.py --sequence ACDEFGHIKLMNPQRSTVWY

# Use more agents for better quality (slower)
python test_protein.py --pdb 1UBQ --agents 50

# More iterations for thoroughness (slower)
python test_protein.py --pdb 1UBQ --iterations 500
```

### What the System Does Automatically

✅ **Auto-configures settings** based on protein size:
- Small proteins (< 50 residues): 15 agents
- Medium proteins (50-100): 20 agents (optimal)
- Large proteins (100-150): 30 agents
- Very large (> 150): 50 agents

✅ **Downloads PDB files** automatically (cached for reuse)

✅ **Validates** against experimental data (if available)

✅ **Saves results** as JSON file for later analysis

### Example Output

```
======================================================================
RESULTS SUMMARY
======================================================================

🔬 STRUCTURAL EXPLORATION:
  - Best Energy: -298.72 kcal/mol
  - Estimated RMSD: 6.54 Å (FAIR)
  - Conformations: 4,000
  - Time: 11.7s
  - Throughput: 341.9 conf/s

📊 QCPP INTEGRATION:
  - Total Analyses: 666
  - Cache Hit Rate: 30.0%
  - Avg Analysis Time: 35.33ms

🎯 PREDICTION ACCURACY:
  - Temperature RMSE: 5.44 °C (GOOD)
  - ΔG RMSE: 0.71 kcal/mol (GOOD)
  - Overall Quality: GOOD

======================================================================
✅ TEST SUCCESSFUL!
   Structure prediction shows promising results
✅ PREDICTION ACCURACY VALIDATED!
   QCPP physics model shows good agreement with experimental data
======================================================================
```

### Troubleshooting

**Problem:** "ModuleNotFoundError"
**Solution:** Install requirements: `pip install -r requirements_qcpp.txt`

**Problem:** "PDB download failed"
**Solution:** Check internet connection, or manually download PDB file to `pdb_cache/`

**Problem:** "Too slow"
**Solution:** Use `--quick` for fast test, or reduce agents: `--agents 10`

**Problem:** "Want better results"
**Solution:** Increase agents: `--agents 50` or iterations: `--iterations 500`

### Files Created

- `test_<PROTEIN>_results.json` - Detailed results
- `pdb_cache/pdb<ID>.ent` - Downloaded protein structure (reused)

### Compare Multiple Proteins

```bash
# Test all available proteins
python test_protein.py --pdb 1UBQ
python test_protein.py --pdb 1CRN
python test_protein.py --pdb 1VII
python test_protein.py --pdb 2MR9
python test_protein.py --pdb 1LYZ

# Results saved as: test_1UBQ_results.json, test_1CRN_results.json, etc.
```

### What Makes This Special?

1. **Physics + AI:** Combines quantum mechanics (QCPP) with intelligent agents (UBF)
2. **Automatic tuning:** Optimizes settings for each protein size
3. **Validated:** Results compared against real experimental data
4. **Fast:** ~10-30 seconds for most proteins
5. **No coding needed:** Just run one command!

---

**Need help?** See full documentation in:
- `OPTIMIZATION_COMPLETE_SUMMARY.md` - Technical details
- `AGENT_SCALING_SUMMARY.md` - How agent count affects results
- `test_protein.py` - The actual code (with comments)
