# Geometric Hypothesis Validation - Action Plan Implementation

## Overview

This directory contains scripts to test whether the geometric patterns (φ ratios, symmetry) observed in protein predictions are:
1. **Real physics** - discovered by QCPP quantum integration
2. **Algorithm bias** - imposed by energy function design  
3. **PDB contamination** - analyzed native structures instead of predictions

## Quick Start

### Test 1: Single Protein Quick Validation (RECOMMENDED)

Tests whether φ patterns exist in PREDICTED structures vs NATIVE PDB:

```cmd
python quick_validation_test.py --pdb 1VII --iterations 500
python quick_validation_test.py --pdb 1CD3 --iterations 300
python quick_validation_test.py --pdb 1UBQ --iterations 500 --output results_1UBQ.json
```

**What it does:**
- Loads native PDB structure and analyzes φ/symmetry
- Runs UBF prediction and analyzes PREDICTED structure
- Calculates TRUE RMSD (with optimal superposition)
- Compares Native vs Predicted to detect contamination

**Expected output:**
```
[1/3] NATIVE PDB STRUCTURE
  φ patterns: 13.2%
  Rotational symmetry: 0.945

[2/3] PREDICTED STRUCTURE
  φ patterns: 11.8%  ← SHOULD DIFFER if analyzing predictions correctly
  Rotational symmetry: 0.912

[3/3] TRUE RMSD
  TRUE RMSD: 4.23 Å  ← Validates quality claims

FINDINGS:
✓ CLEAN ANALYSIS: Δφ = -1.4% (analyzing predictions correctly)
✓ φ PATTERNS PRESENT: 11.8% in prediction
✓ EXCELLENT PREDICTION: RMSD = 4.23 Å
```

### Test 2: Comprehensive Validation (Advanced)

Full test suite including ablation studies and random walk baseline:

```cmd
# Full test (4 experiments: native, predicted, ablation, random)
python validate_geometric_hypothesis.py --proteins 1VII --mode full --iterations 500

# Quick test (native + predicted only)
python validate_geometric_hypothesis.py --proteins 1VII 1CD3 --mode quick --iterations 300

# Ablation test (disable QCPP)
python validate_geometric_hypothesis.py --proteins 1UBQ --mode ablation --iterations 500
```

**What it does:**
1. **Native analysis** - Baseline φ/symmetry from PDB
2. **Predicted (QCPP ON)** - UBF with quantum physics integration
3. **Ablation (QCPP OFF)** - UBF without QCPP (tests if QCPP creates patterns)
4. **Random walk** - Pure random baseline

**Expected findings:**
- **IF φ drops >3% without QCPP** → QCPP integration discovers geometric principles ✓
- **IF φ stays similar without QCPP** → Algorithm bias (energy function imposes order) ⚠️
- **IF predicted φ ≈ native φ** → Previous analysis used native PDB (contamination) ⚠️

## Protein Test Suite

### Ordered Proteins (Should show HIGH φ patterns if hypothesis is correct)
```cmd
python quick_validation_test.py --pdb 1VII  # 36 res, villin headpiece
python quick_validation_test.py --pdb 1ROP  # 56 res, repressor of primer
python quick_validation_test.py --pdb 1HIV  # 98 res, HIV protease
python quick_validation_test.py --pdb 1MBN  # 153 res, myoglobin
```

### Disordered Proteins (Should show LOW φ patterns if hypothesis is correct)
```cmd
python quick_validation_test.py --pdb 1CD3  # IDP, 143 res
python quick_validation_test.py --pdb 1F0R  # p53 TAD, 234 res  
python quick_validation_test.py --pdb 1MVF  # α-synuclein, 127 res
python quick_validation_test.py --pdb 2KJ3  # Calmodulin IDP, 79 res
```

### Expected Results (If Hypothesis is Correct)

| Protein Type | Expected φ (Predicted) | Expected Symmetry | Expected RMSD |
|--------------|------------------------|-------------------|---------------|
| Ordered | **>14%** | **>0.95** | **<5 Å** |
| Disordered | **<10%** | **<0.80** | **>8 Å** |

**Current problem:** IDPs show 12-14% φ and 0.95+ symmetry (SAME as ordered!)

This suggests:
- **Scenario A:** Algorithm bias - System imposes geometric order
- **Scenario B:** PDB contamination - Analyzed native structures not predictions
- **Scenario C:** Real geometry - IDPs do have transient geometric organization

## Critical Tests

### Test A: PDB Contamination Check

Compare native vs predicted φ patterns:

```cmd
python quick_validation_test.py --pdb 1VII --output test_A.json
```

**If Δφ < 2%** → Contamination! Re-run geometric analysis on predictions, not native PDB

**If Δφ > 3%** → Clean! Previous analysis was correct

### Test B: QCPP Mechanism Test

Run with and without QCPP integration:

```cmd
python validate_geometric_hypothesis.py --proteins 1UBQ --mode full --iterations 500
```

**If φ drops >3% without QCPP** → QCPP discovers/enforces geometric patterns (validates hypothesis!)

**If φ unchanged without QCPP** → Energy function creates patterns (algorithm bias)

### Test C: Random Baseline

Compare predictions to random walk:

```cmd
python validate_geometric_hypothesis.py --proteins 1VII --mode full
```

**If predicted φ > 1.5× random** → System creates order (but is it real?)

**If predicted φ ≈ random** → Patterns are statistical artifact

### Test D: RMSD Validation

Validate all 20 proteins' RMSD estimates:

```cmd
# Run on all proteins from research report
for pdb in 1VII 1ROP 1UTG 1GB1 1PGB 1HIV 1CHO 3SSI 1MBN 1CD3 1F0R 1MVF 2KJ3 1LMB 1BTA 1RIS 1BPI 2CI2 1UBQ; do
    python quick_validation_test.py --pdb $pdb --output "validation_${pdb}.json"
done
```

**Compare:**
- Estimated RMSD (from report): 3.0-10.0 Å
- True RMSD (from superposition): ?

**If TRUE RMSD >> estimated** → Quality claims are inflated

## Interpreting Results

### Scenario 1: Hypothesis VALIDATED ✓

```
Finding 1: Predicted φ = 15.2%, Native φ = 13.1% (Δφ = +2.1%) → CLEAN
Finding 2: φ drops to 8.3% without QCPP (Δφ = -6.9%) → QCPP MATTERS
Finding 3: RMSD = 3.8 Å → GOOD PREDICTION
Finding 4: Ordered proteins: φ 14-16%, IDPs: φ 8-11% → DISCRIMINATES
```

**Interpretation:** QCPP integration discovers real geometric principles. Hypothesis is correct!

### Scenario 2: Algorithm Bias Confirmed ⚠️

```
Finding 1: Predicted φ = 13.8%, Native φ = 13.2% (Δφ = +0.6%) → SUSPICIOUS
Finding 2: φ only drops to 12.1% without QCPP (Δφ = -1.7%) → QCPP MINIMAL
Finding 3: RMSD = 8.2 Å → POOR PREDICTION but high φ
Finding 4: Both ordered and IDPs: φ 12-14% → NO DISCRIMINATION
```

**Interpretation:** Geometric patterns are artifacts of energy function design. Hypothesis refuted.

### Scenario 3: PDB Contamination Detected ⚠️

```
Finding 1: Predicted φ = 13.3%, Native φ = 13.1% (Δφ = +0.2%) → CONTAMINATED!
Finding 2: Cannot test (contamination invalidates other tests)
Finding 3: Previous geometric analysis used native PDB structures
Finding 4: Need to re-run φ analysis on PREDICTED structures only
```

**Interpretation:** Previous research report analyzed native PDB files, not predictions. Critical methodological flaw.

## Next Steps After Validation

### If Hypothesis is VALIDATED:

1. Expand to 50+ proteins (ordered + disordered)
2. Compare to AlphaFold2/RoseTTAFold predictions
3. Identify φ pattern locations (functional sites?)
4. Write paper for Nature/Science: "Geometric Attractors in Protein Folding"

### If Algorithm Bias is CONFIRMED:

1. Redesign energy function to remove geometric bias
2. Test pure physics-based approaches (MD simulations)
3. Compare UBF to other de novo methods
4. Document as "effective heuristic" not "fundamental physics"
5. Write paper for Computational Biology journal

### If PDB Contamination DETECTED:

1. **URGENT:** Re-run all φ/symmetry analysis on predicted structures
2. Update geometric_integrity_research_report.md
3. Re-analyze all 20 proteins correctly
4. Check if patterns still exist in predictions
5. If patterns vanish → Hypothesis was artifact of methodology

## Files Generated

```
validation_results.json        # Full multi-test results
validation_1VII.json           # Per-protein quick test
test_A.json, test_B.json       # Specific hypothesis tests
```

## Troubleshooting

**Error: "Cannot import UBF system"**
```cmd
pip install -r ubf_protein\requirements.txt
```

**Error: "Cannot import BioPython"**
```cmd
pip install biopython numpy
```

**Error: "PDB download failed"**
- Check internet connection
- Manually download from https://www.rcsb.org/structure/{PDB_ID}
- Place in `pdb_cache/` directory

**Slow execution:**
- Reduce `--iterations` to 200-300 for testing
- Reduce `--agents` to 5-8 for faster (but less thorough) exploration

## Expected Runtime

| Test Type | Protein Size | Iterations | Runtime |
|-----------|--------------|------------|---------|
| Quick validation | Small (<50 res) | 500 | 2-5 min |
| Quick validation | Medium (50-100) | 500 | 5-10 min |
| Quick validation | Large (>100) | 500 | 10-20 min |
| Full validation | Any size | 500 | 4× quick time |

## Contact / Issues

Report issues in the main repository.

---

**Author:** QCPP-UBF Research Team  
**Date:** November 5, 2025  
**Purpose:** Critical validation of geometric attractor hypothesis
