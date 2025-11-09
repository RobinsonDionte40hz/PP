# Quantum Refinement Validation Suite

## Overview

The quantum refinement validation suite provides comprehensive testing of the refinement engine on standard benchmark proteins. This implements **Task 14** of the quantum refinement engine specification.

## Validation Script

**File**: `ubf_protein/validate_quantum_refinement.py`

### Features

- **Automated PDB Download**: Downloads native structures from RCSB PDB
- **Full Pipeline Testing**: Runs exploration → refinement → validation
- **Comprehensive Metrics**: RMSD, GDT-TS, TM-score, energy, timing
- **Component Breakdown**: Separate RMSD for helix, sheet, loop, core
- **Success Criteria Validation**: Checks all 5 success criteria per protein
- **JSON Output**: Saves detailed results for further analysis

### Test Proteins

| PDB ID | Protein | Residues | Target RMSD | Difficulty |
|--------|---------|----------|-------------|------------|
| 1UBQ | Ubiquitin | 76 | <4Å | Medium |
| 1CRN | Crambin | 46 | <3Å | Easy |
| 2MR9 | Villin headpiece | 35 | <3Å | Easy |

### Success Criteria (per Requirement 10.2)

For each protein, the validation checks:

1. ✓ **RMSD < 5Å**: Final structure is near-native
2. ✓ **RMSD Improvement > 50%**: Significant refinement from initial
3. ✓ **Energy < 0 kcal/mol**: Thermodynamically stable (folded)
4. ✓ **GDT-TS > 50**: Correct global fold topology
5. ✓ **Runtime < 5 minutes**: Practical computation time

## Usage

### Run All Proteins

```bash
python ubf_protein/validate_quantum_refinement.py
```

Output will be saved to `quantum_refinement_validation.json`.

### Run Single Protein

```bash
python ubf_protein/validate_quantum_refinement.py --protein 1UBQ
```

### Custom Output File

```bash
python ubf_protein/validate_quantum_refinement.py --output my_results.json
```

### Verbose Logging

```bash
python ubf_protein/validate_quantum_refinement.py --verbose
```

## Output Format

### Console Output

```
======================================================================
Quantum Refinement Validation: 1UBQ (Ubiquitin)
======================================================================

Protein Info:
  Sequence Length:     76 residues
  
Pre-Refinement:
  RMSD:                10.50 Å
  Energy:              45.20 kcal/mol
  GDT-TS:              32.5
  
Post-Refinement:
  RMSD:                3.80 Å
  Energy:              -45.30 kcal/mol
  GDT-TS:              75.2
  TM-Score:            0.820
  
Improvements:
  RMSD Improvement:    63.8%
  Energy Improvement:  90.50 kcal/mol
  
Component RMSD Breakdown:
  Helix:               2.50 Å
  Sheet:               3.20 Å
  Loop:                4.80 Å
  Core:                2.90 Å
  
Refinement Details:
  Quantum Cores:       8
  Distance Restraints: 24
  Tertiary Contacts:   12
  Runtime:             145.2 seconds
  
Success Criteria:
  RMSD < 5Å:           ✓
  RMSD Improvement >50%: ✓
  Energy < 0:          ✓
  GDT-TS > 50:         ✓
  Time < 5min:         ✓
  
Overall: ✓ PASS
```

### JSON Output

```json
{
  "suite_summary": {
    "total_proteins": 3,
    "successful": 3,
    "success_rate": 100.0,
    "average_rmsd_improvement": 62.0,
    "average_final_rmsd": 3.03,
    "total_runtime_seconds": 450.0
  },
  "individual_results": [
    {
      "pdb_id": "1UBQ",
      "protein_name": "Ubiquitin",
      "sequence_length": 76,
      "initial_rmsd": 10.5,
      "final_rmsd": 3.8,
      "rmsd_improvement_percent": 63.8,
      "energy_improvement": 90.5,
      "helix_rmsd": 2.5,
      "sheet_rmsd": 3.2,
      "loop_rmsd": 4.8,
      "core_rmsd": 2.9,
      "final_gdt_ts": 75.2,
      "final_tm_score": 0.82,
      "refinement_time_seconds": 145.2,
      "quantum_cores_identified": 8,
      "distance_restraints_applied": 24,
      "tertiary_contacts_enforced": 12,
      "meets_rmsd_target": true,
      "meets_improvement_target": true,
      "meets_energy_target": true,
      "meets_gdt_target": true,
      "meets_time_target": true
    }
  ]
}
```

## Unit Tests

**File**: `ubf_protein/tests/test_quantum_refinement_validation.py`

### Test Coverage

- ✅ `TestRefinementValidationResult`: Data class functionality
  - `test_successful_validation`: All criteria met
  - `test_failed_validation_rmsd`: RMSD criterion failed
  - `test_failed_validation_time`: Time limit exceeded

- ✅ `TestValidationSuiteResults`: Aggregated results
  - `test_suite_summary_all_pass`: 100% success rate
  - `test_suite_summary_partial_pass`: Partial failures

- ✅ `TestTestProteinConfiguration`: Configuration validation
  - `test_all_proteins_configured`: All 3 proteins present
  - `test_protein_configs_complete`: All required fields
  - `test_target_rmsd_values`: Reasonable RMSD targets

### Running Tests

```bash
pytest ubf_protein/tests/test_quantum_refinement_validation.py -v
```

**Result**: 8/8 tests passing ✅

## Integration with Quantum Refinement Engine

The validation suite integrates with all refinement components:

1. **Quantum Core Analyzer**: Identifies high-QCP regions
2. **Distance Restraint Manager**: Applies φ-harmonic restraints
3. **Secondary Structure Registrar**: Fixes helix/sheet geometry
4. **Hydrophobic Core Packer**: Optimizes core packing
5. **Loop Refiner**: Applies G(φ,t) dynamics
6. **Tertiary Contact Predictor**: Predicts and enforces contacts
7. **Two-Stage Optimization**: Global fold → local refinement
8. **RMSD Component Diagnostics**: Breaks down RMSD by region

## Performance Expectations

Based on Task 12 performance targets:

| Component | Target | Expected |
|-----------|--------|----------|
| Quantum core identification | <100ms | 50-80ms |
| Secondary structure registration | <200ms | 120-180ms |
| Hydrophobic packing | <500ms | 300-450ms |
| Full refinement | <5 min (100 res) | 2-4 min |

## Next Steps (Task 14.2-14.3)

### Task 14.2: Verify Success Criteria

Run actual validation on test proteins and verify:
- [ ] RMSD improvement > 50% for all proteins
- [ ] Final RMSD < 5Å for all proteins
- [ ] GDT-TS > 50 for all proteins
- [ ] Energy < 0 for all proteins
- [ ] Runtime < 5 minutes for all proteins

### Task 14.3: Generate Validation Reports

Create detailed reports with:
- [ ] RMSD trajectories over time
- [ ] Energy landscape visualization
- [ ] Component breakdown charts
- [ ] Quantum core visualizations
- [ ] Contact map heatmaps

## References

- Requirements: `quantum-refinement-engine/requirements.md`
- Tasks: `quantum-refinement-engine/tasks.md`
- Design: `quantum-refinement-engine/design.md`

## Status

**Task 14.1**: ✅ COMPLETED (November 9, 2025)
- Validation script implemented
- Unit tests created (8/8 passing)
- Ready for integration testing with real proteins
