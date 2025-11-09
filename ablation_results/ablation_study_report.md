# Ablation Studies Report

Generated: 2025-11-09 17:43:48

## QCP Formula Components

### Summary Statistics

- **Mean RMSD Drop**: -14.0%
- **Std RMSD Drop**: 5.2%
- **Most Impactful Component**: baseline
- **Least Impactful Component**: linear_qcp
- **Significant Changes**: 7/7

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -18.0% | 20.00 | Full QCP formula |
| no_exponential | -15.8% | 20.00 | Remove 2^n term |
| no_phi | -17.8% | 20.00 | Remove phi^l term |
| no_hydrophobicity | -9.0% | 20.00 | Remove m term |
| no_base_energy | -17.4% | 20.00 | Remove 4.0 constant |
| linear_qcp | -3.5% | 20.00 | Linear formula: qcp = n + l + m |
| random_qcp | -16.5% | 20.00 | Random QCP values |

## Energy Function Components

### Summary Statistics

- **Mean RMSD Drop**: 27.8%
- **Std RMSD Drop**: 41.5%
- **Most Impactful Component**: no_compactness
- **Least Impactful Component**: no_dihedral
- **Significant Changes**: 10/10

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -3.2% | 20.00 | All energy terms |
| no_bond | 20.2% | 20.00 | Remove bond stretching energy |
| no_angle | 11.5% | 20.00 | Remove angle bending energy |
| no_dihedral | 2.0% | 20.00 | Remove dihedral torsion energy |
| no_vdw | 57.0% | 20.00 | Remove van der Waals energy |
| no_electrostatic | -13.6% | 20.00 | Remove electrostatic energy |
| no_hbond | -24.8% | 20.00 | Remove hydrogen bonding energy |
| no_compactness | 99.5% | 20.00 | Remove compactness bonus |
| harmonic_only | 31.9% | 20.00 | Only harmonic terms (bond + angle) |
| nonbonded_only | 97.0% | 20.00 | Only non-bonded terms (vdw + electrostatic) |

## Exploration Parameters

### Summary Statistics

- **Mean RMSD Drop**: 29.7%
- **Std RMSD Drop**: 32.8%
- **Most Impactful Component**: random_params
- **Least Impactful Component**: baseline
- **Significant Changes**: 6/6

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | 7.2% | 20.00 | Standard consciousness-inspired transformations |
| linear_transform | -11.0% | 20.00 | Linear parameter mapping |
| no_transform | 26.5% | 20.00 | Direct parameter usage |
| random_params | 89.9% | 20.00 | Random parameter assignment |
| fixed_params | 50.4% | 20.00 | Fixed parameter values |
| inverse_transform | 15.3% | 20.00 | Inverse transformation functions |

## Move Evaluation Factors

### Summary Statistics

- **Mean RMSD Drop**: 31.7%
- **Std RMSD Drop**: 35.1%
- **Most Impactful Component**: physical_only
- **Least Impactful Component**: equal_weights
- **Significant Changes**: 9/9

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -10.4% | 20.00 | All 5 evaluation factors |
| no_physical | 61.3% | 20.00 | Remove physical feasibility factor |
| no_quantum | -6.8% | 20.00 | Remove quantum alignment factor |
| no_behavioral | 45.1% | 20.00 | Remove behavioral preference factor |
| no_historical | 13.1% | 20.00 | Remove historical success factor |
| no_goal | 24.7% | 20.00 | Remove goal alignment factor |
| physical_only | 91.8% | 20.00 | Only physical feasibility |
| random_weights | 70.5% | 20.00 | Random factor weights |
| equal_weights | -3.8% | 20.00 | Equal factor weights |

## Validation Metrics

### Summary Statistics

- **Mean RMSD Drop**: 38.9%
- **Std RMSD Drop**: 52.7%
- **Most Impactful Component**: no_validation
- **Least Impactful Component**: baseline
- **Significant Changes**: 7/7

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | 2.4% | 20.00 | RMSD + Energy optimization |
| rmsd_only | -6.1% | 20.00 | RMSD-only optimization |
| energy_only | 57.4% | 20.00 | Energy-only optimization |
| gdt_ts_target | -3.8% | 20.00 | GDT-TS as optimization target |
| tm_score_target | -8.0% | 20.00 | TM-score as optimization target |
| no_validation | 119.6% | 20.00 | No validation feedback |
| random_metric | 111.0% | 20.00 | Random metric selection |

