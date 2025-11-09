# Ablation Studies Report

Generated: 2025-11-09 17:48:01

## QCP Formula Components

### Summary Statistics

- **Mean RMSD Drop**: -66.9%
- **Std RMSD Drop**: 3.0%
- **Most Impactful Component**: no_exponential
- **Least Impactful Component**: baseline
- **Significant Changes**: 7/7

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -60.8% | 8.37 | Full QCP formula |
| no_exponential | -71.2% | 8.37 | Remove 2^n term |
| no_phi | -66.5% | 8.37 | Remove phi^l term |
| no_hydrophobicity | -66.2% | 8.37 | Remove m term |
| no_base_energy | -66.7% | 8.37 | Remove 4.0 constant |
| linear_qcp | -67.2% | 8.37 | Linear formula: qcp = n + l + m |
| random_qcp | -69.6% | 8.37 | Random QCP values |

## Energy Function Components

### Summary Statistics

- **Mean RMSD Drop**: -46.3%
- **Std RMSD Drop**: 19.2%
- **Most Impactful Component**: no_hbond
- **Least Impactful Component**: no_compactness
- **Significant Changes**: 10/10

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -56.7% | 9.09 | All energy terms |
| no_bond | -49.5% | 9.09 | Remove bond stretching energy |
| no_angle | -55.4% | 9.09 | Remove angle bending energy |
| no_dihedral | -55.8% | 9.09 | Remove dihedral torsion energy |
| no_vdw | -24.0% | 9.09 | Remove van der Waals energy |
| no_electrostatic | -66.3% | 9.09 | Remove electrostatic energy |
| no_hbond | -77.0% | 9.09 | Remove hydrogen bonding energy |
| no_compactness | -18.0% | 9.09 | Remove compactness bonus |
| harmonic_only | -41.2% | 9.09 | Only harmonic terms (bond + angle) |
| nonbonded_only | -19.0% | 9.09 | Only non-bonded terms (vdw + electrostatic) |

## Exploration Parameters

### Summary Statistics

- **Mean RMSD Drop**: -46.4%
- **Std RMSD Drop**: 13.3%
- **Most Impactful Component**: linear_transform
- **Least Impactful Component**: random_params
- **Significant Changes**: 6/6

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -54.0% | 8.52 | Standard consciousness-inspired transformations |
| linear_transform | -65.1% | 8.52 | Linear parameter mapping |
| no_transform | -51.8% | 8.52 | Direct parameter usage |
| random_params | -22.6% | 8.52 | Random parameter assignment |
| fixed_params | -39.0% | 8.52 | Fixed parameter values |
| inverse_transform | -46.3% | 8.52 | Inverse transformation functions |

## Move Evaluation Factors

### Summary Statistics

- **Mean RMSD Drop**: -42.5%
- **Std RMSD Drop**: 13.9%
- **Most Impactful Component**: equal_weights
- **Least Impactful Component**: physical_only
- **Significant Changes**: 9/9

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -53.5% | 8.38 | All 5 evaluation factors |
| no_physical | -35.6% | 8.38 | Remove physical feasibility factor |
| no_quantum | -51.2% | 8.38 | Remove quantum alignment factor |
| no_behavioral | -42.2% | 8.38 | Remove behavioral preference factor |
| no_historical | -52.1% | 8.38 | Remove historical success factor |
| no_goal | -46.3% | 8.38 | Remove goal alignment factor |
| physical_only | -14.7% | 8.38 | Only physical feasibility |
| random_weights | -26.0% | 8.38 | Random factor weights |
| equal_weights | -60.9% | 8.38 | Equal factor weights |

## Validation Metrics

### Summary Statistics

- **Mean RMSD Drop**: -40.1%
- **Std RMSD Drop**: 23.6%
- **Most Impactful Component**: baseline
- **Least Impactful Component**: random_metric
- **Significant Changes**: 7/7

### Detailed Results

| Variant | RMSD Drop | Significance | Notes |
|---------|-----------|--------------|-------|
| baseline | -61.9% | 9.01 | RMSD + Energy optimization |
| rmsd_only | -56.2% | 9.01 | RMSD-only optimization |
| energy_only | -30.0% | 9.01 | Energy-only optimization |
| gdt_ts_target | -59.1% | 9.01 | GDT-TS as optimization target |
| tm_score_target | -60.8% | 9.01 | TM-score as optimization target |
| no_validation | -7.3% | 9.01 | No validation feedback |
| random_metric | -5.2% | 9.01 | Random metric selection |

