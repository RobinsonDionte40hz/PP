# Requirements Document

## Introduction

This specification addresses critical scientific validity issues in the UBF Protein Prediction System's energy calculations and validation metrics. Currently, the system produces physically impossible energy values (positive energies for folded proteins) and lacks proper RMSD validation against native structures. This prevents the system from being scientifically credible and makes it impossible to assess prediction accuracy.

The goal is to implement a physically accurate energy function based on established force fields and add comprehensive RMSD validation infrastructure to enable proper benchmarking against known protein structures.

## Glossary

- **Energy Function**: Mathematical model that calculates the potential energy of a protein conformation based on atomic interactions
- **RMSD (Root Mean Square Deviation)**: Standard metric measuring structural similarity between two protein conformations in Ångströms (Å)
- **Force Field**: Set of parameters and equations describing atomic interactions (e.g., AMBER, CHARMM)
- **Native Structure**: Experimentally determined protein structure from PDB database
- **UBF System**: Universal Behavioral Framework protein prediction system
- **Conformation**: Specific 3D arrangement of protein atoms
- **Energy Terms**: Components of total energy (van der Waals, electrostatics, bonds, angles, torsions, hydrogen bonds)
- **PDB Parser**: Tool for reading Protein Data Bank structure files
- **Superimposer**: Algorithm for optimal structural alignment of two protein structures
- **Validation Suite**: Collection of tests comparing predictions against known structures

## Requirements

### Requirement 1: Physically Accurate Energy Function

**User Story:** As a computational biologist, I want the energy function to produce physically realistic values, so that I can trust the system's predictions and compare them to experimental data.

#### Acceptance Criteria

1. WHEN the System calculates energy for a folded protein conformation, THE System SHALL return negative energy values in the range of -200 to -50 kcal/mol for small proteins (50-150 residues)

2. WHEN the System calculates energy for an unfolded protein conformation, THE System SHALL return energy values that are higher (less negative or positive) than the corresponding folded state by at least 20 kcal/mol

3. THE System SHALL include all six standard force field energy terms: van der Waals interactions, electrostatic interactions, bond stretching, angle bending, torsional rotation, and hydrogen bonding

4. WHEN the System calculates van der Waals energy, THE System SHALL use Lennard-Jones 12-6 potential with appropriate atomic radii and well depths

5. WHEN the System calculates electrostatic energy, THE System SHALL use Coulomb's law with distance-dependent dielectric screening

6. THE System SHALL validate energy calculations against at least three known protein structures with documented experimental stability values (ΔG)

7. WHEN the System calculates energy for ubiquitin (1UBQ), THE System SHALL produce values between -120 and -80 kcal/mol for the native structure

### Requirement 2: RMSD Calculation Infrastructure

**User Story:** As a researcher, I want to calculate RMSD between predicted and native structures, so that I can quantitatively assess prediction accuracy using the field's gold standard metric.

#### Acceptance Criteria

1. THE System SHALL implement RMSD calculation between two protein conformations using the formula: RMSD = sqrt(sum((r_i - r'_i)²) / N) where r_i are atomic positions

2. WHEN the System calculates RMSD, THE System SHALL perform optimal structural superposition using Kabsch algorithm before measuring distances

3. THE System SHALL support RMSD calculation for both Cα-only and all-atom representations

4. WHEN the System loads a native PDB structure, THE System SHALL extract atomic coordinates for all residues in the sequence

5. THE System SHALL handle missing residues in PDB files by excluding them from RMSD calculation and reporting the number of excluded residues

6. WHEN the System calculates RMSD for identical structures, THE System SHALL return a value less than 0.01 Å

7. THE System SHALL calculate RMSD in less than 100 milliseconds for proteins up to 500 residues

### Requirement 3: PDB Structure Loading

**User Story:** As a user, I want to load native protein structures from PDB files, so that I can validate predictions against experimentally determined structures.

#### Acceptance Criteria

1. THE System SHALL parse PDB format files and extract Cα atom coordinates for all residues

2. WHEN the System encounters a PDB file with multiple models, THE System SHALL use the first model by default

3. THE System SHALL extract sequence information from PDB SEQRES records or ATOM records

4. WHEN the System loads a PDB file with missing residues, THE System SHALL report which residues are missing and continue processing

5. THE System SHALL support loading PDB files from local filesystem paths

6. THE System SHALL support downloading PDB files from RCSB PDB database using PDB IDs (e.g., "1UBQ")

7. WHEN the System fails to load a PDB file, THE System SHALL provide a clear error message indicating the failure reason

### Requirement 4: Validation Reporting

**User Story:** As a developer, I want comprehensive validation reports comparing predictions to native structures, so that I can track system improvements and identify issues.

#### Acceptance Criteria

1. THE System SHALL generate validation reports including: best RMSD, best energy, energy range, RMSD range, and comparison to native structure

2. WHEN the System completes a prediction run, THE System SHALL report RMSD to native structure if a native structure was provided

3. THE System SHALL report whether the predicted energy is within the physically realistic range for folded proteins

4. THE System SHALL calculate and report the energy difference between predicted and native conformations

5. WHEN the System produces RMSD greater than 5.0 Å, THE System SHALL flag the prediction as "poor quality"

6. WHEN the System produces RMSD between 2.0 and 5.0 Å, THE System SHALL flag the prediction as "moderate quality"

7. WHEN the System produces RMSD less than 2.0 Å, THE System SHALL flag the prediction as "high quality"

8. THE System SHALL include validation metrics in the existing test report output format

### Requirement 5: Energy Function Validation Suite

**User Story:** As a quality assurance engineer, I want automated tests that verify energy calculations against known benchmarks, so that I can ensure the energy function remains accurate across code changes.

#### Acceptance Criteria

1. THE System SHALL include unit tests for each energy term (van der Waals, electrostatics, bonds, angles, torsions, hydrogen bonds)

2. THE System SHALL include integration tests comparing total energy calculations to reference values for at least five known protein structures

3. WHEN the System runs energy validation tests, THE System SHALL verify that folded proteins have lower energy than unfolded states

4. THE System SHALL test energy function performance to ensure calculations complete in less than 50 milliseconds for 100-residue proteins

5. THE System SHALL include tests verifying energy function produces consistent results for identical conformations

6. THE System SHALL test that energy function correctly handles edge cases: single residue, two residues, and very large proteins (>500 residues)

### Requirement 6: Backward Compatibility

**User Story:** As a system maintainer, I want the new energy function to integrate seamlessly with existing code, so that I don't break current functionality while fixing the physics.

#### Acceptance Criteria

1. THE System SHALL maintain the existing IPhysicsCalculator interface for energy calculations

2. THE System SHALL preserve all existing method signatures in physics_integration.py

3. WHEN the System updates energy calculations, THE System SHALL ensure all existing tests continue to pass

4. THE System SHALL maintain compatibility with the existing Conformation data model

5. THE System SHALL update configuration parameters in config.py without removing existing parameters

6. THE System SHALL provide migration documentation if any breaking changes are unavoidable
