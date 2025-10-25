# Requirements Document

## Introduction

The UBF Protein System currently achieves RMSD values of 83Å (target: <5Å), has 0% collective learning benefit, and experiences 97-99% stuck rates. This feature addresses four fundamental architectural issues: (1) random move generation without structural bias, (2) incomplete energy function missing hydrogen bonds and solvation, (3) broken collective learning with no measurable benefit, and (4) lack of guidance toward native-like structures. The goal is to transform the system from a random conformational sampler into a physics-grounded structure predictor capable of achieving <10Å RMSD on benchmark proteins.

## Glossary

- **UBF System**: Universal Behavioral Framework protein structure prediction system located in `ubf_protein/`
- **RMSD**: Root Mean Square Deviation - distance metric between predicted and native structures (Angstroms)
- **Ramachandran Plot**: Map of allowed backbone dihedral angles (phi, psi) for amino acids
- **Hydrophobic Effect**: Tendency of nonpolar residues to cluster in protein core, driven by solvation
- **Fragment Library**: Database of known good local protein structures (3-9 residue segments)
- **Collective Learning**: Multi-agent system where agents share high-significance memories
- **Consciousness Coordinates**: 2D state space (frequency 3-15 Hz, coherence 0.2-1.0) guiding agent behavior
- **Energy Function**: Mathematical function scoring conformation quality (lower = more stable)
- **Stuck Rate**: Percentage of iterations where agents cannot find acceptable moves

## Requirements

### Requirement 1: Ramachandran-Biased Move Generation

**User Story:** As a protein structure researcher, I want the move generator to sample conformations from sterically allowed regions, so that agents explore physically realistic structures instead of random impossible geometries.

#### Acceptance Criteria

1. WHEN the MapplessMoveGenerator generates a backbone rotation move, THE UBF System SHALL sample phi and psi angles from Ramachandran-favored regions with 80% probability
2. WHEN the MapplessMoveGenerator generates a helix formation move, THE UBF System SHALL bias dihedral angles toward canonical helix geometry (phi=-60°, psi=-45°)
3. WHEN the MapplessMoveGenerator generates a sheet formation move, THE UBF System SHALL bias dihedral angles toward canonical beta-sheet geometry (phi=-120°, psi=+120°)
4. WHILE generating moves for residues in predicted secondary structure regions, THE UBF System SHALL apply secondary structure propensity weights based on amino acid type
5. THE UBF System SHALL reduce the percentage of moves sampling disallowed Ramachandran regions from 80% to less than 20%

### Requirement 2: Enhanced Energy Function with Missing Physics

**User Story:** As a computational biologist, I want the energy function to include hydrogen bonding, solvation effects, and entropy penalties, so that low-energy conformations correlate with native-like structures.

#### Acceptance Criteria

1. WHEN the physics integration calculates energy, THE UBF System SHALL compute hydrogen bond energy for backbone N-H···O=C interactions within 3.5Å
2. WHEN the physics integration calculates energy, THE UBF System SHALL compute solvation energy using implicit solvent model with hydrophobic burial rewards
3. WHEN the physics integration calculates energy, THE UBF System SHALL apply entropy penalties for conformations with restricted backbone flexibility
4. THE UBF System SHALL achieve correlation coefficient > 0.6 between energy scores and RMSD-to-native on benchmark proteins
5. WHEN comparing conformations at -300 kcal/mol versus -400 kcal/mol, THE UBF System SHALL demonstrate that lower energy conformations have lower average RMSD

### Requirement 3: Fragment Library Integration

**User Story:** As a structure prediction developer, I want agents to bias moves toward known good local structures from fragment libraries, so that exploration is guided by evolutionary and structural knowledge.

#### Acceptance Criteria

1. THE UBF System SHALL load a fragment library containing 3-mer and 9-mer backbone fragments for each sequence position
2. WHEN the MapplessMoveGenerator generates moves, THE UBF System SHALL sample fragment-based moves with 30% probability
3. WHEN applying a fragment-based move, THE UBF System SHALL copy backbone angles from the selected fragment to the target residue range
4. THE UBF System SHALL score fragment quality using sequence similarity and structural compatibility metrics
5. WHERE fragment library data is available, THE UBF System SHALL demonstrate 20% improvement in RMSD convergence rate compared to non-fragment moves

### Requirement 4: Functional Collective Learning System

**User Story:** As a multi-agent system researcher, I want agents to share memories that provide measurable learning benefit, so that collective exploration outperforms independent agents.

#### Acceptance Criteria

1. WHEN an agent achieves RMSD improvement > 2Å, THE UBF System SHALL store the move as a high-significance memory with RMSD delta recorded
2. WHEN the SharedMemoryPool stores memories, THE UBF System SHALL index memories by sequence region and move type for efficient retrieval
3. WHEN an agent queries shared memories, THE UBF System SHALL retrieve memories matching the current sequence context and behavioral state
4. WHEN an agent applies a shared memory move, THE UBF System SHALL track success rate and update memory influence scores
5. THE UBF System SHALL achieve collective_learning_benefit > 0.15 (15% improvement) in multi-agent runs compared to single-agent baseline

### Requirement 5: RMSD-Aware Move Evaluation

**User Story:** As a structure prediction user, I want move evaluation to consider structural similarity to native folds, so that agents optimize for correct structure rather than just low energy.

#### Acceptance Criteria

1. WHERE native structure is available for validation, THE UBF System SHALL compute RMSD for each evaluated conformation
2. WHEN the CapabilityBasedMoveEvaluator scores moves, THE UBF System SHALL include RMSD improvement as a goal alignment factor
3. WHEN an agent is stuck in a local minimum, THE UBF System SHALL prioritize moves that increase structural diversity over energy minimization
4. THE UBF System SHALL track RMSD trajectory alongside energy trajectory in checkpoint and visualization outputs
5. WHEN running multi-agent exploration, THE UBF System SHALL report best RMSD achieved and RMSD improvement rate in final results

### Requirement 6: Reduced Stuck Rate Through Better Sampling

**User Story:** As a protein folding researcher, I want agents to successfully generate acceptable moves more than 50% of the time, so that exploration is efficient and not dominated by rejection.

#### Acceptance Criteria

1. WHEN the MapplessMoveGenerator generates moves with Ramachandran bias, THE UBF System SHALL achieve stuck rate < 50% (down from 97-99%)
2. WHEN an agent is stuck for more than 100 consecutive iterations, THE UBF System SHALL automatically increase move diversity and reduce acceptance thresholds
3. WHEN the LocalMinimaDetector triggers escape strategy, THE UBF System SHALL use fragment-based large jumps to escape local minima
4. THE UBF System SHALL log stuck event frequency and reasons (energy threshold, clash detection, validation failure) for diagnostic analysis
5. WHILE running 1000-iteration explorations, THE UBF System SHALL demonstrate that agents spend less than 50% of iterations in stuck state

### Requirement 7: Benchmark Validation Framework

**User Story:** As a computational structural biologist, I want to validate predictions against known benchmark proteins, so that I can measure real-world prediction accuracy.

#### Acceptance Criteria

1. THE UBF System SHALL include a benchmark suite with at least 5 proteins of varying sizes (30-100 residues)
2. WHEN running benchmark validation, THE UBF System SHALL load native structures and compute RMSD for all predictions
3. WHEN benchmark validation completes, THE UBF System SHALL report average RMSD, best RMSD, energy-RMSD correlation, and stuck rate statistics
4. THE UBF System SHALL achieve average RMSD < 10Å on at least 3 out of 5 benchmark proteins after fixes
5. THE UBF System SHALL generate comparison plots showing RMSD trajectories, energy landscapes, and Ramachandran distributions

### Requirement 8: Hydrogen Bond Network Tracking

**User Story:** As a protein chemist, I want the system to explicitly track and optimize hydrogen bond networks, so that predicted structures exhibit realistic secondary structure stabilization.

#### Acceptance Criteria

1. WHEN the physics integration evaluates a conformation, THE UBF System SHALL identify all backbone hydrogen bonds meeting geometric criteria (distance < 3.5Å, angle > 120°)
2. WHEN the CapabilityBasedMoveEvaluator scores moves, THE UBF System SHALL reward moves that form new hydrogen bonds in predicted helix/sheet regions
3. THE UBF System SHALL track hydrogen bond count in ConformationalMemory and visualization outputs
4. WHEN comparing native and predicted structures, THE UBF System SHALL report hydrogen bond overlap percentage
5. THE UBF System SHALL demonstrate that low-RMSD conformations have higher hydrogen bond network similarity to native structures

### Requirement 9: Solvation-Driven Hydrophobic Collapse

**User Story:** As a biophysicist, I want hydrophobic residues to be driven toward the protein core by solvation energy, so that predicted structures exhibit realistic hydrophobic effect.

#### Acceptance Criteria

1. WHEN the physics integration calculates solvation energy, THE UBF System SHALL assign positive energy penalties to solvent-exposed hydrophobic residues
2. WHEN the physics integration calculates solvation energy, THE UBF System SHALL assign negative energy rewards to buried hydrophobic residues
3. THE UBF System SHALL use residue-specific hydrophobicity scales (Kyte-Doolittle or equivalent)
4. WHEN running exploration on proteins with hydrophobic cores, THE UBF System SHALL demonstrate progressive burial of hydrophobic residues over iterations
5. THE UBF System SHALL report solvent-accessible surface area (SASA) for hydrophobic residues in visualization outputs

### Requirement 10: Integration Testing with Real Proteins

**User Story:** As a quality assurance engineer, I want comprehensive integration tests using real protein sequences, so that all fixes work together correctly in production scenarios.

#### Acceptance Criteria

1. THE UBF System SHALL include integration tests for 3 benchmark proteins (small, medium, large)
2. WHEN integration tests run, THE UBF System SHALL verify that RMSD improves by at least 10% compared to baseline
3. WHEN integration tests run, THE UBF System SHALL verify that collective learning benefit is greater than 0% 
4. WHEN integration tests run, THE UBF System SHALL verify that stuck rate is less than 60%
5. THE UBF System SHALL execute all integration tests in under 10 minutes on standard hardware
