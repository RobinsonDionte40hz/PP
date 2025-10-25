# Implementation Plan

- [x] 1. Implement Molecular Mechanics Energy Function
  - Create `ubf_protein/energy_function.py` with `MolecularMechanicsEnergy` class implementing `IPhysicsCalculator`
  - Implement bond energy calculation using harmonic potential: E_bond = Σ k_b(r - r_0)²
  - Implement angle energy calculation using harmonic potential: E_angle = Σ k_θ(θ - θ_0)²
  - Implement dihedral energy calculation using periodic potential: E_dihedral = Σ V_n/2 [1 + cos(nφ - γ)]
  - Implement van der Waals energy using Lennard-Jones 12-6 potential with 12Å cutoff
  - Implement electrostatic energy using Coulomb's law with distance-dependent dielectric
  - Implement hydrogen bond energy using 10-12 potential
  - Create `ForceFieldParameters` class with AMBER-like parameters for all energy terms
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.1 Write unit tests for energy function components
  - Test bond energy returns ~0 for ideal geometry
  - Test VDW energy is repulsive (<0) for atoms too close
  - Test VDW energy is attractive (>0) at optimal distance
  - Test electrostatic energy is attractive for opposite charges
  - Test total energy is negative for known folded protein (1UBQ)
  - Test total energy is positive/high for extended chain
  - _Requirements: 5.1, 5.5_

- [x] 2. Integrate Energy Function with UBF System
  - Update `protein_agent.py` to use `MolecularMechanicsEnergy` calculator
  - Modify `_execute_move` method to calculate energy using new function
  - Update `Conformation` model to include `energy_components` field for debugging
  - Add configuration flag `USE_MOLECULAR_MECHANICS_ENERGY` in `config.py`
  - Implement graceful error handling for energy calculation failures
  - Add energy validation checks (warn if |energy| > 10000 kcal/mol)
  - _Requirements: 1.7, 6.1, 6.2, 6.4, 6.5_

- [x] 2.1 Write integration tests for energy function
  - Test energy calculation integrates correctly with agent exploration
  - Test energy decreases during successful moves
  - Test energy validation catches unrealistic values
  - Test backward compatibility with existing tests
  - _Requirements: 5.2, 5.3, 6.3_

- [x] 3. Implement RMSD Calculator
  - Create `ubf_protein/rmsd_calculator.py` with `RMSDCalculator` class
  - Implement basic RMSD calculation: RMSD = sqrt(Σ(r_pred - r_native)² / N)
  - Implement Kabsch algorithm for optimal superposition (rotation + translation)
  - Implement GDT-TS calculation (% residues within 1, 2, 4, 8 Å)
  - Implement TM-score calculation with length-dependent normalization
  - Support both Cα-only and all-atom RMSD calculations
  - Add performance optimization to complete in <100ms for 500 residues
  - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7_

- [x] 3.1 Write unit tests for RMSD calculator
  - Test RMSD = 0 for identical structures
  - Test RMSD ≈ 0 after translation (with alignment)
  - Test RMSD ≈ 0 after rotation (with alignment)
  - Test RMSD ≈ 1.0 for structure with 1Å random noise
  - Test GDT-TS = 100 for identical structures
  - Test TM-score = 1.0 for identical structures
  - _Requirements: 2.6_

- [x] 4. Implement Native Structure Loader
  - Create `NativeStructureLoader` class in `rmsd_calculator.py`
  - Implement PDB file parsing to extract Cα coordinates
  - Implement sequence extraction from PDB SEQRES or ATOM records
  - Handle multiple models (use first model by default)
  - Handle missing residues (report and exclude from RMSD)
  - Implement local PDB file loading from filesystem
  - Implement PDB download from RCSB database using PDB IDs
  - Add clear error messages for loading failures
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 4.1 Write unit tests for PDB loader
  - Test loading local PDB file
  - Test extracting Cα coordinates
  - Test extracting sequence information
  - Test handling missing residues
  - Test error handling for invalid files
  - _Requirements: 3.4, 3.7_

- [x] 5. Integrate RMSD Validation with UBF System
  - Update `Conformation` model to add `native_structure_ref`, `gdt_ts_score`, `tm_score` fields
  - Update `protein_agent.py` to accept optional native structure parameter
  - Modify `_execute_move` to calculate RMSD if native structure provided
  - Update `ExplorationMetrics` to include RMSD metrics
  - Update `ExplorationResults` to include best RMSD and validation quality
  - Implement graceful error handling for RMSD calculation failures
  - _Requirements: 2.4, 2.5, 6.4_

- [x] 5.1 Write integration tests for RMSD validation
  - Test RMSD calculation integrates with agent exploration
  - Test RMSD improves during exploration toward native
  - Test RMSD calculation handles missing native structure gracefully
  - Test RMSD metrics appear in exploration results
  - _Requirements: 2.5_

- [x] 6. Create Validation Suite
  - Create `ubf_protein/validation_suite.py` with `ValidationSuite` class
  - Create test set JSON with 5 proteins: 1UBQ, 1CRN, 2MR9, 1VII, 1LYZ
  - Implement `validate_protein` method to run full validation on single protein
  - Implement `run_test_suite` method to validate all test proteins
  - Create `ValidationReport` dataclass with all metrics (RMSD, energy, GDT-TS, TM-score)
  - Implement quality assessment logic (excellent/good/acceptable/poor)
  - Implement baseline comparison (random sampling, Monte Carlo)
  - _Requirements: 1.6, 4.1, 4.2, 4.3, 4.4_

- [x] 6.1 Write tests for validation suite
  - Test validation runs successfully on test proteins
  - Test validation report includes all required metrics
  - Test quality assessment logic
  - Test baseline comparison
  - _Requirements: 5.2_

- [x] 7. Update Reporting and Documentation
  - Update `run_multi_agent.py` to include RMSD and energy validation in output
  - Modify test report format to show energy components breakdown
  - Add quality flags (high/moderate/poor) to reports based on RMSD
  - Update report to show energy is negative for folded proteins
  - Add comparison to native structure energy if available
  - Update README.md with energy function details and validation metrics
  - Update API.md with new classes and methods
  - Create example script demonstrating validation usage
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [x] 8. Performance Optimization
  - Implement neighbor list for non-bonded interactions (VDW, electrostatics)
  - Add caching for force field parameter lookups
  - Profile energy calculation and optimize hot paths
  - Verify energy calculation <50ms for 100-residue proteins
  - Verify RMSD calculation <100ms for 500-residue proteins
  - Verify agent decision latency remains <2ms
  - _Requirements: 2.7, 5.4_

- [x] 8.1 Write performance tests
  - Benchmark energy calculation speed
  - Benchmark RMSD calculation speed
  - Benchmark overall agent decision latency
  - Test performance with various protein sizes
  - _Requirements: 5.4_

- [x] 9. Validation and Testing
  - Run validation suite on all 5 test proteins
  - Verify ubiquitin (1UBQ) energy is between -120 and -80 kcal/mol
  - Verify all folded proteins have negative energy
  - Verify RMSD improves during exploration
  - Verify energy decreases correlate with RMSD decreases
  - Run all existing tests to ensure backward compatibility
  - Document any breaking changes and provide migration guide
  - _Requirements: 1.6, 1.7, 5.2, 5.3, 6.3, 6.6_
