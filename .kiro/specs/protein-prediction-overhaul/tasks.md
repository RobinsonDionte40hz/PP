# Implementation Plan

- [ ] 1. Implement Ramachandran-biased move generation
  - Create `RamachandranBiasMoveGenerator` class extending `MaplessMoveGenerator`
  - Implement Ramachandran region lookup tables for all 20 amino acids
  - Implement `_sample_ramachandran_angles()` method with 80% favored region probability
  - Implement secondary structure propensity biases (Chou-Fasman scale)
  - Update `_create_move()` methods to use Ramachandran-biased sampling
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 1.1 Write unit tests for Ramachandran sampling
  - Test that 80% of samples fall in favored regions
  - Test secondary structure bias (helix phi=-60, psi=-45)
  - Test residue-specific propensities
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Implement hydrogen bond energy calculator
  - Create `HydrogenBondCalculator` class implementing `IPhysicsCalculator`
  - Implement N-H···O=C geometry detection (distance 2.5-3.5Å, angle >120°)
  - Implement energy calculation (-5.0 kcal/mol × distance_factor × angle_factor)
  - Add hydrogen bond tracking to `ConformationalMemory` model
  - Integrate into `PhysicsIntegration.calculate_total_energy()`
  - _Requirements: 2.1, 8.1, 8.2, 8.5_

- [ ] 2.1 Write unit tests for hydrogen bond detection
  - Test ideal helix H-bond detection (i to i+4)
  - Test ideal sheet H-bond detection (parallel and antiparallel)
  - Test geometry criteria (distance and angle thresholds)
  - _Requirements: 2.1, 8.1_

- [ ] 3. Implement solvation energy calculator
  - Create `SolvationEnergyCalculator` class implementing `IPhysicsCalculator`
  - Implement SASA (solvent-accessible surface area) calculation
  - Implement Kyte-Doolittle hydrophobicity scale
  - Calculate solvation energy (σ_i × SASA_i for each residue)
  - Integrate into `PhysicsIntegration.calculate_total_energy()`
  - _Requirements: 2.2, 9.1, 9.2, 9.3, 9.4_

- [ ] 3.1 Write unit tests for solvation energy
  - Test that buried hydrophobic residues have negative energy
  - Test that exposed hydrophobic residues have positive energy
  - Test SASA calculation accuracy
  - _Requirements: 2.2, 9.1, 9.2_

- [ ] 4. Implement entropy penalty calculator
  - Create `EntropyPenaltyCalculator` class implementing `IPhysicsCalculator`
  - Implement Ramachandran probability lookup (favored vs disallowed)
  - Calculate entropy penalty (k_B × T × Σ -ln(P_i))
  - Integrate into `PhysicsIntegration.calculate_total_energy()`
  - _Requirements: 2.3_

- [ ] 4.1 Write unit tests for entropy penalty
  - Test that disallowed angles have high penalty
  - Test that favored angles have low penalty
  - Test temperature dependence
  - _Requirements: 2.3_

- [ ] 5. Create fragment library manager
  - Create `Fragment` data model with phi/psi angles and quality score
  - Create `FragmentLibraryManager` class
  - Implement simplified canonical fragment library (helix, sheet, turn 3-mers)
  - Implement `get_fragments_for_position()` with sequence matching
  - Implement `apply_fragment_to_conformation()` method
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 5.1 Write unit tests for fragment library
  - Test fragment loading and retrieval
  - Test sequence matching and scoring
  - Test fragment application to conformation
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 6. Integrate fragment-based moves into move generator
  - Add `_create_fragment_based_move()` method to `RamachandranBiasMoveGenerator`
  - Implement 30% probability for fragment-based moves
  - Ensure fragment moves preserve backbone continuity
  - Add fragment move type to `MoveType` enum if needed
  - _Requirements: 3.2, 3.3, 3.5_

- [ ] 7. Implement RMSD-aware move evaluation
  - Add `rmsd_before`, `rmsd_after`, `rmsd_delta` fields to `ConformationalMemory` model
  - Implement `_calculate_structural_goal_alignment()` in `CapabilityBasedMoveEvaluator`
  - Update composite score calculation to include structural goal factor (25% weight)
  - Add RMSD calculation for moves when native structure available
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 7.1 Write unit tests for RMSD-aware evaluation
  - Test structural goal calculation with native structure
  - Test structural goal calculation without native structure (compactness)
  - Test composite score weighting
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 8. Add RMSD tracking to agent and visualization
  - Update `ProteinAgent` to track RMSD trajectory alongside energy
  - Add RMSD to `ConformationSnapshot` model
  - Update `VisualizationExporter` to include RMSD in trajectory JSON
  - Add RMSD column to energy landscape CSV export
  - _Requirements: 5.4, 5.5_

- [ ] 9. Fix collective learning with indexed memory retrieval
  - Add `sequence_region` field to `ConformationalMemory` model
  - Implement `retrieve_memories_by_context()` in `SharedMemoryPool`
  - Add memory indexing by sequence region and move type
  - Implement memory relevance scoring (sequence match × behavioral similarity)
  - _Requirements: 4.2, 4.3_

- [ ] 9.1 Write unit tests for indexed memory retrieval
  - Test memory indexing by sequence region
  - Test memory retrieval by context
  - Test relevance scoring
  - _Requirements: 4.2, 4.3_

- [ ] 10. Implement memory success rate tracking
  - Add `times_applied`, `times_successful`, `success_rate` fields to `ConformationalMemory`
  - Implement `update_memory_success_rate()` in `SharedMemoryPool`
  - Update memory influence weight based on success rate
  - Track memory application and outcomes in `ProteinAgent`
  - _Requirements: 4.4, 4.5_

- [ ] 11. Update memory storage to use RMSD delta for significance
  - Modify `ConformationalMemory.get_structural_significance()` to use RMSD delta
  - Update memory storage threshold logic to prioritize RMSD improvements
  - Ensure memories with RMSD delta > 2Å are always shared
  - _Requirements: 4.1, 4.5_

- [ ] 12. Add configuration flags for new features
  - Add `use_ramachandran_bias`, `use_fragment_library`, `use_enhanced_physics` to `AdaptiveConfig`
  - Add `fragment_move_probability` and `ramachandran_favored_probability` config fields
  - Implement backward compatibility (default to new behavior, allow opt-out)
  - Update config documentation
  - _Requirements: All (configuration)_

- [ ] 13. Implement stuck rate reduction strategies
  - Update `LocalMinimaDetector` to use fragment-based large jumps for escape
  - Implement automatic threshold relaxation after 100 stuck iterations
  - Add stuck event logging with reasons (energy, clash, validation)
  - Track and report stuck rate in final results
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 13.1 Write unit tests for stuck detection improvements
  - Test fragment-based escape strategy
  - Test threshold relaxation
  - Test stuck rate calculation
  - _Requirements: 6.1, 6.2, 6.5_

- [ ] 14. Create benchmark validation framework
  - Create `benchmark_proteins.py` with 5 benchmark proteins (1CRN, 1UBQ, 1LYZ, etc.)
  - Implement `run_benchmark_suite()` function
  - Add native structure loading and RMSD comparison
  - Generate comparison plots (RMSD trajectory, energy landscape, Ramachandran distribution)
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [ ] 14.1 Write benchmark validation tests
  - Test benchmark suite execution
  - Test RMSD comparison with native structures
  - Test plot generation
  - _Requirements: 7.2, 7.3, 7.5_

- [ ] 15. Run benchmark validation and verify targets
  - Run benchmark suite on all 5 proteins (10 agents, 2000 iterations each)
  - Verify average RMSD < 10Å on at least 3 proteins
  - Verify stuck rate < 50% across all benchmarks
  - Verify collective learning benefit > 0.15
  - Generate comprehensive validation report
  - _Requirements: 7.4, 6.5, 4.5_

- [ ] 16. Create integration tests for complete system
  - Create `test_protein_prediction_integration.py`
  - Test end-to-end prediction with all fixes enabled
  - Test collective learning benefit measurement
  - Test RMSD improvement over iterations
  - Verify tests complete in <10 minutes
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 17. Update documentation with new features
  - Update `README.md` with Ramachandran bias, enhanced physics, fragment library
  - Update `API.md` with new classes and methods
  - Add examples for benchmark validation
  - Document configuration options for new features
  - Add troubleshooting section for common issues
  - _Requirements: All (documentation)_
