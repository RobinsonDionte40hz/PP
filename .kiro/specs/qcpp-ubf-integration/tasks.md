no# Implementation Plan

- [x] 1. Create QCPP integration layer
  - Create `ubf_protein/qcpp_integration.py` with `QCPPIntegrationAdapter` class
  - Implement `QCPPMetrics` dataclass with qcp_score, field_coherence, stability_score, phi_match_score fields
  - Implement `analyze_conformation()` method that wraps QCPP predictor calls
  - Implement `calculate_quantum_alignment()` method using formula: 0.5 + min(1.0, (qcp/5.0)*0.4 + (coherence+1.0)*0.3 + phi_match*0.3)
  - Implement `get_stability_score()` and `get_phi_match_score()` helper methods
  - Add LRU cache decorator to `analyze_conformation()` for performance
  - _Requirements: 1.1, 1.2, 1.3, 7.1_

- [x] 1.1 Write unit tests for QCPP integration layer
  - Test `QCPPMetrics` dataclass validation
  - Test `analyze_conformation()` correctly calls QCPP methods
  - Test `calculate_quantum_alignment()` formula with various inputs
  - Test caching behavior with repeated conformations
  - _Requirements: 10.1, 10.2_

- [x] 2. Enhance move evaluator with QCPP
  - Modify `ubf_protein/mapless_moves.py` `CapabilityBasedMoveEvaluator.__init__()` to accept optional `qcpp_integration` parameter
  - Replace `_quantum_alignment_factor()` implementation to use QCPP when available
  - Add fallback to existing QAAP calculation when QCPP is None
  - Implement phi pattern reward logic: if phi_match > 0.8, apply -50 kcal/mol energy bonus
  - Update move weight calculation to use QCPP-derived quantum alignment
  - _Requirements: 1.4, 1.5, 6.1, 6.2_

- [x] 2.1 Write unit tests for enhanced move evaluator
  - Test move evaluator with QCPP integration enabled
  - Test move evaluator falls back to QAAP when QCPP is None
  - Test phi pattern reward application
  - Test move weight calculation with QCPP factors
  - _Requirements: 10.1_

- [x] 3. Implement physics-grounded consciousness
  - Create `ubf_protein/physics_grounded_consciousness.py` with `PhysicsGroundedConsciousness` class extending `ConsciousnessState`
  - Implement `update_from_qcpp_metrics()` method
  - Implement frequency mapping: target_frequency = 15.0 - (qcp_score / 0.5)
  - Implement coherence mapping: target_coherence = 0.2 + (qcpp_coherence * 0.8)
  - Implement exponential smoothing with factor 0.1 for smooth transitions
  - Add bounds enforcement for frequency (3-15 Hz) and coherence (0.2-1.0)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.1 Write unit tests for physics-grounded consciousness
  - Test frequency mapping from various QCP scores
  - Test coherence mapping from various QCPP coherence values
  - Test exponential smoothing transitions
  - Test bounds enforcement
  - _Requirements: 10.3_

- [x] 4. Add QCPP validation to memory system
  - Modify `ubf_protein/models.py` to add `QCPPValidatedMemory` dataclass extending `ConformationalMemory`
  - Add `qcpp_metrics: QCPPMetrics` field to memory
  - Add `qcpp_significance: float` field for QCPP contribution to significance
  - Modify `ubf_protein/memory_system.py` `MemorySystem.store_memory()` to accept QCPP metrics
  - Update significance calculation to include QCPP stability with weight 0.3
  - Implement high-significance detection: qcpp_stability > 1.5 and energy_change < -20
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Write unit tests for QCPP-validated memory
  - Test `QCPPValidatedMemory` dataclass creation
  - Test significance calculation with QCPP metrics
  - Test high-significance threshold detection
  - Test memory storage and retrieval with QCPP data
  - _Requirements: 10.4_

- [x] 5. Implement dynamic parameter adjustment
  - Create `ubf_protein/dynamic_adjustment.py` with `DynamicParameterAdjuster` class
  - Implement `adjust_from_stability()` method
  - Add logic: if stability < 1.0, increase frequency by 2.0 Hz and temperature by 50 K
  - Add logic: if stability > 2.0, decrease frequency by 1.0 Hz and temperature by 20 K
  - Implement bounds enforcement: frequency 3-15 Hz, temperature 100-500 K
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5.1 Write unit tests for dynamic parameter adjustment
  - Test parameter increases in unstable regions (stability < 1.0)
  - Test parameter decreases in stable regions (stability > 2.0)
  - Test bounds enforcement for frequency and temperature
  - _Requirements: 10.1_

- [ ] 6. Integrate QCPP into ProteinAgent
  - Modify `ubf_protein/protein_agent.py` `ProteinAgent.__init__()` to accept optional `qcpp_integration` parameter
  - Replace `ConsciousnessState` with `PhysicsGroundedConsciousness` when QCPP enabled
  - Pass QCPP integration to move evaluator during initialization
  - Add QCPP analysis call after each move execution in `_execute_exploration_step()`
  - Update consciousness from QCPP metrics after each move
  - Apply dynamic parameter adjustment based on QCPP stability
  - _Requirements: 1.1, 3.1, 5.1_

- [ ] 6.1 Write unit tests for integrated ProteinAgent
  - Test agent initialization with QCPP integration
  - Test agent uses physics-grounded consciousness when QCPP enabled
  - Test agent updates consciousness from QCPP metrics
  - Test agent applies dynamic parameter adjustment
  - _Requirements: 10.1, 10.3_

- [ ] 7. Integrate QCPP into MultiAgentCoordinator
  - Modify `ubf_protein/multi_agent_coordinator.py` `MultiAgentCoordinator.__init__()` to accept optional `qcpp_integration` parameter
  - Pass QCPP integration to agents during `initialize_agents()`
  - Store QCPP integration reference for trajectory recording
  - _Requirements: 1.1, 2.1_

- [ ] 7.1 Write unit tests for integrated MultiAgentCoordinator
  - Test coordinator initialization with QCPP integration
  - Test coordinator passes QCPP to agents
  - Test coordinator stores QCPP reference
  - _Requirements: 10.1_

- [ ] 8. Implement integrated trajectory recording
  - Create `ubf_protein/integrated_trajectory.py` with `IntegratedTrajectoryRecorder` class
  - Implement `IntegratedTrajectoryPoint` dataclass with UBF metrics (rmsd, energy, consciousness) and QCPP metrics (qcp, coherence, stability, phi_match)
  - Implement `record_point()` method that captures both UBF and QCPP metrics
  - Implement `export_to_json()` method for trajectory data
  - Implement `TrajectoryAnalyzer` class with correlation analysis methods
  - Add `calculate_qcpp_rmsd_correlation()` method using scipy.stats.pearsonr
  - _Requirements: 2.3, 2.4, 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 8.1 Write unit tests for integrated trajectory recording
  - Test `IntegratedTrajectoryPoint` dataclass creation
  - Test trajectory point recording with both UBF and QCPP metrics
  - Test JSON export format
  - Test correlation analysis calculations
  - _Requirements: 10.5_

- [ ] 9. Wire trajectory recording into coordinator
  - Modify `ubf_protein/multi_agent_coordinator.py` to use `IntegratedTrajectoryRecorder` when QCPP enabled
  - Update `run_parallel_exploration()` to record QCPP metrics at each iteration
  - Add correlation analysis after exploration completes
  - Include correlation results in `ExplorationResults`
  - _Requirements: 2.2, 2.3, 2.5_

- [ ] 9.1 Write unit tests for coordinator trajectory integration
  - Test coordinator records QCPP metrics during exploration
  - Test coordinator computes correlations after exploration
  - Test correlation results included in exploration results
  - _Requirements: 10.5_

- [ ] 10. Add configuration and backward compatibility
  - Create `ubf_protein/qcpp_config.py` with `QCPPIntegrationConfig` dataclass
  - Add configuration fields: enabled, analysis_frequency, cache_size, max_calculation_time_ms, phi_reward_threshold, phi_reward_energy, enable_dynamic_adjustment, stability thresholds, enable_physics_grounding, smoothing_factor
  - Implement configuration validation
  - Ensure all components check `enabled` flag and fall back gracefully when False
  - Add configuration parameter to coordinator and agent constructors
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 10.1 Write unit tests for configuration and backward compatibility
  - Test UBF operates without QCPP when not provided
  - Test configuration flag disables integration
  - Test all components fall back gracefully
  - _Requirements: 9.1, 9.2, 9.5_

- [ ] 11. Create integration example script
  - Create `ubf_protein/examples/integrated_exploration.py`
  - Demonstrate initializing QCPP integration adapter
  - Demonstrate creating coordinator with QCPP integration
  - Demonstrate running exploration with QCPP feedback
  - Demonstrate analyzing trajectory with correlation results
  - Add command-line arguments for protein sequence, iterations, agents
  - _Requirements: 1.1, 2.1, 8.1_

- [ ] 12. Add performance monitoring and optimization
  - Add timing instrumentation to QCPP analysis calls
  - Implement adaptive analysis frequency based on calculation time
  - Add performance metrics to trajectory data (qcpp_calculation_time_ms)
  - Log warnings when QCPP analysis exceeds 5ms threshold
  - Implement cache hit rate monitoring
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 12.1 Write performance tests
  - Test QCPP analysis completes within 5ms per conformation
  - Test multi-agent exploration completes within 5 minutes (10 agents, 2000 iterations)
  - Test throughput maintains ≥50 conformations/second/agent
  - Test memory overhead remains acceptable
  - _Requirements: 7.1, 7.2, 7.5_

- [ ] 13. Update documentation
  - Update `ubf_protein/README.md` with QCPP integration section
  - Add integration architecture diagram
  - Add usage examples for QCPP integration
  - Document configuration options
  - Add performance considerations section
  - Update `ubf_protein/API.md` with new classes and methods
  - Add integration examples to `ubf_protein/EXAMPLES.md`
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_

- [ ] 14. Create end-to-end validation script
  - Create `validate_qcpp_ubf_integration.py` in root directory
  - Run integrated exploration on test protein (ubiquitin)
  - Verify QCPP metrics are recorded in trajectory
  - Verify correlation analysis produces results
  - Verify final RMSD improves compared to non-integrated baseline
  - Generate comparison report with visualizations
  - _Requirements: 2.4, 2.5, 8.5_
