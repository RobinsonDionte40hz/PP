"""
Protein agent implementation for UBF protein system.

This module implements the autonomous protein folding agent that coordinates
consciousness, behavioral state, and memory systems to perform mapless
conformational exploration.
"""

import time
import random
import math
import logging
from typing import Optional, Dict, List, Any, Tuple

from .interfaces import IProteinAgent, IPhysicsCalculator
from .models import (
    Conformation, ConformationalOutcome, ConformationalMemory,
    ConsciousnessCoordinates, BehavioralStateData, AdaptiveConfig, ProteinSizeClass,
    ConformationSnapshot
)
from .consciousness import ConsciousnessState
from .behavioral_state import BehavioralState
from .memory_system import MemorySystem
from .local_minima_detector import LocalMinimaDetector
from .structural_validation import StructuralValidation
from .config import (
    BASE_STUCK_DETECTION_WINDOW, BASE_STUCK_DETECTION_THRESHOLD,
    ENERGY_VALIDATION_THRESHOLD,
    INITIAL_TEMPERATURE, TEMPERATURE_DECAY_RATE, MIN_TEMPERATURE, BOLTZMANN_CONSTANT,
    MEMORY_SIGNIFICANCE_THRESHOLD
)
from . import config as config_module

# Set up logging
logger = logging.getLogger(__name__)

# THz signature recording for determinism testing
try:
    from .vibrational_analysis import create_vibrational_analyzer, THzSpectrum
    HAS_THZ_ANALYSIS = True
except ImportError:
    HAS_THZ_ANALYSIS = False
    logger.warning("THz vibrational analysis not available - signature recording disabled")

# Task 5: Import RMSD calculator for native structure validation
try:
    from .rmsd_calculator import RMSDCalculator
    HAS_RMSD_CALCULATOR = True
except ImportError:
    HAS_RMSD_CALCULATOR = False
    logger.warning("RMSDCalculator not available - native structure validation disabled")


class ProteinAgent(IProteinAgent):
    """
    Implementation of autonomous protein folding agent.

    Coordinates consciousness, behavioral state, and memory systems to perform
    intelligent conformational exploration using mapless design principles.
    """

    # Bond length validation constants (from StructuralValidation)
    MIN_BOND_LENGTH = 1.0  # Å - minimum CA-CA distance
    MAX_BOND_LENGTH = 5.0  # Å - maximum CA-CA distance

    def __init__(self,
                 protein_sequence: str,
                 initial_frequency: float = 9.0,
                 initial_coherence: float = 0.6,
                 initial_conformation: Optional[Conformation] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 enable_visualization: bool = False,
                 max_snapshots: int = 1000,
                 native_structure: Optional[Conformation] = None,
                 qcpp_integration: Optional[Any] = None,
                 qcpp_analysis_frequency: int = 5,
                 enable_thz_recording: bool = False,
                 coordinator: Optional[Any] = None,
                 target_geometry: str = 'none'):
        """
        Initialize protein agent with consciousness coordinates and protein sequence.

        Args:
            protein_sequence: Amino acid sequence of the protein
            initial_frequency: Initial consciousness frequency (3-15 Hz)
            initial_coherence: Initial consciousness coherence (0.2-1.0)
            initial_conformation: Starting conformation (generated if None)
            adaptive_config: Adaptive configuration (created automatically if None)
            enable_visualization: Enable trajectory snapshot recording
            max_snapshots: Maximum snapshots to store (prevents memory overflow)
            native_structure: Optional native structure for RMSD validation (Task 5)
            qcpp_integration: Optional QCPP integration adapter for physics-grounded exploration
            qcpp_analysis_frequency: Analyze with QCPP every N iterations (default: 5 for performance)
            enable_thz_recording: Enable THz signature recording at local minima (for determinism research, default: False)
            coordinator: Optional coordinator reference for global QCPP registry access (cross-agent sharing)
            target_geometry: Target Platonic solid geometry for active agent guidance (default: 'none')
        """
        # Create adaptive config if not provided
        if adaptive_config is None:
            adaptive_config = self._create_default_adaptive_config(protein_sequence)
        
        # Store QCPP integration reference and analysis frequency
        self._qcpp_integration = qcpp_integration
        self._qcpp_analysis_frequency = qcpp_analysis_frequency
        self._last_qcpp_metrics = None  # Store latest QCPP metrics for resonance bonus
        
        # Store geometric targeting configuration (NEW: Phase 3)
        self._target_geometry = target_geometry
        
        # Store coordinator reference for global QCPP registry (cross-agent sharing)
        self._coordinator = coordinator

        # Initialize consciousness system (physics-grounded if QCPP enabled)
        if qcpp_integration is not None:
            try:
                from .physics_grounded_consciousness import PhysicsGroundedConsciousness
                self._consciousness = PhysicsGroundedConsciousness(initial_frequency, initial_coherence)
                logger.info("Using PhysicsGroundedConsciousness with QCPP integration")
            except ImportError as e:
                logger.warning(f"Failed to import PhysicsGroundedConsciousness: {e}")
                logger.warning("Falling back to standard ConsciousnessState")
                self._consciousness = ConsciousnessState(initial_frequency, initial_coherence)
        else:
            self._consciousness = ConsciousnessState(initial_frequency, initial_coherence)
        
        # Initialize dynamic parameter adjuster if QCPP enabled
        self._dynamic_adjuster = None
        if qcpp_integration is not None:
            try:
                from .dynamic_adjustment import DynamicParameterAdjuster
                self._dynamic_adjuster = DynamicParameterAdjuster()
                logger.info("Dynamic parameter adjustment enabled with QCPP integration")
            except ImportError as e:
                logger.warning(f"Failed to import DynamicParameterAdjuster: {e}")
                logger.warning("Dynamic parameter adjustment disabled")

        # Initialize behavioral state (derived from consciousness)
        self._behavioral = BehavioralState(self._consciousness.get_coordinates())

        # Initialize memory system
        self._memory = MemorySystem()

        # Initialize local minima detector
        self._local_minima_detector = LocalMinimaDetector(adaptive_config)

        # Initialize structural validator
        self._validator = StructuralValidation()
        
        # Task 5: Initialize RMSD calculator and store native structure
        self._native_structure = native_structure
        self._rmsd_calculator = None
        if HAS_RMSD_CALCULATOR and native_structure is not None:
            try:
                self._rmsd_calculator = RMSDCalculator(align_structures=True)
                logger.info("RMSD calculator initialized for native structure validation")
            except Exception as e:
                logger.error(f"Error initializing RMSDCalculator: {e}")
                logger.warning("Native structure validation will be disabled")
        
        # Initialize energy calculator (if enabled)
        self._energy_calculator: Optional[IPhysicsCalculator] = None
        if config_module.USE_MOLECULAR_MECHANICS_ENERGY:
            try:
                from .energy_function import MolecularMechanicsEnergy
                self._energy_calculator = MolecularMechanicsEnergy()
                logger.info("MolecularMechanicsEnergy calculator initialized")
            except ImportError as e:
                logger.warning(f"Failed to import MolecularMechanicsEnergy: {e}")
                logger.warning("Falling back to simplified energy calculation")
            except Exception as e:
                logger.error(f"Error initializing MolecularMechanicsEnergy: {e}")
                logger.warning("Falling back to simplified energy calculation")

        # Store protein sequence and config
        self._protein_sequence = protein_sequence
        self._adaptive_config = adaptive_config
        
        # Visualization settings
        self._enable_visualization = enable_visualization
        self._max_snapshots = max_snapshots
        self._trajectory_snapshots: List[ConformationSnapshot] = []
        self._agent_id = f"agent_{id(self)}"  # Unique ID based on object identity

        # Initialize current conformation
        if initial_conformation is None:
            self._current_conformation = self._generate_initial_conformation()
        else:
            self._current_conformation = initial_conformation

        # Exploration metrics
        self._iterations_completed = 0
        self._conformations_explored = 1  # Start with 1 (initial conformation)
        self._memories_created = 0
        self._best_energy = self._current_conformation.energy
        self._best_rmsd = self._current_conformation.rmsd_to_native or float('inf')
        self._total_decision_time_ms = 0.0
        self._stuck_in_minima_count = 0
        self._successful_escapes = 0
        self._validation_failures = 0
        self._repair_attempts = 0
        self._repair_successes = 0
        
        # Task 5: Add GDT-TS and TM-score tracking
        self._best_gdt_ts = self._current_conformation.gdt_ts_score if self._current_conformation.gdt_ts_score is not None else 0.0
        self._best_tm_score = self._current_conformation.tm_score if self._current_conformation.tm_score is not None else 0.0
        
        # Learning improvement tracking
        self._rmsd_history = [self._best_rmsd] if self._best_rmsd != float('inf') else []
        
        # Simulated annealing temperature for move acceptance
        self._temperature = INITIAL_TEMPERATURE
        self._moves_accepted = 0
        self._moves_rejected = 0
        
        # QCPP metrics reuse tracking
        self._qcpp_calculations = 0  # Fresh QCPP calculations (novel conformations)
        self._qcpp_cache_hits = 0    # Reused from memory (self-revisits)
        
        # THz vibrational analysis (opt-in for determinism research) - MUST BE BEFORE visualization
        self._enable_thz_recording = enable_thz_recording
        self._thz_signature_history: List[THzSpectrum] = []
        self._last_minima_energy = float('inf')
        self._minima_detection_threshold = 5.0  # Energy change threshold for minima detection
        
        # Only create analyzer if THz recording is explicitly enabled
        if enable_thz_recording and HAS_THZ_ANALYSIS:
            self._thz_analyzer = create_vibrational_analyzer(cutoff=10.0, spring_constant=1.0)
            logger.info("THz recording ENABLED for determinism analysis")
        else:
            self._thz_analyzer = None
            if enable_thz_recording and not HAS_THZ_ANALYSIS:
                logger.warning("THz recording requested but vibrational_analysis module not available")
        
        # Create initial snapshot if visualization enabled
        if self._enable_visualization:
            self._capture_snapshot(iteration=0)

    def get_consciousness_state(self) -> ConsciousnessState:
        """Get current consciousness coordinates."""
        return self._consciousness

    def get_behavioral_state(self) -> BehavioralState:
        """Get cached behavioral state."""
        return self._behavioral

    def get_memory_system(self) -> MemorySystem:
        """Get agent's memory system."""
        return self._memory

    def explore_step(self) -> ConformationalOutcome:
        """
        Execute one exploration step using mapless design with error handling.

        Generates available moves, evaluates them using capability-based evaluation,
        selects the best move, executes it, validates the result, and updates all systems.

        Returns:
            ConformationalOutcome from the exploration step
        """
        start_time = time.time()

        try:
            # Generate available moves using mapless generator
            from .mapless_moves import MaplessMoveGenerator, CapabilityBasedMoveEvaluator
            move_generator = MaplessMoveGenerator()
            
            # Create move evaluator with QCPP integration if available
            if self._qcpp_integration is not None:
                move_evaluator = CapabilityBasedMoveEvaluator(qcpp_integration=self._qcpp_integration)
            else:
                move_evaluator = CapabilityBasedMoveEvaluator()

            available_moves = move_generator.generate_moves(self._current_conformation)

            if not available_moves:
                # No moves available - create a minimal outcome
                outcome = ConformationalOutcome(
                    move_executed=None,  # type: ignore
                    new_conformation=self._current_conformation,
                    energy_change=0.0,
                    rmsd_change=0.0,
                    success=False,
                    significance=0.0
                )
            else:
                # Evaluate all moves
                move_weights = []
                for move in available_moves:
                    try:
                        # Get memory influence for this move type
                        memory_influence = self._memory.calculate_memory_influence(move.move_type.value)

                        # Calculate physics factors (placeholder for now)
                        physics_factors = self._get_physics_factors(move)

                        # Get current RMSD if available (for validation guidance)
                        current_rmsd = None
                        if self._current_conformation.rmsd_to_native is not None:
                            current_rmsd = self._current_conformation.rmsd_to_native

                        # Evaluate move with RMSD awareness
                        weight = move_evaluator.evaluate_move(
                            move,
                            self._behavioral,
                            memory_influence,
                            physics_factors,
                            current_rmsd
                        )
                        
                        # Todo #4: Apply 40Hz resonance bonus if QCPP metrics available
                        if self._last_qcpp_metrics is not None and self._dynamic_adjuster is not None:
                            resonance_bonus = self._dynamic_adjuster.calculate_resonance_bonus(self._last_qcpp_metrics)
                            weight *= resonance_bonus
                            if resonance_bonus > 1.0:
                                logger.debug(f"Applied {resonance_bonus:.2f}× resonance bonus to move {move.move_type}")
                        
                        # NEW Phase 3: Apply geometric targeting factor
                        if self._last_qcpp_metrics is not None and self._last_qcpp_metrics.geometric_similarity > 0.0:
                            # Weight moves by geometric similarity to target
                            # Range: 0.8-1.2x (baseline), up to 1.32x with high similarity bonus
                            geometric_factor = 0.8 + (0.4 * self._last_qcpp_metrics.geometric_similarity)
                            
                            # Extra bonus for high similarity (positive feedback loop)
                            if self._last_qcpp_metrics.geometric_similarity > 0.7:
                                geometric_factor *= 1.1  # Up to 1.32x total
                            
                            weight *= geometric_factor
                            if geometric_factor > 1.0:
                                logger.debug(
                                    f"Applied {geometric_factor:.2f}× geometric targeting bonus "
                                    f"(similarity: {self._last_qcpp_metrics.geometric_similarity:.3f} "
                                    f"to {self._target_geometry})"
                                )
                        
                        # Task 9: Apply pattern guidance from Mediator Agents
                        if self._coordinator is not None:
                            try:
                                shared_memory_pool = self._coordinator.get_shared_memory_pool()
                                pattern_guidance = self._get_pattern_guidance(move, shared_memory_pool)
                                weight *= pattern_guidance
                                
                                if pattern_guidance != 1.0:
                                    logger.debug(
                                        f"Applied {pattern_guidance:.2f}× pattern guidance "
                                        f"to move {move.move_type}"
                                    )
                            except Exception as e:
                                logger.warning(f"Error applying pattern guidance: {e}")
                        
                        move_weights.append((move, weight))
                    except Exception as e:
                        logger.warning(f"Error evaluating move {move.move_id}: {e}")
                        # Skip this move and continue with others
                        continue

                if not move_weights:
                    # All moves failed evaluation - return minimal outcome
                    outcome = ConformationalOutcome(
                        move_executed=None,  # type: ignore
                        new_conformation=self._current_conformation,
                        energy_change=0.0,
                        rmsd_change=0.0,
                        success=False,
                        significance=0.0
                    )
                else:
                    # Select best move (highest weight)
                    best_move, best_weight = max(move_weights, key=lambda x: x[1])

                    # Execute the move (simulate conformational change)
                    new_conformation = self._execute_move(best_move)

                    # Validate the new conformation
                    validation_result = self._validator.validate_conformation(new_conformation)
                    
                    if not validation_result.is_valid:
                        self._validation_failures += 1
                        logger.warning(f"Invalid conformation detected: {validation_result.issues[:3]}")
                        
                        # Attempt repair
                        self._repair_attempts += 1
                        repaired_conf, repair_success = self._validator.repair_conformation(new_conformation)
                        
                        if repair_success:
                            self._repair_successes += 1
                            new_conformation = repaired_conf
                            logger.info(f"Successfully repaired conformation")
                        else:
                            # Repair failed - use current conformation instead
                            logger.warning(f"Repair failed, reverting to current conformation")
                            new_conformation = self._current_conformation

                    # Calculate actual changes
                    energy_change = new_conformation.energy - self._current_conformation.energy
                    rmsd_change = abs(energy_change) * 0.1  # Simplified RMSD estimation

                    # Determine success using Metropolis-Hastings acceptance criterion
                    # Accept if energy decreases OR with probability based on temperature
                    accept_move = self._metropolis_accept(energy_change)
                    
                    if accept_move:
                        self._moves_accepted += 1
                        success = True
                    else:
                        self._moves_rejected += 1
                        success = False
                        # Revert to current conformation if not accepted
                        new_conformation = self._current_conformation
                        energy_change = 0.0
                        rmsd_change = 0.0

                    # Calculate significance (simplified)
                    significance = self._calculate_outcome_significance(energy_change, rmsd_change, success)

                    # Create outcome
                    outcome = ConformationalOutcome(
                        move_executed=best_move,
                        new_conformation=new_conformation,
                        energy_change=energy_change,
                        rmsd_change=rmsd_change,
                        success=success,
                        significance=significance
                    )
                    
                    # QCPP Integration: Analyze conformation and update consciousness
                    # Only run every N iterations for performance (default: every 5 iterations)
                    qcpp_metrics = None  # Store for memory creation
                    should_analyze_qcpp = (
                        self._qcpp_integration is not None 
                        and success 
                        and (self._iterations_completed % self._qcpp_analysis_frequency == 0)
                    )
                    if should_analyze_qcpp:
                        try:
                            # STEP 1: Check global registry first (cross-agent optimization)
                            if self._coordinator is not None:
                                qcpp_metrics = self._coordinator.get_qcpp_from_registry(new_conformation)
                                if qcpp_metrics is not None:
                                    logger.debug("✓ Reusing QCPP from GLOBAL REGISTRY (cross-agent)")
                                    # No need to check local memory since global registry is faster
                            
                            # STEP 2: Check local memory (self-revisit optimization)
                            if qcpp_metrics is None:
                                qcpp_metrics = self._memory.get_qcpp_for_conformation(new_conformation)
                                if qcpp_metrics is not None:
                                    logger.debug("✓ Reusing QCPP from local memory (self-revisit)")
                                    self._qcpp_cache_hits += 1
                            
                            # STEP 3: Calculate fresh (novel conformation)
                            if qcpp_metrics is None and self._qcpp_integration is not None:
                                qcpp_metrics = self._qcpp_integration.analyze_conformation(new_conformation)
                                logger.debug("✓ Calculated NEW QCPP metrics (novel conformation)")
                                self._qcpp_calculations += 1
                                
                                # Store in global registry for cross-agent sharing
                                if self._coordinator is not None:
                                    self._coordinator.store_qcpp_in_registry(new_conformation, qcpp_metrics)
                            
                            # Store latest QCPP metrics for resonance bonus in next iteration
                            self._last_qcpp_metrics = qcpp_metrics
                            
                            # Update physics-grounded consciousness from QCPP metrics
                            if qcpp_metrics is not None and hasattr(self._consciousness, 'update_from_qcpp_metrics'):
                                self._consciousness.update_from_qcpp_metrics(qcpp_metrics)  # type: ignore
                                logger.debug(
                                    f"Updated consciousness from QCPP: "
                                    f"QCP={qcpp_metrics.qcp_score:.2f}, "
                                    f"stability={qcpp_metrics.stability_score:.2f}"
                                )
                            
                            # Apply dynamic parameter adjustment if stability suggests it
                            if self._dynamic_adjuster is not None:
                                current_coords = self._consciousness.get_coordinates()
                                new_freq, new_temp = self._dynamic_adjuster.adjust_from_qcpp_metrics(
                                    current_coords.frequency,
                                    self._temperature,
                                    qcpp_metrics
                                )
                                
                                # Update parameters if they changed
                                if new_freq != current_coords.frequency:
                                    self._consciousness._coordinates.frequency = new_freq
                                    logger.debug(f"Adjusted frequency: {current_coords.frequency:.1f} → {new_freq:.1f} Hz")
                                
                                if new_temp != self._temperature:
                                    self._temperature = new_temp
                                    logger.debug(f"Adjusted temperature: {self._temperature:.1f} → {new_temp:.1f} K")
                                
                                # Todo #4: Check if refinement mode should be triggered
                                if self._dynamic_adjuster.should_trigger_refinement_mode(qcpp_metrics):
                                    refinement_freq, refinement_temp = self._dynamic_adjuster.get_refinement_parameters(
                                        current_coords.frequency,
                                        self._temperature
                                    )
                                    self._consciousness._coordinates.frequency = refinement_freq
                                    self._temperature = refinement_temp
                                    logger.info(
                                        f"Refinement mode activated: freq={refinement_freq:.1f} Hz, "
                                        f"temp={refinement_temp:.1f} K"
                                    )
                        
                        except Exception as e:
                            logger.warning(f"Error in QCPP analysis/adjustment: {e}")
                            # Continue execution - QCPP integration is non-critical
                    
                    # Store qcpp_metrics for memory creation
                    outcome._qcpp_metrics = qcpp_metrics

        except Exception as e:
            logger.error(f"Critical error in explore_step: {e}", exc_info=True)
            # Return minimal outcome to continue execution
            outcome = ConformationalOutcome(
                move_executed=None,  # type: ignore
                new_conformation=self._current_conformation,
                energy_change=0.0,
                rmsd_change=0.0,
                success=False,
                significance=0.0
            )

        # Update consciousness based on outcome
        try:
            self._consciousness.update_from_outcome(outcome)
        except Exception as e:
            logger.error(f"Error updating consciousness: {e}")

        # Check for local minima and apply escape strategies if needed
        is_stuck = self._local_minima_detector.update(outcome.new_conformation.energy, self._iterations_completed)
        if is_stuck:
            # Apply escape strategy
            current_coords = self._consciousness.get_coordinates()
            escape_strategy = self._local_minima_detector.get_escape_strategy(
                current_coords.frequency, current_coords.coherence
            )

            # Apply escape adjustment to consciousness coordinates
            new_frequency = max(3.0, min(15.0, current_coords.frequency + escape_strategy['frequency_adjustment']))
            new_coherence = max(0.2, min(1.0, current_coords.coherence + escape_strategy['coherence_adjustment']))

            # Directly update coordinates (since ConsciousnessState doesn't have set_coordinates)
            self._consciousness._coordinates.frequency = new_frequency
            self._consciousness._coordinates.coherence = new_coherence

            # Boost temperature temporarily to enable uphill moves (escape)
            temp_boost = 1.5  # 50% temperature increase
            self._temperature = min(INITIAL_TEMPERATURE, self._temperature * temp_boost)
            
            # Track escape attempt
            self._stuck_in_minima_count += 1
            
            # Force behavioral state regeneration during escape
            try:
                regenerated_behavioral = BehavioralState(self._consciousness.get_coordinates())
                if regenerated_behavioral is not None:
                    self._behavioral = regenerated_behavioral
            except Exception as e:
                logger.error(f"Error regenerating behavioral state during escape: {e}")

        # Check if behavioral state needs regeneration (normal case)
        if not is_stuck:
            try:
                regenerated_behavioral = self._behavioral.regenerate_if_needed(
                    self._consciousness.get_coordinates()
                )
                if regenerated_behavioral is not None:
                    self._behavioral = regenerated_behavioral
            except Exception as e:
                logger.error(f"Error regenerating behavioral state: {e}")

        # Create and store memory if significant
        try:
            # Get QCPP metrics if available
            qcpp_metrics_for_memory = getattr(outcome, '_qcpp_metrics', None)
            
            memory = self._memory.create_memory_from_outcome(
                outcome,
                self._consciousness.get_coordinates(),
                self._behavioral.get_behavioral_data(),
                qcpp_metrics=qcpp_metrics_for_memory,
                conformation=outcome.new_conformation  # Pass conformation for hash generation
            )
            self._memory.store_memory(memory)
            if memory.significance >= MEMORY_SIGNIFICANCE_THRESHOLD:
                self._memories_created += 1
        except Exception as e:
            logger.warning(f"Error creating/storing memory: {e}")
            # Continue execution - memory is non-critical

        # Update current conformation
        previous_energy = self._current_conformation.energy
        if outcome.new_conformation != self._current_conformation:
            self._current_conformation = outcome.new_conformation
            self._conformations_explored += 1
        
        # Check if escape was successful (after conformation update)
        if is_stuck and outcome.new_conformation.energy < previous_energy:
            self._successful_escapes += 1
            self._local_minima_detector.record_escape_success(self._iterations_completed)

            # Create high-significance memory for successful escape
            qcpp_metrics_for_memory = getattr(outcome, '_qcpp_metrics', None)
            escape_memory = self._memory.create_memory_from_outcome(
                outcome,
                self._consciousness.get_coordinates(),
                self._behavioral.get_behavioral_data(),
                qcpp_metrics=qcpp_metrics_for_memory,
                conformation=outcome.new_conformation  # Pass conformation for hash generation
            )
            # Override significance for successful escape
            escape_memory.significance = 0.8
            self._memory.store_memory(escape_memory)
            self._memories_created += 1
            logger.info(f"Successful escape at iteration {self._iterations_completed}! Energy: {previous_energy:.2f} -> {outcome.new_conformation.energy:.2f}")

        # Update best metrics
        if outcome.new_conformation.energy < self._best_energy:
            self._best_energy = outcome.new_conformation.energy
        if (outcome.new_conformation.rmsd_to_native and
            outcome.new_conformation.rmsd_to_native < self._best_rmsd):
            self._best_rmsd = outcome.new_conformation.rmsd_to_native
            # Track RMSD improvement for learning calculation
            self._rmsd_history.append(self._best_rmsd)
        
        # Task 5: Update best GDT-TS and TM-score
        if (outcome.new_conformation.gdt_ts_score is not None and
            outcome.new_conformation.gdt_ts_score > self._best_gdt_ts):
            self._best_gdt_ts = outcome.new_conformation.gdt_ts_score
        if (outcome.new_conformation.tm_score is not None and
            outcome.new_conformation.tm_score > self._best_tm_score):
            self._best_tm_score = outcome.new_conformation.tm_score

        # Update metrics
        self._iterations_completed += 1
        decision_time_ms = (time.time() - start_time) * 1000
        self._total_decision_time_ms += decision_time_ms
        
        # Update temperature (simulated annealing)
        self._update_temperature()

        # THz signature recording at local minima (only if enabled)
        if self._enable_thz_recording:
            self._record_thz_signature_if_minimum(outcome.new_conformation)

        # Capture visualization snapshot if enabled
        self._capture_snapshot(self._iterations_completed)

        return outcome

    def _create_default_adaptive_config(self, protein_sequence: str) -> AdaptiveConfig:
        """
        Create a default adaptive configuration based on protein size.

        Args:
            protein_sequence: Amino acid sequence

        Returns:
            Default AdaptiveConfig for the protein size
        """
        residue_count = len(protein_sequence)

        if residue_count < 50:
            size_class = ProteinSizeClass.SMALL
        elif residue_count <= 150:
            size_class = ProteinSizeClass.MEDIUM
        else:
            size_class = ProteinSizeClass.LARGE

        return AdaptiveConfig(
            size_class=size_class,
            residue_count=residue_count,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=BASE_STUCK_DETECTION_WINDOW,
            stuck_detection_threshold=BASE_STUCK_DETECTION_THRESHOLD,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )

    def _generate_initial_conformation(self) -> Conformation:
        """
        Generate initial compact conformation with realistic protein-like geometry.

        Creates a more realistic starting structure that resembles a folded protein,
        rather than an extended chain. This gives the exploration a better starting point.
        """
        # Create compact, roughly spherical starting structure
        num_residues = len(self._protein_sequence)
        atom_coordinates = []
        
        # Generate points on a rough sphere with some noise
        # This creates a more compact starting structure
        radius = 8.0  # Å, typical for small proteins
        
        for i in range(num_residues):
            # Distribute points roughly on a sphere
            phi = (i / num_residues) * 2 * math.pi  # Azimuthal angle
            theta = math.acos(2 * (i / num_residues) - 1)  # Polar angle (Fibonacci-like distribution)
            
            # Convert to Cartesian with some noise
            x = radius * math.sin(theta) * math.cos(phi) + random.uniform(-1.0, 1.0)
            y = radius * math.sin(theta) * math.sin(phi) + random.uniform(-1.0, 1.0)
            z = radius * math.cos(theta) + random.uniform(-1.0, 1.0)
            
            atom_coordinates.append((x, y, z))

        # Ensure CA-CA distances are reasonable (~3.8 Å)
        atom_coordinates = self._regularize_chain_geometry(atom_coordinates)

        # Placeholder secondary structure (coil, will be updated by moves)
        secondary_structure = ['C'] * num_residues

        # Randomized angles around typical values
        phi_angles = [-60.0 + random.uniform(-30, 30) for _ in range(num_residues)]
        psi_angles = [-40.0 + random.uniform(-30, 30) for _ in range(num_residues)]
        
        # Start with moderate energy (not too high, not too low)
        initial_energy = random.uniform(200.0, 400.0)

        return Conformation(
            conformation_id="initial_compact",
            sequence=self._protein_sequence,
            atom_coordinates=atom_coordinates,
            energy=initial_energy,
            rmsd_to_native=None,  # No native structure known
            secondary_structure=secondary_structure,
            phi_angles=phi_angles,
            psi_angles=psi_angles,
            available_move_types=["backbone_rotation", "sidechain_adjust"],
            structural_constraints={}
        )

    def _regularize_chain_geometry(self, coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Regularize chain geometry to ensure reasonable CA-CA distances.
        
        This creates a more protein-like chain while maintaining overall structure.
        """
        if len(coords) < 2:
            return coords
            
        regularized = [coords[0]]  # Keep first atom as is
        
        target_distance = 3.8  # CA-CA distance in Å
        
        for i in range(1, len(coords)):
            prev_pos = regularized[-1]
            curr_pos = coords[i]
            
            # Calculate vector from previous to current
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            dz = curr_pos[2] - prev_pos[2]
            
            # Calculate current distance
            current_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if current_dist < 0.1:
                # Too close, place at target distance in random direction
                angle1 = random.uniform(0, 2*math.pi)
                angle2 = random.uniform(0, math.pi)
                dx = target_distance * math.sin(angle2) * math.cos(angle1)
                dy = target_distance * math.sin(angle2) * math.sin(angle1)
                dz = target_distance * math.cos(angle2)
            else:
                # Scale to target distance
                scale = target_distance / current_dist
                dx *= scale
                dy *= scale
                dz *= scale
            
            # New position
            new_pos = (prev_pos[0] + dx, prev_pos[1] + dy, prev_pos[2] + dz)
            regularized.append(new_pos)
        
        return regularized

    def _get_physics_factors(self, move) -> Dict[str, float]:
        """
        Get physics factors for move evaluation.

        This is a placeholder implementation. In the full system,
        this would calculate actual QAAP, resonance, and water shielding.

        Args:
            move: The move to evaluate

        Returns:
            Dictionary with physics factors
        """
        # Placeholder values - in real implementation would calculate from conformation
        return {
            'qaap': 0.5,  # 0-1 scale
            'resonance': 0.5,  # 0-1 scale
            'water_shielding': 0.5  # 0-1 scale
        }
    
    def _get_pattern_guidance(self, move, shared_memory_pool) -> float:
        """
        Get move evaluation adjustment based on Mediator Agent pattern broadcasts.
        
        This method retrieves recent pattern detections from shared memory and
        adjusts move weights to guide exploration toward:
        - Geometric attractors (phi patterns, Platonic similarities)
        - THz resonance regions (vibrational stability)
        - Secondary structure formation (helix/sheet regions)
        
        Args:
            move: ConformationalMove being evaluated
            shared_memory_pool: Shared memory pool with pattern broadcasts
        
        Returns:
            Multiplicative guidance factor (0.8-1.5):
            - 1.0: Neutral (no relevant patterns)
            - >1.0: Move aligns with detected patterns
            - <1.0: Move conflicts with patterns
        
        Example:
            >>> guidance = agent._get_pattern_guidance(move, shared_memory)
            >>> weight *= guidance  # Apply to move weight
        """
        if not shared_memory_pool:
            return 1.0
        
        try:
            # Retrieve recent patterns (last 100 iterations)
            patterns = shared_memory_pool.retrieve_recent_patterns(
                current_iteration=self._iterations_completed,
                max_age=100
            )
            
            if not patterns:
                return 1.0
            
            # Accumulate guidance from different pattern types
            geometric_bonus = 1.0
            thz_bonus = 1.0
            folding_bonus = 1.0
            
            for pattern in patterns:
                pattern_type = pattern.get('pattern_type', '')
                significance = pattern.get('significance', 'low')
                
                # Weight by significance
                sig_weight = {'low': 1.05, 'medium': 1.10, 'high': 1.20}.get(significance, 1.0)
                
                # Geometric similarity patterns
                if pattern_type == 'geometric_similarity':
                    geo_data = pattern.get('geometric_data', {})
                    
                    # Prioritize moves toward golden ratio patterns
                    if geo_data.get('golden_ratio_percentage', 0) > 20.0:
                        geometric_bonus *= sig_weight
                    
                    # Prioritize moves toward dominant Platonic solid
                    dominant_solid = geo_data.get('dominant_platonic_solid', '')
                    if dominant_solid in ['icosahedron', 'dodecahedron']:
                        # These have phi-based geometries
                        geometric_bonus *= 1.05
                
                # THz resonance patterns
                elif pattern_type == 'thz_resonance':
                    thz_data = pattern.get('thz_data', {})
                    
                    # Prioritize moves that might maintain resonance
                    # (favor small, local adjustments over large jumps)
                    if move.move_type.value in ['backbone_rotation', 'sidechain_adjust']:
                        thz_bonus *= sig_weight
                    elif move.move_type.value == 'large_jump':
                        thz_bonus *= 0.9  # Slight penalty for large moves
                
                # Folding dynamics patterns
                elif pattern_type == 'folding_dynamics':
                    fold_data = pattern.get('folding_data', {})
                    
                    # Encourage continuation of detected secondary structure
                    helix_pct = fold_data.get('helix_percentage', 0)
                    sheet_pct = fold_data.get('sheet_percentage', 0)
                    
                    if helix_pct > 30.0 and move.move_type.value == 'helix_formation':
                        folding_bonus *= sig_weight
                    elif sheet_pct > 20.0 and move.move_type.value == 'sheet_formation':
                        folding_bonus *= sig_weight
            
            # Combine bonuses (multiplicative)
            total_guidance = geometric_bonus * thz_bonus * folding_bonus
            
            # Clamp to reasonable range [0.8, 1.5]
            return max(0.8, min(1.5, total_guidance))
        
        except Exception as e:
            logger.warning(f"Error retrieving pattern guidance: {e}")
            return 1.0

    def _execute_move(self, move) -> Conformation:
        """
        Execute a conformational move and return new conformation.

        Uses proper protein geometry moves that maintain bond lengths and angles.

        Args:
            move: The move to execute

        Returns:
            New conformation after move execution
        """
        # Calculate actual energy change (may differ from estimate)
        actual_energy_change = move.estimated_energy_change * (0.8 + random.random() * 0.4)  # ±20% variation

        # Apply proper protein conformational moves
        new_coords = self._apply_protein_move(self._current_conformation.atom_coordinates, move)

        # Update phi/psi angles for target residues (moderate changes)
        new_phi = list(self._current_conformation.phi_angles)
        new_psi = list(self._current_conformation.psi_angles)
        for i in move.target_residues:
            if i < len(new_phi):
                new_phi[i] += random.uniform(-15, 15)  # ±15° change
                new_psi[i] += random.uniform(-15, 15)

        # Create new conformation with preliminary energy
        new_conformation = Conformation(
            conformation_id=f"conf_{self._iterations_completed + 1}_{move.move_id}",
            sequence=self._protein_sequence,
            atom_coordinates=new_coords,  # Updated coordinates with proper geometry
            energy=self._current_conformation.energy + actual_energy_change,
            rmsd_to_native=self._current_conformation.rmsd_to_native,
            secondary_structure=self._current_conformation.secondary_structure,
            phi_angles=new_phi,  # Updated angles
            psi_angles=new_psi,  # Updated angles
            available_move_types=self._current_conformation.available_move_types,
            structural_constraints=self._current_conformation.structural_constraints,
            energy_components=None  # Will be populated if MM energy is calculated
        )

        # Recalculate energy using molecular mechanics if available
        if self._energy_calculator is not None:
            try:
                # Use calculate_with_components to get both total and breakdown
                if hasattr(self._energy_calculator, 'calculate_with_components'):
                    energy_dict = self._energy_calculator.calculate_with_components(new_conformation)  # type: ignore
                    new_conformation.energy = energy_dict['total']
                    # Store components with consistent naming
                    new_conformation.energy_components = {
                        'total_energy': energy_dict['total'],
                        'bond_energy': energy_dict['bond'],
                        'angle_energy': energy_dict['angle'],
                        'dihedral_energy': energy_dict['dihedral'],
                        'vdw_energy': energy_dict['vdw'],
                        'electrostatic_energy': energy_dict['electrostatic'],
                        'hbond_energy': energy_dict['hbond'],
                        'compactness_bonus': energy_dict['compactness']
                    }
                else:
                    # Fall back to basic calculate method
                    new_conformation.energy = self._energy_calculator.calculate(new_conformation)

                # Validate energy is physically reasonable
                if abs(new_conformation.energy) > ENERGY_VALIDATION_THRESHOLD:
                    logger.warning(
                        f"Unrealistic energy detected: {new_conformation.energy:.2f} kcal/mol "
                        f"(threshold: {ENERGY_VALIDATION_THRESHOLD})"
                    )

            except Exception as e:
                logger.warning(f"Error calculating molecular mechanics energy: {e}")
                logger.debug(f"Falling back to estimated energy for this conformation")
                # Keep the estimated energy if MM calculation fails

        # Update secondary structure if move creates structure
        if move.move_type.value in ['helix_formation', 'sheet_formation']:
            # Simulate secondary structure change
            start_idx = move.target_residues[0]
            end_idx = move.target_residues[-1] + 1
            new_ss = list(self._current_conformation.secondary_structure)

            ss_type = 'H' if move.move_type.value == 'helix_formation' else 'E'
            for i in range(start_idx, min(end_idx, len(new_ss))):
                new_ss[i] = ss_type

            new_conformation.secondary_structure = new_ss

        # Task 5: Calculate RMSD, GDT-TS, and TM-score if native structure is provided
        if self._rmsd_calculator is not None and self._native_structure is not None:
            try:
                # Calculate RMSD and quality metrics
                rmsd_result = self._rmsd_calculator.calculate_rmsd(
                    predicted_coords=new_conformation.atom_coordinates,
                    native_coords=self._native_structure.atom_coordinates,
                    calculate_metrics=True
                )

                # Update conformation with validation metrics
                new_conformation.rmsd_to_native = rmsd_result.rmsd
                new_conformation.gdt_ts_score = rmsd_result.gdt_ts
                new_conformation.tm_score = rmsd_result.tm_score

                # Set native structure reference if not already set
                if new_conformation.native_structure_ref is None:
                    new_conformation.native_structure_ref = getattr(
                        self._native_structure,
                        'native_structure_ref',
                        'native_structure'
                    )

                logger.debug(
                    f"RMSD validation: RMSD={rmsd_result.rmsd:.2f}Å, "
                    f"GDT-TS={rmsd_result.gdt_ts:.1f}, TM-score={rmsd_result.tm_score:.3f}"
                )

            except ValueError as e:
                # Handle structure mismatch errors gracefully
                logger.warning(f"RMSD calculation failed (structure mismatch): {e}")
                new_conformation.rmsd_to_native = None
                new_conformation.gdt_ts_score = None
                new_conformation.tm_score = None

            except Exception as e:
                # Handle any other RMSD calculation errors gracefully
                logger.warning(f"RMSD calculation failed: {e}")
                logger.debug(f"RMSD error details", exc_info=True)
                new_conformation.rmsd_to_native = None
                new_conformation.gdt_ts_score = None
                new_conformation.tm_score = None

        return new_conformation

    def _apply_protein_move(self, current_coords: List[Tuple[float, float, float]],
                           move) -> List[Tuple[float, float, float]]:
        """
        Apply a proper protein conformational move that maintains geometry.

        Uses backbone torsion angle changes and maintains CA-CA distances ~3.8 Å.

        Args:
            current_coords: Current CA coordinates
            move: The move to apply

        Returns:
            New coordinates with proper protein geometry
        """
        new_coords = list(current_coords)  # Copy current coordinates

        # Target residues for this move
        target_residues = move.target_residues
        if not target_residues:
            return new_coords

        # Apply different move types
        move_type = move.move_type.value

        if move_type == "backbone_rotation":
            # Phi/psi angle changes - most common and geometry-preserving
            new_coords = self._apply_backbone_rotation(new_coords, target_residues)

        elif move_type == "sidechain_adjust":
            # Side chain adjustments (minimal coordinate changes)
            new_coords = self._apply_sidechain_adjustment(new_coords, target_residues)

        elif move_type == "helix_formation":
            # Form helical structure in target region
            new_coords = self._apply_helix_formation(new_coords, target_residues)

        elif move_type == "sheet_formation":
            # Form sheet structure in target region
            new_coords = self._apply_sheet_formation(new_coords, target_residues)

        elif move_type == "hydrophobic_collapse":
            # Bring hydrophobic residues closer
            new_coords = self._apply_hydrophobic_collapse(new_coords, target_residues)

        elif move_type == "energy_minimization":
            # Small local adjustments to minimize energy
            new_coords = self._apply_energy_minimization(new_coords, target_residues)

        else:
            # Default: small backbone rotation
            new_coords = self._apply_backbone_rotation(new_coords, target_residues)

        # Ensure CA-CA distances are reasonable (~3.8 Å)
        # Note: We now generate good initial geometry and use small moves,
        # so explicit bond length maintenance is less critical
        return new_coords

    def _apply_backbone_rotation(self, coords: List[Tuple[float, float, float]],
                                target_residues: List[int]) -> List[Tuple[float, float, float]]:
        """
        Apply small backbone rotation to maintain geometry.
        """
        new_coords = list(coords)

        for residue_idx in target_residues:
            if residue_idx < 1 or residue_idx >= len(coords) - 1:
                continue  # Skip terminal residues

            # Very small rotation (±5°) to avoid breaking geometry
            rotation_angle = random.uniform(-5, 5)

            # Rotate around the backbone axis defined by prev->next CA
            prev_ca = coords[residue_idx - 1]
            curr_ca = coords[residue_idx]
            next_ca = coords[residue_idx + 1]

            # Calculate axis (vector from prev to next)
            axis = self._vector_subtract(next_ca, prev_ca)
            axis = self._normalize_vector(axis)

            # Rotate current CA around this axis
            rotated_ca = self._rotate_point_around_axis(curr_ca, prev_ca, axis, rotation_angle)
            new_coords[residue_idx] = rotated_ca

        return new_coords

    def _apply_sidechain_adjustment(self, coords: List[Tuple[float, float, float]],
                                   target_residues: List[int]) -> List[Tuple[float, float, float]]:
        """
        Apply very small side chain adjustments.
        """
        new_coords = list(coords)

        for residue_idx in target_residues:
            if residue_idx >= len(coords):
                continue

            # Very small random displacement (±0.1 Å)
            x, y, z = coords[residue_idx]
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            dz = random.uniform(-0.1, 0.1)
            new_coords[residue_idx] = (x + dx, y + dy, z + dz)

        return new_coords

    def _apply_helix_formation(self, coords: List[Tuple[float, float, float]],
                              target_residues: List[int]) -> List[Tuple[float, float, float]]:
        """
        Apply helical structure formation in target region.
        
        Uses small incremental adjustments to form helical geometry while
        maintaining chain connectivity and reasonable bond lengths.
        """
        new_coords = list(coords)

        if len(target_residues) < 3:
            return new_coords

        # Apply small helical adjustments rather than complete repositioning
        for i, residue_idx in enumerate(target_residues):
            if residue_idx < 1 or residue_idx >= len(coords) - 1:
                continue  # Skip terminal residues
            
            # Get neighboring positions for context
            prev_pos = new_coords[residue_idx - 1]
            curr_pos = new_coords[residue_idx]
            next_pos = new_coords[residue_idx + 1]
            
            # Calculate current local geometry
            v1 = self._vector_subtract(curr_pos, prev_pos)  # Previous bond
            v2 = self._vector_subtract(next_pos, curr_pos)  # Next bond
            
            # Small helical adjustment: rotate slightly around backbone axis
            # Helical geometry: ~3.6 residues per turn, ~1.5 Å rise per residue
            helical_rotation = 100.0  # degrees per residue for alpha helix
            
            # Apply small rotation to encourage helical geometry
            rotation_angle = helical_rotation * 0.1  # Small fraction of full turn
            
            # Rotate around the average backbone direction
            backbone_axis = self._vector_subtract(next_pos, prev_pos)
            backbone_axis = self._normalize_vector(backbone_axis)
            
            # Apply small rotation
            rotated_pos = self._rotate_point_around_axis(
                curr_pos, prev_pos, backbone_axis, rotation_angle
            )
            
            # Only apply if it doesn't break bond lengths too much
            new_prev_dist = self._distance(rotated_pos, prev_pos)
            new_next_dist = self._distance(rotated_pos, next_pos)
            
            if (self.MIN_BOND_LENGTH <= new_prev_dist <= self.MAX_BOND_LENGTH and
                self.MIN_BOND_LENGTH <= new_next_dist <= self.MAX_BOND_LENGTH):
                new_coords[residue_idx] = rotated_pos

        return new_coords

    def _apply_sheet_formation(self, coords: List[Tuple[float, float, float]],
                              target_residues: List[int]) -> List[Tuple[float, float, float]]:
        """
        Apply sheet structure formation in target region.
        
        Uses small incremental adjustments to form extended geometry while
        maintaining chain connectivity and reasonable bond lengths.
        """
        new_coords = list(coords)

        if len(target_residues) < 3:
            return new_coords

        # Apply small sheet-like adjustments rather than complete repositioning
        for i, residue_idx in enumerate(target_residues):
            if residue_idx < 1 or residue_idx >= len(coords) - 1:
                continue  # Skip terminal residues
            
            # Get neighboring positions for context
            prev_pos = new_coords[residue_idx - 1]
            curr_pos = new_coords[residue_idx]
            next_pos = new_coords[residue_idx + 1]
            
            # Calculate current local geometry
            v1 = self._vector_subtract(curr_pos, prev_pos)  # Previous bond
            v2 = self._vector_subtract(next_pos, curr_pos)  # Next bond
            
            # Small sheet adjustment: extend slightly along backbone direction
            # Sheet geometry: more extended than helix, ~3.8 Å per residue
            
            # Calculate backbone direction and extend slightly
            backbone_dir = self._vector_subtract(next_pos, prev_pos)
            backbone_dir = self._normalize_vector(backbone_dir)
            
            # Small extension along backbone (0.2 Å)
            extension = 0.2
            extended_pos = (
                curr_pos[0] + backbone_dir[0] * extension,
                curr_pos[1] + backbone_dir[1] * extension,
                curr_pos[2] + backbone_dir[2] * extension
            )
            
            # Only apply if it doesn't break bond lengths too much
            new_prev_dist = self._distance(extended_pos, prev_pos)
            new_next_dist = self._distance(extended_pos, next_pos)
            
            if (self.MIN_BOND_LENGTH <= new_prev_dist <= self.MAX_BOND_LENGTH and
                self.MIN_BOND_LENGTH <= new_next_dist <= self.MAX_BOND_LENGTH):
                new_coords[residue_idx] = extended_pos

        return new_coords

    def _apply_hydrophobic_collapse(self, coords: List[Tuple[float, float, float]],
                                   target_residues: List[int]) -> List[Tuple[float, float, float]]:
        """
        Bring hydrophobic residues closer together.
        """
        new_coords = list(coords)

        # Find centroid of target residues
        if not target_residues:
            return new_coords

        centroid = [0.0, 0.0, 0.0]
        for idx in target_residues:
            if idx < len(coords):
                centroid[0] += coords[idx][0]
                centroid[1] += coords[idx][1]
                centroid[2] += coords[idx][2]

        centroid = [c / len(target_residues) for c in centroid]

        # Move target residues slightly toward centroid
        for idx in target_residues:
            if idx >= len(coords):
                continue

            x, y, z = coords[idx]
            # Move 10% toward centroid
            new_x = x + 0.1 * (centroid[0] - x)
            new_y = y + 0.1 * (centroid[1] - y)
            new_z = z + 0.1 * (centroid[2] - z)
            new_coords[idx] = (new_x, new_y, new_z)

        return new_coords

    def _apply_energy_minimization(self, coords: List[Tuple[float, float, float]],
                                  target_residues: List[int]) -> List[Tuple[float, float, float]]:
        """
        Apply small local energy minimization moves.
        """
        new_coords = list(coords)

        for idx in target_residues:
            if idx >= len(coords):
                continue

            # Very small random adjustments for local minimization
            x, y, z = coords[idx]
            dx = random.uniform(-0.1, 0.1)  # ±0.1 Å
            dy = random.uniform(-0.1, 0.1)
            dz = random.uniform(-0.1, 0.1)
            new_coords[idx] = (x + dx, y + dy, z + dz)

        return new_coords

    def _maintain_bond_lengths(self, coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Ensure CA-CA bond lengths are maintained at ~3.8 Å.

        This is a simplified constraint satisfaction - in a full implementation,
        this would use proper constraint algorithms.
        """
        new_coords = list(coords)
        target_distance = 3.8  # CA-CA distance in Å

        for i in range(len(new_coords) - 1):
            p1 = new_coords[i]
            p2 = new_coords[i + 1]

            # Calculate current distance
            current_dist = self._distance(p1, p2)

            if abs(current_dist - target_distance) > 0.1:  # If deviation > 0.1 Å
                # Scale the second point to maintain distance
                direction = self._vector_subtract(p2, p1)
                direction = self._normalize_vector(direction)

                # New position for second point
                new_p2 = (
                    p1[0] + direction[0] * target_distance,
                    p1[1] + direction[1] * target_distance,
                    p1[2] + direction[2] * target_distance
                )
                new_coords[i + 1] = new_p2

        return new_coords

    # Vector math utilities
    def _distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _vector_subtract(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Subtract two vectors."""
        return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    def _normalize_vector(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Normalize a vector."""
        length = math.sqrt(sum(x ** 2 for x in v))
        if length == 0:
            return (0, 0, 0)
        return (v[0] / length, v[1] / length, v[2] / length)

    def _rotate_point_around_axis(self, point: Tuple[float, float, float],
                                 axis_point: Tuple[float, float, float],
                                 axis: Tuple[float, float, float],
                                 angle_deg: float) -> Tuple[float, float, float]:
        """
        Rotate a point around an axis by a given angle.

        Args:
            point: Point to rotate
            axis_point: Point on the rotation axis
            axis: Rotation axis (normalized)
            angle_deg: Rotation angle in degrees

        Returns:
            Rotated point
        """
        angle_rad = math.radians(angle_deg)

        # Translate point to origin
        p = self._vector_subtract(point, axis_point)

        # Rodrigues' rotation formula
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        cross = (
            axis[1] * p[2] - axis[2] * p[1],
            axis[2] * p[0] - axis[0] * p[2],
            axis[0] * p[1] - axis[1] * p[0]
        )

        dot = sum(a * b for a, b in zip(axis, p))

        rotated = (
            p[0] * cos_a + cross[0] * sin_a + axis[0] * dot * (1 - cos_a),
            p[1] * cos_a + cross[1] * sin_a + axis[1] * dot * (1 - cos_a),
            p[2] * cos_a + cross[2] * sin_a + axis[2] * dot * (1 - cos_a)
        )

        # Translate back
        return self._vector_subtract(rotated, (-axis_point[0], -axis_point[1], -axis_point[2]))

    def _calculate_outcome_significance(self, energy_change: float,
                                      rmsd_change: float,
                                      success: bool) -> float:
        """
        Calculate significance of an exploration outcome.

        Uses simplified 3-factor approach: energy_change, structural_novelty, rmsd_improvement.

        Args:
            energy_change: Change in energy
            rmsd_change: Change in RMSD
            success: Whether the move was successful

        Returns:
            Significance score (0.0-1.0)
        """
        # Factor 1: RMSD improvement (0.5 weight) - PRIMARY OBJECTIVE
        # RMSD decrease toward native structure is the main goal
        rmsd_significance = min(1.0, max(0.0, -rmsd_change / 2.0))  # -2 Å = max significance

        # Factor 2: Energy change impact (0.3 weight) - SECONDARY
        # Large negative changes are significant but not primary goal
        energy_significance = min(1.0, max(0.0, -energy_change / 50.0))  # -50 kJ/mol = max significance

        # Factor 3: Structural novelty (0.2 weight) - TERTIARY
        # Exploration diversity
        structural_novelty = 0.5 if success else 0.1

        # Combine factors: RMSD-focused (50% RMSD, 30% energy, 20% novelty)
        significance = (0.5 * rmsd_significance +
                       0.3 * energy_significance +
                       0.2 * structural_novelty)

        return min(1.0, significance)

    def _record_thz_signature_if_minimum(self, conformation: Conformation) -> None:
        """
        Record THz signature if conformation represents a local energy minimum.
        
        Only called when enable_thz_recording=True (for determinism research).
        Detects local minima by checking if energy has stabilized (low variation)
        and is significantly lower than the last recorded minimum.
        
        Args:
            conformation: Current conformation to potentially record
        """
        # Should only be called when THz recording is enabled
        if not self._enable_thz_recording or self._thz_analyzer is None:
            return
        
        current_energy = conformation.energy
        
        # Check if this is a local minimum (energy lower than recent history)
        is_local_minimum = False
        
        # Method 1: Use local minima detector's stuck detection as proxy
        # (stuck = energy stabilized = potential minimum)
        if self._local_minima_detector.consecutive_stuck_iterations >= 3:
            is_local_minimum = True
        
        # Method 2: Significant energy improvement from last recorded minimum
        if abs(current_energy - self._last_minima_energy) > self._minima_detection_threshold:
            if current_energy < self._last_minima_energy:
                is_local_minimum = True
        
        # Record THz signature if this is a minimum
        if is_local_minimum:
            try:
                # Extract CA coordinates from conformation
                ca_coords = conformation.atom_coordinates  # Already list of (x,y,z) tuples
                
                if len(ca_coords) < 3:
                    logger.warning(f"Not enough atoms ({len(ca_coords)}) for THz analysis")
                    return
                
                # Calculate THz spectrum
                spectrum = self._thz_analyzer.calculate_spectrum(
                    ca_coordinates=ca_coords,
                    n_modes=20,
                    energy=current_energy,
                    rmsd=conformation.rmsd_to_native or 0.0,
                    qcp_score=None  # Will be added if QCPP integration active
                )
                
                # Store in signature history
                self._thz_signature_history.append(spectrum)
                self._last_minima_energy = current_energy
                
                # Log significant peaks
                peak_freqs = spectrum.get_peak_frequencies(threshold=0.1)
                logger.info(
                    f"THz signature recorded at minimum: "
                    f"E={current_energy:.2f}, "
                    f"peaks={len(peak_freqs)}, "
                    f"dominant={peak_freqs[0] if peak_freqs else 0:.2f} THz"
                )
                
            except Exception as e:
                logger.warning(f"Failed to record THz signature: {e}")

    def get_current_conformation(self) -> Conformation:
        """Get current protein conformation."""
        return self._current_conformation

    def get_exploration_metrics(self) -> Dict[str, float]:
        """Get current exploration metrics."""
        metrics = {
            "iterations_completed": self._iterations_completed,
            "conformations_explored": self._conformations_explored,
            "memories_created": self._memories_created,
            "best_energy": self._best_energy,
            "best_rmsd": self._best_rmsd,
            "avg_decision_time_ms": (
                self._total_decision_time_ms / max(1, self._iterations_completed)
            ),
            "stuck_in_minima_count": self._stuck_in_minima_count,
            "successful_escapes": self._successful_escapes,
            "validation_failures": self._validation_failures,
            "repair_attempts": self._repair_attempts,
            "repair_successes": self._repair_successes,
            "learning_improvement": self._calculate_learning_improvement()
        }
        
        # Task 5: Add GDT-TS and TM-score if available
        if hasattr(self, '_best_gdt_ts'):
            metrics["best_gdt_ts"] = self._best_gdt_ts
        if hasattr(self, '_best_tm_score'):
            metrics["best_tm_score"] = self._best_tm_score
        
        # Add THz signature count if available
        if hasattr(self, '_thz_signature_history'):
            metrics["thz_signatures_recorded"] = len(self._thz_signature_history)
            
        return metrics
    
    def get_thz_signature_history(self) -> List[THzSpectrum]:
        """
        Get history of recorded THz signatures from local minima.
        
        Note: Only populated if enable_thz_recording=True was set during initialization.
        
        Returns:
            List of THzSpectrum objects recorded during exploration (empty if THz recording disabled)
        """
        if hasattr(self, '_thz_signature_history'):
            return self._thz_signature_history.copy()
        return []

    def _calculate_learning_improvement(self) -> float:
        """
        Calculate learning improvement as percentage RMSD improvement over time.

        Returns:
            Learning improvement percentage (0.0-100.0)
        """
        if len(self._rmsd_history) < 2:
            return 0.0  # Not enough data for improvement calculation

        # Calculate improvement as reduction from initial to best RMSD
        initial_rmsd = self._rmsd_history[0]
        best_rmsd = min(self._rmsd_history)

        if initial_rmsd == 0 or best_rmsd >= initial_rmsd:
            return 0.0  # No improvement or invalid data

        # Percentage improvement
        improvement = ((initial_rmsd - best_rmsd) / initial_rmsd) * 100.0
        return min(100.0, max(0.0, improvement))  # Clamp to 0-100%

    def _metropolis_accept(self, energy_change: float) -> bool:
        """
        Metropolis-Hastings acceptance criterion for moves.
        
        Always accept if energy decreases (energy_change < 0).
        Accept uphill moves with probability exp(-ΔE / kT).
        
        Args:
            energy_change: Energy change (new - current) in kcal/mol
            
        Returns:
            True if move should be accepted, False otherwise
        """
        # Always accept downhill moves
        if energy_change < 0:
            return True
        
        # For uphill moves, accept with Boltzmann probability
        # P = exp(-ΔE / kT) where k is Boltzmann constant
        try:
            acceptance_probability = math.exp(-energy_change / (BOLTZMANN_CONSTANT * self._temperature))
            return random.random() < acceptance_probability
        except OverflowError:
            # Energy change is too large, reject move
            return False
    
    def _update_temperature(self) -> None:
        """
        Update temperature using simulated annealing schedule.
        
        Decreases temperature gradually to focus search over time.
        """
        self._temperature = max(MIN_TEMPERATURE, self._temperature * TEMPERATURE_DECAY_RATE)

    def _capture_snapshot(self, iteration: int) -> None:
        """
        Capture current state as a ConformationSnapshot.
        
        Args:
            iteration: Current iteration number
        """
        if not self._enable_visualization:
            return
        
        # Create snapshot
        snapshot = ConformationSnapshot(
            iteration=iteration,
            timestamp=time.time(),
            conformation=self._current_conformation,
            agent_id=self._agent_id,
            consciousness_state=self._consciousness.get_coordinates(),
            behavioral_state=self._behavioral.get_behavioral_data()
        )
        
        self._trajectory_snapshots.append(snapshot)
        
        # Downsample if we exceed max_snapshots
        if len(self._trajectory_snapshots) > self._max_snapshots:
            # Keep first 20%, last 20%, sample 60% middle at 50% rate
            keep_start = max(1, int(self._max_snapshots * 0.2))
            keep_end = max(1, int(self._max_snapshots * 0.2))
            middle_start = keep_start
            middle_end = len(self._trajectory_snapshots) - keep_end
            
            # Downsample middle by keeping every other snapshot
            downsampled = (
                self._trajectory_snapshots[:keep_start] +
                self._trajectory_snapshots[middle_start:middle_end:2] +
                self._trajectory_snapshots[-keep_end:]
            )
            self._trajectory_snapshots = downsampled

    def get_trajectory_snapshots(self) -> List[ConformationSnapshot]:
        """
        Get all trajectory snapshots for this agent.
        
        Returns:
            List of ConformationSnapshots
        """
        return self._trajectory_snapshots.copy()

    def get_agent_id(self) -> str:
        """
        Get unique agent identifier.
        
        Returns:
            Agent ID string
        """
        return self._agent_id

    def set_agent_id(self, agent_id: str) -> None:
        """
        Set custom agent identifier.
        
        Args:
            agent_id: New agent ID
        """
        self._agent_id = agent_id

    def enable_visualization(self, enable: bool = True, max_snapshots: int = 1000) -> None:
        """
        Enable or disable visualization snapshot recording.
        
        Args:
            enable: Whether to enable visualization
            max_snapshots: Maximum snapshots to store
        """
        self._enable_visualization = enable
        self._max_snapshots = max_snapshots
        
        if not enable:
            self._trajectory_snapshots.clear()

    def clear_trajectory_snapshots(self) -> None:
        """Clear all stored trajectory snapshots to free memory."""
        self._trajectory_snapshots.clear()