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
from typing import Optional, Dict, List, Any

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

    def __init__(self,
                 protein_sequence: str,
                 initial_frequency: float = 9.0,
                 initial_coherence: float = 0.6,
                 initial_conformation: Optional[Conformation] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 enable_visualization: bool = False,
                 max_snapshots: int = 1000,
                 native_structure: Optional[Conformation] = None,
                 qcpp_integration: Optional[Any] = None):
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
        """
        # Create adaptive config if not provided
        if adaptive_config is None:
            adaptive_config = self._create_default_adaptive_config(protein_sequence)
        
        # Store QCPP integration reference
        self._qcpp_integration = qcpp_integration

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

                        # Evaluate move
                        weight = move_evaluator.evaluate_move(
                            move,
                            self._behavioral,
                            memory_influence,
                            physics_factors
                        )
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
                    qcpp_metrics = None  # Store for memory creation
                    if self._qcpp_integration is not None and success:
                        try:
                            # Analyze conformation with QCPP
                            qcpp_metrics = self._qcpp_integration.analyze_conformation(new_conformation)
                            
                            # Update physics-grounded consciousness from QCPP metrics
                            if hasattr(self._consciousness, 'update_from_qcpp_metrics'):
                                self._consciousness.update_from_qcpp_metrics(qcpp_metrics)
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
                qcpp_metrics=qcpp_metrics_for_memory
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
                qcpp_metrics=qcpp_metrics_for_memory
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
        Generate initial extended conformation with randomization.

        Each agent gets a slightly different starting conformation
        to enable diverse exploration.
        """
        # Create placeholder 3D coordinates (extended chain with noise)
        num_residues = len(self._protein_sequence)
        atom_coordinates = []
        for i in range(num_residues):
            # Simple extended chain with random perturbations
            x = i * 3.8 + random.uniform(-0.5, 0.5)  # ±0.5 Å noise
            y = random.uniform(-0.5, 0.5)  # ±0.5 Å noise
            z = random.uniform(-0.5, 0.5)  # ±0.5 Å noise
            atom_coordinates.append((x, y, z))

        # Placeholder secondary structure (all coil)
        secondary_structure = ['C'] * num_residues

        # Randomized angles (±20° from alpha helix)
        phi_angles = [-60.0 + random.uniform(-20, 20) for _ in range(num_residues)]
        psi_angles = [-40.0 + random.uniform(-20, 20) for _ in range(num_residues)]
        
        # Randomize initial energy slightly (reduces likelihood of all agents finding same minimum)
        initial_energy = random.uniform(950.0, 1050.0)

        return Conformation(
            conformation_id="initial",
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

    def _execute_move(self, move) -> Conformation:
        """
        Execute a conformational move and return new conformation.

        This is a simplified simulation. In the full system,
        this would perform actual structural calculations.

        Args:
            move: The move to execute

        Returns:
            New conformation after move execution
        """
        # Calculate actual energy change (may differ from estimate)
        actual_energy_change = move.estimated_energy_change * (0.8 + random.random() * 0.4)  # ±20% variation

        # Apply structural changes to coordinates (more substantial moves)
        new_coords = []
        for i, (x, y, z) in enumerate(self._current_conformation.atom_coordinates):
            if i in move.target_residues:
                # Apply moderate random perturbations to move residues
                move_scale = 0.8 if self._stuck_in_minima_count > 10 else 0.5  # Moderate moves
                dx = random.uniform(-1.0, 1.0) * move_scale
                dy = random.uniform(-1.0, 1.0) * move_scale
                dz = random.uniform(-1.0, 1.0) * move_scale
                new_coords.append((x + dx, y + dy, z + dz))
            else:
                # Keep non-target residues with small perturbations
                new_coords.append((
                    x + random.uniform(-0.1, 0.1),
                    y + random.uniform(-0.1, 0.1),
                    z + random.uniform(-0.1, 0.1)
                ))
        
        # Update phi/psi angles for target residues (moderate changes)
        new_phi = list(self._current_conformation.phi_angles)
        new_psi = list(self._current_conformation.psi_angles)
        for i in move.target_residues:
            if i < len(new_phi):
                new_phi[i] += random.uniform(-15, 15)  # ±15° change (reduced from ±30°)
                new_psi[i] += random.uniform(-15, 15)

        # Create new conformation with preliminary energy
        new_conformation = Conformation(
            conformation_id=f"conf_{self._iterations_completed + 1}_{move.move_id}",
            sequence=self._protein_sequence,
            atom_coordinates=new_coords,  # Updated coordinates
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
        # Factor 1: Energy change impact (0.5 weight)
        # Large negative changes are highly significant
        energy_significance = min(1.0, max(0.0, -energy_change / 50.0))  # -50 kJ/mol = max significance

        # Factor 2: Structural novelty (0.3 weight)
        # For now, assume some novelty if successful
        structural_novelty = 0.5 if success else 0.1

        # Factor 3: RMSD improvement (0.2 weight)
        # RMSD decrease is good
        rmsd_significance = min(1.0, max(0.0, -rmsd_change / 2.0))  # -2 Å = max significance

        # Combine factors
        significance = (0.5 * energy_significance +
                       0.3 * structural_novelty +
                       0.2 * rmsd_significance)

        return min(1.0, significance)

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
            
        return metrics

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