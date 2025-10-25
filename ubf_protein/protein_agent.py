"""
Protein agent implementation for UBF protein system.

This module implements the autonomous protein folding agent that coordinates
consciousness, behavioral state, and memory systems to perform mapless
conformational exploration.
"""

import time
import random
import logging
from typing import Optional, Dict, List

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
    ENERGY_VALIDATION_THRESHOLD
)
from . import config as config_module

# Set up logging
logger = logging.getLogger(__name__)


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
                 max_snapshots: int = 1000):
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
        """
        # Create adaptive config if not provided
        if adaptive_config is None:
            adaptive_config = self._create_default_adaptive_config(protein_sequence)

        # Initialize consciousness system
        self._consciousness = ConsciousnessState(initial_frequency, initial_coherence)

        # Initialize behavioral state (derived from consciousness)
        self._behavioral = BehavioralState(self._consciousness.get_coordinates())

        # Initialize memory system
        self._memory = MemorySystem()

        # Initialize local minima detector
        self._local_minima_detector = LocalMinimaDetector(adaptive_config)

        # Initialize structural validator
        self._validator = StructuralValidation()
        
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
        
        # Learning improvement tracking
        self._rmsd_history = [self._best_rmsd] if self._best_rmsd != float('inf') else []
        
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

                    # Determine success (energy decreased or RMSD improved)
                    success = energy_change < 0

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

            # Track escape attempt
            self._stuck_in_minima_count += 1

            # Check if escape was successful (energy improved)
            if outcome.new_conformation.energy < self._current_conformation.energy:
                self._successful_escapes += 1
                self._local_minima_detector.record_escape_success(self._iterations_completed)

                # Create high-significance memory for successful escape
                # Manually create memory with high significance
                escape_memory = self._memory.create_memory_from_outcome(
                    outcome,
                    self._consciousness.get_coordinates(),
                    self._behavioral.get_behavioral_data()
                )
                # Override significance for successful escape
                escape_memory.significance = 0.8
                self._memory.store_memory(escape_memory)
                self._memories_created += 1

        # Check if behavioral state needs regeneration
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
            memory = self._memory.create_memory_from_outcome(
                outcome,
                self._consciousness.get_coordinates(),
                self._behavioral.get_behavioral_data()
            )
            self._memory.store_memory(memory)
            if memory.significance >= 0.3:  # Same threshold as memory system
                self._memories_created += 1
        except Exception as e:
            logger.warning(f"Error creating/storing memory: {e}")
            # Continue execution - memory is non-critical

        # Update current conformation
        if outcome.new_conformation != self._current_conformation:
            self._current_conformation = outcome.new_conformation
            self._conformations_explored += 1

        # Update best metrics
        if outcome.new_conformation.energy < self._best_energy:
            self._best_energy = outcome.new_conformation.energy
        if (outcome.new_conformation.rmsd_to_native and
            outcome.new_conformation.rmsd_to_native < self._best_rmsd):
            self._best_rmsd = outcome.new_conformation.rmsd_to_native
            # Track RMSD improvement for learning calculation
            self._rmsd_history.append(self._best_rmsd)

        # Update metrics
        self._iterations_completed += 1
        decision_time_ms = (time.time() - start_time) * 1000
        self._total_decision_time_ms += decision_time_ms

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
        Generate initial extended conformation.

        This is a simplified placeholder. In a real implementation,
        this would generate a proper 3D structure.
        """
        # Create placeholder 3D coordinates (extended chain)
        num_residues = len(self._protein_sequence)
        atom_coordinates = []
        for i in range(num_residues):
            # Simple extended chain: each residue 3.8 Å apart along x-axis
            x = i * 3.8
            atom_coordinates.append((x, 0.0, 0.0))  # CA atoms only for simplicity

        # Placeholder secondary structure (all coil)
        secondary_structure = ['C'] * num_residues

        # Placeholder angles
        phi_angles = [-60.0] * num_residues  # Alpha helix phi
        psi_angles = [-40.0] * num_residues  # Alpha helix psi

        return Conformation(
            conformation_id="initial",
            sequence=self._protein_sequence,
            atom_coordinates=atom_coordinates,
            energy=1000.0,  # High initial energy
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

        # Create new conformation with preliminary energy
        new_conformation = Conformation(
            conformation_id=f"conf_{self._iterations_completed + 1}_{move.move_id}",
            sequence=self._protein_sequence,
            atom_coordinates=self._current_conformation.atom_coordinates,  # Unchanged for now
            energy=self._current_conformation.energy + actual_energy_change,
            rmsd_to_native=self._current_conformation.rmsd_to_native,
            secondary_structure=self._current_conformation.secondary_structure,
            phi_angles=self._current_conformation.phi_angles,
            psi_angles=self._current_conformation.psi_angles,
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
        return {
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