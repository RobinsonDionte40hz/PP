"""
Protein agent implementation for UBF protein system.

This module implements the autonomous protein folding agent that coordinates
consciousness, behavioral state, and memory systems to perform mapless
conformational exploration.
"""

import time
import random
from typing import Optional, Dict

from .interfaces import IProteinAgent
from .models import (
    Conformation, ConformationalOutcome, ConformationalMemory,
    ConsciousnessCoordinates, BehavioralStateData
)
from .consciousness import ConsciousnessState
from .behavioral_state import BehavioralState
from .memory_system import MemorySystem


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
                 initial_conformation: Optional[Conformation] = None):
        """
        Initialize protein agent with consciousness coordinates and protein sequence.

        Args:
            protein_sequence: Amino acid sequence of the protein
            initial_frequency: Initial consciousness frequency (3-15 Hz)
            initial_coherence: Initial consciousness coherence (0.2-1.0)
            initial_conformation: Starting conformation (generated if None)
        """
        # Initialize consciousness system
        self._consciousness = ConsciousnessState(initial_frequency, initial_coherence)

        # Initialize behavioral state (derived from consciousness)
        self._behavioral = BehavioralState(self._consciousness.get_coordinates())

        # Initialize memory system
        self._memory = MemorySystem()

        # Store protein sequence
        self._protein_sequence = protein_sequence

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
        Execute one exploration step using mapless design.

        Generates available moves, evaluates them using capability-based evaluation,
        selects the best move, executes it, and updates all systems.

        Returns:
            ConformationalOutcome from the exploration step
        """
        start_time = time.time()

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

            # Select best move (highest weight)
            best_move, best_weight = max(move_weights, key=lambda x: x[1])

            # Execute the move (simulate conformational change)
            new_conformation = self._execute_move(best_move)

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

        # Update consciousness based on outcome
        self._consciousness.update_from_outcome(outcome)

        # Check if behavioral state needs regeneration
        regenerated_behavioral = self._behavioral.regenerate_if_needed(
            self._consciousness.get_coordinates()
        )
        if regenerated_behavioral is not None:
            self._behavioral = regenerated_behavioral

        # Create and store memory if significant
        memory = self._memory.create_memory_from_outcome(
            outcome,
            self._consciousness.get_coordinates(),
            self._behavioral.get_behavioral_data()
        )
        self._memory.store_memory(memory)
        if memory.significance >= 0.3:  # Same threshold as memory system
            self._memories_created += 1

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

        # Update metrics
        self._iterations_completed += 1
        decision_time_ms = (time.time() - start_time) * 1000
        self._total_decision_time_ms += decision_time_ms

        return outcome

    def get_current_conformation(self) -> Conformation:
        """Get current protein conformation."""
        return self._current_conformation

    def get_exploration_metrics(self) -> dict:
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
            "successful_escapes": self._successful_escapes
        }

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

    def _simulate_energy_change(self) -> float:
        """
        Simulate energy change based on current behavioral state.

        This is a simplified simulation that will be replaced with
        actual physics calculations in later tasks.
        """
        # Get behavioral preferences
        exploration_energy = self._behavioral.get_exploration_energy()
        risk_tolerance = self._behavioral.get_risk_tolerance()
        hydrophobic_drive = self._behavioral.get_hydrophobic_drive()

        # Simple simulation: higher exploration energy and risk tolerance
        # tend to find better moves, but with more variance
        base_change = (exploration_energy - 0.5) * 100  # -50 to +50
        risk_factor = (risk_tolerance - 0.5) * 50       # -25 to +25
        hydrophobic_factor = (hydrophobic_drive - 0.5) * 30  # -15 to +15

        # Add some randomness
        import random
        noise = random.gauss(0, 20)

        total_change = base_change + risk_factor + hydrophobic_factor + noise

        # Memory influence (if available)
        memory_influence = self._memory.calculate_memory_influence("backbone_rotation")
        total_change *= memory_influence

        return total_change

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

        # Create new conformation
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
            structural_constraints=self._current_conformation.structural_constraints
        )

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