"""
Mapless move system implementation for UBF protein system.

This module implements the mapless conformational exploration system,
including move generation, evaluation, and physics integration.
"""

import random
import math
from typing import List, Dict, Optional

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try package-relative imports first
    from .interfaces import IMoveGenerator, IMoveEvaluator
    from .models import Conformation, ConformationalMove, MoveType
    from .physics_integration import QAAPCalculator, ResonanceCouplingCalculator, WaterShieldingCalculator
except ImportError:
    # Fall back to absolute imports from ubf_protein package
    from ubf_protein.interfaces import IMoveGenerator, IMoveEvaluator
    from ubf_protein.models import Conformation, ConformationalMove, MoveType
    from ubf_protein.physics_integration import QAAPCalculator, ResonanceCouplingCalculator, WaterShieldingCalculator


class MaplessMoveGenerator(IMoveGenerator):
    """
    Mapless move generator that creates conformational moves without spatial pathfinding.

    Uses capability-based filtering to generate feasible moves from current conformation.
    """

    def __init__(self):
        """Initialize move generator."""
        self.move_types = [
            MoveType.BACKBONE_ROTATION,
            MoveType.SIDECHAIN_ADJUST,
            MoveType.HELIX_FORMATION,
            MoveType.SHEET_FORMATION,
            MoveType.TURN_FORMATION,
            MoveType.HYDROPHOBIC_COLLAPSE,
            MoveType.ENERGY_MINIMIZATION
        ]

    def generate_moves(self, current_conformation: Conformation) -> List[ConformationalMove]:
        """
        Generate all feasible moves from current state (mapless - no pathfinding).

        Args:
            current_conformation: Current protein conformation

        Returns:
            List of feasible conformational moves
        """
        moves = []
        capabilities = current_conformation.get_capabilities()

        # Generate moves based on capabilities
        for move_type in self.move_types:
            if self._is_move_feasible(move_type, capabilities, current_conformation):
                move = self._create_move(move_type, current_conformation)
                if move:
                    moves.append(move)

        return moves

    def _is_move_feasible(self, move_type: MoveType,
                         capabilities: Dict[str, bool],
                         conformation: Conformation) -> bool:
        """
        Check if a move type is feasible given current capabilities.

        Args:
            move_type: Type of move to check
            capabilities: Current conformational capabilities
            conformation: Current conformation

        Returns:
            True if move is feasible
        """
        # Basic capability checks
        if move_type == MoveType.HELIX_FORMATION:
            return capabilities.get('can_form_helix', False)
        elif move_type == MoveType.SHEET_FORMATION:
            return capabilities.get('can_form_sheet', False)
        elif move_type == MoveType.HYDROPHOBIC_COLLAPSE:
            return capabilities.get('can_hydrophobic_collapse', False)
        elif move_type == MoveType.LARGE_CONFORMATIONAL_JUMP:
            return capabilities.get('can_large_rotation', False)

        # Default: most moves are feasible with some restrictions
        return len(conformation.sequence) > 3  # Need minimum size

    def _create_move(self, move_type: MoveType,
                    conformation: Conformation) -> Optional[ConformationalMove]:
        """
        Create a specific move instance.

        Args:
            move_type: Type of move to create
            conformation: Current conformation

        Returns:
            New ConformationalMove instance or None if invalid
        """
        n_residues = len(conformation.sequence)

        # Select random target residues based on move type
        if move_type in [MoveType.HELIX_FORMATION, MoveType.SHEET_FORMATION]:
            # Need at least 4-6 consecutive residues
            if n_residues < 6:
                return None
            start_idx = random.randint(0, n_residues - 6)
            target_residues = list(range(start_idx, start_idx + 6))
        elif move_type == MoveType.TURN_FORMATION:
            # Need 3-4 residues
            if n_residues < 4:
                return None
            start_idx = random.randint(0, n_residues - 4)
            target_residues = list(range(start_idx, start_idx + 4))
        else:
            # Random subset of residues
            n_targets = min(5, max(1, n_residues // 4))
            target_residues = random.sample(range(n_residues), n_targets)

        # Estimate energy change (simplified)
        energy_change = self._estimate_energy_change(move_type, target_residues, conformation)

        # Estimate RMSD change (simplified)
        rmsd_change = self._estimate_rmsd_change(move_type, len(target_residues))

        # Determine required capabilities
        required_capabilities = self._get_required_capabilities(move_type)

        # Calculate feasibility (simplified)
        structural_feasibility = self._calculate_structural_feasibility(move_type, conformation)

        # Calculate energy barrier
        energy_barrier = self._calculate_energy_barrier(move_type, len(target_residues))

        move_id = f"{move_type.value}_{random.randint(1000, 9999)}_{len(target_residues)}"

        return ConformationalMove(
            move_id=move_id,
            move_type=move_type,
            target_residues=target_residues,
            estimated_energy_change=energy_change,
            estimated_rmsd_change=rmsd_change,
            required_capabilities=required_capabilities,
            energy_barrier=energy_barrier,
            structural_feasibility=structural_feasibility
        )

    def _estimate_energy_change(self, move_type: MoveType,
                               target_residues: List[int],
                               conformation: Conformation) -> float:
        """Estimate energy change for a move (simplified)."""
        base_change = 0.0

        if move_type == MoveType.HELIX_FORMATION:
            base_change = -15.0  # Helices are stabilizing
        elif move_type == MoveType.SHEET_FORMATION:
            base_change = -10.0  # Sheets are stabilizing
        elif move_type == MoveType.HYDROPHOBIC_COLLAPSE:
            base_change = -20.0  # Hydrophobic collapse very stabilizing
        elif move_type == MoveType.ENERGY_MINIMIZATION:
            base_change = -5.0  # Small improvement
        else:
            base_change = random.uniform(-5.0, 5.0)  # Random small change

        # Scale by number of residues affected
        scale_factor = len(target_residues) / 10.0
        return base_change * scale_factor

    def _estimate_rmsd_change(self, move_type: MoveType, n_residues: int) -> float:
        """Estimate RMSD change for a move (simplified)."""
        if move_type in [MoveType.LARGE_CONFORMATIONAL_JUMP, MoveType.HELIX_FORMATION]:
            return random.uniform(2.0, 5.0)  # Large changes
        elif move_type in [MoveType.SHEET_FORMATION, MoveType.HYDROPHOBIC_COLLAPSE]:
            return random.uniform(1.0, 3.0)  # Medium changes
        else:
            return random.uniform(0.1, 1.0)  # Small changes

    def _get_required_capabilities(self, move_type: MoveType) -> Dict[str, bool]:
        """Get required capabilities for a move type."""
        if move_type == MoveType.HELIX_FORMATION:
            return {'can_form_helix': True}
        elif move_type == MoveType.SHEET_FORMATION:
            return {'can_form_sheet': True}
        elif move_type == MoveType.HYDROPHOBIC_COLLAPSE:
            return {'can_hydrophobic_collapse': True}
        elif move_type == MoveType.LARGE_CONFORMATIONAL_JUMP:
            return {'can_large_rotation': True}
        else:
            return {}  # No special requirements

    def _calculate_structural_feasibility(self, move_type: MoveType,
                                        conformation: Conformation) -> float:
        """Calculate structural feasibility (0.0-1.0)."""
        # Simplified feasibility calculation
        base_feasibility = 0.8  # Most moves are reasonably feasible

        if move_type == MoveType.HELIX_FORMATION:
            # Check if there are coil regions available
            coil_count = conformation.secondary_structure.count('C')
            base_feasibility = min(1.0, coil_count / 10.0)
        elif move_type == MoveType.SHEET_FORMATION:
            coil_count = conformation.secondary_structure.count('C')
            base_feasibility = min(1.0, coil_count / 8.0)

        return max(0.1, base_feasibility)  # Minimum feasibility

    def _calculate_energy_barrier(self, move_type: MoveType, n_residues: int) -> float:
        """Calculate energy barrier for a move."""
        base_barrier = 5.0  # Base barrier in kJ/mol

        if move_type == MoveType.LARGE_CONFORMATIONAL_JUMP:
            base_barrier = 25.0  # High barrier for large jumps
        elif move_type in [MoveType.HELIX_FORMATION, MoveType.SHEET_FORMATION]:
            base_barrier = 15.0  # Moderate barrier for structure formation

        # Scale by size
        return base_barrier * (n_residues / 5.0)


class CapabilityBasedMoveEvaluator(IMoveEvaluator):
    """
    Capability-based move evaluator using 5 composite factors.

    Implements the simplified 5-factor evaluation approach:
    1. Physical Feasibility
    2. Quantum Alignment (QAAP + resonance + water shielding)
    3. Behavioral Preference
    4. Historical Success
    5. Goal Alignment
    """

    def __init__(self):
        """Initialize move evaluator with physics calculators."""
        self.qaap_calculator = QAAPCalculator()
        self.resonance_calculator = ResonanceCouplingCalculator()
        self.water_shielding_calculator = WaterShieldingCalculator()

    def evaluate_move(self,
                     move: ConformationalMove,
                     behavioral_state,  # IBehavioralState
                     memory_influence: float,
                     physics_factors: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weight for move using 5 composite factors.

        Args:
            move: The move to evaluate
            behavioral_state: Current behavioral state
            memory_influence: Memory-based influence multiplier
            physics_factors: Optional pre-calculated physics factors

        Returns:
            Move weight (higher = more desirable)
        """
        # Type hints for PyPy JIT optimization
        physical_feasibility: float
        quantum_alignment: float
        behavioral_preference: float
        historical_success: float
        goal_alignment: float
        total_weight: float
        
        # Factor 1: Physical Feasibility (0.2 weight)
        physical_feasibility = self._calculate_physical_feasibility(move)

        # Factor 2: Quantum Alignment (0.25 weight)
        quantum_alignment = self._calculate_quantum_alignment(move, physics_factors)

        # Factor 3: Behavioral Preference (0.2 weight)
        behavioral_preference = self._calculate_behavioral_preference(move, behavioral_state)

        # Factor 4: Historical Success (0.15 weight)
        historical_success = memory_influence

        # Factor 5: Goal Alignment (0.2 weight)
        goal_alignment = self._calculate_goal_alignment(move)

        # Combine factors with weights - explicit calculation for JIT optimization
        total_weight = (
            0.2 * physical_feasibility +
            0.25 * quantum_alignment +
            0.2 * behavioral_preference +
            0.15 * historical_success +
            0.2 * goal_alignment
        )

        return total_weight

    def _calculate_physical_feasibility(self, move: ConformationalMove) -> float:
        """
        Factor 1: Physical Feasibility
        Combines structural feasibility and energy barrier.
        """
        # Type hints for PyPy JIT optimization
        structural: float = move.structural_feasibility
        barrier_feasibility: float
        result: float

        # Convert energy barrier to feasibility (lower barrier = higher feasibility)
        # Barriers > 50 kJ/mol are very unlikely
        barrier_feasibility = max(0.0, 1.0 - (move.energy_barrier / 50.0))

        # Combine (weighted average) - explicit calculation for JIT
        result = 0.7 * structural + 0.3 * barrier_feasibility
        
        return result

    def _calculate_quantum_alignment(self, move: ConformationalMove,
                                   physics_factors: Optional[Dict[str, float]]) -> float:
        """
        Factor 2: Quantum Alignment
        Combines QAAP, resonance, and water shielding.
        """
        if physics_factors:
            # Use provided factors if available
            qaap = physics_factors.get('qaap', 0.5)
            resonance = physics_factors.get('resonance', 0.5)
            water_shielding = physics_factors.get('water_shielding', 0.5)
        else:
            # Placeholder values - in real implementation would calculate from conformation
            qaap = 0.5
            resonance = 0.5
            water_shielding = 0.5

        # Combine with specified weights from requirements
        # QAAP: 0.7-1.3 range contributes 40%
        qaap_weight = 0.4
        qaap_contribution = qaap_weight * (0.7 + 0.6 * qaap)  # Maps 0-1 to 0.7-1.3

        # Resonance: 0.9-1.2 range contributes 35%
        resonance_weight = 0.35
        resonance_contribution = resonance_weight * (0.9 + 0.3 * resonance)  # Maps 0-1 to 0.9-1.2

        # Water shielding: 0.95-1.05 range contributes 25%
        shielding_weight = 0.25
        shielding_contribution = shielding_weight * (0.95 + 0.1 * water_shielding)  # Maps 0-1 to 0.95-1.05

        return qaap_contribution + resonance_contribution + shielding_contribution

    def _calculate_behavioral_preference(self, move: ConformationalMove,
                                       behavioral_state) -> float:
        """
        Factor 3: Behavioral Preference
        Combines all 5 behavioral dimensions.
        """
        # Get behavioral dimensions
        exploration_energy = behavioral_state.get_exploration_energy()
        structural_focus = behavioral_state.get_structural_focus()
        hydrophobic_drive = behavioral_state.get_hydrophobic_drive()
        risk_tolerance = behavioral_state.get_risk_tolerance()
        native_state_ambition = behavioral_state.get_native_state_ambition()

        # Calculate preference based on move type
        if move.move_type.value == 'hydrophobic_collapse':
            preference = (0.3 * hydrophobic_drive +
                         0.3 * exploration_energy +
                         0.2 * risk_tolerance +
                         0.1 * structural_focus +
                         0.1 * native_state_ambition)
        elif move.move_type.value in ['helix_formation', 'sheet_formation']:
            preference = (0.4 * structural_focus +
                         0.3 * native_state_ambition +
                         0.2 * exploration_energy +
                         0.1 * hydrophobic_drive)
        elif move.move_type.value == 'large_jump':
            preference = (0.4 * risk_tolerance +
                         0.3 * exploration_energy +
                         0.2 * hydrophobic_drive +
                         0.1 * structural_focus)
        else:  # Default moves
            preference = (0.25 * exploration_energy +
                         0.25 * structural_focus +
                         0.2 * risk_tolerance +
                         0.15 * hydrophobic_drive +
                         0.15 * native_state_ambition)

        return preference

    def _calculate_goal_alignment(self, move: ConformationalMove) -> float:
        """
        Factor 5: Goal Alignment
        Based on energy decrease and RMSD improvement potential.
        """
        # Energy alignment (negative energy change = good, positive = bad)
        # Convert energy change to alignment score (higher = better)
        # energy_change of -50 or less = 1.0 (perfect)
        # energy_change of 0 = 0.5 (neutral)
        # energy_change of +50 or more = 0.0 (terrible)
        energy_alignment = max(0.0, min(1.0, 1.0 - (move.estimated_energy_change / 50.0)))

        # RMSD alignment (some RMSD change is good, but not too much)
        optimal_rmsd = 1.0  # Ideal RMSD change
        rmsd_alignment = 1.0 - abs(move.estimated_rmsd_change - optimal_rmsd) / 2.0
        rmsd_alignment = max(0.0, rmsd_alignment)

        # Combine
        return 0.6 * energy_alignment + 0.4 * rmsd_alignment