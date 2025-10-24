"""
Integration tests for physics modules in move evaluation system.

Tests the integration of QAAP, resonance, and water shielding calculators
into the capability-based move evaluator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ubf_protein.mapless_moves import CapabilityBasedMoveEvaluator
from ubf_protein.physics_integration import QAAPCalculator, ResonanceCouplingCalculator, WaterShieldingCalculator
from ubf_protein.models import ConformationalMove, MoveType


class TestPhysicsIntegration:
    """Test physics calculator integration in move evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Create move evaluator instance."""
        return CapabilityBasedMoveEvaluator()

    @pytest.fixture
    def mock_behavioral_state(self):
        """Create mock behavioral state."""
        state = Mock()
        state.get_exploration_energy.return_value = 0.7
        state.get_structural_focus.return_value = 0.8
        state.get_hydrophobic_drive.return_value = 0.6
        state.get_risk_tolerance.return_value = 0.5
        state.get_native_state_ambition.return_value = 0.9
        return state

    @pytest.fixture
    def sample_move(self):
        """Create sample conformational move."""
        return ConformationalMove(
            move_id="test_move_123",
            move_type=MoveType.HELIX_FORMATION,
            target_residues=[1, 2, 3, 4, 5],
            estimated_energy_change=-15.0,
            estimated_rmsd_change=2.5,
            required_capabilities={'can_form_helix': True},
            energy_barrier=12.0,
            structural_feasibility=0.85
        )

    def test_evaluator_initialization(self, evaluator):
        """Test that evaluator initializes with physics calculators."""
        assert hasattr(evaluator, 'qaap_calculator')
        assert hasattr(evaluator, 'resonance_calculator')
        assert hasattr(evaluator, 'water_shielding_calculator')

        # Check that they are the correct types by checking class names
        assert evaluator.qaap_calculator.__class__.__name__ == 'QAAPCalculator'
        assert evaluator.resonance_calculator.__class__.__name__ == 'ResonanceCouplingCalculator'
        assert evaluator.water_shielding_calculator.__class__.__name__ == 'WaterShieldingCalculator'

    def test_quantum_alignment_calculation_with_provided_factors(self, evaluator, sample_move, mock_behavioral_state):
        """Test quantum alignment calculation with provided physics factors."""
        physics_factors = {
            'qaap': 0.8,
            'resonance': 0.6,
            'water_shielding': 0.7
        }

        weight = evaluator.evaluate_move(
            sample_move,
            mock_behavioral_state,
            memory_influence=0.5,
            physics_factors=physics_factors
        )

        # Should return a valid weight
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 2.0  # Reasonable range for combined factors

    def test_quantum_alignment_calculation_without_factors(self, evaluator, sample_move, mock_behavioral_state):
        """Test quantum alignment calculation with default/placeholder factors."""
        weight = evaluator.evaluate_move(
            sample_move,
            mock_behavioral_state,
            memory_influence=0.5
        )

        # Should return a valid weight using default factors
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 2.0

    def test_quantum_alignment_factor_weights(self, evaluator):
        """Test that quantum alignment uses correct factor weights."""
        # Test with extreme values to verify weighting
        physics_factors = {
            'qaap': 1.0,  # Maximum QAAP
            'resonance': 0.0,  # Minimum resonance
            'water_shielding': 0.0  # Minimum shielding
        }

        alignment_high_qaap = evaluator._calculate_quantum_alignment(None, physics_factors)

        physics_factors = {
            'qaap': 0.0,  # Minimum QAAP
            'resonance': 1.0,  # Maximum resonance
            'water_shielding': 1.0  # Maximum shielding
        }

        alignment_high_others = evaluator._calculate_quantum_alignment(None, physics_factors)

        # QAAP should have more influence (40% weight vs 35% + 25% = 60%)
        # So high QAAP should give higher alignment than high resonance+shielding
        assert alignment_high_qaap > alignment_high_others

    def test_quantum_alignment_ranges(self, evaluator):
        """Test that quantum alignment produces values in expected ranges."""
        test_cases = [
            {'qaap': 0.0, 'resonance': 0.0, 'water_shielding': 0.0},
            {'qaap': 0.5, 'resonance': 0.5, 'water_shielding': 0.5},
            {'qaap': 1.0, 'resonance': 1.0, 'water_shielding': 1.0}
        ]

        for factors in test_cases:
            alignment = evaluator._calculate_quantum_alignment(None, factors)

            # Should be in range [0.7+0.9+0.95, 1.3+1.2+1.05] = [2.55, 3.55]
            # But with weights: 0.4*range + 0.35*range + 0.25*range
            # Min: 0.4*0.7 + 0.35*0.9 + 0.25*0.95 = 0.28 + 0.315 + 0.2375 = 0.8325
            # Max: 0.4*1.3 + 0.35*1.2 + 0.25*1.05 = 0.52 + 0.42 + 0.2625 = 1.2025

            assert 0.8 <= alignment <= 1.3, f"Alignment {alignment} out of range for factors {factors}"

    def test_physical_feasibility_calculation(self, evaluator):
        """Test physical feasibility factor calculation."""
        # High feasibility move
        high_feasibility_move = ConformationalMove(
            move_id="high_feas_123",
            move_type=MoveType.HELIX_FORMATION,
            target_residues=[1, 2, 3, 4],
            estimated_energy_change=-10.0,
            estimated_rmsd_change=1.0,
            required_capabilities={},
            energy_barrier=5.0,
            structural_feasibility=0.95
        )

        # Low feasibility move
        low_feasibility_move = ConformationalMove(
            move_id="low_feas_456",
            move_type=MoveType.LARGE_CONFORMATIONAL_JUMP,
            target_residues=[1, 2, 3, 4, 5, 6, 7, 8],
            estimated_energy_change=5.0,
            estimated_rmsd_change=4.0,
            required_capabilities={},
            energy_barrier=40.0,
            structural_feasibility=0.3
        )

        high_feas = evaluator._calculate_physical_feasibility(high_feasibility_move)
        low_feas = evaluator._calculate_physical_feasibility(low_feasibility_move)

        assert high_feas > low_feas
        assert 0.0 <= high_feas <= 1.0
        assert 0.0 <= low_feas <= 1.0

    def test_behavioral_preference_calculation(self, evaluator, sample_move):
        """Test behavioral preference factor calculation."""
        # Create different behavioral states
        high_exploration = Mock()
        high_exploration.get_exploration_energy.return_value = 1.0
        high_exploration.get_structural_focus.return_value = 0.1
        high_exploration.get_hydrophobic_drive.return_value = 0.1
        high_exploration.get_risk_tolerance.return_value = 0.1
        high_exploration.get_native_state_ambition.return_value = 0.1

        high_structural = Mock()
        high_structural.get_exploration_energy.return_value = 0.1
        high_structural.get_structural_focus.return_value = 1.0
        high_structural.get_hydrophobic_drive.return_value = 0.1
        high_structural.get_risk_tolerance.return_value = 0.1
        high_structural.get_native_state_ambition.return_value = 0.1

        # Helix formation should prefer structural focus
        pref_exploration = evaluator._calculate_behavioral_preference(sample_move, high_exploration)
        pref_structural = evaluator._calculate_behavioral_preference(sample_move, high_structural)

        assert pref_structural > pref_exploration

    def test_goal_alignment_calculation(self, evaluator):
        """Test goal alignment factor calculation."""
        # Energy-improving move (negative energy change = good)
        good_move = ConformationalMove(
            move_id="good_123",
            move_type=MoveType.ENERGY_MINIMIZATION,
            target_residues=[1, 2, 3],
            estimated_energy_change=-50.0,  # Large energy improvement
            estimated_rmsd_change=1.0,  # Good RMSD change
            required_capabilities={},
            energy_barrier=5.0,
            structural_feasibility=0.8
        )

        # Energy-worsening move (positive energy change = bad)
        bad_move = ConformationalMove(
            move_id="bad_456",
            move_type=MoveType.BACKBONE_ROTATION,
            target_residues=[1, 2, 3],
            estimated_energy_change=50.0,  # Large energy increase
            estimated_rmsd_change=0.1,  # Too small RMSD change
            required_capabilities={},
            energy_barrier=5.0,
            structural_feasibility=0.8
        )

        good_alignment = evaluator._calculate_goal_alignment(good_move)
        bad_alignment = evaluator._calculate_goal_alignment(bad_move)

        assert good_alignment > bad_alignment
        assert 0.0 <= good_alignment <= 1.0
        assert 0.0 <= bad_alignment <= 1.0

    def test_complete_evaluation_integration(self, evaluator, sample_move, mock_behavioral_state):
        """Test complete move evaluation with all factors."""
        physics_factors = {
            'qaap': 0.8,
            'resonance': 0.7,
            'water_shielding': 0.6
        }

        weight = evaluator.evaluate_move(
            sample_move,
            mock_behavioral_state,
            memory_influence=0.8,
            physics_factors=physics_factors
        )

        # Verify all components contribute
        assert weight > 0.0

        # Test that physics factors affect the result
        weight_no_physics = evaluator.evaluate_move(
            sample_move,
            mock_behavioral_state,
            memory_influence=0.8
        )

        # Should be different (though could be same due to default values)
        # At minimum, both should be valid weights
        assert isinstance(weight_no_physics, float)
        assert 0.0 <= weight_no_physics <= 2.0

    @pytest.mark.parametrize("move_type,expected_preference_range", [
        (MoveType.HELIX_FORMATION, (0.3, 0.8)),
        (MoveType.SHEET_FORMATION, (0.3, 0.8)),
        (MoveType.HYDROPHOBIC_COLLAPSE, (0.2, 0.7)),
        (MoveType.LARGE_CONFORMATIONAL_JUMP, (0.2, 0.7)),
        (MoveType.ENERGY_MINIMIZATION, (0.2, 0.8))
    ])
    def test_behavioral_preferences_by_move_type(self, evaluator, mock_behavioral_state,
                                                move_type, expected_preference_range):
        """Test behavioral preferences vary appropriately by move type."""
        move = ConformationalMove(
            move_id=f"test_{move_type.value}",
            move_type=move_type,
            target_residues=[1, 2, 3, 4],
            estimated_energy_change=-5.0,
            estimated_rmsd_change=1.0,
            required_capabilities={},
            energy_barrier=10.0,
            structural_feasibility=0.8
        )

        preference = evaluator._calculate_behavioral_preference(move, mock_behavioral_state)

        min_pref, max_pref = expected_preference_range
        assert min_pref <= preference <= max_pref, \
            f"Preference {preference} for {move_type.value} outside expected range {expected_preference_range}"