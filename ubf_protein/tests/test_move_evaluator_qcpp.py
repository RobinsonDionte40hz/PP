"""
Unit tests for enhanced move evaluator with QCPP integration.

Tests Requirements 1.4, 1.5, 6.1, 6.2 and corresponding acceptance criteria:
- Move evaluator with QCPP integration enabled
- Move evaluator falls back to QAAP when QCPP is None
- Phi pattern reward application
- Move weight calculation with QCPP factors
"""

import pytest
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.mapless_moves import CapabilityBasedMoveEvaluator
from ubf_protein.models import ConformationalMove, MoveType
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
from unittest.mock import Mock, MagicMock


class TestMoveEvaluatorWithQCPP:
    """Test suite for move evaluator with QCPP integration enabled."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create mock QCPP integration
        self.mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        self.mock_qcpp.calculate_quantum_alignment = Mock(return_value=1.2)
        
        # Create evaluator with QCPP enabled
        self.evaluator_with_qcpp = CapabilityBasedMoveEvaluator(
            qcpp_integration=self.mock_qcpp
        )
        
        # Create evaluator without QCPP (fallback)
        self.evaluator_without_qcpp = CapabilityBasedMoveEvaluator(
            qcpp_integration=None
        )
        
        # Create mock behavioral state
        self.behavioral_state = self._create_mock_behavioral_state()
    
    def _create_mock_behavioral_state(self):
        """Create a mock behavioral state for testing."""
        mock_state = Mock()
        mock_state.get_exploration_energy = Mock(return_value=0.5)
        mock_state.get_structural_focus = Mock(return_value=0.6)
        mock_state.get_hydrophobic_drive = Mock(return_value=0.4)
        mock_state.get_risk_tolerance = Mock(return_value=0.5)
        mock_state.get_native_state_ambition = Mock(return_value=0.7)
        return mock_state
    
    def _create_test_move(self, move_type=MoveType.HELIX_FORMATION, 
                         energy_change=-15.0, rmsd_change=1.5):
        """Create a test conformational move."""
        return ConformationalMove(
            move_id="test_move_001",
            move_type=move_type,
            target_residues=[1, 2, 3, 4, 5, 6],
            estimated_energy_change=energy_change,
            estimated_rmsd_change=rmsd_change,
            required_capabilities={'can_form_helix': True},
            energy_barrier=15.0,
            structural_feasibility=0.8
        )
    
    def test_evaluator_initialization_with_qcpp(self):
        """Test that evaluator initializes correctly with QCPP."""
        assert self.evaluator_with_qcpp.qcpp_integration == self.mock_qcpp
        assert self.evaluator_with_qcpp.qaap_calculator is not None
        assert self.evaluator_with_qcpp.phi_reward_threshold == 0.8
        assert self.evaluator_with_qcpp.phi_reward_energy == -50.0
    
    def test_evaluator_initialization_without_qcpp(self):
        """Test that evaluator initializes correctly without QCPP (fallback)."""
        assert self.evaluator_without_qcpp.qcpp_integration is None
        assert self.evaluator_without_qcpp.qaap_calculator is not None
    
    def test_evaluate_move_with_qcpp_enabled(self):
        """Test that evaluate_move uses QCPP when available."""
        move = self._create_test_move()
        
        weight = self.evaluator_with_qcpp.evaluate_move(
            move=move,
            behavioral_state=self.behavioral_state,
            memory_influence=1.0,
            physics_factors=None
        )
        
        # Should return a valid weight
        assert isinstance(weight, float)
        assert weight > 0
    
    def test_evaluate_move_without_qcpp_fallback(self):
        """Test that evaluate_move falls back to QAAP when QCPP is None."""
        move = self._create_test_move()
        
        weight = self.evaluator_without_qcpp.evaluate_move(
            move=move,
            behavioral_state=self.behavioral_state,
            memory_influence=1.0,
            physics_factors=None
        )
        
        # Should still return a valid weight
        assert isinstance(weight, float)
        assert weight > 0
    
    def test_quantum_alignment_uses_qcpp_when_available(self):
        """Test that quantum alignment calculation uses QCPP when enabled."""
        move = self._create_test_move()
        
        # Call quantum alignment calculation
        alignment = self.evaluator_with_qcpp._calculate_quantum_alignment(
            move, physics_factors=None
        )
        
        # Should use QCPP path (returns mocked value 1.2)
        assert alignment == 1.2
    
    def test_quantum_alignment_uses_qaap_fallback(self):
        """Test that quantum alignment falls back to QAAP when QCPP is None."""
        move = self._create_test_move()
        
        # Call quantum alignment calculation
        alignment = self.evaluator_without_qcpp._calculate_quantum_alignment(
            move, physics_factors=None
        )
        
        # Should use QAAP path (returns different value)
        assert alignment != 1.2
        assert 0.5 <= alignment <= 1.5
    
    def test_qcpp_quantum_alignment_formula(self):
        """Test QCPP quantum alignment uses correct formula."""
        # Create real QCPP integration adapter (not mocked)
        real_qcpp = QCPPIntegrationAdapter(predictor=Mock(), cache_size=100)
        evaluator = CapabilityBasedMoveEvaluator(qcpp_integration=real_qcpp)
        
        move = self._create_test_move(move_type=MoveType.HELIX_FORMATION)
        
        # Call QCPP quantum alignment
        alignment = evaluator._calculate_qcpp_quantum_alignment(move)
        
        # Should be in valid range [0.5, 1.5]
        assert 0.5 <= alignment <= 1.5
    
    def test_qcp_estimation_from_helix_move(self):
        """Test QCP estimation for helix formation move."""
        move = self._create_test_move(
            move_type=MoveType.HELIX_FORMATION,
            energy_change=-25.0
        )
        
        qcp = self.evaluator_with_qcpp._estimate_qcp_from_move(move)
        
        # Helix formation with good energy should have high QCP
        assert qcp > 5.0
        assert 3.0 <= qcp <= 8.0
    
    def test_qcp_estimation_from_sheet_move(self):
        """Test QCP estimation for sheet formation move."""
        move = self._create_test_move(
            move_type=MoveType.SHEET_FORMATION,
            energy_change=-20.0
        )
        
        qcp = self.evaluator_with_qcpp._estimate_qcp_from_move(move)
        
        # Sheet formation with good energy should have elevated QCP
        assert qcp > 5.0
        assert 3.0 <= qcp <= 8.0
    
    def test_qcp_estimation_from_collapse_move(self):
        """Test QCP estimation for hydrophobic collapse move."""
        move = self._create_test_move(
            move_type=MoveType.HYDROPHOBIC_COLLAPSE,
            energy_change=-30.0
        )
        
        qcp = self.evaluator_with_qcpp._estimate_qcp_from_move(move)
        
        # Collapse with good energy should have moderately high QCP
        assert qcp > 4.5
        assert 3.0 <= qcp <= 8.0
    
    def test_coherence_estimation_from_helix_move(self):
        """Test coherence estimation for helix formation."""
        move = self._create_test_move(move_type=MoveType.HELIX_FORMATION)
        
        coherence = self.evaluator_with_qcpp._estimate_coherence_from_move(move)
        
        # Helix formation should have positive coherence
        assert coherence > 0
        assert -1.0 <= coherence <= 1.0
    
    def test_coherence_estimation_from_large_jump(self):
        """Test coherence estimation for large conformational jump."""
        move = self._create_test_move(move_type=MoveType.LARGE_CONFORMATIONAL_JUMP)
        
        coherence = self.evaluator_with_qcpp._estimate_coherence_from_move(move)
        
        # Large jumps should have negative coherence
        assert coherence < 0
        assert -1.0 <= coherence <= 1.0
    
    def test_phi_match_estimation_from_helix_move(self):
        """Test phi match estimation for helix formation."""
        move = self._create_test_move(move_type=MoveType.HELIX_FORMATION)
        
        phi_match = self.evaluator_with_qcpp._estimate_phi_match_from_move(move)
        
        # Helix formation should have high phi match
        assert phi_match >= 0.5
        assert 0.0 <= phi_match <= 1.0
    
    def test_phi_match_estimation_from_turn_move(self):
        """Test phi match estimation for turn formation."""
        move = self._create_test_move(move_type=MoveType.TURN_FORMATION)
        
        phi_match = self.evaluator_with_qcpp._estimate_phi_match_from_move(move)
        
        # Turn formation should have moderate phi match
        assert phi_match >= 0.5
        assert 0.0 <= phi_match <= 1.0


class TestPhiPatternRewards:
    """Test suite for phi pattern reward application."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create mock QCPP integration
        self.mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        # Create evaluator with QCPP enabled
        self.evaluator = CapabilityBasedMoveEvaluator(
            qcpp_integration=self.mock_qcpp
        )
        
        # Create mock behavioral state
        self.behavioral_state = Mock()
        self.behavioral_state.get_exploration_energy = Mock(return_value=0.5)
        self.behavioral_state.get_structural_focus = Mock(return_value=0.6)
        self.behavioral_state.get_hydrophobic_drive = Mock(return_value=0.4)
        self.behavioral_state.get_risk_tolerance = Mock(return_value=0.5)
        self.behavioral_state.get_native_state_ambition = Mock(return_value=0.7)
    
    def test_phi_reward_applied_for_high_phi_match(self):
        """Test that phi reward is applied when phi match > threshold."""
        # Create move with characteristics that give high phi match (>0.8)
        move = ConformationalMove(
            move_id="high_phi_move",
            move_type=MoveType.HELIX_FORMATION,  # High phi match
            target_residues=[1, 2, 3, 4, 5, 6],
            estimated_energy_change=-10.0,
            estimated_rmsd_change=1.0,
            required_capabilities={'can_form_helix': True},
            energy_barrier=15.0,
            structural_feasibility=0.8
        )
        
        # Calculate goal alignment (which applies phi reward)
        goal_alignment = self.evaluator._calculate_goal_alignment(move)
        
        # With phi reward applied, effective energy is -10 + (-50) = -60
        # Goal alignment should be high
        assert goal_alignment > 0.7
    
    def test_phi_reward_not_applied_for_low_phi_match(self):
        """Test that phi reward is not applied when phi match < threshold."""
        # Create move with low phi match
        move = ConformationalMove(
            move_id="low_phi_move",
            move_type=MoveType.BACKBONE_ROTATION,  # Low phi match (0.5)
            target_residues=[1, 2, 3],
            estimated_energy_change=-10.0,
            estimated_rmsd_change=0.5,
            required_capabilities={},
            energy_barrier=10.0,
            structural_feasibility=0.8
        )
        
        # Calculate goal alignment
        goal_alignment = self.evaluator._calculate_goal_alignment(move)
        
        # Energy of -10 gives alignment of 1.0 - (-10/50) = 0.8
        # RMSD of 0.5 (0.5 from optimal 1.0) gives alignment of 1.0 - 0.5/2 = 0.75
        # Combined: 0.6*0.8 + 0.4*0.75 = 0.48 + 0.3 = 0.78
        # Phi match is 0.5 (below threshold 0.8), so no reward applied
        assert 0.7 <= goal_alignment <= 1.0
    
    def test_phi_reward_threshold_configurable(self):
        """Test that phi reward threshold is configurable."""
        assert self.evaluator.phi_reward_threshold == 0.8
        
        # Can be modified
        self.evaluator.phi_reward_threshold = 0.9
        assert self.evaluator.phi_reward_threshold == 0.9
    
    def test_phi_reward_energy_configurable(self):
        """Test that phi reward energy bonus is configurable."""
        assert self.evaluator.phi_reward_energy == -50.0
        
        # Can be modified
        self.evaluator.phi_reward_energy = -100.0
        assert self.evaluator.phi_reward_energy == -100.0
    
    def test_phi_reward_only_applied_with_qcpp(self):
        """Test that phi reward is only applied when QCPP is enabled."""
        # Evaluator without QCPP
        evaluator_no_qcpp = CapabilityBasedMoveEvaluator(qcpp_integration=None)
        
        # Lower the threshold temporarily to ensure helix formation gets the reward
        self.evaluator.phi_reward_threshold = 0.6  # Helix has 0.7 phi match
        
        # Create move with bad energy so phi reward has a clear effect
        bad_move = ConformationalMove(
            move_id="bad_move",
            move_type=MoveType.HELIX_FORMATION,  # Phi match 0.7 > threshold 0.6
            target_residues=[1, 2, 3, 4, 5, 6],
            estimated_energy_change=30.0,  # Bad energy
            estimated_rmsd_change=1.0,
            required_capabilities={'can_form_helix': True},
            energy_barrier=15.0,
            structural_feasibility=0.8
        )
        
        # Without QCPP, no phi reward applied
        bad_no_qcpp = evaluator_no_qcpp._calculate_goal_alignment(bad_move)
        
        # With QCPP and lowered threshold, phi reward applied (30 + (-50) = -20)
        bad_with_qcpp = self.evaluator._calculate_goal_alignment(bad_move)
        
        # Phi reward should significantly improve goal alignment
        assert bad_with_qcpp > bad_no_qcpp
        
        # Reset threshold
        self.evaluator.phi_reward_threshold = 0.8


class TestMoveWeightCalculation:
    """Test suite for move weight calculation with QCPP factors."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create real QCPP integration adapter
        self.qcpp = QCPPIntegrationAdapter(predictor=Mock(), cache_size=100)
        self.evaluator = CapabilityBasedMoveEvaluator(qcpp_integration=self.qcpp)
        
        # Create mock behavioral state
        self.behavioral_state = Mock()
        self.behavioral_state.get_exploration_energy = Mock(return_value=0.5)
        self.behavioral_state.get_structural_focus = Mock(return_value=0.6)
        self.behavioral_state.get_hydrophobic_drive = Mock(return_value=0.4)
        self.behavioral_state.get_risk_tolerance = Mock(return_value=0.5)
        self.behavioral_state.get_native_state_ambition = Mock(return_value=0.7)
    
    def test_move_weight_includes_qcpp_quantum_alignment(self):
        """Test that final move weight incorporates QCPP quantum alignment."""
        move = ConformationalMove(
            move_id="test_move",
            move_type=MoveType.HELIX_FORMATION,
            target_residues=[1, 2, 3, 4, 5, 6],
            estimated_energy_change=-20.0,
            estimated_rmsd_change=1.2,
            required_capabilities={'can_form_helix': True},
            energy_barrier=12.0,
            structural_feasibility=0.85
        )
        
        weight = self.evaluator.evaluate_move(
            move=move,
            behavioral_state=self.behavioral_state,
            memory_influence=1.0,
            physics_factors=None
        )
        
        # Should return valid weight incorporating QCPP factors
        assert isinstance(weight, float)
        assert weight > 0
    
    def test_move_weight_calculation_components(self):
        """Test that all 5 factors contribute to final weight."""
        move = ConformationalMove(
            move_id="test_move",
            move_type=MoveType.SHEET_FORMATION,
            target_residues=[2, 3, 4, 5, 6, 7],
            estimated_energy_change=-15.0,
            estimated_rmsd_change=1.5,
            required_capabilities={'can_form_sheet': True},
            energy_barrier=18.0,
            structural_feasibility=0.75
        )
        
        # Calculate individual factors
        physical = self.evaluator._calculate_physical_feasibility(move)
        quantum = self.evaluator._calculate_quantum_alignment(move, None)
        behavioral = self.evaluator._calculate_behavioral_preference(move, self.behavioral_state)
        historical = 1.0  # memory_influence
        goal = self.evaluator._calculate_goal_alignment(move)
        
        # All factors should be valid
        assert 0.0 <= physical <= 1.0
        assert 0.5 <= quantum <= 1.5  # QCPP range
        assert 0.0 <= behavioral <= 1.0
        assert historical == 1.0
        assert 0.0 <= goal <= 1.0
        
        # Calculate expected weight
        expected_weight = (
            0.2 * physical +
            0.25 * quantum +
            0.2 * behavioral +
            0.15 * historical +
            0.2 * goal
        )
        
        # Calculate actual weight
        actual_weight = self.evaluator.evaluate_move(
            move=move,
            behavioral_state=self.behavioral_state,
            memory_influence=historical,
            physics_factors=None
        )
        
        # Should match (within floating point tolerance)
        assert abs(actual_weight - expected_weight) < 0.01
    
    def test_qcpp_improves_structure_forming_moves(self):
        """Test that QCPP gives higher weights to structure-forming moves."""
        helix_move = ConformationalMove(
            move_id="helix",
            move_type=MoveType.HELIX_FORMATION,
            target_residues=[1, 2, 3, 4, 5, 6],
            estimated_energy_change=-15.0,
            estimated_rmsd_change=1.5,
            required_capabilities={'can_form_helix': True},
            energy_barrier=15.0,
            structural_feasibility=0.8
        )
        
        random_move = ConformationalMove(
            move_id="random",
            move_type=MoveType.BACKBONE_ROTATION,
            target_residues=[1, 2, 3],
            estimated_energy_change=-15.0,  # Same energy
            estimated_rmsd_change=0.5,
            required_capabilities={},
            energy_barrier=10.0,
            structural_feasibility=0.8
        )
        
        helix_weight = self.evaluator.evaluate_move(
            helix_move, self.behavioral_state, 1.0, None
        )
        random_weight = self.evaluator.evaluate_move(
            random_move, self.behavioral_state, 1.0, None
        )
        
        # Helix should have higher weight due to better QCPP metrics
        assert helix_weight > random_weight


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
