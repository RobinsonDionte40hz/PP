"""
Tests for Hierarchical Folding System

Tests for:
- ResidueAnchorManager
- PhaseController
- HierarchicalFoldingManager

Author: UBF Protein System
Date: December 5, 2025
"""

import pytest
import math
from unittest.mock import Mock, MagicMock

# Import the modules under test
from ubf_protein.residue_anchor_manager import (
    ResidueAnchorManager,
    AnchorManagerConfig,
    StructuralState,
    AnchorStrength,
    ResidueAnchorState,
    create_anchor_manager
)
from ubf_protein.phase_controller import (
    PhaseController,
    PhaseControllerConfig,
    ExplorationPhase,
    PhaseMetrics,
    create_phase_controller
)
from ubf_protein.hierarchical_folding import (
    HierarchicalFoldingManager,
    HierarchicalFoldingConfig,
    create_hierarchical_folding_manager
)


class TestResidueAnchorState:
    """Tests for ResidueAnchorState dataclass."""
    
    def test_initial_state(self):
        """Test default initial state."""
        state = ResidueAnchorState(residue_index=0)
        
        assert state.residue_index == 0
        assert state.current_state == StructuralState.UNSTRUCTURED
        assert state.state_confidence == 0.0
        assert state.consecutive_observations == 0
        assert state.anchor_strength == AnchorStrength.FREE
        assert state.target_phi is None
        assert state.target_psi is None
    
    def test_allowed_deviation(self):
        """Test allowed deviation based on anchor strength."""
        state = ResidueAnchorState(residue_index=0)
        
        # Free: 180 degrees
        state.anchor_strength = AnchorStrength.FREE
        assert state.get_allowed_deviation() == 180.0
        
        # Soft: 30 degrees
        state.anchor_strength = AnchorStrength.SOFT
        assert state.get_allowed_deviation() == 30.0
        
        # Medium: 15 degrees
        state.anchor_strength = AnchorStrength.MEDIUM
        assert state.get_allowed_deviation() == 15.0
        
        # Hard: 5 degrees
        state.anchor_strength = AnchorStrength.HARD
        assert state.get_allowed_deviation() == 5.0
        
        # Locked: 2 degrees
        state.anchor_strength = AnchorStrength.LOCKED
        assert state.get_allowed_deviation() == 2.0
    
    def test_is_anchored(self):
        """Test is_anchored method."""
        state = ResidueAnchorState(residue_index=0)
        
        state.anchor_strength = AnchorStrength.FREE
        assert not state.is_anchored()
        
        state.anchor_strength = AnchorStrength.SOFT
        assert state.is_anchored()
        
        state.anchor_strength = AnchorStrength.LOCKED
        assert state.is_anchored()


class TestResidueAnchorManager:
    """Tests for ResidueAnchorManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = ResidueAnchorManager(sequence_length=50)
        
        assert manager.sequence_length == 50
        assert len(manager._residue_states) == 50
        assert manager.total_updates == 0
        assert manager.get_anchoring_percentage() == 0.0
    
    def test_update_residue_state_new(self):
        """Test updating a residue with new structural state."""
        manager = ResidueAnchorManager(sequence_length=20)
        
        # Update residue 5 to helix
        manager.update_residue_state(
            residue_index=5,
            detected_state=StructuralState.HELIX,
            confidence=0.8,
            phi_angle=-60.0,
            psi_angle=-45.0
        )
        
        state = manager.get_residue_state(5)
        assert state.current_state == StructuralState.HELIX
        assert state.consecutive_observations == 1
        assert state.target_phi == -60.0
        assert state.target_psi == -45.0
    
    def test_update_residue_state_reinforcement(self):
        """Test that repeated observations reinforce confidence."""
        manager = ResidueAnchorManager(sequence_length=20)
        
        # Repeat helix observation multiple times
        for i in range(10):
            manager.update_residue_state(
                residue_index=5,
                detected_state=StructuralState.HELIX,
                confidence=0.8
            )
        
        state = manager.get_residue_state(5)
        assert state.consecutive_observations == 10
        assert state.state_confidence > 0.5  # Should have increased
    
    def test_update_residue_state_regression(self):
        """Test that state changes cause confidence regression."""
        manager = ResidueAnchorManager(sequence_length=20)
        
        # First establish helix
        for i in range(5):
            manager.update_residue_state(5, StructuralState.HELIX, 0.8)
        
        initial_confidence = manager.get_residue_state(5).state_confidence
        
        # Now change to sheet
        manager.update_residue_state(5, StructuralState.SHEET, 0.8)
        
        state = manager.get_residue_state(5)
        assert state.current_state == StructuralState.SHEET
        assert state.consecutive_observations == 1
        assert state.state_confidence < initial_confidence
    
    def test_update_from_secondary_structure(self):
        """Test batch update from SS string."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # HHHHCCEECC pattern
        ss_string = "HHHHCCEECC"
        manager.update_from_secondary_structure(ss_string)
        
        # Check helix residues
        for i in range(4):
            assert manager.get_residue_state(i).current_state == StructuralState.HELIX
        
        # Check coil residues
        for i in [4, 5, 8, 9]:
            assert manager.get_residue_state(i).current_state == StructuralState.UNSTRUCTURED
        
        # Check sheet residues
        for i in [6, 7]:
            assert manager.get_residue_state(i).current_state == StructuralState.SHEET
    
    def test_is_move_allowed_free(self):
        """Test that moves are allowed for free residues."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Free residue - any angle should be allowed
        assert manager.is_move_allowed(5, new_phi=-120.0, new_psi=150.0)
        assert manager.is_move_allowed(5, new_phi=60.0, new_psi=-60.0)
    
    def test_is_move_allowed_anchored(self):
        """Test move constraints for anchored residues."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Manually set up an anchored residue
        state = manager._residue_states[5]
        state.anchor_strength = AnchorStrength.HARD  # ±5° allowed
        state.target_phi = -60.0
        state.target_psi = -45.0
        
        # Within tolerance
        assert manager.is_move_allowed(5, new_phi=-58.0, new_psi=-43.0)
        
        # Outside tolerance
        assert not manager.is_move_allowed(5, new_phi=-80.0, new_psi=-45.0)
        assert not manager.is_move_allowed(5, new_phi=-60.0, new_psi=-60.0)
    
    def test_constrain_angles(self):
        """Test angle constraining."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Set up anchored residue
        state = manager._residue_states[5]
        state.anchor_strength = AnchorStrength.MEDIUM  # ±15° allowed
        state.target_phi = -60.0
        state.target_psi = -45.0
        
        # Angles within tolerance - unchanged
        phi, psi = manager.constrain_angles(5, -65.0, -50.0)
        assert phi == -65.0
        assert psi == -50.0
        
        # Angles outside tolerance - clamped
        phi, psi = manager.constrain_angles(5, -100.0, -100.0)
        assert abs(phi - (-75.0)) < 0.1  # Clamped to -60 - 15
        assert abs(psi - (-60.0)) < 0.1  # Clamped to -45 - 15
    
    def test_get_anchored_residues(self):
        """Test getting list of anchored residues."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Anchor some residues
        manager._residue_states[2].anchor_strength = AnchorStrength.SOFT
        manager._residue_states[5].anchor_strength = AnchorStrength.MEDIUM
        manager._residue_states[7].anchor_strength = AnchorStrength.HARD
        
        anchored = manager.get_anchored_residues()
        assert 2 in anchored
        assert 5 in anchored
        assert 7 in anchored
        assert len(anchored) == 3
    
    def test_get_free_residues(self):
        """Test getting list of free residues."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Anchor some residues
        manager._residue_states[2].anchor_strength = AnchorStrength.SOFT
        manager._residue_states[5].anchor_strength = AnchorStrength.MEDIUM
        
        free = manager.get_free_residues()
        assert 0 in free
        assert 1 in free
        assert 2 not in free
        assert 5 not in free
        assert len(free) == 8
    
    def test_get_anchoring_percentage(self):
        """Test anchoring percentage calculation."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        assert manager.get_anchoring_percentage() == 0.0
        
        # Anchor 3 out of 10
        manager._residue_states[0].anchor_strength = AnchorStrength.SOFT
        manager._residue_states[1].anchor_strength = AnchorStrength.SOFT
        manager._residue_states[2].anchor_strength = AnchorStrength.SOFT
        
        assert manager.get_anchoring_percentage() == 30.0
    
    def test_confidence_decay(self):
        """Test confidence decay."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Set up some confidence
        manager._residue_states[5].state_confidence = 0.5
        
        manager.decay_confidence()
        
        assert manager._residue_states[5].state_confidence < 0.5
    
    def test_get_anchoring_summary(self):
        """Test summary generation."""
        manager = ResidueAnchorManager(sequence_length=20)
        
        summary = manager.get_anchoring_summary()
        
        assert 'sequence_length' in summary
        assert 'anchoring_percentage' in summary
        assert 'anchor_distribution' in summary
        assert 'statistics' in summary


class TestPhaseController:
    """Tests for PhaseController."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = PhaseController(sequence_length=50)
        
        assert controller.sequence_length == 50
        assert controller.current_phase == ExplorationPhase.FREE_EXPLORATION
        assert controller.metrics.current_iteration == 0
    
    def test_update_metrics(self):
        """Test metrics update."""
        controller = PhaseController(sequence_length=50)
        
        controller.update_metrics(
            iteration=10,
            energy=-50.0,
            rmsd=15.0,
            helix_pct=20.0,
            sheet_pct=10.0,
            anchoring_pct=5.0
        )
        
        m = controller.metrics
        assert m.current_iteration == 10
        assert m.current_energy == -50.0
        assert m.current_rmsd == 15.0
        assert m.helix_percentage == 20.0
        assert m.sheet_percentage == 10.0
        assert m.structured_percentage == 30.0
        assert m.anchoring_percentage == 5.0
    
    def test_should_transition_free_to_local(self):
        """Test transition from FREE_EXPLORATION to LOCAL_ANCHORING."""
        config = PhaseControllerConfig(
            min_iterations_free=10,
            min_structured_pct_for_anchoring=15.0
        )
        controller = PhaseController(sequence_length=50, config=config)
        
        # Not enough iterations
        for i in range(5):
            controller.update_metrics(i, -50.0, helix_pct=20.0, sheet_pct=10.0)
        assert not controller.should_transition()
        
        # Enough iterations but structure not stable
        for i in range(5, 15):
            controller.update_metrics(i, -50.0, helix_pct=20.0, sheet_pct=10.0)
        
        # Should eventually trigger transition
        # (need stable structure observations)
        controller._metrics.structure_stable_iterations = 15
        assert controller.should_transition()
    
    def test_transition_to_next_phase(self):
        """Test phase transition."""
        controller = PhaseController(sequence_length=50)
        
        assert controller.current_phase == ExplorationPhase.FREE_EXPLORATION
        
        new_phase = controller.transition_to_next_phase()
        assert new_phase == ExplorationPhase.LOCAL_ANCHORING
        assert controller.current_phase == ExplorationPhase.LOCAL_ANCHORING
        
        new_phase = controller.transition_to_next_phase()
        assert new_phase == ExplorationPhase.TERTIARY_PACKING
        
        new_phase = controller.transition_to_next_phase()
        assert new_phase == ExplorationPhase.REFINEMENT
        
        # Can't go past refinement
        new_phase = controller.transition_to_next_phase()
        assert new_phase == ExplorationPhase.REFINEMENT
    
    def test_rollback_phase(self):
        """Test phase rollback."""
        controller = PhaseController(sequence_length=50)
        
        # Advance to TERTIARY_PACKING
        controller.transition_to_next_phase()  # LOCAL_ANCHORING
        controller.transition_to_next_phase()  # TERTIARY_PACKING
        
        assert controller.current_phase == ExplorationPhase.TERTIARY_PACKING
        
        new_phase = controller.rollback_phase()
        assert new_phase == ExplorationPhase.LOCAL_ANCHORING
        
        # Can't roll back past FREE_EXPLORATION
        controller.rollback_phase()  # FREE_EXPLORATION
        new_phase = controller.rollback_phase()
        assert new_phase == ExplorationPhase.FREE_EXPLORATION
    
    def test_get_move_scale(self):
        """Test move scale per phase."""
        controller = PhaseController(sequence_length=50)
        
        assert controller.get_move_scale() == 1.0  # FREE
        
        controller.transition_to_next_phase()
        assert controller.get_move_scale() == 0.8  # LOCAL_ANCHORING
        
        controller.transition_to_next_phase()
        assert controller.get_move_scale() == 0.5  # TERTIARY_PACKING
        
        controller.transition_to_next_phase()
        assert controller.get_move_scale() == 0.2  # REFINEMENT (default)
    
    def test_get_move_type_weights(self):
        """Test move type weights per phase."""
        controller = PhaseController(sequence_length=50)
        
        # FREE_EXPLORATION
        weights = controller.get_move_type_weights()
        assert weights['hydrophobic_collapse'] == 1.5  # Slight boost
        
        # LOCAL_ANCHORING
        controller.transition_to_next_phase()
        weights = controller.get_move_type_weights()
        assert weights['helix_formation'] == 1.5
        assert weights['sheet_formation'] == 1.5
        
        # TERTIARY_PACKING
        controller.transition_to_next_phase()
        weights = controller.get_move_type_weights()
        assert weights['hydrophobic_collapse'] == 2.0
        assert weights['pivot_rotation'] == 1.5
        
        # REFINEMENT
        controller.transition_to_next_phase()
        weights = controller.get_move_type_weights()
        assert weights['sidechain_adjust'] == 2.0
        assert weights['energy_minimization'] == 2.0
        assert weights['pivot_rotation'] == 0.1  # Reduced
    
    def test_is_in_refinement(self):
        """Test refinement phase detection."""
        controller = PhaseController(sequence_length=50)
        
        assert not controller.is_in_refinement()
        
        controller.transition_to_next_phase()  # LOCAL
        controller.transition_to_next_phase()  # TERTIARY
        assert not controller.is_in_refinement()
        
        controller.transition_to_next_phase()  # REFINEMENT
        assert controller.is_in_refinement()
    
    def test_get_phase_summary(self):
        """Test phase summary generation."""
        controller = PhaseController(sequence_length=50)
        
        summary = controller.get_phase_summary()
        
        assert 'current_phase' in summary
        assert 'iterations_in_phase' in summary
        assert 'metrics' in summary
        assert 'move_scale' in summary


class TestHierarchicalFoldingManager:
    """Tests for HierarchicalFoldingManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = HierarchicalFoldingManager(sequence_length=50)
        
        assert manager.sequence_length == 50
        assert manager.anchor_manager is not None
        assert manager.phase_controller is not None
    
    def test_initialization_disabled(self):
        """Test manager with disabled components."""
        config = HierarchicalFoldingConfig(
            enable_anchoring=False,
            enable_phase_control=False
        )
        manager = HierarchicalFoldingManager(sequence_length=50, config=config)
        
        assert manager.anchor_manager is None
        assert manager.phase_controller is None
    
    def test_update(self):
        """Test update method."""
        manager = HierarchicalFoldingManager(sequence_length=20)
        
        # Create mock conformation
        mock_conf = Mock()
        mock_conf.phi_angles = [-60.0] * 20
        mock_conf.psi_angles = [-45.0] * 20
        mock_conf.secondary_structure = ['H'] * 10 + ['C'] * 10
        
        # Should not error
        manager.update(
            iteration=5,
            conformation=mock_conf,
            energy=-50.0,
            rmsd=10.0
        )
    
    def test_is_move_allowed(self):
        """Test move allowance check."""
        manager = HierarchicalFoldingManager(sequence_length=10)
        
        # Create mock move
        mock_move = Mock()
        mock_move.target_residues = [3, 4, 5]
        
        # Initially all free
        assert manager.is_move_allowed(
            mock_move,
            proposed_phi=[-60.0] * 10,
            proposed_psi=[-45.0] * 10
        )
    
    def test_get_move_scale(self):
        """Test move scale retrieval."""
        manager = HierarchicalFoldingManager(sequence_length=50)
        
        # Initially FREE_EXPLORATION
        assert manager.get_move_scale() == 1.0
    
    def test_get_move_type_weights(self):
        """Test move type weights."""
        manager = HierarchicalFoldingManager(sequence_length=50)
        
        weights = manager.get_move_type_weights()
        assert isinstance(weights, dict)
        assert 'hydrophobic_collapse' in weights
    
    def test_get_current_phase(self):
        """Test phase retrieval."""
        manager = HierarchicalFoldingManager(sequence_length=50)
        
        phase = manager.get_current_phase()
        assert phase == ExplorationPhase.FREE_EXPLORATION
    
    def test_get_anchoring_percentage(self):
        """Test anchoring percentage."""
        manager = HierarchicalFoldingManager(sequence_length=50)
        
        assert manager.get_anchoring_percentage() == 0.0
    
    def test_get_free_residues(self):
        """Test free residues retrieval."""
        manager = HierarchicalFoldingManager(sequence_length=10)
        
        free = manager.get_free_residues()
        assert len(free) == 10
    
    def test_get_anchored_residues(self):
        """Test anchored residues retrieval."""
        manager = HierarchicalFoldingManager(sequence_length=10)
        
        anchored = manager.get_anchored_residues()
        assert len(anchored) == 0
    
    def test_get_summary(self):
        """Test summary generation."""
        manager = HierarchicalFoldingManager(sequence_length=50)
        
        summary = manager.get_summary()
        
        assert 'sequence_length' in summary
        assert 'config' in summary
        assert 'anchoring' in summary
        assert 'phase' in summary


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_anchor_manager(self):
        """Test anchor manager factory."""
        manager = create_anchor_manager(50)
        assert manager.sequence_length == 50
        
        aggressive = create_anchor_manager(50, aggressive=True)
        assert aggressive.config.soft_anchor_confidence < manager.config.soft_anchor_confidence
    
    def test_create_phase_controller(self):
        """Test phase controller factory."""
        controller = create_phase_controller(50)
        assert controller.sequence_length == 50
        
        fast = create_phase_controller(50, fast_mode=True)
        assert fast.config.min_iterations_free < controller.config.min_iterations_free
    
    def test_create_hierarchical_folding_manager(self):
        """Test hierarchical manager factory."""
        manager = create_hierarchical_folding_manager(50)
        assert manager.sequence_length == 50
        assert manager.anchor_manager is not None
        assert manager.phase_controller is not None
        
        no_anchoring = create_hierarchical_folding_manager(50, enable_anchoring=False)
        assert no_anchoring.anchor_manager is None


class TestAngleCalculations:
    """Tests for angle-related calculations."""
    
    def test_angle_difference(self):
        """Test angle difference calculation (handling wraparound)."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Simple difference
        assert manager._angle_difference(45.0, 30.0) == 15.0
        
        # Wraparound case
        diff = manager._angle_difference(170.0, -170.0)
        assert abs(diff - 20.0) < 0.1  # Should be ~20, not 340
        
        # Same angle
        assert manager._angle_difference(90.0, 90.0) == 0.0
    
    def test_clamp_angle(self):
        """Test angle clamping."""
        manager = ResidueAnchorManager(sequence_length=10)
        
        # Within range
        result = manager._clamp_angle(35.0, 30.0, 10.0)
        assert result == 35.0
        
        # Above range
        result = manager._clamp_angle(50.0, 30.0, 10.0)
        assert abs(result - 40.0) < 0.1
        
        # Below range
        result = manager._clamp_angle(10.0, 30.0, 10.0)
        assert abs(result - 20.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
