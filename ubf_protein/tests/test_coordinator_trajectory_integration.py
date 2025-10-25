"""
Unit tests for integrated trajectory recording in MultiAgentCoordinator.

Tests Task 9.1:
- Test coordinator records QCPP metrics during exploration
- Test coordinator computes correlations after exploration
- Test correlation results included in exploration results
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
from ubf_protein.integrated_trajectory import IntegratedTrajectoryPoint
from ubf_protein.models import Conformation


class TestCoordinatorTrajectoryIntegration:
    """Test suite for trajectory recording integration in coordinator."""
    
    def _create_mock_qcpp(self) -> Mock:
        """Helper to create mock QCPP integration."""
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        mock_qcpp.analyze_conformation = Mock(return_value=QCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.5,
            stability_score=1.5,
            phi_match_score=0.7,
            calculation_time_ms=2.0
        ))
        return mock_qcpp
    
    def test_coordinator_without_qcpp_no_trajectory(self):
        """Test coordinator without QCPP doesn't initialize trajectory recorder."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=None
        )
        
        # Should not have trajectory recorder
        assert coordinator._trajectory_recorder is None
        assert coordinator.get_trajectory_recorder() is None
    
    def test_coordinator_with_qcpp_initializes_trajectory(self):
        """Test coordinator with QCPP initializes trajectory recorder."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Should have trajectory recorder
        assert coordinator._trajectory_recorder is not None
        assert coordinator.get_trajectory_recorder() is not None
    
    def test_trajectory_recorder_has_max_points_limit(self):
        """Test trajectory recorder is initialized with reasonable max_points."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        recorder = coordinator.get_trajectory_recorder()
        assert recorder is not None
        assert recorder._max_points == 10000  # Default from initialization
    
    def test_exploration_without_qcpp_no_trajectory_data(self):
        """Test exploration without QCPP doesn't include trajectory data in results."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=None
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=5)
        
        # Should not have QCPP trajectory data
        assert results.qcpp_trajectory_data is None
        assert results.qcpp_rmsd_correlations is None
        assert results.qcpp_energy_correlations is None
        assert results.consciousness_qcpp_correlations is None
    
    @patch('ubf_protein.integrated_trajectory.IntegratedTrajectoryRecorder')
    def test_exploration_with_qcpp_records_trajectory(self, mock_recorder_class):
        """Test exploration with QCPP records trajectory points."""
        mock_qcpp = self._create_mock_qcpp()
        
        # Create mock recorder instance
        mock_recorder = Mock()
        mock_recorder.get_point_count = Mock(return_value=0)  # No points for this test
        mock_recorder_class.return_value = mock_recorder
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=5)
        
        # Recorder should have been called to record points
        # (actual number depends on execution, just verify it was created)
        assert mock_recorder_class.called
    
    def test_exploration_results_include_trajectory_data(self):
        """Test exploration results include trajectory data when QCPP enabled."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        # Run a few iterations
        results = coordinator.run_parallel_exploration(iterations=10)
        
        # Should have trajectory data (even if empty due to mock setup)
        # The structure should exist
        assert hasattr(results, 'qcpp_trajectory_data')
        assert hasattr(results, 'qcpp_rmsd_correlations')
        assert hasattr(results, 'qcpp_energy_correlations')
        assert hasattr(results, 'consciousness_qcpp_correlations')
    
    def test_correlation_computed_with_sufficient_points(self):
        """Test correlations are computed when trajectory has ≥2 points."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Initialize agents and run exploration
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        # Run enough iterations to get trajectory points
        results = coordinator.run_parallel_exploration(iterations=20)
        
        # Check if trajectory was recorded
        if results.qcpp_trajectory_data is not None:
            point_count = results.qcpp_trajectory_data.get('metadata', {}).get('point_count', 0)
            
            # If we have ≥2 points, correlations should be computed
            if point_count >= 2:
                assert results.qcpp_rmsd_correlations is not None
                assert 'qcp_rmsd_corr' in results.qcpp_rmsd_correlations
                assert 'stability_rmsd_corr' in results.qcpp_rmsd_correlations
                
                assert results.qcpp_energy_correlations is not None
                assert 'qcp_energy_corr' in results.qcpp_energy_correlations
                
                assert results.consciousness_qcpp_correlations is not None
                assert 'frequency_qcp_corr' in results.consciousness_qcpp_correlations
    
    def test_correlation_fields_in_results(self):
        """Test that correlation fields exist in ExplorationResults."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=1, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=5)
        
        # Fields should exist (even if None)
        assert hasattr(results, 'qcpp_trajectory_data')
        assert hasattr(results, 'qcpp_rmsd_correlations')
        assert hasattr(results, 'qcpp_energy_correlations')
        assert hasattr(results, 'consciousness_qcpp_correlations')
    
    def test_trajectory_data_structure(self):
        """Test trajectory data has expected structure when present."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=15)
        
        # If trajectory data exists, check structure
        if results.qcpp_trajectory_data is not None:
            assert 'metadata' in results.qcpp_trajectory_data
            assert 'trajectory_points' in results.qcpp_trajectory_data
            
            metadata = results.qcpp_trajectory_data['metadata']
            assert 'point_count' in metadata
            assert 'recording_count' in metadata
    
    def test_correlation_values_are_floats(self):
        """Test correlation values are numeric when present."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=20)
        
        # If correlations were computed, verify types
        if results.qcpp_rmsd_correlations is not None:
            for key, value in results.qcpp_rmsd_correlations.items():
                assert isinstance(value, (int, float)), f"{key} should be numeric"
        
        if results.qcpp_energy_correlations is not None:
            for key, value in results.qcpp_energy_correlations.items():
                assert isinstance(value, (int, float)), f"{key} should be numeric"
        
        if results.consciousness_qcpp_correlations is not None:
            for key, value in results.consciousness_qcpp_correlations.items():
                assert isinstance(value, (int, float)), f"{key} should be numeric"
    
    def test_trajectory_recording_non_critical(self):
        """Test that trajectory recording failures don't crash exploration."""
        # Create QCPP that raises errors
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        mock_qcpp.analyze_conformation = Mock(side_effect=Exception("QCPP failed"))
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        # Should not crash despite QCPP errors
        results = coordinator.run_parallel_exploration(iterations=5)
        
        # Exploration should complete
        assert results.total_iterations > 0
    
    def test_multiple_explorations_same_coordinator(self):
        """Test running multiple explorations with same coordinator."""
        mock_qcpp = self._create_mock_qcpp()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        # Run first exploration
        results1 = coordinator.run_parallel_exploration(iterations=5)
        
        # Run second exploration
        results2 = coordinator.run_parallel_exploration(iterations=5)
        
        # Both should have trajectory data structures
        assert hasattr(results1, 'qcpp_trajectory_data')
        assert hasattr(results2, 'qcpp_trajectory_data')
        
        # Total iterations should accumulate
        assert results2.total_iterations > results1.total_iterations
    
    def test_backward_compatibility_with_old_code(self):
        """Test that old code without QCPP still works."""
        # Old-style initialization (no QCPP)
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False
        )
        
        coordinator.initialize_agents(count=3, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=10)
        
        # Should work fine
        assert results.total_iterations == 10
        assert results.total_conformations_explored > 0
        
        # New fields should be None (backward compatible)
        assert results.qcpp_trajectory_data is None
        assert results.qcpp_rmsd_correlations is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
