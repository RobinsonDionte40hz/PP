"""
Integration tests for multi-agent coordination system.

Tests the MultiAgentCoordinator with multiple ProteinAgent instances
working together to explore conformational space.
"""

import pytest
from unittest.mock import Mock, patch

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.memory_system import SharedMemoryPool
from ubf_protein.models import ExplorationResults, Conformation
from ubf_protein.config import AGENT_DIVERSITY_PROFILES


class TestMultiAgentCoordinator:
    """Test suite for MultiAgentCoordinator"""

    def test_initialize_agents_balanced_diversity(self):
        """Test agent initialization with balanced diversity profile"""
        coordinator = MultiAgentCoordinator("TESTSEQ")

        # Test with 9 agents (should give 3 of each type)
        agents = coordinator.initialize_agents(9, "balanced")

        assert len(agents) == 9

        # Count agent types by checking their initial consciousness ranges
        cautious_count = 0
        balanced_count = 0
        aggressive_count = 0

        for agent in agents:
            freq = agent.get_consciousness_state().get_frequency()
            coh = agent.get_consciousness_state().get_coherence()

            if 4.0 <= freq <= 7.0 and 0.7 <= coh <= 1.0:
                cautious_count += 1
            elif 7.0 <= freq <= 10.0 and 0.5 <= coh <= 0.8:
                balanced_count += 1
            elif 10.0 <= freq <= 15.0 and 0.3 <= coh <= 0.6:
                aggressive_count += 1

        assert cautious_count == 3, f"Expected 3 cautious agents, got {cautious_count}"
        assert balanced_count == 3, f"Expected 3 balanced agents, got {balanced_count}"
        assert aggressive_count == 3, f"Expected 3 aggressive agents, got {aggressive_count}"

    def test_initialize_agents_single_profile(self):
        """Test agent initialization with single profile"""
        coordinator = MultiAgentCoordinator("TESTSEQ")

        # Test with single profile
        agents = coordinator.initialize_agents(5, "cautious")

        assert len(agents) == 5

        # All agents should be cautious
        for agent in agents:
            freq = agent.get_consciousness_state().get_frequency()
            coh = agent.get_consciousness_state().get_coherence()

            assert 4.0 <= freq <= 7.0, f"Cautious agent frequency {freq} out of range"
            assert 0.7 <= coh <= 1.0, f"Cautious agent coherence {coh} out of range"

    def test_initialize_agents_invalid_profile(self):
        """Test agent initialization with invalid profile raises error"""
        coordinator = MultiAgentCoordinator("TESTSEQ")

        with pytest.raises(ValueError, match="Unknown diversity profile"):
            coordinator.initialize_agents(3, "invalid_profile")

    @patch('ubf_protein.multi_agent_coordinator.time.time')
    def test_run_parallel_exploration_basic(self, mock_time):
        """Test basic parallel exploration execution"""
        mock_time.return_value = 1000.0  # Fixed time for testing

        coordinator = MultiAgentCoordinator("TESTSEQ")
        coordinator.initialize_agents(2, "balanced")

        # Mock the exploration step to avoid complex simulation
        with patch.object(ProteinAgent, 'explore_step') as mock_explore:
            # Create mock outcome
            mock_outcome = Mock()
            mock_outcome.significance = 0.5  # Below sharing threshold
            mock_outcome.new_conformation.energy = 900.0
            mock_outcome.new_conformation.rmsd_to_native = 5.0
            mock_explore.return_value = mock_outcome

            # Mock get_current_conformation
            mock_conformation = Mock()
            mock_conformation.energy = 900.0
            mock_conformation.rmsd_to_native = 5.0

            with patch.object(ProteinAgent, 'get_current_conformation', return_value=mock_conformation):
                with patch.object(ProteinAgent, 'get_exploration_metrics', return_value={
                    'iterations_completed': 10,
                    'conformations_explored': 10,
                    'memories_created': 2,
                    'best_energy': 900.0,
                    'best_rmsd': 5.0,
                    'avg_decision_time_ms': 50.0,
                    'stuck_in_minima_count': 0,
                    'successful_escapes': 0
                }):
                    results = coordinator.run_parallel_exploration(10)

        # Verify results structure
        assert isinstance(results, ExplorationResults)
        assert results.total_iterations == 10
        assert results.total_conformations_explored == 20  # 2 agents * 10 iterations
        assert results.agent_metrics is not None
        assert len(results.agent_metrics) == 2
        assert results.total_runtime_seconds == 0.0  # Same start/end time in test

    def test_get_best_conformation_no_exploration(self):
        """Test get_best_conformation raises error when no exploration performed"""
        coordinator = MultiAgentCoordinator("TESTSEQ")

        with pytest.raises(ValueError, match="No exploration has been performed yet"):
            coordinator.get_best_conformation()

    def test_get_best_conformation_after_exploration(self):
        """Test get_best_conformation returns best found conformation"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        coordinator.initialize_agents(1, "balanced")

        # Manually set best conformation
        mock_conformation = Mock()
        mock_conformation.energy = 800.0
        mock_conformation.rmsd_to_native = 3.0

        coordinator._best_conformation = mock_conformation
        coordinator._best_energy = 800.0
        coordinator._best_rmsd = 3.0

        best_conf, best_energy, best_rmsd = coordinator.get_best_conformation()

        assert best_conf == mock_conformation
        assert best_energy == 800.0
        assert best_rmsd == 3.0

    def test_shared_memory_pool_integration(self):
        """Test that agents share high-significance memories"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        coordinator.initialize_agents(2, "balanced")

        # Mock exploration with high-significance outcome
        with patch.object(ProteinAgent, 'explore_step') as mock_explore:
            mock_outcome = Mock()
            mock_outcome.significance = 0.8  # Above sharing threshold
            mock_outcome.move_executed = Mock()
            mock_outcome.move_executed.move_type = Mock()
            mock_outcome.move_executed.move_type.value = "backbone_rotation"
            mock_explore.return_value = mock_outcome

            # Mock memory creation
            with patch('ubf_protein.memory_system.MemorySystem.retrieve_relevant_memories') as mock_retrieve:
                mock_memory = Mock()
                mock_memory.significance = 0.8
                mock_retrieve.return_value = [mock_memory]

                # Mock get_current_conformation to return a proper mock with energy
                mock_conformation = Mock()
                mock_conformation.energy = 900.0
                mock_conformation.rmsd_to_native = 5.0

                with patch.object(ProteinAgent, 'get_current_conformation', return_value=mock_conformation):
                    coordinator.run_parallel_exploration(1)

        # Check that memory was shared
        pool = coordinator.get_shared_memory_pool()
        assert pool.get_total_memories() >= 1

    def test_agent_metrics_collection(self):
        """Test that agent metrics are properly collected"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        coordinator.initialize_agents(1, "balanced")

        with patch.object(ProteinAgent, 'explore_step') as mock_explore:
            mock_outcome = Mock()
            mock_outcome.significance = 0.5
            mock_explore.return_value = mock_outcome

            with patch.object(ProteinAgent, 'get_current_conformation') as mock_get_conf:
                mock_conf = Mock()
                mock_conf.energy = 850.0
                mock_conf.rmsd_to_native = 4.0
                mock_get_conf.return_value = mock_conf

                with patch.object(ProteinAgent, 'get_exploration_metrics') as mock_metrics:
                    mock_metrics.return_value = {
                        'iterations_completed': 5,
                        'conformations_explored': 5,
                        'memories_created': 1,
                        'best_energy': 850.0,
                        'best_rmsd': 4.0,
                        'avg_decision_time_ms': 25.0,
                        'stuck_in_minima_count': 0,
                        'successful_escapes': 0
                    }

                    results = coordinator.run_parallel_exploration(5)

        # Verify metrics
        assert len(results.agent_metrics) == 1
        metrics = results.agent_metrics[0]
        assert metrics.iterations_completed == 5
        assert metrics.conformations_explored == 5
        assert metrics.memories_created == 1
        assert metrics.best_energy_found == 850.0
        assert metrics.best_rmsd_found == 4.0
        assert metrics.avg_decision_time_ms == 25.0

    def test_get_agents_and_shared_memory_pool(self):
        """Test accessor methods for agents and shared memory pool"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        agents = coordinator.initialize_agents(3, "balanced")

        assert len(coordinator.get_agents()) == 3
        assert coordinator.get_agents() == agents

        pool = coordinator.get_shared_memory_pool()
        assert isinstance(pool, SharedMemoryPool)

    def test_best_conformation_tracking(self):
        """Test that best conformation is properly tracked across agents"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        coordinator.initialize_agents(2, "balanced")

        # Create mock conformations with different energies
        conf1 = Mock()
        conf1.energy = 900.0
        conf1.rmsd_to_native = 6.0

        conf2 = Mock()
        conf2.energy = 800.0  # Better energy
        conf2.rmsd_to_native = 4.0

        # Simulate exploration where agent 1 finds better conformation
        with patch.object(ProteinAgent, 'explore_step') as mock_explore:
            mock_outcome = Mock()
            mock_outcome.significance = 0.5
            mock_explore.return_value = mock_outcome

            # Agent 1 gets conf1, agent 2 gets conf2
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                result = conf1 if call_count % 2 == 0 else conf2
                call_count += 1
                return result

            with patch.object(ProteinAgent, 'get_current_conformation', side_effect=side_effect):
                coordinator.run_parallel_exploration(1)

        # Best should be conf2 (lower energy)
        best_conf, best_energy, best_rmsd = coordinator.get_best_conformation()
        assert best_energy == 800.0
        assert best_rmsd == 4.0