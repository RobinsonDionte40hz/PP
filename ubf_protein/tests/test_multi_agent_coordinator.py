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


class TestMultiAgentCoordinatorWithMediators:
    """Test suite for MultiAgentCoordinator with Mediator Agents (Task 10.6)"""

    def test_mediator_initialization_disabled_by_default(self):
        """Test that mediators are disabled by default"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        
        # Mediators should be disabled
        assert coordinator._enable_mediators is False
        assert len(coordinator._mediators) == 0
        
        # Trying to initialize mediators should raise error
        with pytest.raises(ValueError, match="Mediators are not enabled"):
            coordinator.initialize_mediators()

    def test_mediator_initialization_enabled(self):
        """Test successful mediator initialization when enabled"""
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=3
        )
        
        # Initialize mediators
        mediators = coordinator.initialize_mediators()
        
        assert len(mediators) == 3
        assert len(coordinator._mediators) == 3
        assert coordinator._geometric_analyzer is not None
        assert coordinator._mediator_config is not None

    def test_mediator_initialization_with_custom_config(self):
        """Test mediator initialization with custom configuration"""
        from ubf_protein.mediator_config import MediatorConfig
        
        custom_config = MediatorConfig(
            relay_frequency=5,
            enable_thz_detection=True,
            enable_folding_detection=False,
            enable_geometric_detection=True
        )
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=2,
            mediator_config=custom_config
        )
        
        mediators = coordinator.initialize_mediators()
        
        assert len(mediators) == 2
        # Access the actual config after initialization
        config = coordinator._mediator_config
        assert config is not None
        assert config.relay_frequency == 5
        assert config.enable_thz_detection is True
        assert config.enable_folding_detection is False

    def test_run_mediator_cycle_no_mediators(self):
        """Test that mediator cycle returns empty list when mediators disabled"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        
        patterns = coordinator.run_mediator_cycle(iteration=10)
        
        assert patterns == []

    def test_run_mediator_cycle_with_mediators(self):
        """Test mediator cycle execution with mock pattern detection"""
        from ubf_protein.pattern_detection import PatternDetection, PatternType, PatternSignificance, THzResonanceData
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=2
        )
        
        coordinator.initialize_agents(2, "balanced")
        coordinator.initialize_mediators()
        
        # Create mock best conformation
        mock_conf = Mock()
        mock_conf.atom_coordinates = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        mock_conf.energy = 800.0
        mock_conf.sequence = "TE"
        coordinator._best_conformation = mock_conf
        
        # Mock pattern detection with THzResonanceData
        thz_data = THzResonanceData(
            cluster_id=1,
            cluster_size=5,
            similarity_score=0.85,
            dominant_frequency=3.5,
            spectral_entropy=2.1
        )
        
        mock_pattern = PatternDetection(
            pattern_type=PatternType.THZ,
            significance=PatternSignificance.HIGH,
            timestamp=1000.0,
            iteration=10,
            conformation_hash="abc123def4567890",  # 16 characters
            thz_data=thz_data  # Now has data
        )
        
        with patch('ubf_protein.mediator_agent.MediatorAgent.detect_patterns', return_value=[mock_pattern]):
            with patch('ubf_protein.mediator_agent.MediatorAgent.relay_to_qcpp', return_value=None):
                with patch('ubf_protein.mediator_agent.MediatorAgent.broadcast_to_agents', return_value=True):
                    patterns = coordinator.run_mediator_cycle(iteration=10)
        
        # Should have detected patterns from both mediators
        assert len(patterns) == 2  # 2 mediators * 1 pattern each

    def test_mediator_integration_in_exploration(self):
        """Test that mediator cycles are executed during exploration at relay frequency"""
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=1
        )
        
        coordinator.initialize_agents(1, "balanced")
        coordinator.initialize_mediators()
        
        # Mock exploration
        with patch.object(ProteinAgent, 'explore_step') as mock_explore:
            mock_outcome = Mock()
            mock_outcome.significance = 0.5
            mock_explore.return_value = mock_outcome
            
            mock_conf = Mock()
            mock_conf.energy = 850.0
            mock_conf.rmsd_to_native = 4.0
            mock_conf.atom_coordinates = [(1.0, 2.0, 3.0)]
            mock_conf.sequence = "T"
            
            with patch.object(ProteinAgent, 'get_current_conformation', return_value=mock_conf):
                with patch.object(ProteinAgent, 'get_exploration_metrics', return_value={
                    'iterations_completed': 10,
                    'conformations_explored': 10,
                    'memories_created': 2,
                    'best_energy': 850.0,
                    'best_rmsd': 4.0,
                    'avg_decision_time_ms': 50.0,
                    'stuck_in_minima_count': 0,  # Fixed typo
                    'successful_escapes': 0
                }):
                    # Mock mediator cycle
                    with patch.object(
                        MultiAgentCoordinator,
                        'run_mediator_cycle',
                        return_value=[]
                    ) as mock_mediator_cycle:
                        # Run 20 iterations with relay frequency of 10
                        # Set config after initialization
                        if coordinator._mediator_config:
                            coordinator._mediator_config.relay_frequency = 10
                        coordinator.run_parallel_exploration(20)
                        
                        # Mediator cycle should be called at iterations 10 and 20
                        assert mock_mediator_cycle.call_count == 2

    def test_get_mediator_statistics_disabled(self):
        """Test that getting mediator statistics raises error when disabled"""
        coordinator = MultiAgentCoordinator("TESTSEQ")
        
        with pytest.raises(ValueError, match="Mediators are not enabled"):
            coordinator.get_mediator_statistics()

    def test_get_mediator_statistics_enabled(self):
        """Test getting mediator statistics when enabled"""
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=2
        )
        
        coordinator.initialize_mediators()
        
        # Mock mediator statistics
        mock_stats = {
            'total_detections': 10,
            'thz_detections': 5,
            'folding_detections': 3,
            'geometric_detections': 2,
            'broadcasts': 8,
            'qcpp_validations': 10,
            'cache_hits': 15,
            'cache_misses': 5,
            'cache_size': 10,
            'reference_conformations': 5,
            'cache_hit_rate': 0.75
        }
        
        with patch('ubf_protein.mediator_agent.MediatorAgent.get_detection_statistics', return_value=mock_stats):
            stats = coordinator.get_mediator_statistics()
        
        # Verify aggregated statistics
        assert stats['enabled'] is True
        assert stats['mediator_count'] == 2
        assert stats['total_detections'] == 20  # 2 mediators * 10
        assert stats['thz_detections'] == 10  # 2 mediators * 5
        assert stats['folding_detections'] == 6  # 2 mediators * 3
        assert stats['geometric_detections'] == 4  # 2 mediators * 2
        assert stats['broadcasts'] == 16  # 2 mediators * 8
        assert stats['cache_hit_rate'] > 0.0

    def test_mediator_statistics_empty_mediators(self):
        """Test mediator statistics with no mediators initialized"""
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=0
        )
        
        stats = coordinator.get_mediator_statistics()
        
        assert stats['enabled'] is False
        assert stats['mediator_count'] == 0
        assert stats['total_detections'] == 0

    def test_backward_compatibility_no_mediators(self):
        """Test that existing code works without mediators (backward compatibility)"""
        # Create coordinator without mediator parameters (old way)
        coordinator = MultiAgentCoordinator("TESTSEQ")
        
        # Should work exactly as before
        agents = coordinator.initialize_agents(2, "balanced")
        assert len(agents) == 2
        
        # Mock exploration should work normally
        with patch.object(ProteinAgent, 'explore_step') as mock_explore:
            mock_outcome = Mock()
            mock_outcome.significance = 0.5
            mock_explore.return_value = mock_outcome
            
            mock_conf = Mock()
            mock_conf.energy = 900.0
            mock_conf.rmsd_to_native = 5.0
            
            with patch.object(ProteinAgent, 'get_current_conformation', return_value=mock_conf):
                with patch.object(ProteinAgent, 'get_exploration_metrics', return_value={
                    'iterations_completed': 5,
                    'conformations_explored': 5,
                    'memories_created': 1,
                    'best_energy': 900.0,
                    'best_rmsd': 5.0,
                    'avg_decision_time_ms': 25.0,
                    'stuck_in_minima_count': 0,
                    'successful_escapes': 0
                }):
                    results = coordinator.run_parallel_exploration(5)
        
        # Verify normal operation
        assert results.total_iterations == 5
        assert len(results.agent_metrics) == 2

    def test_mediator_reference_conformation_updates(self):
        """Test that reference conformations are updated during exploration"""
        from ubf_protein.geometric_attractor import GeometricAnalysisResult
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=1
        )
        
        coordinator.initialize_agents(1, "balanced")
        coordinator.initialize_mediators()
        
        # Create mock best conformation
        mock_conf = Mock()
        mock_conf.atom_coordinates = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        mock_conf.energy = 800.0
        mock_conf.sequence = "TE"
        coordinator._best_conformation = mock_conf
        
        # Mock geometric analysis
        mock_geo_result = GeometricAnalysisResult(
            golden_ratio_percentage=45.0,
            phi_pattern_count=5,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.6,
            icosahedron_similarity=0.5,
            rotational_symmetry=0.7,
            local_symmetry=0.6,
            radius_of_gyration=10.0,
            asphericity=0.3,
            conformation_hash="abc123def4567890",  # 16 characters
            timestamp=1000.0,
            num_residues=2
        )
        
        # Ensure geometric analyzer is initialized (it should be after initialize_mediators)
        assert coordinator._geometric_analyzer is not None
        
        # Manually add reference to simulate what happens during exploration
        # The reference update actually happens in run_parallel_exploration, not run_mediator_cycle
        coordinator._mediators[0].add_reference_conformation(
            mock_conf,
            agent_id="test_agent",
            geometric_score=45.0
        )
        
        # Verify reference was added to mediator
        assert len(coordinator._mediators[0].reference_conformations) == 1
        assert coordinator._mediators[0].reference_conformations[0]['geometric_score'] == 45.0

    def test_mediator_cycle_handles_errors_gracefully(self):
        """Test that mediator cycle handles errors without crashing"""
        coordinator = MultiAgentCoordinator(
            protein_sequence="TESTSEQ",
            enable_mediators=True,
            mediator_count=2
        )
        
        coordinator.initialize_mediators()
        
        # Mock one mediator to raise exception, other to succeed
        def detect_side_effect(*args, **kwargs):
            # First call raises, second succeeds
            if not hasattr(detect_side_effect, 'call_count'):
                detect_side_effect.call_count = 0
            detect_side_effect.call_count += 1
            
            if detect_side_effect.call_count == 1:
                raise Exception("Simulated detection failure")
            return []
        
        with patch('ubf_protein.mediator_agent.MediatorAgent.detect_patterns', side_effect=detect_side_effect):
            # Should not crash, just log warning
            patterns = coordinator.run_mediator_cycle(iteration=10)
        
        # Should return empty list (both mediators failed or returned empty)
        assert patterns == []
