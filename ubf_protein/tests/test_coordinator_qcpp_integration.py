"""
Unit tests for QCPP integration in MultiAgentCoordinator.

Tests Task 7.1:
- Test coordinator initialization with QCPP integration
- Test coordinator passes QCPP to agents
- Test coordinator stores QCPP reference
"""

import pytest
from unittest.mock import Mock, MagicMock
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
from ubf_protein.models import Conformation


class TestCoordinatorQCPPIntegration:
    """Test suite for QCPP integration in MultiAgentCoordinator."""
    
    def test_coordinator_initialization_without_qcpp(self):
        """Test coordinator can be initialized without QCPP (backward compatibility)."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False
        )
        
        # Should initialize successfully
        assert coordinator._protein_sequence == "ACDEFGH"
        
        # QCPP integration should be None
        assert coordinator._qcpp_integration is None
        assert coordinator.get_qcpp_integration() is None
    
    def test_coordinator_initialization_with_qcpp(self):
        """Test coordinator initialization with QCPP integration."""
        # Create mock QCPP integration
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Should initialize successfully
        assert coordinator._protein_sequence == "ACDEFGH"
        
        # QCPP integration should be stored
        assert coordinator._qcpp_integration is mock_qcpp
        assert coordinator.get_qcpp_integration() is mock_qcpp
    
    def test_coordinator_stores_qcpp_reference(self):
        """Test coordinator stores and provides access to QCPP integration reference."""
        # Create mock QCPP integration
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        mock_qcpp.cache_size = 1000
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Should store reference internally
        assert coordinator._qcpp_integration is mock_qcpp
        
        # Should provide access via getter
        retrieved_qcpp = coordinator.get_qcpp_integration()
        assert retrieved_qcpp is mock_qcpp
        assert retrieved_qcpp.cache_size == 1000  # type: ignore
    
    def test_coordinator_passes_qcpp_to_agents(self):
        """Test coordinator passes QCPP integration to agents during initialization."""
        # Create mock QCPP integration with necessary methods
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        mock_qcpp.analyze_conformation = Mock(return_value=QCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.5,
            stability_score=1.5,
            phi_match_score=0.7,
            calculation_time_ms=2.0
        ))
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Initialize agents
        agents = coordinator.initialize_agents(count=3, diversity_profile="balanced")
        
        # All agents should receive QCPP integration
        assert len(agents) == 3
        for agent in agents:
            # Check agent has QCPP integration reference
            assert agent._qcpp_integration is mock_qcpp  # type: ignore
    
    def test_coordinator_agents_without_qcpp(self):
        """Test agents are created without QCPP when coordinator has no QCPP."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=None
        )
        
        # Initialize agents
        agents = coordinator.initialize_agents(count=3, diversity_profile="balanced")
        
        # All agents should have None for QCPP integration
        assert len(agents) == 3
        for agent in agents:
            assert agent._qcpp_integration is None  # type: ignore
    
    def test_coordinator_diversity_profiles_with_qcpp(self):
        """Test coordinator creates diverse agents with QCPP integration."""
        # Create mock QCPP integration
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Initialize agents with balanced diversity
        agents = coordinator.initialize_agents(count=9, diversity_profile="balanced")
        
        # Should create 9 agents
        assert len(agents) == 9
        
        # All should have QCPP integration
        for agent in agents:
            assert agent._qcpp_integration is mock_qcpp  # type: ignore
        
        # Should have diverse consciousness coordinates
        # Use getter methods instead of direct attribute access
        frequencies = [agent._consciousness.get_frequency() for agent in agents]  # type: ignore
        coherences = [agent._consciousness.get_coherence() for agent in agents]  # type: ignore
        
        # Check we have variation (not all identical)
        assert len(set(frequencies)) > 1
        assert len(set(coherences)) > 1
    
    def test_coordinator_qcpp_integration_type_checking(self):
        """Test coordinator accepts any QCPP-like object (duck typing)."""
        # Create a minimal mock that quacks like QCPP
        class MinimalQCPP:
            def analyze_conformation(self, conf):
                return QCPPMetrics(
                    qcp_score=5.0,
                    field_coherence=0.5,
                    stability_score=1.5,
                    phi_match_score=0.7,
                    calculation_time_ms=1.0
                )
        
        mock_qcpp = MinimalQCPP()
        
        # Should accept any object (duck typing)
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        assert coordinator._qcpp_integration is mock_qcpp
    
    def test_coordinator_single_agent_with_qcpp(self):
        """Test coordinator creates single agent with QCPP integration."""
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Initialize single agent
        agents = coordinator.initialize_agents(count=1, diversity_profile="balanced")
        
        assert len(agents) == 1
        assert agents[0]._qcpp_integration is mock_qcpp  # type: ignore
    
    def test_coordinator_many_agents_with_qcpp(self):
        """Test coordinator creates many agents with QCPP integration."""
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Initialize many agents
        agents = coordinator.initialize_agents(count=100, diversity_profile="balanced")
        
        assert len(agents) == 100
        
        # All should have QCPP integration reference
        for agent in agents:
            assert agent._qcpp_integration is mock_qcpp  # type: ignore
    
    def test_coordinator_qcpp_persistence_across_operations(self):
        """Test QCPP integration reference persists across coordinator operations."""
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Initialize agents
        coordinator.initialize_agents(count=3, diversity_profile="balanced")
        
        # QCPP reference should persist
        assert coordinator.get_qcpp_integration() is mock_qcpp
        
        # Get agents
        agents = coordinator.get_agents()
        assert len(agents) == 3
        
        # QCPP reference should still be available
        assert coordinator.get_qcpp_integration() is mock_qcpp
        
        # Check agents still have reference
        for agent in agents:
            assert agent._qcpp_integration is mock_qcpp  # type: ignore
    
    def test_coordinator_mixed_profiles_with_qcpp(self):
        """Test coordinator with different diversity profiles all receive QCPP."""
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        
        for profile in ["cautious", "balanced", "aggressive"]:
            coordinator = MultiAgentCoordinator(
                protein_sequence="ACDEFGH",
                enable_checkpointing=False,
                qcpp_integration=mock_qcpp
            )
            
            agents = coordinator.initialize_agents(count=5, diversity_profile=profile)
            
            # All agents should have QCPP regardless of profile
            assert len(agents) == 5
            for agent in agents:
                assert agent._qcpp_integration is mock_qcpp  # type: ignore
    
    def test_coordinator_qcpp_with_adaptive_config(self):
        """Test QCPP integration works with adaptive configuration."""
        from ubf_protein.adaptive_config import AdaptiveConfigurator
        
        mock_qcpp = Mock(spec=QCPPIntegrationAdapter)
        configurator = AdaptiveConfigurator()
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            adaptive_configurator=configurator,
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        agents = coordinator.initialize_agents(count=3, diversity_profile="balanced")
        
        # Should have QCPP and adaptive config
        assert coordinator._qcpp_integration is mock_qcpp
        assert coordinator._adaptive_config is not None
        
        # Agents should have both
        for agent in agents:
            assert agent._qcpp_integration is mock_qcpp  # type: ignore
            assert agent._adaptive_config is not None  # type: ignore


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
