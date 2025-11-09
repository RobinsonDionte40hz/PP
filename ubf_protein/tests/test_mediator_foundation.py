"""
Unit Tests for Mediator Agent Foundation

Tests MediatorAgent initialization, configuration validation, and
interface compliance (Task 5.4).

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from ubf_protein.mediator_agent import MediatorAgent
from ubf_protein.mediator_config import MediatorConfig
from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer
from ubf_protein.interfaces import IProteinAgent, IConsciousnessState, IBehavioralState, IMemorySystem


# ============================================================================
# MediatorConfig Tests
# ============================================================================

class TestMediatorConfig:
    """Test MediatorConfig dataclass validation and factory methods"""
    
    def test_default_config_initialization(self):
        """Test that default configuration initializes successfully"""
        config = MediatorConfig()
        
        assert config.thz_similarity_threshold == 0.7
        assert config.geometric_similarity_threshold == 2.0
        assert config.secondary_structure_min_length == 4
        assert config.relay_frequency == 20
        assert config.broadcast_throttle_rate == 10
        assert config.cache_size == 10000
        assert config.cache_ttl_seconds == 3600
        assert config.enable_thz_detection is True
        assert config.enable_folding_detection is True
        assert config.enable_geometric_detection is True
        assert config.max_reference_conformations == 100
        assert config.pattern_history_size == 1000
    
    def test_custom_config_initialization(self):
        """Test custom configuration with valid parameters"""
        config = MediatorConfig(
            thz_similarity_threshold=0.8,
            geometric_similarity_threshold=1.5,
            secondary_structure_min_length=5,
            relay_frequency=10,
            cache_size=5000
        )
        
        assert config.thz_similarity_threshold == 0.8
        assert config.geometric_similarity_threshold == 1.5
        assert config.secondary_structure_min_length == 5
        assert config.relay_frequency == 10
        assert config.cache_size == 5000
    
    def test_invalid_thz_threshold_too_low(self):
        """Test that thz_similarity_threshold < 0.0 raises ValueError"""
        with pytest.raises(ValueError, match="thz_similarity_threshold must be in range"):
            MediatorConfig(thz_similarity_threshold=-0.1)
    
    def test_invalid_thz_threshold_too_high(self):
        """Test that thz_similarity_threshold > 1.0 raises ValueError"""
        with pytest.raises(ValueError, match="thz_similarity_threshold must be in range"):
            MediatorConfig(thz_similarity_threshold=1.5)
    
    def test_invalid_geometric_threshold(self):
        """Test that geometric_similarity_threshold <= 0 raises ValueError"""
        with pytest.raises(ValueError, match="geometric_similarity_threshold must be positive"):
            MediatorConfig(geometric_similarity_threshold=0.0)
    
    def test_invalid_secondary_structure_min_length(self):
        """Test that secondary_structure_min_length < 1 raises ValueError"""
        with pytest.raises(ValueError, match="secondary_structure_min_length must be at least 1"):
            MediatorConfig(secondary_structure_min_length=0)
    
    def test_invalid_relay_frequency(self):
        """Test that relay_frequency < 1 raises ValueError"""
        with pytest.raises(ValueError, match="relay_frequency must be at least 1"):
            MediatorConfig(relay_frequency=0)
    
    def test_invalid_broadcast_throttle_rate(self):
        """Test that broadcast_throttle_rate < 1 raises ValueError"""
        with pytest.raises(ValueError, match="broadcast_throttle_rate must be at least 1"):
            MediatorConfig(broadcast_throttle_rate=0)
    
    def test_invalid_cache_size(self):
        """Test that cache_size < 1 raises ValueError"""
        with pytest.raises(ValueError, match="cache_size must be at least 1"):
            MediatorConfig(cache_size=0)
    
    def test_invalid_cache_ttl(self):
        """Test that cache_ttl_seconds <= 0 raises ValueError"""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be positive"):
            MediatorConfig(cache_ttl_seconds=0)
    
    def test_invalid_max_reference_conformations(self):
        """Test that max_reference_conformations < 1 raises ValueError"""
        with pytest.raises(ValueError, match="max_reference_conformations must be at least 1"):
            MediatorConfig(max_reference_conformations=0)
    
    def test_invalid_pattern_history_size(self):
        """Test that pattern_history_size < 1 raises ValueError"""
        with pytest.raises(ValueError, match="pattern_history_size must be at least 1"):
            MediatorConfig(pattern_history_size=0)
    
    def test_all_detections_disabled_raises_error(self):
        """Test that disabling all detection types raises ValueError"""
        with pytest.raises(ValueError, match="At least one detection type must be enabled"):
            MediatorConfig(
                enable_thz_detection=False,
                enable_folding_detection=False,
                enable_geometric_detection=False
            )
    
    def test_to_dict_conversion(self):
        """Test that configuration converts to dictionary correctly"""
        config = MediatorConfig()
        config_dict = config.to_dict()
        
        assert 'detection_thresholds' in config_dict
        assert 'relay_config' in config_dict
        assert 'cache_config' in config_dict
        assert 'features' in config_dict
        assert 'memory_management' in config_dict
        
        assert config_dict['detection_thresholds']['thz_similarity'] == 0.7
        assert config_dict['relay_config']['relay_frequency'] == 20
        assert config_dict['cache_config']['cache_size'] == 10000
        assert config_dict['features']['thz_detection'] is True
    
    def test_high_sensitivity_config(self):
        """Test high sensitivity configuration factory method"""
        config = MediatorConfig.get_high_sensitivity_config()
        
        assert config.thz_similarity_threshold == 0.6
        assert config.geometric_similarity_threshold == 3.0
        assert config.secondary_structure_min_length == 3
        assert config.relay_frequency == 10
        assert config.cache_size == 15000
    
    def test_high_performance_config(self):
        """Test high performance configuration factory method"""
        config = MediatorConfig.get_high_performance_config()
        
        assert config.thz_similarity_threshold == 0.8
        assert config.geometric_similarity_threshold == 1.5
        assert config.secondary_structure_min_length == 5
        assert config.relay_frequency == 50
        assert config.enable_thz_detection is False  # Disabled for performance
        assert config.cache_size == 5000
    
    def test_balanced_config(self):
        """Test balanced configuration factory method"""
        config = MediatorConfig.get_balanced_config()
        
        # Should match default values
        assert config.thz_similarity_threshold == 0.7
        assert config.relay_frequency == 20
        assert config.cache_size == 10000


# ============================================================================
# MediatorAgent Tests
# ============================================================================

class TestMediatorAgent:
    """Test MediatorAgent initialization and interface compliance"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for MediatorAgent"""
        qcpp_adapter = Mock()
        geometric_analyzer = GeometricAttractorAnalyzer()
        shared_memory = Mock()
        
        return qcpp_adapter, geometric_analyzer, shared_memory
    
    def test_mediator_agent_initialization(self, mock_dependencies):
        """Test that MediatorAgent initializes successfully with valid inputs"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGHIKL",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        assert agent.protein_sequence == "ACDEFGHIKL"
        assert agent.qcpp_adapter is qcpp_adapter
        assert agent.geometric_analyzer is geometric_analyzer
        assert agent.shared_memory is shared_memory
        assert isinstance(agent.config, MediatorConfig)
    
    def test_mediator_agent_with_custom_config(self, mock_dependencies):
        """Test MediatorAgent initialization with custom configuration"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        config = MediatorConfig(relay_frequency=10, cache_size=5000)
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        assert agent.config.relay_frequency == 10
        assert agent.config.cache_size == 5000
    
    def test_mediator_agent_empty_sequence_raises_error(self, mock_dependencies):
        """Test that empty protein_sequence raises ValueError"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        with pytest.raises(ValueError, match="protein_sequence cannot be empty"):
            MediatorAgent(
                protein_sequence="",
                qcpp_adapter=qcpp_adapter,
                geometric_analyzer=geometric_analyzer,
                shared_memory=shared_memory
            )
    
    def test_consciousness_state_initialization(self, mock_dependencies):
        """Test that consciousness state is initialized correctly"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        consciousness = agent.get_consciousness_state()
        coords = consciousness.get_coordinates()
        
        # Mediators should have frequency=9.0, coherence=0.8
        assert coords.frequency == 9.0
        assert coords.coherence == 0.8
    
    def test_behavioral_state_initialization(self, mock_dependencies):
        """Test that behavioral state is derived from consciousness"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        behavioral = agent.get_behavioral_state()
        
        # Should have behavioral methods from interface
        assert hasattr(behavioral, 'get_exploration_energy')
        assert hasattr(behavioral, 'get_structural_focus')
        assert hasattr(behavioral, 'get_hydrophobic_drive')
        
        # Should be able to call methods
        exploration_energy = behavioral.get_exploration_energy()
        assert isinstance(exploration_energy, float)
    
    def test_memory_system_initialization(self, mock_dependencies):
        """Test that memory system is initialized"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        memory_system = agent.get_memory_system()
        
        assert memory_system is not None
    
    def test_detection_statistics_initialization(self, mock_dependencies):
        """Test that detection statistics are initialized to zero"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        stats = agent.get_detection_statistics()
        
        assert stats['total_detections'] == 0
        assert stats['thz_detections'] == 0
        assert stats['folding_detections'] == 0
        assert stats['geometric_detections'] == 0
        assert stats['broadcasts'] == 0
        assert stats['qcpp_validations'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['cache_size'] == 0
        assert stats['reference_conformations'] == 0
    
    def test_pattern_cache_initialization(self, mock_dependencies):
        """Test that pattern cache is initialized empty"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        assert len(agent.pattern_cache) == 0
        assert len(agent.reference_conformations) == 0
    
    def test_iprotein_agent_interface_compliance(self, mock_dependencies):
        """Test that MediatorAgent implements IProteinAgent interface"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Check that agent implements all required methods
        assert isinstance(agent, IProteinAgent)
        assert hasattr(agent, 'get_consciousness_state')
        assert hasattr(agent, 'get_behavioral_state')
        assert hasattr(agent, 'get_memory_system')
        assert hasattr(agent, 'explore_step')
        assert hasattr(agent, 'get_current_conformation')
        assert hasattr(agent, 'get_exploration_metrics')
    
    def test_get_exploration_metrics(self, mock_dependencies):
        """Test that get_exploration_metrics returns detection statistics"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        metrics = agent.get_exploration_metrics()
        
        assert 'total_detections' in metrics
        assert 'thz_detections' in metrics
        assert 'folding_detections' in metrics
        assert 'geometric_detections' in metrics
        assert 'broadcasts' in metrics
        assert 'cache_hit_rate' in metrics
        
        # All should be floats
        for key, value in metrics.items():
            assert isinstance(value, float)
    
    def test_mediator_specific_methods_exist(self, mock_dependencies):
        """Test that Mediator-specific methods are present"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Check Mediator-specific methods exist
        assert hasattr(agent, 'detect_patterns')
        assert hasattr(agent, 'relay_to_qcpp')
        assert hasattr(agent, 'broadcast_to_agents')
        assert hasattr(agent, 'get_detection_statistics')
    
    def test_explore_step_not_implemented(self, mock_dependencies):
        """Test that explore_step raises NotImplementedError (Task 5.3)"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # explore_step should raise NotImplementedError until Task 5.3
        with pytest.raises(NotImplementedError, match="will be implemented in Task 5.3"):
            agent.explore_step()
    
    def test_get_current_conformation_not_implemented(self, mock_dependencies):
        """Test that get_current_conformation raises NotImplementedError"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Mediators don't maintain conformations
        with pytest.raises(NotImplementedError, match="don't maintain conformations"):
            agent.get_current_conformation()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
