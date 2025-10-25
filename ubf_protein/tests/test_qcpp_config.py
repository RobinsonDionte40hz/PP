"""
Unit tests for QCPP integration configuration.

Tests Task 10.1:
- Test UBF operates without QCPP when not provided
- Test configuration flag disables integration
- Test all components fall back gracefully
"""

import pytest
from ubf_protein.qcpp_config import (
    QCPPIntegrationConfig,
    get_default_config,
    get_high_performance_config,
    get_high_accuracy_config,
    get_disabled_config,
    get_testing_config,
    create_config,
    is_enabled
)
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from unittest.mock import Mock


class TestQCPPIntegrationConfig:
    """Test suite for QCPPIntegrationConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = QCPPIntegrationConfig()
        
        assert config.enabled is True
        assert config.analysis_frequency == 1
        assert config.cache_size == 1000
        assert config.max_calculation_time_ms == 5.0
        assert config.phi_reward_threshold == 0.8
        assert config.phi_reward_energy == -50.0
        assert config.enable_dynamic_adjustment is True
        assert config.stability_low_threshold == 1.0
        assert config.stability_high_threshold == 2.0
        assert config.enable_physics_grounding is True
        assert config.smoothing_factor == 0.1
        assert config.enable_trajectory_recording is True
        assert config.trajectory_max_points == 10000
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = QCPPIntegrationConfig(
            enabled=False,
            cache_size=5000,
            analysis_frequency=10
        )
        
        assert config.enabled is False
        assert config.cache_size == 5000
        assert config.analysis_frequency == 10
        # Other values should be defaults
        assert config.phi_reward_threshold == 0.8
    
    def test_config_validation_negative_frequency(self):
        """Test validation rejects negative analysis frequency."""
        with pytest.raises(ValueError, match="analysis_frequency must be >= 1"):
            QCPPIntegrationConfig(analysis_frequency=0)
    
    def test_config_validation_negative_cache(self):
        """Test validation rejects negative cache size."""
        with pytest.raises(ValueError, match="cache_size must be >= 1"):
            QCPPIntegrationConfig(cache_size=0)
    
    def test_config_validation_negative_timeout(self):
        """Test validation rejects negative timeout."""
        with pytest.raises(ValueError, match="max_calculation_time_ms must be > 0"):
            QCPPIntegrationConfig(max_calculation_time_ms=0)
    
    def test_config_validation_phi_threshold_bounds(self):
        """Test validation enforces phi threshold bounds [0, 1]."""
        # Too low
        with pytest.raises(ValueError, match="phi_reward_threshold must be in"):
            QCPPIntegrationConfig(phi_reward_threshold=-0.1)
        
        # Too high
        with pytest.raises(ValueError, match="phi_reward_threshold must be in"):
            QCPPIntegrationConfig(phi_reward_threshold=1.1)
        
        # Valid boundaries
        config = QCPPIntegrationConfig(phi_reward_threshold=0.0)
        assert config.phi_reward_threshold == 0.0
        
        config = QCPPIntegrationConfig(phi_reward_threshold=1.0)
        assert config.phi_reward_threshold == 1.0
    
    def test_config_validation_stability_thresholds(self):
        """Test validation enforces stability threshold ordering."""
        # High must be > low
        with pytest.raises(ValueError, match="stability_high_threshold.*must be >"):
            QCPPIntegrationConfig(
                stability_low_threshold=2.0,
                stability_high_threshold=1.0
            )
        
        # Valid ordering
        config = QCPPIntegrationConfig(
            stability_low_threshold=1.0,
            stability_high_threshold=3.0
        )
        assert config.stability_low_threshold == 1.0
        assert config.stability_high_threshold == 3.0
    
    def test_config_validation_smoothing_factor_bounds(self):
        """Test validation enforces smoothing factor bounds (0, 1]."""
        # Too low
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            QCPPIntegrationConfig(smoothing_factor=0.0)
        
        # Too high
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            QCPPIntegrationConfig(smoothing_factor=1.1)
        
        # Valid boundary
        config = QCPPIntegrationConfig(smoothing_factor=1.0)
        assert config.smoothing_factor == 1.0
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = QCPPIntegrationConfig(
            enabled=False,
            cache_size=2000
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['enabled'] is False
        assert config_dict['cache_size'] == 2000
        assert 'analysis_frequency' in config_dict
        assert 'enable_physics_grounding' in config_dict
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'enabled': True,
            'cache_size': 3000,
            'analysis_frequency': 5
        }
        
        config = QCPPIntegrationConfig.from_dict(config_dict)
        
        assert config.enabled is True
        assert config.cache_size == 3000
        assert config.analysis_frequency == 5
    
    def test_config_repr(self):
        """Test string representation of config."""
        config = QCPPIntegrationConfig(enabled=True)
        repr_str = repr(config)
        
        assert "ENABLED" in repr_str
        assert "freq=" in repr_str
        assert "cache=" in repr_str
        
        config_disabled = QCPPIntegrationConfig(enabled=False)
        repr_str_disabled = repr(config_disabled)
        
        assert "DISABLED" in repr_str_disabled


class TestPredefinedConfigs:
    """Test suite for predefined configuration functions."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        
        assert config.enabled is True
        assert config.analysis_frequency == 1
        assert config.cache_size == 1000
    
    def test_get_high_performance_config(self):
        """Test getting high-performance configuration."""
        config = get_high_performance_config()
        
        assert config.enabled is True
        assert config.analysis_frequency > 1  # Less frequent
        assert config.cache_size > 1000  # Larger cache
        assert config.max_calculation_time_ms < 5.0  # Stricter timeout
        assert config.enable_trajectory_recording is False  # Disabled for speed
    
    def test_get_high_accuracy_config(self):
        """Test getting high-accuracy configuration."""
        config = get_high_accuracy_config()
        
        assert config.enabled is True
        assert config.analysis_frequency == 1  # Every iteration
        assert config.cache_size > 1000  # Large cache
        assert config.max_calculation_time_ms > 5.0  # More generous
        assert config.enable_trajectory_recording is True
        assert config.log_qcpp_metrics is True  # Detailed logging
    
    def test_get_disabled_config(self):
        """Test getting disabled configuration."""
        config = get_disabled_config()
        
        assert config.enabled is False
    
    def test_get_testing_config(self):
        """Test getting testing configuration."""
        config = get_testing_config()
        
        assert config.enabled is True
        assert config.cache_size < 1000  # Smaller for tests
        assert config.trajectory_max_points < 10000  # Minimal
    
    def test_create_config_with_overrides(self):
        """Test creating custom config with overrides."""
        config = create_config(
            cache_size=2000,
            enable_dynamic_adjustment=False
        )
        
        assert config.cache_size == 2000
        assert config.enable_dynamic_adjustment is False
        # Other values should be defaults
        assert config.enabled is True
        assert config.analysis_frequency == 1
    
    def test_is_enabled_with_config(self):
        """Test is_enabled helper with various configs."""
        # Enabled config
        enabled_config = QCPPIntegrationConfig(enabled=True)
        assert is_enabled(enabled_config) is True
        
        # Disabled config
        disabled_config = QCPPIntegrationConfig(enabled=False)
        assert is_enabled(disabled_config) is False
        
        # None config (no QCPP)
        assert is_enabled(None) is False


class TestBackwardCompatibility:
    """Test suite for backward compatibility without QCPP."""
    
    def test_coordinator_without_qcpp(self):
        """Test coordinator works without QCPP integration."""
        # Old-style usage (no QCPP)
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False
        )
        
        # Should initialize successfully
        assert coordinator._protein_sequence == "ACDEFGH"
        assert coordinator._qcpp_integration is None
        assert coordinator._trajectory_recorder is None
        
        # Should be able to run exploration
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        results = coordinator.run_parallel_exploration(iterations=5)
        
        # Should complete successfully
        assert results.total_iterations == 5
        assert results.qcpp_trajectory_data is None
    
    def test_coordinator_with_disabled_config(self):
        """Test coordinator with disabled QCPP config behaves like no QCPP."""
        # Create disabled config
        config = get_disabled_config()
        assert config.enabled is False
        
        # In practice, disabled config means not passing QCPP at all
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=None  # Equivalent to disabled
        )
        
        assert coordinator._qcpp_integration is None
        assert coordinator._trajectory_recorder is None
    
    def test_multiple_coordinators_mixed_configs(self):
        """Test multiple coordinators with different QCPP configurations."""
        # Coordinator without QCPP
        coord1 = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False
        )
        
        # Coordinator with QCPP (mocked)
        mock_qcpp = Mock()
        coord2 = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False,
            qcpp_integration=mock_qcpp
        )
        
        # Both should work independently
        assert coord1._qcpp_integration is None
        assert coord2._qcpp_integration is mock_qcpp
    
    def test_config_does_not_affect_core_functionality(self):
        """Test that QCPP config doesn't break core UBF functionality."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=False
        )
        
        # Core functionality should work
        agents = coordinator.initialize_agents(count=3, diversity_profile="balanced")
        assert len(agents) == 3
        
        # Should be able to get shared memory pool
        pool = coordinator.get_shared_memory_pool()
        assert pool is not None
        
        # Should be able to run exploration
        results = coordinator.run_parallel_exploration(iterations=10)
        assert results.total_iterations == 10
        assert results.total_conformations_explored > 0


class TestConfigurationValidation:
    """Test suite for configuration validation edge cases."""
    
    def test_all_validation_rules(self):
        """Test that all validation rules are enforced."""
        # Valid config should pass
        config = QCPPIntegrationConfig()
        config.validate()  # Should not raise
        
        # Invalid configs should fail
        invalid_configs = [
            {'analysis_frequency': 0},
            {'cache_size': 0},
            {'max_calculation_time_ms': 0},
            {'phi_reward_threshold': -0.1},
            {'phi_reward_threshold': 1.1},
            {'smoothing_factor': 0.0},
            {'smoothing_factor': 1.1},
            {'trajectory_max_points': 0},
            {'stability_low_threshold': 2.0, 'stability_high_threshold': 1.0}
        ]
        
        for invalid_params in invalid_configs:
            with pytest.raises(ValueError):
                QCPPIntegrationConfig(**invalid_params)
    
    def test_config_mutation_after_creation(self):
        """Test that config can be mutated after creation (not frozen)."""
        config = QCPPIntegrationConfig()
        
        # Should be able to modify
        config.cache_size = 2000
        assert config.cache_size == 2000
        
        # But validation should still be possible
        config.validate()  # Should pass
        
        # Invalid mutation should be caught by validation
        config.analysis_frequency = -1
        with pytest.raises(ValueError):
            config.validate()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
