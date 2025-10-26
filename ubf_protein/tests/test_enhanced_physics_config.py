"""
Unit tests for enhanced physics configuration system.

Tests configuration dataclass, factory methods, environment variable parsing,
validation, and integration with MultiAgentCoordinator.
"""

import pytest
import os
from dataclasses import replace
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.models import DisulfideBond


class TestConfigInitialization:
    """Tests for configuration initialization and defaults."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = EnhancedPhysicsConfig()
        
        # Feature toggles default to False/True
        assert config.use_enhanced_energy is False
        assert config.enable_side_chains is True
        assert config.enable_solvent is True
        assert config.enable_entropic is True
        assert config.enable_refinement is False
        
        # Empty disulfide bonds
        assert config.disulfide_bonds == []
        
        # Default parameters
        assert config.sidechain_cutoff == 15.0
        assert config.solvent_screening_length == 3.0
        assert config.entropic_temperature == 300.0
        assert config.refinement_max_iterations == 100
        assert config.disulfide_spring_constant == 50.0
        assert config.stuck_window == 30
        assert config.max_iterations == 2000
    
    def test_custom_configuration(self):
        """Test custom parameter values."""
        bonds = [DisulfideBond(6, 22, 3.8)]
        
        config = EnhancedPhysicsConfig(
            use_enhanced_energy=True,
            enable_refinement=True,
            disulfide_bonds=bonds,
            sidechain_cutoff=20.0,
            entropic_temperature=310.0,
            max_iterations=5000
        )
        
        assert config.use_enhanced_energy is True
        assert config.enable_refinement is True
        assert config.disulfide_bonds is not None
        assert len(config.disulfide_bonds) == 1
        assert config.sidechain_cutoff == 20.0
        assert config.entropic_temperature == 310.0
        assert config.max_iterations == 5000
    
    def test_immutable_configuration(self):
        """Test that configuration is immutable."""
        config = EnhancedPhysicsConfig()
        
        with pytest.raises(AttributeError):
            config.use_enhanced_energy = True  # type: ignore
    
    def test_none_disulfide_bonds_becomes_empty_list(self):
        """Test that None disulfide_bonds is converted to empty list."""
        config = EnhancedPhysicsConfig(disulfide_bonds=None)
        assert config.disulfide_bonds == []


class TestFactoryMethods:
    """Tests for configuration factory methods."""
    
    def test_baseline_configuration(self):
        """Test baseline configuration factory."""
        config = EnhancedPhysicsConfig.baseline()
        
        assert config.use_enhanced_energy is False
        assert config.enable_side_chains is False
        assert config.enable_solvent is False
        assert config.enable_entropic is False
        assert config.enable_refinement is False
        assert config.disulfide_bonds == []
    
    def test_enhanced_default_configuration(self):
        """Test enhanced default configuration factory."""
        config = EnhancedPhysicsConfig.enhanced_default()
        
        assert config.use_enhanced_energy is True
        assert config.enable_side_chains is True
        assert config.enable_solvent is True
        assert config.enable_entropic is True
        assert config.enable_refinement is False  # Refinement optional
        assert config.disulfide_bonds == []
    
    def test_enhanced_default_with_disulfides(self):
        """Test enhanced default with disulfide bonds."""
        bonds = [DisulfideBond(6, 22, 3.8), DisulfideBond(11, 40, 3.8)]
        config = EnhancedPhysicsConfig.enhanced_default(bonds)
        
        assert config.use_enhanced_energy is True
        assert config.disulfide_bonds is not None
        assert len(config.disulfide_bonds) == 2
    
    def test_small_protein_configuration(self):
        """Test small protein configuration factory."""
        config = EnhancedPhysicsConfig.for_small_protein(30)
        
        assert config.use_enhanced_energy is True
        assert config.stuck_window == 20
        assert config.stuck_threshold == 5.0
        assert config.max_iterations == 1000
        assert config.refinement_max_iterations == 50
    
    def test_medium_protein_configuration(self):
        """Test medium protein configuration factory."""
        config = EnhancedPhysicsConfig.for_medium_protein(100)
        
        assert config.use_enhanced_energy is True
        assert config.stuck_window == 30
        assert config.stuck_threshold == 10.0
        assert config.max_iterations == 2000
        assert config.refinement_max_iterations == 100
    
    def test_large_protein_configuration(self):
        """Test large protein configuration factory."""
        config = EnhancedPhysicsConfig.for_large_protein(200)
        
        assert config.use_enhanced_energy is True
        assert config.stuck_window == 40
        assert config.stuck_threshold == 15.0
        assert config.max_iterations == 5000
        assert config.refinement_max_iterations == 150
    
    def test_auto_adapt_small(self):
        """Test auto-adapt for small protein."""
        config = EnhancedPhysicsConfig.auto_adapt(30)
        
        assert config.stuck_window == 20
        assert config.max_iterations == 1000
    
    def test_auto_adapt_medium(self):
        """Test auto-adapt for medium protein."""
        config = EnhancedPhysicsConfig.auto_adapt(100)
        
        assert config.stuck_window == 30
        assert config.max_iterations == 2000
    
    def test_auto_adapt_large(self):
        """Test auto-adapt for large protein."""
        config = EnhancedPhysicsConfig.auto_adapt(200)
        
        assert config.stuck_window == 40
        assert config.max_iterations == 5000
    
    def test_auto_adapt_with_disulfides(self):
        """Test auto-adapt with disulfide bonds."""
        bonds = [DisulfideBond(6, 22, 3.8)]
        config = EnhancedPhysicsConfig.auto_adapt(100, bonds)
        
        assert config.disulfide_bonds is not None
        assert len(config.disulfide_bonds) == 1
        assert config.stuck_window == 30


class TestEnvironmentVariables:
    """Tests for environment variable parsing."""
    
    def test_from_environment_default(self):
        """Test from_environment with no env vars set."""
        # Clear any existing env vars
        for key in ['UBF_ENHANCED', 'UBF_SIDECHAINS', 'UBF_SOLVENT', 'UBF_ENTROPIC', 
                    'UBF_REFINEMENT', 'UBF_SIDECHAIN_CUTOFF', 'UBF_MAX_ITERATIONS']:
            os.environ.pop(key, None)
        
        config = EnhancedPhysicsConfig.from_environment()
        
        assert config.use_enhanced_energy is False
        assert config.enable_side_chains is True
        assert config.sidechain_cutoff == 15.0
        assert config.max_iterations == 2000
    
    def test_from_environment_boolean_parsing(self):
        """Test parsing boolean environment variables."""
        test_cases = [
            ('true', True), ('True', True), ('TRUE', True),
            ('1', True), ('yes', True), ('on', True),
            ('false', False), ('False', False), ('0', False),
            ('no', False), ('off', False)
        ]
        
        for value, expected in test_cases:
            os.environ['UBF_ENHANCED'] = value
            config = EnhancedPhysicsConfig.from_environment()
            assert config.use_enhanced_energy == expected, f"Failed for '{value}'"
        
        os.environ.pop('UBF_ENHANCED', None)
    
    def test_from_environment_all_features_enabled(self):
        """Test enabling all features via environment."""
        os.environ.update({
            'UBF_ENHANCED': 'true',
            'UBF_SIDECHAINS': 'true',
            'UBF_SOLVENT': 'true',
            'UBF_ENTROPIC': 'true',
            'UBF_REFINEMENT': 'true'
        })
        
        config = EnhancedPhysicsConfig.from_environment()
        
        assert config.use_enhanced_energy is True
        assert config.enable_side_chains is True
        assert config.enable_solvent is True
        assert config.enable_entropic is True
        assert config.enable_refinement is True
        
        # Cleanup
        for key in ['UBF_ENHANCED', 'UBF_SIDECHAINS', 'UBF_SOLVENT', 'UBF_ENTROPIC', 'UBF_REFINEMENT']:
            os.environ.pop(key, None)
    
    def test_from_environment_numeric_parameters(self):
        """Test parsing numeric environment variables."""
        os.environ.update({
            'UBF_SIDECHAIN_CUTOFF': '20.0',
            'UBF_SOLVENT_SCREENING': '4.0',
            'UBF_ENTROPIC_TEMP': '310.0',
            'UBF_REFINEMENT_ITERATIONS': '150',
            'UBF_STUCK_WINDOW': '40',
            'UBF_STUCK_THRESHOLD': '12.5',
            'UBF_MAX_ITERATIONS': '3000'
        })
        
        config = EnhancedPhysicsConfig.from_environment()
        
        assert config.sidechain_cutoff == 20.0
        assert config.solvent_screening_length == 4.0
        assert config.entropic_temperature == 310.0
        assert config.refinement_max_iterations == 150
        assert config.stuck_window == 40
        assert config.stuck_threshold == 12.5
        assert config.max_iterations == 3000
        
        # Cleanup
        for key in ['UBF_SIDECHAIN_CUTOFF', 'UBF_SOLVENT_SCREENING', 'UBF_ENTROPIC_TEMP',
                    'UBF_REFINEMENT_ITERATIONS', 'UBF_STUCK_WINDOW', 'UBF_STUCK_THRESHOLD', 'UBF_MAX_ITERATIONS']:
            os.environ.pop(key, None)
    
    def test_from_environment_with_disulfides(self):
        """Test from_environment with disulfide bonds parameter."""
        bonds = [DisulfideBond(6, 22, 3.8)]
        config = EnhancedPhysicsConfig.from_environment(bonds)
        
        assert config.disulfide_bonds is not None
        assert len(config.disulfide_bonds) == 1


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_valid_configuration(self):
        """Test that valid configuration passes validation."""
        config = EnhancedPhysicsConfig.enhanced_default()
        config.validate()  # Should not raise
    
    def test_invalid_sidechain_cutoff(self):
        """Test validation fails for invalid sidechain cutoff."""
        config = EnhancedPhysicsConfig(sidechain_cutoff=-5.0)
        
        with pytest.raises(ValueError, match="sidechain_cutoff must be positive"):
            config.validate()
    
    def test_invalid_solvent_screening(self):
        """Test validation fails for invalid solvent screening length."""
        config = EnhancedPhysicsConfig(solvent_screening_length=0.0)
        
        with pytest.raises(ValueError, match="solvent_screening_length must be positive"):
            config.validate()
    
    def test_invalid_dielectric_constants(self):
        """Test validation fails for invalid dielectric constants."""
        # Buried dielectric < 1
        config1 = EnhancedPhysicsConfig(solvent_buried_dielectric=0.5)
        with pytest.raises(ValueError, match="solvent_buried_dielectric must be >= 1"):
            config1.validate()
        
        # Surface < buried
        config2 = EnhancedPhysicsConfig(
            solvent_buried_dielectric=80.0,
            solvent_surface_dielectric=40.0
        )
        with pytest.raises(ValueError, match="solvent_surface_dielectric.*must be >= solvent_buried_dielectric"):
            config2.validate()
    
    def test_invalid_entropic_parameters(self):
        """Test validation fails for invalid entropic parameters."""
        # Temperature <= 0
        config1 = EnhancedPhysicsConfig(entropic_temperature=-100.0)
        with pytest.raises(ValueError, match="entropic_temperature must be positive"):
            config1.validate()
        
        # Window size < 2
        config2 = EnhancedPhysicsConfig(entropic_window_size=1)
        with pytest.raises(ValueError, match="entropic_window_size must be >= 2"):
            config2.validate()
    
    def test_invalid_refinement_parameters(self):
        """Test validation fails for invalid refinement parameters."""
        # Max iterations < 1
        config1 = EnhancedPhysicsConfig(refinement_max_iterations=0)
        with pytest.raises(ValueError, match="refinement_max_iterations must be >= 1"):
            config1.validate()
        
        # Step size <= 0
        config2 = EnhancedPhysicsConfig(refinement_step_size=0.0)
        with pytest.raises(ValueError, match="refinement_step_size must be positive"):
            config2.validate()
        
        # Epsilon <= 0
        config3 = EnhancedPhysicsConfig(refinement_epsilon=-0.01)
        with pytest.raises(ValueError, match="refinement_epsilon must be positive"):
            config3.validate()
        
        # Reduction factor not in (0, 1)
        config4 = EnhancedPhysicsConfig(refinement_reduction_factor=1.5)
        with pytest.raises(ValueError, match="refinement_reduction_factor must be in"):
            config4.validate()
    
    def test_invalid_disulfide_parameters(self):
        """Test validation fails for invalid disulfide parameters."""
        # Spring constant <= 0
        config1 = EnhancedPhysicsConfig(disulfide_spring_constant=0.0)
        with pytest.raises(ValueError, match="disulfide_spring_constant must be positive"):
            config1.validate()
        
        # Target distance <= 0
        config2 = EnhancedPhysicsConfig(disulfide_target_distance=-3.8)
        with pytest.raises(ValueError, match="disulfide_target_distance must be positive"):
            config2.validate()
    
    def test_invalid_adaptive_parameters(self):
        """Test validation fails for invalid adaptive parameters."""
        # Stuck window < 1
        config1 = EnhancedPhysicsConfig(stuck_window=0)
        with pytest.raises(ValueError, match="stuck_window must be >= 1"):
            config1.validate()
        
        # Stuck threshold < 0
        config2 = EnhancedPhysicsConfig(stuck_threshold=-5.0)
        with pytest.raises(ValueError, match="stuck_threshold must be non-negative"):
            config2.validate()
        
        # Max iterations < 1
        config3 = EnhancedPhysicsConfig(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            config3.validate()


class TestConfigModification:
    """Tests for configuration modification methods."""
    
    def test_with_refinement_enable(self):
        """Test enabling refinement."""
        config = EnhancedPhysicsConfig.baseline()
        assert config.enable_refinement is False
        
        new_config = config.with_refinement(True)
        assert new_config.enable_refinement is True
        assert config.enable_refinement is False  # Original unchanged
    
    def test_with_refinement_custom_iterations(self):
        """Test enabling refinement with custom iterations."""
        config = EnhancedPhysicsConfig()
        new_config = config.with_refinement(True, max_iterations=200)
        
        assert new_config.enable_refinement is True
        assert new_config.refinement_max_iterations == 200
    
    def test_with_disulfide_bonds(self):
        """Test updating disulfide bonds."""
        config = EnhancedPhysicsConfig()
        bonds = [DisulfideBond(6, 22, 3.8), DisulfideBond(11, 40, 3.8)]
        
        new_config = config.with_disulfide_bonds(bonds)
        
        assert new_config.disulfide_bonds is not None
        assert len(new_config.disulfide_bonds) == 2
        assert config.disulfide_bonds is not None
        assert len(config.disulfide_bonds) == 0  # Original unchanged
    
    def test_with_custom_parameters(self):
        """Test updating multiple custom parameters."""
        config = EnhancedPhysicsConfig()
        
        new_config = config.with_custom_parameters(
            sidechain_cutoff=20.0,
            entropic_temperature=310.0,
            max_iterations=3000
        )
        
        assert new_config.sidechain_cutoff == 20.0
        assert new_config.entropic_temperature == 310.0
        assert new_config.max_iterations == 3000
        assert config.sidechain_cutoff == 15.0  # Original unchanged


class TestSerialization:
    """Tests for configuration serialization."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        bonds = [DisulfideBond(6, 22, 3.8)]
        config = EnhancedPhysicsConfig(
            use_enhanced_energy=True,
            disulfide_bonds=bonds,
            max_iterations=5000
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['use_enhanced_energy'] is True
        assert config_dict['num_disulfide_bonds'] == 1
        assert config_dict['max_iterations'] == 5000
        assert 'sidechain_cutoff' in config_dict
    
    def test_summary(self):
        """Test human-readable summary."""
        bonds = [DisulfideBond(6, 22, 3.8), DisulfideBond(11, 40, 3.8)]
        config = EnhancedPhysicsConfig.enhanced_default(bonds).with_refinement(True)
        
        summary = config.summary()
        
        assert isinstance(summary, str)
        assert 'Enhanced Physics Configuration' in summary
        assert 'Enhanced Energy: True' in summary
        assert 'Disulfide Bonds: 2' in summary
        assert 'Max Iterations' in summary
        assert 'Refinement Parameters' in summary


class TestIntegrationWithCoordinator:
    """Tests for integration with MultiAgentCoordinator."""
    
    def test_coordinator_with_config_object(self):
        """Test creating coordinator with config object."""
        from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
        
        config = EnhancedPhysicsConfig.for_small_protein(30)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGHIKLMNPQRSTVWY",
            physics_config=config
        )
        
        assert coordinator._physics_config.stuck_window == 20
        assert coordinator._physics_config.max_iterations == 1000
    
    def test_coordinator_backward_compatibility(self):
        """Test coordinator with legacy parameters."""
        from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFG",
            use_enhanced_energy=True,
            enable_side_chains=False
        )
        
        # Should create config from legacy params
        assert coordinator._physics_config.use_enhanced_energy is True
        assert coordinator._physics_config.enable_side_chains is False
    
    def test_coordinator_config_precedence(self):
        """Test that physics_config takes precedence over legacy params."""
        from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
        
        config = EnhancedPhysicsConfig(use_enhanced_energy=True)
        
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFG",
            physics_config=config,
            use_enhanced_energy=False  # Should be ignored
        )
        
        assert coordinator._physics_config.use_enhanced_energy is True
    
    def test_coordinator_with_auto_adapt(self):
        """Test coordinator with auto-adapted configuration."""
        from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
        
        sequence = "A" * 100  # 100-residue protein
        config = EnhancedPhysicsConfig.auto_adapt(len(sequence))
        
        coordinator = MultiAgentCoordinator(
            protein_sequence=sequence,
            physics_config=config
        )
        
        # Should use medium protein settings
        assert coordinator._physics_config.stuck_window == 30
        assert coordinator._physics_config.max_iterations == 2000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
