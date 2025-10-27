"""
Unit Tests for Configuration Management System

Tests ConfigPresets, ConfigValidator, ConfigLoader, and CampaignConfigManager
functionality including preset loading, validation, file I/O, and merging.
"""

import json
import pytest
import tempfile
from pathlib import Path
from dataclasses import asdict

from validation.campaign_config import (
    CampaignConfigManager,
    ConfigPresets,
    ConfigValidator,
    ConfigLoader,
    ValidationError,
    CampaignPreset
)
from validation.large_scale_validation_campaign import CampaignConfig


# ============================================================================
# ConfigPresets Tests
# ============================================================================

class TestConfigPresets:
    """Test preset configuration loading."""
    
    def test_get_default_preset(self):
        """Test loading default preset."""
        config = ConfigPresets.get_default()
        
        assert isinstance(config, CampaignConfig)
        assert config.target_protein_count == 60
        assert config.enable_qcpp is True
        assert config.num_agents == 10
        assert config.iterations_per_agent == 1000
    
    def test_get_fast_preset(self):
        """Test loading fast preset."""
        config = ConfigPresets.get_fast()
        
        assert config.target_protein_count == 50
        assert config.num_agents == 5
        assert config.iterations_per_agent == 500
        assert config.max_parallel_tests == 5
    
    def test_get_high_accuracy_preset(self):
        """Test loading high accuracy preset."""
        config = ConfigPresets.get_high_accuracy()
        
        assert config.target_protein_count == 75
        assert config.num_agents == 20
        assert config.iterations_per_agent == 2000
        assert config.random_seed == 42
    
    def test_get_baseline_preset(self):
        """Test loading baseline preset."""
        config = ConfigPresets.get_baseline()
        
        assert config.enable_qcpp is False  # Key difference
        assert config.target_protein_count == 60
    
    def test_get_benchmark_preset(self):
        """Test loading benchmark preset."""
        config = ConfigPresets.get_benchmark()
        
        assert config.target_protein_count == 30
        assert config.random_seed == 42
    
    def test_get_development_preset(self):
        """Test loading development preset."""
        config = ConfigPresets.get_development()
        
        assert config.target_protein_count == 20
        assert config.num_agents == 5
        assert config.iterations_per_agent == 300
    
    def test_get_production_preset(self):
        """Test loading production preset."""
        config = ConfigPresets.get_production()
        
        assert config.target_protein_count == 60
        assert config.num_agents == 12
        assert config.iterations_per_agent == 1500
    
    def test_get_preset_by_string(self):
        """Test getting preset by string name."""
        config = ConfigPresets.get_preset('default')
        assert isinstance(config, CampaignConfig)
        
        config = ConfigPresets.get_preset('HIGH_ACCURACY')  # Case insensitive
        assert config.num_agents == 20
    
    def test_get_preset_by_enum(self):
        """Test getting preset by enum."""
        config = ConfigPresets.get_preset(CampaignPreset.HIGH_ACCURACY)
        assert config.num_agents == 20
    
    def test_get_preset_invalid_name(self):
        """Test error on invalid preset name."""
        with pytest.raises(ValueError, match="Invalid preset"):
            ConfigPresets.get_preset('nonexistent')
    
    def test_list_presets(self):
        """Test listing all available presets."""
        presets = ConfigPresets.list_presets()
        
        assert isinstance(presets, list)
        assert len(presets) == 7
        assert 'default' in presets
        assert 'fast' in presets
        assert 'high_accuracy' in presets


# ============================================================================
# ConfigValidator Tests
# ============================================================================

class TestConfigValidator:
    """Test configuration validation logic."""
    
    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = ConfigPresets.get_default()
        
        # Should not raise
        ConfigValidator.validate(config)
    
    def test_validate_negative_protein_count(self):
        """Test validation fails for negative protein count."""
        config = ConfigPresets.get_default()
        config.target_protein_count = -10
        
        with pytest.raises(ValidationError, match="target_protein_count must be >= 1"):
            ConfigValidator.validate(config)
    
    def test_validate_zero_parallel_tests(self):
        """Test validation fails for zero parallel tests."""
        config = ConfigPresets.get_default()
        config.max_parallel_tests = 0
        
        with pytest.raises(ValidationError, match="max_parallel_tests must be >= 1"):
            ConfigValidator.validate(config)
    
    def test_validate_negative_agents(self):
        """Test validation fails for negative agents."""
        config = ConfigPresets.get_default()
        config.num_agents = -5
        
        with pytest.raises(ValidationError, match="num_agents must be >= 1"):
            ConfigValidator.validate(config)
    
    def test_validate_low_iterations(self):
        """Test validation fails for too few iterations."""
        config = ConfigPresets.get_default()
        config.iterations_per_agent = 50
        
        with pytest.raises(ValidationError, match="iterations_per_agent must be >= 100"):
            ConfigValidator.validate(config)
    
    def test_validate_invalid_quality_threshold(self):
        """Test validation fails for invalid quality threshold."""
        config = ConfigPresets.get_default()
        
        # Too low
        config.quality_gate_threshold = 0.0
        with pytest.raises(ValidationError, match="quality_gate_threshold must be in"):
            ConfigValidator.validate(config)
        
        # Too high
        config.quality_gate_threshold = 1.5
        with pytest.raises(ValidationError, match="quality_gate_threshold must be in"):
            ConfigValidator.validate(config)
    
    def test_validate_negative_rmsd_threshold(self):
        """Test validation fails for negative RMSD threshold."""
        config = ConfigPresets.get_default()
        config.failure_rmsd_threshold = -1.0
        
        with pytest.raises(ValidationError, match="failure_rmsd_threshold must be > 0"):
            ConfigValidator.validate(config)
    
    def test_validate_invalid_timeout_multiplier(self):
        """Test validation fails for invalid timeout multiplier."""
        config = ConfigPresets.get_default()
        config.timeout_multiplier = 0.0
        
        with pytest.raises(ValidationError, match="timeout_multiplier must be > 0"):
            ConfigValidator.validate(config)
    
    def test_validate_empty_output_dir(self):
        """Test validation fails for empty output directory."""
        config = ConfigPresets.get_default()
        config.output_dir = ""
        
        with pytest.raises(ValidationError, match="output_dir must be a non-empty string"):
            ConfigValidator.validate(config)
    
    def test_validate_multiple_errors(self):
        """Test validation reports multiple errors."""
        config = ConfigPresets.get_default()
        config.num_agents = -5
        config.quality_gate_threshold = 2.0
        config.timeout_multiplier = -1.0
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate(config)
        
        error_msg = str(exc_info.value)
        assert "num_agents" in error_msg
        assert "quality_gate_threshold" in error_msg
        assert "timeout_multiplier" in error_msg


# ============================================================================
# ConfigLoader Tests
# ============================================================================

class TestConfigLoader:
    """Test configuration loading from files."""
    
    def test_load_json(self):
        """Test loading configuration from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"
            
            # Create test config
            test_config = {
                "target_protein_count": 50,
                "enable_qcpp": True,
                "num_agents": 12
            }
            
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            # Load
            loaded = ConfigLoader.load_json(config_path)
            
            assert loaded["target_protein_count"] == 50
            assert loaded["enable_qcpp"] is True
            assert loaded["num_agents"] == 12
    
    def test_load_json_file_not_found(self):
        """Test error when JSON file not found."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_json("nonexistent.json")
    
    def test_load_auto_detect_json(self):
        """Test auto-detection of JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.json"
            
            test_config = {"target_protein_count": 40}
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            loaded = ConfigLoader.load(config_path)
            assert loaded["target_protein_count"] == 40
    
    def test_load_unsupported_format(self):
        """Test error for unsupported file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.txt"
            config_path.touch()
            
            with pytest.raises(ValueError, match="Unsupported configuration format"):
                ConfigLoader.load(config_path)
    
    def test_merge_configs(self):
        """Test merging two configuration dictionaries."""
        base = {
            "target_protein_count": 60,
            "num_agents": 10,
            "enable_qcpp": True
        }
        
        override = {
            "num_agents": 15,
            "random_seed": 42
        }
        
        merged = ConfigLoader.merge(base, override)
        
        assert merged["target_protein_count"] == 60  # From base
        assert merged["num_agents"] == 15  # Overridden
        assert merged["enable_qcpp"] is True  # From base
        assert merged["random_seed"] == 42  # New from override
    
    def test_dict_to_config(self):
        """Test converting dictionary to CampaignConfig."""
        config_dict = {
            "target_protein_count": 50,
            "enable_qcpp": False,
            "num_agents": 8,
            "iterations_per_agent": 800,
            "max_parallel_tests": 2,
            "checkpoint_interval": 10,
            "quality_gate_threshold": 0.55,
            "failure_rmsd_threshold": 9.0,
            "timeout_multiplier": 2.5,
            "random_seed": 123,
            "output_dir": "./test_output"
        }
        
        config = ConfigLoader.dict_to_config(config_dict)
        
        assert isinstance(config, CampaignConfig)
        assert config.target_protein_count == 50
        assert config.enable_qcpp is False
        assert config.num_agents == 8
    
    def test_dict_to_config_filters_invalid_keys(self):
        """Test that invalid keys are filtered out."""
        config_dict = {
            "target_protein_count": 50,
            "invalid_key": "should be ignored",
            "another_invalid": 123
        }
        
        # Should not raise, just filter
        config = ConfigLoader.dict_to_config(config_dict)
        assert config.target_protein_count == 50


# ============================================================================
# CampaignConfigManager Tests
# ============================================================================

class TestCampaignConfigManager:
    """Test main configuration manager interface."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = CampaignConfigManager()
        
        assert manager.presets is not None
        assert manager.validator is not None
        assert manager.loader is not None
    
    def test_get_preset(self):
        """Test getting preset through manager."""
        manager = CampaignConfigManager()
        
        config = manager.get_preset('default')
        assert isinstance(config, CampaignConfig)
        assert config.target_protein_count == 60
    
    def test_list_presets(self):
        """Test listing presets through manager."""
        manager = CampaignConfigManager()
        
        presets = manager.list_presets()
        assert len(presets) == 7
        assert 'default' in presets
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        manager = CampaignConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.json"
            
            # Save a config first
            test_config = ConfigPresets.get_default()
            manager.save(test_config, config_path)
            
            # Load it back
            loaded_config = manager.load(config_path)
            
            assert loaded_config.target_protein_count == test_config.target_protein_count
            assert loaded_config.num_agents == test_config.num_agents
    
    def test_load_with_preset_no_override(self):
        """Test loading preset without override file."""
        manager = CampaignConfigManager()
        
        config = manager.load_with_preset('fast')
        
        assert config.target_protein_count == 50
        assert config.num_agents == 5
    
    def test_load_with_preset_with_override(self):
        """Test loading preset with override file."""
        manager = CampaignConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = Path(tmpdir) / "override.json"
            
            # Create override file
            override_dict = {
                "num_agents": 20,
                "random_seed": 999
            }
            with open(override_path, 'w') as f:
                json.dump(override_dict, f)
            
            # Load with override
            config = manager.load_with_preset('fast', override_path)
            
            assert config.target_protein_count == 50  # From preset
            assert config.num_agents == 20  # Overridden
            assert config.random_seed == 999  # Overridden
    
    def test_override(self):
        """Test overriding configuration parameters."""
        manager = CampaignConfigManager()
        
        base_config = manager.get_preset('default')
        
        custom_config = manager.override(
            base_config,
            num_agents=15,
            random_seed=42,
            output_dir="./custom_output"
        )
        
        assert custom_config.target_protein_count == 60  # Unchanged
        assert custom_config.num_agents == 15  # Overridden
        assert custom_config.random_seed == 42  # Overridden
        assert custom_config.output_dir == "./custom_output"  # Overridden
    
    def test_validate_strict(self):
        """Test strict validation through manager."""
        manager = CampaignConfigManager()
        
        valid_config = manager.get_preset('default')
        # Should not raise
        assert manager.validate(valid_config, strict=True) is True
        
        invalid_config = manager.override(valid_config, num_agents=-5)
        with pytest.raises(ValidationError):
            manager.validate(invalid_config, strict=True)
    
    def test_validate_non_strict(self):
        """Test non-strict validation through manager."""
        manager = CampaignConfigManager()
        
        valid_config = manager.get_preset('default')
        assert manager.validate(valid_config, strict=False) is True
        
        invalid_config = manager.override(valid_config, num_agents=-5)
        assert manager.validate(invalid_config, strict=False) is False
    
    def test_save_json(self):
        """Test saving configuration to JSON file."""
        manager = CampaignConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "saved_config.json"
            
            config = manager.get_preset('default')
            manager.save(config, config_path)
            
            assert config_path.exists()
            
            # Load and verify
            with open(config_path) as f:
                loaded = json.load(f)
            
            assert loaded["target_protein_count"] == 60
            assert loaded["num_agents"] == 10
    
    def test_save_auto_add_extension(self):
        """Test auto-adding .json extension."""
        manager = CampaignConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config_no_ext"
            
            config = manager.get_preset('fast')
            manager.save(config, config_path)
            
            expected_path = config_path.with_suffix('.json')
            assert expected_path.exists()
    
    def test_create_preset_files(self):
        """Test creating all preset configuration files."""
        manager = CampaignConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            manager.create_preset_files(output_dir)
            
            # Check all preset files were created
            for preset_name in manager.list_presets():
                expected_file = output_dir / f"{preset_name}_campaign.json"
                assert expected_file.exists()
    
    def test_round_trip_save_load(self):
        """Test configuration survives save/load cycle."""
        manager = CampaignConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "roundtrip.json"
            
            # Create custom config
            original = manager.override(
                manager.get_preset('default'),
                target_protein_count=55,
                num_agents=13,
                random_seed=777,
                output_dir="./roundtrip_test"
            )
            
            # Save
            manager.save(original, config_path)
            
            # Load
            loaded = manager.load(config_path)
            
            # Verify all fields match
            assert loaded.target_protein_count == 55
            assert loaded.num_agents == 13
            assert loaded.random_seed == 777
            assert loaded.output_dir == "./roundtrip_test"
            assert loaded.enable_qcpp == original.enable_qcpp


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigurationIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test complete configuration workflow."""
        manager = CampaignConfigManager()
        
        # 1. Load preset
        config = manager.get_preset('high_accuracy')
        
        # 2. Customize
        config = manager.override(
            config,
            target_protein_count=60,
            num_agents=15,
            random_seed=2025
        )
        
        # 3. Validate
        assert manager.validate(config) is True
        
        # 4. Save
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "workflow.json"
            manager.save(config, config_path)
            
            # 5. Load and verify
            loaded = manager.load(config_path)
            assert loaded.target_protein_count == 60
            assert loaded.num_agents == 15
            assert loaded.random_seed == 2025
    
    def test_preset_comparison(self):
        """Test comparing different presets."""
        manager = CampaignConfigManager()
        
        fast = manager.get_preset('fast')
        high_acc = manager.get_preset('high_accuracy')
        
        # Fast should be faster
        assert fast.iterations_per_agent < high_acc.iterations_per_agent
        assert fast.num_agents < high_acc.num_agents
        
        # High accuracy should test more proteins
        assert high_acc.target_protein_count >= fast.target_protein_count
    
    def test_error_recovery(self):
        """Test recovery from invalid configurations."""
        manager = CampaignConfigManager()
        
        # Create invalid config
        config = manager.override(
            manager.get_preset('default'),
            num_agents=-5
        )
        
        # Validation should fail
        with pytest.raises(ValidationError):
            manager.validate(config, strict=True)
        
        # Fix and retry
        fixed_config = manager.override(config, num_agents=10)
        assert manager.validate(fixed_config) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
