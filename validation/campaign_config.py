"""
Configuration Management System for Large-Scale Validation Campaigns

This module provides a comprehensive configuration management system with:
- Default configurations for various campaign types
- Configuration validation with detailed error messages
- Configuration loading from JSON/YAML files
- Configuration merging and override capabilities
- Configuration export and persistence
- Configuration presets for common use cases

Classes:
    ConfigValidator: Validates campaign configurations
    ConfigLoader: Loads and merges configurations from files
    ConfigPresets: Provides pre-configured campaign types
    CampaignConfigManager: Main configuration management interface

Example:
    >>> manager = CampaignConfigManager()
    >>> config = manager.get_preset('high_accuracy')
    >>> config = manager.override(config, num_agents=15)
    >>> manager.validate(config)
    >>> manager.save(config, 'my_config.json')
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not installed. YAML support disabled.")

from .large_scale_validation_campaign import CampaignConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Presets
# ============================================================================

class CampaignPreset(Enum):
    """Predefined campaign configuration presets."""
    DEFAULT = "default"
    FAST = "fast"
    HIGH_ACCURACY = "high_accuracy"
    BASELINE = "baseline"
    BENCHMARK = "benchmark"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class ConfigPresets:
    """
    Provides pre-configured campaign types for common use cases.
    
    Available presets:
        - default: Balanced settings for general validation
        - fast: Quick validation for development/testing
        - high_accuracy: Publication-quality results
        - baseline: UBF-only without QCPP integration
        - benchmark: Comparative benchmarking settings
        - development: Fast iteration during development
        - production: Robust production settings
    """
    
    @staticmethod
    def get_default() -> CampaignConfig:
        """
        Get default balanced configuration.
        
        Returns:
            CampaignConfig with standard balanced settings
        """
        return CampaignConfig(
            target_protein_count=60,
            enable_qcpp=True,
            max_parallel_tests=3,
            num_agents=10,
            iterations_per_agent=1000,
            checkpoint_interval=5,
            quality_gate_threshold=0.60,
            failure_rmsd_threshold=8.0,
            timeout_multiplier=2.0,
            random_seed=None,
            output_dir="./campaign_results"
        )
    
    @staticmethod
    def get_fast() -> CampaignConfig:
        """
        Get fast configuration for development and testing.
        
        Optimized for:
        - Quick iteration
        - Development workflow
        - Parameter testing
        
        Returns:
            CampaignConfig with fast settings
        """
        return CampaignConfig(
            target_protein_count=50,
            enable_qcpp=True,
            max_parallel_tests=5,
            num_agents=5,
            iterations_per_agent=500,
            checkpoint_interval=10,
            quality_gate_threshold=0.55,
            failure_rmsd_threshold=8.0,
            timeout_multiplier=1.5,
            random_seed=None,
            output_dir="./fast_campaign_results"
        )
    
    @staticmethod
    def get_high_accuracy() -> CampaignConfig:
        """
        Get high-accuracy configuration for publication-quality results.
        
        Optimized for:
        - Publication-quality data
        - Maximum accuracy
        - Thorough exploration
        
        Returns:
            CampaignConfig with high-accuracy settings
        """
        return CampaignConfig(
            target_protein_count=75,
            enable_qcpp=True,
            max_parallel_tests=2,
            num_agents=20,
            iterations_per_agent=2000,
            checkpoint_interval=3,
            quality_gate_threshold=0.65,
            failure_rmsd_threshold=6.0,
            timeout_multiplier=3.0,
            random_seed=42,
            output_dir="./high_accuracy_results"
        )
    
    @staticmethod
    def get_baseline() -> CampaignConfig:
        """
        Get baseline configuration (UBF-only, no QCPP).
        
        Optimized for:
        - Baseline comparisons
        - UBF-only validation
        - Control experiments
        
        Returns:
            CampaignConfig with QCPP disabled
        """
        return CampaignConfig(
            target_protein_count=60,
            enable_qcpp=False,
            max_parallel_tests=3,
            num_agents=10,
            iterations_per_agent=1000,
            checkpoint_interval=5,
            quality_gate_threshold=0.60,
            failure_rmsd_threshold=8.0,
            timeout_multiplier=2.0,
            random_seed=None,
            output_dir="./baseline_results"
        )
    
    @staticmethod
    def get_benchmark() -> CampaignConfig:
        """
        Get benchmark configuration for comparative analysis.
        
        Optimized for:
        - Comparative benchmarking
        - Statistical testing
        - Performance analysis
        
        Returns:
            CampaignConfig with benchmark settings
        """
        return CampaignConfig(
            target_protein_count=30,
            enable_qcpp=True,
            max_parallel_tests=2,
            num_agents=10,
            iterations_per_agent=1000,
            checkpoint_interval=5,
            quality_gate_threshold=0.60,
            failure_rmsd_threshold=8.0,
            timeout_multiplier=2.0,
            random_seed=42,
            output_dir="./benchmark_results"
        )
    
    @staticmethod
    def get_development() -> CampaignConfig:
        """
        Get development configuration for rapid testing.
        
        Optimized for:
        - Rapid prototyping
        - Feature testing
        - Debug workflows
        
        Returns:
            CampaignConfig with minimal settings
        """
        return CampaignConfig(
            target_protein_count=20,
            enable_qcpp=True,
            max_parallel_tests=4,
            num_agents=5,
            iterations_per_agent=300,
            checkpoint_interval=5,
            quality_gate_threshold=0.50,
            failure_rmsd_threshold=10.0,
            timeout_multiplier=1.0,
            random_seed=123,
            output_dir="./dev_results"
        )
    
    @staticmethod
    def get_production() -> CampaignConfig:
        """
        Get production configuration for robust execution.
        
        Optimized for:
        - Production deployments
        - Automated pipelines
        - Reliable execution
        
        Returns:
            CampaignConfig with production settings
        """
        return CampaignConfig(
            target_protein_count=60,
            enable_qcpp=True,
            max_parallel_tests=3,
            num_agents=12,
            iterations_per_agent=1500,
            checkpoint_interval=5,
            quality_gate_threshold=0.60,
            failure_rmsd_threshold=7.0,
            timeout_multiplier=2.5,
            random_seed=42,
            output_dir="./production_results"
        )
    
    @staticmethod
    def get_preset(preset_name: Union[str, CampaignPreset]) -> CampaignConfig:
        """
        Get configuration by preset name.
        
        Args:
            preset_name: Preset name or CampaignPreset enum
            
        Returns:
            CampaignConfig for specified preset
            
        Raises:
            ValueError: If preset name is invalid
        """
        if isinstance(preset_name, str):
            preset_name_lower = preset_name.lower()
            preset_map_str = {
                'default': ConfigPresets.get_default,
                'fast': ConfigPresets.get_fast,
                'high_accuracy': ConfigPresets.get_high_accuracy,
                'baseline': ConfigPresets.get_baseline,
                'benchmark': ConfigPresets.get_benchmark,
                'development': ConfigPresets.get_development,
                'production': ConfigPresets.get_production,
            }
            
            if preset_name_lower not in preset_map_str:
                available = list(preset_map_str.keys())
                raise ValueError(
                    f"Invalid preset '{preset_name}'. "
                    f"Available presets: {available}"
                )
            
            return preset_map_str[preset_name_lower]()
        else:
            preset_map_enum = {
                CampaignPreset.DEFAULT: ConfigPresets.get_default,
                CampaignPreset.FAST: ConfigPresets.get_fast,
                CampaignPreset.HIGH_ACCURACY: ConfigPresets.get_high_accuracy,
                CampaignPreset.BASELINE: ConfigPresets.get_baseline,
                CampaignPreset.BENCHMARK: ConfigPresets.get_benchmark,
                CampaignPreset.DEVELOPMENT: ConfigPresets.get_development,
                CampaignPreset.PRODUCTION: ConfigPresets.get_production,
            }
            
            if preset_name not in preset_map_enum:
                available = list(preset_map_enum.keys())
                raise ValueError(
                    f"Invalid preset '{preset_name}'. "
                    f"Available presets: {available}"
                )
            
            return preset_map_enum[preset_name]()
    
    @staticmethod
    def list_presets() -> List[str]:
        """
        List all available preset names.
        
        Returns:
            List of preset names
        """
        return [
            'default',
            'fast',
            'high_accuracy',
            'baseline',
            'benchmark',
            'development',
            'production'
        ]


# ============================================================================
# Configuration Validation
# ============================================================================

class ValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigValidator:
    """
    Validates campaign configurations with detailed error messages.
    
    Performs comprehensive validation including:
    - Range checks for numeric parameters
    - Type validation
    - Logical consistency checks
    - Resource feasibility checks
    """
    
    @staticmethod
    def validate(config: CampaignConfig) -> None:
        """
        Validate campaign configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValidationError: If configuration is invalid
        """
        errors: List[str] = []
        
        # Validate protein count
        if config.target_protein_count < 1:
            errors.append(
                f"target_protein_count must be >= 1, got {config.target_protein_count}"
            )
        elif config.target_protein_count < 20:
            logger.warning(
                f"target_protein_count={config.target_protein_count} is very small. "
                f"Consider using at least 20 proteins for meaningful statistics."
            )
        elif config.target_protein_count > 100:
            logger.warning(
                f"target_protein_count={config.target_protein_count} is very large. "
                f"This will require significant computational resources."
            )
        
        # Validate parallel tests
        if config.max_parallel_tests < 1:
            errors.append(
                f"max_parallel_tests must be >= 1, got {config.max_parallel_tests}"
            )
        elif config.max_parallel_tests > 10:
            logger.warning(
                f"max_parallel_tests={config.max_parallel_tests} may overwhelm system resources"
            )
        
        # Validate agents
        if config.num_agents < 1:
            errors.append(
                f"num_agents must be >= 1, got {config.num_agents}"
            )
        elif config.num_agents > 50:
            logger.warning(
                f"num_agents={config.num_agents} may be excessive. "
                f"Typical range is 5-20."
            )
        
        # Validate iterations
        if config.iterations_per_agent < 100:
            errors.append(
                f"iterations_per_agent must be >= 100, got {config.iterations_per_agent}"
            )
        elif config.iterations_per_agent > 10000:
            logger.warning(
                f"iterations_per_agent={config.iterations_per_agent} is very high. "
                f"This will significantly increase runtime."
            )
        
        # Validate checkpoint interval
        if config.checkpoint_interval < 1:
            errors.append(
                f"checkpoint_interval must be >= 1, got {config.checkpoint_interval}"
            )
        
        # Validate quality gate threshold
        if not (0.0 < config.quality_gate_threshold <= 1.0):
            errors.append(
                f"quality_gate_threshold must be in (0, 1], got {config.quality_gate_threshold}"
            )
        elif config.quality_gate_threshold < 0.4:
            logger.warning(
                f"quality_gate_threshold={config.quality_gate_threshold} is very low. "
                f"Most campaigns use 0.5-0.7."
            )
        
        # Validate RMSD threshold
        if config.failure_rmsd_threshold <= 0:
            errors.append(
                f"failure_rmsd_threshold must be > 0, got {config.failure_rmsd_threshold}"
            )
        elif config.failure_rmsd_threshold < 3.0:
            logger.warning(
                f"failure_rmsd_threshold={config.failure_rmsd_threshold}Å is very strict. "
                f"Typical range is 5-10Å."
            )
        
        # Validate timeout multiplier
        if config.timeout_multiplier <= 0:
            errors.append(
                f"timeout_multiplier must be > 0, got {config.timeout_multiplier}"
            )
        elif config.timeout_multiplier < 1.0:
            logger.warning(
                f"timeout_multiplier={config.timeout_multiplier} may cause premature timeouts"
            )
        
        # Validate output directory
        if not config.output_dir or not isinstance(config.output_dir, str):
            errors.append(
                f"output_dir must be a non-empty string, got {config.output_dir}"
            )
        
        # Check logical consistency
        if config.checkpoint_interval > config.target_protein_count:
            logger.warning(
                f"checkpoint_interval ({config.checkpoint_interval}) > "
                f"target_protein_count ({config.target_protein_count}). "
                f"Only one checkpoint will be created at the end."
            )
        
        # Resource feasibility check
        estimated_memory_mb = config.num_agents * config.max_parallel_tests * 50
        if estimated_memory_mb > 8000:
            logger.warning(
                f"Configuration may require ~{estimated_memory_mb}MB memory. "
                f"Ensure sufficient system resources."
            )
        
        # Raise if any errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValidationError(error_msg)
        
        logger.info("Configuration validation passed")


# ============================================================================
# Configuration Loading
# ============================================================================

class ConfigLoader:
    """
    Loads and merges configurations from JSON/YAML files.
    
    Supports:
    - JSON configuration files
    - YAML configuration files (if PyYAML installed)
    - Merging multiple configuration sources
    - Partial configuration overrides
    """
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Loaded configuration from {file_path}")
        return config_dict
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ImportError: If PyYAML not installed
            yaml.YAMLError: If file contains invalid YAML
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {file_path}")
        return config_dict
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file (auto-detect format).
        
        Args:
            file_path: Path to configuration file (.json or .yaml/.yml)
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If file format not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigLoader.load_json(file_path)
        elif suffix in ['.yaml', '.yml']:
            return ConfigLoader.load_yaml(file_path)
        else:
            raise ValueError(
                f"Unsupported configuration format: {suffix}. "
                f"Supported formats: .json, .yaml, .yml"
            )
    
    @staticmethod
    def merge(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration (takes precedence)
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    
    @staticmethod
    def dict_to_config(config_dict: Dict[str, Any]) -> CampaignConfig:
        """
        Convert dictionary to CampaignConfig object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            CampaignConfig instance
            
        Raises:
            TypeError: If dictionary contains invalid keys
        """
        # Filter to only valid CampaignConfig fields
        valid_fields = {f.name for f in fields(CampaignConfig)}
        filtered_dict = {
            k: v for k, v in config_dict.items()
            if k in valid_fields
        }
        
        # Log ignored keys
        ignored_keys = set(config_dict.keys()) - valid_fields
        if ignored_keys:
            logger.warning(f"Ignoring unrecognized config keys: {ignored_keys}")
        
        return CampaignConfig(**filtered_dict)


# ============================================================================
# Main Configuration Manager
# ============================================================================

class CampaignConfigManager:
    """
    Main configuration management interface.
    
    Provides high-level API for:
    - Loading configurations from presets or files
    - Validating configurations
    - Merging and overriding configurations
    - Saving configurations
    
    Example:
        >>> manager = CampaignConfigManager()
        >>> config = manager.get_preset('high_accuracy')
        >>> config = manager.override(config, num_agents=15, random_seed=123)
        >>> manager.validate(config)
        >>> manager.save(config, './my_config.json')
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.presets = ConfigPresets()
        self.validator = ConfigValidator()
        self.loader = ConfigLoader()
        logger.info("CampaignConfigManager initialized")
    
    def get_preset(self, preset_name: Union[str, CampaignPreset]) -> CampaignConfig:
        """
        Get configuration from preset.
        
        Args:
            preset_name: Preset name or CampaignPreset enum
            
        Returns:
            CampaignConfig for specified preset
        """
        return self.presets.get_preset(preset_name)
    
    def list_presets(self) -> List[str]:
        """
        List all available presets.
        
        Returns:
            List of preset names
        """
        return self.presets.list_presets()
    
    def load(self, file_path: Union[str, Path]) -> CampaignConfig:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            CampaignConfig loaded from file
        """
        config_dict = self.loader.load(file_path)
        return self.loader.dict_to_config(config_dict)
    
    def load_with_preset(
        self,
        preset_name: str,
        override_file: Optional[Union[str, Path]] = None
    ) -> CampaignConfig:
        """
        Load preset and override with file.
        
        Args:
            preset_name: Base preset name
            override_file: Optional file with overrides
            
        Returns:
            CampaignConfig with merged settings
        """
        # Start with preset
        config = self.get_preset(preset_name)
        config_dict = asdict(config)
        
        # Override with file if provided
        if override_file:
            override_dict = self.loader.load(override_file)
            config_dict = self.loader.merge(config_dict, override_dict)
        
        return self.loader.dict_to_config(config_dict)
    
    def override(
        self,
        config: CampaignConfig,
        **overrides: Any
    ) -> CampaignConfig:
        """
        Create new configuration with overrides.
        
        Args:
            config: Base configuration
            **overrides: Fields to override
            
        Returns:
            New CampaignConfig with overrides applied
            
        Example:
            >>> config = manager.override(
            ...     base_config,
            ...     num_agents=15,
            ...     random_seed=123
            ... )
        """
        config_dict = asdict(config)
        config_dict.update(overrides)
        return self.loader.dict_to_config(config_dict)
    
    def validate(self, config: CampaignConfig, strict: bool = True) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            strict: If True, raise exception on errors; if False, log warnings
            
        Returns:
            True if valid (when strict=False)
            
        Raises:
            ValidationError: If configuration invalid (when strict=True)
        """
        if strict:
            self.validator.validate(config)
            return True
        else:
            try:
                self.validator.validate(config)
                return True
            except ValidationError as e:
                logger.error(f"Configuration validation failed: {e}")
                return False
    
    def save(
        self,
        config: CampaignConfig,
        file_path: Union[str, Path],
        format: Optional[str] = None
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            file_path: Output file path
            format: File format ('json' or 'yaml'). Auto-detected if None.
        """
        file_path = Path(file_path)
        
        # Auto-detect format
        if format is None:
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix in ['.yaml', '.yml']:
                format = 'yaml'
            else:
                format = 'json'  # default
                file_path = file_path.with_suffix('.json')
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        config_dict = asdict(config)
        
        # Save in requested format
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML not installed. Install with: pip install pyyaml"
                )
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {file_path}")
    
    def create_preset_files(self, output_dir: Union[str, Path] = "./validation/configs") -> None:
        """
        Create all preset configuration files.
        
        Args:
            output_dir: Directory to save preset files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for preset_name in self.list_presets():
            config = self.get_preset(preset_name)
            file_path = output_dir / f"{preset_name}_campaign.json"
            self.save(config, file_path)
        
        logger.info(f"Created {len(self.list_presets())} preset files in {output_dir}")
