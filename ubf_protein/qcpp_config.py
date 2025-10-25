"""
Configuration for QCPP-UBF Integration

This module provides configuration management for the integration between
the Quantum Coherence Protein Predictor (QCPP) and Universal Behavioral
Framework (UBF) systems.

Key Components:
- QCPPIntegrationConfig: Main configuration dataclass
- Default configurations for common use cases
- Configuration validation utilities
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class QCPPIntegrationConfig:
    """
    Configuration for QCPP-UBF integration.
    
    This dataclass controls all aspects of how QCPP and UBF interact,
    including performance tuning, feature flags, and threshold values.
    
    Design Philosophy:
    - Sensible defaults for production use
    - All features enabled by default
    - Conservative performance targets
    - Easy to disable for testing/debugging
    
    Attributes:
        enabled: Master switch for QCPP integration (default: True)
        analysis_frequency: Analyze every N iterations (default: 1, every iteration)
        cache_size: LRU cache size for QCPP analysis (default: 1000)
        max_calculation_time_ms: Timeout for QCPP analysis (default: 5.0ms)
        phi_reward_threshold: Min phi match score for energy reward (default: 0.8)
        phi_reward_energy: Energy bonus for phi patterns (default: -50.0 kcal/mol)
        enable_dynamic_adjustment: Enable stability-based parameter adjustment (default: True)
        stability_low_threshold: Threshold for unstable regions (default: 1.0)
        stability_high_threshold: Threshold for stable regions (default: 2.0)
        enable_physics_grounding: Enable consciousness grounding in QCPP (default: True)
        smoothing_factor: Exponential smoothing for consciousness updates (default: 0.1)
        enable_trajectory_recording: Enable integrated trajectory recording (default: True)
        trajectory_max_points: Maximum trajectory points to store (default: 10000)
        log_qcpp_metrics: Log QCPP metrics to console (default: False)
    """
    
    # Master enable/disable switch
    enabled: bool = True
    
    # QCPP analysis frequency
    analysis_frequency: int = 1  # Analyze every N iterations (1 = every iteration)
    
    # Task 12: Adaptive analysis frequency
    enable_adaptive_frequency: bool = True  # Automatically adjust frequency based on performance
    min_analysis_frequency: int = 1  # Minimum frequency (most frequent)
    max_analysis_frequency: int = 10  # Maximum frequency (least frequent)
    
    # Caching configuration
    cache_size: int = 1000  # LRU cache size for conformations
    
    # Performance constraints
    max_calculation_time_ms: float = 5.0  # Timeout for QCPP analysis
    
    # Phi pattern reward configuration
    phi_reward_threshold: float = 0.8  # Min phi match for reward (0-1)
    phi_reward_energy: float = -50.0  # Energy bonus in kcal/mol
    
    # Dynamic parameter adjustment
    enable_dynamic_adjustment: bool = True
    stability_low_threshold: float = 1.0  # Below this = unstable
    stability_high_threshold: float = 2.0  # Above this = stable
    frequency_adjustment_unstable: float = 2.0  # Hz increase when unstable
    frequency_adjustment_stable: float = -1.0  # Hz decrease when stable
    temperature_adjustment_unstable: float = 50.0  # K increase when unstable
    temperature_adjustment_stable: float = -20.0  # K decrease when stable
    
    # Physics-grounded consciousness
    enable_physics_grounding: bool = True
    smoothing_factor: float = 0.1  # Exponential smoothing (0-1)
    
    # Trajectory recording
    enable_trajectory_recording: bool = True
    trajectory_max_points: int = 10000  # Maximum points to store
    
    # Logging
    log_qcpp_metrics: bool = False  # Log detailed QCPP metrics
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration values are within acceptable ranges.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Analysis frequency must be positive
        if self.analysis_frequency < 1:
            raise ValueError(f"analysis_frequency must be >= 1, got {self.analysis_frequency}")
        
        # Task 12: Validate adaptive frequency bounds
        if self.enable_adaptive_frequency:
            if self.min_analysis_frequency < 1:
                raise ValueError(f"min_analysis_frequency must be >= 1, got {self.min_analysis_frequency}")
            if self.max_analysis_frequency < self.min_analysis_frequency:
                raise ValueError(
                    f"max_analysis_frequency ({self.max_analysis_frequency}) must be >= "
                    f"min_analysis_frequency ({self.min_analysis_frequency})"
                )
        
        # Cache size must be positive
        if self.cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {self.cache_size}")
        
        # Max calculation time must be positive
        if self.max_calculation_time_ms <= 0:
            raise ValueError(f"max_calculation_time_ms must be > 0, got {self.max_calculation_time_ms}")
        
        # Phi reward threshold must be in [0, 1]
        if not (0.0 <= self.phi_reward_threshold <= 1.0):
            raise ValueError(f"phi_reward_threshold must be in [0, 1], got {self.phi_reward_threshold}")
        
        # Stability thresholds must be positive and ordered correctly
        if self.stability_low_threshold <= 0:
            raise ValueError(f"stability_low_threshold must be > 0, got {self.stability_low_threshold}")
        
        if self.stability_high_threshold <= self.stability_low_threshold:
            raise ValueError(
                f"stability_high_threshold ({self.stability_high_threshold}) must be > "
                f"stability_low_threshold ({self.stability_low_threshold})"
            )
        
        # Smoothing factor must be in (0, 1]
        if not (0.0 < self.smoothing_factor <= 1.0):
            raise ValueError(f"smoothing_factor must be in (0, 1], got {self.smoothing_factor}")
        
        # Trajectory max points must be positive
        if self.trajectory_max_points < 1:
            raise ValueError(f"trajectory_max_points must be >= 1, got {self.trajectory_max_points}")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'enabled': self.enabled,
            'analysis_frequency': self.analysis_frequency,
            'enable_adaptive_frequency': self.enable_adaptive_frequency,
            'min_analysis_frequency': self.min_analysis_frequency,
            'max_analysis_frequency': self.max_analysis_frequency,
            'cache_size': self.cache_size,
            'max_calculation_time_ms': self.max_calculation_time_ms,
            'phi_reward_threshold': self.phi_reward_threshold,
            'phi_reward_energy': self.phi_reward_energy,
            'enable_dynamic_adjustment': self.enable_dynamic_adjustment,
            'stability_low_threshold': self.stability_low_threshold,
            'stability_high_threshold': self.stability_high_threshold,
            'frequency_adjustment_unstable': self.frequency_adjustment_unstable,
            'frequency_adjustment_stable': self.frequency_adjustment_stable,
            'temperature_adjustment_unstable': self.temperature_adjustment_unstable,
            'temperature_adjustment_stable': self.temperature_adjustment_stable,
            'enable_physics_grounding': self.enable_physics_grounding,
            'smoothing_factor': self.smoothing_factor,
            'enable_trajectory_recording': self.enable_trajectory_recording,
            'trajectory_max_points': self.trajectory_max_points,
            'log_qcpp_metrics': self.log_qcpp_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'QCPPIntegrationConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            QCPPIntegrationConfig instance
        """
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """String representation for logging."""
        status = "ENABLED" if self.enabled else "DISABLED"
        return (
            f"QCPPIntegrationConfig({status}, "
            f"freq={self.analysis_frequency}, "
            f"cache={self.cache_size}, "
            f"physics_grounding={self.enable_physics_grounding}, "
            f"trajectory={self.enable_trajectory_recording})"
        )


# ============================================================================
# Predefined Configurations
# ============================================================================

def get_default_config() -> QCPPIntegrationConfig:
    """
    Get default QCPP integration configuration.
    
    Balanced configuration suitable for most use cases:
    - All features enabled
    - Conservative performance limits
    - Standard thresholds
    
    Returns:
        QCPPIntegrationConfig with default settings
    """
    return QCPPIntegrationConfig()


def get_high_performance_config() -> QCPPIntegrationConfig:
    """
    Get high-performance QCPP integration configuration.
    
    Optimized for speed:
    - Larger cache (5000)
    - Less frequent analysis (every 5 iterations)
    - Faster timeout (2ms)
    - Trajectory recording disabled
    
    Returns:
        QCPPIntegrationConfig optimized for performance
    """
    return QCPPIntegrationConfig(
        enabled=True,
        analysis_frequency=5,  # Analyze less frequently
        cache_size=5000,  # Larger cache
        max_calculation_time_ms=2.0,  # Stricter timeout
        enable_trajectory_recording=False,  # Disable to save memory
        log_qcpp_metrics=False
    )


def get_high_accuracy_config() -> QCPPIntegrationConfig:
    """
    Get high-accuracy QCPP integration configuration.
    
    Optimized for accuracy:
    - Analyze every iteration
    - Larger timeout (10ms)
    - All features enabled
    - Detailed logging
    
    Returns:
        QCPPIntegrationConfig optimized for accuracy
    """
    return QCPPIntegrationConfig(
        enabled=True,
        analysis_frequency=1,  # Analyze every iteration
        cache_size=10000,  # Very large cache
        max_calculation_time_ms=10.0,  # More generous timeout
        enable_dynamic_adjustment=True,
        enable_physics_grounding=True,
        enable_trajectory_recording=True,
        trajectory_max_points=50000,  # Store more trajectory points
        log_qcpp_metrics=True  # Detailed logging
    )


def get_disabled_config() -> QCPPIntegrationConfig:
    """
    Get disabled QCPP integration configuration.
    
    Disables all QCPP integration, equivalent to not providing
    QCPP integration to the coordinator.
    
    Returns:
        QCPPIntegrationConfig with integration disabled
    """
    return QCPPIntegrationConfig(enabled=False)


def get_testing_config() -> QCPPIntegrationConfig:
    """
    Get testing QCPP integration configuration.
    
    Minimal configuration for unit testing:
    - Small cache (100)
    - Fast timeout (1ms)
    - Minimal trajectory recording
    
    Returns:
        QCPPIntegrationConfig suitable for testing
    """
    return QCPPIntegrationConfig(
        enabled=True,
        analysis_frequency=1,
        cache_size=100,  # Small cache for tests
        max_calculation_time_ms=1.0,
        enable_trajectory_recording=True,
        trajectory_max_points=100,  # Minimal trajectory
        log_qcpp_metrics=False
    )


# ============================================================================
# Configuration Helpers
# ============================================================================

def create_config(**kwargs) -> QCPPIntegrationConfig:
    """
    Create a custom configuration with specified overrides.
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        QCPPIntegrationConfig with specified overrides
        
    Example:
        config = create_config(
            cache_size=2000,
            enable_dynamic_adjustment=False
        )
    """
    return QCPPIntegrationConfig(**kwargs)


def is_enabled(config: Optional[QCPPIntegrationConfig]) -> bool:
    """
    Check if QCPP integration is enabled in configuration.
    
    Args:
        config: Configuration to check (None means disabled)
        
    Returns:
        True if integration is enabled, False otherwise
    """
    return config is not None and config.enabled
