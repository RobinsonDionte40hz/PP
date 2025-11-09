"""
Mediator Agent Configuration Module

This module provides configuration dataclasses for Mediator Agents, which are specialized
agents responsible for pattern detection and information relay in the UBF protein system.

Mediator Agents detect:
- THz resonance patterns (vibrational signatures)
- Folding dynamics (secondary structure formation)
- Geometric similarities (convergent pathways)

Author: UBF Protein System
Date: November 9, 2025
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MediatorConfig:
    """
    Configuration for Mediator Agent behavior.
    
    This dataclass controls all aspects of Mediator Agent operation, including
    detection thresholds, relay frequency, caching parameters, and feature flags.
    
    Attributes:
        # Detection Thresholds
        thz_similarity_threshold: Minimum correlation for THz resonance (0.0-1.0)
        geometric_similarity_threshold: Maximum RMSD for geometric similarity (Ångströms)
        secondary_structure_min_length: Minimum consecutive residues for helix/sheet
        
        # Relay Configuration
        relay_frequency: Detect patterns every N iterations (higher = less overhead)
        broadcast_throttle_rate: Maximum broadcasts per second (prevents memory overflow)
        
        # Cache Configuration
        cache_size: Maximum number of cached pattern detections
        cache_ttl_seconds: Time-to-live for cache entries in seconds
        
        # Performance Tuning (Feature Flags)
        enable_thz_detection: Enable/disable THz resonance detection
        enable_folding_detection: Enable/disable secondary structure detection
        enable_geometric_detection: Enable/disable geometric similarity detection
        
        # Memory Management
        max_reference_conformations: Maximum reference conformations for RMSD comparison
        pattern_history_size: Maximum number of patterns to track
    
    Example:
        >>> # Default configuration (balanced)
        >>> config = MediatorConfig()
        
        >>> # High sensitivity configuration
        >>> config = MediatorConfig(
        ...     thz_similarity_threshold=0.6,
        ...     geometric_similarity_threshold=3.0,
        ...     relay_frequency=10
        ... )
        
        >>> # Performance-optimized configuration
        >>> config = MediatorConfig(
        ...     relay_frequency=50,
        ...     enable_thz_detection=False,
        ...     cache_size=5000
        ... )
    """
    
    # Detection thresholds
    thz_similarity_threshold: float = 0.7
    geometric_similarity_threshold: float = 2.0  # Ångströms RMSD
    secondary_structure_min_length: int = 4      # Minimum helix/sheet length
    
    # Relay configuration
    relay_frequency: int = 20           # Detect patterns every N iterations
    broadcast_throttle_rate: int = 10   # Max broadcasts per second
    
    # Cache configuration
    cache_size: int = 10000
    cache_ttl_seconds: int = 3600       # 1 hour
    
    # Performance tuning (feature flags)
    enable_thz_detection: bool = True
    enable_folding_detection: bool = True
    enable_geometric_detection: bool = True
    
    # Memory management
    max_reference_conformations: int = 100
    pattern_history_size: int = 1000
    
    def __post_init__(self):
        """Validate all configuration parameters are within valid ranges."""
        
        # Validate thz_similarity_threshold (0.0-1.0)
        if not (0.0 <= self.thz_similarity_threshold <= 1.0):
            raise ValueError(
                f"thz_similarity_threshold must be in range [0.0, 1.0], "
                f"got {self.thz_similarity_threshold}"
            )
        
        # Validate geometric_similarity_threshold (positive)
        if self.geometric_similarity_threshold <= 0.0:
            raise ValueError(
                f"geometric_similarity_threshold must be positive, "
                f"got {self.geometric_similarity_threshold}"
            )
        
        # Validate secondary_structure_min_length (positive integer)
        if self.secondary_structure_min_length < 1:
            raise ValueError(
                f"secondary_structure_min_length must be at least 1, "
                f"got {self.secondary_structure_min_length}"
            )
        
        # Validate relay_frequency (positive integer)
        if self.relay_frequency < 1:
            raise ValueError(
                f"relay_frequency must be at least 1, "
                f"got {self.relay_frequency}"
            )
        
        # Validate broadcast_throttle_rate (positive integer)
        if self.broadcast_throttle_rate < 1:
            raise ValueError(
                f"broadcast_throttle_rate must be at least 1, "
                f"got {self.broadcast_throttle_rate}"
            )
        
        # Validate cache_size (positive integer)
        if self.cache_size < 1:
            raise ValueError(
                f"cache_size must be at least 1, "
                f"got {self.cache_size}"
            )
        
        # Validate cache_ttl_seconds (positive)
        if self.cache_ttl_seconds <= 0:
            raise ValueError(
                f"cache_ttl_seconds must be positive, "
                f"got {self.cache_ttl_seconds}"
            )
        
        # Validate max_reference_conformations (positive integer)
        if self.max_reference_conformations < 1:
            raise ValueError(
                f"max_reference_conformations must be at least 1, "
                f"got {self.max_reference_conformations}"
            )
        
        # Validate pattern_history_size (positive integer)
        if self.pattern_history_size < 1:
            raise ValueError(
                f"pattern_history_size must be at least 1, "
                f"got {self.pattern_history_size}"
            )
        
        # Validate at least one detection is enabled
        if not any([
            self.enable_thz_detection,
            self.enable_folding_detection,
            self.enable_geometric_detection
        ]):
            raise ValueError(
                "At least one detection type must be enabled "
                "(enable_thz_detection, enable_folding_detection, or enable_geometric_detection)"
            )
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for JSON serialization.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'detection_thresholds': {
                'thz_similarity': self.thz_similarity_threshold,
                'geometric_similarity_rmsd': self.geometric_similarity_threshold,
                'secondary_structure_min_length': self.secondary_structure_min_length,
            },
            'relay_config': {
                'relay_frequency': self.relay_frequency,
                'broadcast_throttle_rate': self.broadcast_throttle_rate,
            },
            'cache_config': {
                'cache_size': self.cache_size,
                'cache_ttl_seconds': self.cache_ttl_seconds,
            },
            'features': {
                'thz_detection': self.enable_thz_detection,
                'folding_detection': self.enable_folding_detection,
                'geometric_detection': self.enable_geometric_detection,
            },
            'memory_management': {
                'max_reference_conformations': self.max_reference_conformations,
                'pattern_history_size': self.pattern_history_size,
            }
        }
    
    @classmethod
    def get_high_sensitivity_config(cls) -> 'MediatorConfig':
        """
        Get configuration optimized for high sensitivity pattern detection.
        
        Use this for small proteins or when you want to detect subtle patterns.
        Increased computational cost due to more frequent detection and lower thresholds.
        
        Returns:
            MediatorConfig with high sensitivity settings
        """
        return cls(
            thz_similarity_threshold=0.6,
            geometric_similarity_threshold=3.0,
            secondary_structure_min_length=3,
            relay_frequency=10,
            cache_size=15000,
        )
    
    @classmethod
    def get_high_performance_config(cls) -> 'MediatorConfig':
        """
        Get configuration optimized for performance.
        
        Use this for large proteins or when computational resources are limited.
        Reduced overhead due to less frequent detection and disabled THz analysis.
        
        Returns:
            MediatorConfig with high performance settings
        """
        return cls(
            thz_similarity_threshold=0.8,
            geometric_similarity_threshold=1.5,
            secondary_structure_min_length=5,
            relay_frequency=50,
            broadcast_throttle_rate=5,
            cache_size=5000,
            enable_thz_detection=False,  # Disable expensive THz analysis
        )
    
    @classmethod
    def get_balanced_config(cls) -> 'MediatorConfig':
        """
        Get balanced configuration (same as default).
        
        Use this for general-purpose protein folding with good balance
        between sensitivity and performance.
        
        Returns:
            MediatorConfig with balanced settings (default values)
        """
        return cls()
