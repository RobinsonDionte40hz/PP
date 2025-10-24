"""
Adaptive configuration system for UBF protein system.

This module implements adaptive parameter tuning based on protein size
and properties, automatically scaling thresholds and limits for optimal
performance across different protein sizes.
"""

import math
from typing import Tuple

from .interfaces import IAdaptiveConfigurator
from .models import AdaptiveConfig, ProteinSizeClass
from .config import (
    BASE_STUCK_DETECTION_WINDOW,
    BASE_STUCK_DETECTION_THRESHOLD,
    MEMORY_SIGNIFICANCE_THRESHOLD,
    MAX_MEMORIES_PER_AGENT,
    FREQUENCY_MIN,
    FREQUENCY_MAX,
    COHERENCE_MIN,
    COHERENCE_MAX
)


class AdaptiveConfigurator(IAdaptiveConfigurator):
    """
    Adaptive configuration generator based on protein properties.
    
    Automatically scales parameters based on protein size:
    - Small proteins (<50 residues): Tighter thresholds, faster convergence
    - Medium proteins (50-150 residues): Balanced parameters
    - Large proteins (>150 residues): Relaxed thresholds, more exploration
    """

    def __init__(self):
        """Initialize adaptive configurator with default scaling factors."""
        # Size-specific multipliers for various parameters
        self._size_multipliers = {
            ProteinSizeClass.SMALL: {
                'stuck_window': 0.67,      # 20 -> 13
                'stuck_threshold': 0.5,    # 10 -> 5
                'max_memories': 0.6,       # 50 -> 30
                'convergence_energy': 0.5, # 10 -> 5
                'convergence_rmsd': 0.75,  # 2.0 -> 1.5
                'max_iterations': 0.5      # 2000 -> 1000
            },
            ProteinSizeClass.MEDIUM: {
                'stuck_window': 1.0,       # 20 -> 20
                'stuck_threshold': 1.0,    # 10 -> 10
                'max_memories': 1.0,       # 50 -> 50
                'convergence_energy': 1.0, # 10 -> 10
                'convergence_rmsd': 1.0,   # 2.0 -> 2.0
                'max_iterations': 1.0      # 2000 -> 2000
            },
            ProteinSizeClass.LARGE: {
                'stuck_window': 1.5,       # 20 -> 30
                'stuck_threshold': 1.5,    # 10 -> 15
                'max_memories': 1.5,       # 50 -> 75
                'convergence_energy': 1.5, # 10 -> 15
                'convergence_rmsd': 1.5,   # 2.0 -> 3.0
                'max_iterations': 2.5      # 2000 -> 5000
            }
        }

        # Base values for medium proteins (reference point)
        self._base_values = {
            'stuck_window': 20,
            'stuck_threshold': 10.0,
            'max_memories': 50,
            'convergence_energy': 10.0,
            'convergence_rmsd': 2.0,
            'max_iterations': 2000,
            'checkpoint_interval': 100
        }

    def classify_protein_size(self, sequence: str) -> ProteinSizeClass:
        """
        Classify protein size based on sequence length.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            ProteinSizeClass enum (SMALL, MEDIUM, or LARGE)
        """
        residue_count = len(sequence)
        
        if residue_count < 50:
            return ProteinSizeClass.SMALL
        elif residue_count <= 150:
            return ProteinSizeClass.MEDIUM
        else:
            return ProteinSizeClass.LARGE

    def scale_threshold(self, base_threshold: float, residue_count: int) -> float:
        """
        Scale threshold proportionally to protein size using square root scaling.
        
        Larger proteins need proportionally larger thresholds, but not linearly.
        Uses sqrt scaling: threshold ∝ √(residue_count)
        
        Args:
            base_threshold: Base threshold value (for 100 residues reference)
            residue_count: Number of residues in protein
            
        Returns:
            Scaled threshold value
        """
        # Reference size: 100 residues
        reference_size = 100.0
        
        # Square root scaling for better proportionality
        # Prevents thresholds from becoming too large for big proteins
        scale_factor = math.sqrt(residue_count / reference_size)
        
        return base_threshold * scale_factor

    def get_config_for_protein(self, sequence: str) -> AdaptiveConfig:
        """
        Generate adaptive configuration optimized for protein size.
        
        Automatically scales all parameters based on protein size classification.
        Small proteins get tighter thresholds and faster convergence criteria,
        while large proteins get more relaxed parameters to handle increased
        conformational complexity.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            AdaptiveConfig with size-appropriate parameters
        """
        residue_count = len(sequence)
        size_class = self.classify_protein_size(sequence)
        multipliers = self._size_multipliers[size_class]
        
        # Calculate scaled parameters
        stuck_window = int(self._base_values['stuck_window'] * multipliers['stuck_window'])
        stuck_threshold = self._base_values['stuck_threshold'] * multipliers['stuck_threshold']
        max_memories = int(self._base_values['max_memories'] * multipliers['max_memories'])
        convergence_energy = self._base_values['convergence_energy'] * multipliers['convergence_energy']
        convergence_rmsd = self._base_values['convergence_rmsd'] * multipliers['convergence_rmsd']
        max_iterations = int(self._base_values['max_iterations'] * multipliers['max_iterations'])
        
        # Fine-tune stuck threshold using square root scaling for very large proteins
        if residue_count > 150:
            stuck_threshold = self.scale_threshold(stuck_threshold, residue_count)
        
        # Checkpoint interval scales with max iterations (checkpoint every 5% of total)
        checkpoint_interval = max(50, int(max_iterations * 0.05))
        
        # Consciousness parameter ranges remain constant across sizes
        # (agents will naturally adapt their exploration based on outcomes)
        initial_frequency_range = (FREQUENCY_MIN, FREQUENCY_MAX)
        initial_coherence_range = (COHERENCE_MIN, COHERENCE_MAX)
        
        return AdaptiveConfig(
            size_class=size_class,
            residue_count=residue_count,
            initial_frequency_range=initial_frequency_range,
            initial_coherence_range=initial_coherence_range,
            stuck_detection_window=stuck_window,
            stuck_detection_threshold=stuck_threshold,
            memory_significance_threshold=MEMORY_SIGNIFICANCE_THRESHOLD,
            max_memories_per_agent=max_memories,
            convergence_energy_threshold=convergence_energy,
            convergence_rmsd_threshold=convergence_rmsd,
            max_iterations=max_iterations,
            checkpoint_interval=checkpoint_interval
        )

    def get_config_summary(self, config: AdaptiveConfig) -> str:
        """
        Generate human-readable summary of configuration.
        
        Args:
            config: AdaptiveConfig to summarize
            
        Returns:
            Formatted string with configuration details
        """
        summary = f"""
Adaptive Configuration Summary
==============================
Protein Size: {config.size_class.value.upper()} ({config.residue_count} residues)

Consciousness Parameters:
  - Frequency Range: {config.initial_frequency_range[0]:.1f} - {config.initial_frequency_range[1]:.1f} Hz
  - Coherence Range: {config.initial_coherence_range[0]:.1f} - {config.initial_coherence_range[1]:.1f}

Local Minima Detection:
  - Detection Window: {config.stuck_detection_window} iterations
  - Energy Threshold: {config.stuck_detection_threshold:.1f} kJ/mol

Memory System:
  - Significance Threshold: {config.memory_significance_threshold:.2f}
  - Max Memories per Agent: {config.max_memories_per_agent}

Convergence Criteria:
  - Energy Threshold: {config.convergence_energy_threshold:.1f} kJ/mol
  - RMSD Threshold: {config.convergence_rmsd_threshold:.1f} Å

Performance:
  - Max Iterations: {config.max_iterations}
  - Checkpoint Interval: {config.checkpoint_interval} iterations
==============================
        """
        return summary.strip()


# Global singleton instance for easy access
_default_configurator = None


def get_default_configurator() -> AdaptiveConfigurator:
    """
    Get the default global configurator instance.
    
    Returns:
        Singleton AdaptiveConfigurator instance
    """
    global _default_configurator
    if _default_configurator is None:
        _default_configurator = AdaptiveConfigurator()
    return _default_configurator


def create_config_for_sequence(sequence: str, verbose: bool = False) -> AdaptiveConfig:
    """
    Convenience function to create configuration for a sequence.
    
    Args:
        sequence: Amino acid sequence
        verbose: If True, print configuration summary
        
    Returns:
        AdaptiveConfig optimized for the sequence
    """
    configurator = get_default_configurator()
    config = configurator.get_config_for_protein(sequence)
    
    if verbose:
        print(configurator.get_config_summary(config))
    
    return config
