"""
Local minima detection and escape system for UBF protein system.

This module implements adaptive local minima detection using moving averages
and multiple escape strategies to help agents escape stuck states.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
import statistics

from .models import AdaptiveConfig


class EscapeStrategy(Enum):
    """Types of escape strategies for local minima"""
    FREQUENCY_BOOST = "frequency_boost"
    COHERENCE_REDUCTION = "coherence_reduction"
    LARGE_JUMP_BIAS = "large_jump_bias"
    COMBINED_ADJUSTMENT = "combined_adjustment"


class LocalMinimaDetector:
    """
    Adaptive local minima detector using moving averages and configurable thresholds.

    Tracks energy history and detects when agents are stuck in local minima,
    providing appropriate escape strategies based on the situation.
    """

    def __init__(self, adaptive_config: AdaptiveConfig):
        """
        Initialize local minima detector with adaptive configuration.

        Args:
            adaptive_config: Configuration with size-appropriate parameters
        """
        self.adaptive_config = adaptive_config

        # Energy history tracking
        self.energy_history: List[float] = []
        self.window_size = adaptive_config.stuck_detection_window
        self.threshold = adaptive_config.stuck_detection_threshold

        # Stuck state tracking
        self.stuck_count = 0
        self.last_escape_iteration = 0
        self.consecutive_stuck_iterations = 0

        # Escape strategy tracking
        self.escape_attempts = 0
        self.successful_escapes = 0

    def update(self, current_energy: float, iteration: int) -> bool:
        """
        Update energy history and check for stuck state.

        Uses moving average of energy changes over the detection window.
        Considers agent stuck if energy variation is below threshold for
        the entire window.

        Args:
            current_energy: Current conformation energy
            iteration: Current iteration number

        Returns:
            True if agent is detected as stuck in local minimum
        """
        # Add current energy to history
        self.energy_history.append(current_energy)

        # Keep only the recent window
        if len(self.energy_history) > self.window_size:
            self.energy_history = self.energy_history[-self.window_size:]

        # Need at least window_size energies to detect stuck state
        if len(self.energy_history) < self.window_size:
            return False

        # Calculate energy variation (standard deviation of recent energies)
        try:
            energy_variation = statistics.stdev(self.energy_history)
        except statistics.StatisticsError:
            # All energies are the same
            energy_variation = 0.0

        # Check if variation is below threshold (stuck in local minimum)
        is_stuck = energy_variation < self.threshold

        if is_stuck:
            self.consecutive_stuck_iterations += 1
            self.stuck_count += 1
        else:
            self.consecutive_stuck_iterations = 0

        return is_stuck

    def get_escape_strategy(self, current_frequency: float, current_coherence: float) -> Dict[str, Any]:
        """
        Get appropriate escape strategy based on current state and history.

        Returns consciousness coordinate adjustments and behavioral hints.

        Args:
            current_frequency: Current consciousness frequency
            current_coherence: Current consciousness coherence

        Returns:
            Dictionary with escape strategy parameters
        """
        self.escape_attempts += 1

        # Choose strategy based on consecutive stuck iterations and agent state
        if self.consecutive_stuck_iterations >= 10:  # Very stuck
            strategy = EscapeStrategy.COMBINED_ADJUSTMENT
        elif current_frequency < 8.0:  # Low energy state
            strategy = EscapeStrategy.FREQUENCY_BOOST
        elif current_coherence > 0.7:  # High focus state
            strategy = EscapeStrategy.COHERENCE_REDUCTION
        else:
            strategy = EscapeStrategy.LARGE_JUMP_BIAS

        # Generate strategy parameters
        if strategy == EscapeStrategy.FREQUENCY_BOOST:
            return {
                'strategy': strategy.value,
                'frequency_adjustment': 1.0,  # Boost exploration energy
                'coherence_adjustment': 0.0,
                'behavioral_hints': {
                    'risk_tolerance_boost': 0.2,
                    'exploration_energy_boost': 0.3
                }
            }
        elif strategy == EscapeStrategy.COHERENCE_REDUCTION:
            return {
                'strategy': strategy.value,
                'frequency_adjustment': 0.5,
                'coherence_adjustment': -0.1,  # Reduce structural focus
                'behavioral_hints': {
                    'structural_focus_reduction': 0.2,
                    'hydrophobic_drive_boost': 0.1
                }
            }
        elif strategy == EscapeStrategy.LARGE_JUMP_BIAS:
            return {
                'strategy': strategy.value,
                'frequency_adjustment': 0.8,
                'coherence_adjustment': -0.05,
                'behavioral_hints': {
                    'large_jump_preference': 0.5,
                    'risk_tolerance_boost': 0.3
                }
            }
        else:  # COMBINED_ADJUSTMENT
            return {
                'strategy': strategy.value,
                'frequency_adjustment': 1.5,  # Major boost
                'coherence_adjustment': -0.15,  # Major reduction
                'behavioral_hints': {
                    'risk_tolerance_boost': 0.4,
                    'exploration_energy_boost': 0.4,
                    'structural_focus_reduction': 0.3,
                    'large_jump_preference': 0.7
                }
            }

    def record_escape_success(self, iteration: int) -> None:
        """
        Record a successful escape from local minimum.

        Args:
            iteration: Iteration where escape was successful
        """
        self.successful_escapes += 1
        self.last_escape_iteration = iteration
        self.consecutive_stuck_iterations = 0

    def get_stuck_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stuck detection and escape performance.

        Returns:
            Dictionary with stuck detection statistics
        """
        return {
            'total_stuck_count': self.stuck_count,
            'escape_attempts': self.escape_attempts,
            'successful_escapes': self.successful_escapes,
            'escape_success_rate': (
                self.successful_escapes / max(1, self.escape_attempts)
            ),
            'consecutive_stuck_iterations': self.consecutive_stuck_iterations,
            'last_escape_iteration': self.last_escape_iteration
        }

    def reset(self) -> None:
        """
        Reset the detector state (useful for testing or restarting).
        """
        self.energy_history.clear()
        self.stuck_count = 0
        self.last_escape_iteration = 0
        self.consecutive_stuck_iterations = 0
        self.escape_attempts = 0
        self.successful_escapes = 0