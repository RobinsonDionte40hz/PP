"""
Dynamic parameter adjustment based on QCPP stability metrics.

This module implements adaptive parameter tuning that responds to
quantum physics-based stability feedback. Agents increase exploration
in unstable regions and exploit stable regions.
"""

import logging
from typing import Tuple, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)


class DynamicParameterAdjuster:
    """
    Adjusts exploration parameters based on QCPP stability metrics.
    
    Implements adaptive behavior:
    - Low stability (< 1.0): Increase exploration (higher freq, higher temp)
    - High stability (> 2.0): Increase exploitation (lower freq, lower temp)
    - Normal stability (1.0-2.0): No adjustment
    
    Enforces bounds:
    - Frequency: 3-15 Hz
    - Temperature: 100-500 K
    """
    
    # Parameter adjustment constants
    LOW_STABILITY_THRESHOLD = 1.0
    HIGH_STABILITY_THRESHOLD = 2.0
    
    FREQUENCY_INCREASE_UNSTABLE = 2.0  # Hz
    FREQUENCY_DECREASE_STABLE = 1.0    # Hz
    
    TEMPERATURE_INCREASE_UNSTABLE = 50.0  # K
    TEMPERATURE_DECREASE_STABLE = 20.0    # K
    
    # Parameter bounds
    MIN_FREQUENCY = 3.0   # Hz
    MAX_FREQUENCY = 15.0  # Hz
    
    MIN_TEMPERATURE = 100.0   # K
    MAX_TEMPERATURE = 1200.0  # K (raised to support high-temp annealing)
    
    def __init__(self):
        """Initialize dynamic parameter adjuster."""
        self._adjustment_count = 0
        self._total_frequency_change = 0.0
        self._total_temperature_change = 0.0
    
    def adjust_from_stability(self,
                            current_frequency: float,
                            current_temperature: float,
                            stability_score: float) -> Tuple[float, float]:
        """
        Adjust parameters based on QCPP stability score.
        
        Args:
            current_frequency: Current consciousness frequency (Hz)
            current_temperature: Current temperature parameter (K)
            stability_score: QCPP stability score (typically 0-3 range)
            
        Returns:
            Tuple of (adjusted_frequency, adjusted_temperature)
            
        Raises:
            ValueError: If current parameters are out of valid ranges
        """
        # Validate input parameters
        if not (self.MIN_FREQUENCY <= current_frequency <= self.MAX_FREQUENCY):
            raise ValueError(
                f"Current frequency {current_frequency} outside valid range "
                f"[{self.MIN_FREQUENCY}, {self.MAX_FREQUENCY}]"
            )
        
        if not (self.MIN_TEMPERATURE <= current_temperature <= self.MAX_TEMPERATURE):
            raise ValueError(
                f"Current temperature {current_temperature} outside valid range "
                f"[{self.MIN_TEMPERATURE}, {self.MAX_TEMPERATURE}]"
            )
        
        # Start with current values
        new_frequency = current_frequency
        new_temperature = current_temperature
        
        # Determine adjustment based on stability
        if stability_score < self.LOW_STABILITY_THRESHOLD:
            # Unstable region: Increase exploration
            new_frequency = current_frequency + self.FREQUENCY_INCREASE_UNSTABLE
            new_temperature = current_temperature + self.TEMPERATURE_INCREASE_UNSTABLE
            
            logger.debug(
                f"Low stability ({stability_score:.2f}): Increasing exploration "
                f"(freq +{self.FREQUENCY_INCREASE_UNSTABLE} Hz, "
                f"temp +{self.TEMPERATURE_INCREASE_UNSTABLE} K)"
            )
            
        elif stability_score > self.HIGH_STABILITY_THRESHOLD:
            # Stable region: Increase exploitation
            new_frequency = current_frequency - self.FREQUENCY_DECREASE_STABLE
            new_temperature = current_temperature - self.TEMPERATURE_DECREASE_STABLE
            
            logger.debug(
                f"High stability ({stability_score:.2f}): Increasing exploitation "
                f"(freq -{self.FREQUENCY_DECREASE_STABLE} Hz, "
                f"temp -{self.TEMPERATURE_DECREASE_STABLE} K)"
            )
        else:
            # Normal stability: No adjustment
            logger.debug(
                f"Normal stability ({stability_score:.2f}): No adjustment"
            )
        
        # Enforce bounds
        adjusted_frequency = self._enforce_frequency_bounds(new_frequency)
        adjusted_temperature = self._enforce_temperature_bounds(new_temperature)
        
        # Track changes for statistics
        if adjusted_frequency != current_frequency or adjusted_temperature != current_temperature:
            self._adjustment_count += 1
            self._total_frequency_change += (adjusted_frequency - current_frequency)
            self._total_temperature_change += (adjusted_temperature - current_temperature)
        
        return adjusted_frequency, adjusted_temperature
    
    def adjust_from_qcpp_metrics(self,
                                 current_frequency: float,
                                 current_temperature: float,
                                 qcpp_metrics: Any) -> Tuple[float, float]:
        """
        Adjust parameters from QCPPMetrics object.
        
        Convenience method that extracts stability score from QCPPMetrics
        and calls adjust_from_stability().
        
        Args:
            current_frequency: Current consciousness frequency (Hz)
            current_temperature: Current temperature parameter (K)
            qcpp_metrics: QCPPMetrics instance with stability_score attribute
            
        Returns:
            Tuple of (adjusted_frequency, adjusted_temperature)
            
        Raises:
            ValueError: If qcpp_metrics lacks stability_score attribute
        """
        if not hasattr(qcpp_metrics, 'stability_score'):
            raise ValueError("qcpp_metrics must have stability_score attribute")
        
        return self.adjust_from_stability(
            current_frequency,
            current_temperature,
            qcpp_metrics.stability_score
        )
    
    def _enforce_frequency_bounds(self, frequency: float) -> float:
        """
        Enforce frequency bounds [3-15 Hz].
        
        Args:
            frequency: Proposed frequency value
            
        Returns:
            Frequency clamped to valid range
        """
        if frequency < self.MIN_FREQUENCY:
            logger.debug(
                f"Frequency {frequency:.2f} below minimum, clamping to {self.MIN_FREQUENCY}"
            )
            return self.MIN_FREQUENCY
        
        if frequency > self.MAX_FREQUENCY:
            logger.debug(
                f"Frequency {frequency:.2f} above maximum, clamping to {self.MAX_FREQUENCY}"
            )
            return self.MAX_FREQUENCY
        
        return frequency
    
    def _enforce_temperature_bounds(self, temperature: float) -> float:
        """
        Enforce temperature bounds [100-500 K].
        
        Args:
            temperature: Proposed temperature value
            
        Returns:
            Temperature clamped to valid range
        """
        if temperature < self.MIN_TEMPERATURE:
            logger.debug(
                f"Temperature {temperature:.2f} below minimum, clamping to {self.MIN_TEMPERATURE}"
            )
            return self.MIN_TEMPERATURE
        
        if temperature > self.MAX_TEMPERATURE:
            logger.debug(
                f"Temperature {temperature:.2f} above maximum, clamping to {self.MAX_TEMPERATURE}"
            )
            return self.MAX_TEMPERATURE
        
        return temperature
    
    def get_adjustment_statistics(self) -> dict:
        """
        Get statistics about parameter adjustments.
        
        Returns:
            Dictionary with adjustment counts and total changes
        """
        return {
            'adjustment_count': self._adjustment_count,
            'total_frequency_change': self._total_frequency_change,
            'total_temperature_change': self._total_temperature_change,
            'avg_frequency_change': (
                self._total_frequency_change / self._adjustment_count
                if self._adjustment_count > 0 else 0.0
            ),
            'avg_temperature_change': (
                self._total_temperature_change / self._adjustment_count
                if self._adjustment_count > 0 else 0.0
            )
        }
    
    def reset_statistics(self):
        """Reset adjustment statistics."""
        self._adjustment_count = 0
        self._total_frequency_change = 0.0
        self._total_temperature_change = 0.0
    
    def should_reduce_iterations(self, determinism_score: float) -> bool:
        """
        Check if exploration should terminate early based on determinism.
        
        High determinism (>0.8) indicates the structure is settling into
        a predictable, stable attractor. Further random exploration is
        unlikely to improve quality, so terminate early to save time.
        
        Args:
            determinism_score: THz determinism score (0-1)
            
        Returns:
            True if iterations should be reduced by 50%
        """
        return determinism_score > 0.8
    
    def calculate_resonance_bonus(self, qcpp_metrics: Any) -> float:
        """
        Calculate move weight bonus for 40 Hz resonance maintenance.
        
        The 40 Hz "gamma band" frequency is associated with consciousness
        and protein stability. Moves that maintain resonance near this
        frequency should be preferred.
        
        Args:
            qcpp_metrics: QCPPMetrics with qcp_score (correlates with THz)
            
        Returns:
            Weight multiplier (1.0 = neutral, 1.3 = high resonance bonus)
        """
        try:
            # QCP score correlates with THz resonance strength
            # QCP typically ranges 3-8, with higher = stronger resonance
            # Map to resonance strength: QCP < 4 = weak, QCP > 6 = strong
            
            if not hasattr(qcpp_metrics, 'qcp_score'):
                return 1.0  # Neutral if no data
            
            qcp = qcpp_metrics.qcp_score
            
            # Strong resonance (QCP > 6): 1.3× bonus
            if qcp > 6.0:
                logger.debug(f"Strong 40Hz resonance (QCP={qcp:.2f}): 1.3× weight bonus")
                return 1.3
            
            # Moderate resonance (QCP 4-6): 1.15× bonus
            elif qcp > 4.0:
                logger.debug(f"Moderate 40Hz resonance (QCP={qcp:.2f}): 1.15× weight bonus")
                return 1.15
            
            # Weak resonance (QCP < 4): 1.0× neutral
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Error calculating resonance bonus: {e}")
            return 1.0
    
    def should_trigger_refinement_mode(self, qcpp_metrics: Any) -> bool:
        """
        Check if refinement mode should be triggered based on field coherence.
        
        High field coherence (>0.7) indicates quantum fields are well-aligned,
        suggesting a high-quality local minimum. Switch from exploration to
        refinement: smaller moves, lower temperature, higher precision.
        
        Args:
            qcpp_metrics: QCPPMetrics with field_coherence
            
        Returns:
            True if refinement mode should be activated
        """
        try:
            if not hasattr(qcpp_metrics, 'field_coherence'):
                return False
            
            coherence = qcpp_metrics.field_coherence
            
            # Field coherence ranges [-1, 1], normalize to [0, 1]
            normalized_coherence = (coherence + 1.0) / 2.0
            
            if normalized_coherence > 0.7:
                logger.info(
                    f"High field coherence ({normalized_coherence:.3f}): "
                    f"Triggering refinement mode"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking refinement mode trigger: {e}")
            return False
    
    def get_refinement_parameters(self,
                                  current_frequency: float,
                                  current_temperature: float) -> Tuple[float, float]:
        """
        Get parameters for refinement mode (smaller moves, lower temp).
        
        Refinement mode focuses on local optimization around a high-quality
        minimum rather than broad exploration.
        
        Args:
            current_frequency: Current consciousness frequency
            current_temperature: Current temperature
            
        Returns:
            Tuple of (refinement_frequency, refinement_temperature)
        """
        # Reduce frequency by 50% (slower, more careful exploration)
        refinement_frequency = max(self.MIN_FREQUENCY, current_frequency * 0.5)
        
        # Reduce temperature by 70% (less randomness, more exploitation)
        refinement_temperature = max(self.MIN_TEMPERATURE, current_temperature * 0.3)
        
        logger.debug(
            f"Refinement parameters: freq={refinement_frequency:.1f} Hz, "
            f"temp={refinement_temperature:.1f} K"
        )
        
        return refinement_frequency, refinement_temperature
