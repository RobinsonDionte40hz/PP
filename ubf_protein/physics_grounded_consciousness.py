"""
Physics-grounded consciousness implementation.

This module extends the consciousness system to map QCPP physics metrics
(QCP score and field coherence) to consciousness coordinates, grounding
agent behavior in quantum physics principles.
"""

import time
from typing import TYPE_CHECKING, Optional

from .consciousness import ConsciousnessState
from .models import ConsciousnessCoordinates, ConformationalOutcome
from .config import FREQUENCY_MIN, FREQUENCY_MAX, COHERENCE_MIN, COHERENCE_MAX

if TYPE_CHECKING:
    from .qcpp_integration import QCPPMetrics


class PhysicsGroundedConsciousness(ConsciousnessState):
    """
    Physics-grounded consciousness that maps QCPP metrics to coordinates.
    
    Extends ConsciousnessState to incorporate QCPP quantum physics metrics,
    creating a bidirectional mapping where:
    - High QCP scores → Lower frequency (more settled/stable)
    - Low QCP scores → Higher frequency (more exploring/searching)
    - QCPP coherence → Consciousness coherence (direct mapping)
    
    Uses exponential smoothing for gradual transitions between states,
    preventing abrupt changes in agent behavior.
    
    Formulas:
    - target_frequency = 15.0 - (qcp_score / 0.5)  # Range: 3-15 Hz
    - target_coherence = 0.2 + (qcpp_coherence * 0.8)  # Range: 0.2-1.0
    - new_value = old_value + (target - old_value) * smoothing_factor
    """
    
    def __init__(self, frequency: float, coherence: float, 
                 smoothing_factor: float = 0.1):
        """
        Initialize physics-grounded consciousness state.
        
        Args:
            frequency: Initial frequency in Hz (3-15)
            coherence: Initial coherence (0.2-1.0)
            smoothing_factor: Exponential smoothing factor (0-1, default 0.1)
                            Lower = smoother transitions, Higher = faster updates
        
        Raises:
            ValueError: If coordinates are outside valid bounds or smoothing_factor invalid
        """
        super().__init__(frequency, coherence)
        
        # Validate smoothing factor
        if not (0.0 < smoothing_factor <= 1.0):
            raise ValueError(f"Smoothing factor must be in (0, 1], got {smoothing_factor}")
        
        self.smoothing_factor = smoothing_factor
        
        # Track target values for smooth transitions
        self._target_frequency = frequency
        self._target_coherence = coherence
        
        # Statistics for monitoring
        self.qcpp_updates_count = 0
        self.total_frequency_change = 0.0
        self.total_coherence_change = 0.0
    
    def update_from_qcpp_metrics(self, qcpp_metrics: 'QCPPMetrics') -> None:
        """
        Update consciousness coordinates from QCPP metrics.
        
        Maps QCPP physics measurements to consciousness coordinates using:
        1. QCP score → Target frequency (inverse relationship)
        2. QCPP coherence → Target consciousness coherence (direct)
        3. Exponential smoothing for gradual transitions
        
        Args:
            qcpp_metrics: QCPP analysis results containing QCP score and coherence
        """
        # Calculate target frequency from QCP score
        # High QCP (stable) → Low frequency (settled)
        # Low QCP (unstable) → High frequency (exploring)
        self._target_frequency = self._map_qcp_to_frequency(qcpp_metrics.qcp_score)
        
        # Calculate target coherence from QCPP coherence
        # Direct mapping with range transformation
        self._target_coherence = self._map_qcpp_coherence_to_consciousness(
            qcpp_metrics.field_coherence
        )
        
        # Apply exponential smoothing for gradual transitions
        old_frequency = self._coordinates.frequency
        old_coherence = self._coordinates.coherence
        
        new_frequency = self._smooth_transition(
            current=old_frequency,
            target=self._target_frequency,
            smoothing=self.smoothing_factor
        )
        
        new_coherence = self._smooth_transition(
            current=old_coherence,
            target=self._target_coherence,
            smoothing=self.smoothing_factor
        )
        
        # Enforce bounds
        new_frequency = max(FREQUENCY_MIN, min(FREQUENCY_MAX, new_frequency))
        new_coherence = max(COHERENCE_MIN, min(COHERENCE_MAX, new_coherence))
        
        # Update coordinates
        self._coordinates.frequency = new_frequency
        self._coordinates.coherence = new_coherence
        self._coordinates.last_update_timestamp = int(time.time() * 1000)
        
        # Update statistics
        self.qcpp_updates_count += 1
        self.total_frequency_change += abs(new_frequency - old_frequency)
        self.total_coherence_change += abs(new_coherence - old_coherence)
    
    def get_target_frequency(self, qcp_score: float) -> float:
        """
        Calculate target frequency from QCP score.
        
        Formula: target_frequency = 15.0 - (qcp_score / 0.5)
        
        This creates an inverse relationship:
        - QCP = 0 → freq = 15 Hz (maximum exploration)
        - QCP = 3 → freq = 9 Hz (moderate)
        - QCP = 6 → freq = 3 Hz (settled in stable state)
        
        Args:
            qcp_score: QCP score from QCPP analysis
            
        Returns:
            Target frequency in Hz (clamped to 3-15 range)
        """
        return self._map_qcp_to_frequency(qcp_score)
    
    def get_target_coherence(self, qcpp_coherence: float) -> float:
        """
        Calculate target coherence from QCPP coherence.
        
        Formula: target_coherence = 0.2 + (qcpp_coherence * 0.8)
        
        Maps QCPP coherence [-1, 1] to consciousness coherence [0.2, 1.0]:
        - QCPP coherence = -1 → consciousness = 0.2 (minimum)
        - QCPP coherence = 0 → consciousness = 0.6 (moderate)
        - QCPP coherence = 1 → consciousness = 1.0 (maximum)
        
        Args:
            qcpp_coherence: Field coherence from QCPP analysis
            
        Returns:
            Target coherence (clamped to 0.2-1.0 range)
        """
        return self._map_qcpp_coherence_to_consciousness(qcpp_coherence)
    
    def _map_qcp_to_frequency(self, qcp_score: float) -> float:
        """
        Map QCP score to target frequency (inverse relationship).
        
        Args:
            qcp_score: QCP score (typically 3-8)
            
        Returns:
            Target frequency in Hz (3-15)
        """
        # Formula: 15.0 - (qcp_score / 0.5)
        # QCP of 0 → 15 Hz, QCP of 6 → 3 Hz
        target = 15.0 - (qcp_score / 0.5)
        
        # Clamp to valid range
        return max(FREQUENCY_MIN, min(FREQUENCY_MAX, target))
    
    def _map_qcpp_coherence_to_consciousness(self, qcpp_coherence: float) -> float:
        """
        Map QCPP coherence to consciousness coherence (direct with offset).
        
        Args:
            qcpp_coherence: Field coherence from QCPP (typically -1 to 1)
            
        Returns:
            Target consciousness coherence (0.2-1.0)
        """
        # Formula: 0.2 + (qcpp_coherence * 0.8)
        # But QCPP coherence needs to be normalized to [0, 1] first
        # Since QCPP coherence is in [-1, 1], normalize it
        normalized = (qcpp_coherence + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
        target = 0.2 + (normalized * 0.8)
        
        # Clamp to valid range
        return max(COHERENCE_MIN, min(COHERENCE_MAX, target))
    
    def _smooth_transition(self, current: float, target: float, 
                          smoothing: float) -> float:
        """
        Apply exponential smoothing for gradual transitions.
        
        Formula: new_value = current + (target - current) * smoothing
        
        Args:
            current: Current value
            target: Target value
            smoothing: Smoothing factor (0-1)
            
        Returns:
            Smoothed new value
        """
        return current + (target - current) * smoothing
    
    def get_physics_grounding_stats(self) -> dict:
        """
        Get statistics about physics grounding updates.
        
        Returns:
            Dictionary with update statistics:
            - qcpp_updates_count: Number of QCPP-based updates
            - avg_frequency_change: Average frequency change per update
            - avg_coherence_change: Average coherence change per update
            - target_frequency: Current target frequency
            - target_coherence: Current target coherence
            - smoothing_factor: Current smoothing factor
        """
        return {
            'qcpp_updates_count': self.qcpp_updates_count,
            'avg_frequency_change': (
                self.total_frequency_change / self.qcpp_updates_count 
                if self.qcpp_updates_count > 0 else 0.0
            ),
            'avg_coherence_change': (
                self.total_coherence_change / self.qcpp_updates_count
                if self.qcpp_updates_count > 0 else 0.0
            ),
            'target_frequency': self._target_frequency,
            'target_coherence': self._target_coherence,
            'smoothing_factor': self.smoothing_factor,
            'current_frequency': self._coordinates.frequency,
            'current_coherence': self._coordinates.coherence
        }
    
    def set_smoothing_factor(self, smoothing_factor: float) -> None:
        """
        Update the smoothing factor for future transitions.
        
        Args:
            smoothing_factor: New smoothing factor (0-1)
            
        Raises:
            ValueError: If smoothing_factor is outside (0, 1]
        """
        if not (0.0 < smoothing_factor <= 1.0):
            raise ValueError(f"Smoothing factor must be in (0, 1], got {smoothing_factor}")
        
        self.smoothing_factor = smoothing_factor
