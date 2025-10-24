"""
Behavioral state implementation for UBF protein system.

This module implements the behavioral state that is derived from consciousness
coordinates and cached to avoid recomputation.
"""

import time
from typing import Optional

from .interfaces import IBehavioralState
from .models import ConsciousnessCoordinates, BehavioralStateData
from .config import BEHAVIORAL_STATE_REGEN_THRESHOLD


class BehavioralState(IBehavioralState):
    """
    Implementation of cached behavioral state derived from consciousness coordinates.

    Behavioral state represents the agent's current behavioral preferences and
    tendencies, derived from consciousness coordinates. This state is cached
    to avoid recomputation on every access.
    """

    def __init__(self, consciousness_coordinates: ConsciousnessCoordinates):
        """
        Initialize behavioral state from consciousness coordinates.

        Args:
            consciousness_coordinates: The consciousness coordinates to derive behavior from
        """
        self._behavioral_data = BehavioralStateData.from_consciousness(
            consciousness_coordinates.frequency,
            consciousness_coordinates.coherence
        )
        # Make a copy to avoid reference issues
        self._last_coordinates = ConsciousnessCoordinates(
            frequency=consciousness_coordinates.frequency,
            coherence=consciousness_coordinates.coherence,
            last_update_timestamp=consciousness_coordinates.last_update_timestamp
        )
        self._cached_timestamp = consciousness_coordinates.last_update_timestamp

    @staticmethod
    def from_consciousness(frequency: float, coherence: float) -> 'BehavioralState':
        """
        Create behavioral state from consciousness coordinates.

        Args:
            frequency: Consciousness frequency (3-15 Hz)
            coherence: Consciousness coherence (0.2-1.0)

        Returns:
            New BehavioralState instance
        """
        coordinates = ConsciousnessCoordinates(
            frequency=frequency,
            coherence=coherence,
            last_update_timestamp=int(time.time() * 1000)
        )
        return BehavioralState(coordinates)

    def get_exploration_energy(self) -> float:
        """Energy level for conformational exploration (0.0-1.0)."""
        return self._behavioral_data.exploration_energy

    def get_structural_focus(self) -> float:
        """Focus/precision for structural refinement (0.0-1.0)."""
        return self._behavioral_data.structural_focus

    def get_hydrophobic_drive(self) -> float:
        """Drive toward hydrophobic collapse (0.0-1.0)."""
        return self._behavioral_data.hydrophobic_drive

    def get_risk_tolerance(self) -> float:
        """Willingness to try radical moves (0.0-1.0)."""
        return self._behavioral_data.risk_tolerance

    def get_native_state_ambition(self) -> float:
        """Drive toward goal-directed behavior (0.0-1.0)."""
        return self._behavioral_data.native_state_ambition

    def should_regenerate(self, coordinate_change: float) -> bool:
        """
        Check if behavioral state needs regeneration based on coordinate change.

        Args:
            coordinate_change: The magnitude of change in consciousness coordinates

        Returns:
            True if regeneration is needed (change >= threshold)
        """
        return coordinate_change >= BEHAVIORAL_STATE_REGEN_THRESHOLD

    def get_behavioral_data(self) -> BehavioralStateData:
        """Get the underlying behavioral state data."""
        return self._behavioral_data

    def get_cached_timestamp(self) -> int:
        """Get the timestamp when this behavioral state was cached."""
        return self._cached_timestamp

    def regenerate_if_needed(self, new_coordinates: ConsciousnessCoordinates) -> Optional['BehavioralState']:
        """
        Regenerate behavioral state if coordinates have changed significantly.

        Args:
            new_coordinates: New consciousness coordinates

        Returns:
            New BehavioralState if regeneration occurred, None otherwise
        """
        # Calculate coordinate change magnitude
        freq_change = abs(new_coordinates.frequency - self._last_coordinates.frequency)
        coh_change = abs(new_coordinates.coherence - self._last_coordinates.coherence)
        max_change = max(freq_change, coh_change)

        if self.should_regenerate(max_change):
            return BehavioralState(new_coordinates)

        return None