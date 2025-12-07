"""
Search Strategy Implementation (Derived from Exploration Parameters)

Copyright (c) 2025 Dionte Robinson
MIT License - See LICENSE file for details

If you use this work in academic research, please cite:
    Robinson, D. (2025). UBF Protein System - Behavioral State Derivation.
    https://github.com/RobinsonDionte40hz/PP

Key formulas:
- 2D→5D behavioral transformation
- Geometric mean combinations for hydrophobic drive
- Adaptive risk tolerance calculations

---

⚠️ IMPORTANT DISCLAIMER:
"Behavioral state" is derived from exploration parameters (metaphorically
called "consciousness coordinates"). This is NOT about protein behavior or
consciousness - it's a mathematical transformation.

This module implements HEURISTIC derivations:
- 5 search dimensions derived from 2 base parameters
- Transformations are AD HOC, not theoretically derived
- No claim these are optimal or unique transformations

Mathematical Reality:
- Input: 2D parameter space (aggressiveness, consistency)
- Output: 5D search strategy space
- Transformation: Heuristic functions (geometric mean, linear combinations)
"""

import time
from typing import Optional

from .interfaces import IBehavioralState
from .models import ConsciousnessCoordinates, BehavioralStateData
from .config import BEHAVIORAL_STATE_REGEN_THRESHOLD


class BehavioralState(IBehavioralState):
    """
    Search strategy state derived from exploration parameters.

    The 5D search strategy is derived from 2D exploration parameters using 
    proprietary transformation functions optimized through extensive testing.

    Derivation transforms 2D parameters into:
    - exploration_energy: How much to explore vs exploit
    - structural_focus: Precision of structural moves
    - hydrophobic_drive: Tendency toward core formation
    - risk_tolerance: Willingness for bold moves
    - native_state_ambition: Goal-directedness

    Specific transformation functions are proprietary.
    """

    def __init__(self, consciousness_coordinates: ConsciousnessCoordinates):
        """
        Initialize search strategy from exploration parameters.

        Args:
            consciousness_coordinates: Exploration parameters (aggressiveness, consistency)
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
        Create search strategy from exploration parameters.

        Args:
            frequency: Aggressiveness parameter (3-15, dimensionless)
            coherence: Consistency parameter (0.2-1.0, dimensionless)

        Returns:
            New BehavioralState (search strategy) instance
            
        Note: "frequency" and "coherence" are metaphorical parameter names.
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