"""
Exploration parameter system implementation (consciousness-inspired metaphor).

⚠️ IMPORTANT DISCLAIMER:
This module uses "consciousness" as a METAPHORICAL FRAMEWORK for organizing
exploration parameters. This is NOT a claim about consciousness in proteins.

The "consciousness" terminology provides intuitive parameter names:
- "frequency" = exploration aggressiveness (dimensionless, scaled 3-15)
- "coherence" = behavioral consistency (dimensionless, scaled 0.2-1.0)

This is a heuristic optimization framework inspired by how conscious systems
balance exploration vs exploitation, NOT a physical model of protein cognition.

Mathematical Reality:
- These are optimization parameters in a 2D search space
- No connection to neuroscience or quantum consciousness
- Parameter evolution is based on heuristic update rules
"""

import time
from typing import TYPE_CHECKING

from .interfaces import IConsciousnessState
from .models import ConsciousnessCoordinates, ConformationalOutcome
from .config import (
    FREQUENCY_MIN, FREQUENCY_MAX, COHERENCE_MIN, COHERENCE_MAX,
    CONSCIOUSNESS_UPDATE_RULES
)

if TYPE_CHECKING:
    from .models import ConformationalOutcome


class ConsciousnessState(IConsciousnessState):
    """
    Exploration parameter state manager (metaphorically called "consciousness").

    ⚠️ METAPHOR ALERT: "Consciousness" is a design pattern, not a physical claim.
    
    This class manages two optimization parameters:
    - frequency: Exploration aggressiveness (dimensionless, 3-15)
    - coherence: Behavioral consistency (dimensionless, 0.2-1.0)
    
    Updates parameters based on exploration outcomes using heuristic rules.
    The "Hz" notation for frequency is metaphorical - these are dimensionless
    parameters inspired by dynamical systems theory, not physical frequencies.
    """

    def __init__(self, frequency: float, coherence: float):
        """
        Initialize exploration parameter state.

        Args:
            frequency: Aggressiveness parameter (3-15, dimensionless)
            coherence: Consistency parameter (0.2-1.0, dimensionless)

        Raises:
            ValueError: If parameters are outside valid bounds
        """
        # Validate bounds
        if not (FREQUENCY_MIN <= frequency <= FREQUENCY_MAX):
            raise ValueError(f"Frequency must be between {FREQUENCY_MIN}-{FREQUENCY_MAX} Hz, got {frequency}")

        if not (COHERENCE_MIN <= coherence <= COHERENCE_MAX):
            raise ValueError(f"Coherence must be between {COHERENCE_MIN}-{COHERENCE_MAX}, got {coherence}")

        self._coordinates = ConsciousnessCoordinates(
            frequency=frequency,
            coherence=coherence,
            last_update_timestamp=int(time.time() * 1000)  # milliseconds
        )

    def get_frequency(self) -> float:
        """
        Returns current aggressiveness parameter (3-15, dimensionless).
        
        Note: Called "frequency" for historical/metaphorical reasons.
        This is NOT a physical frequency in Hz.
        """
        return self._coordinates.frequency

    def get_coherence(self) -> float:
        """
        Returns current consistency parameter (0.2-1.0, dimensionless).
        
        Note: Called "coherence" for historical/metaphorical reasons.
        This is NOT quantum coherence.
        """
        return self._coordinates.coherence

    def update_from_outcome(self, outcome: ConformationalOutcome) -> None:
        """
        Updates exploration parameters based on outcome.

        Uses HEURISTIC update rules (not derived from first principles) to
        adjust aggressiveness and consistency based on the outcome type.

        Update philosophy:
        - Success → Increase aggressiveness and consistency (reward)
        - Failure → Decrease aggressiveness and consistency (punishment)
        - These rules are EMPIRICAL, not theoretically derived

        Args:
            outcome: The result of a conformational exploration step
        """
        # Determine outcome type based on outcome characteristics
        outcome_type = self._classify_outcome(outcome)

        # Get update rules for this outcome type
        if outcome_type in CONSCIOUSNESS_UPDATE_RULES:
            updates = CONSCIOUSNESS_UPDATE_RULES[outcome_type]
            freq_delta = updates.get('frequency', 0.0)
            coh_delta = updates.get('coherence', 0.0)
        else:
            # Default: no change for unknown outcomes
            freq_delta = 0.0
            coh_delta = 0.0

        # Apply updates with bounds checking
        new_frequency = max(FREQUENCY_MIN, min(FREQUENCY_MAX, self._coordinates.frequency + freq_delta))
        new_coherence = max(COHERENCE_MIN, min(COHERENCE_MAX, self._coordinates.coherence + coh_delta))

        # Update coordinates
        self._coordinates.frequency = new_frequency
        self._coordinates.coherence = new_coherence
        self._coordinates.last_update_timestamp = int(time.time() * 1000)

    def get_coordinates(self) -> ConsciousnessCoordinates:
        """Get the current consciousness coordinates."""
        return self._coordinates

    def _classify_outcome(self, outcome: ConformationalOutcome) -> str:
        """
        Classify the outcome type based on its characteristics.

        This is a simplified classification - in a full implementation,
        this would analyze the conformational changes in detail.

        Args:
            outcome: The exploration outcome to classify

        Returns:
            String key for CONSCIOUSNESS_UPDATE_RULES
        """
        # Simplified classification based on energy change and success
        if outcome.energy_change < -100:
            return 'energy_decrease_large'
        elif outcome.energy_change < -50:
            return 'energy_decrease_small'
        elif outcome.energy_change > 50:
            return 'energy_increase'
        elif outcome.energy_change > 100:
            return 'structure_collapse'
        elif outcome.success and outcome.energy_change < -10:
            return 'stable_minimum_found'
        else:
            # Default case
            return 'energy_decrease_small'  # Assume small improvement