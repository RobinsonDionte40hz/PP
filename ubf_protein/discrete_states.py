"""
Discrete consciousness states module for UBF protein system.

This module implements discrete frequency states based on archive research
documenting the ionic/chemical frequency channels.

Archive Insight (THREE_CHANNEL_TYPES_SUMMARY.md):
- Chemical frequency channels are MASS-INDEPENDENT (quantum constants)
- Ca²⁺: 10 Hz, Mg²⁺: 7 Hz, Na⁺: 16 Hz, Zn²⁺: 40 Hz
- These frequencies appear universally because they're properties of ions themselves

For protein exploration (consciousness range 3-15):
- State 1: 4 Hz  (deep theta - minimal exploration)
- State 2: 7 Hz  (Mg²⁺ resonance - moderate exploration)
- State 3: 10 Hz (Ca²⁺ resonance - balanced exploration)
- State 4: 12 Hz (alpha peak - focused exploration)
- State 5: 15 Hz (max aggressiveness - maximum exploration)

The discrete states provide "attractor basins" that help agents escape local minima
by jumping between stable frequency modes rather than continuous interpolation.
"""

import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Discrete frequency states based on archive ionic frequencies
# Mapped to consciousness range 3-15 Hz
DISCRETE_FREQUENCY_STATES = [4.0, 7.0, 10.0, 12.0, 15.0]

# State names for logging/debugging
STATE_NAMES = {
    4.0: "theta_deep",      # Minimal exploration, consolidation
    7.0: "theta_Mg",        # Mg²⁺ resonance, moderate exploration
    10.0: "alpha_Ca",       # Ca²⁺ resonance, balanced exploration
    12.0: "alpha_peak",     # Alpha peak, focused exploration
    15.0: "beta_max",       # Maximum aggressiveness
}

# Information capacity per state (from THREE_CHANNEL_TYPES_SUMMARY.md)
# Based on N_states ∝ (bandwidth / min_resolution) × Q-factor
STATE_INFO_CAPACITY = {
    4.0: 8,     # Low frequency = fewer states but high Q
    7.0: 14,    # Mg²⁺ resonance = moderate states
    10.0: 20,   # Ca²⁺ resonance = balanced
    12.0: 24,   # Alpha = higher density
    15.0: 30,   # Max = maximum states but lower Q
}


@dataclass
class DiscreteStateTransition:
    """Record of a state transition."""
    from_state: float
    to_state: float
    reason: str
    energy_before: float
    energy_after: Optional[float] = None
    successful: bool = False


def snap_to_nearest_state(frequency: float) -> float:
    """
    Snap a continuous frequency to the nearest discrete state.
    
    Archive insight: Agents should operate at discrete resonant frequencies
    rather than arbitrary continuous values. This provides more stable
    exploration dynamics.
    
    Args:
        frequency: Continuous frequency value (3-15 range)
    
    Returns:
        Nearest discrete state frequency
    """
    min_dist = float('inf')
    nearest = DISCRETE_FREQUENCY_STATES[2]  # Default to 10 Hz (Ca²⁺)
    
    for state in DISCRETE_FREQUENCY_STATES:
        dist = abs(frequency - state)
        if dist < min_dist:
            min_dist = dist
            nearest = state
    
    return nearest


def get_state_index(frequency: float) -> int:
    """Get the index of the nearest discrete state (0-4)."""
    nearest = snap_to_nearest_state(frequency)
    return DISCRETE_FREQUENCY_STATES.index(nearest)


def get_adjacent_states(current_state: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the adjacent lower and higher states.
    
    Returns:
        Tuple of (lower_state, higher_state), None if at boundary
    """
    idx = get_state_index(current_state)
    
    lower = DISCRETE_FREQUENCY_STATES[idx - 1] if idx > 0 else None
    higher = DISCRETE_FREQUENCY_STATES[idx + 1] if idx < len(DISCRETE_FREQUENCY_STATES) - 1 else None
    
    return lower, higher


def jump_to_state(target_state: float) -> float:
    """
    Jump directly to a target discrete state.
    
    Unlike continuous frequency changes, this provides a discrete
    "quantum jump" between attractor basins.
    """
    return snap_to_nearest_state(target_state)


def select_escape_state(current_frequency: float, consecutive_stuck: int) -> float:
    """
    Select a new state for escaping local minima.
    
    Archive insight: When stuck, jump to a DIFFERENT discrete state
    rather than making small continuous adjustments. The further stuck,
    the larger the jump.
    
    Args:
        current_frequency: Current frequency (will be snapped to nearest state)
        consecutive_stuck: Number of consecutive stuck iterations
    
    Returns:
        New state frequency to jump to
    """
    current_state = snap_to_nearest_state(current_frequency)
    current_idx = get_state_index(current_state)
    
    # Calculate jump magnitude based on how stuck we are
    # 1-5 iterations: small jump (±1 state)
    # 6-10 iterations: medium jump (±2 states)
    # 10+ iterations: large jump (random state)
    
    if consecutive_stuck < 5:
        # Small jump: move to adjacent state
        lower, higher = get_adjacent_states(current_state)
        candidates = [s for s in [lower, higher] if s is not None]
        if candidates:
            return random.choice(candidates)
        return current_state
    
    elif consecutive_stuck < 10:
        # Medium jump: move 2 states away
        jump_dir = 1 if random.random() > 0.5 else -1
        new_idx = current_idx + (2 * jump_dir)
        new_idx = max(0, min(len(DISCRETE_FREQUENCY_STATES) - 1, new_idx))
        return DISCRETE_FREQUENCY_STATES[new_idx]
    
    else:
        # Large jump: random state different from current
        candidates = [s for s in DISCRETE_FREQUENCY_STATES if s != current_state]
        return random.choice(candidates)


def calculate_state_resonance_bonus(frequency: float, target_state: Optional[float] = None) -> float:
    """
    Calculate a bonus for being at or near a discrete state.
    
    Archive insight: Discrete states represent resonant frequencies
    with higher stability and information capacity.
    
    Args:
        frequency: Current frequency
        target_state: Optional target state (if None, uses nearest)
    
    Returns:
        Bonus multiplier (1.0 = at state, decreasing with distance)
    """
    if target_state is None:
        target_state = snap_to_nearest_state(frequency)
    
    distance = abs(frequency - target_state)
    
    # Gaussian decay from resonant state
    # Full bonus at state, 50% at 1 Hz away, 25% at 2 Hz away
    sigma = 1.0
    bonus = math.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    return bonus


def get_state_info_capacity(frequency: float) -> int:
    """
    Get the information capacity for a frequency state.
    
    Archive insight (THREE_CHANNEL_TYPES_SUMMARY.md):
    N_states = (bandwidth / min_resolution) × Q-factor
    
    Different states have different information capacities.
    """
    state = snap_to_nearest_state(frequency)
    return STATE_INFO_CAPACITY.get(state, 20)


def get_exploration_strategy(frequency: float) -> dict:
    """
    Get exploration strategy parameters based on discrete state.
    
    Different states have different optimal exploration strategies.
    """
    state = snap_to_nearest_state(frequency)
    
    strategies = {
        4.0: {  # theta_deep
            'risk_tolerance': 0.2,
            'exploration_radius': 'small',
            'memory_weight': 0.8,  # Rely heavily on memory
            'description': 'Conservative consolidation'
        },
        7.0: {  # theta_Mg
            'risk_tolerance': 0.4,
            'exploration_radius': 'medium',
            'memory_weight': 0.6,
            'description': 'Moderate exploration (Mg²⁺ resonance)'
        },
        10.0: {  # alpha_Ca
            'risk_tolerance': 0.5,
            'exploration_radius': 'medium',
            'memory_weight': 0.5,  # Balanced
            'description': 'Balanced exploration (Ca²⁺ resonance)'
        },
        12.0: {  # alpha_peak
            'risk_tolerance': 0.6,
            'exploration_radius': 'large',
            'memory_weight': 0.4,
            'description': 'Focused aggressive exploration'
        },
        15.0: {  # beta_max
            'risk_tolerance': 0.8,
            'exploration_radius': 'very_large',
            'memory_weight': 0.2,  # More independent
            'description': 'Maximum exploration (boundary probing)'
        }
    }
    
    return strategies.get(state, strategies[10.0])


class DiscreteStateManager:
    """
    Manager for tracking and controlling discrete state transitions.
    
    This class maintains state transition history and provides
    intelligent state selection based on exploration outcomes.
    """
    
    def __init__(self, initial_frequency: float = 10.0, enable_discrete: bool = True):
        """
        Initialize discrete state manager.
        
        Args:
            initial_frequency: Starting frequency (will be snapped if enable_discrete)
            enable_discrete: If True, use discrete states; if False, continuous
        """
        self._enable_discrete = enable_discrete
        self._current_state = snap_to_nearest_state(initial_frequency) if enable_discrete else initial_frequency
        self._transition_history: List[DiscreteStateTransition] = []
        self._state_performance: dict = {s: [] for s in DISCRETE_FREQUENCY_STATES}
        self._consecutive_stuck = 0
    
    @property
    def current_state(self) -> float:
        return self._current_state
    
    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(snap_to_nearest_state(self._current_state), "unknown")
    
    def update_frequency(self, new_frequency: float, reason: str = "update") -> float:
        """
        Update frequency, snapping to discrete state if enabled.
        
        Returns:
            The actual new frequency (snapped if discrete mode)
        """
        if self._enable_discrete:
            new_state = snap_to_nearest_state(new_frequency)
        else:
            new_state = new_frequency
        
        if new_state != self._current_state:
            transition = DiscreteStateTransition(
                from_state=self._current_state,
                to_state=new_state,
                reason=reason,
                energy_before=0.0  # Will be filled by caller
            )
            self._transition_history.append(transition)
        
        self._current_state = new_state
        return self._current_state
    
    def record_stuck(self) -> None:
        """Record that agent is stuck in current state."""
        self._consecutive_stuck += 1
    
    def record_progress(self, energy_improvement: float) -> None:
        """Record that agent made progress."""
        self._consecutive_stuck = 0
        state = snap_to_nearest_state(self._current_state)
        self._state_performance[state].append(energy_improvement)
    
    def suggest_escape_state(self) -> float:
        """
        Suggest a new state for escaping local minima.
        
        Uses history to prefer states that have worked well before.
        """
        if not self._enable_discrete:
            # In continuous mode, just increase frequency
            return min(15.0, self._current_state + 1.0)
        
        # Get base suggestion from escape algorithm
        base_suggestion = select_escape_state(self._current_state, self._consecutive_stuck)
        
        # If we have performance history, bias toward better states
        if len(self._transition_history) > 5:
            best_state = max(
                DISCRETE_FREQUENCY_STATES,
                key=lambda s: sum(self._state_performance[s]) / max(len(self._state_performance[s]), 1)
            )
            
            # 30% chance to jump to historically best state
            if random.random() < 0.3 and best_state != self._current_state:
                return best_state
        
        return base_suggestion
    
    def get_statistics(self) -> dict:
        """Get statistics about state transitions and performance."""
        return {
            'current_state': self._current_state,
            'state_name': self.state_name,
            'total_transitions': len(self._transition_history),
            'consecutive_stuck': self._consecutive_stuck,
            'state_performance': {
                STATE_NAMES.get(s, str(s)): {
                    'visits': len(self._state_performance[s]),
                    'avg_improvement': sum(self._state_performance[s]) / max(len(self._state_performance[s]), 1)
                }
                for s in DISCRETE_FREQUENCY_STATES
            }
        }
