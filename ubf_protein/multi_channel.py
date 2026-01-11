"""
Multi-channel energy exploration for UBF protein system.

This module implements parallel energy evaluation using the three-channel pattern
from archive research: voltage, current, and resistance channels.

Archive Insight (THREE_CHANNEL_TYPES_SUMMARY.md):
- Type 1 (Chemical): Mass-independent, ionic resonances (10 Hz, 40 Hz)
- Type 2 (Acoustic): f ∝ M^(-1/3), mechanical modes
- Type 3 (Quantum): f = ΔE/ℏ, energy-determined

For protein exploration, we map these to:
- Voltage Channel (Type 3): Total energy (potential energy landscape)
- Current Channel (Type 2): Energy flow/gradient (exploration direction)
- Resistance Channel (Type 1): Structural stability (impedance to change)

Cross-channel coupling enables more effective exploration by:
1. Using voltage (energy) for absolute quality
2. Using current (gradient) for direction
3. Using resistance (stability) to avoid unstable states

Information Capacity:
N_states = (bandwidth / min_resolution) × Q-factor
Higher capacity = more information per evaluation
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ChannelReading:
    """Reading from a single energy channel."""
    channel_type: str  # 'voltage', 'current', 'resistance'
    value: float
    uncertainty: float
    q_factor: float
    timestamp: int = 0


@dataclass
class MultiChannelScore:
    """Combined score from all three channels."""
    voltage_score: float   # Total energy quality (lower = better)
    current_score: float   # Gradient quality (steeper descent = better)
    resistance_score: float  # Stability quality (moderate = better)
    
    # Combined scores
    combined_score: float  # Weighted combination
    information_content: float  # Bits of information
    
    # Channel coupling
    coupling_strength: float  # V-I-R coupling factor
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'voltage': self.voltage_score,
            'current': self.current_score,
            'resistance': self.resistance_score,
            'combined': self.combined_score,
            'information': self.information_content,
            'coupling': self.coupling_strength
        }


class VoltageChannel:
    """
    Voltage channel - measures total energy landscape.
    
    Type 3 (Quantum-like): Energy-determined evaluation
    Lower voltage = lower energy = better conformation
    """
    
    def __init__(self, q_factor: float = 100.0):
        self._q_factor = q_factor
        self._baseline = 0.0  # Will be set from first reading
        self._readings: List[float] = []
        self._min_reading = float('inf')
        self._max_reading = float('-inf')
    
    def evaluate(self, energy: float) -> ChannelReading:
        """
        Evaluate conformation energy as voltage.
        
        Normalized to 0-1 range where 0 = best energy seen
        """
        # Track min/max for normalization
        if energy < self._min_reading:
            self._min_reading = energy
        if energy > self._max_reading:
            self._max_reading = energy
        
        self._readings.append(energy)
        
        # Normalize to 0-1 (0 = best)
        energy_range = self._max_reading - self._min_reading
        if energy_range > 0:
            normalized = (energy - self._min_reading) / energy_range
        else:
            normalized = 0.5
        
        # Calculate uncertainty based on Q-factor
        # Higher Q = lower uncertainty
        uncertainty = 1.0 / self._q_factor
        
        return ChannelReading(
            channel_type='voltage',
            value=normalized,
            uncertainty=uncertainty,
            q_factor=self._q_factor
        )


class CurrentChannel:
    """
    Current channel - measures energy gradient/flow.
    
    Type 2 (Acoustic-like): Rate of change evaluation
    Negative current (downhill) = good exploration direction
    """
    
    def __init__(self, q_factor: float = 50.0, window_size: int = 5):
        self._q_factor = q_factor
        self._window_size = window_size
        self._energy_history: List[float] = []
    
    def evaluate(self, energy: float) -> ChannelReading:
        """
        Evaluate energy gradient as current.
        
        Negative = downhill (good)
        Positive = uphill (exploring)
        Zero = stuck
        """
        self._energy_history.append(energy)
        
        # Keep only recent history
        if len(self._energy_history) > self._window_size:
            self._energy_history = self._energy_history[-self._window_size:]
        
        # Calculate gradient (current)
        if len(self._energy_history) >= 2:
            # Use linear regression slope for smoother estimate
            n = len(self._energy_history)
            x_mean = (n - 1) / 2
            y_mean = sum(self._energy_history) / n
            
            numerator = sum(
                (i - x_mean) * (e - y_mean) 
                for i, e in enumerate(self._energy_history)
            )
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator > 0:
                gradient = numerator / denominator
            else:
                gradient = 0.0
        else:
            gradient = 0.0
        
        # Normalize gradient (-1 to 1, negative = good)
        # Use tanh for smooth normalization
        normalized = math.tanh(gradient / 100.0)  # Scale factor for typical energies
        
        uncertainty = 1.0 / self._q_factor
        
        return ChannelReading(
            channel_type='current',
            value=normalized,
            uncertainty=uncertainty,
            q_factor=self._q_factor
        )


class ResistanceChannel:
    """
    Resistance channel - measures structural stability/impedance.
    
    Type 1 (Chemical-like): Stability evaluation
    Moderate resistance = stable but explorable
    Too low = unstable, too high = stuck
    """
    
    def __init__(self, q_factor: float = 1000.0):
        self._q_factor = q_factor
        self._stability_history: List[float] = []
    
    def evaluate(self, energy: float, variance: float = 0.0, 
                 structural_quality: float = 0.5) -> ChannelReading:
        """
        Evaluate structural resistance/stability.
        
        Args:
            energy: Current energy
            variance: Energy variance over recent history
            structural_quality: Quality metric (0-1) from validation
        
        Returns:
            Resistance reading (0 = unstable, 0.5 = optimal, 1 = stuck)
        """
        # Calculate resistance as combination of variance and quality
        # Low variance + high quality = high resistance (stuck in good state)
        # High variance + low quality = low resistance (unstable)
        # Moderate = optimal for exploration
        
        # Normalize variance (assume typical range 0-1000)
        norm_variance = 1.0 - math.exp(-variance / 100.0)
        
        # Combine factors
        # Optimal resistance is around 0.5
        stability = 0.3 * (1.0 - norm_variance) + 0.7 * structural_quality
        
        self._stability_history.append(stability)
        
        # Use Q-factor for uncertainty
        uncertainty = 1.0 / self._q_factor
        
        return ChannelReading(
            channel_type='resistance',
            value=stability,
            uncertainty=uncertainty,
            q_factor=self._q_factor
        )


class MultiChannelEvaluator:
    """
    Multi-channel energy evaluator combining V-I-R channels.
    
    Uses impedance matching from archive research to combine channels
    effectively. Cross-channel coupling provides more information than
    any single channel alone.
    """
    
    def __init__(self, 
                 voltage_weight: float = 0.5,
                 current_weight: float = 0.3,
                 resistance_weight: float = 0.2):
        """
        Initialize multi-channel evaluator.
        
        Args:
            voltage_weight: Weight for energy quality (default 0.5)
            current_weight: Weight for gradient quality (default 0.3)
            resistance_weight: Weight for stability (default 0.2)
        """
        self._voltage = VoltageChannel(q_factor=100.0)
        self._current = CurrentChannel(q_factor=50.0, window_size=5)
        self._resistance = ResistanceChannel(q_factor=1000.0)
        
        # Normalize weights
        total = voltage_weight + current_weight + resistance_weight
        self._v_weight = voltage_weight / total
        self._i_weight = current_weight / total
        self._r_weight = resistance_weight / total
        
        # History for information calculation
        self._evaluation_count = 0
    
    def evaluate(self, energy: float, 
                 energy_variance: float = 0.0,
                 structural_quality: float = 0.5) -> MultiChannelScore:
        """
        Evaluate conformation using all three channels.
        
        Args:
            energy: Total energy of conformation
            energy_variance: Variance over recent history
            structural_quality: Structural validation score (0-1)
        
        Returns:
            MultiChannelScore with combined evaluation
        """
        self._evaluation_count += 1
        
        # Get individual channel readings
        v_reading = self._voltage.evaluate(energy)
        i_reading = self._current.evaluate(energy)
        r_reading = self._resistance.evaluate(energy, energy_variance, structural_quality)
        
        # Calculate coupling strength (impedance matching)
        # Strong coupling when all channels agree
        v_i_coupling = 1.0 - abs(v_reading.value - (-i_reading.value))  # V and -I should align
        i_r_coupling = 1.0 - abs((i_reading.value + 1) / 2 - (1 - r_reading.value))  # Complex coupling
        
        coupling_strength = (v_i_coupling + i_r_coupling) / 2
        
        # Combined score (lower = better)
        # Voltage: 0 = best energy
        # Current: negative = good direction
        # Resistance: 0.5 = optimal
        
        voltage_score = v_reading.value
        current_score = (i_reading.value + 1) / 2  # Normalize to 0-1
        resistance_score = abs(r_reading.value - 0.5) * 2  # Distance from optimal
        
        combined = (
            self._v_weight * voltage_score +
            self._i_weight * current_score +
            self._r_weight * resistance_score
        )
        
        # Calculate information content
        # Higher Q-factor and coupling = more information
        avg_q = (v_reading.q_factor + i_reading.q_factor + r_reading.q_factor) / 3
        information = math.log2(avg_q) * coupling_strength
        
        return MultiChannelScore(
            voltage_score=voltage_score,
            current_score=current_score,
            resistance_score=resistance_score,
            combined_score=combined,
            information_content=information,
            coupling_strength=coupling_strength
        )
    
    def get_exploration_guidance(self) -> Dict[str, Any]:
        """
        Get guidance for next exploration step based on channel states.
        
        Returns:
            Dictionary with guidance parameters
        """
        # Analyze channel states
        v_len = len(self._voltage._readings)
        
        if v_len < 3:
            return {
                'action': 'explore',
                'confidence': 0.3,
                'reason': 'Insufficient data for guidance'
            }
        
        # Recent voltage trend
        recent_v = self._voltage._readings[-3:]
        v_trend = recent_v[-1] - recent_v[0]
        
        # Recent current (gradient)
        recent_i = self._current._energy_history
        avg_current = sum(recent_i) / len(recent_i) if recent_i else 0
        
        if v_trend < 0 and avg_current < 0:
            return {
                'action': 'continue',
                'confidence': 0.8,
                'reason': 'Downhill progress detected'
            }
        elif v_trend > 0 and avg_current > 0:
            return {
                'action': 'jump',
                'confidence': 0.6,
                'reason': 'Uphill - consider large move'
            }
        else:
            return {
                'action': 'explore',
                'confidence': 0.5,
                'reason': 'Mixed signals - explore alternatives'
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            'evaluations': self._evaluation_count,
            'voltage': {
                'min': self._voltage._min_reading,
                'max': self._voltage._max_reading,
                'readings': len(self._voltage._readings)
            },
            'current': {
                'history_size': len(self._current._energy_history)
            },
            'resistance': {
                'history_size': len(self._resistance._stability_history)
            }
        }


def calculate_information_capacity(bandwidth: float, resolution: float, q_factor: float) -> float:
    """
    Calculate information capacity of a frequency channel.
    
    Archive formula (THREE_CHANNEL_TYPES_SUMMARY.md):
    N_states = (bandwidth / min_resolution) × Q-factor
    
    Args:
        bandwidth: Frequency bandwidth (Hz)
        resolution: Minimum frequency resolution (Hz)
        q_factor: Quality factor of the channel
    
    Returns:
        Information capacity in bits
    """
    n_states = (bandwidth / max(resolution, 0.001)) * q_factor
    return math.log2(max(n_states, 1))


def calculate_bit_density(information_capacity: float, volume: float) -> float:
    """
    Calculate bit density for a region.
    
    Args:
        information_capacity: Total bits of information
        volume: Volume of region (Å³)
    
    Returns:
        Bit density (bits/Å³)
    """
    return information_capacity / max(volume, 0.001)
