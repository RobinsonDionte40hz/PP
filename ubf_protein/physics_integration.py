"""
Physics integration module for UBF protein system.

This module provides physics calculator adapters that wrap existing
physics modules to implement the standardized interfaces.
"""

import math
from typing import Optional, List

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try package-relative imports first
    from .interfaces import IQAAPCalculator, IResonanceCoupling, IWaterShielding
    from .models import Conformation
except ImportError:
    # Fall back to absolute imports from ubf_protein package
    from ubf_protein.interfaces import IQAAPCalculator, IResonanceCoupling, IWaterShielding
    from ubf_protein.models import Conformation


def _euclidean_distance(coord1: tuple, coord2: tuple) -> float:
    """
    Calculate Euclidean distance between two 3D coordinates.
    
    Args:
        coord1: First coordinate (x, y, z)
        coord2: Second coordinate (x, y, z)
    
    Returns:
        Distance
    """
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    dz = coord2[2] - coord1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _vector_norm(coord: tuple) -> float:
    """
    Calculate the Euclidean norm of a vector/coordinate.
    
    Args:
        coord: Coordinate (x, y, z)
    
    Returns:
        Norm (magnitude) of the vector
    """
    return math.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2)


def _mean(values: list) -> float:
    """
    Calculate the mean of a list of values.
    
    Args:
        values: List of numerical values
    
    Returns:
        Mean value
    """
    return sum(values) / len(values) if values else 0.0


class QAAPCalculator(IQAAPCalculator):
    """
    Adapter for Quantum Amino Acid Potential (QAAP) calculator.

    Wraps the existing QCP calculation from QuantumCoherenceProteinPredictor.
    """

    def __init__(self):
        """Initialize QAAP calculator."""
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.base_energy = 4.0

    def calculate(self, conformation: Conformation) -> float:
        """Calculate physics-based score/energy."""
        return self.calculate_qaap_potential(conformation)

    def calculate_qaap_potential(self, conformation: Conformation) -> float:
        """
        Calculate quantum potential: QCP = 4 + (2^n × φ^l × m)

        This is the core quantum amino acid potential calculation.

        Args:
            conformation: The protein conformation to analyze

        Returns:
            Average QAAP potential across all residues
        """
        if not conformation.atom_coordinates:
            return 0.0

        # Type hints for PyPy JIT optimization
        qcp_values: List[float] = []
        n: int
        neighbors: int
        l: int
        m: float
        qcp: float
        dist: float

        for i, (coord, ss_type) in enumerate(zip(conformation.atom_coordinates,
                                                conformation.secondary_structure)):
            # Determine structural hierarchy level (n)
            if ss_type == 'H':  # Helix
                n = 1
            elif ss_type == 'E':  # Sheet
                n = 2
            else:  # Coil/other
                n = 0

            # Calculate neighbor count (l) - simplified
            neighbors = 0
            for j, other_coord in enumerate(conformation.atom_coordinates):
                if i != j:
                    dist = _euclidean_distance(coord, other_coord)
                    if dist < 8.0:  # Within 8Å
                        neighbors += 1

            l = min(max(1, neighbors // 3), 3)  # Scale to 1-3

            # Calculate hydrophobicity factor (m) - simplified
            # This would normally use residue-specific hydrophobicity
            # For now, use a simple approximation based on position
            hydrophobicity_scale: float = math.sin(i * 0.5)  # Pseudo-random between -1 and 1
            m = hydrophobicity_scale

            # Calculate QCP for this residue
            qcp = self.base_energy + (2**n * (self.phi**l) * m)
            qcp_values.append(qcp)

        # Return average QAAP potential
        return _mean(qcp_values)


class ResonanceCouplingCalculator(IResonanceCoupling):
    """
    Adapter for 40 Hz gamma resonance coupling calculator.

    Implements the resonance coupling formula: R(E₁,E₂) = exp[-(E₁ - E₂ - ℏω_γ)²/(2ℏω_γ)]
    """

    def __init__(self, gamma_frequency_hz: float = 40.0):
        """
        Initialize resonance coupling calculator.

        Args:
            gamma_frequency_hz: Gamma oscillation frequency in Hz
        """
        self.gamma_frequency_hz = gamma_frequency_hz
        self.plank_reduced = 1.0545718e-34  # Reduced Planck's constant (J⋅s)
        self.h_gamma = self.plank_reduced * gamma_frequency_hz

    def calculate(self, conformation: Conformation) -> float:
        """Calculate physics-based score/energy."""
        # For a single conformation, calculate average resonance coupling
        return self._calculate_average_resonance(conformation)

    def calculate_resonance(self, residue1: int, residue2: int, conformation: Conformation) -> float:
        """
        Calculate resonance coupling between two residues.

        Args:
            residue1: Index of first residue
            residue2: Index of second residue
            conformation: Protein conformation

        Returns:
            Resonance coupling strength (0-1)
        """
        if (residue1 >= len(conformation.atom_coordinates) or
            residue2 >= len(conformation.atom_coordinates)):
            return 0.0

        # For now, use simplified energy approximation
        # In a real implementation, this would use actual energy calculations
        energy1 = _vector_norm(conformation.atom_coordinates[residue1])
        energy2 = _vector_norm(conformation.atom_coordinates[residue2])

        return self._resonance_coupling(energy1, energy2)

    def _resonance_coupling(self, energy1: float, energy2: float) -> float:
        """
        Calculate resonance coupling strength between two energy states.

        R(E₁,E₂) = exp[-(E₁ - E₂ - ℏω_γ)²/(2ℏω_γ)]

        Args:
            energy1: Energy of first state
            energy2: Energy of second state

        Returns:
            Coupling strength (0-1)
        """
        energy_diff = abs(energy1 - energy2)
        coupling = math.exp(-(energy_diff - self.h_gamma)**2 / (2 * self.h_gamma))

        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, coupling))

    def _calculate_average_resonance(self, conformation: Conformation) -> float:
        """
        Calculate average resonance coupling across the conformation.

        Args:
            conformation: Protein conformation

        Returns:
            Average resonance coupling strength
        """
        if len(conformation.atom_coordinates) < 2:
            return 0.0

        couplings = []
        n_residues = len(conformation.atom_coordinates)

        # Sample a subset of residue pairs to avoid O(n²) complexity
        sample_size = min(100, n_residues * (n_residues - 1) // 2)
        pairs_checked = 0

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                if pairs_checked >= sample_size:
                    break
                coupling = self.calculate_resonance(i, j, conformation)
                couplings.append(coupling)
                pairs_checked += 1
            if pairs_checked >= sample_size:
                break

        return _mean(couplings)


class WaterShieldingCalculator(IWaterShielding):
    """
    Adapter for water shielding effects calculator.

    Implements water shielding with 408 fs coherence time and 3.57 nm⁻¹ factor.
    """

    def __init__(self,
                 coherence_time_fs: float = 408.0,
                 shielding_factor: float = 3.57):
        """
        Initialize water shielding calculator.

        Args:
            coherence_time_fs: Water coherence time in femtoseconds
            shielding_factor: Water shielding factor in nm⁻¹
        """
        self.coherence_time_fs = coherence_time_fs
        self.shielding_factor = shielding_factor

    def calculate(self, conformation: Conformation) -> float:
        """Calculate physics-based score/energy."""
        return self.calculate_shielding(conformation)

    def calculate_shielding(self, conformation: Conformation) -> float:
        """
        Calculate water shielding effects.

        Uses the formula incorporating coherence time and shielding factor.

        Args:
            conformation: Protein conformation

        Returns:
            Water shielding factor (0-1, higher = better shielding)
        """
        if not conformation.atom_coordinates:
            return 0.0

        # Calculate solvent accessible surface area approximation
        # This is a simplified calculation - in reality would use proper SASA calculation
        total_shielding = 0.0
        n_residues = len(conformation.atom_coordinates)

        for i, coord in enumerate(conformation.atom_coordinates):
            # Count nearby residues (simulating buried vs exposed)
            nearby_count = 0
            for j, other_coord in enumerate(conformation.atom_coordinates):
                if i != j:
                    dist = _euclidean_distance(coord, other_coord)
                    if dist < 8.0:  # Within 8Å
                        nearby_count += 1

            # Higher nearby count = more buried = better shielding
            local_shielding = min(1.0, nearby_count / 10.0)
            total_shielding += local_shielding

        # Average shielding across all residues
        avg_shielding = total_shielding / n_residues if n_residues > 0 else 0.0

        # Apply coherence time and shielding factor
        # This is a simplified model - real implementation would be more complex
        coherence_factor = math.exp(-self.coherence_time_fs / 1000.0)  # Decay with time
        shielding_effect = avg_shielding * coherence_factor * (self.shielding_factor / 10.0)

        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, shielding_effect))