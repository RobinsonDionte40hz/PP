"""
Entropic corrections for protein folding free energy.

This module implements entropic contributions to the energy function:
- Coherence entropy: From variance in quantum coherence pattern (QCP values)
- Configurational entropy: From structural diversity in conformation ensemble
- Temperature-dependent free energy: ΔG = ΔH - TΔS at 300K

These entropy terms capture disorder and ensemble effects that pure
enthalpy-based scoring misses.
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try package-relative imports first
    from .models import Conformation
except ImportError:
    # Fall back to absolute imports from ubf_protein package
    from ubf_protein.models import Conformation


@dataclass
class EntropicContributions:
    """
    Container for entropic energy contributions.
    
    Attributes:
        coherence_entropy: Entropy from QCP variance (kcal/mol at 300K)
        configurational_entropy: Entropy from RMSD diversity (kcal/mol at 300K)
        total_entropic_energy: Sum of all entropic contributions (kcal/mol)
        qcp_variance: Raw variance of QCP values
        avg_pairwise_rmsd: Average RMSD among recent conformations (Å)
        n_conformations: Number of conformations used for configurational entropy
    """
    coherence_entropy: float
    configurational_entropy: float
    total_entropic_energy: float
    qcp_variance: float
    avg_pairwise_rmsd: float
    n_conformations: int


class EntropicCalculator:
    """
    Calculator for entropic contributions to protein folding free energy.
    
    Implements two entropy terms:
    
    1. Coherence Entropy: Measures disorder in quantum coherence pattern
       - Computed from variance of QCP (Quantum Coherence Pattern) values
       - Higher variance = more disorder = higher entropy
       - S_coh = k_B * ln(1 + σ²_QCP)
       
    2. Configurational Entropy: Measures structural diversity in ensemble
       - Computed from RMSD diversity among recent conformations
       - Higher RMSD diversity = more accessible states = higher entropy
       - S_conf = k_B * ln(1 + <RMSD>)
    
    Free energy contributions are computed using: ΔG = -T * ΔS
    At 300K, entropy contributions are significant for ensemble sampling.
    
    Attributes:
        boltzmann_constant: k_B in kcal/(mol·K) (default 0.001987)
        temperature: Temperature in Kelvin (default 300.0)
        max_variance: Maximum QCP variance for normalization (default 10.0)
        window_size: Number of recent conformations to consider (default 50)
        min_conformations: Minimum conformations needed for config entropy (default 2)
    
    Example:
        >>> calc = EntropicCalculator()
        >>> qcp_values = [4.0, 4.2, 3.8, 4.5, 4.1]
        >>> contributions = calc.calculate_entropic_contributions(
        ...     qcp_values=qcp_values,
        ...     recent_conformations=[conf1, conf2, conf3]
        ... )
        >>> contributions.total_entropic_energy  # In kcal/mol
        -2.5
    """
    
    def __init__(self,
                 boltzmann_constant: float = 0.001987,
                 temperature: float = 300.0,
                 max_variance: float = 10.0,
                 window_size: int = 50,
                 min_conformations: int = 2):
        """
        Initialize entropic calculator.
        
        Args:
            boltzmann_constant: k_B in kcal/(mol·K) (default 0.001987)
            temperature: Temperature in Kelvin (default 300.0)
            max_variance: Maximum QCP variance for normalization (default 10.0)
            window_size: Number of recent conformations for diversity (default 50)
            min_conformations: Minimum conformations needed (default 2)
        """
        if boltzmann_constant <= 0:
            raise ValueError(f"boltzmann_constant must be positive, got {boltzmann_constant}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if max_variance <= 0:
            raise ValueError(f"max_variance must be positive, got {max_variance}")
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if min_conformations < 2:
            raise ValueError(f"min_conformations must be >= 2, got {min_conformations}")
        
        self.boltzmann_constant = boltzmann_constant
        self.temperature = temperature
        self.max_variance = max_variance
        self.window_size = window_size
        self.min_conformations = min_conformations
        
        # Cache thermal energy: k_B * T
        self._kt = boltzmann_constant * temperature
    
    def calculate_qcp_variance(self, qcp_values: List[float]) -> float:
        """
        Calculate variance of QCP values across protein structure.
        
        High variance indicates disorder in quantum coherence pattern.
        Variance is normalized to max_variance for numerical stability.
        
        Args:
            qcp_values: List of QCP values for each residue
            
        Returns:
            Variance of QCP values (normalized to max_variance)
            
        Raises:
            ValueError: If qcp_values is empty or has invalid values
        """
        if not qcp_values:
            raise ValueError("qcp_values cannot be empty")
        
        if len(qcp_values) == 1:
            return 0.0  # Single value has zero variance
        
        # Calculate mean
        n = len(qcp_values)
        mean = sum(qcp_values) / n
        
        # Calculate variance: σ² = Σ(x - μ)² / n
        variance = sum((x - mean) ** 2 for x in qcp_values) / n
        
        # Normalize to max_variance for stability
        normalized_variance = min(variance, self.max_variance)
        
        return normalized_variance
    
    def calculate_coherence_entropy(self, qcp_values: List[float]) -> Tuple[float, float]:
        """
        Calculate coherence entropy from QCP variance.
        
        Uses logarithmic relationship between variance and entropy:
        S_coherence = k_B * ln(1 + σ²_QCP)
        
        The "+1" ensures entropy is zero when variance is zero (perfect order).
        Free energy contribution: ΔG = -T * S
        
        Args:
            qcp_values: List of QCP values for each residue
            
        Returns:
            Tuple of (entropy in kcal/(mol·K), free energy contribution in kcal/mol)
            
        Example:
            >>> calc = EntropicCalculator()
            >>> entropy, free_energy = calc.calculate_coherence_entropy([4.0, 4.2, 3.8])
            >>> entropy > 0  # Positive entropy
            True
            >>> free_energy < 0  # Negative free energy (favorable)
            True
        """
        if not qcp_values:
            return 0.0, 0.0
        
        variance = self.calculate_qcp_variance(qcp_values)
        
        # Entropy: S = k_B * ln(1 + σ²)
        # Using ln(1 + x) ensures S=0 when σ²=0
        entropy = self.boltzmann_constant * math.log(1.0 + variance)
        
        # Free energy: ΔG = -T * S (negative means favorable)
        free_energy = -self.temperature * entropy
        
        return entropy, free_energy
    
    def calculate_pairwise_rmsd(self,
                                conf1: Conformation,
                                conf2: Conformation) -> float:
        """
        Calculate RMSD between two conformations.
        
        Uses standard RMSD formula:
        RMSD = sqrt(Σ(r_i - r_j)² / N)
        
        Args:
            conf1: First conformation
            conf2: Second conformation
            
        Returns:
            RMSD in Angstroms
            
        Raises:
            ValueError: If conformations have different lengths
        """
        coords1 = conf1.atom_coordinates
        coords2 = conf2.atom_coordinates
        
        if len(coords1) != len(coords2):
            raise ValueError(
                f"Conformations must have same length: {len(coords1)} vs {len(coords2)}"
            )
        
        if len(coords1) == 0:
            return 0.0
        
        # Calculate sum of squared distances
        sum_sq_dist = 0.0
        for c1, c2 in zip(coords1, coords2):
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dz = c2[2] - c1[2]
            sum_sq_dist += dx**2 + dy**2 + dz**2
        
        # RMSD = sqrt(mean squared distance)
        rmsd = math.sqrt(sum_sq_dist / len(coords1))
        
        return rmsd
    
    def calculate_average_pairwise_rmsd(self,
                                       conformations: List[Conformation]) -> float:
        """
        Calculate average RMSD among all pairs of conformations.
        
        This measures the structural diversity of the ensemble.
        Higher average RMSD indicates more diverse conformations.
        
        Args:
            conformations: List of conformations (uses up to window_size most recent)
            
        Returns:
            Average pairwise RMSD in Angstroms
            
        Example:
            >>> calc = EntropicCalculator(window_size=50)
            >>> conformations = [conf1, conf2, conf3, ...]
            >>> avg_rmsd = calc.calculate_average_pairwise_rmsd(conformations)
        """
        if len(conformations) < 2:
            return 0.0
        
        # Use only most recent conformations (up to window_size)
        recent = conformations[-self.window_size:]
        
        # Calculate all pairwise RMSDs
        total_rmsd = 0.0
        n_pairs = 0
        
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                rmsd = self.calculate_pairwise_rmsd(recent[i], recent[j])
                total_rmsd += rmsd
                n_pairs += 1
        
        # Average over all pairs
        if n_pairs == 0:
            return 0.0
        
        return total_rmsd / n_pairs
    
    def calculate_configurational_entropy(self,
                                         conformations: List[Conformation]) -> Tuple[float, float, float]:
        """
        Calculate configurational entropy from ensemble diversity.
        
        Uses logarithmic relationship with average RMSD:
        S_config = k_B * ln(1 + <RMSD>)
        
        Higher RMSD diversity indicates more accessible conformational states,
        thus higher entropy. This captures the entropic benefit of sampling
        diverse conformations.
        
        Args:
            conformations: List of recent conformations
            
        Returns:
            Tuple of (entropy in kcal/(mol·K), 
                     free energy contribution in kcal/mol,
                     average pairwise RMSD in Å)
            
        Example:
            >>> calc = EntropicCalculator()
            >>> entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
            >>> entropy > 0  # Higher diversity = higher entropy
            True
        """
        if len(conformations) < self.min_conformations:
            return 0.0, 0.0, 0.0
        
        # Calculate average pairwise RMSD
        avg_rmsd = self.calculate_average_pairwise_rmsd(conformations)
        
        # Entropy: S = k_B * ln(1 + <RMSD>)
        # Using ln(1 + x) ensures S=0 when RMSD=0 (no diversity)
        entropy = self.boltzmann_constant * math.log(1.0 + avg_rmsd)
        
        # Free energy: ΔG = -T * S (negative means favorable)
        free_energy = -self.temperature * entropy
        
        return entropy, free_energy, avg_rmsd
    
    def calculate_entropic_contributions(self,
                                        qcp_values: Optional[List[float]] = None,
                                        recent_conformations: Optional[List[Conformation]] = None) -> EntropicContributions:
        """
        Calculate all entropic contributions.
        
        Combines coherence entropy and configurational entropy into
        total entropic free energy contribution. Either or both can
        be provided; missing terms contribute zero.
        
        Args:
            qcp_values: QCP values for coherence entropy (optional)
            recent_conformations: Recent conformations for configurational entropy (optional)
            
        Returns:
            EntropicContributions object with all entropy terms
            
        Example:
            >>> calc = EntropicCalculator()
            >>> contributions = calc.calculate_entropic_contributions(
            ...     qcp_values=[4.0, 4.2, 3.8, 4.5],
            ...     recent_conformations=[conf1, conf2, conf3]
            ... )
            >>> contributions.total_entropic_energy  # Total ΔG in kcal/mol
            -3.2
        """
        # Calculate coherence entropy
        if qcp_values is not None and len(qcp_values) > 0:
            coh_entropy, coh_free_energy = self.calculate_coherence_entropy(qcp_values)
            qcp_var = self.calculate_qcp_variance(qcp_values)
        else:
            coh_entropy = 0.0
            coh_free_energy = 0.0
            qcp_var = 0.0
        
        # Calculate configurational entropy
        if recent_conformations is not None and len(recent_conformations) >= self.min_conformations:
            conf_entropy, conf_free_energy, avg_rmsd = self.calculate_configurational_entropy(
                recent_conformations
            )
            n_conf = len(recent_conformations)
        else:
            conf_entropy = 0.0
            conf_free_energy = 0.0
            avg_rmsd = 0.0
            n_conf = 0 if recent_conformations is None else len(recent_conformations)
        
        # Total entropic free energy
        total_entropic_energy = coh_free_energy + conf_free_energy
        
        return EntropicContributions(
            coherence_entropy=coh_entropy,
            configurational_entropy=conf_entropy,
            total_entropic_energy=total_entropic_energy,
            qcp_variance=qcp_var,
            avg_pairwise_rmsd=avg_rmsd,
            n_conformations=n_conf
        )
    
    def get_free_energy_contribution(self,
                                    qcp_values: Optional[List[float]] = None,
                                    recent_conformations: Optional[List[Conformation]] = None) -> float:
        """
        Get total entropic free energy contribution.
        
        Convenience method that returns just the total ΔG value.
        
        Args:
            qcp_values: QCP values for coherence entropy (optional)
            recent_conformations: Recent conformations for configurational entropy (optional)
            
        Returns:
            Total entropic free energy in kcal/mol (negative = favorable)
            
        Example:
            >>> calc = EntropicCalculator()
            >>> delta_g = calc.get_free_energy_contribution(qcp_values=[4.0, 4.2, 3.8])
            >>> delta_g < 0  # Entropy favors exploration
            True
        """
        contributions = self.calculate_entropic_contributions(qcp_values, recent_conformations)
        return contributions.total_entropic_energy
