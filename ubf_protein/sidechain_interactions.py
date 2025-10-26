"""
Side-chain interaction energy calculations for UBF protein system.

This module implements detailed side-chain interaction energies including:
- Steric repulsion from overlapping van der Waals fields
- Hydrophobic attraction between nonpolar residues
- Hydrophobic-hydrophilic repulsion
- Electrostatic interactions with Coulomb's law

All interactions use a 15.0 Å cutoff for computational efficiency.
"""

import math
from typing import Dict, List, Tuple, Optional

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try package-relative imports first
    from .models import SideChainField
except ImportError:
    # Fall back to absolute imports from ubf_protein package
    from ubf_protein.models import SideChainField


class SideChainInteractionCalculator:
    """
    Calculator for side-chain interaction energies.
    
    Implements physical models for:
    - Steric repulsion: Prevents atoms from overlapping
    - Hydrophobic effect: Attraction between nonpolar residues
    - Electrostatics: Coulomb interactions between charged residues
    - Hydrophobic-hydrophilic repulsion: Penalty for unfavorable contacts
    
    All energies are in kcal/mol for consistency with molecular mechanics.
    
    Attributes:
        cutoff_distance: Maximum interaction distance in Angstroms (default 15.0)
        sigma: Gaussian width for field calculations (default 2.0)
        k_coulomb: Coulomb constant (332.06 kcal·Å/(mol·e²))
        k_steric: Steric repulsion strength (10.0 kcal/mol)
        k_hydrophobic: Hydrophobic attraction strength (0.5 kcal/mol)
        k_repulsion: Hydrophobic-hydrophilic repulsion strength (0.3 kcal/mol)
    
    Example:
        >>> calc = SideChainInteractionCalculator()
        >>> field1 = SideChainField(0, 'L', (0.0, 0.0, 0.0), 0.0, 3.8, 166.7)
        >>> field2 = SideChainField(5, 'K', (5.0, 0.0, 0.0), 1.0, -3.9, 168.6)
        >>> energy = calc.calculate_total_interaction(field1, field2)
    """
    
    def __init__(self,
                 cutoff_distance: float = 15.0,
                 sigma: float = 2.0,
                 dielectric: float = 4.0):
        """
        Initialize interaction calculator.
        
        Args:
            cutoff_distance: Maximum interaction distance in Angstroms (default 15.0)
            sigma: Gaussian width for field calculations (default 2.0)
            dielectric: Dielectric constant for electrostatics (default 4.0 for protein interior)
        """
        if cutoff_distance <= 0:
            raise ValueError(f"cutoff_distance must be positive, got {cutoff_distance}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if dielectric < 1.0:
            raise ValueError(f"dielectric must be >= 1.0, got {dielectric}")
        
        self.cutoff_distance = cutoff_distance
        self.sigma = sigma
        self.dielectric = dielectric
        
        # Physical constants (kcal/mol units)
        self.k_coulomb = 332.06  # kcal·Å/(mol·e²) - Coulomb constant
        self.k_steric = 10.0     # kcal/mol - Steric repulsion strength
        self.k_hydrophobic = 0.5 # kcal/mol - Hydrophobic attraction strength
        self.k_repulsion = 0.3   # kcal/mol - Hydrophobic-hydrophilic repulsion
        
        # Cache for efficiency
        self._sigma_squared_2 = 2.0 * sigma * sigma
        self._sigma_repulsion = 0.5  # Å - Width of steric repulsion
        self._sigma_rep_squared = self._sigma_repulsion * self._sigma_repulsion
    
    def calculate_distance(self, field1: SideChainField, field2: SideChainField) -> float:
        """
        Calculate distance between two fields.
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            
        Returns:
            Distance in Angstroms
        """
        return field1.calculate_distance_to(field2.position)
    
    def calculate_steric_repulsion(self, field1: SideChainField, field2: SideChainField,
                                   distance: Optional[float] = None) -> float:
        """
        Calculate steric repulsion energy from overlapping van der Waals fields.
        
        Uses a Gaussian repulsion model that becomes strong when atoms approach
        closer than the sum of their van der Waals radii. The model:
        
        E = k_steric * exp(-(r - σ_sum)² / σ_rep²)
        
        where σ_sum is estimated from volumes and σ_rep = 0.5 Å controls sharpness.
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            distance: Pre-calculated distance (optional, for efficiency)
            
        Returns:
            Steric repulsion energy in kcal/mol (always >= 0)
        """
        if distance is None:
            distance = self.calculate_distance(field1, field2)
        
        # Beyond cutoff, no interaction
        if distance > self.cutoff_distance:
            return 0.0
        
        # Estimate van der Waals radii from volumes (V ≈ 4/3 π r³)
        # Using cube root for rough estimate
        r1 = (field1.volume / 4.18879) ** (1.0/3.0)  # 4/3π ≈ 4.18879
        r2 = (field2.volume / 4.18879) ** (1.0/3.0)
        sigma_sum = r1 + r2
        
        # Gaussian repulsion centered at contact distance
        deviation = distance - sigma_sum
        exponent = -(deviation * deviation) / self._sigma_rep_squared
        
        return self.k_steric * math.exp(exponent)
    
    def calculate_hydrophobic_attraction(self, field1: SideChainField, field2: SideChainField,
                                        distance: Optional[float] = None) -> float:
        """
        Calculate hydrophobic attraction between nonpolar residues.
        
        Hydrophobic residues (positive Kyte-Doolittle values) attract each other
        through the hydrophobic effect, driven by entropy gain from water exclusion.
        
        E = -k_hphob * h_i * h_j * exp(-r²/(2σ²))
        
        Only applies when both residues are hydrophobic (h > 0).
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            distance: Pre-calculated distance (optional, for efficiency)
            
        Returns:
            Hydrophobic attraction energy in kcal/mol (negative = favorable)
        """
        if distance is None:
            distance = self.calculate_distance(field1, field2)
        
        # Beyond cutoff, no interaction
        if distance > self.cutoff_distance:
            return 0.0
        
        # Only attractive if both are hydrophobic
        if field1.hydrophobicity <= 0.0 or field2.hydrophobicity <= 0.0:
            return 0.0
        
        # Gaussian decay from field center
        exponent = -(distance * distance) / self._sigma_squared_2
        field_strength = math.exp(exponent)
        
        # Energy scales with product of hydrophobicities
        energy = -self.k_hydrophobic * field1.hydrophobicity * field2.hydrophobicity * field_strength
        
        return energy
    
    def calculate_hydrophobic_repulsion(self, field1: SideChainField, field2: SideChainField,
                                       distance: Optional[float] = None) -> float:
        """
        Calculate repulsion between hydrophobic and hydrophilic residues.
        
        Unfavorable contacts between hydrophobic (h > 0) and hydrophilic (h < 0)
        residues incur an energy penalty, as they disrupt favorable solvation patterns.
        
        E = k_rep * |h_i * h_j| * exp(-r²/(2σ²))
        
        Only applies when residues have opposite hydrophobicity signs.
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            distance: Pre-calculated distance (optional, for efficiency)
            
        Returns:
            Repulsion energy in kcal/mol (positive = unfavorable)
        """
        if distance is None:
            distance = self.calculate_distance(field1, field2)
        
        # Beyond cutoff, no interaction
        if distance > self.cutoff_distance:
            return 0.0
        
        # Only repulsive if opposite signs (one hydrophobic, one hydrophilic)
        product = field1.hydrophobicity * field2.hydrophobicity
        if product >= 0.0:  # Same sign or zero
            return 0.0
        
        # Gaussian decay from field center
        exponent = -(distance * distance) / self._sigma_squared_2
        field_strength = math.exp(exponent)
        
        # Energy scales with magnitude of product
        energy = self.k_repulsion * abs(product) * field_strength
        
        return energy
    
    def calculate_electrostatic(self, field1: SideChainField, field2: SideChainField,
                               distance: Optional[float] = None,
                               dielectric: Optional[float] = None) -> float:
        """
        Calculate electrostatic interaction using Coulomb's law.
        
        Charges interact through Coulomb's law with a distance-dependent or
        constant dielectric:
        
        E = k_e * q_i * q_j / (ε * r)
        
        where k_e = 332.06 kcal·Å/(mol·e²) and ε is the dielectric constant.
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            distance: Pre-calculated distance (optional, for efficiency)
            dielectric: Dielectric constant override (uses self.dielectric if None)
            
        Returns:
            Electrostatic energy in kcal/mol (negative = favorable for opposite charges)
        """
        if distance is None:
            distance = self.calculate_distance(field1, field2)
        
        # Beyond cutoff, no interaction
        if distance > self.cutoff_distance:
            return 0.0
        
        # No interaction if either charge is zero
        if field1.charge == 0.0 or field2.charge == 0.0:
            return 0.0
        
        # Avoid division by zero at very small distances
        if distance < 0.1:  # 0.1 Å minimum distance
            distance = 0.1
        
        # Use provided dielectric or default
        eps = dielectric if dielectric is not None else self.dielectric
        
        # Coulomb's law: E = k * q1 * q2 / (ε * r)
        energy = self.k_coulomb * field1.charge * field2.charge / (eps * distance)
        
        return energy
    
    def calculate_total_interaction(self, field1: SideChainField, field2: SideChainField,
                                   include_components: bool = False) -> Dict[str, float]:
        """
        Calculate total interaction energy combining all terms.
        
        Computes steric repulsion, hydrophobic effects, and electrostatics,
        returning either just the total or a full breakdown.
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            include_components: If True, return breakdown of all components
            
        Returns:
            Dictionary with 'total' energy and optionally individual components:
            - 'total': Sum of all interaction energies
            - 'steric': Steric repulsion (if include_components=True)
            - 'hydrophobic_attraction': Hydrophobic attraction
            - 'hydrophobic_repulsion': Hydrophobic-hydrophilic repulsion
            - 'electrostatic': Coulomb interaction
        """
        # Calculate distance once for efficiency
        distance = self.calculate_distance(field1, field2)
        
        # Beyond cutoff, no interaction
        if distance > self.cutoff_distance:
            if include_components:
                return {
                    'total': 0.0,
                    'steric': 0.0,
                    'hydrophobic_attraction': 0.0,
                    'hydrophobic_repulsion': 0.0,
                    'electrostatic': 0.0
                }
            else:
                return {'total': 0.0}
        
        # Calculate all components
        steric = self.calculate_steric_repulsion(field1, field2, distance)
        hphob_attract = self.calculate_hydrophobic_attraction(field1, field2, distance)
        hphob_repel = self.calculate_hydrophobic_repulsion(field1, field2, distance)
        electro = self.calculate_electrostatic(field1, field2, distance)
        
        # Sum for total
        total = steric + hphob_attract + hphob_repel + electro
        
        if include_components:
            return {
                'total': total,
                'steric': steric,
                'hydrophobic_attraction': hphob_attract,
                'hydrophobic_repulsion': hphob_repel,
                'electrostatic': electro
            }
        else:
            return {'total': total}
    
    def calculate_all_pairwise_interactions(self,
                                           fields: List[SideChainField],
                                           sequence_separation: int = 3) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total interaction energy for all field pairs.
        
        Computes pairwise interactions between all fields, respecting the
        sequence separation filter (typically skip i,i+1, i+2, i+3 pairs
        to avoid spurious backbone effects).
        
        Args:
            fields: List of side-chain fields
            sequence_separation: Minimum sequence separation (default 3)
            
        Returns:
            Tuple of (total_energy, energy_breakdown) where breakdown contains
            sums of each component type
        """
        total_energy = 0.0
        breakdown = {
            'steric': 0.0,
            'hydrophobic_attraction': 0.0,
            'hydrophobic_repulsion': 0.0,
            'electrostatic': 0.0
        }
        
        n_fields = len(fields)
        
        # Loop over all unique pairs with sequence separation
        for i in range(n_fields):
            for j in range(i + sequence_separation + 1, n_fields):
                # Calculate interaction with full breakdown
                result = self.calculate_total_interaction(fields[i], fields[j], include_components=True)
                
                # Accumulate totals
                total_energy += result['total']
                breakdown['steric'] += result['steric']
                breakdown['hydrophobic_attraction'] += result['hydrophobic_attraction']
                breakdown['hydrophobic_repulsion'] += result['hydrophobic_repulsion']
                breakdown['electrostatic'] += result['electrostatic']
        
        return total_energy, breakdown
