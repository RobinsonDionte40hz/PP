"""
Solvent field correction for electrostatic interactions in UBF protein system.

This module implements solvent screening effects on electrostatic interactions:
- Distance-dependent dielectric: Models screening by solvent at different distances
- Burial factor: Adjusts dielectric based on residue exposure to solvent
- Sigmoidal transitions: Smooth interpolation between buried and surface environments

These corrections provide more realistic electrostatics by accounting for the
protein's heterogeneous dielectric environment.
"""

import math
from typing import List, Tuple, Dict, Optional

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


class SolventFieldCorrection:
    """
    Calculator for solvent screening corrections to electrostatic interactions.
    
    Implements a physically motivated model for the dielectric environment:
    
    1. Distance-dependent dielectric: Screening increases with distance as more
       solvent molecules can intervene between charges.
       
    2. Burial factor: Core residues experience lower dielectric (ε ≈ 4, like
       hydrophobic interior) while surface residues see higher dielectric
       (ε ≈ 80, like bulk water).
       
    3. Smooth transitions: Sigmoidal functions provide continuous interpolation
       to avoid discontinuities in energy and forces.
    
    Attributes:
        screening_length: Characteristic length for distance-dependent screening (3.0 Å)
        neighbor_cutoff: Distance to count neighbors for burial (8.0 Å)
        epsilon_buried: Dielectric constant for buried residues (4.0)
        epsilon_surface: Dielectric constant for surface residues (80.0)
        burial_midpoint: Neighbor count for 50% buried (12 neighbors)
        burial_steepness: Steepness of sigmoidal burial transition (0.3)
    
    Example:
        >>> corrector = SolventFieldCorrection()
        >>> burial = corrector.calculate_burial_factor(field, all_fields)
        >>> epsilon = corrector.calculate_effective_dielectric(5.0, burial)
    """
    
    def __init__(self,
                 screening_length: float = 3.0,
                 neighbor_cutoff: float = 8.0,
                 epsilon_buried: float = 4.0,
                 epsilon_surface: float = 80.0,
                 burial_midpoint: int = 12,
                 burial_steepness: float = 0.3):
        """
        Initialize solvent correction calculator.
        
        Args:
            screening_length: Distance decay constant in Angstroms (default 3.0)
            neighbor_cutoff: Maximum distance to count neighbors in Angstroms (default 8.0)
            epsilon_buried: Dielectric for buried residues (default 4.0)
            epsilon_surface: Dielectric for surface residues (default 80.0)
            burial_midpoint: Neighbor count for 50% burial (default 12)
            burial_steepness: Sigmoid steepness parameter (default 0.3)
        """
        if screening_length <= 0:
            raise ValueError(f"screening_length must be positive, got {screening_length}")
        if neighbor_cutoff <= 0:
            raise ValueError(f"neighbor_cutoff must be positive, got {neighbor_cutoff}")
        if epsilon_buried < 1.0 or epsilon_surface < epsilon_buried:
            raise ValueError(f"Need 1.0 <= epsilon_buried <= epsilon_surface")
        if burial_midpoint < 0:
            raise ValueError(f"burial_midpoint must be non-negative, got {burial_midpoint}")
        if burial_steepness <= 0:
            raise ValueError(f"burial_steepness must be positive, got {burial_steepness}")
        
        self.screening_length = screening_length
        self.neighbor_cutoff = neighbor_cutoff
        self.epsilon_buried = epsilon_buried
        self.epsilon_surface = epsilon_surface
        self.burial_midpoint = burial_midpoint
        self.burial_steepness = burial_steepness
    
    def count_neighbors(self, field: SideChainField, all_fields: List[SideChainField],
                       sequence_separation: int = 3) -> int:
        """
        Count neighbors within cutoff distance.
        
        Counts residues that are close in space but separated in sequence.
        This gives a measure of local packing density around a residue.
        
        Args:
            field: Target side-chain field
            all_fields: List of all side-chain fields
            sequence_separation: Minimum sequence separation to count (default 3)
            
        Returns:
            Number of neighbors within neighbor_cutoff distance
        """
        count = 0
        
        for other_field in all_fields:
            # Skip self
            if other_field.residue_index == field.residue_index:
                continue
            
            # Skip close sequence neighbors (connected by backbone)
            if abs(other_field.residue_index - field.residue_index) <= sequence_separation:
                continue
            
            # Check distance
            distance = field.calculate_distance_to(other_field.position)
            if distance <= self.neighbor_cutoff:
                count += 1
        
        return count
    
    def calculate_burial_factor(self, field: SideChainField,
                                all_fields: List[SideChainField],
                                sequence_separation: int = 3) -> float:
        """
        Calculate burial factor based on neighbor count.
        
        Uses a sigmoidal function to smoothly transition from surface (0.0)
        to buried (1.0) based on local packing density:
        
        burial = 1 / (1 + exp(-k * (n - n_0)))
        
        where n is neighbor count, n_0 is midpoint, k is steepness.
        
        Args:
            field: Target side-chain field
            all_fields: List of all side-chain fields
            sequence_separation: Minimum sequence separation to count (default 3)
            
        Returns:
            Burial factor in range [0.0, 1.0]
            - 0.0: Fully surface-exposed (few neighbors)
            - 1.0: Fully buried (many neighbors)
            
        Example:
            >>> corrector = SolventFieldCorrection()
            >>> # Surface residue with 5 neighbors
            >>> burial = corrector.calculate_burial_factor(field, all_fields)
            >>> burial < 0.5  # Less than 50% buried
            True
        """
        # Count neighbors
        neighbor_count = self.count_neighbors(field, all_fields, sequence_separation)
        
        # Sigmoidal burial factor
        # burial = 1 / (1 + exp(-k * (n - n_0)))
        deviation = neighbor_count - self.burial_midpoint
        exponent = -self.burial_steepness * deviation
        
        # Avoid overflow for large negative exponents
        if exponent > 20:
            return 0.0
        elif exponent < -20:
            return 1.0
        
        burial = 1.0 / (1.0 + math.exp(exponent))
        return burial
    
    def calculate_distance_dependent_dielectric(self, distance: float) -> float:
        """
        Calculate distance-dependent dielectric constant.
        
        Models the screening effect where distant charges are better shielded
        by intervening solvent molecules:
        
        ε(r) = ε_buried + (ε_surface - ε_buried) * (1 - exp(-r/λ))
        
        where λ is the screening length (3.0 Å).
        
        Args:
            distance: Inter-residue distance in Angstroms
            
        Returns:
            Dielectric constant in range [epsilon_buried, epsilon_surface]
            
        Example:
            >>> corrector = SolventFieldCorrection()
            >>> corrector.calculate_distance_dependent_dielectric(0.0)
            4.0  # At zero distance, ε = ε_buried
            >>> corrector.calculate_distance_dependent_dielectric(10.0)
            # Much closer to 80.0 at large distance
        """
        if distance < 0:
            raise ValueError(f"distance must be non-negative, got {distance}")
        
        # Exponential screening: ε(r) = ε_min + Δε * (1 - exp(-r/λ))
        delta_epsilon = self.epsilon_surface - self.epsilon_buried
        screening_factor = 1.0 - math.exp(-distance / self.screening_length)
        
        epsilon = self.epsilon_buried + delta_epsilon * screening_factor
        return epsilon
    
    def calculate_effective_dielectric(self, distance: float, burial_factor: float) -> float:
        """
        Calculate effective dielectric combining distance and burial effects.
        
        The effective dielectric interpolates between:
        - Distance-dependent dielectric for surface interactions
        - Low buried dielectric for core interactions
        
        Formula:
        ε_eff = (1 - burial) * ε_distance(r) + burial * ε_buried
        
        This captures the physics that:
        - Surface-surface interactions: High ε, strong screening
        - Core-core interactions: Low ε, weak screening
        - Surface-core interactions: Intermediate ε
        
        Args:
            distance: Inter-residue distance in Angstroms
            burial_factor: Burial factor in range [0.0, 1.0]
            
        Returns:
            Effective dielectric constant
            
        Example:
            >>> corrector = SolventFieldCorrection()
            >>> # Core-core interaction (both buried)
            >>> eps_core = corrector.calculate_effective_dielectric(5.0, 0.9)
            >>> eps_core  # Close to epsilon_buried
            4.5
            >>> # Surface-surface interaction (both exposed)
            >>> eps_surf = corrector.calculate_effective_dielectric(5.0, 0.1)
            >>> eps_surf  # Much higher due to solvent screening
            70.0
        """
        if not 0.0 <= burial_factor <= 1.0:
            raise ValueError(f"burial_factor must be in [0, 1], got {burial_factor}")
        
        # Distance-dependent component (for exposed residues)
        epsilon_distance = self.calculate_distance_dependent_dielectric(distance)
        
        # Interpolate based on burial
        # More buried → use lower dielectric (less screening)
        # More exposed → use distance-dependent dielectric (more screening)
        epsilon_eff = (1.0 - burial_factor) * epsilon_distance + burial_factor * self.epsilon_buried
        
        return epsilon_eff
    
    def calculate_pairwise_effective_dielectric(self,
                                               field1: SideChainField,
                                               field2: SideChainField,
                                               all_fields: List[SideChainField],
                                               burial_cache: Optional[Dict[int, float]] = None) -> float:
        """
        Calculate effective dielectric for a pair of residues.
        
        Computes burial factors for both residues (using cache if available)
        and returns the effective dielectric for their interaction.
        
        Args:
            field1: First side-chain field
            field2: Second side-chain field
            all_fields: List of all fields (for burial calculation)
            burial_cache: Optional dict mapping residue_index to burial_factor
            
        Returns:
            Effective dielectric constant for the pair interaction
        """
        distance = field1.calculate_distance_to(field2.position)
        
        # Get or calculate burial factors
        if burial_cache is not None:
            burial1 = burial_cache.get(field1.residue_index)
            burial2 = burial_cache.get(field2.residue_index)
            
            if burial1 is None:
                burial1 = self.calculate_burial_factor(field1, all_fields)
                burial_cache[field1.residue_index] = burial1
            
            if burial2 is None:
                burial2 = self.calculate_burial_factor(field2, all_fields)
                burial_cache[field2.residue_index] = burial2
        else:
            burial1 = self.calculate_burial_factor(field1, all_fields)
            burial2 = self.calculate_burial_factor(field2, all_fields)
        
        # Average burial for the pair
        avg_burial = (burial1 + burial2) / 2.0
        
        # Calculate effective dielectric
        epsilon_eff = self.calculate_effective_dielectric(distance, avg_burial)
        
        return epsilon_eff
    
    def apply_correction_to_electrostatic(self,
                                         electrostatic_energy: float,
                                         field1: SideChainField,
                                         field2: SideChainField,
                                         all_fields: List[SideChainField],
                                         original_dielectric: float = 4.0,
                                         burial_cache: Optional[Dict[int, float]] = None) -> float:
        """
        Apply solvent correction to an electrostatic energy.
        
        Rescales the electrostatic energy based on the ratio of effective
        to original dielectric constants:
        
        E_corrected = E_original * (ε_original / ε_effective)
        
        Args:
            electrostatic_energy: Original electrostatic energy (kcal/mol)
            field1: First side-chain field
            field2: Second side-chain field
            all_fields: List of all fields (for burial calculation)
            original_dielectric: Dielectric used in original calculation (default 4.0)
            burial_cache: Optional burial factor cache for efficiency
            
        Returns:
            Corrected electrostatic energy (kcal/mol)
        """
        # If no charge interaction, no correction needed
        if electrostatic_energy == 0.0:
            return 0.0
        
        # Calculate effective dielectric
        epsilon_eff = self.calculate_pairwise_effective_dielectric(
            field1, field2, all_fields, burial_cache
        )
        
        # Rescale energy: E_new = E_old * (ε_old / ε_new)
        # Higher ε_eff → weaker interaction (more screening)
        correction_factor = original_dielectric / epsilon_eff
        corrected_energy = electrostatic_energy * correction_factor
        
        return corrected_energy
