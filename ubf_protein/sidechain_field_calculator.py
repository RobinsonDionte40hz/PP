"""
Side-chain field calculator for UBF protein system.

This module implements the calculation of side-chain electromagnetic and steric fields,
which are used to model inter-residue interactions beyond simple backbone geometry.

The field model uses:
- Gaussian decay with sigma = 2.0 Å
- Amino acid-specific properties (charge, hydrophobicity, volume)
- 3D spatial positioning for accurate distance calculations
"""

import math
from typing import List, Tuple, Optional

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try package-relative imports first
    from .models import SideChainField, Conformation
    from .amino_acid_properties import get_all_properties
except ImportError:
    # Fall back to absolute imports from ubf_protein package
    from ubf_protein.models import SideChainField, Conformation
    from ubf_protein.amino_acid_properties import get_all_properties


class SideChainFieldCalculator:
    """
    Calculator for side-chain electromagnetic and steric fields.
    
    Creates SideChainField objects from amino acid properties and implements
    Gaussian field strength calculations for modeling inter-residue interactions.
    
    The Gaussian decay model represents the effective interaction range of
    side-chains, with sigma=2.0 Å providing realistic falloff behavior.
    
    Attributes:
        sigma: Gaussian standard deviation in Angstroms (default 2.0)
    
    Example:
        >>> calculator = SideChainFieldCalculator()
        >>> field = calculator.create_field_for_residue(5, 'W', (10.0, 20.0, 15.0))
        >>> strength = calculator.calculate_field_strength(field.field_strength, 3.0)
    """
    
    def __init__(self, sigma: float = 2.0):
        """
        Initialize field calculator.
        
        Args:
            sigma: Gaussian standard deviation in Angstroms (default 2.0)
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self._sigma_squared_2 = 2.0 * sigma * sigma  # Cache for efficiency
    
    def create_field_for_residue(self,
                                 residue_index: int,
                                 amino_acid: str,
                                 position: Tuple[float, float, float],
                                 field_strength: float = 1.0) -> SideChainField:
        """
        Create a side-chain field for a single residue.
        
        Looks up amino acid properties from the database and creates a
        SideChainField object with the appropriate parameters.
        
        Args:
            residue_index: Index of residue in sequence (0-based)
            amino_acid: Single-letter amino acid code
            position: 3D coordinates (x, y, z) in Angstroms
            field_strength: Base field strength multiplier (default 1.0)
            
        Returns:
            SideChainField object with all properties initialized
            
        Raises:
            ValueError: If amino acid code is invalid or parameters are invalid
            
        Example:
            >>> calc = SideChainFieldCalculator()
            >>> field = calc.create_field_for_residue(10, 'L', (5.0, 10.0, 15.0))
            >>> field.hydrophobicity
            3.8
        """
        # Look up amino acid properties
        charge, hydrophobicity, volume = get_all_properties(amino_acid)
        
        # Create and return field
        return SideChainField(
            residue_index=residue_index,
            amino_acid=amino_acid.upper(),
            position=position,
            charge=charge,
            hydrophobicity=hydrophobicity,
            volume=volume,
            field_strength=field_strength
        )
    
    def calculate_field_strength(self, base_strength: float, distance: float) -> float:
        """
        Calculate Gaussian field strength at a given distance.
        
        Uses the formula: strength = base_strength * exp(-distance²/(2σ²))
        
        This provides a smooth falloff that represents the effective interaction
        range of side-chains, with the field decaying to ~61% at distance=σ,
        ~14% at distance=2σ, and ~1% at distance=3σ.
        
        Args:
            base_strength: Base field strength (typically 1.0)
            distance: Distance from field center in Angstroms
            
        Returns:
            Field strength at the given distance (0.0 to base_strength)
            
        Example:
            >>> calc = SideChainFieldCalculator(sigma=2.0)
            >>> calc.calculate_field_strength(1.0, 0.0)  # At field center
            1.0
            >>> calc.calculate_field_strength(1.0, 2.0)  # At 1 sigma
            0.6065...
            >>> calc.calculate_field_strength(1.0, 6.0)  # At 3 sigma
            0.0111...
        """
        if distance < 0:
            raise ValueError(f"distance must be non-negative, got {distance}")
        
        # Gaussian decay: exp(-r²/(2σ²))
        exponent = -(distance * distance) / self._sigma_squared_2
        return base_strength * math.exp(exponent)
    
    def calculate_field_strength_between(self,
                                        field: SideChainField,
                                        target_position: Tuple[float, float, float]) -> float:
        """
        Calculate field strength at a target position.
        
        Convenience method that combines distance calculation and field strength
        computation in one call.
        
        Args:
            field: Source side-chain field
            target_position: Target 3D coordinates (x, y, z)
            
        Returns:
            Field strength at target position
            
        Example:
            >>> calc = SideChainFieldCalculator()
            >>> field = calc.create_field_for_residue(0, 'W', (0.0, 0.0, 0.0))
            >>> strength = calc.calculate_field_strength_between(field, (2.0, 0.0, 0.0))
        """
        distance = field.calculate_distance_to(target_position)
        return self.calculate_field_strength(field.field_strength, distance)
    
    def create_fields_for_conformation(self,
                                      conformation: Conformation,
                                      use_ca_positions: bool = True) -> List[SideChainField]:
        """
        Create side-chain fields for all residues in a conformation.
        
        Generates a list of SideChainField objects, one for each residue in the
        protein. By default uses CA (alpha carbon) positions as field centers,
        though side-chain centroids could be used with more detailed geometry.
        
        Args:
            conformation: Protein conformation with sequence and coordinates
            use_ca_positions: If True, use CA atom positions as field centers
                            (default True, since CA is typically available)
            
        Returns:
            List of SideChainField objects, one per residue
            
        Raises:
            ValueError: If sequence length doesn't match coordinate count
            
        Example:
            >>> calc = SideChainFieldCalculator()
            >>> # Assuming conformation is a valid Conformation object
            >>> fields = calc.create_fields_for_conformation(conformation)
            >>> len(fields) == len(conformation.sequence)
            True
        """
        if len(conformation.sequence) != len(conformation.atom_coordinates):
            raise ValueError(
                f"Sequence length ({len(conformation.sequence)}) doesn't match "
                f"coordinate count ({len(conformation.atom_coordinates)})"
            )
        
        fields = []
        for i, amino_acid in enumerate(conformation.sequence):
            # Use CA position (typically first atom for each residue)
            position = conformation.atom_coordinates[i]
            
            # Create field for this residue
            field = self.create_field_for_residue(
                residue_index=i,
                amino_acid=amino_acid,
                position=position,
                field_strength=1.0
            )
            fields.append(field)
        
        return fields
    
    def get_interacting_pairs(self,
                             fields: List[SideChainField],
                             cutoff_distance: float = 15.0,
                             sequence_separation: int = 3) -> List[Tuple[int, int, float]]:
        """
        Find all pairs of fields within interaction distance.
        
        Returns pairs of residue indices that are close enough to interact,
        along with their distances. Filters out pairs that are too close in
        sequence (typically within 3 residues) to avoid spurious backbone effects.
        
        Args:
            fields: List of side-chain fields
            cutoff_distance: Maximum interaction distance in Angstroms (default 15.0)
            sequence_separation: Minimum sequence separation to consider (default 3)
            
        Returns:
            List of (index_i, index_j, distance) tuples for interacting pairs
            
        Example:
            >>> calc = SideChainFieldCalculator()
            >>> fields = calc.create_fields_for_conformation(conformation)
            >>> pairs = calc.get_interacting_pairs(fields, cutoff_distance=10.0)
            >>> len(pairs)  # Number of interacting pairs
            42
        """
        pairs = []
        n_fields = len(fields)
        
        for i in range(n_fields):
            for j in range(i + sequence_separation + 1, n_fields):
                distance = fields[i].calculate_distance_to(fields[j].position)
                
                if distance <= cutoff_distance:
                    pairs.append((i, j, distance))
        
        return pairs
