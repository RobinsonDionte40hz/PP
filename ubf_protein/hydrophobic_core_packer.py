"""
Quantum-guided hydrophobic core packing optimizer.

This module implements hydrophobic core optimization using water exclusion
zones and QCP-weighted force constants. Hydrophobic residues pack at optimal
distances determined by quantum-derived water molecule spacing, creating preferred
packing intervals.

Key Concepts:
    - Water Exclusion Zones: Hydrophobic residues exclude water, creating
      discrete packing distances at multiples of water spacing (2.8Å)
    - QCP-Weighted Forces: Force constants scaled by average QCP of residue
      pairs, prioritizing high-coherence interactions
    - Optimal Packing: Distances selected from water-spaced intervals closest
      to ideal contact distance (~6.0Å)

Design Principles:
    - Pure Python implementation (PyPy-optimized)
    - Immutable data models (PackingConstraint dataclass)
    - O(N²) pairwise evaluation for hydrophobic residues only
    - Physics-grounded constraints based on water shielding
"""

from typing import List, Dict, Tuple, Optional
import math

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .models import Conformation, PackingConstraint
except ImportError:
    from ubf_protein.models import Conformation, PackingConstraint


class HydrophobicCorePacker:
    """
    Quantum-guided hydrophobic core packing optimizer.
    
    Optimizes packing of hydrophobic residues using water exclusion zones
    and QCP-weighted force constants. Generates distance constraints that
    guide hydrophobic residues to optimal packing geometries.
    
    Attributes:
        water_spacing_nm: Water molecule spacing in nanometers (proprietary)
        water_spacing_angstrom: Water spacing in Ångströms (proprietary)
        base_force_constant: Base force constant for packing (proprietary)
        ideal_contact_distance: Ideal hydrophobic contact distance (proprietary)
        min_sequence_separation: Minimum residue separation for contacts
    
    Example:
        >>> packer = HydrophobicCorePacker()
        >>> constraints = packer.quantum_hydrophobic_packing(
        ...     structure=current_conformation,
        ...     qcp_values={0: 7.5, 5: 8.2, 10: 6.8}
        ... )
        >>> print(f"Generated {len(constraints)} packing constraints")
    """
    
    def __init__(self):
        """Initialize hydrophobic core packer with physics constants."""
        # Water spacing from quantum coherence theory (proprietary values)
        self.water_spacing_nm = 0.28  # proprietary
        self.water_spacing_angstrom = 2.8  # proprietary
        
        # Packing parameters
        self.base_force_constant = 10.0  # kcal/mol/Ř
        self.ideal_contact_distance = 6.0  # Ångströms
        self.min_sequence_separation = 5  # Minimum residue separation
        
        # Hydrophobic residues (standard amino acid codes)
        self.hydrophobic_residues = {
            'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'
        }
    
    def quantum_hydrophobic_packing(
        self,
        structure: Conformation,
        qcp_values: Dict[int, float]
    ) -> List[PackingConstraint]:
        """
        Generate packing constraints for hydrophobic residues.
        
        Strategy:
        1. Identify hydrophobic residues in sequence
        2. Generate all pairwise combinations (i < j, |j-i| >= 5)
        3. Calculate current distances from structure
        4. Find optimal water-excluded distances (2.8Å intervals)
        5. Calculate QCP coupling factors (avg QCP / 10)
        6. Create PackingConstraint objects with scaled force constants
        
        Args:
            structure: Current protein conformation
            qcp_values: QCP values for each residue index
        
        Returns:
            List of PackingConstraint objects for hydrophobic pairs
        
        Example:
            >>> structure = Conformation(
            ...     sequence="AVLIFMPW",
            ...     coordinates=[[0,0,0], [3,0,0], ...],
            ...     energy=-50.0
            ... )
            >>> qcp_values = {0: 7.5, 1: 8.0, 2: 6.5, ...}
            >>> constraints = packer.quantum_hydrophobic_packing(structure, qcp_values)
            >>> assert all(c.qcp_coupling > 0 for c in constraints)
        """
        constraints = []
        
        # Step 1: Identify hydrophobic residue indices
        hydrophobic_indices = self._identify_hydrophobic_residues(structure.sequence)
        
        if len(hydrophobic_indices) < 2:
            # Need at least 2 hydrophobic residues for pairwise constraints
            return constraints
        
        # Step 2 & 3: Generate pairwise combinations and calculate distances
        residue_pairs = self._generate_hydrophobic_pairs(hydrophobic_indices)
        
        # Step 4 & 5 & 6: For each pair, calculate optimal distance and create constraint
        for residue_i, residue_j in residue_pairs:
            # Get current distance
            current_distance = self._calculate_distance(
                structure.atom_coordinates[residue_i],
                structure.atom_coordinates[residue_j]
            )
            
            # Find optimal water-excluded distance
            optimal_distance = self._find_optimal_packing_distance(current_distance)
            
            # Calculate QCP coupling factor (ensure positive value)
            qcp_i = max(0.1, qcp_values.get(residue_i, 4.0))  # Default QCP if missing, min 0.1
            qcp_j = max(0.1, qcp_values.get(residue_j, 4.0))
            qcp_coupling = (qcp_i + qcp_j) / 2.0
            
            # Scale force constant by QCP coupling (minimum 1.0 to ensure positive)
            force_constant = max(1.0, self.base_force_constant * (qcp_coupling / 10.0))
            
            # Create packing constraint
            constraint = PackingConstraint(
                residue_i=residue_i,
                residue_j=residue_j,
                target_distance=optimal_distance,
                force_constant=force_constant,
                qcp_coupling=qcp_coupling
            )
            
            constraints.append(constraint)
        
        return constraints
    
    def calculate_water_exclusion_zones(
        self,
        residue_pairs: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate optimal distances based on water spacing.
        
        Water molecules create discrete spacing at quantum-derived intervals,
        leading to preferred packing distances at multiples of water spacing.
        This method is primarily for analysis - actual optimization
        happens in quantum_hydrophobic_packing().
        
        Args:
            residue_pairs: List of (residue_i, residue_j) tuples
        
        Returns:
            Dictionary mapping residue pairs to optimal packing distances
        
        Example:
            >>> packer = HydrophobicCorePacker()
            >>> pairs = [(0, 5), (1, 6), (2, 7)]
            >>> zones = packer.calculate_water_exclusion_zones(pairs)
            >>> # All distances should be multiples of water spacing
            >>> for dist in zones.values():
            ...     assert dist % 2.8 < 0.01 or (dist % 2.8) > 2.79
        """
        exclusion_zones = {}
        
        for residue_i, residue_j in residue_pairs:
            # Default optimal distance (closest to ideal contact)
            optimal_distance = self._find_optimal_packing_distance(
                self.ideal_contact_distance
            )
            exclusion_zones[(residue_i, residue_j)] = optimal_distance
        
        return exclusion_zones
    
    def _identify_hydrophobic_residues(self, sequence: str) -> List[int]:
        """
        Identify indices of hydrophobic residues in sequence.
        
        Args:
            sequence: Protein sequence (single-letter codes)
        
        Returns:
            List of 0-based residue indices that are hydrophobic
        """
        hydrophobic_indices = []
        
        for i, residue in enumerate(sequence):
            if residue.upper() in self.hydrophobic_residues:
                hydrophobic_indices.append(i)
        
        return hydrophobic_indices
    
    def _generate_hydrophobic_pairs(
        self,
        hydrophobic_indices: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Generate all valid hydrophobic residue pairs.
        
        Pairs must satisfy:
        - i < j (avoid duplicates)
        - |j - i| >= min_sequence_separation (avoid local contacts)
        
        Args:
            hydrophobic_indices: List of hydrophobic residue indices
        
        Returns:
            List of (residue_i, residue_j) tuples
        """
        pairs = []
        
        for i in range(len(hydrophobic_indices)):
            for j in range(i + 1, len(hydrophobic_indices)):
                res_i = hydrophobic_indices[i]
                res_j = hydrophobic_indices[j]
                
                # Check sequence separation
                if abs(res_j - res_i) >= self.min_sequence_separation:
                    pairs.append((res_i, res_j))
        
        return pairs
    
    def _calculate_distance(
        self,
        coord1: Tuple[float, float, float],
        coord2: Tuple[float, float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two coordinates.
        
        Args:
            coord1: First coordinate (x, y, z)
            coord2: Second coordinate (x, y, z)
        
        Returns:
            Distance in Ångströms
        """
        dx = coord2[0] - coord1[0]
        dy = coord2[1] - coord1[1]
        dz = coord2[2] - coord1[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _find_optimal_packing_distance(self, current_distance: float) -> float:
        """
        Find optimal packing distance from water-excluded intervals.
        
        Strategy:
        1. Generate candidate distances at 2.8Å intervals around current distance
        2. Select candidate closest to ideal_contact_distance (6.0Å)
        3. Ensure distance is reasonable (>= 2.8Å)
        
        Args:
            current_distance: Current distance between residues (Å)
        
        Returns:
            Optimal packing distance in Ångströms
        """
        # Generate candidates: [..., d-2.8, d, d+2.8, ...]
        # We want to find the water-spaced distance closest to 6.0Å
        
        # Find the water-spaced multiple closest to current distance
        n = round(current_distance / self.water_spacing_angstrom)
        
        # Generate candidates around this multiple
        candidates = []
        for offset in [-2, -1, 0, 1, 2]:
            multiplier = n + offset
            if multiplier >= 1:  # Ensure distance >= 2.8Å
                candidates.append(multiplier * self.water_spacing_angstrom)
        
        # Select candidate closest to ideal contact distance
        if not candidates:
            return self.water_spacing_angstrom  # Minimum distance
        
        optimal_distance = min(
            candidates,
            key=lambda d: abs(d - self.ideal_contact_distance)
        )
        
        return optimal_distance
