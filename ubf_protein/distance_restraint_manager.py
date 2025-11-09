"""
Distance Restraint Manager for UBF Protein System

This module manages φ-harmonic distance restraints for high-QCP residue pairs
during quantum refinement. Distance restraints enforce golden ratio geometric
patterns that emerge in stable protein structures.

Key Concepts:
    - φ-Harmonic Distance: Distances at golden ratio multiples (d/φ, d, d×φ)
    - High-QCP Pairs: Residue pairs where both QCP > threshold (default 7.0)
    - Harmonic Restraint: E = weight × (distance - target)² when outside tolerance
    - Optimal Contact Distance: 6.0Å (typical inter-residue contact)

Physics Background:
    - Golden ratio patterns emerge in quantum-coherent structures
    - High-QCP regions have enhanced structural stability
    - Distance restraints preserve coherence during optimization
    - φ-harmonic spacing minimizes conformational entropy

Restraint Parameters:
    - Weight: 100.0 kcal/mol/Å² (strong restraint)
    - Tolerance: 0.5Å (tight constraint)
    - QCP Threshold: 7.0 (high coherence)
    - Target: Nearest φ-harmonic to 6.0Å

Performance Targets:
    - Restraint generation: <50ms for 100 residues
    - Restraint application: <10ms per optimization step
    - Memory: <1MB for typical protein

Requirements Addressed:
    - 7.1: Identify high-QCP pairs (both residues QCP > 7)
    - 7.2: Calculate φ-harmonic distances [d/φ, d, d×φ]
    - 7.3: Select distance closest to 6.0Å
    - 7.4: Apply weight 100.0 and tolerance 0.5Å
    - 7.5: Maintain restraints throughout optimization
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

try:
    from .models import Conformation, DistanceRestraint
    from .qcpp_integration import QCPPIntegrationAdapter
except ImportError:
    from ubf_protein.models import Conformation, DistanceRestraint
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter

# Setup logger
logger = logging.getLogger(__name__)


class DistanceRestraintManager:
    """
    Manages golden ratio distance restraints for quantum refinement.
    
    This manager identifies high-QCP residue pairs and generates φ-harmonic
    distance restraints to maintain quantum coherence during optimization.
    
    The restraint selection process:
        1. Identify pairs where both residues have QCP > threshold
        2. Calculate current inter-residue distance
        3. Generate φ-harmonic options: [d/φ, d, d×φ]
        4. Select option closest to optimal distance (6.0Å)
        5. Apply strong harmonic restraint (weight=100, tolerance=0.5Å)
    
    Attributes:
        qcpp_adapter: QCPP integration for QCP values
        phi: Golden ratio constant (1.618033988749895)
        default_weight: Default restraint force constant (100.0)
        default_tolerance: Default distance tolerance (0.5Å)
        optimal_distance: Target contact distance (6.0Å)
    
    Example:
        >>> manager = DistanceRestraintManager(qcpp_adapter)
        >>> restraints = manager.add_quantum_distance_restraints(
        ...     structure=conformation,
        ...     qcp_values={0: 8.5, 1: 9.2, 2: 6.3, 3: 8.8},
        ...     qcp_threshold=7.0
        ... )
        >>> print(f"Generated {len(restraints)} φ-harmonic restraints")
    """
    
    def __init__(
        self,
        qcpp_adapter: Optional[QCPPIntegrationAdapter] = None,
        phi: float = 1.618033988749895,
        default_weight: float = 100.0,
        default_tolerance: float = 0.5,
        optimal_distance: float = 6.0
    ):
        """
        Initialize distance restraint manager.
        
        Args:
            qcpp_adapter: Optional QCPP integration for QCP calculations
            phi: Golden ratio constant (default: 1.618033988749895)
            default_weight: Default restraint force constant (default: 100.0)
            default_tolerance: Default distance tolerance in Å (default: 0.5)
            optimal_distance: Target contact distance in Å (default: 6.0)
        """
        self.qcpp_adapter = qcpp_adapter
        self.phi = phi
        self.default_weight = default_weight
        self.default_tolerance = default_tolerance
        self.optimal_distance = optimal_distance
        
        # Statistics
        self.total_restraints_generated = 0
        self.total_pairs_evaluated = 0
        
        logger.info(
            f"Initialized DistanceRestraintManager with φ={self.phi:.6f}, "
            f"weight={self.default_weight:.1f}, tolerance={self.default_tolerance:.2f}Å"
        )
    
    def add_quantum_distance_restraints(
        self,
        structure: Conformation,
        qcp_values: Dict[int, float],
        qcp_threshold: float = 7.0,
        min_sequence_separation: int = 5
    ) -> List[DistanceRestraint]:
        """
        Generate distance restraints for high-QCP residue pairs.
        
        Identifies all pairs where both residues have QCP > threshold and
        generates φ-harmonic distance restraints to maintain quantum coherence.
        
        Algorithm:
            1. Iterate over all residue pairs (i, j) with |j-i| >= min_separation
            2. Check if both QCP_i and QCP_j > threshold
            3. Calculate current distance d_ij
            4. Generate φ-harmonic options: [d/φ, d, d×φ]
            5. Select option closest to optimal_distance (6.0Å)
            6. Create restraint with weight=100.0, tolerance=0.5Å
        
        Args:
            structure: Protein conformation with coordinates
            qcp_values: Dictionary mapping residue_index -> QCP value
            qcp_threshold: Minimum QCP for restraint (default: 7.0)
            min_sequence_separation: Minimum |j-i| for long-range restraints (default: 5)
        
        Returns:
            List of DistanceRestraint objects for high-QCP pairs
        
        Raises:
            ValueError: If structure has no coordinates
            ValueError: If qcp_values is empty
        
        Example:
            >>> qcp_vals = {0: 8.5, 1: 9.2, 2: 6.3, 3: 8.8, 4: 9.5}
            >>> restraints = manager.add_quantum_distance_restraints(
            ...     structure=conf,
            ...     qcp_values=qcp_vals,
            ...     qcp_threshold=7.0,
            ...     min_sequence_separation=2
            ... )
            >>> # Generates restraints for pairs: (0,1), (0,3), (0,4), (1,3), (1,4), (3,4)
            >>> # Skips pair (0,2) because QCP[2]=6.3 < 7.0
        """
        # Validation
        if not hasattr(structure, 'atom_coordinates') or not structure.atom_coordinates:
            raise ValueError("Structure must have atom_coordinates")
        
        if not qcp_values:
            raise ValueError("qcp_values cannot be empty")
        
        restraints = []
        n_residues = len(structure.atom_coordinates)
        
        logger.info(
            f"Generating restraints for {n_residues} residues with QCP threshold {qcp_threshold:.1f}"
        )
        
        # Iterate over all residue pairs
        for i in range(n_residues):
            for j in range(i + min_sequence_separation, n_residues):
                self.total_pairs_evaluated += 1
                
                # Check if both residues have high QCP
                qcp_i = qcp_values.get(i, 0.0)
                qcp_j = qcp_values.get(j, 0.0)
                
                if qcp_i < qcp_threshold or qcp_j < qcp_threshold:
                    continue
                
                # Calculate current distance
                coord_i = structure.atom_coordinates[i]
                coord_j = structure.atom_coordinates[j]
                current_distance = self._calculate_distance(coord_i, coord_j)
                
                # Find optimal φ-harmonic target distance
                phi_distances = self.find_phi_harmonic_distances(current_distance)
                target_distance = self._select_optimal_distance(phi_distances, self.optimal_distance)
                
                # Create restraint
                restraint = DistanceRestraint(
                    residue_i=i,
                    residue_j=j,
                    target_distance=target_distance,
                    weight=self.default_weight,
                    tolerance=self.default_tolerance,
                    is_phi_harmonic=True
                )
                
                restraints.append(restraint)
                self.total_restraints_generated += 1
                
                logger.debug(
                    f"Restraint {i}-{j}: QCP=({qcp_i:.2f}, {qcp_j:.2f}), "
                    f"distance={current_distance:.2f}Å → target={target_distance:.2f}Å"
                )
        
        logger.info(
            f"Generated {len(restraints)} restraints from {self.total_pairs_evaluated} pairs "
            f"({len(restraints)/max(1, self.total_pairs_evaluated)*100:.1f}% pass QCP threshold)"
        )
        
        return restraints
    
    def find_phi_harmonic_distances(self, current_distance: float) -> List[float]:
        """
        Calculate φ-harmonic distance options.
        
        Given a current distance d, generates three φ-harmonic options:
            - d/φ: Contracted by golden ratio
            - d:   Current distance (unchanged)
            - d×φ: Expanded by golden ratio
        
        These options span the φ-harmonic series, ensuring the final
        distance maintains golden ratio geometric relationships.
        
        Args:
            current_distance: Current inter-residue distance in Ångströms
        
        Returns:
            List of three φ-harmonic distances [d/φ, d, d×φ]
        
        Example:
            >>> manager = DistanceRestraintManager()
            >>> manager.find_phi_harmonic_distances(8.0)
            [4.944271909999159, 8.0, 12.944271909999159]
            
            >>> # For distance near optimal (6.0Å):
            >>> manager.find_phi_harmonic_distances(6.5)
            [4.017397626..., 6.5, 10.517397626...]
        """
        if current_distance <= 0:
            raise ValueError(f"current_distance must be > 0, got {current_distance}")
        
        return [
            current_distance / self.phi,  # Contracted
            current_distance,             # Unchanged
            current_distance * self.phi   # Expanded
        ]
    
    def apply_restraints(
        self,
        structure: Conformation,
        restraints: List[DistanceRestraint]
    ) -> float:
        """
        Apply distance restraints and calculate total restraint energy.
        
        Evaluates all restraints against current structure and sums the
        harmonic penalty energies. This energy is added to the total
        system energy during optimization to guide the structure toward
        φ-harmonic geometric patterns.
        
        The restraint energy for each pair is:
            E = 0                                  if |d - d₀| ≤ tolerance
            E = weight × (d - d₀ - tolerance)²     if d > d₀ + tolerance
            E = weight × (d₀ - d - tolerance)²     if d < d₀ - tolerance
        
        Args:
            structure: Protein conformation with current coordinates
            restraints: List of DistanceRestraint objects to apply
        
        Returns:
            Total restraint energy in kcal/mol
        
        Raises:
            ValueError: If structure has no coordinates
            ValueError: If restraints list is empty
        
        Example:
            >>> restraints = [
            ...     DistanceRestraint(0, 5, 6.0, 100.0, 0.5, True),
            ...     DistanceRestraint(2, 8, 5.5, 100.0, 0.5, True),
            ... ]
            >>> total_energy = manager.apply_restraints(structure, restraints)
            >>> print(f"Restraint energy: {total_energy:.2f} kcal/mol")
        """
        # Validation
        if not hasattr(structure, 'atom_coordinates') or not structure.atom_coordinates:
            raise ValueError("Structure must have atom_coordinates")
        
        if not restraints:
            logger.warning("apply_restraints called with empty restraints list")
            return 0.0
        
        total_energy = 0.0
        applied_count = 0
        violations = 0
        
        for restraint in restraints:
            # Get coordinates
            try:
                coord_i = structure.atom_coordinates[restraint.residue_i]
                coord_j = structure.atom_coordinates[restraint.residue_j]
            except (IndexError, KeyError) as e:
                logger.warning(
                    f"Skipping restraint {restraint.residue_i}-{restraint.residue_j}: {e}"
                )
                continue
            
            # Calculate current distance
            current_distance = self._calculate_distance(coord_i, coord_j)
            
            # Calculate restraint energy
            energy = restraint.calculate_energy(current_distance)
            total_energy += energy
            applied_count += 1
            
            if energy > 0:
                violations += 1
                logger.debug(
                    f"Restraint violation {restraint.residue_i}-{restraint.residue_j}: "
                    f"distance={current_distance:.2f}Å, target={restraint.target_distance:.2f}Å, "
                    f"energy={energy:.2f} kcal/mol"
                )
        
        logger.info(
            f"Applied {applied_count}/{len(restraints)} restraints: "
            f"total_energy={total_energy:.2f} kcal/mol, violations={violations}"
        )
        
        return total_energy
    
    def _calculate_distance(
        self,
        coord1: Tuple[float, float, float],
        coord2: Tuple[float, float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two 3D coordinates.
        
        Args:
            coord1: (x, y, z) coordinates in Ångströms
            coord2: (x, y, z) coordinates in Ångströms
        
        Returns:
            Distance in Ångströms
        """
        dx = coord2[0] - coord1[0]
        dy = coord2[1] - coord1[1]
        dz = coord2[2] - coord1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _select_optimal_distance(
        self,
        phi_distances: List[float],
        target: float
    ) -> float:
        """
        Select φ-harmonic distance closest to target.
        
        Given multiple φ-harmonic options, selects the one that minimizes
        deviation from the target distance (typically 6.0Å for contacts).
        
        Args:
            phi_distances: List of φ-harmonic distance options
            target: Target distance to match (default: 6.0Å)
        
        Returns:
            φ-harmonic distance closest to target
        
        Example:
            >>> manager = DistanceRestraintManager()
            >>> options = [4.5, 8.0, 12.5]  # [d/φ, d, d×φ]
            >>> manager._select_optimal_distance(options, 6.0)
            8.0  # Closest to 6.0
            
            >>> options = [3.7, 6.0, 9.7]
            >>> manager._select_optimal_distance(options, 6.0)
            6.0  # Exact match
        """
        return min(phi_distances, key=lambda d: abs(d - target))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get distance restraint manager statistics.
        
        Returns:
            Dictionary with statistics:
                - total_restraints_generated: Total restraints created
                - total_pairs_evaluated: Total residue pairs checked
                - acceptance_rate: Percentage passing QCP threshold
                - phi: Golden ratio constant
                - default_weight: Default force constant
                - default_tolerance: Default tolerance
                - optimal_distance: Target contact distance
        """
        acceptance_rate = 0.0
        if self.total_pairs_evaluated > 0:
            acceptance_rate = (
                self.total_restraints_generated / self.total_pairs_evaluated * 100
            )
        
        return {
            "total_restraints_generated": self.total_restraints_generated,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "acceptance_rate": acceptance_rate,
            "phi": self.phi,
            "default_weight": self.default_weight,
            "default_tolerance": self.default_tolerance,
            "optimal_distance": self.optimal_distance,
        }
