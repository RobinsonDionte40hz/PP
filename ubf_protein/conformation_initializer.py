"""
Conformation Initializer for UBF Protein System.

Generates reasonable initial conformations instead of random coordinates.
This prevents catastrophic energy explosion in enhanced physics mode.

Key strategies:
1. Ideal backbone geometry (3.8 Å CA-CA spacing)
2. Secondary structure biased angles (helix/sheet regions)
3. Disulfide bond proximity initialization (if bonds known)
4. Compact sphere initialization (prevents 100+ Å extended chains)
"""

import math
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from .models import DisulfideBond
except ImportError:
    from ubf_protein.models import DisulfideBond


@dataclass
class InitializationStrategy:
    """Configuration for conformation initialization."""
    mode: str = "compact_sphere"  # compact_sphere, extended_chain, helical_bias
    sphere_radius: float = 15.0  # Å for compact sphere mode
    ca_ca_distance: float = 3.8  # Å ideal CA-CA spacing
    add_noise: bool = True  # Add small random perturbations
    noise_std: float = 0.5  # Å standard deviation of noise
    respect_disulfide_proximity: bool = True  # Bias toward disulfide satisfaction
    disulfide_proximity_radius: float = 10.0  # Å initial separation for S-S pairs


class ConformationInitializer:
    """
    Generate reasonable initial protein conformations.
    
    Prevents catastrophic energy in enhanced physics mode by ensuring:
    - Reasonable backbone geometry (3.8 Å CA-CA spacing)
    - Compact initial structure (not 100+ Å extended)
    - Disulfide pairs start near each other (~10 Å, not 140 Å)
    - Biased toward native-like geometry
    """
    
    def __init__(self, strategy: Optional[InitializationStrategy] = None):
        """
        Initialize conformation generator.
        
        Args:
            strategy: Initialization strategy configuration
        """
        self.strategy = strategy or InitializationStrategy()
    
    def generate_initial_coordinates(
        self,
        sequence_length: int,
        disulfide_bonds: Optional[List[DisulfideBond]] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Generate initial CA coordinates for protein.
        
        Args:
            sequence_length: Number of residues
            disulfide_bonds: Optional disulfide bond constraints
            
        Returns:
            List of (x, y, z) coordinates for CA atoms
        """
        if self.strategy.mode == "compact_sphere":
            coords = self._generate_compact_sphere(sequence_length)
        elif self.strategy.mode == "extended_chain":
            coords = self._generate_extended_chain(sequence_length)
        elif self.strategy.mode == "helical_bias":
            coords = self._generate_helical_bias(sequence_length)
        else:
            # Default to compact sphere
            coords = self._generate_compact_sphere(sequence_length)
        
        # Apply disulfide proximity constraint if requested
        if (disulfide_bonds and 
            self.strategy.respect_disulfide_proximity and 
            len(disulfide_bonds) > 0):
            coords = self._adjust_for_disulfide_proximity(coords, disulfide_bonds)
        
        # Add small random noise to break symmetry
        if self.strategy.add_noise:
            coords = self._add_noise(coords)
        
        return coords
    
    def _generate_compact_sphere(
        self, 
        sequence_length: int
    ) -> List[Tuple[float, float, float]]:
        """
        Generate coordinates in compact sphere with ideal CA-CA spacing.
        
        Places residues on a random walk constrained to sphere radius.
        Maintains ~3.8 Å CA-CA distance while keeping structure compact.
        
        Args:
            sequence_length: Number of residues
            
        Returns:
            List of CA coordinates
        """
        coords = []
        radius = self.strategy.sphere_radius
        ca_dist = self.strategy.ca_ca_distance
        
        # Start at center
        current = (0.0, 0.0, 0.0)
        coords.append(current)
        
        for i in range(1, sequence_length):
            # Random direction
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            
            # Unit vector
            dx = math.sin(phi) * math.cos(theta)
            dy = math.sin(phi) * math.sin(theta)
            dz = math.cos(phi)
            
            # Step ca_dist in random direction
            next_x = current[0] + ca_dist * dx
            next_y = current[1] + ca_dist * dy
            next_z = current[2] + ca_dist * dz
            
            # If outside sphere, reflect back toward center
            dist_from_center = math.sqrt(next_x**2 + next_y**2 + next_z**2)
            if dist_from_center > radius:
                scale = radius / dist_from_center
                next_x *= scale
                next_y *= scale
                next_z *= scale
            
            current = (next_x, next_y, next_z)
            coords.append(current)
        
        return coords
    
    def _generate_extended_chain(
        self,
        sequence_length: int
    ) -> List[Tuple[float, float, float]]:
        """
        Generate extended chain along x-axis.
        
        Simple linear chain with ideal CA-CA spacing.
        Useful for baseline testing.
        
        Args:
            sequence_length: Number of residues
            
        Returns:
            List of CA coordinates
        """
        coords = []
        ca_dist = self.strategy.ca_ca_distance
        
        for i in range(sequence_length):
            x = i * ca_dist
            y = 0.0
            z = 0.0
            coords.append((x, y, z))
        
        return coords
    
    def _generate_helical_bias(
        self,
        sequence_length: int
    ) -> List[Tuple[float, float, float]]:
        """
        Generate coordinates with helical bias.
        
        Creates alpha-helix-like geometry:
        - 3.6 residues per turn
        - 5.4 Å pitch
        - Right-handed helix
        
        Args:
            sequence_length: Number of residues
            
        Returns:
            List of CA coordinates
        """
        coords = []
        
        # Alpha helix parameters
        radius = 2.3  # Å helix radius
        pitch = 5.4   # Å per turn
        residues_per_turn = 3.6
        
        for i in range(sequence_length):
            angle = (i / residues_per_turn) * 2 * math.pi
            z = (i / residues_per_turn) * pitch
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            coords.append((x, y, z))
        
        return coords
    
    def _adjust_for_disulfide_proximity(
        self,
        coords: List[Tuple[float, float, float]],
        disulfide_bonds: List[DisulfideBond]
    ) -> List[Tuple[float, float, float]]:
        """
        Adjust coordinates to bring disulfide-bonded cysteines closer.
        
        Moves bonded pairs toward their midpoint to achieve
        initial separation of ~10 Å instead of random separation.
        
        Args:
            coords: Initial coordinates
            disulfide_bonds: Disulfide bond constraints
            
        Returns:
            Adjusted coordinates
        """
        coords = list(coords)  # Make mutable copy
        target_dist = self.strategy.disulfide_proximity_radius
        
        for bond in disulfide_bonds:
            i, j = bond.residue_i, bond.residue_j
            
            if i >= len(coords) or j >= len(coords):
                continue
            
            # Current positions
            pos_i = coords[i]
            pos_j = coords[j]
            
            # Current distance
            dx = pos_j[0] - pos_i[0]
            dy = pos_j[1] - pos_i[1]
            dz = pos_j[2] - pos_i[2]
            current_dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # If too far, bring closer
            if current_dist > target_dist:
                # Midpoint
                mid_x = (pos_i[0] + pos_j[0]) / 2
                mid_y = (pos_i[1] + pos_j[1]) / 2
                mid_z = (pos_i[2] + pos_j[2]) / 2
                
                # Unit vector from i to j
                if current_dist > 0.01:
                    ux = dx / current_dist
                    uy = dy / current_dist
                    uz = dz / current_dist
                else:
                    ux, uy, uz = 1.0, 0.0, 0.0
                
                # Place i and j at target_dist/2 from midpoint
                half_target = target_dist / 2
                
                new_i = (
                    mid_x - half_target * ux,
                    mid_y - half_target * uy,
                    mid_z - half_target * uz
                )
                new_j = (
                    mid_x + half_target * ux,
                    mid_y + half_target * uy,
                    mid_z + half_target * uz
                )
                
                coords[i] = new_i
                coords[j] = new_j
        
        return coords
    
    def _add_noise(
        self,
        coords: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Add small random noise to coordinates.
        
        Breaks perfect symmetry and provides diversity
        for multiple agents.
        
        Args:
            coords: Input coordinates
            
        Returns:
            Coordinates with added noise
        """
        noisy_coords = []
        std = self.strategy.noise_std
        
        for x, y, z in coords:
            nx = x + random.gauss(0, std)
            ny = y + random.gauss(0, std)
            nz = z + random.gauss(0, std)
            noisy_coords.append((nx, ny, nz))
        
        return noisy_coords
    
    @staticmethod
    def calculate_initial_energy_estimate(
        coords: List[Tuple[float, float, float]],
        disulfide_bonds: Optional[List[DisulfideBond]] = None
    ) -> dict:
        """
        Estimate initial energy to verify reasonableness.
        
        Quick sanity check without full energy calculation.
        
        Args:
            coords: Coordinates to check
            disulfide_bonds: Optional disulfide constraints
            
        Returns:
            Dictionary with energy estimates
        """
        # Check CA-CA distances
        ca_distances = []
        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            dz = coords[i+1][2] - coords[i][2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            ca_distances.append(dist)
        
        avg_ca_dist = sum(ca_distances) / len(ca_distances) if ca_distances else 0
        
        # Check disulfide distances
        disulfide_info = {}
        if disulfide_bonds:
            for bond in disulfide_bonds:
                i, j = bond.residue_i, bond.residue_j
                if i < len(coords) and j < len(coords):
                    dx = coords[j][0] - coords[i][0]
                    dy = coords[j][1] - coords[i][1]
                    dz = coords[j][2] - coords[i][2]
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Estimate disulfide energy
                    k = 50.0
                    r0 = 3.8
                    energy = 0.5 * k * (dist - r0) ** 2
                    
                    disulfide_info[f"bond_{i}_{j}"] = {
                        "distance": dist,
                        "energy": energy
                    }
        
        return {
            "avg_ca_distance": avg_ca_dist,
            "ca_distance_range": (min(ca_distances), max(ca_distances)) if ca_distances else (0, 0),
            "disulfide_bonds": disulfide_info,
            "total_disulfide_energy": sum(info["energy"] for info in disulfide_info.values())
        }


def create_default_initializer(
    protein_size: int,
    has_disulfide_bonds: bool = False
) -> ConformationInitializer:
    """
    Create initializer with sensible defaults based on protein characteristics.
    
    Args:
        protein_size: Number of residues
        has_disulfide_bonds: Whether protein has disulfide bonds
        
    Returns:
        Configured ConformationInitializer
    """
    # Adjust sphere radius based on protein size
    # Rule of thumb: radius ≈ 2 * (N residues)^(1/3)
    radius = 2.0 * (protein_size ** (1/3)) + 5.0
    radius = min(radius, 30.0)  # Cap at 30 Å for very large proteins
    
    strategy = InitializationStrategy(
        mode="compact_sphere",
        sphere_radius=radius,
        ca_ca_distance=3.8,
        add_noise=True,
        noise_std=0.5,
        respect_disulfide_proximity=has_disulfide_bonds,
        disulfide_proximity_radius=10.0 if has_disulfide_bonds else 15.0
    )
    
    return ConformationInitializer(strategy)
