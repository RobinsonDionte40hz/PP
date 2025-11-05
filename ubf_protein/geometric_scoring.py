"""
Geometric Scoring Module for Prescriptive Geometric Targeting

Fast geometric similarity calculations (<2ms) for real-time move evaluation.
Supports targeting Platonic solid geometries: octahedron, icosahedron, 
dodecahedron, tetrahedron, cube.

Design Philosophy:
- Fast approximations (not full Platonic solid analysis from test_geometric_attractors.py)
- O(N) complexity for N atoms (no pairwise distance matrices)
- Uses distance ratios, angle distributions, symmetry measures
- Performance target: <2ms per conformation

Usage:
    scorer = GeometricScorer(target_geometry='octahedron')
    similarity = scorer.calculate_similarity(coords)  # Returns 0.0-1.0
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import time


@dataclass
class GeometricFeatures:
    """Geometric features extracted from protein structure (cached for performance)."""
    centroid: np.ndarray  # Center of mass (3,)
    radius_of_gyration: float  # RMS distance from centroid
    distance_ratios: List[float]  # Sample of distance ratios (for shape matching)
    angle_distribution: np.ndarray  # Histogram of backbone angles (10 bins)
    symmetry_score: float  # Rotational symmetry measure (0-1)
    asphericity: float  # Deviation from spherical shape (0-1)


class GeometricScorer:
    """
    Fast geometric similarity scorer for prescriptive geometric targeting.
    
    Performance Targets:
    - Calculation time: <2ms per conformation
    - Memory: <10KB per scorer instance
    - Cache efficiency: Reuses intermediate calculations
    
    Attributes:
        target_geometry: 'none', 'octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube'
        phi: Golden ratio constant (1.618...) for φ-containing geometries
    """
    
    def __init__(self, target_geometry: str = 'none'):
        """
        Initialize geometric scorer.
        
        Args:
            target_geometry: Target Platonic solid ('none', 'octahedron', 'icosahedron', etc.)
        """
        self.target_geometry = target_geometry.lower()
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        
        # Ideal distance ratios for each Platonic solid (used for shape matching)
        self.ideal_ratios = {
            'tetrahedron': [1.0, 1.0, 1.0],  # All edges equal
            'cube': [1.0, np.sqrt(2), np.sqrt(3)],  # Edge, face diagonal, body diagonal
            'octahedron': [1.0, np.sqrt(2), np.sqrt(2)],  # Edge, face diagonal
            'dodecahedron': [1.0, self.phi, self.phi**2],  # Contains golden ratio
            'icosahedron': [1.0, self.phi, self.phi],  # Contains golden ratio
        }
        
        # Ideal symmetry scores (number of rotational symmetries)
        self.ideal_symmetry = {
            'tetrahedron': 12,  # T_d symmetry (24 operations, 12 rotations)
            'cube': 24,  # O_h symmetry (48 operations, 24 rotations)
            'octahedron': 24,  # O_h symmetry (same as cube)
            'dodecahedron': 60,  # I_h symmetry (120 operations, 60 rotations)
            'icosahedron': 60,  # I_h symmetry (same as dodecahedron)
        }
        
        # Performance tracking
        self.calculation_count = 0
        self.total_time_ms = 0.0
        self.max_time_ms = 0.0
    
    def calculate_similarity(self, coords: List[np.ndarray]) -> float:
        """
        Calculate geometric similarity to target Platonic solid.
        
        Fast approximation using:
        1. Distance ratio matching (40% weight)
        2. Symmetry scoring (30% weight)
        3. Asphericity (20% weight)
        4. Angle distribution (10% weight)
        
        Args:
            coords: List of CA atom coordinates (N x 3 numpy arrays)
            
        Returns:
            Similarity score 0.0-1.0 (0.0 if target_geometry='none')
        """
        start_time = time.perf_counter()
        
        # Return 0 if no geometric target
        if self.target_geometry == 'none':
            return 0.0
        
        # Extract features (cached intermediate results)
        features = self._extract_features(coords)
        
        # Calculate component scores
        distance_score = self._score_distance_ratios(features.distance_ratios)
        symmetry_score = self._score_symmetry(features.symmetry_score)
        asphericity_score = self._score_asphericity(features.asphericity)
        angle_score = self._score_angles(features.angle_distribution)
        
        # Weighted combination
        similarity = (
            distance_score * 0.40 +
            symmetry_score * 0.30 +
            asphericity_score * 0.20 +
            angle_score * 0.10
        )
        
        # Track performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.calculation_count += 1
        self.total_time_ms += elapsed_ms
        self.max_time_ms = max(self.max_time_ms, elapsed_ms)
        
        # Warn if slow (target: <2ms)
        if elapsed_ms > 2.0:
            print(f"⚠️  Geometric scoring slow: {elapsed_ms:.2f}ms (target: <2ms)")
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _extract_features(self, coords: List[np.ndarray]) -> GeometricFeatures:
        """
        Extract geometric features from coordinates (O(N) complexity).
        
        Args:
            coords: List of CA coordinates
            
        Returns:
            GeometricFeatures with cached intermediate results
        """
        # Convert to numpy array for vectorized operations
        coords_array = np.array([np.array(c) for c in coords], dtype=np.float32)
        n_atoms = len(coords_array)
        
        # Centroid (center of mass)
        centroid = np.mean(coords_array, axis=0)
        
        # Radius of gyration (RMS distance from centroid)
        distances_from_center = np.linalg.norm(coords_array - centroid, axis=1)
        radius_of_gyration = float(np.sqrt(np.mean(distances_from_center**2)))
        
        # Sample distance ratios (avoid O(N²) pairwise distances)
        # Sample ~100 pairs uniformly across the structure
        sample_size = min(100, n_atoms * (n_atoms - 1) // 2)
        distance_ratios = []
        
        if n_atoms > 1:
            # Sample pairs uniformly
            step = max(1, n_atoms // 10)
            for i in range(0, n_atoms, step):
                for j in range(i + step, n_atoms, step):
                    if len(distance_ratios) >= sample_size:
                        break
                    dist_ij = np.linalg.norm(coords_array[i] - coords_array[j])
                    dist_i_center = distances_from_center[i]
                    if dist_i_center > 0.01:  # Avoid division by zero
                        distance_ratios.append(dist_ij / dist_i_center)
                if len(distance_ratios) >= sample_size:
                    break
        
        # Angle distribution (backbone angles for i, i+1, i+2)
        angles = []
        for i in range(n_atoms - 2):
            v1 = coords_array[i+1] - coords_array[i]
            v2 = coords_array[i+2] - coords_array[i+1]
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        # Create angle histogram (10 bins from 0 to π)
        if len(angles) > 0:
            angle_distribution, _ = np.histogram(angles, bins=10, range=(0, np.pi))
            angle_distribution = angle_distribution / np.sum(angle_distribution)  # Normalize
        else:
            angle_distribution = np.zeros(10)
        
        # Symmetry score (rotational symmetry approximation)
        # Check if distances are similar in different directions
        if n_atoms > 3:
            # Sample 8 directions uniformly on unit sphere
            directions = np.array([
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
                [1, 1, 1]/np.sqrt(3), [-1, -1, -1]/np.sqrt(3)
            ])
            
            # Average distance in each direction
            direction_distances = []
            for direction in directions:
                # Project all atoms onto this direction
                projections = np.dot(coords_array - centroid, direction)
                # Average absolute projection distance
                avg_dist = float(np.mean(np.abs(projections)))
                direction_distances.append(avg_dist)
            
            # Symmetry = uniformity of direction distances
            if np.std(direction_distances) > 0:
                symmetry_score = 1.0 - min(1.0, np.std(direction_distances) / np.mean(direction_distances))
            else:
                symmetry_score = 1.0
        else:
            symmetry_score = 0.5
        
        # Asphericity (deviation from sphere)
        # Calculate moment of inertia tensor
        coords_centered = coords_array - centroid
        I_xx = np.sum(coords_centered[:, 1]**2 + coords_centered[:, 2]**2)
        I_yy = np.sum(coords_centered[:, 0]**2 + coords_centered[:, 2]**2)
        I_zz = np.sum(coords_centered[:, 0]**2 + coords_centered[:, 1]**2)
        
        # Principal moments (simplified - just diagonal)
        moments = np.array([I_xx, I_yy, I_zz])
        if np.sum(moments) > 0:
            moments = moments / np.sum(moments)  # Normalize
            # Asphericity: variance of moments (0 = sphere, 1 = rod/disk)
            asphericity = float(np.std(moments))
        else:
            asphericity = 0.0
        
        return GeometricFeatures(
            centroid=centroid,
            radius_of_gyration=radius_of_gyration,
            distance_ratios=distance_ratios,
            angle_distribution=angle_distribution,
            symmetry_score=symmetry_score,
            asphericity=asphericity
        )
    
    def _score_distance_ratios(self, ratios: List[float]) -> float:
        """
        Score distance ratios against ideal Platonic solid ratios.
        
        Args:
            ratios: Observed distance ratios
            
        Returns:
            Score 0.0-1.0 (1.0 = perfect match)
        """
        if not ratios or self.target_geometry not in self.ideal_ratios:
            return 0.5
        
        ideal = self.ideal_ratios[self.target_geometry]
        
        # Find matches (within 10% tolerance)
        matches = 0
        for ratio in ratios:
            for ideal_ratio in ideal:
                if abs(ratio - ideal_ratio) / ideal_ratio < 0.10:
                    matches += 1
                    break
        
        # Score: fraction of ratios matching ideal
        score = matches / len(ratios)
        return score
    
    def _score_symmetry(self, observed_symmetry: float) -> float:
        """
        Score symmetry against ideal Platonic solid symmetry.
        
        Args:
            observed_symmetry: Observed symmetry score (0-1)
            
        Returns:
            Score 0.0-1.0 (1.0 = perfect symmetry)
        """
        if self.target_geometry not in self.ideal_symmetry:
            return 0.5
        
        # Higher symmetry Platonic solids (icosahedron, dodecahedron) need higher observed symmetry
        ideal_sym = self.ideal_symmetry[self.target_geometry]
        
        # Normalize ideal symmetry to 0-1 range (max is 60 for icosahedron)
        normalized_ideal = ideal_sym / 60.0
        
        # Score: how close is observed to ideal
        diff = abs(observed_symmetry - normalized_ideal)
        score = max(0.0, 1.0 - diff)
        
        return score
    
    def _score_asphericity(self, observed_asphericity: float) -> float:
        """
        Score asphericity (deviation from sphere).
        
        Most Platonic solids are approximately spherical when embedded in 3D,
        so lower asphericity is better (except for very asymmetric proteins).
        
        Args:
            observed_asphericity: Observed asphericity (0-1)
            
        Returns:
            Score 0.0-1.0 (1.0 = ideal asphericity for target)
        """
        # Platonic solids are fairly spherical (asphericity ~0.1-0.3)
        ideal_asphericity = 0.2
        
        # Score: closer to ideal = higher score
        diff = abs(observed_asphericity - ideal_asphericity)
        score = max(0.0, 1.0 - diff / 0.5)  # Normalize by max expected difference
        
        return score
    
    def _score_angles(self, angle_distribution: np.ndarray) -> float:
        """
        Score angle distribution against expected patterns.
        
        Platonic solids have characteristic angle distributions.
        
        Args:
            angle_distribution: Histogram of backbone angles (10 bins, 0 to π)
            
        Returns:
            Score 0.0-1.0 (1.0 = ideal angle distribution)
        """
        # Ideal angle distributions (approximate - based on Platonic solid geometries)
        if self.target_geometry == 'tetrahedron':
            # Tetrahedron: ~109.5° angles (bin 6)
            ideal_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0])
        elif self.target_geometry == 'cube' or self.target_geometry == 'octahedron':
            # Cube/Octahedron: 90° and 180° angles (bins 5, 10)
            ideal_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.4])
        elif self.target_geometry == 'dodecahedron' or self.target_geometry == 'icosahedron':
            # Dodecahedron/Icosahedron: ~72° and ~108° angles (bins 4, 6)
            ideal_dist = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.4, 0.3, 0.0, 0.0, 0.0])
        else:
            return 0.5
        
        # Normalize ideal distribution
        if np.sum(ideal_dist) > 0:
            ideal_dist = ideal_dist / np.sum(ideal_dist)
        
        # Score: similarity between observed and ideal distributions (using KL divergence inverse)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        obs_dist = angle_distribution + epsilon
        ideal_dist_safe = ideal_dist + epsilon
        
        # Normalize
        obs_dist = obs_dist / np.sum(obs_dist)
        ideal_dist_safe = ideal_dist_safe / np.sum(ideal_dist_safe)
        
        # KL divergence: D_KL(ideal || observed)
        kl_div = np.sum(ideal_dist_safe * np.log(ideal_dist_safe / obs_dist))
        
        # Convert KL divergence to similarity score (0 = identical, higher = more different)
        # Map to 0-1 range: score = exp(-kl_div)
        score = np.exp(-kl_div)
        
        return float(score)
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with calculation count, avg/max times
        """
        avg_time_ms = self.total_time_ms / self.calculation_count if self.calculation_count > 0 else 0.0
        
        return {
            'target_geometry': self.target_geometry,
            'calculation_count': self.calculation_count,
            'avg_time_ms': avg_time_ms,
            'max_time_ms': self.max_time_ms,
            'within_target': avg_time_ms < 2.0
        }


def create_scorer(target_geometry: str) -> GeometricScorer:
    """
    Factory function to create geometric scorer.
    
    Args:
        target_geometry: 'none', 'octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube'
        
    Returns:
        Configured GeometricScorer instance
    """
    return GeometricScorer(target_geometry=target_geometry)
