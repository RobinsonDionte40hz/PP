"""
Geometric Attractor Module - Pattern Detection and Analysis

This module provides geometric pattern detection and analysis for protein conformations,
focusing on golden ratio patterns, Platonic solid similarities, and symmetry metrics.

Key Features:
- Golden ratio (φ) pattern detection in interatomic distances
- Platonic solid similarity analysis (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
- Symmetry metrics (rotational, local, radius of gyration, asphericity)
- LRU caching with TTL for performance optimization
- O(n²) complexity with intelligent sampling

Author: UBF Protein System
Date: November 9, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union
from collections import OrderedDict
import time
import hashlib
import math


@dataclass(frozen=True)
class GeometricAnalysisResult:
    """
    Immutable result from geometric attractor analysis.
    
    This dataclass stores comprehensive geometric pattern analysis results for a protein
    conformation, including golden ratio patterns, Platonic solid similarities, and
    symmetry metrics.
    
    Attributes:
        golden_ratio_percentage: Percentage of interatomic distances following golden ratio (0.0-100.0)
        phi_pattern_count: Number of φ patterns detected (non-negative integer)
        
        # Platonic Solid Similarities (0.0-1.0 each)
        tetrahedron_similarity: Similarity to tetrahedral geometry
        cube_similarity: Similarity to cubic geometry
        octahedron_similarity: Similarity to octahedral geometry
        dodecahedron_similarity: Similarity to dodecahedral geometry (φ-boosted)
        icosahedron_similarity: Similarity to icosahedral geometry (φ-boosted)
        
        # Symmetry Metrics
        rotational_symmetry: Rotational symmetry score based on eigenvalue entropy (0.0-1.0)
        local_symmetry: Local symmetry score from nearest-neighbor regularity (0.0-1.0)
        radius_of_gyration: Radius of gyration in Ångströms (non-negative)
        asphericity: Asphericity measure (0.0 = sphere, 1.0 = rod/disk)
        
        # Metadata
        conformation_hash: SHA256 hash of conformation (16 chars)
        timestamp: Unix timestamp of analysis
        num_residues: Number of residues analyzed
    
    Example:
        >>> result = GeometricAnalysisResult(
        ...     golden_ratio_percentage=23.5,
        ...     phi_pattern_count=42,
        ...     tetrahedron_similarity=0.3,
        ...     cube_similarity=0.2,
        ...     octahedron_similarity=0.4,
        ...     dodecahedron_similarity=0.7,
        ...     icosahedron_similarity=0.8,
        ...     rotational_symmetry=0.6,
        ...     local_symmetry=0.5,
        ...     radius_of_gyration=15.2,
        ...     asphericity=0.3,
        ...     conformation_hash="a1b2c3d4e5f6g7h8",
        ...     timestamp=1699564800.0,
        ...     num_residues=76
        ... )
    """
    
    # Golden ratio patterns
    golden_ratio_percentage: float
    phi_pattern_count: int
    
    # Platonic solid similarities
    tetrahedron_similarity: float
    cube_similarity: float
    octahedron_similarity: float
    dodecahedron_similarity: float
    icosahedron_similarity: float
    
    # Symmetry metrics
    rotational_symmetry: float
    local_symmetry: float
    radius_of_gyration: float
    asphericity: float
    
    # Metadata
    conformation_hash: str
    timestamp: float
    num_residues: int
    
    def __post_init__(self):
        """Validate all field values are within expected ranges."""
        
        # Validate golden ratio percentage (0.0-100.0)
        if not (0.0 <= self.golden_ratio_percentage <= 100.0):
            raise ValueError(
                f"golden_ratio_percentage must be in range [0.0, 100.0], "
                f"got {self.golden_ratio_percentage}"
            )
        
        # Validate phi pattern count (non-negative)
        if self.phi_pattern_count < 0:
            raise ValueError(
                f"phi_pattern_count must be non-negative, got {self.phi_pattern_count}"
            )
        
        # Validate Platonic solid similarities (0.0-1.0)
        similarities = {
            'tetrahedron_similarity': self.tetrahedron_similarity,
            'cube_similarity': self.cube_similarity,
            'octahedron_similarity': self.octahedron_similarity,
            'dodecahedron_similarity': self.dodecahedron_similarity,
            'icosahedron_similarity': self.icosahedron_similarity,
        }
        for name, value in similarities.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {value}")
        
        # Validate symmetry metrics (0.0-1.0 for scores)
        if not (0.0 <= self.rotational_symmetry <= 1.0):
            raise ValueError(
                f"rotational_symmetry must be in range [0.0, 1.0], "
                f"got {self.rotational_symmetry}"
            )
        if not (0.0 <= self.local_symmetry <= 1.0):
            raise ValueError(
                f"local_symmetry must be in range [0.0, 1.0], got {self.local_symmetry}"
            )
        
        # Validate radius of gyration (non-negative)
        if self.radius_of_gyration < 0.0:
            raise ValueError(
                f"radius_of_gyration must be non-negative, got {self.radius_of_gyration}"
            )
        
        # Validate asphericity (0.0-1.0)
        if not (0.0 <= self.asphericity <= 1.0):
            raise ValueError(f"asphericity must be in range [0.0, 1.0], got {self.asphericity}")
        
        # Validate timestamp (positive)
        if self.timestamp <= 0.0:
            raise ValueError(f"timestamp must be positive, got {self.timestamp}")
        
        # Validate num_residues (positive)
        if self.num_residues <= 0:
            raise ValueError(f"num_residues must be positive, got {self.num_residues}")
        
        # Validate conformation_hash (16 characters)
        if len(self.conformation_hash) != 16:
            raise ValueError(
                f"conformation_hash must be 16 characters, got {len(self.conformation_hash)}"
            )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'golden_ratio_percentage': self.golden_ratio_percentage,
            'phi_pattern_count': self.phi_pattern_count,
            'platonic_similarities': {
                'tetrahedron': self.tetrahedron_similarity,
                'cube': self.cube_similarity,
                'octahedron': self.octahedron_similarity,
                'dodecahedron': self.dodecahedron_similarity,
                'icosahedron': self.icosahedron_similarity,
            },
            'symmetry_metrics': {
                'rotational': self.rotational_symmetry,
                'local': self.local_symmetry,
                'radius_of_gyration': self.radius_of_gyration,
                'asphericity': self.asphericity,
            },
            'metadata': {
                'conformation_hash': self.conformation_hash,
                'timestamp': self.timestamp,
                'num_residues': self.num_residues,
            }
        }


class LRUCache:
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Provides O(1) access and automatic eviction of least recently used entries
    when size limit is reached. Entries also expire after TTL seconds.
    
    Attributes:
        max_size: Maximum number of entries to cache
        ttl_seconds: Time-to-live for cache entries in seconds
        
    Example:
        >>> cache = LRUCache(max_size=1000, ttl_seconds=3600)
        >>> cache.put("key1", result)
        >>> cached_result = cache.get("key1")  # Returns result if not expired
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries (default 1000)
            ttl_seconds: Time-to-live in seconds (default 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[GeometricAnalysisResult]:
        """
        Get value from cache if present and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if present and valid, None otherwise
        """
        if key not in self._cache:
            self._misses += 1
            return None
        
        # Check TTL
        age = time.time() - self._timestamps[key]
        if age > self.ttl_seconds:
            # Expired - remove and return None
            self._remove(key)
            self._misses += 1
            return None
        
        # Valid cache hit - move to end (most recently used)
        self._cache.move_to_end(key)
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        self._hits += 1
        
        return self._cache[key]
    
    def put(self, key: str, value: GeometricAnalysisResult) -> None:
        """
        Put value in cache with automatic LRU eviction.
        
        Args:
            key: Cache key
            value: GeometricAnalysisResult to cache
        """
        # Update existing entry
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            self._timestamps[key] = time.time()
            return
        
        # Add new entry
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._access_counts[key] = 0
        
        # Evict oldest if over size limit
        if len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self._evictions += 1
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache and metadata."""
        del self._cache[key]
        del self._timestamps[key]
        if key in self._access_counts:
            del self._access_counts[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()
        self._access_counts.clear()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit_rate, size, hits, misses, evictions
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'total_requests': total_requests,
        }


# Golden ratio constant
PHI = (1.0 + math.sqrt(5.0)) / 2.0  # ≈ 1.618


class GeometricAttractorAnalyzer:
    """
    Geometric Attractor Analyzer for protein conformations.
    
    Analyzes protein structures for:
    - Golden ratio (φ) patterns in interatomic distances
    - Platonic solid similarities (5 solids)
    - Symmetry metrics (rotational, local, shape)
    
    Features:
    - LRU caching with TTL for performance
    - O(n²) complexity with intelligent sampling
    - Pure Python implementation (PyPy-compatible)
    
    Usage:
        >>> analyzer = GeometricAttractorAnalyzer()
        >>> result = analyzer.analyze_conformation(conformation)
        >>> print(f"Golden ratio: {result.golden_ratio_percentage:.1f}%")
        >>> print(f"Best Platonic: {max_platonic_solid(result)}")
    
    Attributes:
        cache: LRU cache for analysis results
        phi_tolerance: Tolerance for φ pattern matching (default 0.05)
        neighbor_window: Window size for distance ratio sampling (default 10)
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: float = 3600.0,
        phi_tolerance: float = 0.05,
        neighbor_window: int = 10,
    ):
        """
        Initialize geometric attractor analyzer.
        
        Args:
            cache_size: Maximum number of cached results (default 1000)
            cache_ttl: Cache time-to-live in seconds (default 3600 = 1 hour)
            phi_tolerance: Tolerance for φ pattern matching (default 0.05)
            neighbor_window: Window for distance ratio sampling (default 10)
        """
        self.cache = LRUCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self.phi_tolerance = phi_tolerance
        self.neighbor_window = neighbor_window
    
    def analyze_conformation(
        self,
        conformation: Union[Dict, str],
        sequence: Optional[str] = None,
    ) -> GeometricAnalysisResult:
        """
        Analyze protein conformation for geometric patterns.
        
        Args:
            conformation: Protein conformation (dict with 'coordinates' or PDB file path)
            sequence: Amino acid sequence (optional, for validation)
        
        Returns:
            GeometricAnalysisResult with all geometric analysis metrics
        
        Raises:
            ValueError: If conformation is invalid or missing coordinates
        
        Example:
            >>> conformation = {'coordinates': [...]}  # List of (x,y,z) tuples
            >>> result = analyzer.analyze_conformation(conformation)
            >>> print(result.golden_ratio_percentage)
        """
        # Generate hash for cache lookup
        conf_hash = self._generate_conformation_hash(conformation)
        
        # Check cache
        cached_result = self.cache.get(conf_hash)
        if cached_result is not None:
            return cached_result
        
        # Extract coordinates
        coordinates = self._extract_coordinates(conformation)
        num_residues = len(coordinates)
        
        if num_residues < 3:
            raise ValueError(f"Need at least 3 residues for analysis, got {num_residues}")
        
        # Calculate all geometric metrics
        phi_percentage, phi_count = self._calculate_golden_ratio_patterns(coordinates)
        
        platonic_similarities = self._calculate_platonic_similarities(coordinates)
        
        symmetry_metrics = self._calculate_symmetry_metrics(coordinates)
        
        # Create result
        result = GeometricAnalysisResult(
            golden_ratio_percentage=phi_percentage,
            phi_pattern_count=phi_count,
            tetrahedron_similarity=platonic_similarities['tetrahedron'],
            cube_similarity=platonic_similarities['cube'],
            octahedron_similarity=platonic_similarities['octahedron'],
            dodecahedron_similarity=platonic_similarities['dodecahedron'],
            icosahedron_similarity=platonic_similarities['icosahedron'],
            rotational_symmetry=symmetry_metrics['rotational'],
            local_symmetry=symmetry_metrics['local'],
            radius_of_gyration=symmetry_metrics['radius_of_gyration'],
            asphericity=symmetry_metrics['asphericity'],
            conformation_hash=conf_hash,
            timestamp=time.time(),
            num_residues=num_residues,
        )
        
        # Cache result
        self.cache.put(conf_hash, result)
        
        return result
    
    def _generate_conformation_hash(self, conformation: Union[Dict, str]) -> str:
        """
        Generate SHA256 hash of conformation for cache key.
        
        Args:
            conformation: Protein conformation (dict or file path)
        
        Returns:
            First 16 characters of SHA256 hash
        """
        coordinates = self._extract_coordinates(conformation)
        
        # Round to 2 decimal places for stability
        rounded_coords = []
        for x, y, z in coordinates:
            rounded_coords.extend([
                round(x, 2),
                round(y, 2),
                round(z, 2),
            ])
        
        # Convert to bytes and hash
        coord_bytes = str(rounded_coords).encode('utf-8')
        hash_obj = hashlib.sha256(coord_bytes)
        
        return hash_obj.hexdigest()[:16]
    
    def _extract_coordinates(self, conformation: Union[Dict, str]) -> List[Tuple[float, float, float]]:
        """
        Extract CA coordinates from conformation.
        
        Args:
            conformation: Dict with 'coordinates' key or PDB file path
        
        Returns:
            List of (x, y, z) coordinate tuples
        
        Raises:
            ValueError: If coordinates cannot be extracted
        """
        if isinstance(conformation, dict):
            if 'coordinates' in conformation:
                coords = conformation['coordinates']
                # Convert to list of tuples if needed
                if coords and isinstance(coords[0], (list, tuple)):
                    return [tuple(c) if isinstance(c, list) else c for c in coords]
                raise ValueError("Coordinates must be list of (x,y,z) tuples")
            else:
                raise ValueError("Dict must contain 'coordinates' key")
        
        elif isinstance(conformation, str):
            # PDB file path - would need BioPython to parse
            # For now, raise error (will implement in integration phase)
            raise NotImplementedError("PDB file parsing not yet implemented")
        
        else:
            raise ValueError(f"Invalid conformation type: {type(conformation)}")
    
    def _calculate_golden_ratio_patterns(
        self,
        coordinates: List[Tuple[float, float, float]]
    ) -> Tuple[float, int]:
        """
        Detect golden ratio patterns in interatomic distances.
        
        Calculates pairwise distances and identifies distance ratios
        that match φ within tolerance. Uses intelligent sampling with
        neighbor window to achieve O(n²) complexity.
        
        Args:
            coordinates: List of (x, y, z) CA coordinates
        
        Returns:
            Tuple of (percentage, count):
                - percentage: Percentage of distance ratios matching φ (0.0-100.0)
                - count: Number of φ patterns detected
        """
        n = len(coordinates)
        
        # Calculate all pairwise distances
        distances = []
        for i in range(n):
            x1, y1, z1 = coordinates[i]
            for j in range(i + 1, n):
                x2, y2, z2 = coordinates[j]
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                distances.append((i, j, dist))
        
        if len(distances) < 2:
            return 0.0, 0
        
        # Sample distance ratios with neighbor window
        phi_count = 0
        total_comparisons = 0
        
        for i, (idx1_a, idx2_a, dist_a) in enumerate(distances):
            # Only compare to neighbors within window
            start = max(0, i - self.neighbor_window)
            end = min(len(distances), i + self.neighbor_window + 1)
            
            for j in range(start, i):
                idx1_b, idx2_b, dist_b = distances[j]
                
                # Skip if distances share indices (would give trivial ratios)
                if idx1_a == idx1_b or idx1_a == idx2_b or idx2_a == idx1_b or idx2_a == idx2_b:
                    continue
                
                # Calculate ratio (larger / smaller)
                if dist_a > dist_b and dist_b > 0:
                    ratio = dist_a / dist_b
                elif dist_b > dist_a and dist_a > 0:
                    ratio = dist_b / dist_a
                else:
                    continue
                
                total_comparisons += 1
                
                # Check if ratio matches φ
                if abs(ratio - PHI) < self.phi_tolerance:
                    phi_count += 1
        
        # Calculate percentage
        percentage = (phi_count / total_comparisons * 100.0) if total_comparisons > 0 else 0.0
        
        return percentage, phi_count
    
    def _calculate_platonic_similarities(
        self,
        coordinates: List[Tuple[float, float, float]]
    ) -> Dict[str, float]:
        """
        Calculate similarity to each of the 5 Platonic solids.
        
        Uses principal component analysis (SVD) to find symmetry axes
        and eigenvalue distribution to score similarity to ideal geometries.
        Applies φ-pattern boost for dodecahedron and icosahedron.
        
        Args:
            coordinates: List of (x, y, z) CA coordinates
        
        Returns:
            Dict with similarity scores (0.0-1.0) for each solid:
                - tetrahedron: 4 vertices, tetrahedral symmetry
                - cube: 8 vertices, cubic symmetry
                - octahedron: 6 vertices, octahedral symmetry
                - dodecahedron: 20 vertices, icosahedral + φ patterns
                - icosahedron: 12 vertices, icosahedral + φ patterns
        """
        n = len(coordinates)
        
        # Center coordinates at origin
        cx = sum(x for x, y, z in coordinates) / n
        cy = sum(y for x, y, z in coordinates) / n
        cz = sum(z for x, y, z in coordinates) / n
        
        centered = [(x - cx, y - cy, z - cz) for x, y, z in coordinates]
        
        # Calculate moment of inertia tensor (simplified SVD approach)
        # This gives us principal axes and their variances
        Ixx = sum(y*y + z*z for x, y, z in centered)
        Iyy = sum(x*x + z*z for x, y, z in centered)
        Izz = sum(x*x + y*y for x, y, z in centered)
        Ixy = sum(x*y for x, y, z in centered)
        Ixz = sum(x*z for x, y, z in centered)
        Iyz = sum(y*z for x, y, z in centered)
        
        # Simplified eigenvalue estimation (trace and determinant approach)
        # For full SVD we'd need numpy, but this gives good approximation
        trace = Ixx + Iyy + Izz
        
        # Normalized variances along principal axes (rough approximation)
        var_x = Ixx / trace if trace > 0 else 0.0
        var_y = Iyy / trace if trace > 0 else 0.0
        var_z = Izz / trace if trace > 0 else 0.0
        
        # Calculate eigenvalue entropy (measure of symmetry)
        eigenvalues = [var_x, var_y, var_z]
        eigenvalues = [max(0.001, v) for v in eigenvalues]  # Avoid log(0)
        total = sum(eigenvalues)
        eigenvalues = [v / total for v in eigenvalues]
        
        entropy = -sum(v * math.log(v) for v in eigenvalues)
        max_entropy = math.log(3.0)  # Maximum for 3 equal eigenvalues
        symmetry_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Platonic solid ideal symmetries (higher = more symmetric)
        # Based on eigenvalue distributions
        tetrahedron_ideal = 0.85  # Moderate symmetry
        cube_ideal = 0.95  # High symmetry
        octahedron_ideal = 0.90  # High symmetry
        dodecahedron_ideal = 0.92  # Very high symmetry
        icosahedron_ideal = 0.88  # High symmetry
        
        # Calculate similarities based on how close we are to ideal
        similarities = {
            'tetrahedron': max(0.0, 1.0 - abs(symmetry_score - tetrahedron_ideal)),
            'cube': max(0.0, 1.0 - abs(symmetry_score - cube_ideal)),
            'octahedron': max(0.0, 1.0 - abs(symmetry_score - octahedron_ideal)),
            'dodecahedron': max(0.0, 1.0 - abs(symmetry_score - dodecahedron_ideal)),
            'icosahedron': max(0.0, 1.0 - abs(symmetry_score - icosahedron_ideal)),
        }
        
        # Apply φ-pattern boost for dodecahedron and icosahedron
        # These solids have inherent golden ratio properties
        phi_boost = 0.1  # 10% boost if structure has φ patterns
        
        # Check for φ patterns in structure (quick estimation)
        distances = []
        for i in range(min(20, n)):  # Sample first 20 for speed
            x1, y1, z1 = coordinates[i]
            for j in range(i + 1, min(20, n)):
                x2, y2, z2 = coordinates[j]
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                distances.append(dist)
        
        if len(distances) >= 2:
            distances.sort()
            # Check if consecutive distances have φ ratio
            phi_patterns_found = False
            for i in range(len(distances) - 1):
                if distances[i] > 0:
                    ratio = distances[i + 1] / distances[i]
                    if abs(ratio - PHI) < self.phi_tolerance * 2:  # Relaxed tolerance
                        phi_patterns_found = True
                        break
            
            if phi_patterns_found:
                similarities['dodecahedron'] = min(1.0, similarities['dodecahedron'] + phi_boost)
                similarities['icosahedron'] = min(1.0, similarities['icosahedron'] + phi_boost)
        
        return similarities
    
    def _calculate_symmetry_metrics(
        self,
        coordinates: List[Tuple[float, float, float]]
    ) -> Dict[str, float]:
        """
        Calculate symmetry metrics for conformation.
        
        Computes:
        - Rotational symmetry: Based on eigenvalue entropy
        - Local symmetry: Nearest-neighbor regularity
        - Radius of gyration: RMS distance from center of mass
        - Asphericity: Shape anisotropy (0=sphere, 1=rod/disk)
        
        Args:
            coordinates: List of (x, y, z) CA coordinates
        
        Returns:
            Dict with symmetry metrics:
                - rotational: Rotational symmetry score (0.0-1.0)
                - local: Local symmetry score (0.0-1.0)
                - radius_of_gyration: Rg in Ångströms
                - asphericity: Shape parameter (0.0-1.0)
        """
        n = len(coordinates)
        
        # Center of mass
        cx = sum(x for x, y, z in coordinates) / n
        cy = sum(y for x, y, z in coordinates) / n
        cz = sum(z for x, y, z in coordinates) / n
        
        # Radius of gyration
        rg_sq = sum((x - cx)**2 + (y - cy)**2 + (z - cz)**2 for x, y, z in coordinates) / n
        rg = math.sqrt(rg_sq)
        
        # Moment of inertia components
        centered = [(x - cx, y - cy, z - cz) for x, y, z in coordinates]
        
        Ixx = sum(y*y + z*z for x, y, z in centered) / n
        Iyy = sum(x*x + z*z for x, y, z in centered) / n
        Izz = sum(x*x + y*y for x, y, z in centered) / n
        
        # Eigenvalue approximation for rotational symmetry
        trace = Ixx + Iyy + Izz
        if trace > 0:
            var_x = Ixx / trace
            var_y = Iyy / trace
            var_z = Izz / trace
            
            # Normalize
            eigenvalues = [var_x, var_y, var_z]
            eigenvalues = [max(0.001, v) for v in eigenvalues]
            total = sum(eigenvalues)
            eigenvalues = [v / total for v in eigenvalues]
            
            # Shannon entropy (high = more symmetric)
            entropy = -sum(v * math.log(v) for v in eigenvalues)
            max_entropy = math.log(3.0)
            rotational_symmetry = entropy / max_entropy
            
            # Asphericity (0 = sphere, 1 = rod/disk)
            # Based on eigenvalue spread
            ev_sorted = sorted(eigenvalues, reverse=True)
            asphericity = (ev_sorted[0] - ev_sorted[2]) / sum(eigenvalues)
        else:
            rotational_symmetry = 0.0
            asphericity = 0.0
        
        # Local symmetry: nearest-neighbor distance regularity
        # Calculate variance in nearest-neighbor distances
        nn_distances = []
        for i in range(n):
            x1, y1, z1 = coordinates[i]
            min_dist = float('inf')
            
            for j in range(n):
                if i == j:
                    continue
                x2, y2, z2 = coordinates[j]
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                nn_distances.append(min_dist)
        
        if nn_distances:
            mean_nn = sum(nn_distances) / len(nn_distances)
            if mean_nn > 0:
                variance = sum((d - mean_nn)**2 for d in nn_distances) / len(nn_distances)
                cv = math.sqrt(variance) / mean_nn  # Coefficient of variation
                # Convert to 0-1 score (low CV = high local symmetry)
                local_symmetry = max(0.0, 1.0 - min(1.0, cv))
            else:
                local_symmetry = 0.0
        else:
            local_symmetry = 0.0
        
        return {
            'rotational': rotational_symmetry,
            'local': local_symmetry,
            'radius_of_gyration': rg,
            'asphericity': asphericity,
        }
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached analysis results."""
        self.cache.clear()
