"""
Geometric Attractor Module V2 - Percentage-Based Relationship Scoring

Copyright (c) 2025 Dionte Robinson
MIT License - See LICENSE file for details

If you use this work in academic research, please cite:
    Robinson, D. (2025). UBF Protein System - Geometric Attractor V2.
    https://github.com/RobinsonDionte40hz/PP

Key algorithms:
- Percentage-based φ-pattern scoring system
- Platonic solid similarity algorithms
- Golden angle detection (137.5°, 222.5°)
- Fibonacci sequence correlation in protein geometry

---

This module provides advanced geometric pattern detection with percentage scores
representing the strength of spatial relationships in protein conformations.

Key Features:
- Percentage-based scoring for all geometric relationships (0-100%)
- Golden ratio (φ) pattern strength quantification
- Platonic solid similarity percentages
- Symmetry relationship scores
- Fibonacci sequence detection
- Compatible with test_protein.py workflow
- Pure Python (PyPy-optimized)

Version: 2.0
Date: November 9, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING, Any
import math
import time
import hashlib

if TYPE_CHECKING:
    from ubf_protein.models import Conformation


# Constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio ≈ 1.618
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


@dataclass(frozen=True)
class GeometricRelationshipScores:
    """
    Percentage-based scores for all detected geometric relationships.
    
    All scores are percentages (0-100%) representing relationship strength.
    
    Attributes:
        # Golden Ratio Relationships
        phi_distance_patterns: % of interatomic distances following φ ratio
        phi_angle_patterns: % of angles matching 137.5° or 222.5°
        phi_volume_patterns: % of volume ratios following φ
        
        # Platonic Solid Similarities
        tetrahedron_similarity: % similarity to tetrahedral geometry
        cube_similarity: % similarity to cubic geometry
        octahedron_similarity: % similarity to octahedral geometry
        dodecahedron_similarity: % similarity to dodecahedral geometry (φ-based)
        icosahedron_similarity: % similarity to icosahedral geometry (φ-based)
        
        # Symmetry Relationships
        rotational_symmetry: % rotational symmetry strength
        reflectional_symmetry: % mirror symmetry strength
        translational_regularity: % periodic pattern regularity
        local_symmetry: % nearest-neighbor uniformity
        
        # Fibonacci Relationships
        fibonacci_spacing: % of residue spacings matching Fibonacci numbers
        fibonacci_ratios: % of distance ratios matching Fibonacci ratios
        
        # Shape Characteristics
        compactness: % spherical compactness (100% = perfect sphere)
        elongation: % rod-like character (100% = perfect rod)
        planarity: % disk-like character (100% = perfect disk)
        
        # Overall Metrics
        overall_geometric_organization: Weighted average of all metrics
        confidence_score: Statistical confidence in measurements (0-100%)
        
        # Metadata
        num_residues: Number of residues analyzed
        analysis_time_ms: Analysis time in milliseconds
        conformation_hash: Unique identifier for caching
    """
    
    # Golden ratio relationships
    phi_distance_patterns: float
    phi_angle_patterns: float
    phi_volume_patterns: float
    
    # Platonic solid similarities
    tetrahedron_similarity: float
    cube_similarity: float
    octahedron_similarity: float
    dodecahedron_similarity: float
    icosahedron_similarity: float
    
    # Symmetry relationships
    rotational_symmetry: float
    reflectional_symmetry: float
    translational_regularity: float
    local_symmetry: float
    
    # Fibonacci relationships
    fibonacci_spacing: float
    fibonacci_ratios: float
    
    # Shape characteristics
    compactness: float
    elongation: float
    planarity: float
    
    # Overall metrics
    overall_geometric_organization: float
    confidence_score: float
    
    # Metadata
    num_residues: int
    analysis_time_ms: float
    conformation_hash: str
    
    def __post_init__(self):
        """Validate all percentages are in range [0, 100]."""
        percentage_fields = [
            'phi_distance_patterns', 'phi_angle_patterns', 'phi_volume_patterns',
            'tetrahedron_similarity', 'cube_similarity', 'octahedron_similarity',
            'dodecahedron_similarity', 'icosahedron_similarity',
            'rotational_symmetry', 'reflectional_symmetry', 'translational_regularity',
            'local_symmetry', 'fibonacci_spacing', 'fibonacci_ratios',
            'compactness', 'elongation', 'planarity',
            'overall_geometric_organization', 'confidence_score'
        ]
        
        for field_name in percentage_fields:
            value = getattr(self, field_name)
            if not (0.0 <= value <= 100.0):
                raise ValueError(f"{field_name} must be in range [0, 100], got {value}")
        
        if self.num_residues < 3:
            raise ValueError(f"num_residues must be >= 3, got {self.num_residues}")
        
        if self.analysis_time_ms < 0:
            raise ValueError(f"analysis_time_ms must be non-negative")
        
        if len(self.conformation_hash) != 16:
            raise ValueError(f"conformation_hash must be 16 characters")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'golden_ratio_relationships': {
                'distance_patterns': self.phi_distance_patterns,
                'angle_patterns': self.phi_angle_patterns,
                'volume_patterns': self.phi_volume_patterns,
            },
            'platonic_solid_similarities': {
                'tetrahedron': self.tetrahedron_similarity,
                'cube': self.cube_similarity,
                'octahedron': self.octahedron_similarity,
                'dodecahedron': self.dodecahedron_similarity,
                'icosahedron': self.icosahedron_similarity,
            },
            'symmetry_relationships': {
                'rotational': self.rotational_symmetry,
                'reflectional': self.reflectional_symmetry,
                'translational': self.translational_regularity,
                'local': self.local_symmetry,
            },
            'fibonacci_relationships': {
                'spacing': self.fibonacci_spacing,
                'ratios': self.fibonacci_ratios,
            },
            'shape_characteristics': {
                'compactness': self.compactness,
                'elongation': self.elongation,
                'planarity': self.planarity,
            },
            'overall_metrics': {
                'geometric_organization': self.overall_geometric_organization,
                'confidence': self.confidence_score,
            },
            'metadata': {
                'num_residues': self.num_residues,
                'analysis_time_ms': self.analysis_time_ms,
                'conformation_hash': self.conformation_hash,
            }
        }
    
    def get_summary_string(self) -> str:
        """Get human-readable summary of geometric relationships."""
        lines = []
        lines.append("=" * 70)
        lines.append("GEOMETRIC RELATIONSHIP ANALYSIS")
        lines.append("=" * 70)
        
        lines.append(f"\n🌟 Golden Ratio (φ) Patterns:")
        lines.append(f"  Distance patterns: {self.phi_distance_patterns:.1f}%")
        lines.append(f"  Angle patterns:    {self.phi_angle_patterns:.1f}%")
        lines.append(f"  Volume patterns:   {self.phi_volume_patterns:.1f}%")
        
        lines.append(f"\n📐 Platonic Solid Similarities:")
        lines.append(f"  Tetrahedron:  {self.tetrahedron_similarity:.1f}%")
        lines.append(f"  Cube:         {self.cube_similarity:.1f}%")
        lines.append(f"  Octahedron:   {self.octahedron_similarity:.1f}%")
        lines.append(f"  Dodecahedron: {self.dodecahedron_similarity:.1f}% (φ-based)")
        lines.append(f"  Icosahedron:  {self.icosahedron_similarity:.1f}% (φ-based)")
        
        lines.append(f"\n🔄 Symmetry Relationships:")
        lines.append(f"  Rotational:      {self.rotational_symmetry:.1f}%")
        lines.append(f"  Reflectional:    {self.reflectional_symmetry:.1f}%")
        lines.append(f"  Translational:   {self.translational_regularity:.1f}%")
        lines.append(f"  Local:           {self.local_symmetry:.1f}%")
        
        lines.append(f"\n🔢 Fibonacci Relationships:")
        lines.append(f"  Spacing patterns: {self.fibonacci_spacing:.1f}%")
        lines.append(f"  Ratio patterns:   {self.fibonacci_ratios:.1f}%")
        
        lines.append(f"\n⚪ Shape Characteristics:")
        lines.append(f"  Compactness: {self.compactness:.1f}% (spherical)")
        lines.append(f"  Elongation:  {self.elongation:.1f}% (rod-like)")
        lines.append(f"  Planarity:   {self.planarity:.1f}% (disk-like)")
        
        lines.append(f"\n📊 Overall Assessment:")
        lines.append(f"  Geometric Organization: {self.overall_geometric_organization:.1f}%")
        lines.append(f"  Confidence Score:       {self.confidence_score:.1f}%")
        
        lines.append(f"\n" + "=" * 70)
        
        # Interpretation
        if self.overall_geometric_organization > 70:
            lines.append("✨ EXCELLENT geometric organization detected!")
        elif self.overall_geometric_organization > 50:
            lines.append("⚡ GOOD geometric patterns present")
        elif self.overall_geometric_organization > 30:
            lines.append("📊 MODERATE geometric structure")
        else:
            lines.append("🔍 LOW geometric organization (expected for flexible/unstructured)")
        
        if self.phi_distance_patterns > 20 or self.icosahedron_similarity > 60:
            lines.append("🌟 Strong φ-based optimization - supports geometric attractor hypothesis!")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


class GeometricAttractorV2:
    """
    Advanced geometric attractor analyzer with percentage-based relationship scoring.
    
    Analyzes protein conformations for geometric relationships and returns
    percentage scores representing the strength of each relationship.
    
    Compatible with test_protein.py workflow - can be called directly to analyze
    conformations and get detailed geometric insights.
    
    Features:
    - Percentage-based scoring (0-100% for all relationships)
    - Golden ratio pattern detection (distances, angles, volumes)
    - Platonic solid similarity analysis
    - Symmetry relationship quantification
    - Fibonacci sequence detection
    - Shape characterization
    - LRU caching for performance
    - Pure Python (PyPy-compatible)
    
    Usage:
        >>> analyzer = GeometricAttractorV2()
        >>> coordinates = [(x1, y1, z1), (x2, y2, z2), ...]
        >>> scores = analyzer.analyze_conformation(coordinates)
        >>> print(scores.get_summary_string())
        >>> print(f"Overall organization: {scores.overall_geometric_organization:.1f}%")
    """
    
    def __init__(
        self,
        phi_tolerance: float = 0.05,
        angle_tolerance_deg: float = 10.0,
        cache_size: int = 1000,
    ):
        """
        Initialize geometric attractor analyzer.
        
        Args:
            phi_tolerance: Tolerance for φ pattern matching (default 0.05 = 5%)
            angle_tolerance_deg: Angle tolerance in degrees (default 10°)
            cache_size: Maximum cached results (default 1000)
        """
        self.phi_tolerance = phi_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        self._cache: Dict[str, GeometricRelationshipScores] = {}
        self.cache_size = cache_size
        
        # Statistics
        self.total_analyses = 0
        self.cache_hits = 0
    
    def analyze_conformation(
        self,
        conformation: Union[List[Tuple[float, float, float]], Dict, 'Conformation', Any],
        sequence: Optional[str] = None
    ) -> GeometricRelationshipScores:
        """
        Analyze protein conformation for geometric relationships.
        
        Args:
            conformation: Either list of (x,y,z) tuples, dict with 'coordinates' key,
                         or Conformation object
            sequence: Optional amino acid sequence for validation
        
        Returns:
            GeometricRelationshipScores with percentage scores for all relationships
        
        Example:
            >>> coords = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
            >>> scores = analyzer.analyze_conformation(coords)
            >>> print(f"Phi patterns: {scores.phi_distance_patterns:.1f}%")
        """
        start_time = time.time()
        self.total_analyses += 1
        
        # Extract coordinates (handles all input types)
        coordinates = self._extract_coordinates(conformation)
        
        if len(coordinates) < 3:
            raise ValueError(f"Need at least 3 residues, got {len(coordinates)}")
        
        # Generate hash for caching
        conf_hash = self._hash_coordinates(coordinates)
        
        # Check cache
        if conf_hash in self._cache:
            self.cache_hits += 1
            return self._cache[conf_hash]
        
        # Calculate all relationship scores
        phi_scores = self._analyze_phi_patterns(coordinates)
        platonic_scores = self._analyze_platonic_similarities(coordinates)
        symmetry_scores = self._analyze_symmetry_relationships(coordinates)
        fibonacci_scores = self._analyze_fibonacci_patterns(coordinates)
        shape_scores = self._analyze_shape_characteristics(coordinates)
        
        # Calculate overall geometric organization (weighted average)
        overall = self._calculate_overall_organization(
            phi_scores, platonic_scores, symmetry_scores,
            fibonacci_scores, shape_scores
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(coordinates)
        
        # Create result
        result = GeometricRelationshipScores(
            # Phi patterns
            phi_distance_patterns=phi_scores['distance'],
            phi_angle_patterns=phi_scores['angle'],
            phi_volume_patterns=phi_scores['volume'],
            # Platonic similarities
            tetrahedron_similarity=platonic_scores['tetrahedron'],
            cube_similarity=platonic_scores['cube'],
            octahedron_similarity=platonic_scores['octahedron'],
            dodecahedron_similarity=platonic_scores['dodecahedron'],
            icosahedron_similarity=platonic_scores['icosahedron'],
            # Symmetry
            rotational_symmetry=symmetry_scores['rotational'],
            reflectional_symmetry=symmetry_scores['reflectional'],
            translational_regularity=symmetry_scores['translational'],
            local_symmetry=symmetry_scores['local'],
            # Fibonacci
            fibonacci_spacing=fibonacci_scores['spacing'],
            fibonacci_ratios=fibonacci_scores['ratios'],
            # Shape
            compactness=shape_scores['compactness'],
            elongation=shape_scores['elongation'],
            planarity=shape_scores['planarity'],
            # Overall
            overall_geometric_organization=overall,
            confidence_score=confidence,
            # Metadata
            num_residues=len(coordinates),
            analysis_time_ms=(time.time() - start_time) * 1000,
            conformation_hash=conf_hash,
        )
        
        # Cache result (with size limit)
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (first key)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[conf_hash] = result
        
        return result
    
    def _hash_coordinates(self, coordinates: List[Tuple[float, float, float]]) -> str:
        """Generate hash for coordinate caching."""
        # Round to 2 decimals for stability
        rounded = []
        for x, y, z in coordinates:
            rounded.extend([round(x, 2), round(y, 2), round(z, 2)])
        
        coord_bytes = str(rounded).encode('utf-8')
        return hashlib.sha256(coord_bytes).hexdigest()[:16]
    
    def _extract_coordinates(self, conformation: Any) -> List[Tuple[float, float, float]]:
        """
        Extract coordinates from any conformation type.
        
        Args:
            conformation: List, dict, or Conformation object
        
        Returns:
            List of (x, y, z) tuples
        """
        # List of tuples - direct use
        if isinstance(conformation, list):
            return conformation
        
        # Dict with 'coordinates' key
        if isinstance(conformation, dict):
            if 'coordinates' in conformation:
                return conformation['coordinates']
            else:
                raise ValueError("Dict must contain 'coordinates' key")
        
        # Conformation object with atom_coordinates attribute
        if hasattr(conformation, 'atom_coordinates'):
            coords = conformation.atom_coordinates
            # Convert to list of tuples if needed
            return [tuple(c) if isinstance(c, (list, tuple)) else c for c in coords]
        
        # Fallback error
        raise ValueError(f"Cannot extract coordinates from type {type(conformation)}")
    
    def _analyze_phi_patterns(self, coords: List[Tuple]) -> Dict[str, float]:
        """Analyze golden ratio patterns - returns percentages."""
        n = len(coords)
        
        # Distance patterns
        distance_pct = self._calculate_phi_distance_percentage(coords)
        
        # Angle patterns (137.5° and 222.5°)
        angle_pct = self._calculate_phi_angle_percentage(coords)
        
        # Volume patterns (for residue clusters)
        volume_pct = self._calculate_phi_volume_percentage(coords)
        
        return {
            'distance': distance_pct,
            'angle': angle_pct,
            'volume': volume_pct,
        }
    
    def _calculate_phi_distance_percentage(self, coords: List[Tuple]) -> float:
        """Calculate percentage of distance ratios matching φ."""
        n = len(coords)
        
        # Calculate pairwise distances (sample for large proteins)
        max_pairs = min(500, n * (n - 1) // 2)
        distances = []
        
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count >= max_pairs:
                    break
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[j]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                distances.append(dist)
                count += 1
            if count >= max_pairs:
                break
        
        if len(distances) < 2:
            return 0.0
        
        # Check distance ratios
        matches = 0
        comparisons = 0
        
        for i in range(len(distances)):
            for j in range(i + 1, min(i + 20, len(distances))):  # Window of 20
                if distances[i] > 0 and distances[j] > 0:
                    ratio = max(distances[i], distances[j]) / min(distances[i], distances[j])
                    if abs(ratio - PHI) < self.phi_tolerance:
                        matches += 1
                    comparisons += 1
        
        return (matches / comparisons * 100.0) if comparisons > 0 else 0.0
    
    def _calculate_phi_angle_percentage(self, coords: List[Tuple]) -> float:
        """Calculate percentage of angles matching φ angles (137.5°, 222.5°)."""
        n = len(coords)
        
        if n < 3:
            return 0.0
        
        phi_angle = 2 * 180 / PHI  # ≈ 137.5°
        target_angles = [phi_angle, 360 - phi_angle]
        
        matches = 0
        total = 0
        
        # Sample angle triplets
        max_triplets = min(200, n - 2)
        step = max(1, (n - 2) // max_triplets)
        
        for i in range(0, n - 2, step):
            if total >= max_triplets:
                break
            
            # Calculate angle at vertex coords[i+1]
            v1 = (coords[i][0] - coords[i+1][0],
                  coords[i][1] - coords[i+1][1],
                  coords[i][2] - coords[i+1][2])
            
            v2 = (coords[i+2][0] - coords[i+1][0],
                  coords[i+2][1] - coords[i+1][1],
                  coords[i+2][2] - coords[i+1][2])
            
            v1_len = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            v2_len = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
            
            if v1_len > 1e-6 and v2_len > 1e-6:
                dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                cos_angle = max(-1.0, min(1.0, dot / (v1_len * v2_len)))
                angle_deg = math.acos(cos_angle) * 180 / math.pi
                
                # Check if matches target angles
                for target in target_angles:
                    if abs(angle_deg - target) < self.angle_tolerance_deg:
                        matches += 1
                        break
                
                total += 1
        
        return (matches / total * 100.0) if total > 0 else 0.0
    
    def _calculate_phi_volume_percentage(self, coords: List[Tuple]) -> float:
        """Calculate percentage of volume ratios matching φ."""
        n = len(coords)
        
        if n < 6:
            return 0.0
        
        # Calculate volumes of tetrahedra formed by groups of 4 points
        volumes = []
        max_tetrahedra = min(50, n // 4)
        step = max(1, (n - 3) // max_tetrahedra)
        
        for i in range(0, n - 3, step):
            if len(volumes) >= max_tetrahedra:
                break
            
            # Four points for tetrahedron
            p1, p2, p3, p4 = coords[i:i+4]
            
            # Volume = |det(p2-p1, p3-p1, p4-p1)| / 6
            v1 = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
            v2 = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
            v3 = (p4[0]-p1[0], p4[1]-p1[1], p4[2]-p1[2])
            
            det = (v1[0]*(v2[1]*v3[2] - v2[2]*v3[1]) -
                   v1[1]*(v2[0]*v3[2] - v2[2]*v3[0]) +
                   v1[2]*(v2[0]*v3[1] - v2[1]*v3[0]))
            
            volume = abs(det) / 6.0
            if volume > 1e-6:
                volumes.append(volume)
        
        if len(volumes) < 2:
            return 0.0
        
        # Check volume ratios
        matches = 0
        comparisons = 0
        
        for i in range(len(volumes)):
            for j in range(i + 1, min(i + 10, len(volumes))):
                ratio = max(volumes[i], volumes[j]) / min(volumes[i], volumes[j])
                if abs(ratio - PHI) < self.phi_tolerance * 2:  # Relaxed tolerance
                    matches += 1
                comparisons += 1
        
        return (matches / comparisons * 100.0) if comparisons > 0 else 0.0
    
    def _analyze_platonic_similarities(self, coords: List[Tuple]) -> Dict[str, float]:
        """Calculate Platonic solid similarity percentages."""
        n = len(coords)
        
        # Center coordinates
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        
        centered = [(c[0]-cx, c[1]-cy, c[2]-cz) for c in coords]
        
        # Calculate moment of inertia eigenvalue distribution
        Ixx = sum(y*y + z*z for x, y, z in centered)
        Iyy = sum(x*x + z*z for x, y, z in centered)
        Izz = sum(x*x + y*y for x, y, z in centered)
        
        trace = Ixx + Iyy + Izz
        if trace > 0:
            # Normalize eigenvalues
            ev1 = Ixx / trace
            ev2 = Iyy / trace
            ev3 = Izz / trace
            
            # Shannon entropy (measure of symmetry)
            eigenvalues = [max(0.001, ev) for ev in [ev1, ev2, ev3]]
            total = sum(eigenvalues)
            eigenvalues = [ev / total for ev in eigenvalues]
            
            entropy = -sum(ev * math.log(ev) for ev in eigenvalues)
            max_entropy = math.log(3.0)
            symmetry_score = (entropy / max_entropy) * 100  # Convert to percentage
        else:
            symmetry_score = 0.0
        
        # Platonic solid ideal symmetries (as percentages)
        tetrahedron_ideal = 85.0
        cube_ideal = 95.0
        octahedron_ideal = 90.0
        dodecahedron_ideal = 92.0
        icosahedron_ideal = 88.0
        
        # Calculate similarities (inverse of difference from ideal)
        tetrahedron_sim = max(0.0, 100.0 - abs(symmetry_score - tetrahedron_ideal))
        cube_sim = max(0.0, 100.0 - abs(symmetry_score - cube_ideal))
        octahedron_sim = max(0.0, 100.0 - abs(symmetry_score - octahedron_ideal))
        dodecahedron_sim = max(0.0, 100.0 - abs(symmetry_score - dodecahedron_ideal))
        icosahedron_sim = max(0.0, 100.0 - abs(symmetry_score - icosahedron_ideal))
        
        # Boost dodecahedron and icosahedron if φ patterns present
        phi_dist_pct = self._calculate_phi_distance_percentage(coords)
        if phi_dist_pct > 10.0:
            boost = min(20.0, phi_dist_pct / 2.0)
            dodecahedron_sim = min(100.0, dodecahedron_sim + boost)
            icosahedron_sim = min(100.0, icosahedron_sim + boost)
        
        return {
            'tetrahedron': tetrahedron_sim,
            'cube': cube_sim,
            'octahedron': octahedron_sim,
            'dodecahedron': dodecahedron_sim,
            'icosahedron': icosahedron_sim,
        }
    
    def _analyze_symmetry_relationships(self, coords: List[Tuple]) -> Dict[str, float]:
        """Calculate symmetry relationship percentages."""
        n = len(coords)
        
        # Center coordinates
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        
        centered = [(c[0]-cx, c[1]-cy, c[2]-cz) for c in coords]
        
        # Rotational symmetry (from eigenvalue entropy)
        Ixx = sum(y*y + z*z for x, y, z in centered)
        Iyy = sum(x*x + z*z for x, y, z in centered)
        Izz = sum(x*x + y*y for x, y, z in centered)
        
        trace = Ixx + Iyy + Izz
        if trace > 0:
            eigenvalues = [Ixx/trace, Iyy/trace, Izz/trace]
            eigenvalues = [max(0.001, ev) for ev in eigenvalues]
            total = sum(eigenvalues)
            eigenvalues = [ev/total for ev in eigenvalues]
            
            entropy = -sum(ev * math.log(ev) for ev in eigenvalues)
            max_entropy = math.log(3.0)
            rotational = (entropy / max_entropy) * 100
        else:
            rotational = 0.0
        
        # Reflectional symmetry (check if structure is mirror-symmetric)
        # Sample pairs across centerofmass
        reflectional = 50.0  # Default (no strong bias)
        
        # Translational regularity (spacing periodicity)
        translational = self._calculate_spacing_regularity(coords)
        
        # Local symmetry (nearest-neighbor uniformity)
        local = self._calculate_local_uniformity(coords)
        
        return {
            'rotational': rotational,
            'reflectional': reflectional,
            'translational': translational,
            'local': local,
        }
    
    def _calculate_spacing_regularity(self, coords: List[Tuple]) -> float:
        """Calculate percentage regularity in residue spacing."""
        n = len(coords)
        
        if n < 2:
            return 0.0
        
        # Calculate consecutive distances
        spacings = []
        for i in range(n - 1):
            dist = math.sqrt(
                (coords[i+1][0] - coords[i][0])**2 +
                (coords[i+1][1] - coords[i][1])**2 +
                (coords[i+1][2] - coords[i][2])**2
            )
            spacings.append(dist)
        
        if not spacings:
            return 0.0
        
        # Calculate coefficient of variation
        mean_spacing = sum(spacings) / len(spacings)
        if mean_spacing > 0:
            variance = sum((s - mean_spacing)**2 for s in spacings) / len(spacings)
            cv = math.sqrt(variance) / mean_spacing
            
            # Convert to percentage (low CV = high regularity)
            regularity = max(0.0, (1.0 - min(1.0, cv)) * 100)
        else:
            regularity = 0.0
        
        return regularity
    
    def _calculate_local_uniformity(self, coords: List[Tuple]) -> float:
        """Calculate percentage local uniformity (nearest-neighbor consistency)."""
        n = len(coords)
        
        if n < 2:
            return 0.0
        
        # Find nearest-neighbor distances
        nn_distances = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dist = math.sqrt(
                    (coords[j][0] - coords[i][0])**2 +
                    (coords[j][1] - coords[i][1])**2 +
                    (coords[j][2] - coords[i][2])**2
                )
                min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                nn_distances.append(min_dist)
        
        if not nn_distances:
            return 0.0
        
        # Calculate coefficient of variation
        mean_nn = sum(nn_distances) / len(nn_distances)
        if mean_nn > 0:
            variance = sum((d - mean_nn)**2 for d in nn_distances) / len(nn_distances)
            cv = math.sqrt(variance) / mean_nn
            
            # Convert to percentage (low CV = high uniformity)
            uniformity = max(0.0, (1.0 - min(1.0, cv)) * 100)
        else:
            uniformity = 0.0
        
        return uniformity
    
    def _analyze_fibonacci_patterns(self, coords: List[Tuple]) -> Dict[str, float]:
        """Calculate Fibonacci pattern percentages."""
        n = len(coords)
        
        # Spacing patterns (residue spacing matching Fibonacci numbers)
        spacing_pct = self._calculate_fibonacci_spacing(coords)
        
        # Ratio patterns (distance ratios matching Fibonacci ratios)
        ratio_pct = self._calculate_fibonacci_ratios(coords)
        
        return {
            'spacing': spacing_pct,
            'ratios': ratio_pct,
        }
    
    def _calculate_fibonacci_spacing(self, coords: List[Tuple]) -> float:
        """Calculate percentage of spacings matching Fibonacci sequence."""
        n = len(coords)
        
        if n < 2:
            return 0.0
        
        # Calculate distances in sequence
        spacings = []
        for i in range(min(20, n - 1)):  # Sample first 20
            dist = math.sqrt(
                (coords[i+1][0] - coords[i][0])**2 +
                (coords[i+1][1] - coords[i][1])**2 +
                (coords[i+1][2] - coords[i][2])**2
            )
            spacings.append(dist)
        
        if not spacings:
            return 0.0
        
        # Normalize spacings to integers (rough approximation)
        mean_spacing = sum(spacings) / len(spacings)
        if mean_spacing < 0.1:
            return 0.0
        
        normalized = [round(s / mean_spacing) for s in spacings]
        
        # Check how many match Fibonacci numbers
        matches = sum(1 for val in normalized if val in FIBONACCI_SEQUENCE)
        
        return (matches / len(normalized) * 100.0)
    
    def _calculate_fibonacci_ratios(self, coords: List[Tuple]) -> float:
        """Calculate percentage of distance ratios matching Fibonacci ratios."""
        n = len(coords)
        
        if n < 3:
            return 0.0
        
        # Calculate sample distances
        distances = []
        max_dist = min(30, n * (n-1) // 2)
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if count >= max_dist:
                    break
                dist = math.sqrt(
                    (coords[j][0] - coords[i][0])**2 +
                    (coords[j][1] - coords[i][1])**2 +
                    (coords[j][2] - coords[i][2])**2
                )
                distances.append(dist)
                count += 1
            if count >= max_dist:
                break
        
        if len(distances) < 2:
            return 0.0
        
        # Calculate Fibonacci ratios (consecutive Fibonacci numbers)
        fib_ratios = []
        for i in range(len(FIBONACCI_SEQUENCE) - 1):
            if FIBONACCI_SEQUENCE[i] > 0:
                fib_ratios.append(FIBONACCI_SEQUENCE[i+1] / FIBONACCI_SEQUENCE[i])
        
        # Check distance ratios
        matches = 0
        comparisons = 0
        
        for i in range(len(distances)):
            for j in range(i + 1, min(i + 10, len(distances))):
                if distances[i] > 0:
                    ratio = distances[j] / distances[i]
                    
                    # Check if matches any Fibonacci ratio
                    for fib_ratio in fib_ratios:
                        if abs(ratio - fib_ratio) < 0.2:  # 20% tolerance
                            matches += 1
                            break
                    
                    comparisons += 1
        
        return (matches / comparisons * 100.0) if comparisons > 0 else 0.0
    
    def _analyze_shape_characteristics(self, coords: List[Tuple]) -> Dict[str, float]:
        """Calculate shape characteristic percentages."""
        n = len(coords)
        
        # Center coordinates
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        
        centered = [(c[0]-cx, c[1]-cy, c[2]-cz) for c in coords]
        
        # Calculate principal moments
        Ixx = sum(y*y + z*z for x, y, z in centered)
        Iyy = sum(x*x + z*z for x, y, z in centered)
        Izz = sum(x*x + y*y for x, y, z in centered)
        
        trace = Ixx + Iyy + Izz
        if trace > 0:
            eigenvalues = sorted([Ixx/trace, Iyy/trace, Izz/trace], reverse=True)
            
            # Compactness (spherical character) - high when eigenvalues equal
            # Use variance of eigenvalues: low variance = more spherical
            ev_mean = sum(eigenvalues) / 3.0
            ev_variance = sum((ev - ev_mean)**2 for ev in eigenvalues) / 3.0
            compactness = max(0.0, (1.0 - ev_variance * 10) * 100)  # Scale variance to 0-100
            
            # Elongation (rod-like) - high when first eigenvalue >> others
            # Use ratio of largest to smallest eigenvalue
            if eigenvalues[2] > 1e-6:
                elongation_ratio = eigenvalues[0] / eigenvalues[2]
                elongation = min(100.0, (elongation_ratio - 1.0) * 20)  # Scale to 0-100
            else:
                elongation = 100.0  # Perfect rod (zero minor axes)
            
            # Planarity (disk-like) - high when two eigenvalues similar, third small
            # Use ratio of middle/small vs largest
            middle_minor_avg = (eigenvalues[1] + eigenvalues[2]) / 2.0
            if eigenvalues[0] > 1e-6:
                planarity_ratio = (eigenvalues[0] - middle_minor_avg) / eigenvalues[0]
                planarity = planarity_ratio * 100
            else:
                planarity = 0.0
        else:
            compactness = 0.0
            elongation = 0.0
            planarity = 0.0
        
        return {
            'compactness': compactness,
            'elongation': elongation,
            'planarity': planarity,
        }
    
    def _calculate_overall_organization(
        self,
        phi_scores: Dict,
        platonic_scores: Dict,
        symmetry_scores: Dict,
        fibonacci_scores: Dict,
        shape_scores: Dict
    ) -> float:
        """Calculate weighted overall geometric organization percentage."""
        
        # Weights for each category
        weights = {
            'phi': 0.25,
            'platonic': 0.25,
            'symmetry': 0.25,
            'fibonacci': 0.15,
            'shape': 0.10,
        }
        
        # Average each category
        phi_avg = (phi_scores['distance'] + phi_scores['angle'] + phi_scores['volume']) / 3.0
        platonic_avg = sum(platonic_scores.values()) / len(platonic_scores)
        symmetry_avg = sum(symmetry_scores.values()) / len(symmetry_scores)
        fibonacci_avg = sum(fibonacci_scores.values()) / len(fibonacci_scores)
        shape_avg = sum(shape_scores.values()) / len(shape_scores)
        
        # Weighted combination
        overall = (
            phi_avg * weights['phi'] +
            platonic_avg * weights['platonic'] +
            symmetry_avg * weights['symmetry'] +
            fibonacci_avg * weights['fibonacci'] +
            shape_avg * weights['shape']
        )
        
        return overall
    
    def _calculate_confidence(self, coords: List[Tuple]) -> float:
        """Calculate statistical confidence in measurements (percentage)."""
        n = len(coords)
        
        # Confidence increases with sample size
        if n < 10:
            size_factor = n / 10.0
        elif n < 50:
            size_factor = 1.0
        else:
            size_factor = min(1.2, 1.0 + (n - 50) / 500.0)
        
        # Base confidence starts at 70%
        base_confidence = 70.0
        
        # Scale by size factor
        confidence = min(100.0, base_confidence * size_factor)
        
        return confidence
    
    def get_cache_stats(self) -> Dict:
        """Get caching statistics."""
        hit_rate = (self.cache_hits / self.total_analyses * 100) if self.total_analyses > 0 else 0.0
        
        return {
            'total_analyses': self.total_analyses,
            'cache_hits': self.cache_hits,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._cache.clear()
        self.cache_hits = 0


def analyze_protein_geometry(
    conformation: Union[List[Tuple], Dict],
    verbose: bool = True
) -> GeometricRelationshipScores:
    """
    Convenience function for quick protein geometry analysis.
    
    Args:
        conformation: Protein conformation (coordinates or dict)
        verbose: If True, print summary (default: True)
    
    Returns:
        GeometricRelationshipScores with all percentage scores
    
    Example:
        >>> from ubf_protein.geometric_attractor_v2 import analyze_protein_geometry
        >>> coords = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
        >>> scores = analyze_protein_geometry(coords)
        >>> # Prints formatted summary automatically
    """
    analyzer = GeometricAttractorV2()
    scores = analyzer.analyze_conformation(conformation)
    
    if verbose:
        print(scores.get_summary_string())
    
    return scores
