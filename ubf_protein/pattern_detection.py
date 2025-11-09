"""
Pattern Detection Module - Data Models

This module defines data models for pattern detection in protein conformations,
including THz resonance patterns, folding dynamics, and geometric similarities.

Key Features:
- Three pattern types: THz, Folding, Geometric
- Immutable pattern detection results
- Significance scoring (low, medium, high)
- Rich metadata for each pattern type

Author: UBF Protein System
Date: November 9, 2025
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum


class PatternType(Enum):
    """
    Types of patterns detected in protein conformations.
    
    Attributes:
        THZ: THz resonance patterns from QCPP analysis
        FOLDING: Secondary structure folding dynamics
        GEOMETRIC: Geometric similarity to reference conformations
    """
    THZ = "thz_resonance"
    FOLDING = "folding_dynamics"
    GEOMETRIC = "geometric_similarity"


class PatternSignificance(Enum):
    """
    Significance levels for detected patterns.
    
    Attributes:
        LOW: Minor pattern, low priority for broadcast
        MEDIUM: Moderate pattern, standard broadcast
        HIGH: Major pattern, high priority for relay and broadcast
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class THzResonanceData:
    """
    THz resonance pattern data from QCPP analysis.
    
    Attributes:
        cluster_id: ID of the THz signature cluster (non-negative)
        cluster_size: Number of conformations in cluster (positive)
        similarity_score: Spectral correlation score (0.0-1.0)
        dominant_frequency: Dominant THz frequency in the cluster (GHz)
        spectral_entropy: Shannon entropy of THz spectrum (non-negative)
    
    Example:
        >>> thz_data = THzResonanceData(
        ...     cluster_id=3,
        ...     cluster_size=12,
        ...     similarity_score=0.85,
        ...     dominant_frequency=2.45,
        ...     spectral_entropy=1.23
        ... )
    """
    cluster_id: int
    cluster_size: int
    similarity_score: float
    dominant_frequency: float
    spectral_entropy: float
    
    def __post_init__(self):
        """Validate field values."""
        if self.cluster_id < 0:
            raise ValueError(f"cluster_id must be non-negative, got {self.cluster_id}")
        
        if self.cluster_size <= 0:
            raise ValueError(f"cluster_size must be positive, got {self.cluster_size}")
        
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError(
                f"similarity_score must be in range [0.0, 1.0], got {self.similarity_score}"
            )
        
        if self.dominant_frequency < 0.0:
            raise ValueError(
                f"dominant_frequency must be non-negative, got {self.dominant_frequency}"
            )
        
        if self.spectral_entropy < 0.0:
            raise ValueError(
                f"spectral_entropy must be non-negative, got {self.spectral_entropy}"
            )


@dataclass(frozen=True)
class FoldingDynamicsData:
    """
    Folding dynamics pattern data from secondary structure analysis.
    
    Attributes:
        helix_percentage: Percentage of residues in helical conformation (0.0-100.0)
        sheet_percentage: Percentage of residues in sheet conformation (0.0-100.0)
        turn_percentage: Percentage of residues in turn conformation (0.0-100.0)
        coil_percentage: Percentage of residues in coil conformation (0.0-100.0)
        helix_regions: List of (start, end) tuples for continuous helix regions
        sheet_regions: List of (start, end) tuples for continuous sheet regions
        turn_regions: List of (start, end) tuples for turn regions
    
    Example:
        >>> folding_data = FoldingDynamicsData(
        ...     helix_percentage=35.2,
        ...     sheet_percentage=22.1,
        ...     turn_percentage=12.5,
        ...     coil_percentage=30.2,
        ...     helix_regions=[(5, 18), (25, 38)],
        ...     sheet_regions=[(42, 48), (55, 62)],
        ...     turn_regions=[(19, 22)]
        ... )
    """
    helix_percentage: float
    sheet_percentage: float
    turn_percentage: float
    coil_percentage: float
    helix_regions: List[Tuple[int, int]]
    sheet_regions: List[Tuple[int, int]]
    turn_regions: List[Tuple[int, int]]
    
    def __post_init__(self):
        """Validate field values."""
        # Validate percentages (0.0-100.0)
        percentages = {
            'helix_percentage': self.helix_percentage,
            'sheet_percentage': self.sheet_percentage,
            'turn_percentage': self.turn_percentage,
            'coil_percentage': self.coil_percentage,
        }
        for name, value in percentages.items():
            if not (0.0 <= value <= 100.0):
                raise ValueError(f"{name} must be in range [0.0, 100.0], got {value}")
        
        # Validate percentages sum to ~100% (allow 1% tolerance for rounding)
        total = sum(percentages.values())
        if not (99.0 <= total <= 101.0):
            raise ValueError(
                f"Percentages must sum to ~100%, got {total:.1f}% "
                f"(helix={self.helix_percentage}, sheet={self.sheet_percentage}, "
                f"turn={self.turn_percentage}, coil={self.coil_percentage})"
            )
        
        # Validate regions (start < end)
        for region_name, regions in [
            ('helix_regions', self.helix_regions),
            ('sheet_regions', self.sheet_regions),
            ('turn_regions', self.turn_regions),
        ]:
            for start, end in regions:
                if start >= end:
                    raise ValueError(
                        f"{region_name} must have start < end, got ({start}, {end})"
                    )


@dataclass(frozen=True)
class GeometricSimilarityData:
    """
    Geometric similarity pattern data from RMSD analysis.
    
    Attributes:
        rmsd_to_reference: RMSD to reference conformation in Ångströms (non-negative)
        overlap_percentage: Percentage of residues within 2.0 Å of reference (0.0-100.0)
        reference_conformation_hash: Hash of reference conformation (16 chars)
        golden_ratio_percentage: Golden ratio pattern percentage (0.0-100.0)
        dominant_platonic_solid: Name of most similar Platonic solid
        platonic_similarity_score: Similarity score to dominant solid (0.0-1.0)
    
    Example:
        >>> geo_data = GeometricSimilarityData(
        ...     rmsd_to_reference=1.85,
        ...     overlap_percentage=78.5,
        ...     reference_conformation_hash="a1b2c3d4e5f6g7h8",
        ...     golden_ratio_percentage=24.3,
        ...     dominant_platonic_solid="icosahedron",
        ...     platonic_similarity_score=0.82
        ... )
    """
    rmsd_to_reference: float
    overlap_percentage: float
    reference_conformation_hash: str
    golden_ratio_percentage: float
    dominant_platonic_solid: str
    platonic_similarity_score: float
    
    def __post_init__(self):
        """Validate field values."""
        if self.rmsd_to_reference < 0.0:
            raise ValueError(
                f"rmsd_to_reference must be non-negative, got {self.rmsd_to_reference}"
            )
        
        if not (0.0 <= self.overlap_percentage <= 100.0):
            raise ValueError(
                f"overlap_percentage must be in range [0.0, 100.0], "
                f"got {self.overlap_percentage}"
            )
        
        if len(self.reference_conformation_hash) != 16:
            raise ValueError(
                f"reference_conformation_hash must be 16 characters, "
                f"got {len(self.reference_conformation_hash)}"
            )
        
        if not (0.0 <= self.golden_ratio_percentage <= 100.0):
            raise ValueError(
                f"golden_ratio_percentage must be in range [0.0, 100.0], "
                f"got {self.golden_ratio_percentage}"
            )
        
        valid_solids = {'tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'}
        if self.dominant_platonic_solid not in valid_solids:
            raise ValueError(
                f"dominant_platonic_solid must be one of {valid_solids}, "
                f"got {self.dominant_platonic_solid}"
            )
        
        if not (0.0 <= self.platonic_similarity_score <= 1.0):
            raise ValueError(
                f"platonic_similarity_score must be in range [0.0, 1.0], "
                f"got {self.platonic_similarity_score}"
            )


@dataclass(frozen=True)
class PatternDetection:
    """
    Immutable pattern detection result with rich metadata.
    
    This dataclass represents a detected pattern in a protein conformation,
    including the pattern type, significance, and type-specific data.
    
    Attributes:
        pattern_type: Type of pattern detected (THz, Folding, Geometric)
        significance: Significance level (LOW, MEDIUM, HIGH)
        timestamp: Unix timestamp when pattern was detected
        iteration: Exploration iteration when detected
        conformation_hash: Hash of conformation where pattern was found (16 chars)
        
        # Type-specific data (only one should be non-None)
        thz_data: THz resonance data (if pattern_type == THZ)
        folding_data: Folding dynamics data (if pattern_type == FOLDING)
        geometric_data: Geometric similarity data (if pattern_type == GEOMETRIC)
    
    Example:
        >>> thz_pattern = PatternDetection(
        ...     pattern_type=PatternType.THZ,
        ...     significance=PatternSignificance.HIGH,
        ...     timestamp=1699564800.0,
        ...     iteration=150,
        ...     conformation_hash="a1b2c3d4e5f6g7h8",
        ...     thz_data=THzResonanceData(...),
        ...     folding_data=None,
        ...     geometric_data=None
        ... )
    """
    pattern_type: PatternType
    significance: PatternSignificance
    timestamp: float
    iteration: int
    conformation_hash: str
    
    # Type-specific data (only one should be non-None)
    thz_data: Optional[THzResonanceData] = None
    folding_data: Optional[FoldingDynamicsData] = None
    geometric_data: Optional[GeometricSimilarityData] = None
    
    def __post_init__(self):
        """Validate field values and consistency."""
        # Validate timestamp (positive)
        if self.timestamp <= 0.0:
            raise ValueError(f"timestamp must be positive, got {self.timestamp}")
        
        # Validate iteration (non-negative)
        if self.iteration < 0:
            raise ValueError(f"iteration must be non-negative, got {self.iteration}")
        
        # Validate conformation_hash (16 characters)
        if len(self.conformation_hash) != 16:
            raise ValueError(
                f"conformation_hash must be 16 characters, "
                f"got {len(self.conformation_hash)}"
            )
        
        # Validate pattern type and data consistency
        data_count = sum([
            self.thz_data is not None,
            self.folding_data is not None,
            self.geometric_data is not None,
        ])
        
        if data_count == 0:
            raise ValueError("At least one pattern data field must be non-None")
        
        if data_count > 1:
            raise ValueError("Only one pattern data field should be non-None")
        
        # Check pattern type matches data
        if self.pattern_type == PatternType.THZ and self.thz_data is None:
            raise ValueError("pattern_type is THZ but thz_data is None")
        
        if self.pattern_type == PatternType.FOLDING and self.folding_data is None:
            raise ValueError("pattern_type is FOLDING but folding_data is None")
        
        if self.pattern_type == PatternType.GEOMETRIC and self.geometric_data is None:
            raise ValueError("pattern_type is GEOMETRIC but geometric_data is None")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'pattern_type': self.pattern_type.value,
            'significance': self.significance.value,
            'timestamp': self.timestamp,
            'iteration': self.iteration,
            'conformation_hash': self.conformation_hash,
        }
        
        # Add type-specific data
        if self.thz_data:
            result['thz_data'] = {
                'cluster_id': self.thz_data.cluster_id,
                'cluster_size': self.thz_data.cluster_size,
                'similarity_score': self.thz_data.similarity_score,
                'dominant_frequency': self.thz_data.dominant_frequency,
                'spectral_entropy': self.thz_data.spectral_entropy,
            }
        elif self.folding_data:
            result['folding_data'] = {
                'helix_percentage': self.folding_data.helix_percentage,
                'sheet_percentage': self.folding_data.sheet_percentage,
                'turn_percentage': self.folding_data.turn_percentage,
                'coil_percentage': self.folding_data.coil_percentage,
                'helix_regions': self.folding_data.helix_regions,
                'sheet_regions': self.folding_data.sheet_regions,
                'turn_regions': self.folding_data.turn_regions,
            }
        elif self.geometric_data:
            result['geometric_data'] = {
                'rmsd_to_reference': self.geometric_data.rmsd_to_reference,
                'overlap_percentage': self.geometric_data.overlap_percentage,
                'reference_conformation_hash': self.geometric_data.reference_conformation_hash,
                'golden_ratio_percentage': self.geometric_data.golden_ratio_percentage,
                'dominant_platonic_solid': self.geometric_data.dominant_platonic_solid,
                'platonic_similarity_score': self.geometric_data.platonic_similarity_score,
            }
        
        return result
