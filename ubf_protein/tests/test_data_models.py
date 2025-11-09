"""
Unit Tests for Geometric Attractor and Pattern Detection Data Models

Tests cover:
- GeometricAnalysisResult validation and immutability
- PatternDetection and all sub-dataclasses
- LRUCache functionality and statistics
- Edge cases and error handling

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import time
from ubf_protein.geometric_attractor import GeometricAnalysisResult, LRUCache
from ubf_protein.pattern_detection import (
    PatternType,
    PatternSignificance,
    THzResonanceData,
    FoldingDynamicsData,
    GeometricSimilarityData,
    PatternDetection,
)


class TestGeometricAnalysisResult:
    """Tests for GeometricAnalysisResult dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid GeometricAnalysisResult."""
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=1699564800.0,
            num_residues=76,
        )
        
        assert result.golden_ratio_percentage == 23.5
        assert result.phi_pattern_count == 42
        assert result.dodecahedron_similarity == 0.7
        assert result.num_residues == 76
    
    def test_immutability(self):
        """Test that dataclass is frozen (immutable)."""
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=1699564800.0,
            num_residues=76,
        )
        
        with pytest.raises(AttributeError):
            result.golden_ratio_percentage = 50.0
    
    def test_invalid_golden_ratio_percentage(self):
        """Test validation of golden_ratio_percentage range."""
        with pytest.raises(ValueError, match="golden_ratio_percentage must be in range"):
            GeometricAnalysisResult(
                golden_ratio_percentage=150.0,  # Invalid: > 100
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_phi_pattern_count(self):
        """Test validation of phi_pattern_count (non-negative)."""
        with pytest.raises(ValueError, match="phi_pattern_count must be non-negative"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=-5,  # Invalid: negative
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_platonic_similarity(self):
        """Test validation of Platonic solid similarity range."""
        with pytest.raises(ValueError, match="dodecahedron_similarity must be in range"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=1.5,  # Invalid: > 1.0
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_symmetry_metric(self):
        """Test validation of symmetry metric range."""
        with pytest.raises(ValueError, match="rotational_symmetry must be in range"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=-0.1,  # Invalid: < 0.0
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_radius_of_gyration(self):
        """Test validation of radius_of_gyration (non-negative)."""
        with pytest.raises(ValueError, match="radius_of_gyration must be non-negative"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=-5.0,  # Invalid: negative
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_asphericity(self):
        """Test validation of asphericity range."""
        with pytest.raises(ValueError, match="asphericity must be in range"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=1.5,  # Invalid: > 1.0
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_conformation_hash(self):
        """Test validation of conformation_hash length."""
        with pytest.raises(ValueError, match="conformation_hash must be 16 characters"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="short",  # Invalid: < 16 chars
                timestamp=1699564800.0,
                num_residues=76,
            )
    
    def test_invalid_timestamp(self):
        """Test validation of timestamp (positive)."""
        with pytest.raises(ValueError, match="timestamp must be positive"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=-1.0,  # Invalid: negative
                num_residues=76,
            )
    
    def test_invalid_num_residues(self):
        """Test validation of num_residues (positive)."""
        with pytest.raises(ValueError, match="num_residues must be positive"):
            GeometricAnalysisResult(
                golden_ratio_percentage=23.5,
                phi_pattern_count=42,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash="a1b2c3d4e5f6g7h8",
                timestamp=1699564800.0,
                num_residues=0,  # Invalid: zero
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=1699564800.0,
            num_residues=76,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['golden_ratio_percentage'] == 23.5
        assert result_dict['phi_pattern_count'] == 42
        assert result_dict['platonic_similarities']['dodecahedron'] == 0.7
        assert result_dict['symmetry_metrics']['rotational'] == 0.6
        assert result_dict['metadata']['num_residues'] == 76


class TestTHzResonanceData:
    """Tests for THzResonanceData dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid THzResonanceData."""
        thz_data = THzResonanceData(
            cluster_id=3,
            cluster_size=12,
            similarity_score=0.85,
            dominant_frequency=2.45,
            spectral_entropy=1.23,
        )
        
        assert thz_data.cluster_id == 3
        assert thz_data.cluster_size == 12
        assert thz_data.similarity_score == 0.85
    
    def test_invalid_cluster_id(self):
        """Test validation of cluster_id (non-negative)."""
        with pytest.raises(ValueError, match="cluster_id must be non-negative"):
            THzResonanceData(
                cluster_id=-1,
                cluster_size=12,
                similarity_score=0.85,
                dominant_frequency=2.45,
                spectral_entropy=1.23,
            )
    
    def test_invalid_cluster_size(self):
        """Test validation of cluster_size (positive)."""
        with pytest.raises(ValueError, match="cluster_size must be positive"):
            THzResonanceData(
                cluster_id=3,
                cluster_size=0,
                similarity_score=0.85,
                dominant_frequency=2.45,
                spectral_entropy=1.23,
            )
    
    def test_invalid_similarity_score(self):
        """Test validation of similarity_score range."""
        with pytest.raises(ValueError, match="similarity_score must be in range"):
            THzResonanceData(
                cluster_id=3,
                cluster_size=12,
                similarity_score=1.5,
                dominant_frequency=2.45,
                spectral_entropy=1.23,
            )


class TestFoldingDynamicsData:
    """Tests for FoldingDynamicsData dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid FoldingDynamicsData."""
        folding_data = FoldingDynamicsData(
            helix_percentage=35.0,
            sheet_percentage=25.0,
            turn_percentage=15.0,
            coil_percentage=25.0,
            helix_regions=[(5, 18), (25, 38)],
            sheet_regions=[(42, 48)],
            turn_regions=[(19, 22)],
        )
        
        assert folding_data.helix_percentage == 35.0
        assert len(folding_data.helix_regions) == 2
    
    def test_invalid_percentage_range(self):
        """Test validation of percentage range."""
        with pytest.raises(ValueError, match="helix_percentage must be in range"):
            FoldingDynamicsData(
                helix_percentage=150.0,
                sheet_percentage=25.0,
                turn_percentage=15.0,
                coil_percentage=25.0,
                helix_regions=[],
                sheet_regions=[],
                turn_regions=[],
            )
    
    def test_invalid_percentage_sum(self):
        """Test validation of percentage sum."""
        with pytest.raises(ValueError, match="Percentages must sum to ~100%"):
            FoldingDynamicsData(
                helix_percentage=30.0,
                sheet_percentage=20.0,
                turn_percentage=10.0,
                coil_percentage=10.0,  # Sum = 70%, invalid
                helix_regions=[],
                sheet_regions=[],
                turn_regions=[],
            )
    
    def test_invalid_region_bounds(self):
        """Test validation of region bounds (start < end)."""
        with pytest.raises(ValueError, match="must have start < end"):
            FoldingDynamicsData(
                helix_percentage=35.0,
                sheet_percentage=25.0,
                turn_percentage=15.0,
                coil_percentage=25.0,
                helix_regions=[(10, 5)],  # Invalid: start > end
                sheet_regions=[],
                turn_regions=[],
            )


class TestGeometricSimilarityData:
    """Tests for GeometricSimilarityData dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid GeometricSimilarityData."""
        geo_data = GeometricSimilarityData(
            rmsd_to_reference=1.85,
            overlap_percentage=78.5,
            reference_conformation_hash="a1b2c3d4e5f6g7h8",
            golden_ratio_percentage=24.3,
            dominant_platonic_solid="icosahedron",
            platonic_similarity_score=0.82,
        )
        
        assert geo_data.rmsd_to_reference == 1.85
        assert geo_data.dominant_platonic_solid == "icosahedron"
    
    def test_invalid_rmsd(self):
        """Test validation of RMSD (non-negative)."""
        with pytest.raises(ValueError, match="rmsd_to_reference must be non-negative"):
            GeometricSimilarityData(
                rmsd_to_reference=-1.0,
                overlap_percentage=78.5,
                reference_conformation_hash="a1b2c3d4e5f6g7h8",
                golden_ratio_percentage=24.3,
                dominant_platonic_solid="icosahedron",
                platonic_similarity_score=0.82,
            )
    
    def test_invalid_dominant_platonic_solid(self):
        """Test validation of dominant_platonic_solid."""
        with pytest.raises(ValueError, match="dominant_platonic_solid must be one of"):
            GeometricSimilarityData(
                rmsd_to_reference=1.85,
                overlap_percentage=78.5,
                reference_conformation_hash="a1b2c3d4e5f6g7h8",
                golden_ratio_percentage=24.3,
                dominant_platonic_solid="sphere",  # Invalid solid
                platonic_similarity_score=0.82,
            )


class TestPatternDetection:
    """Tests for PatternDetection dataclass."""
    
    def test_valid_thz_pattern(self):
        """Test creating valid THz pattern detection."""
        thz_data = THzResonanceData(
            cluster_id=3,
            cluster_size=12,
            similarity_score=0.85,
            dominant_frequency=2.45,
            spectral_entropy=1.23,
        )
        
        pattern = PatternDetection(
            pattern_type=PatternType.THZ,
            significance=PatternSignificance.HIGH,
            timestamp=1699564800.0,
            iteration=150,
            conformation_hash="a1b2c3d4e5f6g7h8",
            thz_data=thz_data,
        )
        
        assert pattern.pattern_type == PatternType.THZ
        assert pattern.thz_data is not None
        assert pattern.folding_data is None
    
    def test_valid_folding_pattern(self):
        """Test creating valid folding pattern detection."""
        folding_data = FoldingDynamicsData(
            helix_percentage=35.0,
            sheet_percentage=25.0,
            turn_percentage=15.0,
            coil_percentage=25.0,
            helix_regions=[(5, 18)],
            sheet_regions=[(42, 48)],
            turn_regions=[(19, 22)],
        )
        
        pattern = PatternDetection(
            pattern_type=PatternType.FOLDING,
            significance=PatternSignificance.MEDIUM,
            timestamp=1699564800.0,
            iteration=150,
            conformation_hash="a1b2c3d4e5f6g7h8",
            folding_data=folding_data,
        )
        
        assert pattern.pattern_type == PatternType.FOLDING
        assert pattern.folding_data is not None
    
    def test_no_data_raises_error(self):
        """Test that missing data raises ValueError."""
        with pytest.raises(ValueError, match="At least one pattern data field must be non-None"):
            PatternDetection(
                pattern_type=PatternType.THZ,
                significance=PatternSignificance.HIGH,
                timestamp=1699564800.0,
                iteration=150,
                conformation_hash="a1b2c3d4e5f6g7h8",
            )
    
    def test_multiple_data_raises_error(self):
        """Test that multiple data fields raises ValueError."""
        thz_data = THzResonanceData(
            cluster_id=3,
            cluster_size=12,
            similarity_score=0.85,
            dominant_frequency=2.45,
            spectral_entropy=1.23,
        )
        folding_data = FoldingDynamicsData(
            helix_percentage=35.0,
            sheet_percentage=25.0,
            turn_percentage=15.0,
            coil_percentage=25.0,
            helix_regions=[],
            sheet_regions=[],
            turn_regions=[],
        )
        
        with pytest.raises(ValueError, match="Only one pattern data field should be non-None"):
            PatternDetection(
                pattern_type=PatternType.THZ,
                significance=PatternSignificance.HIGH,
                timestamp=1699564800.0,
                iteration=150,
                conformation_hash="a1b2c3d4e5f6g7h8",
                thz_data=thz_data,
                folding_data=folding_data,
            )
    
    def test_mismatched_type_and_data(self):
        """Test that mismatched pattern type and data raises ValueError."""
        folding_data = FoldingDynamicsData(
            helix_percentage=35.0,
            sheet_percentage=25.0,
            turn_percentage=15.0,
            coil_percentage=25.0,
            helix_regions=[],
            sheet_regions=[],
            turn_regions=[],
        )
        
        with pytest.raises(ValueError, match="pattern_type is THZ but thz_data is None"):
            PatternDetection(
                pattern_type=PatternType.THZ,
                significance=PatternSignificance.HIGH,
                timestamp=1699564800.0,
                iteration=150,
                conformation_hash="a1b2c3d4e5f6g7h8",
                folding_data=folding_data,
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        thz_data = THzResonanceData(
            cluster_id=3,
            cluster_size=12,
            similarity_score=0.85,
            dominant_frequency=2.45,
            spectral_entropy=1.23,
        )
        
        pattern = PatternDetection(
            pattern_type=PatternType.THZ,
            significance=PatternSignificance.HIGH,
            timestamp=1699564800.0,
            iteration=150,
            conformation_hash="a1b2c3d4e5f6g7h8",
            thz_data=thz_data,
        )
        
        pattern_dict = pattern.to_dict()
        
        assert pattern_dict['pattern_type'] == 'thz_resonance'
        assert pattern_dict['significance'] == 'high'
        assert pattern_dict['thz_data']['cluster_id'] == 3


class TestLRUCache:
    """Tests for LRUCache class."""
    
    def test_cache_initialization(self):
        """Test LRU cache initialization."""
        cache = LRUCache(max_size=100, ttl_seconds=600)
        
        assert cache.max_size == 100
        assert cache.ttl_seconds == 600
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hit_rate'] == 0.0
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=10, ttl_seconds=3600)
        
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=time.time(),
            num_residues=76,
        )
        
        cache.put("test_key", result)
        cached_result = cache.get("test_key")
        
        assert cached_result is not None
        assert cached_result.golden_ratio_percentage == 23.5
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=10, ttl_seconds=3600)
        
        cached_result = cache.get("nonexistent_key")
        
        assert cached_result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when size limit reached."""
        cache = LRUCache(max_size=3, ttl_seconds=3600)
        
        # Add 4 entries (should evict oldest)
        for i in range(4):
            result = GeometricAnalysisResult(
                golden_ratio_percentage=float(i),
                phi_pattern_count=i,
                tetrahedron_similarity=0.3,
                cube_similarity=0.2,
                octahedron_similarity=0.4,
                dodecahedron_similarity=0.7,
                icosahedron_similarity=0.8,
                rotational_symmetry=0.6,
                local_symmetry=0.5,
                radius_of_gyration=15.2,
                asphericity=0.3,
                conformation_hash=f"{i:016d}",
                timestamp=time.time(),
                num_residues=76,
            )
            cache.put(f"key_{i}", result)
        
        # First entry should be evicted
        assert cache.get("key_0") is None
        # Later entries should still be present
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None
        assert cache.get("key_3") is not None
        
        stats = cache.get_stats()
        assert stats['evictions'] == 1
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration removes stale entries."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=time.time(),
            num_residues=76,
        )
        
        cache.put("test_key", result)
        
        # Should be present immediately
        assert cache.get("test_key") is not None
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("test_key") is None
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = LRUCache(max_size=10, ttl_seconds=3600)
        
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=time.time(),
            num_residues=76,
        )
        
        cache.put("test_key", result)
        
        # Hit
        cache.get("test_key")
        # Miss
        cache.get("nonexistent")
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['total_requests'] == 2
        assert stats['hit_rate'] == 0.5
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=10, ttl_seconds=3600)
        
        result = GeometricAnalysisResult(
            golden_ratio_percentage=23.5,
            phi_pattern_count=42,
            tetrahedron_similarity=0.3,
            cube_similarity=0.2,
            octahedron_similarity=0.4,
            dodecahedron_similarity=0.7,
            icosahedron_similarity=0.8,
            rotational_symmetry=0.6,
            local_symmetry=0.5,
            radius_of_gyration=15.2,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=time.time(),
            num_residues=76,
        )
        
        cache.put("test_key", result)
        assert cache.get_stats()['size'] == 1
        
        cache.clear()
        assert cache.get_stats()['size'] == 0
        assert cache.get("test_key") is None
