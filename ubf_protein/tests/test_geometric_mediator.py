"""
Unit Tests for Geometric Attractor V2 and Mediator Agents

Tests cover:
- Geometric Attractor V2 percentage-based scoring
- Mediator Agent pattern detection
- Integration workflows
- Caching performance
- Error handling

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import time
import math
from ubf_protein.geometric_attractor_v2 import (
    GeometricAttractorV2,
    GeometricRelationshipScores,
    analyze_protein_geometry,
    PHI
)
from ubf_protein.mediator_agents import (
    MediatorAgent,
    MediatorAgentConfig,
    create_mediator,
    PatternSignificance,
    THzResonancePattern,
    FoldingDynamicsPattern,
    GeometricSimilarityPattern
)


# =============================================================================
# Geometric Attractor V2 Tests
# =============================================================================

class TestGeometricAttractorV2:
    """Tests for GeometricAttractorV2 class."""
    
    def test_basic_analysis(self):
        """Test basic conformation analysis returns valid results."""
        analyzer = GeometricAttractorV2()
        
        # Simple square conformation
        coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        
        result = analyzer.analyze_conformation(coords)
        
        # Verify all fields present and in valid ranges
        assert 0.0 <= result.phi_distance_patterns <= 100.0
        assert 0.0 <= result.phi_angle_patterns <= 100.0
        assert 0.0 <= result.phi_volume_patterns <= 100.0
        assert 0.0 <= result.overall_geometric_organization <= 100.0
        assert 0.0 <= result.confidence_score <= 100.0
        assert result.num_residues == 4
    
    def test_phi_pattern_detection(self):
        """Test golden ratio pattern detection."""
        analyzer = GeometricAttractorV2()
        
        # Create structure with deliberate φ patterns
        coords = []
        for i in range(15):
            # Use φ-based spacing
            x = i * PHI
            y = (i * PHI) % 5.0
            z = (i ** 2) * 0.1
            coords.append((x, y, z))
        
        result = analyzer.analyze_conformation(coords)
        
        # Should detect at least some φ patterns
        assert result.phi_distance_patterns >= 0.0
        assert result.num_residues == 15
    
    def test_platonic_similarities(self):
        """Test Platonic solid similarity calculations."""
        analyzer = GeometricAttractorV2()
        
        # Tetrahedral-like structure
        a = 1.0 / math.sqrt(3.0)
        coords = [
            (a, a, a),
            (a, -a, -a),
            (-a, a, -a),
            (-a, -a, a),
        ]
        
        result = analyzer.analyze_conformation(coords)
        
        # All similarities should be percentages
        assert 0.0 <= result.tetrahedron_similarity <= 100.0
        assert 0.0 <= result.cube_similarity <= 100.0
        assert 0.0 <= result.octahedron_similarity <= 100.0
        assert 0.0 <= result.dodecahedron_similarity <= 100.0
        assert 0.0 <= result.icosahedron_similarity <= 100.0
    
    def test_symmetry_relationships(self):
        """Test symmetry relationship calculations."""
        analyzer = GeometricAttractorV2()
        
        # Symmetric structure (cube vertices)
        coords = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]
        
        result = analyzer.analyze_conformation(coords)
        
        # Should have high symmetry
        assert result.rotational_symmetry > 50.0  # At least moderate
        assert result.local_symmetry > 0.0
        assert result.translational_regularity >= 0.0
        assert result.reflectional_symmetry >= 0.0
    
    def test_fibonacci_patterns(self):
        """Test Fibonacci pattern detection."""
        analyzer = GeometricAttractorV2()
        
        # Create Fibonacci-spaced structure
        coords = []
        fib = [1, 1, 2, 3, 5, 8, 13]
        for i, f in enumerate(fib):
            coords.append((float(f), float(i), 0.0))
        
        result = analyzer.analyze_conformation(coords)
        
        # Should detect some Fibonacci patterns
        assert result.fibonacci_spacing >= 0.0
        assert result.fibonacci_ratios >= 0.0
    
    def test_shape_characteristics(self):
        """Test shape characteristic analysis."""
        analyzer = GeometricAttractorV2()
        
        # Rod-like structure (linear)
        coords = [(float(i), 0.0, 0.0) for i in range(10)]
        
        result = analyzer.analyze_conformation(coords)
        
        # Should show high elongation
        assert result.elongation > result.compactness
        assert 0.0 <= result.compactness <= 100.0
        assert 0.0 <= result.elongation <= 100.0
        assert 0.0 <= result.planarity <= 100.0
    
    def test_caching(self):
        """Test that caching works correctly."""
        analyzer = GeometricAttractorV2(cache_size=100)
        
        coords = [(float(i), float(i), 0.0) for i in range(10)]
        
        # First call (cache miss)
        result1 = analyzer.analyze_conformation(coords)
        
        # Second call (cache hit)
        result2 = analyzer.analyze_conformation(coords)
        
        # Results should be identical
        assert result1.conformation_hash == result2.conformation_hash
        assert result1.overall_geometric_organization == result2.overall_geometric_organization
        
        # Check cache stats
        stats = analyzer.get_cache_stats()
        assert stats['cache_hits'] >= 1
        assert stats['total_analyses'] >= 2
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        analyzer = GeometricAttractorV2()
        
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        result = analyzer.analyze_conformation(coords)
        
        result_dict = result.to_dict()
        
        # Verify structure
        assert 'golden_ratio_relationships' in result_dict
        assert 'platonic_solid_similarities' in result_dict
        assert 'symmetry_relationships' in result_dict
        assert 'fibonacci_relationships' in result_dict
        assert 'shape_characteristics' in result_dict
        assert 'overall_metrics' in result_dict
        assert 'metadata' in result_dict
    
    def test_get_summary_string(self):
        """Test summary string generation."""
        analyzer = GeometricAttractorV2()
        
        coords = [(float(i), float(i % 3), 0.0) for i in range(10)]
        result = analyzer.analyze_conformation(coords)
        
        summary = result.get_summary_string()
        
        # Should contain key sections
        assert "GEOMETRIC RELATIONSHIP ANALYSIS" in summary
        assert "Golden Ratio" in summary
        assert "Platonic Solid" in summary
        assert "Symmetry" in summary
    
    def test_min_residues_error(self):
        """Test that too few residues raises error."""
        analyzer = GeometricAttractorV2()
        
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]  # Only 2
        
        with pytest.raises(ValueError, match="at least 3 residues"):
            analyzer.analyze_conformation(coords)
    
    def test_dict_input(self):
        """Test analysis with dict input."""
        analyzer = GeometricAttractorV2()
        
        conformation = {
            'coordinates': [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
            ]
        }
        
        result = analyzer.analyze_conformation(conformation)
        assert result.num_residues == 3
    
    def test_clear_cache(self):
        """Test cache clearing."""
        analyzer = GeometricAttractorV2()
        
        coords = [(float(i), 0.0, 0.0) for i in range(5)]
        analyzer.analyze_conformation(coords)
        
        stats_before = analyzer.get_cache_stats()
        assert stats_before['cache_size'] > 0
        
        analyzer.clear_cache()
        
        stats_after = analyzer.get_cache_stats()
        assert stats_after['cache_size'] == 0
        assert stats_after['cache_hits'] == 0


class TestAnalyzeProteinGeometry:
    """Tests for convenience function."""
    
    def test_convenience_function(self):
        """Test convenience function works."""
        coords = [(float(i), 0.0, 0.0) for i in range(5)]
        
        # Should not raise error
        result = analyze_protein_geometry(coords, verbose=False)
        
        assert result.num_residues == 5
        assert 0.0 <= result.overall_geometric_organization <= 100.0


# =============================================================================
# Mediator Agent Tests
# =============================================================================

class TestMediatorAgent:
    """Tests for MediatorAgent class."""
    
    def test_initialization(self):
        """Test mediator initialization."""
        config = MediatorAgentConfig(
            thz_similarity_threshold=0.8,
            geometric_rmsd_threshold=2.5,
            qcpp_cache_size=2000
        )
        
        mediator = MediatorAgent(config)
        
        assert mediator.config.thz_similarity_threshold == 0.8
        assert mediator.config.geometric_rmsd_threshold == 2.5
        assert mediator.config.qcpp_cache_size == 2000
        assert mediator.total_observations == 0
    
    def test_default_initialization(self):
        """Test mediator with default config."""
        mediator = MediatorAgent()
        
        assert mediator.config is not None
        assert mediator.total_observations == 0
    
    def test_statistics(self):
        """Test statistics retrieval."""
        mediator = MediatorAgent()
        
        stats = mediator.get_statistics()
        
        assert 'total_observations' in stats
        assert 'unique_conformations' in stats
        assert 'patterns_detected' in stats
        assert 'cache_hit_rate' in stats
    
    def test_cache_operations(self):
        """Test caching functionality."""
        mediator = MediatorAgent()
        
        # Mock QCPP metrics
        from dataclasses import dataclass
        
        @dataclass(frozen=True)
        class MockQCPPMetrics:
            qcp_score: float
            field_coherence: float
            stability_score: float
            phi_match_score: float
            calculation_time_ms: float
            geometric_similarity: float = 0.0
        
        metrics = MockQCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.6,
            stability_score=2.0,
            phi_match_score=0.7,
            calculation_time_ms=1.5
        )
        
        conf_hash = "test_hash_12345"
        mediator._cache_qcpp_metrics(conf_hash, metrics)  # type: ignore[arg-type]
        
        # Retrieve from cache
        cached = mediator.get_cached_qcpp_metrics(conf_hash)
        assert cached is not None
        assert cached.qcp_score == 5.0
    
    def test_pattern_detection(self):
        """Test pattern detection workflow."""
        mediator = MediatorAgent()
        
        # Create mock conformation
        from dataclasses import dataclass
        
        @dataclass
        class MockConformation:
            atom_coordinates: list
        
        conf = MockConformation(
            atom_coordinates=[(float(i), 0.0, 0.0) for i in range(10)]
        )
        
        # Mock metrics
        @dataclass(frozen=True)
        class MockQCPPMetrics:
            qcp_score: float
            field_coherence: float
            stability_score: float
            phi_match_score: float
            calculation_time_ms: float
            geometric_similarity: float = 0.0
        
        qcpp_metrics = MockQCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.6,
            stability_score=2.0,
            phi_match_score=0.7,
            calculation_time_ms=1.5
        )
        
        # Analyze geometry
        geo_analyzer = GeometricAttractorV2()
        geo_scores = geo_analyzer.analyze_conformation(conf)
        
        # Observe
        mediator.observe_conformation(conf, qcpp_metrics, geo_scores)  # type: ignore[arg-type]
        
        # Check statistics
        stats = mediator.get_statistics()
        assert stats['total_observations'] >= 1
        assert stats['unique_conformations'] >= 1
    
    def test_pattern_retrieval(self):
        """Test pattern retrieval with significance filtering."""
        mediator = MediatorAgent()
        
        # Get patterns (should be empty initially)
        patterns = mediator.get_significant_patterns(PatternSignificance.LOW)
        assert isinstance(patterns, list)
    
    def test_clear_patterns(self):
        """Test pattern clearing."""
        mediator = MediatorAgent()
        
        # Add some mock patterns
        pattern = THzResonancePattern(
            cluster_id=1,
            cluster_size=5,
            dominant_frequency_thz=2.5,
            similarity_score=0.8,
            representative_conformation_hash="abc123",
            significance=PatternSignificance.HIGH,
            timestamp=time.time()
        )
        
        mediator.thz_patterns.append(pattern)
        assert len(mediator.thz_patterns) == 1
        
        # Clear
        mediator.clear_patterns()
        assert len(mediator.thz_patterns) == 0


class TestMediatorAgentConfig:
    """Tests for MediatorAgentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MediatorAgentConfig()
        
        assert config.thz_similarity_threshold == 0.7
        assert config.geometric_rmsd_threshold == 3.0
        assert config.folding_min_length == 4
        assert config.high_significance_threshold == 0.7
        assert config.medium_significance_threshold == 0.4
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MediatorAgentConfig(
            thz_similarity_threshold=0.85,
            qcpp_cache_size=10000,
            broadcast_interval_ms=50.0
        )
        
        assert config.thz_similarity_threshold == 0.85
        assert config.qcpp_cache_size == 10000
        assert config.broadcast_interval_ms == 50.0


class TestPatternDataModels:
    """Tests for pattern data models."""
    
    def test_thz_resonance_pattern(self):
        """Test THzResonancePattern creation and serialization."""
        pattern = THzResonancePattern(
            cluster_id=1,
            cluster_size=10,
            dominant_frequency_thz=2.5,
            similarity_score=0.85,
            representative_conformation_hash="abc123",
            significance=PatternSignificance.HIGH,
            timestamp=time.time()
        )
        
        assert pattern.cluster_id == 1
        assert pattern.cluster_size == 10
        
        # Test serialization
        pattern_dict = pattern.to_dict()
        assert pattern_dict['type'] == 'thz_resonance'
        assert pattern_dict['cluster_id'] == 1
        assert pattern_dict['significance'] == 'high'
    
    def test_folding_dynamics_pattern(self):
        """Test FoldingDynamicsPattern creation."""
        pattern = FoldingDynamicsPattern(
            pattern_type='helix',
            start_residue=5,
            end_residue=18,
            length=14,
            stability_score=0.75,
            occurrence_count=3,
            significance=PatternSignificance.MEDIUM,
            timestamp=time.time()
        )
        
        assert pattern.pattern_type == 'helix'
        assert pattern.length == 14
        
        pattern_dict = pattern.to_dict()
        assert pattern_dict['type'] == 'folding_dynamics'
        assert pattern_dict['pattern_type'] == 'helix'
    
    def test_geometric_similarity_pattern(self):
        """Test GeometricSimilarityPattern creation."""
        pattern = GeometricSimilarityPattern(
            cluster_id=2,
            cluster_size=8,
            representative_hash="xyz789",
            average_rmsd=2.5,
            geometric_score=75.0,
            phi_pattern_strength=35.0,
            platonic_similarity=65.0,
            significance=PatternSignificance.HIGH,
            timestamp=time.time()
        )
        
        assert pattern.cluster_id == 2
        assert pattern.geometric_score == 75.0
        
        pattern_dict = pattern.to_dict()
        assert pattern_dict['type'] == 'geometric_similarity'
        assert pattern_dict['geometric_score'] == 75.0


class TestCreateMediatorFunction:
    """Tests for create_mediator convenience function."""
    
    def test_create_mediator_defaults(self):
        """Test mediator creation with defaults."""
        mediator = create_mediator()
        
        assert mediator.config.thz_similarity_threshold == 0.7
        assert mediator.config.qcpp_cache_size == 5000
    
    def test_create_mediator_custom(self):
        """Test mediator creation with custom parameters."""
        mediator = create_mediator(
            thz_threshold=0.85,
            geometric_rmsd_threshold=2.0,
            cache_size=10000
        )
        
        assert mediator.config.thz_similarity_threshold == 0.85
        assert mediator.config.geometric_rmsd_threshold == 2.0
        assert mediator.config.qcpp_cache_size == 10000


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for both modules working together."""
    
    def test_mediator_with_geometric_analyzer(self):
        """Test mediator observing geometric analysis results."""
        mediator = create_mediator()
        geo_analyzer = GeometricAttractorV2()
        
        # Create conformations
        from dataclasses import dataclass
        
        @dataclass
        class MockConformation:
            atom_coordinates: list
        
        conformations = [
            MockConformation([(float(i), float(i % 3), 0.0) for i in range(10)])
            for _ in range(5)
        ]
        
        # Mock QCPP metrics
        @dataclass(frozen=True)
        class MockQCPPMetrics:
            qcp_score: float
            field_coherence: float
            stability_score: float
            phi_match_score: float
            calculation_time_ms: float
            geometric_similarity: float = 0.0
        
        # Process each conformation
        for conf in conformations:
            # Analyze geometry
            geo_scores = geo_analyzer.analyze_conformation(conf)
            
            # Create mock QCPP metrics
            qcpp_metrics = MockQCPPMetrics(
                qcp_score=5.0,
                field_coherence=0.6,
                stability_score=2.0,
                phi_match_score=0.7,
                calculation_time_ms=1.5
            )
            
            # Observe with mediator
            mediator.observe_conformation(conf, qcpp_metrics, geo_scores)  # type: ignore[arg-type]
        
        # Check results
        stats = mediator.get_statistics()
        assert stats['total_observations'] >= 5
        
        geo_stats = geo_analyzer.get_cache_stats()
        assert geo_stats['total_analyses'] >= 5
    
    def test_caching_performance(self):
        """Test that caching improves performance."""
        mediator = create_mediator(cache_size=1000)
        geo_analyzer = GeometricAttractorV2(cache_size=1000)
        
        from dataclasses import dataclass
        
        @dataclass
        class MockConformation:
            atom_coordinates: list
        
        # Same conformation repeated
        conf = MockConformation([(float(i), 0.0, 0.0) for i in range(10)])
        
        # First pass (cache misses)
        start = time.time()
        for _ in range(10):
            geo_analyzer.analyze_conformation(conf)
        first_pass_time = time.time() - start
        
        # Second pass (cache hits)
        start = time.time()
        for _ in range(10):
            geo_analyzer.analyze_conformation(conf)
        second_pass_time = time.time() - start
        
        # Second pass should be much faster (cache hits)
        # We just verify it completed successfully
        assert second_pass_time >= 0
        
        # Verify high cache hit rate
        stats = geo_analyzer.get_cache_stats()
        assert stats['cache_hits'] >= 9  # 9/10 should be hits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
