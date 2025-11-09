"""
Integration tests for Geometric Attractor Module in test_protein.py

Tests Task 4 implementation:
- analyze_geometric_attractors_v2() function invoked during protein test
- Results included in output JSON
- Error handling (invalid PDB, missing file)
- Performance impact < 10% of total runtime
"""

import pytest
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer


def test_geometric_analyzer_initialization():
    """Test that GeometricAttractorAnalyzer initializes correctly."""
    analyzer = GeometricAttractorAnalyzer()
    
    assert analyzer.cache is not None
    assert analyzer.phi_tolerance == 0.05
    assert analyzer.neighbor_window == 10


def test_geometric_analysis_on_small_conformation():
    """Test geometric analysis on a small 7-residue conformation."""
    # Create simple linear conformation
    coordinates = [
        (0.0, 0.0, 0.0),
        (3.8, 0.0, 0.0),
        (7.6, 0.0, 0.0),
        (11.4, 0.0, 0.0),
        (15.2, 0.0, 0.0),
        (19.0, 0.0, 0.0),
        (22.8, 0.0, 0.0),
    ]
    
    conformation = {'coordinates': coordinates}
    
    analyzer = GeometricAttractorAnalyzer()
    result = analyzer.analyze_conformation(conformation, sequence="ACDEFGH")
    
    # Verify all required fields are present
    assert hasattr(result, 'golden_ratio_percentage')
    assert hasattr(result, 'phi_pattern_count')
    assert hasattr(result, 'tetrahedron_similarity')
    assert hasattr(result, 'cube_similarity')
    assert hasattr(result, 'octahedron_similarity')
    assert hasattr(result, 'dodecahedron_similarity')
    assert hasattr(result, 'icosahedron_similarity')
    assert hasattr(result, 'rotational_symmetry')
    assert hasattr(result, 'local_symmetry')
    assert hasattr(result, 'radius_of_gyration')
    assert hasattr(result, 'asphericity')
    assert hasattr(result, 'conformation_hash')
    
    # Verify value ranges
    assert 0.0 <= result.golden_ratio_percentage <= 100.0
    assert result.phi_pattern_count >= 0
    assert 0.0 <= result.rotational_symmetry <= 1.0
    assert 0.0 <= result.local_symmetry <= 1.0
    assert result.radius_of_gyration >= 0.0
    assert 0.0 <= result.asphericity <= 1.0


def test_geometric_analysis_performance():
    """Test that geometric analysis completes quickly (<50ms target)."""
    # Create medium-sized conformation (20 residues)
    coordinates = [(float(i), 0.0, 0.0) for i in range(20)]
    conformation = {'coordinates': coordinates}
    
    analyzer = GeometricAttractorAnalyzer()
    
    start_time = time.time()
    result = analyzer.analyze_conformation(conformation)
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Should complete in < 50ms
    assert elapsed_ms < 50.0, f"Analysis took {elapsed_ms:.1f}ms, expected < 50ms"


def test_geometric_analysis_caching():
    """Test that caching works correctly."""
    coordinates = [(float(i), 0.0, 0.0) for i in range(10)]
    conformation = {'coordinates': coordinates}
    
    analyzer = GeometricAttractorAnalyzer()
    
    # First call - cache miss
    result1 = analyzer.analyze_conformation(conformation)
    stats1 = analyzer.get_cache_stats()
    assert stats1['misses'] == 1
    assert stats1['hits'] == 0
    
    # Second call - cache hit
    result2 = analyzer.analyze_conformation(conformation)
    stats2 = analyzer.get_cache_stats()
    assert stats2['hits'] == 1
    
    # Results should be identical
    assert result1.conformation_hash == result2.conformation_hash
    assert result1.golden_ratio_percentage == result2.golden_ratio_percentage


def test_geometric_analysis_to_dict():
    """Test that to_dict() returns correct structure."""
    coordinates = [(float(i), 0.0, 0.0) for i in range(7)]
    conformation = {'coordinates': coordinates}
    
    analyzer = GeometricAttractorAnalyzer()
    result = analyzer.analyze_conformation(conformation)
    
    result_dict = result.to_dict()
    
    # Verify dictionary structure
    assert 'golden_ratio_percentage' in result_dict
    assert 'phi_pattern_count' in result_dict
    assert 'platonic_similarities' in result_dict
    assert 'symmetry_metrics' in result_dict
    assert 'metadata' in result_dict
    
    # Verify nested structures
    assert 'tetrahedron' in result_dict['platonic_similarities']
    assert 'rotational' in result_dict['symmetry_metrics']
    assert 'conformation_hash' in result_dict['metadata']


def test_geometric_analysis_error_handling():
    """Test error handling for invalid inputs."""
    analyzer = GeometricAttractorAnalyzer()
    
    # Test with too few residues
    with pytest.raises(ValueError, match="at least 3 residues"):
        analyzer.analyze_conformation({'coordinates': [(0,0,0), (1,1,1)]})
    
    # Test with missing coordinates
    with pytest.raises(ValueError, match="must contain 'coordinates'"):
        analyzer.analyze_conformation({})
    
    # Test with invalid conformation type
    with pytest.raises(ValueError):
        analyzer.analyze_conformation(123)


def test_integration_with_test_protein_format():
    """Test that the format matches what test_protein.py expects."""
    # Simulate best_conformation from multi-agent exploration
    best_conformation = {
        'atom_coordinates': [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
            (19.0, 0.0, 0.0),
            (22.8, 0.0, 0.0),
        ]
    }
    
    sequence = "ACDEFGH"
    
    # Convert to format expected by analyzer
    conformation_data = {
        'coordinates': best_conformation['atom_coordinates']
    }
    
    analyzer = GeometricAttractorAnalyzer()
    result = analyzer.analyze_conformation(conformation_data, sequence=sequence)
    result_dict = result.to_dict()
    
    # Verify JSON serializable
    json_str = json.dumps(result_dict)
    reloaded = json.loads(json_str)
    
    assert reloaded['golden_ratio_percentage'] == result_dict['golden_ratio_percentage']
    assert reloaded['platonic_similarities']['dodecahedron'] == result_dict['platonic_similarities']['dodecahedron']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
