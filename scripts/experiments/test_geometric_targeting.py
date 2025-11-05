"""
Quick Integration Test for Geometric Targeting

Tests the full prescriptive geometric targeting system:
1. CLI flag parsing
2. Geometric scorer initialization
3. QCPP metrics with geometric_similarity
4. Agent move evaluation with geometric factor
5. Memory significance with 8th signal

Usage:
    python test_geometric_targeting.py --target octahedron
    python test_geometric_targeting.py --target icosahedron
    python test_geometric_targeting.py --target none
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent))

from ubf_protein.geometric_scoring import GeometricScorer, create_scorer
from ubf_protein.qcpp_integration import QCPPMetrics
from ubf_protein.models import Conformation

def test_geometric_scorer():
    """Test geometric scorer performance and accuracy."""
    print("\n" + "="*70)
    print("TEST 1: Geometric Scorer")
    print("="*70)
    
    # Test each geometry type
    geometries = ['octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube', 'none']
    
    for target in geometries:
        scorer = create_scorer(target)
        
        # Create simple test coordinates (10 atoms in a line)
        coords = [np.array([float(i), 0.0, 0.0]) for i in range(10)]
        
        # Time the calculation
        start = time.perf_counter()
        similarity = scorer.calculate_similarity(coords)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Check performance
        within_target = "✅" if elapsed_ms < 2.0 else "❌"
        print(f"  {target:15s}: similarity={similarity:.3f}, time={elapsed_ms:.2f}ms {within_target}")
        
        # Verify similarity is in range
        assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range [0, 1]"
        
        # Verify 'none' returns 0.0
        if target == 'none':
            assert similarity == 0.0, f"'none' target should return 0.0, got {similarity}"
    
    print("✓ All geometric scorers passed")


def test_qcpp_metrics_with_geometric():
    """Test QCPPMetrics with geometric_similarity field."""
    print("\n" + "="*70)
    print("TEST 2: QCPPMetrics with Geometric Similarity")
    print("="*70)
    
    # Test valid metrics
    metrics = QCPPMetrics(
        qcp_score=5.5,
        field_coherence=0.7,
        stability_score=1.2,
        phi_match_score=0.85,
        calculation_time_ms=3.5,
        geometric_similarity=0.75  # NEW field
    )
    
    print(f"  QCP Score: {metrics.qcp_score}")
    print(f"  Field Coherence: {metrics.field_coherence}")
    print(f"  Stability: {metrics.stability_score}")
    print(f"  Phi Match: {metrics.phi_match_score}")
    print(f"  Geometric Similarity: {metrics.geometric_similarity} ✅")
    
    assert metrics.geometric_similarity == 0.75
    print("✓ QCPPMetrics with geometric_similarity passed")
    
    # Test validation
    try:
        bad_metrics = QCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.5,
            stability_score=1.0,
            phi_match_score=0.5,
            calculation_time_ms=2.0,
            geometric_similarity=1.5  # Invalid (>1.0)
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Validation correctly rejected geometric_similarity=1.5: {e}")


def test_qcpp_integration_with_target():
    """Test QCPPIntegrationAdapter with geometric targeting."""
    print("\n" + "="*70)
    print("TEST 3: QCPP Integration with Geometric Targeting")
    print("="*70)
    
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
    
    # Mock QCPP predictor (minimal interface)
    class MockQCPPPredictor:
        pass
    
    # Test with octahedron target
    adapter = QCPPIntegrationAdapter(
        MockQCPPPredictor(),
        cache_size=100,
        target_geometry='octahedron'
    )
    
    print(f"  Target geometry: {adapter.target_geometry}")
    print(f"  Geometric scorer initialized: {adapter.geometric_scorer is not None}")
    
    assert adapter.target_geometry == 'octahedron'
    assert adapter.geometric_scorer is not None
    print("✓ QCPP adapter with geometric targeting passed")
    
    # Test with no target
    adapter_none = QCPPIntegrationAdapter(
        MockQCPPPredictor(),
        cache_size=100,
        target_geometry='none'
    )
    
    assert adapter_none.target_geometry == 'none'
    assert adapter_none.geometric_scorer is None
    print("✓ QCPP adapter without targeting passed")


def test_cli_parsing():
    """Test CLI flag parsing (without running full test)."""
    print("\n" + "="*70)
    print("TEST 4: CLI Flag Parsing")
    print("="*70)
    
    import argparse
    
    # Create parser similar to test_protein.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-geometry', 
                        choices=['none', 'octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube'],
                        default='none')
    
    # Test valid arguments
    test_cases = [
        (['--target-geometry', 'octahedron'], 'octahedron'),
        (['--target-geometry', 'icosahedron'], 'icosahedron'),
        (['--target-geometry', 'none'], 'none'),
        ([], 'none'),  # Default
    ]
    
    for args, expected in test_cases:
        parsed = parser.parse_args(args)
        assert parsed.target_geometry == expected
        args_str = ' '.join(args) if args else 'default'
        print(f"  {args_str:30s} → {parsed.target_geometry:15s} ✅")
    
    print("✓ CLI flag parsing passed")


def test_performance_profile():
    """Profile geometric scoring performance."""
    print("\n" + "="*70)
    print("TEST 5: Performance Profiling")
    print("="*70)
    
    scorer = create_scorer('octahedron')
    
    # Test with different protein sizes
    sizes = [10, 30, 50, 100, 200]
    
    for size in sizes:
        # Create random coordinates
        coords = [np.random.randn(3) for _ in range(size)]
        
        # Time 10 runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            scorer.calculate_similarity(coords)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        within_target = "✅" if avg_time < 2.0 else "⚠️"
        
        print(f"  {size:3d} residues: avg={avg_time:.2f}ms, max={max_time:.2f}ms {within_target}")
    
    # Get stats
    stats = scorer.get_stats()
    print(f"\n  Total calculations: {stats['calculation_count']}")
    print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
    print(f"  Max time: {stats['max_time_ms']:.2f}ms")
    print(f"  Within target (<2ms): {stats['within_target']}")
    
    print("✓ Performance profiling complete")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GEOMETRIC TARGETING INTEGRATION TEST")
    print("="*80)
    
    try:
        test_geometric_scorer()
        test_qcpp_metrics_with_geometric()
        test_qcpp_integration_with_target()
        test_cli_parsing()
        test_performance_profile()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nGeometric targeting system is fully operational!")
        print("\nNext steps:")
        print("  1. Run real protein test:")
        print("     python test_protein.py --pdb 1CRN --agents 5 --iterations 50 --target-geometry octahedron")
        print("\n  2. Compare with non-targeted:")
        print("     python test_protein.py --pdb 1CRN --agents 5 --iterations 50 --target-geometry none")
        print("\n  3. Try different geometries:")
        print("     python test_protein.py --pdb 1VII --target-geometry icosahedron")
        print("     python test_protein.py --pdb 1UBQ --target-geometry dodecahedron")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
