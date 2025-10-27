#!/usr/bin/env python3
"""
Test script for Task 11: Integration Example Script

This script verifies that the integrated_exploration.py example works correctly
and demonstrates the key features of QCPP-UBF integration.
"""

import sys
from pathlib import Path

# Add ubf_protein to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from ubf_protein.examples.integrated_exploration import (
            MockQCPPPredictor,
            get_config_by_name,
            print_config_summary,
            run_integrated_exploration
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_management():
    """Test configuration management functions."""
    print("\nTesting configuration management...")
    
    try:
        from ubf_protein.examples.integrated_exploration import get_config_by_name
        from ubf_protein.qcpp_config import QCPPIntegrationConfig
        
        # Test getting predefined configs
        configs = ['default', 'high_performance', 'high_accuracy']
        for config_name in configs:
            config = get_config_by_name(config_name)
            assert isinstance(config, QCPPIntegrationConfig)
            assert config.enabled is True
            print(f"  ✓ {config_name} config loaded successfully")
        
        # Test invalid config name
        try:
            get_config_by_name('invalid')
            print("  ✗ Should have raised ValueError for invalid config")
            return False
        except ValueError:
            print("  ✓ Invalid config name handled correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_mock_qcpp_predictor():
    """Test the mock QCPP predictor."""
    print("\nTesting mock QCPP predictor...")
    
    try:
        from ubf_protein.examples.integrated_exploration import MockQCPPPredictor
        
        predictor = MockQCPPPredictor()
        assert predictor.analysis_count == 0
        
        # Test methods
        qcp = predictor.calculate_qcp(None)
        assert isinstance(qcp, float)
        assert predictor.analysis_count == 1
        
        coherence = predictor.calculate_coherence(None)
        assert isinstance(coherence, float)
        
        stability = predictor.predict_stability(None)
        assert isinstance(stability, float)
        
        print("  ✓ Mock predictor works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Mock predictor test failed: {e}")
        return False


def test_integration_adapter():
    """Test QCPP integration adapter initialization."""
    print("\nTesting QCPP integration adapter...")
    
    try:
        from ubf_protein.examples.integrated_exploration import MockQCPPPredictor
        from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
        
        # Create mock predictor
        predictor = MockQCPPPredictor()
        
        # Create adapter
        adapter = QCPPIntegrationAdapter(predictor=predictor, cache_size=100)
        assert adapter.cache_size == 100
        assert adapter.analysis_count == 0
        
        # Test cache stats
        stats = adapter.get_cache_stats()
        assert 'total_analyses' in stats
        assert 'cache_hits' in stats
        assert 'cache_hit_rate' in stats
        
        print("  ✓ Integration adapter initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_exploration():
    """Test a simple exploration run."""
    print("\nTesting simple integrated exploration...")
    print("  (This may take 10-30 seconds...)")
    
    try:
        from ubf_protein.examples.integrated_exploration import run_integrated_exploration
        from ubf_protein.qcpp_config import get_testing_config
        
        # Run with minimal parameters for testing
        config = get_testing_config()
        config.cache_size = 10  # Small cache for test
        
        results = run_integrated_exploration(
            sequence="ACDEFGH",  # Very short sequence
            num_agents=2,  # Minimal agents
            iterations=5,  # Minimal iterations
            diversity='balanced',
            native_pdb=None,
            qcpp_config=config,
            output_file=None,
            verbose=False  # Suppress output during test
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'sequence' in results
        assert 'best_energy' in results
        assert 'qcpp_integration' in results
        assert results['sequence'] == "ACDEFGH"
        assert results['num_agents'] == 2
        assert results['iterations_per_agent'] == 5
        
        # Verify QCPP stats
        qcpp_stats = results['qcpp_integration']
        assert 'total_analyses' in qcpp_stats
        assert 'cache_hit_rate' in qcpp_stats
        assert qcpp_stats['enabled'] is True
        
        print("  ✓ Simple exploration completed successfully")
        print(f"    - Best energy: {results['best_energy']:.2f} kcal/mol")
        print(f"    - QCPP analyses: {qcpp_stats['total_analyses']}")
        print(f"    - Cache hit rate: {qcpp_stats['cache_hit_rate']:.1f}%")
        return True
        
    except Exception as e:
        print(f"  ✗ Exploration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line_parsing():
    """Test command-line argument parsing."""
    print("\nTesting command-line interface...")
    
    try:
        import argparse
        from ubf_protein.examples.integrated_exploration import main
        
        # Test that main function exists and is callable
        assert callable(main)
        
        print("  ✓ CLI interface defined correctly")
        print("  ℹ Full CLI test requires running as subprocess (skipped)")
        return True
        
    except Exception as e:
        print(f"  ✗ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("TASK 11 VERIFICATION: Integration Example Script")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Management", test_config_management),
        ("Mock QCPP Predictor", test_mock_qcpp_predictor),
        ("Integration Adapter", test_integration_adapter),
        ("Simple Exploration", test_simple_exploration),
        ("Command-Line Interface", test_command_line_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✓ Task 11 verification SUCCESSFUL!")
        print("\nThe integrated_exploration.py example script:")
        print("  ✓ Imports all required components")
        print("  ✓ Manages QCPP integration configuration")
        print("  ✓ Initializes QCPP adapter with mock predictor")
        print("  ✓ Creates multi-agent coordinator with QCPP")
        print("  ✓ Runs exploration with QCPP feedback")
        print("  ✓ Analyzes and reports QCPP statistics")
        print("  ✓ Provides command-line interface")
        print("\nTask 11 is COMPLETE!")
        return 0
    else:
        print(f"\n✗ Task 11 verification FAILED: {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
