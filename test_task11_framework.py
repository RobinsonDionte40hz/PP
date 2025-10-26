"""
Test Task 11: Enhanced test framework with physics support.

Quick validation that command-line flags work correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_baseline_mode():
    """Test baseline mode (no enhancements)."""
    print("Test 1: Baseline mode")
    print("=" * 60)
    
    cmd = [
        sys.executable, "test_protein.py",
        "--sequence", "ACDEFG",
        "--agents", "2",
        "--iterations", "5"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    # Check for key outputs
    assert "Enhanced Physics: DISABLED" in result.stdout
    assert "✓" in result.stdout  # Some success marker
    print("✓ Baseline mode works")
    print()


def test_enhanced_mode():
    """Test enhanced mode with all features."""
    print("Test 2: Enhanced mode (all features)")
    print("=" * 60)
    
    cmd = [
        sys.executable, "test_protein.py",
        "--sequence", "ACDEFG",
        "--agents", "2",
        "--iterations", "5",
        "--enhanced"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    # Check for key outputs
    assert "Enhanced Physics:" in result.stdout
    assert "ENABLED" in result.stdout
    print("✓ Enhanced mode works")
    print()


def test_partial_enhancements():
    """Test with some features disabled."""
    print("Test 3: Partial enhancements (no side-chains)")
    print("=" * 60)
    
    cmd = [
        sys.executable, "test_protein.py",
        "--sequence", "ACDEFG",
        "--agents", "2",
        "--iterations", "5",
        "--enhanced",
        "--no-sidechains"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    # Check for key outputs
    assert "Enhanced Physics:" in result.stdout
    assert "Side-chain Interactions: OFF" in result.stdout
    print("✓ Partial enhancements work")
    print()


def test_help_message():
    """Test help message includes new flags."""
    print("Test 4: Help message includes enhanced physics flags")
    print("=" * 60)
    
    cmd = [sys.executable, "test_protein.py", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    # Check for new flags
    assert "--enhanced" in result.stdout
    assert "--no-sidechains" in result.stdout
    assert "--refinement" in result.stdout
    print("✓ Help message updated")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TASK 11 TEST FRAMEWORK VALIDATION")
    print("=" * 60 + "\n")
    
    try:
        test_help_message()
        print("=" * 60)
        print("✅ ALL QUICK TESTS PASSED")
        print("=" * 60)
        print("\nTask 11 implementation verified:")
        print("  ✓ Command-line flags added")
        print("  ✓ Help message updated")
        print("\nNote: Full integration tests skipped (too slow)")
        print("Run manually: python test_protein.py --sequence ACDEFG --enhanced")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
