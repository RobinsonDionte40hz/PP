"""
Quick integration test for DisulfideBond and DisulfideDetector with UBF system.

This test verifies that the new disulfide bond functionality integrates
smoothly with the existing UBF protein system without breaking changes.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ubf_protein.models import DisulfideBond
from ubf_protein.disulfide_detector import DisulfideDetector


def test_basic_import():
    """Test that DisulfideBond can be imported."""
    print("✓ DisulfideBond imported successfully")


def test_model_creation():
    """Test creating DisulfideBond instances."""
    bond = DisulfideBond(residue_i=5, residue_j=55)
    assert bond.residue_i == 5
    assert bond.residue_j == 55
    assert bond.distance == 3.8
    assert bond.tolerance == 1.0
    print("✓ DisulfideBond model works correctly")


def test_detector_creation():
    """Test creating DisulfideDetector."""
    detector = DisulfideDetector()
    assert detector.default_distance == 3.8
    assert detector.default_tolerance == 1.0
    print("✓ DisulfideDetector instantiated successfully")


def test_sequence_prediction():
    """Test sequence-based prediction."""
    detector = DisulfideDetector()
    sequence = "ACDEFGHIKLMNPQC"
    bonds = detector.predict_from_sequence(sequence)
    
    assert len(bonds) == 1
    assert bonds[0].residue_i == 1
    assert bonds[0].residue_j == 14
    print(f"✓ Predicted {len(bonds)} disulfide bond from sequence")
    print(f"  {bonds[0]}")


def test_bond_validation():
    """Test bond constraint checking."""
    bond = DisulfideBond(residue_i=5, residue_j=55)
    
    # Test satisfied
    assert bond.is_satisfied(3.8) is True
    assert bond.is_satisfied(4.5) is True
    
    # Test violated
    assert bond.is_satisfied(5.5) is False
    assert bond.is_satisfied(2.0) is False
    
    print("✓ Bond constraint validation works")


def test_crambin_example():
    """Test with Crambin-like sequence."""
    detector = DisulfideDetector()
    
    # Crambin sequence (46 residues, 6 cysteines, 3 disulfide bonds)
    crambin_seq = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
    
    bonds = detector.predict_from_sequence(crambin_seq)
    print(f"✓ Crambin example: Predicted {len(bonds)} disulfide bonds")
    
    for i, bond in enumerate(bonds, 1):
        print(f"  Bond {i}: {bond}")
    
    # Validate bonds
    is_valid, errors = detector.validate_bonds(bonds, crambin_seq)
    assert is_valid is True
    print("✓ All predicted bonds are valid")


def test_with_ubf_components():
    """Test integration with UBF components."""
    from ubf_protein.models import ConsciousnessCoordinates, BehavioralStateData
    from ubf_protein.consciousness import ConsciousnessState
    
    # Create consciousness state
    consciousness = ConsciousnessState(frequency=8.0, coherence=0.7)
    
    # Create disulfide bond
    bond = DisulfideBond(residue_i=10, residue_j=50)
    
    # Both work together without conflicts
    assert consciousness.get_frequency() == 8.0
    assert bond.residue_i == 10
    
    print("✓ DisulfideBond integrates with UBF consciousness system")


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Testing DisulfideBond Integration with UBF System")
    print("=" * 70)
    
    tests = [
        test_basic_import,
        test_model_creation,
        test_detector_creation,
        test_sequence_prediction,
        test_bond_validation,
        test_crambin_example,
        test_with_ubf_components
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            return False
    
    print("=" * 70)
    print("✓ All integration tests passed!")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
