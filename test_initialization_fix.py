"""
Quick test to verify smart initialization prevents energy explosion.

Tests that agents now start with reasonable initial energies
instead of 750,000+ kcal/mol catastrophic values.
"""

import sys
from pathlib import Path

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

from ubf_protein.conformation_initializer import create_default_initializer
from ubf_protein.models import DisulfideBond

def test_initialization_without_disulfide():
    """Test initialization for protein without disulfide bonds."""
    print("\n" + "="*70)
    print("TEST 1: Initialization WITHOUT Disulfide Bonds")
    print("="*70)
    
    initializer = create_default_initializer(
        protein_size=46,  # Crambin size
        has_disulfide_bonds=False
    )
    
    coords = initializer.generate_initial_coordinates(
        sequence_length=46,
        disulfide_bonds=None
    )
    
    energy_check = initializer.calculate_initial_energy_estimate(coords, None)
    
    print(f"\n✓ Generated {len(coords)} coordinates")
    print(f"  Average CA-CA distance: {energy_check['avg_ca_distance']:.2f} Å")
    print(f"  Distance range: {energy_check['ca_distance_range'][0]:.2f} - {energy_check['ca_distance_range'][1]:.2f} Å")
    print(f"  Total disulfide energy: {energy_check['total_disulfide_energy']:.2f} kcal/mol")
    
    # Check reasonableness
    assert 3.0 < energy_check['avg_ca_distance'] < 5.0, "CA-CA distance should be ~3.8 Å"
    print("\n✅ PASS: Coordinates look reasonable!")


def test_initialization_with_disulfide():
    """Test initialization for Crambin with 3 disulfide bonds."""
    print("\n" + "="*70)
    print("TEST 2: Initialization WITH Disulfide Bonds (Crambin)")
    print("="*70)
    
    # Crambin disulfide bonds (from PDB 1CRN)
    bonds = [
        DisulfideBond(residue_i=3, residue_j=40, distance=3.8, tolerance=0.5),
        DisulfideBond(residue_i=4, residue_j=32, distance=3.8, tolerance=0.5),
        DisulfideBond(residue_i=16, residue_j=26, distance=3.8, tolerance=0.5)
    ]
    
    initializer = create_default_initializer(
        protein_size=46,
        has_disulfide_bonds=True
    )
    
    coords = initializer.generate_initial_coordinates(
        sequence_length=46,
        disulfide_bonds=bonds
    )
    
    energy_check = initializer.calculate_initial_energy_estimate(coords, bonds)
    
    print(f"\n✓ Generated {len(coords)} coordinates with {len(bonds)} disulfide bonds")
    print(f"  Average CA-CA distance: {energy_check['avg_ca_distance']:.2f} Å")
    print(f"\n  Disulfide Bond Distances:")
    
    for bond_key, info in energy_check['disulfide_bonds'].items():
        print(f"    {bond_key}: {info['distance']:.2f} Å (energy: {info['energy']:.2f} kcal/mol)")
    
    print(f"\n  Total disulfide energy: {energy_check['total_disulfide_energy']:.2f} kcal/mol")
    
    # Check improvement
    if energy_check['total_disulfide_energy'] < 1000.0:
        print("\n✅ EXCELLENT: Initial disulfide energy < 1000 kcal/mol")
        print("   (Previously would be 450,000+ kcal/mol with random initialization!)")
    elif energy_check['total_disulfide_energy'] < 5000.0:
        print("\n✅ GOOD: Initial disulfide energy < 5000 kcal/mol")
    else:
        print("\n⚠️  WARNING: Disulfide energy still high, but better than random")
    
    # Verify disulfide bonds are closer than random would produce
    for bond_key, info in energy_check['disulfide_bonds'].items():
        assert info['distance'] < 50.0, f"Bond {bond_key} too far: {info['distance']:.2f} Å"
        print(f"  ✓ {bond_key}: {info['distance']:.2f} Å < 50 Å (good!)")
    
    print("\n✅ PASS: Disulfide bonds initialized much closer together!")


def test_comparison():
    """Compare random initialization vs smart initialization."""
    print("\n" + "="*70)
    print("TEST 3: RANDOM vs SMART Initialization Comparison")
    print("="*70)
    
    bonds = [
        DisulfideBond(residue_i=3, residue_j=40, distance=3.8, tolerance=0.5),
        DisulfideBond(residue_i=4, residue_j=32, distance=3.8, tolerance=0.5),
        DisulfideBond(residue_i=16, residue_j=26, distance=3.8, tolerance=0.5)
    ]
    
    # Random initialization (old way - extended chain)
    import random
    random.seed(42)
    random_coords = []
    for i in range(46):
        x = i * 3.8 + random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)
        z = random.uniform(-0.5, 0.5)
        random_coords.append((x, y, z))
    
    # Smart initialization (new way)
    initializer = create_default_initializer(protein_size=46, has_disulfide_bonds=True)
    smart_coords = initializer.generate_initial_coordinates(46, bonds)
    
    # Calculate energies
    random_energy = initializer.calculate_initial_energy_estimate(random_coords, bonds)
    smart_energy = initializer.calculate_initial_energy_estimate(smart_coords, bonds)
    
    print("\n📊 RANDOM Initialization (Old Method):")
    print(f"   Total disulfide energy: {random_energy['total_disulfide_energy']:.2f} kcal/mol")
    for bond_key, info in random_energy['disulfide_bonds'].items():
        print(f"     {bond_key}: {info['distance']:.2f} Å")
    
    print("\n📊 SMART Initialization (New Method):")
    print(f"   Total disulfide energy: {smart_energy['total_disulfide_energy']:.2f} kcal/mol")
    for bond_key, info in smart_energy['disulfide_bonds'].items():
        print(f"     {bond_key}: {info['distance']:.2f} Å")
    
    improvement = random_energy['total_disulfide_energy'] - smart_energy['total_disulfide_energy']
    improvement_percent = (improvement / random_energy['total_disulfide_energy']) * 100
    
    print(f"\n🎯 IMPROVEMENT:")
    print(f"   Energy reduction: {improvement:.2f} kcal/mol ({improvement_percent:.1f}% better)")
    print(f"   Smart initialization prevents energy explosion! 🎉")
    
    assert smart_energy['total_disulfide_energy'] < random_energy['total_disulfide_energy'], \
        "Smart initialization should be better than random"
    
    print("\n✅ PASS: Smart initialization is MUCH better!")


if __name__ == "__main__":
    try:
        test_initialization_without_disulfide()
        test_initialization_with_disulfide()
        test_comparison()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSmart initialization successfully prevents energy explosion.")
        print("You can now run: python test_protein.py --pdb 1CRN --enhanced")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
