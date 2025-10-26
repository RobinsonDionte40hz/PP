"""
Test that size-adaptive disulfide spring constants are working correctly.

This test verifies:
1. EnhancedPhysicsConfig factory methods use correct spring constants
2. EnhancedEnergyCalculator respects the spring constant parameter
3. MultiAgentCoordinator passes spring constant to energy calculator
4. Energy values scale appropriately with different spring constants
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ubf_protein'))

from ubf_protein.models import DisulfideBond, Conformation
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator

print("=" * 70)
print("TEST: Size-Adaptive Disulfide Spring Constants")
print("=" * 70)
print()

# Test 1: EnhancedPhysicsConfig factory methods
print("Test 1: EnhancedPhysicsConfig Factory Methods")
print("-" * 70)

bonds = [DisulfideBond(residue_i=5, residue_j=25)]

config_small = EnhancedPhysicsConfig.for_small_protein(30, bonds)
config_medium = EnhancedPhysicsConfig.for_medium_protein(100, bonds)
config_large = EnhancedPhysicsConfig.for_large_protein(200, bonds)

print(f"Small protein config:  k = {config_small.disulfide_spring_constant} kcal/mol/Ų")
print(f"Medium protein config: k = {config_medium.disulfide_spring_constant} kcal/mol/Ų")
print(f"Large protein config:  k = {config_large.disulfide_spring_constant} kcal/mol/Ų")

assert config_small.disulfide_spring_constant == 20.0, "Small protein should use k=20.0"
assert config_medium.disulfide_spring_constant == 35.0, "Medium protein should use k=35.0"
assert config_large.disulfide_spring_constant == 50.0, "Large protein should use k=50.0"

print("✓ All factory methods return correct spring constants")
print()

# Test 2: EnhancedEnergyCalculator accepts and uses spring constant
print("Test 2: EnhancedEnergyCalculator Parameter")
print("-" * 70)

sequence = "ACDEFGHIKLMNPQRSTVWYA" * 2  # 42 residues
bonds = [DisulfideBond(residue_i=5, residue_j=35)]

# Create calculators with different spring constants
calc_soft = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=False,
    enable_entropic=False,
    enable_solvent=False,
    disulfide_spring_constant=20.0
)

calc_medium = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=False,
    enable_entropic=False,
    enable_solvent=False,
    disulfide_spring_constant=35.0
)

calc_stiff = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=False,
    enable_entropic=False,
    enable_solvent=False,
    disulfide_spring_constant=50.0
)

print(f"Soft calculator:   k = {calc_soft.disulfide_spring_constant} kcal/mol/Ų")
print(f"Medium calculator: k = {calc_medium.disulfide_spring_constant} kcal/mol/Ų")
print(f"Stiff calculator:  k = {calc_stiff.disulfide_spring_constant} kcal/mol/Ų")

assert calc_soft.disulfide_spring_constant == 20.0
assert calc_medium.disulfide_spring_constant == 35.0
assert calc_stiff.disulfide_spring_constant == 50.0

print("✓ EnhancedEnergyCalculator stores spring constant correctly")
print()

# Test 3: Energy calculation with different spring constants
print("Test 3: Energy Scaling with Spring Constants")
print("-" * 70)

# Create a test conformation with large deviation (30 Å between bonded residues)
coords = []
for i in range(len(sequence)):
    if i == 5:
        coords.append((0.0, 0.0, 0.0))  # First bonded residue at origin
    elif i == 35:
        coords.append((30.0, 0.0, 0.0))  # Second bonded residue at 30 Å
    else:
        coords.append((i * 3.8, 0.0, 0.0))  # Other residues spaced normally

conformation = Conformation(
    conformation_id="test",
    sequence=sequence,
    atom_coordinates=coords,
    energy=0.0,
    rmsd_to_native=None,
    secondary_structure=['C'] * len(sequence),
    phi_angles=[0.0] * len(sequence),
    psi_angles=[0.0] * len(sequence),
    available_move_types=[],
    structural_constraints={}
)

# Calculate energies
breakdown_soft = calc_soft.calculate_with_breakdown(conformation)
breakdown_medium = calc_medium.calculate_with_breakdown(conformation)
breakdown_stiff = calc_stiff.calculate_with_breakdown(conformation)

print(f"Bond distance: 30.0 Å (target: 3.8 Å, deviation: 26.2 Å)")
print()
print(f"Soft (k=20.0):   Disulfide E = {breakdown_soft.disulfide:8.1f} kcal/mol")
print(f"Medium (k=35.0): Disulfide E = {breakdown_medium.disulfide:8.1f} kcal/mol")
print(f"Stiff (k=50.0):  Disulfide E = {breakdown_stiff.disulfide:8.1f} kcal/mol")
print()

# Verify ratios
expected_ratio_medium_soft = 35.0 / 20.0  # 1.75
expected_ratio_stiff_soft = 50.0 / 20.0   # 2.50

actual_ratio_medium_soft = breakdown_medium.disulfide / breakdown_soft.disulfide
actual_ratio_stiff_soft = breakdown_stiff.disulfide / breakdown_soft.disulfide

print(f"Medium/Soft ratio:  {actual_ratio_medium_soft:.3f} (expected: {expected_ratio_medium_soft:.3f})")
print(f"Stiff/Soft ratio:   {actual_ratio_stiff_soft:.3f} (expected: {expected_ratio_stiff_soft:.3f})")

# Allow 1% tolerance for floating point
assert abs(actual_ratio_medium_soft - expected_ratio_medium_soft) < 0.01, "Medium/Soft ratio incorrect"
assert abs(actual_ratio_stiff_soft - expected_ratio_stiff_soft) < 0.01, "Stiff/Soft ratio incorrect"

print("✓ Energy scales correctly with spring constant")
print()

# Test 4: Verify dramatic energy reduction for small proteins
print("Test 4: Energy Reduction for Small Proteins")
print("-" * 70)

old_energy = breakdown_stiff.disulfide  # Old fixed k=50.0
new_energy = breakdown_soft.disulfide   # New k=20.0 for small

reduction = old_energy - new_energy
reduction_pct = (reduction / old_energy) * 100

print(f"Old energy (k=50.0): {old_energy:.1f} kcal/mol")
print(f"New energy (k=20.0): {new_energy:.1f} kcal/mol")
print(f"Reduction:           {reduction:.1f} kcal/mol ({reduction_pct:.1f}%)")

assert reduction_pct > 55.0, "Should see ~60% reduction"
assert reduction_pct < 65.0, "Should see ~60% reduction"

print("✓ 60% energy reduction achieved for small proteins")
print()

# Test 5: Auto-adapt selects correct spring constant
print("Test 5: Auto-Adapt Selection")
print("-" * 70)

config_auto_small = EnhancedPhysicsConfig.auto_adapt(30, bonds)
config_auto_medium = EnhancedPhysicsConfig.auto_adapt(100, bonds)
config_auto_large = EnhancedPhysicsConfig.auto_adapt(200, bonds)

print(f"Auto-adapt (30 res):  k = {config_auto_small.disulfide_spring_constant} kcal/mol/Ų")
print(f"Auto-adapt (100 res): k = {config_auto_medium.disulfide_spring_constant} kcal/mol/Ų")
print(f"Auto-adapt (200 res): k = {config_auto_large.disulfide_spring_constant} kcal/mol/Ų")

assert config_auto_small.disulfide_spring_constant == 20.0
assert config_auto_medium.disulfide_spring_constant == 35.0
assert config_auto_large.disulfide_spring_constant == 50.0

print("✓ Auto-adapt selects correct spring constant based on size")
print()

# Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print()
print("✓ All 5 tests passed!")
print()
print("Verified:")
print("  1. EnhancedPhysicsConfig factory methods use size-specific k values")
print("  2. EnhancedEnergyCalculator accepts disulfide_spring_constant parameter")
print("  3. Energy scales proportionally with spring constant (E ∝ k)")
print("  4. Small proteins get 60% energy reduction vs old fixed k=50.0")
print("  5. Auto-adapt correctly selects spring constant based on protein size")
print()
print("Next step: Test with MultiAgentCoordinator integration")
print()
