#!/usr/bin/env python3
"""
Test enhanced energy with CLASHING structures (atoms too close).
This simulates what happens during bad moves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

from ubf_protein.models import Conformation
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
import numpy as np

# Create a small test protein
sequence = "ACDEFGH"
n = len(sequence)

print("="*70)
print("TESTING ENHANCED ENERGY WITH CLASHING ATOMS")
print("="*70)

# Test Case 1: Normal extended chain
print("\n1. NORMAL EXTENDED CHAIN (baseline)")
print("-" * 70)
coords_normal = []
for i in range(n):
    x = i * 3.8  
    y = 0.0
    z = 0.0
    coords_normal.append((x, y, z))

conf_normal = Conformation(
    conformation_id="normal",
    sequence=sequence,
    atom_coordinates=coords_normal,
    energy=0.0,
    rmsd_to_native=0.0,
    secondary_structure=["C"] * n,
    phi_angles=[0.0] * n,
    psi_angles=[0.0] * n,
    available_move_types=[],
    structural_constraints={}
)

calc = EnhancedEnergyCalculator(
    sequence=sequence,
    enable_sidechains=True,
    enable_disulfide=False,
    enable_entropic=False,
    enable_solvent=False
)

breakdown = calc.calculate_with_breakdown(conf_normal)
print(f"Total Energy: {breakdown.total:.2f} kcal/mol")
print(f"  Base MM:    {breakdown.base:.2f}")
print(f"  Side-chain: {breakdown.sidechain:.2f}")

# Test Case 2: MILD CLASH (atoms 2Å apart instead of 3.8Å)
print("\n2. MILD CLASH (2Å spacing - atoms slightly too close)")
print("-" * 70)
coords_mild = []
for i in range(n):
    x = i * 2.0  # Much closer!
    y = 0.0
    z = 0.0
    coords_mild.append((x, y, z))

conf_mild = Conformation(
    conformation_id="mild_clash",
    sequence=sequence,
    atom_coordinates=coords_mild,
    energy=0.0,
    rmsd_to_native=0.0,
    secondary_structure=["C"] * n,
    phi_angles=[0.0] * n,
    psi_angles=[0.0] * n,
    available_move_types=[],
    structural_constraints={}
)

breakdown_mild = calc.calculate_with_breakdown(conf_mild)
print(f"Total Energy: {breakdown_mild.total:.2e} kcal/mol")
print(f"  Base MM:    {breakdown_mild.base:.2e}")
print(f"  Side-chain: {breakdown_mild.sidechain:.2e}")

ratio = abs(breakdown_mild.sidechain / breakdown.sidechain) if breakdown.sidechain != 0 else 0
print(f"Side-chain energy increased by {ratio:.0f}x")

if abs(breakdown_mild.total) > 10000:
    print("🔴 PROBLEM: Energy exploded with mild clash!")

# Test Case 3: SEVERE CLASH (atoms 1Å apart)
print("\n3. SEVERE CLASH (1Å spacing - severe overlap)")
print("-" * 70)
coords_severe = []
for i in range(n):
    x = i * 1.0  # Very close!
    y = 0.0
    z = 0.0
    coords_severe.append((x, y, z))

conf_severe = Conformation(
    conformation_id="severe_clash",
    sequence=sequence,
    atom_coordinates=coords_severe,
    energy=0.0,
    rmsd_to_native=0.0,
    secondary_structure=["C"] * n,
    phi_angles=[0.0] * n,
    psi_angles=[0.0] * n,
    available_move_types=[],
    structural_constraints={}
)

breakdown_severe = calc.calculate_with_breakdown(conf_severe)
print(f"Total Energy: {breakdown_severe.total:.2e} kcal/mol")
print(f"  Base MM:    {breakdown_severe.base:.2e}")
print(f"  Side-chain: {breakdown_severe.sidechain:.2e}")

ratio_severe = abs(breakdown_severe.sidechain / breakdown.sidechain) if breakdown.sidechain != 0 else 0
print(f"Side-chain energy increased by {ratio_severe:.0f}x")

if abs(breakdown_severe.total) > 1e9:
    print("🔴 CRITICAL: Energy is in BILLIONS! This is the problem!")
elif abs(breakdown_severe.total) > 1e6:
    print("🔴 PROBLEM: Energy is in MILLIONS! Way too high!")
elif abs(breakdown_severe.total) > 100000:
    print("⚠️  WARNING: Energy is over 100k - very high but not insane")
elif abs(breakdown_severe.total) > 10000:
    print("⚠️  Energy is over 10k - high penalty but maybe acceptable")
else:
    print("✓ Energy penalty is reasonable even with severe clash")

# Test Case 4: EXTREME OVERLAP (all atoms at same position!)
print("\n4. EXTREME OVERLAP (all atoms at origin)")
print("-" * 70)
coords_overlap = [(0.0, 0.0, 0.0)] * n

conf_overlap = Conformation(
    conformation_id="extreme_overlap",
    sequence=sequence,
    atom_coordinates=coords_overlap,
    energy=0.0,
    rmsd_to_native=0.0,
    secondary_structure=["C"] * n,
    phi_angles=[0.0] * n,
    psi_angles=[0.0] * n,
    available_move_types=[],
    structural_constraints={}
)

breakdown_overlap = calc.calculate_with_breakdown(conf_overlap)
print(f"Total Energy: {breakdown_overlap.total:.2e} kcal/mol")
print(f"  Base MM:    {breakdown_overlap.base:.2e}")
print(f"  Side-chain: {breakdown_overlap.sidechain:.2e}")

if abs(breakdown_overlap.total) > 1e15:
    print("🔴 ASTRONOMICALLY HIGH: This explains the 10^15 energies you saw!")
elif abs(breakdown_overlap.total) > 1e12:
    print("🔴 CRITICAL: Energy in TRILLIONS!")
elif abs(breakdown_overlap.total) > 1e9:
    print("🔴 CRITICAL: Energy in BILLIONS!")
elif abs(breakdown_overlap.total) > 1e6:
    print("🔴 PROBLEM: Energy in MILLIONS!")
else:
    print("✓ Even extreme overlap gives reasonable penalty")

print()
print("="*70)
print("DIAGNOSIS:")
print("="*70)
print("The enhanced energy calculator likely has no upper bound on repulsion.")
print("When atoms clash severely (like during exploration), energies explode.")
print()
print("SOLUTION NEEDED:")
print("1. Add energy capping/softening for steric repulsion")
print("2. Use softer potential (e.g., Lennard-Jones 6-12 instead of Gaussian)")
print("3. Add maximum energy cutoff (e.g., cap at 5000 kcal/mol)")
print("="*70)
