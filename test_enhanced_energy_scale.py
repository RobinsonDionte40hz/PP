#!/usr/bin/env python3
"""
Quick test to diagnose enhanced energy scale issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

from ubf_protein.models import Conformation
from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
import numpy as np

# Create a small test protein (7 residues)
sequence = "ACDEFGH"
n = len(sequence)

# Create random but reasonable coordinates (roughly extended chain)
coords = []
for i in range(n):
    x = i * 3.8  # Roughly C-alpha spacing
    y = np.random.rand() * 2.0 - 1.0  # Small random offset
    z = np.random.rand() * 2.0 - 1.0
    coords.append((x, y, z))

# Create conformation
conf = Conformation(
    conformation_id="test_1",
    sequence=sequence,
    atom_coordinates=coords,
    energy=0.0,
    rmsd_to_native=0.0,
    secondary_structure=["C"] * n,
    phi_angles=[0.0] * n,
    psi_angles=[0.0] * n,
    available_move_types=[],
    structural_constraints={}
)

print("="*70)
print("ENERGY SCALE DIAGNOSTIC TEST")
print("="*70)
print(f"Sequence: {sequence} ({n} residues)")
print(f"Structure: Extended chain with small random perturbations")
print()

# Test 1: Baseline energy
print("1. BASELINE ENERGY (MolecularMechanicsEnergy)")
print("-" * 70)
try:
    baseline_calc = MolecularMechanicsEnergy()
    baseline_energy = baseline_calc.calculate(conf)
    print(f"   Total Energy: {baseline_energy:.2f} kcal/mol")
    
    if hasattr(baseline_calc, 'calculate_with_components'):
        components = baseline_calc.calculate_with_components(conf)
        print(f"   Components:")
        for key, value in components.items():
            if key != 'total':
                print(f"     {key:15s}: {value:12.2f} kcal/mol")
except Exception as e:
    print(f"   ERROR: {e}")
    baseline_energy = None

print()

# Test 2: Enhanced energy (all features disabled - should match baseline)
print("2. ENHANCED ENERGY (All features OFF - should match baseline)")
print("-" * 70)
try:
    enhanced_off_calc = EnhancedEnergyCalculator(
        sequence=sequence,
        enable_sidechains=False,
        enable_disulfide=False,
        enable_entropic=False,
        enable_solvent=False
    )
    enhanced_off_energy = enhanced_off_calc.calculate(conf)
    print(f"   Total Energy: {enhanced_off_energy:.2f} kcal/mol")
    
    if baseline_energy is not None:
        diff = abs(enhanced_off_energy - baseline_energy)
        print(f"   Difference from baseline: {diff:.2f} kcal/mol")
        if diff < 0.1:
            print(f"   ✓ MATCH - Enhanced calculator works correctly when features disabled")
        else:
            print(f"   ✗ MISMATCH - There's a problem with the base implementation")
except Exception as e:
    print(f"   ERROR: {e}")
    enhanced_off_energy = None

print()

# Test 3: Enhanced energy (only sidechains ON)
print("3. ENHANCED ENERGY (Side-chains ON)")
print("-" * 70)
try:
    enhanced_sc_calc = EnhancedEnergyCalculator(
        sequence=sequence,
        enable_sidechains=True,
        enable_disulfide=False,
        enable_entropic=False,
        enable_solvent=False
    )
    breakdown = enhanced_sc_calc.calculate_with_breakdown(conf)
    print(f"   Total Energy: {breakdown.total:.2f} kcal/mol")
    print(f"   Components:")
    print(f"     Base MM       : {breakdown.base:.2f} kcal/mol")
    print(f"     Side-chains   : {breakdown.sidechain:.2f} kcal/mol")
    print(f"     Disulfide     : {breakdown.disulfide:.2f} kcal/mol")
    print(f"     Entropic      : {breakdown.entropic:.2f} kcal/mol")
    
    if baseline_energy is not None:
        ratio = breakdown.sidechain / baseline_energy if baseline_energy != 0 else 0
        print(f"   Side-chain / Base ratio: {ratio:.2f}x")
        
        if abs(breakdown.sidechain) > 10000:
            print(f"   ⚠️  WARNING: Side-chain energy is HUGE ({breakdown.sidechain:.0f})!")
            print(f"      This is {abs(breakdown.sidechain/baseline_energy):.0f}x larger than baseline")
        elif abs(breakdown.sidechain) < abs(baseline_energy):
            print(f"   ✓ Side-chain energy is reasonable relative to baseline")
        else:
            print(f"   ? Side-chain energy is {ratio:.1f}x baseline (may be too high)")
            
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Enhanced energy (ALL features ON)
print("4. ENHANCED ENERGY (ALL features ON)")
print("-" * 70)
try:
    enhanced_all_calc = EnhancedEnergyCalculator(
        sequence=sequence,
        enable_sidechains=True,
        enable_disulfide=False,  # No disulfide bonds in test sequence
        enable_entropic=True,
        enable_solvent=True
    )
    breakdown = enhanced_all_calc.calculate_with_breakdown(conf)
    print(f"   Total Energy: {breakdown.total:.2f} kcal/mol")
    print(f"   Components:")
    print(f"     Base MM       : {breakdown.base:.2f} kcal/mol")
    print(f"     Side-chains   : {breakdown.sidechain:.2f} kcal/mol")
    print(f"     Disulfide     : {breakdown.disulfide:.2f} kcal/mol")
    print(f"     Entropic      : {breakdown.entropic:.2f} kcal/mol")
    
    if baseline_energy is not None:
        ratio = breakdown.total / baseline_energy if baseline_energy != 0 else 0
        print(f"   Total / Base ratio: {ratio:.2f}x")
        
        if abs(breakdown.total) > 100000:
            print(f"   🔴 CRITICAL: Total energy is ASTRONOMICALLY HIGH!")
            print(f"      Agents will reject 100% of moves with this scale")
        elif abs(breakdown.total) > 10000:
            print(f"   ⚠️  WARNING: Total energy is very high")
            print(f"      Agents will likely reject most moves")
        elif abs(breakdown.total) < 10 * abs(baseline_energy):
            print(f"   ✓ Total energy scale is reasonable")
        else:
            print(f"   ? Total energy is {ratio:.1f}x baseline")
            
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
