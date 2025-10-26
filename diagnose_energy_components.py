"""
Diagnose which energy components are causing the high total energy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ubf_protein'))

from ubf_protein.models import DisulfideBond, Conformation
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
from ubf_protein.disulfide_detector import DisulfideDetector

print("=" * 80)
print("ENERGY COMPONENT DIAGNOSIS")
print("=" * 80)
print()

# Use 1UBQ (Ubiquitin) - 76 residues
sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
pdb_file = "pdb_cache/pdb1ubq.ent"

print(f"Protein: Ubiquitin (1UBQ)")
print(f"Sequence length: {len(sequence)} residues")
print()

# Detect disulfide bonds
detector = DisulfideDetector()
try:
    bonds = detector.detect_from_pdb(pdb_file)
    print(f"Disulfide bonds detected: {len(bonds)}")
    if bonds:
        for bond in bonds:
            print(f"  {bond}")
except:
    bonds = []
    print(f"No PDB file found or no disulfide bonds")
print()

# Create config for medium protein (76 residues)
config = EnhancedPhysicsConfig.auto_adapt(len(sequence), bonds)
print(f"Auto-adapt selected:")
print(f"  Spring constant: {config.disulfide_spring_constant} kcal/mol/Ų")
print()

# Create test conformation with realistic but not-perfect structure
# Spread residues along a line with some variation
coords = []
for i in range(len(sequence)):
    x = i * 3.8  # Ideal spacing
    y = 0.0
    z = 0.0
    coords.append((x, y, z))

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

print("=" * 80)
print("TEST 1: All Components Enabled")
print("=" * 80)
print()

calculator_full = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=True,
    enable_solvent=True,
    enable_entropic=True,
    disulfide_spring_constant=config.disulfide_spring_constant
)

breakdown = calculator_full.calculate_with_breakdown(conformation)

print("Energy Breakdown:")
print(f"  Total:                {breakdown.total:10.2f} kcal/mol")
print(f"  ├─ Base MM:           {breakdown.base:10.2f} kcal/mol")
print(f"  │  ├─ Bond:           {breakdown.bond:10.2f} kcal/mol")
print(f"  │  ├─ Angle:          {breakdown.angle:10.2f} kcal/mol")
print(f"  │  ├─ Dihedral:       {breakdown.dihedral:10.2f} kcal/mol")
print(f"  │  ├─ VdW:            {breakdown.vdw:10.2f} kcal/mol")
print(f"  │  ├─ Electrostatic:  {breakdown.electrostatic:10.2f} kcal/mol")
print(f"  │  ├─ H-bond:         {breakdown.hbond:10.2f} kcal/mol")
print(f"  │  └─ Compactness:    {breakdown.compactness:10.2f} kcal/mol")
print(f"  ├─ Side-chains:       {breakdown.sidechain:10.2f} kcal/mol")
print(f"  ├─ Disulfide:         {breakdown.disulfide:10.2f} kcal/mol")
print(f"  └─ Entropic:          {breakdown.entropic:10.2f} kcal/mol")
print()

# Identify the largest component
components = {
    'Base MM': breakdown.base,
    'Side-chains': breakdown.sidechain,
    'Disulfide': breakdown.disulfide,
    'Entropic': breakdown.entropic
}
max_component = max(components.items(), key=lambda x: abs(x[1]))
print(f"⚠️  Largest component: {max_component[0]} = {max_component[1]:.2f} kcal/mol")
print()

print("=" * 80)
print("TEST 2: Component-by-Component Analysis")
print("=" * 80)
print()

# Test with only base MM
calc_base = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=[],
    enable_sidechains=False,
    enable_solvent=False,
    enable_entropic=False
)
energy_base = calc_base.calculate(conformation)
print(f"Base MM only:           {energy_base:10.2f} kcal/mol")

# Test with base + sidechains
calc_sc = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=[],
    enable_sidechains=True,
    enable_solvent=False,
    enable_entropic=False
)
energy_sc = calc_sc.calculate(conformation)
sidechain_contribution = energy_sc - energy_base
print(f"  + Side-chains:        {sidechain_contribution:+10.2f} kcal/mol  (Total: {energy_sc:.2f})")

# Test with base + disulfide (if bonds exist)
if bonds:
    calc_ss = EnhancedEnergyCalculator(
        sequence=sequence,
        disulfide_bonds=bonds,
        enable_sidechains=False,
        enable_solvent=False,
        enable_entropic=False,
        disulfide_spring_constant=config.disulfide_spring_constant
    )
    energy_ss = calc_ss.calculate(conformation)
    disulfide_contribution = energy_ss - energy_base
    print(f"  + Disulfide (k={config.disulfide_spring_constant}): {disulfide_contribution:+10.2f} kcal/mol  (Total: {energy_ss:.2f})")

print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

if abs(breakdown.disulfide) > 1000:
    print("🔴 DISULFIDE ENERGY TOO HIGH")
    print("   Issue: Large deviations from 3.8 Å target")
    print("   Solutions:")
    print("   1. Initialize with better starting structure")
    print("   2. Add disulfide-guided moves earlier in exploration")
    print("   3. Use even softer spring constant for exploration phase")
    print(f"   4. Current k={config.disulfide_spring_constant}, try k=10.0 for early exploration")
    print()

if abs(breakdown.sidechain) > 500:
    print("🔴 SIDECHAIN ENERGY TOO HIGH")
    print("   Issue: Strong repulsive interactions or electrostatic clashes")
    print("   Solutions:")
    print("   1. Reduce sidechain interaction strength (scale by 0.5)")
    print("   2. Increase cutoff distance to reduce pairwise interactions")
    print("   3. Disable sidechains during early exploration")
    print()

if abs(breakdown.base) > 200:
    print("🟡 BASE MM ENERGY ELEVATED")
    print("   Issue: Bond/angle strain or steric clashes")
    print("   Solutions:")
    print("   1. Improve move generator to avoid geometry violations")
    print("   2. Add structure repair after moves")
    print("   3. Use looser validation thresholds during exploration")
    print()

print("💡 STRATEGY:")
print("   Phase 1 (iterations 0-200): Exploration with weak constraints")
print("     - Disable sidechains")
print("     - Use k=10.0 for disulfides")
print("     - Focus on gross structure")
print()
print("   Phase 2 (iterations 200-500): Refinement")
print("     - Enable sidechains")
print("     - Use size-adaptive k (20/35/50)")
print("     - Add local refinement")
print()
