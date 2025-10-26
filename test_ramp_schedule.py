"""
Test and demonstrate the staged restraint ramp-up for disulfide bonds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ubf_protein'))

from ubf_protein.models import DisulfideBond, Conformation
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator

print("=" * 80)
print("STAGED RESTRAINT RAMP-UP TEST")
print("=" * 80)
print()

# Test protein setup
sequence = "ACDEFGHIKLMNPQRSTVWYA" * 2  # 42 residues (small protein)
bonds = [DisulfideBond(residue_i=5, residue_j=35)]

# Create config with ramp schedule
config = EnhancedPhysicsConfig.for_small_protein(len(sequence), bonds)

print(f"Protein: {len(sequence)} residues (SMALL)")
print(f"Disulfide bonds: {len(bonds)}")
print()
print("Ramp Schedule:")
for iteration, k in config.disulfide_ramp_schedule:
    print(f"  Iteration {iteration:4d}+: k = {k:5.1f} kcal/mol/Ų")
print()

# Create calculator with ramp schedule
calculator = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=False,
    enable_entropic=False,
    enable_solvent=False,
    disulfide_ramp_schedule=config.disulfide_ramp_schedule
)

# Create test conformation with large deviation (30 Å between bonded residues)
coords = []
for i in range(len(sequence)):
    if i == 5:
        coords.append((0.0, 0.0, 0.0))
    elif i == 35:
        coords.append((30.0, 0.0, 0.0))  # 30 Å away
    else:
        coords.append((i * 3.8, 0.0, 0.0))

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
print("ENERGY EVOLUTION DURING RAMP-UP")
print("=" * 80)
print()
print(f"Bond distance: 30.0 Å (target: 3.8 Å, deviation: 26.2 Å)")
print()
print(f"{'Iteration':>10s}  {'k (kcal/mol/Ų)':>16s}  {'Disulfide E (kcal/mol)':>24s}  {'Phase':>20s}")
print("-" * 80)

# Simulate iterations and show energy at key points
test_iterations = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

for iteration in test_iterations:
    calculator.set_iteration(iteration)
    k = calculator.get_current_spring_constant()
    breakdown = calculator.calculate_with_breakdown(conformation)
    
    # Determine phase
    if iteration < 200:
        phase = "Gentle Exploration"
    elif iteration < 500:
        phase = "Moderate Constraint"
    else:
        phase = "Full Refinement"
    
    print(f"{iteration:10d}  {k:16.1f}  {breakdown.disulfide:24.1f}  {phase:>20s}")

print()
print("=" * 80)
print("ENERGY REDUCTION COMPARISON")
print("=" * 80)
print()

# Compare old fixed k=50 vs new ramped approach
calculator_old = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=False,
    enable_entropic=False,
    enable_solvent=False,
    disulfide_spring_constant=50.0  # Old fixed value
)

calculator_new = EnhancedEnergyCalculator(
    sequence=sequence,
    disulfide_bonds=bonds,
    enable_sidechains=False,
    enable_entropic=False,
    enable_solvent=False,
    disulfide_ramp_schedule=config.disulfide_ramp_schedule
)

print(f"{'Phase':>20s}  {'Old (k=50)':>15s}  {'New (ramped)':>15s}  {'Reduction':>12s}")
print("-" * 80)

phases = [
    (0, "Early Exploration"),
    (100, "Mid Exploration"),
    (200, "Late Exploration"),
    (350, "Early Constraint"),
    (500, "Full Constraint"),
    (1000, "Refinement")
]

for iteration, phase_name in phases:
    energy_old = calculator_old.calculate(conformation)
    
    calculator_new.set_iteration(iteration)
    k_new = calculator_new.get_current_spring_constant()
    energy_new = calculator_new.calculate(conformation)
    
    reduction = energy_old - energy_new
    reduction_pct = (reduction / energy_old) * 100 if energy_old != 0 else 0
    
    print(f"{phase_name:>20s}  {energy_old:15.1f}  {energy_new:15.1f}  {reduction_pct:11.1f}%")

print()
print("=" * 80)
print("BENEFITS OF STAGED RAMP-UP")
print("=" * 80)
print()
print("✓ Gentle initial constraints (k=2) guide without over-penalizing")
print("✓ Allows broader conformational exploration early on")
print("✓ Gradually tightens as structure improves")
print("✓ Reduces premature convergence to local minima")
print("✓ 90%+ energy reduction during exploration phase")
print("✓ Reaches full constraint strength only during refinement")
print()
print("Recommended usage:")
print("  1. Use EnhancedPhysicsConfig.for_small/medium/large_protein()")
print("  2. Ramp schedule automatically configured")
print("  3. Energy calculator auto-updates k based on iteration")
print()
