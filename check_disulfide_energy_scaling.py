"""
Demonstrate the impact of size-adaptive disulfide spring constants.

This script shows how energy penalties scale differently for small, medium,
and large proteins with the new adaptive spring constants.
"""

print("=" * 70)
print("DISULFIDE ENERGY SCALING ANALYSIS")
print("=" * 70)
print()
print("Formula: E = 0.5 * k * (r - r₀)²")
print("Target distance (r₀): 3.8 Å")
print()

# Test case 1: Large deviation (40 Å - typical for unfolded small proteins)
print("-" * 70)
print("Case 1: Large deviation (40 Å from target)")
print("-" * 70)
deviation = 40.0 - 3.8
print(f"Deviation: {deviation:.1f} Å")
print()

configs = [
    ("Small (<50 res)", 20.0),
    ("Medium (50-150 res)", 35.0),
    ("Large (>150 res)", 50.0)
]

for size, k in configs:
    energy = 0.5 * k * deviation ** 2
    print(f"  {size:25s}  k={k:4.1f} kcal/mol/Ų  →  E={energy:8.1f} kcal/mol")

print()
print("Comparison to previous fixed k=50.0:")
old_energy = 0.5 * 50.0 * deviation ** 2
new_energy_small = 0.5 * 20.0 * deviation ** 2
reduction = old_energy - new_energy_small
print(f"  Old (all sizes): {old_energy:.1f} kcal/mol")
print(f"  New (small):     {new_energy_small:.1f} kcal/mol")
print(f"  Reduction:       {reduction:.1f} kcal/mol ({reduction/old_energy*100:.1f}%)")

# Test case 2: Moderate deviation (10 Å)
print()
print("-" * 70)
print("Case 2: Moderate deviation (10 Å from target)")
print("-" * 70)
deviation = 10.0 - 3.8
print(f"Deviation: {deviation:.1f} Å")
print()

for size, k in configs:
    energy = 0.5 * k * deviation ** 2
    print(f"  {size:25s}  k={k:4.1f} kcal/mol/Ų  →  E={energy:7.1f} kcal/mol")

print()
print("Comparison to previous fixed k=50.0:")
old_energy = 0.5 * 50.0 * deviation ** 2
new_energy_small = 0.5 * 20.0 * deviation ** 2
reduction = old_energy - new_energy_small
print(f"  Old (all sizes): {old_energy:.1f} kcal/mol")
print(f"  New (small):     {new_energy_small:.1f} kcal/mol")
print(f"  Reduction:       {reduction:.1f} kcal/mol ({reduction/old_energy*100:.1f}%)")

# Test case 3: Small deviation (5 Å)
print()
print("-" * 70)
print("Case 3: Small deviation (5 Å from target)")
print("-" * 70)
deviation = 5.0 - 3.8
print(f"Deviation: {deviation:.1f} Å")
print()

for size, k in configs:
    energy = 0.5 * k * deviation ** 2
    print(f"  {size:25s}  k={k:4.1f} kcal/mol/Ų  →  E={energy:7.2f} kcal/mol")

# Summary
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("✓ Small proteins now use k=20.0 (60% softer than before)")
print("✓ Medium proteins use k=35.0 (30% softer than before)")
print("✓ Large proteins keep k=50.0 (unchanged)")
print()
print("Benefits:")
print("  • Reduced over-penalization of small proteins")
print("  • More realistic energy landscapes for different protein sizes")
print("  • Better conformational sampling in early exploration")
print("  • Maintained constraint strength for large proteins where needed")
print()
print("Configuration:")
print("  • EnhancedPhysicsConfig.for_small_protein()  → k=20.0")
print("  • EnhancedPhysicsConfig.for_medium_protein() → k=35.0")
print("  • EnhancedPhysicsConfig.for_large_protein()  → k=50.0")
print("  • EnhancedPhysicsConfig.auto_adapt()         → Automatic selection")
print()
