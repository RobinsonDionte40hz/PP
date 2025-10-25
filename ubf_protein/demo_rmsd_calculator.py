"""
Demo: RMSD Calculator

This script demonstrates the RMSD calculator with various test cases:
- Identical structures
- Translated structures
- Rotated structures
- Structures with noise
- Quality assessment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ubf_protein.rmsd_calculator import RMSDCalculator
import math


def main():
    print("=" * 80)
    print("RMSD Calculator Demo")
    print("=" * 80)
    
    # Test 1: Identical structures
    print("\n1. Identical Structures")
    print("-" * 80)
    
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    
    calculator = RMSDCalculator(align_structures=False)
    result = calculator.calculate_rmsd(coords, coords, calculate_metrics=True)
    
    print(f"Structure: Square (4 atoms)")
    print(f"RMSD: {result.rmsd:.6f} Å")
    print(f"GDT-TS: {result.gdt_ts:.2f}")
    print(f"TM-score: {result.tm_score:.4f}")
    print(f"Quality: {calculator.get_quality_assessment(result.rmsd, result.gdt_ts, result.tm_score)}")
    
    # Test 2: Translated structure (without alignment)
    print("\n2. Translated Structure (No Alignment)")
    print("-" * 80)
    
    coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    coords2 = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (6.0, 6.0, 5.0), (5.0, 6.0, 5.0)]
    
    calculator_no_align = RMSDCalculator(align_structures=False)
    result_no_align = calculator_no_align.calculate_rmsd(coords1, coords2, calculate_metrics=True)
    
    print(f"Translation: (5, 5, 5)")
    print(f"RMSD: {result_no_align.rmsd:.3f} Å")
    print(f"Expected: ~{math.sqrt(75):.3f} Å (sqrt(5²+5²+5²))")
    print(f"GDT-TS: {result_no_align.gdt_ts:.2f}")
    print(f"TM-score: {result_no_align.tm_score:.4f}")
    
    # Test 3: Translated structure (with alignment)
    print("\n3. Translated Structure (With Alignment)")
    print("-" * 80)
    
    calculator_align = RMSDCalculator(align_structures=True)
    result_align = calculator_align.calculate_rmsd(coords1, coords2, calculate_metrics=True)
    
    print(f"Translation: (5, 5, 5)")
    print(f"RMSD (unaligned): {result_no_align.rmsd:.3f} Å")
    print(f"RMSD (aligned): {result_align.rmsd:.3f} Å")
    print(f"Improvement: {result_no_align.rmsd - result_align.rmsd:.3f} Å")
    print(f"GDT-TS: {result_align.gdt_ts:.2f}")
    print(f"TM-score: {result_align.tm_score:.4f}")
    print(f"Quality: {calculator_align.get_quality_assessment(result_align.rmsd, result_align.gdt_ts, result_align.tm_score)}")
    
    # Test 4: Structure with noise
    print("\n4. Structure with Random Noise")
    print("-" * 80)
    
    coords_original = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), 
                      (3.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    coords_noisy = [(0.2, -0.1, 0.15), (1.1, 0.15, -0.2), (1.9, -0.15, 0.1),
                   (3.15, 0.1, -0.05), (3.8, -0.2, 0.25)]
    
    result_noise = calculator_align.calculate_rmsd(coords_original, coords_noisy, calculate_metrics=True)
    
    print(f"Structure: Linear chain (5 atoms)")
    print(f"Noise level: ~0.2 Å per coordinate")
    print(f"RMSD: {result_noise.rmsd:.3f} Å")
    print(f"GDT-TS: {result_noise.gdt_ts:.2f}")
    print(f"TM-score: {result_noise.tm_score:.4f}")
    print(f"Quality: {calculator_align.get_quality_assessment(result_noise.rmsd, result_noise.gdt_ts, result_noise.tm_score)}")
    
    # Test 5: Large protein simulation
    print("\n5. Large Protein Simulation (100 residues)")
    print("-" * 80)
    
    # Simulate a compact protein structure
    coords_large = []
    for i in range(100):
        # Helical-like pattern
        angle = i * 0.1
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        z = i * 1.5
        coords_large.append((x, y, z))
    
    # Create slightly perturbed version
    coords_perturbed = []
    for x, y, z in coords_large:
        coords_perturbed.append((x + 0.5, y + 0.5, z + 0.5))
    
    import time
    start_time = time.time()
    result_large = calculator_align.calculate_rmsd(coords_large, coords_perturbed, calculate_metrics=True)
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"Number of atoms: {result_large.n_atoms}")
    print(f"RMSD: {result_large.rmsd:.3f} Å")
    print(f"GDT-TS: {result_large.gdt_ts:.2f}")
    print(f"TM-score: {result_large.tm_score:.4f}")
    print(f"Calculation time: {elapsed_ms:.2f} ms")
    print(f"Performance target: < 100 ms for 500 residues")
    print(f"Quality: {calculator_align.get_quality_assessment(result_large.rmsd, result_large.gdt_ts, result_large.tm_score)}")
    
    # Test 6: Quality assessment examples
    print("\n6. Quality Assessment Examples")
    print("-" * 80)
    
    quality_examples = [
        ("Excellent prediction", 1.5, 85.0, 0.85),
        ("Good prediction", 3.0, 65.0, 0.65),
        ("Acceptable prediction", 5.0, 40.0, 0.4),
        ("Poor prediction", 10.0, 15.0, 0.15),
    ]
    
    for label, rmsd, gdt_ts, tm_score in quality_examples:
        quality = calculator.get_quality_assessment(rmsd, gdt_ts, tm_score)
        print(f"{label:25s} RMSD={rmsd:4.1f}Å  GDT-TS={gdt_ts:5.1f}  TM={tm_score:.2f}  → {quality}")
    
    # Test 7: Distance matrix
    print("\n7. Distance Matrix Example")
    print("-" * 80)
    
    coords_small1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    coords_small2 = [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 0.0, 0.0)]
    
    distances = calculator.calculate_distance_matrix(coords_small1, coords_small2)
    
    print("Structure 1: [(0,0,0), (1,0,0), (2,0,0)]")
    print("Structure 2: [(0.5,0,0), (1.5,0,0), (2.5,0,0)] (shifted by 0.5 Å)")
    print("\nDistance Matrix (Å):")
    print("         Atom 0   Atom 1   Atom 2")
    for i, row in enumerate(distances):
        print(f"Atom {i}:  ", end="")
        for dist in row:
            print(f"{dist:7.3f}  ", end="")
        print()
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    print("\nKey Features Demonstrated:")
    print("✓ Basic RMSD calculation")
    print("✓ Kabsch alignment for optimal superposition")
    print("✓ GDT-TS calculation (residues within 1,2,4,8 Å)")
    print("✓ TM-score calculation with length normalization")
    print("✓ Quality assessment (excellent/good/acceptable/poor)")
    print("✓ Performance: < 100 ms for 100 residues")
    print("✓ Distance matrix computation")


if __name__ == '__main__':
    main()
