"""Profile RMSD calculation performance to ensure <100ms for 500 residues."""
import time
import cProfile
import pstats
from io import StringIO

from ubf_protein.rmsd_calculator import RMSDCalculator

def create_test_structures(num_residues: int):
    """Create two test structures for RMSD calculation."""
    # Structure 1: extended chain
    coords1 = [(i * 3.8, 0.0, 0.0) for i in range(num_residues)]
    
    # Structure 2: helical (shifted and rotated)
    import math
    coords2 = []
    for i in range(num_residues):
        # Helical geometry: rise of 1.5Å per residue, 100° rotation
        angle = math.radians(100 * i)
        radius = 2.3  # Distance from helix axis
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = i * 1.5
        coords2.append((x, y, z))
    
    return coords1, coords2

def profile_rmsd_calculation(num_residues: int, num_iterations: int = 50):
    """Profile RMSD calculation for proteins of given size."""
    print(f"\n{'='*60}")
    print(f"Profiling {num_residues}-residue protein RMSD")
    print(f"{'='*60}")
    
    calculator = RMSDCalculator(align_structures=True)
    coords1, coords2 = create_test_structures(num_residues)
    
    # Warmup
    for _ in range(5):
        calculator.calculate_rmsd(coords1, coords2, calculate_metrics=True)
    
    # Timing test
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=True)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nTiming Results ({num_iterations} iterations):")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Std Dev: {std_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print(f"  RMSD value: {result.rmsd:.2f}Å")
    
    # Target check
    target = 100.0 if num_residues >= 500 else 50.0
    status = "✓ PASS" if avg_time < target else "✗ FAIL"
    print(f"\n  Target:  <{target:.0f}ms ... {status}")
    
    # Detailed profiling for 500-residue case
    if num_residues >= 500:
        print(f"\n{'='*60}")
        print(f"Detailed Profile (500-residue RMSD)")
        print(f"{'='*60}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(20):
            calculator.calculate_rmsd(coords1, coords2, calculate_metrics=True)
        
        profiler.disable()
        
        # Print top 15 time-consuming functions
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(15)
        
        print(s.getvalue())
    
    return avg_time

def main():
    """Run RMSD performance profiling."""
    print("\n" + "="*60)
    print("RMSD CALCULATION PERFORMANCE PROFILING")
    print("="*60)
    
    # Test various protein sizes
    sizes = [10, 50, 100, 250, 500]
    results = {}
    
    for size in sizes:
        iterations = 50 if size <= 250 else 20
        avg_time = profile_rmsd_calculation(size, iterations)
        results[size] = avg_time
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Size (residues)':<20} {'Avg Time':<15} {'Target':<15} {'Status':<10}")
    print("-"*60)
    
    for size, avg_time in results.items():
        target = 100.0 if size >= 500 else 50.0
        status = "✓ PASS" if avg_time < target else "✗ FAIL"
        print(f"{size:<20} {avg_time:>10.2f}ms {target:>10.0f}ms     {status}")
    
    # Complexity analysis
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS")
    print("="*60)
    
    if len(results) >= 3:
        sizes_list = list(results.keys())
        times_list = list(results.values())
        
        for i in range(len(sizes_list) - 1):
            size_ratio = sizes_list[i+1] / sizes_list[i]
            time_ratio = times_list[i+1] / times_list[i]
            expected_linear = size_ratio
            expected_quadratic = size_ratio ** 2
            
            if time_ratio > expected_linear * 1.5:
                complexity = "O(N²)" if abs(time_ratio - expected_quadratic) < abs(time_ratio - expected_linear) else "O(N log N)"
            else:
                complexity = "O(N)"
            
            print(f"{sizes_list[i]:>3d} → {sizes_list[i+1]:>3d} residues:")
            print(f"  Size ratio: {size_ratio:.2f}x")
            print(f"  Time ratio: {time_ratio:.2f}x")
            print(f"  Apparent complexity: {complexity}")
            print()

if __name__ == "__main__":
    main()
