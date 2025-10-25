"""Profile energy calculation performance to identify optimization targets."""
import time
import cProfile
import pstats
from io import StringIO

from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.models import Conformation

def create_test_protein(num_residues: int) -> Conformation:
    """Create a test protein of specified size."""
    sequence = "A" * num_residues
    
    # Create extended chain geometry
    atom_coords = []
    for i in range(num_residues):
        # CA atom at ~3.8Å spacing along x-axis
        atom_coords.append([i * 3.8, 0.0, 0.0])
    
    # Simple backbone angles
    phi_angles = [-60.0] * num_residues  # Alpha helix-like
    psi_angles = [-45.0] * num_residues
    
    return Conformation(
        sequence=sequence,
        conformation_id="prof_0",
        atom_coordinates=atom_coords,
        phi_angles=phi_angles,
        psi_angles=psi_angles,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=["C"] * num_residues,
        available_move_types=[],
        structural_constraints={}
    )

def profile_energy_calculation(num_residues: int, num_iterations: int = 100):
    """Profile energy calculation for protein of given size."""
    print(f"\n{'='*60}")
    print(f"Profiling {num_residues}-residue protein")
    print(f"{'='*60}")
    
    energy_calc = MolecularMechanicsEnergy()
    conformation = create_test_protein(num_residues)
    
    # Warmup
    for _ in range(5):
        energy_calc.calculate(conformation)
    
    # Timing test
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = energy_calc.calculate(conformation)
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
    
    # Target check
    target = 50.0 if num_residues <= 100 else 100.0
    status = "✓ PASS" if avg_time < target else "✗ FAIL"
    print(f"\n  Target:  <{target:.0f}ms ... {status}")
    
    # Detailed profiling for largest case
    if num_residues >= 100:
        print(f"\n{'='*60}")
        print(f"Detailed Profile (100-residue protein)")
        print(f"{'='*60}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(50):
            energy_calc.calculate(conformation)
        
        profiler.disable()
        
        # Print top 20 time-consuming functions
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        print(s.getvalue())
        
        # Print just the key methods
        print("\nKey Method Breakdown:")
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        stats_dict = ps.stats  # type: ignore - pstats.Stats has stats attribute
        for func, stat_tuple in list(stats_dict.items())[:20]:
            if 'energy_function' in str(func):
                cc, nc, tt, ct, callers = stat_tuple
                print(f"  {func[2]:40s} {ct*1000/50:8.2f}ms/call ({nc:4d} calls)")
    
    return avg_time

def main():
    """Run performance profiling."""
    print("\n" + "="*60)
    print("ENERGY CALCULATION PERFORMANCE PROFILING")
    print("="*60)
    
    # Test various protein sizes
    sizes = [10, 25, 50, 100, 200]
    results = {}
    
    for size in sizes:
        iterations = 100 if size <= 100 else 50
        avg_time = profile_energy_calculation(size, iterations)
        results[size] = avg_time
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Size (residues)':<20} {'Avg Time':<15} {'Target':<15} {'Status':<10}")
    print("-"*60)
    
    for size, avg_time in results.items():
        target = 50.0 if size <= 100 else 100.0
        status = "✓ PASS" if avg_time < target else "✗ FAIL"
        print(f"{size:<20} {avg_time:>10.2f}ms {target:>10.0f}ms     {status}")
    
    # Complexity analysis
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS")
    print("="*60)
    
    # Check if growth is quadratic (O(N²))
    if len(results) >= 3:
        sizes_list = list(results.keys())
        times_list = list(results.values())
        
        # Compare time ratios to size ratios
        for i in range(len(sizes_list) - 1):
            size_ratio = sizes_list[i+1] / sizes_list[i]
            time_ratio = times_list[i+1] / times_list[i]
            expected_linear = size_ratio
            expected_quadratic = size_ratio ** 2
            
            if time_ratio > expected_linear * 1.2:
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
