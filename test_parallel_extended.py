"""
Extended Parallel Multi-Agent Performance Test
Tests parallel execution with larger proteins and more iterations.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def run_test(sequence_length, num_agents, num_iterations, test_name):
    """Run a single test configuration."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Generate test sequence
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    test_sequence = (amino_acids * (sequence_length // len(amino_acids) + 1))[:sequence_length]
    
    print(f"\nConfiguration:")
    print(f"  Protein sequence length: {sequence_length} residues")
    print(f"  Number of agents: {num_agents}")
    print(f"  Iterations per agent: {num_iterations}")
    print(f"  Total conformations: {num_agents * num_iterations:,}")
    
    # Create coordinator
    print(f"\n⚙️  Initializing coordinator...")
    coordinator = MultiAgentCoordinator(
        protein_sequence=test_sequence,
        enable_checkpointing=False  # Disable for performance testing
    )
    
    # Initialize agents
    print(f"⚙️  Initializing {num_agents} agents...")
    coordinator.initialize_agents(
        count=num_agents,
        diversity_profile='balanced'
    )
    
    # Run exploration
    print(f"\n🚀 Starting parallel exploration...")
    print(f"{'='*70}")
    
    start_time = time.time()
    results = coordinator.run_parallel_exploration(iterations=num_iterations)
    end_time = time.time()
    
    runtime = end_time - start_time
    
    # Calculate metrics
    total_conformations = num_agents * num_iterations
    time_per_iteration = (runtime / num_iterations) * 1000  # ms
    throughput = total_conformations / runtime
    
    # Estimate sequential time (based on 17ms per iteration from baseline)
    sequential_estimate = (num_iterations * num_agents * 0.017)
    speedup = sequential_estimate / runtime
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {test_name}")
    print(f"{'='*70}")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"  Total runtime: {format_time(runtime)}")
    print(f"  Time per iteration: {time_per_iteration:.2f} ms")
    print(f"  Throughput: {throughput:.0f} conformations/second")
    
    print(f"\n🔍 Exploration Results:")
    print(f"  Conformations explored: {total_conformations:,}")
    print(f"  Best energy: {results.best_energy:.2f} kcal/mol")
    print(f"  Best RMSD: {results.best_rmsd if results.best_rmsd else 'N/A (no native structure)'}")
    
    print(f"\n🤖 Agent Statistics:")
    print(f"  Total agents: {len(coordinator._agents)}")
    
    print(f"\n⚡ Parallelization Efficiency:")
    print(f"  Estimated sequential time: {format_time(sequential_estimate)}")
    print(f"  Actual parallel time: {format_time(runtime)}")
    print(f"  Speedup factor: {speedup:.1f}x")
    
    if speedup >= num_agents * 0.8:
        assessment = "✅ Excellent - Near-perfect scaling!"
    elif speedup >= num_agents * 0.6:
        assessment = "✅ Good - Strong parallelization"
    elif speedup >= num_agents * 0.4:
        assessment = "⚠️  Moderate - Some overhead"
    else:
        assessment = "❌ Poor - High overhead"
    
    print(f"  Assessment: {assessment}")
    
    return {
        'test_name': test_name,
        'sequence_length': sequence_length,
        'num_agents': num_agents,
        'num_iterations': num_iterations,
        'runtime': runtime,
        'throughput': throughput,
        'speedup': speedup,
        'best_energy': results.best_energy
    }

def main():
    """Run extended performance tests."""
    print("="*70)
    print("EXTENDED PARALLEL MULTI-AGENT PERFORMANCE TEST SUITE")
    print("="*70)
    print("\nThis will test parallel execution at scale with:")
    print("  • Larger proteins (50-100 residues)")
    print("  • More iterations (500-1000)")
    print("  • Multiple agent configurations")
    
    all_results = []
    
    # Test 1: Medium protein, extended iterations
    result1 = run_test(
        sequence_length=50,
        num_agents=15,
        num_iterations=500,
        test_name="Medium Protein (50 res) - Extended Run"
    )
    all_results.append(result1)
    
    # Test 2: Large protein, moderate iterations
    result2 = run_test(
        sequence_length=100,
        num_agents=15,
        num_iterations=300,
        test_name="Large Protein (100 res) - Moderate Run"
    )
    all_results.append(result2)
    
    # Test 3: Medium protein, different agent count
    result3 = run_test(
        sequence_length=50,
        num_agents=10,
        num_iterations=500,
        test_name="Medium Protein (50 res) - 10 Agents"
    )
    all_results.append(result3)
    
    # Test 4: Medium protein, max agents
    result4 = run_test(
        sequence_length=50,
        num_agents=20,
        num_iterations=300,
        test_name="Medium Protein (50 res) - 20 Agents"
    )
    all_results.append(result4)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*70}\n")
    
    print(f"{'Test':<40} {'Runtime':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for result in all_results:
        runtime_str = format_time(result['runtime'])
        throughput_str = f"{result['throughput']:.0f} conf/s"
        speedup_str = f"{result['speedup']:.1f}x"
        test_name = result['test_name'][:38]
        
        print(f"{test_name:<40} {runtime_str:<12} {throughput_str:<15} {speedup_str:<10}")
    
    # Calculate averages
    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    avg_throughput = sum(r['throughput'] for r in all_results) / len(all_results)
    total_conformations = sum(r['num_agents'] * r['num_iterations'] for r in all_results)
    total_runtime = sum(r['runtime'] for r in all_results)
    
    print("-" * 70)
    print(f"\n📊 Overall Statistics:")
    print(f"  Total conformations explored: {total_conformations:,}")
    print(f"  Total runtime: {format_time(total_runtime)}")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  Average throughput: {avg_throughput:.0f} conformations/second")
    
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE ✅")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
