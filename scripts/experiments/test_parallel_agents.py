"""
Test parallel vs sequential multi-agent performance.

This script demonstrates the speedup from running agents in parallel.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

def test_parallel_performance():
    """Test the parallel multi-agent system."""
    
    # Test with a small protein for quick results
    test_sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues
    num_agents = 15
    iterations = 100
    
    print("=" * 70)
    print("PARALLEL MULTI-AGENT PERFORMANCE TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Protein sequence: {test_sequence}")
    print(f"  Sequence length: {len(test_sequence)} residues")
    print(f"  Number of agents: {num_agents}")
    print(f"  Iterations: {iterations}")
    print(f"  Total conformations: {num_agents * iterations} = {num_agents * iterations:,}")
    
    # Create coordinator with parallel execution
    print(f"\n{'=' * 70}")
    print("Starting parallel exploration...")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    
    coordinator = MultiAgentCoordinator(
        protein_sequence=test_sequence,
        enable_checkpointing=False  # Disable for pure speed test
    )
    
    # Initialize agents
    coordinator.initialize_agents(count=num_agents, diversity_profile='balanced')
    
    # Run parallel exploration
    results = coordinator.run_parallel_exploration(iterations=iterations)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    conformations = results.total_conformations_explored
    time_per_iteration = (total_time / iterations) * 1000  # ms
    throughput = conformations / total_time
    
    # Display results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"\n⏱️  Performance Metrics:")
    print(f"  Total runtime: {total_time:.2f} seconds")
    print(f"  Time per iteration: {time_per_iteration:.2f} ms")
    print(f"  Throughput: {throughput:.0f} conformations/second")
    
    print(f"\n🔍 Exploration Results:")
    print(f"  Conformations explored: {conformations:,}")
    print(f"  Best energy: {results.best_energy:.2f} kcal/mol")
    print(f"  Best RMSD: {results.best_rmsd:.2f} Å" if results.best_rmsd != float('inf') else "  Best RMSD: N/A (no native structure)")
    
    print(f"\n🤖 Agent Statistics:")
    print(f"  Total agents: {len(coordinator.get_agents())}")
    print(f"  Shared memories: {results.shared_memories_created}")
    
    # Estimate speedup
    sequential_time = (num_agents * time_per_iteration * iterations) / 1000  # seconds
    speedup = sequential_time / total_time
    
    print(f"\n⚡ Parallelization Efficiency:")
    print(f"  Estimated sequential time: {sequential_time:.2f} seconds")
    print(f"  Actual parallel time: {total_time:.2f} seconds")
    print(f"  Speedup factor: {speedup:.1f}x")
    
    if speedup > 8:
        print(f"  Assessment: ✅ Excellent - Super-linear speedup!")
    elif speedup > 5:
        print(f"  Assessment: ✅ Excellent - Great parallelization!")
    elif speedup > 3:
        print(f"  Assessment: ✅ Good - Effective parallelization")
    elif speedup > 1.5:
        print(f"  Assessment: ⚠️  Fair - Some parallelization overhead")
    else:
        print(f"  Assessment: ❌ Poor - High overhead")
    
    print(f"\n{'=' * 70}")
    print("TEST COMPLETE ✅")
    print(f"{'=' * 70}\n")
    
    return results

if __name__ == "__main__":
    try:
        results = test_parallel_performance()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
