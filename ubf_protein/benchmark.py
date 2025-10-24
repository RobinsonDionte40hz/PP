"""
Performance benchmark for UBF protein system.

Measures key performance metrics:
- Decision latency per move evaluation
- Memory retrieval time
- Agent memory footprint
- Multi-agent throughput
- Optional CPython vs PyPy comparison
"""

import time
import sys
import os
import tracemalloc
import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.models import AdaptiveConfig, ProteinSizeClass
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.memory_system import MemorySystem


class PerformanceBenchmark:
    """Performance benchmarking suite for UBF protein system."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.results: Dict[str, float] = {}
    
    def benchmark_move_evaluation_latency(self, iterations: int = 1000) -> float:
        """
        Benchmark move evaluation latency.
        
        Target: < 2ms per evaluation
        
        Args:
            iterations: Number of evaluations to perform
            
        Returns:
            Average latency in milliseconds
        """
        print(f"\n=== Move Evaluation Latency Benchmark ===")
        print(f"Target: < 2ms per evaluation")
        print(f"Running {iterations} iterations...")
        
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=30,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
            adaptive_config=config
        )
        
        # Warm-up
        for _ in range(10):
            agent.explore_step()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            agent.explore_step()
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / iterations
        
        print(f"Total time: {total_time_ms:.2f}ms")
        print(f"Average latency: {avg_latency_ms:.3f}ms per evaluation")
        print(f"Status: {'✅ PASS' if avg_latency_ms < 2.0 else '❌ FAIL'}")
        
        self.results['move_evaluation_latency_ms'] = avg_latency_ms
        return avg_latency_ms
    
    def benchmark_memory_retrieval(self, iterations: int = 10000) -> float:
        """
        Benchmark memory retrieval performance.
        
        Target: < 10μs (0.01ms)
        
        Args:
            iterations: Number of retrievals to perform
            
        Returns:
            Average retrieval time in microseconds
        """
        print(f"\n=== Memory Retrieval Benchmark ===")
        print(f"Target: < 10μs per retrieval")
        print(f"Running {iterations} iterations...")
        
        memory_system = MemorySystem()
        
        # Populate with some memories
        from ubf_protein.models import ConformationalMemory, ConsciousnessCoordinates, BehavioralStateData
        for i in range(50):
            memory = ConformationalMemory(
                memory_id=f"mem_{i}",
                move_type="backbone_rotation",
                significance=0.5 + (i % 5) / 10.0,
                energy_change=-10.0 * (i % 3),
                rmsd_change=-0.5,
                success=True,
                timestamp=1000 + i,
                consciousness_state=ConsciousnessCoordinates(8.0, 0.6, 1000),
                behavioral_state=BehavioralStateData(0.5, 0.6, 0.5, 0.4, 0.6, 0.8, 1000)
            )
            memory_system.store_memory(memory)
        
        # Warm-up
        for _ in range(100):
            memory_system.retrieve_relevant_memories("backbone_rotation")
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            memory_system.retrieve_relevant_memories("backbone_rotation")
        end_time = time.perf_counter()
        
        total_time_us = (end_time - start_time) * 1_000_000
        avg_time_us = total_time_us / iterations
        
        print(f"Total time: {total_time_us:.2f}μs")
        print(f"Average time: {avg_time_us:.3f}μs per retrieval")
        print(f"Status: {'✅ PASS' if avg_time_us < 10.0 else '❌ FAIL'}")
        
        self.results['memory_retrieval_us'] = avg_time_us
        return avg_time_us
    
    def benchmark_agent_memory_footprint(self) -> float:
        """
        Benchmark agent memory footprint.
        
        Target: < 50 MB per agent with 50 memories
        
        Returns:
            Memory footprint in MB
        """
        print(f"\n=== Agent Memory Footprint Benchmark ===")
        print(f"Target: < 50 MB per agent")
        
        tracemalloc.start()
        
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=30,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        snapshot_before = tracemalloc.take_snapshot()
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
            adaptive_config=config
        )
        
        # Fill up memory
        for _ in range(100):
            agent.explore_step()
        
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate difference
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_bytes = sum(stat.size_diff for stat in stats)
        total_mb = total_bytes / (1024 * 1024)
        
        print(f"Memory footprint: {total_mb:.2f} MB")
        print(f"Status: {'✅ PASS' if total_mb < 50.0 else '❌ FAIL'}")
        
        self.results['agent_memory_mb'] = total_mb
        return total_mb
    
    def benchmark_multi_agent_throughput(self, num_agents: int = 10, iterations: int = 100) -> float:
        """
        Benchmark multi-agent exploration throughput.
        
        Target: 100 agents complete 500K conformations in < 2 minutes
        (5000 conformations/second)
        
        Args:
            num_agents: Number of agents
            iterations: Iterations per agent
            
        Returns:
            Conformations per second
        """
        print(f"\n=== Multi-Agent Throughput Benchmark ===")
        print(f"Target: 5000 conformations/second")
        print(f"Running {num_agents} agents for {iterations} iterations each...")
        
        coordinator = MultiAgentCoordinator("ACDEFGHIKLMNPQRSTVWYACDEFGHIKL")
        coordinator.initialize_agents(num_agents, "balanced")
        
        start_time = time.perf_counter()
        results = coordinator.run_parallel_exploration(iterations)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_conformations = results.total_conformations_explored
        throughput = total_conformations / total_time
        
        print(f"Total time: {total_time:.2f}s")
        print(f"Total conformations: {total_conformations}")
        print(f"Throughput: {throughput:.0f} conformations/second")
        print(f"Status: {'✅ PASS' if throughput >= 5000 else '❌ FAIL'}")
        
        self.results['throughput_conf_per_sec'] = throughput
        return throughput
    
    def run_all_benchmarks(self) -> Dict[str, float]:
        """
        Run all performance benchmarks.
        
        Returns:
            Dictionary of benchmark results
        """
        print("=" * 60)
        print("UBF Protein System Performance Benchmark")
        print(f"Python: {sys.version}")
        print("=" * 60)
        
        self.benchmark_move_evaluation_latency()
        self.benchmark_memory_retrieval()
        self.benchmark_agent_memory_footprint()
        self.benchmark_multi_agent_throughput()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for metric, value in self.results.items():
            print(f"{metric}: {value:.3f}")
        
        return self.results


def export_benchmark_results(results: Dict[str, float], output_file: str) -> None:
    """
    Export benchmark results to JSON file.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Path to save JSON results
    """
    output_data = {
        'metadata': {
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
            'platform': platform.platform(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': results,
        'pass_fail': {
            'move_evaluation_latency': results.get('move_evaluation_latency_ms', 999) < 2.0,
            'memory_retrieval': results.get('memory_retrieval_us', 999) < 10.0,
            'agent_memory_footprint': results.get('agent_memory_mb', 999) < 50.0,
            'throughput': results.get('throughput_conf_per_sec', 0) >= 5000
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBenchmark results exported to: {output_file}")


def compare_cpython_pypy() -> Dict[str, Any]:
    """
    Compare performance between CPython and PyPy.
    
    Note: This function detects which implementation is currently running
    and provides comparison guidance.
    
    Returns:
        Dictionary with implementation info and performance notes
    """
    implementation = platform.python_implementation()
    
    comparison = {
        'current_implementation': implementation,
        'python_version': sys.version,
        'speedup_target': 2.0,
        'notes': []
    }
    
    if implementation == 'PyPy':
        comparison['notes'].append("Running on PyPy - JIT compilation enabled")
        comparison['notes'].append("Expected 2x+ speedup over CPython for long-running operations")
        comparison['notes'].append("To compare with CPython, run this script with: python benchmark.py")
    elif implementation == 'CPython':
        comparison['notes'].append("Running on CPython - standard Python interpreter")
        comparison['notes'].append("To compare with PyPy, run this script with: pypy3 benchmark.py")
        comparison['notes'].append("PyPy should show ~2x speedup for computational workloads")
    else:
        comparison['notes'].append(f"Running on {implementation} - unknown implementation")
    
    return comparison


def main():
    """Main entry point for benchmark script with CLI support."""
    parser = argparse.ArgumentParser(
        description='Benchmark UBF protein system performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with default parameters
  python benchmark.py

  # Run with custom parameters
  python benchmark.py --agents 20 --iterations 200

  # Run and export results to JSON
  python benchmark.py --output results.json

  # Show CPython vs PyPy comparison info
  python benchmark.py --compare-cpython

  # Run on PyPy for comparison
  pypy3 benchmark.py --output pypy_results.json
        """
    )

    parser.add_argument('--agents',
                       type=int,
                       default=10,
                       help='Number of agents for throughput test (default: 10)')
    parser.add_argument('--iterations',
                       type=int,
                       default=100,
                       help='Iterations per agent for throughput test (default: 100)')
    parser.add_argument('--output', '-o',
                       help='Output file path for benchmark results (JSON)')
    parser.add_argument('--compare-cpython',
                       action='store_true',
                       help='Show CPython vs PyPy comparison information')
    parser.add_argument('--skip-memory',
                       action='store_true',
                       help='Skip memory footprint test (faster but incomplete)')
    parser.add_argument('--skip-throughput',
                       action='store_true',
                       help='Skip throughput test (faster but incomplete)')

    args = parser.parse_args()

    print("=" * 60)
    print("UBF Protein System Performance Benchmark")
    print(f"Python: {sys.version}")
    print(f"Implementation: {platform.python_implementation()}")
    print(f"Platform: {platform.platform()}")
    print("=" * 60)

    # Show comparison info if requested
    if args.compare_cpython:
        print("\n" + "=" * 60)
        print("CPython vs PyPy Comparison")
        print("=" * 60)
        comparison = compare_cpython_pypy()
        print(f"Current Implementation: {comparison['current_implementation']}")
        print(f"Speedup Target: {comparison['speedup_target']}x")
        print("\nNotes:")
        for note in comparison['notes']:
            print(f"  - {note}")
        print("=" * 60)

    # Run benchmarks
    benchmark = PerformanceBenchmark()
    
    # Always run core benchmarks
    benchmark.benchmark_move_evaluation_latency()
    benchmark.benchmark_memory_retrieval()
    
    # Optional benchmarks
    if not args.skip_memory:
        benchmark.benchmark_agent_memory_footprint()
    else:
        print("\n=== Agent Memory Footprint Benchmark ===")
        print("SKIPPED (--skip-memory flag)")
    
    if not args.skip_throughput:
        benchmark.benchmark_multi_agent_throughput(
            num_agents=args.agents,
            iterations=args.iterations
        )
    else:
        print("\n=== Multi-Agent Throughput Benchmark ===")
        print("SKIPPED (--skip-throughput flag)")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for metric, value in benchmark.results.items():
        print(f"{metric}: {value:.3f}")
    
    # Calculate overall pass/fail
    passes = 0
    total = 0
    
    if 'move_evaluation_latency_ms' in benchmark.results:
        total += 1
        if benchmark.results['move_evaluation_latency_ms'] < 2.0:
            passes += 1
    
    if 'memory_retrieval_us' in benchmark.results:
        total += 1
        if benchmark.results['memory_retrieval_us'] < 10.0:
            passes += 1
    
    if 'agent_memory_mb' in benchmark.results:
        total += 1
        if benchmark.results['agent_memory_mb'] < 50.0:
            passes += 1
    
    if 'throughput_conf_per_sec' in benchmark.results:
        total += 1
        if benchmark.results['throughput_conf_per_sec'] >= 5000:
            passes += 1
    
    print(f"\nOverall: {passes}/{total} benchmarks passed")
    print("=" * 60)

    # Export results if requested
    if args.output:
        export_benchmark_results(benchmark.results, args.output)

    # Exit with appropriate code
    sys.exit(0 if passes == total else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode with arguments
        main()
    else:
        # Legacy mode for backward compatibility
        benchmark = PerformanceBenchmark()
        results = benchmark.run_all_benchmarks()
