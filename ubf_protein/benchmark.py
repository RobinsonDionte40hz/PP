"""
Performance benchmark for UBF protein system.

Measures key performance metrics:
- Decision latency per move evaluation
- Memory retrieval time
- Agent memory footprint
- Multi-agent throughput
"""

import time
import sys
import tracemalloc
from typing import Dict, List

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


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
