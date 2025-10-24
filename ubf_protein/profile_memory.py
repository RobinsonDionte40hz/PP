"""
Profiling script to identify bottlenecks in memory retrieval.
"""

import cProfile
import pstats
from io import StringIO

from ubf_protein.memory_system import MemorySystem
from ubf_protein.models import ConformationalMemory, ConsciousnessCoordinates, BehavioralStateData


def profile_memory_retrieval():
    """Profile memory retrieval operations."""
    memory_system = MemorySystem()
    
    # Populate with some memories
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
    
    # Profile retrieval
    def retrieve_many_times():
        for _ in range(10000):
            memory_system.retrieve_relevant_memories("backbone_rotation")
    
    profiler = cProfile.Profile()
    profiler.enable()
    retrieve_many_times()
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(s.getvalue())
    
    # Also sort by time
    print("\n\n=== Sorted by Time ===\n")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('time')
    ps.print_stats(20)
    
    print(s.getvalue())


if __name__ == "__main__":
    profile_memory_retrieval()
