"""
Example Usage: BatchExecutor

Demonstrates how to use BatchExecutor for parallel protein validation testing with:
- Parallel execution
- Resource monitoring
- Size-based prioritization  
- Checkpointing and resume
- Progress tracking
- Completion time estimation

Examples include:
1. Basic batch execution
2. Batch with prioritization
3. Monitoring progress during execution
4. Checkpoint and resume
5. Resource monitoring
"""

import sys
import os
import time
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from batch_executor import BatchExecutor


@dataclass
class MockProtein:
    """Mock protein metadata for examples."""
    pdb_id: str
    sequence_length: int
    size_category: str


def mock_test_function(protein):
    """
    Mock test function that simulates protein structure prediction.
    
    Simulates different execution times based on protein size.
    """
    # Simulate work (larger proteins take longer)
    delay = protein.sequence_length / 500.0
    time.sleep(delay)
    
    # Simulate result
    return {
        "pdb_id": protein.pdb_id,
        "rmsd": 2.5 + (protein.sequence_length / 100.0),
        "gdt_ts": 75.0 - (protein.sequence_length / 10.0),
        "tm_score": 0.65,
        "final_energy": -45.0,
        "execution_time": delay
    }


def example_1_basic_execution():
    """Example 1: Basic batch execution."""
    print("\n" + "="*80)
    print("Example 1: Basic Batch Execution")
    print("="*80)
    
    # Create test proteins
    proteins = [
        MockProtein("1UBQ", 76, "small"),
        MockProtein("1CRN", 46, "tiny"),
        MockProtein("2MR9", 35, "tiny"),
        MockProtein("1VII", 36, "tiny"),
        MockProtein("1LYZ", 129, "medium"),
    ]
    
    # Create executor
    executor = BatchExecutor(
        max_parallel=3,
        checkpoint_interval=2,
        checkpoint_dir="example_checkpoints",
        enable_throttling=False
    )
    
    print(f"\n✓ Created BatchExecutor:")
    print(f"  - Max parallel: {executor.max_parallel}")
    print(f"  - Checkpoint interval: {executor.checkpoint_interval}")
    print(f"  - Throttling: {'enabled' if executor.enable_throttling else 'disabled'}")
    
    # Execute batch
    print(f"\n✓ Executing batch of {len(proteins)} proteins...")
    start_time = time.time()
    
    results = executor.execute_batch(
        proteins=proteins,
        test_function=mock_test_function,
        batch_id="example_batch_001",
        prioritize=False
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Batch complete in {elapsed:.2f}s:")
    print(f"  - Total proteins: {len(proteins)}")
    print(f"  - Completed: {len([r for r in results if r is not None])}")
    print(f"  - Failed: {len([r for r in results if r is None])}")
    print(f"  - Average time per protein: {elapsed / len(proteins):.2f}s")


def example_2_prioritized_execution():
    """Example 2: Batch execution with size prioritization."""
    print("\n" + "="*80)
    print("Example 2: Batch with Size Prioritization")
    print("="*80)
    
    proteins = [
        MockProtein("1UBQ", 76, "small"),
        MockProtein("1CRN", 46, "tiny"),
        MockProtein("2MR9", 35, "tiny"),
        MockProtein("1VII", 36, "tiny"),
        MockProtein("1LYZ", 129, "medium"),
        MockProtein("1TIM", 247, "large"),
    ]
    
    print(f"\n✓ Original order:")
    for p in proteins:
        print(f"  - {p.pdb_id}: {p.sequence_length} residues ({p.size_category})")
    
    executor = BatchExecutor(
        max_parallel=3,
        checkpoint_interval=2,
        checkpoint_dir="example_checkpoints",
        enable_throttling=False
    )
    
    # Show prioritized order
    prioritized = executor.prioritize_by_size(proteins)
    print(f"\n✓ Prioritized order (small first):")
    for p in prioritized:
        print(f"  - {p.pdb_id}: {p.sequence_length} residues ({p.size_category})")
    
    # Execute with prioritization
    print(f"\n✓ Executing batch with prioritization...")
    results = executor.execute_batch(
        proteins=proteins,
        test_function=mock_test_function,
        batch_id="example_batch_002",
        prioritize=True
    )
    
    print(f"\n✓ Results returned in original order:")
    for i, result in enumerate(results):
        if result:
            print(f"  - {proteins[i].pdb_id}: RMSD={result['rmsd']:.2f}Å, "
                  f"execution_time={result['execution_time']:.2f}s")


def example_3_progress_monitoring():
    """Example 3: Monitor progress during execution."""
    print("\n" + "="*80)
    print("Example 3: Progress Monitoring")
    print("="*80)
    
    proteins = [
        MockProtein(f"TEST{i:03d}", 50 + i*10, "small")
        for i in range(10)
    ]
    
    executor = BatchExecutor(
        max_parallel=3,
        checkpoint_interval=3,
        checkpoint_dir="example_checkpoints",
        enable_throttling=False
    )
    
    print(f"\n✓ Starting batch of {len(proteins)} proteins...")
    print(f"  (Execute batch in background and check progress)\n")
    
    # In real usage, you'd check progress from another thread
    # Here we just execute and show final progress
    results = executor.execute_batch(
        proteins=proteins,
        test_function=mock_test_function,
        batch_id="example_batch_003",
        prioritize=True
    )
    
    # Get final progress
    progress = executor.get_progress()
    
    print(f"✓ Final Progress:")
    print(f"  - Total: {progress.total_proteins}")
    print(f"  - Completed: {progress.completed}")
    print(f"  - In progress: {progress.in_progress}")
    print(f"  - Pending: {progress.pending}")
    print(f"  - Failed: {progress.failed}")
    print(f"  - Average time/protein: {progress.average_time_per_protein:.2f}s")
    print(f"  - Elapsed time: {progress.elapsed_time:.2f}s")


def example_4_checkpoint_resume():
    """Example 4: Checkpoint and resume functionality."""
    print("\n" + "="*80)
    print("Example 4: Checkpoint and Resume")
    print("="*80)
    
    proteins = [
        MockProtein(f"PROTEIN{i:03d}", 40 + i*5, "small")
        for i in range(6)
    ]
    
    batch_id = "example_batch_checkpoint"
    
    # Execute batch (creates checkpoints)
    print(f"\n✓ Executing batch with checkpointing...")
    executor1 = BatchExecutor(
        max_parallel=2,
        checkpoint_interval=2,  # Checkpoint every 2 proteins
        checkpoint_dir="example_checkpoints",
        enable_throttling=False
    )
    
    results1 = executor1.execute_batch(
        proteins=proteins,
        test_function=mock_test_function,
        batch_id=batch_id,
        prioritize=False
    )
    
    print(f"\n✓ Batch complete. Checkpoints created:")
    checkpoint_file = f"example_checkpoints/{batch_id}_checkpoint.json"
    print(f"  - {checkpoint_file}")
    
    # Show checkpoint content
    import json
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    
    print(f"\n✓ Checkpoint contains:")
    print(f"  - Batch ID: {checkpoint_data['batch_id']}")
    print(f"  - Total proteins: {checkpoint_data['total_proteins']}")
    print(f"  - Completed: {len(checkpoint_data['completed_proteins'])}")
    print(f"  - Pending: {len(checkpoint_data['pending_proteins'])}")


def example_5_resource_monitoring():
    """Example 5: Resource monitoring."""
    print("\n" + "="*80)
    print("Example 5: Resource Monitoring")
    print("="*80)
    
    executor = BatchExecutor(
        max_parallel=3,
        checkpoint_interval=5,
        checkpoint_dir="example_checkpoints",
        cpu_threshold=80.0,
        memory_threshold=80.0,
        enable_throttling=True
    )
    
    print(f"\n✓ Resource monitoring configuration:")
    print(f"  - CPU threshold: {executor.cpu_threshold}%")
    print(f"  - Memory threshold: {executor.memory_threshold}%")
    print(f"  - Throttling: {'enabled' if executor.enable_throttling else 'disabled'}")
    
    # Monitor current resources
    metrics = executor.monitor_resources()
    
    print(f"\n✓ Current system resources:")
    print(f"  - CPU usage: {metrics.cpu_usage_percent:.1f}%")
    print(f"  - Memory usage: {metrics.memory_usage_mb:.1f} MB ({metrics.memory_usage_percent:.1f}%)")
    print(f"  - Disk usage: {metrics.disk_usage_mb:.1f} MB")
    print(f"  - Active processes: {metrics.active_processes}")
    print(f"  - Throttle recommended: {'Yes' if metrics.throttle_recommended else 'No'}")
    
    # Test completion time estimation
    proteins = [MockProtein(f"P{i}", 50, "small") for i in range(3)]
    executor.execute_batch(proteins, mock_test_function, prioritize=False)
    
    estimate = executor.estimate_completion_time(remaining=10)
    if estimate:
        print(f"\n✓ Completion time estimate for 10 remaining proteins:")
        print(f"  - Estimated time: {estimate.total_seconds():.1f}s ({estimate.total_seconds()/60:.1f} minutes)")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BatchExecutor Usage Examples")
    print("="*80)
    
    # Run examples
    example_1_basic_execution()
    example_2_prioritized_execution()
    example_3_progress_monitoring()
    example_4_checkpoint_resume()
    example_5_resource_monitoring()
    
    print("\n" + "="*80)
    print("All Examples Complete!")
    print("="*80)
    print("\nCheck the 'example_checkpoints' directory for checkpoint files.")


if __name__ == "__main__":
    main()
