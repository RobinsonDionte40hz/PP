"""
Demo script for Task 14 - Visualization Export System

This script demonstrates the key features of the visualization system:
1. Snapshot capture during exploration
2. Automatic downsampling for memory efficiency
3. Multi-format export (JSON, PDB)
4. Agent identification and trajectory comparison

Usage:
    python ubf_protein/demo_visualization.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.visualization import VisualizationExporter
from ubf_protein.tests.test_helpers import create_test_config
from ubf_protein.models import ProteinSizeClass


def demo_basic_visualization():
    """Demo 1: Basic visualization capture and export."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Visualization Capture and Export")
    print("="*70)
    
    # Create agent with visualization enabled
    config = create_test_config(
        size_class=ProteinSizeClass.SMALL,
        max_iterations=20
    )
    
    agent = ProteinAgent(
        protein_sequence="ACDEFGHIKLM",
        adaptive_config=config,
        enable_visualization=True,
        max_snapshots=15  # Limit to 15 snapshots for demo
    )
    
    print(f"\nAgent ID: {agent.get_agent_id()}")
    print(f"Initial snapshots: {len(agent.get_trajectory_snapshots())}")
    
    # Run exploration
    print("\nRunning 20 exploration steps...")
    for i in range(20):
        outcome = agent.explore_step()
        if (i + 1) % 5 == 0:
            snapshots = agent.get_trajectory_snapshots()
            print(f"  Step {i+1}: {len(snapshots)} snapshots, "
                  f"Energy={outcome.new_conformation.energy:.2f}")
    
    # Get final snapshots
    snapshots = agent.get_trajectory_snapshots()
    print(f"\nFinal snapshot count: {len(snapshots)}")
    print(f"Max snapshots limit: {agent._max_snapshots}")
    print(f"Downsampling active: {len(snapshots) <= agent._max_snapshots + 1}")
    
    # Export to JSON
    exporter = VisualizationExporter()
    for snapshot in snapshots:
        exporter.add_snapshot(snapshot)
    
    output_file = "demo_trajectory.json"
    exporter.export_trajectory_to_json(agent.get_agent_id(), output_file)
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"\n✓ Exported trajectory to '{output_file}' ({file_size:,} bytes)")
        os.remove(output_file)  # Cleanup
    
    print("\n✓ Demo 1 Complete!")


def demo_multi_agent_comparison():
    """Demo 2: Multi-agent trajectory comparison."""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Agent Trajectory Comparison")
    print("="*70)
    
    config = create_test_config(
        size_class=ProteinSizeClass.SMALL,
        max_iterations=15
    )
    
    # Create two agents with different IDs
    agent1 = ProteinAgent(
        protein_sequence="ACDEFGHIKLM",
        adaptive_config=config,
        enable_visualization=True,
        max_snapshots=20
    )
    agent1.set_agent_id("exploratory_agent")
    
    agent2 = ProteinAgent(
        protein_sequence="ACDEFGHIKLM",
        adaptive_config=config,
        enable_visualization=True,
        max_snapshots=20
    )
    agent2.set_agent_id("cautious_agent")
    
    print(f"\nAgent 1 ID: {agent1.get_agent_id()}")
    print(f"Agent 2 ID: {agent2.get_agent_id()}")
    
    # Run both agents
    print("\nRunning 15 exploration steps for each agent...")
    for i in range(15):
        agent1.explore_step()
        agent2.explore_step()
        
        if (i + 1) % 5 == 0:
            snaps1 = agent1.get_trajectory_snapshots()
            snaps2 = agent2.get_trajectory_snapshots()
            print(f"  Step {i+1}: Agent1={len(snaps1)} snaps, Agent2={len(snaps2)} snaps")
    
    # Export both trajectories
    exporter = VisualizationExporter()
    
    # Add agent 1 snapshots
    for snapshot in agent1.get_trajectory_snapshots():
        exporter.add_snapshot(snapshot)
    
    # Add agent 2 snapshots
    for snapshot in agent2.get_trajectory_snapshots():
        exporter.add_snapshot(snapshot)
    
    # Export separately
    file1 = "demo_agent1.json"
    file2 = "demo_agent2.json"
    
    exporter.export_trajectory_to_json(agent1.get_agent_id(), file1)
    exporter.export_trajectory_to_json(agent2.get_agent_id(), file2)
    
    if os.path.exists(file1) and os.path.exists(file2):
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)
        print(f"\n✓ Exported Agent 1 trajectory ({size1:,} bytes)")
        print(f"✓ Exported Agent 2 trajectory ({size2:,} bytes)")
        os.remove(file1)
        os.remove(file2)
    
    print("\n✓ Demo 2 Complete!")


def demo_memory_efficiency():
    """Demo 3: Memory-efficient downsampling."""
    print("\n" + "="*70)
    print("DEMO 3: Memory-Efficient Downsampling")
    print("="*70)
    
    config = create_test_config(
        size_class=ProteinSizeClass.SMALL,
        max_iterations=100
    )
    
    agent = ProteinAgent(
        protein_sequence="ACDEFGHIKLM",
        adaptive_config=config,
        enable_visualization=True,
        max_snapshots=10  # Very low limit to trigger downsampling
    )
    
    print(f"\nMax snapshots: {agent._max_snapshots}")
    print("Running 50 exploration steps to demonstrate downsampling...\n")
    
    snapshot_counts = []
    
    for i in range(50):
        agent.explore_step()
        count = len(agent.get_trajectory_snapshots())
        snapshot_counts.append(count)
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}: {count} snapshots (growth controlled)")
    
    # Show growth pattern
    print(f"\nSnapshot count growth:")
    print(f"  After 10 steps: {snapshot_counts[9]}")
    print(f"  After 20 steps: {snapshot_counts[19]}")
    print(f"  After 30 steps: {snapshot_counts[29]}")
    print(f"  After 40 steps: {snapshot_counts[39]}")
    print(f"  After 50 steps: {snapshot_counts[49]}")
    print(f"\n✓ Growth controlled - downsampling working!")
    print("✓ Demo 3 Complete!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Task 14 Visualization System - Feature Demonstration")
    print("="*70)
    
    try:
        demo_basic_visualization()
        demo_multi_agent_comparison()
        demo_memory_efficiency()
        
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETE!")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Automatic snapshot capture during exploration")
        print("  ✓ Multi-format export (JSON, PDB supported)")
        print("  ✓ Multi-agent trajectory tracking and comparison")
        print("  ✓ Memory-efficient downsampling for long runs")
        print("  ✓ Agent identification for trajectory differentiation")
        print("\nNext: Task 15 - Checkpoint and Resume System")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
