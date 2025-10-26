# UBF Protein System - Usage Examples

Practical examples demonstrating common use cases and advanced features of the UBF Protein System.

## Table of Contents

- [Example 1: Basic Single Agent](#example-1-basic-single-agent)
- [Example 2: Multi-Agent with Diversity](#example-2-multi-agent-with-diversity)
- [Example 3: Checkpoint and Resume](#example-3-checkpoint-and-resume)
- [Example 4: Custom Configuration](#example-4-custom-configuration)
- [Example 5: Memory Analysis](#example-5-memory-analysis)
- [Example 6: Visualization Export](#example-6-visualization-export)
- [Example 7: Performance Benchmarking](#example-7-performance-benchmarking)
- [Example 8: Validation Against Native](#example-8-validation-against-native)
- [Example 9: Real-Time Monitoring](#example-9-real-time-monitoring)
- [Example 10: Advanced Multi-Agent Strategies](#example-10-advanced-multi-agent-strategies)
- [Example 11: QCPP Integration - Basic](#example-11-qcpp-integration---basic) **← NEW**
- [Example 12: QCPP Integration - Performance Tuning](#example-12-qcpp-integration---performance-tuning) **← NEW**
- [Example 13: QCPP Integration - Physics-Grounded Consciousness](#example-13-qcpp-integration---physics-grounded-consciousness) **← NEW**

---

## Example 1: Basic Single Agent

Simple single-agent exploration with default configuration.

```python
#!/usr/bin/env python3
"""
Example 1: Basic single agent exploration.

Demonstrates:
- Creating a protein agent with default configuration
- Running exploration loop
- Monitoring progress
- Extracting results
"""

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import AdaptiveConfig

def main():
    # Protein sequence (20 residue test peptide)
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    
    # Create agent with default configuration
    agent = ProteinAgent(
        protein_sequence=sequence,
        initial_frequency=9.0,    # Balanced frequency
        initial_coherence=0.6     # Balanced coherence
    )
    
    print(f"Exploring conformational space for {len(sequence)}-residue protein")
    print(f"Initial consciousness: f={agent._consciousness.get_frequency():.2f} Hz, "
          f"c={agent._consciousness.get_coherence():.2f}")
    
    # Run exploration
    iterations = 500
    for i in range(iterations):
        outcome = agent.explore_step()
        
        # Print progress every 50 iterations
        if (i + 1) % 50 == 0:
            consciousness = agent._consciousness.get_coordinates()
            print(f"\nIteration {i + 1}/{iterations}")
            print(f"  Best energy: {agent._best_energy:.2f} kcal/mol")
            print(f"  Best RMSD: {agent._best_rmsd:.2f} Å")
            print(f"  Consciousness: f={consciousness.frequency:.2f} Hz, "
                  f"c={consciousness.coherence:.2f}")
            print(f"  Memories created: {agent._memories_created}")
            print(f"  Stuck count: {agent._stuck_in_minima_count}")
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best energy found: {agent._best_energy:.2f} kcal/mol")
    print(f"Best RMSD: {agent._best_rmsd:.2f} Å")
    print(f"Total conformations explored: {agent._conformations_explored}")
    print(f"Significant memories created: {agent._memories_created}")
    print(f"Times stuck in local minima: {agent._stuck_in_minima_count}")
    print(f"Successful escapes: {agent._successful_escapes}")
    
    # Get final consciousness state
    final_consciousness = agent._consciousness.get_coordinates()
    print(f"\nFinal consciousness state:")
    print(f"  Frequency: {final_consciousness.frequency:.2f} Hz")
    print(f"  Coherence: {final_consciousness.coherence:.2f}")
    
    # Get final behavioral state
    behavioral = agent._behavioral.get_behavioral_data()
    print(f"\nFinal behavioral state:")
    print(f"  Exploration energy: {behavioral.exploration_energy:.2f}")
    print(f"  Structural focus: {behavioral.structural_focus:.2f}")
    print(f"  Hydrophobic drive: {behavioral.hydrophobic_drive:.2f}")
    print(f"  Risk tolerance: {behavioral.risk_tolerance:.2f}")
    print(f"  Native state ambition: {behavioral.native_state_ambition:.2f}")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
Exploring conformational space for 20-residue protein
Initial consciousness: f=9.00 Hz, c=0.60

Iteration 50/500
  Best energy: -45.23 kcal/mol
  Best RMSD: 2.8 Å
  Consciousness: f=9.30 Hz, c=0.62
  Memories created: 12
  Stuck count: 0

...

FINAL RESULTS
============================================================
Best energy found: -78.45 kcal/mol
Best RMSD: 1.2 Å
Total conformations explored: 500
Significant memories created: 45
Times stuck in local minima: 3
Successful escapes: 2
```

---

## Example 2: Multi-Agent with Diversity

Multi-agent system with diverse consciousness profiles for parallel exploration.

```python
#!/usr/bin/env python3
"""
Example 2: Multi-agent exploration with diversity.

Demonstrates:
- Initializing multiple agents with diverse profiles
- Parallel exploration
- Shared memory collective learning
- Results aggregation
"""

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

def main():
    # Protein sequence
    sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGH"
    
    # Initialize coordinator
    print(f"Initializing multi-agent system for {len(sequence)}-residue protein")
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence
    )
    
    # Initialize agents with balanced diversity
    agent_count = 20
    coordinator.initialize_agents(
        count=agent_count,
        diversity_profile="balanced"  # 33% cautious, 34% balanced, 33% aggressive
    )
    
    print(f"\nInitialized {agent_count} agents with diversity profile: balanced")
    print(f"  Cautious agents (low freq, high coh): ~7 agents")
    print(f"  Balanced agents (mid freq, mid coh): ~7 agents")
    print(f"  Aggressive agents (high freq, low coh): ~6 agents")
    
    # Run parallel exploration
    iterations = 500
    print(f"\nRunning {iterations} iterations of parallel exploration...")
    
    results = coordinator.run_parallel_exploration(iterations=iterations)
    
    # Display results
    print(f"\n{'='*60}")
    print("MULTI-AGENT RESULTS")
    print(f"{'='*60}")
    print(f"Best energy found: {results.best_energy:.2f} kcal/mol")
    print(f"Best RMSD: {results.best_rmsd:.2f} Å")
    print(f"Best agent ID: {results.best_agent_id}")
    print(f"\nCollective learning benefit: {results.collective_learning_benefit:.2%}")
    print(f"Total conformations explored: {results.total_conformations_explored}")
    print(f"Total memories created: {results.total_memories_created}")
    print(f"Shared memories created: {results.shared_memories_created}")
    
    # Analyze per-agent performance
    print(f"\n{'='*60}")
    print("PER-AGENT ANALYSIS")
    print(f"{'='*60}")
    
    best_agents = sorted(
        coordinator._agents,
        key=lambda a: a._best_energy
    )[:5]
    
    print("\nTop 5 agents by energy:")
    for i, agent in enumerate(best_agents, 1):
        consciousness = agent._consciousness.get_coordinates()
        print(f"\n{i}. Agent {agent.get_agent_id()}")
        print(f"   Energy: {agent._best_energy:.2f} kcal/mol")
        print(f"   RMSD: {agent._best_rmsd:.2f} Å")
        print(f"   Consciousness: f={consciousness.frequency:.2f} Hz, "
              f"c={consciousness.coherence:.2f}")
        print(f"   Memories: {agent._memories_created}")
        print(f"   Stuck/Escaped: {agent._stuck_in_minima_count}/{agent._successful_escapes}")
    
    # Analyze diversity impact
    print(f"\n{'='*60}")
    print("DIVERSITY ANALYSIS")
    print(f"{'='*60}")
    
    cautious_agents = [a for a in coordinator._agents 
                       if a._consciousness.get_frequency() < 7.0]
    balanced_agents = [a for a in coordinator._agents 
                       if 7.0 <= a._consciousness.get_frequency() < 11.0]
    aggressive_agents = [a for a in coordinator._agents 
                         if a._consciousness.get_frequency() >= 11.0]
    
    print(f"\nCautious agents: {len(cautious_agents)}")
    if cautious_agents:
        avg_energy = sum(a._best_energy for a in cautious_agents) / len(cautious_agents)
        print(f"  Avg best energy: {avg_energy:.2f} kcal/mol")
    
    print(f"\nBalanced agents: {len(balanced_agents)}")
    if balanced_agents:
        avg_energy = sum(a._best_energy for a in balanced_agents) / len(balanced_agents)
        print(f"  Avg best energy: {avg_energy:.2f} kcal/mol")
    
    print(f"\nAggressive agents: {len(aggressive_agents)}")
    if aggressive_agents:
        avg_energy = sum(a._best_energy for a in aggressive_agents) / len(aggressive_agents)
        print(f"  Avg best energy: {avg_energy:.2f} kcal/mol")

if __name__ == "__main__":
    main()
```

---

## Example 3: Checkpoint and Resume

Save exploration state and resume from checkpoint.

```python
#!/usr/bin/env python3
"""
Example 3: Checkpoint and resume exploration.

Demonstrates:
- Enabling checkpointing
- Setting auto-save interval
- Manual checkpoint saving
- Resuming from checkpoint
- State preservation validation
"""

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
import os

def main():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    checkpoint_dir = "./example_checkpoints"
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("="*60)
    print("PHASE 1: Initial Exploration with Checkpointing")
    print("="*60)
    
    # Create coordinator with checkpointing enabled
    coordinator1 = MultiAgentCoordinator(
        protein_sequence=sequence,
        enable_checkpointing=True,
        checkpoint_dir=checkpoint_dir
    )
    
    # Set auto-save every 50 iterations
    coordinator1._checkpoint_manager.set_auto_save_interval(50)
    print(f"Auto-save enabled: checkpoint every 50 iterations")
    
    # Initialize agents
    coordinator1.initialize_agents(count=10, diversity_profile="balanced")
    print(f"Initialized 10 agents")
    
    # Run first phase
    print(f"\nRunning Phase 1: 200 iterations...")
    results1 = coordinator1.run_parallel_exploration(iterations=200)
    
    print(f"\nPhase 1 Results:")
    print(f"  Best energy: {results1.best_energy:.2f} kcal/mol")
    print(f"  Best RMSD: {results1.best_rmsd:.2f} Å")
    print(f"  Conformations explored: {results1.total_conformations_explored}")
    
    # Manual checkpoint save
    print(f"\nSaving manual checkpoint...")
    checkpoint_file = coordinator1.save_checkpoint(checkpoint_name="phase1_complete")
    print(f"Checkpoint saved: {checkpoint_file}")
    
    # Get agent state before resuming
    agent1_energy = coordinator1._agents[0]._best_energy
    agent1_iterations = coordinator1._agents[0]._iterations_completed
    
    print(f"\nAgent 0 state before resume:")
    print(f"  Iterations completed: {agent1_iterations}")
    print(f"  Best energy: {agent1_energy:.2f} kcal/mol")
    
    # Simulate restart - create new coordinator
    print(f"\n{'='*60}")
    print("PHASE 2: Resume from Checkpoint")
    print(f"{'='*60}")
    
    coordinator2 = MultiAgentCoordinator(
        protein_sequence=sequence,
        enable_checkpointing=True,
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume from checkpoint
    print(f"Resuming from checkpoint...")
    coordinator2.resume_from_checkpoint(checkpoint_file)
    
    # Verify state restoration
    agent2_energy = coordinator2._agents[0]._best_energy
    agent2_iterations = coordinator2._agents[0]._iterations_completed
    
    print(f"\nAgent 0 state after resume:")
    print(f"  Iterations completed: {agent2_iterations}")
    print(f"  Best energy: {agent2_energy:.2f} kcal/mol")
    print(f"\nState preservation verified: {agent1_iterations == agent2_iterations and agent1_energy == agent2_energy}")
    
    # Continue exploration
    print(f"\nRunning Phase 2: additional 200 iterations...")
    results2 = coordinator2.run_parallel_exploration(iterations=200)
    
    print(f"\nPhase 2 Results:")
    print(f"  Best energy: {results2.best_energy:.2f} kcal/mol")
    print(f"  Best RMSD: {results2.best_rmsd:.2f} Å")
    print(f"  Conformations explored: {results2.total_conformations_explored}")
    
    # List all checkpoints
    print(f"\n{'='*60}")
    print("CHECKPOINT HISTORY")
    print(f"{'='*60}")
    
    checkpoints = coordinator2._checkpoint_manager.list_checkpoints()
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  {cp['name']} (iteration {cp['iteration']})")

if __name__ == "__main__":
    main()
```

---

## Example 4: Custom Configuration

Create custom configuration for specific protein sizes and exploration strategies.

```python
#!/usr/bin/env python3
"""
Example 4: Custom adaptive configuration.

Demonstrates:
- Creating size-specific configurations
- Customizing exploration parameters
- Adaptive threshold scaling
- Configuration for different protein types
"""

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import AdaptiveConfig, ProteinSizeClass

def create_aggressive_config(sequence: str) -> AdaptiveConfig:
    """Create aggressive exploration configuration."""
    residue_count = len(sequence)
    
    # Classify size
    if residue_count < 50:
        size_class = ProteinSizeClass.SMALL
    elif residue_count <= 150:
        size_class = ProteinSizeClass.MEDIUM
    else:
        size_class = ProteinSizeClass.LARGE
    
    return AdaptiveConfig(
        size_class=size_class,
        residue_count=residue_count,
        initial_frequency_range=(10.0, 15.0),  # High frequency = aggressive
        initial_coherence_range=(0.2, 0.4),     # Low coherence = exploratory
        stuck_window=15,                         # Shorter stuck window
        stuck_threshold=3.0,                     # Lower stuck threshold
        max_memories=75,                         # More memories
        convergence_energy_threshold=20.0,       # Relaxed convergence
        convergence_rmsd_threshold=3.0,
        checkpoint_interval=50,
        max_iterations=3000
    )

def create_cautious_config(sequence: str) -> AdaptiveConfig:
    """Create cautious exploration configuration."""
    residue_count = len(sequence)
    
    if residue_count < 50:
        size_class = ProteinSizeClass.SMALL
    elif residue_count <= 150:
        size_class = ProteinSizeClass.MEDIUM
    else:
        size_class = ProteinSizeClass.LARGE
    
    return AdaptiveConfig(
        size_class=size_class,
        residue_count=residue_count,
        initial_frequency_range=(3.0, 7.0),    # Low frequency = cautious
        initial_coherence_range=(0.7, 1.0),     # High coherence = focused
        stuck_window=30,                         # Longer stuck window
        stuck_threshold=10.0,                    # Higher stuck threshold
        max_memories=30,                         # Fewer memories
        convergence_energy_threshold=5.0,        # Strict convergence
        convergence_rmsd_threshold=1.0,
        checkpoint_interval=100,
        max_iterations=5000
    )

def main():
    # Test sequence (medium size)
    sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN"
    
    print("="*60)
    print("AGGRESSIVE CONFIGURATION")
    print("="*60)
    
    # Create agent with aggressive config
    aggressive_config = create_aggressive_config(sequence)
    agent_aggressive = ProteinAgent(
        protein_sequence=sequence,
        adaptive_config=aggressive_config
    )
    
    print(f"\nConfiguration:")
    print(f"  Frequency range: {aggressive_config.initial_frequency_range}")
    print(f"  Coherence range: {aggressive_config.initial_coherence_range}")
    print(f"  Stuck window: {aggressive_config.stuck_window}")
    print(f"  Stuck threshold: {aggressive_config.stuck_threshold}")
    print(f"  Max memories: {aggressive_config.max_memories}")
    
    # Run exploration
    print(f"\nRunning 300 iterations...")
    for i in range(300):
        agent_aggressive.explore_step()
    
    print(f"\nResults:")
    print(f"  Best energy: {agent_aggressive._best_energy:.2f} kcal/mol")
    print(f"  Conformations explored: {agent_aggressive._conformations_explored}")
    print(f"  Stuck count: {agent_aggressive._stuck_in_minima_count}")
    
    # Compare with cautious configuration
    print(f"\n{'='*60}")
    print("CAUTIOUS CONFIGURATION")
    print(f"{'='*60}")
    
    cautious_config = create_cautious_config(sequence)
    agent_cautious = ProteinAgent(
        protein_sequence=sequence,
        adaptive_config=cautious_config
    )
    
    print(f"\nConfiguration:")
    print(f"  Frequency range: {cautious_config.initial_frequency_range}")
    print(f"  Coherence range: {cautious_config.initial_coherence_range}")
    print(f"  Stuck window: {cautious_config.stuck_window}")
    print(f"  Stuck threshold: {cautious_config.stuck_threshold}")
    print(f"  Max memories: {cautious_config.max_memories}")
    
    # Run exploration
    print(f"\nRunning 300 iterations...")
    for i in range(300):
        agent_cautious.explore_step()
    
    print(f"\nResults:")
    print(f"  Best energy: {agent_cautious._best_energy:.2f} kcal/mol")
    print(f"  Conformations explored: {agent_cautious._conformations_explored}")
    print(f"  Stuck count: {agent_cautious._stuck_in_minima_count}")
    
    # Compare strategies
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"\nAggressive strategy:")
    print(f"  Final energy: {agent_aggressive._best_energy:.2f} kcal/mol")
    print(f"  Exploration speed: {agent_aggressive._conformations_explored / 300:.2f} conf/iter")
    print(f"  Stuck frequency: {agent_aggressive._stuck_in_minima_count / 300:.2%}")
    
    print(f"\nCautious strategy:")
    print(f"  Final energy: {agent_cautious._best_energy:.2f} kcal/mol")
    print(f"  Exploration speed: {agent_cautious._conformations_explored / 300:.2f} conf/iter")
    print(f"  Stuck frequency: {agent_cautious._stuck_in_minima_count / 300:.2%}")

if __name__ == "__main__":
    main()
```

---

## Example 5: Memory Analysis

Analyze agent memory system to understand learning patterns.

```python
#!/usr/bin/env python3
"""
Example 5: Memory system analysis.

Demonstrates:
- Memory creation and storage
- Memory significance calculation
- Memory influence on move selection
- Memory pruning behavior
- Shared memory analysis
"""

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from collections import defaultdict

def analyze_agent_memories(agent):
    """Analyze individual agent's memory system."""
    memory_system = agent.get_memory_system()
    
    print(f"\nAgent {agent.get_agent_id()} Memory Analysis:")
    print(f"{'='*50}")
    
    # Count memories by move type
    move_type_counts = defaultdict(int)
    total_memories = 0
    
    for move_type, memories in memory_system._memories.items():
        count = len(memories)
        move_type_counts[move_type] = count
        total_memories += count
    
    print(f"\nTotal memories: {total_memories}")
    print(f"Memory count: {memory_system._memory_count}")
    
    # Show memory distribution by move type
    print(f"\nMemory distribution by move type:")
    for move_type in sorted(move_type_counts.keys(), key=lambda k: move_type_counts[k], reverse=True):
        count = move_type_counts[move_type]
        percentage = (count / total_memories * 100) if total_memories > 0 else 0
        print(f"  {move_type:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Analyze significance distribution
    all_memories = []
    for memories in memory_system._memories.values():
        all_memories.extend(memories)
    
    if all_memories:
        significances = [m.significance for m in all_memories]
        avg_significance = sum(significances) / len(significances)
        max_significance = max(significances)
        min_significance = min(significances)
        
        print(f"\nSignificance statistics:")
        print(f"  Average: {avg_significance:.3f}")
        print(f"  Maximum: {max_significance:.3f}")
        print(f"  Minimum: {min_significance:.3f}")
        
        # Histogram of significance ranges
        high_sig = sum(1 for s in significances if s >= 0.7)
        med_sig = sum(1 for s in significances if 0.5 <= s < 0.7)
        low_sig = sum(1 for s in significances if s < 0.5)
        
        print(f"\nSignificance distribution:")
        print(f"  High (≥0.7): {high_sig} ({high_sig/len(significances)*100:.1f}%)")
        print(f"  Medium (0.5-0.7): {med_sig} ({med_sig/len(significances)*100:.1f}%)")
        print(f"  Low (<0.5): {low_sig} ({low_sig/len(significances)*100:.1f}%)")
        
        # Show top 5 most significant memories
        top_memories = sorted(all_memories, key=lambda m: m.significance, reverse=True)[:5]
        print(f"\nTop 5 most significant memories:")
        for i, memory in enumerate(top_memories, 1):
            print(f"\n  {i}. {memory.move_type}")
            print(f"     Significance: {memory.significance:.3f}")
            print(f"     Energy change: {memory.energy_change:.2f} kcal/mol")
            print(f"     RMSD change: {memory.rmsd_change:.2f} Å")
            print(f"     Success: {memory.success}")

def analyze_shared_memory_pool(coordinator):
    """Analyze shared memory pool."""
    shared_pool = coordinator.get_shared_memory_pool()
    
    print(f"\n{'='*60}")
    print("SHARED MEMORY POOL ANALYSIS")
    print(f"{'='*60}")
    
    total_shared = shared_pool._memory_count
    print(f"\nTotal shared memories: {total_shared}")
    
    if total_shared > 0:
        # Count by move type
        move_type_counts = defaultdict(int)
        for memory in shared_pool._shared_memories:
            move_type_counts[memory.move_type] += 1
        
        print(f"\nShared memory distribution:")
        for move_type in sorted(move_type_counts.keys(), key=lambda k: move_type_counts[k], reverse=True):
            count = move_type_counts[move_type]
            percentage = (count / total_shared * 100)
            print(f"  {move_type:20s}: {count:3d} ({percentage:5.1f}%)")
        
        # Analyze significance
        significances = [m.significance for m in shared_pool._shared_memories]
        avg_sig = sum(significances) / len(significances)
        print(f"\nAverage significance: {avg_sig:.3f}")
        print(f"(Note: All shared memories have significance ≥ 0.7)")

def main():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    
    print("="*60)
    print("MEMORY SYSTEM ANALYSIS")
    print("="*60)
    
    # Create multi-agent system
    coordinator = MultiAgentCoordinator(protein_sequence=sequence)
    coordinator.initialize_agents(count=5, diversity_profile="balanced")
    
    # Run exploration
    print(f"\nRunning 200 iterations...")
    coordinator.run_parallel_exploration(iterations=200)
    
    # Analyze each agent's memories
    for agent in coordinator._agents[:3]:  # Analyze first 3 agents
        analyze_agent_memories(agent)
    
    # Analyze shared memory pool
    analyze_shared_memory_pool(coordinator)
    
    # Test memory influence
    print(f"\n{'='*60}")
    print("MEMORY INFLUENCE TESTING")
    print(f"{'='*60}")
    
    test_agent = coordinator._agents[0]
    memory_system = test_agent.get_memory_system()
    
    # Test different move types
    test_move_types = ["helix_formation", "hydrophobic_collapse", "backbone_rotation"]
    
    for move_type in test_move_types:
        memories = memory_system.retrieve_relevant_memories(move_type, max_count=10)
        if memories:
            influence = memory_system.calculate_memory_influence(memories)
            print(f"\n{move_type}:")
            print(f"  Relevant memories: {len(memories)}")
            print(f"  Memory influence: {influence:.3f}")
            print(f"  Interpretation: {'Promising' if influence > 1.0 else 'Risky' if influence < 1.0 else 'Neutral'}")

if __name__ == "__main__":
    main()
```

---

## Example 6: Visualization Export

Export trajectories and energy landscapes for analysis and visualization.

```python
#!/usr/bin/env python3
"""
Example 6: Visualization export.

Demonstrates:
- Enabling trajectory recording
- Exporting trajectory to JSON
- Exporting energy landscape (2D projection)
- Multiple export formats
- Real-time streaming
"""

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.visualization import VisualizationExporter
import os

def main():
    sequence = "ACDEFGH"
    output_dir = "./example_viz"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("VISUALIZATION EXPORT EXAMPLE")
    print("="*60)
    
    # Create agent with visualization enabled
    print(f"\nCreating agent with visualization enabled...")
    agent = ProteinAgent(
        protein_sequence=sequence,
        enable_visualization=True,
        max_snapshots=500  # Limit snapshots to prevent memory overflow
    )
    
    print(f"Max snapshots: 500")
    print(f"Snapshot frequency: every iteration")
    
    # Run exploration
    iterations = 300
    print(f"\nRunning {iterations} iterations with trajectory recording...")
    
    for i in range(iterations):
        agent.explore_step()
        
        if (i + 1) % 50 == 0:
            print(f"  Iteration {i + 1}/{iterations} - "
                  f"Snapshots recorded: {len(agent.get_trajectory_snapshots())}")
    
    # Get trajectory snapshots
    snapshots = agent.get_trajectory_snapshots()
    print(f"\nTotal snapshots recorded: {len(snapshots)}")
    
    # Create visualization exporter
    exporter = VisualizationExporter(output_dir=output_dir)
    
    # Export trajectory to JSON
    print(f"\nExporting trajectory to JSON...")
    json_file = os.path.join(output_dir, "trajectory.json")
    exporter.export_trajectory_to_json(
        snapshots=snapshots,
        output_file=json_file
    )
    print(f"  Exported: {json_file}")
    
    # Export energy landscape (PCA projection)
    print(f"\nExporting energy landscape (PCA projection)...")
    pca_file = os.path.join(output_dir, "energy_landscape_pca.csv")
    exporter.export_energy_landscape(
        snapshots=snapshots,
        output_file=pca_file,
        projection_method="pca"
    )
    print(f"  Exported: {pca_file}")
    
    # Export energy landscape (t-SNE projection)
    print(f"\nExporting energy landscape (t-SNE projection)...")
    tsne_file = os.path.join(output_dir, "energy_landscape_tsne.csv")
    exporter.export_energy_landscape(
        snapshots=snapshots,
        output_file=tsne_file,
        projection_method="tsne"
    )
    print(f"  Exported: {tsne_file}")
    
    # Analyze trajectory
    print(f"\n{'='*60}")
    print("TRAJECTORY ANALYSIS")
    print(f"{'='*60}")
    
    energies = [s.energy for s in snapshots]
    rmsds = [s.rmsd for s in snapshots]
    
    print(f"\nEnergy progression:")
    print(f"  Initial: {energies[0]:.2f} kcal/mol")
    print(f"  Final: {energies[-1]:.2f} kcal/mol")
    print(f"  Best: {min(energies):.2f} kcal/mol")
    print(f"  Improvement: {energies[0] - min(energies):.2f} kcal/mol")
    
    print(f"\nRMSD progression:")
    print(f"  Initial: {rmsds[0]:.2f} Å")
    print(f"  Final: {rmsds[-1]:.2f} Å")
    print(f"  Best: {min(rmsds):.2f} Å")
    print(f"  Improvement: {rmsds[0] - min(rmsds):.2f} Å")
    
    # Consciousness trajectory
    frequencies = [s.consciousness_coords.frequency for s in snapshots]
    coherences = [s.consciousness_coords.coherence for s in snapshots]
    
    print(f"\nConsciousness trajectory:")
    print(f"  Frequency range: {min(frequencies):.2f} - {max(frequencies):.2f} Hz")
    print(f"  Coherence range: {min(coherences):.2f} - {max(coherences):.2f}")
    print(f"  Final state: f={frequencies[-1]:.2f} Hz, c={coherences[-1]:.2f}")
    
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"\nAll files saved to: {output_dir}")
    print(f"\nTo visualize:")
    print(f"  1. Load {json_file} for full trajectory")
    print(f"  2. Plot {pca_file} for energy landscape (PCA)")
    print(f"  3. Plot {tsne_file} for energy landscape (t-SNE)")

if __name__ == "__main__":
    main()
```

---

## Example 7: Performance Benchmarking

Benchmark system performance and compare PyPy vs CPython.

```python
#!/usr/bin/env python3
"""
Example 7: Performance benchmarking.

Demonstrates:
- Move evaluation latency
- Memory retrieval performance
- Agent throughput
- Memory footprint
- PyPy vs CPython comparison
"""

import time
import tracemalloc
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.models import AdaptiveConfig, ProteinSizeClass

def benchmark_move_evaluation(iterations=1000):
    """Benchmark move evaluation latency."""
    print("\n" + "="*60)
    print("MOVE EVALUATION LATENCY BENCHMARK")
    print("="*60)
    print(f"Target: < 2ms per evaluation")
    print(f"Running {iterations} iterations...\n")
    
    agent = ProteinAgent(
        protein_sequence="ACDEFGHIKLMNPQRSTVWY",
        adaptive_config=AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=20,
            initial_frequency_range=(9.0, 9.0),
            initial_coherence_range=(0.6, 0.6)
        )
    )
    
    # Warmup (JIT compilation)
    print("Warming up JIT compiler...")
    for _ in range(100):
        agent.explore_step()
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        agent.explore_step()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000  # ms
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average latency: {avg_latency:.3f} ms/evaluation")
    print(f"  Status: {'PASS ✓' if avg_latency < 2.0 else 'FAIL ✗'}")
    
    return avg_latency

def benchmark_memory_retrieval(iterations=10000):
    """Benchmark memory retrieval performance."""
    print("\n" + "="*60)
    print("MEMORY RETRIEVAL BENCHMARK")
    print("="*60)
    print(f"Target: < 10μs per retrieval")
    print(f"Running {iterations} retrievals...\n")
    
    agent = ProteinAgent(protein_sequence="ACDEFGHIKLMNPQRSTVWY")
    
    # Create some memories
    for _ in range(50):
        agent.explore_step()
    
    memory_system = agent.get_memory_system()
    
    # Benchmark retrieval
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        memory_system.retrieve_relevant_memories("helix_formation", max_count=10)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1_000_000  # μs
    
    print(f"Results:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average latency: {avg_latency:.3f} μs/retrieval")
    print(f"  Status: {'PASS ✓' if avg_latency < 10.0 else 'FAIL ✗'}")
    
    return avg_latency

def benchmark_agent_throughput(agents=100, iterations=5000):
    """Benchmark multi-agent throughput."""
    print("\n" + "="*60)
    print("MULTI-AGENT THROUGHPUT BENCHMARK")
    print("="*60)
    print(f"Target: {agents} agents × {iterations} conformations < 2 minutes")
    print(f"Total conformations: {agents * iterations:,}\n")
    
    coordinator = MultiAgentCoordinator(protein_sequence="ACDEFGH")
    coordinator.initialize_agents(count=agents, diversity_profile="balanced")
    
    print(f"Running {agents} agents for {iterations} iterations...")
    start_time = time.perf_counter()
    
    coordinator.run_parallel_exploration(iterations=iterations)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_conformations = agents * iterations
    throughput = total_conformations / total_time
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Total conformations: {total_conformations:,}")
    print(f"  Throughput: {throughput:.0f} conformations/second")
    print(f"  Status: {'PASS ✓' if total_time < 120 else 'FAIL ✗'}")
    
    return total_time, throughput

def benchmark_memory_footprint():
    """Benchmark agent memory footprint."""
    print("\n" + "="*60)
    print("MEMORY FOOTPRINT BENCHMARK")
    print("="*60)
    print(f"Target: < 50 MB per agent with 50 memories\n")
    
    tracemalloc.start()
    
    # Create agent and generate memories
    agent = ProteinAgent(protein_sequence="ACDEFGHIKLMNPQRSTVWY")
    
    # Run until we have ~50 memories
    while agent._memories_created < 50:
        agent.explore_step()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    current_mb = current / 1024 / 1024
    peak_mb = peak / 1024 / 1024
    
    print(f"Results:")
    print(f"  Current memory: {current_mb:.2f} MB")
    print(f"  Peak memory: {peak_mb:.2f} MB")
    print(f"  Memories stored: {agent._memories_created}")
    print(f"  Status: {'PASS ✓' if peak_mb < 50 else 'FAIL ✗'}")
    
    return peak_mb

def main():
    print("="*60)
    print("UBF PROTEIN SYSTEM - PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Run benchmarks
    move_latency = benchmark_move_evaluation(iterations=1000)
    memory_latency = benchmark_memory_retrieval(iterations=10000)
    throughput_time, throughput = benchmark_agent_throughput(agents=20, iterations=1000)
    memory_footprint = benchmark_memory_footprint()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\nMove Evaluation Latency:")
    print(f"  Measured: {move_latency:.3f} ms")
    print(f"  Target: < 2.0 ms")
    print(f"  Status: {'PASS ✓' if move_latency < 2.0 else 'FAIL ✗'}")
    
    print(f"\nMemory Retrieval Latency:")
    print(f"  Measured: {memory_latency:.3f} μs")
    print(f"  Target: < 10.0 μs")
    print(f"  Status: {'PASS ✓' if memory_latency < 10.0 else 'FAIL ✗'}")
    
    print(f"\nMulti-Agent Throughput:")
    print(f"  Measured: {throughput_time:.2f} seconds")
    print(f"  Target: < 120 seconds")
    print(f"  Throughput: {throughput:.0f} conf/sec")
    print(f"  Status: {'PASS ✓' if throughput_time < 120 else 'FAIL ✗'}")
    
    print(f"\nMemory Footprint:")
    print(f"  Measured: {memory_footprint:.2f} MB")
    print(f"  Target: < 50 MB")
    print(f"  Status: {'PASS ✓' if memory_footprint < 50 else 'FAIL ✗'}")
    
    # Overall result
    all_pass = (move_latency < 2.0 and memory_latency < 10.0 and 
                throughput_time < 120 and memory_footprint < 50)
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL TESTS PASS ✓' if all_pass else 'SOME TESTS FAIL ✗'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

---

## Example 11: QCPP Integration - Basic

Basic integration with Quantum Coherence Protein Predictor for physics-based validation.

```python
#!/usr/bin/env python3
"""
Example 11: QCPP Integration - Basic Usage

Demonstrates:
- Setting up QCPP integration
- Running exploration with real-time physics feedback
- Monitoring QCPP metrics alongside UBF exploration
- Analyzing physics-grounded results
"""

from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
from ubf_protein.qcpp_config import QCPPIntegrationConfig
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from protein_predictor import QuantumCoherenceProteinPredictor
import time

def main():
    # Protein sequence (ubiquitin N-terminal fragment)
    sequence = "MQIFVKTLTG"  # 10 residues
    
    print("="*70)
    print("EXAMPLE 11: QCPP INTEGRATION - BASIC USAGE")
    print("="*70)
    print(f"\nProtein: {sequence} ({len(sequence)} residues)")
    
    # Step 1: Initialize QCPP predictor
    print("\n[1] Initializing QCPP predictor...")
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    
    # Step 2: Configure integration (default: 10% analysis frequency)
    print("[2] Configuring integration...")
    qcpp_config = QCPPIntegrationConfig.default()
    print(f"    Analysis frequency: {qcpp_config.analysis_frequency*100:.0f}%")
    print(f"    Cache size: {qcpp_config.cache_size}")
    print(f"    Adaptive frequency: {qcpp_config.enable_adaptive_frequency}")
    
    # Step 3: Create QCPP adapter
    print("[3] Creating QCPP adapter...")
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, qcpp_config)
    
    # Step 4: Create coordinator with QCPP integration
    print("[4] Creating multi-agent coordinator...")
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        qcpp_adapter=qcpp_adapter  # Enable QCPP integration
    )
    
    # Step 5: Initialize agents
    num_agents = 5
    iterations = 200
    print(f"[5] Initializing {num_agents} agents...")
    coordinator.initialize_agents(count=num_agents, diversity_profile="balanced")
    
    # Step 6: Run exploration with QCPP feedback
    print(f"[6] Running {iterations} iterations with QCPP feedback...")
    print("    (QCPP provides real-time stability predictions)")
    
    start_time = time.perf_counter()
    coordinator.run_parallel_exploration(iterations=iterations)
    elapsed = time.perf_counter() - start_time
    
    # Step 7: Analyze results
    print(f"\n[7] Exploration complete in {elapsed:.2f}s")
    
    # Get best result
    best_agent = min(coordinator._agents, key=lambda a: a._current_energy)
    print(f"\nBest Result:")
    print(f"  Energy: {best_agent._current_energy:.3f} kcal/mol")
    print(f"  RMSD: {best_agent._current_rmsd:.3f} Å")
    
    # Get QCPP statistics
    cache_stats = qcpp_adapter.get_cache_stats()
    print(f"\nQCPP Analysis Statistics:")
    print(f"  Total requests: {cache_stats['total_requests']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"  Avg calculation time: {cache_stats['avg_calculation_time_ms']:.2f}ms")
    
    # Check performance recommendations
    recommendations = qcpp_adapter.get_performance_recommendations()
    if recommendations:
        print(f"\n⚠️  Performance Recommendations:")
        for rec in recommendations:
            print(f"    • {rec}")
    else:
        print(f"\n✓ Performance is within target ranges")
    
    # Step 8: Analyze final conformation
    print(f"\n[8] Analyzing best conformation with QCPP...")
    final_coords = best_agent._current_conformation.atom_coordinates
    final_metrics = qcpp_adapter.analyze_conformation(sequence, final_coords)
    
    print(f"\nFinal QCPP Metrics:")
    print(f"  Stability prediction: {final_metrics.stability_prediction:.3f}")
    print(f"  Average QCP: {final_metrics.average_qcp():.3f}")
    print(f"  Average coherence: {final_metrics.average_coherence():.3f}")
    print(f"  Calculation time: {final_metrics.calculation_time_ms:.2f}ms")
    
    print(f"\n{'='*70}")
    print("INTEGRATION SUCCESSFUL ✓")
    print("="*70)

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
======================================================================
EXAMPLE 11: QCPP INTEGRATION - BASIC USAGE
======================================================================

Protein: MQIFVKTLTG (10 residues)

[1] Initializing QCPP predictor...
[2] Configuring integration...
    Analysis frequency: 10%
    Cache size: 1000
    Adaptive frequency: True
[3] Creating QCPP adapter...
[4] Creating multi-agent coordinator...
[5] Initializing 5 agents...
[6] Running 200 iterations with QCPP feedback...
    (QCPP provides real-time stability predictions)

[7] Exploration complete in 8.42s

Best Result:
  Energy: -124.567 kcal/mol
  RMSD: 3.821 Å

QCPP Analysis Statistics:
  Total requests: 98
  Cache hits: 42
  Cache misses: 56
  Hit rate: 42.9%
  Avg calculation time: 0.35ms

✓ Performance is within target ranges

[8] Analyzing best conformation with QCPP...

Final QCPP Metrics:
  Stability prediction: 0.742
  Average QCP: 5.123
  Average coherence: 0.856
  Calculation time: 0.32ms

======================================================================
INTEGRATION SUCCESSFUL ✓
======================================================================
```

---

## Example 12: QCPP Integration - Performance Tuning

Optimize QCPP integration for different performance requirements.

```python
#!/usr/bin/env python3
"""
Example 12: QCPP Integration - Performance Tuning

Demonstrates:
- Comparing different configuration presets
- Performance vs accuracy tradeoffs
- Adaptive frequency adjustment
- Custom configuration tuning
"""

from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.qcpp_config import QCPPIntegrationConfig
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from protein_predictor import QuantumCoherenceProteinPredictor
import time

def benchmark_configuration(
    config_name: str,
    config: QCPPIntegrationConfig,
    sequence: str,
    agents: int,
    iterations: int
):
    """Benchmark a specific QCPP configuration."""
    print(f"\n{'='*70}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*70}")
    print(f"Analysis frequency: {config.analysis_frequency*100:.0f}%")
    print(f"Cache size: {config.cache_size}")
    print(f"Min energy change: {config.min_energy_change} kcal/mol")
    print(f"Adaptive frequency: {config.enable_adaptive_frequency}")
    
    # Setup
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, config)
    
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        qcpp_adapter=qcpp_adapter
    )
    coordinator.initialize_agents(count=agents, diversity_profile="balanced")
    
    # Run
    start_time = time.perf_counter()
    coordinator.run_parallel_exploration(iterations=iterations)
    elapsed = time.perf_counter() - start_time
    
    # Results
    best_agent = min(coordinator._agents, key=lambda a: a._current_energy)
    cache_stats = qcpp_adapter.get_cache_stats()
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {(agents*iterations)/elapsed:.1f} conf/s")
    print(f"  Best energy: {best_agent._current_energy:.3f} kcal/mol")
    print(f"  Best RMSD: {best_agent._current_rmsd:.3f} Å")
    print(f"  QCPP requests: {cache_stats['total_requests']}")
    print(f"  Cache hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"  Avg QCPP time: {cache_stats['avg_calculation_time_ms']:.2f}ms")
    
    return {
        'config_name': config_name,
        'elapsed': elapsed,
        'throughput': (agents*iterations)/elapsed,
        'best_energy': best_agent._current_energy,
        'best_rmsd': best_agent._current_rmsd,
        'qcpp_requests': cache_stats['total_requests'],
        'cache_hit_rate': cache_stats['hit_rate'],
        'avg_qcpp_time': cache_stats['avg_calculation_time_ms']
    }

def main():
    print("="*70)
    print("EXAMPLE 12: QCPP INTEGRATION - PERFORMANCE TUNING")
    print("="*70)
    
    # Test parameters
    sequence = "MQIFVKTLTGK"  # 11 residues (ubiquitin fragment)
    agents = 10
    iterations = 100
    
    print(f"\nTest setup:")
    print(f"  Sequence: {sequence} ({len(sequence)} residues)")
    print(f"  Agents: {agents}")
    print(f"  Iterations: {iterations}")
    
    # Test 3 configurations
    configs = [
        ("High Performance", QCPPIntegrationConfig.high_performance()),
        ("Default (Balanced)", QCPPIntegrationConfig.default()),
        ("High Accuracy", QCPPIntegrationConfig.high_accuracy())
    ]
    
    results = []
    for config_name, config in configs:
        result = benchmark_configuration(
            config_name, config, sequence, agents, iterations
        )
        results.append(result)
    
    # Comparison summary
    print(f"\n{'='*70}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Configuration':<25} {'Time':<10} {'Throughput':<15} {'Energy':<12} {'RMSD':<10}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['config_name']:<25} {r['elapsed']:>6.2f}s   "
              f"{r['throughput']:>8.1f} c/s   "
              f"{r['best_energy']:>8.3f}   "
              f"{r['best_rmsd']:>6.3f}Å")
    
    print(f"\n{'Configuration':<25} {'QCPP Reqs':<12} {'Hit Rate':<12} {'Avg Time':<10}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['config_name']:<25} {r['qcpp_requests']:>6}       "
              f"{r['cache_hit_rate']*100:>6.1f}%      "
              f"{r['avg_qcpp_time']:>6.2f}ms")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("\n1. High Performance:")
    print("   • Use for: Large-scale screening, high throughput needs")
    print("   • Pros: Fastest execution, minimal overhead")
    print("   • Cons: Less frequent physics validation")
    
    print("\n2. Default (Balanced):")
    print("   • Use for: General-purpose exploration")
    print("   • Pros: Good balance of speed and accuracy")
    print("   • Cons: Moderate physics validation frequency")
    
    print("\n3. High Accuracy:")
    print("   • Use for: Final refinement, critical predictions")
    print("   • Pros: Maximum physics validation")
    print("   • Cons: Slower execution, higher computational cost")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
```

---

## Example 13: QCPP Integration - Physics-Grounded Consciousness

Use QCPP physics to directly guide consciousness evolution.

```python
#!/usr/bin/env python3
"""
Example 13: QCPP Integration - Physics-Grounded Consciousness

Demonstrates:
- Physics-grounded consciousness updates
- QCPP stability influencing behavioral state
- Dynamic parameter adjustment based on stability
- Integrated trajectory recording
"""

from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.qcpp_config import QCPPIntegrationConfig
from ubf_protein.physics_grounded_consciousness import PhysicsGroundedConsciousness
from ubf_protein.dynamic_adjustment import DynamicParameterAdjuster
from ubf_protein.integrated_trajectory import IntegratedTrajectoryRecorder
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.config import SystemConfig
from protein_predictor import QuantumCoherenceProteinPredictor

def main():
    print("="*70)
    print("EXAMPLE 13: QCPP INTEGRATION - PHYSICS-GROUNDED CONSCIOUSNESS")
    print("="*70)
    
    # Setup
    sequence = "ACDEFGHIKL"  # 10 residues
    print(f"\nProtein: {sequence} ({len(sequence)} residues)")
    
    # Step 1: Initialize QCPP
    print("\n[1] Initializing QCPP system...")
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    qcpp_config = QCPPIntegrationConfig.default()
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, qcpp_config)
    
    # Step 2: Create physics-grounded consciousness
    print("[2] Creating physics-grounded consciousness...")
    agent = ProteinAgent(protein_sequence=sequence)
    base_consciousness = agent.get_consciousness_state()
    
    physics_consciousness = PhysicsGroundedConsciousness(
        base_consciousness=base_consciousness,
        config=qcpp_config
    )
    
    # Step 3: Create dynamic parameter adjuster
    print("[3] Creating dynamic parameter adjuster...")
    base_config = SystemConfig()
    parameter_adjuster = DynamicParameterAdjuster(base_config)
    
    # Step 4: Create integrated trajectory recorder
    print("[4] Creating integrated trajectory recorder...")
    trajectory = IntegratedTrajectoryRecorder()
    
    # Step 5: Run exploration with physics grounding
    print(f"[5] Running 200 iterations with physics-grounded consciousness...")
    
    for iteration in range(200):
        # Perform move
        outcome = agent.explore_step()
        
        # Analyze with QCPP (every 10th move)
        qcpp_metrics = None
        if iteration % 10 == 0:
            coords = agent._current_conformation.atom_coordinates
            qcpp_metrics = qcpp_adapter.analyze_conformation(sequence, coords)
            
            # Update consciousness based on physics
            new_consciousness = physics_consciousness.update_with_qcpp_metrics(
                qcpp_metrics, outcome
            )
            agent._consciousness = new_consciousness
            
            # Adjust parameters based on stability
            current_config = base_config  # In practice, get from agent
            new_config = parameter_adjuster.adjust_parameters(
                qcpp_metrics, current_config
            )
            # Apply new_config to agent (implementation-specific)
        
        # Record trajectory
        trajectory.record_step(
            iteration=iteration,
            conformation=agent._current_conformation,
            consciousness=agent.get_consciousness_state(),
            behavioral=agent._behavioral_state,
            energy=agent._current_energy,
            rmsd=agent._current_rmsd,
            qcpp_metrics=qcpp_metrics
        )
        
        # Progress update
        if iteration % 50 == 0:
            consciousness = agent.get_consciousness_state()
            print(f"    Iter {iteration}: Energy={agent._current_energy:.2f}, "
                  f"RMSD={agent._current_rmsd:.2f}, "
                  f"f={consciousness.get_frequency():.2f}Hz, "
                  f"c={consciousness.get_coherence():.3f}")
            if qcpp_metrics:
                print(f"              QCPP Stability={qcpp_metrics.stability_prediction:.3f}")
    
    # Step 6: Export results
    print(f"\n[6] Exporting integrated trajectory...")
    trajectory.export_json("integrated_trajectory.json")
    trajectory.export_csv("integrated_trajectory.csv")
    print("    • integrated_trajectory.json")
    print("    • integrated_trajectory.csv")
    
    # Step 7: Analysis
    print(f"\n[7] Analyzing physics-consciousness correlation...")
    
    # Get QCPP points (every 10th iteration)
    qcpp_points = [p for p in trajectory.points if p.qcpp_stability is not None]
    
    if qcpp_points:
        # Calculate correlations
        stabilities = [p.qcpp_stability for p in qcpp_points]
        coherences = [p.consciousness_coherence for p in qcpp_points]
        
        avg_stability = sum(stabilities) / len(stabilities)
        avg_coherence = sum(coherences) / len(coherences)
        
        print(f"\nPhysics-Consciousness Statistics:")
        print(f"  QCPP measurements: {len(qcpp_points)}")
        print(f"  Average stability: {avg_stability:.3f}")
        print(f"  Average coherence: {avg_coherence:.3f}")
        print(f"  Final stability: {stabilities[-1]:.3f}")
        print(f"  Final coherence: {coherences[-1]:.3f}")
        
        # Check if coherence tracks stability
        stability_change = stabilities[-1] - stabilities[0]
        coherence_change = coherences[-1] - coherences[0]
        
        print(f"\nEvolution:")
        print(f"  Stability change: {stability_change:+.3f}")
        print(f"  Coherence change: {coherence_change:+.3f}")
        
        if (stability_change > 0 and coherence_change > 0) or \
           (stability_change < 0 and coherence_change < 0):
            print(f"  ✓ Coherence correlates with stability (physics-grounded)")
        else:
            print(f"  ⚠ Coherence diverges from stability")
    
    # Final result
    print(f"\n{'='*70}")
    print(f"Final Result:")
    print(f"  Energy: {agent._current_energy:.3f} kcal/mol")
    print(f"  RMSD: {agent._current_rmsd:.3f} Å")
    print(f"  Consciousness: f={agent.get_consciousness_state().get_frequency():.2f}Hz, "
          f"c={agent.get_consciousness_state().get_coherence():.3f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

**Key Insights:**

1. **Physics Grounding**: Consciousness evolution directly influenced by quantum physics
2. **Stability Tracking**: Coherence increases/decreases with QCPP stability predictions
3. **Dynamic Adjustment**: Exploration parameters adapt to physical stability landscape
4. **Integrated Recording**: Combined UBF + QCPP metrics in single trajectory

---

## Example 14: Physics Enhancements - Disulfide Detection

Detect and enforce disulfide bond constraints for improved folding accuracy.

```python
#!/usr/bin/env python3
"""
Example 14: Disulfide bond detection and constraint enforcement.

Demonstrates:
- Detecting disulfide bonds from PDB files
- Predicting disulfide bonds from sequence
- Checking bond satisfaction
- Using disulfide constraints in enhanced energy calculator
"""

from ubf_protein.disulfide_detector import DisulfideDetector
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
from ubf_protein.models import DisulfideBond
from ubf_protein.protein_agent import ProteinAgent

def main():
    print("="*70)
    print("EXAMPLE 14: Disulfide Bond Detection and Constraint Enforcement")
    print("="*70)
    
    # Method 1: Detect from PDB file (Crambin - 1CRN)
    print("\n[1] Detecting disulfide bonds from PDB file...")
    detector = DisulfideDetector()
    
    pdb_file = "pdb_cache/1CRN.pdb"  # Download first if needed
    try:
        bonds_from_pdb = detector.detect_from_pdb(pdb_file)
        print(f"    Found {len(bonds_from_pdb)} disulfide bonds:")
        for bond in bonds_from_pdb:
            print(f"      CYS{bond.residue_i+1}-CYS{bond.residue_j+1}: "
                  f"target {bond.distance:.1f} Å, tolerance ±{bond.tolerance:.1f} Å")
    except FileNotFoundError:
        print(f"    ⚠ PDB file not found. Skipping PDB detection.")
        bonds_from_pdb = []
    
    # Method 2: Predict from sequence
    print("\n[2] Predicting disulfide bonds from sequence...")
    sequence = "ACDEFGHCIKLMNPCQRST"  # 3 cysteines at positions 1, 7, 14
    bonds_from_seq = detector.predict_from_sequence(sequence)
    print(f"    Predicted {len(bonds_from_seq)} disulfide bonds:")
    for bond in bonds_from_seq:
        print(f"      CYS{bond.residue_i+1}-CYS{bond.residue_j+1}: "
              f"target {bond.distance:.1f} Å, tolerance ±{bond.tolerance:.1f} Å")
    
    # Method 3: Check satisfaction in conformation
    print("\n[3] Checking bond satisfaction...")
    
    # Create mock coordinates (for demonstration)
    import random
    num_residues = len(sequence)
    coords = [(random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10))
              for _ in range(num_residues)]
    
    satisfied, violations = detector.check_satisfaction(bonds_from_seq, coords)
    
    if satisfied:
        print(f"    ✓ All disulfide bonds satisfied!")
    else:
        print(f"    ✗ {len(violations)} bond violation(s):")
        for msg in violations:
            print(f"      {msg}")
    
    # Method 4: Use in enhanced energy calculator
    print("\n[4] Using disulfide constraints in energy calculator...")
    
    calculator = EnhancedEnergyCalculator(
        sequence=sequence,
        disulfide_bonds=bonds_from_seq,
        enable_side_chains=True,
        enable_solvent=True,
        enable_entropic=False  # No trajectory yet
    )
    
    print(f"    Created enhanced energy calculator with:")
    print(f"      • {len(bonds_from_seq)} disulfide constraints")
    print(f"      • Side-chain field interactions")
    print(f"      • Solvent screening corrections")
    
    # Calculate energy breakdown (mock conformation)
    from ubf_protein.models import Conformation
    mock_conf = Conformation(
        conformation_id="test_001",
        sequence=sequence,
        atom_coordinates=coords,
        secondary_structure=['C'] * num_residues,
        energy=0.0,
        rmsd=5.0,
        phi_angles=[0.0] * num_residues,
        psi_angles=[0.0] * num_residues
    )
    
    breakdown = calculator.calculate_energy_breakdown(mock_conf)
    
    print(f"\n    Energy breakdown:")
    for component, energy in breakdown.items():
        print(f"      {component:15s}: {energy:8.2f} kcal/mol")
    
    # Method 5: Calculate violation energy
    print("\n[5] Analyzing individual bond violation energies...")
    
    for i, bond in enumerate(bonds_from_seq):
        # Get CA-CA distance
        pos_i = coords[bond.residue_i]
        pos_j = coords[bond.residue_j]
        distance = ((pos_i[0] - pos_j[0])**2 + 
                   (pos_i[1] - pos_j[1])**2 + 
                   (pos_i[2] - pos_j[2])**2) ** 0.5
        
        violation_energy = bond.violation_energy(distance)
        is_satisfied = bond.is_satisfied(distance)
        
        print(f"    Bond {i+1} (CYS{bond.residue_i+1}-CYS{bond.residue_j+1}):")
        print(f"      Current distance: {distance:.2f} Å")
        print(f"      Target distance: {bond.distance:.2f} Å")
        print(f"      Satisfied: {is_satisfied}")
        print(f"      Violation energy: {violation_energy:.2f} kcal/mol")
    
    print(f"\n{'='*70}")
    print("Key Takeaways:")
    print("  • Disulfide bonds enforce spatial constraints (CA-CA ≈ 3.8 Å)")
    print("  • PDB detection extracts SSBOND records (most accurate)")
    print("  • Sequence prediction provides fallback for Cys pairs")
    print("  • Violation energy penalizes unsatisfied constraints")
    print("  • Enhanced calculator integrates all constraints automatically")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

---

## Example 15: Physics Enhancements - Enhanced Energy Calculator

Compare baseline vs enhanced energy calculations with all physics improvements.

```python
#!/usr/bin/env python3
"""
Example 15: Enhanced energy calculator with full physics.

Demonstrates:
- Baseline vs enhanced energy comparison
- Energy component breakdown
- Side-chain field interactions
- Solvent screening corrections
- Entropic contributions
- Performance analysis
"""

from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
from ubf_protein.models import Conformation, DisulfideBond
import time

def main():
    print("="*70)
    print("EXAMPLE 15: Enhanced Energy Calculator - Full Physics Comparison")
    print("="*70)
    
    # Test protein: small peptide with diverse residues
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 amino acids
    num_residues = len(sequence)
    
    # Create mock conformation
    import random
    random.seed(42)  # Reproducible
    coords = [(random.uniform(0, 15), random.uniform(0, 15), random.uniform(0, 15))
              for _ in range(num_residues)]
    
    mock_conf = Conformation(
        conformation_id="comparison_test",
        sequence=sequence,
        atom_coordinates=coords,
        secondary_structure=['C'] * num_residues,
        energy=0.0,
        rmsd=5.0,
        phi_angles=[-60.0] * num_residues,  # Alpha-helix region
        psi_angles=[-45.0] * num_residues
    )
    
    # Setup disulfide bonds (Cys at position 1)
    # In real sequence, C is at index 1, no other cysteines
    disulfide_bonds = []  # No disulfide bonds in this sequence
    
    # [1] Baseline energy calculation
    print("\n[1] Baseline molecular mechanics energy...")
    baseline_calculator = MolecularMechanicsEnergy()
    
    start = time.perf_counter()
    baseline_energy = baseline_calculator.calculate_energy(mock_conf)
    baseline_time = (time.perf_counter() - start) * 1000
    
    baseline_breakdown = baseline_calculator.calculate_detailed_energy(mock_conf)
    
    print(f"    Total energy: {baseline_energy:.2f} kcal/mol")
    print(f"    Calculation time: {baseline_time:.3f} ms")
    print(f"\n    Component breakdown:")
    for component, energy in baseline_breakdown.items():
        if component != "total":
            print(f"      {component:15s}: {energy:8.2f} kcal/mol")
    
    # [2] Enhanced energy calculation (all features)
    print("\n[2] Enhanced energy with all physics improvements...")
    enhanced_calculator = EnhancedEnergyCalculator(
        sequence=sequence,
        disulfide_bonds=disulfide_bonds,
        enable_side_chains=True,
        enable_solvent=True,
        enable_entropic=True
    )
    
    # Create mock trajectory for entropic calculation
    from ubf_protein.models import ConformationSnapshot
    trajectory = [
        ConformationSnapshot(
            iteration=i,
            conformation_id=f"snap_{i}",
            energy=baseline_energy + random.uniform(-5, 5),
            rmsd=5.0 + random.uniform(-1, 1),
            timestamp=time.time_ns() // 1_000_000,
            consciousness_frequency=9.0,
            consciousness_coherence=0.6
        )
        for i in range(50)
    ]
    
    start = time.perf_counter()
    enhanced_energy = enhanced_calculator.calculate_energy(mock_conf, trajectory)
    enhanced_time = (time.perf_counter() - start) * 1000
    
    enhanced_breakdown = enhanced_calculator.calculate_energy_breakdown(mock_conf, trajectory)
    
    print(f"    Total energy: {enhanced_energy:.2f} kcal/mol")
    print(f"    Calculation time: {enhanced_time:.3f} ms")
    print(f"\n    Component breakdown:")
    for component, energy in enhanced_breakdown.items():
        if component != "total":
            print(f"      {component:15s}: {energy:8.2f} kcal/mol")
    
    # [3] Feature ablation study
    print("\n[3] Feature ablation study (incremental enhancements)...")
    
    configs = [
        ("Baseline only", False, False, False),
        ("+ Side chains", True, False, False),
        ("+ Solvent", True, True, False),
        ("+ Entropic", True, True, True)
    ]
    
    print(f"\n    {'Configuration':<25s} {'Energy (kcal/mol)':>18s} {'Time (ms)':>12s}")
    print(f"    {'-'*58}")
    
    for config_name, side_chains, solvent, entropic in configs:
        calc = EnhancedEnergyCalculator(
            sequence=sequence,
            disulfide_bonds=disulfide_bonds,
            enable_side_chains=side_chains,
            enable_solvent=solvent,
            enable_entropic=entropic
        )
        
        start = time.perf_counter()
        energy = calc.calculate_energy(mock_conf, trajectory if entropic else None)
        calc_time = (time.perf_counter() - start) * 1000
        
        print(f"    {config_name:<25s} {energy:>18.2f} {calc_time:>12.3f}")
    
    # [4] Analysis
    print(f"\n[4] Comparative analysis...")
    
    energy_diff = enhanced_energy - baseline_energy
    time_overhead = enhanced_time - baseline_time
    overhead_percent = (time_overhead / baseline_time) * 100 if baseline_time > 0 else 0
    
    print(f"\n    Baseline total:   {baseline_energy:.2f} kcal/mol")
    print(f"    Enhanced total:   {enhanced_energy:.2f} kcal/mol")
    print(f"    Difference:       {energy_diff:+.2f} kcal/mol")
    print(f"\n    Baseline time:    {baseline_time:.3f} ms")
    print(f"    Enhanced time:    {enhanced_time:.3f} ms")
    print(f"    Overhead:         {time_overhead:+.3f} ms ({overhead_percent:+.1f}%)")
    
    # Extract enhancement contributions
    side_chain_contrib = enhanced_breakdown.get("side_chains", 0.0)
    entropic_contrib = enhanced_breakdown.get("entropic", 0.0)
    
    print(f"\n    Enhancement contributions:")
    print(f"      Side-chain interactions: {side_chain_contrib:.2f} kcal/mol")
    print(f"      Entropic correction:     {entropic_contrib:.2f} kcal/mol")
    print(f"      Solvent screening:       (integrated into base terms)")
    
    print(f"\n{'='*70}")
    print("Key Insights:")
    print("  • Enhanced energy includes side-chain, solvent, and entropic terms")
    print("  • Side-chain interactions capture hydrophobic/electrostatic effects")
    print("  • Solvent corrections reduce buried charge interactions")
    print("  • Entropic terms account for conformational diversity")
    print(f"  • Computational overhead: {overhead_percent:.1f}% (acceptable for accuracy gain)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

---

## Example 16: Physics Enhancements - Local Refinement

Optimize conformations using gradient descent local refinement.

```python
#!/usr/bin/env python3
"""
Example 16: Local refinement with gradient descent.

Demonstrates:
- Creating local refinement optimizer
- Refining crude conformation to local minimum
- Analyzing refinement statistics
- Convergence behavior
- Step size adaptation
"""

from ubf_protein.local_refinement import LocalRefinement
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
from ubf_protein.models import Conformation
import random
import time

def main():
    print("="*70)
    print("EXAMPLE 16: Local Refinement with Gradient Descent")
    print("="*70)
    
    # Create test protein
    sequence = "ACDEFGHIKLMNPQ"  # 14 residues
    num_residues = len(sequence)
    
    # Create deliberately poor initial conformation (random positions)
    print("\n[1] Creating initial crude conformation...")
    random.seed(42)
    coords = [(random.uniform(0, 20), random.uniform(0, 20), random.uniform(0, 20))
              for _ in range(num_residues)]
    
    initial_conf = Conformation(
        conformation_id="crude_001",
        sequence=sequence,
        atom_coordinates=coords,
        secondary_structure=['C'] * num_residues,
        energy=0.0,
        rmsd=10.0,
        phi_angles=[random.uniform(-180, 180) for _ in range(num_residues)],
        psi_angles=[random.uniform(-180, 180) for _ in range(num_residues)]
    )
    
    # Setup energy calculator
    calculator = EnhancedEnergyCalculator(
        sequence=sequence,
        enable_side_chains=True,
        enable_solvent=True,
        enable_entropic=False  # No trajectory for initial refinement
    )
    
    initial_energy = calculator.calculate_energy(initial_conf)
    print(f"    Initial energy: {initial_energy:.2f} kcal/mol")
    print(f"    Initial RMSD: {initial_conf.rmsd:.2f} Å")
    
    # [2] Create local refinement optimizer
    print("\n[2] Initializing local refinement optimizer...")
    refiner = LocalRefinement(
        energy_calculator=calculator,
        max_iterations=100,
        convergence_threshold=0.001,  # kcal/mol
        initial_step_size=0.01  # Angstroms
    )
    print(f"    Max iterations: 100")
    print(f"    Convergence threshold: 0.001 kcal/mol")
    print(f"    Initial step size: 0.01 Å")
    
    # [3] Refine conformation
    print("\n[3] Running gradient descent refinement...")
    print(f"    (This may take 10-30 seconds...)")
    
    start = time.perf_counter()
    refined_conf, stats = refiner.refine(initial_conf)
    refinement_time = time.perf_counter() - start
    
    print(f"\n    ✓ Refinement complete in {refinement_time:.2f} seconds")
    
    # [4] Analyze refinement statistics
    print("\n[4] Refinement statistics:")
    print(f"    {'='*50}")
    print(f"    Iterations completed:   {stats['iterations']}")
    print(f"    Converged:              {stats['converged']}")
    print(f"\n    Energy evolution:")
    print(f"      Initial:              {stats['initial_energy']:.3f} kcal/mol")
    print(f"      Final:                {stats['final_energy']:.3f} kcal/mol")
    print(f"      Improvement:          {stats['energy_improvement']:.3f} kcal/mol")
    print(f"\n    Optimization details:")
    print(f"      Final step size:      {stats['final_step_size']:.4f} Å")
    print(f"      Time per iteration:   {refinement_time*1000/stats['iterations']:.2f} ms")
    print(f"    {'='*50}")
    
    # [5] Energy breakdown comparison
    print("\n[5] Energy breakdown comparison...")
    
    initial_breakdown = calculator.calculate_energy_breakdown(initial_conf)
    refined_breakdown = calculator.calculate_energy_breakdown(refined_conf)
    
    print(f"\n    {'Component':<20s} {'Initial':>12s} {'Refined':>12s} {'Change':>12s}")
    print(f"    {'-'*58}")
    
    for component in initial_breakdown.keys():
        if component == "total":
            continue
        init_val = initial_breakdown[component]
        refined_val = refined_breakdown[component]
        change = refined_val - init_val
        
        print(f"    {component:<20s} {init_val:>12.2f} {refined_val:>12.2f} {change:>12.2f}")
    
    print(f"    {'-'*58}")
    total_init = initial_breakdown["total"]
    total_refined = refined_breakdown["total"]
    total_change = total_refined - total_init
    print(f"    {'TOTAL':<20s} {total_init:>12.2f} {total_refined:>12.2f} {total_change:>12.2f}")
    
    # [6] Convergence check
    print("\n[6] Convergence analysis...")
    
    if stats['converged']:
        print(f"    ✓ Refinement converged successfully")
        print(f"      Final energy change below threshold: {stats['convergence_threshold']:.4f} kcal/mol")
    else:
        print(f"    ⚠ Refinement reached max iterations without convergence")
        print(f"      Consider increasing max_iterations or loosening threshold")
    
    # [7] Geometric validation
    print("\n[7] Geometric validation...")
    
    # Check bond lengths (simplified - just check CA-CA distances)
    initial_violations = 0
    refined_violations = 0
    
    for i in range(num_residues - 1):
        # CA-CA distance should be ~3.8 Å for sequential residues
        init_dist = ((initial_conf.atom_coordinates[i][0] - initial_conf.atom_coordinates[i+1][0])**2 +
                     (initial_conf.atom_coordinates[i][1] - initial_conf.atom_coordinates[i+1][1])**2 +
                     (initial_conf.atom_coordinates[i][2] - initial_conf.atom_coordinates[i+1][2])**2) ** 0.5
        
        refined_dist = ((refined_conf.atom_coordinates[i][0] - refined_conf.atom_coordinates[i+1][0])**2 +
                       (refined_conf.atom_coordinates[i][1] - refined_conf.atom_coordinates[i+1][1])**2 +
                       (refined_conf.atom_coordinates[i][2] - refined_conf.atom_coordinates[i+1][2])**2) ** 0.5
        
        # Acceptable range: 3.2 - 4.4 Å
        if not (3.2 <= init_dist <= 4.4):
            initial_violations += 1
        if not (3.2 <= refined_dist <= 4.4):
            refined_violations += 1
    
    print(f"    Sequential CA-CA distance violations:")
    print(f"      Initial:   {initial_violations}/{num_residues-1}")
    print(f"      Refined:   {refined_violations}/{num_residues-1}")
    print(f"      Improvement: {initial_violations - refined_violations} violations resolved")
    
    print(f"\n{'='*70}")
    print("Key Takeaways:")
    print("  • Local refinement significantly improves crude conformations")
    print(f"  • Energy improvement: {stats['energy_improvement']:.2f} kcal/mol")
    print(f"  • Typical refinement: 20-50 iterations, 10-30 seconds")
    print("  • Adaptive step size prevents geometry violations")
    print("  • Use after global search for local minimum optimization")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

---

## Example 17: Physics Enhancements - Configuration System

Use `EnhancedPhysicsConfig` for flexible enhancement control.

```python
#!/usr/bin/env python3
"""
Example 17: Enhanced physics configuration system.

Demonstrates:
- Using factory methods (baseline, enhanced_default, auto_adapt)
- Size-based configuration (small/medium/large proteins)
- Environment variable configuration
- Custom configuration creation
- Configuration comparison
"""

from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
import os

def main():
    print("="*70)
    print("EXAMPLE 17: Enhanced Physics Configuration System")
    print("="*70)
    
    # [1] Factory presets
    print("\n[1] Using factory method presets...")
    
    # Baseline (no enhancements)
    baseline = EnhancedPhysicsConfig.baseline()
    print("\n  Baseline configuration:")
    print(f"    Enhanced energy: {baseline.use_enhanced_energy}")
    print(f"    Side chains:     {baseline.enable_side_chains}")
    print(f"    Solvent:         {baseline.enable_solvent}")
    print(f"    Entropic:        {baseline.enable_entropic}")
    print(f"    Refinement:      {baseline.enable_refinement}")
    
    # Enhanced default
    enhanced = EnhancedPhysicsConfig.enhanced_default()
    print("\n  Enhanced default configuration:")
    print(f"    Enhanced energy: {enhanced.use_enhanced_energy}")
    print(f"    Side chains:     {enhanced.enable_side_chains}")
    print(f"    Solvent:         {enhanced.enable_solvent}")
    print(f"    Entropic:        {enhanced.enable_entropic}")
    print(f"    Refinement:      {enhanced.enable_refinement}")
    print(f"    Refinement iterations: {enhanced.refinement_max_iterations}")
    
    # [2] Size-based adaptation
    print("\n[2] Size-based auto-adaptation...")
    
    sequences = {
        "Small": "ACDEFG",                    # 6 residues
        "Medium": "ACDEFGHIKLMNPQRSTVWY" * 5,  # 100 residues
        "Large": "ACDEFGHIKLMNPQRSTVWY" * 10   # 200 residues
    }
    
    for size_name, seq in sequences.items():
        config = EnhancedPhysicsConfig.auto_adapt(seq)
        print(f"\n  {size_name} protein ({len(seq)} residues):")
        print(f"    Refinement iterations: {config.refinement_max_iterations}")
        print(f"    Trajectory window:     {config.trajectory_window}")
        print(f"    Side-chain cutoff:     {config.sidechain_cutoff:.1f} Å")
    
    # [3] Environment variable configuration
    print("\n[3] Loading from environment variables...")
    
    # Set environment variables
    os.environ["UBF_USE_ENHANCED_ENERGY"] = "true"
    os.environ["UBF_ENABLE_SIDE_CHAINS"] = "true"
    os.environ["UBF_ENABLE_SOLVENT"] = "false"
    os.environ["UBF_TEMPERATURE"] = "310.0"
    os.environ["UBF_REFINEMENT_MAX_ITERATIONS"] = "150"
    
    env_config = EnhancedPhysicsConfig.from_environment()
    print(f"\n  Configuration from environment:")
    print(f"    Enhanced energy: {env_config.use_enhanced_energy}")
    print(f"    Side chains:     {env_config.enable_side_chains}")
    print(f"    Solvent:         {env_config.enable_solvent}")
    print(f"    Temperature:     {env_config.temperature:.1f} K")
    print(f"    Refinement iters: {env_config.refinement_max_iterations}")
    
    # Clean up environment
    for key in list(os.environ.keys()):
        if key.startswith("UBF_"):
            del os.environ[key]
    
    # [4] Custom configuration
    print("\n[4] Creating custom configurations...")
    
    # Start with default and modify
    custom = EnhancedPhysicsConfig.enhanced_default()
    custom = custom.with_refinement(False)  # Disable refinement
    
    print(f"\n  Custom configuration (no refinement):")
    print(f"    Enhanced energy: {custom.use_enhanced_energy}")
    print(f"    Refinement:      {custom.enable_refinement}")
    
    # Add disulfide bonds
    from ubf_protein.models import DisulfideBond
    bonds = [DisulfideBond(2, 10), DisulfideBond(5, 15)]
    custom_with_bonds = custom.with_disulfide_bonds(bonds)
    
    print(f"\n  Custom with disulfide bonds:")
    print(f"    Number of bonds: {len(custom_with_bonds.disulfide_bonds)}")
    for i, bond in enumerate(custom_with_bonds.disulfide_bonds):
        print(f"      Bond {i+1}: CYS{bond.residue_i+1}-CYS{bond.residue_j+1}")
    
    # [5] Configuration summary
    print("\n[5] Configuration summaries...")
    
    print("\n" + "="*70)
    print(enhanced.summary())
    print("="*70)
    
    # [6] Using configurations with MultiAgentCoordinator
    print("\n[6] Using configurations with multi-agent system...")
    
    sequence = "ACDEFGHIKLMNPQ"
    
    # Baseline run
    print(f"\n  Creating baseline coordinator...")
    coordinator_baseline = MultiAgentCoordinator(
        protein_sequence=sequence,
        physics_config=baseline
    )
    print(f"    ✓ Baseline coordinator created (no enhancements)")
    
    # Enhanced run
    print(f"\n  Creating enhanced coordinator...")
    coordinator_enhanced = MultiAgentCoordinator(
        protein_sequence=sequence,
        physics_config=enhanced
    )
    print(f"    ✓ Enhanced coordinator created (full physics)")
    
    # Auto-adapted run
    print(f"\n  Creating auto-adapted coordinator...")
    auto_config = EnhancedPhysicsConfig.auto_adapt(sequence)
    coordinator_auto = MultiAgentCoordinator(
        protein_sequence=sequence,
        physics_config=auto_config
    )
    print(f"    ✓ Auto-adapted coordinator created")
    print(f"      (optimized for {len(sequence)}-residue protein)")
    
    # [7] Configuration comparison
    print("\n[7] Configuration parameter comparison...")
    
    configs_to_compare = {
        "Baseline": baseline,
        "Enhanced": enhanced,
        "Small": EnhancedPhysicsConfig.small_protein(20),
        "Large": EnhancedPhysicsConfig.large_protein(200)
    }
    
    print(f"\n  {'Parameter':<30s} {'Baseline':>10s} {'Enhanced':>10s} {'Small':>10s} {'Large':>10s}")
    print(f"  {'-'*74}")
    
    params = [
        ("use_enhanced_energy", lambda c: "Yes" if c.use_enhanced_energy else "No"),
        ("refinement_max_iterations", lambda c: str(c.refinement_max_iterations)),
        ("trajectory_window", lambda c: str(c.trajectory_window)),
        ("sidechain_cutoff", lambda c: f"{c.sidechain_cutoff:.1f}"),
        ("temperature", lambda c: f"{c.temperature:.0f}"),
    ]
    
    for param_name, getter in params:
        values = [getter(configs_to_compare[name]) for name in ["Baseline", "Enhanced", "Small", "Large"]]
        print(f"  {param_name:<30s} {values[0]:>10s} {values[1]:>10s} {values[2]:>10s} {values[3]:>10s}")
    
    print(f"\n{'='*70}")
    print("Key Insights:")
    print("  • Factory methods provide sensible defaults")
    print("  • auto_adapt() automatically tunes for protein size")
    print("  • Environment variables allow external configuration")
    print("  • Immutable configs ensure thread-safe parallel execution")
    print("  • Configuration comparison helps optimize for specific cases")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

---

These examples demonstrate the major features and use cases of the UBF Protein System, including both the consciousness-based exploration and the comprehensive physics enhancements. Each example is self-contained and can be run independently. For more advanced usage and integration examples, see the test suite in `ubf_protein/tests/` and the complete documentation in `API.md`.
