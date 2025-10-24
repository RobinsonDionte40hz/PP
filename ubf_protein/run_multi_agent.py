#!/usr/bin/env python3
"""
Multi-agent exploration script for UBF protein system.

This script runs multiple autonomous protein agents in parallel to collectively
explore conformational space and predict protein structure using consciousness-
based navigation with shared memory exchange.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.models import ExplorationResults


def run_multi_agent(sequence: str,
                   agents: int,
                   iterations: int,
                   diversity_profile: str = "balanced",
                   output_file: Optional[str] = None,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Run multi-agent parallel exploration for specified iterations.

    Args:
        sequence: Protein amino acid sequence
        agents: Number of agents to run in parallel
        iterations: Number of exploration iterations per agent
        diversity_profile: Agent diversity profile ("balanced", "cautious", "aggressive")
        output_file: Path to save results JSON (optional)
        verbose: Print detailed progress information

    Returns:
        Dictionary with exploration results
    """
    print(f"Starting multi-agent exploration")
    print(f"Sequence length: {len(sequence)} residues")
    print(f"Agents: {agents}")
    print(f"Iterations per agent: {iterations}")
    print(f"Total iterations: {agents * iterations}")
    print(f"Diversity profile: {diversity_profile}")

    # Initialize coordinator
    print("\nInitializing multi-agent coordinator...")
    coordinator = MultiAgentCoordinator(sequence)
    
    # Initialize agents with diversity
    print(f"Initializing {agents} agents with {diversity_profile} diversity...")
    agent_list = coordinator.initialize_agents(agents, diversity_profile)
    print(f"Agents initialized: {len(agent_list)}")
    
    if diversity_profile == "balanced":
        cautious = len([a for a in agent_list if 3.0 <= a.get_consciousness_state().get_frequency() < 7.0])
        balanced = len([a for a in agent_list if 7.0 <= a.get_consciousness_state().get_frequency() < 11.0])
        aggressive = len([a for a in agent_list if 11.0 <= a.get_consciousness_state().get_frequency() <= 15.0])
        print(f"  Cautious (low freq): {cautious} agents")
        print(f"  Balanced (mid freq): {balanced} agents")
        print(f"  Aggressive (high freq): {aggressive} agents")

    # Run parallel exploration
    print("\nStarting parallel exploration...")
    print("-" * 60)
    
    results = coordinator.run_parallel_exploration(iterations)
    
    print("-" * 60)
    print("Exploration complete!")

    # Compile detailed results
    detailed_results = {
        'metadata': {
            'sequence': sequence,
            'sequence_length': len(sequence),
            'num_agents': agents,
            'iterations_per_agent': iterations,
            'total_iterations': results.total_iterations,
            'diversity_profile': diversity_profile,
            'total_runtime_seconds': results.total_runtime_seconds,
            'avg_time_per_iteration_ms': (results.total_runtime_seconds / results.total_iterations) * 1000 if results.total_iterations > 0 else 0
        },
        'best_results': {
            'best_energy': results.best_energy,
            'best_rmsd': results.best_rmsd,
            'total_conformations_explored': results.total_conformations_explored
        },
        'collective_learning': {
            'collective_learning_benefit': results.collective_learning_benefit,
            'shared_memories_created': results.shared_memories_created,
            'avg_learning_improvement': sum(m.learning_improvement for m in results.agent_metrics) / max(1, len(results.agent_metrics))
        },
        'agent_summary': {
            'total_agents': len(results.agent_metrics),
            'avg_conformations_per_agent': results.total_conformations_explored / max(1, len(results.agent_metrics)),
            'avg_memories_per_agent': sum(m.memories_created for m in results.agent_metrics) / max(1, len(results.agent_metrics)),
            'avg_decision_time_ms': sum(m.avg_decision_time_ms for m in results.agent_metrics) / max(1, len(results.agent_metrics)),
            'total_stuck_events': sum(m.stuck_in_minima_count for m in results.agent_metrics),
            'total_escapes': sum(m.successful_escapes for m in results.agent_metrics)
        },
        'per_agent_metrics': [
            {
                'agent_id': m.agent_id,
                'iterations': m.iterations_completed,
                'conformations': m.conformations_explored,
                'memories': m.memories_created,
                'best_energy': m.best_energy_found,
                'best_rmsd': m.best_rmsd_found,
                'learning_improvement': m.learning_improvement,
                'decision_time_ms': m.avg_decision_time_ms,
                'stuck_count': m.stuck_in_minima_count,
                'escape_count': m.successful_escapes
            }
            for m in results.agent_metrics
        ]
    }

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Best Energy Found: {results.best_energy:.2f} kcal/mol")
    print(f"Best RMSD: {results.best_rmsd:.2f} Å")
    print(f"Total Conformations Explored: {results.total_conformations_explored}")
    print(f"Total Runtime: {results.total_runtime_seconds:.2f} seconds")
    print(f"Avg Time per Iteration: {(results.total_runtime_seconds / results.total_iterations) * 1000:.2f} ms")
    print(f"\nCollective Learning:")
    print(f"  Collective Learning Benefit: {results.collective_learning_benefit:.2%}")
    print(f"  Shared Memories Created: {results.shared_memories_created}")
    print(f"  Avg Learning Improvement: {detailed_results['collective_learning']['avg_learning_improvement']:.1f}%")
    print(f"\nAgent Performance:")
    print(f"  Total Agents: {len(results.agent_metrics)}")
    print(f"  Avg Conformations per Agent: {detailed_results['agent_summary']['avg_conformations_per_agent']:.0f}")
    print(f"  Avg Memories per Agent: {detailed_results['agent_summary']['avg_memories_per_agent']:.1f}")
    print(f"  Avg Decision Time: {detailed_results['agent_summary']['avg_decision_time_ms']:.2f} ms")
    print(f"  Total Stuck/Escape Events: {detailed_results['agent_summary']['total_stuck_events']}/{detailed_results['agent_summary']['total_escapes']}")
    print("=" * 60)

    # Print top 5 performing agents
    if verbose and results.agent_metrics:
        print("\nTop 5 Performing Agents (by best energy):")
        print("-" * 60)
        sorted_agents = sorted(results.agent_metrics, key=lambda m: m.best_energy_found)[:5]
        for i, m in enumerate(sorted_agents, 1):
            print(f"{i}. {m.agent_id}: Energy={m.best_energy_found:.2f}, "
                  f"RMSD={m.best_rmsd_found:.2f}, "
                  f"Learning={m.learning_improvement:.1f}%")

    # Save to file if requested
    if output_file:
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print("Results saved successfully!")
        
        # Auto-generate user-friendly report
        report_file = output_file.replace('.json', '_REPORT.txt')
        try:
            import subprocess
            report_script = Path(__file__).parent.parent / 'generate_protein_report.py'
            if report_script.exists():
                subprocess.run([sys.executable, str(report_script), output_file], 
                             capture_output=True, check=False)
                print(f"✓ User-friendly report generated: {report_file}")
        except Exception as e:
            print(f"Note: Could not auto-generate report: {e}")

    return detailed_results


def main():
    """Main entry point for multi-agent exploration script."""
    parser = argparse.ArgumentParser(
        description='Run multi-agent UBF protein folding exploration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 10 agents
  python run_multi_agent.py ACDEFGHIKLMNPQRSTVWY --agents 10 --iterations 500

  # With custom diversity profile
  python run_multi_agent.py MVLSEDK --agents 20 --iterations 1000 --diversity aggressive

  # Save results with verbose output
  python run_multi_agent.py ACDEFGHIKLM --agents 15 --iterations 500 --output results.json --verbose

  # Small exploration for testing
  python run_multi_agent.py ACDEF --agents 5 --iterations 100
        """
    )

    parser.add_argument('sequence',
                       help='Protein amino acid sequence (single-letter codes)')
    parser.add_argument('--agents', '-a',
                       type=int,
                       default=10,
                       help='Number of agents to run in parallel (default: 10)')
    parser.add_argument('--iterations', '-i',
                       type=int,
                       default=500,
                       help='Number of exploration iterations per agent (default: 500)')
    parser.add_argument('--diversity', '-d',
                       choices=['balanced', 'cautious', 'aggressive'],
                       default='balanced',
                       help='Agent diversity profile (default: balanced)')
    parser.add_argument('--output', '-o',
                       help='Output file path for results JSON')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Print detailed progress and agent information')

    args = parser.parse_args()

    # Validate inputs
    if args.agents < 1:
        print(f"Error: Number of agents must be positive (got {args.agents})")
        sys.exit(1)

    if args.iterations < 1:
        print(f"Error: Iterations must be positive (got {args.iterations})")
        sys.exit(1)

    if not args.sequence or len(args.sequence) < 3:
        print("Error: Sequence must contain at least 3 amino acids")
        sys.exit(1)

    # Validate sequence contains only valid amino acid codes
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(c in valid_aa for c in args.sequence.upper()):
        print("Error: Sequence contains invalid amino acid codes")
        print("Valid codes: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y")
        sys.exit(1)

    # Warn about large computational load
    total_iterations = args.agents * args.iterations
    if total_iterations > 50000:
        print(f"Warning: Running {total_iterations} total iterations may take significant time.")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted by user.")
            sys.exit(0)

    try:
        results = run_multi_agent(
            sequence=args.sequence.upper(),
            agents=args.agents,
            iterations=args.iterations,
            diversity_profile=args.diversity,
            output_file=args.output,
            verbose=args.verbose
        )

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)

    except Exception as e:
        print(f"\nError during exploration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
