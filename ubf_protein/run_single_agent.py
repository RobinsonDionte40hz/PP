#!/usr/bin/env python3
"""
Single agent exploration script for UBF protein system.

This script runs a single autonomous protein agent to explore conformational
space and predict protein structure using consciousness-based navigation.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import AdaptiveConfig, ProteinSizeClass, ConformationalOutcome


def determine_protein_size_class(sequence: str) -> ProteinSizeClass:
    """
    Determine protein size class from sequence length.

    Args:
        sequence: Protein amino acid sequence

    Returns:
        ProteinSizeClass enum value
    """
    residue_count = len(sequence)
    if residue_count < 50:
        return ProteinSizeClass.SMALL
    elif residue_count <= 150:
        return ProteinSizeClass.MEDIUM
    else:
        return ProteinSizeClass.LARGE


def create_adaptive_config(sequence: str) -> AdaptiveConfig:
    """
    Create adaptive configuration based on protein size.

    Args:
        sequence: Protein amino acid sequence

    Returns:
        AdaptiveConfig with size-appropriate parameters
    """
    residue_count = len(sequence)
    size_class = determine_protein_size_class(sequence)

    # Size-specific parameter scaling
    if size_class == ProteinSizeClass.SMALL:
        stuck_window = 20
        stuck_threshold = 5.0
        max_memories = 30
        convergence_energy = 5.0
        convergence_rmsd = 1.5
        max_iterations = 1000
    elif size_class == ProteinSizeClass.MEDIUM:
        stuck_window = 30
        stuck_threshold = 10.0
        max_memories = 50
        convergence_energy = 10.0
        convergence_rmsd = 2.0
        max_iterations = 2000
    else:  # LARGE
        stuck_window = 40
        stuck_threshold = 15.0
        max_memories = 75
        convergence_energy = 15.0
        convergence_rmsd = 3.0
        max_iterations = 5000

    return AdaptiveConfig(
        size_class=size_class,
        residue_count=residue_count,
        initial_frequency_range=(3.0, 15.0),
        initial_coherence_range=(0.2, 1.0),
        stuck_detection_window=stuck_window,
        stuck_detection_threshold=stuck_threshold,
        memory_significance_threshold=0.3,
        max_memories_per_agent=max_memories,
        convergence_energy_threshold=convergence_energy,
        convergence_rmsd_threshold=convergence_rmsd,
        max_iterations=max_iterations,
        checkpoint_interval=100
    )


def run_single_agent(sequence: str,
                    iterations: int,
                    initial_frequency: float = 9.0,
                    initial_coherence: float = 0.6,
                    output_file: Optional[str] = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    Run single agent exploration for specified iterations.

    Args:
        sequence: Protein amino acid sequence
        iterations: Number of exploration iterations to perform
        initial_frequency: Initial consciousness frequency (3-15 Hz)
        initial_coherence: Initial consciousness coherence (0.2-1.0)
        output_file: Path to save results JSON (optional)
        verbose: Print detailed progress information

    Returns:
        Dictionary with exploration results
    """
    print(f"Starting single agent exploration")
    print(f"Sequence length: {len(sequence)} residues")
    print(f"Iterations: {iterations}")
    print(f"Initial consciousness: frequency={initial_frequency} Hz, coherence={initial_coherence}")

    # Create adaptive configuration
    config = create_adaptive_config(sequence)
    print(f"Protein size class: {config.size_class.value}")
    print(f"Adaptive parameters:")
    print(f"  - Stuck detection window: {config.stuck_detection_window}")
    print(f"  - Max memories: {config.max_memories_per_agent}")
    print(f"  - Convergence thresholds: energy={config.convergence_energy_threshold}, RMSD={config.convergence_rmsd_threshold}")

    # Initialize agent
    print("\nInitializing protein agent...")
    agent = ProteinAgent(
        protein_sequence=sequence,
        initial_frequency=initial_frequency,
        initial_coherence=initial_coherence,
        adaptive_config=config
    )

    # Track exploration metrics
    outcomes = []
    best_energy = float('inf')
    best_rmsd = float('inf')
    best_iteration = 0
    start_time = time.time()

    print("\nStarting exploration...")
    print("-" * 60)

    # Run exploration
    for i in range(iterations):
        outcome = agent.explore_step()
        outcomes.append(outcome)

        # Update best found
        current_energy = outcome.new_conformation.energy
        current_rmsd = outcome.new_conformation.rmsd_to_native or float('inf')
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_rmsd = current_rmsd
            best_iteration = i

        # Print progress
        if verbose and (i + 1) % 10 == 0:
            consciousness = agent.get_consciousness_state()
            behavioral = agent.get_behavioral_state()
            print(f"Iteration {i+1}/{iterations}: "
                  f"Energy={current_energy:.2f}, "
                  f"RMSD={current_rmsd:.2f}, "
                  f"F={consciousness.get_frequency():.1f}Hz, "
                  f"C={consciousness.get_coherence():.2f}")
        elif (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{iterations} iterations ({(i+1)/iterations*100:.1f}%)")

    end_time = time.time()
    total_runtime = end_time - start_time

    print("-" * 60)
    print("Exploration complete!")

    # Calculate summary metrics
    agent_metrics_dict = agent.get_exploration_metrics()
    
    # Calculate learning improvement
    early_rmsds = [o.new_conformation.rmsd_to_native or 0.0 for o in outcomes[:20]] if len(outcomes) >= 20 else []
    late_rmsds = [o.new_conformation.rmsd_to_native or 0.0 for o in outcomes[-20:]] if len(outcomes) >= 20 else []
    
    learning_improvement = 0.0
    if early_rmsds and late_rmsds:
        early_avg = sum(early_rmsds) / len(early_rmsds)
        late_avg = sum(late_rmsds) / len(late_rmsds)
        if early_avg > 0:
            learning_improvement = ((early_avg - late_avg) / early_avg) * 100

    # Compile results
    results = {
        'metadata': {
            'sequence': sequence,
            'sequence_length': len(sequence),
            'protein_size_class': config.size_class.value,
            'iterations_performed': iterations,
            'total_runtime_seconds': total_runtime,
            'avg_time_per_iteration_ms': (total_runtime / iterations) * 1000,
            'initial_consciousness': {
                'frequency': initial_frequency,
                'coherence': initial_coherence
            }
        },
        'best_results': {
            'best_energy': best_energy,
            'best_rmsd': best_rmsd,
            'best_iteration': best_iteration
        },
        'agent_metrics': {
            'total_iterations': agent_metrics_dict['iterations_completed'],
            'conformations_explored': agent_metrics_dict['conformations_explored'],
            'memories_created': agent_metrics_dict['memories_created'],
            'stuck_count': agent_metrics_dict['stuck_in_minima_count'],
            'escape_count': agent_metrics_dict['successful_escapes'],
            'avg_decision_time_ms': agent_metrics_dict['avg_decision_time_ms'],
            'learning_improvement_percent': learning_improvement
        },
        'final_state': {
            'consciousness': {
                'frequency': agent.get_consciousness_state().get_frequency(),
                'coherence': agent.get_consciousness_state().get_coherence()
            },
            'behavioral': {
                'exploration_energy': agent.get_behavioral_state().get_exploration_energy(),
                'structural_focus': agent.get_behavioral_state().get_structural_focus(),
                'hydrophobic_drive': agent.get_behavioral_state().get_hydrophobic_drive(),
                'risk_tolerance': agent.get_behavioral_state().get_risk_tolerance(),
                'native_state_ambition': agent.get_behavioral_state().get_native_state_ambition()
            }
        }
    }

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Best Energy Found: {best_energy:.2f} kcal/mol (iteration {best_iteration})")
    print(f"Best RMSD: {best_rmsd:.2f} Å")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Avg Time per Iteration: {(total_runtime / iterations) * 1000:.2f} ms")
    print(f"Conformations Explored: {agent_metrics_dict['conformations_explored']}")
    print(f"Memories Created: {agent_metrics_dict['memories_created']}")
    print(f"Learning Improvement: {learning_improvement:.1f}%")
    print(f"Stuck/Escape Events: {agent_metrics_dict['stuck_in_minima_count']}/{agent_metrics_dict['successful_escapes']}")
    print(f"Final Consciousness: F={agent.get_consciousness_state().get_frequency():.1f}Hz, "
          f"C={agent.get_consciousness_state().get_coherence():.2f}")
    print("=" * 60)

    # Save to file if requested
    if output_file:
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully!")

    return results


def main():
    """Main entry point for single agent exploration script."""
    parser = argparse.ArgumentParser(
        description='Run single agent UBF protein folding exploration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with sequence
  python run_single_agent.py ACDEFGHIKLMNPQRSTVWY --iterations 500

  # With custom consciousness parameters
  python run_single_agent.py MVLSEDK --iterations 1000 --frequency 12.0 --coherence 0.8

  # Save results to file with verbose output
  python run_single_agent.py ACDEFGHIKLM --iterations 500 --output results.json --verbose
        """
    )

    parser.add_argument('sequence',
                       help='Protein amino acid sequence (single-letter codes)')
    parser.add_argument('--iterations', '-i',
                       type=int,
                       default=500,
                       help='Number of exploration iterations (default: 500)')
    parser.add_argument('--frequency', '-f',
                       type=float,
                       default=9.0,
                       help='Initial consciousness frequency in Hz, range 3-15 (default: 9.0)')
    parser.add_argument('--coherence', '-c',
                       type=float,
                       default=0.6,
                       help='Initial consciousness coherence, range 0.2-1.0 (default: 0.6)')
    parser.add_argument('--output', '-o',
                       help='Output file path for results JSON')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Print detailed progress information')

    args = parser.parse_args()

    # Validate inputs
    if args.frequency < 3.0 or args.frequency > 15.0:
        print(f"Error: Frequency must be between 3.0 and 15.0 Hz (got {args.frequency})")
        sys.exit(1)

    if args.coherence < 0.2 or args.coherence > 1.0:
        print(f"Error: Coherence must be between 0.2 and 1.0 (got {args.coherence})")
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

    try:
        results = run_single_agent(
            sequence=args.sequence.upper(),
            iterations=args.iterations,
            initial_frequency=args.frequency,
            initial_coherence=args.coherence,
            output_file=args.output,
            verbose=args.verbose
        )

        # Exit with success
        sys.exit(0)

    except Exception as e:
        print(f"\nError during exploration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
