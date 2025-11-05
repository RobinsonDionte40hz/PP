"""
Folding Determinism Experiment - Multi-Trial THz Signature Analysis

Tests the hypothesis: If protein folding is deterministic, multiple independent 
trials should converge to the same THz signature at energy minima.

This is the "catch the protein lying" test - if folding pathways are truly 
deterministic (Anfinsen's dogma), all agents should find conformations with 
the same vibrational "fingerprint."
"""

import sys
import os
import argparse
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ubf_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ubf_dir)
sys.path.insert(0, project_root)

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.signature_analysis import create_determinism_tester, DeterminismScore
from ubf_protein.vibrational_analysis import THzSpectrum


def run_single_trial(
    sequence: str, 
    trial_number: int, 
    iterations: int,
    native_pdb: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a single folding trial and collect THz signatures.
    
    Args:
        sequence: Protein sequence
        trial_number: Trial identifier for random seed
        iterations: Number of exploration iterations
        native_pdb: Optional native structure PDB code
        
    Returns:
        Dictionary with trial results
    """
    print(f"   Trial {trial_number + 1}: ", end="", flush=True)
    
    # Create agent with unique seed
    agent = ProteinAgent(
        protein_sequence=sequence,
        initial_frequency=9.0,
        initial_coherence=0.6,
        enable_visualization=False
    )
    
    # Run exploration
    start_time = time.time()
    for _ in range(iterations):
        try:
            agent.explore_step()
        except Exception as e:
            print(f"Error: {e}")
            break
    
    elapsed = time.time() - start_time
    
    # Get results
    metrics = agent.get_exploration_metrics()
    thz_history = agent.get_thz_signature_history()
    
    # Extract frequencies and intensities from all signatures
    all_frequencies = []
    all_intensities = []
    for spectrum in thz_history:
        all_frequencies.append(spectrum.frequencies)
        all_intensities.append(spectrum.intensities)
    
    print(f"✓ ({len(thz_history)} signatures, {elapsed:.1f}s, E={metrics['best_energy']:.2f})")
    
    return {
        'trial_number': trial_number,
        'sequence': sequence,
        'iterations': iterations,
        'elapsed_time': elapsed,
        'best_energy': metrics['best_energy'],
        'best_rmsd': metrics.get('best_rmsd', None),
        'signatures_recorded': len(thz_history),
        'thz_frequencies': all_frequencies,
        'thz_intensities': all_intensities,
        'conformations_explored': metrics['conformations_explored'],
        'stuck_count': metrics['stuck_in_minima_count']
    }


def run_determinism_experiment(
    sequence: str,
    n_trials: int = 100,
    iterations_per_trial: int = 500,
    native_pdb: Optional[str] = None,
    output_file: Optional[str] = None
) -> DeterminismScore:
    """
    Run full determinism experiment with multiple trials.
    
    Args:
        sequence: Protein sequence to test
        n_trials: Number of independent trials
        iterations_per_trial: Exploration iterations per trial
        native_pdb: Optional native structure for validation
        output_file: Optional JSON output file
        
    Returns:
        DeterminismScore with analysis results
    """
    print("=" * 70)
    print("FOLDING DETERMINISM EXPERIMENT")
    print("=" * 70)
    print(f"\nSequence: {sequence} ({len(sequence)} residues)")
    print(f"Trials: {n_trials}")
    print(f"Iterations per trial: {iterations_per_trial}")
    if native_pdb:
        print(f"Native structure: {native_pdb}")
    print()
    
    # Run all trials
    print(f"Running {n_trials} independent folding trials...")
    print()
    
    all_trial_data = []
    all_frequencies = []
    all_intensities = []
    
    for trial_num in range(n_trials):
        trial_data = run_single_trial(
            sequence, 
            trial_num, 
            iterations_per_trial,
            native_pdb
        )
        all_trial_data.append(trial_data)
        
        # Collect all THz signatures from this trial
        all_frequencies.extend(trial_data['thz_frequencies'])
        all_intensities.extend(trial_data['thz_intensities'])
    
    print()
    print(f"✅ All trials complete!")
    print(f"   Total signatures collected: {len(all_frequencies)}")
    
    # Analyze determinism
    print()
    print("Analyzing signature clustering...")
    
    tester = create_determinism_tester(similarity_threshold=0.7)
    determinism_score = tester.calculate_determinism_score(
        all_frequencies,
        all_intensities
    )
    
    # Calculate additional statistics
    avg_energy = sum(t['best_energy'] for t in all_trial_data) / n_trials
    avg_signatures = sum(t['signatures_recorded'] for t in all_trial_data) / n_trials
    avg_conformations = sum(t['conformations_explored'] for t in all_trial_data) / n_trials
    
    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Determinism Analysis:")
    print(f"   Trials: {determinism_score.n_trials}")
    print(f"   Signature clusters: {determinism_score.n_clusters}")
    print(f"   Largest cluster: {determinism_score.largest_cluster_size} trials ({determinism_score.convergence_ratio:.1%})")
    print(f"   Intra-cluster similarity: {determinism_score.average_intra_cluster_similarity:.3f}")
    print(f"   Determinism score: {determinism_score.determinism_score:.3f}")
    print()
    print(f"Interpretation:")
    print(f"   {determinism_score.interpret()}")
    print()
    print(f"Trial Statistics:")
    print(f"   Average best energy: {avg_energy:.2f} kcal/mol")
    print(f"   Average signatures/trial: {avg_signatures:.1f}")
    print(f"   Average conformations explored: {avg_conformations:.1f}")
    print()
    
    # Save results
    if output_file:
        results = {
            'experiment_config': {
                'sequence': sequence,
                'n_trials': n_trials,
                'iterations_per_trial': iterations_per_trial,
                'native_pdb': native_pdb
            },
            'determinism_analysis': {
                'n_trials': determinism_score.n_trials,
                'n_clusters': determinism_score.n_clusters,
                'largest_cluster_size': determinism_score.largest_cluster_size,
                'convergence_ratio': determinism_score.convergence_ratio,
                'average_intra_cluster_similarity': determinism_score.average_intra_cluster_similarity,
                'determinism_score': determinism_score.determinism_score,
                'interpretation': determinism_score.interpret()
            },
            'trial_statistics': {
                'avg_best_energy': avg_energy,
                'avg_signatures_per_trial': avg_signatures,
                'avg_conformations_explored': avg_conformations
            },
            'all_trial_data': all_trial_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        print()
    
    return determinism_score


def main():
    """Run determinism experiment from command line."""
    parser = argparse.ArgumentParser(
        description="Test protein folding determinism using THz signature clustering"
    )
    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Protein amino acid sequence (e.g., MQIFVKTLTGKT)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of independent trials (default: 100)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=500,
        help='Exploration iterations per trial (default: 500)'
    )
    parser.add_argument(
        '--native',
        type=str,
        default=None,
        help='Native structure PDB code for validation (e.g., 1UBQ)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"determinism_{len(args.sequence)}res_{timestamp}.json"
    
    try:
        score = run_determinism_experiment(
            sequence=args.sequence,
            n_trials=args.trials,
            iterations_per_trial=args.iterations,
            native_pdb=args.native,
            output_file=args.output
        )
        
        # Exit code based on determinism level
        if score.determinism_score > 0.8:
            return 0  # Strong determinism
        elif score.determinism_score > 0.6:
            return 0  # Moderate determinism
        else:
            return 1  # Weak/stochastic
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
