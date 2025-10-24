#!/usr/bin/env python3
"""
Validation script for UBF protein system.

This script validates the UBF protein folding system by comparing predicted
conformations against experimental data and native structures.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.models import ExplorationResults


def load_native_structure(pdb_file: str) -> Optional[Dict[str, Any]]:
    """
    Load native protein structure from PDB file.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Dictionary with native structure data or None if loading fails
    """
    try:
        # Placeholder implementation - in real system would parse PDB
        # For now, return mock data
        return {
            'sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR',
            'length': 141,
            'experimental_stability': {
                'melting_temperature': 50.5,  # °C
                'delta_g_unfolding': -5.2,     # kcal/mol
                'stability_class': 'stable'
            }
        }
    except Exception as e:
        print(f"Error loading PDB file {pdb_file}: {e}")
        return None


def calculate_validation_metrics(results: ExplorationResults,
                               native_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate validation metrics comparing predictions to experimental data.

    Args:
        results: Exploration results from multi-agent system
        native_data: Native structure and experimental data

    Returns:
        Dictionary with validation metrics
    """
    metrics = {
        'prediction_accuracy': {},
        'convergence_quality': {},
        'exploration_efficiency': {},
        'learning_assessment': {}
    }

    # Prediction accuracy metrics
    if results.best_conformation and native_data:
        # RMSD accuracy (lower is better)
        predicted_rmsd = results.best_rmsd
        native_rmsd_baseline = 2.0  # Typical RMSD for good predictions
        rmsd_accuracy = max(0.0, (native_rmsd_baseline - predicted_rmsd) / native_rmsd_baseline)

        # Energy accuracy (compare to experimental stability)
        predicted_energy = results.best_energy
        experimental_dg = native_data.get('experimental_stability', {}).get('delta_g_unfolding', 0)
        energy_accuracy = 1.0 - abs(predicted_energy - experimental_dg) / 10.0  # ±10 kcal/mol tolerance
        energy_accuracy = max(0.0, min(1.0, energy_accuracy))

        metrics['prediction_accuracy'] = {
            'rmsd_accuracy': rmsd_accuracy,
            'energy_accuracy': energy_accuracy,
            'overall_accuracy': (rmsd_accuracy + energy_accuracy) / 2.0
        }

    # Convergence quality metrics
    total_iterations = results.total_iterations
    best_energy_iterations = len([m for m in results.agent_metrics if m.best_energy_found < -10])  # Arbitrary threshold
    convergence_rate = best_energy_iterations / max(1, len(results.agent_metrics))

    metrics['convergence_quality'] = {
        'total_iterations': total_iterations,
        'convergence_rate': convergence_rate,
        'avg_iterations_per_agent': total_iterations / max(1, len(results.agent_metrics))
    }

    # Exploration efficiency metrics
    total_conformations = results.total_conformations_explored
    unique_conformations = total_conformations  # Placeholder - would calculate actual unique states
    exploration_efficiency = unique_conformations / max(1, total_conformations)

    metrics['exploration_efficiency'] = {
        'total_conformations_explored': total_conformations,
        'exploration_efficiency': exploration_efficiency,
        'avg_conformations_per_agent': total_conformations / max(1, len(results.agent_metrics))
    }

    # Learning assessment metrics
    avg_learning_improvement = sum(m.learning_improvement for m in results.agent_metrics) / max(1, len(results.agent_metrics))
    collective_benefit = results.collective_learning_benefit

    metrics['learning_assessment'] = {
        'avg_learning_improvement': avg_learning_improvement,
        'collective_learning_benefit': collective_benefit,
        'learning_effectiveness': (avg_learning_improvement + collective_benefit) / 2.0
    }

    return metrics


def run_validation(sequence: str,
                  native_pdb: Optional[str] = None,
                  agents: int = 10,
                  iterations: int = 100,
                  output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run complete validation pipeline.

    Args:
        sequence: Protein amino acid sequence
        native_pdb: Path to native PDB file (optional)
        agents: Number of agents to use
        iterations: Number of iterations per agent
        output_file: Path to save results (optional)

    Returns:
        Validation results dictionary
    """
    print(f"Starting UBF validation for protein sequence (length: {len(sequence)})")
    print(f"Using {agents} agents for {iterations} iterations each")

    # Load native structure if provided
    native_data = None
    if native_pdb:
        print(f"Loading native structure from {native_pdb}")
        native_data = load_native_structure(native_pdb)
        if native_data:
            print(f"Native structure loaded: {native_data['length']} residues")
        else:
            print("Warning: Could not load native structure")

    # Initialize and run multi-agent exploration
    print("Initializing multi-agent coordinator...")
    coordinator = MultiAgentCoordinator(sequence)
    coordinator.initialize_agents(agents)

    print("Running exploration...")
    results = coordinator.run_parallel_exploration(iterations)

    # Calculate validation metrics
    print("Calculating validation metrics...")
    validation_metrics = calculate_validation_metrics(results, native_data or {})

    # Prepare final results
    validation_results = {
        'metadata': {
            'sequence_length': len(sequence),
            'agents_used': agents,
            'iterations_per_agent': iterations,
            'native_structure_provided': native_pdb is not None,
            'validation_timestamp': '2025-01-01T00:00:00Z'  # Would use actual timestamp
        },
        'exploration_results': {
            'best_energy': results.best_energy,
            'best_rmsd': results.best_rmsd,
            'total_conformations': results.total_conformations_explored,
            'collective_learning_benefit': results.collective_learning_benefit,
            'total_runtime_seconds': results.total_runtime_seconds
        },
        'validation_metrics': validation_metrics,
        'agent_summary': {
            'total_agents': len(results.agent_metrics),
            'avg_learning_improvement': sum(m.learning_improvement for m in results.agent_metrics) / max(1, len(results.agent_metrics)),
            'total_memories_created': sum(m.memories_created for m in results.agent_metrics),
            'avg_decision_time_ms': sum(m.avg_decision_time_ms for m in results.agent_metrics) / max(1, len(results.agent_metrics))
        }
    }

    # Save results if requested
    if output_file:
        print(f"Saving validation results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Best Energy Found: {results.best_energy:.2f} kcal/mol")
    print(f"Best RMSD: {results.best_rmsd:.2f} Å")
    print(f"Total Conformations Explored: {results.total_conformations_explored}")
    print(f"Overall Accuracy: {validation_metrics['prediction_accuracy'].get('overall_accuracy', 0):.2%}")
    print(f"Learning Effectiveness: {validation_metrics['learning_assessment']['learning_effectiveness']:.2%}")

    return validation_results


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(description='Validate UBF protein folding system')
    parser.add_argument('sequence', help='Protein amino acid sequence')
    parser.add_argument('--native-pdb', help='Path to native PDB structure file')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents to use (default: 10)')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per agent (default: 100)')
    parser.add_argument('--output', help='Output file for validation results (JSON)')

    args = parser.parse_args()

    try:
        results = run_validation(
            sequence=args.sequence,
            native_pdb=args.native_pdb,
            agents=args.agents,
            iterations=args.iterations,
            output_file=args.output
        )

        # Exit with success/failure based on accuracy
        overall_accuracy = results['validation_metrics']['prediction_accuracy'].get('overall_accuracy', 0)
        if overall_accuracy > 0.7:  # 70% accuracy threshold
            print("✓ Validation PASSED")
            sys.exit(0)
        else:
            print("✗ Validation FAILED - Low accuracy")
            sys.exit(1)

    except Exception as e:
        print(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()