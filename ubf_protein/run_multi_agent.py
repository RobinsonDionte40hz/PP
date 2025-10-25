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


def assess_prediction_quality(rmsd: float, gdt_ts: Optional[float]) -> str:
    """
    Assess prediction quality based on RMSD and GDT-TS.
    
    Args:
        rmsd: Best RMSD value
        gdt_ts: Best GDT-TS score (optional)
        
    Returns:
        Quality string: 'excellent', 'good', 'acceptable', 'poor', or 'unknown'
    """
    if rmsd == float('inf') or rmsd > 100:
        return 'unknown'
    
    if gdt_ts is not None:
        # Use GDT-TS for more accurate assessment
        if rmsd < 2.0 and gdt_ts > 80:
            return 'excellent'
        elif rmsd < 3.0 and gdt_ts > 70:
            return 'good'
        elif rmsd < 5.0 and gdt_ts > 50:
            return 'acceptable'
        else:
            return 'poor'
    else:
        # Fallback to RMSD-only assessment
        if rmsd < 2.0:
            return 'excellent'
        elif rmsd < 3.0:
            return 'good'
        elif rmsd < 5.0:
            return 'acceptable'
        else:
            return 'poor'


def run_multi_agent(sequence: str,
                   agents: int,
                   iterations: int,
                   diversity_profile: str = "balanced",
                   output_file: Optional[str] = None,
                   verbose: bool = False,
                   native_pdb_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run multi-agent parallel exploration for specified iterations.

    Args:
        sequence: Protein amino acid sequence
        agents: Number of agents to run in parallel
        iterations: Number of exploration iterations per agent
        diversity_profile: Agent diversity profile ("balanced", "cautious", "aggressive")
        output_file: Path to save results JSON (optional)
        verbose: Print detailed progress information
        native_pdb_id: Optional PDB ID for native structure validation

    Returns:
        Dictionary with exploration results
    """
    print(f"Starting multi-agent exploration")
    print(f"Sequence length: {len(sequence)} residues")
    print(f"Agents: {agents}")
    print(f"Iterations per agent: {iterations}")
    print(f"Total iterations: {agents * iterations}")
    print(f"Diversity profile: {diversity_profile}")
    if native_pdb_id:
        print(f"Native structure: {native_pdb_id} (RMSD validation enabled)")

    # Load native structure if provided
    native_structure = None
    if native_pdb_id:
        try:
            from ubf_protein.rmsd_calculator import NativeStructureLoader
            from ubf_protein.models import Conformation
            
            print(f"\nLoading native structure {native_pdb_id}...")
            loader = NativeStructureLoader()
            native_struct_obj = loader.load_from_pdb_id(native_pdb_id, ca_only=True)
            
            # Convert to Conformation
            native_structure = Conformation(
                conformation_id=f"native_{native_pdb_id}",
                sequence=native_struct_obj.sequence,
                atom_coordinates=native_struct_obj.ca_coords,
                energy=-100.0,  # Placeholder
                rmsd_to_native=0.0,
                secondary_structure=['C'] * len(native_struct_obj.sequence),
                phi_angles=[0.0] * len(native_struct_obj.sequence),
                psi_angles=[0.0] * len(native_struct_obj.sequence),
                available_move_types=[],
                structural_constraints={},
                native_structure_ref=native_pdb_id
            )
            print(f"✓ Native structure loaded: {len(native_structure.sequence)} residues")
        except Exception as e:
            print(f"⚠ Warning: Could not load native structure: {e}")
            print("  Continuing without RMSD validation")
            native_structure = None

    # Initialize coordinator
    print("\nInitializing multi-agent coordinator...")
    coordinator = MultiAgentCoordinator(sequence)
    
    # Initialize agents with diversity
    print(f"Initializing {agents} agents with {diversity_profile} diversity...")
    
    # If native structure provided, create agents manually with native structure
    if native_structure:
        from ubf_protein.protein_agent import ProteinAgent
        agent_list = []
        for i in range(agents):
            # Determine agent type based on diversity profile
            if diversity_profile == "balanced":
                if i < agents // 3:
                    freq, coh = 5.0, 0.4  # Cautious
                elif i < 2 * agents // 3:
                    freq, coh = 9.0, 0.6  # Balanced
                else:
                    freq, coh = 13.0, 0.8  # Aggressive
            else:
                freq, coh = 9.0, 0.6  # Default balanced
            
            agent = ProteinAgent(
                protein_sequence=sequence,
                initial_frequency=freq,
                initial_coherence=coh,
                native_structure=native_structure  # Pass native structure
            )
            agent_list.append(agent)
        coordinator._agents = agent_list
    else:
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

    # Extract energy components from best conformation if available
    energy_components = None
    if results.best_conformation and results.best_conformation.energy_components:
        energy_components = results.best_conformation.energy_components
    
    # Extract validation metrics if available
    best_gdt_ts = results.best_gdt_ts if hasattr(results, 'best_gdt_ts') else None
    best_tm_score = results.best_tm_score if hasattr(results, 'best_tm_score') else None
    validation_quality = results.validation_quality if hasattr(results, 'validation_quality') else None
    
    # If not in results, assess quality from metrics
    if validation_quality is None and results.best_rmsd != float('inf'):
        validation_quality = assess_prediction_quality(results.best_rmsd, best_gdt_ts)
    
    # Determine if energy is physically realistic
    energy_is_negative = results.best_energy < 0
    energy_status = "✓ Negative (folded)" if energy_is_negative else "⚠ Positive (unfolded)"

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
            'avg_time_per_iteration_ms': (results.total_runtime_seconds / results.total_iterations) * 1000 if results.total_iterations > 0 else 0,
            'native_structure': native_pdb_id if native_pdb_id else None
        },
        'best_results': {
            'best_energy': results.best_energy,
            'energy_is_negative': energy_is_negative,
            'best_rmsd': results.best_rmsd,
            'total_conformations_explored': results.total_conformations_explored,
            'best_gdt_ts': best_gdt_ts,
            'best_tm_score': best_tm_score,
            'validation_quality': validation_quality
        },
        'energy_components': energy_components if energy_components else {},
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
                'best_gdt_ts': getattr(m, 'best_gdt_ts_score', None),
                'best_tm_score': getattr(m, 'best_tm_score', None),
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
    print(f"Best Energy Found: {results.best_energy:.2f} kcal/mol {energy_status}")
    
    # Show energy components if available
    if energy_components:
        print(f"\nEnergy Components Breakdown:")
        if 'bond_energy' in energy_components:
            print(f"  Bond Energy:          {energy_components['bond_energy']:.2f} kcal/mol")
        if 'angle_energy' in energy_components:
            print(f"  Angle Energy:         {energy_components['angle_energy']:.2f} kcal/mol")
        if 'dihedral_energy' in energy_components:
            print(f"  Dihedral Energy:      {energy_components['dihedral_energy']:.2f} kcal/mol")
        if 'vdw_energy' in energy_components:
            print(f"  Van der Waals:        {energy_components['vdw_energy']:.2f} kcal/mol")
        if 'electrostatic_energy' in energy_components:
            print(f"  Electrostatic:        {energy_components['electrostatic_energy']:.2f} kcal/mol")
        if 'hbond_energy' in energy_components:
            print(f"  Hydrogen Bonds:       {energy_components['hbond_energy']:.2f} kcal/mol")
    
    print(f"\nStructural Metrics:")
    print(f"  Best RMSD: {results.best_rmsd:.2f} Å")
    
    # Show validation metrics if available
    if best_gdt_ts is not None:
        print(f"  Best GDT-TS: {best_gdt_ts:.1f}")
    if best_tm_score is not None:
        print(f"  Best TM-score: {best_tm_score:.3f}")
    if validation_quality and validation_quality != 'unknown':
        quality_symbol = {'excellent': '★★★★', 'good': '★★★', 'acceptable': '★★', 'poor': '★'}.get(validation_quality, '')
        print(f"  Quality Assessment: {validation_quality.upper()} {quality_symbol}")
    
    print(f"\nExploration Stats:")
    print(f"  Total Conformations Explored: {results.total_conformations_explored}")
    print(f"  Total Runtime: {results.total_runtime_seconds:.2f} seconds")
    print(f"  Avg Time per Iteration: {(results.total_runtime_seconds / results.total_iterations) * 1000:.2f} ms")
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

  # With native structure validation
  python run_multi_agent.py ACDEFGHIKLM --agents 10 --iterations 500 --native 1UBQ

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
    parser.add_argument('--native', '-n',
                       help='PDB ID of native structure for RMSD validation (e.g., 1UBQ)')

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
            verbose=args.verbose,
            native_pdb_id=args.native
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
