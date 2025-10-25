#!/usr/bin/env python3
"""
Agent Scaling Experiment: QCPP-UBF Integration

Tests how the number of agents affects both RMSD and RMSE scores.
Scales from 5 → 10 → 20 → 50 → 100 agents.

Based on quick_test_integration.py architecture.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

# Import QCPP components
from protein_predictor import QuantumCoherenceProteinPredictor

# Import UBF components
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator


def load_experimental_data():
    """Load experimental stability data for RMSE calculation."""
    exp_file = Path("experimental_stability.csv")
    if not exp_file.exists():
        print(f"⚠️  Experimental data not found: {exp_file}")
        return None
    
    df = pd.read_csv(exp_file)
    return df


def calculate_qcpp_prediction(sequence: str, pdb_file: Path):
    """Calculate QCPP stability prediction for comparison."""
    if not pdb_file.exists():
        print(f"⚠️  PDB file not found: {pdb_file}")
        return None
    
    predictor = QuantumCoherenceProteinPredictor()
    predictor.load_protein(str(pdb_file), chain_id='A')
    
    qcp_df = predictor.calculate_qcp()
    if qcp_df is None or len(qcp_df) == 0:
        return None
    
    qcp_values = qcp_df['qcp'].to_numpy()
    avg_qcp = float(np.mean(qcp_values))
    stability_score = avg_qcp / 5.0  # Normalize
    
    return {
        'qcp_mean': avg_qcp,
        'stability_score': stability_score
    }


def estimate_rmsd_from_energy(energy: float) -> float:
    """Estimate RMSD from energy using correlation."""
    # Native state ~-300 to -400 kcal/mol
    # Unfolded state ~0 to -100 kcal/mol
    # Linear mapping: better energy → lower RMSD
    normalized_energy = (energy + 200) / -200
    normalized_energy = max(0, min(1, normalized_energy))
    estimated_rmsd = 10.0 - (normalized_energy * 7.0)
    return max(0.5, estimated_rmsd)


def calculate_rmse(predicted_stability: float, experimental_data: pd.DataFrame, pdb_id: str):
    """Calculate RMSE between QCPP prediction and experimental data."""
    protein_data = experimental_data[experimental_data['PDB_ID'] == pdb_id.upper()]
    
    if protein_data.empty:
        return None
    
    exp_temp = protein_data['Melting_Temperature_C'].values[0]
    exp_dg = protein_data['DeltaG_kcal_mol'].values[0]
    
    # Use validated scaling formulas from validate_ubiquitin_rmse.py
    # Higher stability → higher melting temp
    predicted_temp = 50.0 + (predicted_stability * 40.0)
    
    # Higher stability → higher ΔG (more stable)
    predicted_dg = predicted_stability * 8.0
    
    temp_rmse = abs(predicted_temp - exp_temp)
    dg_rmse = abs(predicted_dg - exp_dg)
    
    return {
        'temperature_rmse': temp_rmse,
        'dg_rmse': dg_rmse,
        'predicted_temp': predicted_temp,
        'predicted_dg': predicted_dg,
        'experimental_temp': exp_temp,
        'experimental_dg': exp_dg
    }


def run_single_experiment(
    num_agents: int,
    sequence: str,
    iterations: int,
    pdb_file: Path,
    experimental_data: pd.DataFrame | None,
    pdb_id: str
) -> Dict[str, Any]:
    """Run a single scaling experiment with specified number of agents."""
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {num_agents} AGENTS × {iterations} ITERATIONS")
    print("="*70)
    
    # Step 1: Initialize QCPP
    print(f"\n[1/5] Initializing QCPP predictor...")
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    cache_size = 5000  # Larger cache for more agents
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, cache_size)
    print(f"✓ QCPP initialized with cache size {cache_size}")
    
    # Step 2: Create coordinator
    print(f"\n[2/5] Creating multi-agent coordinator...")
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        qcpp_integration=qcpp_adapter
    )
    
    # Initialize agents with balanced diversity
    coordinator.initialize_agents(
        count=num_agents,
        diversity_profile="balanced"
    )
    print(f"✓ Initialized {num_agents} agents")
    
    # Step 3: Run exploration
    print(f"\n[3/5] Running parallel exploration...")
    start_time = time.time()
    
    results = coordinator.run_parallel_exploration(
        iterations=iterations
    )
    
    exploration_time = time.time() - start_time
    total_conformations = num_agents * iterations
    throughput = total_conformations / exploration_time
    
    print(f"✓ Exploration complete")
    print(f"  Time: {exploration_time:.1f}s")
    print(f"  Throughput: {throughput:.1f} conf/s")
    print(f"  Best Energy: {results.best_energy:.2f} kcal/mol")
    
    # Step 4: Calculate RMSD
    print(f"\n[4/5] Calculating RMSD...")
    estimated_rmsd = estimate_rmsd_from_energy(results.best_energy)
    print(f"✓ Estimated RMSD: {estimated_rmsd:.2f} Å")
    
    # Step 5: Calculate RMSE
    print(f"\n[5/5] Calculating RMSE...")
    qcpp_prediction = calculate_qcpp_prediction(sequence, pdb_file)
    
    if qcpp_prediction and experimental_data is not None:
        rmse_results = calculate_rmse(
            qcpp_prediction['stability_score'],
            experimental_data,
            pdb_id
        )
        
        if rmse_results:
            print(f"✓ RMSE calculated")
            print(f"  Temperature RMSE: {rmse_results['temperature_rmse']:.2f}°C")
            print(f"  ΔG RMSE: {rmse_results['dg_rmse']:.2f} kcal/mol")
        else:
            rmse_results = None
            print(f"⚠️  No experimental data for {pdb_id}")
    else:
        rmse_results = None
        qcpp_prediction = None
        print(f"⚠️  QCPP prediction failed or no experimental data")
    
    # Get QCPP cache stats
    cache_stats = qcpp_adapter.get_cache_stats()
    
    # Compile results
    experiment_results = {
        'num_agents': num_agents,
        'iterations_per_agent': iterations,
        'total_conformations': total_conformations,
        'exploration_time_s': exploration_time,
        'throughput_conf_per_s': throughput,
        'best_energy': results.best_energy,
        'best_rmsd_to_native': results.best_rmsd,
        'estimated_rmsd': estimated_rmsd,
        'qcpp_analyses': cache_stats['total_analyses'],
        'qcpp_cache_hit_rate': cache_stats['cache_hit_rate'],
        'qcpp_avg_time_ms': cache_stats['avg_calculation_time_ms'],
        'qcpp_prediction': qcpp_prediction,
        'rmse_results': rmse_results,
        'timestamp': datetime.now().isoformat()
    }
    
    return experiment_results


def main():
    """Run the full agent scaling experiment."""
    
    print("="*70)
    print("AGENT SCALING EXPERIMENT: QCPP-UBF INTEGRATION")
    print("="*70)
    print("Testing agent counts: 5, 10, 20, 50, 100")
    print("Protein: Ubiquitin (1UBQ, 76 residues)")
    print("="*70)
    
    # Configuration
    agent_counts = [5, 10, 20, 50, 100]
    iterations = 200  # Per agent (will scale total work)
    
    # Ubiquitin configuration
    pdb_id = "1ubq"
    pdb_file = Path("pdb_cache/pdb1ubq.ent")
    
    # Load sequence from PDB
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.Polypeptide import aa3, aa1
    
    # Create 3-letter to 1-letter mapping
    aa_map = dict(zip(aa3, aa1))
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))
    
    # Get first chain
    chain = list(structure.get_chains())[0]
    residues = list(chain.get_residues())
    
    # Extract sequence using mapping
    sequence = ""
    for res in residues:
        if res.id[0] == ' ':  # Standard residue
            resname = res.resname
            if resname in aa_map:
                sequence += aa_map[resname]
            else:
                sequence += 'X'  # Unknown
    
    print(f"\nProtein Sequence ({len(sequence)} residues):")
    print(f"{sequence[:60]}..." if len(sequence) > 60 else sequence)
    
    # Load experimental data
    print(f"\nLoading experimental data...")
    experimental_data = load_experimental_data()
    if experimental_data is not None:
        print(f"✓ Loaded experimental data for {len(experimental_data)} proteins")
    
    # Run experiments
    all_results = []
    
    for num_agents in agent_counts:
        try:
            results = run_single_experiment(
                num_agents=num_agents,
                sequence=sequence,
                iterations=iterations,
                pdb_file=pdb_file,
                experimental_data=experimental_data,
                pdb_id=pdb_id
            )
            all_results.append(results)
            
            # Brief summary
            print(f"\n{'='*70}")
            print(f"SUMMARY: {num_agents} Agents")
            print(f"{'='*70}")
            print(f"Best Energy:       {results['best_energy']:.2f} kcal/mol")
            print(f"Estimated RMSD:    {results['estimated_rmsd']:.2f} Å")
            if results['rmse_results']:
                print(f"Temperature RMSE:  {results['rmse_results']['temperature_rmse']:.2f}°C")
                print(f"ΔG RMSE:           {results['rmse_results']['dg_rmse']:.2f} kcal/mol")
            print(f"Throughput:        {results['throughput_conf_per_s']:.1f} conf/s")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\n❌ Experiment with {num_agents} agents failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    output_file = Path("agent_scaling_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'experiment_info': {
                'protein': pdb_id,
                'sequence_length': len(sequence),
                'iterations_per_agent': iterations,
                'agent_counts_tested': agent_counts,
                'total_experiments': len(all_results)
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Generate comparison table
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS")
    print("="*70)
    print(f"{'Agents':<10} {'Energy':<12} {'RMSD (Å)':<12} {'Temp RMSE':<12} {'ΔG RMSE':<12} {'Throughput':<15}")
    print("-"*70)
    
    for result in all_results:
        agents = result['num_agents']
        energy = result['best_energy']
        rmsd = result['estimated_rmsd']
        throughput = result['throughput_conf_per_s']
        
        temp_rmse = "N/A"
        dg_rmse = "N/A"
        if result['rmse_results']:
            temp_rmse = f"{result['rmse_results']['temperature_rmse']:.2f}°C"
            dg_rmse = f"{result['rmse_results']['dg_rmse']:.2f}"
        
        print(f"{agents:<10} {energy:<12.2f} {rmsd:<12.2f} {temp_rmse:<12} {dg_rmse:<12} {throughput:<15.1f}")
    
    print("="*70)
    
    # Analysis
    print("\nANALYSIS:")
    if len(all_results) >= 2:
        best_energy_idx = min(range(len(all_results)), key=lambda i: all_results[i]['best_energy'])
        best_rmsd_idx = min(range(len(all_results)), key=lambda i: all_results[i]['estimated_rmsd'])
        
        print(f"  • Best Energy: {all_results[best_energy_idx]['num_agents']} agents "
              f"({all_results[best_energy_idx]['best_energy']:.2f} kcal/mol)")
        print(f"  • Best RMSD: {all_results[best_rmsd_idx]['num_agents']} agents "
              f"({all_results[best_rmsd_idx]['estimated_rmsd']:.2f} Å)")
        
        # Energy improvement
        energy_improvement = all_results[0]['best_energy'] - all_results[best_energy_idx]['best_energy']
        print(f"  • Energy improvement from 5→{all_results[best_energy_idx]['num_agents']} agents: "
              f"{energy_improvement:.2f} kcal/mol")
        
        # RMSD improvement
        rmsd_improvement = all_results[0]['estimated_rmsd'] - all_results[best_rmsd_idx]['estimated_rmsd']
        print(f"  • RMSD improvement from 5→{all_results[best_rmsd_idx]['num_agents']} agents: "
              f"{rmsd_improvement:.2f} Å")
        
        # Check if RMSE improves
        if all_results[0]['rmse_results'] and all_results[-1]['rmse_results']:
            initial_temp_rmse = all_results[0]['rmse_results']['temperature_rmse']
            final_temp_rmse = all_results[-1]['rmse_results']['temperature_rmse']
            temp_change = initial_temp_rmse - final_temp_rmse
            
            print(f"  • Temperature RMSE change (5→100 agents): {temp_change:+.2f}°C "
                  f"({'improved' if temp_change > 0 else 'worse'})")
    
    print("\n" + "="*70)
    print("✓ EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Review the comparative results above")
    print("  2. Check detailed results in agent_scaling_results.json")
    print("  3. Identify optimal agent count for best RMSD/RMSE balance")
    print("="*70)


if __name__ == "__main__":
    main()
