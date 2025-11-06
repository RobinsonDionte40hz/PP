"""Deep mechanistic analysis: Diversity, Consciousness Trajectories, and Mixing.

This script analyzes three key questions about inverse scaling:
1. Exploration Diversity: Do large proteins explore more unique conformations?
2. Conformational Mixing: Do large proteins make bigger jumps in conformational space?
3. Consciousness Adaptation: Do large proteins drive more behavioral complexity?
"""
import json
import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator


def run_deep_analysis(protein_id: str, sequence: str, protein_size: int, iterations: int = 500, num_agents: int = 10):
    """Run deep mechanism analysis on a single protein."""
    
    print(f"\n{'='*70}")
    print(f"DEEP MECHANISM ANALYSIS: {protein_id} ({protein_size} residues)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Create coordinator
    coordinator = MultiAgentCoordinator(protein_sequence=sequence)
    agents = coordinator.initialize_agents(count=num_agents, diversity_profile='balanced')
    
    # Tracking structures
    unique_conformations = set()
    consciousness_trajectories = []
    energy_history = []
    previous_hashes = {}
    conformational_transitions = []
    mixing_events = 0
    
    print(f"\n[1/3] Running {iterations} iterations with {num_agents} agents...")
    
    # Run exploration
    for iteration in range(iterations):
        # Execute one iteration
        coordinator.run_parallel_exploration(iterations=1)
        
        # Track metrics for each agent
        for agent_id, agent in enumerate(agents):
            # Get agent state
            consciousness = agent.get_consciousness_state()
            conformation = agent.get_current_conformation()
            metrics = agent.get_exploration_metrics()
            energy = metrics.get('current_energy', 0.0)
            
            # Track consciousness
            consciousness_trajectories.append({
                'iteration': iteration,
                'agent_id': agent_id,
                'frequency': consciousness.get_frequency(),
                'coherence': consciousness.get_coherence(),
                'energy': energy
            })
            
            # Track unique conformations (hash by phi/psi angles)
            angles = [f"{phi:.1f},{psi:.1f}" for phi, psi in zip(conformation.phi_angles, conformation.psi_angles)]
            conf_hash = "|".join(angles)
            unique_conformations.add(conf_hash)
            
            # Track conformational transitions
            agent_key = f"agent_{agent_id}"
            if agent_key in previous_hashes:
                # Simple diversity measure: how many angles changed significantly
                prev_angles = previous_hashes[agent_key].split("|")
                curr_angles = angles
                
                changes = 0
                for prev, curr in zip(prev_angles, curr_angles):
                    prev_phi, prev_psi = map(float, prev.split(","))
                    curr_phi, curr_psi = map(float, curr.split(","))
                    
                    phi_change = abs(curr_phi - prev_phi)
                    psi_change = abs(curr_psi - prev_psi)
                    
                    # Wrap angles at 360
                    if phi_change > 180:
                        phi_change = 360 - phi_change
                    if psi_change > 180:
                        psi_change = 360 - psi_change
                    
                    if phi_change > 10 or psi_change > 10:  # Significant change threshold
                        changes += 1
                
                transition_fraction = changes / len(angles) if len(angles) > 0 else 0.0
                conformational_transitions.append(transition_fraction)
                
                # Count mixing events (>50% of residues changed significantly)
                if transition_fraction > 0.5:
                    mixing_events += 1
            
            previous_hashes[agent_key] = conf_hash
            
            # Track energy
            energy_history.append(energy)
        
        # Progress reporting
        if (iteration + 1) % 100 == 0:
            unique_count = len(unique_conformations)
            diversity_ratio = unique_count / ((iteration + 1) * num_agents)
            print(f"  Iteration {iteration+1}/{iterations}: "
                  f"{unique_count} unique conformations "
                  f"(diversity: {diversity_ratio:.3f})")
    
    elapsed = time.time() - start_time
    
    print(f"\n[2/3] Analyzing exploration patterns...")
    
    # Compute diversity metrics
    total_conformations = iterations * num_agents
    unique_count = len(unique_conformations)
    diversity_ratio = unique_count / total_conformations
    
    # Analyze consciousness trajectories
    freq_values = [t['frequency'] for t in consciousness_trajectories]
    coh_values = [t['coherence'] for t in consciousness_trajectories]
    
    # Compute consciousness trajectory complexity (path length in 2D space)
    consciousness_path_length = 0.0
    for i in range(1, len(consciousness_trajectories)):
        if consciousness_trajectories[i]['agent_id'] == consciousness_trajectories[i-1]['agent_id']:
            df = consciousness_trajectories[i]['frequency'] - consciousness_trajectories[i-1]['frequency']
            dc = consciousness_trajectories[i]['coherence'] - consciousness_trajectories[i-1]['coherence']
            consciousness_path_length += np.sqrt(df**2 + dc**2)
    
    # Analyze mixing
    mean_transition = np.mean(conformational_transitions) if conformational_transitions else 0.0
    std_transition = np.std(conformational_transitions) if conformational_transitions else 0.0
    mixing_rate = mixing_events / total_conformations if total_conformations > 0 else 0.0
    
    # Energy metrics
    best_energy = min(energy_history) if energy_history else float('inf')
    energy_improvement = energy_history[0] - best_energy if energy_history else 0.0
    
    print(f"\n[3/3] Compiling results...")
    
    results = {
        'protein_info': {
            'id': protein_id,
            'size': protein_size,
            'sequence': sequence
        },
        'exploration_diversity': {
            'total_conformations': total_conformations,
            'unique_conformations': unique_count,
            'diversity_ratio': diversity_ratio,
            'redundancy_ratio': 1.0 - diversity_ratio
        },
        'conformational_mixing': {
            'total_transitions': len(conformational_transitions),
            'mean_transition_fraction': mean_transition,
            'std_transition_fraction': std_transition,
            'mixing_events': mixing_events,
            'mixing_rate': mixing_rate
        },
        'consciousness_dynamics': {
            'mean_frequency': float(np.mean(freq_values)),
            'std_frequency': float(np.std(freq_values)),
            'frequency_range': [float(np.min(freq_values)), float(np.max(freq_values))],
            'mean_coherence': float(np.mean(coh_values)),
            'std_coherence': float(np.std(coh_values)),
            'coherence_range': [float(np.min(coh_values)), float(np.max(coh_values))],
            'trajectory_complexity': float(consciousness_path_length),
            'trajectory_length_per_step': float(consciousness_path_length / len(consciousness_trajectories))
        },
        'performance': {
            'best_energy': float(best_energy),
            'energy_improvement': float(energy_improvement),
            'exploration_time_seconds': elapsed,
            'conformations_per_second': total_conformations / elapsed
        }
    }
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Diversity Ratio:           {diversity_ratio:.4f}")
    print(f"    Mixing Rate:               {mixing_rate:.4f}")
    print(f"    Consciousness Complexity:  {consciousness_path_length:.2f}")
    print(f"    Best Energy:               {best_energy:.2f} kcal/mol")
    
    return results


def run_comparative_analysis(proteins: list, iterations: int = 500):
    """Run analysis on multiple proteins and compare."""
    all_results = []
    
    print("\n" + "="*70)
    print("COMPARATIVE DEEP MECHANISM ANALYSIS")
    print("="*70)
    print(f"Proteins: {len(proteins)}")
    print(f"Iterations per protein: {iterations}")
    print(f"Total runtime estimate: {len(proteins) * 2} minutes")
    
    for i, protein in enumerate(proteins, 1):
        print(f"\n\n### PROTEIN {i}/{len(proteins)} ###")
        
        results = run_deep_analysis(
            protein_id=protein['id'],
            sequence=protein['sequence'],
            protein_size=protein['size'],
            iterations=iterations,
            num_agents=10
        )
        
        all_results.append(results)
    
    # Save results
    output_dir = Path('results/deep_mechanism')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in all_results:
        filename = f"{result['protein_info']['id']}_deep_analysis.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Comparative analysis
    print(f"\n\n{'='*70}")
    print("COMPARATIVE RESULTS")
    print(f"{'='*70}\n")
    
    print(f"{'Protein':<8} | {'Size':>4} | {'Diversity':>9} | {'Mixing':>6} | {'Consciousness':>13}")
    print("-" * 70)
    
    for result in all_results:
        protein_id = result['protein_info']['id']
        size = result['protein_info']['size']
        diversity = result['exploration_diversity']['diversity_ratio']
        mixing = result['conformational_mixing']['mixing_rate']
        consciousness = result['consciousness_dynamics']['trajectory_complexity']
        
        print(f"{protein_id:<8} | {size:>4d} | {diversity:>9.4f} | {mixing:>6.4f} | {consciousness:>13.2f}")
    
    # Correlation analysis
    sizes = [r['protein_info']['size'] for r in all_results]
    diversities = [r['exploration_diversity']['diversity_ratio'] for r in all_results]
    mixings = [r['conformational_mixing']['mixing_rate'] for r in all_results]
    consciousness_vals = [r['consciousness_dynamics']['trajectory_complexity'] for r in all_results]
    
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*70}\n")
    
    r_div = np.corrcoef(sizes, diversities)[0, 1]
    r_mix = np.corrcoef(sizes, mixings)[0, 1]
    r_con = np.corrcoef(sizes, consciousness_vals)[0, 1]
    
    print(f"Size vs Exploration Diversity:      r = {r_div:+.3f}")
    print(f"Size vs Conformational Mixing:      r = {r_mix:+.3f}")
    print(f"Size vs Consciousness Complexity:   r = {r_con:+.3f}")
    
    print(f"\n{'='*70}")
    print("MECHANISTIC INSIGHTS")
    print(f"{'='*70}\n")
    
    if abs(r_div) > 0.7:
        direction = "POSITIVE" if r_div > 0 else "NEGATIVE"
        print(f"✓ Exploration Diversity shows {direction} correlation with size")
        if r_div > 0:
            print("  → Large proteins explore MORE unique conformations")
        else:
            print("  → Small proteins explore MORE unique conformations (revisit more)")
    
    if abs(r_mix) > 0.7:
        direction = "POSITIVE" if r_mix > 0 else "NEGATIVE"
        print(f"✓ Conformational Mixing shows {direction} correlation with size")
        if r_mix > 0:
            print("  → Large proteins make BIGGER conformational jumps")
        else:
            print("  → Small proteins make BIGGER conformational jumps")
    
    if abs(r_con) > 0.7:
        direction = "POSITIVE" if r_con > 0 else "NEGATIVE"
        print(f"✓ Consciousness Complexity shows {direction} correlation with size")
        if r_con > 0:
            print("  → Large proteins drive MORE behavioral adaptation")
        else:
            print("  → Small proteins drive MORE behavioral adaptation")
    
    # Save summary
    summary = {
        'metadata': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'proteins_analyzed': len(proteins),
            'iterations_per_protein': iterations
        },
        'correlations': {
            'size_vs_diversity': float(r_div),
            'size_vs_mixing': float(r_mix),
            'size_vs_consciousness': float(r_con)
        },
        'results': all_results
    }
    
    with open(output_dir / 'comparative_deep_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return summary


if __name__ == '__main__':
    # Test proteins spanning size range
    test_proteins = [
        {
            'id': '1VII',
            'name': 'Villin Headpiece',
            'size': 36,
            'sequence': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF'
        },
        {
            'id': '1CRN',
            'name': 'Crambin',
            'size': 46,
            'sequence': 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN'
        },
        {
            'id': '1UBQ',
            'name': 'Ubiquitin',
            'size': 76,
            'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
        },
        {
            'id': '1LYZ',
            'name': 'Lysozyme',
            'size': 129,
            'sequence': 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL'
        },
        {
            'id': '1MBN',
            'name': 'Myoglobin',
            'size': 153,
            'sequence': 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG'
        }
    ]
    
    # Run analysis
    summary = run_comparative_analysis(test_proteins, iterations=500)
