"""Deep mechanistic analysis: Diversity, Consciousness Trajectories, and Mixing."""
import json
import sys
from pathlib import Path
import time
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator


class DeepMechanismAnalyzer:
    """Analyze exploration diversity, consciousness dynamics, and conformational mixing."""
    
    def __init__(self, protein_id: str, protein_size: int, sequence: str, native_pdb: str):
        self.protein_id = protein_id
        self.protein_size = protein_size
        self.sequence = sequence
        self.native_pdb = native_pdb
        
        # Tracking structures
        self.unique_conformations = set()  # Hash-based uniqueness
        self.energy_history = []
        self.consciousness_trajectories = []  # (freq, coh) per iteration
        self.conformational_transitions = []  # RMSD distances between steps
        self.mixing_events = 0  # Count of large conformational changes
        
    def _hash_conformation(self, conformation) -> str:
        """Create hash of conformation for uniqueness tracking."""
        # Round phi/psi to 1 decimal to avoid float precision issues
        angles = []
        for res in conformation.residues:
            angles.append(f"{res.phi:.1f},{res.psi:.1f}")
        return "|".join(angles)
    
    def _compute_rmsd(self, conf1, conf2) -> float:
        """Compute RMSD between two conformations (simple CA-based)."""
        if len(conf1.residues) != len(conf2.residues):
            return float('inf')
        
        # Simple RMSD without alignment (tracks movement magnitude)
        sum_sq = 0.0
        count = 0
        for r1, r2 in zip(conf1.residues, conf2.residues):
            dx = r1.ca_coords[0] - r2.ca_coords[0]
            dy = r1.ca_coords[1] - r2.ca_coords[1]
            dz = r1.ca_coords[2] - r2.ca_coords[2]
            sum_sq += dx*dx + dy*dy + dz*dz
            count += 1
        
        if count == 0:
            return 0.0
        return np.sqrt(sum_sq / count)
    
    def run_analysis(self, iterations: int = 500, num_agents: int = 10) -> dict:
        """Run exploration with detailed tracking."""
        print(f"\n{'='*70}")
        print(f"DEEP MECHANISM ANALYSIS: {self.protein_id} ({self.protein_size} residues)")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Create coordinator with tracking hooks
        coordinator = MultiAgentCoordinator(
            protein_sequence=self.sequence,
            num_agents=num_agents,
            diversity_mode='balanced'
        )
        
        print(f"\n[1/3] Running {iterations} iterations with {num_agents} agents...")
        
        # Track previous structures for transition analysis
        previous_structures = {}
        
        # Run exploration with detailed logging
        for iteration in range(iterations):
            # Get current states before move
            current_states = []
            for agent_id, agent in enumerate(coordinator.agents):
                current_states.append({
                    'structure': agent.current_structure,
                    'frequency': agent.consciousness.frequency,
                    'coherence': agent.consciousness.coherence,
                    'energy': agent.current_energy
                })
            
            # Execute one iteration
            coordinator.run_parallel_exploration(iterations=1)
            
            # Analyze post-move states
            for agent_id, agent in enumerate(coordinator.agents):
                # Track consciousness trajectory
                self.consciousness_trajectories.append({
                    'iteration': iteration,
                    'agent_id': agent_id,
                    'frequency': agent.consciousness.frequency,
                    'coherence': agent.consciousness.coherence,
                    'energy': agent.current_energy
                })
                
                # Track unique conformations
                conf_hash = self._hash_conformation(agent.current_structure)
                self.unique_conformations.add(conf_hash)
                
                # Track conformational transitions
                prev_key = f"{iteration-1}_{agent_id}"
                if prev_key in previous_structures:
                    rmsd = self._compute_rmsd(
                        previous_structures[prev_key],
                        agent.current_structure
                    )
                    self.conformational_transitions.append(rmsd)
                    
                    # Count mixing events (large jumps > 5 Å)
                    if rmsd > 5.0:
                        self.mixing_events += 1
                
                # Store current structure
                current_key = f"{iteration}_{agent_id}"
                previous_structures[current_key] = agent.current_structure
                
                # Track energy
                self.energy_history.append(agent.current_energy)
            
            # Progress reporting
            if (iteration + 1) % 100 == 0:
                unique_count = len(self.unique_conformations)
                diversity_ratio = unique_count / ((iteration + 1) * num_agents)
                print(f"  Iteration {iteration+1}/{iterations}: "
                      f"{unique_count} unique conformations "
                      f"(diversity: {diversity_ratio:.3f})")
        
        elapsed = time.time() - start_time
        
        print(f"\n[2/3] Analyzing exploration patterns...")
        
        # Compute diversity metrics
        total_conformations = iterations * num_agents
        unique_count = len(self.unique_conformations)
        diversity_ratio = unique_count / total_conformations
        
        # Analyze consciousness trajectories
        freq_values = [t['frequency'] for t in self.consciousness_trajectories]
        coh_values = [t['coherence'] for t in self.consciousness_trajectories]
        
        # Compute consciousness trajectory complexity (path length in 2D space)
        consciousness_path_length = 0.0
        for i in range(1, len(self.consciousness_trajectories)):
            if self.consciousness_trajectories[i]['agent_id'] == \
               self.consciousness_trajectories[i-1]['agent_id']:
                df = self.consciousness_trajectories[i]['frequency'] - \
                     self.consciousness_trajectories[i-1]['frequency']
                dc = self.consciousness_trajectories[i]['coherence'] - \
                     self.consciousness_trajectories[i-1]['coherence']
                consciousness_path_length += np.sqrt(df**2 + dc**2)
        
        # Analyze mixing quality
        if self.conformational_transitions:
            mean_transition = np.mean(self.conformational_transitions)
            std_transition = np.std(self.conformational_transitions)
            max_transition = np.max(self.conformational_transitions)
        else:
            mean_transition = 0.0
            std_transition = 0.0
            max_transition = 0.0
        
        mixing_rate = self.mixing_events / total_conformations if total_conformations > 0 else 0.0
        
        # Energy landscape metrics
        best_energy = min(self.energy_history) if self.energy_history else float('inf')
        energy_improvement = self.energy_history[0] - best_energy if self.energy_history else 0.0
        
        print(f"\n[3/3] Compiling results...")
        
        results = {
            'protein_info': {
                'id': self.protein_id,
                'size': self.protein_size,
                'sequence': self.sequence,
                'native_pdb': self.native_pdb
            },
            'exploration_diversity': {
                'total_conformations': total_conformations,
                'unique_conformations': unique_count,
                'diversity_ratio': diversity_ratio,
                'redundancy_ratio': 1.0 - diversity_ratio,
                'explanation': 'Higher diversity_ratio = more exploration vs exploitation'
            },
            'conformational_mixing': {
                'total_transitions': len(self.conformational_transitions),
                'mean_transition_rmsd': mean_transition,
                'std_transition_rmsd': std_transition,
                'max_transition_rmsd': max_transition,
                'mixing_events': self.mixing_events,
                'mixing_rate': mixing_rate,
                'explanation': 'Higher mixing_rate = more large conformational jumps'
            },
            'consciousness_dynamics': {
                'mean_frequency': np.mean(freq_values),
                'std_frequency': np.std(freq_values),
                'frequency_range': [np.min(freq_values), np.max(freq_values)],
                'mean_coherence': np.mean(coh_values),
                'std_coherence': np.std(coh_values),
                'coherence_range': [np.min(coh_values), np.max(coh_values)],
                'trajectory_complexity': consciousness_path_length,
                'trajectory_length_per_step': consciousness_path_length / len(self.consciousness_trajectories),
                'explanation': 'Higher trajectory_complexity = more behavioral adaptation'
            },
            'performance': {
                'best_energy': best_energy,
                'energy_improvement': energy_improvement,
                'exploration_time_seconds': elapsed,
                'conformations_per_second': total_conformations / elapsed
            }
        }
        
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
        
        analyzer = DeepMechanismAnalyzer(
            protein_id=protein['id'],
            protein_size=protein['size'],
            sequence=protein['sequence'],
            native_pdb=protein['pdb']
        )
        
        results = analyzer.run_analysis(iterations=iterations)
        all_results.append(results)
        
        # Print summary
        div = results['exploration_diversity']['diversity_ratio']
        mix = results['conformational_mixing']['mixing_rate']
        comp = results['consciousness_dynamics']['trajectory_complexity']
        
        print(f"\n  Summary:")
        print(f"    Diversity Ratio:  {div:.4f}")
        print(f"    Mixing Rate:      {mix:.4f}")
        print(f"    Consciousness Complexity: {comp:.2f}")
    
    # Save individual results
    output_dir = Path('results/deep_mechanism')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in all_results:
        filename = f"{result['protein_info']['id']}_deep_analysis.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Create comparative analysis
    print(f"\n\n{'='*70}")
    print("COMPARATIVE RESULTS")
    print(f"{'='*70}\n")
    
    print("Protein | Size | Diversity | Mixing | Consciousness")
    print("-" * 70)
    
    for result in all_results:
        protein_id = result['protein_info']['id']
        size = result['protein_info']['size']
        diversity = result['exploration_diversity']['diversity_ratio']
        mixing = result['conformational_mixing']['mixing_rate']
        consciousness = result['consciousness_dynamics']['trajectory_complexity']
        
        print(f"{protein_id:7} | {size:4d} | {diversity:9.4f} | {mixing:6.4f} | {consciousness:13.2f}")
    
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
    
    print(f"Size vs Exploration Diversity:  r = {r_div:+.3f}")
    print(f"Size vs Conformational Mixing:  r = {r_mix:+.3f}")
    print(f"Size vs Consciousness Complexity: r = {r_con:+.3f}")
    
    print(f"\n{'='*70}")
    print("MECHANISTIC INSIGHTS")
    print(f"{'='*70}\n")
    
    if abs(r_div) > 0.7:
        direction = "POSITIVE" if r_div > 0 else "NEGATIVE"
        print(f"✓ Exploration Diversity shows {direction} correlation with size")
        if r_div > 0:
            print("  → Large proteins explore more unique conformations")
        else:
            print("  → Small proteins explore more unique conformations")
    
    if abs(r_mix) > 0.7:
        direction = "POSITIVE" if r_mix > 0 else "NEGATIVE"
        print(f"✓ Conformational Mixing shows {direction} correlation with size")
        if r_mix > 0:
            print("  → Large proteins make bigger conformational jumps")
        else:
            print("  → Small proteins make bigger conformational jumps")
    
    if abs(r_con) > 0.7:
        direction = "POSITIVE" if r_con > 0 else "NEGATIVE"
        print(f"✓ Consciousness Complexity shows {direction} correlation with size")
        if r_con > 0:
            print("  → Large proteins drive more behavioral adaptation")
        else:
            print("  → Small proteins drive more behavioral adaptation")
    
    # Save summary
    summary = {
        'metadata': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'proteins_analyzed': len(proteins),
            'iterations_per_protein': iterations
        },
        'correlations': {
            'size_vs_diversity': r_div,
            'size_vs_mixing': r_mix,
            'size_vs_consciousness': r_con
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
            'sequence': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
            'pdb': 'pdb1vii.ent'
        },
        {
            'id': '1CRN',
            'name': 'Crambin',
            'size': 46,
            'sequence': 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
            'pdb': 'pdb1crn.ent'
        },
        {
            'id': '1UBQ',
            'name': 'Ubiquitin',
            'size': 76,
            'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
            'pdb': 'pdb1ubq.ent'
        },
        {
            'id': '1LYZ',
            'name': 'Lysozyme',
            'size': 129,
            'sequence': 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
            'pdb': 'pdb1lyz.ent'
        },
        {
            'id': '1MBN',
            'name': 'Myoglobin',
            'size': 153,
            'sequence': 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG',
            'pdb': 'pdb1mbn.ent'
        }
    ]
    
    # Run analysis
    summary = run_comparative_analysis(test_proteins, iterations=500)
