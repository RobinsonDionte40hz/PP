"""
Test Multi-Start on LARGE Protein: 1MBN (153 residues)

Question: Do large proteins also show uniform basin quality?
Hypothesis: If uniform → inverse scaling is about MEAN basin quality
            If variable → large proteins have rare deep basins
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

# Test configuration
TEST_PROTEIN = {
    'id': '1MBN',
    'sequence': 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG',
    'size': 153,
    'category': 'large'
}

N_STARTS_LIST = [1, 5, 10, 20, 50]
ITERATIONS_PER_START = 500
N_AGENTS_PER_START = 10

print("="*80)
print("LARGE PROTEIN MULTI-START TEST")
print("="*80)
print(f"\nProtein: {TEST_PROTEIN['id']} ({TEST_PROTEIN['size']} residues)")
print(f"Landscape: Smooth (2.2 minima/residue)")
print(f"Hypothesis: Basin quality uniformity vs size")
print(f"Test range: {N_STARTS_LIST[0]}-{N_STARTS_LIST[-1]} starts")
print("="*80)

all_results = []

for n_starts in N_STARTS_LIST:
    print(f"\n{'='*80}")
    print(f"Testing {n_starts} random start(s)")
    print(f"{'='*80}")
    
    start_time = time.time()
    start_results = []
    
    for start_idx in range(n_starts):
        print(f"\n  Start {start_idx + 1}/{n_starts}:")
        
        # Create coordinator (each start = new random initialization)
        coordinator = MultiAgentCoordinator(
            protein_sequence=TEST_PROTEIN['sequence'],
            enable_checkpointing=False
        )
        
        # Initialize agents
        agents = coordinator.initialize_agents(
            count=N_AGENTS_PER_START,
            diversity_profile='balanced'
        )
        
        # Run exploration
        try:
            results = coordinator.run_parallel_exploration(iterations=ITERATIONS_PER_START)
            best_energy = results.best_energy
            
            start_results.append({
                'start_index': start_idx,
                'best_energy': best_energy,
                'best_rmsd': None,
                'agents': N_AGENTS_PER_START,
                'iterations': ITERATIONS_PER_START
            })
            
            print(f"    Best energy: {best_energy:.2f} kcal/mol")
            
        except Exception as e:
            print(f"    Error in start {start_idx + 1}: {e}")
            start_results.append({
                'start_index': start_idx,
                'best_energy': None,
                'best_rmsd': None,
                'error': str(e)
            })
    
    # Analyze results
    elapsed = time.time() - start_time
    valid_energies = [r['best_energy'] for r in start_results if r['best_energy'] is not None]
    
    if valid_energies:
        best_energy_overall = min(valid_energies)
        mean_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)
    else:
        best_energy_overall = None
        mean_energy = None
        std_energy = None
    
    result = {
        'n_starts': n_starts,
        'total_agents': n_starts * N_AGENTS_PER_START,
        'iterations_per_start': ITERATIONS_PER_START,
        'elapsed_seconds': elapsed,
        'best_energy_overall': best_energy_overall,
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'best_rmsd_overall': None,
        'mean_rmsd': None,
        'std_rmsd': None,
        'start_results': start_results,
        'success_rate': len(valid_energies) / n_starts if n_starts > 0 else 0
    }
    
    all_results.append(result)
    
    # Print summary
    print(f"\n  Summary for {n_starts} start(s):")
    print(f"    Time: {elapsed:.1f}s, Success: {result['success_rate']*100:.1f}%")
    if best_energy_overall is not None:
        print(f"    Best: {best_energy_overall:.2f} kcal/mol")
        print(f"    Mean: {mean_energy:.2f} ± {std_energy:.2f} kcal/mol")

# Save results
output_dir = project_root / 'results' / 'multistart_experiment'
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f'{TEST_PROTEIN["id"]}_multistart_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'date': datetime.now().isoformat(),
            'protein_id': TEST_PROTEIN['id'],
            'protein_size': TEST_PROTEIN['size'],
            'iterations_per_start': ITERATIONS_PER_START,
            'agents_per_start': N_AGENTS_PER_START
        },
        'results': all_results
    }, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved: {output_file}")
print(f"{'='*80}")

# Analysis
if len(all_results) >= 2:
    best_energies = [r['best_energy_overall'] for r in all_results if r['best_energy_overall'] is not None]
    
    if len(best_energies) > 1:
        baseline = best_energies[0]
        final = best_energies[-1]
        improvement = ((baseline - final) / abs(baseline)) * 100
        
        print(f"\nLARGE PROTEIN (1MBN) RESULTS:")
        print(f"  1 start:  {baseline:.2f} kcal/mol")
        print(f"  50 starts: {final:.2f} kcal/mol")
        print(f"  Improvement: {improvement:+.1f}%")
        
        # Compare to small protein
        print(f"\nCOMPARISON:")
        print(f"  Small (1VII):  +0.6% improvement (uniform basins)")
        print(f"  Large (1MBN): {improvement:+.1f}% improvement")
        
        if abs(improvement) < 5:
            print(f"\n✓ UNIFORM BASINS confirmed for large proteins!")
            print(f"  Interpretation: Inverse scaling = MEAN basin quality, not variance")
        else:
            print(f"\n✗ VARIABLE BASINS found in large proteins!")
            print(f"  Interpretation: Large proteins have rare deep basins")

print(f"{'='*80}\n")
