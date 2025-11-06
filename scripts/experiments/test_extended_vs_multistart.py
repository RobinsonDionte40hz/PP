"""
Test Extended Single Run vs Multi-Start

Question: Does longer exploration beat multi-start?
Hypothesis: If basins are deep, single long run > multiple short runs
            If basins are shallow, multi-start wins by sampling
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

# Test configuration
TEST_PROTEIN = {
    'id': '1VII',
    'sequence': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
    'size': 36
}

# Compare strategies with SAME total computation
TOTAL_AGENTS_ITERATIONS = 50000  # 10 agents × 5000 iterations

STRATEGIES = [
    {
        'name': 'Single Long Run',
        'n_starts': 1,
        'agents_per_start': 10,
        'iterations_per_start': 5000,
        'description': 'Deep exploration of one basin'
    },
    {
        'name': 'Multi-Start (10×)',
        'n_starts': 10,
        'agents_per_start': 10,
        'iterations_per_start': 500,
        'description': 'Sample 10 different basins'
    }
]

print("="*80)
print("EXTENDED RUN vs MULTI-START COMPARISON")
print("="*80)
print(f"\nProtein: {TEST_PROTEIN['id']} ({TEST_PROTEIN['size']} residues)")
print(f"Total computation: {TOTAL_AGENTS_ITERATIONS} agent-iterations (equal)")
print(f"Question: Deep exploration vs broad sampling?")
print("="*80)

all_strategy_results = []

for strategy in STRATEGIES:
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy['name']}")
    print(f"  {strategy['description']}")
    print(f"  {strategy['n_starts']} start(s) × {strategy['agents_per_start']} agents × {strategy['iterations_per_start']} iterations")
    print(f"{'='*80}")
    
    strategy_start_time = time.time()
    start_results = []
    
    for start_idx in range(strategy['n_starts']):
        print(f"\n  Start {start_idx + 1}/{strategy['n_starts']}:")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            protein_sequence=TEST_PROTEIN['sequence'],
            enable_checkpointing=False
        )
        
        # Initialize agents
        agents = coordinator.initialize_agents(
            count=strategy['agents_per_start'],
            diversity_profile='balanced'
        )
        
        # Run exploration
        try:
            start_time = time.time()
            results = coordinator.run_parallel_exploration(
                iterations=strategy['iterations_per_start']
            )
            elapsed = time.time() - start_time
            
            best_energy = results.best_energy
            
            start_results.append({
                'start_index': start_idx,
                'best_energy': best_energy,
                'elapsed_seconds': elapsed
            })
            
            print(f"    Best energy: {best_energy:.2f} kcal/mol ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"    Error: {e}")
            start_results.append({
                'start_index': start_idx,
                'best_energy': None,
                'error': str(e)
            })
    
    # Analyze strategy
    strategy_elapsed = time.time() - strategy_start_time
    valid_energies = [r['best_energy'] for r in start_results if r['best_energy'] is not None]
    
    if valid_energies:
        strategy_best = min(valid_energies)
        strategy_mean = np.mean(valid_energies)
        strategy_std = np.std(valid_energies)
    else:
        strategy_best = None
        strategy_mean = None
        strategy_std = None
    
    strategy_result = {
        'strategy_name': strategy['name'],
        'n_starts': strategy['n_starts'],
        'agents_per_start': strategy['agents_per_start'],
        'iterations_per_start': strategy['iterations_per_start'],
        'total_agent_iterations': strategy['n_starts'] * strategy['agents_per_start'] * strategy['iterations_per_start'],
        'elapsed_seconds': strategy_elapsed,
        'best_energy': strategy_best,
        'mean_energy': strategy_mean,
        'std_energy': strategy_std,
        'start_results': start_results,
        'success_rate': len(valid_energies) / strategy['n_starts'] if strategy['n_starts'] > 0 else 0
    }
    
    all_strategy_results.append(strategy_result)
    
    # Print summary
    print(f"\n  Strategy Summary:")
    print(f"    Total time: {strategy_elapsed:.1f}s")
    print(f"    Success rate: {strategy_result['success_rate']*100:.1f}%")
    if strategy_best is not None:
        print(f"    Best energy: {strategy_best:.2f} kcal/mol")
        if strategy['n_starts'] > 1:
            print(f"    Mean energy: {strategy_mean:.2f} ± {strategy_std:.2f} kcal/mol")

# Save results
output_dir = project_root / 'results' / 'multistart_experiment'
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f'{TEST_PROTEIN["id"]}_extended_vs_multistart.json'
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'date': datetime.now().isoformat(),
            'protein_id': TEST_PROTEIN['id'],
            'protein_size': TEST_PROTEIN['size'],
            'total_computation': TOTAL_AGENTS_ITERATIONS
        },
        'strategies': all_strategy_results
    }, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved: {output_file}")
print(f"{'='*80}")

# Final comparison
if len(all_strategy_results) == 2:
    single = all_strategy_results[0]
    multi = all_strategy_results[1]
    
    print(f"\nFINAL COMPARISON (Equal Computation):")
    print(f"{'='*80}")
    
    if single['best_energy'] is not None and multi['best_energy'] is not None:
        print(f"\nSingle Long Run (1 × 5000 iter):")
        print(f"  Best energy: {single['best_energy']:.2f} kcal/mol")
        print(f"  Time: {single['elapsed_seconds']:.1f}s")
        
        print(f"\nMulti-Start (10 × 500 iter):")
        print(f"  Best energy: {multi['best_energy']:.2f} kcal/mol")
        print(f"  Time: {multi['elapsed_seconds']:.1f}s")
        
        diff = single['best_energy'] - multi['best_energy']
        diff_pct = (diff / abs(single['best_energy'])) * 100
        
        print(f"\nDifference: {diff:+.2f} kcal/mol ({diff_pct:+.1f}%)")
        
        if diff > 2:  # Single is worse by >2 kcal/mol
            print(f"\n✓ MULTI-START WINS!")
            print(f"  Interpretation: Sampling many basins > deep exploration of one")
            print(f"  Landscape: Shallow basins, multi-start finds better ones")
        elif diff < -2:  # Single is better by >2 kcal/mol
            print(f"\n✓ SINGLE LONG RUN WINS!")
            print(f"  Interpretation: Deep exploration > broad sampling")
            print(f"  Landscape: Deep basins, need time to descend")
        else:  # Within 2 kcal/mol
            print(f"\n✓ TIE - No clear winner")
            print(f"  Interpretation: Basins are shallow AND uniform")
            print(f"  Landscape: Flat plateau at ~200 kcal/mol")
            print(f"  Implication: Neither strategy helps much!")

print(f"{'='*80}\n")
