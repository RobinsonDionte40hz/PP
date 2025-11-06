"""Monitor multi-start experiment progress."""
from pathlib import Path
import json
import time

output_dir = Path('results/multistart_experiment')
output_file = output_dir / '1VII_multistart_results.json'

print("\n" + "="*70)
print("MULTI-START EXPERIMENT - PROGRESS MONITOR")
print("="*70)
print(f"Protein: 1VII (36 residues)")
print(f"Testing: 1, 5, 10, 20, 50 random starts")
print(f"Total computation: 86 starts × 10 agents × 500 iterations")
print("="*70)

while True:
    if output_file.exists():
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        print(f"\n✓ Progress: {len(results)}/5 configurations complete")
        
        for r in results:
            n_starts = r['n_starts']
            best_energy = r.get('best_energy_overall')
            
            if best_energy is not None:
                print(f"  {n_starts:2d} start(s): Best energy = {best_energy:.2f} kcal/mol")
            else:
                print(f"  {n_starts:2d} start(s): IN PROGRESS")
        
        if len(results) >= 5:
            print("\n✅ EXPERIMENT COMPLETE!")
            
            # Show final results
            baseline = results[0]['best_energy_overall']
            print(f"\nBaseline (1 start):  {baseline:.2f} kcal/mol")
            
            for r in results[1:]:
                n = r['n_starts']
                best = r['best_energy_overall']
                improvement = ((baseline - best) / abs(baseline)) * 100 if baseline != 0 else 0
                print(f"{n:2d} starts:         {best:.2f} kcal/mol ({improvement:+.1f}%)")
            
            break
    else:
        print("\n⏳ Waiting for experiment to start...")
    
    time.sleep(10)
    print(".", end="", flush=True)
