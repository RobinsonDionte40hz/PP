"""
Test Multi-Start Hypothesis: Do more random initializations help small proteins?

Experiment Design:
- Protein: 1VII (36 residues, small protein with rough landscape)
- Test: 1, 5, 10, 20, 50 random starts
- Measure: Best energy achieved across all starts
- Hypothesis: Quality improves with more starts, saturates ~20

If this works:
- Small proteins benefit from multi-start
- Large proteins already succeed (smooth basins)
- Simple algorithmic fix validated!
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
from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructureLoader

# Test configuration
TEST_PROTEIN = {
    'id': '1VII',
    'sequence': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
    'size': 36,
    'pdb': 'pdb1vii.ent',
    'native_structure': 'quantum_coherence_proteins/pdb_files/pdb1vii.ent'
}

N_STARTS_LIST = [1, 5, 10, 20, 50]
ITERATIONS_PER_START = 500  # Quick runs to sample initial basin
N_AGENTS_PER_START = 10     # Standard setup

print("="*80)
print("MULTI-START HYPOTHESIS TEST")
print("="*80)
print(f"\nProtein: {TEST_PROTEIN['id']} ({TEST_PROTEIN['size']} residues)")
print(f"Hypothesis: More random starts → better predictions (find better basins)")
print(f"Test range: {N_STARTS_LIST[0]}-{N_STARTS_LIST[-1]} starts")
print("="*80)

all_results = []  # Changed name to avoid confusion

for n_starts in N_STARTS_LIST:
    print(f"\n{'='*80}")
    print(f"Testing {n_starts} random start(s)")
    print(f"{'='*80}")
    
    start_time = time.time()
    start_results = []
    
    for start_idx in range(n_starts):
        print(f"\n  Start {start_idx + 1}/{n_starts}:")
        
        # Create coordinator (NOTE: each start is a NEW random initialization)
        coordinator = MultiAgentCoordinator(
            protein_sequence=TEST_PROTEIN['sequence'],
            enable_checkpointing=False  # Speed up
        )
        
        # Initialize agents with diversity profile
        agents = coordinator.initialize_agents(
            count=N_AGENTS_PER_START,
            diversity_profile='balanced'
        )
        
        # Run exploration (this samples the initial basin)
        try:
            # Run exploration
            results = coordinator.run_parallel_exploration(iterations=ITERATIONS_PER_START)
            
            # Get best energy
            best_energy = results.best_energy
            
            # Calculate RMSD to native if possible
            rmsd = None  # Skip RMSD for speed - focus on energy
            
            start_results.append({
                'start_index': start_idx,
                'best_energy': best_energy,
                'best_rmsd': rmsd,
                'agents': N_AGENTS_PER_START,
                'iterations': ITERATIONS_PER_START
            })
            
            print(f"    Best energy: {best_energy:.2f} kcal/mol")
            if rmsd is not None:
                print(f"    Best RMSD: {rmsd:.2f} Å")
            
        except Exception as e:
            print(f"    Error in start {start_idx + 1}: {e}")
            start_results.append({
                'start_index': start_idx,
                'best_energy': None,
                'best_rmsd': None,
                'error': str(e)
            })
    
    # Analyze results for this N_starts
    elapsed = time.time() - start_time
    
    valid_energies = [r['best_energy'] for r in start_results if r['best_energy'] is not None]
    valid_rmsds = [r['best_rmsd'] for r in start_results if r['best_rmsd'] is not None]
    
    if valid_energies:
        best_energy_overall = min(valid_energies)
        mean_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)
    else:
        best_energy_overall = None
        mean_energy = None
        std_energy = None
    
    if valid_rmsds:
        best_rmsd_overall = min(valid_rmsds)
        mean_rmsd = np.mean(valid_rmsds)
        std_rmsd = np.std(valid_rmsds)
    else:
        best_rmsd_overall = None
        mean_rmsd = None
        std_rmsd = None
    
    result = {
        'n_starts': n_starts,
        'total_agents': n_starts * N_AGENTS_PER_START,
        'iterations_per_start': ITERATIONS_PER_START,
        'elapsed_seconds': elapsed,
        'best_energy_overall': best_energy_overall,
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'best_rmsd_overall': best_rmsd_overall,
        'mean_rmsd': mean_rmsd,
        'std_rmsd': std_rmsd,
        'start_results': start_results,
        'success_rate': len(valid_energies) / n_starts if n_starts > 0 else 0
    }
    
    all_results.append(result)
    
    # Print summary
    print(f"\n  Summary for {n_starts} start(s):")
    print(f"    Total computation: {n_starts * N_AGENTS_PER_START} agents × {ITERATIONS_PER_START} iterations")
    print(f"    Time elapsed: {elapsed:.1f} seconds")
    print(f"    Success rate: {result['success_rate']*100:.1f}%")
    if best_energy_overall is not None:
        print(f"    Best energy found: {best_energy_overall:.2f} kcal/mol")
        print(f"    Mean energy: {mean_energy:.2f} ± {std_energy:.2f} kcal/mol")
    if best_rmsd_overall is not None:
        print(f"    Best RMSD found: {best_rmsd_overall:.2f} Å")
        print(f"    Mean RMSD: {mean_rmsd:.2f} ± {std_rmsd:.2f} Å")

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
print("RESULTS SAVED")
print(f"{'='*80}")
print(f"File: {output_file}")

# Analysis and Visualization
print(f"\n{'='*80}")
print("ANALYSIS: Does Multi-Start Help?")
print(f"{'='*80}")

# Extract data for plotting
n_starts_array = np.array([r['n_starts'] for r in all_results])
best_energies = np.array([r['best_energy_overall'] for r in all_results if r['best_energy_overall'] is not None])
mean_energies = np.array([r['mean_energy'] for r in all_results if r['mean_energy'] is not None])
std_energies = np.array([r['std_energy'] for r in all_results if r['std_energy'] is not None])

best_rmsds = np.array([r['best_rmsd_overall'] for r in all_results if r['best_rmsd_overall'] is not None])
mean_rmsds = np.array([r['mean_rmsd'] for r in all_results if r['mean_rmsd'] is not None])

# Calculate improvement
if len(best_energies) > 1:
    baseline_energy = best_energies[0]  # 1 start
    improvements_energy = ((baseline_energy - best_energies) / abs(baseline_energy)) * 100
    
    print(f"\nEnergy Improvement over 1-start baseline:")
    for i, n in enumerate(n_starts_array[:len(best_energies)]):
        print(f"  {n:2d} starts: {best_energies[i]:8.2f} kcal/mol ({improvements_energy[i]:+6.2f}%)")

if len(best_rmsds) > 1:
    baseline_rmsd = best_rmsds[0]
    improvements_rmsd = ((baseline_rmsd - best_rmsds) / baseline_rmsd) * 100
    
    print(f"\nRMSD Improvement over 1-start baseline:")
    for i, n in enumerate(n_starts_array[:len(best_rmsds)]):
        print(f"  {n:2d} starts: {best_rmsds[i]:8.2f} Å ({improvements_rmsd[i]:+6.2f}%)")

# Check for saturation
if len(best_energies) >= 3:
    # Compare last two results
    improvement_last = abs(best_energies[-1] - best_energies[-2])
    improvement_first = abs(best_energies[1] - best_energies[0])
    
    if improvement_last < 0.1 * improvement_first:
        print(f"\n✓ SATURATION DETECTED around {n_starts_array[-2]} starts")
        print(f"  (Improvement {improvement_last:.2f} is <10% of initial {improvement_first:.2f})")
    else:
        print(f"\n⚠ NO SATURATION YET - more starts may help")
        print(f"  (Still seeing {improvement_last:.2f} kcal/mol improvement)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Best energy vs N_starts
ax = axes[0, 0]
if len(best_energies) > 0:
    ax.plot(n_starts_array[:len(best_energies)], best_energies, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Number of Random Starts', fontsize=12)
    ax.set_ylabel('Best Energy Found (kcal/mol)', fontsize=12)
    ax.set_title(f'A. Multi-Start Improvement\n{TEST_PROTEIN["id"]} ({TEST_PROTEIN["size"]} residues)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add baseline reference
    if len(best_energies) > 0:
        ax.axhline(y=best_energies[0], color='red', linestyle='--', alpha=0.5, label='1-start baseline')
        ax.legend()

# Panel B: Mean ± std energy
ax = axes[0, 1]
if len(mean_energies) > 0:
    ax.errorbar(n_starts_array[:len(mean_energies)], mean_energies, yerr=std_energies, 
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='green')
    ax.set_xlabel('Number of Random Starts', fontsize=12)
    ax.set_ylabel('Mean Energy ± Std (kcal/mol)', fontsize=12)
    ax.set_title('B. Basin Quality Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Panel C: Best RMSD vs N_starts
ax = axes[1, 0]
if len(best_rmsds) > 0:
    ax.plot(n_starts_array[:len(best_rmsds)], best_rmsds, 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Random Starts', fontsize=12)
    ax.set_ylabel('Best RMSD to Native (Å)', fontsize=12)
    ax.set_title('C. Structural Quality Improvement', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if len(best_rmsds) > 0:
        ax.axhline(y=best_rmsds[0], color='red', linestyle='--', alpha=0.5, label='1-start baseline')
        ax.legend()

# Panel D: Summary statistics
ax = axes[1, 1]
summary_text = f"""
MULTI-START HYPOTHESIS TEST

Protein: {TEST_PROTEIN['id']} ({TEST_PROTEIN['size']} residues)
Landscape: Rough (9.3 minima/residue)

Results:
"""

if len(best_energies) > 1:
    improvement_pct = improvements_energy[-1] if len(improvements_energy) > 0 else 0
    summary_text += f"""
Energy Improvement:
  1 start:  {best_energies[0]:.2f} kcal/mol
  {n_starts_array[-1]:2d} starts: {best_energies[-1]:.2f} kcal/mol
  Change:   {improvement_pct:+.1f}%
"""

if len(best_rmsds) > 1:
    improvement_pct_rmsd = improvements_rmsd[-1] if len(improvements_rmsd) > 0 else 0
    summary_text += f"""
RMSD Improvement:
  1 start:  {best_rmsds[0]:.2f} Å
  {n_starts_array[-1]:2d} starts: {best_rmsds[-1]:.2f} Å
  Change:   {improvement_pct_rmsd:+.1f}%
"""

# Saturation check
if len(best_energies) >= 3:
    if improvement_last < 0.1 * improvement_first:
        summary_text += f"""
Saturation: YES (~{n_starts_array[-2]} starts)
"""
    else:
        summary_text += f"""
Saturation: NO (needs more starts)
"""

summary_text += f"""
Conclusion:
{'✓ Multi-start HELPS small proteins!' if len(best_energies) > 1 and best_energies[-1] < best_energies[0] else '✗ Multi-start showed no improvement'}
"""

ax.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=10,
        family='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.axis('off')

plt.tight_layout()
plot_file = output_dir / f'{TEST_PROTEIN["id"]}_multistart_analysis.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved: {plot_file}")

# Final verdict
print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")

if len(best_energies) > 1:
    if best_energies[-1] < best_energies[0]:
        print("✓ HYPOTHESIS CONFIRMED: Multi-start improves small protein predictions!")
        print(f"  Best improvement: {improvements_energy[-1]:.1f}%")
        print(f"  Mechanism: Finding better initial basins via random sampling")
        print(f"\n  Recommendation: Use {n_starts_array[-2] if len(n_starts_array) > 1 else n_starts_array[-1]} starts for small proteins")
    else:
        print("✗ HYPOTHESIS REJECTED: Multi-start did not help")
        print("  Possible reasons: Basin quality uniform, or need more iterations per start")
else:
    print("⚠ INSUFFICIENT DATA: Need successful runs to compare")

print(f"{'='*80}\n")
