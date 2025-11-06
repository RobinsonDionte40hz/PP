"""Monitor enhanced exploration progress."""
from pathlib import Path
import json

output_dir = Path('results/enhanced_exploration')

proteins = ['1VII', '1CRN', '1UBQ', '1LYZ', '1MBN']
baseline_diversity = 0.002  # From original analysis

print("\n" + "="*70)
print("ENHANCED EXPLORATION - PROGRESS MONITOR")
print("="*70)
print(f"Baseline diversity: {baseline_diversity:.4f} (0.2% - agents stuck)")
print(f"Goal: Increase diversity through perturbations")
print("="*70)

completed = []
for protein_id in proteins:
    filename = f"{protein_id}_enhanced.json"
    filepath = output_dir / filename
    
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
        completed.append(protein_id)
        
        div = data['exploration_diversity']['diversity_ratio']
        mix = data['conformational_mixing']['mixing_rate']
        con = data['consciousness_dynamics']['trajectory_complexity']
        pert_effect = data['exploration_diversity'].get('perturbation_effectiveness', 0.0)
        
        # Calculate improvement over baseline
        div_improvement = ((div - baseline_diversity) / baseline_diversity * 100) if baseline_diversity > 0 else 0
        
        print(f"\n✓ {protein_id} ({data['protein_info']['size']} res) - COMPLETE")
        print(f"    Diversity: {div:.4f} ({div_improvement:+.1f}% vs baseline)")
        print(f"    Mixing: {mix:.4f} (baseline: 0.0000)")
        print(f"    Consciousness: {con:.2f} (baseline: 0.00)")
        print(f"    Perturbation Effect: {pert_effect:+.6f}")
    else:
        print(f"\n⏳ {protein_id} - IN PROGRESS")

print(f"\n{'='*70}")
print(f"Progress: {len(completed)}/{len(proteins)} complete")

if len(completed) == len(proteins):
    print("\n✅ ALL PROTEINS COMPLETE!")
    
    # Check for summary
    summary_file = output_dir / 'comparative_enhanced_analysis.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"\n{'='*70}")
        print("ENHANCED vs BASELINE COMPARISON")
        print(f"{'='*70}")
        
        div_improvement = summary['improvements']['diversity_improvement_percent']
        mean_div = summary['improvements']['mean_diversity_enhanced']
        mean_mix = summary['improvements']['mean_mixing_enhanced']
        mean_con = summary['improvements']['mean_consciousness_enhanced']
        
        print(f"\nDiversity improvement:     {div_improvement:+.1f}%")
        print(f"  Baseline: {baseline_diversity:.4f}")
        print(f"  Enhanced: {mean_div:.4f}")
        
        print(f"\nMixing rate:")
        print(f"  Baseline: 0.0000")
        print(f"  Enhanced: {mean_mix:.4f}")
        
        print(f"\nConsciousness complexity:")
        print(f"  Baseline: 0.00")
        print(f"  Enhanced: {mean_con:.2f}")
        
        print(f"\n{'='*70}")
        print("CORRELATION RESULTS (WITH PERTURBATIONS)")
        print(f"{'='*70}")
        
        r_div = summary['correlations']['size_vs_diversity']
        r_mix = summary['correlations']['size_vs_mixing']
        r_con = summary['correlations']['size_vs_consciousness']
        
        print(f"\nSize vs Diversity:      r = {r_div:+.3f}")
        print(f"Size vs Mixing:         r = {r_mix:+.3f}")
        print(f"Size vs Consciousness:  r = {r_con:+.3f}")
    else:
        print("\n⚠️  Waiting for summary file...")
else:
    remaining = len(proteins) - len(completed)
    print(f"\n⏳ {remaining} protein(s) remaining...")
    print(f"   Estimated time: ~{remaining * 3} minutes")

print(f"{'='*70}\n")
