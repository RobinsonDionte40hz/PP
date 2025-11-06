"""Monitor progress of deep mechanism analysis."""
from pathlib import Path
import json
import time

output_dir = Path('results/deep_mechanism')

proteins = ['1VII', '1CRN', '1UBQ', '1LYZ', '1MBN']

print("\n" + "="*70)
print("DEEP MECHANISM ANALYSIS - PROGRESS MONITOR")
print("="*70)

completed = []
for protein_id in proteins:
    filename = f"{protein_id}_deep_analysis.json"
    filepath = output_dir / filename
    
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
        completed.append(protein_id)
        
        div = data['exploration_diversity']['diversity_ratio']
        mix = data['conformational_mixing']['mixing_rate']
        con = data['consciousness_dynamics']['trajectory_complexity']
        
        print(f"\n✓ {protein_id} ({data['protein_info']['size']} res) - COMPLETE")
        print(f"    Diversity: {div:.4f} | Mixing: {mix:.4f} | Consciousness: {con:.2f}")
    else:
        print(f"\n⏳ {protein_id} - IN PROGRESS")

print(f"\n{'='*70}")
print(f"Progress: {len(completed)}/{len(proteins)} complete")

if len(completed) == len(proteins):
    print("\n✅ ALL PROTEINS COMPLETE!")
    print("\nRunning correlation analysis...")
    
    # Check for summary
    summary_file = output_dir / 'comparative_deep_analysis.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"\n{'='*70}")
        print("CORRELATION RESULTS")
        print(f"{'='*70}")
        
        r_div = summary['correlations']['size_vs_diversity']
        r_mix = summary['correlations']['size_vs_mixing']
        r_con = summary['correlations']['size_vs_consciousness']
        
        print(f"\nSize vs Exploration Diversity:      r = {r_div:+.3f}")
        print(f"Size vs Conformational Mixing:      r = {r_mix:+.3f}")
        print(f"Size vs Consciousness Complexity:   r = {r_con:+.3f}")
        
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}\n")
        
        if abs(r_div) > 0.7:
            print(f"✓ Diversity: {'POSITIVE' if r_div > 0 else 'NEGATIVE'} correlation (|r| = {abs(r_div):.3f})")
        else:
            print(f"✗ Diversity: No strong correlation (|r| = {abs(r_div):.3f})")
            
        if abs(r_mix) > 0.7:
            print(f"✓ Mixing: {'POSITIVE' if r_mix > 0 else 'NEGATIVE'} correlation (|r| = {abs(r_mix):.3f})")
        else:
            print(f"✗ Mixing: No strong correlation (|r| = {abs(r_mix):.3f})")
            
        if abs(r_con) > 0.7:
            print(f"✓ Consciousness: {'POSITIVE' if r_con > 0 else 'NEGATIVE'} correlation (|r| = {abs(r_con):.3f})")
        else:
            print(f"✗ Consciousness: No strong correlation (|r| = {abs(r_con):.3f})")
    else:
        print("\n⚠️  Waiting for summary file...")
else:
    remaining = len(proteins) - len(completed)
    print(f"\n⏳ {remaining} protein(s) remaining...")
    print(f"   Estimated time: ~{remaining * 2} minutes")

print(f"{'='*70}\n")
