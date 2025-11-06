#!/usr/bin/env python3
"""
Full Inverse Scaling Investigation Study

Runs comprehensive investigation on 5 test proteins spanning size range:
- 1VII (36 residues) - Small
- 1CRN (46 residues) - Small-Medium  
- 1UBQ (76 residues) - Medium
- 1LYZ (129 residues) - Medium-Large
- 1MBN (153 residues) - Large

Estimated time: 4-5 hours
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiments.investigate_inverse_scaling import InverseScalingInvestigator
from scipy.stats import pearsonr, spearmanr

# Test suite - proteins with known sequences
TEST_PROTEINS = [
    {
        'id': '1VII',
        'name': 'Villin Headpiece',
        'size': 36,
        'sequence': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
        'pdb': 'pdb1vii.ent',
        'category': 'small'
    },
    {
        'id': '1CRN',
        'name': 'Crambin',
        'size': 46,
        'sequence': 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
        'pdb': 'pdb1crn.ent',
        'category': 'small-medium'
    },
    {
        'id': '1UBQ',
        'name': 'Ubiquitin',
        'size': 76,
        'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        'pdb': 'pdb1ubq.ent',
        'category': 'medium'
    },
    {
        'id': '1LYZ',
        'name': 'Lysozyme',
        'size': 129,
        'sequence': 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
        'pdb': 'pdb1lyz.ent',
        'category': 'medium-large'
    },
    {
        'id': '1MBN',
        'name': 'Myoglobin',
        'size': 153,
        'sequence': 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG',
        'pdb': 'pdb1mbn.ent',
        'category': 'large'
    }
]


def run_single_investigation(protein_info: Dict, iterations: int = 2000, 
                             landscape_samples: int = 1000) -> Dict:
    """Run investigation on single protein."""
    print(f"\n{'='*80}")
    print(f"PROTEIN {protein_info['id']}: {protein_info['name']}")
    print(f"{'='*80}")
    print(f"Size: {protein_info['size']} residues")
    print(f"Category: {protein_info['category']}")
    
    # Check for native structure
    pdb_path = Path(f"pdb_cache/{protein_info['pdb']}")
    if not pdb_path.exists():
        # Try alternative location
        pdb_path = Path(f"quantum_coherence_proteins/pdb_files/{protein_info['pdb']}")
    
    if not pdb_path.exists():
        print(f"⚠️  Native PDB not found: {protein_info['pdb']}")
        pdb_path = None
    
    # Create investigator
    investigator = InverseScalingInvestigator(
        protein_sequence=protein_info['sequence'],
        native_pdb_path=str(pdb_path) if pdb_path else None,
        protein_id=protein_info['id']
    )
    
    # Run full investigation
    start_time = time.time()
    results = investigator.run_full_investigation(
        iterations=iterations,
        n_landscape_samples=landscape_samples
    )
    elapsed = time.time() - start_time
    
    print(f"\n✓ Investigation complete in {elapsed/60:.1f} minutes")
    
    # Convert to dict for JSON serialization
    results_dict = {
        'protein_info': protein_info,
        'results': {
            'protein_id': results.protein_id,
            'protein_size': results.protein_size,
            'best_energy': results.best_energy,
            'best_rmsd': results.best_rmsd,
            'initial_rmsd': results.initial_rmsd,
            'improvement_ratio': results.improvement_ratio,
            'exploration_time_seconds': results.exploration_time_seconds,
            'iterations_completed': results.iterations_completed,
            'total_conformations': results.total_conformations_explored
        }
    }
    
    # Add hypothesis testing data
    if results.energy_landscape:
        results_dict['energy_landscape'] = {
            'local_minima_density': results.energy_landscape.local_minima_density,
            'mean_energy_barrier': results.energy_landscape.mean_energy_barrier,
            'gradient_smoothness': results.energy_landscape.energy_gradient_smoothness,
            'autocorrelation_length': results.energy_landscape.energy_autocorrelation_length
        }
    
    return results_dict


def analyze_correlations(all_results: List[Dict]):
    """Analyze correlations between protein size and various metrics."""
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Extract data
    sizes = [r['protein_info']['size'] for r in all_results]
    energies = [r['results']['best_energy'] for r in all_results]
    
    # Landscape metrics
    minima_densities = [r['energy_landscape']['local_minima_density'] 
                       for r in all_results if 'energy_landscape' in r]
    barriers = [r['energy_landscape']['mean_energy_barrier'] 
               for r in all_results if 'energy_landscape' in r]
    smoothness = [r['energy_landscape']['gradient_smoothness'] 
                 for r in all_results if 'energy_landscape' in r]
    
    print("\n[H1] Energy Landscape Smoothness")
    print("-" * 60)
    
    if len(minima_densities) > 2:
        r_minima, p_minima = pearsonr(sizes[:len(minima_densities)], minima_densities)
        print(f"  Size vs Local Minima Density: r = {r_minima:.3f}, p = {p_minima:.4f}")
        if p_minima < 0.05:
            print(f"    ✓ SIGNIFICANT - Larger proteins have {'fewer' if r_minima < 0 else 'more'} local minima per residue")
        else:
            print(f"    ✗ Not significant")
        
        r_barrier, p_barrier = pearsonr(sizes[:len(barriers)], barriers)
        print(f"  Size vs Energy Barriers: r = {r_barrier:.3f}, p = {p_barrier:.4f}")
        
        r_smooth, p_smooth = pearsonr(sizes[:len(smoothness)], smoothness)
        print(f"  Size vs Gradient Smoothness: r = {r_smooth:.3f}, p = {p_smooth:.4f}")
    
    print("\n[Overall Performance]")
    print("-" * 60)
    r_energy, p_energy = pearsonr(sizes, energies)
    print(f"  Size vs Best Energy: r = {r_energy:.3f}, p = {p_energy:.4f}")
    if p_energy < 0.05:
        print(f"    ✓ SIGNIFICANT - Larger proteins achieve {'lower' if r_energy < 0 else 'higher'} energies")
    
    # Create summary DataFrame
    df = pd.DataFrame({
        'Protein': [r['protein_info']['id'] for r in all_results],
        'Size': sizes,
        'Category': [r['protein_info']['category'] for r in all_results],
        'Best Energy': energies,
        'Minima Density': minima_densities + [None] * (len(all_results) - len(minima_densities)),
        'Energy Barrier': barriers + [None] * (len(all_results) - len(barriers)),
        'Gradient Smoothness': smoothness + [None] * (len(all_results) - len(smoothness))
    })
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate correlation plots."""
    print(f"\nGenerating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Size vs Energy
    ax = axes[0, 0]
    ax.scatter(df['Size'], df['Best Energy'], s=100, alpha=0.7)
    for i, txt in enumerate(df['Protein']):
        ax.annotate(txt, (df['Size'].iloc[i], df['Best Energy'].iloc[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Protein Size (residues)', fontsize=12)
    ax.set_ylabel('Best Energy (kcal/mol)', fontsize=12)
    ax.set_title('Size vs Energy (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Size vs Minima Density
    ax = axes[0, 1]
    valid_data = df.dropna(subset=['Minima Density'])
    if len(valid_data) > 0:
        ax.scatter(valid_data['Size'], valid_data['Minima Density'], s=100, alpha=0.7, color='orange')
        for i, row in valid_data.iterrows():
            ax.annotate(row['Protein'], (row['Size'], row['Minima Density']),
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Protein Size (residues)', fontsize=12)
        ax.set_ylabel('Local Minima per Residue', fontsize=12)
        ax.set_title('H1: Landscape Smoothness (Lower is Smoother)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Size vs Energy Barrier
    ax = axes[1, 0]
    valid_data = df.dropna(subset=['Energy Barrier'])
    if len(valid_data) > 0:
        ax.scatter(valid_data['Size'], valid_data['Energy Barrier'], s=100, alpha=0.7, color='green')
        for i, row in valid_data.iterrows():
            ax.annotate(row['Protein'], (row['Size'], row['Energy Barrier']),
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Protein Size (residues)', fontsize=12)
        ax.set_ylabel('Mean Energy Barrier (kcal/mol)', fontsize=12)
        ax.set_title('H1: Energy Barriers (Lower is Smoother)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Size vs Gradient Smoothness
    ax = axes[1, 1]
    valid_data = df.dropna(subset=['Gradient Smoothness'])
    if len(valid_data) > 0:
        ax.scatter(valid_data['Size'], valid_data['Gradient Smoothness'], s=100, alpha=0.7, color='red')
        for i, row in valid_data.iterrows():
            ax.annotate(row['Protein'], (row['Size'], row['Gradient Smoothness']),
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Protein Size (residues)', fontsize=12)
        ax.set_ylabel('Gradient Smoothness', fontsize=12)
        ax.set_title('H1: Gradient Smoothness (Higher is Smoother)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "inverse_scaling_correlations.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_file}")


def main():
    """Run full investigation study."""
    print(f"\n{'='*80}")
    print("FULL INVERSE SCALING INVESTIGATION STUDY")
    print(f"{'='*80}")
    print(f"Proteins: {len(TEST_PROTEINS)}")
    print(f"Iterations per protein: 2000")
    print(f"Landscape samples: 1000")
    print(f"Estimated total time: 4-5 hours")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = Path("results/inverse_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run investigations
    all_results = []
    start_time = time.time()
    
    for i, protein in enumerate(TEST_PROTEINS, 1):
        print(f"\n[{i}/{len(TEST_PROTEINS)}] Starting investigation...")
        
        try:
            results = run_single_investigation(
                protein,
                iterations=2000,
                landscape_samples=1000
            )
            all_results.append(results)
            
            # Save individual results
            output_file = output_dir / f"{protein['id']}_investigation.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"❌ Error investigating {protein['id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Total time
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ALL INVESTIGATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Proteins completed: {len(all_results)}/{len(TEST_PROTEINS)}")
    
    # Analyze correlations
    if len(all_results) >= 3:
        df = analyze_correlations(all_results)
        
        # Save summary
        summary_file = output_dir / "investigation_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")
        
        # Generate visualizations
        try:
            generate_visualizations(df, output_dir)
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
    
    # Save complete results
    complete_file = output_dir / "COMPLETE_INVESTIGATION_RESULTS.json"
    with open(complete_file, 'w') as f:
        json.dump({
            'metadata': {
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_time_hours': total_time / 3600,
                'proteins_tested': len(all_results)
            },
            'proteins': all_results
        }, f, indent=2)
    
    print(f"✓ Saved complete results: {complete_file}")
    print(f"\n{'='*80}")
    print("INVESTIGATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Review results in: {output_dir}")
    print(f"2. Analyze correlations to identify mechanism")
    print(f"3. Update PUBLICATION_DRAFT.md with findings")
    print(f"4. Prepare manuscript for submission")


if __name__ == "__main__":
    main()
