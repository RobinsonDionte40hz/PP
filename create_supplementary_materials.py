#!/usr/bin/env python3
"""
Generate Supplementary Materials for bioRxiv Paper

Creates additional figures and formatted tables for supplementary section.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_individual_results(results_dir: Path) -> list:
    """Load all individual JSON results."""
    results = []
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return results


def create_supp_fig_s1_energy_decomposition(df: pd.DataFrame, output_dir: Path):
    """Supplementary Figure S1: Energy component analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    energy_components = [
        ('best_bond_energy', 'Bond Energy', 'skyblue'),
        ('best_angle_energy', 'Angle Energy', 'lightcoral'),
        ('best_dihedral_energy', 'Dihedral Energy', 'lightgreen'),
        ('best_vdw_energy', 'Van der Waals', 'plum'),
        ('best_electrostatic_energy', 'Electrostatic', 'gold'),
        ('best_hbond_energy', 'H-bond Energy', 'pink')
    ]
    
    for idx, (col, label, color) in enumerate(energy_components):
        ax = axes[idx // 3, idx % 3]
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0 and not (data == 0).all():
                ax.hist(data, bins=20, color=color, edgecolor='black', alpha=0.7)
                ax.set_xlabel(f'{label} (kcal/mol)', fontweight='bold')
                ax.set_ylabel('Count', fontweight='bold')
                ax.set_title(label, fontweight='bold')
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_figure_s1_energy_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'supp_figure_s1_energy_decomposition.pdf', bbox_inches='tight')
    print(f"✓ Saved Supplementary Figure S1")
    plt.close()


def create_supp_fig_s2_size_vs_metrics(df: pd.DataFrame, output_dir: Path):
    """Supplementary Figure S2: Size correlations with multiple metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSD vs Size (duplicate of main but with more detail)
    ax = axes[0, 0]
    scatter = ax.scatter(df['sequence_length'], df['best_rmsd'], 
                        c=df['best_rmsd'], cmap='viridis_r', s=100, alpha=0.6,
                        edgecolors='black', linewidth=0.5)
    z = np.polyfit(df['sequence_length'], df['best_rmsd'], 1)
    p = np.poly1d(z)
    ax.plot(df['sequence_length'], p(df['sequence_length']), "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('RMSD (Å)', fontweight='bold')
    ax.set_title('RMSD vs Size', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # GDT-TS vs Size
    ax = axes[0, 1]
    if 'gdt_ts_score' in df.columns:
        valid_data = df[['sequence_length', 'gdt_ts_score']].dropna()
        if len(valid_data) > 0:
            ax.scatter(valid_data['sequence_length'], valid_data['gdt_ts_score'],
                      c='orange', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Sequence Length', fontweight='bold')
            ax.set_ylabel('GDT-TS Score', fontweight='bold')
            ax.set_title('GDT-TS vs Size', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # TM-score vs Size
    ax = axes[1, 0]
    if 'tm_score' in df.columns:
        valid_data = df[['sequence_length', 'tm_score']].dropna()
        if len(valid_data) > 0:
            ax.scatter(valid_data['sequence_length'], valid_data['tm_score'],
                      c='green', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Sequence Length', fontweight='bold')
            ax.set_ylabel('TM-score', fontweight='bold')
            ax.set_title('TM-score vs Size', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Execution Time vs Size (log scale)
    ax = axes[1, 1]
    ax.scatter(df['sequence_length'], df['execution_time_seconds'],
              c='purple', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Execution Time (s, log scale)', fontweight='bold')
    ax.set_title('Performance Scaling', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_figure_s2_size_correlations.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'supp_figure_s2_size_correlations.pdf', bbox_inches='tight')
    print(f"✓ Saved Supplementary Figure S2")
    plt.close()


def create_supp_fig_s3_quality_breakdown(df: pd.DataFrame, output_dir: Path):
    """Supplementary Figure S3: Quality metrics breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Quality distribution pie chart
    ax = axes[0]
    quality_counts = df['quality'].value_counts()
    colors = ['green', 'yellowgreen', 'orange', 'red']
    ax.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Prediction Quality Distribution', fontweight='bold', fontsize=12)
    
    # Size category distribution
    ax = axes[1]
    size_counts = df['size_category'].value_counts()
    colors_size = sns.color_palette("husl", len(size_counts))
    ax.bar(range(len(size_counts)), size_counts.values, color=colors_size, edgecolor='black')
    ax.set_xticks(range(len(size_counts)))
    ax.set_xticklabels(size_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Protein Size Distribution', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_figure_s3_quality_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'supp_figure_s3_quality_breakdown.pdf', bbox_inches='tight')
    print(f"✓ Saved Supplementary Figure S3")
    plt.close()


def create_supp_fig_s4_performance_metrics(df: pd.DataFrame, output_dir: Path):
    """Supplementary Figure S4: Detailed performance analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Conformations per second vs size
    ax = axes[0]
    if 'conformations_per_second' in df.columns:
        ax.scatter(df['sequence_length'], df['conformations_per_second'],
                  c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Sequence Length', fontweight='bold')
        ax.set_ylabel('Conformations/Second', fontweight='bold')
        ax.set_title('Sampling Throughput', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Success rate by size category
    ax = axes[1]
    if 'size_category' in df.columns:
        size_success = df.groupby('size_category').agg({
            'success': 'sum',
            'pdb_id': 'count'
        })
        size_success['success_rate'] = (size_success['success'] / size_success['pdb_id'] * 100)
        
        colors = sns.color_palette("RdYlGn", len(size_success))
        bars = ax.bar(range(len(size_success)), size_success['success_rate'].values,
                      color=colors, edgecolor='black')
        ax.set_xticks(range(len(size_success)))
        ax.set_xticklabels(size_success.index, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Success Rate by Size Category', fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_figure_s4_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'supp_figure_s4_performance.pdf', bbox_inches='tight')
    print(f"✓ Saved Supplementary Figure S4")
    plt.close()


def prepare_supplementary_tables(df: pd.DataFrame, output_dir: Path):
    """Create properly formatted supplementary tables."""
    
    # Supplementary Table S1: Complete results (48 proteins)
    table_s1 = df[[
        'pdb_id', 'protein_name', 'sequence_length', 
        'best_rmsd', 'gdt_ts_score', 'tm_score',
        'execution_time_seconds', 'conformations_per_second',
        'best_total_energy'
    ]].round(2)
    
    table_s1.columns = [
        'PDB ID', 'Protein Name', 'Length (residues)',
        'RMSD (Å)', 'GDT-TS', 'TM-score',
        'Time (s)', 'Conformations/s', 'Energy (kcal/mol)'
    ]
    
    table_s1.to_csv(output_dir / 'supp_table_s1_complete_results.csv', index=False)
    print(f"✓ Saved Supplementary Table S1: Complete results (48 proteins)")
    
    # Supplementary Table S2: Summary statistics by size
    summary = df.groupby('size_category').agg({
        'pdb_id': 'count',
        'sequence_length': ['mean', 'std'],
        'best_rmsd': ['mean', 'std', 'min', 'max'],
        'execution_time_seconds': ['mean', 'std'],
        'conformations_per_second': ['mean', 'std']
    }).round(2)
    
    summary.to_csv(output_dir / 'supp_table_s2_summary_by_size.csv')
    print(f"✓ Saved Supplementary Table S2: Summary statistics")
    
    # Supplementary Table S3: Top 20 best predictions
    table_s3 = df.nsmallest(20, 'best_rmsd')[[
        'pdb_id', 'protein_name', 'sequence_length',
        'best_rmsd', 'gdt_ts_score', 'tm_score', 'execution_time_seconds'
    ]].round(2)
    
    table_s3.to_csv(output_dir / 'supp_table_s3_top20.csv', index=False)
    print(f"✓ Saved Supplementary Table S3: Top 20 predictions")
    
    # Supplementary Table S4: Worst 10 for failure analysis
    table_s4 = df.nlargest(10, 'best_rmsd')[[
        'pdb_id', 'protein_name', 'sequence_length',
        'best_rmsd', 'gdt_ts_score', 'tm_score', 'execution_time_seconds'
    ]].round(2)
    
    table_s4.to_csv(output_dir / 'supp_table_s4_worst10.csv', index=False)
    print(f"✓ Saved Supplementary Table S4: Worst 10 predictions (failure analysis)")


def main():
    """Generate all supplementary materials."""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY MATERIALS")
    print("="*80 + "\n")
    
    # Load data
    print("📊 Loading benchmark data...")
    df = pd.read_csv("benchmark_results/summaries/complete_benchmark.csv")
    df = df[df['success'] == True].copy()
    
    # Add categorizations
    df['size_category'] = pd.cut(
        df['sequence_length'],
        bins=[0, 50, 100, 150, 200, 1000],
        labels=['Very Small (<50)', 'Small (50-100)', 'Medium (100-150)', 'Large (150-200)', 'Very Large (>200)']
    )
    
    df['quality'] = pd.cut(
        df['best_rmsd'],
        bins=[0, 5, 10, 15, 1000],
        labels=['Excellent (<5Å)', 'Good (5-10Å)', 'Fair (10-15Å)', 'Poor (>15Å)']
    )
    
    print(f"✓ Loaded {len(df)} successful predictions\n")
    
    # Create output directory
    output_dir = Path("benchmark_results/supplementary_materials")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Generate figures
    print("📈 Generating supplementary figures...")
    create_supp_fig_s1_energy_decomposition(df, output_dir)
    create_supp_fig_s2_size_vs_metrics(df, output_dir)
    create_supp_fig_s3_quality_breakdown(df, output_dir)
    create_supp_fig_s4_performance_metrics(df, output_dir)
    print()
    
    # Generate tables
    print("📋 Generating supplementary tables...")
    prepare_supplementary_tables(df, output_dir)
    print()
    
    print("="*80)
    print("✅ SUPPLEMENTARY MATERIALS COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {output_dir}/")
    print("\nGenerated:")
    print("  Figures:")
    print("    - supp_figure_s1_energy_decomposition.png/pdf")
    print("    - supp_figure_s2_size_correlations.png/pdf")
    print("    - supp_figure_s3_quality_breakdown.png/pdf")
    print("    - supp_figure_s4_performance.png/pdf")
    print("\n  Tables:")
    print("    - supp_table_s1_complete_results.csv (48 proteins)")
    print("    - supp_table_s2_summary_by_size.csv")
    print("    - supp_table_s3_top20.csv")
    print("    - supp_table_s4_worst10.csv")
    print("\n✨ Ready to add to your Word document!")


if __name__ == "__main__":
    main()
