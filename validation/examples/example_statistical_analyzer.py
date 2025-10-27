"""
Example demonstrating StatisticalAnalyzer usage.

Shows:
1. Correlation analysis between protein characteristics and accuracy
2. Size category comparisons with ANOVA
3. Distribution plot generation
4. Predictive feature identification
5. Confidence interval calculations
6. Export functionality
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.statistical_analyzer import StatisticalAnalyzer


def example1_correlation_analysis():
    """Example 1: Correlation analysis between characteristics and accuracy."""
    print("=" * 60)
    print("Example 1: Correlation Analysis")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # Sample validation results
    results = [
        {'pdb_id': '1UBQ', 'protein_length': 76, 'rmsd': 2.5, 'gdt_ts': 75.0, 'tm_score': 0.65, 
         'energy': -45.0, 'resolution': 1.8, 'helix_fraction': 0.3, 'sheet_fraction': 0.4},
        {'pdb_id': '1CRN', 'protein_length': 46, 'rmsd': 1.8, 'gdt_ts': 85.0, 'tm_score': 0.82, 
         'energy': -52.0, 'resolution': 1.5, 'helix_fraction': 0.4, 'sheet_fraction': 0.3},
        {'pdb_id': '2MR9', 'protein_length': 35, 'rmsd': 3.2, 'gdt_ts': 68.0, 'tm_score': 0.58, 
         'energy': -38.0, 'resolution': 2.0, 'helix_fraction': 0.5, 'sheet_fraction': 0.2},
        {'pdb_id': '1VII', 'protein_length': 36, 'rmsd': 4.5, 'gdt_ts': 55.0, 'tm_score': 0.52, 
         'energy': -25.0, 'resolution': 2.2, 'helix_fraction': 0.2, 'sheet_fraction': 0.5},
        {'pdb_id': '1LYZ', 'protein_length': 129, 'rmsd': 2.9, 'gdt_ts': 72.0, 'tm_score': 0.61, 
         'energy': -41.0, 'resolution': 2.5, 'helix_fraction': 0.35, 'sheet_fraction': 0.15},
    ]
    
    # Calculate correlations
    correlations = analyzer.calculate_correlations(results)
    
    print("\n📊 Correlation Coefficients (Pearson):")
    print(f"  Size vs RMSD:         {correlations.size_vs_rmsd:+.3f}")
    print(f"  Size vs GDT-TS:       {correlations.size_vs_gdt_ts:+.3f}")
    print(f"  SS Content vs RMSD:   {correlations.secondary_structure_vs_rmsd:+.3f}")
    print(f"  Resolution vs RMSD:   {correlations.resolution_vs_accuracy:+.3f}")
    
    print("\n📈 Interpretation:")
    if abs(correlations.size_vs_rmsd) > 0.5:
        direction = "positive" if correlations.size_vs_rmsd > 0 else "negative"
        print(f"  • Strong {direction} correlation between size and RMSD")
    else:
        print(f"  • Weak correlation between size and RMSD")
    
    print()


def example2_size_category_comparison():
    """Example 2: Size category comparisons with ANOVA."""
    print("=" * 60)
    print("Example 2: Size Category Comparison (ANOVA)")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # Sample results with different size categories
    results = [
        {'pdb_id': '1UBQ', 'protein_length': 76, 'rmsd': 2.5, 'gdt_ts': 75.0, 'size_category': 'tiny'},
        {'pdb_id': '1CRN', 'protein_length': 46, 'rmsd': 1.8, 'gdt_ts': 85.0, 'size_category': 'tiny'},
        {'pdb_id': '2MR9', 'protein_length': 35, 'rmsd': 3.2, 'gdt_ts': 68.0, 'size_category': 'tiny'},
        {'pdb_id': '1LYZ', 'protein_length': 129, 'rmsd': 2.9, 'gdt_ts': 72.0, 'size_category': 'small'},
        {'pdb_id': 'TEST1', 'protein_length': 200, 'rmsd': 3.5, 'gdt_ts': 65.0, 'size_category': 'medium'},
        {'pdb_id': 'TEST2', 'protein_length': 210, 'rmsd': 3.8, 'gdt_ts': 62.0, 'size_category': 'medium'},
        {'pdb_id': 'TEST3', 'protein_length': 350, 'rmsd': 5.0, 'gdt_ts': 50.0, 'size_category': 'large'},
        {'pdb_id': 'TEST4', 'protein_length': 400, 'rmsd': 5.5, 'gdt_ts': 48.0, 'size_category': 'large'},
    ]
    
    # Compare RMSD across size categories
    print("\n🔬 Comparing RMSD across size categories...")
    comparison = analyzer.compare_size_categories(results, metric='rmsd')
    
    print("\n📊 Category Statistics (RMSD in Å):")
    for category in sorted(comparison.category_means.keys()):
        mean = comparison.category_means[category]
        std = comparison.category_stds[category]
        print(f"  {category:10s}: {mean:.2f} ± {std:.2f}")
    
    print("\n📈 Statistical Significance:")
    for test, p_value in comparison.p_values.items():
        significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else "ns"
        print(f"  {test}: p={p_value:.3f} {significance}")
    
    if comparison.effect_sizes:
        print("\n📐 Effect Sizes (Cohen's d):")
        for pair, effect_size in comparison.effect_sizes.items():
            magnitude = "large" if effect_size > 0.8 else "medium" if effect_size > 0.5 else "small"
            print(f"  {pair}: d={effect_size:.2f} ({magnitude})")
    
    print()


def example3_distribution_analysis():
    """Example 3: Distribution plot generation."""
    print("=" * 60)
    print("Example 3: Distribution Analysis")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # Sample results
    results = [
        {'pdb_id': f'TEST{i}', 'rmsd': 2.0 + i * 0.3, 'gdt_ts': 80.0 - i * 2.0, 
         'tm_score': 0.7 - i * 0.02, 'energy': -50.0 + i * 2.0}
        for i in range(10)
    ]
    
    # Generate distribution plots
    output_dir = "./statistical_analysis_output"
    plot_files = analyzer.generate_distribution_plots(results, output_dir=output_dir)
    
    print(f"\n📊 Generated {len(plot_files)} distribution plots:")
    for plot_file in plot_files:
        print(f"  ✅ {Path(plot_file).name}")
        
        # Display distribution statistics
        with open(plot_file, 'r') as f:
            stats = json.load(f)
        
        metric = stats['metric']
        print(f"     {metric.upper()}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
              f"std={stats['stdev']:.2f}")
        print(f"     Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"     Quartiles: Q1={stats['quartiles']['q1']:.2f}, "
              f"Q2={stats['quartiles']['q2']:.2f}, Q3={stats['quartiles']['q3']:.2f}")
    
    print()


def example4_predictive_features():
    """Example 4: Predictive feature identification."""
    print("=" * 60)
    print("Example 4: Predictive Feature Identification")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # Sample results with varying characteristics
    results = [
        {'pdb_id': '1UBQ', 'protein_length': 76, 'rmsd': 2.5, 'resolution': 1.8, 
         'helix_fraction': 0.3, 'sheet_fraction': 0.4},
        {'pdb_id': '1CRN', 'protein_length': 46, 'rmsd': 1.8, 'resolution': 1.5, 
         'helix_fraction': 0.4, 'sheet_fraction': 0.3},
        {'pdb_id': '2MR9', 'protein_length': 35, 'rmsd': 3.2, 'resolution': 2.0, 
         'helix_fraction': 0.5, 'sheet_fraction': 0.2},
        {'pdb_id': '1VII', 'protein_length': 36, 'rmsd': 4.5, 'resolution': 2.2, 
         'helix_fraction': 0.2, 'sheet_fraction': 0.5},
        {'pdb_id': '1LYZ', 'protein_length': 129, 'rmsd': 2.9, 'resolution': 2.5, 
         'helix_fraction': 0.35, 'sheet_fraction': 0.15},
        {'pdb_id': 'TEST1', 'protein_length': 200, 'rmsd': 3.5, 'resolution': 2.8, 
         'helix_fraction': 0.45, 'sheet_fraction': 0.25},
    ]
    
    # Identify predictive features for RMSD
    importance = analyzer.identify_predictive_features(results, target_metric='rmsd')
    
    print("\n🎯 Feature Importance Ranking (for RMSD prediction):")
    for i, (feature, score) in enumerate(zip(importance.features, importance.importance_scores), 1):
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        power = importance.predictive_power[feature]
        print(f"  {i}. {feature:20s} {bar} {score:.3f} (R²={power:.3f})")
    
    print("\n💡 Interpretation:")
    top_feature = importance.features[0]
    top_score = importance.importance_scores[0]
    print(f"  • Most predictive feature: {top_feature} (importance={top_score:.3f})")
    
    if importance.predictive_power[top_feature] > 0.5:
        print(f"  • Strong predictive power (R²={importance.predictive_power[top_feature]:.3f})")
    else:
        print(f"  • Moderate predictive power (R²={importance.predictive_power[top_feature]:.3f})")
    
    print()


def example5_confidence_intervals():
    """Example 5: Confidence interval calculations."""
    print("=" * 60)
    print("Example 5: Confidence Intervals")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # Sample results
    results = [
        {'pdb_id': f'TEST{i}', 'rmsd': 2.5 + i * 0.1, 'gdt_ts': 75.0 - i * 0.5, 
         'tm_score': 0.65 + i * 0.01, 'energy': -45.0 - i * 0.5}
        for i in range(20)
    ]
    
    # Calculate 95% confidence intervals
    ci_95 = analyzer.calculate_confidence_intervals(results, confidence_level=0.95)
    
    print("\n📊 95% Confidence Intervals:")
    print(f"  RMSD:     [{ci_95.rmsd_ci[0]:.2f}, {ci_95.rmsd_ci[1]:.2f}] Å")
    print(f"  GDT-TS:   [{ci_95.gdt_ts_ci[0]:.1f}, {ci_95.gdt_ts_ci[1]:.1f}]")
    print(f"  TM-score: [{ci_95.tm_score_ci[0]:.3f}, {ci_95.tm_score_ci[1]:.3f}]")
    print(f"  Energy:   [{ci_95.energy_ci[0]:.1f}, {ci_95.energy_ci[1]:.1f}] kcal/mol")
    
    # Calculate 99% confidence intervals
    ci_99 = analyzer.calculate_confidence_intervals(results, confidence_level=0.99)
    
    print("\n📊 99% Confidence Intervals (wider):")
    print(f"  RMSD:     [{ci_99.rmsd_ci[0]:.2f}, {ci_99.rmsd_ci[1]:.2f}] Å")
    print(f"  GDT-TS:   [{ci_99.gdt_ts_ci[0]:.1f}, {ci_99.gdt_ts_ci[1]:.1f}]")
    
    print("\n💡 Interpretation:")
    print("  • 95% CI: We're 95% confident the true mean lies within this range")
    print("  • 99% CI: Wider interval with higher confidence")
    
    # Compare interval widths
    width_95 = ci_95.rmsd_ci[1] - ci_95.rmsd_ci[0]
    width_99 = ci_99.rmsd_ci[1] - ci_99.rmsd_ci[0]
    print(f"  • RMSD interval width: 95%={width_95:.2f}Å, 99%={width_99:.2f}Å")
    
    print()


def example6_export_comprehensive_analysis():
    """Example 6: Export comprehensive analysis."""
    print("=" * 60)
    print("Example 6: Export Comprehensive Analysis")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # Sample results
    results = [
        {'pdb_id': '1UBQ', 'protein_length': 76, 'rmsd': 2.5, 'gdt_ts': 75.0, 'tm_score': 0.65, 
         'energy': -45.0, 'resolution': 1.8, 'helix_fraction': 0.3, 'sheet_fraction': 0.4, 
         'size_category': 'tiny'},
        {'pdb_id': '1CRN', 'protein_length': 46, 'rmsd': 1.8, 'gdt_ts': 85.0, 'tm_score': 0.82, 
         'energy': -52.0, 'resolution': 1.5, 'helix_fraction': 0.4, 'sheet_fraction': 0.3, 
         'size_category': 'tiny'},
        {'pdb_id': '2MR9', 'protein_length': 35, 'rmsd': 3.2, 'gdt_ts': 68.0, 'tm_score': 0.58, 
         'energy': -38.0, 'resolution': 2.0, 'helix_fraction': 0.5, 'sheet_fraction': 0.2, 
         'size_category': 'tiny'},
        {'pdb_id': '1LYZ', 'protein_length': 129, 'rmsd': 2.9, 'gdt_ts': 72.0, 'tm_score': 0.61, 
         'energy': -41.0, 'resolution': 2.5, 'helix_fraction': 0.35, 'sheet_fraction': 0.15, 
         'size_category': 'small'},
        {'pdb_id': 'TEST1', 'protein_length': 200, 'rmsd': 3.5, 'gdt_ts': 65.0, 'tm_score': 0.55, 
         'energy': -35.0, 'resolution': 2.8, 'helix_fraction': 0.45, 'sheet_fraction': 0.25, 
         'size_category': 'medium'},
    ]
    
    print("\n🔬 Running comprehensive statistical analysis...")
    
    # Perform all analyses
    correlations = analyzer.calculate_correlations(results)
    comparison = analyzer.compare_size_categories(results, metric='rmsd')
    importance = analyzer.identify_predictive_features(results, target_metric='rmsd')
    ci = analyzer.calculate_confidence_intervals(results)
    
    # Create comprehensive analysis dictionary
    analysis_results = {
        'correlations': correlations,
        'size_comparison': comparison,
        'feature_importance': importance,
        'confidence_intervals': ci,
        'summary': {
            'total_proteins': len(results),
            'size_categories': len(set(r['size_category'] for r in results)),
            'features_analyzed': len(importance.features)
        }
    }
    
    # Export to JSON
    output_dir = Path("./statistical_analysis_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "comprehensive_analysis.json"
    
    analyzer.export_analysis(str(output_file), analysis_results)
    
    print(f"\n✅ Analysis exported: {output_file}")
    
    # Display summary
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Analysis Summary:")
    print(f"  Total Proteins: {data['summary']['total_proteins']}")
    print(f"  Size Categories: {data['summary']['size_categories']}")
    print(f"  Features Analyzed: {data['summary']['features_analyzed']}")
    print(f"\n  Key Correlations:")
    print(f"    • Size vs RMSD: {data['correlations']['size_vs_rmsd']:.3f}")
    print(f"    • Size vs GDT-TS: {data['correlations']['size_vs_gdt_ts']:.3f}")
    print(f"\n  Feature Importance (top 3):")
    for i in range(min(3, len(data['feature_importance']['features']))):
        feat = data['feature_importance']['features'][i]
        score = data['feature_importance']['importance_scores'][i]
        print(f"    {i+1}. {feat}: {score:.3f}")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "🔬 StatisticalAnalyzer Examples ".center(60, "="))
    print()
    
    example1_correlation_analysis()
    example2_size_category_comparison()
    example3_distribution_analysis()
    example4_predictive_features()
    example5_confidence_intervals()
    example6_export_comprehensive_analysis()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
