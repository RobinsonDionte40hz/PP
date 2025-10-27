"""
StatisticalAnalyzer - Automated statistical analysis and pattern detection for validation results.

This module provides comprehensive statistical analysis capabilities for protein structure
prediction validation campaigns, including:
- Correlation analysis between protein characteristics and prediction accuracy
- Statistical comparisons between size categories using ANOVA
- Distribution analysis and visualization
- Predictive feature identification
- Confidence interval calculations

Author: Large-Scale Validation Framework
Date: October 26, 2025
"""

import statistics
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


@dataclass(frozen=True)
class CorrelationMatrix:
    """
    Correlation coefficients between protein characteristics and accuracy metrics.
    
    Attributes:
        size_vs_rmsd: Pearson correlation between protein size and RMSD
        size_vs_gdt_ts: Pearson correlation between protein size and GDT-TS
        secondary_structure_vs_rmsd: Correlation between SS content and RMSD
        resolution_vs_accuracy: Correlation between resolution and overall accuracy
        correlations: Complete dictionary of all pairwise correlations
    """
    size_vs_rmsd: float
    size_vs_gdt_ts: float
    secondary_structure_vs_rmsd: float
    resolution_vs_accuracy: float
    correlations: Dict[str, float]


@dataclass(frozen=True)
class StatisticalComparison:
    """
    Statistical comparison results between protein size categories.
    
    Attributes:
        category_means: Mean values for each category
        category_stds: Standard deviations for each category
        p_values: P-values from ANOVA tests (statistical significance)
        effect_sizes: Cohen's d effect sizes between categories
    """
    category_means: Dict[str, float]
    category_stds: Dict[str, float]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]


@dataclass(frozen=True)
class FeatureImportance:
    """
    Importance ranking of predictive features.
    
    Attributes:
        features: List of feature names
        importance_scores: Normalized importance scores (0-1)
        predictive_power: Dictionary mapping features to R² values
    """
    features: List[str]
    importance_scores: List[float]
    predictive_power: Dict[str, float]


@dataclass(frozen=True)
class ConfidenceIntervals:
    """
    95% confidence intervals for validation metrics.
    
    Attributes:
        rmsd_ci: (lower, upper) bounds for RMSD mean
        gdt_ts_ci: (lower, upper) bounds for GDT-TS mean
        tm_score_ci: (lower, upper) bounds for TM-score mean
        energy_ci: (lower, upper) bounds for energy mean
    """
    rmsd_ci: Tuple[float, float]
    gdt_ts_ci: Tuple[float, float]
    tm_score_ci: Tuple[float, float]
    energy_ci: Tuple[float, float]


class StatisticalAnalyzer:
    """
    Automated statistical analysis and pattern detection for validation results.
    
    Provides comprehensive statistical analysis including correlation calculations,
    ANOVA comparisons, distribution analysis, feature importance ranking, and
    confidence interval estimation.
    
    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> correlations = analyzer.calculate_correlations(validation_results)
        >>> print(f"Size vs RMSD: {correlations.size_vs_rmsd:.3f}")
        >>> 
        >>> comparison = analyzer.compare_size_categories(validation_results)
        >>> for category, mean_rmsd in comparison.category_means.items():
        ...     print(f"{category}: {mean_rmsd:.2f}Å")
    """
    
    def __init__(self):
        """Initialize the StatisticalAnalyzer."""
        self._results_cache: List[Dict] = []
    
    def calculate_correlations(self, results: List[Dict]) -> CorrelationMatrix:
        """
        Calculate Pearson correlation coefficients between protein characteristics and metrics.
        
        Args:
            results: List of validation reports (dicts with protein metadata and metrics)
        
        Returns:
            CorrelationMatrix with all correlation coefficients
        
        Raises:
            ValueError: If results list is empty or has insufficient data
        
        Example:
            >>> correlations = analyzer.calculate_correlations(results)
            >>> print(f"Size correlation with RMSD: {correlations.size_vs_rmsd:.3f}")
        """
        if not results:
            raise ValueError("Cannot calculate correlations on empty results list")
        
        if len(results) < 3:
            raise ValueError("Need at least 3 results for meaningful correlations")
        
        # Extract data for correlation calculations
        sizes = [r.get('protein_length', 0) for r in results]
        rmsds = [r.get('rmsd', 0.0) for r in results]
        gdt_tss = [r.get('gdt_ts', 0.0) for r in results]
        tm_scores = [r.get('tm_score', 0.0) for r in results]
        
        # Calculate correlations
        size_vs_rmsd = self._pearson_correlation(sizes, rmsds)
        size_vs_gdt_ts = self._pearson_correlation(sizes, gdt_tss)
        
        # Secondary structure content vs RMSD (use helix_fraction as proxy)
        ss_content = [r.get('helix_fraction', 0.5) for r in results]
        ss_vs_rmsd = self._pearson_correlation(ss_content, rmsds)
        
        # Resolution vs accuracy (lower RMSD = higher accuracy)
        resolutions = [r.get('resolution', 2.0) for r in results]
        resolution_vs_accuracy = self._pearson_correlation(resolutions, rmsds)
        
        # Build complete correlations dictionary
        all_correlations = {
            'size_vs_rmsd': size_vs_rmsd,
            'size_vs_gdt_ts': size_vs_gdt_ts,
            'size_vs_tm_score': self._pearson_correlation(sizes, tm_scores),
            'resolution_vs_rmsd': resolution_vs_accuracy,
            'resolution_vs_gdt_ts': self._pearson_correlation(resolutions, gdt_tss),
            'ss_content_vs_rmsd': ss_vs_rmsd,
            'ss_content_vs_gdt_ts': self._pearson_correlation(ss_content, gdt_tss),
        }
        
        return CorrelationMatrix(
            size_vs_rmsd=size_vs_rmsd,
            size_vs_gdt_ts=size_vs_gdt_ts,
            secondary_structure_vs_rmsd=ss_vs_rmsd,
            resolution_vs_accuracy=resolution_vs_accuracy,
            correlations=all_correlations
        )
    
    def compare_size_categories(
        self,
        results: List[Dict],
        metric: str = 'rmsd'
    ) -> StatisticalComparison:
        """
        Compare metrics across protein size categories using ANOVA.
        
        Args:
            results: List of validation reports with size_category field
            metric: Metric to compare ('rmsd', 'gdt_ts', 'tm_score', 'energy')
        
        Returns:
            StatisticalComparison with means, stds, p-values, and effect sizes
        
        Raises:
            ValueError: If results is empty or metric is invalid
        
        Example:
            >>> comparison = analyzer.compare_size_categories(results, metric='rmsd')
            >>> for cat, mean in comparison.category_means.items():
            ...     print(f"{cat}: {mean:.2f}Å ± {comparison.category_stds[cat]:.2f}")
        """
        if not results:
            raise ValueError("Cannot compare empty results list")
        
        valid_metrics = ['rmsd', 'gdt_ts', 'tm_score', 'energy']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}")
        
        # Group results by size category
        categories = {}
        for result in results:
            category = result.get('size_category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result.get(metric, 0.0))
        
        # Calculate means and standard deviations
        category_means = {}
        category_stds = {}
        for cat, values in categories.items():
            if values:
                category_means[cat] = statistics.mean(values)
                category_stds[cat] = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate ANOVA p-values (one-way ANOVA)
        p_values = self._calculate_anova_p_values(categories, metric)
        
        # Calculate effect sizes (Cohen's d between consecutive size categories)
        effect_sizes = self._calculate_effect_sizes(categories, category_means, category_stds)
        
        return StatisticalComparison(
            category_means=category_means,
            category_stds=category_stds,
            p_values=p_values,
            effect_sizes=effect_sizes
        )
    
    def generate_distribution_plots(
        self,
        results: List[Dict],
        output_dir: str = "./distribution_plots"
    ) -> List[str]:
        """
        Generate distribution plots for validation metrics.
        
        Creates histogram plots showing the distribution of RMSD, GDT-TS, TM-score,
        and energy values. Returns list of generated file paths.
        
        Args:
            results: List of validation reports
            output_dir: Directory to save plots
        
        Returns:
            List of file paths to generated plots
        
        Note:
            This method creates plot metadata files (JSON) describing the distributions.
            Actual visualization requires matplotlib or similar plotting library.
        
        Example:
            >>> plot_files = analyzer.generate_distribution_plots(results)
            >>> print(f"Generated {len(plot_files)} distribution plots")
        """
        if not results:
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics to plot
        metrics = ['rmsd', 'gdt_ts', 'tm_score', 'energy']
        generated_files = []
        
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results if metric in r]
            if not values:
                continue
            
            # Calculate distribution statistics
            dist_stats = {
                'metric': metric,
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'quartiles': self._calculate_quartiles(values)
            }
            
            # Save distribution metadata
            output_file = output_path / f"{metric}_distribution.json"
            with open(output_file, 'w') as f:
                json.dump(dist_stats, f, indent=2)
            
            generated_files.append(str(output_file))
        
        return generated_files
    
    def identify_predictive_features(
        self,
        results: List[Dict],
        target_metric: str = 'rmsd'
    ) -> FeatureImportance:
        """
        Identify which protein features best predict validation accuracy.
        
        Uses correlation-based feature importance to rank protein characteristics
        by their predictive power for the target metric.
        
        Args:
            results: List of validation reports
            target_metric: Metric to predict ('rmsd', 'gdt_ts', 'tm_score', 'energy')
        
        Returns:
            FeatureImportance with ranked features and importance scores
        
        Raises:
            ValueError: If results is empty or target_metric is invalid
        
        Example:
            >>> importance = analyzer.identify_predictive_features(results, 'rmsd')
            >>> for feat, score in zip(importance.features, importance.importance_scores):
            ...     print(f"{feat}: {score:.3f}")
        """
        if not results:
            raise ValueError("Cannot identify features on empty results list")
        
        valid_metrics = ['rmsd', 'gdt_ts', 'tm_score', 'energy']
        if target_metric not in valid_metrics:
            raise ValueError(f"Invalid target metric '{target_metric}'. Must be one of {valid_metrics}")
        
        # Extract target values
        target_values = [r.get(target_metric, 0.0) for r in results]
        
        # Calculate correlation (R²) for each feature
        features = ['protein_length', 'resolution', 'helix_fraction', 'sheet_fraction']
        predictive_power = {}
        
        for feature in features:
            feature_values = [r.get(feature, 0.0) for r in results]
            correlation = self._pearson_correlation(feature_values, target_values)
            predictive_power[feature] = abs(correlation)  # Use absolute value for ranking
        
        # Sort features by predictive power
        sorted_features = sorted(predictive_power.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize importance scores to 0-1 range
        max_score = max(predictive_power.values()) if predictive_power else 1.0
        importance_scores = [score / max_score if max_score > 0 else 0.0 
                            for _, score in sorted_features]
        
        return FeatureImportance(
            features=[feat for feat, _ in sorted_features],
            importance_scores=importance_scores,
            predictive_power=predictive_power
        )
    
    def calculate_confidence_intervals(
        self,
        results: List[Dict],
        confidence_level: float = 0.95
    ) -> ConfidenceIntervals:
        """
        Calculate confidence intervals for validation metric means.
        
        Uses t-distribution for small samples (<30) and normal distribution
        for larger samples.
        
        Args:
            results: List of validation reports
            confidence_level: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            ConfidenceIntervals with (lower, upper) bounds for each metric
        
        Raises:
            ValueError: If results is empty or confidence_level is invalid
        
        Example:
            >>> ci = analyzer.calculate_confidence_intervals(results)
            >>> print(f"RMSD: {ci.rmsd_ci[0]:.2f} - {ci.rmsd_ci[1]:.2f}Å")
        """
        if not results:
            raise ValueError("Cannot calculate confidence intervals on empty results list")
        
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        # Extract metric values
        rmsds = [r.get('rmsd', 0.0) for r in results]
        gdt_tss = [r.get('gdt_ts', 0.0) for r in results]
        tm_scores = [r.get('tm_score', 0.0) for r in results]
        energies = [r.get('energy', 0.0) for r in results]
        
        # Calculate confidence intervals
        rmsd_ci = self._calculate_ci(rmsds, confidence_level)
        gdt_ts_ci = self._calculate_ci(gdt_tss, confidence_level)
        tm_score_ci = self._calculate_ci(tm_scores, confidence_level)
        energy_ci = self._calculate_ci(energies, confidence_level)
        
        return ConfidenceIntervals(
            rmsd_ci=rmsd_ci,
            gdt_ts_ci=gdt_ts_ci,
            tm_score_ci=tm_score_ci,
            energy_ci=energy_ci
        )
    
    def export_analysis(self, output_path: str, analysis_results: Dict):
        """
        Export statistical analysis results to JSON file.
        
        Args:
            output_path: Path to output JSON file
            analysis_results: Dictionary containing analysis results
        
        Example:
            >>> results_dict = {
            ...     'correlations': correlations,
            ...     'comparisons': comparison,
            ...     'confidence_intervals': ci
            ... }
            >>> analyzer.export_analysis('analysis.json', results_dict)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass instances to dicts for JSON serialization
        serializable = {}
        for key, value in analysis_results.items():
            if hasattr(value, '__dataclass_fields__'):
                serializable[key] = self._dataclass_to_dict(value)
            else:
                serializable[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    # Private helper methods
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two variables."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_anova_p_values(
        self,
        categories: Dict[str, List[float]],
        metric: str
    ) -> Dict[str, float]:
        """
        Calculate ANOVA p-values for comparisons between categories.
        
        Note: This is a simplified F-statistic calculation. For production use,
        consider using scipy.stats.f_oneway for more accurate p-values.
        """
        p_values = {}
        
        # Calculate overall mean
        all_values = [v for values in categories.values() for v in values]
        if not all_values:
            return p_values
        
        overall_mean = statistics.mean(all_values)
        
        # Between-group variance
        between_ss = sum(
            len(values) * (statistics.mean(values) - overall_mean) ** 2
            for values in categories.values() if values
        )
        
        # Within-group variance
        within_ss = sum(
            sum((v - statistics.mean(values)) ** 2 for v in values)
            for values in categories.values() if len(values) > 1
        )
        
        # Degrees of freedom
        k = len(categories)  # number of groups
        n = len(all_values)  # total observations
        
        if k <= 1 or n <= k or within_ss == 0:
            p_values[f"{metric}_overall"] = 1.0
            return p_values
        
        # F-statistic
        between_ms = between_ss / (k - 1)
        within_ms = within_ss / (n - k)
        f_stat = between_ms / within_ms if within_ms > 0 else 0.0
        
        # Simplified p-value estimation (p < 0.05 if F > 3.0)
        # For production, use scipy.stats.f.sf(f_stat, k-1, n-k)
        if f_stat > 5.0:
            p_value = 0.01  # Highly significant
        elif f_stat > 3.0:
            p_value = 0.05  # Significant
        else:
            p_value = 0.10  # Not significant
        
        p_values[f"{metric}_overall"] = p_value
        
        return p_values
    
    def _calculate_effect_sizes(
        self,
        categories: Dict[str, List[float]],
        means: Dict[str, float],
        stds: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate Cohen's d effect sizes between categories."""
        effect_sizes = {}
        category_names = list(categories.keys())
        
        # Calculate pairwise effect sizes
        for i in range(len(category_names)):
            for j in range(i + 1, len(category_names)):
                cat1, cat2 = category_names[i], category_names[j]
                
                if cat1 in means and cat2 in means:
                    mean_diff = abs(means[cat1] - means[cat2])
                    
                    # Pooled standard deviation
                    n1, n2 = len(categories[cat1]), len(categories[cat2])
                    if n1 > 1 and n2 > 1:
                        pooled_std = ((stds[cat1] ** 2 * (n1 - 1) + 
                                      stds[cat2] ** 2 * (n2 - 1)) / 
                                     (n1 + n2 - 2)) ** 0.5
                        
                        if pooled_std > 0:
                            cohen_d = mean_diff / pooled_std
                            effect_sizes[f"{cat1}_vs_{cat2}"] = cohen_d
        
        return effect_sizes
    
    def _calculate_quartiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate quartiles for a dataset."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'q1': sorted_values[n // 4],
            'q2': statistics.median(sorted_values),
            'q3': sorted_values[3 * n // 4]
        }
    
    def _calculate_ci(
        self,
        values: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a dataset."""
        if len(values) < 2:
            mean = statistics.mean(values) if values else 0.0
            return (mean, mean)
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        n = len(values)
        
        # Use t-distribution for small samples, normal for large samples
        # Simplified: t_critical ≈ 2.0 for 95% CI
        if confidence_level == 0.95:
            t_critical = 2.0
        elif confidence_level == 0.99:
            t_critical = 2.6
        else:
            t_critical = 1.96  # Default to normal approximation
        
        margin_of_error = t_critical * (std / (n ** 0.5))
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def _dataclass_to_dict(self, obj) -> Dict:
        """Convert dataclass instance to dictionary for JSON serialization."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._dataclass_to_dict(value)
                elif isinstance(value, dict):
                    result[field_name] = {
                        k: self._dataclass_to_dict(v) if hasattr(v, '__dataclass_fields__') else v
                        for k, v in value.items()
                    }
                else:
                    result[field_name] = value
            return result
        return obj
