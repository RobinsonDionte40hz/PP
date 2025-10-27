"""
Unit tests for StatisticalAnalyzer

Tests cover:
- Initialization
- Correlation calculations (Pearson coefficients)
- Size category comparisons (ANOVA)
- Distribution plot generation
- Predictive feature identification
- Confidence interval calculations
- Export functionality
- Edge cases and error handling
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from validation.statistical_analyzer import (
    StatisticalAnalyzer,
    CorrelationMatrix,
    StatisticalComparison,
    FeatureImportance,
    ConfidenceIntervals
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def analyzer():
    """Create StatisticalAnalyzer instance."""
    return StatisticalAnalyzer()


@pytest.fixture
def sample_results():
    """Create sample validation results."""
    return [
        {
            'pdb_id': '1UBQ',
            'protein_length': 76,
            'rmsd': 2.5,
            'gdt_ts': 75.0,
            'tm_score': 0.65,
            'energy': -45.0,
            'resolution': 1.8,
            'helix_fraction': 0.3,
            'sheet_fraction': 0.4,
            'size_category': 'tiny'
        },
        {
            'pdb_id': '1CRN',
            'protein_length': 46,
            'rmsd': 1.8,
            'gdt_ts': 85.0,
            'tm_score': 0.82,
            'energy': -52.0,
            'resolution': 1.5,
            'helix_fraction': 0.4,
            'sheet_fraction': 0.3,
            'size_category': 'tiny'
        },
        {
            'pdb_id': '2MR9',
            'protein_length': 35,
            'rmsd': 3.2,
            'gdt_ts': 68.0,
            'tm_score': 0.58,
            'energy': -38.0,
            'resolution': 2.0,
            'helix_fraction': 0.5,
            'sheet_fraction': 0.2,
            'size_category': 'tiny'
        },
        {
            'pdb_id': '1VII',
            'protein_length': 36,
            'rmsd': 4.5,
            'gdt_ts': 55.0,
            'tm_score': 0.52,
            'energy': -25.0,
            'resolution': 2.2,
            'helix_fraction': 0.2,
            'sheet_fraction': 0.5,
            'size_category': 'tiny'
        },
        {
            'pdb_id': '1LYZ',
            'protein_length': 129,
            'rmsd': 2.9,
            'gdt_ts': 72.0,
            'tm_score': 0.61,
            'energy': -41.0,
            'resolution': 2.5,
            'helix_fraction': 0.35,
            'sheet_fraction': 0.15,
            'size_category': 'small'
        },
        {
            'pdb_id': 'TEST1',
            'protein_length': 200,
            'rmsd': 3.5,
            'gdt_ts': 65.0,
            'tm_score': 0.55,
            'energy': -35.0,
            'resolution': 2.8,
            'helix_fraction': 0.45,
            'sheet_fraction': 0.25,
            'size_category': 'medium'
        },
        {
            'pdb_id': 'TEST2',
            'protein_length': 350,
            'rmsd': 5.0,
            'gdt_ts': 50.0,
            'tm_score': 0.48,
            'energy': -30.0,
            'resolution': 3.0,
            'helix_fraction': 0.3,
            'sheet_fraction': 0.4,
            'size_category': 'large'
        },
    ]


class TestInitialization:
    """Test StatisticalAnalyzer initialization."""
    
    def test_create_analyzer(self, analyzer):
        """Test analyzer creation."""
        assert analyzer is not None
        assert hasattr(analyzer, '_results_cache')
        assert analyzer._results_cache == []


class TestCorrelationCalculations:
    """Test correlation coefficient calculations."""
    
    def test_calculate_correlations_basic(self, analyzer, sample_results):
        """Test basic correlation calculation."""
        correlations = analyzer.calculate_correlations(sample_results)
        
        assert isinstance(correlations, CorrelationMatrix)
        assert -1.0 <= correlations.size_vs_rmsd <= 1.0
        assert -1.0 <= correlations.size_vs_gdt_ts <= 1.0
        assert -1.0 <= correlations.secondary_structure_vs_rmsd <= 1.0
        assert -1.0 <= correlations.resolution_vs_accuracy <= 1.0
    
    def test_calculate_correlations_complete_dict(self, analyzer, sample_results):
        """Test that all correlations are calculated."""
        correlations = analyzer.calculate_correlations(sample_results)
        
        expected_keys = [
            'size_vs_rmsd',
            'size_vs_gdt_ts',
            'size_vs_tm_score',
            'resolution_vs_rmsd',
            'resolution_vs_gdt_ts',
            'ss_content_vs_rmsd',
            'ss_content_vs_gdt_ts'
        ]
        
        for key in expected_keys:
            assert key in correlations.correlations
            assert -1.0 <= correlations.correlations[key] <= 1.0
    
    def test_calculate_correlations_empty_results(self, analyzer):
        """Test correlation calculation with empty results."""
        with pytest.raises(ValueError, match="empty results"):
            analyzer.calculate_correlations([])
    
    def test_calculate_correlations_insufficient_data(self, analyzer):
        """Test correlation calculation with insufficient data."""
        results = [
            {'pdb_id': 'TEST1', 'protein_length': 50, 'rmsd': 2.5, 'gdt_ts': 75.0, 'tm_score': 0.65}
        ]
        
        with pytest.raises(ValueError, match="at least 3 results"):
            analyzer.calculate_correlations(results)
    
    def test_pearson_correlation_perfect_positive(self, analyzer):
        """Test perfect positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        
        corr = analyzer._pearson_correlation(x, y)
        assert abs(corr - 1.0) < 0.01  # Should be ~1.0
    
    def test_pearson_correlation_perfect_negative(self, analyzer):
        """Test perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        
        corr = analyzer._pearson_correlation(x, y)
        assert abs(corr - (-1.0)) < 0.01  # Should be ~-1.0
    
    def test_pearson_correlation_no_correlation(self, analyzer):
        """Test zero correlation."""
        x = [1, 2, 3, 4, 5]
        y = [5, 3, 5, 3, 5]
        
        corr = analyzer._pearson_correlation(x, y)
        assert abs(corr) < 0.5  # Should be close to 0


class TestSizeCategoryComparison:
    """Test size category statistical comparisons."""
    
    def test_compare_size_categories_rmsd(self, analyzer, sample_results):
        """Test RMSD comparison across size categories."""
        comparison = analyzer.compare_size_categories(sample_results, metric='rmsd')
        
        assert isinstance(comparison, StatisticalComparison)
        assert len(comparison.category_means) > 0
        assert 'tiny' in comparison.category_means
        assert comparison.category_means['tiny'] > 0
    
    def test_compare_size_categories_gdt_ts(self, analyzer, sample_results):
        """Test GDT-TS comparison across size categories."""
        comparison = analyzer.compare_size_categories(sample_results, metric='gdt_ts')
        
        assert isinstance(comparison, StatisticalComparison)
        assert all(mean > 0 for mean in comparison.category_means.values())
    
    def test_compare_size_categories_all_metrics(self, analyzer, sample_results):
        """Test comparison for all supported metrics."""
        metrics = ['rmsd', 'gdt_ts', 'tm_score', 'energy']
        
        for metric in metrics:
            comparison = analyzer.compare_size_categories(sample_results, metric=metric)
            assert isinstance(comparison, StatisticalComparison)
            assert len(comparison.category_means) > 0
    
    def test_compare_size_categories_empty_results(self, analyzer):
        """Test comparison with empty results."""
        with pytest.raises(ValueError, match="empty results"):
            analyzer.compare_size_categories([])
    
    def test_compare_size_categories_invalid_metric(self, analyzer, sample_results):
        """Test comparison with invalid metric."""
        with pytest.raises(ValueError, match="Invalid metric"):
            analyzer.compare_size_categories(sample_results, metric='invalid')
    
    def test_compare_size_categories_p_values(self, analyzer, sample_results):
        """Test that p-values are calculated."""
        comparison = analyzer.compare_size_categories(sample_results, metric='rmsd')
        
        assert len(comparison.p_values) > 0
        for p_value in comparison.p_values.values():
            assert 0 <= p_value <= 1.0
    
    def test_compare_size_categories_effect_sizes(self, analyzer, sample_results):
        """Test that effect sizes are calculated."""
        comparison = analyzer.compare_size_categories(sample_results, metric='rmsd')
        
        # Should have pairwise effect sizes
        assert isinstance(comparison.effect_sizes, dict)


class TestDistributionPlots:
    """Test distribution plot generation."""
    
    def test_generate_distribution_plots_basic(self, analyzer, sample_results, temp_dir):
        """Test basic distribution plot generation."""
        plot_files = analyzer.generate_distribution_plots(sample_results, output_dir=temp_dir)
        
        assert len(plot_files) > 0
        assert all(Path(f).exists() for f in plot_files)
    
    def test_generate_distribution_plots_all_metrics(self, analyzer, sample_results, temp_dir):
        """Test that plots are generated for all metrics."""
        plot_files = analyzer.generate_distribution_plots(sample_results, output_dir=temp_dir)
        
        # Should have plots for rmsd, gdt_ts, tm_score, energy
        assert len(plot_files) == 4
        
        metrics = ['rmsd', 'gdt_ts', 'tm_score', 'energy']
        for metric in metrics:
            assert any(metric in f for f in plot_files)
    
    def test_generate_distribution_plots_content(self, analyzer, sample_results, temp_dir):
        """Test distribution plot file contents."""
        plot_files = analyzer.generate_distribution_plots(sample_results, output_dir=temp_dir)
        
        # Check that JSON files contain expected statistics
        for plot_file in plot_files:
            with open(plot_file, 'r') as f:
                data = json.load(f)
            
            assert 'metric' in data
            assert 'mean' in data
            assert 'median' in data
            assert 'stdev' in data
            assert 'quartiles' in data
    
    def test_generate_distribution_plots_empty_results(self, analyzer, temp_dir):
        """Test plot generation with empty results."""
        plot_files = analyzer.generate_distribution_plots([], output_dir=temp_dir)
        assert plot_files == []


class TestPredictiveFeatures:
    """Test predictive feature identification."""
    
    def test_identify_predictive_features_rmsd(self, analyzer, sample_results):
        """Test feature identification for RMSD prediction."""
        importance = analyzer.identify_predictive_features(sample_results, target_metric='rmsd')
        
        assert isinstance(importance, FeatureImportance)
        assert len(importance.features) > 0
        assert len(importance.features) == len(importance.importance_scores)
    
    def test_identify_predictive_features_all_metrics(self, analyzer, sample_results):
        """Test feature identification for all metrics."""
        metrics = ['rmsd', 'gdt_ts', 'tm_score', 'energy']
        
        for metric in metrics:
            importance = analyzer.identify_predictive_features(sample_results, target_metric=metric)
            assert isinstance(importance, FeatureImportance)
            assert len(importance.features) > 0
    
    def test_identify_predictive_features_scores_normalized(self, analyzer, sample_results):
        """Test that importance scores are normalized."""
        importance = analyzer.identify_predictive_features(sample_results, target_metric='rmsd')
        
        # Scores should be normalized to 0-1
        assert all(0 <= score <= 1.0 for score in importance.importance_scores)
        # Top feature should have score of 1.0
        assert max(importance.importance_scores) == 1.0
    
    def test_identify_predictive_features_sorted(self, analyzer, sample_results):
        """Test that features are sorted by importance."""
        importance = analyzer.identify_predictive_features(sample_results, target_metric='rmsd')
        
        # Importance scores should be in descending order
        for i in range(len(importance.importance_scores) - 1):
            assert importance.importance_scores[i] >= importance.importance_scores[i + 1]
    
    def test_identify_predictive_features_empty_results(self, analyzer):
        """Test feature identification with empty results."""
        with pytest.raises(ValueError, match="empty results"):
            analyzer.identify_predictive_features([])
    
    def test_identify_predictive_features_invalid_metric(self, analyzer, sample_results):
        """Test feature identification with invalid metric."""
        with pytest.raises(ValueError, match="Invalid target metric"):
            analyzer.identify_predictive_features(sample_results, target_metric='invalid')
    
    def test_identify_predictive_features_power_dict(self, analyzer, sample_results):
        """Test that predictive power dictionary is complete."""
        importance = analyzer.identify_predictive_features(sample_results, target_metric='rmsd')
        
        expected_features = ['protein_length', 'resolution', 'helix_fraction', 'sheet_fraction']
        for feature in expected_features:
            assert feature in importance.predictive_power
            assert 0 <= importance.predictive_power[feature] <= 1.0


class TestConfidenceIntervals:
    """Test confidence interval calculations."""
    
    def test_calculate_confidence_intervals_basic(self, analyzer, sample_results):
        """Test basic confidence interval calculation."""
        ci = analyzer.calculate_confidence_intervals(sample_results)
        
        assert isinstance(ci, ConfidenceIntervals)
        assert len(ci.rmsd_ci) == 2
        assert len(ci.gdt_ts_ci) == 2
        assert len(ci.tm_score_ci) == 2
        assert len(ci.energy_ci) == 2
    
    def test_calculate_confidence_intervals_bounds(self, analyzer, sample_results):
        """Test that CI bounds are reasonable."""
        ci = analyzer.calculate_confidence_intervals(sample_results)
        
        # Lower bound should be less than upper bound
        assert ci.rmsd_ci[0] < ci.rmsd_ci[1]
        assert ci.gdt_ts_ci[0] < ci.gdt_ts_ci[1]
        assert ci.tm_score_ci[0] < ci.tm_score_ci[1]
        assert ci.energy_ci[0] < ci.energy_ci[1]
    
    def test_calculate_confidence_intervals_95(self, analyzer, sample_results):
        """Test 95% confidence intervals."""
        ci = analyzer.calculate_confidence_intervals(sample_results, confidence_level=0.95)
        
        # Calculate sample mean for RMSD
        rmsds = [r['rmsd'] for r in sample_results]
        mean_rmsd = sum(rmsds) / len(rmsds)
        
        # Mean should be within CI
        assert ci.rmsd_ci[0] <= mean_rmsd <= ci.rmsd_ci[1]
    
    def test_calculate_confidence_intervals_99(self, analyzer, sample_results):
        """Test 99% confidence intervals (should be wider than 95%)."""
        ci_95 = analyzer.calculate_confidence_intervals(sample_results, confidence_level=0.95)
        ci_99 = analyzer.calculate_confidence_intervals(sample_results, confidence_level=0.99)
        
        # 99% CI should be wider than 95% CI
        width_95 = ci_95.rmsd_ci[1] - ci_95.rmsd_ci[0]
        width_99 = ci_99.rmsd_ci[1] - ci_99.rmsd_ci[0]
        assert width_99 > width_95
    
    def test_calculate_confidence_intervals_empty_results(self, analyzer):
        """Test CI calculation with empty results."""
        with pytest.raises(ValueError, match="empty results"):
            analyzer.calculate_confidence_intervals([])
    
    def test_calculate_confidence_intervals_invalid_level(self, analyzer, sample_results):
        """Test CI calculation with invalid confidence level."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            analyzer.calculate_confidence_intervals(sample_results, confidence_level=1.5)


class TestExportFunctionality:
    """Test export functionality."""
    
    def test_export_analysis_basic(self, analyzer, temp_dir):
        """Test basic analysis export."""
        output_path = Path(temp_dir) / "analysis.json"
        
        analysis_results = {
            'test_key': 'test_value',
            'test_number': 42
        }
        
        analyzer.export_analysis(str(output_path), analysis_results)
        
        assert output_path.exists()
    
    def test_export_analysis_with_dataclasses(self, analyzer, sample_results, temp_dir):
        """Test export with dataclass instances."""
        output_path = Path(temp_dir) / "analysis.json"
        
        correlations = analyzer.calculate_correlations(sample_results)
        ci = analyzer.calculate_confidence_intervals(sample_results)
        
        analysis_results = {
            'correlations': correlations,
            'confidence_intervals': ci
        }
        
        analyzer.export_analysis(str(output_path), analysis_results)
        
        assert output_path.exists()
        
        # Verify JSON is valid and readable
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'correlations' in data
        assert 'confidence_intervals' in data
    
    def test_export_analysis_content(self, analyzer, sample_results, temp_dir):
        """Test exported analysis content."""
        output_path = Path(temp_dir) / "analysis.json"
        
        correlations = analyzer.calculate_correlations(sample_results)
        
        analysis_results = {
            'correlations': correlations
        }
        
        analyzer.export_analysis(str(output_path), analysis_results)
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'correlations' in data
        assert 'size_vs_rmsd' in data['correlations']
        assert 'size_vs_gdt_ts' in data['correlations']


class TestHelperMethods:
    """Test helper methods."""
    
    def test_calculate_quartiles(self, analyzer):
        """Test quartile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        quartiles = analyzer._calculate_quartiles(values)
        
        assert 'q1' in quartiles
        assert 'q2' in quartiles
        assert 'q3' in quartiles
        assert quartiles['q1'] < quartiles['q2'] < quartiles['q3']
    
    def test_calculate_quartiles_empty(self, analyzer):
        """Test quartile calculation with empty list."""
        quartiles = analyzer._calculate_quartiles([])
        assert quartiles == {}
    
    def test_dataclass_to_dict(self, analyzer, sample_results):
        """Test dataclass to dictionary conversion."""
        correlations = analyzer.calculate_correlations(sample_results)
        result = analyzer._dataclass_to_dict(correlations)
        
        assert isinstance(result, dict)
        assert 'size_vs_rmsd' in result
        assert 'size_vs_gdt_ts' in result
        assert 'correlations' in result


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_category(self, analyzer):
        """Test comparison with single category."""
        results = [
            {'pdb_id': 'TEST1', 'protein_length': 50, 'rmsd': 2.5, 'size_category': 'tiny'},
            {'pdb_id': 'TEST2', 'protein_length': 51, 'rmsd': 2.6, 'size_category': 'tiny'},
            {'pdb_id': 'TEST3', 'protein_length': 52, 'rmsd': 2.7, 'size_category': 'tiny'},
        ]
        
        comparison = analyzer.compare_size_categories(results, metric='rmsd')
        assert len(comparison.category_means) == 1
        assert 'tiny' in comparison.category_means
    
    def test_missing_fields(self, analyzer):
        """Test with missing fields in results."""
        results = [
            {'pdb_id': 'TEST1'},  # Missing most fields
            {'pdb_id': 'TEST2', 'rmsd': 2.5},
            {'pdb_id': 'TEST3', 'protein_length': 50},
        ]
        
        # Should handle missing fields gracefully
        correlations = analyzer.calculate_correlations(results)
        assert isinstance(correlations, CorrelationMatrix)
    
    def test_identical_values(self, analyzer):
        """Test correlation with identical values."""
        x = [1, 1, 1, 1, 1]
        y = [2, 2, 2, 2, 2]
        
        # Should return 0 (no variance)
        corr = analyzer._pearson_correlation(x, y)
        assert corr == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
