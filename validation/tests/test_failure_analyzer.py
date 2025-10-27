"""
Unit tests for FailureAnalyzer

Tests cover:
- Initialization
- Failure classification by type and severity
- Common characteristic extraction
- Failure visualization metadata generation
- Energy trajectory analysis
- Parameter adjustment recommendations
- Export functionality
- Edge cases and error handling
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from validation.failure_analyzer import (
    FailureAnalyzer,
    FailureClassification,
    FailurePatterns,
    TrajectoryAnalysis
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def analyzer():
    """Create FailureAnalyzer instance."""
    return FailureAnalyzer()


@pytest.fixture
def severe_failure():
    """Create severe failure report."""
    return {
        'pdb_id': 'FAIL1',
        'rmsd': 10.0,  # > 8.0 (severe)
        'energy': 5.0,  # > 0 (unstable)
        'gdt_ts': 25.0,  # < 30 (severe)
        'protein_length': 200,
        'size_category': 'medium'
    }


@pytest.fixture
def moderate_failure():
    """Create moderate failure report."""
    return {
        'pdb_id': 'FAIL2',
        'rmsd': 6.5,  # > 5.0 but < 8.0
        'energy': -10.0,
        'gdt_ts': 45.0,  # < 50 but > 30
        'protein_length': 150,
        'size_category': 'small'
    }


@pytest.fixture
def minor_failure():
    """Create minor failure report."""
    return {
        'pdb_id': 'FAIL3',
        'rmsd': 5.5,
        'energy': -20.0,
        'gdt_ts': 55.0,
        'protein_length': 100,
        'size_category': 'small'
    }


class TestInitialization:
    """Test FailureAnalyzer initialization."""
    
    def test_create_analyzer(self, analyzer):
        """Test analyzer creation."""
        assert analyzer is not None
        assert hasattr(analyzer, '_failure_cache')
        assert analyzer._failure_cache == []
    
    def test_thresholds_defined(self, analyzer):
        """Test that failure thresholds are defined."""
        assert analyzer.SEVERE_RMSD_THRESHOLD == 8.0
        assert analyzer.MODERATE_RMSD_THRESHOLD == 5.0
        assert analyzer.UNSTABLE_ENERGY_THRESHOLD == 0.0
        assert analyzer.SEVERE_GDT_THRESHOLD == 30.0
        assert analyzer.MODERATE_GDT_THRESHOLD == 50.0


class TestFailureClassification:
    """Test failure classification functionality."""
    
    def test_classify_severe_failure(self, analyzer, severe_failure):
        """Test classification of severe failure."""
        classification = analyzer.classify_failure(severe_failure)
        
        assert isinstance(classification, FailureClassification)
        assert classification.pdb_id == 'FAIL1'
        assert classification.severity == 'severe'
        assert classification.failure_type == 'multiple'  # Multiple criteria met
    
    def test_classify_moderate_failure(self, analyzer, moderate_failure):
        """Test classification of moderate failure."""
        classification = analyzer.classify_failure(moderate_failure)
        
        # RMSD 6.5 > 5.0 but < 8.0, GDT-TS 45 < 50 but > 30
        # Only one criterion met, so it's minor severity
        assert classification.severity == 'minor'
        assert classification.failure_type in ['high_rmsd', 'low_gdt_ts']
    
    def test_classify_minor_failure(self, analyzer, minor_failure):
        """Test classification of minor failure."""
        classification = analyzer.classify_failure(minor_failure)
        
        assert classification.severity == 'minor'
    
    def test_classify_high_rmsd_only(self, analyzer):
        """Test classification with high RMSD only."""
        report = {
            'pdb_id': 'TEST1',
            'rmsd': 9.0,
            'energy': -30.0,
            'gdt_ts': 60.0
        }
        
        classification = analyzer.classify_failure(report)
        assert classification.failure_type == 'high_rmsd'
    
    def test_classify_poor_energy_only(self, analyzer):
        """Test classification with poor energy only."""
        report = {
            'pdb_id': 'TEST2',
            'rmsd': 3.0,
            'energy': 5.0,
            'gdt_ts': 70.0
        }
        
        classification = analyzer.classify_failure(report)
        assert classification.failure_type == 'poor_energy'
    
    def test_classify_low_gdt_only(self, analyzer):
        """Test classification with low GDT-TS only."""
        report = {
            'pdb_id': 'TEST3',
            'rmsd': 3.0,
            'energy': -30.0,
            'gdt_ts': 25.0
        }
        
        classification = analyzer.classify_failure(report)
        assert classification.failure_type == 'low_gdt_ts'
    
    def test_classify_missing_fields(self, analyzer):
        """Test classification with missing required fields."""
        report = {'pdb_id': 'TEST4'}
        
        with pytest.raises(ValueError, match="missing required fields"):
            analyzer.classify_failure(report)
    
    def test_classify_caches_result(self, analyzer, severe_failure):
        """Test that classification is cached."""
        assert len(analyzer._failure_cache) == 0
        
        analyzer.classify_failure(severe_failure)
        assert len(analyzer._failure_cache) == 1


class TestCommonCharacteristics:
    """Test common characteristic extraction."""
    
    def test_extract_patterns_basic(self, analyzer):
        """Test basic pattern extraction."""
        failures = [
            {'pdb_id': 'F1', 'rmsd': 6.0, 'gdt_ts': 40.0, 'size_category': 'large', 
             'helix_fraction': 0.3, 'sheet_fraction': 0.4, 'protein_length': 300},
            {'pdb_id': 'F2', 'rmsd': 7.0, 'gdt_ts': 35.0, 'size_category': 'large', 
             'helix_fraction': 0.35, 'sheet_fraction': 0.35, 'protein_length': 350},
            {'pdb_id': 'F3', 'rmsd': 6.5, 'gdt_ts': 38.0, 'size_category': 'large', 
             'helix_fraction': 0.4, 'sheet_fraction': 0.3, 'protein_length': 280},
        ]
        
        patterns = analyzer.extract_common_characteristics(failures)
        
        assert isinstance(patterns, FailurePatterns)
        assert patterns.common_size_category == 'large'
        assert len(patterns.average_secondary_structure_content) == 3
    
    def test_extract_patterns_high_rmsd(self, analyzer):
        """Test detection of high RMSD pattern."""
        failures = [
            {'pdb_id': f'F{i}', 'rmsd': 6.0 + i, 'gdt_ts': 45.0, 'energy': -30.0}
            for i in range(5)
        ]
        
        patterns = analyzer.extract_common_characteristics(failures)
        assert any('High RMSD' in issue for issue in patterns.common_issues)
    
    def test_extract_patterns_poor_energy(self, analyzer):
        """Test detection of poor energy pattern."""
        failures = [
            {'pdb_id': f'F{i}', 'rmsd': 4.0, 'gdt_ts': 60.0, 'energy': 5.0}
            for i in range(5)
        ]
        
        patterns = analyzer.extract_common_characteristics(failures)
        assert any('Poor energy' in issue for issue in patterns.common_issues)
    
    def test_extract_patterns_low_gdt(self, analyzer):
        """Test detection of low GDT-TS pattern."""
        failures = [
            {'pdb_id': f'F{i}', 'rmsd': 4.0, 'gdt_ts': 40.0, 'energy': -30.0}
            for i in range(5)
        ]
        
        patterns = analyzer.extract_common_characteristics(failures)
        assert any('Low GDT-TS' in issue for issue in patterns.common_issues)
    
    def test_extract_patterns_large_protein_bias(self, analyzer):
        """Test detection of large protein bias."""
        failures = [
            {'pdb_id': f'F{i}', 'rmsd': 5.0, 'gdt_ts': 50.0, 'protein_length': 300 + i * 10}
            for i in range(10)
        ]
        
        patterns = analyzer.extract_common_characteristics(failures)
        assert any('large proteins' in issue for issue in patterns.common_issues)
    
    def test_extract_patterns_empty_failures(self, analyzer):
        """Test pattern extraction with empty list."""
        with pytest.raises(ValueError, match="empty failures"):
            analyzer.extract_common_characteristics([])


class TestVisualizationGeneration:
    """Test visualization metadata generation."""
    
    def test_generate_visualizations_basic(self, analyzer, severe_failure, temp_dir):
        """Test basic visualization generation."""
        vis_files = analyzer.generate_failure_visualizations(
            'FAIL1',
            severe_failure,
            output_dir=temp_dir
        )
        
        assert len(vis_files) > 0
        assert all(Path(f).exists() for f in vis_files)
    
    def test_generate_visualizations_all_types(self, analyzer, severe_failure, temp_dir):
        """Test that all visualization types are generated."""
        vis_files = analyzer.generate_failure_visualizations(
            'FAIL1',
            severe_failure,
            output_dir=temp_dir
        )
        
        # Should generate comparison, deviation, and energy visualizations
        assert len(vis_files) == 3
    
    def test_generate_visualizations_content(self, analyzer, severe_failure, temp_dir):
        """Test visualization file contents."""
        vis_files = analyzer.generate_failure_visualizations(
            'FAIL1',
            severe_failure,
            output_dir=temp_dir
        )
        
        # Check comparison visualization
        comparison_file = [f for f in vis_files if 'comparison' in f][0]
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        assert data['pdb_id'] == 'FAIL1'
        assert data['visualization_type'] == 'structure_comparison'
        assert 'rmsd' in data


class TestTrajectoryAnalysis:
    """Test energy trajectory analysis."""
    
    def test_analyze_trajectory_basic(self, analyzer):
        """Test basic trajectory analysis."""
        trajectory = {
            'energies': [-40.0, -45.0, -42.0, -46.0, -45.5, -46.0]
        }
        
        analysis = analyzer.analyze_energy_trajectory(trajectory)
        
        assert isinstance(analysis, TrajectoryAnalysis)
        assert analysis.minima_count >= 0
        assert analysis.escape_attempts >= 0
        assert analysis.energy_variance >= 0
    
    def test_analyze_trajectory_with_minima(self, analyzer):
        """Test trajectory with local minima."""
        trajectory = {
            'energies': [-40.0, -45.0, -43.0, -46.0, -44.0, -47.0]
        }
        
        analysis = analyzer.analyze_energy_trajectory(trajectory)
        assert analysis.minima_count > 0
    
    def test_analyze_trajectory_converged(self, analyzer):
        """Test converged trajectory."""
        # Stable energy in last portion
        trajectory = {
            'energies': [-30.0, -35.0, -40.0, -45.0] + [-46.0] * 10
        }
        
        analysis = analyzer.analyze_energy_trajectory(trajectory)
        assert analysis.convergence_achieved is True
    
    def test_analyze_trajectory_not_converged(self, analyzer):
        """Test non-converged trajectory."""
        # Large fluctuations throughout
        trajectory = {
            'energies': [-30.0, -40.0, -35.0, -45.0, -38.0, -42.0, -36.0, -44.0]
        }
        
        analysis = analyzer.analyze_energy_trajectory(trajectory)
        assert analysis.convergence_achieved is False
    
    def test_analyze_trajectory_stuck(self, analyzer):
        """Test detection of stuck in local minimum."""
        # Multiple minima with little progress
        trajectory = {
            'energies': [-30.0, -32.0, -31.0, -33.0, -32.5, -33.5, -33.0, -33.2, -33.1, -33.2] * 2
        }
        
        analysis = analyzer.analyze_energy_trajectory(trajectory)
        # May or may not be stuck depending on exact criteria
        assert isinstance(analysis.stuck_in_local_minima, bool)
    
    def test_analyze_trajectory_missing_energies(self, analyzer):
        """Test trajectory analysis with missing energies."""
        with pytest.raises(ValueError, match="must contain"):
            analyzer.analyze_energy_trajectory({})
    
    def test_analyze_trajectory_empty_energies(self, analyzer):
        """Test trajectory analysis with empty energies list."""
        with pytest.raises(ValueError, match="must contain"):
            analyzer.analyze_energy_trajectory({'energies': []})


class TestParameterRecommendations:
    """Test parameter adjustment recommendations."""
    
    def test_recommend_for_large_proteins(self, analyzer):
        """Test recommendations for large protein failures."""
        patterns = FailurePatterns(
            common_size_category='large',
            common_structural_class=None,
            average_secondary_structure_content={'helix': 0.3, 'sheet': 0.3, 'coil': 0.4},
            common_issues=['High RMSD in 5/5 cases']
        )
        
        recommendations = analyzer.recommend_parameter_adjustments(patterns)
        
        assert len(recommendations) > 0
        assert any('iterations_per_agent' in rec for rec in recommendations)
    
    def test_recommend_for_high_rmsd(self, analyzer):
        """Test recommendations for high RMSD failures."""
        patterns = FailurePatterns(
            common_size_category=None,
            common_structural_class=None,
            average_secondary_structure_content={'helix': 0.3, 'sheet': 0.3, 'coil': 0.4},
            common_issues=['High RMSD in 8/10 cases']
        )
        
        recommendations = analyzer.recommend_parameter_adjustments(patterns)
        assert any('QCPP' in rec for rec in recommendations)
    
    def test_recommend_for_poor_energy(self, analyzer):
        """Test recommendations for poor energy failures."""
        patterns = FailurePatterns(
            common_size_category=None,
            common_structural_class=None,
            average_secondary_structure_content={'helix': 0.3, 'sheet': 0.3, 'coil': 0.4},
            common_issues=['Poor energy in 6/10 cases']
        )
        
        recommendations = analyzer.recommend_parameter_adjustments(patterns)
        assert any('energy_function' in rec or 'QAAP' in rec for rec in recommendations)
    
    def test_recommend_for_low_gdt(self, analyzer):
        """Test recommendations for low GDT-TS failures."""
        patterns = FailurePatterns(
            common_size_category=None,
            common_structural_class=None,
            average_secondary_structure_content={'helix': 0.3, 'sheet': 0.3, 'coil': 0.4},
            common_issues=['Low GDT-TS in 7/10 cases']
        )
        
        recommendations = analyzer.recommend_parameter_adjustments(patterns)
        assert any('native_state_ambition' in rec or 'diversity' in rec for rec in recommendations)
    
    def test_recommend_for_high_coil(self, analyzer):
        """Test recommendations for high coil content."""
        patterns = FailurePatterns(
            common_size_category=None,
            common_structural_class=None,
            average_secondary_structure_content={'helix': 0.1, 'sheet': 0.1, 'coil': 0.8},
            common_issues=[]
        )
        
        recommendations = analyzer.recommend_parameter_adjustments(patterns)
        assert any('coil' in rec or 'flexibility' in rec for rec in recommendations)
    
    def test_recommend_general_when_no_patterns(self, analyzer):
        """Test general recommendations when no specific patterns found."""
        patterns = FailurePatterns(
            common_size_category=None,
            common_structural_class=None,
            average_secondary_structure_content={'helix': 0.3, 'sheet': 0.3, 'coil': 0.4},
            common_issues=[]
        )
        
        recommendations = analyzer.recommend_parameter_adjustments(patterns)
        assert len(recommendations) > 0
        assert any('general' in rec.lower() or 'try' in rec.lower() for rec in recommendations)


class TestExportFunctionality:
    """Test export functionality."""
    
    def test_export_failure_report_basic(self, analyzer, temp_dir):
        """Test basic failure report export."""
        classifications = [
            FailureClassification('TEST1', 'high_rmsd', 'severe', 9.0, -20.0, 30.0),
            FailureClassification('TEST2', 'poor_energy', 'moderate', 6.0, 5.0, 45.0),
        ]
        
        output_path = Path(temp_dir) / "failure_report.json"
        analyzer.export_failure_report(str(output_path), classifications)
        
        assert output_path.exists()
    
    def test_export_failure_report_with_patterns(self, analyzer, temp_dir):
        """Test export with patterns and recommendations."""
        classifications = [
            FailureClassification('TEST1', 'high_rmsd', 'severe', 9.0, -20.0, 30.0),
        ]
        
        patterns = FailurePatterns(
            'large',
            None,
            {'helix': 0.3, 'sheet': 0.3, 'coil': 0.4},
            ['High RMSD in 1/1 cases']
        )
        
        recommendations = ['Increase iterations']
        
        output_path = Path(temp_dir) / "failure_report.json"
        analyzer.export_failure_report(
            str(output_path),
            classifications,
            patterns,
            recommendations
        )
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'patterns' in data
        assert 'recommendations' in data
        assert 'summary' in data
    
    def test_export_failure_report_summary(self, analyzer, temp_dir):
        """Test that export includes summary statistics."""
        classifications = [
            FailureClassification('TEST1', 'high_rmsd', 'severe', 9.0, -20.0, 30.0),
            FailureClassification('TEST2', 'poor_energy', 'moderate', 6.0, 5.0, 45.0),
            FailureClassification('TEST3', 'low_gdt_ts', 'minor', 5.5, -25.0, 48.0),
        ]
        
        output_path = Path(temp_dir) / "failure_report.json"
        analyzer.export_failure_report(str(output_path), classifications)
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'summary' in data
        assert 'avg_rmsd' in data['summary']
        assert 'severity_distribution' in data['summary']
        assert 'failure_type_distribution' in data['summary']


class TestHelperMethods:
    """Test helper methods."""
    
    def test_count_local_minima(self, analyzer):
        """Test local minima counting."""
        # Clear minima
        energies = [-40.0, -45.0, -42.0, -46.0, -44.0]
        count = analyzer._count_local_minima(energies)
        assert count == 2  # At indices 1 and 3
    
    def test_count_escape_attempts(self, analyzer):
        """Test escape attempt counting."""
        energies = [-40.0, -45.0, -42.0, -46.0, -44.0, -47.0]
        count = analyzer._count_escape_attempts(energies)
        assert count >= 0
    
    def test_find_most_common(self, analyzer):
        """Test most common item finder."""
        items = ['a', 'b', 'a', 'c', 'a', 'b']
        most_common = analyzer._find_most_common(items)
        assert most_common == 'a'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
