"""
Unit tests for comparative benchmarking functionality (Task 11).

Tests cover:
- BaselineResult and IntegratedResult data models
- ComparisonMetrics calculation
- StatisticalSignificance testing
- BenchmarkReport generation
- ComparativeBenchmark analysis
- Error handling and edge cases
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from comparative_benchmarking import (
    BaselineResult,
    IntegratedResult,
    ComparisonMetrics,
    StatisticalSignificance,
    BenchmarkReport,
    ComparativeBenchmark
)
from protein_selector import ProteinMetadata


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_baseline_result():
    """Sample baseline (UBF-only) result."""
    return BaselineResult(
        pdb_id="1TST",
        rmsd=2.5,
        gdt_ts=75.0,
        tm_score=0.75,
        final_energy=-50.0,
        runtime_seconds=120.0,
        conformations_explored=1000,
        success=True
    )


@pytest.fixture
def sample_integrated_result():
    """Sample integrated (UBF+QCPP) result."""
    return IntegratedResult(
        pdb_id="1TST",
        rmsd=2.0,
        gdt_ts=80.0,
        tm_score=0.80,
        final_energy=-55.0,
        runtime_seconds=150.0,
        conformations_explored=1000,
        qcpp_overhead_seconds=30.0,
        qcpp_cache_hit_rate=0.35,
        success=True
    )


@pytest.fixture
def sample_protein_metadata():
    """Sample protein metadata."""
    return ProteinMetadata(
        pdb_id="1TST",
        sequence_length=50,
        resolution=2.0,
        experimental_method="X-RAY DIFFRACTION",
        structural_class="all-alpha",
        size_category="small",
        missing_residues_pct=0.0,
        organism="Test organism",
        description="Test protein"
    )


# ============================================================================
# Test BaselineResult Data Model
# ============================================================================

class TestBaselineResult:
    """Tests for BaselineResult dataclass."""
    
    def test_baseline_result_creation(self, sample_baseline_result):
        """Test creating a baseline result."""
        assert sample_baseline_result.pdb_id == "1TST"
        assert sample_baseline_result.rmsd == 2.5
        assert sample_baseline_result.gdt_ts == 75.0
        assert sample_baseline_result.tm_score == 0.75
        assert sample_baseline_result.success is True
    
    def test_baseline_result_serialization(self, sample_baseline_result):
        """Test serializing baseline result to dict."""
        from dataclasses import asdict
        result_dict = asdict(sample_baseline_result)
        
        assert isinstance(result_dict, dict)
        assert result_dict['pdb_id'] == "1TST"
        assert result_dict['rmsd'] == 2.5


# ============================================================================
# Test IntegratedResult Data Model
# ============================================================================

class TestIntegratedResult:
    """Tests for IntegratedResult dataclass."""
    
    def test_integrated_result_creation(self, sample_integrated_result):
        """Test creating an integrated result."""
        assert sample_integrated_result.pdb_id == "1TST"
        assert sample_integrated_result.rmsd == 2.0
        assert sample_integrated_result.qcpp_overhead_seconds == 30.0
        assert sample_integrated_result.qcpp_cache_hit_rate == 0.35
    
    def test_integrated_result_with_qcpp_data(self):
        """Test integrated result includes QCPP-specific data."""
        result = IntegratedResult(
            pdb_id="2TST",
            rmsd=1.8,
            gdt_ts=85.0,
            tm_score=0.85,
            final_energy=-60.0,
            runtime_seconds=180.0,
            conformations_explored=1200,
            qcpp_overhead_seconds=40.0,
            qcpp_cache_hit_rate=0.42,
            success=True
        )
        
        assert result.qcpp_overhead_seconds > 0
        assert 0 <= result.qcpp_cache_hit_rate <= 1


# ============================================================================
# Test ComparisonMetrics Calculation
# ============================================================================

class TestComparisonMetrics:
    """Tests for ComparisonMetrics dataclass."""
    
    def test_comparison_metrics_creation(self):
        """Test creating comparison metrics."""
        metrics = ComparisonMetrics(
            pdb_id="1TST",
            rmsd_delta=-0.5,  # Improved (lower)
            rmsd_improvement_pct=20.0,
            gdt_ts_delta=5.0,  # Improved (higher)
            gdt_ts_improvement_pct=6.67,
            tm_score_delta=0.05,  # Improved (higher)
            energy_delta=-5.0,  # Improved (lower)
            runtime_overhead_seconds=30.0,
            runtime_overhead_pct=25.0,
            success_change="maintained_success"
        )
        
        assert metrics.rmsd_delta < 0  # Improvement
        assert metrics.gdt_ts_delta > 0  # Improvement
        assert metrics.runtime_overhead_pct > 0
    
    def test_improvement_percentages(self):
        """Test improvement percentage calculations."""
        metrics = ComparisonMetrics(
            pdb_id="2TST",
            rmsd_delta=-1.0,
            rmsd_improvement_pct=40.0,
            gdt_ts_delta=10.0,
            gdt_ts_improvement_pct=15.0,
            tm_score_delta=0.10,
            energy_delta=-10.0,
            runtime_overhead_seconds=50.0,
            runtime_overhead_pct=30.0,
            success_change="gained_success"
        )
        
        assert metrics.rmsd_improvement_pct == 40.0
        assert metrics.gdt_ts_improvement_pct == 15.0
        assert metrics.success_change == "gained_success"


# ============================================================================
# Test StatisticalSignificance
# ============================================================================

class TestStatisticalSignificance:
    """Tests for StatisticalSignificance dataclass."""
    
    def test_statistical_significance_creation(self):
        """Test creating statistical significance result."""
        sig = StatisticalSignificance(
            metric_name="RMSD",
            t_statistic=-3.5,
            p_value=0.002,
            significant=True,
            effect_size=0.85,
            mean_difference=-0.5,
            confidence_interval=(-0.8, -0.2)
        )
        
        assert sig.metric_name == "RMSD"
        assert sig.p_value < 0.05
        assert sig.significant is True
        assert sig.effect_size > 0
    
    def test_non_significant_result(self):
        """Test non-significant statistical result."""
        sig = StatisticalSignificance(
            metric_name="Energy",
            t_statistic=-1.2,
            p_value=0.15,
            significant=False,
            effect_size=0.25,
            mean_difference=-2.0,
            confidence_interval=(-5.0, 1.0)
        )
        
        assert sig.p_value > 0.05
        assert sig.significant is False


# ============================================================================
# Test BenchmarkReport
# ============================================================================

class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""
    
    def test_benchmark_report_creation(self, sample_baseline_result, 
                                      sample_integrated_result):
        """Test creating a benchmark report."""
        metrics = ComparisonMetrics(
            pdb_id="1TST",
            rmsd_delta=-0.5,
            rmsd_improvement_pct=20.0,
            gdt_ts_delta=5.0,
            gdt_ts_improvement_pct=6.67,
            tm_score_delta=0.05,
            energy_delta=-5.0,
            runtime_overhead_seconds=30.0,
            runtime_overhead_pct=25.0,
            success_change="maintained_success"
        )
        
        sig = StatisticalSignificance(
            metric_name="RMSD",
            t_statistic=-3.5,
            p_value=0.002,
            significant=True,
            effect_size=0.85,
            mean_difference=-0.5,
            confidence_interval=(-0.8, -0.2)
        )
        
        report = BenchmarkReport(
            benchmark_id="bench_001",
            timestamp="2024-01-15T10:00:00",
            total_proteins=1,
            baseline_results=[sample_baseline_result],
            integrated_results=[sample_integrated_result],
            comparison_metrics=[metrics],
            statistical_tests=[sig],
            summary_statistics={"mean_rmsd_improvement": 20.0},
            computational_overhead={"mean_overhead_pct": 25.0}
        )
        
        assert report.benchmark_id == "bench_001"
        assert report.total_proteins == 1
        assert len(report.baseline_results) == 1
        assert len(report.integrated_results) == 1
        assert len(report.comparison_metrics) == 1
        assert len(report.statistical_tests) == 1


# ============================================================================
# Test ComparativeBenchmark Core Functionality
# ============================================================================

class TestComparativeBenchmarkInit:
    """Tests for ComparativeBenchmark initialization."""
    
    def test_benchmark_creation(self, temp_output_dir):
        """Test creating a comparative benchmark."""
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        assert benchmark.output_dir == Path(temp_output_dir)
        assert benchmark.output_dir.exists()
    
    def test_benchmark_with_custom_config(self, temp_output_dir):
        """Test benchmark with custom configuration."""
        benchmark = ComparativeBenchmark(
            output_dir=temp_output_dir
        )
        
        assert benchmark.output_dir.exists()
        # Configuration is passed to run_benchmark method, not constructor


class TestComparativeBenchmarkExecution:
    """Tests for benchmark execution."""
    
    @patch('validation.comparative_benchmarking.ComparativeBenchmark._run_baseline_test')
    @patch('validation.comparative_benchmarking.ComparativeBenchmark._run_integrated_test')
    def test_run_single_protein_benchmark(self, mock_integrated, mock_baseline, 
                                         temp_output_dir, sample_protein_metadata,
                                         sample_baseline_result, sample_integrated_result):
        """Test running benchmark on single protein."""
        mock_baseline.return_value = sample_baseline_result
        mock_integrated.return_value = sample_integrated_result
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        report = benchmark.run_benchmark(proteins=[sample_protein_metadata])
        
        assert isinstance(report, BenchmarkReport)
        assert report.total_proteins == 1
        assert len(report.baseline_results) == 1
        assert len(report.integrated_results) == 1
        
        mock_baseline.assert_called_once()
        mock_integrated.assert_called_once()
    
    @patch('validation.comparative_benchmarking.ComparativeBenchmark._run_baseline_test')
    @patch('validation.comparative_benchmarking.ComparativeBenchmark._run_integrated_test')
    def test_run_multiple_protein_benchmark(self, mock_integrated, mock_baseline, 
                                           temp_output_dir):
        """Test running benchmark on multiple proteins."""
        proteins = [
            ProteinMetadata(
                pdb_id=f"1TST{i}",
                sequence_length=50 + i * 10,
                resolution=2.0,
                experimental_method="X-RAY DIFFRACTION",
                structural_class="all-alpha",
                size_category="small",
                missing_residues_pct=0.0,
                organism="Test organism",
                description="Test protein"
            )
            for i in range(3)
        ]
        
        # Mock returns
        mock_baseline.side_effect = [
            BaselineResult("1TST0", 2.5, 75.0, 0.75, -50.0, 120.0, 1000, True),
            BaselineResult("1TST1", 2.3, 77.0, 0.77, -52.0, 125.0, 1100, True),
            BaselineResult("1TST2", 2.7, 73.0, 0.73, -48.0, 115.0, 950, True)
        ]
        mock_integrated.side_effect = [
            IntegratedResult("1TST0", 2.0, 80.0, 0.80, -55.0, 150.0, 1000, 30.0, 0.35, True),
            IntegratedResult("1TST1", 1.8, 82.0, 0.82, -58.0, 155.0, 1100, 30.0, 0.38, True),
            IntegratedResult("1TST2", 2.2, 78.0, 0.78, -52.0, 145.0, 950, 30.0, 0.32, True)
        ]
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        report = benchmark.run_benchmark(proteins=proteins)
        
        assert report.total_proteins == 3
        assert len(report.baseline_results) == 3
        assert len(report.integrated_results) == 3
        assert len(report.comparison_metrics) == 3


class TestComparativeBenchmarkAnalysis:
    """Tests for comparative analysis functionality."""
    
    def test_calculate_comparison_metrics(self, temp_output_dir, 
                                         sample_baseline_result, sample_integrated_result):
        """Test calculating comparison metrics."""
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        metrics_list = benchmark._calculate_comparison_metrics(
            [sample_baseline_result],
            [sample_integrated_result]
        )
        
        assert isinstance(metrics_list, list)
        assert len(metrics_list) == 1
        metrics = metrics_list[0]
        assert isinstance(metrics, ComparisonMetrics)
        assert metrics.pdb_id == "1TST"
        assert metrics.rmsd_delta < 0  # Improved
        assert metrics.gdt_ts_delta > 0  # Improved
        assert metrics.runtime_overhead_seconds > 0
    
    def test_perform_statistical_tests(self, temp_output_dir):
        """Test performing statistical significance tests."""
        baseline_results = [
            BaselineResult(f"1TST{i}", 2.5 - i*0.2, 75.0, 0.75, -50.0, 120.0, 1000, True)
            for i in range(5)
        ]
        integrated_results = [
            IntegratedResult(f"1TST{i}", 2.0 - i*0.2, 80.0, 0.80, -55.0, 150.0, 1000, 30.0, 0.35, True)
            for i in range(5)
        ]
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        tests = benchmark._perform_statistical_tests(
            baseline_results=baseline_results,
            integrated_results=integrated_results
        )
        
        assert isinstance(tests, list)
        # Should have tests for RMSD, GDT-TS, TM-score, Energy
        assert len(tests) >= 1
        for test in tests:
            assert isinstance(test, StatisticalSignificance)
            assert test.p_value >= 0
    
    def test_calculate_summary_statistics(self, temp_output_dir):
        """Test calculating summary statistics."""
        metrics = [
            ComparisonMetrics(
                pdb_id=f"1TST{i}",
                rmsd_delta=-0.5,
                rmsd_improvement_pct=20.0,
                gdt_ts_delta=5.0,
                gdt_ts_improvement_pct=6.67,
                tm_score_delta=0.05,
                energy_delta=-5.0,
                runtime_overhead_seconds=30.0,
                runtime_overhead_pct=25.0,
                success_change="maintained_success"
            )
            for i in range(3)
        ]
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        summary = benchmark._calculate_summary_statistics(metrics)
        
        assert isinstance(summary, dict)
        assert 'mean_rmsd_improvement' in summary
        assert 'mean_runtime_overhead' in summary


class TestComparativeBenchmarkReporting:
    """Tests for report generation and export."""
    
    def test_export_report_json(self, temp_output_dir, sample_baseline_result, 
                               sample_integrated_result):
        """Test exporting report to JSON."""
        metrics = ComparisonMetrics(
            pdb_id="1TST",
            rmsd_delta=-0.5,
            rmsd_improvement_pct=20.0,
            gdt_ts_delta=5.0,
            gdt_ts_improvement_pct=6.67,
            tm_score_delta=0.05,
            energy_delta=-5.0,
            runtime_overhead_seconds=30.0,
            runtime_overhead_pct=25.0,
            success_change="maintained_success"
        )
        
        report = BenchmarkReport(
            benchmark_id="bench_001",
            timestamp="2024-01-15T10:00:00",
            total_proteins=1,
            baseline_results=[sample_baseline_result],
            integrated_results=[sample_integrated_result],
            comparison_metrics=[metrics],
            statistical_tests=[],
            summary_statistics={},
            computational_overhead={}
        )
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        output_file = Path(temp_output_dir) / "report.json"
        
        benchmark.export_report(report, str(output_file))
        
        assert output_file.exists()
        
        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
        
        assert data['benchmark_id'] == "bench_001"
        assert data['total_proteins'] == 1
    
    def test_generate_text_summary(self, temp_output_dir, sample_baseline_result, 
                                   sample_integrated_result):
        """Test generating text summary."""
        metrics = ComparisonMetrics(
            pdb_id="1TST",
            rmsd_delta=-0.5,
            rmsd_improvement_pct=20.0,
            gdt_ts_delta=5.0,
            gdt_ts_improvement_pct=6.67,
            tm_score_delta=0.05,
            energy_delta=-5.0,
            runtime_overhead_seconds=30.0,
            runtime_overhead_pct=25.0,
            success_change="maintained_success"
        )
        
        report = BenchmarkReport(
            benchmark_id="bench_001",
            timestamp="2024-01-15T10:00:00",
            total_proteins=1,
            baseline_results=[sample_baseline_result],
            integrated_results=[sample_integrated_result],
            comparison_metrics=[metrics],
            statistical_tests=[],
            summary_statistics={"mean_rmsd_improvement": 20.0},
            computational_overhead={"mean_overhead_pct": 25.0}
        )
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        text_summary = benchmark.generate_markdown_report(report)
        
        assert isinstance(text_summary, str)
        assert len(text_summary) > 0


class TestComparativeBenchmarkErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_benchmark_with_empty_protein_list(self, temp_output_dir):
        """Test benchmarking with empty protein list."""
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        with pytest.raises((ValueError, IndexError)):
            benchmark.run_benchmark(proteins=[])
    
    def test_benchmark_with_failed_proteins(self, temp_output_dir):
        """Test handling proteins that fail prediction."""
        baseline_failed = BaselineResult(
            pdb_id="FAIL",
            rmsd=999.0,
            gdt_ts=0.0,
            tm_score=0.0,
            final_energy=0.0,
            runtime_seconds=60.0,
            conformations_explored=0,
            success=False
        )
        
        integrated_failed = IntegratedResult(
            pdb_id="FAIL",
            rmsd=999.0,
            gdt_ts=0.0,
            tm_score=0.0,
            final_energy=0.0,
            runtime_seconds=90.0,
            conformations_explored=0,
            qcpp_overhead_seconds=30.0,
            qcpp_cache_hit_rate=0.0,
            success=False
        )
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        # Should handle failed proteins gracefully
        metrics_list = benchmark._calculate_comparison_metrics(
            [baseline_failed],
            [integrated_failed]
        )
        
        assert len(metrics_list) == 1
        assert metrics_list[0].success_change == "maintained_failure"
    
    def test_mismatched_protein_ids(self, temp_output_dir):
        """Test handling mismatched protein IDs."""
        baseline = BaselineResult("1TST", 2.5, 75.0, 0.75, -50.0, 120.0, 1000, True)
        integrated = IntegratedResult("2TST", 2.0, 80.0, 0.80, -55.0, 150.0, 1000, 30.0, 0.35, True)
        
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        
        with pytest.raises((ValueError, AssertionError)):
            benchmark._calculate_comparison_metrics([baseline], [integrated])


# ============================================================================
# Integration Tests
# ============================================================================

class TestComparativeBenchmarkIntegration:
    """Integration tests for complete benchmarking workflows."""
    
    @patch('validation.comparative_benchmarking.ComparativeBenchmark._run_baseline_test')
    @patch('validation.comparative_benchmarking.ComparativeBenchmark._run_integrated_test')
    def test_full_benchmark_workflow(self, mock_integrated, mock_baseline, temp_output_dir):
        """Test complete benchmark workflow."""
        # Setup
        proteins = [
            ProteinMetadata(
                pdb_id=f"1TST{i}",
                sequence_length=50 + i * 10,
                resolution=2.0,
                experimental_method="X-RAY DIFFRACTION",
                structural_class="all-alpha",
                size_category="small",
                missing_residues_pct=0.0,
                organism="Test organism",
                description="Test protein"
            )
            for i in range(2)
        ]
        
        mock_baseline.side_effect = [
            BaselineResult("1TST0", 2.5, 75.0, 0.75, -50.0, 120.0, 1000, True),
            BaselineResult("1TST1", 2.3, 77.0, 0.77, -52.0, 125.0, 1100, True)
        ]
        mock_integrated.side_effect = [
            IntegratedResult("1TST0", 2.0, 80.0, 0.80, -55.0, 150.0, 1000, 30.0, 0.35, True),
            IntegratedResult("1TST1", 1.8, 82.0, 0.82, -58.0, 155.0, 1100, 30.0, 0.38, True)
        ]
        
        # Run benchmark
        benchmark = ComparativeBenchmark(output_dir=temp_output_dir)
        report = benchmark.run_benchmark(proteins=proteins)
        
        # Verify report
        assert isinstance(report, BenchmarkReport)
        assert report.total_proteins == 2
        
        # Export report
        output_file = Path(temp_output_dir) / "benchmark_report.json"
        benchmark.export_report(report, str(output_file))
        assert output_file.exists()
        
        # Generate markdown report
        md_file = Path(temp_output_dir) / "benchmark_report.md"
        benchmark.generate_markdown_report(report, str(md_file))
        assert md_file.exists()
