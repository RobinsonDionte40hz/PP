"""
Unit tests for ProgressTracker

Tests cover:
- Initialization and configuration
- Progress updates
- Running average calculations
- Outlier detection
- Dashboard data generation
- Interim report generation
- Trend analysis
- Issue detection
- Recommendations
- Export functionality
- Summary statistics
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from validation.progress_tracker import (
    ProgressTracker,
    DashboardData,
    InterimReport,
    RunningAverages
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def tracker():
    """Create ProgressTracker instance."""
    return ProgressTracker(
        total_tests=10,
        phase=1,
        success_threshold=5.0,
        outlier_threshold=2.0,
        recent_window=5
    )


@pytest.fixture
def sample_reports():
    """Create sample validation reports."""
    return [
        {"pdb_id": "1UBQ", "rmsd": 2.5, "gdt_ts": 75.0, "tm_score": 0.65, "final_energy": -45.0},
        {"pdb_id": "1CRN", "rmsd": 1.8, "gdt_ts": 85.0, "tm_score": 0.82, "final_energy": -52.0},
        {"pdb_id": "2MR9", "rmsd": 3.2, "gdt_ts": 68.0, "tm_score": 0.58, "final_energy": -38.0},
        {"pdb_id": "1VII", "rmsd": 4.5, "gdt_ts": 55.0, "tm_score": 0.52, "final_energy": -25.0},
        {"pdb_id": "1LYZ", "rmsd": 2.9, "gdt_ts": 72.0, "tm_score": 0.61, "final_energy": -41.0},
    ]


class TestInitialization:
    """Test ProgressTracker initialization."""
    
    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        tracker = ProgressTracker(total_tests=60)
        
        assert tracker.total_tests == 60
        assert tracker.phase == 1
        assert tracker.success_threshold == 5.0
        assert tracker.outlier_threshold == 2.0
        assert tracker.recent_window == 10
        assert tracker.interim_report_intervals == [0.25, 0.5, 0.75]
    
    def test_create_with_custom_params(self):
        """Test creation with custom parameters."""
        tracker = ProgressTracker(
            total_tests=30,
            phase=2,
            success_threshold=4.0,
            outlier_threshold=3.0,
            recent_window=15,
            interim_report_intervals=[0.33, 0.67]
        )
        
        assert tracker.total_tests == 30
        assert tracker.phase == 2
        assert tracker.success_threshold == 4.0
        assert tracker.outlier_threshold == 3.0
        assert tracker.recent_window == 15
        assert tracker.interim_report_intervals == [0.33, 0.67]


class TestProgressUpdates:
    """Test progress update functionality."""
    
    def test_update_single_report(self, tracker):
        """Test updating with a single report."""
        report = {"pdb_id": "1UBQ", "rmsd": 2.5, "gdt_ts": 75.0, "tm_score": 0.65, "final_energy": -45.0}
        
        tracker.update_progress(report)
        
        assert len(tracker._completed_tests) == 1
        assert len(tracker._rmsds) == 1
        assert tracker._rmsds[0] == 2.5
        assert len(tracker._gdt_tss) == 1
        assert tracker._gdt_tss[0] == 75.0
    
    def test_update_multiple_reports(self, tracker, sample_reports):
        """Test updating with multiple reports."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        assert len(tracker._completed_tests) == 5
        assert len(tracker._rmsds) == 5
        assert len(tracker._gdt_tss) == 5
        assert len(tracker._tm_scores) == 5
        assert len(tracker._energies) == 5
    
    def test_recent_completions_window(self, tracker):
        """Test that recent completions respects window size."""
        # Add more reports than window size
        for i in range(10):
            tracker.update_progress({
                "pdb_id": f"TEST{i}",
                "rmsd": 2.5,
                "gdt_ts": 75.0,
                "tm_score": 0.65,
                "final_energy": -45.0
            })
        
        # Should only keep last 5 (recent_window=5)
        assert len(tracker._recent_completions) == 5
        assert tracker._recent_completions[-1] == "TEST9"


class TestRunningAverages:
    """Test running average calculations."""
    
    def test_calculate_averages_empty(self, tracker):
        """Test averages with no data."""
        averages = tracker.calculate_running_averages()
        
        assert isinstance(averages, RunningAverages)
        assert averages.rmsd == 0.0
        assert averages.gdt_ts == 0.0
        assert averages.tm_score == 0.0
        assert averages.energy == 0.0
        assert averages.count == 0
    
    def test_calculate_averages_with_data(self, tracker, sample_reports):
        """Test averages with data."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        averages = tracker.calculate_running_averages()
        
        assert averages.count == 5
        assert 2.0 < averages.rmsd < 4.0
        assert 60.0 < averages.gdt_ts < 80.0
        assert 0.5 < averages.tm_score < 0.7
        assert -55.0 < averages.energy < -30.0


class TestOutlierDetection:
    """Test outlier detection functionality."""
    
    def test_outlier_detection_insufficient_data(self, tracker):
        """Test outlier detection with insufficient data."""
        # Add only 2 reports (need 3 for outlier detection)
        tracker.update_progress({"pdb_id": "TEST1", "rmsd": 2.5, "gdt_ts": 75.0, "tm_score": 0.65, "final_energy": -45.0})
        tracker.update_progress({"pdb_id": "TEST2", "rmsd": 2.6, "gdt_ts": 74.0, "tm_score": 0.64, "final_energy": -44.0})
        
        outliers = tracker.identify_outliers()
        assert outliers == []
    
    def test_outlier_detection_with_outlier(self, tracker):
        """Test outlier detection with clear outlier."""
        # Add normal reports
        for i in range(5):
            tracker.update_progress({
                "pdb_id": f"NORMAL{i}",
                "rmsd": 2.5 + i * 0.1,
                "gdt_ts": 75.0,
                "tm_score": 0.65,
                "final_energy": -45.0
            })
        
        # Add outlier (very high RMSD)
        tracker.update_progress({
            "pdb_id": "OUTLIER",
            "rmsd": 15.0,  # Much higher than others
            "gdt_ts": 30.0,
            "tm_score": 0.3,
            "final_energy": 10.0
        })
        
        outliers = tracker.identify_outliers(threshold_std=2.0)
        assert "OUTLIER" in outliers
    
    def test_outlier_tracked_during_update(self, tracker):
        """Test that outliers are tracked during update_progress."""
        # Add normal reports
        for i in range(5):
            tracker.update_progress({
                "pdb_id": f"NORMAL{i}",
                "rmsd": 2.5,
                "gdt_ts": 75.0,
                "tm_score": 0.65,
                "final_energy": -45.0
            })
        
        # Add outlier
        tracker.update_progress({
            "pdb_id": "OUTLIER",
            "rmsd": 20.0,
            "gdt_ts": 20.0,
            "tm_score": 0.2,
            "final_energy": 20.0
        })
        
        assert "OUTLIER" in tracker._outliers


class TestDashboardData:
    """Test dashboard data generation."""
    
    def test_dashboard_empty(self, tracker):
        """Test dashboard with no data."""
        dashboard = tracker.get_dashboard_data()
        
        assert isinstance(dashboard, DashboardData)
        assert dashboard.phase == 1
        assert dashboard.completed_tests == 0
        assert dashboard.pending_tests == 10
        assert dashboard.success_rate == 0.0
        assert dashboard.running_avg_rmsd == 0.0
    
    def test_dashboard_with_data(self, tracker, sample_reports):
        """Test dashboard with data."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        dashboard = tracker.get_dashboard_data()
        
        assert dashboard.phase == 1
        assert dashboard.completed_tests == 5
        assert dashboard.pending_tests == 5
        assert 0.0 < dashboard.success_rate <= 1.0
        assert dashboard.running_avg_rmsd > 0
        assert dashboard.running_avg_gdt_ts > 0
        assert len(dashboard.recent_completions) == 5


class TestInterimReports:
    """Test interim report generation."""
    
    def test_interim_report_empty(self, tracker):
        """Test interim report with no data."""
        report = tracker.generate_interim_report()
        
        assert isinstance(report, InterimReport)
        assert report.phase == 1
        assert report.completion_percentage == 0.0
        assert report.tests_completed == 0
        assert report.tests_remaining == 10
        assert report.current_success_rate == 0.0
    
    def test_interim_report_with_data(self, tracker, sample_reports):
        """Test interim report with data."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        interim = tracker.generate_interim_report()
        
        assert interim.phase == 1
        assert interim.completion_percentage == 50.0  # 5/10
        assert interim.tests_completed == 5
        assert interim.tests_remaining == 5
        assert 0.0 < interim.current_success_rate <= 1.0
        assert isinstance(interim.trends, dict)
        assert isinstance(interim.issues_detected, list)
        assert isinstance(interim.recommendations, list)
    
    def test_interim_report_milestone_tracking(self, tracker):
        """Test that interim report milestones are tracked."""
        # Add reports to reach 25% (2.5 -> 3 reports for 30%)
        for i in range(3):
            tracker.update_progress({
                "pdb_id": f"TEST{i}",
                "rmsd": 2.5,
                "gdt_ts": 75.0,
                "tm_score": 0.65,
                "final_energy": -45.0
            })
        
        # 25% milestone should be marked
        assert 0.25 in tracker._interim_reports_generated


class TestTrendAnalysis:
    """Test trend analysis functionality."""
    
    def test_trend_insufficient_data(self, tracker):
        """Test trend analysis with insufficient data."""
        # Add only 3 reports (need 5 for meaningful trends)
        for i in range(3):
            tracker.update_progress({
                "pdb_id": f"TEST{i}",
                "rmsd": 2.5,
                "gdt_ts": 75.0,
                "tm_score": 0.65,
                "final_energy": -45.0
            })
        
        trends = tracker._analyze_trends()
        
        assert trends['rmsd'] == 'insufficient_data'
        assert trends['gdt_ts'] == 'insufficient_data'
    
    def test_trend_improving(self, tracker):
        """Test detection of improving trend."""
        # Add reports with improving RMSD (decreasing)
        for i in range(10):
            tracker.update_progress({
                "pdb_id": f"TEST{i}",
                "rmsd": 5.0 - i * 0.3,  # Decreasing RMSD
                "gdt_ts": 60.0 + i * 2.0,  # Increasing GDT-TS
                "tm_score": 0.5 + i * 0.02,
                "final_energy": -30.0 - i * 1.0
            })
        
        trends = tracker._analyze_trends()
        
        assert trends['rmsd'] == 'improving'
        assert trends['gdt_ts'] == 'improving'
    
    def test_trend_declining(self, tracker):
        """Test detection of declining trend."""
        # Add reports with declining metrics
        for i in range(10):
            tracker.update_progress({
                "pdb_id": f"TEST{i}",
                "rmsd": 2.0 + i * 0.3,  # Increasing RMSD (worse)
                "gdt_ts": 80.0 - i * 2.0,  # Decreasing GDT-TS (worse)
                "tm_score": 0.7 - i * 0.02,
                "final_energy": -45.0 + i * 2.0
            })
        
        trends = tracker._analyze_trends()
        
        assert trends['rmsd'] == 'declining'
        assert trends['gdt_ts'] == 'declining'


class TestIssueDetection:
    """Test issue detection functionality."""
    
    def test_detect_low_success_rate(self, tracker):
        """Test detection of low success rate."""
        # Add reports with high RMSD (failures)
        for i in range(10):
            tracker.update_progress({
                "pdb_id": f"TEST{i}",
                "rmsd": 8.0,  # Above success threshold
                "gdt_ts": 40.0,
                "tm_score": 0.4,
                "final_energy": -20.0
            })
        
        trends = tracker._analyze_trends()
        issues = tracker._detect_issues(trends, success_rate=0.0)
        
        assert any("Low success rate" in issue for issue in issues)
    
    def test_detect_declining_trends(self, tracker):
        """Test detection of declining trends."""
        trends = {'rmsd': 'declining', 'gdt_ts': 'declining'}
        issues = tracker._detect_issues(trends, success_rate=0.8)
        
        assert any("RMSD trend declining" in issue for issue in issues)
        assert any("GDT-TS trend declining" in issue for issue in issues)
    
    def test_detect_high_outlier_rate(self, tracker):
        """Test detection of high outlier rate."""
        # Add normal reports
        for i in range(5):
            tracker.update_progress({
                "pdb_id": f"NORMAL{i}",
                "rmsd": 2.5,
                "gdt_ts": 75.0,
                "tm_score": 0.65,
                "final_energy": -45.0
            })
        
        # Manually add outliers
        tracker._outliers = ["OUT1", "OUT2", "OUT3"]
        
        trends = {}
        issues = tracker._detect_issues(trends, success_rate=0.8)
        
        assert any("High outlier rate" in issue for issue in issues)


class TestRecommendations:
    """Test recommendation generation."""
    
    def test_recommendations_no_issues(self, tracker):
        """Test recommendations when no issues detected."""
        trends = {'rmsd': 'stable', 'gdt_ts': 'stable'}
        issues = []
        
        recommendations = tracker._generate_recommendations(trends, issues)
        
        assert len(recommendations) > 0
        assert any("No issues detected" in rec for rec in recommendations)
    
    def test_recommendations_low_success_rate(self, tracker):
        """Test recommendations for low success rate."""
        trends = {}
        issues = ["Low success rate: 40.0% (threshold: 60%)"]
        
        recommendations = tracker._generate_recommendations(trends, issues)
        
        assert any("iterations_per_agent" in rec for rec in recommendations)
        assert any("QCPP" in rec for rec in recommendations)
    
    def test_recommendations_declining_trends(self, tracker):
        """Test recommendations for declining trends."""
        trends = {'rmsd': 'declining', 'gdt_ts': 'declining'}
        issues = ["RMSD trend declining", "GDT-TS trend declining"]
        
        recommendations = tracker._generate_recommendations(trends, issues)
        
        assert any("stuck_threshold" in rec for rec in recommendations)
        assert any("adaptive configuration" in rec for rec in recommendations)


class TestExport:
    """Test export functionality."""
    
    def test_export_dashboard(self, tracker, sample_reports, temp_dir):
        """Test exporting dashboard data."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        output_path = Path(temp_dir) / "dashboard.json"
        tracker.export_dashboard(str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data['phase'] == 1
        assert data['completed_tests'] == 5
        assert 'running_avg_rmsd' in data
    
    def test_export_interim_report(self, tracker, sample_reports, temp_dir):
        """Test exporting interim report."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        output_path = Path(temp_dir) / "interim_report.json"
        tracker.export_interim_report(str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data['phase'] == 1
        assert data['tests_completed'] == 5
        assert 'trends' in data
        assert 'recommendations' in data


class TestSummaryStatistics:
    """Test summary statistics functionality."""
    
    def test_summary_empty(self, tracker):
        """Test summary with no data."""
        summary = tracker.get_summary_statistics()
        
        assert summary['completed_tests'] == 0
        assert summary['metrics_available'] is False
    
    def test_summary_with_data(self, tracker, sample_reports):
        """Test summary with data."""
        for report in sample_reports:
            tracker.update_progress(report)
        
        summary = tracker.get_summary_statistics()
        
        assert summary['completed_tests'] == 5
        assert summary['metrics_available'] is True
        assert summary['rmsd']['mean'] is not None
        assert summary['rmsd']['median'] is not None
        assert summary['rmsd']['stdev'] is not None
        assert summary['rmsd']['min'] is not None
        assert summary['rmsd']['max'] is not None
        assert 0.0 <= summary['success_rate'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
