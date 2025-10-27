"""
Progress Tracker for Large-Scale Validation

Real-time monitoring and visualization of testing progress with:
- Running average calculations for all metrics
- Outlier detection using statistical thresholds
- Interim report generation at phase milestones
- Dashboard data for web/CLI display
- Trend analysis (improving/stable/declining)
- Recent completions tracking

Key Features:
- Track progress across 50-75 protein validation tests
- Calculate running averages for RMSD, GDT-TS, TM-score, energy
- Identify outliers (>2 std deviations from mean)
- Generate interim reports at 25%, 50%, 75% completion
- Detect trends in validation metrics
- Export dashboard data for visualization
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    """
    Real-time dashboard data for progress visualization.
    
    Attributes:
        phase: Current phase number (1-4)
        completed_tests: Number of completed tests
        pending_tests: Number of pending tests
        success_rate: Success rate (0-1)
        running_avg_rmsd: Running average RMSD
        running_avg_gdt_ts: Running average GDT-TS
        running_avg_tm_score: Running average TM-score
        running_avg_energy: Running average energy
        outliers: List of outlier protein IDs
        recent_completions: List of recently completed protein IDs
        timestamp: When dashboard was generated
    """
    phase: int
    completed_tests: int
    pending_tests: int
    success_rate: float
    running_avg_rmsd: float
    running_avg_gdt_ts: float
    running_avg_tm_score: float
    running_avg_energy: float
    outliers: List[str]
    recent_completions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class InterimReport:
    """
    Interim report generated at phase milestones.
    
    Attributes:
        phase: Phase number
        completion_percentage: Percentage complete (0-100)
        tests_completed: Number of tests completed
        tests_remaining: Number of tests remaining
        current_success_rate: Current success rate (0-1)
        trends: Trend analysis for each metric (improving/stable/declining)
        issues_detected: List of detected issues
        recommendations: List of recommendations based on trends
        timestamp: When report was generated
    """
    phase: int
    completion_percentage: float
    tests_completed: int
    tests_remaining: int
    current_success_rate: float
    trends: Dict[str, str]
    issues_detected: List[str]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RunningAverages:
    """
    Running averages for all validation metrics.
    
    Attributes:
        rmsd: Average RMSD across all tests
        gdt_ts: Average GDT-TS across all tests
        tm_score: Average TM-score across all tests
        energy: Average final energy across all tests
        count: Number of tests included
    """
    rmsd: float
    gdt_ts: float
    tm_score: float
    energy: float
    count: int


class ProgressTracker:
    """
    Real-time progress tracking and monitoring for validation campaigns.
    
    Tracks progress across all validation tests and provides:
    - Running averages for metrics
    - Outlier detection
    - Trend analysis
    - Interim reports
    - Dashboard data
    
    Usage:
        tracker = ProgressTracker(
            total_tests=60,
            phase=1,
            interim_report_intervals=[0.25, 0.5, 0.75]
        )
        
        # Update with each completed test
        tracker.update_progress(validation_report)
        
        # Get dashboard data
        dashboard = tracker.get_dashboard_data()
        
        # Generate interim report
        report = tracker.generate_interim_report()
    """
    
    def __init__(self,
                 total_tests: int,
                 phase: int = 1,
                 success_threshold: float = 5.0,
                 outlier_threshold: float = 2.0,
                 recent_window: int = 10,
                 interim_report_intervals: Optional[List[float]] = None):
        """
        Initialize ProgressTracker.
        
        Args:
            total_tests: Total number of tests in current phase
            phase: Current phase number (1-4)
            success_threshold: RMSD threshold for success (Angstroms)
            outlier_threshold: Number of standard deviations for outlier detection
            recent_window: Number of recent completions to track
            interim_report_intervals: List of completion percentages for interim reports (e.g., [0.25, 0.5, 0.75])
        """
        self.total_tests = total_tests
        self.phase = phase
        self.success_threshold = success_threshold
        self.outlier_threshold = outlier_threshold
        self.recent_window = recent_window
        self.interim_report_intervals = interim_report_intervals or [0.25, 0.5, 0.75]
        
        # Progress tracking
        self._completed_tests: List[Dict[str, Any]] = []
        self._recent_completions: deque = deque(maxlen=recent_window)
        self._outliers: List[str] = []
        self._interim_reports_generated: set = set()
        
        # Metric tracking
        self._rmsds: List[float] = []
        self._gdt_tss: List[float] = []
        self._tm_scores: List[float] = []
        self._energies: List[float] = []
        
        # Historical data for trend analysis
        self._rmsd_history: deque = deque(maxlen=20)
        self._gdt_ts_history: deque = deque(maxlen=20)
        
        logger.info(f"ProgressTracker initialized: phase={phase}, "
                   f"total_tests={total_tests}, success_threshold={success_threshold}Å")
    
    def update_progress(self, report: Dict[str, Any]) -> None:
        """
        Update progress with a completed test report.
        
        Args:
            report: Validation report dictionary with keys:
                - pdb_id: str
                - rmsd: float
                - gdt_ts: float
                - tm_score: float
                - final_energy: float
        """
        pdb_id = report.get('pdb_id', 'unknown')
        
        # Add to completed tests
        self._completed_tests.append(report)
        self._recent_completions.append(pdb_id)
        
        # Extract and store metrics
        rmsd = report.get('rmsd')
        if rmsd is not None:
            self._rmsds.append(float(rmsd))
            self._rmsd_history.append(float(rmsd))
        
        gdt_ts = report.get('gdt_ts')
        if gdt_ts is not None:
            self._gdt_tss.append(float(gdt_ts))
            self._gdt_ts_history.append(float(gdt_ts))
        
        tm_score = report.get('tm_score')
        if tm_score is not None:
            self._tm_scores.append(float(tm_score))
        
        energy = report.get('final_energy')
        if energy is not None:
            self._energies.append(float(energy))
        
        # Check if outlier
        if self._is_outlier(report):
            self._outliers.append(pdb_id)
            logger.warning(f"Outlier detected: {pdb_id} (RMSD={rmsd:.2f}Å)")
        
        # Check if interim report should be generated
        completion_pct = len(self._completed_tests) / self.total_tests
        for interval in self.interim_report_intervals:
            if completion_pct >= interval and interval not in self._interim_reports_generated:
                self._interim_reports_generated.add(interval)
                logger.info(f"Interim report milestone reached: {interval*100:.0f}%")
        
        logger.debug(f"Progress updated: {pdb_id} completed "
                    f"({len(self._completed_tests)}/{self.total_tests})")
    
    def _is_outlier(self, report: Dict[str, Any]) -> bool:
        """
        Check if report metrics are outliers.
        
        Args:
            report: Validation report
        
        Returns:
            True if report is an outlier
        """
        # Need at least 3 data points for meaningful outlier detection
        if len(self._rmsds) < 3:
            return False
        
        rmsd = report.get('rmsd')
        if rmsd is None:
            return False
        
        try:
            mean_rmsd = statistics.mean(self._rmsds)
            std_rmsd = statistics.stdev(self._rmsds)
            
            # Check if beyond threshold
            z_score = abs((rmsd - mean_rmsd) / std_rmsd) if std_rmsd > 0 else 0
            return z_score > self.outlier_threshold
        
        except statistics.StatisticsError:
            return False
    
    def get_dashboard_data(self) -> DashboardData:
        """
        Get current dashboard data for visualization.
        
        Returns:
            DashboardData with current progress and metrics
        """
        completed = len(self._completed_tests)
        pending = self.total_tests - completed
        
        # Calculate success rate (RMSD < threshold)
        if self._rmsds:
            successes = sum(1 for rmsd in self._rmsds if rmsd < self.success_threshold)
            success_rate = successes / len(self._rmsds)
        else:
            success_rate = 0.0
        
        # Calculate running averages
        averages = self.calculate_running_averages()
        
        # Get recent completions
        recent = list(self._recent_completions)
        
        return DashboardData(
            phase=self.phase,
            completed_tests=completed,
            pending_tests=pending,
            success_rate=success_rate,
            running_avg_rmsd=averages.rmsd,
            running_avg_gdt_ts=averages.gdt_ts,
            running_avg_tm_score=averages.tm_score,
            running_avg_energy=averages.energy,
            outliers=list(self._outliers),
            recent_completions=recent
        )
    
    def calculate_running_averages(self) -> RunningAverages:
        """
        Calculate running averages for all metrics.
        
        Returns:
            RunningAverages with current averages
        """
        count = len(self._completed_tests)
        
        return RunningAverages(
            rmsd=statistics.mean(self._rmsds) if self._rmsds else 0.0,
            gdt_ts=statistics.mean(self._gdt_tss) if self._gdt_tss else 0.0,
            tm_score=statistics.mean(self._tm_scores) if self._tm_scores else 0.0,
            energy=statistics.mean(self._energies) if self._energies else 0.0,
            count=count
        )
    
    def identify_outliers(self, threshold_std: float = 2.0) -> List[str]:
        """
        Identify all outlier proteins based on statistical threshold.
        
        Args:
            threshold_std: Number of standard deviations for outlier classification
        
        Returns:
            List of outlier protein IDs
        """
        if len(self._rmsds) < 3:
            return []
        
        try:
            mean_rmsd = statistics.mean(self._rmsds)
            std_rmsd = statistics.stdev(self._rmsds)
            
            outliers = []
            for i, rmsd in enumerate(self._rmsds):
                z_score = abs((rmsd - mean_rmsd) / std_rmsd) if std_rmsd > 0 else 0
                if z_score > threshold_std:
                    pdb_id = self._completed_tests[i].get('pdb_id', f'test_{i}')
                    outliers.append(pdb_id)
            
            return outliers
        
        except statistics.StatisticsError:
            return []
    
    def generate_interim_report(self) -> InterimReport:
        """
        Generate interim report with current status and trends.
        
        Returns:
            InterimReport with analysis and recommendations
        """
        completed = len(self._completed_tests)
        remaining = self.total_tests - completed
        completion_pct = (completed / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        # Calculate success rate
        if self._rmsds:
            successes = sum(1 for rmsd in self._rmsds if rmsd < self.success_threshold)
            success_rate = successes / len(self._rmsds)
        else:
            success_rate = 0.0
        
        # Analyze trends
        trends = self._analyze_trends()
        
        # Detect issues
        issues = self._detect_issues(trends, success_rate)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trends, issues)
        
        return InterimReport(
            phase=self.phase,
            completion_percentage=completion_pct,
            tests_completed=completed,
            tests_remaining=remaining,
            current_success_rate=success_rate,
            trends=trends,
            issues_detected=issues,
            recommendations=recommendations
        )
    
    def _analyze_trends(self) -> Dict[str, str]:
        """
        Analyze trends in metrics over recent history.
        
        Returns:
            Dictionary mapping metric names to trend strings
        """
        trends = {}
        
        # Analyze RMSD trend
        if len(self._rmsd_history) >= 5:
            recent_rmsd = list(self._rmsd_history)
            first_half = statistics.mean(recent_rmsd[:len(recent_rmsd)//2])
            second_half = statistics.mean(recent_rmsd[len(recent_rmsd)//2:])
            
            if second_half < first_half * 0.9:
                trends['rmsd'] = 'improving'
            elif second_half > first_half * 1.1:
                trends['rmsd'] = 'declining'
            else:
                trends['rmsd'] = 'stable'
        else:
            trends['rmsd'] = 'insufficient_data'
        
        # Analyze GDT-TS trend
        if len(self._gdt_ts_history) >= 5:
            recent_gdt = list(self._gdt_ts_history)
            first_half = statistics.mean(recent_gdt[:len(recent_gdt)//2])
            second_half = statistics.mean(recent_gdt[len(recent_gdt)//2:])
            
            if second_half > first_half * 1.05:
                trends['gdt_ts'] = 'improving'
            elif second_half < first_half * 0.95:
                trends['gdt_ts'] = 'declining'
            else:
                trends['gdt_ts'] = 'stable'
        else:
            trends['gdt_ts'] = 'insufficient_data'
        
        return trends
    
    def _detect_issues(self, trends: Dict[str, str], success_rate: float) -> List[str]:
        """
        Detect issues based on trends and success rate.
        
        Args:
            trends: Trend analysis results
            success_rate: Current success rate
        
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check success rate
        if success_rate < 0.6:
            issues.append(f"Low success rate: {success_rate*100:.1f}% (threshold: 60%)")
        
        # Check declining trends
        if trends.get('rmsd') == 'declining':
            issues.append("RMSD trend declining (getting worse)")
        
        if trends.get('gdt_ts') == 'declining':
            issues.append("GDT-TS trend declining (getting worse)")
        
        # Check outliers
        outlier_rate = len(self._outliers) / len(self._completed_tests) if self._completed_tests else 0
        if outlier_rate > 0.2:
            issues.append(f"High outlier rate: {outlier_rate*100:.1f}% of tests")
        
        return issues
    
    def _generate_recommendations(self, trends: Dict[str, str], issues: List[str]) -> List[str]:
        """
        Generate recommendations based on trends and issues.
        
        Args:
            trends: Trend analysis results
            issues: Detected issues
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not issues:
            recommendations.append("No issues detected. Continue with current parameters.")
            return recommendations
        
        # Recommendations based on success rate
        if "Low success rate" in str(issues):
            recommendations.append("Consider increasing iterations_per_agent")
            recommendations.append("Consider enabling QCPP integration if disabled")
        
        # Recommendations based on trends
        if trends.get('rmsd') == 'declining' or trends.get('gdt_ts') == 'declining':
            recommendations.append("Consider adjusting stuck_threshold and stuck_window")
            recommendations.append("Review adaptive configuration for protein size categories")
        
        # Recommendations based on outliers
        if "High outlier rate" in str(issues):
            recommendations.append("Investigate outlier proteins for common characteristics")
            recommendations.append("Consider size-specific parameter tuning")
        
        return recommendations
    
    def export_dashboard(self, output_path: str) -> None:
        """
        Export dashboard data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        dashboard = self.get_dashboard_data()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(asdict(dashboard), f, indent=2)
            
            logger.info(f"Dashboard exported to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise
    
    def export_interim_report(self, output_path: str) -> None:
        """
        Export interim report to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        report = self.generate_interim_report()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(asdict(report), f, indent=2)
            
            logger.info(f"Interim report exported to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to export interim report: {e}")
            raise
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._completed_tests:
            return {
                "completed_tests": 0,
                "metrics_available": False
            }
        
        summary = {
            "completed_tests": len(self._completed_tests),
            "metrics_available": True,
            "rmsd": {
                "mean": statistics.mean(self._rmsds) if self._rmsds else None,
                "median": statistics.median(self._rmsds) if self._rmsds else None,
                "stdev": statistics.stdev(self._rmsds) if len(self._rmsds) > 1 else None,
                "min": min(self._rmsds) if self._rmsds else None,
                "max": max(self._rmsds) if self._rmsds else None
            },
            "gdt_ts": {
                "mean": statistics.mean(self._gdt_tss) if self._gdt_tss else None,
                "median": statistics.median(self._gdt_tss) if self._gdt_tss else None,
                "stdev": statistics.stdev(self._gdt_tss) if len(self._gdt_tss) > 1 else None,
                "min": min(self._gdt_tss) if self._gdt_tss else None,
                "max": max(self._gdt_tss) if self._gdt_tss else None
            },
            "tm_score": {
                "mean": statistics.mean(self._tm_scores) if self._tm_scores else None,
                "median": statistics.median(self._tm_scores) if self._tm_scores else None,
                "stdev": statistics.stdev(self._tm_scores) if len(self._tm_scores) > 1 else None,
                "min": min(self._tm_scores) if self._tm_scores else None,
                "max": max(self._tm_scores) if self._tm_scores else None
            },
            "energy": {
                "mean": statistics.mean(self._energies) if self._energies else None,
                "median": statistics.median(self._energies) if self._energies else None,
                "stdev": statistics.stdev(self._energies) if len(self._energies) > 1 else None,
                "min": min(self._energies) if self._energies else None,
                "max": max(self._energies) if self._energies else None
            },
            "outliers": len(self._outliers),
            "success_rate": sum(1 for rmsd in self._rmsds if rmsd < self.success_threshold) / len(self._rmsds) if self._rmsds else 0.0
        }
        
        return summary
