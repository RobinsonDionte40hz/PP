"""
Example demonstrating ProgressTracker usage.

Shows:
1. Basic progress tracking
2. Real-time dashboard data
3. Interim reports at milestones
4. Trend analysis
5. Outlier detection
6. Export functionality
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.progress_tracker import ProgressTracker


def example1_basic_tracking():
    """Example 1: Basic progress tracking."""
    print("=" * 60)
    print("Example 1: Basic Progress Tracking")
    print("=" * 60)
    
    tracker = ProgressTracker(total_tests=10, phase=1)
    
    # Simulate test completions
    test_results = [
        {"pdb_id": "1UBQ", "rmsd": 2.5, "gdt_ts": 75.0, "tm_score": 0.65, "final_energy": -45.0},
        {"pdb_id": "1CRN", "rmsd": 1.8, "gdt_ts": 85.0, "tm_score": 0.82, "final_energy": -52.0},
        {"pdb_id": "2MR9", "rmsd": 3.2, "gdt_ts": 68.0, "tm_score": 0.58, "final_energy": -38.0},
    ]
    
    for result in test_results:
        tracker.update_progress(result)
        print(f"✅ Completed: {result['pdb_id']} (RMSD: {result['rmsd']:.2f}Å)")
    
    print(f"\n📊 Progress: {len(tracker._completed_tests)}/{tracker.total_tests} tests")
    print()


def example2_dashboard_data():
    """Example 2: Real-time dashboard data."""
    print("=" * 60)
    print("Example 2: Real-Time Dashboard Data")
    print("=" * 60)
    
    tracker = ProgressTracker(total_tests=20, phase=2, recent_window=5)
    
    # Simulate multiple test completions
    for i in range(10):
        tracker.update_progress({
            "pdb_id": f"TEST{i}",
            "rmsd": 2.5 + i * 0.2,
            "gdt_ts": 75.0 - i * 1.5,
            "tm_score": 0.65 - i * 0.02,
            "final_energy": -45.0 + i * 1.0
        })
    
    # Get dashboard data
    dashboard = tracker.get_dashboard_data()
    
    print(f"Phase: {dashboard.phase}")
    print(f"Progress: {dashboard.completed_tests}/{dashboard.completed_tests + dashboard.pending_tests}")
    print(f"Success Rate: {dashboard.success_rate:.1%}")
    print(f"\nRunning Averages:")
    print(f"  RMSD: {dashboard.running_avg_rmsd:.2f}Å")
    print(f"  GDT-TS: {dashboard.running_avg_gdt_ts:.1f}")
    print(f"  TM-score: {dashboard.running_avg_tm_score:.3f}")
    print(f"  Energy: {dashboard.running_avg_energy:.1f} kcal/mol")
    print(f"\nRecent Completions: {', '.join(dashboard.recent_completions)}")
    print()


def example3_interim_reports():
    """Example 3: Interim reports at milestones."""
    print("=" * 60)
    print("Example 3: Interim Reports at Milestones")
    print("=" * 60)
    
    tracker = ProgressTracker(
        total_tests=20,
        phase=1,
        interim_report_intervals=[0.25, 0.5, 0.75]
    )
    
    # Simulate test completions to reach milestones
    print("📝 Simulating test completions...\n")
    
    for i in range(20):
        tracker.update_progress({
            "pdb_id": f"PROTEIN_{i}",
            "rmsd": 2.5 + (i % 3) * 0.5,
            "gdt_ts": 75.0 - (i % 4) * 2.0,
            "tm_score": 0.65 + (i % 2) * 0.05,
            "final_energy": -45.0 + (i % 3) * 2.0
        })
        
        # Check if we hit a milestone
        progress = (i + 1) / tracker.total_tests
        if progress in [0.25, 0.5, 0.75] and progress in tracker._interim_reports_generated:
            print(f"\n🎯 Milestone: {progress:.0%} Complete")
            report = tracker.generate_interim_report()
            print(f"Tests Completed: {report.tests_completed}")
            print(f"Tests Remaining: {report.tests_remaining}")
            print(f"Success Rate: {report.current_success_rate:.1%}")
            print(f"Trends: {report.trends}")
            print(f"Issues: {len(report.issues_detected)} detected")
            print(f"Recommendations: {len(report.recommendations)} provided")
    
    print()


def example4_trend_analysis():
    """Example 4: Trend analysis."""
    print("=" * 60)
    print("Example 4: Trend Analysis")
    print("=" * 60)
    
    tracker = ProgressTracker(total_tests=15, phase=1, recent_window=10)
    
    # Simulate improving trend
    print("📈 Simulating improving trend (RMSD decreasing)...\n")
    for i in range(10):
        tracker.update_progress({
            "pdb_id": f"IMPROVING_{i}",
            "rmsd": 5.0 - i * 0.3,  # Decreasing
            "gdt_ts": 60.0 + i * 2.0,  # Increasing
            "tm_score": 0.5 + i * 0.02,
            "final_energy": -30.0 - i * 1.5
        })
    
    trends = tracker._analyze_trends()
    print("Trend Analysis:")
    for metric, trend in trends.items():
        emoji = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
        print(f"  {emoji} {metric}: {trend}")
    
    # Get recommendations
    report = tracker.generate_interim_report()
    print("\n💡 Recommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print()


def example5_outlier_detection():
    """Example 5: Outlier detection."""
    print("=" * 60)
    print("Example 5: Outlier Detection")
    print("=" * 60)
    
    tracker = ProgressTracker(total_tests=15, phase=1, outlier_threshold=2.0)
    
    # Add normal results
    print("📊 Adding normal test results...\n")
    for i in range(8):
        tracker.update_progress({
            "pdb_id": f"NORMAL_{i}",
            "rmsd": 2.5 + i * 0.1,
            "gdt_ts": 75.0 + i * 0.5,
            "tm_score": 0.65 + i * 0.01,
            "final_energy": -45.0 - i * 0.5
        })
    
    # Add outliers
    print("⚠️  Adding outlier results...\n")
    outlier_results = [
        {"pdb_id": "OUTLIER_1", "rmsd": 15.0, "gdt_ts": 30.0, "tm_score": 0.3, "final_energy": 10.0},
        {"pdb_id": "OUTLIER_2", "rmsd": 0.5, "gdt_ts": 95.0, "tm_score": 0.95, "final_energy": -80.0},
    ]
    
    for result in outlier_results:
        tracker.update_progress(result)
    
    # Identify outliers
    outliers = tracker.identify_outliers(threshold_std=2.0)
    
    print("Outlier Analysis:")
    print(f"  Total Tests: {len(tracker._completed_tests)}")
    print(f"  Outliers Detected: {len(outliers)}")
    print(f"  Outlier Rate: {len(outliers)/len(tracker._completed_tests):.1%}")
    print(f"\n  Outlier IDs: {', '.join(outliers)}")
    
    # Get recommendations for high outlier rate
    report = tracker.generate_interim_report()
    if report.issues_detected:
        print(f"\n⚠️  Issues Detected:")
        for issue in report.issues_detected:
            print(f"    • {issue}")
    
    print()


def example6_export_functionality():
    """Example 6: Export functionality."""
    print("=" * 60)
    print("Example 6: Export Functionality")
    print("=" * 60)
    
    tracker = ProgressTracker(total_tests=10, phase=3)
    
    # Add test results
    for i in range(7):
        tracker.update_progress({
            "pdb_id": f"EXPORT_TEST_{i}",
            "rmsd": 2.5 + i * 0.3,
            "gdt_ts": 75.0 - i * 1.0,
            "tm_score": 0.65 - i * 0.02,
            "final_energy": -45.0 + i * 0.8
        })
    
    # Create output directory
    output_dir = Path("validation_tracking_output")
    output_dir.mkdir(exist_ok=True)
    
    # Export dashboard
    dashboard_path = output_dir / "dashboard.json"
    tracker.export_dashboard(str(dashboard_path))
    print(f"✅ Dashboard exported: {dashboard_path}")
    
    # Display dashboard contents
    with open(dashboard_path, 'r') as f:
        dashboard = json.load(f)
    print(f"\nDashboard Data:")
    print(f"  Phase: {dashboard['phase']}")
    print(f"  Progress: {dashboard['completed_tests']}/{dashboard['completed_tests'] + dashboard['pending_tests']}")
    print(f"  Success Rate: {dashboard['success_rate']:.1%}")
    
    # Export interim report
    report_path = output_dir / "interim_report.json"
    tracker.export_interim_report(str(report_path))
    print(f"\n✅ Interim report exported: {report_path}")
    
    # Display report summary
    with open(report_path, 'r') as f:
        report = json.load(f)
    print(f"\nInterim Report:")
    print(f"  Completion: {report['completion_percentage']:.1f}%")
    print(f"  Success Rate: {report['current_success_rate']:.1%}")
    print(f"  Trends: {len(report['trends'])} metrics analyzed")
    print(f"  Recommendations: {len(report['recommendations'])} provided")
    
    # Get summary statistics
    summary = tracker.get_summary_statistics()
    print(f"\n📊 Summary Statistics:")
    print(f"  RMSD: {summary['rmsd']['mean']:.2f} ± {summary['rmsd']['stdev']:.2f}Å")
    print(f"  GDT-TS: {summary['gdt_ts']['mean']:.1f} ± {summary['gdt_ts']['stdev']:.1f}")
    print(f"  TM-score: {summary['tm_score']['mean']:.3f} ± {summary['tm_score']['stdev']:.3f}")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "🔬 ProgressTracker Examples ".center(60, "="))
    print()
    
    example1_basic_tracking()
    example2_dashboard_data()
    example3_interim_reports()
    example4_trend_analysis()
    example5_outlier_detection()
    example6_export_functionality()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
