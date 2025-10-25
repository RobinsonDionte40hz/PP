"""
Unit tests for integrated trajectory recording.

Tests Task 8.1:
- Test IntegratedTrajectoryPoint dataclass creation
- Test trajectory point recording with both UBF and QCPP metrics
- Test JSON export format
- Test correlation analysis calculations
"""

import pytest
import json
import tempfile
import os
from ubf_protein.integrated_trajectory import (
    IntegratedTrajectoryPoint,
    IntegratedTrajectoryRecorder,
    TrajectoryAnalyzer
)


class TestIntegratedTrajectoryPoint:
    """Test suite for IntegratedTrajectoryPoint dataclass."""
    
    def test_trajectory_point_creation(self):
        """Test creating a valid trajectory point."""
        point = IntegratedTrajectoryPoint(
            iteration=100,
            rmsd=5.2,
            energy=-150.0,
            consciousness_frequency=8.5,
            consciousness_coherence=0.6,
            qcp_score=5.0,
            field_coherence=0.3,
            stability_score=1.5,
            phi_match_score=0.7,
            timestamp=1234567890
        )
        
        # Verify all fields
        assert point.iteration == 100
        assert point.rmsd == 5.2
        assert point.energy == -150.0
        assert point.consciousness_frequency == 8.5
        assert point.consciousness_coherence == 0.6
        assert point.qcp_score == 5.0
        assert point.field_coherence == 0.3
        assert point.stability_score == 1.5
        assert point.phi_match_score == 0.7
        assert point.timestamp == 1234567890
    
    def test_trajectory_point_immutability(self):
        """Test that trajectory points are immutable (frozen dataclass)."""
        point = IntegratedTrajectoryPoint(
            iteration=1,
            rmsd=5.0,
            energy=-100.0,
            consciousness_frequency=10.0,
            consciousness_coherence=0.5,
            qcp_score=4.5,
            field_coherence=0.0,
            stability_score=1.0,
            phi_match_score=0.5,
            timestamp=1000000
        )
        
        # Attempt to modify should raise error
        with pytest.raises(Exception):  # FrozenInstanceError
            point.rmsd = 10.0  # type: ignore
    
    def test_trajectory_point_validation_negative_iteration(self):
        """Test validation rejects negative iteration."""
        with pytest.raises(ValueError, match="iteration.*must be non-negative"):
            IntegratedTrajectoryPoint(
                iteration=-1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=0.5,
                timestamp=1000000
            )
    
    def test_trajectory_point_validation_negative_rmsd(self):
        """Test validation rejects negative RMSD."""
        with pytest.raises(ValueError, match="rmsd.*must be non-negative"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=-1.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=0.5,
                timestamp=1000000
            )
    
    def test_trajectory_point_validation_frequency_bounds(self):
        """Test validation enforces frequency bounds [3, 15]."""
        # Too low
        with pytest.raises(ValueError, match="consciousness_frequency.*must be in range"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=2.0,  # Below 3.0
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=0.5,
                timestamp=1000000
            )
        
        # Too high
        with pytest.raises(ValueError, match="consciousness_frequency.*must be in range"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=16.0,  # Above 15.0
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=0.5,
                timestamp=1000000
            )
    
    def test_trajectory_point_validation_coherence_bounds(self):
        """Test validation enforces coherence bounds [0.2, 1.0]."""
        # Too low
        with pytest.raises(ValueError, match="consciousness_coherence.*must be in range"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.1,  # Below 0.2
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=0.5,
                timestamp=1000000
            )
        
        # Too high
        with pytest.raises(ValueError, match="consciousness_coherence.*must be in range"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=1.1,  # Above 1.0
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=0.5,
                timestamp=1000000
            )
    
    def test_trajectory_point_validation_phi_match_bounds(self):
        """Test validation enforces phi match bounds [0, 1]."""
        # Too low
        with pytest.raises(ValueError, match="phi_match_score.*must be in range"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=-0.1,  # Below 0
                timestamp=1000000
            )
        
        # Too high
        with pytest.raises(ValueError, match="phi_match_score.*must be in range"):
            IntegratedTrajectoryPoint(
                iteration=1,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.0,
                stability_score=1.0,
                phi_match_score=1.1,  # Above 1
                timestamp=1000000
            )


class TestIntegratedTrajectoryRecorder:
    """Test suite for IntegratedTrajectoryRecorder."""
    
    def test_recorder_initialization(self):
        """Test recorder initializes correctly."""
        recorder = IntegratedTrajectoryRecorder()
        
        assert recorder.get_point_count() == 0
        assert len(recorder.get_points()) == 0
    
    def test_recorder_initialization_with_max_points(self):
        """Test recorder with max points limit."""
        recorder = IntegratedTrajectoryRecorder(max_points=100)
        
        assert recorder._max_points == 100
        assert recorder.get_point_count() == 0
    
    def test_record_single_point(self):
        """Test recording a single trajectory point."""
        recorder = IntegratedTrajectoryRecorder()
        
        point = recorder.record_point(
            iteration=1,
            rmsd=5.0,
            energy=-100.0,
            consciousness_frequency=10.0,
            consciousness_coherence=0.5,
            qcp_score=4.5,
            field_coherence=0.2,
            stability_score=1.2,
            phi_match_score=0.6
        )
        
        assert recorder.get_point_count() == 1
        assert point.iteration == 1
        assert point.rmsd == 5.0
        assert point.qcp_score == 4.5
    
    def test_record_multiple_points(self):
        """Test recording multiple trajectory points."""
        recorder = IntegratedTrajectoryRecorder()
        
        for i in range(10):
            recorder.record_point(
                iteration=i,
                rmsd=5.0 - i * 0.1,
                energy=-100.0 - i * 5.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.2,
                stability_score=1.2,
                phi_match_score=0.6
            )
        
        assert recorder.get_point_count() == 10
        
        points = recorder.get_points()
        assert len(points) == 10
        assert points[0].iteration == 0
        assert points[9].iteration == 9
    
    def test_max_points_enforcement(self):
        """Test that max_points limit is enforced (FIFO)."""
        recorder = IntegratedTrajectoryRecorder(max_points=5)
        
        # Record 10 points
        for i in range(10):
            recorder.record_point(
                iteration=i,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.2,
                stability_score=1.2,
                phi_match_score=0.6
            )
        
        # Should only have last 5 points
        assert recorder.get_point_count() == 5
        
        points = recorder.get_points()
        assert points[0].iteration == 5  # Oldest point
        assert points[4].iteration == 9  # Newest point
    
    def test_export_to_json(self):
        """Test exporting trajectory to JSON file."""
        recorder = IntegratedTrajectoryRecorder()
        
        # Record some points
        for i in range(5):
            recorder.record_point(
                iteration=i,
                rmsd=5.0 - i * 0.5,
                energy=-100.0 - i * 10.0,
                consciousness_frequency=10.0 + i * 0.5,
                consciousness_coherence=0.5 + i * 0.05,
                qcp_score=4.5 + i * 0.1,
                field_coherence=0.2 + i * 0.05,
                stability_score=1.2 + i * 0.1,
                phi_match_score=0.6 + i * 0.05
            )
        
        # Export to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            recorder.export_to_json(temp_file)
            
            # Verify file exists and has content
            assert os.path.exists(temp_file)
            
            # Load and verify structure
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'trajectory_points' in data
            assert data['metadata']['point_count'] == 5
            assert len(data['trajectory_points']) == 5
            
            # Verify first point
            first_point = data['trajectory_points'][0]
            assert first_point['iteration'] == 0
            assert first_point['rmsd'] == 5.0
            assert first_point['qcp_score'] == 4.5
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_to_dict(self):
        """Test exporting trajectory to dictionary."""
        recorder = IntegratedTrajectoryRecorder()
        
        # Record some points
        for i in range(3):
            recorder.record_point(
                iteration=i,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.2,
                stability_score=1.2,
                phi_match_score=0.6
            )
        
        data = recorder.export_to_dict()
        
        assert 'metadata' in data
        assert 'trajectory_points' in data
        assert data['metadata']['point_count'] == 3
        assert len(data['trajectory_points']) == 3
    
    def test_clear_trajectory(self):
        """Test clearing all trajectory points."""
        recorder = IntegratedTrajectoryRecorder()
        
        # Record some points
        for i in range(5):
            recorder.record_point(
                iteration=i,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.2,
                stability_score=1.2,
                phi_match_score=0.6
            )
        
        assert recorder.get_point_count() == 5
        
        recorder.clear()
        
        assert recorder.get_point_count() == 0
        assert len(recorder.get_points()) == 0
    
    def test_recording_stats(self):
        """Test getting recording performance statistics."""
        recorder = IntegratedTrajectoryRecorder()
        
        # Record some points
        for i in range(10):
            recorder.record_point(
                iteration=i,
                rmsd=5.0,
                energy=-100.0,
                consciousness_frequency=10.0,
                consciousness_coherence=0.5,
                qcp_score=4.5,
                field_coherence=0.2,
                stability_score=1.2,
                phi_match_score=0.6
            )
        
        stats = recorder.get_recording_stats()
        
        assert stats['total_points'] == 10
        assert stats['recording_count'] == 10
        assert stats['avg_recording_time_ms'] >= 0
        assert stats['total_recording_time_ms'] >= 0


class TestTrajectoryAnalyzer:
    """Test suite for TrajectoryAnalyzer."""
    
    def _create_test_trajectory(self, n_points: int = 10) -> list:
        """Helper to create test trajectory."""
        points = []
        for i in range(n_points):
            point = IntegratedTrajectoryPoint(
                iteration=i,
                rmsd=10.0 - i * 0.5,  # Decreasing RMSD
                energy=-50.0 - i * 10.0,  # Decreasing energy
                consciousness_frequency=min(15.0, 5.0 + i * 0.5),  # Increasing, capped at 15.0
                consciousness_coherence=min(1.0, 0.3 + i * 0.05),  # Increasing, capped at 1.0
                qcp_score=4.0 + i * 0.2,  # Increasing QCP
                field_coherence=0.0 + i * 0.1,  # Increasing coherence
                stability_score=1.0 + i * 0.15,  # Increasing stability
                phi_match_score=min(1.0, 0.5 + i * 0.03),  # Increasing phi match, capped at 1.0
                timestamp=1000000 + i * 1000
            )
            points.append(point)
        return points
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes with trajectory points."""
        points = self._create_test_trajectory(5)
        analyzer = TrajectoryAnalyzer(points)
        
        assert analyzer._points == points
    
    def test_qcpp_rmsd_correlation(self):
        """Test calculating QCPP-RMSD correlations."""
        points = self._create_test_trajectory(20)
        analyzer = TrajectoryAnalyzer(points)
        
        correlations = analyzer.calculate_qcpp_rmsd_correlation()
        
        # Should have all correlation fields
        assert 'qcp_rmsd_corr' in correlations
        assert 'coherence_rmsd_corr' in correlations
        assert 'stability_rmsd_corr' in correlations
        assert 'phi_rmsd_corr' in correlations
        
        # With our test data (QCP increases, RMSD decreases)
        # Correlation should be negative
        assert correlations['qcp_rmsd_corr'] < 0
        assert correlations['stability_rmsd_corr'] < 0
    
    def test_qcpp_energy_correlation(self):
        """Test calculating QCPP-energy correlations."""
        points = self._create_test_trajectory(20)
        analyzer = TrajectoryAnalyzer(points)
        
        correlations = analyzer.calculate_qcpp_energy_correlation()
        
        # Should have all correlation fields
        assert 'qcp_energy_corr' in correlations
        assert 'coherence_energy_corr' in correlations
        assert 'stability_energy_corr' in correlations
        assert 'phi_energy_corr' in correlations
        
        # With our test data (both QCP and energy increase)
        # Correlation should be negative (energy becomes more negative)
        assert correlations['qcp_energy_corr'] < 0
    
    def test_consciousness_qcpp_correlation(self):
        """Test calculating consciousness-QCPP correlations."""
        points = self._create_test_trajectory(20)
        analyzer = TrajectoryAnalyzer(points)
        
        correlations = analyzer.calculate_consciousness_qcpp_correlation()
        
        # Should have correlation fields
        assert 'frequency_qcp_corr' in correlations
        assert 'coherence_coherence_corr' in correlations
        
        # With our test data (both increase together)
        # Correlations should be positive
        assert correlations['frequency_qcp_corr'] > 0
        assert correlations['coherence_coherence_corr'] > 0
    
    def test_trajectory_summary(self):
        """Test getting trajectory summary statistics."""
        points = self._create_test_trajectory(10)
        analyzer = TrajectoryAnalyzer(points)
        
        summary = analyzer.get_trajectory_summary()
        
        # Check all required fields
        assert 'point_count' in summary
        assert 'iteration_range' in summary
        assert 'rmsd_best' in summary
        assert 'rmsd_worst' in summary
        assert 'rmsd_avg' in summary
        assert 'energy_best' in summary
        assert 'qcp_best' in summary
        assert 'stability_avg' in summary
        
        # Verify values make sense
        assert summary['point_count'] == 10
        assert summary['iteration_range'] == [0, 9]
        assert summary['rmsd_best'] < summary['rmsd_worst']
        assert summary['energy_best'] < summary['energy_worst']
    
    def test_correlation_with_insufficient_points(self):
        """Test that correlation analysis fails with <2 points."""
        points = self._create_test_trajectory(1)
        analyzer = TrajectoryAnalyzer(points)
        
        with pytest.raises(ValueError, match="at least 2 trajectory points"):
            analyzer.calculate_qcpp_rmsd_correlation()
    
    def test_correlation_with_two_points(self):
        """Test correlation analysis works with exactly 2 points."""
        points = self._create_test_trajectory(2)
        analyzer = TrajectoryAnalyzer(points)
        
        # Should not raise error
        correlations = analyzer.calculate_qcpp_rmsd_correlation()
        
        # Should have correlation values
        assert 'qcp_rmsd_corr' in correlations
        # With 2 points, correlation is perfect (±1.0), allowing for floating point error
        assert abs(abs(correlations['qcp_rmsd_corr']) - 1.0) < 1e-6
    
    def test_empty_trajectory_summary(self):
        """Test summary handles empty trajectory."""
        analyzer = TrajectoryAnalyzer([])
        
        summary = analyzer.get_trajectory_summary()
        
        assert 'error' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
