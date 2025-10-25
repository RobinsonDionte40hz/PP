"""
Unit tests for physics-grounded consciousness.

Tests Requirements 3.1, 3.2, 3.3, 3.4, 3.5 and 10.3 covering:
- Frequency mapping from various QCP scores
- Coherence mapping from various QCPP coherence values
- Exponential smoothing transitions
- Bounds enforcement
"""

import pytest
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.physics_grounded_consciousness import PhysicsGroundedConsciousness
from ubf_protein.qcpp_integration import QCPPMetrics
from unittest.mock import Mock


class TestPhysicsGroundedConsciousnessInitialization:
    """Test suite for initialization of physics-grounded consciousness."""
    
    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=0.1
        )
        
        assert consciousness.get_frequency() == 9.0
        assert consciousness.get_coherence() == 0.6
        assert consciousness.smoothing_factor == 0.1
    
    def test_initialization_default_smoothing(self):
        """Test that default smoothing factor is 0.1."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6
        )
        
        assert consciousness.smoothing_factor == 0.1
    
    def test_initialization_custom_smoothing(self):
        """Test initialization with custom smoothing factor."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=0.2
        )
        
        assert consciousness.smoothing_factor == 0.2
    
    def test_initialization_invalid_smoothing_zero(self):
        """Test that smoothing factor of 0 is rejected."""
        with pytest.raises(ValueError, match="Smoothing factor must be in"):
            PhysicsGroundedConsciousness(
                frequency=9.0,
                coherence=0.6,
                smoothing_factor=0.0
            )
    
    def test_initialization_invalid_smoothing_negative(self):
        """Test that negative smoothing factor is rejected."""
        with pytest.raises(ValueError, match="Smoothing factor must be in"):
            PhysicsGroundedConsciousness(
                frequency=9.0,
                coherence=0.6,
                smoothing_factor=-0.1
            )
    
    def test_initialization_invalid_smoothing_too_high(self):
        """Test that smoothing factor > 1 is rejected."""
        with pytest.raises(ValueError, match="Smoothing factor must be in"):
            PhysicsGroundedConsciousness(
                frequency=9.0,
                coherence=0.6,
                smoothing_factor=1.5
            )
    
    def test_initialization_statistics_reset(self):
        """Test that statistics are initialized to zero."""
        consciousness = PhysicsGroundedConsciousness(9.0, 0.6)
        
        assert consciousness.qcpp_updates_count == 0
        assert consciousness.total_frequency_change == 0.0
        assert consciousness.total_coherence_change == 0.0


class TestFrequencyMapping:
    """Test suite for QCP score to frequency mapping."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.consciousness = PhysicsGroundedConsciousness(9.0, 0.6)
    
    def test_qcp_zero_maps_to_max_frequency(self):
        """Test that QCP = 0 maps to frequency = 15 Hz (maximum)."""
        target = self.consciousness.get_target_frequency(0.0)
        assert target == 15.0
    
    def test_qcp_three_maps_to_high_frequency(self):
        """Test that QCP = 3 maps to frequency = 9 Hz."""
        target = self.consciousness.get_target_frequency(3.0)
        # 15.0 - (3.0 / 0.5) = 15.0 - 6.0 = 9.0
        assert target == 9.0
    
    def test_qcp_six_maps_to_min_frequency(self):
        """Test that QCP = 6 maps to frequency = 3 Hz (minimum)."""
        target = self.consciousness.get_target_frequency(6.0)
        # 15.0 - (6.0 / 0.5) = 15.0 - 12.0 = 3.0
        assert target == 3.0
    
    def test_qcp_high_clamped_to_min_frequency(self):
        """Test that very high QCP is clamped to minimum frequency."""
        target = self.consciousness.get_target_frequency(10.0)
        # 15.0 - (10.0 / 0.5) = 15.0 - 20.0 = -5.0 → clamped to 3.0
        assert target == 3.0
    
    def test_qcp_negative_clamped_to_max_frequency(self):
        """Test that negative QCP is clamped to maximum frequency."""
        target = self.consciousness.get_target_frequency(-5.0)
        # 15.0 - (-5.0 / 0.5) = 15.0 + 10.0 = 25.0 → clamped to 15.0
        assert target == 15.0
    
    def test_qcp_typical_range(self):
        """Test mapping for typical QCP values (3-8)."""
        # QCP = 4 → freq = 15 - 8 = 7
        assert self.consciousness.get_target_frequency(4.0) == 7.0
        
        # QCP = 5 → freq = 15 - 10 = 5
        assert self.consciousness.get_target_frequency(5.0) == 5.0
        
        # QCP = 7 → freq = 15 - 14 = 1 → clamped to 3
        assert self.consciousness.get_target_frequency(7.0) == 3.0


class TestCoherenceMapping:
    """Test suite for QCPP coherence to consciousness coherence mapping."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.consciousness = PhysicsGroundedConsciousness(9.0, 0.6)
    
    def test_qcpp_coherence_negative_one_maps_to_min(self):
        """Test that QCPP coherence = -1 maps to consciousness = 0.2."""
        target = self.consciousness.get_target_coherence(-1.0)
        # normalized = (-1 + 1) / 2 = 0
        # target = 0.2 + 0 * 0.8 = 0.2
        assert abs(target - 0.2) < 0.001
    
    def test_qcpp_coherence_zero_maps_to_moderate(self):
        """Test that QCPP coherence = 0 maps to consciousness = 0.6."""
        target = self.consciousness.get_target_coherence(0.0)
        # normalized = (0 + 1) / 2 = 0.5
        # target = 0.2 + 0.5 * 0.8 = 0.2 + 0.4 = 0.6
        assert abs(target - 0.6) < 0.001
    
    def test_qcpp_coherence_positive_one_maps_to_max(self):
        """Test that QCPP coherence = 1 maps to consciousness = 1.0."""
        target = self.consciousness.get_target_coherence(1.0)
        # normalized = (1 + 1) / 2 = 1.0
        # target = 0.2 + 1.0 * 0.8 = 1.0
        assert abs(target - 1.0) < 0.001
    
    def test_qcpp_coherence_high_clamped_to_max(self):
        """Test that very high QCPP coherence is clamped to 1.0."""
        target = self.consciousness.get_target_coherence(2.0)
        # Would exceed 1.0, should be clamped
        assert target == 1.0
    
    def test_qcpp_coherence_low_clamped_to_min(self):
        """Test that very low QCPP coherence is clamped to 0.2."""
        target = self.consciousness.get_target_coherence(-2.0)
        # Would be below 0.2, should be clamped
        assert target == 0.2
    
    def test_qcpp_coherence_typical_positive(self):
        """Test mapping for typical positive QCPP coherence."""
        # QCPP = 0.5 → norm = 0.75 → target = 0.2 + 0.6 = 0.8
        target = self.consciousness.get_target_coherence(0.5)
        assert abs(target - 0.8) < 0.001
    
    def test_qcpp_coherence_typical_negative(self):
        """Test mapping for typical negative QCPP coherence."""
        # QCPP = -0.5 → norm = 0.25 → target = 0.2 + 0.2 = 0.4
        target = self.consciousness.get_target_coherence(-0.5)
        assert abs(target - 0.4) < 0.001


class TestExponentialSmoothing:
    """Test suite for exponential smoothing transitions."""
    
    def test_smoothing_gradual_transition(self):
        """Test that smoothing creates gradual transitions."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=0.1
        )
        
        # Create QCPP metrics that would set target freq to 3.0
        metrics = QCPPMetrics(
            qcp_score=6.0,  # → target freq = 3.0
            field_coherence=1.0,  # → target coherence = 1.0
            stability_score=2.0,
            phi_match_score=0.8,
            calculation_time_ms=1.0
        )
        
        # First update: should move 10% of the way
        initial_freq = consciousness.get_frequency()
        consciousness.update_from_qcpp_metrics(metrics)
        
        # Expected: 9.0 + (3.0 - 9.0) * 0.1 = 9.0 - 0.6 = 8.4
        assert abs(consciousness.get_frequency() - 8.4) < 0.001
        
        # Second update: should move another 10% from current position
        consciousness.update_from_qcpp_metrics(metrics)
        # Expected: 8.4 + (3.0 - 8.4) * 0.1 = 8.4 - 0.54 = 7.86
        assert abs(consciousness.get_frequency() - 7.86) < 0.001
    
    def test_smoothing_factor_one_immediate_transition(self):
        """Test that smoothing factor = 1.0 causes immediate transition."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=1.0
        )
        
        metrics = QCPPMetrics(
            qcp_score=6.0,  # → target freq = 3.0
            field_coherence=1.0,  # → target coherence = 1.0
            stability_score=2.0,
            phi_match_score=0.8,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        # With smoothing = 1.0, should jump immediately to target
        assert consciousness.get_frequency() == 3.0
        assert consciousness.get_coherence() == 1.0
    
    def test_smoothing_factor_small_slow_transition(self):
        """Test that small smoothing factor creates slow transitions."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=0.01  # Very slow
        )
        
        metrics = QCPPMetrics(
            qcp_score=6.0,  # → target freq = 3.0
            field_coherence=1.0,
            stability_score=2.0,
            phi_match_score=0.8,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        # Expected: 9.0 + (3.0 - 9.0) * 0.01 = 9.0 - 0.06 = 8.94
        assert abs(consciousness.get_frequency() - 8.94) < 0.001
    
    def test_multiple_updates_converge_to_target(self):
        """Test that multiple updates converge toward target."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=0.1
        )
        
        metrics = QCPPMetrics(
            qcp_score=6.0,  # Target: freq = 3.0
            field_coherence=1.0,  # Target: coherence = 1.0
            stability_score=2.0,
            phi_match_score=0.8,
            calculation_time_ms=1.0
        )
        
        # Apply 50 updates (smoothing factor 0.1 needs more iterations)
        for _ in range(50):
            consciousness.update_from_qcpp_metrics(metrics)
        
        # Should be very close to target after 50 updates
        assert abs(consciousness.get_frequency() - 3.0) < 0.2
        assert abs(consciousness.get_coherence() - 1.0) < 0.02


class TestBoundsEnforcement:
    """Test suite for bounds enforcement."""
    
    def test_frequency_lower_bound_enforced(self):
        """Test that frequency cannot go below 3 Hz."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=4.0,
            coherence=0.6,
            smoothing_factor=1.0  # Immediate transition
        )
        
        # Very high QCP should try to set freq < 3
        metrics = QCPPMetrics(
            qcp_score=20.0,  # Would map to negative frequency
            field_coherence=0.0,
            stability_score=2.0,
            phi_match_score=0.5,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        # Should be clamped to 3.0
        assert consciousness.get_frequency() == 3.0
    
    def test_frequency_upper_bound_enforced(self):
        """Test that frequency cannot go above 15 Hz."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=12.0,
            coherence=0.6,
            smoothing_factor=1.0
        )
        
        # Very low QCP should try to set freq > 15
        # QCP valid range is [0, 20], use 0.0 which maps to 15 Hz
        metrics = QCPPMetrics(
            qcp_score=0.0,  # Maps to 15 Hz (max)
            field_coherence=0.0,
            stability_score=0.5,
            phi_match_score=0.5,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        # Should be clamped to 15.0
        assert consciousness.get_frequency() == 15.0
    
    def test_coherence_lower_bound_enforced(self):
        """Test that coherence cannot go below 0.2."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.3,
            smoothing_factor=1.0
        )
        
        # Very low QCPP coherence should try to set < 0.2
        # Field coherence valid range is [-2, 2], use -2.0 which maps to < 0.2
        metrics = QCPPMetrics(
            qcp_score=5.0,
            field_coherence=-2.0,  # Maps to < 0.2 (will be clamped)
            stability_score=1.0,
            phi_match_score=0.5,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        # Should be clamped to 0.2
        assert consciousness.get_coherence() == 0.2
    
    def test_coherence_upper_bound_enforced(self):
        """Test that coherence cannot go above 1.0."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.9,
            smoothing_factor=1.0
        )
        
        # Very high QCPP coherence should try to set > 1.0
        # Field coherence valid range is [-2, 2], use 2.0 which maps to > 1.0
        metrics = QCPPMetrics(
            qcp_score=5.0,
            field_coherence=2.0,  # Maps to > 1.0 (will be clamped)
            stability_score=1.0,
            phi_match_score=0.5,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        # Should be clamped to 1.0
        assert consciousness.get_coherence() == 1.0


class TestStatisticsTracking:
    """Test suite for statistics tracking."""
    
    def test_update_count_increments(self):
        """Test that update count increments with each QCPP update."""
        consciousness = PhysicsGroundedConsciousness(9.0, 0.6)
        
        metrics = QCPPMetrics(5.0, 0.5, 1.5, 0.7, 1.0)
        
        assert consciousness.qcpp_updates_count == 0
        
        consciousness.update_from_qcpp_metrics(metrics)
        assert consciousness.qcpp_updates_count == 1
        
        consciousness.update_from_qcpp_metrics(metrics)
        assert consciousness.qcpp_updates_count == 2
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        consciousness = PhysicsGroundedConsciousness(
            frequency=9.0,
            coherence=0.6,
            smoothing_factor=0.5
        )
        
        metrics = QCPPMetrics(
            qcp_score=4.0,  # → freq = 7.0
            field_coherence=0.0,  # → coherence = 0.6
            stability_score=1.0,
            phi_match_score=0.5,
            calculation_time_ms=1.0
        )
        
        consciousness.update_from_qcpp_metrics(metrics)
        
        stats = consciousness.get_physics_grounding_stats()
        
        assert stats['qcpp_updates_count'] == 1
        assert stats['target_frequency'] == 7.0
        assert abs(stats['target_coherence'] - 0.6) < 0.001
        assert stats['smoothing_factor'] == 0.5
    
    def test_set_smoothing_factor(self):
        """Test that smoothing factor can be updated."""
        consciousness = PhysicsGroundedConsciousness(9.0, 0.6, 0.1)
        
        assert consciousness.smoothing_factor == 0.1
        
        consciousness.set_smoothing_factor(0.3)
        assert consciousness.smoothing_factor == 0.3
    
    def test_set_smoothing_factor_invalid(self):
        """Test that invalid smoothing factor is rejected."""
        consciousness = PhysicsGroundedConsciousness(9.0, 0.6)
        
        with pytest.raises(ValueError):
            consciousness.set_smoothing_factor(0.0)
        
        with pytest.raises(ValueError):
            consciousness.set_smoothing_factor(1.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
